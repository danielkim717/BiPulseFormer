"""Subject-filtered UBFC-PHYS clips on the verified native-FPS transfer cache."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PHYSCacheDataset(Dataset):
    dataset_name = 'UBFC-PHYS'

    def __init__(self, plan_path, subjects, clip_len=160, step=80, random_hflip=False):
        plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
        self.clip_len, self.random_hflip = clip_len, random_hflip
        self.cache = Path(plan['cache'])
        self.records = {}
        self.samples = []
        self._current_id, self._current_rgb = None, None
        selected = set(subjects)
        for record in plan['records']:
            vid = record['video_id']
            if vid.split('_')[0] not in selected:
                continue
            meta = json.loads((self.cache / f'{vid}.json').read_text(encoding='utf-8'))
            if meta['record'] != record:
                raise ValueError(f'PHYS cache inventory mismatch: {vid}')
            signal_path = Path(record['bvp'])
            if hashlib.sha256(signal_path.read_bytes()).hexdigest() != record['bvp_sha256']:
                raise ValueError(f'PHYS BVP changed: {vid}')
            raw = np.loadtxt(signal_path, dtype=np.float64)
            if raw.ndim != 1 or not np.isfinite(raw).all():
                raise ValueError(f'Invalid PHYS BVP: {vid}')
            # Physical seconds, identical to evaluate_phys_transfer.py.
            end = min((record['frames'] - 1) / record['fps'], (len(raw) - 1) / 64.)
            times = np.arange(int(np.floor(end * 30)) + 1) / 30.
            bvp = np.interp(times, np.arange(len(raw)) / 64., raw).astype(np.float32)
            rgb = np.load(self.cache / f'{vid}.npy', mmap_mode='r')
            if rgb.shape != (record['frames'], 128, 128, 3) or rgb.dtype != np.uint8:
                raise ValueError(f'Invalid PHYS cache shape: {vid}')
            if hashlib.sha256(rgb).hexdigest() != meta['pixel_sha256']:
                raise ValueError(f'PHYS cache pixels changed: {vid}')
            del rgb
            self.records[vid] = dict(positions=times * record['fps'], bvp=bvp)
            self.samples.extend(dict(video_id=vid, first_frame_idx=start)
                                for start in range(0, len(times) - clip_len, step))
        actual = {s['video_id'].split('_')[0] for s in self.samples}
        if actual != selected:
            raise ValueError(f'PHYS subject inventory mismatch: {actual} != {selected}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        vid, start = sample['video_id'], sample['first_frame_idx']
        if self._current_id != vid:
            self._current_rgb = np.load(self.cache / f'{vid}.npy', mmap_mode='r')
            self._current_id = vid
        rgb = self._current_rgb
        record = self.records[vid]
        positions = record['positions'][start:start + self.clip_len + 1]
        lo = np.floor(positions).astype(int)
        hi = np.minimum(lo + 1, len(rgb) - 1)
        w = (positions - lo).astype(np.float32)[:, None, None, None]
        frames = (rgb[lo].astype(np.float32) * (1 - w) + rgb[hi].astype(np.float32) * w) / 255.
        x = torch.from_numpy(frames).permute(3, 0, 1, 2)
        x = (x[:, 1:] - x[:, :-1]) / (x[:, 1:] + x[:, :-1] + 1e-7)
        x = x / (x.std() + 1e-7)
        if self.random_hflip and torch.rand(()) < .5:
            x = x.flip(-1)
        y = torch.from_numpy(record['bvp'][start:start + self.clip_len + 1].copy()).diff()
        if y.std() < 1e-9:
            raise ValueError(f'Flat PHYS BVP: {vid}/{start}')
        return x, y / (y.std() + 1e-7)
