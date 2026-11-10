"""Recording-level reference preprocessing, shared immutable caches and subject splits."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from src.toolbox_reference import BaseLoader, PURELoader, UBFCrPPGLoader, REFERENCE, COMMIT


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    import time
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(30):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 29: raise
            time.sleep(.1)


def inventory(dataset, root, split='all'):
    if dataset == 'UBFC-PHYS':
        from src.phys_selection import validate_inventory
        plan = json.loads(Path(root).read_text(encoding='utf-8'))
        validate_inventory(plan)
        ready = json.loads(Path(plan['preparation_status']).read_text(encoding='utf-8'))
        if ready['status'] != 'complete': raise ValueError('PHYS preparation incomplete')
        return [dict(id=r['video_id'], subject=r['video_id'].split('_')[0],
                     video=r['video'], label=r['bvp'], raw=r) for r in plan['records']]
    cls = PURELoader if dataset == 'PURE' else UBFCrPPGLoader
    loader = object.__new__(cls)
    loader.dataset_name = dataset
    raw = loader.get_raw_data(str(root))
    # UBFC upstream uses glob order, not numeric sort. Persist the exact selected order.
    if split != 'all':
        raw = loader.split_raw_data(raw, *( (0, .8) if split == 'train' else (.8, 1)))
    records = []
    for r in raw:
        p = Path(r['path']); name = p.name
        records.append(dict(id=name, subject=name.split('-')[0] if dataset == 'PURE' else name,
                            video=str(p/name) if dataset == 'PURE' else str(p/'vid.avi'),
                            label=str(p/(name+'.json')) if dataset == 'PURE' else str(p/'ground_truth.txt')))
    return records


def cropped_frames(record, dataset):
    """Stream decode with exactly the HC first-frame box and INTER_AREA recipe."""
    helper = BaseLoader()
    box = None
    frames = []
    if dataset == 'PURE':
        paths = sorted(Path(record['video']).glob('*.png'))
        iterator = (cv2.imread(str(p)) for p in paths)
        cap = None
    else:
        cap = cv2.VideoCapture(record['video'])
        if not cap.isOpened(): raise ValueError(f'Cannot open {record["video"]}')
        def decoded():
            while True:
                ok, frame = cap.read()
                if not ok: break
                yield frame
        iterator = decoded()
    try:
        for frame in iterator:
            if frame is None: raise ValueError('Image decode failed')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if box is None:
                box = np.asarray(helper.face_detection(frame, 'HC', True, 1.5), dtype=int)
            x,y,w,h = box
            crop = frame[max(y,0):min(y+h,frame.shape[0]), max(x,0):min(x+w,frame.shape[1])]
            frames.append(cv2.resize(crop, (128,128), interpolation=cv2.INTER_AREA))
    finally:
        if cap is not None: cap.release()
    if len(frames) < 160: raise ValueError(f'Too few decoded frames: {record["id"]}')
    return np.asarray(frames), box.tolist()


def ubfc_chunk_alignment(decoded_frames, label_samples, metadata_frames=None):
    """Match BaseLoader.chunk: video determines chunk count; do not resample labels.

    UBFCrPPGLoader normalizes the original full label before slicing its prefix.
    A truncated video can therefore supply full, aligned chunks. Missing labels
    for any such chunk remain an error, rather than padding invented observations.
    """
    usable = decoded_frames // 160 * 160
    if usable < 160 or label_samples < usable:
        raise ValueError(f'Insufficient labels for decoded full chunks: video={decoded_frames}, labels={label_samples}, usable={usable}')
    return dict(policy='toolbox_video_determined_full_chunks',decoded_frames=decoded_frames,
                metadata_frames=metadata_frames,label_samples=label_samples,usable_frames=usable,
                full_chunks=usable//160,unused_decoded_tail=decoded_frames-usable,
                unused_label_tail=label_samples-usable,
                decode_shortfall=None if metadata_frames is None else metadata_frames-decoded_frames)


def prepare(records, dataset, cache, log):
    cache = Path(cache)/dataset
    cache.mkdir(parents=True, exist_ok=True)
    results = []
    for index, record in enumerate(records, 1):
        video = Path(record['video'])
        files = sorted(video.glob('*.png')) if video.is_dir() else [video]
        identity = dict(commit=COMMIT, record=record, inputs=[(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files],
                        label_sha256=digest(record['label']), version='matched-stream-v1')
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        base = cache/(record['id']+'_'+key[:16])
        meta_path = base.with_suffix('.json')
        log(f'preprocessing {dataset} {record["id"]} {index}/{len(records)}',
            phase='preprocessing', recording=index, recordings=len(records))
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            if meta['identity'] != json.loads(json.dumps(identity)): raise ValueError('Cache identity mismatch')
            for p, sha in meta['hashes'].items():
                if digest(p) != sha: raise ValueError(f'Cache corruption: {p}')
        else:
            rgb, box = cropped_frames(record, dataset)
            alignment = None
            if dataset == 'PURE':
                labels = BaseLoader.resample_ppg(PURELoader.read_wave(record['label']), len(rgb))
            elif dataset == 'UBFC-rPPG':
                labels = UBFCrPPGLoader.read_wave(record['label'])
                cap=cv2.VideoCapture(record['video']);declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
                alignment=ubfc_chunk_alignment(len(rgb),len(labels),declared)
                if len(labels)!=len(rgb) or declared!=len(rgb):
                    log(f'WARNING {record["id"]}: {json.dumps(alignment)}',phase='preprocessing',
                        recording=index,recordings=len(records),input_warning=alignment)
            else:
                # PHYS-specific adapter: native AVI FPS and 64-Hz BVP -> common 30 Hz.
                cap = cv2.VideoCapture(record['video']); fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
                labels = np.loadtxt(record['label'], dtype=np.float64)
                raw = record['raw']
                if video.stat().st_size != raw['video_size'] or video.stat().st_mtime_ns != raw['video_mtime_ns'] or digest(record['label']) != raw['bvp_sha256']:
                    raise ValueError('PHYS input changed after verification')
                duration = min((len(rgb)-1)/fps, (len(labels)-1)/64)
                times = np.arange(int(np.floor(duration*30))+1)/30
                pos = times*fps; lo = np.floor(pos).astype(int); hi = np.minimum(lo+1,len(rgb)-1)
                common = np.empty((len(times),128,128,3), dtype=np.float64)
                for start in range(0,len(times),128):
                    end=start+128; w=(pos[start:end]-lo[start:end])[:,None,None,None]
                    common[start:end]=rgb[lo[start:end]]*(1-w)+rgb[hi[start:end]]*w
                rgb=common
                labels=np.interp(times,np.arange(len(labels))/64,labels)
            data = BaseLoader.diff_normalize_data(rgb.astype(np.float64, copy=False))
            labels = BaseLoader.diff_normalize_label(labels)
            if not np.isfinite(data).all() or not np.isfinite(labels).all(): raise ValueError('Nonfinite preprocessed data')
            dp,lp=base.with_suffix('.data.npy'),base.with_suffix('.label.npy')
            np.save(dp,data); np.save(lp,labels)
            meta=dict(identity=identity, data=str(dp.resolve()), label=str(lp.resolve()), frames=len(data), box=box,alignment=alignment,
                      hashes={str(p.resolve()):digest(p) for p in (dp,lp)})
            atomic_json(meta_path,meta)
            del data, rgb
        results.append(dict(record, **{k:meta[k] for k in ('data','label','frames')}, cache_key=key,
                            alignment=meta.get('alignment')))
    return results


class Clips(Dataset):
    def __init__(self, records):
        self.samples=[]
        self.records=records
        self.arrays={r['id']:(np.load(r['data'],mmap_mode='r'),np.load(r['label'],mmap_mode='r')) for r in records}
        for r in records:
            if len(self.arrays[r['id']][1]) < r['frames']//160*160:
                raise ValueError(f'Cached labels are insufficient for full clips: {r["id"]}')
            for start in range(0,r['frames']-159,160):
                self.samples.append(dict(video_id=r['id'],subject=r['subject'],start=start,end=start+160))
        if not self.samples: raise ValueError('Empty clip set')
    def __len__(self): return len(self.samples)
    def __getitem__(self,index):
        s=self.samples[index]; x,y=self.arrays[s['video_id']]; start=s['start']
        return torch.from_numpy(np.array(x[start:start+160].transpose(3,0,1,2))), torch.tensor(y[start:start+160],dtype=torch.float32)
