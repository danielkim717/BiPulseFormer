"""Bounded lossless cache before random augmentation; single-process dataset only."""
import os
from pathlib import Path
import shutil


class RawFrameCache:
    def __init__(self, directory, max_bytes=8 * 1024**3):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.max_bytes = max_bytes
        self.bytes = 0
        self.hits = self.misses = self.writes = 0

    def get(self, index, decode):
        import numpy as np
        import torch
        path = self.directory / f'{index:06d}.npy'
        if path.exists():
            array = np.load(path, allow_pickle=False)
            if array.dtype != np.float32 or array.shape != (161, 3, 128, 128):
                raise ValueError('Invalid prepared PURE input cache')
            self.hits += 1
            # Recreate the original stack(T,C,H,W).permute(C,T,H,W) strides.
            return torch.from_numpy(array).permute(1, 0, 2, 3)
        self.misses += 1
        frames = decode()
        array = frames.permute(1, 0, 2, 3).numpy()
        if array.dtype != np.float32 or array.shape != (161, 3, 128, 128) or not array.flags.c_contiguous:
            raise ValueError('Unexpected original PURE input layout')
        size = array.nbytes + 256
        if self.bytes + size <= self.max_bytes and shutil.disk_usage(self.directory).free > size + 3*1024**3:
            temporary = path.with_suffix('.tmp')
            try:
                with temporary.open('wb') as stream:
                    np.save(stream, array, allow_pickle=False)
                actual_size = temporary.stat().st_size
                os.replace(temporary, path)
                self.bytes += actual_size
                self.writes += 1
            except OSError:
                # Cache capacity/IO failure must not remove training samples.
                temporary.unlink(missing_ok=True)
        return frames
