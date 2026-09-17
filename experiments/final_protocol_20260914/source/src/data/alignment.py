"""Align PURE PPG measurements to camera timestamps without assuming a 2:1 rate."""
from pathlib import Path

import numpy as np


def align_pure(ppg_entries, image_paths):
    if len(ppg_entries) < 2 or not image_paths:
        raise ValueError('Insufficient PURE timestamps')
    sensor_ns = np.asarray([entry['Timestamp'] for entry in ppg_entries], dtype=np.int64)
    values = np.asarray([entry['Value']['waveform'] for entry in ppg_entries], dtype=np.float64)
    frame_ns = np.asarray([int(Path(path).stem.removeprefix('Image')) for path in image_paths], dtype=np.int64)
    if np.any(np.diff(sensor_ns) <= 0) or np.any(np.diff(frame_ns) <= 0):
        raise ValueError('PURE timestamps must be strictly increasing')
    if not np.isfinite(values).all():
        raise ValueError('Non-finite PURE PPG')
    # Do not extrapolate labels beyond the measured PPG interval.
    keep = (frame_ns >= sensor_ns[0]) & (frame_ns <= sensor_ns[-1])
    if not keep.any():
        raise ValueError('No overlapping camera and PPG timestamps')
    frame_ns = frame_ns[keep]
    paths = [path for path, valid in zip(image_paths, keep) if valid]
    # Subtract the epoch while still int64 to preserve nanosecond precision.
    origin = sensor_ns[0]
    labels = np.interp((frame_ns - origin) / 1e9, (sensor_ns - origin) / 1e9, values)
    return paths, labels.tolist(), int((~keep).sum())
