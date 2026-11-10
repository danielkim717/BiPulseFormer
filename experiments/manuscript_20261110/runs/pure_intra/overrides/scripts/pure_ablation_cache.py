"""Share the unchanged lossless PURE cache across sequential K experiments."""
import json
from pathlib import Path

from pure_frame_cache import RawFrameCache


class SharedFrameCache(RawFrameCache):
    def __init__(self, directory, identity, max_bytes=8 * 1024**3):
        # JSON stores tuples as lists; compare the same representation on reopen.
        identity = json.loads(json.dumps(identity, allow_nan=False))
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = self.directory / 'identity.json'
        if stamp.exists():
            if json.loads(stamp.read_text()) != identity:
                raise ValueError('Shared cache input identity changed')
        else:
            if any(self.directory.iterdir()):
                raise ValueError('Nonempty cache without input identity')
            stamp.write_text(json.dumps(identity, indent=2), encoding='utf-8')
        self.max_bytes = max_bytes
        self.bytes = sum(p.stat().st_size for p in self.directory.glob('*.npy'))
        self.hits = self.misses = self.writes = 0
