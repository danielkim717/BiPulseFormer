"""Read-only package, source and subject-split checks; no data or GPU required."""
from pathlib import Path
import hashlib, json
ROOT = Path(__file__).resolve().parent

def verify():
    manifest = json.loads((ROOT / 'SHA256SUMS.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        path = (ROOT / name).resolve()
        path.relative_to(ROOT)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    sources = json.loads((ROOT / 'SOURCE_HASHES.json').read_text(encoding='utf-8'))
    for name, expected in sources.items():
        assert manifest[name] == expected, name
    splits = json.loads((ROOT / 'splits.json').read_text(encoding='utf-8'))
    def check(name, split):
        split = split.get('subjects_and_recordings', split)
        sets = [set(map(str, split.get(role, []))) for role in ('train', 'valid', 'test')]
        assert not any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)), name
    for name, split in splits['splits'].items():
        check(name, split)
    for name, split in splits['UBFC_to_PURE'].items():
        check(name, split)
    for name, task in json.loads((ROOT / 'run_index.json').read_text(encoding='utf-8')).items():
        assert (ROOT / task['protocol']).is_file(), name
    print(f'PASS: {len(manifest)} file hashes, {len(sources)} unchanged scientific sources, subject-disjoint splits and task configurations')

if __name__ == '__main__':
    verify()
