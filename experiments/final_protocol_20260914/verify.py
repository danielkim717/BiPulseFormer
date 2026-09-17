"""Read-only integrity checks for the public fixed-protocol archive."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    source = root / 'source'
    manifest = json.loads((root / 'snapshot_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        actual = hashlib.sha256((source / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Implementation hash mismatch: {name}')
    vendor = source / 'third_party/rppg_toolbox_b7500b8'
    upstream = json.loads((vendor / 'PINNED_MANIFEST.json').read_text(encoding='utf-8'))
    for name, expected in upstream['sha256'].items():
        if hashlib.sha256((vendor / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Upstream hash mismatch: {name}')
    results = json.loads((root / 'results.json').read_text(encoding='utf-8'))
    if hashlib.sha256((root / 'protocol.json').read_bytes()).hexdigest() != results['protocol_sha256']:
        raise ValueError('Protocol hash mismatch')
    assert len(results['results']) == 4
    for row in results['results']:
        rec = row['test']['per_recording']
        errors = [abs(p-g) for p, g in zip(rec['pred_hrs'], rec['gt_hrs'])]
        assert len(errors) == len(rec['recording_ids']) == rec['n_recordings']
        assert abs(sum(errors) / len(errors) - rec['MAE_bpm']) < 1e-10
        assert abs((sum(e*e for e in errors) / len(errors))**0.5 - rec['RMSE_bpm']) < 1e-10
    print(f'Verified {len(manifest)} source files, upstream hashes, protocol and four result aggregates.')


if __name__ == '__main__':
    main()
