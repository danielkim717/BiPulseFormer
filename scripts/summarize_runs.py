"""Aggregate comparable completed runs; never choose the best test seed."""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.protocol import fingerprint


def summarize(paths):
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    if not runs:
        raise ValueError('Supply completed summary.json files')
    signatures = {fingerprint({"config": r["config"], "split": r["split"],
                               "code_hash": r['code_hash'], 'samples': r['sample_hashes']}) for r in runs}
    if len(signatures) != 1:
        raise ValueError('Do not aggregate different protocols or subject splits')
    identities = [json.loads(Path(p).with_name('config.json').read_text(encoding='utf-8')) for p in paths]
    if len({r['model'] for r in identities}) != 1:
        raise ValueError('Aggregate each model separately')
    seeds = [r['seed'] for r in runs]
    if len(set(seeds)) != len(seeds) or set(seeds) != set(runs[0]['config']['report_seeds']):
        raise ValueError('Supply exactly one completed run for each report seed')
    output = {'model': identities[0]['model'], 'seeds': sorted(seeds), 'metrics': {}}
    for level, names in [('per_clip', ['MAE_bpm_clip', 'RMSE_bpm_clip', 'MAPE_pct_clip', 'Pearson_clip']),
                         ('per_recording', ['MAE_bpm', 'RMSE_bpm', 'MAPE_pct', 'Pearson', 'signal_Pearson_mean'])]:
        output['metrics'][level] = {}
        for name in names:
            values = [r['test'][level][name] for r in runs]
            output['metrics'][level][name] = {'mean': statistics.mean(values), 'std': statistics.stdev(values)}
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summaries', nargs='+')
    args = parser.parse_args()
    print(json.dumps(summarize(args.summaries), indent=2, allow_nan=False))
