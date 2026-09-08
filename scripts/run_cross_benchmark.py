"""Frozen-code, sequential cross benchmark and automatically updated comparison."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment import write_json
from src.protocol import fingerprint, read_protocol


def update_comparison(output, jobs):
    completed = []
    for job in jobs:
        path = output / job['name'] / 'summary.json'
        if path.is_file():
            completed.append((job, json.loads(path.read_text(encoding='utf-8'))))
    lines = ['# Cross-dataset comparison', '',
             'Protocol: source subject 80/20 train/validation, entire target test; 20 epochs.',
             'Epoch selection uses validation clip HR RMSE only. Target metrics below are final-test results.', '',
             f'Completed runs: {len(completed)} / {len(jobs)}. Partial seed results are provisional.', '',
             '| Direction | Model | Seed | Best epoch | Clip MAE | Clip RMSE | Recording MAE | Recording RMSE | HR r |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for job, result in completed:
        c, r = result['test']['per_clip'], result['test']['per_recording']
        lines.append(f"| {job['source']} → {job['target']} | {job['model']} | {job['seed']} | "
                     f"{result['best_epoch']} | {c['MAE_bpm_clip']:.3f} | {c['RMSE_bpm_clip']:.3f} | "
                     f"{r['MAE_bpm']:.3f} | {r['RMSE_bpm']:.3f} | {r['Pearson']:.4f} |")
    for source in ('PURE', 'UBFC-rPPG'):
        groups = {model: [(job, r) for job, r in completed if job['source'] == source and job['model'] == model]
                  for model in ('bipulseformer', 'physformer')}
        if all(len(group) == 3 for group in groups.values()):
            from scripts.summarize_runs import summarize
            summaries = {model: summarize([output / job['name'] / 'summary.json' for job, _ in group])
                         for model, group in groups.items()}
            write_json(output / f'{source}_aggregate.json', summaries)
            lines.extend(['', f'## {source} source: all three seeds', '',
                          '| Model | Recording MAE mean ± SD | Clip MAE mean ± SD |', '|---|---:|---:|'])
            for model, summary in summaries.items():
                m = summary['metrics']
                r, c = m['per_recording']['MAE_bpm'], m['per_clip']['MAE_bpm_clip']
                lines.append(f"| {model} | {r['mean']:.3f} ± {r['std']:.3f} | {c['mean']:.3f} ± {c['std']:.3f} |")
            bi = summaries['bipulseformer']['metrics']['per_recording']['MAE_bpm']['mean']
            base = summaries['physformer']['metrics']['per_recording']['MAE_bpm']['mean']
            if base > 0:
                lines.extend(['', f'BiPulseFormer recording MAE reduction versus baseline: {(base-bi)/base*100:.2f}%.'])
    temp = output / 'comparison.md.tmp'
    temp.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    temp.replace(output / 'comparison.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pure-root', default='D:/PURE')
    parser.add_argument('--ubfc-root', default='D:/UBFC-rPPG')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a new benchmark output: {output}')
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / 'source_snapshot'
    for directory in ('src', 'scripts', 'configs'):
        shutil.copytree(ROOT / directory, frozen / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    config = read_protocol(frozen / 'configs/protocol_v1.json')
    paths = {'PURE': str(Path(args.pure_root).resolve()), 'UBFC-rPPG': str(Path(args.ubfc_root).resolve())}
    jobs = []
    # Complete both model comparisons at seed 42 first, then remaining seeds.
    for seed in config['report_seeds']:
        for source, target in [('PURE', 'UBFC-rPPG'), ('UBFC-rPPG', 'PURE')]:
            for model in ('bipulseformer', 'physformer'):
                name = f"{source}_to_{target}_{model}_s{seed}"
                jobs.append({'name': name, 'source': source, 'target': target,
                             'model': model, 'seed': seed, 'status': 'pending'})
    state = {'protocol_hash': fingerprint(config), 'jobs': jobs, 'pid': os.getpid(),
             'started_at': time.time(), 'status': 'running'}
    write_json(output / 'status.json', state)
    update_comparison(output, jobs)
    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    for job in jobs:
        job['status'], job['started_at'] = 'running', time.time()
        write_json(output / 'status.json', state)
        command = [sys.executable, '-u', str(frozen / 'scripts/run_experiment.py'),
                   '--config', str(frozen / 'configs/protocol_v1.json'), '--model', job['model'],
                   '--source', job['source'], '--target', job['target'],
                   '--source-root', paths[job['source']], '--target-root', paths[job['target']],
                   '--seed', str(job['seed']), '--workers', str(args.workers), '--threads', '2',
                   '--output', str(output / job['name'])]
        print(f"Starting {job['name']}", flush=True)
        with (output / f"{job['name']}.stdout.log").open('w', encoding='utf-8') as handle:
            process = subprocess.Popen(command, cwd=frozen, env=env, stdout=handle, stderr=subprocess.STDOUT)
            job['pid'] = process.pid
            write_json(output / 'status.json', state)
            code = process.wait()
        job['exit_code'], job['finished_at'] = code, time.time()
        job['status'] = 'complete' if code == 0 else 'failed'
        write_json(output / 'status.json', state)
        update_comparison(output, jobs)
        if code:
            state['status'] = 'failed'
            write_json(output / 'status.json', state)
            raise SystemExit(f"Experiment failed: {job['name']}; inspect its stdout log")
    state['status'], state['finished_at'] = 'complete', time.time()
    write_json(output / 'status.json', state)
    print(f"Complete: {output / 'comparison.md'}", flush=True)


if __name__ == '__main__':
    main()
