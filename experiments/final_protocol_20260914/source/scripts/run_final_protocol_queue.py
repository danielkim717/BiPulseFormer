"""Sequential, resumable source-only comparison and fixed-evaluator target queue."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from run_final_protocol import atomic_json


def render(root, state):
    lines = ['# Fixed-protocol BiPulseFormer vs reported PhysFormer', '',
             'Seed42; source-only recording RMSE selection; common journal GT/coverage/36-198 BPM scorer; v1 inputs and overlap inference.',
             'Test results never select the loss variant. Literature entries are external reproductions.', '',
             '| Direction | Loss | Status | Unit | MAE | RMSE | MAPE | Pearson |',
             '|---|---|---|---|---:|---:|---:|---:|']
    results = []
    for job in state['jobs']:
        if job['kind'] != 'test': continue
        path = root/job['name']/'summary.json'
        if not path.exists():
            lines.append(f'| {job["source"]} → {job["target"]} | {job["variant"]} | {job["status"]} | — | — | — | — | — |')
            continue
        d = json.loads(path.read_text(encoding='utf-8'))
        results.append(dict(job=job['name'], summary=str(path), **d))
        for unit, suffix in [('per_recording',''), ('per_clip','_clip')]:
            vals = [d['test'][unit][k+suffix] for k in ['MAE_bpm','RMSE_bpm','MAPE_pct','Pearson']]
            formatted = ['N/A' if v is None else f'{v:.6f}' for v in vals]
            lines.append(f'| {job["source"]} → {job["target"]} | {job["variant"]} | {job["status"]} | {unit} | '+' | '.join(formatted)+' |')
        for task, row in d['test'].get('by_task', {}).items():
            vals = [row['per_recording'][k] for k in ['MAE_bpm','RMSE_bpm','MAPE_pct','Pearson']]
            lines.append(f'| {job["source"]} → {job["target"]} {task} | {job["variant"]} | {row["n_recordings"]} recordings | recording | '+
                         ' | '.join('N/A' if v is None else f'{v:.6f}' for v in vals)+' |')
    lines += ['', '## PhysFormer literature reference (video-level, not local measurements)', '',
              '| Source | Direction | MAE | RMSE | MAPE | Pearson |', '|---|---|---:|---:|---:|---:|',
              '| Toolbox Table7 | PURE → UBFC-rPPG | 1.44 | 3.77 | 1.66 | .98 |',
              '| FactorizePhys Table2, PhysFormer reproduction | PURE → UBFC-rPPG | 1.01 | 2.40 | 1.21 | .990 |',
              '| Toolbox Table8 | PURE → UBFC-PHYS | 6.04 | 9.77 | 7.67 | .65 |',
              '| Reviewed sources | PHYS → PURE / UBFC-rPPG | unreported | unreported | unreported | unreported |', '',
              '[Toolbox](https://proceedings.neurips.cc/paper_files/paper/2023/file/d7d0d548a6317407e02230f15ce75817-Paper-Datasets_and_Benchmarks.pdf)',
              '[FactorizePhys Table2](https://arxiv.org/html/2411.01542v1#S4.T2)', '',
              'PHYS: selected48 subjects/101 recordings (T1=42,T2=26,T3=33), FS35; source train38/valid10 subjects.',
              'PURE source split03–10/01–02; 20 epochs, constant loss weights1, train-only flip, clip normalization, stride80 inference. UBFC subject37 uses7 decoded complete clips. PURE target has59 recordings.',
              'Loss, source split, training length, input crop/resize/normalization and overlap inference are explicit method differences. Pinned2026 code is not proof of paper-environment equivalence.',
              'One seed; no seed SD. No routing ablation: these numbers do not isolate routing causally.', '', '## Source-only variant selection', '']
    for source in ('PURE','UBFC-PHYS'):
        p = root/(source+'_selection.json')
        if p.exists(): lines.append('`'+json.dumps(json.loads(p.read_text()), ensure_ascii=False)+'`')
    lines += ['', '## Failures and ineligible work', '']
    lines.extend(f'- {j["name"]}: {j["status"]}; {j.get("error",j.get("reason",""))}' for j in state['jobs']
                 if j['status'] in ('failed','ineligible','skipped'))
    text = '\n'.join(lines)+'\n'
    for name in ('comparison.md','evening_report.md'): (root/name).write_text(text, encoding='utf-8')
    atomic_json(root/'summary.json', dict(status=state['status'], results=results, jobs=state['jobs'], updated_at=time.time()))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--plan', type=Path, required=True)
    args=ap.parse_args(); plan=json.loads(args.plan.read_text())
    root=Path(plan['output']); path=root/'status.json'
    state=json.loads(path.read_text()); state.update(status='running', phase='running', pid=os.getpid(), started_at=time.time())
    env=dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
             CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    def publish():
        state['heartbeat_at']=time.time(); atomic_json(path,state)
    def select_source(source):
        candidates=[]
        for j in state['jobs']:
            if j['source']==source and j['kind']=='train' and j['status']=='complete':
                s=json.loads((root/j['name']/'selection.json').read_text())
                candidates.append(dict(variant=j['variant'], **s))
        if not candidates: raise ValueError('No eligible trained source candidate')
        best=min(candidates,key=lambda d:(d['best_rmse'], d['variant']!='direct'))
        atomic_json(root/(source+'_selection.json'), dict(source=source, candidates=candidates, selected=best,
                    target_used_for_selection=False, selected_at=time.time()))
    proc=None
    try:
        for job in state['jobs']:
            if job['status'] in ('complete','ineligible','skipped'): continue
            if job['kind']=='test':
                dependency=next(j for j in state['jobs'] if j['name']==job['training_job'])
                if dependency['status']!='complete':
                    job.update(status='skipped', reason='Source training '+dependency['status'], finished_at=time.time()); publish(); continue
                if not (root/(job['source']+'_selection.json')).exists(): select_source(job['source'])
            request=json.loads(Path(job['request']).read_text())
            command=[sys.executable,'-B','-u',plan['worker'],'--request',job['request']]
            stream=(root/(job['name']+'.stdout.log')).open('a',encoding='utf-8')
            proc=subprocess.Popen(command,cwd=plan['snapshot'],env=env,stdout=stream,stderr=subprocess.STDOUT,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
            job.update(status='running',pid=proc.pid,started_at=time.time()); publish(); render(root,state)
            print(f'Started {job["name"]} pid={proc.pid}',flush=True)
            while proc.poll() is None:
                time.sleep(5)
                # A transient reporting failure must not orphan a running child.
                try: publish()
                except OSError as e: print(f'Status publish delayed: {e}',flush=True)
            code=proc.returncode; proc=None; stream.close()
            summary_path=root/job['name']/'summary.json'
            if code or not summary_path.exists():
                job.update(status='failed',exit_code=code,error='See job stdout/progress',finished_at=time.time())
                publish(); render(root,state); continue
            d=json.loads(summary_path.read_text(encoding='utf-8'))
            job.update(status='ineligible' if d.get('status')=='ineligible' else 'complete',finished_at=d.get('finished_at',time.time()))
            if job['status']=='ineligible': job['reason']=d['reason']
            if job['kind']=='train' and job['status']=='complete':
                modeldir=Path(plan['models'])/job['name']; modeldir.mkdir(parents=True,exist_ok=True)
                for name in ('best.pt','bundle.json','config.json','split.json','selection.json'):
                    shutil.copy2(root/job['name']/name,modeldir/name)
                for target in plan['targets'][job['source']]:
                    req=dict(request,mode='test',bundle=str(modeldir),target=target,output='CHOOSE_NEW_OUTPUT')
                    atomic_json(modeldir/(target+'_test_request.json'),req)
                job['bundle']=str(modeldir)
            publish();render(root,state)
        state.update(status='failed' if any(j['status']=='failed' for j in state['jobs']) else 'complete',finished_at=time.time())
        publish();render(root,state)
    except BaseException as error:
        # Never declare a terminal queue while its child is still running.
        if proc is not None:
            print('Controller error; awaiting current child exit before marking failure',flush=True)
            proc.wait()
        state.update(status='failed',error=f'{type(error).__name__}: {error}',finished_at=time.time())
        publish();render(root,state)
        raise


if __name__=='__main__': main()
