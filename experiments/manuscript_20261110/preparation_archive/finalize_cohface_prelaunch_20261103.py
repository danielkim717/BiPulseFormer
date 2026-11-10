"""Refresh only the unlaunched COHFACE scaffold after implementation fixes."""
import hashlib
from pathlib import Path
import shutil
import time
from status_json_io import read,write
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/cohface_intra_20261103'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert read(OUT/'status.json')['status']=='prepared'
    assert not (OUT/'checkpoints').exists() and not (OUT/'pipeline.lock').exists()
    rev=OUT/'prelaunch_revisions'/str(time.time_ns());rev.mkdir(parents=True)
    for name in ('snapshot_manifest.json','control_manifest.json','run_cohface_intra_20261103.py','cohface_intra_data.py'):
        shutil.copy2(OUT/name,rev/name)
    docs=OUT/'source_snapshot/docs'
    if not docs.exists():shutil.copytree(ROOT/'results/pure_k4_seed43_20261005/source_snapshot/docs',docs,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('run_cohface_intra_20261103.py','cohface_intra_data.py'):
        shutil.copy2(ROOT/'scripts'/name,OUT/name)
    snap=OUT/'source_snapshot'
    write(OUT/'snapshot_manifest.json',{p.relative_to(snap).as_posix():sha(p) for p in snap.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'})
    names=('plan.json','protocol.json','inventory_audit.json','snapshot_manifest.json',
           'run_cohface_intra_20261103.py','cohface_intra_data.py','status_json_io.py','literature_comparison.json','literature_comparison.md')
    write(OUT/'control_manifest.json',{n:sha(OUT/n) for n in names})
    write(rev/'reason.json',dict(reason='Prelaunch packaging/runtime fixes only; original scientific model, loss, input and scorer sources unchanged',time=time.time()))
    print('Unlaunched scaffold verified/refrozen',OUT)
if __name__=='__main__':main()
