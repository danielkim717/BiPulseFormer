"""Materialize exact scientific files in a NEW directory; never launches training.

Generated manifests describe this public subset, not a historical manifest.
Paths/receipts/canonical cache must be prepared on the recipient's machine.
"""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--experiment',choices=['pure_intra','ubfc_intra_posthoc','cohface_intra','ubfc_to_pure'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
out=a.output.resolve()
if out.exists():raise FileExistsError('Use a new output directory; existing results are protected')
src=ROOT/'runs'/a.experiment;out.mkdir(parents=True);snap=out/'source_snapshot'
shutil.copytree(ROOT/'implementation',snap)
if (src/'overrides').exists():shutil.copytree(src/'overrides',snap,dirs_exist_ok=True)
if (src/'controls').exists():shutil.copytree(src/'controls',out,dirs_exist_ok=True)
for name in ['protocol.executed.json','inventory.template.json']:
 if (src/name).exists():shutil.copy2(src/name,out/name)
if (src/'request_templates').exists():shutil.copytree(src/'request_templates',out/'request_templates')
manifest={p.relative_to(snap).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(snap.rglob('*')) if p.is_file()}
(out/'snapshot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
(out/'PREPARATION_REQUIRED.txt').write_text('This is a source export, not a ready/running experiment. Configure raw-data roots and request paths; regenerate canonical labels and input caches; run actual parity/source-target/GPU checks. Never create a passed receipt manually. See the release TRAINING.md.\n',encoding='utf-8')
print('Materialized',a.experiment,'at',out,'; training not started')
