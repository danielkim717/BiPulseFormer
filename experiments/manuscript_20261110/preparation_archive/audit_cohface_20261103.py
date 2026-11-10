"""Read-only COHFACE inventory audit. Writes evidence only to the run directory."""
import json
from pathlib import Path
import hashlib
import time
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'results/cohface_intra_20261103/deps'))
import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('D:/cohface')
OUT = ROOT / 'results/cohface_intra_20261103'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def scalar(v):
    if isinstance(v,np.ndarray): return scalar(v.item()) if v.size==1 else [scalar(x) for x in v.tolist()]
    if isinstance(v,np.generic): return scalar(v.item())
    return v.decode() if isinstance(v,bytes) else v

def main():
    OUT.mkdir(exist_ok=True)
    splits = {}
    for role in ('train', 'dev', 'test'):
        p = DATA / 'protocols/all' / (role + '.txt')
        ids = p.read_text().splitlines()
        assert len(ids) == len(set(ids))
        splits[role] = dict(records=ids, subjects=sorted({int(x.split('/')[0]) for x in ids}), sha256=sha(p))
    for a,b in (('train','dev'),('train','test'),('dev','test')):
        assert not set(splits[a]['subjects']) & set(splits[b]['subjects'])
    assert sorted(sum([s['subjects'] for s in splits.values()], [])) == list(range(1,41))
    records=[]
    for role, partition in splits.items():
        for rel in partition['records']:
            base=DATA/rel; video=base.with_suffix('.avi'); label=base.with_suffix('.hdf5')
            assert video.exists() and label.exists(), rel
            cap=cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
            fps=cap.get(cv2.CAP_PROP_FPS); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ok,first=cap.read(); cap.release()
            assert ok and first.shape==(480,640,3) and 19.99<fps<20.01 and count>=1100, rel
            with h5py.File(label,'r') as h:
                y=h['pulse'][:]; t=h['time'][:]
                attrs={k:scalar(v) for k,v in h.attrs.items()
                       if k in ('client-id','session','illumination','sample-rate-hz','scale')}
                attrs={k:v.decode() if isinstance(v,bytes) else v for k,v in attrs.items()}
                assert y.ndim==t.ndim==1 and len(y)==len(t) and np.isfinite(y).all() and np.isfinite(t).all()
                assert np.all(np.diff(t)>0) and np.std(y)>1e-9, rel
                row=dict(id=rel.replace('/data','').replace('/','_'), subject=int(rel.split('/')[0]),
                    session=int(rel.split('/')[1]), official_role=role, video=str(video), label=str(label),
                    video_sha256=sha(video), label_sha256=sha(label), fps=fps, declared_frames=count,
                    video_last_sample_seconds=(count-1)/fps, signal_samples=len(y),
                    signal_time_first=float(t[0]), signal_time_last=float(t[-1]),
                    signal_step_median=float(np.median(np.diff(t))), attrs=attrs)
            records.append(row)
    result=dict(status='metadata_passed',dataset='COHFACE',records=records,official_splits=splits,
                readme_sha256=sha(DATA/'README.rst'),created_at=time.time(),
                note='Only video first frame read; full sequential decoding and model preflight still required.')
    (OUT/'inventory_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(records=len(records),subjects=len({r['subject'] for r in records}),
       splits={k:dict(subjects=len(v['subjects']),records=len(v['records'])) for k,v in splits.items()},
       frames_range=[min(r['declared_frames'] for r in records),max(r['declared_frames'] for r in records)],
       time_first_range=[min(r['signal_time_first'] for r in records),max(r['signal_time_first'] for r in records)],
       time_last_range=[min(r['signal_time_last'] for r in records),max(r['signal_time_last'] for r in records)],
       example=records[0]),indent=2))

if __name__=='__main__': main()
