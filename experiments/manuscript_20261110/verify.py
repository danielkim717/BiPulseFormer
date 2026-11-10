"""Read-only integrity/aggregate checks; no data, GPU, training or network needed."""
from pathlib import Path
import hashlib,json,math
ROOT=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def verify():
 manifest=json.loads((ROOT/'SHA256SUMS.json').read_text(encoding='utf-8'))
 for name,sha in manifest.items():
  p=ROOT/name
  if digest(p)!=sha:raise ValueError('Changed release file: '+name)
 data=json.loads((ROOT/'results_recording.json').read_text(encoding='utf-8'))
 for name,exp in data['experiments'].items():
  unit=exp['units']['per_recording'];rows=unit['per_unit_HR'];m=unit['metrics'];n=len(rows)
  pred=[r['pred_hr'] for r in rows];gt=[r['gt_hr'] for r in rows];error=[a-b for a,b in zip(pred,gt)]
  pm=sum(pred)/n;gm=sum(gt)/n
  r=sum((a-pm)*(b-gm) for a,b in zip(pred,gt))/math.sqrt(sum((a-pm)**2 for a in pred)*sum((b-gm)**2 for b in gt))
  actual={'MAE_bpm':sum(abs(e) for e in error)/n,'RMSE_bpm':math.sqrt(sum(e*e for e in error)/n),'MAPE_pct':sum(abs(e)/g for e,g in zip(error,gt))/n*100,'Pearson':r}
  for k,v in actual.items():assert math.isclose(v,m[k],abs_tol=1e-10), (name,k,v,m[k])
  assert n==m['n_units'] and len({str(r['subject']) for r in rows})==m['n_subjects']
  print(name, 'records=',n,'MAE=',round(actual['MAE_bpm'],6),'RMSE=',round(actual['RMSE_bpm'],6))
 splits=json.loads((ROOT/'splits.json').read_text(encoding='utf-8'))
 for name,split in splits['splits'].items():
  split=split.get('subjects_and_recordings',split);sets=[set(map(str,split.get(k,[]))) for k in ['train','valid','test']]
  assert not any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)),name
 print('PASS:',len(manifest),'file hashes, 4 result aggregates, subject-disjoint splits')
if __name__=='__main__':verify()
