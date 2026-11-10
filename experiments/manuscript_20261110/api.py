"""Small public API using the unchanged manuscript baseline and HR scorer.

Input must already have dataset-specific cropping, resize and time alignment.
No implicit face detector or resize occurs in this API.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'implementation'))

def model(checkpoint=None,device='cpu'):
 import torch
 from src.models.bipulseformer import BiPulseFormer
 net=BiPulseFormer(image_size=(160,128,128),patches=(4,4,4),dim=96,ff_dim=144,num_heads=4,num_layers=12,dropout_rate=.1,theta=.7,n_win=(1,4,4),topk=4,routing_mode='fft_magnitude',diff_routing=True,routing_tau=.5,routing_band=(40/60,3.))
 for layer in net.modules():
  if hasattr(layer,'routing_band') and hasattr(layer,'fps'):layer.fps=7.5
 if checkpoint:
  state=torch.load(checkpoint,map_location='cpu',weights_only=True)
  net.load_state_dict(state['model'] if 'model' in state else state,strict=True)
 return net.to(device)

def normalize_frames(frames):
 """Float RGB [B,3,161,128,128] -> baseline differential [B,3,160,128,128]."""
 import torch
 if frames.ndim!=5 or tuple(frames.shape[1:])!=(3,161,128,128):raise ValueError('Expected B,3,161,128,128 already processed RGB')
 if not frames.is_floating_point() or not torch.isfinite(frames).all():raise ValueError('Expected finite float RGB')
 d=(frames[:,:,1:]-frames[:,:,:-1])/(frames[:,:,1:]+frames[:,:,:-1]+1e-7)
 return torch.stack([torch.nan_to_num(x/(x.std()+1e-7),nan=0.) for x in d])

def score_record(predicted_derivative,reference_derivative,fs=30):
 """Whole covered-record derivative waves; do not pass a mean of clip HRs."""
 import numpy as np
 from src import journal_toolbox as jt
 from src.hr_evaluation_40_180 import calculate_metric_per_video
 jt.verify_vendor();jt.activate_adapters()
 p,g=np.asarray(predicted_derivative),np.asarray(reference_derivative)
 if p.ndim!=1 or g.shape!=p.shape or len(p)<160 or not np.isfinite(p).all() or not np.isfinite(g).all():raise ValueError('Aligned finite 1-D covered-record signals required')
 gt,hr,_,_=calculate_metric_per_video(p,g,fs=fs,diff_flag=True)
 return {'pred_hr':float(hr),'gt_hr':float(gt),'absolute_error_bpm':float(abs(hr-gt)),'hr_bpm':[40,180]}

if __name__=='__main__':
 import argparse,json
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--waves',type=Path);a=p.parse_args()
 if a.smoke:
  import torch
  torch.set_num_threads(2);torch.manual_seed(43)
  net=model().eval();layers=[x for x in net.modules() if hasattr(x,'topk')]
  assert len(layers)==12 and all(x.topk==4 and x.n_win==(1,4,4) for x in layers)
  assert sum(p.numel() for p in net.parameters())==7380865
  with torch.inference_mode():y=net(normalize_frames(torch.rand(1,3,161,128,128)),2.)[0]
  assert y.shape==(1,160) and torch.isfinite(y).all()
  print('PASS: CPU baseline K4, 12 layers, 7380865 parameters, finite [1,160] output')
 elif a.waves:
  import numpy as np
  x=np.load(a.waves,allow_pickle=False);print(json.dumps(score_record(x['prediction'],x['reference']),indent=2))
 else:p.print_help()
