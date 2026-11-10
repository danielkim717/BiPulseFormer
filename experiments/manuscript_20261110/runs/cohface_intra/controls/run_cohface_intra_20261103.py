"""One isolated baseline COHFACE preparation/train/fixed-final-test pipeline."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'deps'))
sys.path.insert(0,str(HERE/'source_snapshot'))
from status_json_io import read,write
from cohface_intra_data import sha,prepare_record,get_frames,normalize,derivative,windows,merge,TrainDataset,helper
import numpy as np
import torch
from torch.utils.data import DataLoader
from src import journal_toolbox as jt
from src.train import FrequencyLoss,NegPearsonLoss

ROOT=HERE
SNAP=HERE/'source_snapshot'

def log(message,**fields):
    p=read(ROOT/'progress.json') if (ROOT/'progress.json').exists() else {}
    p.update(fields,updated_at=time.time(),message=message)
    write(ROOT/'progress.json',p)
    line=time.strftime('[%Y-%m-%d %H:%M:%S] ')+message
    with (ROOT/'log.txt').open('a',encoding='utf-8') as f:f.write(line+'\n')
    print(line,flush=True)

def stopped():
    if (ROOT/'queue.stop').exists():raise InterruptedError('User stop marker present')

def seed_all():
    random.seed(43);np.random.seed(43);torch.manual_seed(43);torch.cuda.manual_seed_all(43)

def configure():
    torch.set_num_threads(2);jt.cv2.setNumThreads(1)
    jt.verify_vendor();jt.activate_adapters()
    from src.hr_evaluation_40_180 import calculate_metric_per_video
    jt.post.calculate_metric_per_video=calculate_metric_per_video

def verify():
    for name,digest in read(ROOT/'control_manifest.json').items():assert sha(ROOT/name)==digest,name
    for name,digest in read(ROOT/'snapshot_manifest.json').items():assert sha(SNAP/name)==digest,name
    p=read(ROOT/'protocol.json')
    assert p['selection_policy']=='fixed_epoch30' and p['epochs']==30 and p['seed']==43
    assert len(p['train_subjects'])==24 and len(p['test_subjects'])==16
    assert not set(p['train_subjects'])&set(p['test_subjects'])
    assert p['test_hr_bpm']==p['routing_band_bpm']==[40,180]
    return p

def model():
    net=jt.model_factory((160,128,128),(4,4,4),96,144,4,12,.1,.7,topk=4)
    layers=[x for x in net.modules() if hasattr(x,'topk')]
    assert len(layers)==12 and all(x.topk==4 and x.routing_mode=='fft_magnitude' and x.n_win==(1,4,4) for x in layers)
    jt.set_input_fps(net,30)
    assert all(x.fps==7.5 and tuple(x.routing_band)==(40/60,3.) for x in layers)
    assert sum(p.numel() for p in net.parameters())==7380865
    return net

def state_hash(net):
    h=hashlib.sha256()
    for n,t in net.state_dict().items():h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def scorer(pred,y):
    gt,hr,_,_=jt.post.calculate_metric_per_video(pred,y,fs=30,diff_flag=True,hr_method='FFT')
    return float(gt),float(hr)

def preflight():
    p=verify();configure();seed_all();stopped()
    row=read(ROOT/'inventory_audit.json')['records'][0]
    assert row['subject'] in p['train_subjects']
    record=prepare_record(row,ROOT,SNAP)
    data=TrainDataset([record],jt.OfficialTrainer)
    # Exact cached source-time sample and independently composed fractional sample.
    frames=get_frames(record,0)
    cached=np.load(record['cache_rgb'],mmap_mode='r')
    assert torch.equal(frames[:,0],torch.from_numpy(np.array(cached[0])))
    reference=cached[0].astype(np.float64)/3+cached[1].astype(np.float64)*2/3
    interpolation_error=float(np.max(np.abs(frames[:,1].numpy()-reference)))
    assert interpolation_error<=np.finfo(np.float32).eps
    # Compare first cached frame with an independent decode and original transform.
    cap=jt.cv2.VideoCapture(row['video'],jt.cv2.CAP_FFMPEG);ok,raw=cap.read();cap.release();assert ok
    h=helper(SNAP);h._face_box_cache[row['id']]=tuple(record['crop_box'])
    raw=jt.cv2.cvtColor(raw,jt.cv2.COLOR_BGR2RGB)
    assert torch.equal(frames[:,0],h.transform(h._crop_face(raw,row['id'],0)))
    wave=np.sin(2*np.pi*1.2*np.arange(1801)/30)
    derivative_wave=np.diff(wave)
    gt,pr=scorer(derivative_wave,derivative_wave)
    assert abs(gt-72)<1 and gt==pr
    length=record['usable_frames'];starts=windows(length)
    assert np.array_equal(merge([np.ones(160) for _ in starts],starts,length),np.ones(length))
    seed_all();net=model().cuda();initial=state_hash(net);net.train()
    batch=[data[i] for i in range(4)]
    x=torch.stack([b[0] for b in batch]).cuda();y=torch.stack([b[1] for b in batch]).cuda()
    target=torch.tensor([b[2] for b in batch],dtype=torch.float32,device='cuda')
    pred=net(x,2.)[0];assert pred.shape==(4,160)
    pred=(pred-pred.mean(-1,keepdim=True))/pred.std(-1,keepdim=True).clamp_min(1e-8)
    lp=NegPearsonLoss()(pred,y);ce,kl=FrequencyLoss(fps=30,diff_flag=False).cuda()(pred,target)
    loss=lp+ce+kl;assert torch.isfinite(loss);loss.backward()
    grads=[v.grad for v in net.parameters() if v.grad is not None]
    assert grads and all(torch.isfinite(v).all() for v in grads) and any((v!=0).any() for v in grads)
    result=dict(status='passed',protocol_sha256=sha(ROOT/'protocol.json'),plan_sha256=sha(ROOT/'plan.json'),
        initial_model_state_sha256=initial,model_file_unchanged=True,parameters=7380865,
        raw_cache_input_parity=True,physical_time_interpolation=True,recording_merge_coverage=True,
        interpolation_max_absolute_error=interpolation_error,
        environment=dict(torch=torch.__version__,numpy=np.__version__,opencv=jt.cv2.__version__,cuda=torch.version.cuda),
        synthetic_72bpm_scorer=dict(gt=gt,pred=pr),batch4_forward_backward=True,
        loss_components=dict(pearson=float(lp.detach()),ce=float(ce.detach()),kl=float(kl.detach())),
        inspected_training_record=row['id'],full_inventory_preparation_required_before_train=True,finished_at=time.time())
    write(ROOT/'validation_checks.json',result)
    write(ROOT/'status.json',dict(status='prepared',phase='ready_for_data_preparation',automatic_reporting=False))
    print(json.dumps(result,indent=2),flush=True)

def evaluate(net,records):
    net.eval();names=[];pred_hr=[];gt_hr=[];saved={};coverage=[]
    for i,r in enumerate(records,1):
        stopped();L=r['usable_frames'];starts=windows(L);waves=[]
        log('Test recording '+r['id'],phase='test',recording=i,recordings=len(records),batch=0,batches=math.ceil(len(starts)/4))
        with torch.inference_mode():
            for j in range(0,len(starts),4):
                stopped();x=torch.stack([normalize(get_frames(r,s)) for s in starts[j:j+4]]).cuda()
                y=net(x,2.)[0];y=(y-y.mean(-1,keepdim=True))/y.std(-1,keepdim=True).clamp_min(1e-8)
                assert torch.isfinite(y).all();waves.extend(y.cpu().numpy())
                log('Test inference',batch=j//4+1)
        pred=merge(waves,starts,L);label=derivative(np.load(r['cache_label'])[:L+1]).numpy()
        gt,pr=scorer(pred,label);names.append(r['id']);gt_hr.append(gt);pred_hr.append(pr)
        saved[r['id']]=dict(prediction=torch.from_numpy(pred),label=torch.from_numpy(label),subject=r['subject'])
        coverage.append(dict(id=r['id'],subject=r['subject'],illumination=r['attrs']['illumination'],
            usable_frames=L,duration_seconds=L/30,windows=len(starts),crop_box=r['crop_box']))
    result=jt.hr_metrics(pred_hr,gt_hr)
    result.update(recording_ids=names,pred_hrs=pred_hr,gt_hrs=gt_hr,n_recordings=len(names),n_subjects=len({r['subject'] for r in records}))
    bylight={}
    for light in ('lamp','natural'):
        ids=[i for i,r in enumerate(records) if r['attrs']['illumination']==light]
        assert len(ids)==32
        bylight[light]=dict(jt.hr_metrics(np.array(pred_hr)[ids],np.array(gt_hr)[ids]),n_recordings=len(ids))
    torch.save(saved,ROOT/'test_predictions.pt');write(ROOT/'test_coverage.json',coverage)
    return dict(per_recording=result,by_illumination=bylight)

def comparison(result):
    refs=read(ROOT/'literature_comparison.json')
    def fmt(value):return '--' if value is None else f'{value:.2f}'
    m=result['per_recording'];lines=['# COHFACE intra result','',
        'BiPulseFormer: official train+dev24/test16, seed43, K4, fixed epoch30, common HR40-180 BPM.',
        'All model output metrics below are recording-level. Published references are not matched reruns.','',
        '| Method | MAE (BPM) | RMSE (BPM) | MAPE (%) | Pearson | Evidence |',
        '|---|---:|---:|---:|---:|---|',
        '| BiPulseFormer | '+' | '.join(fmt(m[k]) for k in ('MAE_bpm','RMSE_bpm','MAPE_pct','Pearson'))+' | This run;64 recordings |']
    for row in refs['included_hr_references']:
        lines.append(f"| {row['model']} | {row['MAE_bpm']:.2f} | {row['RMSE_bpm']:.2f} | -- | {row['Pearson']:.2f} | [{row['source_label']}]({row['url']}) |")
    lines+=['','Reference protocol differences and exclusions: see literature_comparison.md.',
        'No performance-based split/checkpoint/model changes; no model superiority or routing causality inferred from unmatched literature scores.']
    (ROOT/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def run():
    import msvcrt
    with (ROOT/'pipeline.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'1');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        p=verify();configure();stopped()
        check=read(ROOT/'validation_checks.json');assert check['status']=='passed' and check['protocol_sha256']==sha(ROOT/'protocol.json')
        assert read(ROOT/'status.json')['status']=='prepared','No duplicate launch or implicit retry'
        state=dict(status='running',phase='data_preparation',pid=os.getpid(),started_at=time.time(),automatic_reporting=False)
        write(ROOT/'status.json',state)
        try:
            records=[];audit=read(ROOT/'inventory_audit.json')
            for i,row in enumerate(audit['records'],1):
                stopped();assert shutil.disk_usage(ROOT).free>5*1024**3
                log('Preparing '+row['id'],phase='data_preparation',recording=i,recordings=160)
                records.append(prepare_record(row,ROOT,SNAP))
            write(ROOT/'prepared_inventory.json',records)
            train=[r for r in records if r['subject'] in p['train_subjects']]
            test=[r for r in records if r['subject'] in p['test_subjects']]
            assert len(train)==96 and len(test)==64
            data=TrainDataset(train,jt.OfficialTrainer)
            write(ROOT/'train_samples.json',[dict(recording=r['id'],start=s,hr=h) for (r,s),h in zip(data.samples,data.hrs)])
            write(ROOT/'source_preflight.json',dict(status='passed',records=96,samples=len(data),
                min_hr=min(data.hrs),max_hr=max(data.hrs),invalid_targets=0,full_decode_records=160,
                no_subject_overlap=True,finished_at=time.time()))
            state.update(phase='training',source_preflight_sha256=sha(ROOT/'source_preflight.json'));write(ROOT/'status.json',state)
            seed_all();net=model().cuda();assert state_hash(net)==check['initial_model_state_sha256']
            loader=DataLoader(data,batch_size=4,shuffle=True,num_workers=0,drop_last=False,generator=torch.Generator().manual_seed(43))
            opt=torch.optim.Adam(net.parameters(),lr=1e-4,weight_decay=5e-5)
            pearson=NegPearsonLoss();frequency=FrequencyLoss(fps=30,diff_flag=False).cuda()
            history=[];(ROOT/'checkpoints').mkdir(exist_ok=False)
            for epoch in range(1,31):
                stopped();started=time.time();net.train();losses=[]
                log('Training epoch started',phase='training',epoch=epoch,epochs=30,batch=0,batches=len(loader))
                for i,(x,y,h) in enumerate(loader,1):
                    stopped();opt.zero_grad(set_to_none=True)
                    pred=net(x.cuda(),2.)[0]
                    pred=(pred-pred.mean(-1,keepdim=True))/pred.std(-1,keepdim=True).clamp_min(1e-8)
                    lp=pearson(pred,y.cuda());ce,kl=frequency(pred,h.float().cuda());loss=lp+ce+kl
                    assert torch.isfinite(loss),'Nonfinite training loss'
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                    assert torch.isfinite(norm),'Nonfinite gradient';opt.step()
                    losses.append([float(v.detach()) for v in (lp,ce,kl)])
                    if i==1 or i%25==0 or i==len(loader):log('Training progress',batch=i,loss=float(loss.detach()))
                path=ROOT/'checkpoints'/f'epoch_{epoch:02d}.pt';assert not path.exists()
                torch.save(dict(model=net.state_dict(),epoch=epoch,protocol_sha256=sha(ROOT/'protocol.json')),path)
                history.append(dict(epoch=epoch,checkpoint=str(path),checkpoint_sha256=sha(path),seconds=time.time()-started,
                    loss_components=dict(zip(('pearson','frequency_ce','distribution_kl'),np.mean(losses,axis=0).tolist()))))
                write(ROOT/'history.json',history)
                torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,
                    python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(),loader_rng=loader.generator.get_state(),
                    protocol_sha256=sha(ROOT/'protocol.json')),ROOT/'last.pt')
                log('Epoch complete',phase='epoch_complete',epoch=epoch,epochs=30)
            selected=ROOT/'checkpoints/epoch_30.pt';shutil.copy2(selected,ROOT/'best.pt')
            write(ROOT/'selection.json',dict(criterion='fixed_epoch30',epoch=30,checkpoint_sha256=sha(selected),test_used=False,validation_used=False))
            write(ROOT/'bundle.json',dict(model='BiPulseFormer',dataset='COHFACE',seed=43,epoch=30,checkpoint_sha256=sha(selected),
                protocol_sha256=sha(ROOT/'protocol.json'),snapshot_manifest_sha256=sha(ROOT/'snapshot_manifest.json')))
            stopped();state.update(phase='test');write(ROOT/'status.json',state)
            net.load_state_dict(torch.load(selected,map_location='cpu',weights_only=True)['model'])
            result=evaluate(net,test)
            summary=dict(status='complete',model='BiPulseFormer',dataset='COHFACE',seed=43,selected_epoch=30,
                checkpoint_sha256=sha(selected),protocol_sha256=sha(ROOT/'protocol.json'),test=result,finished_at=time.time())
            write(ROOT/'summary.json',summary);comparison(result)
            state.update(status='complete',phase='complete',finished_at=time.time());write(ROOT/'status.json',state)
            log('COHFACE train and test complete',phase='complete')
        except BaseException as exc:
            status='held' if isinstance(exc,InterruptedError) else 'failed'
            state.update(status=status,phase=status,error=f'{type(exc).__name__}: {exc}',finished_at=time.time())
            write(ROOT/'status.json',state);log(state['error'],phase=status);traceback.print_exc();raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    if args.preflight:preflight()
    else:run()
