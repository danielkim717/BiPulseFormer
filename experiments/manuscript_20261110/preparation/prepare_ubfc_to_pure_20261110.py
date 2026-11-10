"""Audit split/GT/decoded coverage, lossless source inputs and a real GPU batch."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/ubfc_to_pure_20261110'
sys.path.insert(0,str(ROOT/'scripts'))
from status_json_io import read,write


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def main():
    state=read(RUN/'status.json');assert state['status']=='prepared'
    plan=read(RUN/'plan.json');req=read(plan['jobs'][0]['request']);protocol=read(req['protocol'])
    snap=Path(req['snapshot']);sys.path.insert(0,str(snap));sys.path.insert(0,str(snap/'scripts'))
    state.update(phase='input_preflight',preparation_pid=os.getpid(),updated_at=time.time());write(RUN/'status.json',state)
    try:
        from src import journal_toolbox as jt
        from src.train import NegPearsonLoss,FrequencyLoss
        from pure_frame_cache import RawFrameCache
        from pure_m2_hr_evaluation_40_180 import calculate_metric_per_video
        import numpy as np
        import torch
        import cv2
        torch.set_num_threads(2);cv2.setNumThreads(1)
        jt.verify_vendor();jt.activate_adapters();jt.post.calculate_metric_per_video=calculate_metric_per_video
        os.chdir(jt.VENDOR)
        worker=load('cross_worker_audit',plan['worker'])
        legacy=load('cross_legacy_audit',snap/'legacy/rppg_dataset.py')
        for name,h in read(RUN/'snapshot_manifest.json').items():assert sha(snap/name)==h,name
        pre=RUN/'preflight_source_v2'
        assert read(pre/'progress.json')['phase']=='complete'
        assert not read(pre/'loss_target_audit.json')['invalid']
        train=read(pre/'train_inventory.json');valid=read(pre/'valid_inventory.json')
        split=read(pre/'split.json')
        assert len(train)==33 and len(valid)==9 and not set(split['train'])&set(split['valid'])
        roots=read(req['roots_file']);cfg,_=jt.make_config('UBFC-rPPG','PURE',RUN,roots)
        target=worker.canonical_records(jt,'PURE',roots['PURE'],req['cache'],cfg.TEST.DATA.PREPROCESS)
        assert len(target)==59 and len({r['subject'] for r in target})==10
        write(RUN/'target_inventory.json',target)
        # Check canonical target arrays and actual boundary frames without evaluating model predictions.
        target_audit=[]
        for r in target:
            gt=np.load(r['cached_label'],allow_pickle=False)
            assert gt.shape==(r['coverage']['usable_frames']//160,160) and np.isfinite(gt).all()
            paths=sorted(Path(r['video']).glob('*.png'))
            assert len(paths)==r['coverage']['decoded_frames']
            for p in (paths[0],paths[-1]):assert cv2.imread(str(p)) is not None,str(p)
            target_audit.append(dict(recording=r['id'],frames=len(paths),gt_sha256=sha(r['cached_label'])))
        # Full native source decode verifies coverage. Compare seek-based windows at boundaries
        # to sequential pixels so an AVI seek error cannot silently enter the cache.
        decode_audit=[]
        samples=read(pre/'train_samples.json')
        keys=[(s['recording'],s['start']) for s in samples]
        for r in valid:
            keys.extend((r['id'],s) for s in worker.window_starts(r['coverage']['usable_frames']))
        assert len(set(keys))==len(keys)
        rows={r['id']:r for r in train+valid}
        read_frames,boxes=worker.make_reader(legacy,train+valid,'UBFC-rPPG')
        for ri,r in enumerate(train+valid,1):
            starts=[s for rid,s in keys if rid==r['id']]
            probes={0,min(starts),max(starts),min(max(starts)+160,r['coverage']['decoded_frames']-1)}
            cap=cv2.VideoCapture(r['video']);n=0;pixels={}
            while True:
                ok,frame=cap.read()
                if not ok:break
                if n in probes:pixels[n]=frame.copy()
                n+=1
            cap.release()
            assert n==r['coverage']['decoded_frames'],(r['id'],n,r['coverage'])
            for idx,frame in pixels.items():
                cap=cv2.VideoCapture(r['video']);assert cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
                ok,seek=cap.read();cap.release();assert ok and np.array_equal(frame,seek),(r['id'],idx)
            decode_audit.append(dict(recording=r['id'],decoded_frames=n,seek_probes=sorted(pixels)))
            print('SOURCE_DECODE',ri,42,r['id'],n,flush=True)
        # Complete cache, never a performance-dependent sample subset.
        cache=RawFrameCache(RUN/'input_cache',max_bytes=32*1024**3)
        cache_manifest={};parity=[]
        for i,(rid,start) in enumerate(keys):
            r=rows[rid];allow_tail=rid in set(split['valid'])
            before=torch.get_rng_state().clone()
            raw=read_frames(r,start,allow_tail)
            cold=cache.get(i,lambda:raw)
            warm=cache.get(i,lambda:(_ for _ in ()).throw(AssertionError('warm cache miss')))
            assert torch.equal(before,torch.get_rng_state())
            assert torch.equal(raw,cold) and torch.equal(raw,warm) and raw.stride()==warm.stride()
            if i in (0,len(samples)-1,len(keys)-1):
                for seed in (0,1):
                    torch.manual_seed(seed);a=raw.flip(-1) if torch.rand(1).item()<.5 else raw
                    ra=torch.get_rng_state().clone()
                    torch.manual_seed(seed);b=warm.flip(-1) if torch.rand(1).item()<.5 else warm
                    assert torch.equal(worker.normalize_input(a),worker.normalize_input(b))
                    assert torch.equal(ra,torch.get_rng_state())
                parity.append(dict(recording=rid,start=start,input_rng_exact=True))
            path=RUN/'input_cache'/f'{i:06d}.npy';assert path.exists()
            cache_manifest[str(path)]=sha(path)
            if i%25==0 or i+1==len(keys):
                state.update(phase='building_source_cache',completed=i+1,total=len(keys),updated_at=time.time())
                write(RUN/'status.json',state);print('CACHE',i+1,len(keys),flush=True)
        write(RUN/'input_cache/identity.json',dict(status='passed',samples=[dict(recording=r,start=s) for r,s in keys],
            protocol_sha256=sha(RUN/'protocol.json'),raw_float32_exact=True,source_only=True))
        write(RUN/'cache_manifest.json',cache_manifest)
        state.update(phase='gpu_preflight',updated_at=time.time());write(RUN/'status.json',state)
        random.seed(43);np.random.seed(43);torch.manual_seed(43);torch.cuda.manual_seed_all(43)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True,warn_only=True)
        net=jt.model_factory((160,128,128),(4,4,4),96,144,4,12,.1,.7);jt.set_input_fps(net,30)
        routing=[m for m in net.modules() if hasattr(m,'topk')]
        assert len(routing)==12 and all(m.topk==4 and m.n_win==(1,4,4) and
            m.routing_mode=='fft_magnitude' and m.routing_band==(40/60,3.) and m.fps==7.5 for m in routing)
        initial=hashlib.sha256()
        for name,t in net.state_dict().items():initial.update(name.encode());initial.update(t.cpu().contiguous().numpy().tobytes())
        xb=[];yb=[];hb=[]
        for i in (0,len(samples)//3,len(samples)//2,len(samples)-1):
            rid,start=keys[i];r=rows[rid]
            frames=torch.from_numpy(np.load(RUN/'input_cache'/f'{i:06d}.npy')).permute(1,0,2,3)
            y=torch.tensor(jt.UBFCrPPGLoader.read_wave(r['label'])[start:start+161],dtype=torch.float32).diff()
            y=torch.nan_to_num(y/(y.std()+1e-7),nan=0.)
            xb.append(worker.normalize_input(frames));yb.append(y)
            hb.append(jt.OfficialTrainer.get_hr(None,y.numpy(),sr=30,min=40,max=180))
        net.cuda().train();pred=net(torch.stack(xb).cuda(),2.)[0]
        pred=(pred-pred.mean(-1,keepdim=True))/pred.std(-1,keepdim=True).clamp_min(1e-8)
        ce,kl=FrequencyLoss(fps=30,diff_flag=False).cuda()(pred,torch.tensor(hb,dtype=torch.float32).cuda())
        loss=NegPearsonLoss()(pred,torch.stack(yb).cuda())+ce+kl
        assert torch.isfinite(loss);loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        assert all(any(p.grad is not None and p.grad.abs().sum()>0 for p in block.parameters())
                   for m in routing for block in (m.proj_q,m.proj_k,m.proj_v))
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        opt=torch.optim.Adam(net.parameters(),lr=1e-4,weight_decay=5e-5);opt.step();torch.cuda.synchronize()
        # Selection function rejects incomplete or test-derived histories and resolves ties early.
        history=[dict(epoch=i,dataset_role='source_validation',recording_test_rmse=1.) for i in range(1,31)]
        assert worker.choose(history)['epoch']==1
        for bad in (history[:-1],[dict(r,dataset_role='test') for r in history]):
            try:worker.choose(bad)
            except ValueError:pass
            else:raise AssertionError('Invalid selection accepted')
        checks=dict(status='passed',plan_sha256=sha(RUN/'plan.json'),finished_at=time.time(),
            source_train_subjects=33,source_validation_subjects=9,target_subjects=10,target_recordings=59,
            train_samples=len(samples),source_cache_windows=len(keys),cache_bytes=cache.bytes,
            all_source_decode=decode_audit,target_gt=target_audit,cache_parity=parity,
            gpu_batch4=dict(loss=float(loss.detach()),finite_gradients=True,optimizer_step=True,
                            all_12_blocks_K4=True,initial_state_sha256=initial.hexdigest()),
            source_only_selection_verified=True,weights_reused=False)
        write(RUN/'validation_checks.json',checks)
        state.update(phase='ready_for_launch',updated_at=time.time());write(RUN/'status.json',state)
        print('READY_FOR_LAUNCH',flush=True)
    except BaseException as exc:
        state.update(status='failed',phase='preflight_failed',error=f'{type(exc).__name__}: {exc}',updated_at=time.time())
        write(RUN/'status.json',state);raise


if __name__=='__main__':main()
