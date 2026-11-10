"""Pinned upstream algorithms with explicit model, storage and reporting adapters.

No upstream source is rewritten. Unrelated eager package imports are bypassed.
"""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'third_party/rppg_toolbox_b7500b8'
COMMIT = 'b7500b848f84ad7f86e277b4612563b69f4f88f9'
sys.path.insert(0, str(ROOT / '.journal-deps'))
sys.path.insert(0, str(VENDOR))
for name in ('dataset', 'dataset.data_loader', 'neural_methods', 'neural_methods.model',
             'neural_methods.loss', 'neural_methods.trainer', 'evaluation',
             'unsupervised_methods', 'unsupervised_methods.methods'):
    package = types.ModuleType(name)
    package.__path__ = [str(VENDOR.joinpath(*name.split('.')))]
    sys.modules[name] = package

import cv2
import numpy as np
import torch
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
from torch.utils.data import Dataset
from dataset.data_loader.BaseLoader import BaseLoader
from dataset.data_loader.PURELoader import PURELoader
from dataset.data_loader.UBFCrPPGLoader import UBFCrPPGLoader
from dataset.data_loader.UBFCPHYSLoader import UBFCPHYSLoader
from evaluation import metrics as official_metrics
from evaluation import post_process as post
trainer_module = importlib.import_module('neural_methods.trainer.PhysFormerTrainer')
OfficialTrainer = trainer_module.PhysFormerTrainer
from src.matched_data import atomic_json, digest
from src.models.bipulseformer import BiPulseFormer
from src.phys_selection import validate_inventory

spec = importlib.util.spec_from_file_location('journal_official_config', VENDOR/'config.py')
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)


def verify_vendor():
    manifest = json.loads((VENDOR/'PINNED_MANIFEST.json').read_text())
    if manifest['commit'] != COMMIT:
        raise ValueError('Wrong upstream commit')
    for name, sha in manifest['sha256'].items():
        if digest(VENDOR/name) != sha:
            raise ValueError(f'Upstream source modified: {name}')


def model_factory(image_size, patches, dim, ff_dim, num_heads, num_layers, dropout_rate, theta):
    return BiPulseFormer(image_size=image_size, patches=patches, dim=dim, ff_dim=ff_dim,
                         num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_rate,
                         theta=theta, n_win=(1,4,4), topk=4, routing_mode='fft_magnitude',
                         diff_routing=True, routing_tau=.5, routing_band=(40/60,180/60))


def set_input_fps(model, fps):
    # Architecture's FFT routing uses temporal tokens downsampled by four.
    # The physical band is fixed; this changes only sampling metadata, never weights.
    for layer in model.modules():
        if hasattr(layer,'routing_band') and hasattr(layer,'fps'):
            layer.fps=fps/4


def sparse_detrend(signal, lambda_value):
    """Same upstream linear system; avoids a dense inverse for long videos."""
    y = np.asarray(signal, dtype=np.float64)
    n = len(y)
    d = diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)], [0,1,2], shape=(n-2,n))
    return y - spsolve(eye(n, format='csc') + lambda_value**2*(d.T@d).tocsc(), y)


def activate_adapters():
    # The original Trainer.train/valid/test methods are executed intact.
    trainer_module.ViT_ST_ST_Compact3_TDC_gra_sharp = model_factory
    post._detrend = sparse_detrend


def make_config(source, target, output, roots, checkpoint=None):
    template = ('PURE_PURE_UBFC-PHYS_PHYSFORMER_BASIC .yaml' if target == 'UBFC-PHYS'
                else 'PURE_PURE_UBFC-rPPG_PHYSFORMER_BASIC.yaml')
    path = VENDOR/'configs/train_configs'/template
    cfg = config_module.get_config(types.SimpleNamespace(config_file=str(path)))
    cfg.defrost()
    cfg.TOOLBOX_MODE = 'only_test' if checkpoint else 'train_and_test'
    cfg.MODEL.NAME = 'BiPulseFormer'
    cfg.TRAIN.MODEL_FILE_NAME = f'{source}_{source}_{target}_bipulseformer'
    cfg.MODEL.MODEL_DIR = str(output/'checkpoints')
    cfg.LOG.PATH = str(output/'official_logs')
    cfg.TEST.OUTPUT_SAVE_DIR = str(output/'test_outputs')
    if checkpoint:
        # Upstream report filenames split this path on '/'.
        cfg.INFERENCE.MODEL_PATH = Path(checkpoint).as_posix()
    for role, dataset in [('TRAIN',source),('VALID',source),('TEST',target)]:
        data = getattr(cfg,role).DATA
        data.DATASET = dataset
        data.DATA_PATH = str(roots[dataset])
        data.FS = 35 if dataset == 'UBFC-PHYS' else 30
        data.EXP_DATA_NAME = dataset
        data.DO_PREPROCESS = False  # our streaming cache is built before upstream training
    cfg.freeze()
    return cfg, template


def inventory(dataset, root, role='all'):
    if dataset == 'UBFC-PHYS':
        plan = json.loads(Path(root).read_text(encoding='utf-8'))
        validate_inventory(plan)
        ready = json.loads(Path(plan['preparation_status']).read_text(encoding='utf-8'))
        if ready['status'] != 'complete' or len(plan['records']) != 101:
            raise ValueError('Verified selected PHYS inventory required')
        records = [dict(id=r['video_id'], toolbox_id=r['video_id'], subject=r['video_id'].split('_')[0],
                        video=r['video'], label=r['bvp'], receipt=r) for r in plan['records']]
        if role != 'all':
            from src.protocol import make_split
            split = make_split(sorted({r['subject'] for r in records}), 'cross',
                               dict(split_seed=42, cross_train_fraction=.8))
            records = [r for r in records if r['subject'] in split[role]]
        return records
    cls = PURELoader if dataset == 'PURE' else UBFCrPPGLoader
    obj = object.__new__(cls)
    obj.dataset_name = dataset
    raw = obj.get_raw_data(str(root))
    if role != 'all':
        raw = obj.split_raw_data(raw, *((0,.8) if role == 'train' else (.8,1)))
    rows = []
    for r in raw:
        p = Path(r['path']); name = p.name
        rows.append(dict(id=name, toolbox_id=str(r['index']),
                         subject=name.split('-')[0] if dataset == 'PURE' else name,
                         video=str(p/name) if dataset == 'PURE' else str(p/'vid.avi'),
                         label=str(p/(name+'.json')) if dataset == 'PURE' else str(p/'ground_truth.txt')))
    return rows


def stream_crop(record, dataset):
    """Equivalent static HC crop, decoded one frame at a time to bound raw memory."""
    helper = object.__new__(BaseLoader)
    cap = None
    if dataset == 'PURE':
        frames = (cv2.imread(str(p)) for p in sorted(Path(record['video']).glob('*.png')))
        declared = None; native_fps = 30
    else:
        cap = cv2.VideoCapture(record['video'])
        if not cap.isOpened(): raise ValueError('Cannot open '+record['video'])
        declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        native_fps = float(cap.get(cv2.CAP_PROP_FPS))
        def read():
            while True:
                ok, frame = cap.read()
                if not ok: break
                yield frame
        frames = read()
    result = []; box = None
    try:
        for frame in frames:
            if frame is None: raise ValueError('Image decode failed')
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            if box is None:
                box = np.asarray(helper.face_detection(frame,'HC',True,1.5),dtype=int)
            x,y,w,h = box
            cropped = frame[max(y,0):min(y+h,frame.shape[0]),max(x,0):min(x+w,frame.shape[1])]
            result.append(cv2.resize(cropped,(128,128),interpolation=cv2.INTER_AREA))
    finally:
        if cap is not None: cap.release()
    if len(result)<160: raise ValueError('Fewer than one full clip: '+record['id'])
    return np.asarray(result), dict(decoded_frames=len(result), metadata_frames=declared,
                                  native_fps=native_fps, crop_box=box.tolist())


def prepare(records, dataset, cache, preprocess, log):
    cache = Path(cache)/dataset; cache.mkdir(parents=True,exist_ok=True)
    prepared = []
    cache_settings=preprocess.clone();cache_settings.defrost()
    if not cache_settings.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION:
        # This frequency is unused for static crops (official YAMLs use 30/32).
        cache_settings.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY=0
    cache_settings.freeze()
    for i, record in enumerate(records,1):
        files = sorted(Path(record['video']).glob('*.png')) if dataset=='PURE' else [Path(record['video'])]
        identity = dict(commit=COMMIT,version='journal-official-v1',record=record,
                        input_stats=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in files],
                        label_sha256=digest(record['label']),preprocess=cache_settings.dump())
        import hashlib
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        base=cache/(record['id']+'_'+key[:16]);meta_path=base.with_suffix('.json')
        log(f'preprocessing {dataset} {record["id"]}',phase='preprocessing',recording=i,recordings=len(records))
        if meta_path.exists():
            meta=json.loads(meta_path.read_text())
            if meta['identity'] != json.loads(json.dumps(identity)): raise ValueError('Cache identity mismatch')
            for p,sha in meta['hashes'].items():
                if digest(p)!=sha: raise ValueError('Cache data hash mismatch')
        else:
            if dataset=='UBFC-PHYS':
                receipt=record['receipt'];v=Path(record['video'])
                if (v.stat().st_size!=receipt['video_size'] or v.stat().st_mtime_ns!=receipt['video_mtime_ns']
                        or digest(record['label'])!=receipt['bvp_sha256']):
                    raise ValueError('PHYS data changed after verification')
            frames,coverage=stream_crop(record,dataset)
            cls={'PURE':PURELoader,'UBFC-rPPG':UBFCrPPGLoader,'UBFC-PHYS':UBFCPHYSLoader}[dataset]
            wave=cls.read_wave(record['label'])
            coverage['original_label_samples']=len(wave)
            if dataset in ('PURE','UBFC-PHYS'):
                wave=BaseLoader.resample_ppg(wave,len(frames))
            if len(wave)<len(frames)//160*160: raise ValueError('Labels shorter than complete video chunks')
            settings=preprocess.clone();settings.defrost();settings.CROP_FACE.DO_CROP_FACE=False;settings.freeze()
            # Exact upstream preprocessing after the numerically identical streamed crop.
            data,labels=object.__new__(BaseLoader).preprocess(frames,wave,settings)
            if not np.isfinite(data).all() or not np.isfinite(labels).all(): raise ValueError('Nonfinite cache')
            dp,lp=base.with_suffix('.data.npy'),base.with_suffix('.label.npy')
            np.save(dp,data);np.save(lp,labels)
            coverage.update(full_chunks=len(data),usable_frames=len(data)*160,evaluation_fps=35 if dataset=='UBFC-PHYS' else 30)
            meta=dict(identity=identity,data=str(dp.resolve()),label=str(lp.resolve()),coverage=coverage,
                      hashes={str(p.resolve()):digest(p) for p in (dp,lp)})
            atomic_json(meta_path,meta)
            del frames,data,labels
        prepared.append(dict(record,data=meta['data'],cached_label=meta['label'],coverage=meta['coverage'],cache_key=key))
    return prepared


class CachedClips(Dataset):
    def __init__(self, records):
        self.records=records
        self.arrays={r['id']:(np.load(r['data'],mmap_mode='r'),np.load(r['cached_label'],mmap_mode='r')) for r in records}
        self.samples=[dict(video_id=r['id'],toolbox_id=r['toolbox_id'],subject=r['subject'],chunk=c,start=c*160,end=(c+1)*160)
                      for r in records for c in range(len(self.arrays[r['id']][0]))]
        # BaseLoader.load_preprocessed_data sorts input filenames lexicographically.
        self.samples.sort(key=lambda s:f'{s["toolbox_id"]}_input{s["chunk"]}.npy')
        if not self.samples:raise ValueError('Empty dataset')
    def __len__(self):return len(self.samples)
    def __getitem__(self,index):
        s=self.samples[index];x,y=self.arrays[s['video_id']]
        return (torch.from_numpy(np.array(x[s['chunk']].transpose(3,0,1,2),dtype=np.float32)),
                torch.from_numpy(np.array(y[s['chunk']],dtype=np.float32)),s['toolbox_id'],str(s['chunk']))


def hr_metrics(pred,gt):
    p,g=np.asarray(pred),np.asarray(gt);e=p-g
    if not np.isfinite(p).all() or not np.isfinite(g).all() or np.any(g<=0):raise ValueError('Invalid HR values')
    r=float(np.corrcoef(p,g)[0,1]) if len(p)>1 and p.std()>0 and g.std()>0 else None
    return dict(MAE_bpm=float(np.mean(abs(e))),RMSE_bpm=float(np.sqrt(np.mean(e**2))),
                MAPE_pct=float(np.mean(abs(e)/g)*100),Pearson=r)


def summarize_predictions(predictions, labels, records, fs):
    lookup={r['toolbox_id']:r for r in records};names=[];pr=[];gr=[];snrs=[];pc=[];gc=[];clip_ids=[]
    for name in predictions:
        if set(predictions[name])!=set(labels[name]):raise ValueError('Prediction/label chunk mismatch')
        p=official_metrics._reform_data_from_dict(predictions[name])
        g=official_metrics._reform_data_from_dict(labels[name])
        truth,guess,snr,_=post.calculate_metric_per_video(p,g,fs=fs,diff_flag=True,hr_method='FFT')
        names.append(lookup[name]['id']);pr.append(guess);gr.append(truth);snrs.append(snr)
        for k in sorted(predictions[name]):
            truth,guess,_,_=post.calculate_metric_per_video(predictions[name][k].detach().cpu().numpy(),
                labels[name][k].detach().cpu().numpy(),fs=fs,diff_flag=True,hr_method='FFT')
            pc.append(guess);gc.append(truth);clip_ids.append(lookup[name]['id'])
    recording=dict(hr_metrics(pr,gr),recording_ids=names,pred_hrs=pr,gt_hrs=gr,n_recordings=len(names),
                   n_subjects=len({lookup[k]['subject'] for k in predictions}),SNR_db=float(np.mean(snrs)))
    clips={k+'_clip':v for k,v in hr_metrics(pc,gc).items()};clips['n_clips']=len(pc)
    tasks={}
    for task in ('T1','T2','T3'):
        ri=[i for i,n in enumerate(names) if n.endswith('_'+task)]
        ci=[i for i,n in enumerate(clip_ids) if n.endswith('_'+task)]
        if ri:
            tasks[task]=dict(per_recording=hr_metrics(np.array(pr)[ri],np.array(gr)[ri]),
                            per_clip={k+'_clip':v for k,v in hr_metrics(np.array(pc)[ci],np.array(gc)[ci]).items()},
                            n_recordings=len(ri),n_clips=len(ci))
    subjects=[lookup[k]['subject'] for k in predictions]
    groups=[np.flatnonzero(np.array(subjects)==s) for s in sorted(set(subjects))]
    rng=np.random.default_rng(20260911);ae=abs(np.array(pr)-np.array(gr))
    draws=[float(ae[np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))])].mean()) for _ in range(10000)]
    recording['subject_bootstrap_MAE_95pct']=np.quantile(draws,[.025,.975]).tolist()
    return dict(per_recording=recording,per_clip=clips,by_task=tasks)
