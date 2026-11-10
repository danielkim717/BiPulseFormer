"""COHFACE adapter: native AVI -> static legacy crop -> physical 30-Hz inputs.

Raw files are read only. Native resized RGB is cached losslessly as float32.
No test metric is computed during preparation.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch
import cv2
import h5py
from torch.utils.data import Dataset
from status_json_io import read, write

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def helper(snapshot):
    spec=importlib.util.spec_from_file_location('cohface_legacy_input',Path(snapshot)/'legacy/rppg_dataset.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Helper(module.RPPGDataset):
        def _prepare_data(self): pass
    h=Helper('COHFACE','',face_crop=True,data_type='diff_normalized')
    h._haar=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    return h

def prepare_record(row,root,snapshot):
    root=Path(root);cache=root/'input_cache';cache.mkdir(exist_ok=True)
    receipt=cache/(row['id']+'.json')
    if receipt.exists():
        m=read(receipt)
        assert m['raw_video_sha256']==row['video_sha256'] and m['raw_label_sha256']==row['label_sha256']
        for name,digest in m['file_hashes'].items(): assert sha(name)==digest,name
        return m
    assert sha(row['video'])==row['video_sha256'] and sha(row['label'])==row['label_sha256']
    h=helper(snapshot);cap=cv2.VideoCapture(row['video'],cv2.CAP_FFMPEG)
    ok,frame=cap.read();assert ok,row['id']
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);box,found=h._detect_face_box_raw(rgb)
    h._face_box_cache[row['id']]=box
    dest=cache/(row['id']+'.rgb.npy');tmp=cache/(row['id']+'.rgb.partial.npy')
    assert not dest.exists() and not tmp.exists(),'Preserve partial attempts; use a new path for recovery'
    arr=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(row['declared_frames'],3,128,128))
    count=0
    try:
        while ok:
            assert count<row['declared_frames'],'Decoded more frames than declared'
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            arr[count]=h.transform(h._crop_face(rgb,row['id'],count)).numpy()
            count+=1;ok,frame=cap.read()
    finally:cap.release()
    assert count==row['declared_frames'],(row['id'],count,row['declared_frames'])
    arr.flush();del arr;tmp.rename(dest)
    with h5py.File(row['label'],'r') as f:
        pulse=f['pulse'][:];clock=f['time'][:]
    last=min((count-1)/row['fps'],float(clock[-1]))
    length30=int(np.floor(last*30+1e-8))+1
    times=np.arange(length30,dtype=np.float64)/30
    aligned=np.interp(times,clock,pulse)
    assert np.isfinite(aligned).all() and aligned.std()>1e-9
    labels=cache/(row['id']+'.bvp30.npy');assert not labels.exists();np.save(labels,aligned)
    usable=((length30-1)//160)*160
    assert usable>=160 and usable+1<=length30
    meta=dict(row,raw_video_sha256=row['video_sha256'],raw_label_sha256=row['label_sha256'],
        cache_rgb=str(dest),cache_label=str(labels),decoded_frames=count,frame_shape=[3,128,128],
        crop_box=list(box),face_detected=bool(found),samples_30hz=length30,usable_frames=usable,
        last_evaluated_time=usable/30,first_label_endpoint_hold_seconds=float(clock[0]),
        file_hashes={str(dest):sha(dest),str(labels):sha(labels)},prepared_at=time.time())
    write(receipt,meta);return meta

def get_frames(record,start,count=161):
    assert 0<=start and start+count<=record['samples_30hz']
    cache=np.load(record['cache_rgb'],mmap_mode='r')
    pos=np.arange(start,start+count,dtype=np.float64)*record['fps']/30
    lo=np.floor(pos+1e-9).astype(np.int64);hi=np.minimum(lo+1,len(cache)-1)
    alpha=(pos-lo).astype(np.float32)[:,None,None,None]
    values=np.asarray(cache[lo])*(1-alpha)+np.asarray(cache[hi])*alpha
    assert values.shape==(count,3,128,128) and np.isfinite(values).all()
    return torch.from_numpy(values).permute(1,0,2,3).contiguous()

def normalize(frames):
    diff=(frames[:,1:]-frames[:,:-1])/(frames[:,1:]+frames[:,:-1]+1e-7)
    return torch.nan_to_num(diff/(diff.std()+1e-7),nan=0.)

def derivative(y):
    values=torch.tensor(np.asarray(y).copy(),dtype=torch.float32).diff()
    assert torch.isfinite(values).all() and values.std()>1e-9
    return values/(values.std()+1e-7)

def windows(length):
    assert length>=160 and length%160==0
    return list(range(0,length-159,80))

def merge(waves,starts,length):
    assert starts==windows(length) and len(waves)==len(starts)
    total=np.zeros(length,dtype=np.float64);weight=np.zeros(length,dtype=np.int32)
    for p,s in zip(waves,starts):
        assert np.asarray(p).shape==(160,) and np.isfinite(p).all()
        total[s:s+160]+=p;weight[s:s+160]+=1
    assert (weight>0).all()
    return total/weight

class TrainDataset(Dataset):
    def __init__(self,records,trainer):
        self.samples=[];self.labels=[];self.hrs=[]
        for r in records:
            wave=np.load(r['cache_label'])
            for s in range(0,r['samples_30hz']-160,160):
                y=derivative(wave[s:s+161])
                hr=float(trainer.get_hr(None,y.numpy(),sr=30,min=40,max=180))
                assert np.isfinite(hr) and 40<hr<180 and 0<=int(hr-40)<140,(r['id'],s,hr)
                self.samples.append((r,s));self.labels.append(y);self.hrs.append(hr)
    def __len__(self):return len(self.samples)
    def __getitem__(self,i):
        r,s=self.samples[i];frames=get_frames(r,s)
        if torch.rand(1).item()<.5:frames=frames.flip(-1)
        return normalize(frames),self.labels[i],self.hrs[i]
