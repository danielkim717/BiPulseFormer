"""Fixed 2026-09-14 source protocol; immutable common test labels and coverage."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

from run_source_optimized import atomic_json


def choose(rows):
    import math
    if [r['epoch'] for r in rows] != list(range(1, 21)):
        raise ValueError('All twenty source checkpoints are required')
    if any(r['dataset_role'] != 'source_validation' or not math.isfinite(r['recording_test_rmse']) for r in rows):
        raise ValueError('Invalid source-only selection history')
    return min(rows, key=lambda r: (r['recording_test_rmse'], r['epoch']))


def window_starts(length):
    if length < 160 or length % 160:
        raise ValueError('Common coverage must consist of complete 160-frame chunks')
    return list(range(0, length - 159, 80))


def merge_windows(windows, starts, length):
    import numpy as np
    if starts != window_starts(length) or len(windows) != len(starts):
        raise ValueError('Missing, duplicated or changed inference windows')
    total = np.zeros(length, dtype=np.float64)
    weight = np.zeros(length, dtype=np.int32)
    for wave, start in zip(windows, starts):
        if np.asarray(wave).shape != (160,) or not np.isfinite(wave).all():
            raise ValueError('Invalid prediction')
        total[start:start+160] += wave
        weight[start:start+160] += 1
    if not (weight > 0).all(): raise ValueError('Uncovered common evaluation sample')
    return total / weight


def canonical_records(jt, dataset, root, cache, preprocess, role='all'):
    """Reuse verified GT only. Image-cache contents are neither read nor modified."""
    records = jt.inventory(dataset, root, role if dataset == 'UBFC-PHYS' else 'all')
    if dataset == 'PURE' and role != 'all':
        ids = {f'{i:02d}' for i in (range(3, 11) if role == 'train' else range(1, 3))}
        records = [r for r in records if r['subject'] in ids]
    settings = preprocess.clone(); settings.defrost()
    settings.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY = 0; settings.freeze()
    prepared = []
    for r in records:
        files = sorted(Path(r['video']).glob('*.png')) if dataset == 'PURE' else [Path(r['video'])]
        identity = dict(commit=jt.COMMIT, version='journal-official-v1', record=r,
                        input_stats=[(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files],
                        label_sha256=jt.digest(r['label']), preprocess=settings.dump())
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = Path(cache)/dataset/(r['id']+'_'+key[:16]+'.json')
        meta = json.loads(path.read_text(encoding='utf-8'))
        if meta['identity'] != json.loads(json.dumps(identity)):
            raise ValueError('Common evaluation identity changed: '+r['id'])
        label = meta['label']
        if jt.digest(label) != meta['hashes'][label]:
            raise ValueError('Common ground truth changed: '+r['id'])
        prepared.append(dict(r, cached_label=label, coverage=meta['coverage'],
                             common_label_sha256=meta['hashes'][label], cache_key=key))
    return prepared


def make_reader(legacy, records, dataset):
    """Decode native frames; v1 HC/float tensor resize, with bounded one-clip memory."""
    import cv2
    import numpy as np
    import torch
    class Helper(legacy.RPPGDataset):
        def _prepare_data(self): pass
    helper = Helper(dataset, '', face_crop=True, data_type='diff_normalized')
    helper._haar = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    paths = {r['id']: sorted(Path(r['video']).glob('*.png')) for r in records} if dataset == 'PURE' else {}
    for r in records:
        if dataset == 'PURE': frame = cv2.imread(str(paths[r['id']][0]))
        else:
            cap = cv2.VideoCapture(r['video']); ok, frame = cap.read(); cap.release()
            if not ok: raise ValueError('First frame decode failed: '+r['id'])
        if frame is None: raise ValueError('First frame missing: '+r['id'])
        box, _ = helper._detect_face_box_raw(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        helper._face_box_cache[r['id']] = box
    helper._haar = None
    def read(record, start, allow_tail=False):
        frames = []; cap = None
        available = record['coverage']['decoded_frames']
        if start + 161 > available and not (allow_tail and start + 160 == available):
            raise ValueError('Window outside verified decoded coverage')
        if dataset != 'PURE':
            cap = cv2.VideoCapture(record['video'])
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
                cap.release(); raise ValueError('Cannot seek native video')
        try:
            for index in range(start, min(start+161, available)):
                if dataset == 'PURE': frame = cv2.imread(str(paths[record['id']][index]))
                else:
                    ok, frame = cap.read()
                    if not ok: raise ValueError('Incomplete native clip: '+record['id'])
                if frame is None: raise ValueError('Frame decode failed')
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(helper.transform(helper._crop_face(frame, record['id'], index)))
        finally:
            if cap is not None: cap.release()
        if len(frames) == 160 and allow_tail: frames.append(frames[-1].clone())
        if len(frames) != 161: raise ValueError('Incomplete input window')
        return torch.stack(frames).permute(1, 0, 2, 3)
    return read, helper._face_box_cache


def normalize_input(frames):
    import torch
    difference = (frames[:, 1:]-frames[:, :-1])/(frames[:, 1:]+frames[:, :-1]+1e-7)
    return torch.nan_to_num(difference/(difference.std()+1e-7), nan=0.)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--request', type=Path, required=True)
    ap.add_argument('--output', type=Path); ap.add_argument('--preflight-only', action='store_true')
    args = ap.parse_args(); req = json.loads(args.request.read_text(encoding='utf-8'))
    if args.output: req['output'] = str(args.output.resolve())
    out = Path(req['output']); out.mkdir(parents=True, exist_ok=False)
    snap = Path(req['snapshot']); sys.path.insert(0, str(snap))
    manifest_path = snap.parent/'snapshot_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name, sha in manifest.items():
        if name.split('/')[0] in ('src','scripts','legacy','configs'):
            if hashlib.sha256((snap/name).read_bytes()).hexdigest() != sha:
                raise ValueError('Frozen implementation changed: '+name)
    from src import journal_toolbox as jt
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from src.train import FrequencyLoss, NegPearsonLoss, estimate_hr_targets
    spec = importlib.util.spec_from_file_location('final_legacy_dataset', snap/'legacy/rppg_dataset.py')
    legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
    roots = json.loads(Path(req['roots_file']).read_text())
    protocol = json.loads(Path(req['protocol']).read_text())
    if jt.digest(req['protocol']) != req['protocol_sha256']: raise ValueError('Frozen protocol changed')
    jt.verify_vendor(); jt.activate_adapters(); os.chdir(jt.VENDOR)
    torch.set_num_threads(2); jt.cv2.setNumThreads(1)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    source = req['source']; target = req.get('target', 'UBFC-rPPG'); phase = {}
    def log(message, **fields):
        if fields.get('phase') and fields['phase'] != phase.get('phase'): phase.clear()
        phase.update(fields, updated_at=time.time()); atomic_json(out/'progress.json', phase)
        line = time.strftime('[%Y-%m-%d %H:%M:%S] ')+message
        with (out/'log.txt').open('a', encoding='utf-8') as stream: stream.write(line+'\n')
        print(line, flush=True)
    cfg, _ = jt.make_config(source, target, out, roots)
    atomic_json(out/'config.json', dict(protocol, source=source, target=target, variant=req['variant'],
                                      protocol_sha256=req['protocol_sha256']))
    def records(dataset, role='all'):
        return canonical_records(jt, dataset, roots[dataset], req['cache'], cfg.TEST.DATA.PREPROCESS, role)
    def model():
        net = jt.model_factory((160,128,128), (4,4,4), 96, 144, 4, 12, .1, .7)
        jt.set_input_fps(net, 35 if source == 'UBFC-PHYS' else 30)
        return net.cuda()
    def score(net, rows, dataset, epoch, save=False):
        read, boxes = make_reader(legacy, rows, dataset)
        net.eval(); jt.set_input_fps(net, 35 if dataset == 'UBFC-PHYS' else 30)
        predictions = {}; labels = {}; provenance = []
        for ri, r in enumerate(rows, 1):
            gt = np.load(r['cached_label'])
            length = r['coverage']['usable_frames']
            if gt.shape != (length//160,160): raise ValueError('Canonical GT coverage mismatch')
            starts = window_starts(length); waves = []
            log('Recording inference '+r['id'], phase='test' if save else 'source_validation',
                checkpoint_epoch=epoch, checkpoints=20, recording=ri, recordings=len(rows), batch=0,
                batches=(len(starts)+3)//4)
            with torch.inference_mode():
                for offset in range(0, len(starts), 4):
                    batch = torch.stack([normalize_input(read(r, s, True)) for s in starts[offset:offset+4]])
                    pred = net(batch.cuda(), 2.0)[0]
                    pred = (pred-pred.mean(-1, keepdim=True))/pred.std(-1, keepdim=True).clamp_min(1e-8)
                    waves.extend(pred.cpu().numpy())
                    if offset == 0 or offset//4 % 25 == 0 or offset+4 >= len(starts):
                        log('Inference progress', batch=offset//4+1)
            merged = merge_windows(waves, starts, length)
            predictions[r['toolbox_id']] = {k:torch.tensor(p) for k,p in enumerate(merged.reshape(-1,160))}
            # Match CachedClips' float32 conversion before the common scorer,
            # including when the stored preprocessing label array is float64.
            labels[r['toolbox_id']] = {k:torch.tensor(y.copy(),dtype=torch.float32) for k,y in enumerate(gt)}
            provenance.append(dict(recording=r['id'], start=0, end=length, windows=len(starts),
                                   tail_input_repeat=length == r['coverage']['decoded_frames'],
                                   gt_sha256=r['common_label_sha256'], crop_box=boxes[r['id']]))
        result = jt.summarize_predictions(predictions, labels, rows, 35 if dataset == 'UBFC-PHYS' else 30)
        if save:
            torch.save(dict(predictions=predictions, labels=labels), out/'test_predictions.pt')
            atomic_json(out/'test_inventory.json', rows); atomic_json(out/'coverage.json', provenance)
        jt.set_input_fps(net, 35 if source == 'UBFC-PHYS' else 30)
        return result
    try:
        if req['mode'] == 'test':
            bundle = json.loads((Path(req['bundle'])/'bundle.json').read_text())
            ck = Path(req['bundle'])/'best.pt'
            if (bundle['protocol_sha256'] != req['protocol_sha256'] or bundle['source'] != source
                    or jt.digest(ck) != bundle['checkpoint_sha256']
                    or bundle['snapshot_manifest_sha256'] != jt.digest(manifest_path)):
                raise ValueError('Frozen model bundle mismatch')
            net = model(); net.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True)['model'])
            result = score(net, records(target), target, bundle['best_epoch'], True)
            atomic_json(out/'summary.json', dict(status='complete', model='bipulseformer', source=source,
                        target=target, seed=42, variant=req['variant'], best_epoch=bundle['best_epoch'],
                        test=result, checkpoint_sha256=bundle['checkpoint_sha256'], protocol=protocol,
                        protocol_sha256=req['protocol_sha256'], finished_at=time.time()))
            log('Final test complete', phase='complete'); return
        train_rows = records(source, 'train'); valid_rows = records(source, 'valid')
        split = dict(source=source, train=sorted({r['subject'] for r in train_rows}),
                     valid=sorted({r['subject'] for r in valid_rows}),
                     train_records=[r['id'] for r in train_rows], valid_records=[r['id'] for r in valid_rows])
        if set(split['train']) & set(split['valid']): raise ValueError('Subject leakage')
        atomic_json(out/'split.json', split); atomic_json(out/'valid_inventory.json', valid_rows)
        if source == 'PURE':
            ds = legacy.RPPGDataset('PURE', roots[source], face_crop=True, subjects_filter=split['train'],
                     data_type='diff_normalized', random_hflip=True, chunk_step=160, fps=30)
            raw_labels = [s['bvp'] for s in ds.samples]
            sample_ids = [dict(recording=s['video_id'], start=s['first_frame_idx']) for s in ds.samples]
            if {s['video_id'] for s in ds.samples} != set(split['train_records']): raise ValueError('PURE inventory mismatch')
        else:
            read, _ = make_reader(legacy, train_rows, source)
            samples = []; raw_labels = []; sample_ids = []
            for r in train_rows:
                wave = jt.UBFCPHYSLoader.read_wave(r['label'])
                n = r['coverage']['decoded_frames']
                aligned = np.interp(np.linspace(0,1,n), np.linspace(0,1,len(wave)), wave)
                for start in range(0, n-160, 160):
                    samples.append((r,start)); raw_labels.append(aligned[start:start+161])
                    sample_ids.append(dict(recording=r['id'], start=start))
            class NativeDataset(Dataset):
                def __len__(self): return len(samples)
                def __getitem__(self, index):
                    r,s = samples[index]; frames = read(r,s)
                    if torch.rand(1).item() < .5: frames = frames.flip(-1)
                    return normalize_input(frames), diff_labels[index]
            ds = NativeDataset()
        diff_labels = []
        for y in raw_labels:
            y = torch.tensor(np.asarray(y), dtype=torch.float32).diff()
            diff_labels.append(torch.nan_to_num(y/(y.std()+1e-7), nan=0.))
        yall = torch.stack(diff_labels).numpy(); fps = 35 if source == 'UBFC-PHYS' else 30
        hrs = (estimate_hr_targets(yall, fps=fps) if req['variant']=='restored' else
               np.array([jt.OfficialTrainer.get_hr(None, y, sr=fps) for y in yall]))
        invalid = [dict(sample_ids[i], hr=float(h)) for i,h in enumerate(hrs) if not np.isfinite(h) or not 0 <= int(h-40) < 140]
        audit = dict(variant=req['variant'], count=len(hrs), invalid=invalid, min=float(min(hrs)), max=float(max(hrs)))
        atomic_json(out/'loss_target_audit.json', audit); atomic_json(out/'train_samples.json', sample_ids)
        if invalid:
            atomic_json(out/'summary.json', dict(status='ineligible', source=source, variant=req['variant'],
                        reason='Source HR outside unchanged loss CE bins; no clipping or record exclusion', finished_at=time.time()))
            log('Source candidate ineligible', phase='ineligible'); return
        if args.preflight_only:
            log('Preflight passed', phase='complete'); return
        class Indexed(Dataset):
            def __len__(self): return len(ds)
            def __getitem__(self, index):
                x,y = ds[index]; return x,y,float(hrs[index])
        loader = DataLoader(Indexed(), batch_size=4, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(42), drop_last=False)
        net = model(); opt = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=5e-5)
        pearson = NegPearsonLoss(); frequency = FrequencyLoss(fps=fps, diff_flag=req['variant']=='restored').cuda()
        history = []; (out/'checkpoints').mkdir()
        for epoch in range(1,21):
            started = time.time(); net.train(); losses = []
            log('Training epoch started', phase='training', epoch=epoch, epochs=20, batch=0, batches=len(loader))
            for i,(x,y,h) in enumerate(loader,1):
                opt.zero_grad(set_to_none=True)
                pred = net(x.cuda(), 2.0)[0]
                pred = (pred-pred.mean(-1,keepdim=True))/pred.std(-1,keepdim=True).clamp_min(1e-8)
                lp = pearson(pred,y.cuda()); ce,kl = frequency(pred,h.float().cuda()); loss = lp+ce+kl
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step()
                losses.append([float(v.detach()) for v in (lp,ce,kl)])
                if i==1 or i%25==0 or i==len(loader): log('Training progress', batch=i, loss=float(loss.detach()))
            ck = out/'checkpoints'/f'epoch_{epoch:02d}.pt'
            torch.save(dict(model=net.state_dict(), epoch=epoch, protocol_sha256=req['protocol_sha256']),ck)
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
            result = score(net, valid_rows, source, epoch)
            random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
            parts = dict(zip(('pearson','frequency_ce','distribution_kl'),np.mean(losses,axis=0).tolist()))
            history.append(dict(epoch=epoch,dataset_role='source_validation',checkpoint=str(ck),
                checkpoint_sha256=jt.digest(ck),recording_test_rmse=result['per_recording']['RMSE_bpm'],
                test_style_validation=result,loss_components=parts,seconds=time.time()-started))
            atomic_json(out/'history.json',history)
            torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,
                            protocol_sha256=req['protocol_sha256']),out/'last.pt')
            log('Validation complete',phase='validation_complete',epoch=epoch,epochs=20,
                recording_rmse=history[-1]['recording_test_rmse'])
        best = choose(history); shutil.copy2(best['checkpoint'],out/'best.pt')
        selection = dict(dataset_role='source_validation',criterion='recording_test_rmse',best_epoch=best['epoch'],
            best_rmse=best['recording_test_rmse'],target_used_for_selection=False,tie_break='earliest_epoch',selected_at=time.time())
        atomic_json(out/'selection.json',selection)
        atomic_json(out/'bundle.json',dict(source=source,model='bipulseformer',variant=req['variant'],seed=42,
            best_epoch=best['epoch'],split=split,selection=selection,checkpoint_sha256=jt.digest(out/'best.pt'),
            protocol_sha256=req['protocol_sha256'],snapshot=str(snap),
            snapshot_manifest_sha256=jt.digest(manifest_path)))
        atomic_json(out/'summary.json',dict(status='complete',source=source,variant=req['variant'],selection=selection,finished_at=time.time()))
        log('Source training complete',phase='complete')
    except BaseException as e:
        log(f'{type(e).__name__}: {e}',phase='failed'); raise


if __name__ == '__main__': main()
