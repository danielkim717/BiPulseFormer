"""Frozen-evaluator source-only loss comparison. Target tests are separate jobs."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import uuid


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    for attempt in range(120):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 119:
                raise
            time.sleep(.25)


def choose(rows):
    import math
    if len(rows) != 10 or [r['epoch'] for r in rows] != list(range(1, 11)):
        raise ValueError('All ten source-validation checkpoints are required')
    if any(r['dataset_role'] != 'source_validation' or not math.isfinite(r['recording_test_rmse']) for r in rows):
        raise ValueError('Invalid source-only selection history')
    return min(rows, key=lambda r: (r['recording_test_rmse'], r['epoch']))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--request', type=Path, required=True)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    req = json.loads(args.request.read_text(encoding='utf-8'))
    if args.output: req['output'] = str(args.output.resolve())
    snap = Path(req['snapshot']); out = Path(req['output'])
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(snap))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    from src import journal_toolbox as jt
    from src import matched_data
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from src.train import FrequencyLoss, estimate_hr_targets, _complex_absolute_one
    jt.atomic_json = matched_data.atomic_json = atomic_json
    progress = {}

    def log(message, **fields):
        if fields.get('phase') and fields['phase'] != progress.get('phase'):
            progress.clear()
        progress.update(fields, updated_at=time.time())
        atomic_json(out/'progress.json', progress)
        line = time.strftime('[%Y-%m-%d %H:%M:%S] ') + message
        with (out/'log.txt').open('a', encoding='utf-8') as f:
            f.write(line+'\n')
        print(line, flush=True)

    try:
        jt.verify_vendor(); jt.activate_adapters(); os.chdir(jt.VENDOR)
        torch.set_num_threads(2); jt.cv2.setNumThreads(1)
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        roots = json.loads(Path(req['roots_file']).read_text())
        source = req['source']; target = req.get('target', 'UBFC-rPPG')
        is_test = req['mode'] == 'test'
        bundle = json.loads((Path(req['bundle'])/'bundle.json').read_text()) if is_test else None
        checkpoint = Path(req['bundle'])/'best.pt' if is_test else None
        if is_test and (bundle['source'] != source or bundle['seed'] != 42 or jt.digest(checkpoint) != bundle['checkpoint_sha256']):
            raise ValueError('Source bundle identity mismatch')
        if is_test:
            for name, sha in bundle['source_hashes'].items():
                if jt.digest(snap/name) != sha: raise ValueError('Frozen bundle code mismatch: '+name)
        cfg, template = jt.make_config(source, target, out, roots, checkpoint)
        config = dict(protocol_version='source-optimized-fixed-eval-v1', model='bipulseformer', seed=42,
                      source=source, target=target, reference_commit=jt.COMMIT, paper_equivalence=False,
                      loss_variant=req.get('variant'), selection='source_recording_RMSE',
                      toolbox=jt.config_module.yaml.safe_load(cfg.dump()))
        atomic_json(out/'config.json', config)

        class LoggedLoader:
            def __init__(self, loader, phase):
                self.loader = loader; self.phase = phase; self.epoch = 0
            def __len__(self):
                return len(self.loader)
            def __iter__(self):
                if self.phase == 'training': self.epoch += 1
                log(self.phase+' started', phase=self.phase, epoch=self.epoch, epochs=10, batch=0, batches=len(self))
                for i, batch in enumerate(self.loader, 1):
                    yield batch
                    if i == 1 or i % 25 == 0 or i == len(self):
                        log(f'{self.phase} batch {i}/{len(self)}', batch=i)

        class TestTrainer(jt.OfficialTrainer):
            def save_test_outputs(self, predictions, labels, config):
                super().save_test_outputs(predictions, labels, config)
                self.saved_predictions = predictions; self.saved_labels = labels

        if is_test:
            records = jt.prepare(jt.inventory(target, roots[target]), target, req['cache'], cfg.TEST.DATA.PREPROCESS, log)
            ds = jt.CachedClips(records)
            atomic_json(out/'test_inventory.json', records); atomic_json(out/'test_samples.json', ds.samples)
            atomic_json(out/'split.json', bundle['split'])
            trainer = TestTrainer(cfg, {})
            jt.set_input_fps(trainer.model, cfg.TEST.DATA.FS)
            trainer.test({'test': LoggedLoader(DataLoader(ds, batch_size=4, shuffle=False, num_workers=0,
                           generator=torch.Generator().manual_seed(42)), 'test')})
            result = jt.summarize_predictions(trainer.saved_predictions, trainer.saved_labels, records, cfg.TEST.DATA.FS)
            atomic_json(out/'summary.json', dict(config=config, model='bipulseformer', seed=42, source=source,
                        target=target, reference_commit=jt.COMMIT, split=bundle['split'], best_epoch=bundle['best_epoch'],
                        checkpoint_sha256=bundle['checkpoint_sha256'], source_validation_selection='source_recording_RMSE',
                        test=result, finished_at=time.time(), input_coverage=[dict(recording_id=r['id'], **r['coverage']) for r in records]))
            log('Final test complete', phase='complete')
            return

        records = {}; datasets = {}; loaders = {}
        for role in ('train', 'valid'):
            records[role] = jt.prepare(jt.inventory(source, roots[source], role), source, req['cache'],
                                      getattr(cfg, role.upper()).DATA.PREPROCESS, log)
            datasets[role] = jt.CachedClips(records[role])
            atomic_json(out/(role+'_inventory.json'), records[role])
            atomic_json(out/(role+'_samples.json'), datasets[role].samples)
            loaders[role] = LoggedLoader(DataLoader(datasets[role], batch_size=4, shuffle=role=='train', num_workers=0,
                                        generator=torch.Generator().manual_seed(42)), 'training' if role=='train' else 'validation')
        split = dict(source=source, train=sorted({r['subject'] for r in records['train']}),
                     valid=sorted({r['subject'] for r in records['valid']}),
                     train_records=[r['id'] for r in records['train']], valid_records=[r['id'] for r in records['valid']])
        if set(split['train']) & set(split['valid']): raise ValueError('Source subject overlap')
        atomic_json(out/'split.json', split)
        # Check every training HR before CUDA CE, with no clamping or exclusions.
        audits = {}
        for variant in ('direct', 'restored'):
            failures = []; hrs = []
            for r in records['train']:
                waves = np.load(r['cached_label'], mmap_mode='r')
                if variant == 'restored':
                    values = estimate_hr_targets(waves, fps=cfg.TRAIN.DATA.FS).tolist()
                else:
                    values = [jt.OfficialTrainer.get_hr(None, y, sr=cfg.TRAIN.DATA.FS) for y in waves]
                hrs.extend(values)
                failures.extend(dict(recording=r['id'], chunk=i, hr=float(h)) for i, h in enumerate(values)
                                if not np.isfinite(h) or not 0 <= int(h-40) < 140)
            audits[variant] = dict(eligible=not failures, count=len(hrs), min=float(min(hrs)), max=float(max(hrs)), invalid=failures)
        atomic_json(out/'loss_target_audit.json', audits)
        if not audits[req['variant']]['eligible']:
            atomic_json(out/'summary.json', dict(status='ineligible', source=source, variant=req['variant'],
                        reason='Source HR targets outside unchanged CE bins; no training or target test', finished_at=time.time()))
            log('Candidate ineligible: source HR outside CE bins', phase='ineligible')
            return
        if req['mode'] == 'preflight':
            atomic_json(out/'summary.json', dict(status='complete', source=source, audit=audits, finished_at=time.time()))
            log('Source loss target preflight complete', phase='complete')
            return

        history = []; pearson_values = []; spectral_values = []

        class Trainer(jt.OfficialTrainer):
            def save_model(self, index):
                super().save_model(index); self.observed_epoch = index+1
            def get_hr(self, y, sr=30, min=30, max=180):
                if req['variant'] == 'restored' and not getattr(self, 'in_validation', False):
                    return float(estimate_hr_targets(np.asarray(y)[None], fps=sr)[0])
                return super().get_hr(y, sr, min, max)
            def valid(self, data_loader):
                self.in_validation = True
                try:
                    official_rmse = super().valid(data_loader)
                finally:
                    self.in_validation = False
                # Preserve the exact upstream training state and RNG across the
                # additional recording-level assessment of the saved checkpoint.
                state = copy.deepcopy(self.model.state_dict()); was_training = self.model.training
                rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
                ck = Path(cfg.MODEL.MODEL_DIR)/(cfg.TRAIN.MODEL_FILE_NAME+f'_Epoch{self.observed_epoch-1}.pth')
                try:
                    self.model.load_state_dict(torch.load(ck, map_location=self.device, weights_only=True), strict=True)
                    self.model.eval(); predictions = {}; labels = {}
                    valid_loader = DataLoader(datasets['valid'], batch_size=4, shuffle=False, num_workers=0,
                                              generator=torch.Generator().manual_seed(42))
                    log('Source recording validation', phase='source_validation', checkpoint_epoch=self.observed_epoch,
                        checkpoints=10, batch=0, batches=len(valid_loader))
                    with torch.inference_mode():
                        for i, (x, y, names, chunks) in enumerate(valid_loader, 1):
                            p = self.model(x.float().to(self.device), 2.0)[0]
                            p = (p-p.mean(axis=-1).view(-1, 1))/torch.std(p).view(-1, 1)
                            for pp, yy, name, chunk in zip(p.cpu(), y, names, chunks):
                                predictions.setdefault(name, {})[int(chunk)] = pp.clone()
                                labels.setdefault(name, {})[int(chunk)] = yy.clone()
                            if i == 1 or i % 25 == 0 or i == len(valid_loader): log('Source validation progress', batch=i)
                    score = jt.summarize_predictions(predictions, labels, records['valid'], cfg.VALID.DATA.FS)
                finally:
                    self.model.load_state_dict(state); self.model.train(was_training)
                    random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
                parts = dict(pearson=float(np.mean(pearson_values)), frequency_ce=float(np.mean([r[1] for r in spectral_values])),
                             distribution_kl=float(np.mean([r[0] for r in spectral_values])))
                history.append(dict(epoch=self.observed_epoch, dataset_role='source_validation',
                    checkpoint=str(ck), checkpoint_sha256=jt.digest(ck), official_clip_rmse=float(official_rmse),
                    recording_test_rmse=score['per_recording']['RMSE_bpm'], test_style_validation=score,
                    loss_components=parts, seconds=time.time()-self.epoch_started))
                atomic_json(out/'history.json', history)
                log(f'epoch {self.observed_epoch}: source recording RMSE={history[-1]["recording_test_rmse"]:.6f}',
                    phase='validation_complete', epoch=self.observed_epoch, epochs=10)
                pearson_values.clear(); spectral_values.clear(); self.epoch_started=time.time()
                return official_rmse

        trainer = Trainer(cfg, loaders); jt.set_input_fps(trainer.model, cfg.TRAIN.DATA.FS)
        loss_cls = jt.trainer_module.TorchLossComputer
        original = loss_cls.cross_entropy_power_spectrum_DLDL_softmax2
        restored = FrequencyLoss(fps=cfg.TRAIN.DATA.FS, diff_flag=True).to(trainer.device)
        def spectral(pred, hr, fs, std=1.0):
            if req['variant'] == 'restored':
                ce, kl = restored(pred[None], hr[None])
                with torch.no_grad():
                    wave = pred.cumsum(-1) @ restored._detrend_operator.T
                    spectrum = _complex_absolute_one(wave, fs, torch.arange(40, 180, device=pred.device))
                    hr_error = abs(hr-40-spectrum.argmax())
                result = (kl, ce, hr_error)
            else:
                result = original(pred, hr, fs, std)
            spectral_values.append((float(result[0].detach()), float(result[1].detach())))
            return result
        loss_cls.cross_entropy_power_spectrum_DLDL_softmax2 = staticmethod(spectral)
        hook = trainer.criterion_Pearson.register_forward_hook(lambda m, x, y: pearson_values.append(float(y.detach())))
        trainer.epoch_started = time.time()
        try: trainer.train(loaders)
        finally:
            hook.remove(); loss_cls.cross_entropy_power_spectrum_DLDL_softmax2 = original
        chosen = choose(history)
        selection = dict(dataset_role='source_validation', criterion='recording_test_rmse',
                         best_epoch=chosen['epoch'], best_rmse=chosen['recording_test_rmse'],
                         target_used_for_selection=False, selected_at=time.time(), tie_break='earliest_epoch')
        atomic_json(out/'selection.json', selection)
        shutil.copy2(chosen['checkpoint'], out/'best.pt')
        hashes = {str(p.relative_to(snap)).replace('\\','/'):jt.digest(p) for folder in ('src','configs')
                  for p in (snap/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        bundle = dict(model='bipulseformer', source=source, seed=42, variant=req['variant'],
                      best_epoch=chosen['epoch'], checkpoint_sha256=jt.digest(out/'best.pt'), reference_commit=jt.COMMIT,
                      split=split, source_hashes=hashes, training_config=config['toolbox'],
                      source_validation_selection='source_recording_RMSE', selection=selection)
        atomic_json(out/'bundle.json', bundle)
        atomic_json(out/'summary.json', dict(status='complete', source=source, variant=req['variant'], selection=selection,
                    checkpoint_sha256=bundle['checkpoint_sha256'], finished_at=time.time()))
        log('Source training and selection complete', phase='complete')
    except BaseException as error:
        log(f'{type(error).__name__}: {error}', phase='failed')
        raise


if __name__ == '__main__':
    main()
