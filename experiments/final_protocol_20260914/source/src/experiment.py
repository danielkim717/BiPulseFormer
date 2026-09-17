"""Shared training engine. The target is evaluated only after validation selection."""
import json
import os
import platform
import random
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

from src.protocol import discover_subjects, fingerprint, make_split, read_protocol, subject_id


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def seed_worker(_):
    import numpy as np
    import torch
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def normalize(x):
    return (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True).clamp_min(1e-8)


def build_model(config, kind):
    from src.models.bipulseformer import ViT_BiPulseFormer
    from src.models.physformer_baseline import PhysFormer
    common = dict(patches=(4, 4, 4), dim=config["dim"], ff_dim=config["ff_dim"],
                  num_heads=config["num_heads"], num_layers=config["num_layers"],
                  dropout_rate=config["dropout"], theta=config["theta"],
                  image_size=(config["clip_len"], config["image_size"], config["image_size"]))
    if kind == "physformer":
        return PhysFormer(**common)
    return ViT_BiPulseFormer(**common, frame=config["clip_len"], fps=config["fps"],
                            n_win=tuple(config["n_win"]), topk=config["topk"],
                            routing_mode=config["routing_mode"], diff_routing=config["diff_routing"],
                            routing_tau=config["routing_tau"],
                            routing_band=(config["bpm_low"] / 60, config["bpm_high"] / 60))


def build_loader(name, root, ids, config, args, train=False):
    import torch
    from torch.utils.data import DataLoader
    from src.data.rppg_dataset import RPPGDataset
    if name == 'UBFC-PHYS':
        from src.data.phys_cached_dataset import PHYSCacheDataset
        dataset = PHYSCacheDataset(root, ids, config['clip_len'],
                                   config['clip_len'] if train else config['eval_step'], train)
    else:
        dataset = RPPGDataset(name, root, subjects_filter=ids,
                          clip_len=config["clip_len"], img_size=config["image_size"],
                          face_crop=True, dynamic_detection_freq=0,
                          data_type="diff_normalized", random_hflip=train,
                          chunk_step=config["clip_len"] if train else config["eval_step"],
                          hr_filter=False, fps=config["fps"])
    actual = {subject_id(name, sample["video_id"]) for sample in dataset.samples}
    if actual != set(ids):
        raise ValueError(f"Subject inventory mismatch: expected={sorted(ids)}, actual={sorted(actual)}")
    if not len(dataset):
        raise ValueError("Empty dataset")
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=train,
                      num_workers=args.workers, pin_memory=str(args.device).startswith("cuda"),
                      drop_last=False, worker_init_fn=seed_worker, generator=generator,
                      persistent_workers=False)


def evaluate(model, loader, config, device):
    import numpy as np
    import torch
    from src.evaluation import evaluate_per_subject
    from src.evaluation_per_clip import evaluate_per_clip
    model.eval()
    predictions, labels = [], []
    with torch.inference_mode():
        for x, y in loader:
            pred = normalize(model(x.to(device), gra_sharp=config["gra_sharp"])[0])
            if not torch.isfinite(pred).all() or not torch.isfinite(y).all():
                raise FloatingPointError("Non-finite evaluation signals")
            predictions.append(pred.cpu().numpy())
            labels.append(y.numpy())
    predictions, labels = np.concatenate(predictions), np.concatenate(labels)
    kwargs = dict(fs=config["fps"], diff_flag=True,
                  low_pass=config["bpm_low"] / 60, high_pass=config["bpm_high"] / 60)
    clips = evaluate_per_clip(predictions, labels, **kwargs)
    recordings = evaluate_per_subject(predictions, labels, loader.dataset.samples, **kwargs)
    clips.pop("pred_hrs")
    clips.pop("gt_hrs")
    recordings.pop("n_subjects")  # Historical alias counted recordings, not people.
    recordings["n_subjects"] = len({subject_id(loader.dataset.dataset_name, s["video_id"])
                                       for s in loader.dataset.samples})
    return {"per_clip": clips, "per_recording": recordings}, predictions, labels


def run(args):
    config = read_protocol(args.config)
    if args.source == 'UBFC-PHYS' or args.target == 'UBFC-PHYS':
        plan_path = args.source_root if args.source == 'UBFC-PHYS' else args.target_root
        phys_plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
        config = dict(config, phys_data_protocol=phys_plan['protocol'],
                      phys_inventory_hash=fingerprint(phys_plan['records']))
    source_ids = discover_subjects(args.source, args.source_root)
    split = make_split(source_ids, args.mode, config)
    target = args.target if args.mode == "cross" else args.source
    target_root = args.target_root if args.mode == "cross" else args.source_root
    if args.mode == "cross":
        split["test"] = discover_subjects(target, target_root)
    manifest = {"source": args.source, "target": target, "mode": args.mode,
                "subject_ids": split, "split_seed": config["split_seed"]}
    identity = {"config": config, "split": manifest, "model": args.model, "seed": args.seed}
    run_id = fingerprint(identity)
    if args.dry_run:
        print(json.dumps({"run_id": run_id, **identity}, indent=2))
        return
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory; existing results are protected: {output}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from src.train import FrequencyLoss, NegPearsonLoss, estimate_hr_targets
    torch.set_num_threads(getattr(args, 'threads', 2))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; explicitly use --device cpu for CPU runs")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # CUDA MaxPool3d backward has no deterministic implementation in the tested
    # PyTorch build. Seed everything and warn for unsupported kernels.
    torch.use_deterministic_algorithms(True, warn_only=True)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", {**identity, "run_id": run_id,
                                      "source_root": str(Path(args.source_root).resolve()),
                                      "target_root": str(Path(target_root).resolve())})
    write_json(output / "split.json", manifest)
    repo = Path(__file__).resolve().parents[1]
    def git(*argv):
        result = subprocess.run(["git", *argv], cwd=repo, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    source_hashes = {str(p.relative_to(repo)): fingerprint(p.read_text(encoding="utf-8"))
                     for folder in ("src", "scripts") for p in (repo / folder).rglob("*.py")}
    write_json(output / "environment.json", {
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "device": str(device), "cuda": torch.version.cuda,
        "packages": {name: version(name) for name in ('torch', 'torchvision', 'numpy', 'scipy', 'opencv-python', 'matplotlib')},
        "gpu": torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        "determinism": "seeded; deterministic algorithms warn_only (CUDA MaxPool3d backward)",
        "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
        "source_hashes": source_hashes, "workers": args.workers,
        "cpu_threads": torch.get_num_threads(),
        "selection_metric": "validation per-clip HR RMSE", "planned_test_evaluations": 1})
    def log(value):
        value = time.strftime('[%Y-%m-%d %H:%M:%S] ') + value
        print(value, flush=True)
        with (output / "log.txt").open("a", encoding="utf-8") as handle:
            handle.write(value + "\n")
    train = build_loader(args.source, args.source_root, split["train"], config, args, train=True)
    valid = build_loader(args.source, args.source_root, split["valid"], config, args)
    def sample_manifest(loader):
        return [{"video_id": s["video_id"], "start": s["first_frame_idx"]}
                for s in loader.dataset.samples]
    train_samples, valid_samples = sample_manifest(train), sample_manifest(valid)
    write_json(output / "train_samples.json", train_samples)
    write_json(output / "valid_samples.json", valid_samples)
    model = build_model(config, args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    pearson = NegPearsonLoss()
    frequency = FrequencyLoss(fps=config["fps"], bpm_low=config["bpm_low"],
                              bpm_high=config["bpm_high"], diff_flag=True).to(device)
    history, best_score, best_epoch = [], float("inf"), None
    log(f"run={run_id} model={args.model} params={sum(p.numel() for p in model.parameters())}")
    for epoch in range(1, config["epochs"] + 1):
        started = time.monotonic()
        model.train()
        total, count = np.zeros(3), 0
        for step, (x, y) in enumerate(train, 1):
            targets = estimate_hr_targets(y.numpy(), fps=config["fps"],
                                          bpm_low=config["bpm_low"], bpm_high=config["bpm_high"])
            x, y = x.to(device), y.to(device)
            targets = torch.as_tensor(targets, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            pred = normalize(model(x, gra_sharp=config["gra_sharp"])[0])
            lp = pearson(pred, y)
            ce, kl = frequency(pred, targets)
            loss = config["alpha"] * lp + config["beta"] * (ce + kl)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            total += np.array([lp.item(), ce.item(), kl.item()]) * len(y)
            count += len(y)
            if step == 1 or step % 25 == 0:
                log(f"epoch={epoch} batch={step}/{len(train)} loss={loss.item():.5f}")
                write_json(output / 'progress.json', {'phase': 'training', 'epoch': epoch,
                    'epochs': config['epochs'], 'batch': step, 'batches': len(train),
                    'epoch_elapsed_seconds': time.monotonic() - started, 'updated_at': time.time()})
        write_json(output / 'progress.json', {'phase': 'validation', 'epoch': epoch,
                   'epochs': config['epochs'], 'updated_at': time.time()})
        metrics, _, _ = evaluate(model, valid, config, device)
        score = metrics["per_clip"]["RMSE_bpm_clip"]
        if not np.isfinite(score):
            raise FloatingPointError("Non-finite validation score")
        row = {"epoch": epoch, "loss_components": (total / count).tolist(),
               "valid": metrics, "seconds": time.monotonic() - started}
        history.append(row)
        if score < best_score:
            best_score, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": config,
                        "model_name": args.model, "run_id": run_id}, output / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "config": config, "run_id": run_id}, output / "last.pt")
        write_json(output / "history.json", history)
        log(f"epoch={epoch}/{config['epochs']} valid_clip_RMSE={score:.4f} best_epoch={best_epoch}")
    # Only now open target signals and perform the single held-out evaluation.
    write_json(output / 'progress.json', {'phase': 'test', 'best_epoch': best_epoch, 'updated_at': time.time()})
    state = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    test = build_loader(target, target_root, split["test"], config, args)
    test_samples = sample_manifest(test)
    write_json(output / "test_samples.json", test_samples)
    test_metrics, predictions, labels = evaluate(model, test, config, device)
    by_task = {}
    if target == 'UBFC-PHYS':
        from src.evaluation import evaluate_per_subject
        from src.evaluation_per_clip import evaluate_per_clip
        kwargs = dict(fs=config['fps'], diff_flag=True,
                      low_pass=config['bpm_low']/60, high_pass=config['bpm_high']/60)
        for task in ('T1', 'T2', 'T3'):
            indices = [i for i,s in enumerate(test_samples) if s['video_id'].endswith('_'+task)]
            chosen = [test.dataset.samples[i] for i in indices]
            c = evaluate_per_clip(predictions[indices], labels[indices], **kwargs)
            r = evaluate_per_subject(predictions[indices], labels[indices], chosen, **kwargs)
            c.pop('pred_hrs'); c.pop('gt_hrs')
            r['n_subjects'] = len({s['video_id'].split('_')[0] for s in chosen})
            by_task[task] = dict(per_clip=c, per_recording=r)
    np.savez_compressed(output / "test_predictions.npz", predictions=predictions, labels=labels)
    write_json(output / "summary.json", {"run_id": run_id, "best_epoch": best_epoch,
               "best_valid_clip_rmse": best_score, "test": test_metrics,
               "history": history, "config": config, "split": manifest, "seed": args.seed, "by_task": by_task,
               "code_hash": fingerprint(source_hashes),
               "sample_hashes": {"train": fingerprint(train_samples),
                                 "valid": fingerprint(valid_samples), "test": fingerprint(test_samples)}})
    log(f"complete: best_epoch={best_epoch} test={test_metrics['per_clip']}")
    write_json(output / 'progress.json', {'phase': 'complete', 'best_epoch': best_epoch, 'updated_at': time.time()})
