"""Serializable experiment protocol and subject-exclusive split manifests."""
import hashlib
import json
import random
from pathlib import Path


def read_protocol(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["protocol_version"] != "bipulseformer-v1":
        raise ValueError("Unsupported protocol version")
    if config['pure_alignment'] != 'timestamp_interpolation' or config['frequency_input'] != 'restored_ppg':
        raise ValueError('Protocol v1 requires timestamp alignment and restored spectral inputs')
    if config['fps'] != 30:
        raise ValueError('Protocol-v1 dataset alignment requires 30 fps')
    if config["clip_len"] < 32 or config["clip_len"] % 4 or config["image_size"] != 128:
        raise ValueError("Current pipeline requires 128px inputs and T >= 32 divisible by 4")
    if not 0 < config["bpm_low"] < config["bpm_high"] < config["fps"] * 30:
        raise ValueError("Invalid HR band / Nyquist frequency")
    for key in ("epochs", "batch_size", "lr", "eval_step", "routing_tau", "grad_clip", "gra_sharp"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ('epochs', 'batch_size', 'clip_len', 'eval_step', 'dim', 'ff_dim', 'num_heads', 'num_layers'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if config['num_layers'] % 3 or config['dim'] % config['num_heads']:
        raise ValueError('Invalid transformer layer or head dimensions')
    if config['routing_mode'] not in ('mean', 'fft_magnitude', 'fft_power'):
        raise ValueError('Unknown routing mode')
    if type(config['diff_routing']) is not bool:
        raise ValueError('diff_routing must be boolean')
    windows = config['n_win']
    if len(windows) != 3 or any(type(v) is not int or v < 1 for v in windows):
        raise ValueError('Invalid routing windows')
    import math
    if type(config['topk']) is not int or not 1 <= config['topk'] <= math.prod(windows):
        raise ValueError('Invalid top-k')
    if any(size % win for size, win in zip((config['clip_len'] // 4, 4, 4), windows)):
        raise ValueError('Routing windows must divide the feature grid')
    if config['bpm_high'] / 60 >= config['fps'] / 8:
        raise ValueError('Routing band exceeds token Nyquist frequency')
    if not 0 <= config['dropout'] < 1 or min(config['alpha'], config['beta'], config['weight_decay']) < 0:
        raise ValueError('Invalid regularization or loss weights')
    if config["eval_step"] > config["clip_len"]:
        raise ValueError("Evaluation windows must cover the recording without gaps")
    fractions = config["intra_fractions"]
    if len(fractions) != 3 or min(fractions) <= 0 or abs(sum(fractions) - 1) > 1e-8:
        raise ValueError("Invalid intra fractions")
    if not 0 < config["cross_train_fraction"] < 1:
        raise ValueError("Invalid cross train fraction")
    seeds = config['report_seeds']
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError('Use at least two distinct integer report seeds')
    return config


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def subject_id(dataset, video_id):
    if dataset == "PURE":
        return video_id.split("-")[0]
    if dataset == "UBFC-PHYS":
        return video_id.split("_")[0]
    return video_id


def discover_subjects(dataset, root):
    root = Path(root)
    if dataset == 'UBFC-PHYS':
        plan = json.loads(root.read_text(encoding='utf-8'))
        if plan.get('preparation_status'):
            ready = json.loads(Path(plan['preparation_status']).read_text(encoding='utf-8'))
            if ready['status'] != 'complete':
                raise ValueError('UBFC-PHYS full preparation must complete before training')
        from .phys_selection import validate_inventory
        return validate_inventory(plan)
    if dataset == "UBFC-rPPG" and (root / "DATASET_2").is_dir():
        root = root / "DATASET_2"
    if not root.is_dir():
        raise FileNotFoundError(root)
    if dataset == "PURE":
        videos = [p.name for p in root.iterdir() if p.is_dir()
                  and (p / f"{p.name}.json").is_file() and (p / p.name).is_dir()]
    elif dataset == "UBFC-rPPG":
        videos = [p.name for p in root.iterdir() if p.is_dir()
                  and p.name.startswith("subject") and p.name[7:].isdigit()
                  and (p / "vid.avi").is_file() and (p / "ground_truth.txt").is_file()]
    else:
        raise ValueError('Unknown dataset')
    subjects = sorted({subject_id(dataset, v) for v in videos})
    if not subjects:
        raise ValueError(f"No complete recordings in {root}")
    return subjects


def make_split(subjects, mode, config):
    ids = sorted(set(subjects))
    random.Random(config["split_seed"]).shuffle(ids)
    n = len(ids)
    if mode == "cross":
        cut = int(n * config["cross_train_fraction"])
        split = {"train": ids[:cut], "valid": ids[cut:]}
    elif mode == "intra":
        first = int(n * config["intra_fractions"][0])
        second = int(n * sum(config["intra_fractions"][:2]))
        split = {"train": ids[:first], "valid": ids[first:second], "test": ids[second:]}
    else:
        raise ValueError(mode)
    if any(not ids for ids in split.values()):
        raise ValueError("Too few subjects for the frozen split fractions")
    return split
