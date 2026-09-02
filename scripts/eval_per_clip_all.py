"""Re-evaluate all relevant BiPhysFormer checkpoints with per-clip MAE
(PhysFormer / RhythmFormer paper standard metric).

Targets:
  1. UBFC intra 60/40 E8       (이전 학습)
  2. PURE intra 80/20 E9       (직전 학습)
  3. PURE intra 7/1/2 best     (현재 학습 중 - 끝나면 자동)
  4. UBFC intra 7/1/2 best     (현재 학습 중 - 끝나면 자동)
"""
import os, sys, io, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import torch
from src.models.biphysformer import ViT_BiPhysFormer
from src.data.rppg_dataset import get_dataloader
from src.evaluation import evaluate_per_subject
from src.evaluation_per_clip import evaluate_per_clip


def load_model(ckpt_path, device):
    model = ViT_BiPhysFormer(
        patches=(4, 4, 4), dim=96, ff_dim=144, num_heads=4, num_layers=12,
        dropout_rate=0.1, theta=0.7, image_size=(160, 128, 128),
        n_win=(2, 2, 2), topk=4,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def run_inference(model, loader, device):
    all_p, all_g = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            rPPG, _, _, _ = model(inputs, gra_sharp=2.0)
            rPPG = (rPPG - rPPG.mean(-1, keepdim=True)) / (rPPG.std(-1, keepdim=True) + 1e-8)
            all_p.append(rPPG.cpu().numpy()); all_g.append(labels.numpy())
    return np.concatenate(all_p), np.concatenate(all_g)


def find_best_epoch(summary_path):
    """Find epoch with lowest VALID RMSE from a summary.json."""
    with open(summary_path) as f:
        data = json.load(f)
    # support both list[dict] (older) and single dict (new)
    if isinstance(data, list):
        data = data[0]
    if 'best_epoch' in data:
        return data['best_epoch']
    return None


def evaluate_checkpoint(label, ckpt, dataset, path, split_range,
                       pure_split_mode='subject_exclusive'):
    print(f"\n{'='*70}")
    print(f"[*] {label}")
    print(f"    ckpt: {ckpt}")
    print(f"    test: {dataset} at {path}, split={split_range}")
    print('='*70)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    common_kwargs = dict(face_crop=True, dynamic_detection_freq=0,
                         data_type='diff_normalized', num_workers=4, pin_memory=True)
    if dataset == 'PURE':
        common_kwargs['pure_split_mode'] = pure_split_mode
    test_loader = get_dataloader(
        dataset, path, batch_size=4, clip_len=160,
        shuffle=False, chunk_step=80,
        split_range=split_range, **common_kwargs,
    )
    print(f"  test clips: {len(test_loader.dataset)}")

    model = load_model(ckpt, device)
    preds, gts = run_inference(model, test_loader, device)
    print(f"  preds shape: {preds.shape}")

    # PAPER STANDARD: per-clip MAE
    pc = evaluate_per_clip(preds, gts, fs=30, diff_flag=True,
                           low_pass=0.75, high_pass=2.5)
    print(f"\n  --- PER-CLIP (PhysFormer/RhythmFormer 표준) ---")
    print(f"  MAE_clip   = {pc['MAE_bpm_clip']:.4f} BPM")
    print(f"  RMSE_clip  = {pc['RMSE_bpm_clip']:.4f} BPM")
    print(f"  MAPE_clip  = {pc['MAPE_pct_clip']:.4f} %")
    print(f"  Pearson_clip = {pc['Pearson_clip']:.4f}")
    print(f"  n_clips    = {pc['n_clips']}")

    # 참고: per-subject (rPPG-Toolbox 표준)
    ps = evaluate_per_subject(preds, gts, test_loader.dataset.samples,
                              fs=30, diff_flag=True, low_pass=0.75, high_pass=2.5)
    print(f"\n  --- PER-SUBJECT (rPPG-Toolbox 표준, 참고) ---")
    print(f"  MAE_subj   = {ps['MAE_bpm']:.4f} BPM")
    print(f"  RMSE_subj  = {ps['RMSE_bpm']:.4f} BPM")
    print(f"  Pearson_subj = {ps['Pearson']:.4f}")
    print(f"  signal_Pearson = {ps['signal_Pearson_mean']:.4f}")
    print(f"  n_subjects = {ps['n_subjects']}")
    return {'label': label, 'per_clip': pc, 'per_subject': ps}


def main():
    targets = []
    # ONLY 현재 학습 (7/1/2) 결과만 평가. 기존 checkpoint 는 건너뜀.

    # 1. PURE intra 7/1/2 - best (valid 기준)
    sum_path = 'results/intra_pure_biphysformer_712/summary.json'
    if os.path.exists(sum_path):
        be = find_best_epoch(sum_path)
        if be:
            ckpt = f'results/intra_pure_biphysformer_712/checkpoints/PURE_epoch{be}.pt'
            if os.path.exists(ckpt):
                targets.append((f'PURE intra 7/1/2 (separate valid) E{be}',
                                ckpt, 'PURE', 'D:\\PURE', (0.8, 1.0)))

    # 2. UBFC intra 7/1/2 - best (valid 기준)
    sum_path = 'results/intra_ubfc_biphysformer_712/summary.json'
    if os.path.exists(sum_path):
        be = find_best_epoch(sum_path)
        if be:
            ckpt = f'results/intra_ubfc_biphysformer_712/checkpoints/UBFC-rPPG_epoch{be}.pt'
            if os.path.exists(ckpt):
                targets.append((f'UBFC intra 7/1/2 (separate valid) E{be}',
                                ckpt, 'UBFC-rPPG', 'D:\\UBFC-rPPG', (0.8, 1.0)))

    print(f"Will evaluate {len(targets)} checkpoints with per-clip metric")
    print()
    results = []
    for label, ckpt, ds, path, sr in targets:
        try:
            r = evaluate_checkpoint(label, ckpt, ds, path, sr)
            results.append(r)
        except Exception as e:
            print(f"[!] {label} failed: {e}")
            import traceback; traceback.print_exc()

    # Final paper-comparable table
    print("\n" + "=" * 70)
    print(" FINAL: Paper-comparable per-clip metrics (PhysFormer/RhythmFormer 표준)")
    print("=" * 70)
    print(f"{'Setup':<45} | {'MAE':>8} | {'RMSE':>8} | {'Pearson':>8}")
    print("-" * 80)
    for r in results:
        c = r['per_clip']
        print(f"{r['label']:<45} | {c['MAE_bpm_clip']:>7.3f}  | {c['RMSE_bpm_clip']:>7.3f}  | {c['Pearson_clip']:>7.4f}")

    # Save to JSON
    out = []
    for r in results:
        out.append({
            'label': r['label'],
            'per_clip': {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else None
                         for k, v in r['per_clip'].items() if k not in ('pred_hrs', 'gt_hrs')},
            'per_subject': {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else None
                            for k, v in r['per_subject'].items()},
        })
    with open('results/per_clip_evaluation.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: results/per_clip_evaluation.json")


if __name__ == '__main__':
    main()
