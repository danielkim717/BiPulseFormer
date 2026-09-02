"""Eval MAPE for all paper-reportable checkpoints.

per-subject MAPE = mean(|pred_hr - gt_hr| / gt_hr) * 100
"""
import os, sys, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import json
import numpy as np
import torch
from src.models.bipulseformer import ViT_BiPulseFormer
from src.data.rppg_dataset import get_dataloader
from src.evaluation import evaluate_per_subject, get_subject_signals


def _load_run_config(ckpt_path):
    """ckpt_path = results/<run>/checkpoints/x.pt 패턴에서 <run>/summary.json 의
    'config' 를 읽어온다. 없으면 None."""
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))
    summary_path = os.path.join(run_dir, 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, encoding='utf-8') as f:
            return json.load(f).get('config')
    return None


def _check_config_drift(ckpt_path, n_win, topk, routing_mode):
    """n_win/topk/routing_mode 는 state_dict 밖에 있어서 load_state_dict(strict=False)
    로는 절대 못 잡는 config drift 버그 클래스다 — 체크포인트의 학습 config 와
    이 스크립트가 쓰려는 값이 다르면 크게 경고한다."""
    cfg = _load_run_config(ckpt_path)
    if cfg is None:
        return
    mismatches = []
    if list(cfg.get('n_win', [])) != list(n_win):
        mismatches.append(f"n_win: 체크포인트={cfg.get('n_win')} vs 사용={list(n_win)}")
    if cfg.get('topk') != topk:
        mismatches.append(f"topk: 체크포인트={cfg.get('topk')} vs 사용={topk}")
    if cfg.get('routing_mode', 'mean') != routing_mode:
        mismatches.append(f"routing_mode: 체크포인트={cfg.get('routing_mode', 'mean')} vs 사용={routing_mode}")
    if mismatches:
        print('=' * 78)
        print(f'[!!!] CONFIG MISMATCH for {ckpt_path} — 크래시 없이 조용히 잘못된 결과가 나온다:')
        for m in mismatches:
            print(f'      - {m}')
        print('=' * 78)


def eval_ckpt(ckpt, dataset, path, split_range, label, pure_mode='subject_exclusive',
              n_win=(2, 2, 2), topk=4, routing_mode='mean'):
    _check_config_drift(ckpt, n_win, topk, routing_mode)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    common = dict(face_crop=True, dynamic_detection_freq=0,
                  data_type='diff_normalized', num_workers=4, pin_memory=True)
    if dataset == 'PURE':
        common['pure_split_mode'] = pure_mode
    loader = get_dataloader(dataset, path, batch_size=4, clip_len=160,
                            shuffle=False, chunk_step=80,
                            split_range=split_range, **common)
    model = ViT_BiPulseFormer(
        patches=(4, 4, 4), dim=96, ff_dim=144, num_heads=4, num_layers=12,
        dropout_rate=0.1, theta=0.7, image_size=(160, 128, 128),
        n_win=n_win, topk=topk, routing_mode=routing_mode,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=False)
    model.eval()
    all_p, all_g = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            rPPG, _, _, _ = model(inputs, gra_sharp=2.0)
            rPPG = (rPPG - rPPG.mean(-1, keepdim=True)) / (rPPG.std(-1, keepdim=True) + 1e-8)
            all_p.append(rPPG.cpu().numpy()); all_g.append(labels.numpy())
    preds = np.concatenate(all_p); gts = np.concatenate(all_g)
    m = evaluate_per_subject(preds, gts, loader.dataset.samples,
                             fs=30, diff_flag=True, low_pass=0.75, high_pass=2.5)
    # Get per-subject HRs to compute test HR distribution
    sigs = get_subject_signals(preds, gts, loader.dataset.samples,
                               fs=30, diff_flag=True, low_pass=0.75, high_pass=2.5)
    pred_hrs = np.array([s['hr_pred'] for s in sigs.values()])
    gt_hrs = np.array([s['hr_gt'] for s in sigs.values()])
    print(f"\n=== {label} ===")
    print(f"  ckpt: {ckpt}")
    print(f"  Test n_subjects: {m['n_subjects']}")
    print(f"  Test GT HR: mean={gt_hrs.mean():.2f}, std={gt_hrs.std():.2f}, "
          f"range=[{gt_hrs.min():.1f}, {gt_hrs.max():.1f}]")
    print(f"  --- Paper-reportable metrics (per-subject) ---")
    print(f"  MAE         = {m['MAE_bpm']:.4f} BPM")
    print(f"  RMSE        = {m['RMSE_bpm']:.4f} BPM")
    print(f"  MAPE        = {m['MAPE_pct']:.4f} %")
    print(f"  Pearson(HR) = {m['Pearson']:.4f}")
    print(f"  signal_Pearson = {m['signal_Pearson_mean']:.4f}")
    return m


def main():
    print("=" * 75)
    print("BiPulseFormer — Per-subject (paper-comparable) MAPE 계산")
    print("=" * 75)

    targets = [
        # (label, ckpt, dataset, path, split_range, pure_mode)
        ("UBFC intra 60/40 RhythmFormer protocol (valid=test, 10 ep, StepLR) E8",
         "results/intra_ubfc_bipulseformer/checkpoints/UBFC-rPPG_to_UBFC-rPPG_epoch8.pt",
         "UBFC-rPPG", "D:\\UBFC-rPPG", (0.6, 1.0), 'subject_exclusive'),
        ("UBFC intra 7/1/2 (separate valid, 20 ep, OneCycleLR) E10",
         "results/intra_ubfc_bipulseformer_712_oc20/checkpoints/UBFC-rPPG_epoch10.pt",
         "UBFC-rPPG", "D:\\UBFC-rPPG", (0.8, 1.0), 'subject_exclusive'),
        ("PURE intra 7/1/2 random (separate valid, 20 ep, OneCycleLR) E11",
         "results/intra_pure_bipulseformer_712_oc20/checkpoints/PURE_epoch11.pt",
         "PURE", "D:\\PURE", (0.8, 1.0), 'subject_exclusive_random'),
        ("PURE intra 80/20 (valid=test, 10 ep, StepLR) E9",
         "results/intra_pure_bipulseformer_80_20/checkpoints/PURE_to_PURE_epoch9.pt",
         "PURE", "D:\\PURE", (0.8, 1.0), 'subject_exclusive'),
    ]

    results = []
    for label, ckpt, ds, path, sr, pm in targets:
        if not os.path.exists(ckpt):
            print(f"\n[!] missing: {ckpt}"); continue
        try:
            m = eval_ckpt(ckpt, ds, path, sr, label, pure_mode=pm)
            results.append((label, m))
        except Exception as e:
            print(f"[!] {label} failed: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "=" * 90)
    print(" FINAL SUMMARY (per-subject, paper-comparable)")
    print("=" * 90)
    print(f"{'Setup':<70} | {'MAE':>6} | {'RMSE':>6} | {'MAPE%':>6} | {'Pearson':>7}")
    print("-" * 120)
    for lbl, m in results:
        print(f"{lbl:<70} | {m['MAE_bpm']:>5.3f}  | {m['RMSE_bpm']:>5.3f}  | "
              f"{m['MAPE_pct']:>5.3f}  | {m['Pearson']:>7.4f}")


if __name__ == '__main__':
    main()
