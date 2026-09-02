"""For each 7/1/2 run, compute per-clip MAE on:
  (a) VALID-best epoch (이미 선택됨, 다시 출력)
  (b) TEST-best epoch (per-subject MAE 가 가장 낮았던 epoch)
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


def infer(model, loader, device):
    all_p, all_g = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            rPPG, _, _, _ = model(inputs, gra_sharp=2.0)
            rPPG = (rPPG - rPPG.mean(-1, keepdim=True)) / (rPPG.std(-1, keepdim=True) + 1e-8)
            all_p.append(rPPG.cpu().numpy()); all_g.append(labels.numpy())
    return np.concatenate(all_p), np.concatenate(all_g)


def eval_ckpt(ckpt, dataset, path, split_range, pure_mode='subject_exclusive'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    common = dict(face_crop=True, dynamic_detection_freq=0,
                  data_type='diff_normalized', num_workers=4, pin_memory=True)
    if dataset == 'PURE':
        common['pure_split_mode'] = pure_mode
    loader = get_dataloader(dataset, path, batch_size=4, clip_len=160,
                            shuffle=False, chunk_step=80,
                            split_range=split_range, **common)
    model = load_model(ckpt, device)
    preds, gts = infer(model, loader, device)
    pc = evaluate_per_clip(preds, gts, fs=30, diff_flag=True,
                           low_pass=0.75, high_pass=2.5)
    ps = evaluate_per_subject(preds, gts, loader.dataset.samples,
                              fs=30, diff_flag=True, low_pass=0.75, high_pass=2.5)
    return pc, ps


def main():
    print("=" * 75)
    print("BiPhysFormer 7/1/2 — VALID-best vs TEST-best epoch")
    print("=" * 75)

    configs = [
        ('PURE 7/1/2', 'results/intra_pure_biphysformer_712',
         'PURE', 'D:\\PURE', (0.8, 1.0)),
        ('UBFC 7/1/2', 'results/intra_ubfc_biphysformer_712',
         'UBFC-rPPG', 'D:\\UBFC-rPPG', (0.8, 1.0)),
    ]

    for label, result_dir, ds, path, sr in configs:
        with open(os.path.join(result_dir, 'summary.json')) as f:
            summary = json.load(f)
        # support both old (list) and new (dict)
        d = summary[0] if isinstance(summary, list) else summary
        valid_best_ep = d['best_epoch']
        # Find test-best by per-subject MAE
        test_maes = [(h['epoch'], h['test']['MAE_bpm']) for h in d['history']]
        test_best_ep, test_best_mae = min(test_maes, key=lambda x: x[1])

        print(f"\n=== {label} ===")
        print(f"  VALID-best: E{valid_best_ep}")
        print(f"  TEST-best:  E{test_best_ep}  (per-subj MAE {test_best_mae:.3f})")

        for tag, ep in [('VALID-best', valid_best_ep), ('TEST-best', test_best_ep)]:
            if ds == 'PURE':
                ckpt = f'{result_dir}/checkpoints/PURE_epoch{ep}.pt'
            else:
                ckpt = f'{result_dir}/checkpoints/UBFC-rPPG_epoch{ep}.pt'
            if not os.path.exists(ckpt):
                print(f"  [!] missing: {ckpt}"); continue
            pc, ps = eval_ckpt(ckpt, ds, path, sr)
            print(f"\n  ----- {tag} (E{ep}) -----")
            print(f"  per-clip   MAE  = {pc['MAE_bpm_clip']:.4f} BPM")
            print(f"             RMSE = {pc['RMSE_bpm_clip']:.4f} BPM")
            print(f"             Pearson = {pc['Pearson_clip']:.4f}")
            print(f"  per-subj   MAE  = {ps['MAE_bpm']:.4f} BPM")
            print(f"             RMSE = {ps['RMSE_bpm']:.4f} BPM")
            print(f"             Pearson = {ps['Pearson']:.4f}")
            print(f"             signal_Pearson = {ps['signal_Pearson_mean']:.4f}")


if __name__ == '__main__':
    main()
