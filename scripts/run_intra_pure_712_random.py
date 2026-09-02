"""BiPulseFormer PURE intra 7/1/2 — RANDOM subject assignment (seed=42).

이전 sort-based 와 달리 subject 07 (high-HR outlier) 위치가 다름:
  TRAIN (0.0-0.7): subjects 03,04,06,07,08,09,10  (subject 07 train 포함)
  VALID (0.7-0.8): subject 05
  TEST  (0.8-1.0): subjects 01,02 (normal HR 73-74 BPM)
"""
import os, sys, io, json, time, random
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import torch
import torch.optim as optim
from scipy.signal import welch

from src.models.bipulseformer import ViT_BiPulseFormer
from src.data.rppg_dataset import get_dataloader
from src.train import NegPearsonLoss, FrequencyLoss
from src.evaluation import evaluate_per_subject, get_subject_signals
from src.evaluation_per_clip import evaluate_per_clip


def get_hr_welch(y, sr=30, hr_min=30, hr_max=180):
    y = np.asarray(y, dtype=np.float64)
    if np.std(y) < 1e-9:
        return 0.0
    p, q = welch(y, sr, nfft=1e5 / sr, nperseg=int(np.min((len(y) - 1, 256))))
    mask = (p > hr_min / 60) & (p < hr_max / 60)
    if not mask.any():
        return 0.0
    return float(p[mask][np.argmax(q[mask])] * 60)


EPOCHS = 10
BATCH_SIZE = 4
LR = 1e-4
WD = 5e-5
ALPHA = 1.0
BETA = 1.0
GRA_SHARP = 2.0
DETECTION_FREQ = 0
GRAD_CLIP = 1.0
SEED = 42
FPS = 30
PURE_PATH = 'D:\\PURE'

RESULT_DIR = 'results/intra_pure_biphysformer_712_random'


def per_clip_pearson(pred, gt):
    out = []
    for i in range(pred.shape[0]):
        p = pred[i] - pred[i].mean(); g = gt[i] - gt[i].mean()
        denom = (np.sqrt((p*p).sum()) * np.sqrt((g*g).sum())) + 1e-9
        out.append(float((p*g).sum() / denom))
    return float(np.mean(out)) if out else 0.0


def _seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def evaluate_loader(model, loader, device):
    model.eval()
    all_p, all_g = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            rPPG, _, _, _ = model(inputs, gra_sharp=GRA_SHARP)
            rPPG = (rPPG - torch.mean(rPPG, dim=-1, keepdim=True)) / \
                   (torch.std(rPPG, dim=-1, keepdim=True) + 1e-8)
            all_p.append(rPPG.cpu().numpy()); all_g.append(labels.numpy())
    preds = np.concatenate(all_p); gts = np.concatenate(all_g)
    pc = evaluate_per_clip(preds, gts, fs=FPS, diff_flag=True,
                           low_pass=0.75, high_pass=2.5)
    ps = evaluate_per_subject(preds, gts, loader.dataset.samples,
                              fs=FPS, diff_flag=True, low_pass=0.75, high_pass=2.5)
    return {
        'MAE_clip': pc['MAE_bpm_clip'], 'RMSE_clip': pc['RMSE_bpm_clip'],
        'Pearson_clip': pc['Pearson_clip'], 'n_clips': pc['n_clips'],
        'MAE_subj': ps['MAE_bpm'], 'RMSE_subj': ps['RMSE_bpm'],
        'Pearson_subj': ps['Pearson'], 'signal_Pearson': ps['signal_Pearson_mean'],
        'n_subjects': ps['n_subjects'],
    }, preds, gts


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    log_file = os.path.join(RESULT_DIR, 'log.txt')
    def log(msg):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
        print(msg, flush=True)
    open(log_file, 'w', encoding='utf-8').close()

    _seed_everything(SEED)
    log(f"[*] BiPhysFormer PURE intra 7/1/2 RANDOM (seed={SEED})")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    common = dict(face_crop=True, dynamic_detection_freq=DETECTION_FREQ,
                  data_type='diff_normalized',
                  pure_split_mode='subject_exclusive_random')
    train_loader = get_dataloader('PURE', PURE_PATH, BATCH_SIZE, clip_len=160,
                                  shuffle=True, random_hflip=True, hr_filter=True, fps=FPS,
                                  split_range=(0.0, 0.7), **common)
    valid_loader = get_dataloader('PURE', PURE_PATH, BATCH_SIZE, clip_len=160,
                                  shuffle=False, chunk_step=80,
                                  split_range=(0.7, 0.8), **common)
    test_loader = get_dataloader('PURE', PURE_PATH, BATCH_SIZE, clip_len=160,
                                 shuffle=False, chunk_step=80,
                                 split_range=(0.8, 1.0), **common)
    log(f"  train clips: {len(train_loader.dataset)} (70% of PURE, random subjects)")
    log(f"  valid clips: {len(valid_loader.dataset)} (10%)")
    log(f"  test  clips: {len(test_loader.dataset)} (20%)")

    model = ViT_BiPulseFormer(
        patches=(4, 4, 4), dim=96, ff_dim=144, num_heads=4, num_layers=12,
        dropout_rate=0.1, theta=0.7, image_size=(160, 128, 128),
        n_win=(2, 2, 2), topk=4,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    pearson_criterion = NegPearsonLoss()
    freq_criterion = FrequencyLoss(fps=FPS)
    log(f"  model params: {sum(p.numel() for p in model.parameters())}")

    best_valid_rmse = float('inf')
    best_test = None
    best_epoch = 0
    history = []

    for epoch in range(EPOCHS):
        log(f"\n[*] Epoch {epoch+1}/{EPOCHS}")
        model.train()
        epoch_loss, nb = 0.0, 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            hr_target = torch.tensor(
                [get_hr_welch(labels[k].cpu().numpy(), sr=FPS) for k in range(labels.shape[0])],
                dtype=torch.float32, device=device,
            )
            optimizer.zero_grad()
            rPPG, _, _, _ = model(inputs, gra_sharp=GRA_SHARP)
            rPPG = (rPPG - torch.mean(rPPG, dim=-1, keepdim=True)) / \
                   (torch.std(rPPG, dim=-1, keepdim=True) + 1e-8)
            loss_p = pearson_criterion(rPPG, labels)
            loss_ce, loss_ld = freq_criterion(rPPG, hr_target)
            loss = ALPHA * loss_p + BETA * (loss_ce + loss_ld)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += float(loss.item()); nb += 1
        scheduler.step()
        avg_loss = epoch_loss / max(1, nb)

        valid_m, _, _ = evaluate_loader(model, valid_loader, device)
        test_m, test_preds, test_gts = evaluate_loader(model, test_loader, device)

        ckpt_dir = os.path.join(RESULT_DIR, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f'PURE_epoch{epoch+1}.pt'))

        history.append({'epoch': epoch + 1, 'avg_loss': avg_loss,
                        'valid': valid_m, 'test': test_m})
        log(f"\nEpoch {epoch+1}/{EPOCHS}: Train loss {avg_loss:.4f}")
        log(f"  VALID per-clip MAE {valid_m['MAE_clip']:.3f}  per-subj MAE {valid_m['MAE_subj']:.3f}")
        log(f"  TEST  per-clip MAE {test_m['MAE_clip']:.3f}  RMSE {test_m['RMSE_clip']:.3f}  "
            f"Pearson {test_m['Pearson_clip']:.4f}  (n_clips={test_m['n_clips']})")
        log(f"  TEST  per-subj MAE {test_m['MAE_subj']:.3f}  RMSE {test_m['RMSE_subj']:.3f}  "
            f"Pearson {test_m['Pearson_subj']:.4f}  sig_P {test_m['signal_Pearson']:.4f}  "
            f"(n_subj={test_m['n_subjects']})")

        if valid_m['RMSE_clip'] < best_valid_rmse:
            best_valid_rmse = valid_m['RMSE_clip']
            best_test = test_m
            best_epoch = epoch + 1

    log(f"\n-> Best for PURE intra 7/1/2 random: epoch {best_epoch}")
    log(f"   per-clip  MAE {best_test['MAE_clip']:.3f}  RMSE {best_test['RMSE_clip']:.3f}  "
        f"Pearson {best_test['Pearson_clip']:.4f}")
    log(f"   per-subj  MAE {best_test['MAE_subj']:.3f}  Pearson {best_test['Pearson_subj']:.4f}")

    with open(os.path.join(RESULT_DIR, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump({'name': 'PURE intra 7/1/2 random',
                   'best_epoch': best_epoch, 'best': best_test,
                   'history': history}, f, indent=2)


if __name__ == '__main__':
    main()
