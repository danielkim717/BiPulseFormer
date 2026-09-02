"""Cross-dataset BiPulseFormer — fft_power routing, fps 버그 수정 후 재학습.

run_cross_82_phase12.py 와 완전히 동일한 프로토콜(n_win=(1,4,4), topk=4,
routing_mode='fft_power', StepLR 15ep)이지만, src/models/bipulseformer.py 의
_fft_power_region fps 버그(ViT_BiPulseFormer 가 raw fps 를 그대로 BiLevelRoutingAttention
에 넘겨서 HR-band FFT 마스크가 실제로는 [0.7,3.0]/patch_stride Hz 대역을 선택하던
문제)가 수정된 이후 처음부터 다시 학습한 버전이다.

phase12 의 기존 결과(results/cross_82_*_phase12/)는 버그 있는 축으로 학습된 비교
기준선으로 그대로 보존하고, 이 스크립트는 별도 결과 디렉토리(*_fftfix)에 저장해서
덮어쓰지 않는다.

목표 (.claude/plans/encapsulated-booping-micali.md Phase 2):
  MAE_subj/Pearson_subj 가 기존 phase12 floor 이상이어야 함
    PURE→UBFC: MAE ≤ 8.224, Pearson ≥ 0.526
    UBFC→PURE: MAE ≤ 11.605, Pearson ≥ 0.518
"""
import os, sys, io, json, random, re, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import torch
import torch.optim as optim

from src.models.bipulseformer import ViT_BiPulseFormer
from src.data.rppg_dataset import get_dataloader
from src.train import NegPearsonLoss, FrequencyLoss
from src.evaluation import evaluate_per_subject
from src.evaluation_per_clip import evaluate_per_clip


# 학습 셋업 (phase12 와 동일)
EPOCHS = 15
BATCH_SIZE = 4
LR = 1e-4
WD = 5e-5
STEP_SIZE = 50
GAMMA = 0.5
ALPHA = 1.0
BETA = 1.0
GRA_SHARP = 2.0
DETECTION_FREQ = 0
GRAD_CLIP = 1.0
SEED = 42
FPS = 30
PURE_PATH = 'D:\\PURE'
UBFC_PATH = 'D:\\UBFC-rPPG'

N_WIN = (1, 4, 4)
TOPK = 4
ROUTING_MODE = 'fft_power'


def _seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_hr_welch(y, sr=30, hr_min=30, hr_max=180):
    from scipy.signal import welch
    y = np.asarray(y, dtype=np.float64)
    if np.std(y) < 1e-9:
        return 0.0
    p, q = welch(y, sr, nfft=1e5 / sr, nperseg=int(np.min((len(y) - 1, 256))))
    mask = (p > hr_min / 60) & (p < hr_max / 60)
    if not mask.any():
        return 0.0
    return float(p[mask][np.argmax(q[mask])] * 60)


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
        'MAPE_clip': pc['MAPE_pct_clip'],
        'Pearson_clip': pc['Pearson_clip'], 'n_clips': pc['n_clips'],
        'MAE_subj': ps['MAE_bpm'], 'RMSE_subj': ps['RMSE_bpm'],
        'MAPE_subj': ps['MAPE_pct'],
        'Pearson_subj': ps['Pearson'], 'signal_Pearson': ps['signal_Pearson_mean'],
        'n_subjects': ps['n_subjects'],
    }, preds, gts


def run_cross(src_name, src_path, tgt_name, tgt_path, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    log_file = os.path.join(result_dir, 'log.txt')
    open(log_file, 'w', encoding='utf-8').close()

    label = f"{src_name} -> {tgt_name}"
    safe_label = label.replace(' -> ', '_to_').replace(' ', '')

    def log(msg):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
        print(msg, flush=True)

    log("=" * 70)
    log(f"[*] {label}  (fft_power routing, fps-bug FIXED: n_win={N_WIN}, topk={TOPK})")
    log("=" * 70)
    log(f"  StepLR(step={STEP_SIZE}, gamma={GAMMA}) → 15 epoch 동안 LR={LR} constant")
    log(f"  Loss: constant α={ALPHA}, β={BETA}")
    log(f"  routing_mode='{ROUTING_MODE}', token_fps = {FPS}/patches[0] (fps 버그 수정됨)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    src_kw = dict(face_crop=True, dynamic_detection_freq=DETECTION_FREQ,
                  data_type='diff_normalized', fps=FPS)
    tgt_kw = dict(face_crop=True, dynamic_detection_freq=DETECTION_FREQ,
                  data_type='diff_normalized')
    if src_name == 'PURE':
        src_kw['pure_split_mode'] = 'subject_exclusive'
    if tgt_name == 'PURE':
        tgt_kw['pure_split_mode'] = 'subject_exclusive'

    train_loader = get_dataloader(src_name, src_path, BATCH_SIZE, clip_len=160,
                                  shuffle=True, random_hflip=True, hr_filter=False,
                                  split_range=(0.0, 0.8), **src_kw)
    valid_loader = get_dataloader(src_name, src_path, BATCH_SIZE, clip_len=160,
                                  shuffle=False, chunk_step=80,
                                  split_range=(0.8, 1.0), **src_kw)
    test_loader = get_dataloader(tgt_name, tgt_path, BATCH_SIZE, clip_len=160,
                                 shuffle=False, chunk_step=80,
                                 split_range=None, **tgt_kw)
    log(f"  train clips: {len(train_loader.dataset)}  (80% of {src_name})")
    log(f"  valid clips: {len(valid_loader.dataset)}  (20% of {src_name})")
    log(f"  test  clips: {len(test_loader.dataset)}  (ENTIRE {tgt_name})")

    model = ViT_BiPulseFormer(
        patches=(4, 4, 4), dim=96, ff_dim=144, num_heads=4, num_layers=12,
        dropout_rate=0.1, theta=0.7, image_size=(160, 128, 128),
        n_win=N_WIN, topk=TOPK, routing_mode=ROUTING_MODE, fps=FPS,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    pearson_criterion = NegPearsonLoss()
    freq_criterion = FrequencyLoss(fps=FPS)
    log(f"  model params: {sum(p.numel() for p in model.parameters())}")

    best_valid_rmse = float('inf')
    best_test = None
    best_epoch = 0
    history = []

    for epoch in range(EPOCHS):
        a, b = ALPHA, BETA
        log(f"\n[*] Epoch {epoch+1}/{EPOCHS}  alpha={a}, beta={b}")
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
            loss = a * loss_p + b * (loss_ce + loss_ld)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += float(loss.item()); nb += 1
        scheduler.step()
        avg_loss = epoch_loss / max(1, nb)
        current_lr = optimizer.param_groups[0]['lr']

        valid_m, _, _ = evaluate_loader(model, valid_loader, device)
        test_m, _, _ = evaluate_loader(model, test_loader, device)

        ckpt_dir = os.path.join(result_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f'{safe_label}_epoch{epoch+1}.pt'))

        history.append({'epoch': epoch + 1, 'avg_loss': avg_loss, 'lr': current_lr,
                        'valid': valid_m, 'test': test_m})
        log(f"\nEpoch {epoch+1}/{EPOCHS}: Train loss {avg_loss:.4f}  LR={current_lr:.2e}")
        log(f"  VALID per-clip MAE {valid_m['MAE_clip']:.3f}  RMSE {valid_m['RMSE_clip']:.3f}  "
            f"Pearson {valid_m['Pearson_clip']:.4f}")
        log(f"  TEST  per-subj MAE {test_m['MAE_subj']:.3f}  RMSE {test_m['RMSE_subj']:.3f}  "
            f"MAPE {test_m['MAPE_subj']:.3f}%  Pearson {test_m['Pearson_subj']:.4f}  "
            f"(n_subj={test_m['n_subjects']})")

        if valid_m['RMSE_clip'] < best_valid_rmse:
            best_valid_rmse = valid_m['RMSE_clip']
            best_test = test_m
            best_epoch = epoch + 1

    log(f"\n-> Best for {label}: epoch {best_epoch}  (valid per-clip RMSE {best_valid_rmse:.3f})")
    log(f"   per-subj  MAE {best_test['MAE_subj']:.3f}  RMSE {best_test['RMSE_subj']:.3f}  "
        f"MAPE {best_test['MAPE_subj']:.3f}%  Pearson {best_test['Pearson_subj']:.4f}")

    with open(os.path.join(result_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump({'name': label, 'best_epoch': best_epoch,
                   'best': best_test, 'history': history,
                   'config': {'n_win': N_WIN, 'topk': TOPK, 'routing_mode': ROUTING_MODE,
                              'fps_bug_fixed': True}}, f, indent=2)
    return best_test, best_epoch


def main():
    parser = argparse.ArgumentParser(description='Cross 8:2 BiPulseFormer fft_power (fps-fixed)')
    parser.add_argument('--direction', choices=['pure2ubfc', 'ubfc2pure', 'both'],
                       default='both')
    args = parser.parse_args()
    _seed_everything(SEED)
    print(f"[*] Cross 8:2 BiPulseFormer fft_power fps-fixed (n_win={N_WIN}, topk={TOPK})")
    print(f"    direction={args.direction}")

    experiments = []
    if args.direction in ('pure2ubfc', 'both'):
        experiments.append(('PURE', PURE_PATH, 'UBFC-rPPG', UBFC_PATH,
                           'results/cross_82_pure_to_ubfc_fftfix'))
    if args.direction in ('ubfc2pure', 'both'):
        experiments.append(('UBFC-rPPG', UBFC_PATH, 'PURE', PURE_PATH,
                           'results/cross_82_ubfc_to_pure_fftfix'))

    results = []
    for src, srp, tgt, tgp, out in experiments:
        try:
            best, be = run_cross(src, srp, tgt, tgp, out)
            results.append((f"{src} -> {tgt}", best, be))
        except Exception as e:
            print(f"[!] {src}->{tgt} failed: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "=" * 80)
    print("[*] Final cross 8:2 fft_power fps-fixed — per-subject (paper-comparable)")
    print("=" * 80)
    for name, b, e in results:
        print(f"  {name}: MAE={b['MAE_subj']:.3f}  RMSE={b['RMSE_subj']:.3f}  "
              f"MAPE={b['MAPE_subj']:.3f}%  Pearson={b['Pearson_subj']:.4f}  (E{e})")


if __name__ == '__main__':
    main()
