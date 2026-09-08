"""Cross-dataset BiPulseFormer — 8:2 protocol, StepLR + 15 epochs, constant α=β=1.0,
   diff_routing (STE) FIX 검증용 재학습.

run_cross_82_steplr15.py 와 하이퍼파라미터는 100% 동일 (StepLR(50,0.5), 15ep, α=β=1.0).
유일한 차이: src/models/bipulseformer.py 의 BiLevelRoutingAttention_TDC_gra_sharp 가
이제 diff_routing=True(default) — 학습 중 top-k 선택에 STE gradient 가 흐름
(기존 run_cross_82_steplr15.py 결과는 diff_routing 이전 코드로 학습된 것이므로
결과 폴더를 분리해 A/B 비교 가능하게 함).

결과 폴더: results/cross_82_{pure_to_ubfc,ubfc_to_pure}_diffrouting
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
        'MAPE_clip': pc['MAPE_pct_clip'],
        'Pearson_clip': pc['Pearson_clip'], 'n_clips': pc['n_clips'],
        'MAE_subj': ps['MAE_bpm'], 'RMSE_subj': ps['RMSE_bpm'],
        'MAPE_subj': ps['MAPE_pct'],
        'Pearson_subj': ps['Pearson'], 'signal_Pearson': ps['signal_Pearson_mean'],
        'n_subjects': ps['n_subjects'],
    }, preds, gts


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
    log(f"[*] {label}  (StepLR + 15ep, const α=β=1, diff_routing=True STE FIX)")
    log("=" * 70)
    log(f"  StepLR(step={STEP_SIZE}, gamma={GAMMA}) → 15 epoch 동안 LR={LR} constant")
    log(f"  Loss: constant α={ALPHA}, β={BETA}")
    log(f"  diff_routing=True, routing_tau=0.5 (STE gradient to BiLevel routing)")

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
                                  shuffle=True, random_hflip=True, hr_filter=True,
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
        n_win=(2, 2, 2), topk=4, diff_routing=True, routing_tau=0.5,
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
        test_m, test_preds, test_gts = evaluate_loader(model, test_loader, device)

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
                   'config': {'n_win': [2, 2, 2], 'topk': 4, 'routing_mode': 'mean',
                              'diff_routing': True, 'routing_tau': 0.5}}, f, indent=2)
    return best_test, best_epoch


def main():
    parser = argparse.ArgumentParser(description='Cross 8:2 BiPulseFormer diff_routing STE fix 검증')
    parser.add_argument('--direction', choices=['ubfc2pure', 'pure2ubfc', 'both'],
                       default='both')
    args = parser.parse_args()
    _seed_everything(SEED)
    print(f"[*] Cross 8:2 BiPulseFormer (diff_routing STE fix)  direction={args.direction}")

    experiments = []
    if args.direction in ('pure2ubfc', 'both'):
        experiments.append(('PURE', PURE_PATH, 'UBFC-rPPG', UBFC_PATH,
                           'results/cross_82_pure_to_ubfc_diffrouting'))
    if args.direction in ('ubfc2pure', 'both'):
        experiments.append(('UBFC-rPPG', UBFC_PATH, 'PURE', PURE_PATH,
                           'results/cross_82_ubfc_to_pure_diffrouting'))

    results = []
    for src, srp, tgt, tgp, out in experiments:
        try:
            best, be = run_cross(src, srp, tgt, tgp, out)
            results.append((f"{src} -> {tgt}", best, be))
        except Exception as e:
            print(f"[!] {src}->{tgt} failed: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "=" * 80)
    print("[*] Final cross 8:2 diff_routing fix — per-subject (paper-comparable)")
    print("=" * 80)
    for name, b, e in results:
        print(f"  {name}: MAE={b['MAE_subj']:.3f}  RMSE={b['RMSE_subj']:.3f}  "
              f"MAPE={b['MAPE_subj']:.3f}%  Pearson={b['Pearson_subj']:.4f}  (E{e})")


if __name__ == '__main__':
    main()
