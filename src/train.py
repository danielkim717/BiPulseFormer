"""
Loss functions used by the active training pipeline.

NegPearsonLoss + FrequencyLoss — rPPG-Toolbox PhysFormer recipe 와 동일하게 정렬.

Reference (rPPG-Toolbox):
  - PhysFormerTrainer.py: https://github.com/ubicomplab/rPPG-Toolbox/blob/main/neural_methods/trainer/PhysFormerTrainer.py
  - PhysFormerLossComputer.py: https://github.com/ubicomplab/rPPG-Toolbox/blob/main/neural_methods/loss/PhysFormerLossComputer.py
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NegPearsonLoss(nn.Module):
    """rPPG-Toolbox PhysNetNegPearsonLoss 와 동일.
    Pearson 의 (1 - r) 를 클립별 평균. label/pred 모두 (B, T)."""
    def __init__(self):
        super().__init__()

    def forward(self, preds, labels):
        preds_mean = torch.mean(preds, dim=1, keepdim=True)
        labels_mean = torch.mean(labels, dim=1, keepdim=True)

        preds_std = preds - preds_mean
        labels_std = labels - labels_mean

        cov = torch.sum(preds_std * labels_std, dim=1)
        var_preds = torch.sqrt(torch.sum(preds_std ** 2, dim=1) + 1e-8)
        var_labels = torch.sqrt(torch.sum(labels_std ** 2, dim=1) + 1e-8)

        pearson = cov / (var_preds * var_labels)
        return 1.0 - torch.mean(pearson)


# =============================================================================
# rPPG-Toolbox PhysFormerLossComputer 재현 (per-sample 호출 방식)
# =============================================================================

def _normal_sampling(mean, label_k, std):
    """Gaussian PDF (un-normalized). rPPG-Toolbox normal_sampling 과 동일."""
    return math.exp(-(label_k - mean) ** 2 / (2 * std ** 2)) / (math.sqrt(2 * math.pi) * std)


def _kl_loss(inputs, labels):
    """rPPG-Toolbox kl_loss: KLDivLoss(reduction='sum') with log_softmax(inputs).
    inputs: (140,) frequency distribution
    labels: (140,) target gaussian distribution"""
    labels = labels.view(1, -1)
    criterion = nn.KLDivLoss(reduction='sum')
    return criterion(F.log_softmax(inputs, dim=-1), labels)


def _compute_complex_absolute_given_k(output, k, N):
    """Hanning-windowed DFT magnitude squared at frequencies k.
    output: (T,)   k: (num_bpm,)   →  (1, num_bpm)
    rPPG-Toolbox compute_complex_absolute_given_k 과 동일 (per-sample)."""
    device = output.device
    two_pi_n_over_N = 2.0 * math.pi * torch.arange(0, N, dtype=torch.float32, device=device) / N
    hanning = torch.from_numpy(np.hanning(N).astype(np.float32)).to(device).view(1, -1)
    windowed = output.view(1, -1) * hanning            # (1, N)
    windowed = windowed.view(1, 1, -1)                 # (1, 1, N)
    k = k.view(1, -1, 1)                               # (1, num_bpm, 1)
    two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)   # (1, 1, N)
    sin_part = torch.sum(windowed * torch.sin(k * two_pi_n_over_N), dim=-1)
    cos_part = torch.sum(windowed * torch.cos(k * two_pi_n_over_N), dim=-1)
    return sin_part ** 2 + cos_part ** 2               # (1, num_bpm)


def _complex_absolute_one(output, fps, bpm_range):
    """Single-sample softmax-like normalized power spectrum over bpm_range."""
    N = output.size(-1)
    unit_per_hz = fps / N
    feasible_bpm = bpm_range / 60.0
    k = feasible_bpm / unit_per_hz
    ca = _compute_complex_absolute_given_k(output, k, N)
    return (1.0 / (ca.sum() + 1e-12)) * ca              # (1, num_bpm)  sum=1


def cross_entropy_power_spectrum_DLDL_softmax2(rppg, target_hr_bpm, fps, std=1.0,
                                               bpm_low=40, bpm_high=180):
    """rPPG-Toolbox cross_entropy_power_spectrum_DLDL_softmax2 와 동일.

    rppg: (T,) — 단일 샘플 rPPG signal (이미 정규화됨)
    target_hr_bpm: scalar tensor — Welch peak HR (BPM, e.g. 72.5)
    fps: scalar (e.g. 30)

    Returns: (loss_distribution_kl, loss_ce, hr_mae)
    """
    device = rppg.device
    target = target_hr_bpm.view(1, -1) if target_hr_bpm.dim() == 0 else target_hr_bpm.view(1, -1)
    int_target = int(target_hr_bpm.item())

    # target_distribution: Gaussian centered at HR_BPM over bins [40, 180) BPM (140 bins)
    target_dist = [_normal_sampling(int_target, i, std) for i in range(bpm_low, bpm_high)]
    target_dist = [v if v > 1e-15 else 1e-15 for v in target_dist]
    target_dist = torch.tensor(target_dist, dtype=torch.float32, device=device)

    bpm_range = torch.arange(bpm_low, bpm_high, dtype=torch.float32, device=device)
    ca = _complex_absolute_one(rppg, fps, bpm_range)        # (1, 140), sum=1

    fre_distribution = ca / torch.sum(ca)                   # already sum=1, idempotent
    loss_kl = _kl_loss(fre_distribution, target_dist)

    whole_max_idx = ca.view(-1).argmax().type(torch.float32)
    target_idx = (target - bpm_low).view(1).type(torch.long)
    loss_ce = F.cross_entropy(ca, target_idx)
    hr_mae = torch.abs(target.view(-1)[0] - bpm_low - whole_max_idx)
    return loss_kl, loss_ce, hr_mae


class FrequencyLoss(nn.Module):
    """rPPG-Toolbox PhysFormerTrainer 의 frequency loss 호출 패턴 재현.

    forward(preds, target_hr_bpm) → (loss_ce, loss_ld)
      preds: (B, T) — per-sample 정규화된 rPPG (학습 loop 에서 이미 정규화됨)
      target_hr_bpm: (B,) — label 신호에서 Welch periodogram 으로 추출한 HR (BPM)

    내부적으로 sample 별로 cross_entropy_power_spectrum_DLDL_softmax2 호출 후 평균.
    """
    def __init__(self, fps=30, bpm_low=40, bpm_high=180, std=1.0):
        super().__init__()
        self.fps = fps
        self.bpm_low = bpm_low
        self.bpm_high = bpm_high
        self.std = std

    def forward(self, preds, target_hr_bpm):
        B = preds.shape[0]
        loss_ce_total = 0.0
        loss_kl_total = 0.0
        for b in range(B):
            loss_kl, loss_ce, _ = cross_entropy_power_spectrum_DLDL_softmax2(
                preds[b], target_hr_bpm[b], self.fps, std=self.std,
                bpm_low=self.bpm_low, bpm_high=self.bpm_high
            )
            loss_ce_total = loss_ce_total + loss_ce
            loss_kl_total = loss_kl_total + loss_kl
        return loss_ce_total / B, loss_kl_total / B
