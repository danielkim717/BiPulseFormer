"""
BiPulseFormer — frequency-guided region routing for video-based pulse estimation.

The reported configuration summarizes temporal FFT magnitudes of learned Q/K
features inside a configured HR band, ranks region affinities, and gathers K/V
tokens from top-k regions for sparse attention at inference. This biases routing
toward pulse-related temporal activity; anatomical relevance is not established
by the routing mechanism alone. See docs/method.md for the executed settings.

원본:
  https://github.com/ZitongYu/PhysFormer/blob/main/model/transformer_layer.py
  https://github.com/ZitongYu/PhysFormer/blob/main/model/physformer.py

차이점 (PhysFormer 원본 대비):
  - Block_ST_TDC_gra_sharp.attn 만 BiLevelRoutingAttention_TDC_gra_sharp 로 교체
  - 그 외 (CDC_T, FFN_ST, Block 의 norm/proj/pwff, Stem0/1/2,
    patch_embedding, transformer1/2/3, upsample, ConvBlockLast,
    init_weights, forward signature) 모두 PhysFormer 원본 그대로

BiFormer-inspired region routing (Zhu et al., CVPR 2023):
  Q,K,V 추출은 동일 (TDC-Q, TDC-K, Conv1x1-V), 단 attention 단계에서
    1) fft_magnitude: spatial mean then HR-band temporal FFT magnitude descriptors;
       mean mode remains available as a separate configuration
    2) q_region @ k_region.T 로 region similarity 계산, 각 query window 마다
       상위 k 개의 key window 만 참조 (top-k routing)
    3) 그 k×win_size 토큰만 key/value 로 사용해 multi-head softmax 수행
  Retains PhysFormer's gra_sharp (=2.0) scale. Matching one parameter does not
  establish full protocol equivalence. STE training uses dense attention tensors.
"""
import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CDC_T — PhysFormer 원본 그대로
# =============================================================================
class CDC_T(nn.Module):
    """Temporal Center-difference based 3D Convolution (CDC_T)."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.6):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        if self.conv.weight.shape[2] > 1:
            kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + \
                          self.conv.weight[:, :, 2, :, :].sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None, None]
            out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias,
                                stride=self.conv.stride, padding=0,
                                dilation=self.conv.dilation, groups=self.conv.groups)
            return out_normal - self.theta * out_diff
        return out_normal


# =============================================================================
# split_last / merge_last — PhysFormer 원본 그대로
# =============================================================================
def split_last(x, shape):
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


# =============================================================================
# BiLevelRoutingAttention_TDC_gra_sharp
#   PhysFormer MHSA_TDC_gra_sharp 의 drop-in replacement.
#   - 입력/출력 shape, return signature, forward(x, gra_sharp) 시그니처
#     모두 원본 MHSA 와 동일.
#   - 컨텐츠 어텐션만 BiLevel Routing 으로 교체.
# =============================================================================
class BiLevelRoutingAttention_TDC_gra_sharp(nn.Module):
    """BiFormer-style BRA, drop-in for PhysFormer MHSA_TDC_gra_sharp.

    routing_mode:
      - 'mean': window mean embedding (default, BiFormer paper)
      - 'fft_magnitude': configured HR-band mean FFT magnitude per window
      - 'fft_power': historical alias for fft_magnitude (not squared power)

    diff_routing (STE):
      Hard top-k(torch.topk)는 미분 불가능해 proj_q/proj_k가 "어떤 region을
      골라야 하는지"에 대한 gradient를 받지 못하고, content-attention 경로로만
      간접 학습된다 (routing_analysis 실측 결과 self/same-quadrant/top-half 비율이
      모두 random 수준). diff_routing=True(default) + self.training 시,
      forward 값은 기존 hard top-k와 동일하게 유지하되 backward gradient만
      soft region log-prior를 사용하는 straight-through surrogate로 전달.
      eval() 시에는 항상 기존 gather 기반 sparse 경로 (체크포인트 호환 + 속도).
    """
    def __init__(self, dim, num_heads, dropout, theta,
                 n_win=(2, 2, 2), topk=4, routing_mode='mean', fps=30,
                 diff_routing=True, routing_tau=0.5, routing_band=(0.7, 3.0)):
        super().__init__()
        if num_heads < 1 or dim % num_heads:
            raise ValueError('dim must be divisible by num_heads')
        if len(n_win) != 3 or any(v < 1 for v in n_win):
            raise ValueError('n_win must contain three positive integers')
        if routing_mode not in ('mean', 'fft_power', 'fft_magnitude'):
            raise ValueError(f'Unknown routing mode: {routing_mode}')
        if topk < 1 or topk > math.prod(n_win) or routing_tau <= 0:
            raise ValueError('Invalid topk or routing temperature')
        if not 0 < routing_band[0] < routing_band[1] < fps / 2:
            raise ValueError('Routing band must be below token Nyquist frequency')
        self.proj_q = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),
            nn.BatchNorm3d(dim),
        )
        self.proj_k = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),
            nn.BatchNorm3d(dim),
        )
        self.proj_v = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),
        )
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.dim = dim
        self.n_win = n_win
        self.topk = topk
        self.routing_mode = routing_mode
        self.fps = fps
        self.diff_routing = diff_routing
        self.routing_tau = routing_tau
        self.routing_band = routing_band
        self.scores = None  # for visualization

    def _window_partition(self, feat_3d, t, h, w):
        """feat_3d: (B, C, t, h, w) → (B, S, win, C), S=n_win 원소의 곱 (window 개수), win=window 당 토큰 수."""
        B, C = feat_3d.shape[:2]
        wt, wh, ww = self.n_win
        if t % wt or h % wh or w % ww:
            raise ValueError('Feature dimensions must be divisible by n_win')
        lt, lh, lw = t // wt, h // wh, w // ww
        feat = feat_3d.view(B, C, wt, lt, wh, lh, ww, lw)
        feat = feat.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()  # (B, wt, wh, ww, lt, lh, lw, C)
        S = wt * wh * ww
        win = lt * lh * lw
        return feat.view(B, S, win, C), (lt, lh, lw)

    def _fft_power_region(self, feat_w, lt, lh, lw):
        """Mean FFT magnitude in routing_band; historical method name retained.

        feat_w: (B, S, win=lt*lh*lw, C) → (B, S, C) magnitude descriptor.
        The pulse-band descriptor guides routing; it is not an anatomical label.
        """
        B, S, win, C = feat_w.shape
        # Reshape to separate temporal/spatial inside window
        feat = feat_w.view(B, S, lt, lh * lw, C)
        # Spatial mean within window → (B, S, lt, C)
        feat_t = feat.mean(dim=3)
        # Temporal FFT along lt → (B, S, lt//2+1, C)
        fft = torch.fft.rfft(feat_t.float(), dim=2).abs()
        # Configured HR-band mask, evaluated using the feature-token FPS.
        freqs = torch.fft.rfftfreq(lt, 1.0 / self.fps).to(feat_w.device)
        hr_mask = (freqs >= self.routing_band[0]) & (freqs <= self.routing_band[1])
        if hr_mask.sum() == 0:
            # Window too short → fallback to mean
            return feat_w.mean(dim=2)
        # Mean magnitude in HR band → (B, S, C), not squared power.
        return fft[:, :, hr_mask].mean(dim=2)

    def _window_reverse(self, x_w, t, h, w, lt, lh, lw):
        """(B, S, win, C) → (B, t*h*w, C)."""
        B = x_w.shape[0]
        C = x_w.shape[-1]
        wt, wh, ww = self.n_win
        x = x_w.view(B, wt, wh, ww, lt, lh, lw, C)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()  # (B, wt, lt, wh, lh, ww, lw, C)
        x = x.view(B, t * h * w, C)
        return x

    def forward(self, x, gra_sharp):
        """x: (B, P, C), P = t*h*w. PhysFormer 원본 패턴 그대로 사용.
        Return: (h, scores) — scores 는 region routing 결과 (B, S, S) 로 반환."""
        B, P, C = x.shape
        # PhysFormer 원본은 P = 16*t (h=w=4) 로 reshape — 우리도 동일.
        x_3d = x.transpose(1, 2).view(B, C, P // 16, 4, 4)
        t, h, w = P // 16, 4, 4

        q_3d = self.proj_q(x_3d)
        k_3d = self.proj_k(x_3d)
        v_3d = self.proj_v(x_3d)

        # Window partition
        q_w, (lt, lh, lw) = self._window_partition(q_3d, t, h, w)   # (B, S, win, C)
        k_w, _ = self._window_partition(k_3d, t, h, w)
        v_w, _ = self._window_partition(v_3d, t, h, w)
        S = q_w.shape[1]
        win = q_w.shape[2]

        # Region routing
        if self.routing_mode in ('fft_power', 'fft_magnitude'):
            # HR-band temporal FFT magnitude descriptor per window.
            q_r = self._fft_power_region(q_w, lt, lh, lw)
            k_r = self._fft_power_region(k_w, lt, lh, lw)
        else:
            # Default: window mean
            q_r = q_w.mean(dim=2)                                   # (B, S, C)
            k_r = k_w.mean(dim=2)
        a_r = q_r @ k_r.transpose(-2, -1) / math.sqrt(C)            # (B, S, S)
        topk = min(self.topk, S)
        H = self.n_heads
        d = C // H

        if self.diff_routing and self.training:
            # STE differentiable routing: forward == hard top-k (below), backward ==
            # gradient of softmax(a_r/tau). Computed as dense attention over ALL S
            # key-windows (not just gathered top-k) with a region-level gate, so the
            # gate tensor participates in the graph the same way for every window.
            _, topk_idx = torch.topk(a_r, k=topk, dim=-1)
            mask_hard = torch.zeros_like(a_r).scatter_(-1, topk_idx, 1.0)   # (B, S, S), no grad

            def split_heads(t_):
                return t_.view(B, S * win, H, d).permute(0, 2, 1, 3)        # (B, H, P, d)
            q_h = split_heads(q_w.reshape(B, S * win, C))
            k_h = split_heads(k_w.reshape(B, S * win, C))
            v_h = split_heads(v_w.reshape(B, S * win, C))
            scores = (q_h @ k_h.transpose(-2, -1)) / gra_sharp             # (B, H, P, P)
            hard_tokens = mask_hard.bool().repeat_interleave(win, dim=1).repeat_interleave(win, dim=2)
            # Normalize ONLY selected logits: dense-softmax then masking can
            # underflow all selected probabilities when an unselected logit wins.
            hard_attn = F.softmax(scores.masked_fill(~hard_tokens.unsqueeze(1), float('-inf')), dim=-1)
            # Straight-through surrogate: forward remains exactly hard routing.
            # The soft region prior carries routing gradients; detached content
            # logits avoid adding a second content-attention gradient path.
            log_prior = F.log_softmax(a_r / self.routing_tau, dim=-1)
            log_prior = log_prior.repeat_interleave(win, dim=1).repeat_interleave(win, dim=2)
            soft_attn = F.softmax(scores.detach() + log_prior.unsqueeze(1), dim=-1)
            attn = hard_attn + (soft_attn - soft_attn.detach())
            scores = self.drop(attn)
            out = scores @ v_h                                             # (B, H, P, d)
            out = out.permute(0, 2, 1, 3).reshape(B, S, win, C)            # back to window-partitioned
        else:
            # Efficient sparse path (eval / diff_routing=False): gather K, V from
            # top-k routed windows only — identical numerics to the branch above.
            _, topk_idx = torch.topk(a_r, k=topk, dim=-1)               # (B, S, topk)
            idx = topk_idx.view(B, S, topk, 1, 1).expand(B, S, topk, win, C)
            k_src = k_w.unsqueeze(1).expand(B, S, S, win, C)
            v_src = v_w.unsqueeze(1).expand(B, S, S, win, C)
            k_g = torch.gather(k_src, 2, idx).view(B, S, topk * win, C)  # (B, S, k*win, C)
            v_g = torch.gather(v_src, 2, idx).view(B, S, topk * win, C)

            q_h = q_w.view(B, S, win, H, d).permute(0, 1, 3, 2, 4)             # (B, S, H, win, d)
            k_h = k_g.view(B, S, topk * win, H, d).permute(0, 1, 3, 2, 4)      # (B, S, H, k*win, d)
            v_h = v_g.view(B, S, topk * win, H, d).permute(0, 1, 3, 2, 4)
            # PhysFormer recipe: scores = q @ k.T / gra_sharp  (NOT /sqrt(d))
            scores = (q_h @ k_h.transpose(-2, -1)) / gra_sharp                  # (B, S, H, win, k*win)
            scores = self.drop(F.softmax(scores, dim=-1))
            out = scores @ v_h                                                  # (B, S, H, win, d)
            out = out.permute(0, 1, 3, 2, 4).contiguous().view(B, S, win, C)    # (B, S, win, C)

        h_out = self._window_reverse(out, t, h, w, lt, lh, lw)              # (B, P, C)
        # Score (region routing) for visualization compatibility
        self.scores = a_r.detach()
        return h_out, a_r


# =============================================================================
# PositionWiseFeedForward_ST — PhysFormer 원본 그대로
# =============================================================================
class PositionWiseFeedForward_ST(nn.Module):
    """1x1 Conv → BN → ELU → depthwise 3x3 STConv → BN → ELU → 1x1 Conv → BN."""
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Conv3d(dim, ff_dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )
        self.STConv = nn.Sequential(
            nn.Conv3d(ff_dim, ff_dim, 3, stride=1, padding=1, groups=ff_dim, bias=False),
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv3d(ff_dim, dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(dim),
        )

    def forward(self, x):
        B, P, C = x.shape
        x = x.transpose(1, 2).view(B, C, P // 16, 4, 4)
        x = self.fc1(x)
        x = self.STConv(x)
        x = self.fc2(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# =============================================================================
# Block_ST_TDC_gra_sharp_Bi — PhysFormer 원본 Block 에서 attn 만 BRA 로 교체
# =============================================================================
class Block_ST_TDC_gra_sharp_Bi(nn.Module):
    """Transformer Block (BiLevel Routing Attention 적용)."""
    def __init__(self, dim, num_heads, ff_dim, dropout, theta,
                 n_win=(2, 2, 2), topk=4, routing_mode='mean', fps=30,
                 diff_routing=True, routing_tau=0.5, routing_band=(0.7, 3.0)):
        super().__init__()
        self.attn = BiLevelRoutingAttention_TDC_gra_sharp(
            dim, num_heads, dropout, theta, n_win=n_win, topk=topk,
            routing_mode=routing_mode, fps=fps,
            diff_routing=diff_routing, routing_tau=routing_tau, routing_band=routing_band,
        )
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward_ST(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, gra_sharp):
        Atten, Score = self.attn(self.norm1(x), gra_sharp)
        h = self.drop(self.proj(Atten))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x, Score


# =============================================================================
# Transformer_ST_TDC_gra_sharp_Bi — PhysFormer 원본 Transformer 에서
#   Block 만 Bi 로 교체
# =============================================================================
class Transformer_ST_TDC_gra_sharp_Bi(nn.Module):
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout, theta,
                 n_win=(2, 2, 2), topk=4, routing_mode='mean', fps=30,
                 diff_routing=True, routing_tau=0.5, routing_band=(0.7, 3.0)):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block_ST_TDC_gra_sharp_Bi(dim, num_heads, ff_dim, dropout, theta,
                                      n_win=n_win, topk=topk,
                                      routing_mode=routing_mode, fps=fps,
                                      diff_routing=diff_routing, routing_tau=routing_tau, routing_band=routing_band)
            for _ in range(num_layers)
        ])

    def forward(self, x, gra_sharp):
        for block in self.blocks:
            x, Score = block(x, gra_sharp)
        return x, Score


# =============================================================================
# ViT_BiPulseFormer — PhysFormer 원본 ViT_ST_ST_Compact3_TDC_gra_sharp 와
#   Stem/PE/upsample/init_weights/forward 전부 동일.
#   transformer1/2/3 만 Bi 버전 사용.
# =============================================================================
def _as_tuple(x):
    return x if isinstance(x, tuple) else (x, x, x)


class ViT_BiPulseFormer(nn.Module):
    """PhysFormer + BiLevel Routing Attention. 학습/forward 인터페이스는
    원본 PhysFormer 와 100% 동일 (forward(x, gra_sharp) → rPPG, S1, S2, S3)."""

    def __init__(
        self,
        patches=(4, 4, 4),
        dim: int = 96,
        ff_dim: int = 144,
        num_heads: int = 4,
        num_layers: int = 12,
        dropout_rate: float = 0.1,
        in_channels: int = 3,
        frame: int = 160,
        theta: float = 0.7,
        image_size=(160, 128, 128),
        n_win=(2, 2, 2),
        topk: int = 4,
        routing_mode: str = 'mean',
        fps: int = 30,
        diff_routing: bool = True,
        routing_tau: float = 0.5,
        routing_band=(0.7, 3.0),
    ):
        super().__init__()
        if tuple(patches) != (4, 4, 4) or tuple(image_size[1:]) != (128, 128):
            raise ValueError('Current stem/head require patches=(4,4,4) and 128x128 inputs')
        if num_layers < 3 or num_layers % 3:
            raise ValueError('num_layers must be a positive multiple of three')
        self.image_size = image_size
        self.frame = frame
        self.dim = dim

        ft, fh, fw = patches if isinstance(patches, tuple) else (patches, patches, patches)
        self.patch_embedding = nn.Conv3d(dim, dim, kernel_size=(ft, fh, fw), stride=(ft, fh, fw))

        # BRA 라우팅(routing_mode='fft_power')이 보는 lt 축은 patch_embedding에서
        # 이미 시간축 stride=ft 만큼 다운샘플링된 토큰이므로, 실제 토큰 샘플링
        # 레이트는 fps 가 아니라 fps/ft 다. 원본 fps를 그대로 넘기면 HR-band
        # [0.7,3.0]Hz 마스크가 엉뚱한 주파수 축(실제로는 ~[0.7,3.0]/ft Hz)에
        # 적용되는 버그가 생긴다 — token_fps로 보정해서 넘긴다.
        token_fps = fps / ft
        self.transformer1 = Transformer_ST_TDC_gra_sharp_Bi(
            num_layers=num_layers // 3, dim=dim, num_heads=num_heads,
            ff_dim=ff_dim, dropout=dropout_rate, theta=theta,
            n_win=n_win, topk=topk, routing_mode=routing_mode, fps=token_fps,
            diff_routing=diff_routing, routing_tau=routing_tau, routing_band=routing_band,
        )
        self.transformer2 = Transformer_ST_TDC_gra_sharp_Bi(
            num_layers=num_layers // 3, dim=dim, num_heads=num_heads,
            ff_dim=ff_dim, dropout=dropout_rate, theta=theta,
            n_win=n_win, topk=topk, routing_mode=routing_mode, fps=token_fps,
            diff_routing=diff_routing, routing_tau=routing_tau, routing_band=routing_band,
        )
        self.transformer3 = Transformer_ST_TDC_gra_sharp_Bi(
            num_layers=num_layers // 3, dim=dim, num_heads=num_heads,
            ff_dim=ff_dim, dropout=dropout_rate, theta=theta,
            n_win=n_win, topk=topk, routing_mode=routing_mode, fps=token_fps,
            diff_routing=diff_routing, routing_tau=routing_tau, routing_band=routing_band,
        )

        self.Stem0 = nn.Sequential(
            nn.Conv3d(3, dim // 4, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(dim // 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem1 = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 2, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem2 = nn.Sequential(
            nn.Conv3d(dim // 2, dim, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(dim, dim, [3, 1, 1], stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(dim),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(dim, dim // 2, [3, 1, 1], stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(dim // 2),
            nn.ELU(),
        )
        self.ConvBlockLast = nn.Conv1d(dim // 2, 1, 1, stride=1, padding=0)

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        """PhysFormer 원본: Linear xavier_uniform + bias normal_(std=1e-6)."""
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)

    def forward(self, x, gra_sharp=2.0):
        """x: (B, 3, T, H, W). Returns (rPPG, Score1, Score2, Score3) — PhysFormer 와 동일."""
        b, c, t, fh, fw = x.shape
        if c != 3 or t % 4 or (fh, fw) != (128, 128):
            raise ValueError('Expected (B,3,T,128,128) with T divisible by 4')

        x = self.Stem0(x)
        x = self.Stem1(x)
        x = self.Stem2(x)
        x = self.patch_embedding(x)        # (B, dim, t/4, 4, 4)
        x = x.flatten(2).transpose(1, 2)   # (B, t/4*4*4, dim)

        T1, S1 = self.transformer1(x, gra_sharp)
        T2, S2 = self.transformer2(T1, gra_sharp)
        T3, S3 = self.transformer3(T2, gra_sharp)

        features_last = T3.transpose(1, 2).view(b, self.dim, t // 4, 4, 4)
        features_last = self.upsample(features_last)
        features_last = self.upsample2(features_last)
        features_last = torch.mean(features_last, 3)
        features_last = torch.mean(features_last, 3)
        rPPG = self.ConvBlockLast(features_last).squeeze(1)
        return rPPG, S1, S2, S3

    def load_pretrained_pe(self, path):
        """Optional: load Stem0/1/2 + patch_embedding weights."""
        sd = torch.load(path, map_location='cpu')
        own_sd = self.state_dict()
        loaded = 0
        for k, v in sd.items():
            if k in own_sd and own_sd[k].shape == v.shape:
                own_sd[k] = v
                loaded += 1
        self.load_state_dict(own_sd)
        print(f"[BiPulseFormer] Loaded pretrained PE: {loaded} tensors from {path}")


# Backward-compat alias (older training scripts import BiPulseFormer)
BiPulseFormer = ViT_BiPulseFormer
