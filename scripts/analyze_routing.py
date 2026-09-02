"""BiLevel Routing Attention 분석 — 모델이 선택한 region 이 rPPG 측면에서 의미 있나?

분석 내용:
  1. 12개 BLRA layer 각각에서 routing 선택 빈도 집계
  2. n_win=(2,2,2) → 8 windows 인덱싱:
       window s = wt*4 + wh*2 + ww
       wh=0 (top), wh=1 (bottom),  ww=0 (left), ww=1 (right)
       wt=0 (전반), wt=1 (후반) — 시간축
  3. Spatial 4 quadrant (TL/TR/BL/BR) 별 선택 빈도 → 얼굴 영역 의미 분석
       - rPPG 의미 영역: top 절반 (forehead/cheek 포함), bottom 절반 (mouth/chin)
       - Cross-quadrant 선택 패턴 vs self-quadrant 패턴
  4. Stage 별 (1: low-level, 3: high-level) 패턴 비교
  5. 시각화: 8x8 selection matrix per layer + spatial heatmap + face overlay

실행:
  python scripts/analyze_routing.py \
      --ckpt results/cross_82_pure_to_ubfc_const30/checkpoints/PURE_to_UBFC-rPPG_epoch7.pt \
      --dataset UBFC-rPPG --n_clips 50

출력: results/routing_analysis/
"""
import os
import sys
import io
import json
import argparse
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.bipulseformer import (
    ViT_BiPulseFormer, BiLevelRoutingAttention_TDC_gra_sharp,
)
from src.data.rppg_dataset import get_dataloader


def s_to_wt_wh_ww(s, n_win=(2, 2, 2)):
    """window index s → (wt, wh, ww). s = wt*(wh*ww) + wh*ww + ww."""
    wt, wh, ww = n_win
    iwt = s // (wh * ww)
    rest = s % (wh * ww)
    iwh = rest // ww
    iww = rest % ww
    return iwt, iwh, iww


QUADRANT_NAMES = {
    (0, 0): 'TL (top-left, 이마 좌반)',
    (0, 1): 'TR (top-right, 이마 우반)',
    (1, 0): 'BL (bottom-left, 입/턱 좌반)',
    (1, 1): 'BR (bottom-right, 입/턱 우반)',
}


class RoutingCapture:
    """모든 BLRA layer 의 topk_idx 와 routing matrix 를 hook 으로 수집."""
    def __init__(self, model):
        self.model = model
        self.captures = []   # list of {layer_idx, stage, topk_idx, scores}
        self.hooks = []
        self.layer_idx = 0
        for stage, transformer in enumerate(
                [model.transformer1, model.transformer2, model.transformer3], start=1):
            for block in transformer.blocks:
                attn = block.attn
                if isinstance(attn, BiLevelRoutingAttention_TDC_gra_sharp):
                    h = attn.register_forward_hook(self._make_hook(self.layer_idx, stage))
                    self.hooks.append(h)
                    self.layer_idx += 1

    def _make_hook(self, layer_idx, stage):
        captures = self.captures
        def hook(module, inputs, output):
            # output: (h_out, a_r)  where a_r = (B, S, S) region routing scores
            a_r = output[1]
            topk = min(module.topk, a_r.shape[-1])
            _, topk_idx = torch.topk(a_r, k=topk, dim=-1)   # (B, S, topk)
            captures.append({
                'layer': layer_idx, 'stage': stage,
                'topk_idx': topk_idx.detach().cpu().numpy(),
                'scores': a_r.detach().cpu().numpy(),
                'n_win': module.n_win, 'topk': topk,
            })
        return hook

    def reset(self):
        self.captures = []

    def remove(self):
        for h in self.hooks:
            h.remove()


def aggregate(captures, n_layers=12, S=8):
    """Return per-layer selection matrix (S, S) — count[query=s_q][selected=s_k] / total."""
    by_layer = defaultdict(lambda: np.zeros((S, S), dtype=np.float64))
    counts = defaultdict(int)
    for c in captures:
        topk_idx = c['topk_idx']    # (B, S, topk)
        B = topk_idx.shape[0]
        for b in range(B):
            for s_q in range(S):
                for k_idx in topk_idx[b, s_q]:
                    by_layer[c['layer']][s_q, k_idx] += 1
        counts[c['layer']] += B
    out = {}
    for layer, mat in by_layer.items():
        if counts[layer] > 0:
            out[layer] = mat / counts[layer]
    return out


def spatial_heatmap(selection_mat, n_win=(2, 2, 2)):
    """8x8 selection matrix → 2x2 spatial selection (collapse temporal wt and source query)."""
    S = selection_mat.shape[0]
    wt, wh, ww = n_win
    spatial = np.zeros((wh, ww), dtype=np.float64)
    for s_k in range(S):
        _, iwh, iww = s_to_wt_wh_ww(s_k, n_win)
        spatial[iwh, iww] += selection_mat[:, s_k].sum()
    spatial = spatial / spatial.sum()  # normalize
    return spatial


def temporal_heatmap(selection_mat, n_win=(2, 2, 2)):
    """8x8 → 2 temporal halves."""
    S = selection_mat.shape[0]
    wt = n_win[0]
    temporal = np.zeros(wt, dtype=np.float64)
    for s_k in range(S):
        iwt, _, _ = s_to_wt_wh_ww(s_k, n_win)
        temporal[iwt] += selection_mat[:, s_k].sum()
    temporal = temporal / temporal.sum()
    return temporal


def self_routing_ratio(selection_mat, n_win=(2, 2, 2)):
    """각 query window 가 자기 자신을 top-k 안에 선택한 비율."""
    S = selection_mat.shape[0]
    diag = selection_mat.diagonal().sum()
    total = selection_mat.sum()
    return float(diag / total) if total > 0 else 0.0


def same_quadrant_ratio(selection_mat, n_win=(2, 2, 2)):
    """같은 spatial quadrant 안에서 query→key routing 비율 (다른 시간대 포함)."""
    S = selection_mat.shape[0]
    same = 0.0
    total = selection_mat.sum()
    for s_q in range(S):
        _, q_wh, q_ww = s_to_wt_wh_ww(s_q, n_win)
        for s_k in range(S):
            _, k_wh, k_ww = s_to_wt_wh_ww(s_k, n_win)
            if q_wh == k_wh and q_ww == k_ww:
                same += selection_mat[s_q, s_k]
    return float(same / total) if total > 0 else 0.0


def plot_per_layer_matrices(layer_mats, out_path, n_win=(2, 2, 2)):
    n_layers = len(layer_mats)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for layer in range(n_layers):
        ax = axes[layer // 4, layer % 4]
        mat = layer_mats[layer]
        im = ax.imshow(mat, cmap='YlOrRd', vmin=0, vmax=mat.max())
        stage = layer // 4 + 1
        ax.set_title(f'L{layer+1} (stage {stage})\nself={self_routing_ratio(mat):.2f}, same-Q={same_quadrant_ratio(mat, n_win):.2f}')
        ax.set_xlabel('selected key window')
        ax.set_ylabel('query window')
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle(f'BiLevel Routing Attention — per-layer selection frequency (top-{4} of 8 windows)', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_spatial_heatmap(layer_mats, out_path, n_win=(2, 2, 2)):
    n_layers = len(layer_mats)
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for layer in range(n_layers):
        ax = axes[layer // 4, layer % 4]
        spatial = spatial_heatmap(layer_mats[layer], n_win)
        im = ax.imshow(spatial, cmap='YlGn', vmin=spatial.min(), vmax=spatial.max())
        stage = layer // 4 + 1
        ax.set_title(f'L{layer+1} (stage {stage}) — spatial routing distribution')
        for ih in range(spatial.shape[0]):
            for iw in range(spatial.shape[1]):
                quad = 'TL' if (ih, iw) == (0, 0) else 'TR' if (ih, iw) == (0, 1) else 'BL' if (ih, iw) == (1, 0) else 'BR'
                ax.text(iw, ih, f'{quad}\n{spatial[ih,iw]:.3f}',
                       ha='center', va='center', fontsize=11, color='black')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle('Spatial selection (temporal-collapsed) — TL/TR=top half (forehead area), BL/BR=bottom half (mouth/chin)',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='BiLevel Routing Attention 분석')
    parser.add_argument('--ckpt', required=True, help='Best checkpoint .pt path')
    parser.add_argument('--dataset', default='UBFC-rPPG',
                       choices=['PURE', 'UBFC-rPPG'])
    parser.add_argument('--data_path', default=None,
                       help='Dataset root (default: D:\\PURE or D:\\UBFC-rPPG)')
    parser.add_argument('--n_clips', type=int, default=50,
                       help='분석에 사용할 클립 수')
    parser.add_argument('--out_dir', default='results/routing_analysis')
    parser.add_argument('--split_range', nargs=2, type=float, default=None,
                       help='Subject split range (e.g., 0.8 1.0)')
    args = parser.parse_args()

    if args.data_path is None:
        args.data_path = 'D:\\PURE' if args.dataset == 'PURE' else 'D:\\UBFC-rPPG'

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 로드
    model = ViT_BiPulseFormer(
        patches=(4, 4, 4), dim=96, ff_dim=144, num_heads=4, num_layers=12,
        dropout_rate=0.1, theta=0.7, image_size=(160, 128, 128),
        n_win=(2, 2, 2), topk=4,
    ).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f'[*] Loaded checkpoint: {args.ckpt}')

    # 데이터로더
    common = dict(face_crop=True, dynamic_detection_freq=0,
                  data_type='diff_normalized')
    if args.split_range:
        common['split_range'] = tuple(args.split_range)
    loader = get_dataloader(args.dataset, args.data_path, batch_size=4, clip_len=160,
                            shuffle=False, chunk_step=80, **common)
    print(f'[*] {args.dataset}: {len(loader.dataset)} clips total')

    # Hook 등록
    cap = RoutingCapture(model)
    print(f'[*] Hooked {cap.layer_idx} BiLevelRoutingAttention layers')

    # 추론 (n_clips 만큼만)
    n_done = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            _ = model(inputs, gra_sharp=2.0)
            n_done += inputs.shape[0]
            if n_done >= args.n_clips:
                break
    cap.remove()
    print(f'[*] Processed {n_done} clips, {len(cap.captures)} layer captures')

    # 집계
    layer_mats = aggregate(cap.captures, n_layers=12, S=8)

    # 통계 출력
    stats = {}
    for layer in sorted(layer_mats.keys()):
        mat = layer_mats[layer]
        spatial = spatial_heatmap(mat)
        temporal = temporal_heatmap(mat)
        stats[f'layer_{layer+1}'] = {
            'stage': layer // 4 + 1,
            'self_routing_ratio': self_routing_ratio(mat),
            'same_quadrant_ratio': same_quadrant_ratio(mat),
            'top_half_ratio': float(spatial[0].sum()),    # rPPG 의미: forehead 영역
            'bottom_half_ratio': float(spatial[1].sum()),
            'left_half_ratio': float(spatial[:, 0].sum()),
            'right_half_ratio': float(spatial[:, 1].sum()),
            'first_half_temporal': float(temporal[0]),
            'second_half_temporal': float(temporal[1]),
            'spatial_matrix': spatial.tolist(),
        }

    # 평균 (전체 layer)
    avg_self = np.mean([stats[k]['self_routing_ratio'] for k in stats])
    avg_same_q = np.mean([stats[k]['same_quadrant_ratio'] for k in stats])
    avg_top_half = np.mean([stats[k]['top_half_ratio'] for k in stats])

    print()
    print('=' * 78)
    print(' Per-layer routing analysis summary')
    print('=' * 78)
    print(f'{"Layer":<7} {"Stage":<6} {"Self":<7} {"Same-Q":<8} {"Top%":<7} {"Bot%":<7} {"Left%":<7} {"Right%":<7}')
    print('-' * 78)
    for layer in sorted(layer_mats.keys()):
        s = stats[f'layer_{layer+1}']
        print(f'L{layer+1:<6} {s["stage"]:<6} {s["self_routing_ratio"]:.3f}  '
              f'{s["same_quadrant_ratio"]:.3f}   '
              f'{s["top_half_ratio"]:.3f}  {s["bottom_half_ratio"]:.3f}  '
              f'{s["left_half_ratio"]:.3f}  {s["right_half_ratio"]:.3f}')
    print('-' * 78)
    print(f'AVG     —      {avg_self:.3f}  {avg_same_q:.3f}   {avg_top_half:.3f}  '
          f'{1-avg_top_half:.3f}  —      —')

    # 의미 해석
    print()
    print('해석:')
    print(f'  - Self-routing (자기 window 선택): {avg_self:.3f}  '
          f'(random=0.125 이면 학습 X, 0.5+ 면 강한 self-preference)')
    print(f'  - Same-quadrant (같은 spatial quadrant): {avg_same_q:.3f}  '
          f'(random=0.25 이면 균등, 0.5+ 면 같은 영역 강조)')
    print(f'  - Top half (이마 영역) 선택 비율: {avg_top_half:.3f}  '
          f'(rPPG 의미 영역. 0.5 이상이면 forehead 강조)')

    # JSON 저장
    json_path = os.path.join(args.out_dir, 'routing_stats.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'ckpt': args.ckpt, 'dataset': args.dataset, 'n_clips': n_done,
            'avg_self_routing': float(avg_self),
            'avg_same_quadrant': float(avg_same_q),
            'avg_top_half': float(avg_top_half),
            'per_layer': stats,
        }, f, indent=2)
    print(f'\n[*] Saved stats: {json_path}')

    # Plots
    sel_path = os.path.join(args.out_dir, 'selection_matrices.png')
    plot_per_layer_matrices(layer_mats, sel_path)
    print(f'[*] Saved selection matrices: {sel_path}')

    sp_path = os.path.join(args.out_dir, 'spatial_heatmaps.png')
    plot_spatial_heatmap(layer_mats, sp_path)
    print(f'[*] Saved spatial heatmaps: {sp_path}')


if __name__ == '__main__':
    main()
