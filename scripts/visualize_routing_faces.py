"""BiLevel Routing Attention 이 실제 얼굴의 어느 영역을 선택하는지 시각화.

`analyze_routing.py` 는 8x8/2x2 quadrant 통계만 숫자로 보여준다. 이 스크립트는
실제 얼굴 crop 이미지 위에 top-k routing 이 선택한 window 를 heatmap + 하이라이트로
오버레이해서, "이마/머리카락"이 아니라 "볼-코 중안부"가 실제로 선택되는지
육안으로 확인할 수 있게 한다.

n_win 의 spatial 분할 수 (wh, ww) 만큼 얼굴 이미지를 grid 로 나누고, row 위치를
  - 상단 1/3: forehead/hairline (이마/헤어라인)
  - 중단 1/3: mid-face, cheek/nose (중안부: 볼-코)
  - 하단 1/3: mouth/chin (입/턱)
로 라벨링한다. n_win 이 (2,2,2) 처럼 spatial row 가 2개뿐이면 top/bottom 만
구분 가능하고 "중안부"는 별도로 분리되지 않으므로, 진짜 중안부 여부를 보려면
n_win spatial >= 4 (예: phase12 의 (1,4,4)) 체크포인트를 사용해야 한다.

실행 예:
  python scripts/visualize_routing_faces.py \
      --ckpt results/cross_82_pure_to_ubfc_phase12/checkpoints/PURE_to_UBFC-rPPG_epoch4.pt \
      --dataset UBFC-rPPG --n_win 1 4 4 --topk 4 --routing_mode fft_power \
      --n_clips 8 --out_dir results/routing_face_viz/phase12_fft_power

출력 (out_dir 아래):
  grid.png       — 클립별 [원본 얼굴 | 선택빈도 heatmap 오버레이 | top window 하이라이트]
  aggregate.png  — 전체 클립 평균 얼굴 + 평균 heatmap
  stats.json     — row-band(이마/중안부/입턱) 별 평균 선택 비율, 최고 선택 영역
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches

from src.models.bipulseformer import (
    ViT_BiPulseFormer, BiLevelRoutingAttention_TDC_gra_sharp,
)
from src.data.rppg_dataset import RPPGDataset


def s_to_wt_wh_ww(s, n_win):
    wt, wh, ww = n_win
    iwt = s // (wh * ww)
    rest = s % (wh * ww)
    iwh = rest // ww
    iww = rest % ww
    return iwt, iwh, iww


def row_band_label(iwh, wh):
    # English-only: matplotlib default font (DejaVu Sans) has no Hangul glyphs.
    frac = (iwh + 0.5) / wh
    if frac < 1 / 3:
        return 'forehead/hairline'
    elif frac < 2 / 3:
        return 'mid-face (cheek/nose)'
    else:
        return 'mouth/chin'


class RoutingCapture:
    """모든 BLRA layer 의 topk_idx 를 hook 으로 수집 (1 clip 처리마다 reset)."""
    def __init__(self, model):
        self.captures = []
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
        def hook(module, inputs, output):
            a_r = output[1]
            topk = min(module.topk, a_r.shape[-1])
            _, topk_idx = torch.topk(a_r, k=topk, dim=-1)
            self.captures.append({
                'layer': layer_idx, 'stage': stage,
                'topk_idx': topk_idx.detach().cpu().numpy()[0],  # (S, topk), batch=1
                'n_win': module.n_win,
            })
        return hook

    def reset(self):
        self.captures = []

    def remove(self):
        for h in self.hooks:
            h.remove()


def spatial_selection_count(captures, n_win, stages=None):
    """이 clip 의 모든(또는 지정 stage) layer 를 합산한 (wh, ww) 선택 빈도 grid."""
    wt, wh, ww = n_win
    grid = np.zeros((wh, ww), dtype=np.float64)
    total = 0
    for c in captures:
        if stages is not None and c['stage'] not in stages:
            continue
        topk_idx = c['topk_idx']  # (S, topk)
        S, topk = topk_idx.shape
        for s_q in range(S):
            for s_k in topk_idx[s_q]:
                _, iwh, iww = s_to_wt_wh_ww(int(s_k), n_win)
                grid[iwh, iww] += 1
                total += 1
    if total > 0:
        grid = grid / total
    return grid


def overlay_heatmap(face_img_u8, grid, alpha=0.5):
    """grid (wh, ww) in [0,1] -> upsample to face image resolution, jet colormap alpha blend."""
    H, W = face_img_u8.shape[:2]
    wh, ww = grid.shape
    cell_h, cell_w = H // wh, W // ww
    heat = np.kron(grid, np.ones((cell_h, cell_w)))
    # pad to exact H,W if not divisible
    heat = np.pad(heat, ((0, H - heat.shape[0]), (0, W - heat.shape[1])), mode='edge')
    norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
    colored = (cm.jet(norm)[..., :3] * 255).astype(np.uint8)
    out = (face_img_u8.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def highlight_top_regions(face_img_u8, grid, dim_factor=0.35):
    """최고 선택 grid cell(들)만 원래 밝기 유지, 나머지는 어둡게 + 테두리 표시."""
    H, W = face_img_u8.shape[:2]
    wh, ww = grid.shape
    cell_h, cell_w = H // wh, W // ww
    top_val = grid.max()
    is_top = grid >= (top_val - 1e-9)

    out = face_img_u8.astype(np.float32).copy()
    for r in range(wh):
        for c in range(ww):
            if not is_top[r, c]:
                y0, y1 = r * cell_h, (r + 1) * cell_h if r < wh - 1 else H
                x0, x1 = c * cell_w, (c + 1) * cell_w if c < ww - 1 else W
                out[y0:y1, x0:x1] *= dim_factor
    return np.clip(out, 0, 255).astype(np.uint8), is_top


def draw_grid_and_boxes(ax, is_top, wh, ww, img_size):
    cell_h, cell_w = img_size // wh, img_size // ww
    for r in range(wh):
        for c in range(ww):
            y0, x0 = r * cell_h, c * cell_w
            h_ = cell_h if r < wh - 1 else img_size - y0
            w_ = cell_w if c < ww - 1 else img_size - x0
            color = 'lime' if is_top[r, c] else 'white'
            lw = 3 if is_top[r, c] else 0.5
            rect = mpatches.Rectangle((x0, y0), w_, h_, fill=False, edgecolor=color,
                                       linewidth=lw, alpha=0.9 if is_top[r, c] else 0.3)
            ax.add_patch(rect)


def main():
    parser = argparse.ArgumentParser(description='BiLevel Routing — 얼굴 영역 시각화')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--dataset', default='UBFC-rPPG', choices=['PURE', 'UBFC-rPPG'])
    parser.add_argument('--data_path', default=None)
    parser.add_argument('--split_range', nargs=2, type=float, default=None)
    parser.add_argument('--n_win', nargs=3, type=int, default=[2, 2, 2])
    parser.add_argument('--topk', type=int, default=4)
    parser.add_argument('--routing_mode', default='mean', choices=['mean', 'fft_power'])
    parser.add_argument('--n_clips', type=int, default=8, help='시각화할 clip 수')
    parser.add_argument('--stride', type=int, default=None,
                        help='dataset 내에서 clip 을 뽑는 간격 (subject 다양성 확보). '
                             'None 이면 len(dataset)//n_clips')
    parser.add_argument('--out_dir', default='results/routing_face_viz/default')
    args = parser.parse_args()

    if args.data_path is None:
        args.data_path = 'D:\\PURE' if args.dataset == 'PURE' else 'D:\\UBFC-rPPG'
    n_win = tuple(args.n_win)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ViT_BiPulseFormer(
        patches=(4, 4, 4), dim=96, ff_dim=144, num_heads=4, num_layers=12,
        dropout_rate=0.1, theta=0.7, image_size=(160, 128, 128),
        n_win=n_win, topk=args.topk, routing_mode=args.routing_mode,
    ).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f'[*] Loaded checkpoint: {args.ckpt}  (n_win={n_win}, topk={args.topk}, routing_mode={args.routing_mode})')

    common_kw = dict(root_dir=args.data_path, clip_len=160, img_size=128,
                      face_crop=True, dynamic_detection_freq=0,
                      chunk_step=80, random_hflip=False)
    if args.split_range:
        common_kw['split_range'] = tuple(args.split_range)

    ds_input = RPPGDataset(args.dataset, data_type='diff_normalized', **common_kw)
    ds_raw = RPPGDataset(args.dataset, data_type='raw', **common_kw)
    assert len(ds_input) == len(ds_raw), 'input/raw dataset 길이 불일치 — split 설정 확인'
    n_total = len(ds_input)
    print(f'[*] {args.dataset}: {n_total} clips total')

    stride = args.stride or max(1, n_total // args.n_clips)
    indices = list(range(0, n_total, stride))[:args.n_clips]

    cap = RoutingCapture(model)
    print(f'[*] Hooked {cap.layer_idx} BiLevelRoutingAttention layers')

    wh, ww = n_win[1], n_win[2]
    per_clip_grids = []
    per_clip_faces = []

    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4 * len(indices)))
    if len(indices) == 1:
        axes = axes[None, :]

    for row, idx in enumerate(indices):
        diff_frames, _ = ds_input[idx]
        raw_frames, _ = ds_raw[idx]
        mid_t = raw_frames.shape[1] // 2
        face_img = (raw_frames[:, mid_t].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

        cap.reset()
        with torch.no_grad():
            _ = model(diff_frames.unsqueeze(0).to(device), gra_sharp=2.0)
        grid = spatial_selection_count(cap.captures, n_win)
        per_clip_grids.append(grid)
        per_clip_faces.append(face_img)

        heat_img = overlay_heatmap(face_img, grid)
        hl_img, is_top = highlight_top_regions(face_img, grid)

        axes[row, 0].imshow(face_img)
        axes[row, 0].set_title(f'clip {idx} — original face' if row == 0 else '')
        axes[row, 1].imshow(heat_img)
        axes[row, 1].set_title('routing selection heatmap' if row == 0 else '')
        axes[row, 2].imshow(hl_img)
        draw_grid_and_boxes(axes[row, 2], is_top, wh, ww, face_img.shape[0])
        top_r, top_c = np.unravel_index(np.argmax(grid), grid.shape)
        label = row_band_label(top_r, wh)
        axes[row, 2].set_title(f'top region: {label}' if row == 0 else label, fontsize=9)
        for ax in axes[row]:
            ax.axis('off')

    plt.tight_layout()
    grid_path = os.path.join(args.out_dir, 'grid.png')
    plt.savefig(grid_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f'[*] Saved per-clip grid: {grid_path}')

    # ---- Aggregate across clips ----
    avg_grid = np.mean(per_clip_grids, axis=0)
    avg_face = np.mean(np.stack(per_clip_faces), axis=0).astype(np.uint8)
    avg_heat = overlay_heatmap(avg_face, avg_grid)
    avg_hl, avg_is_top = highlight_top_regions(avg_face, avg_grid)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(avg_face); axes[0].set_title(f'avg face (n={len(indices)} clips)')
    axes[1].imshow(avg_heat); axes[1].set_title('avg routing selection heatmap')
    axes[2].imshow(avg_hl)
    draw_grid_and_boxes(axes[2], avg_is_top, wh, ww, avg_face.shape[0])
    top_r, top_c = np.unravel_index(np.argmax(avg_grid), avg_grid.shape)
    axes[2].set_title(f'top region: {row_band_label(top_r, wh)}', fontsize=9)
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    agg_path = os.path.join(args.out_dir, 'aggregate.png')
    plt.savefig(agg_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f'[*] Saved aggregate: {agg_path}')

    # ---- Row-band stats ----
    row_ratio = avg_grid.sum(axis=1)  # (wh,)
    band_totals = defaultdict(float)
    for r in range(wh):
        band_totals[row_band_label(r, wh)] += row_ratio[r]

    print()
    print('=' * 70)
    print(' Row-band selection ratio (average over clips)')
    print('=' * 70)
    for band, val in sorted(band_totals.items(), key=lambda kv: -kv[1]):
        print(f'  {band:<25} {val:.3f}')
    if wh < 4:
        print('  [!] n_win spatial rows < 4 -> only top/bottom distinguishable, '
              'no separate "mid-face" band. Use n_win=(*,4,4)+ to test mid-face selection.')

    stats = {
        'ckpt': args.ckpt, 'dataset': args.dataset, 'n_win': list(n_win),
        'topk': args.topk, 'routing_mode': args.routing_mode,
        'n_clips': len(indices), 'clip_indices': indices,
        'avg_grid': avg_grid.tolist(),
        'row_band_ratio': {band: v for band, v in band_totals.items()},
        'top_region': row_band_label(int(top_r), wh),
    }
    stats_path = os.path.join(args.out_dir, 'stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f'\n[*] Saved stats: {stats_path}')

    cap.remove()


if __name__ == '__main__':
    main()
