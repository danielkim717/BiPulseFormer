"""Inspect validation-face routing using checkpoint-embedded model settings."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data-root', help='Override recorded source root after moving datasets')
    parser.add_argument('--clips', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.clips < 1:
        parser.error('--clips must be positive')
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from src.experiment import build_loader, build_model, write_json
    from src.protocol import subject_id
    from src.models.bipulseformer import BiLevelRoutingAttention_TDC_gra_sharp
    run = json.loads((args.run / 'config.json').read_text(encoding='utf-8'))
    state = torch.load(args.run / 'best.pt', map_location=args.device, weights_only=True)
    if state['model_name'] != 'bipulseformer' or state['run_id'] != run['run_id']:
        raise ValueError('Expected the matching BiPulseFormer checkpoint')
    config = state['config']
    model = build_model(config, 'bipulseformer').to(args.device).eval()
    model.load_state_dict(state['model'], strict=True)
    loader = build_loader(run['split']['source'], args.data_root or run['source_root'],
                          run['split']['subject_ids']['valid'], config,
                          SimpleNamespace(seed=run['seed'], workers=0, device=args.device))
    dataset = loader.dataset
    # Round-robin across people so a long recording cannot fill the whole panel.
    by_person = {}
    for i, sample in enumerate(dataset.samples):
        by_person.setdefault(subject_id(dataset.dataset_name, sample['video_id']), []).append(i)
    selected = []
    for depth in range(max(map(len, by_person.values()))):
        for ids in by_person.values():
            if depth < len(ids):
                selected.append(ids[depth])
                if len(selected) == args.clips:
                    break
        if len(selected) == args.clips:
            break
    wt, wh, ww = config['n_win']
    fig, axes = plt.subplots(len(selected), 1, figsize=(4, 3 * len(selected)), squeeze=False)
    records = []
    for axis, i in zip(axes[:, 0], selected):
        sample = dataset.samples[i]
        x, _ = dataset[i]
        with torch.inference_mode():
            model(x.unsqueeze(0).to(args.device), gra_sharp=config['gra_sharp'])
        votes = np.zeros(wt * wh * ww)
        for layer in model.modules():
            if isinstance(layer, BiLevelRoutingAttention_TDC_gra_sharp):
                indices = layer.scores.topk(config['topk'], dim=-1).indices.cpu().numpy().ravel()
                votes += np.bincount(indices, minlength=len(votes))
        votes /= votes.sum()
        heatmap = votes.reshape(wt, wh, ww).sum(axis=0)
        image_path = dataset._video_img_list[sample['video_id']][sample['first_frame_idx']]
        face = cv2.imread(image_path)
        if face is None:
            raise OSError(image_path)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = dataset._crop_face(face, sample['video_id'], sample['first_frame_idx'])
        face = cv2.resize(face, (128, 128))
        axis.imshow(face)
        axis.imshow(heatmap, extent=(-.5, 127.5, 127.5, -.5), alpha=.45,
                    cmap='magma', interpolation='nearest', vmin=0, vmax=max(heatmap.max(), 1e-8))
        axis.set_title(f"Validation {sample['video_id']} @ {sample['first_frame_idx']}")
        axis.axis('off')
        records.append({'video_id': sample['video_id'], 'start': sample['first_frame_idx'],
                        'selection_fraction': heatmap.tolist()})
    folder = args.run / 'routing_validation'
    folder.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder / 'faces.png', dpi=140)
    plt.close(fig)
    write_json(folder / 'routing.json', {'run_id': run['run_id'], 'split': 'valid', 'clips': records})
    print(folder)


if __name__ == '__main__':
    main()
