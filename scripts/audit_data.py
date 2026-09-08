"""Audit recording inventory and all train-sized PPG clips before long runs."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pure-root', default='D:/PURE')
    parser.add_argument('--ubfc-root', default='D:/UBFC-rPPG')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from src.data.rppg_dataset import RPPGDataset
    from src.protocol import subject_id
    from src.train import estimate_hr_targets
    torch.set_num_threads(2)
    report = {}
    for name, root in [('PURE', args.pure_root), ('UBFC-rPPG', args.ubfc_root)]:
        dataset = RPPGDataset(name, root, clip_len=160, img_size=128,
                              face_crop=True, data_type='diff_normalized')
        hrs, invalid = [], []
        for sample in dataset.samples:
            try:
                y = np.diff(np.asarray(sample['bvp'], dtype=np.float64))[None]
                hrs.append(float(estimate_hr_targets(y)[0]))
                if 'img_paths' in sample:
                    paths = sample['img_paths']
                else:
                    paths = [str(Path(sample['frames_dir']) / f'{sample["start_idx"] + j:05d}.png')
                             for j in range(161)]
                if not all(Path(p).is_file() for p in paths):
                    raise FileNotFoundError('Incomplete frame cache')
            except Exception as error:
                invalid.append({'video': sample['video_id'], 'start': sample['first_frame_idx'],
                                'error': str(error)})
        report[name] = {'n_subjects': len({subject_id(name, s['video_id']) for s in dataset.samples}),
                        'n_recordings': len({s['video_id'] for s in dataset.samples}),
                        'n_clips': len(dataset), 'invalid': invalid,
                        'target_bpm_range': [min(hrs), max(hrs)] if hrs else None}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    if any(r['invalid'] for r in report.values()):
        raise SystemExit('Data audit failed')


if __name__ == '__main__':
    main()
