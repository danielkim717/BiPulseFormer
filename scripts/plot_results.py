"""Plot saved recording HR pairs; no model inference or test-set reselection."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    result = json.loads(args.summary.read_text(encoding='utf-8'))
    metrics = result['test']['per_recording']
    pred, gt = np.asarray(metrics['pred_hrs']), np.asarray(metrics['gt_hrs'])
    if len(pred) == 0 or pred.shape != gt.shape:
        raise ValueError('No matching saved recording HR pairs')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(gt, pred, s=20)
    limits = [min(gt.min(), pred.min()), max(gt.max(), pred.max())]
    axes[0].plot(limits, limits, '--', color='gray')
    axes[0].set(xlabel='Reference HR (BPM)', ylabel='Predicted HR (BPM)', title='Recording HR')
    difference = pred - gt
    axes[1].scatter((pred + gt) / 2, difference, s=20)
    bias = difference.mean()
    axes[1].axhline(bias, color='gray')
    if len(pred) > 1:
        for bound in (bias - 1.96 * difference.std(ddof=1), bias + 1.96 * difference.std(ddof=1)):
            axes[1].axhline(bound, linestyle='--', color='gray')
    axes[1].set(xlabel='Mean HR (BPM)', ylabel='Prediction - reference (BPM)', title='Bland–Altman')
    fig.tight_layout()
    destination = args.summary.with_name('hr_plots.png')
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    print(destination)


if __name__ == '__main__':
    main()
