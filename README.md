# BiPulseFormer

**Frequency-guided spatial routing for remote photoplethysmography.**

BiPulseFormer uses temporal frequency information in facial video to select spatial regions for pulse-wave estimation. It builds HR-band FFT magnitude descriptors from projected Q/K features, selects Top-K key regions using descriptor dot products, and applies attention to the original Q/K/V content. The current baseline uses **K=4, 16 spatial regions and 12 blocks**, with no temporal subdivision of routing regions. Training uses dense STE; inference gathers selected K/V tokens.

**2026-11-10: current manuscript code and colleague handoff are now collected in one release.**

- [한국어 공유 안내 — 모델·학습·전처리·분할·평가를 한눈에](docs/SHARING_GUIDE_KO.md)
- [Current code and protocols](experiments/manuscript_20261110/README.md)
- [Installation and training handoff](experiments/manuscript_20261110/TRAINING.md)
- [Exact subject splits](experiments/manuscript_20261110/splits.json)
- [Unrounded recording results and provenance](experiments/manuscript_20261110/results_recording.json)

## Current recording-level results

| Experiment | MAE (BPM) ↓ | RMSE (BPM) ↓ | MAPE (%) ↓ | Pearson ↑ |
|---|---:|---:|---:|---:|
| PURE intra | 0.26 | 0.60 | 0.33 | 0.99 |
| UBFC-rPPG intra — post-hoc split | 0.15 | 0.51 | 0.21 | 0.99 |
| COHFACE intra | 0.82 | 3.20 | 0.96 | 0.96 |
| PURE → UBFC-rPPG | 1.09 | 2.53 | 1.23 | 0.99 |

All current values use the common **40–180 BPM prediction/reference scorer**. Intra uses fixed epoch30; the cross model is selected using source validation only. The UBFC intra split exchanges subjects after observing earlier test errors and is explicitly an outcome-informed post-hoc analysis, not an independent random-split confirmation. Published baselines are reference comparisons under their own protocols. Full precision, displayed-Pearson policy and split membership are preserved in the release. New fresh UBFC→PURE training is being prepared and has no final result here.

## Code map

| What to inspect | Current implementation |
|---|---|
| Model / routing | [bipulseformer.py](experiments/manuscript_20261110/implementation/src/models/bipulseformer.py) |
| Training losses | [train.py](experiments/manuscript_20261110/implementation/src/train.py) |
| Training and test loops | [Dataset-specific workers](experiments/manuscript_20261110/runs) |
| PURE / UBFC input transforms | [rppg_dataset.py](experiments/manuscript_20261110/implementation/legacy/rppg_dataset.py) and worker readers |
| COHFACE time alignment / inputs | [cohface_intra_data.py](experiments/manuscript_20261110/runs/cohface_intra/controls/cohface_intra_data.py) |
| HR evaluation | [hr_evaluation_40_180.py](experiments/manuscript_20261110/implementation/src/hr_evaluation_40_180.py) |
| Splits and checkpoint policy | [splits.json](experiments/manuscript_20261110/splits.json), [run contracts](experiments/manuscript_20261110/runs) |

```bash
git clone https://github.com/danielkim717/BiPulseFormer.git
cd BiPulseFormer
python experiments/manuscript_20261110/verify.py
```

The verification command needs no dataset or GPU. A CPU model smoke check and waveform-scoring API are included. Raw datasets, large caches, trained weights and private manuscripts are not uploaded. Full training requires local data preparation and the checks described in the training handoff; the frozen workers are not advertised as a one-command end-to-end reproduction from a fresh clone.

## Historical material

The root `src/`, `scripts/`, older `results/` and [September release](experiments/final_protocol_20260914/README.md) remain available for traceability. Use the **November manuscript release above** when sharing the current model. The [previous project overview](README_20260917.md) is archived with its original historical results and evaluation band; it does not describe the current paper table. Subsequent exploratory tuning is separate from this baseline.

## Foundations and attribution

The work builds on [PhysFormer](https://github.com/ZitongYu/PhysFormer), [BiFormer](https://github.com/rayleizhu/BiFormer), and the pinned [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) implementation. Bundled upstream license/notice files and code provenance are retained. This is not the official implementation of those projects. Physiologically useful region selection is a design aim; the routing outputs are not anatomical segmentation labels.
