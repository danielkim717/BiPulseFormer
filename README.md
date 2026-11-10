# BiPulseFormer

**Frequency-guided spatial routing for remote photoplethysmography.**

BiPulseFormer extracts temporal FFT magnitude descriptors from spatially pooled Q/K features, compares the descriptors to select Top-K regions, and applies attention to the original Q/K/V tokens. The baseline uses **K=4, a 4×4 spatial grid and 12 blocks**, preserving the full temporal extent within each region.

## Start here

- [한국어 코드·실행 안내](docs/SHARING_GUIDE_KO.md)
- [Model, preprocessing and training code](experiments/manuscript_20261110/README.md)
- [Installation and training](experiments/manuscript_20261110/TRAINING.md)
- [Subject splits](experiments/manuscript_20261110/splits.json) · [Evaluation protocol](experiments/manuscript_20261110/evaluation_protocol.json)

```bash
git clone https://github.com/danielkim717/BiPulseFormer.git
cd BiPulseFormer
python -m pip install -r requirements.txt
python experiments/manuscript_20261110/verify.py
python experiments/manuscript_20261110/api.py --smoke
```

Install a compatible PyTorch/torchvision build before installing the remaining dependencies. The smoke check runs on CPU without a dataset. Training requires local datasets and the preparation steps in the training guide.

## Current model results

Recording-level evaluation; MAE/RMSE in BPM, MAPE in percent, prediction/reference HR search at 40–180 BPM.

| Evaluation | MAE ↓ | RMSE ↓ | MAPE ↓ | Pearson ↑ |
|---|---:|---:|---:|---:|
| PURE intra | 0.26 | 0.60 | 0.33 | 0.99 |
| UBFC-rPPG intra | 0.15 | 0.51 | 0.21 | 0.99 |
| COHFACE intra | 0.82 | 3.20 | 0.96 | 0.96 |
| PURE → UBFC-rPPG | 1.09 | 2.53 | 1.23 | 0.99 |

[Full-precision values](experiments/manuscript_20261110/results_recording.json) and [exact splits](experiments/manuscript_20261110/splits.json) accompany the table. PURE/UBFC intra Pearson is truncated to two decimals, matching the manuscript; other cells are rounded. Intra uses fixed epoch30; cross checkpoints use source validation.

## Code map

| Component | File |
|---|---|
| Architecture and routing | [bipulseformer.py](experiments/manuscript_20261110/implementation/src/models/bipulseformer.py) |
| Training losses | [train.py](experiments/manuscript_20261110/implementation/src/train.py) |
| Dataset training/test workers | [runs/](experiments/manuscript_20261110/runs) |
| PURE/UBFC input transforms | [rppg_dataset.py](experiments/manuscript_20261110/implementation/legacy/rppg_dataset.py) |
| COHFACE time alignment | [cohface_intra_data.py](experiments/manuscript_20261110/runs/cohface_intra/controls/cohface_intra_data.py) |
| HR scoring | [hr_evaluation_40_180.py](experiments/manuscript_20261110/implementation/src/hr_evaluation_40_180.py) |

## Attribution

Built on [PhysFormer](https://github.com/ZitongYu/PhysFormer), [BiFormer](https://github.com/rayleizhu/BiFormer), and [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox). Upstream source notices and licenses are retained. This is the BiPulseFormer research implementation, not the official repository of those projects.
