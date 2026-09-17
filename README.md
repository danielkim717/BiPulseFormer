# BiPulseFormer

**Frequency-guided region routing and sparse attention for remote photoplethysmography (rPPG).**

BiPulseFormer estimates a pulse waveform from facial video. It uses frequency-domain information in learned features to guide routing toward physiologically relevant spatio-temporal regions, then applies token attention to the selected regions. The design builds on PhysFormer's temporal-difference features and BiFormer's region-routing principle.

주파수 정보를 이용해 맥박 추정에 유용한 시공간 영역을 선택하고, 선택된 영역에 sparse attention을 적용하는 영상 기반 맥파 추정 모델입니다. [한국어 결과 보고서](docs/results_20260917.md)

## How it works

```mermaid
flowchart LR
    A[Facial video clips] --> B[Spatio-temporal features]
    B --> C[Temporal-difference Q/K and V]
    C --> D[HR-band temporal FFT descriptors]
    D --> E[Region scores and top-k routing]
    C --> F[Selected-region K/V]
    E --> F
    F --> G[Sparse token attention at inference]
    C --> G
    G --> H[Pulse waveform]
```

1. **Extract features.** A video backbone forms spatio-temporal tokens, with temporal-difference projections for queries and keys.
2. **Build frequency descriptors.** Within each region, spatially averaged Q/K features are transformed along time. Their mean FFT **magnitude** in a configured heart-rate band forms a channel-wise descriptor.
3. **Route regions.** Query/key descriptor dot products select the top-k key regions for each query region.
4. **Attend to selected tokens.** At inference, only K/V tokens from those regions are gathered for attention; the network produces a pulse waveform.

The current experiment uses 16 regions (`n_win=(1,4,4)`) and top-k=4. Each region spans the clip's feature-time axis: this configuration selects spatial regions using temporal evidence, not independent short time segments. Physiological relevance is the design objective, not a verified anatomical segmentation. Training uses a **dense straight-through estimator (STE)**; sparse inference does not imply reduced training cost.

[Method, equations and code map →](docs/method.md)

## Results at a glance

**As of 2026-09-17:** PURE training and four target tests are complete. PHYS-source training and its two eligible target tests remain unfinished. These are recording-level HR results from seed42; MAE/RMSE are BPM and MAPE is percent. Published baselines are video-level reference comparisons.

| Train → Test | Model / source | MAE ↓ | RMSE ↓ | MAPE ↓ | Pearson ↑ |
|---|---|---:|---:|---:|---:|
| PURE → UBFC-rPPG | **BiPulseFormer direct — main** | **1.088** | **2.530** | **1.226** | **0.989** |
| PURE → UBFC-rPPG | BiPulseFormer restored | 1.423 | 3.110 | 1.590 | 0.985 |
| PURE → UBFC-rPPG | PhysFormer / Toolbox Table 7 | 1.440 | 3.770 | 1.660 | 0.980 |
| PURE → UBFC-rPPG | PhysFormer / FactorizePhys Table 2 | 1.010 | 2.400 | 1.210 | 0.990 |
| PURE → UBFC-PHYS | **BiPulseFormer direct — main** | **4.901** | **9.007** | **6.588** | **0.743** |
| PURE → UBFC-PHYS | BiPulseFormer restored | 5.767 | 10.678 | 7.608 | 0.647 |
| PURE → UBFC-PHYS | PhysFormer / Toolbox Table 8 | 6.040 | 9.770 | 7.670 | 0.650 |

References: [rPPG-Toolbox, NeurIPS 2023, Tables 7–8](https://proceedings.neurips.cc/paper_files/paper/2023/file/d7d0d548a6317407e02230f15ce75817-Paper-Datasets_and_Benchmarks.pdf#page=21), [FactorizePhys, arXiv v1, Table 2](https://arxiv.org/html/2411.01542v1#S4.T2). These are those authors' PhysFormer reproductions, not local matched measurements.

Direct has lower HR errors than the Toolbox reference in both directions, but does not exceed the stronger FactorizePhys PhysFormer reference on PURE→UBFC-rPPG. Source-validation tie rules selected direct and epoch2 **before target testing**. Both loss candidates are reported; `direct/restored` changes spectral supervision, not the routing architecture.

[Full results, clip metrics, PHYS tasks and uncertainty →](docs/results_20260917.md)

## Start here

| Your goal | Entry point |
|---|---|
| Understand the routing method | [Method and implementation](docs/method.md) |
| Inspect the reported numbers | [Result report](docs/results_20260917.md) · [Machine-readable results](experiments/final_protocol_20260914/results.json) |
| Check the exact experiment settings | [Fixed protocol](docs/final_protocol_20260914.md) |
| Verify the released code and aggregates | [Getting started](docs/getting_started.md) |
| Inspect the implementation used for these results | [Frozen code archive](experiments/final_protocol_20260914/README.md) |
| Explore earlier experiments | [Historical v1 commands](docs/legacy_protocol_v1.md) · [Documentation index](docs/README.md) |

The current result package is `experiments/final_protocol_20260914/`. Its worker, source files and upstream evaluator are preserved with SHA256 manifests. The root `src/`, `scripts/` and `configs/protocol_v1.json` also contain earlier entry points; those v1 commands do not reproduce the new table by themselves.

```text
experiments/final_protocol_20260914/
  source/                 Exact training, model and evaluation implementation
  protocol.json           Executed experiment contract
  results.json            All completed PURE candidates and provenance
  snapshot_manifest.json  Implementation hashes
  verify.py               Data-free integrity and aggregate checks
  test_contract.py        CPU coverage, input and selection tests
docs/                     Method, results, setup and historical notes
src/models/               Model implementation for development
```

## Reproducibility and scope

- **Data:** PURE→rPPG uses 42 target recordings; PURE→PHYS uses the Appendix H selection of 48 subjects / 101 recordings (T1=42, T2=26, T3=33). PHYS-source train/validation is 38/10 subjects.
- **Shared evaluation:** fixed GT and coverage, 36–198 BPM HR scoring. The 40–180 BPM routing band and training loss classes are separate settings; see the [method guide](docs/method.md).
- **Comparison limits:** source split, 20epoch budget, supervision, normalization and overlap differ from published experiments. One seed and no routing-removal ablation do not establish architectural superiority, statistical significance or SOTA. Previously observed targets are not a new blind test.
- **Artifacts:** code and result aggregates are public. Datasets, canonical caches and trained checkpoint bundles remain local; a fresh clone alone cannot regenerate the numerical results. [Requirements and execution limits](docs/getting_started.md).

## Foundations

BiPulseFormer builds on [PhysFormer](https://github.com/ZitongYu/PhysFormer) for temporal-difference video features and [BiFormer](https://github.com/rayleizhu/BiFormer) for region-level routing. Evaluation uses a pinned [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) implementation. Bundled upstream code retains its LICENSE files and commit manifest. This repository should not be presented as the official implementation of those projects.
