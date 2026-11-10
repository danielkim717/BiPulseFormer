# BiPulseFormer

**얼굴 영상의 주파수 정보를 이용해 맥박 추정에 유용한 영역을 선택하는 비접촉 맥파 추정 모델입니다.**

일반 카메라로 촬영한 얼굴 영상에서 시간에 따라 변하는 맥파를 추정하고, 정답 센서 파형과 심박수 오차를 비교하는 연구 프로젝트입니다. PhysFormer의 시간 변화 특징과 BiFormer의 영역 선택 아이디어를 바탕으로, 주파수 정보를 영역 선택 기준에 반영했습니다.

[프로젝트 설명](#프로젝트-한눈에-보기) · [파일별 역할](docs/FILE_GUIDE.md) · [전체 파일 목록](docs/FILE_INDEX.md) · [한국어 결과 보고서](docs/results_20260917.md) · [실행 안내](docs/getting_started.md)

## 프로젝트 한눈에 보기

| 항목 | 내용 |
|---|---|
| 해결하려는 문제 | 얼굴 영상에서 맥박 관련 변화와 움직임·배경의 영향을 구분해 맥파 추정 |
| 모델 입력 | 시간 순서가 있는 얼굴 영상 클립 |
| 모델 출력 | 시간에 따른 맥파. 심박수는 별도 주파수 기반 평가로 계산 |
| 핵심 설계 | 특징의 시간축 FFT 요약 → 영역 점수 계산 → top-k 영역 선택 → 선택 영역의 토큰 attention |
| 공개한 현재 결과 | 2026-09-17 기준 PURE 학습 모델의 교차 데이터셋 평가 기록 |
| 먼저 볼 코드 | `experiments/final_protocol_20260914/`의 보존본 |
| 개발·과거 코드 | 루트 `src/`, `scripts/`, `configs/protocol_v1.json` |

## 각 폴더는 무엇인가요?

| 위치 | 하는 일 |
|---|---|
| [고정 실험 묶음](experiments/final_protocol_20260914/README.md) | 현재 결과표를 만든 모델·학습·평가 구현, 설정, 결과, 검증 코드 |
| [src](src/README.md) | 개발용 모델, 영상·정답 로더, 손실, 평가, 피험자 분할 |
| [scripts](scripts/README.md) | 과거 v1 학습 실행, 데이터 점검, 영역 시각화, 결과 집계 |
| [docs](docs/README.md) | 연구 방법·조건·성능·재현 범위·파일별 설명 |
| [results](results/README.md) | 이전 실험의 로그·모델·결과 표·그림. 현재 결과와 구분 |

## 읽을 때 구분할 점

현재 설정은 영상 특징을 공간 방향의 16개 영역으로 나누고, 각 영역에서 다른 영역 4개를 선택합니다. 주파수 정보는 시간축에서 계산하지만 짧은 시간 구간을 독립적으로 나누어 선택하는 설정은 아닙니다. 추론에서는 선택한 영역의 토큰만 모으지만 학습에서는 dense STE를 사용하므로, 학습 비용이나 실제 장치의 전력 절감이 확인되었다는 뜻은 아닙니다.

아래 방법과 결과표는 기존 공개 근거를 유지했습니다. `direct/restored`는 주파수 학습 신호의 처리 차이이며 서로 다른 라우팅 구조가 아닙니다. PHYS를 학습 원본으로 사용하는 실험의 완료 여부도 결과 스냅샷의 날짜를 기준으로 읽어야 합니다.

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
