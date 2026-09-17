# BiPulseFormer

PhysFormer의 temporal-difference attention에 BiLevel Routing Attention을 결합한 영상 기반 rPPG 연구 코드입니다. 목표는 고정된 실험 조건에서 성능을 개선하고 동일 조건의 비교 모델 및 공개 벤치마크로 검증하는 것입니다.

## 현재 cross 결과 — 2026-09-17

**PURE로 학습한 BiPulseFormer direct는 rPPG-Toolbox의 PhysFormer 보고값보다 PURE→UBFC-rPPG 및 PURE→UBFC-PHYS에서 recording HR 지표가 수치상 좋았습니다.** 다른 논문의 더 강한 PhysFormer 재현값과 조건 차이도 함께 공개합니다. 이는 문헌 참고 비교이며 동일 조건 PhysFormer 실측이나 라우팅 기여의 검증은 아닙니다.

아래는 seed42, source validation으로 선택한 epoch2의 결과입니다. MAE/RMSE는 BPM, MAPE는 %입니다. 문헌은 video 단위, 우리 결과는 recording 단위입니다.

| 학습 → 평가 | 모델 / 출처 | MAE ↓ | RMSE ↓ | MAPE ↓ | Pearson ↑ |
|---|---|---:|---:|---:|---:|
| PURE → UBFC-rPPG | **BiPulseFormer direct (주 결과)** | **1.088** | **2.530** | **1.226** | **0.989** |
| PURE → UBFC-rPPG | BiPulseFormer restored | 1.423 | 3.110 | 1.590 | 0.985 |
| PURE → UBFC-rPPG | PhysFormer, Toolbox Table 7 | 1.440 | 3.770 | 1.660 | 0.980 |
| PURE → UBFC-rPPG | PhysFormer, FactorizePhys Table 2 | 1.010 | 2.400 | 1.210 | 0.990 |
| PURE → UBFC-PHYS | **BiPulseFormer direct (주 결과)** | **4.901** | **9.007** | **6.588** | **0.743** |
| PURE → UBFC-PHYS | BiPulseFormer restored | 5.767 | 10.678 | 7.608 | 0.647 |
| PURE → UBFC-PHYS | PhysFormer, Toolbox Table 8 | 6.040 | 9.770 | 7.670 | 0.650 |

문헌 출처: [rPPG-Toolbox, NeurIPS 2023, Table 7·8](https://proceedings.neurips.cc/paper_files/paper/2023/file/d7d0d548a6317407e02230f15ce75817-Paper-Datasets_and_Benchmarks.pdf#page=21), [FactorizePhys, arXiv v1, Table 2](https://arxiv.org/html/2411.01542v1#S4.T2). 모두 해당 저자들의 PhysFormer 재현값입니다.

Toolbox 중심값 대비 direct의 MAE/RMSE는 rPPG에서 **24.4%/32.9%**, PHYS에서 **18.9%/7.8%** 낮습니다. FactorizePhys의 PhysFormer PURE→rPPG 재현값보다는 오차가 높습니다. 통계적 유의성이나 SOTA를 주장하지 않습니다.

- PURE 두 loss 후보의 20epoch 학습과 네 target test가 완료됐습니다. 두 후보 모두 source recording RMSE 동률이어서 사전에 고정한 direct 우선 규칙으로 주 결과를 정했습니다.
- PHYS는 원본 56명·168영상을 검증하고, Toolbox Appendix H 제외 목록을 적용한 **48명·101영상(T1=42/T2=26/T3=33)**을 평가했습니다. PHYS source는 train38명/valid10명입니다.
- PHYS restored의 20epoch 학습 및 PHYS→PURE·UBFC-rPPG test는 진행/대기 중입니다. PHYS direct는 학습 HR이 고정 loss 구간 밖이어서 부적격이며 해당 두 test 결과는 생성하지 않습니다.
- 이 결과는 **2026-09-14 고정 프로토콜**입니다. 아래 v1 실행기와 구분합니다. PURE train03–10/valid01–02, 20epochs, source recording RMSE 선택, stride80 추론, 공통 GT·평가 구간·36–198 BPM scorer를 사용합니다. 학습·loss·crop/resize·정규화·overlap 및 논문 당시 실행 환경 차이를 명시합니다.
- 한 seed라 seed 간 SD는 산출할 수 없습니다. target은 이전 탐색에서도 관찰됐으므로 완전히 새로운 blind test로 표현하지 않습니다.

[현재 결과·PHYS task·clip·남은 실험](docs/results_20260917.md) · [실행 코드와 해시가 포함된 공개 패키지](experiments/final_protocol_20260914/README.md) · [고정 프로토콜](docs/final_protocol_20260914.md)

## 코드 구조

```text
configs/protocol_v1.json     공통 실험 조건
scripts/run_experiment.py   통합 학습 실행기
scripts/summarize_runs.py   여러 seed의 완료 결과 집계
scripts/plot_results.py     저장된 test HR 결과 시각화
scripts/inspect_routing.py  validation 얼굴 라우팅 시각화
scripts/check_setup.py      실제 입력 크기 forward/backward 검사
src/protocol.py             설정 검증, 피험자 split, 실험 식별자
src/experiment.py           공통 학습·validation·최종 test
src/train.py                Pearson + 주파수 loss
src/models/                 BiPulseFormer / PhysFormer
src/data/rppg_dataset.py    데이터 로딩과 전처리
src/evaluation*.py          영상별 및 클립별 HR 평가
tests/                     split·수치 안정성·학습 경로 검증
docs/legacy_scripts/       이전 실험 소스의 비실행 텍스트 보관본
```

## 이전 고정 프로토콜 v1

이 절과 아래 실행 예시는 이전 v1 실험용입니다. 위의 현재 결과를 생성하는 코드는 `experiments/final_protocol_20260914/source/`에 별도로 보존했습니다.

| 항목 | 조건 |
|---|---|
| 데이터 | PURE, UBFC-rPPG, 30fps |
| Cross | source 피험자 80% train / 20% valid, target 전체 test |
| Intra | 피험자 60% train / 20% valid / 20% test |
| Split | 피험자 ID 정렬 후 seed 42로 셔플, 실제 ID 저장 |
| 반복 | 학습 seed 42, 43, 44; split은 동일 |
| 입력 | 160 frames × 128 × 128, DiffNormalized |
| PURE 정렬 | PPG timestamp를 영상 timestamp에 보간, 측정 범위 밖 영상은 제외 |
| 얼굴 | 첫 프레임 HaarCascade, 1.5배 box, 실패 시 중앙 crop |
| 증강 | train에만 수평 반전 |
| Optimizer | Adam, lr=1e-4 고정, weight decay=5e-5 |
| 학습 예산 | 20 epochs, batch=4, FP32, gradient clip=1.0 |
| Loss | NegPearson + CE_frequency + KL_frequency, 가중치 각 1 |
| 주파수 입력 | 차분 PPG를 cumsum + detrend로 복원한 뒤 loss/HR target 계산 |
| HR 대역 | loss target·loss bins·평가 40–180 BPM (loss 상한 제외) |
| 라우팅 | FFT magnitude, (1,4,4) windows, top-k=4, STE, tau=0.5 |
| 모델 선택 | validation 클립별 HR RMSE 최소, 동률이면 이전 epoch 유지 |
| Test | 학습 종료 후 validation-best 모델로 한 번 |
| 지표 | 클립별 및 영상별 MAE/RMSE/MAPE/HR Pearson, 파형 Pearson |

20epoch와 6:2:2는 공통 예산과 소규모 PURE validation 크기를 고려한 연구 설계입니다. 최적 조건이나 특정 논문의 재현 조건으로 주장하지 않습니다. [rPPG-Toolbox PhysFormer 학습기](https://github.com/ubicomplab/rPPG-Toolbox/blob/main/neural_methods/trainer/PhysFormerTrainer.py)를 참고했으며, v1에서는 고정 loss 가중치·공통 gradient clipping·일관된 HR 대역을 명시적으로 사용합니다.

## 실행

```powershell
python -m pip install -r requirements.txt

# 설정과 피험자 split 검사 (학습하지 않음)
python scripts/run_experiment.py --source PURE --target UBFC-rPPG --source-root D:/PURE --target-root D:/UBFC-rPPG --output results/v1/pure_to_ubfc_bi_s42 --dry-run

# BiPulseFormer cross 학습
python scripts/run_experiment.py --source PURE --target UBFC-rPPG --source-root D:/PURE --target-root D:/UBFC-rPPG --output results/v1/pure_to_ubfc_bi_s42 --seed 42

# 동일 조건 PhysFormer 비교
python scripts/run_experiment.py --model physformer --source PURE --target UBFC-rPPG --source-root D:/PURE --target-root D:/UBFC-rPPG --output results/v1/pure_to_ubfc_phys_s42 --seed 42

# Intra: source 내부에서 train / valid / test 분리
python scripts/run_experiment.py --mode intra --source PURE --source-root D:/PURE --output results/v1/intra_pure_bi_s42

python -m unittest discover -s tests -v
python scripts/check_setup.py
```

반대 방향은 source/target과 경로를 바꿉니다. 각 모델·방향을 seed 42/43/44로 반복한 뒤 `python scripts/summarize_runs.py <seed42의 summary.json> <seed43의 summary.json> <seed44의 summary.json>`으로 집계합니다. 출력 폴더는 매번 새로 지정하며 기존 결과를 덮어쓰지 않습니다. Windows 기본 worker=0, 기본 장치 CUDA입니다. `--workers`, `--device cpu`로 변경할 수 있습니다.

저장물: 설정과 hash, 피험자 split, 실제 클립 목록, 환경/소스 hash, epoch별 train loss와 validation 지표, `best.pt`, `last.pt`, 최종 test 예측과 `summary.json`. 체크포인트에 모델 설정을 포함합니다. `last.pt`는 optimizer를 포함하지만 RNG/loader 상태까지 복원하는 자동 재개는 아직 지원하지 않습니다.

## 결과 해석

- PURE의 영상 수(`n_recordings`)와 피험자 수(`n_subjects`)를 구분합니다. 영상 전체 HR 지표와 짧은 클립 HR·파형 복원 지표를 함께 봅니다.
- 논문과 비교할 때 split, 입력/평가 길이, 전처리, HR 대역, GT 산출법을 맞춰야 합니다. 기존 숫자만으로 SOTA 달성을 선언하지 않습니다.
- target 결과를 보고 epoch나 split을 다시 고르지 않습니다. 이미 여러 번 관찰한 기존 target은 완전히 새로운 blind test가 아니므로 최종 주장은 새 외부 데이터 또는 사전 고정 추가 split에서도 검증합니다.
- STE 학습은 dense attention, 추론은 sparse gather를 사용합니다. 학습 비용 절감이나 에너지 절감을 현재 주장하지 않습니다.
- `fft_magnitude`는 실제 연산을 표현하는 이름입니다. `fft_power`는 기존 설정 호환용 별칭이며 제곱 power가 아닙니다.
- 과거 프로토콜의 성능과 실패는 해당 실행 이력으로 보존합니다. 현재 cross 성능을 과거의 낮은 결과 하나로 요약하거나 서로 다른 프로토콜의 수치를 합산하지 않습니다.
- 현재 PHYS 평가에는 검증을 마친 48명·101영상 선별 집합과 FS35를 사용합니다. v1의 PURE/UBFC-rPPG 30fps 실행 예시와 구분합니다.

자세한 변경 기준은 [연구 프로토콜](docs/research_protocol.md)을 참고하세요.

시각화: `python scripts/plot_results.py <summary.json>`은 저장된 test 예측만 사용합니다. `python scripts/inspect_routing.py --run <실험 폴더>`는 체크포인트 설정으로 source validation 얼굴의 라우팅을 보여줍니다.
