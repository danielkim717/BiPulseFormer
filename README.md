# BiPulseFormer

PhysFormer의 temporal-difference attention에 BiLevel Routing Attention을 결합한 영상 기반 rPPG 연구 코드입니다. 목표는 고정된 실험 조건에서 성능을 개선하고 동일 조건의 비교 모델 및 공개 벤치마크로 검증하는 것입니다.

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

## 고정 프로토콜 v1

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
- 기존 `results/`는 이전 프로토콜의 기록입니다. v1과 조건이 달라 직접 합쳐 비교하지 않습니다. 이전 실행 소스는 `docs/legacy_scripts/`에 보관했습니다.
- UBFC-PHYS 로더는 보존했지만 영상별 FPS/시간 정렬 및 전체 manifest 검증 전에는 v1 대상에 포함하지 않습니다.

자세한 변경 기준은 [연구 프로토콜](docs/research_protocol.md)을 참고하세요.

시각화: `python scripts/plot_results.py <summary.json>`은 저장된 test 예측만 사용합니다. `python scripts/inspect_routing.py --run <실험 폴더>`는 체크포인트 설정으로 source validation 얼굴의 라우팅을 보여줍니다.
