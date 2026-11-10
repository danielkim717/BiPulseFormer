# BiPulseFormer — 현재 논문 모델 공유 안내

**먼저 이 문서와 [실험·코드 묶음](../experiments/manuscript_20261110/README.md)을 보시면 됩니다.** 2026-11-10 기준으로 현재 논문에 사용한 baseline K=4의 모델, 학습 루프, 전처리, 실제 피험자 분할, 평가기와 결과를 한곳에 정리했습니다. 이후의 RMS/log 튜닝 후보나 과거 PHYS 결과를 현재 모델과 섞지 않았습니다.

## 선배님께 전달할 짧은 설명

> BiPulseFormer는 얼굴 영상의 시간적 특징에서 얻은 주파수 정보를 이용해 attention에 사용할 영역을 선택하는 rPPG 모델입니다. Q/K 특징을 영역 내 공간 방향으로 평균한 뒤 시간축 FFT를 적용하고, 40–180 BPM 대역의 magnitude를 평균해 영역별 descriptor를 만듭니다. Descriptor 간 dot product로 각 query의 Top-4 key 영역을 고른 뒤, 원래 Q/K/V 특징으로 attention을 계산하여 맥파를 예측합니다. 시간축은 region 분할하지 않고 얼굴 공간을 4×4로 나눕니다. 모델·학습·전처리 코드와 피험자별 split, 체크포인트 선택 방식, HR 평가 프로토콜은 아래 저장소에 함께 정리했습니다.

- 저장소: https://github.com/danielkim717/BiPulseFormer
- 이 안내: https://github.com/danielkim717/BiPulseFormer/blob/main/docs/SHARING_GUIDE_KO.md
- 코드 및 프로토콜: https://github.com/danielkim717/BiPulseFormer/tree/main/experiments/manuscript_20261110

## 어디를 보면 되나요?

아래 경로는 `experiments/manuscript_20261110/` 기준입니다.

| 확인할 내용 | 파일 | 간단한 설명 |
|---|---|---|
| 모델 구조 | `implementation/src/models/bipulseformer.py` | Q/K 투영, FFT descriptor, Top-K, STE, attention, 맥파 head |
| 실제 baseline 생성 설정 | `api.py`, `implementation/src/journal_toolbox.py` | K=4, 12 blocks, 4×4 공간 영역, 입력 30 Hz/feature 7.5 Hz |
| 손실 | `implementation/src/train.py` | negative Pearson + frequency CE + distribution KL, 가중치 각각 1 |
| PURE 학습·평가 | `runs/pure_intra/overrides/scripts/run_pure_k_ablation.py` | seed43, 30 epochs, 최종 epoch 고정 |
| UBFC intra 학습·평가 | `runs/ubfc_intra_posthoc/overrides/scripts/run_ubfc_intra_30_12.py` | seed42, 30/12명 분할, 최종 epoch 고정 |
| COHFACE 학습·평가 | `runs/cohface_intra/controls/run_cohface_intra_20261103.py` | seed43, 24/16명, 최종 epoch 고정 |
| 기본 영상 전처리 | `implementation/legacy/rppg_dataset.py` | 첫 프레임 얼굴 crop, 128×128 float resize, 차분·정규화, 학습 flip |
| COHFACE 전처리 | `runs/cohface_intra/controls/cohface_intra_data.py` | native 20 Hz 영상과 HDF5 BVP를 실제 시간 기준 30 Hz로 보간 |
| 정확한 분할 | `splits.json` | 실제 피험자 ID 및 사용 가능한 recording 목록 |
| 평가 방식 | `evaluation_protocol.json`, `implementation/src/hr_evaluation_40_180.py` | 예측·정답에 동일한 40–180 BPM 처리, recording 단위 HR |
| 수치·검증 근거 | `results_recording.json`, `source_provenance.json`, `verify.py` | 반올림 전 지표, 기록별 HR, 체크포인트·코드 hash |
| 새 UBFC→PURE cross | `runs/ubfc_to_pure/` | 33/9명 source 분할, source validation으로 선택; 아직 최종 결과 없음 |

## 실험별 분할과 선택 정책

| 실험 | 학습 / 검증 / 테스트 | 모델 선택 |
|---|---|---|
| PURE intra | 01–06 / 없음 / 07–10 (35/24 recordings; 06-02 미존재) | 30 epoch 고정 |
| UBFC-rPPG intra | 30 / 없음 / 12명 | 30 epoch 고정 |
| COHFACE intra | 공식 train+dev 24명 / 없음 / 공식 test 16명 (96/64 recordings) | 30 epoch 고정 |
| PURE→UBFC-rPPG | PURE03–10 / PURE01–02 / UBFC 42명 | 20개 source validation recording RMSE 중 최소, 동률은 먼저 나온 epoch |
| UBFC-rPPG→PURE (준비 중) | UBFC 33 / 9명 / PURE 10명·59 recordings | 30개 source validation recording RMSE 중 최소 |

UBFC intra는 기존 test 오차를 확인한 뒤 subject44·47을 학습으로 이동하고 subject12·32와 교환한 **사후 탐색 분할**입니다. 전체 분할을 무작위 분할이나 문헌 분할의 정확한 재현으로 설명하지 않습니다. 원래 30/12 분할과 결과는 `historical_ubfc/`에 남겼습니다. 새로운 UBFC→PURE 실험은 이 가중치를 재사용하지 않습니다.

학습은 batch4, Adam(lr=1e-4, weight decay=5e-5), gradient clipping=1, 수평 flip 확률 0.5를 사용합니다. Intra는 test 점수로 epoch를 고르지 않습니다. 학습 시 dense STE를 사용하고 추론 시 선택된 K/V를 gather하므로, sparse inference를 학습 연산량 절감과 동일하게 설명하지 않습니다.

## 현재 recording 결과

| 실험 | MAE (BPM) | RMSE (BPM) | MAPE (%) | Pearson |
|---|---:|---:|---:|---:|
| PURE intra | 0.26 | 0.60 | 0.33 | 0.99 |
| UBFC-rPPG intra — 사후 분할 | 0.15 | 0.51 | 0.21 | 0.99 |
| COHFACE intra | 0.82 | 3.20 | 0.96 | 0.96 |
| PURE→UBFC-rPPG | 1.09 | 2.53 | 1.23 | 0.99 |

현재 보고값은 예측과 정답 모두 **40–180 BPM**으로 산출했습니다. PURE/UBFC의 예전 worker에는 당시의 36–198 BPM 평가 호출이 남아 있으므로, 현재 표를 재현할 때는 저장 파형에 동봉한 40–180 평가기를 적용해야 합니다. 기존 source 선택은 그대로 유지합니다. 과거 PURE/UBFC 학습 HR target 검색은 원래 derivative-Welch 기본 범위 30–180, loss class는 40…179였고, COHFACE와 신규 UBFC→PURE는 학습 검색도 명시적으로 40–180입니다. 이를 서로 다른 역할의 대역으로 구분합니다.

문헌 수치는 해당 논문이 보고한 reference이며 동일 환경에서 재실행한 baseline이 아닙니다. COHFACE 비교값은 RhythmFormer Table 1을 사용합니다. PURE·UBFC intra의 자체 Pearson은 원고 정책에 따라 소수 둘째 자리에서 절삭하고, 실제 full-precision 값은 JSON에 보존했습니다.

## 실행·공유 범위

```bash
git clone https://github.com/danielkim717/BiPulseFormer.git
cd BiPulseFormer
python experiments/manuscript_20261110/verify.py
```

위 검증은 데이터나 GPU 없이 코드 무결성, 결과 집계, 분할 중복을 확인합니다. 환경 설치 후 `python experiments/manuscript_20261110/api.py --smoke`로 CPU 모델 입출력을 확인할 수 있습니다. 상세 설치 및 학습 준비는 [TRAINING.md](../experiments/manuscript_20261110/TRAINING.md)에 있습니다.

**공개한 것은 코드·분할·프로토콜·수치 근거입니다.** 원시 데이터, 대형 입력 cache, 학습 가중치와 비공개 Overleaf 원고는 업로드하지 않았습니다. 실제 재학습에는 데이터 이용권한과 로컬 경로 설정, canonical GT/cache 재생성, 입력·GPU 사전 검증이 필요합니다. 원래 작업 폴더에 종속된 frozen worker를 clone만으로 바로 학습할 수 있다고 보장하지 않습니다. 가중치 전달이 필요하면 JSON의 체크포인트 SHA256에 맞는 파일을 별도로 전달하면 됩니다.
