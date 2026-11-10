# BiPulseFormer 코드 공유 안내

현재 모델의 구현과 실행에 필요한 파일은 [`experiments/manuscript_20261110`](../experiments/manuscript_20261110)에 모았습니다. 현재 성능표는 [첫 화면](../README.md#current-model-results)에서 확인할 수 있습니다.

## 모델 구조

Q/K 특징을 영역 안에서 공간 평균한 뒤 시간축 FFT 크기로 라우팅 표현을 만듭니다. 표현 간 내적으로 Top-K 영역을 선택하고, 선택된 영역의 원래 Q/K/V 토큰으로 attention을 수행합니다. 기본 설정은 4×4 영역, K=4, 12개 블록이며 영역을 시간축으로 나누지 않습니다.

## 확인할 코드

| 내용 | 위치 (`experiments/manuscript_20261110/` 기준) |
|---|---|
| 모델과 라우팅 | `implementation/src/models/bipulseformer.py` |
| 학습 손실 | `implementation/src/train.py` |
| PURE·UBFC 입력 전처리 | `implementation/legacy/rppg_dataset.py` 및 각 worker의 reader |
| COHFACE 입력·시간 정렬 | `runs/cohface_intra/controls/cohface_intra_data.py` |
| 데이터셋별 학습·평가 루프 | `runs/` |
| 학습·검증·test 피험자 | `splits.json` |
| 설정 템플릿 | `runs/*/protocol.template.json` |
| HR 평가 | `implementation/src/hr_evaluation_40_180.py`, `evaluation_protocol.json` |

## 실행 순서

1. Python 3.11과 환경에 맞는 PyTorch/torchvision을 설치하고 루트의 `requirements.txt`를 설치합니다.
2. `python experiments/manuscript_20261110/verify.py`로 파일 무결성과 분할을 검사합니다.
3. `python experiments/manuscript_20261110/api.py --smoke`로 CPU 입력·출력 동작을 확인합니다.
4. [TRAINING.md](../experiments/manuscript_20261110/TRAINING.md)에 따라 데이터를 준비하고 `materialize.py`로 작업 폴더를 만듭니다. 로컬 경로와 설정 해시를 채우고 전처리·입력 검증을 마친 뒤 해당 worker를 실행합니다.

지원 작업은 PURE/UBFC-rPPG/COHFACE intra와 양방향 PURE↔UBFC-rPPG cross입니다. Intra는 30 epoch 학습 후 마지막 가중치를 사용하고, cross는 source validation으로 선택합니다. 분할은 `splits.json`에 있는 고정 피험자 목록을 사용합니다.

모델 입력은 128×128 RGB 161프레임에서 만든 160프레임 차분 입력입니다. 성능표는 recording 단위의 40–180 BPM 공통 scorer를 사용합니다. 보존된 일부 worker의 기본 출력 대역은 다르므로, 표와 같은 평가에는 안내된 공통 scorer를 적용해야 합니다. 데이터·캐시·가중치는 저장소에 포함하지 않습니다.
