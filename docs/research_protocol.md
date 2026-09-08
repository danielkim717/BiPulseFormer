# 연구 프로토콜과 변경 기준

## 고정과 탐색

`configs/protocol_v1.json`을 BiPulseFormer와 PhysFormer의 공통 기준으로 사용한다. 20epoch, 고정 LR/loss 가중치를 사용한다. PURE에서 한 사람만으로 validation을 구성하는 변동성을 줄이기 위해 intra 6:2:2를 사용한다. Cross는 source 8:2와 target 전체이다. split seed는 학습 seed와 분리한다.

새 아이디어는 설정 파일을 복사해 한 요인만 바꾼다. 설정 hash, subject ID, clip ID를 저장한다. target을 보고 epoch를 다시 고르거나 split을 옮기지 않는다. 최종 후보는 seed 42/43/44의 평균·표준편차를 모두 보고한다. 학습 seed 반복은 새로운 피험자 분할에 대한 검증을 대신하지 않는다.

## 우선 ablation

1. 동일 조건의 PhysFormer full attention과 기본 BiPulseFormer.
2. 공간 window를 고정한 mean / FFT magnitude routing.
3. STE on / off.
4. top-k 4 / 8 / 16.
5. 고정 중앙 mask / 선택 영역 routing (고정 mask는 아직 미구현).

loss schedule, window, optimizer 등을 동시에 바꾸지 않는다. 아직 수행하지 않은 학습 결과를 완료로 기록하지 않는다. 공개 SOTA 비교는 해당 논문의 평가 조건을 확인한 별도 표로 수행한다.

## 평가 원칙

- 선택: validation 160프레임 클립 HR RMSE, 겹침 step=80.
- Test: 클립 지표와 overlap-average로 복원한 영상 전체 HR 지표 모두 저장.
- HR Pearson과 파형 Pearson은 별도로 보고.
- PURE 영상 수와 실제 피험자 수를 함께 기록.
- 영상별 SE는 참고 통계이며 독립 피험자 단위 confidence interval이 아니다. RMSE SE는 delta-method 근사, 단위 BPM.
- 범위 밖 GT를 이유로 test 클립을 제거하지 않는다. HR 추정은 지정한 40–180 BPM 대역 내 peak이므로 대역 밖 정확도를 검증하지 않는다.
- 누락 피험자·읽기 실패·비정상 수치·평탄한 학습 GT는 실패로 처리한다.

## 데이터와 재현성

PURE는 원본 PPG timestamp를 각 영상 파일의 camera timestamp에 보간한다. 단순 2:1 샘플 건너뛰기를 제거했고, PPG 측정 범위를 벗어난 프레임은 보간하지 않고 수를 로그에 기록한다. UBFC는 기존 frame별 GT 정렬을 유지한다. 두 데이터셋 영상은 30fps로 처리한다. UBFC-PHYS는 시간축 정규화가 없어 고정 프로토콜에서 제외했다.

주파수 target은 차분 GT를 cumsum + smoothness-prior detrend로 복원한 뒤 Welch로 추정한다. 주파수 loss도 예측의 동일한 복원을 미분 가능한 선형 연산으로 수행한다. 차분 자체의 주파수 peak가 2차 고조파를 고르는 문제를 줄이기 위한 수정이다. 실제 PURE 01-01 첫 클립에서 기존 target은 약 153 BPM, 원본/복원 PPG peak는 약 76 BPM으로 달랐다. 따라서 이전 학습과 loss 입력 조건도 다르다.

데이터 경로는 인자로 받는다. UBFC 프레임 캐시가 없으면 로더가 데이터 위치에 캐시를 생성하므로 쓰기 권한이 필요하다. 사용 피험자와 클립 manifest를 저장하며, 원본 데이터 버전/checksum은 최종 배포 때 별도로 확보해야 한다.

Python/NumPy/PyTorch와 DataLoader seed를 고정하고 cuDNN benchmark를 끈다. 현재 PyTorch CUDA의 MaxPool3d backward에는 결정론적 구현이 없어 deterministic algorithms는 warn_only로 설정한다. 따라서 GPU bitwise 재현성을 보장하지 않으며, 이 조건을 환경 기록에 남기고 여러 seed 결과를 보고한다.

STE는 학습 시 dense 연산이다. 효율성 주장은 동일 장치·입력·batch에서 별도 추론 벤치마크가 필요하다.

STE forward는 선택 영역 logits에 masked softmax를 적용해 sparse 추론과 일치시킨다. backward 라우팅 gradient는 detached content logits에 soft region log-prior를 더한 soft attention surrogate로 전달한다. 기존 dense softmax 후 hard mask와 epsilon으로 재정규화하던 방식의 underflow를 수정한 것으로, 이전 STE 학습과 backward 규칙이 다르므로 과거 체크포인트 재학습 결과와 구분한다.

## 참고 구현

- [PhysFormer 공식 저장소](https://github.com/ZitongYu/PhysFormer)
- [rPPG-Toolbox PhysFormerTrainer](https://github.com/ubicomplab/rPPG-Toolbox/blob/main/neural_methods/trainer/PhysFormerTrainer.py)

기존 결과·권장사항 문서는 과거 기록이며 v1 실행 지침이 아니다.
