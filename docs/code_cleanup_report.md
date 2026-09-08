# 코드 정리 결과 — 2026-09-08

## 변경

- 이전 자동 튜닝/워크플로우, 관련 계획서·논문 파일·전용 결과 및 잔여 compiled 모듈을 제거했다. 사용하지 않는 의존성도 제거했다.
- 이전 ANN 실험 스크립트는 `docs/legacy_scripts/*.py.txt`로 옮겨 결과 추적용으로 보관했다. 기존 ANN 결과와 체크포인트는 유지했다.
- 공통 설정과 학습 실행기를 추가했다. 모델 선택은 source validation만 사용하고 test는 학습 후 한 번 평가한다. 설정·subject split·실제 클립·코드 hash를 저장한다.
- PURE의 단순 샘플 건너뛰기를 timestamp 보간으로 교체했다.
- 주파수 loss/HR target은 차분 신호를 복원한 PPG에서 계산한다. PURE 01-01 첫 클립에서 차분 직접 추정 약 153 BPM, 수정 후 정렬·복원 target 약 75 BPM을 확인했다. 이 한 클립 관찰은 성능 개선 결과가 아니다.
- STE attention의 underflow를 재현하고 안정적인 hard forward와 soft routing surrogate로 수정했다. 이전 학습과 backward 규칙이 다르다.
- 입력 형상·설정·빈 데이터·읽기 실패·비정상 loss 검사를 추가했다. 영상 수와 사람 수를 분리하고 RMSE 표준오차의 단위를 수정했다.
- 체크포인트 설정을 사용하는 validation 라우팅 시각화, 저장 결과 그래프, 여러 seed 집계를 추가했다. 조건·코드·사용 클립이 다른 결과의 혼합 집계를 거부한다.

## 검증

- `python -m unittest discover -s tests -v`: 12개 통과. 피험자 split, timestamp 보간, 고조파 target, STE/sparse 일치, 유한 gradient, 평가 입력 검증, 집계 조건 검증, 합성 데이터 1epoch 학습→선택→최종 test 포함.
- `python -m compileall -q src scripts tests`: 통과.
- `git diff --check`: 통과.
- PURE/UBFC 실제 각 1개 영상 로딩 및 `(3,160,128,128)` 출력 확인.
- 실제 데이터 디렉터리의 피험자 split dry-run 확인.
- RTX 4060, batch=4, 160프레임, 128px, 기본 모델 크기에서 두 모델의 forward/backward/Adam step/추론 통과. CUDA 최대 할당 메모리: PhysFormer 약 5823 MiB, BiPulseFormer 약 6203 MiB. 장기 학습 메모리나 처리량 벤치마크는 아니다.

## 현재 범위

실제 데이터셋 장기 재학습과 새 성능 평가는 아직 수행하지 않았다. 이전 결과는 새 프로토콜 성능이 아니다. CUDA MaxPool3d backward의 비결정성 때문에 bitwise 재현을 보장하지 않으며 seed·환경 기록과 반복 실험을 사용한다. 자동 학습 재개와 UBFC-PHYS 시간축 정규화는 현재 지원 범위 밖이다.
