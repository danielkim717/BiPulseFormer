# 개발 코드 안내

이 폴더는 개발과 과거 v1 실험에 사용한 코드입니다. 현재 발표된 수치의 정확한 구현은 [고정 실험 보존본](../experiments/final_protocol_20260914/README.md)에 있습니다.

| 파일 | 역할 | 연결 |
|---|---|---|
| [src/__init__.py](../src/__init__.py) | 개발 코드 패키지 표시 | src 아래 모듈을 import할 수 있게 하는 패키지 파일입니다. |
| [src/data/alignment.py](../src/data/alignment.py) | PURE 영상과 센서 시각 정렬 | 카메라 타임스탬프에 PPG 파형을 보간하며 관측 구간 밖 라벨을 외삽하지 않습니다. |
| [src/data/rppg_dataset.py](../src/data/rppg_dataset.py) | 영상 클립과 정답 맥파 로딩 | 영상·센서 자료를 읽어 학습과 평가에 사용할 입력 클립과 라벨을 만듭니다. |
| [src/evaluation.py](../src/evaluation.py) | 녹화 단위 맥파와 심박 지표 평가 | 예측 파형과 정답을 바탕으로 심박 지표를 계산합니다. 현재 고정 실험은 별도 평가 어댑터를 사용합니다. |
| [src/evaluation_per_clip.py](../src/evaluation_per_clip.py) | 짧은 클립 단위 심박 오차 평가 | 클립별 주파수 기반 심박 추정과 MAE 등을 계산합니다. 녹화 단위 점수와 직접 섞지 않습니다. |
| [src/experiment.py](../src/experiment.py) | 과거 v1 실험의 학습 평가 흐름 | 설정과 피험자 분할을 고정하고 검증 자료에서 모델을 고른 뒤 테스트 결과를 저장합니다. |
| [src/models/bipulseformer.py](../src/models/bipulseformer.py) | BiPulseFormer 주파수 기반 영역 선택 모델 | 시공간 특징과 주파수 요약으로 영역을 선택하고 맥파를 출력합니다. 현재 표의 정확한 코드는 experiments의 고정본을 봅니다. |
| [src/models/physformer_baseline.py](../src/models/physformer_baseline.py) | PhysFormer 비교 모델 구조 | 동일 프로젝트 환경에서 비교할 PhysFormer 계열 모델을 정의합니다. README의 논문 인용 성능과 구분합니다. |
| [src/protocol.py](../src/protocol.py) | 실험 설정과 피험자 분할 관리 | 프로토콜 읽기, 피험자 목록·분할, 설정 지문 생성 기능을 모읍니다. |
| [src/train.py](../src/train.py) | 맥파 학습 손실과 주파수 정답 계산 | 파형 상관관계와 주파수 손실에 필요한 계산을 제공합니다. |
| [src/utils/__init__.py](../src/utils/__init__.py) | 공통 유틸리티 패키지 표시 | 향후 공통 도구를 위한 패키지 경로이며 현재는 별도 실행 기능이 없습니다. |

[전체 파일 안내](../docs/FILE_GUIDE.md)
