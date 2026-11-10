> **2026-11-10 현재 코드:** [공유 안내](SHARING_GUIDE_KO.md)와 [현재 논문 코드 묶음](../experiments/manuscript_20261110/README.md)을 먼저 확인하세요. 아래 내용은 기존 문서 및 과거 실행에 대한 안내입니다.

# BiPulseFormer 파일 안내

얼굴 영상에서 맥파를 추정하는 연구 코드입니다. 현재 결과표를 이해하려면 **고정 실험 묶음**부터 읽고, 구현을 개발하려면 루트의 `src/`와 `scripts/`를 확인하세요. 두 경로는 같은 버전의 실험이 아닙니다.

## 먼저 볼 파일

| 파일 | 역할 |
|---|---|
| [experiments/final_protocol_20260914/README.md](../experiments/final_protocol_20260914/README.md) | 현재 공개 결과를 만든 보존 코드·의존성·실행 한계 설명 |
| [experiments/final_protocol_20260914/protocol.json](../experiments/final_protocol_20260914/protocol.json) | 실제로 사용한 데이터 분할·선정·학습·평가 계약 |
| [experiments/final_protocol_20260914/results.json](../experiments/final_protocol_20260914/results.json) | 완료한 PURE direct/restored의 목표 데이터셋 결과와 녹화별 정답·예측 |
| [experiments/final_protocol_20260914/snapshot_manifest.json](../experiments/final_protocol_20260914/snapshot_manifest.json) | 실행에 사용한 소스 파일의 SHA-256 검증 목록 |
| [experiments/final_protocol_20260914/verify.py](../experiments/final_protocol_20260914/verify.py) | 학습 없이 파일 해시와 결과 집계값을 검증하는 시작점 |
| [experiments/final_protocol_20260914/test_contract.py](../experiments/final_protocol_20260914/test_contract.py) | 피험자 분할·전체 평가 범위·입력 정렬 등을 확인하는 CPU 테스트 |
| [experiments/final_protocol_20260914/source/scripts/run_final_protocol.py](../experiments/final_protocol_20260914/source/scripts/run_final_protocol.py) | 고정 프로토콜의 학습·모델 선정·목표 데이터셋 평가 실행기 |
| [experiments/final_protocol_20260914/source/scripts/run_final_protocol_queue.py](../experiments/final_protocol_20260914/source/scripts/run_final_protocol_queue.py) | 고정 프로토콜 실험의 실행 순서와 상태를 관리하는 도구 |
| [experiments/final_protocol_20260914/source/scripts/run_source_optimized.py](../experiments/final_protocol_20260914/source/scripts/run_source_optimized.py) | 소스 데이터 기준의 후보 학습·선정 실험 실행기 |
| [experiments/final_protocol_20260914/source/src/journal_toolbox.py](../experiments/final_protocol_20260914/source/src/journal_toolbox.py) | 고정 rPPG-Toolbox 평가 함수 연결과 녹화·클립·태스크 집계 |
| [experiments/final_protocol_20260914/source/src/matched_data.py](../experiments/final_protocol_20260914/source/src/matched_data.py) | 평가 조건을 맞춘 데이터와 라벨 준비 도구 |
| [experiments/final_protocol_20260914/source/src/phys_selection.py](../experiments/final_protocol_20260914/source/src/phys_selection.py) | UBFC-PHYS 평가 대상 선정 규칙 |
| [experiments/final_protocol_20260914/source/src/toolbox_reference.py](../experiments/final_protocol_20260914/source/src/toolbox_reference.py) | 참조 rPPG-Toolbox 전처리·평가 함수 어댑터 |
| [experiments/final_protocol_20260914/source/src/data/phys_cached_dataset.py](../experiments/final_protocol_20260914/source/src/data/phys_cached_dataset.py) | 캐시된 UBFC-PHYS 입력과 라벨 읽기 |
| [experiments/final_protocol_20260914/source/src/models/bipulseformer.py](../experiments/final_protocol_20260914/source/src/models/bipulseformer.py) | 현재 공개 결과에서 사용한 BiPulseFormer 모델의 정확한 보존본 |
| [experiments/final_protocol_20260914/source/src/train.py](../experiments/final_protocol_20260914/source/src/train.py) | 현재 공개 결과의 파형·주파수 학습 손실 보존본 |

`experiments/final_protocol_20260914/source/`와 해시 목록은 실험 근거를 보존하는 영역입니다. 설명 보완을 위해 이 파일들을 수정하지 않았습니다. 기준 파일과 다른 구현을 시험할 때는 별도 실험으로 기록해야 합니다.

## 개발 및 과거 v1 코드

| 파일 | 역할 | 입력 또는 연결 관계 |
|---|---|---|
| [scripts/run_experiment.py](../scripts/run_experiment.py) | 과거 v1 학습과 테스트 실행 | configs/protocol_v1.json과 데이터 경로를 받아 src/experiment.py를 호출합니다. 현재 결과표의 고정 실험과 구분합니다. |
| [scripts/run_cross_benchmark.py](../scripts/run_cross_benchmark.py) | 과거 v1 교차 데이터 실험 일괄 실행 | 서로 다른 데이터셋·모델·시드의 실험을 순차 실행하고 비교 표를 만듭니다. 실제 학습을 시작하는 파일입니다. |
| [scripts/check_setup.py](../scripts/check_setup.py) | 모델 순전파 역전파와 학습 환경 확인 | 합성 입력으로 optimizer step까지 점검합니다. 데이터셋 학습 결과를 만드는 파일은 아닙니다. |
| [scripts/audit_data.py](../scripts/audit_data.py) | 학습 전 영상 목록과 PPG 라벨 점검 | PURE·UBFC-rPPG 경로를 읽어 시간 정렬과 학습 구간의 라벨 상태를 집계합니다. |
| [scripts/inspect_routing.py](../scripts/inspect_routing.py) | 학습 모델이 선택한 얼굴 영역 확인 | 저장 모델과 검증 영상에서 라우팅 선택을 시각화합니다. checkpoint의 모델 설정을 사용합니다. |
| [scripts/plot_results.py](../scripts/plot_results.py) | 저장된 심박 추정 결과 그래프 생성 | summary.json의 녹화별 정답·예측 쌍으로 산점도와 Bland–Altman 그림을 만듭니다. 재학습하지 않습니다. |
| [scripts/summarize_runs.py](../scripts/summarize_runs.py) | 동일 조건 반복 실험의 결과 집계 | 여러 summary.json의 설정·분할·코드·표본 일치를 확인한 뒤 평균과 변동을 계산합니다. |
| [scripts/report_cross_progress.py](../scripts/report_cross_progress.py) | 과거 실험 진행 보고 보조 도구 | 기존 작업으로 상태 보고 요청을 보내는 로컬 운영용 파일입니다. 모델 재현의 필수 단계가 아닙니다. |
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
| [tests/test_research.py](../tests/test_research.py) | 개발 코드의 분할과 모델 동작 검사 | 연구 코드의 계약과 처리 동작을 확인하는 테스트입니다. 실제 영상 정확도 평가와 다릅니다. |

## 설정과 기록

| 파일 | 역할 |
|---|---|
| [configs/protocol_v1.json](../configs/protocol_v1.json) | 과거 v1의 학습·분할·평가 설정. 현재 결과표 재현 설정이 아님 |
| [requirements.txt](../requirements.txt) | 기본 개발 의존성. 고정 실험 worker의 추가 의존성은 실행 안내 참고 |
| [results/README.md](../results/README.md) | 이전 실험의 로그·집계·모델·그림을 읽는 안내 |
| [docs/README.md](../docs/README.md) | 현재 방법·결과와 과거 문서의 목차 |
| [docs/FILE_INDEX.md](../docs/FILE_INDEX.md) | 보존한 외부 코드와 결과 이미지까지 포함한 전체 파일 목록 |

## 목적별 읽는 순서

1. **모델 이해:** 첫 화면 → [방법](method.md) → 고정본의 `bipulseformer.py`.
2. **성능 확인:** [날짜가 명시된 결과](results_20260917.md) → 고정본의 `results.json` → `verify.py`.
3. **실행 준비:** [설치와 검증](getting_started.md). 데이터·가중치 없이 할 수 있는 검증과 실제 학습을 구분합니다.
4. **이전 실험 확인:** [v1 명령](legacy_protocol_v1.md)과 `results/`. 현재 표와 서로 다른 조건의 수치를 합치지 않습니다.

[첫 화면](../README.md) · [전체 파일 목록](FILE_INDEX.md)
