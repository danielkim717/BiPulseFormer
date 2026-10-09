# 실행 도구 안내

아래는 개발·과거 v1 프로토콜의 도구입니다. 학습 실행 파일과 이미 저장된 결과를 읽는 도구를 구분해 사용합니다. 현재 결과표의 실행기는 [고정 실험](../experiments/final_protocol_20260914/README.md)에 있습니다.

| 파일 | 역할 | 사용 조건 |
|---|---|---|
| [scripts/run_experiment.py](../scripts/run_experiment.py) | 과거 v1 학습과 테스트 실행 | configs/protocol_v1.json과 데이터 경로를 받아 src/experiment.py를 호출합니다. 현재 결과표의 고정 실험과 구분합니다. |
| [scripts/run_cross_benchmark.py](../scripts/run_cross_benchmark.py) | 과거 v1 교차 데이터 실험 일괄 실행 | 서로 다른 데이터셋·모델·시드의 실험을 순차 실행하고 비교 표를 만듭니다. 실제 학습을 시작하는 파일입니다. |
| [scripts/check_setup.py](../scripts/check_setup.py) | 모델 순전파 역전파와 학습 환경 확인 | 합성 입력으로 optimizer step까지 점검합니다. 데이터셋 학습 결과를 만드는 파일은 아닙니다. |
| [scripts/audit_data.py](../scripts/audit_data.py) | 학습 전 영상 목록과 PPG 라벨 점검 | PURE·UBFC-rPPG 경로를 읽어 시간 정렬과 학습 구간의 라벨 상태를 집계합니다. |
| [scripts/inspect_routing.py](../scripts/inspect_routing.py) | 학습 모델이 선택한 얼굴 영역 확인 | 저장 모델과 검증 영상에서 라우팅 선택을 시각화합니다. checkpoint의 모델 설정을 사용합니다. |
| [scripts/plot_results.py](../scripts/plot_results.py) | 저장된 심박 추정 결과 그래프 생성 | summary.json의 녹화별 정답·예측 쌍으로 산점도와 Bland–Altman 그림을 만듭니다. 재학습하지 않습니다. |
| [scripts/summarize_runs.py](../scripts/summarize_runs.py) | 동일 조건 반복 실험의 결과 집계 | 여러 summary.json의 설정·분할·코드·표본 일치를 확인한 뒤 평균과 변동을 계산합니다. |
| [scripts/report_cross_progress.py](../scripts/report_cross_progress.py) | 과거 실험 진행 보고 보조 도구 | 기존 작업으로 상태 보고 요청을 보내는 로컬 운영용 파일입니다. 모델 재현의 필수 단계가 아닙니다. |

진행 보고 보조 도구는 로컬 작업 연동용이며 재현에 필수적인 명령이 아닙니다.

[전체 파일 안내](../docs/FILE_GUIDE.md)
