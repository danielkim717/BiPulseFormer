# 이전 실험 기록 안내

이 폴더는 모델을 개발하며 수행한 이전 실험의 기록입니다. 현재 README의 표는 [고정 프로토콜 결과](../experiments/final_protocol_20260914/results.json)를 근거로 하므로 이곳에서 가장 좋은 숫자를 골라 합치지 않습니다.

| 파일 형태 | 의미 |
|---|---|
| `각 실행 폴더/summary.json` | 해당 설정·분할에서 얻은 지표와 실행 요약 |
| `각 실행 폴더/log.txt` | 학습 진행과 당시 출력 기록 |
| `각 실행 폴더/checkpoints/*.pt` | 해당 과거 실행의 학습 가중치. 현재 고정 실험의 가중치와 다름 |
| `plots/bipulseformer/scatter_*.png` | 정답과 예측 심박수의 산점도 |
| `plots/bipulseformer/bland_altman_*.png` | 두 값의 평균에 따른 예측 오차 분포 |
| `plots/bipulseformer/per_subject_*.csv` | 피험자 단위 집계 자료 |
| `routing_analysis*/routing_stats.json` | 영역 선택의 통계 기록 |
| `routing_face_viz/*/aggregate.png`, `grid.png`, `stats.json` | 영역 선택의 집계 그림, 사례 모음, 통계 |
| `routing_meaningfulness.md` | 이전 영역 선택 분석의 해석 |
| `hyperparameter_recommendations.md` | 이전 설정 탐색에서 작성한 검토 기록 |
| `mape_paper_report.txt` | 과거 MAPE 비교 보고 기록 |

파일명에 포함된 `intra/cross`, 데이터셋, epoch, 실험 태그를 함께 확인하세요. 파일명만으로 서로 동일한 평가 조건이라고 판단할 수 없습니다.

[모든 결과 파일의 위치](../docs/FILE_INDEX.md) · [현재 결과 설명](../docs/results_20260917.md)
