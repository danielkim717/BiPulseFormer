# BiLevel Routing — "의미있는 라우팅"의 수치 기준과 실측값

작성일: 2026-09-03
관련: `.claude/plans/encapsulated-booping-micali.md` Phase 1e

## 1. 기준 (Criterion)

`scripts/visualize_routing_faces.py`가 계산하는 `row_band_ratio`를 사용:

> 특정 (체크포인트, n_win, topk, routing_mode) 조합에 대해, 서로 다른 subject를
> 아우르는 클립 8개 이상 평균에서 **중안부(mid-face: cheek/nose) row-band 비율이
> 0.70 이상**이면 "의미있는 라우팅"으로 판정한다.
>
> 4행 grid에서 중안부는 4행 중 가운데 2행을 차지하므로 균등-랜덤 기준선은 ≈0.5.
> 0.70은 그보다 확실히 높은 수준.
>
> **전제조건**: `n_win`의 spatial row가 4 이상이어야 "이마" vs "중안부" vs "입/턱"이
> 분리되어 측정 가능하다. 2×2 grid(예: `n_win=(2,2,2)`)는 top/bottom만 구분되고
> 중안부 자체를 채점할 수 없다 — 이런 config는 이 기준으로 "의미있다/없다"를 판정하지
> 않는다 (아래 표에서 "채점 불가"로 표기).

## 2. 실측값

| 실행 | n_win | routing_mode | fft_power fps 버그 | 클립 수 | mid-face 비율 | 판정 |
|---|---|---|---|---:|---:|---|
| `steplr15_mean_baseline` | (2,2,2) | mean | 해당없음 | 8 | top/bottom 50/50 (중안부 채점 불가) | 채점 불가 — 음성 대조군 (top/bottom 균등 = 신호 없음의 증거) |
| `phase12_fft_power` | (1,4,4) | fft_power | 있음 (수정 전) | 8 | **0.986** | 기준(0.70) 통과 — 단, 버그 있는 주파수축으로 학습+평가됨 |
| `phase12_fft_power_fixed` | (1,4,4) | fft_power | **없음 (수정 후 코드로 재평가만)** | 8 | **1.000** | 기준(0.70) 통과 — sanity check, 최종 판정 아님 |

## 3. 해석과 주의사항

- **`steplr15_mean_baseline`**: `n_win=(2,2,2)` 라 애초에 중안부를 못 봄. 다만
  top/bottom이 정확히 50/50으로 나온 것 자체가 `routing_analysis/routing_stats.json`의
  기존 발견("self/same-quadrant/top-half 비율이 모두 random 수준")과 일치하는
  음성 대조군 역할을 한다 — mean-routing 은 유의미한 선택을 하지 않는다는 근거.

- **`phase12_fft_power`**: Phase 1a에서 발견된 fps 버그(`_fft_power_region`이
  `fps`를 그대로 써서 실제로는 `[0.7,3.0]/4 ≈ [0.175,0.75]Hz` 대역을 선택하던 문제)가
  **있는 상태**로 학습되고 평가된 체크포인트. 그럼에도 98.6%로 기준을 통과함 —
  즉 "잘못된 주파수 대역"이었어도 중안부가 어쨌든 신호가 강한 영역이라 우연히
  잘 골랐을 가능성이 있다.

- **`phase12_fft_power_fixed`**: 같은 체크포인트(가중치는 버그 있는 축으로 학습됨)를
  **fps 수정된 코드로 eval만 다시** 돌린 것. 100%로 더 높게 나왔지만, 이건
  "학습도 fps 수정된 코드로 했을 때" 결과가 아니라 **코드 경로 자체가 정상 작동하는지의
  sanity check**일 뿐이다. `visualize_routing_faces.py`의 8개 클립 시각화
  (`results/routing_face_viz/phase12_fft_power_fixed/grid.png`)를 육안으로도 확인함 —
  8명 전원에서 배경/이마/머리카락이 아니라 눈-코-볼-입 중앙 2행이 선택되고, subject마다
  선택 패턴이 조금씩 다름 (퇴화된 상수 출력이 아님).

- **진짜 판정은 아직 안 나왔다**: fps가 수정된 코드로 **처음부터 다시 학습**한
  체크포인트에 대해 이 표를 다시 채워야 한다 (Phase 2). 지금까지의 실측값은 전부
  "버그 있는 축으로 학습된 가중치"를 다룬 것이라, 학습 자체가 잘못된 신호를 최적화한
  결과일 수도 있다.

## 4. 다음 단계

Phase 2에서 fps 수정된 `fft_power` 설정으로 cross-dataset 재학습 후, 이 표에
`phase12_fixed_retrain` 행을 추가하고 0.70 기준으로 최종 판정한다.
