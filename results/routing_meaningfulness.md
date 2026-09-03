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
| `fftfix_retrain` (**진짜 재학습**) | (1,4,4) | fft_power | **없음 (처음부터 fps 수정된 코드로 학습)** | 8 | **1.000** | **기준(0.70) 통과 — 최종 판정** |

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

- **`fftfix_retrain` — 최종 판정 (2026-09-03)**: `scripts/run_cross_82_fftfix.py`로
  fps 수정된 코드로 **처음부터** 재학습한 체크포인트
  (`results/cross_82_pure_to_ubfc_fftfix/checkpoints/PURE_to_UBFC-rPPG_epoch3.pt`,
  best_epoch=3). 중안부 비율 100%, `results/routing_face_viz/fftfix_retrain/grid.png`
  로 8명 전원 육안 확인 — 배경/이마/머리카락이 아니라 눈-코-볼-입 중앙 2행이
  일관되게 선택되고 subject마다 세부 패턴이 다름 (퇴화 없음). **0.70 기준을 통과하며,
  이번엔 학습 자체가 fps 수정된 신호로 이루어졌으므로 sanity check가 아닌 진짜 판정이다.**
  → **Phase 1 목표(라우팅이 의미있는 얼굴 영역을 선택하는가) 달성.**

  단, cross-dataset 성능(Phase 2 목표)은 방향에 따라 다르게 나왔다 — PURE→UBFC는
  기존 phase12 대비 MAE 34% 개선(8.224→5.462)됐지만 UBFC→PURE는 거의 그대로
  (11.605→11.634). 라우팅이 옳은 곳을 보게 됐다고 cross-domain 성능이 자동으로
  비례해서 좋아지는 건 아니라는 뜻 — 자세한 원인 후보는 대화 기록 참고
  (top-k sparse attention의 cross-domain trade-off, loss schedule 미스매치,
  BatchNorm domain shift, 작은 학습 데이터, diff_routing STE 기본 활성화 confound 등).

## 4. 다음 단계

Phase 1은 완료. Phase 3(baseline 초월)에서 BN 재추정, loss schedule, BiFormer
파라미터 스윕 등을 시도해 cross-dataset 성능 자체를 개선한다. STE(diff_routing)를
껐을 때와 켰을 때의 라우팅 의미성/성능 차이는 사용자 요청에 따라 별도 ablation으로
나중에 진행한다.
