# BiPulseFormer Cross-Dataset 학습 — Hyperparameter 권장사항

> Historical experiment notes. For the current method and results, start with the [repository overview](../README.md) and [documentation map](../docs/README.md). These notes are not the current execution plan.

작성일: 2026-05-24
근거: `cross_82_pure_to_ubfc_const30` (E1~E28) 학습 추세 분석

---

## 1. 핵심 발견 — 무엇이 잘못됐나

### 1.1 LR Schedule 진단
현재 학습 (PURE→UBFC, 30ep) 추세:

| 단계 | Epoch | LR 범위 | TEST MAE | 비고 |
|---|---|---|---|---|
| **Phase 1 (낭비)** | 1-3 | 7e-6 ~ 3e-5 | 33-35 | LR 너무 낮아 학습 무의미 |
| **Phase 2 (도약)** | 4-7 | 4e-5 ~ 9e-5 | 19→**13.4** | 급격한 개선, E7 best |
| **Phase 3 (과적합)** | 8-28 | 1e-4 → 2e-6 | 17→30 | Source overfit, TEST 계속 악화 |

**문제 요약:**
- OneCycleLR(epochs=30) → warmup 3 epoch + peak 부근 1-2 epoch + decay 25 epoch
- **실제 유용한 epoch는 4-7번 단 4번뿐**
- 나머지 24 epoch (≈80%) 는 GPU 시간 낭비 + source overfit 유발

### 1.2 학습 패턴 관찰
- **Train loss 단조 감소** (9.46→8.43): 모델 capacity 학습 잘 됨
- **VALID Pearson E7 peak (0.695) → 30ep 0.44** : 30% 감소
- **TEST Pearson E7 peak (0.384) → 30ep -0.23**: 부호 뒤집힘
- **TEST MAE E7 13.4 → 30ep 29 BPM**: 2배 악화
- **Phase oscillation**: 매 epoch 마다 TEST phase 부호 진동

---

## 2. 권장 Hyperparameter Settings

### 2.1 LR Scheduler (최우선 변경)

**🥇 Option A: StepLR (rPPG-Toolbox / PhysFormer paper 표준) — 추천**
```python
EPOCHS = 10                    # rPPG-Toolbox default
LR = 1e-4
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
# 효과: 10 epoch 동안 LR=1e-4 상수, 학습 시간 1/3, paper와 직접 비교 가능
```
- 장점: paper와 동일 → fair comparison, 학습 시간 절약
- 단점: 학습 후반 수렴 미세 조정 부재

**🥈 Option B: OneCycleLR with short warmup**
```python
EPOCHS = 15
LR = 1e-4
scheduler = OneCycleLR(optimizer, max_lr=LR, epochs=EPOCHS,
                       pct_start=0.1,        # default 0.3 → 0.1 (warmup 1.5 epoch만)
                       div_factor=10,         # initial_lr = max_lr/10
                       final_div_factor=100)  # final_lr = max_lr/1000
# 효과: epoch 2부터 peak LR 도달, 나머지 13 epoch cosine decay
```
- 장점: warmup 짧고 충분한 decay 시간
- 단점: paper와 다른 schedule

**🥉 Option C: Cosine annealing with warm restarts**
```python
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
# 5, 10, 20 epoch마다 LR 재시작 → phase oscillation 완화 가능
```

**❌ 피해야 할 셋업:**
- OneCycleLR(epochs≥25): Phase 1 낭비
- StepLR(step≤5): 너무 빠른 decay
- ReduceLROnPlateau: cross-dataset에서는 valid 기준이라 부정확

### 2.2 Epoch 수

| 셋업 | 권장 Epochs | 이유 |
|---|---|---|
| StepLR (constant LR) | **10** | 7-8 epoch에서 best 도달 후 거의 변화 없음 |
| OneCycleLR (short warmup) | **15** | peak 이후 7-8 epoch decay 적절 |
| CosineAnnealingWarmRestarts | **20** | 두 사이클 (5+10+...) 활용 |

**경고**: 30 epoch는 무조건 source overfit. 절대 사용 X.

### 2.3 Loss 가중치 (α, β)

현재: α=β=1.0 constant (rPPG-Toolbox)
- 결과: phase oscillation 심함, source overfit

**권장:**

| 설정 | α | β schedule | 기대 효과 |
|---|---|---|---|
| **현재** | 1.0 | 1.0 constant | 빠른 학습 but phase 진동 |
| **PhysFormer 원논문** | 0.1 | 1.0·5.0^(e/25) | β 점진 증가 → freq 학습 강조, 안정적 |
| **권장 (절충)** | **0.5** | **1.0·2.0^(e/E)** | 적절한 balance, phase 안정 |

### 2.4 Early Stopping

**필수 추가:**
```python
EARLY_STOP_PATIENCE = 5   # VALID best 5 epoch째 갱신 없으면 종료
```
현재 학습에서 E7 best 후 21 epoch 갱신 없음 → patience=5 면 E12에서 멈춤, **18 epoch 절약 (50시간)**.

### 2.5 Gradient Clipping

현재: max_norm=1.0 ✅ 유지
- gradient explosion 방지 효과 있음

### 2.6 Data Augmentation

현재: random_hflip + hr_filter ✅
추가 권장:
- **Random temporal crop** (clip 시작 위치 jitter): cross-dataset 일반화
- **Brightness/contrast jitter** (±10%): camera 차이 보강
- ⚠️ Random spatial crop은 face crop과 충돌 — 신중하게

---

## 3. BiLevel Routing 관련 Hyperparameter

### 3.1 Window 분할 (n_win)
현재: `n_win=(2,2,2)` → 8 windows total
- 시공간 모두 2분할
- 각 window: 20 frames × 2×2 spatial = 80 tokens

**대안 시도 가치:**
- `n_win=(1,2,2)`: 시간 분할 X, spatial만 → 더 긴 시간 컨텍스트
- `n_win=(4,2,2)`: 시간 더 세분 → 더 짧은 시간 패턴 학습
- `n_win=(2,4,4)`: spatial 더 세분 → 더 작은 region attention (계산량 ↑)

### 3.2 Top-k
현재: `topk=4` (8 windows 중 절반 선택)

| topk | 효과 | 장단점 |
|---|---|---|
| 2 | 매우 sparse | 정보 손실 ↑, 계산 ↓ |
| **4 (현재)** | 절반 sparse | 균형 |
| 6 | 거의 dense | sparse 효과 ↓ |
| 8 | dense = full attention | BiFormer 효과 없음 |

**권장 실험:**
- topk=2: sparser routing → 더 강한 region selection (의미 있는 region만 선택?)
- topk=6: cross-domain일 때는 보수적으로 많이 보는 게 안정적일 수 있음

### 3.3 Gradient sharpness (gra_sharp)
현재: 2.0 (PhysFormer paper)
- 변경 비추 (paper 표준)

---

## 4. Cross-Dataset 특화 권장

### 4.1 BatchNorm 처리 (중요)
- **현재**: BN running_stats 그대로 사용 → source 분포 통계로 target 평가
- **권장 1 (안정)**: target 데이터로 BN running_stats 재추정 (label 없이 가능)
  ```python
  model.train()  # BN running stats 업데이트 모드
  for x, _ in target_loader:
      model(x)   # forward only, no backward
  model.eval()
  ```
- **권장 2 (강건)**: `track_running_stats=False` 로 모델 학습 → eval 시 batch 통계 사용

### 4.2 Domain Adaptation 기법 (대규모 변경)
현재 setup으로는 paper 격차 극복 어려움. 다음 중 하나 도입 권장:
- **DANN** (Domain-Adversarial NN): 별도 domain classifier 추가
- **MMD** (Maximum Mean Discrepancy): feature 분포 정규화 loss 추가
- **Test-time adaptation**: SHOT, TENT 등

---

## 5. 다음 실험 추천 — 우선순위

### 🥇 우선순위 1 (즉시 시도, 30시간 → 5시간)
```python
EPOCHS = 10
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
ALPHA = 1.0  # constant
BETA = 1.0   # constant
EARLY_STOP_PATIENCE = 5
```
- 결과 예상: 현재 E7 best (MAE 13.4) 수준에서 시작 후 미세 개선
- 시간 절약: 30시간 → 5-8시간 (epoch당 50분 × 6-10 epoch)

### 🥈 우선순위 2 (loss schedule 개선)
```python
ALPHA = 0.1
BETA_schedule = lambda e: 1.0 * 5.0 ** (e / EPOCHS)
# β: 1.0 → 5.0 점진 증가
```
- 효과: freq loss 점진 강조, phase 안정화 기대

### 🥉 우선순위 3 (BN running stats 재추정)
- 추가 1시간으로 cross-domain 1-2 BPM 개선 가능

### 우선순위 4 (BiFormer 파라미터 sweep)
- `topk=2`, `n_win=(1,2,2)` 등 시도

---

## 6. 즉시 액션 권장

1. **현재 PURE→UBFC 학습 끝까지 진행** (남은 2 epoch, 1-2시간)
2. **UBFC→PURE 자동 시작 차단** (보류 결정에 따라)
3. **Best checkpoint (E7) 으로 routing analysis 먼저 진행**
4. 분석 결과 + 본 권장사항 검토 후 다음 실험 셋업 결정

---

## 부록: 현재 학습 vs 권장 셋업 시간/품질 비교 표

| 항목 | 현재 (OneCycleLR 30ep) | 권장 (StepLR 10ep) | 차이 |
|---|---|---|---|
| **학습 시간** | ~30시간 | ~8시간 | -73% |
| **GPU 시간** | 8.5 × 30시간 = 25 kWh | 8.5 × 8 = 6.8 kWh | -73% |
| **Best epoch 도달** | E7 (피크) | E5-8 (예상) | 비슷 |
| **Source overfit** | 심각 (E8+ 모두) | 미미 (E10 종료) | 개선 |
| **Paper와 비교** | 다른 schedule | 동일 (rPPG-Toolbox) | 명확 |

**핵심 결론**: OneCycleLR + 30 epoch 셋업은 8시간 일 5시간 안에 끝낼 수 있는 결과를 25시간 추가 GPU 시간으로 만들고 있음. **즉시 StepLR + 10 epoch + early stopping 으로 변경 권장**.
