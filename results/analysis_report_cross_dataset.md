# BiPulseFormer Cross-Dataset 학습 결과 분석 리포트

작성일: 2026-05-23
작성 사유: 현재 진행 중인 `cross_82_pure_to_ubfc_const30` 결과가 paper 수준 (MAE 1.44) 대비 23x 격차

---

## 1. 현재 학습 진행 상황

**실험**: `cross_82_pure_to_ubfc_const30` (PURE → UBFC-rPPG, BiPulseFormer)
**진행률**: Epoch 3/30 완료 (5/23 14:51 기준)

| Epoch | VALID MAE | VALID ρ | TEST MAE | TEST ρ | TEST MAPE | Train Loss |
|---|---|---|---|---|---|---|
| 1 | 16.96 | 0.085 | 35.03 | 0.003 | 32.87% | 9.457 |
| 2 | 17.93 | 0.004 | 35.16 | 0.047 | 33.18% | 9.436 |
| 3 | 15.73 | 0.141 | 32.88 | -0.007 | 30.80% | 9.409 |

페이스: ~50분/epoch, 27 epoch 남음 → ~22시간 추가 예상

---

## 2. PhysFormer 논문 보고치 vs 우리 결과

| 출처 | Train→Test | MAE | RMSE | Pearson ρ | MAPE |
|---|---|---|---|---|---|
| **PhysFormer 원논문** (Yu 2022) | VIPL→MMSE-HR | **2.84** | 5.36 | **0.92** | — |
| **Spiking-PhysFormer 재현** (Table 3, rPPG-Toolbox) | PURE→UBFC | **1.44** | — | **0.98** | 1.66% |
| **우리 5/14** (`cross_82_pure2ubfc`) | PURE→UBFC (20%) | **3.81** | 10.58 | **0.625** | — |
| **우리 5/17** (`cross_82_pure_to_ubfc`) | PURE→UBFC (full) | 19.00 | 27.85 | 0.171 | — |
| **현재 5/23** (`cross_82_const30`) E3 | PURE→UBFC (full) | 32.88 | 38.22 | -0.007 | 30.80% |

⚠️ **PhysFormer 원논문 자체는 PURE↔UBFC cross-dataset 실험을 보고하지 않음.** MAE 1.44는 Spiking-PhysFormer 논문이 rPPG-Toolbox로 PhysFormer를 재현한 수치.

---

## 3. 우리 학습 방식 vs 논문 방식 비교

### 3.1 5/14 GOOD 셋업 vs 현재 5/23 셋업 차이

| 항목 | 5/14 (cross_82_pure2ubfc) | 현재 (cross_82_const30) |
|---|---|---|
| **Test set 크기** | **20% UBFC** (9 subj, 214 clips) | **100% UBFC** (42 subj, 929 clips) ❗ |
| Valid set | valid = test (RhythmFormer protocol) | 20% PURE (296 clips) |
| Train split | 80% PURE subject-exclusive | 동일 |
| Epochs | 10 | 30 |
| LR scheduler | **StepLR(50, 0.5)** — 사실상 constant | **OneCycleLR(1e-4, 30)** |
| α, β | 1.0, 1.0 constant | 1.0, 1.0 constant |
| Loss | NegPearson + Freq(CE+LD) | 동일 |
| Output norm | per-sample (axis=-1) | 동일 |
| Augmentation | random_hflip, hr_filter | 동일 |
| Data | DiffNormalized, static face crop | 동일 |
| Model params | 7.38M (BiPulseFormer) | 동일 |

**핵심 차이 2개:**
1. **Test set 크기 5배 (9 → 42 subjects)** — 가장 큰 영향
2. **LR scheduler: StepLR → OneCycleLR**

### 3.2 PhysFormer 원논문 (Yu et al., CVPR 2022) 셋업

```
Model: dim=96, ff_dim=144, num_heads=4, num_layers=12, theta=0.7, dropout=0.1
Optimizer: Adam(lr=1e-4, wd=5e-5)
Scheduler: StepLR(step_size=50, gamma=0.5)  # 25ep 동안 constant
Batch: 4
Epochs: 25
Loss: a·NegPearson + b·(CE_freq + KL_freq)
  - a = 0.1 (epoch≤25), 0.05 (epoch>25)
  - b = 1.0·5.0^(epoch/25)  # 1.0→4.78 (지수 증가)
Output norm: rPPG = (rPPG-mean)/std (global batch)
Augmentation: RandomHorizontalFlip + Normaliztion
Target: VIPL-HR (5-fold CV), Cross: VIPL→MMSE-HR
```

### 3.3 rPPG-Toolbox PhysFormerTrainer (MAE 1.44 출처 추정)

```
Model: 동일 PhysFormer
Optimizer: Adam(lr=1e-4, wd=5e-5)
Scheduler: StepLR(step_size=50, gamma=0.5)
Batch: 4
Epochs: 10 (config default — paper 25보다 적음)
Loss: a·NegPearson + b·(CE+LD)
  - a_start=1.0, exp_a=1.0 → a=1.0 constant (paper의 0.1과 다름!)
  - b_start=1.0, exp_b=1.0 → b=1.0 constant (paper의 5.0 schedule과 다름!)
Test: ENTIRE target dataset (assumed)
```

⚠️ **rPPG-Toolbox 셋업과 PhysFormer 원논문이 다름!**
- α: 0.1 (paper) vs 1.0 (rPPG-Toolbox)
- β: 1.0→4.78 (paper, exp) vs 1.0 constant (rPPG-Toolbox)
- Epochs: 25 (paper) vs 10 (rPPG-Toolbox)
- 우리는 **rPPG-Toolbox 셋업**을 따라가고 있음 (paper의 MAE 1.44가 거기서 나왔으므로)

### 3.4 우리 셋업 (현재)

```
Model: BiPulseFormer (= PhysFormer + BiLevel Routing Attention) 7.38M params
Optimizer: Adam(lr=1e-4, wd=5e-5)
Scheduler: OneCycleLR(max_lr=1e-4, epochs=30)  ← StepLR 아님!
Batch: 4
Epochs: 30 (rPPG-Toolbox 10, paper 25보다 많음)
Loss: a·NegPearson + b·(CE+LD)
  - a = 1.0 constant (rPPG-Toolbox와 일치)
  - b = 1.0 constant (rPPG-Toolbox와 일치)
Train: PURE 80% subject-exclusive
Valid: PURE 20%
Test: ENTIRE UBFC (929 clips, 42 subj)
Output norm: per-sample (rPPG-Toolbox 와 일치)
Augmentation: random_hflip + hr_filter
```

---

## 4. 왜 결과가 안 좋은가? — 원인 분석

### 원인 1: Test set 크기 (가장 큰 영향) 🔴

**5/14 결과 (MAE 3.81)**: UBFC 20% subjects (9명) 만 test → "쉬운" subset
**현재 결과 (MAE 33)**: UBFC 100% (42명) 전체 test → **모든 subject 다양성 포함**

Cross-dataset에서 일부 subject (특히 motion 많거나 어두운 영상) 가 systematic error를 만들어 평균 MAE가 크게 올라감. Spiking-PhysFormer paper의 MAE 1.44는 전체 UBFC인 것으로 추정되므로 우리 셋업과 같지만, **paper 모델은 본질적으로 더 잘 학습됐다는 의미**.

### 원인 2: OneCycleLR vs StepLR 🟡

**Paper/rPPG-Toolbox**: StepLR(50, 0.5) → 10-30 epoch 동안 LR=1e-4 constant
**현재**: OneCycleLR(1e-4, 30) → warm-up + cosine decay
- 현재 epoch 3에서 LR=2.80e-05 (peak의 28%)
- OneCycleLR은 7.5 epoch 부근에서 peak LR 도달

**문제**: 초반 epoch에서 LR이 너무 낮아 학습이 느리게 진행. StepLR이었다면 epoch 1부터 LR=1e-4로 빠른 학습 가능.

### 원인 3: BiLevel Routing Attention의 cross-domain 제약 🟡

BiPulseFormer는 PhysFormer의 full attention을 BiLevel Routing (top-k=4, 8 windows) 으로 교체:
- **장점**: intra-dataset에서 0.05-0.5 BPM MAE (우수)
- **단점**: top-k window 선택이 source-specific 패턴에 의존 → cross-domain에서 잘못된 region 선택 가능

### 원인 4: 30 epoch가 오히려 너무 많음 🟢

5/14는 10 epoch 학습 후 best epoch 3에서 MAE 3.81 달성. 30 epoch 학습은 **source overfitting** 위험 증가.

### 원인 5: Subject-exclusive 8:2 valid split의 문제 🟢

- 5/14: valid = test (target dataset 20%) — 같은 분포라 valid가 best epoch 잘 선정
- 현재: valid = source 20% (PURE) — source에서 best epoch 선정, target에선 다를 수 있음

---

## 5. 권장 조치

### 즉시 적용 가능 (학습 중단 후)
1. **OneCycleLR → StepLR(50, 0.5)** 로 변경
2. **30 → 10 epoch** 으로 단축 (rPPG-Toolbox 표준)
3. Test set은 그대로 (paper와 fair comparison 위해 full UBFC 유지)

### 비교 검증 (paper 셋업 충실 재현)
4. **별도 실험**: α=0.1, β=5^(epoch/25) schedule + 25 epoch (PhysFormer 원논문)
5. **Test set 20% 유지** (5/14 셋업 재현해서 sanity check)

### 모델 차원
6. BiPulseFormer가 cross-domain에서 약한 경우, **PhysFormer baseline (full attention) 직접 학습**해서 ablation 비교 (BiFormer가 진짜 cross에서 도움 되는지)

---

## 6. 현재 학습 진행 결정

**옵션 A**: 30 epoch 끝까지 진행 (~22시간), best epoch 추적
- 장점: 완전 결과 확보
- 단점: OneCycleLR이 후반 LR 너무 낮아져 결과 개선 불투명

**옵션 B**: 즉시 중단, StepLR + 10 epoch 셋업으로 재시작
- 장점: rPPG-Toolbox 셋업 정확 재현
- 단점: 또 처음부터, 5시간 추가 소요

**옵션 C**: 옵션 A 진행하되 epoch 10 best 확인 후 결정
- 절충안

### 결과 정리 시 보고할 표 (최종)

| Setup | Train | Test | MAE | RMSE | Pearson | MAPE |
|---|---|---|---|---|---|---|
| Paper PhysFormer (Yu 2022) | VIPL | MMSE-HR | 2.84 | 5.36 | 0.92 | — |
| Paper PhysFormer (rPPG-Toolbox) | PURE | UBFC | 1.44 | — | 0.98 | 1.66% |
| BiPulseFormer (ours, 5/14) | PURE | UBFC 20% | 3.81 | 10.58 | 0.625 | — |
| BiPulseFormer (ours, 5/17) | PURE | UBFC full | 19.00 | 27.85 | 0.171 | — |
| BiPulseFormer (ours, 5/23 const30) | PURE | UBFC full | _진행 중_ | | | |

---

## 7. 결론

현재 학습 결과가 paper 수준에 못 미치는 핵심 원인은:

1. **Test set 크기 차이** (5/14 GOOD 셋업은 UBFC 20%, 현재는 100%) — 가장 큰 영향
2. **OneCycleLR 선택 실수** (paper/rPPG-Toolbox 모두 StepLR 사용) — 초반 epoch LR 너무 낮음
3. **30 epoch 학습**이 source overfit 위험 증가 (paper 10/25 epoch 보다 김)

**우선순위 권장**: 옵션 C (현재 학습 계속, epoch 10 best 확인 후 옵션 B 결정).
