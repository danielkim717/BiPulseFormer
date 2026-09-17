# 2026-09-14 고정 실험 프로토콜

사용자가 항목별 처리 방식을 승인한 뒤 새 실험을 요청했다. 계획과 수치는 `results/final_protocol_20260914`에 별도로 저장한다. `protocol.json`과 frozen source의 SHA-256을 실행 시 확인하며, target 결과에 따라 프로토콜·선택 기준을 변경하지 않는다.

## 범위와 선택

- BiPulseFormer seed42, PURE→UBFC-rPPG·UBFC-PHYS, UBFC-PHYS→PURE·UBFC-rPPG의 네 방향.
- 각 source에서 direct/restored spectral loss 두 후보를 검토한다. 적격 후보별 20 epochs, 모든 epoch 가중치 보존. 최대 학습 4개 + test 8개, 총 12 jobs이며 방향은 4개다.
- best는 해당 source의 validation recording RMSE 최저 epoch, 동률이면 이른 epoch. 후보 간 source RMSE 동률이면 direct. 두 후보 학습 후 target test 전에 선택을 저장한다. 공식 선택 후보가 주 결과이며 다른 후보도 모두 보조 결과로 공개한다.
- CE target이 기존 40~180 BPM bin 밖이면 해당 후보는 ineligible로 기록한다. clamp·영상 제외·loss bin 변경으로 통과시키지 않는다. 다른 적격 후보는 계속 진행한다.
- 낮은 test 결과나 실패를 숨기지 않는다. 한 seed의 결과이며 seed 간 SD와 라우팅의 인과적 기여는 주장하지 않는다. 여러 차례 본 target이므로 완전히 새로운 blind test라고 표현하지 않는다.

## 고정 항목

| 항목 | 처리 |
|---|---|
| PURE source 분할 | train03~10, valid01~02; 피험자 중복 없음 |
| PHYS source 분할 | 기존 검증된 48명·101기록 중 train38명·80기록, valid10명·21기록 |
| 학습 | 20 epochs, batch4, Adam lr1e-4, weight decay5e-5, gradient clip1, scheduler 없음 |
| Loss | negative Pearson + frequency CE + distribution KL; alpha=beta=1을 20 epochs 내내 유지 |
| Direct/restored 차이 | direct는 derivative spectrum/official Welch HR; restored는 cumsum+detrend 후 spectrum/복원 Welch HR. Pearson은 derivative에 적용 |
| 증강 | 학습에서만 horizontal flip p=0.5 |
| 입력 전처리 | v1 정적 HC1.5 얼굴 box, square/center fallback, torchvision float tensor resize128·antialias; 161 raw frames→160 diff frames, clip std 정규화 |
| 학습 라벨 | PURE timestamp interpolation; PHYS native video 길이로 BVP 보간. 각 clip에서 diff/std |
| 추론 | 160-frame 출력, stride80, clip별 예측 정규화 후 겹친 위치의 derivative를 산술 평균 |
| 평가 구간 | 기존 journal의 native frame0부터 완성 160-frame chunks 전체. subject37은 decoded1170 중1120 frames/7 clips |
| 마지막 입력 frame | 완성 구간 끝이 decoded 길이와 정확히 같으면 마지막 raw frame을 한 번 반복. 평가 sample은 추가·제외하지 않음 |
| 평가 정답 | 기존 journal cache 라벨 byte/hash 그대로 사용. timestamp 재정렬·겹침 평균·후보별 재산출 금지 |
| 평가 HR·후처리 | pinned b7500b8의 FFT, cumsum, detrend100, Butterworth .6~3.3Hz: 정답과 예측 모두36~198 BPM |
| FS | PURE/UBFC-rPPG30, PHYS35; PHYS native frames 유지 |
| 최종 지표 | recording/clip MAE·RMSE(BPM), MAPE(%), Pearson; PHYS T1/T2/T3, 피험자 bootstrap |

입력 crop/resize도 과거 v1 구현을 사용한다. 따라서 현재 journal의 OpenCV INTER_AREA 입력과 동일하다고 쓰지 않는다. 평가 label·시간 구간·scorer는 공통으로 유지한다. PHYS 원본 56명·168기록 검증 완료와 실제 선별48명·101기록(T1=42,T2=26,T3=33)을 구분한다.

## 재사용·저장·실행 순서

v1은 best.pt와 last.pt만 있어 20개 checkpoint 전체의 새 recording 선택을 재현할 수 없다. 최근 source-optimized는 분할·길이·입력이 다르다. 이번 주 실험용으로 조건이 완전히 일치하는 학습 가중치는 없으므로 새로 학습한다. 기존 결과와 가중치는 그대로 보존하며, 공통 평가 라벨·영상 목록과 pinned 평가 코드는 재사용한다.

현재 `source_optimized_20260913`의 controller·child가 종료되고 terminal status가 확인되면 새 큐가 시작한다. 기존 학습을 중단하거나 frozen source를 변경하지 않는다. PHYS 원본을 streaming/seek로 읽어 대형 캐시를 추가하지 않는다.

새 가중치는 `models/final_protocol_20260914/<source>_<variant>_s42`의 best/bundle/config/split/selection과 결과 폴더의 20개 epoch checkpoint로 남긴다. 각 model 폴더의 `evaluate.ps1 -Target ... -Output <새폴더>`로 재학습 없이 같은 evaluator를 실행할 수 있다.

보고는 기존 완료·실패 이벤트 정책(progress_enabled=false)을 유지한다. 이전 큐의 reporter가 종료된 뒤 동일 thread에서 새 root로 하나만 연결한다. 보고 지연이나 퇴근 상태 때문에 새 학습이 대기하지 않도록 학습 큐와 보고 인계를 분리한다. `reporting.stop`은 자동으로 지우지 않는다.

## 문헌 비교의 범위

Toolbox Table7의 PhysFormer PURE→UBFC-rPPG 1.44/3.77/1.66/.98, Table8의 PURE→PHYS 6.04/9.77/7.67/.65는 외부 재현의 video 수치다. FactorizePhys Table2의 PhysFormer PURE→UBFC-rPPG 1.01/2.40/1.21/.990은 별도 재현이다. 본 실험의 분할·학습·loss·입력·overlap·실행 환경 차이를 명시하고 참고 비교한다. pinned 최신 코드36~198 BPM을 2023 논문 환경과 완전히 동일하다고 주장하지 않는다. PHYS-source 두 방향의 미보고 수치를 다른 방향으로 채우지 않는다.

- Toolbox: https://proceedings.neurips.cc/paper_files/paper/2023/file/d7d0d548a6317407e02230f15ce75817-Paper-Datasets_and_Benchmarks.pdf
- FactorizePhys Table2: https://arxiv.org/html/2411.01542v1#S4.T2
