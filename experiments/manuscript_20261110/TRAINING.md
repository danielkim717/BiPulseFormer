# Installation and training handoff

## Environment and scope

Use Python 3.11. The executed environment used PyTorch 2.6.0 + CUDA12.4 and torchvision 0.21.0 on Windows/RTX4060. Install PyTorch/torchvision appropriate to your platform, then:

```bash
python -m pip install -r experiments/manuscript_20261110/requirements.txt
python experiments/manuscript_20261110/verify.py
python experiments/manuscript_20261110/api.py --smoke
```

`environment_observed.json` records installed versions. It is an environment record, not a portable lockfile. Frozen training workers call `.cuda()` and the COHFACE worker uses a Windows process lock; Linux training has not been validated. CPU inference checks do not certify a complete CUDA training run.

## Data layout

Acquire each dataset with its original permissions. Keep raw files unchanged.

```text
PURE/01-01/01-01/*.png + PURE/01-01/01-01.json   (native timestamp filenames)
UBFC-rPPG/subject1/vid.avi + ground_truth.txt    (42 available subject folders)
COHFACE/1/0/data.avi + data.hdf5                (subjects 1..40, sessions 0..3)
```

The exact PURE directory lookup is in `implementation/src/journal_toolbox.py::inventory` and `implementation/legacy/rppg_dataset.py`; preserve the released dataset nesting. The available PURE inventory contains 59 recordings: 06-02 is absent. In the executed UBFC copy, subject37 decodes 1170 frames despite 2032 declared frames. Scoring covers 1120 real frames; do not stretch its waveform to the declared duration.

## Preprocessing and caches

- PURE/UBFC inputs: first-frame Haar crop, 1.5 enlargement, square/center fallback, float bilinear resize to128² with antialiasing. Read161 RGB frames to produce160 normalized temporal differences. Train stride160; inference stride80; repeat the last input frame only for the frozen worker's exact-boundary support case.
- The **training input transform is not the Toolbox image-cache transform**. `legacy/rppg_dataset.py` and the worker's `make_reader` are authoritative. `journal_toolbox.py::prepare` creates the pinned canonical **evaluation labels and coverage**; its image cache is not substituted for the legacy model input.
- PURE training BVP uses timestamps; canonical test labels use the preserved Toolbox preparation contract. UBFC training uses native frame-index alignment. Canonical labels are normalized by the original upstream whole-record routine then chunked; do not replace them with independent per-window normalization.
- COHFACE: `cohface_intra_data.py` decodes20Hz frames, caches resized RGB losslessly in float32, interpolates RGB and timestamped HDF5 pulse to physical30Hz, and retains complete160-frame derivative chunks with a real support frame. Use the original inventory order, not a newly shuffled file traversal.

The code that performs each transform is included. Dataset files/canonical labels/pre-augmentation caches are not. `implementation/src/journal_toolbox.py::prepare` is the original canonical-cache preparation implementation for PURE/UBFC; regenerating all its image files can require substantial disk space. Run it in a new cache directory using the protocol's `make_config(...).TEST.DATA.PREPROCESS`, rather than pointing it at historical or active caches. Old receipt hashes cannot certify a newly generated cache.

## Materialize one experiment

From the repository root, choose a fresh output path:

```bash
python experiments/manuscript_20261110/materialize.py --experiment pure_intra --output work/pure_intra
```

Other choices are `ubfc_intra_posthoc`, `cohface_intra`, `ubfc_to_pure`. This creates exact source files plus path templates, **not a ready-to-train run**. It does not modify the model, create success receipts, start training, download data or resume any process.

For PURE/UBFC request-driven workers:

1. Set `roots.json` to the local absolute raw-data roots for the requested datasets. Replace `${WORKSPACE}`/`${DATA}` fields in the selected request templates. Set `snapshot` to the materialized `source_snapshot`, `cache` to the newly verified canonical cache, and `output` to a new job directory.
2. Keep all scientific fields in `protocol.executed.json`; relocate its paths and compute `protocol_sha256` from the exact resulting bytes. Point requests to that protocol. Materialized `snapshot_manifest.json` describes the exported source subset and is distinct from the historical manifest.
3. Regenerate canonical labels/coverage. Verify raw decoding and seek parity, exact raw/cached input equality including random flip/RNG behavior, subject disjointness, source HR CE targets, and a real batch4 CUDA forward/backward. The worker requires genuine preparation receipts; do not manually create `status='passed'` or bypass these checks. Worker `--preflight-only` audits source targets but is **not** the full cache/GPU check.
4. Use the unchanged worker for sequential training then testing, with separate new output directories. Set test request `bundle` to the completed train bundle. All intra workers select the prespecified final epoch30; do not search test scores.

Example worker commands after those preparations:

```bash
python work/pure_intra/source_snapshot/scripts/run_pure_k_ablation.py --request work/pure_intra/train.request.json
python work/pure_intra/source_snapshot/scripts/run_pure_k_ablation.py --request work/pure_intra/test.request.json
```

UBFC intra uses `run_ubfc_intra_30_12.py`; UBFC→PURE uses `run_ubfc_to_pure.py` and additionally requires its full prepared source cache. COHFACE uses the materialized root's `run_cohface_intra_20261103.py`, after creating `protocol.json`, relocated `inventory_audit.json`, `snapshot_manifest.json`, `control_manifest.json` and initial status. Its `--preflight` performs the worker's source/real-GPU checks before the combined prepare/train/test run. Inventory/control manifests must be regenerated against actual local files.

These launch commands document the executed entrypoints; a new machine's end-to-end data preparation/retraining has not been completed by this code-sharing task. If a recipient needs ready-to-run data or weights, transfer authorized artifacts separately and verify the published identifiers. Never copy a stale passed receipt merely to make a worker start.

## Current HR reporting

All current report values use the included `hr_evaluation_40_180.py`: reconstruct by cumulative sum, detrend100, order1 zero-phase Butterworth40–180BPM, next-power-of-two periodogram, inclusive peak search, separately for reference and prediction. Merge overlapping normalized derivative predictions before recording-level HR; do not average clip HR estimates.

For older workers' saved `test_predictions.pt`, concatenate chunk dictionaries in numerical key order, as `journal_toolbox.py::summarize_predictions` does, and apply this scorer. `api.score_record` accepts the merged derivative waves. Do **not** overwrite the original summary or reselect the source checkpoint when rescoring. COHFACE and the new UBFC→PURE worker already use the common40–180 scorer.

A checkpoint's `best.pt` name is only an epoch30 compatibility alias for intra. PURE→UBFC uses source-only selection over20epochs; new UBFC→PURE uses source-only selection over30epochs. The public result JSON provides exact checkpoint SHA256 values. No checkpoint binary is uploaded with this release.
