# Fixed-protocol experiment archive — 2026-09-14

This package contains the exact implementation used for the completed PURE seed42 results reported on 2026-09-17. It is an audit archive, not a replacement for an active experiment directory. Training and data preparation do not run when importing this package or checking its hashes.

New readers: [method and routing equations](../../docs/method.md) · [getting started](../../docs/getting_started.md) · [documentation map](../../docs/README.md).

## Contents

- `results.json`: all four completed PURE target tests, including both direct/restored candidates, per-recording HR pairs, clip metrics, PHYS task metrics, subject bootstrap intervals, checkpoint and original summary hashes, and the remaining queue status at export time.
- `protocol.json`: byte-identical executed protocol. SHA256: `decc73a886d45bdd3d64be23031a1b30075cf97709846e2d9126092cddbcce96`.
- `source/scripts/run_final_protocol.py`: actual 20epoch training, source-only checkpoint selection, native input reader, overlap reconstruction and target evaluation worker.
- `source/src/train.py`: Pearson, frequency CE/KL and restored-label HR calculation.
- `source/src/journal_toolbox.py`: pinned evaluator adapters and recording/clip/task aggregation.
- `source/legacy/rppg_dataset.py`: the exact v1 input implementation reused here.
- `source/third_party/rppg_toolbox_b7500b8/`: upstream code and LICENSE, pinned to `b7500b848f84ad7f86e277b4612563b69f4f88f9`, with its original file hash manifest.
- `source/docs/protocol_audit_sources/`: upstream functions required by the reference adapter.
- `snapshot_manifest.json`: SHA256 of each exported implementation file. The runtime worker verifies this manifest before loading a model.
- `test_contract.py`: CPU tests for complete evaluation coverage, source-only 20epoch selection, v1 input equivalence, and exact-tail input handling.

The source files are copied byte-for-byte from the executed frozen snapshot. The machine-specific attendance/reporting controller and process handoff are intentionally outside this research package. The root `src/` and v1 commands are older entry points; use this archive when inspecting the new results.

## Direct versus restored

Both candidates use the same BiPulseFormer architecture, seed42, data split, 20epoch budget, Adam settings and final test evaluator. Both optimize `negative Pearson + frequency CE + distribution KL` with constant weights of one.

| Training operation | Direct | Restored |
|---|---|---|
| Prediction entering frequency loss | Normalized derivative signal | Cumulative sum, then detrend with lambda100 |
| HR target for frequency loss | Official Welch on derivative label, 30–180 BPM search | Welch on restored/detrended label, 40–180 BPM search |
| CE classes | 40–179 BPM, reject invalid targets | Same |
| Pearson term | Derivative prediction versus derivative label | Same |
| Final test | Shared GT, full common coverage, FFT and 36–198 BPM | Same |

`restored` refers to restoring a waveform before spectral supervision. It does not mean resuming an earlier checkpoint. Both PURE candidates were trained from scratch and selected epoch2 using source recording RMSE. Direct was selected as the main candidate by the predeclared source-score tie rule, before the target tests. A later epoch may have a better clip score without changing this selection.

## Verification and execution limits

From the repository root, with the project Python dependencies available:

```powershell
python experiments/final_protocol_20260914/verify.py
python experiments/final_protocol_20260914/test_contract.py
python experiments/final_protocol_20260914/source/scripts/run_final_protocol.py --help
```

The verification script uses only the standard library. The contract tests require the repository's PyTorch/torchvision, NumPy, SciPy and OpenCV dependencies. Full worker execution additionally imports upstream dependencies such as yacs, PyYAML, pandas, scikit-image, scikit-learn, tqdm, matplotlib and timm; the vendor requirements file records upstream requirements, not a tested modern environment lock.

**A fresh checkout alone cannot reproduce the numerical test results.** Raw datasets, verified PHYS inventory, canonical journal GT caches (including original file-stat identities), and checkpoint bundles are local artifacts and are not shipped here. A dataset path substitution does not recreate their identities. The original common-label hashes and coverage must be verified; do not bypass these checks or silently regenerate different labels. This archive supports code/result auditing; a portable dataset-preparation release remains separate work.

On the existing experiment machine, saved models under `models/final_protocol_20260914/PURE_direct_s42/` and `PURE_restored_s42/` include `best.pt`, config, split, selection and target request files. Their generated wrapper uses the original frozen snapshot:

```powershell
& ./models/final_protocol_20260914/PURE_direct_s42/evaluate.ps1 -Target UBFC-rPPG -Output results/manual_test/new_unique_output
```

This command starts evaluation and requires a new output directory; it is documentation, not part of the verification checks. It must not be run concurrently merely to regenerate an already reported result.

## Interpretation

See [the result report](../../docs/results_20260917.md). Published PhysFormer numbers are external reproductions, not local matched measurements. The stronger FactorizePhys reproduction is included alongside Toolbox. Different source splits, training budgets, normalization, overlap inference and historical evaluator conditions prevent a claim of controlled architectural superiority. No routing ablation or multi-seed claim is made. Older results are retained as history; this package covers only the new fixed protocol.
