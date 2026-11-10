# Model and experiment code

This package contains the K=4 BiPulseFormer baseline, preprocessing, training workers, fixed subject lists and recording-level HR evaluation.

| Path | Purpose |
|---|---|
| `implementation/src/models/bipulseformer.py` | Model and frequency-guided routing |
| `implementation/src/train.py` | Negative Pearson + frequency CE + distribution KL loss |
| `implementation/legacy/rppg_dataset.py` | Face crop, resize, time alignment and input normalization |
| `implementation/src/hr_evaluation_40_180.py` | Prediction/reference HR scorer |
| `runs/*/overrides` | Task-specific source files over the common implementation |
| `runs/cohface_intra/controls` | COHFACE loader, time adapter and training worker |
| `runs/*/protocol.template.json` | Training/evaluation configuration templates |
| `preparation/` | Dataset inventory, cache and prelaunch preparation helpers |
| `splits.json` | Explicit subject lists and recording coverage |
| `evaluation_protocol.json` | Architecture, normalization and HR evaluation settings |
| `results_recording.json` | Current aggregate performance values |
| `SOURCE_HASHES.json`, `SHA256SUMS.json` | Scientific source and package integrity |
| `api.py` | Model construction, normalization and waveform scoring |
| `materialize.py` | Export a task's source and configuration to a new directory |

[Installation and training](TRAINING.md) · [한국어 안내](../../docs/SHARING_GUIDE_KO.md)

Supported tasks: `pure_intra`, `ubfc_intra`, `cohface_intra`, `pure_to_ubfc`, `ubfc_to_pure`.

Scientific source files retain their original bytes. Configuration templates omit operational commentary and require local paths/hashes to be filled in; they are not original execution receipts. The common evaluation scorer uses 40–180 BPM. Some preserved workers emit 36–198 BPM summaries; apply the common scorer to saved merged waveforms for the table above, as described in the training guide.

Raw datasets, caches and model weights are supplied separately. The pinned rPPG-Toolbox revision is `b7500b848f84ad7f86e277b4612563b69f4f88f9`; its licenses and integrity manifest are retained. `implementation/docs/protocol_audit_sources` contains runtime loader/loss dependencies and must remain with the source tree.
