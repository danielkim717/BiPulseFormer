# Current manuscript baseline — 2026-11-10

Start with the [Korean colleague handoff](../../docs/SHARING_GUIDE_KO.md). This release collects **the executed K=4 baseline**, not the later routing-tuning candidates. It includes model/loss/preprocessing source, dataset-specific training workers, exact splits, current recording metrics and the common 40–180 BPM scorer.

| Location | Purpose |
|---|---|
| `implementation/src/models/bipulseformer.py` | Baseline architecture and routing |
| `implementation/src/train.py` | Direct Pearson + CE + KL loss |
| `implementation/legacy/rppg_dataset.py` | Original crop/resize/normalization and PURE training loader |
| `implementation/src/hr_evaluation_40_180.py` | Current prediction/reference HR scorer |
| `runs/*/overrides` | Exact run-specific code layered over `implementation` |
| `runs/cohface_intra/controls` | COHFACE physical-time adapter and original worker |
| `runs/*/protocol.executed.json` | Executed contracts; local path strings relocated only |
| `splits.json` | Actual train/validation/test subjects, not just ratios |
| `results_recording.json` | Unrounded final metrics, per-record HR and model identifiers |
| `evaluation_protocol.json` | Scoring, normalization and architecture details |
| `historical_ubfc` | Original UBFC split/result and common-subject comparison |
| `source_provenance.json` / `SHA256SUMS.json` | Exact source-copy provenance / public artifact integrity |
| `api.py` | CPU smoke check, baseline constructor, normalization and waveform scoring |
| `materialize.py` | Assemble an experiment's source files into a new directory; no launch |
| `TRAINING.md` | Setup, input layouts, worker commands, preparation requirements |

```bash
python experiments/manuscript_20261110/verify.py
python experiments/manuscript_20261110/api.py --smoke
```

`verify.py` needs only Python's standard library. The smoke command needs PyTorch and the dependencies in `requirements.txt`. It uses random synthetic input and **does not reproduce an accuracy result**. `api.py --waves example.npz` accepts two aligned, whole-record derivative arrays named `prediction` and `reference` and applies the current HR scorer.

The original PURE→UBFC cross implementation remains in [the 2026-09-14 source archive](../final_protocol_20260914/README.md). That archive's historical score is not silently overwritten: this release records rescoring of the same saved predictions at 40–180 BPM. The PURE/UBFC intra frozen workers also retain their executed historical scorer; use the current scorer for the current manuscript table.

**Provenance:** model/training source bytes are unchanged. `protocol.executed.json` and request templates replace machine-specific workspace/data prefixes with placeholders, so their public hashes differ from the original executed JSON hashes. Original identifiers are recorded in `run_index.json`. `materialize.py` generates a new source-subset manifest and does not pretend it is the historical full snapshot manifest. Old comments referring to earlier loss variants are retained; the worker's actual `variant='direct'` invocation and the executed contract define this release.

**Scientific scope:** the UBFC intra row is an outcome-informed post-hoc subject exchange, not a new random split; see the handoff and `splits.json`. New UBFC→PURE source training uses fresh weights and 33/9 subjects and has no final result in this release. Published baselines retain their own protocols. Raw data, caches, weights and private manuscript files are not distributed here.

Upstream PhysFormer, BiFormer and rPPG-Toolbox attribution remains in the source files and pinned vendor licenses. This is not their official repository. The Toolbox revision is `b7500b848f84ad7f86e277b4612563b69f4f88f9`; exact baseline model SHA256 is `3c6f576c83fa7c57b2ed1bc6f47d30c875970daab2b3e841dbe942ba84a6b7e7`.
