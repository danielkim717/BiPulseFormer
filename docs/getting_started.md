# Getting started

[Overview](../README.md) · [Method](method.md) · [Result report](results_20260917.md)

## Choose an entry point

The **current reported experiment** lives under `experiments/final_protocol_20260914/`. It contains frozen model/training/evaluation code and a result snapshot. The root `scripts/run_experiment.py` and `configs/protocol_v1.json` are earlier v1 entry points. Use [historical commands](legacy_protocol_v1.md) only when working on that protocol.

## Inspect results without datasets or a GPU

```sh
git clone https://github.com/danielkim717/BiPulseFormer.git
cd BiPulseFormer
python experiments/final_protocol_20260914/verify.py
```

This uses the Python standard library to verify implementation hashes, the upstream manifest, the fixed protocol, and recording MAE/RMSE recomputed from the exported HR pairs. It does not run a model, contact a dataset server or modify the experiment queue. Read [results.json](../experiments/final_protocol_20260914/results.json) for all completed candidates, task metrics and provenance.

## Run CPU contract tests

Create a virtual environment and install the base project dependencies. For GPU training, install a PyTorch build appropriate for your CUDA environment; the following command is not a CUDA environment specification.

```sh
python -m pip install -r requirements.txt
python experiments/final_protocol_20260914/test_contract.py
python experiments/final_protocol_20260914/source/scripts/run_final_protocol.py --help
```

The tests cover source-only selection, complete evaluation coverage, legacy input equivalence and tail handling. Passing them does not reproduce cross-test metrics. Full worker execution has additional upstream dependencies; see the [archive requirements and limitations](../experiments/final_protocol_20260914/README.md#verification-and-execution-limits).

## Train or evaluate with data

The published worker expects a request JSON containing the snapshot path, protocol and protocol SHA256, dataset-root file, canonical cache directory, source, variant, mode and a new output directory. Test mode also needs a trained model bundle. These paths and cache identities are machine-specific.

Raw datasets, checkpoint weights, verified PHYS inventory and canonical GT caches are **not included** in Git. In particular, the cache identity includes source file metadata and preprocessing configuration. Replacing a path in an old request is not a portable reproduction procedure. Do not bypass identity checks or call regenerated labels identical to the reported ones.

On the original experiment machine, source bundles contain best weights, all required provenance and a test-only wrapper; [the archive guide](../experiments/final_protocol_20260914/README.md#verification-and-execution-limits) shows its use. Each evaluation requires a new output directory. A portable data-preparation and checkpoint-distribution release is not yet provided.

## Interpret the outputs

- `per_recording` computes one HR pair per recording; `per_clip` computes short-clip HR pairs. Compare published video metrics with recording results, while disclosing differences in scoring and coverage.
- PHYS tasks share subjects. Its bootstrap resamples subjects, not tasks as separate people.
- `direct/restored` changes supervision, not the attention architecture. Best checkpoint and main variant are selected on source validation, never target metrics.
- `ineligible` denotes a failed source-label preflight; it differs from a training crash and does not create a target result.
- The public tables are dated snapshots. Older files under `results/` and `docs/legacy_scripts/` describe earlier experiments, not the live queue.
