# Method: frequency-guided region routing

[Overview](../README.md) · [Results](results_20260917.md) · [Getting started](getting_started.md)

BiPulseFormer uses frequency-domain information in learned video features to guide routing toward physiologically relevant spatio-temporal regions, followed by selected-region token attention. Here, physiological relevance means an inductive bias toward pulse-band temporal activity; it is not a claim that the learned regions have been validated as anatomical pulse sources.

## From video features to routes

The backbone maps a facial clip to spatio-temporal tokens. Temporal center-difference convolutions form Q and K; a pointwise convolution forms V. Q/K/V are partitioned into regions.

For region i and feature channel c, spatial positions inside the region are averaged first. A temporal real FFT then produces a channel-wise magnitude spectrum. Averaging the magnitudes of bins inside the configured HR band gives the region descriptor:

```text
q_region[i,c] = mean over f in HR band of abs(rFFT_time(spatial_mean(Q[i])))[f,c]
k_region[j,c] = mean over f in HR band of abs(rFFT_time(spatial_mean(K[j])))[f,c]
route_score[i,j] = dot(q_region[i], k_region[j]) / sqrt(channels)
selected[i] = topk(route_score[i,:])
```

The descriptor retains feature channels but averages the selected frequency bins. It is not a frequency-by-frequency similarity score, raw-video spectrum, phase-coherence measure or squared power spectrum. The legacy identifier `fft_power` is an alias for `fft_magnitude`; `_fft_power_region` also retains its historical name. If no FFT bin lies inside the band, the implementation falls back to a window mean.

The current experiment configures `(time,height,width)=(1,4,4)` windows and top-k=4. The feature map therefore has 16 spatial regions, each spanning the full feature-time axis of the clip. Temporal information guides spatial-region selection. A phrase such as “spatio-temporal region routing” describes these feature volumes; it should not imply independently learned temporal segment boundaries.

## Attention after routing

For each query region, the inference path gathers K/V tokens from its four selected key regions. Multi-head attention compares the query tokens with that gathered token set, using the retained `gra_sharp=2.0` scale. The output is returned to the spatio-temporal layout and processed by subsequent blocks and the waveform head.

This differs from simply replacing a component by name: the region descriptors deliberately summarize pulse-band temporal activity rather than the standard mean-only region embedding. Spectral similarity is used for routing; token-level content attention still operates on learned Q/K/V features.

During training, hard top-k selection is used in the forward computation, while a soft routing prior supplies surrogate gradients through an STE. This branch computes dense attention tensors. During evaluation the implementation uses actual top-k K/V gathering. Sparse inference does **not** establish training-time memory savings, latency gains or energy efficiency; those require separate measurement.

## Three frequency settings with different jobs

| Setting | Current use |
|---|---|
| Routing band | 40–180 BPM, applied to latent Q/K descriptors; token FPS is input FPS / 4 |
| Frequency-loss classes | 40–179 BPM; out-of-range labels make a candidate ineligible |
| Evaluation band | 36–198 BPM, shared by prediction and GT HR estimation |

PURE and UBFC-rPPG use input FS30, PHYS uses FS35, so routing token FPS is respectively 7.5 and 8.75. The adapters update the routing sampling metadata for the dataset under evaluation. Ground-truth HR is not used to choose target routes.

## Direct and restored are supervision choices

Both variants use the same routing architecture. Their objective is `negative Pearson + frequency CE + distribution KL`, with constant weights of one.

| Operation | Direct | Restored |
|---|---|---|
| Prediction before frequency loss | Normalized derivative | Cumulative sum followed by detrending |
| Training HR target | Official Welch on derivative label, 30–180 BPM search | Welch on restored/detrended label, 40–180 BPM search |
| Pearson term | Derivative prediction versus derivative label | Same |
| Final target scorer and GT | Shared fixed evaluation | Same |

“Restored” means waveform restoration before spectral supervision, not restoration of a previous checkpoint. Both PURE models were trained from scratch. Source recording RMSE over all 20 checkpoints determines best; ties use the earliest epoch. Candidate ties use direct. These rules selected epoch2 for both PURE candidates and direct as the main candidate before target testing.

## Read the exact implementation

| Part | Source in the executed archive |
|---|---|
| FFT descriptors, top-k, STE and sparse gathering | [bipulseformer.py](../experiments/final_protocol_20260914/source/src/models/bipulseformer.py), `BiLevelRoutingAttention_TDC_gra_sharp` |
| Model config and routing token FPS | [journal_toolbox.py](../experiments/final_protocol_20260914/source/src/journal_toolbox.py), `model_factory`, `set_input_fps` |
| Frequency supervision | [train.py](../experiments/final_protocol_20260914/source/src/train.py), `FrequencyLoss`, `estimate_hr_targets` |
| Selection, normalization and overlap reconstruction | [run_final_protocol.py](../experiments/final_protocol_20260914/source/scripts/run_final_protocol.py) |
| Full fixed configuration | [protocol.json](../experiments/final_protocol_20260914/protocol.json) |

The archived implementation remains byte-identical to the executed snapshot, including historical comments. This guide explains its current configuration. The result comparison supports a statement about the complete method under the disclosed protocol; a routing-removal ablation is needed to isolate routing's contribution.
