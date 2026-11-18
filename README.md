# FM-FWI

![FM-FWI](img/cover.png)

A foundation-model objective function for full waveform inversion (FWI): modeled and observed seismic traces are
compared in the feature space of a frozen, pretrained SeisLM encoder instead of (or in addition to) sample by sample,
with [Devito](https://www.devitoproject.org/) and [Devito-fwi](https://github.com/DIG-Kaust/Devito-fwi) for wave
modeling. This repository contains the code of the paper

> M. Alfarhan, M. Ravasi, F. Chen, G. Turkiyyah and D. Keyes, *Foundation Model-Assisted Full Waveform Inversion*,
> JGR: Machine Learning and Computation (manuscript 2026JH001637).

The package vendors only the part of [liutianlin0121/seisLM](https://github.com/liutianlin0121/seisLM) needed to load
the pretrained SeisLM encoder and run it: `fm_fwi/model/` is a stripped-down copy of seisLM's
`seisLM/model/foundation/` without the `lightning`, `seisbench` and `torchtune` training dependencies (checkpoints
are loaded with `torch.load`). Changes from upstream: imports renamed to `fm_fwi.model`; `torch.nn.RMSNorm` replaces
torchtune's `RMSNorm` (unused by the pretrained checkpoint, which uses LayerNorm); `pretrained.py` is new (checkpoint
loader without `lightning`, and a randomly initialized encoder of the same architecture); upstream's
`pretrained_models.py` is left out and its tests are in `pytests/`. If you use the encoder, please also cite

> T. Liu, J. Münchmeyer, L. Laurenti, C. Marone, M. V. de Hoop and I. Dokmanić, *SeisLM: A foundation model for
> seismic waveforms*, arXiv:2410.15765 (2024), https://doi.org/10.48550/arXiv.2410.15765.

## Contents

* **`fm_fwi/`**, the installable package:
  * `losses.py`: `SeisLMFeatureLoss` (full feature sequence, `pool="none"`, or time-averaged features, `pool="mean"`;
    per-trace standardization; trace chunking; optional resampling and feature-layer choice), `L2` and `HybridLoss`
    (scheduled blend of the two, each term normalized by its value summed over all shots at the starting model).
    All objectives are devitofwi `NonlinearOperator`s, so the SeisLM gradient enters adjoint-state FWI as an adjoint
    source computed by backpropagation through the frozen encoder. The observed data `b` are passed already masked
    (`b[i] = M d_obs[i]`, the mask `M` also being the operator `Op` applied to the modeled data).
  * `baselines.py`: the established misfits compared in the paper (standardized L2, envelope, trace-wise W2,
    frequency continuation, soft-DTW) and the noise models.
  * `model/`: the vendored SeisLM encoder; `plotting.py`, `tee.py`: helpers.
* **`scripts/`**, the drivers of every experiment in the paper:
  * `marm2d.py`: Marmousi (Figures 5-8);
  * `overthrust2d.py`: 2D Overthrust and all 2D tests (Figures 9-13, 18, Tables 2-3, Supporting Information);
  * `overthrust3d.py`: 3D Overthrust (Figures 14-18);
  * `boa1d.py`, `shift_diagnostics.py`, `layerwise/`: the time-shift diagnostics (Figures 3, 4, 12a, 18a, S4, S12);
  * `background_accuracy.py`: background errors of Table S3;
  * `paper_figs/`: one script per figure, reading the driver outputs in `runs/`.
* **`pytests/`**: unit tests (below).
* **`REPRODUCE.md`**: the command behind every run, figure and table of the paper.

## Installation

```
./install_env.sh
conda activate fmfwi
```
`install_env.sh` creates the `fmfwi` conda environment from `environment.yml`, installs the PyTorch build matching the
local CUDA driver (CPU-only if no GPU is found), Devito, devitofwi and this package in editable mode. The versions are
those of the runs in the paper: Python 3.10, PyTorch 2.5.1, Devito 4.8.22, devitofwi
[662ee6d](https://github.com/DIG-Kaust/Devito-fwi/commit/662ee6d2215bb12a00a4d48fe8789f71c382232e), NumPy 2.2.6,
SciPy 1.15.2, PyLops 2.7.0 and tslearn 0.9.0. The experiments were run on NVIDIA H200, A100 and RTX 3090 GPUs; the
L2 runs need only CPUs.

Then place the data and the SeisLM checkpoint as described in `data/README.md`: `data/Marm.bin`,
`data/overthrust.vites` and `pretrained_seislm_base/checkpoints/epoch=39-step=1203000.ckpt`.

## Tests

```
pip install -e .[dev]
pytest pytests/
```
* `test_baselines.py`: float64 gradient checks of the established misfits (autograd `gradcheck`) and of the SeisLM and
  hybrid losses (directional finite differences on the real encoder, both feature forms, with resampling and a mask);
  noise SNR and spectra.
* `test_losses.py`: the stage-1 time-space mask; chunked against unchunked execution of the feature loss; and
  equivalence with the module that produced the 3D results of the paper (`pytests/reference/`), loss and gradient,
  for both feature forms.
* `test_softdtw.py`: soft-DTW against a float64 reference.
* `test_conv_encoder.py`, `test_mask_utils.py`, `test_multidim_wav2vec2.py`: the vendored encoder against the
  HuggingFace Wav2Vec2 reference (these need `transformers` 4.57.6, installed by `pip install -e .[dev]`, and
  network access).

The SeisLM tests are skipped when the checkpoint is absent.

## Outputs

Each 2D run writes `runs/<name>/run.json` with its configuration, the software versions and source checksums, the
optimizer messages, and the loss and relative velocity error at every accepted iteration, together with the models
after each stage. The 3D driver writes the same information to `runs/3d/<name>/` (`params.json`, `results.json`,
`loss_history.npy`, `vp_error_local_history.npy`, `vp_current.npy`).

## License

MIT (see `LICENSE`). The vendored SeisLM code in `fm_fwi/model/` is MIT-licensed, Copyright (c) 2024 Tianlin Liu
([liutianlin0121/seisLM](https://github.com/liutianlin0121/seisLM)); its license is in `fm_fwi/model/LICENSE`.
