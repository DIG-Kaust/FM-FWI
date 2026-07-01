# Scripts

## Overthrust3D.py

3D acoustic full waveform inversion on the SEG/EAGE Overthrust model, using
either an L2 data-misfit objective or a SeisLM feature-space objective.

### Prerequisites

- `conda activate fmfwi` (or follow `install_env.sh` from the repo root)
- `data/overthrust.vites` placed in the repo's `data/` folder
  (see `data/README.md` for download instructions)
- For `--loss feature` or `--loss hybrid`: the SeisLM pretrained checkpoint
  under `pretrained_seislm_base/checkpoints/` (see `data/README.md`)

Run all commands from the **repo root**.

---

### Quick start — L2 loss, 10 iterations

```bash
python scripts/Overthrust3D.py \
  --loss l2 \
  --maxiter 10 \
  --exp_name test_l2
```

### Feature-space loss

```bash
python scripts/Overthrust3D.py \
  --loss feature \
  --maxiter 50 \
  --trace_chunk 512 \
  --cache_obs_embeddings_to_disk \
  --exp_name test_feature
```

### Hybrid loss (α × L2 + (1−α) × SeisLM)

```bash
python scripts/Overthrust3D.py \
  --loss hybrid \
  --alpha 0.5 \
  --maxiter 50 \
  --exp_name test_hybrid
```

### Multi-GPU (SeisLM inference across all visible GPUs)

```bash
python scripts/Overthrust3D.py \
  --loss feature \
  --maxiter 50 \
  --exp_name test_feature_multigpu
```

Use `--single_gpu` to restrict to `cuda:0` even when more GPUs are visible.

---

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--loss` | `feature` | Objective: `l2`, `feature`, or `hybrid` |
| `--maxiter` | `10` | L-BFGS-B iteration limit |
| `--exp_name` | auto | Output directory |
| `--sigma SX SY SZ` | `10 10 5` | Gaussian smoothing for the initial model (grid cells) |
| `--grad_sigma SX SY SZ` | `4 4 2` | Gradient smoothing in `PostProcessVP` |
| `--water_depth_m` | `500.0` | Water-column thickness prepended above the Overthrust model (m) |
| `--water_velocity` | `1.5` | Water velocity (km/s) |
| `--toff` | `0.5` | Time offset added to the direct-wave arrival for the refraction mute (s) |
| `--no_mask` | off | Disable the refraction mute entirely |
| `--alpha` | `0.5` | Blending weight for hybrid loss (1 = pure L2, 0 = pure SeisLM) |
| `--alpha_start/end` | — | Linearly schedule α over iterations for hybrid loss |
| `--trace_chunk` | `768` | Traces per SeisLM forward pass (reduce if GPU OOM) |
| `--cache_obs_embeddings_to_disk` | off | Persist observed SeisLM embeddings to disk for reuse |
| `--clear_dobs_cache` | off | Delete and recompute the cached observed data |
| `--ckpt_path` | auto | Path to SeisLM `.ckpt` (default: `pretrained_seislm_base/checkpoints/…`) |
| `--cpu` | off | Force CPU for SeisLM inference (slow) |
| `--single_gpu` | off | Use only `cuda:0` even when multiple GPUs are visible |
| `--init_model` | — | Path to a `.npy` file to use as initial model |
| `--linear_init` | off | Linear-gradient initial model instead of smoothed true model |

### Output files

All results are saved to `<exp_name>/`:

| File | Description |
|---|---|
| `overthrust_true.png` / `overthrust_init.png` | True and initial velocity slices |
| `gradient.png` | Normalised initial gradient |
| `dobs_parts/` | Cached per-source observed data (reused on subsequent runs) |
| `vp_current.npy` / `vp_current.png` | Latest iterate (updated after each L-BFGS step) |
| `loss_history.npy` / `vp_error_history.npy` | Running histories (updated each step) |
| `vp_inv.npy` | Final inverted model |
| `loss_plot.png` / `vp_error_plot.png` | Convergence curves |
| `result_inv.png` | Final model slices |
| `params.json` | Full run configuration |
