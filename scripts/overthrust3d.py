#!/usr/bin/env python3
r"""
3D Acoustic FWI on the SEG/EAGE Overthrust Model with a water column, no MPI,
with SeisLM Feature Loss and optional single-GPU or multi-GPU SeisLM feature
extraction.

This script is adapted from the MPI version, but deliberately removes mpi4py,
pylops_mpi, base_comm, gather/allgather, and rank-based source splitting.

Key design choices (memory on one 128-GB node):
  * One Python process only: no MPI object duplication across ranks.
  * Observed data are cached as one file per source, not as one huge MPI gather.
  * SeisLMFeatureLoss receives either one device or all visible CUDA devices.
  * Masks are forced back to float32 after filtfilt to avoid silent float64 RAM growth.
  * PerShotLazyData: observed data never loaded en masse — each shot is read from
    disk on demand, keeping ~27 GB of dobs off the heap entirely.
  * LazyMaskOpList: Diagonal mask operators built on demand, not pre-allocated —
    saves ~25 GB compared to pre-building all 70 operators at once.
  * The same mask M is applied to the modeled and the observed data: both losses get Op=M (applied to the
    modeled data) and b[i] = M d_obs[i] (PerShotLazyData with mask_ops). The runs of the submitted paper passed
    the unmasked d_obs to the L2 objective, which then compared M d_mod with d_obs; that differs only inside
    the mask taper and was corrected for the revision.
  * A constant water column is prepended above the Overthrust model. Sources and
    receivers stay at zero depth, the initial model keeps the water cells at the
    water velocity, and the gradient mask zeros the water layer.

Typical run:
  export DEVITO_LANGUAGE=openmp
  export DEVITO_MPI=0
  export OMP_NUM_THREADS=30
  python 3D_Overthrust_lbfgs_water.py --maxiter 10 --trace_chunk 768
"""

import os
import sys
import json
import shutil
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from pylops.basicoperators import Diagonal
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.signal import filtfilt
from devito import configuration

from devitofwi.waveengine.acoustic3d import AcousticWave3D
from devitofwi.postproc.acoustic import PostProcessVP

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from fm_fwi.losses import L2, SeisLMFeatureLoss  # noqa: E402
from fm_fwi.plotting import plot_slices  # noqa: E402
from fm_fwi.model.pretrained import load_pretrained_multidim_wav2vec2  # noqa: E402

configuration["log-level"] = "ERROR"


def parse_args():
    parser = argparse.ArgumentParser(
        description="No-MPI 3D Overthrust FWI with SeisLM feature loss, single or multi-GPU."
    )
    parser.add_argument("--maxiter", type=int, default=10, help="L-BFGS-B max iterations.")
    parser.add_argument("--pool", choices=["mean", "none"], default="mean",
                        help="SeisLM feature pooling: mean = time-averaged (June runs), none = full sequence.")
    parser.add_argument("--trace_chunk", type=int, default=768, help="Trace chunk used by SeisLMFeatureLoss.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Run name; output goes to runs/3d/<exp_name>. Default: overthrust3d_<loss>_<maxiter>it.",
    )
    parser.add_argument(
        "--clear_dobs_cache",
        action="store_true",
        help="Delete cached observed data before running. Equivalent to CLEAR_DOBS_CACHE=1.",
    )
    parser.add_argument(
        "--no_cache_obs_embeddings",
        action="store_true",
        help="Disable observed embedding cache in SeisLMFeatureLoss.",
    )
    parser.add_argument(
        "--cache_obs_on_gpu",
        action="store_true",
        help="Keep cached observed embeddings on GPU instead of CPU, if caching is enabled.",
    )
    parser.add_argument(
        "--gain_power",
        type=float,
        default=0.0,
        help="Time-gain exponent k in t^k folded into the refraction mask operator. "
             "0.0=off, 0.5=2D-style (cancels 1/sqrt(t)), 1.0=3D-style (cancels 1/t). "
             "Applied to both observed and modelled sides via Op for consistency. "
             "Default: 0.0",
    )
    parser.add_argument(
        "--gain_t0",
        type=float,
        default=0.0,
        help="Time offset (s) added before computing t^k. Use a positive value to "
             "soften the late-vs-early gain ratio (e.g. --gain_t0 1.0 with --gain_power 0.5 "
             "gives sqrt(t+1) instead of sqrt(t)). Default: 0.0",
    )
    parser.add_argument(
        "--single_gpu",
        action="store_true",
        help="Force SeisLMFeatureLoss to use only cuda:0 even if multiple GPUs are visible.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU for SeisLMFeatureLoss. Mostly for debugging; slow.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=os.path.join(REPO, "pretrained_seislm_base/checkpoints/epoch=39-step=1203000.ckpt"),
        help="Path to SeisLM checkpoint.",
    )
    # --- Refraction mask ---
    parser.add_argument(
        "--toff", type=float, default=0.5,
        help="Time offset (s) added to direct-wave arrival for refraction mute. Default: 0.5",
    )
    parser.add_argument(
        "--water_depth_m",
        type=float,
        default=500.0,
        help="Water-column thickness in meters prepended above the Overthrust model. Default: 500.",
    )
    parser.add_argument(
        "--water_velocity",
        type=float,
        default=1.5,
        help="Water velocity in km/s used in true/initial models and direct-arrival times. Default: 1.5.",
    )
    # --- Loss function ---
    parser.add_argument(
        "--loss", type=str, default="feature", choices=["feature", "l2"],
        help="Stage-1 objective: 'feature' (SeisLM feature loss) or 'l2'. Default: feature",
    )
    parser.add_argument(
        "--no_mask", action="store_true",
        help="Disable refraction mask (Op=None). By default the mask is applied.",
    )
    # --- Initial model smoothing ---
    parser.add_argument(
        "--sigma", type=float, nargs=3, default=[10.0, 10.0, 5.0],
        metavar=("SX", "SY", "SZ"),
        help="Gaussian sigma (x y z) in grid cells for initial model smoothing. Default: 10 10 5",
    )
    parser.add_argument(
        "--init_model", type=str, default=None,
        help="Path to a .npy file to use as the initial model instead of the smoothed true model.",
    )
    parser.add_argument(
        "--linear_init",
        action="store_true",
        help="Use a laterally constant initial subsurface model that increases linearly "
             "with depth from vp_true.min() to vp_true.max(), then prepend the water layer. "
             "By default, the initial model is a Gaussian-smoothed true model.",
    )
    parser.add_argument(
        "--init_true_top_m",
        type=float,
        default=0.0,
        help="Depth in meters from the surface to copy from the true model into the initial model. "
             "Default: 0.0 (disabled).",
    )
    # --- Gradient post-processing ---
    parser.add_argument(
        "--grad_sigma", type=float, nargs=3, default=[4.0, 4.0, 2.0],
        metavar=("SX", "SY", "SZ"),
        help="Gaussian sigma (x y z) for gradient smoothing in PostProcessVP. Default: 4 4 2",
    )
    parser.add_argument(
        "--grad_mask_type",
        type=str,
        default="shallow_divide",
        choices=["shallow_divide", "soft"],
        help="Depth mask used for gradient post-processing. Default: shallow_divide",
    )
    parser.add_argument(
        "--grad_shallow_divide_to_km",
        type=float,
        default=0.20,
        help="Depth in km for the shallow-divide gradient mask. Cells shallower than "
             "or equal to this depth have the gradient divided by --grad_shallow_divide_k. "
             "Default: 0.20.",
    )
    parser.add_argument(
        "--grad_shallow_divide_k",
        type=float,
        default=10.0,
        help="Divisor k for the shallow-divide gradient mask. Default: 10.0.",
    )
    parser.add_argument(
        "--grad_soft_taper_to_km",
        type=float,
        default=0.50,
        help="Depth in km where the soft gradient mask reaches 1.0. Default: 0.50",
    )
    parser.add_argument(
        "--grad_soft_power",
        type=float,
        default=1.5,
        help="Power applied to the raised-cosine soft gradient mask. Default: 1.5",
    )
    # --- Model-error metric window ---
    # The relative model-error norm `||(xk - vp_true) / vp_true||` is dominated
    # by whichever cells have the largest error. Without a window, that's
    # almost always the source-region near-surface (top) or the
    # poorly-illuminated bottom, neither of which reflect inversion quality
    # in the region we care about. These margins (in km) clip the metric
    # to the well-illuminated interior. Set to 0 to disable any margin.
    parser.add_argument(
        "--metric_margin_x", type=float, default=0.5,
        help="Margin in km excluded from the model-error metric on each x-edge. Default: 0.5",
    )
    parser.add_argument(
        "--metric_margin_y", type=float, default=0.5,
        help="Margin in km excluded from the model-error metric on each y-edge. Default: 0.5",
    )
    parser.add_argument(
        "--metric_margin_top", type=float, default=0.10,
        help="Depth in km from the surface excluded from the model-error metric. Default: 0.10",
    )
    parser.add_argument(
        "--metric_margin_bottom", type=float, default=0.50,
        help="Thickness in km at the bottom of the model excluded from the model-error metric. Default: 0.50",
    )
    return parser.parse_args()


def log_memory(tag):
    # Current RSS (not high-watermark)
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                current_rss_gb = int(line.split()[1]) / 1024**2
                break
    print(f"[memory] {tag}: CPU current RSS = {current_rss_gb:.2f} GB", flush=True)

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(
                f"[memory] {tag}: cuda:{i} allocated={alloc:.2f} GB, reserved={reserved:.2f} GB",
                flush=True,
            )


class TimeSpaceMasking3D:
    """Memory-aware refraction mask for 3D acquisition.

    Stores arrival times and builds an (nt, nrec) mask one shot at a time.
    The mask is always returned as contiguous float32. This avoids filtfilt
    promoting masks to float64 and quietly doubling RAM use.
    """

    def __init__(self, arrival, nt, dt, toff=0.0, nsmooth=None):
        self.arrival = arrival.astype(np.float32, copy=False)
        self.nt = int(nt)
        self.dt = float(dt)
        self.toff = float(toff)
        self.nsmooth = nsmooth
        self._t_idx = np.arange(self.nt)

    def mask_oneshot(self, isrc):
        it_cut = np.round((self.arrival[isrc] + self.toff) / self.dt).astype(int).clip(0, self.nt)
        mask = (self._t_idx[:, None] < it_cut[None, :]).astype(np.float32)

        if self.nsmooth is not None:
            b = np.ones(self.nsmooth, dtype=np.float32) / np.float32(self.nsmooth)
            mask = filtfilt(b, 1, mask, axis=0)

        return np.ascontiguousarray(mask, dtype=np.float32)


class LazyMaskOpList(list):
    """Build Diagonal mask operators on demand.

    Inherits from `list` (not `Sequence`) so that `isinstance(self.Op, list)`
    checks in third-party code (e.g. devitofwi.loss.l2.L2) succeed and the
    per-shot operator is correctly retrieved via `self.Op[i]`.
    """

    def __init__(self, mask_fn, n, dtype="float32"):
        super().__init__([None] * n)   # stub list of correct length
        self._mask_fn = mask_fn
        self._dtype = dtype
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return self._mask_fn(i)


class PerShotLazyData:
    """Load one shot at a time from per-source .npy files.

    Avoids holding the full observed data array in RAM (~27 GB).
    Returns a flat (nt*nrec,) float32 array, the shape that SeisLMFeatureLoss and L2 expect from b[i]:
    the shot masked by mask_ops[i] when mask_ops is given, the raw shot otherwise.
    """

    def __init__(self, source_file_fn, n, nt, nrec, mask_ops=None):
        self._source_file_fn = source_file_fn
        self._n = n
        self._nt = nt
        self._nrec = nrec
        self._mask_ops = mask_ops

    def __len__(self):
        return self._n

    @property
    def size(self):
        return self._n * self._nt * self._nrec

    def __getitem__(self, i):
        shot = np.load(self._source_file_fn(i)).astype(np.float32).ravel()   # (nt*nrec,)
        if self._mask_ops is not None:
            shot = (self._mask_ops[i] @ shot).astype(np.float32)
        return shot


def build_depth_shallow_divide(nz, dz, shallow_to_km=0.20, k=10.0):
    """1-D depth mask that scales shallow gradients by 1/k."""
    z = np.arange(nz) * dz
    mask = np.ones(nz, dtype=np.float32)
    mask[z <= shallow_to_km + 1e-6] = np.float32(1.0 / k)
    return mask


def build_depth_taper_soft(nz, dz, taper_to_km=0.50, power=1.5):
    """1-D raised-cosine depth mask that smoothly grows from 0 to 1."""
    z = np.arange(nz) * dz
    base = np.where(
        z < taper_to_km,
        0.5 * (1 - np.cos(np.pi * z / taper_to_km)),
        1.0,
    )
    return (base ** power).astype(np.float32)


def prepend_water_column(vp_subsurface, nwater, water_velocity):
    """Prepend a constant water layer along z and return a float32 model."""
    if nwater <= 0:
        return vp_subsurface.astype(np.float32, copy=False)
    water = np.full(
        (vp_subsurface.shape[0], vp_subsurface.shape[1], nwater),
        np.float32(water_velocity),
        dtype=np.float32,
    )
    return np.concatenate([water, vp_subsurface.astype(np.float32, copy=False)], axis=2)


def main():
    args = parse_args()
    if args.init_model is not None and args.linear_init:
        raise ValueError("--init_model and --linear_init are mutually exclusive")
    if args.init_true_top_m < 0.0:
        raise ValueError("--init_true_top_m must be non-negative")
    if args.grad_shallow_divide_to_km < 0.0:
        raise ValueError("--grad_shallow_divide_to_km must be non-negative")
    if args.grad_shallow_divide_k <= 0.0:
        raise ValueError("--grad_shallow_divide_k must be positive")
    if args.grad_soft_taper_to_km <= 0.0:
        raise ValueError("--grad_soft_taper_to_km must be positive")
    if args.grad_soft_power <= 0.0:
        raise ValueError("--grad_soft_power must be positive")
    if args.water_depth_m < 0.0:
        raise ValueError("--water_depth_m must be non-negative")
    if args.water_velocity <= 0.0:
        raise ValueError("--water_velocity must be positive")

    base_nz = 90
    dz_km = 0.05
    water_depth_km = float(args.water_depth_m) / 1000.0
    nwater = int(round(water_depth_km / dz_km))
    actual_water_depth_km = nwater * dz_km
    model_nz = base_nz + nwater

    maxiter = args.maxiter
    exp_name = os.path.join(REPO, "runs/3d", args.exp_name or f"overthrust3d_{args.loss}_{maxiter}it")

    print("3D Acoustic FWI — Overthrust SeisLM Feature Loss, no MPI")
    print(f"Experiment: {exp_name}")
    os.makedirs(exp_name, exist_ok=True)

    ##################################################################
    # Save experiment parameters
    ##################################################################
    import datetime
    params_dict = {
        "experiment": exp_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "loss": args.loss,
        "pool": args.pool,
        "maxiter": args.maxiter,
        "trace_chunk": args.trace_chunk,
        "toff": args.toff,
        "water_depth_m": args.water_depth_m,
        "water_velocity": args.water_velocity,
        "no_mask": args.no_mask,
        "sigma": args.sigma,
        "grad_sigma": args.grad_sigma,
        "init_model": args.init_model,
        "linear_init": args.linear_init,
        "init_true_top_m": args.init_true_top_m,
        "ckpt_path": args.ckpt_path,
        "cache_obs_embeddings": not args.no_cache_obs_embeddings,
        "cache_obs_on_gpu": args.cache_obs_on_gpu,
        "single_gpu": args.single_gpu,
        "cpu": args.cpu,
        "metric_window": {
            "margin_x_km":      args.metric_margin_x,
            "margin_y_km":      args.metric_margin_y,
            "margin_top_km":    args.metric_margin_top,
            "margin_bottom_km": args.metric_margin_bottom,
        },
        "time_gain": {
            "power": args.gain_power,
            "t0_s":  args.gain_t0,
        },
        "model": {
            "nx": 200, "ny": 140, "nz": model_nz,
            "base_nz": base_nz, "water_nz": nwater,
            "dx": 0.05, "dy": 0.05, "dz": 0.05,
            "nbl": 20, "space_order": 4,
            "freq": 5.0, "t0": 0.0, "tn": 5.0,
        },
        "water_column": {
            "requested_depth_m": args.water_depth_m,
            "actual_depth_m": actual_water_depth_km * 1000.0,
            "n_cells": nwater,
            "velocity_km_s": args.water_velocity,
            "source_receiver_depth_km": 0.0,
        },
        "acquisition": {
            "nsrc_x": 10, "nsrc_y": 7, "nrec_x": 200, "nrec_y": 140,
            "dsx": 1.0, "dsy": 1.0, "drx": 0.05, "dry": 0.05,
        },
        "postproc": {
            "grad_mask": args.grad_mask_type,
            "shallow_divide_to_km": args.grad_shallow_divide_to_km,
            "shallow_divide_k": args.grad_shallow_divide_k,
            "soft_taper_to_km": args.grad_soft_taper_to_km,
            "soft_power": args.grad_soft_power,
            "grad_sigmas": args.grad_sigma,
            "init_true_top_m": args.init_true_top_m,
        },
        "checkpointing_factor": 4,
    }
    params_path = os.path.join(exp_name, "params.json")
    with open(params_path, "w") as f:
        json.dump(params_dict, f, indent=2)
    print(f"Parameters saved to {params_path}")

    ##################################################################
    # Parameters
    ##################################################################
    par = {
        "nx": 200, "dx": 0.05, "ox": 0.0,
        "ny": 140, "dy": 0.05, "oy": 0.0,
        "nz": model_nz, "dz": dz_km, "oz": 0.0,
        "nsx": 10, "dsx": 1.0, "osx": 0.5,
        "nsy": 7,  "dsy": 1.0, "osy": 0.5,
        "sz": 0.0,
        "nrx": 200, "drx": 0.05, "orx": 0.0,
        "nry": 140, "dry": 0.05, "ory": 0.0,
        "rz": 0.0,
        "t0": 0.0, "tn": 5.0,
        "freq": 5.0,
    }

    shape = (par["nx"], par["ny"], par["nz"])
    spacing = (par["dx"], par["dy"], par["dz"])
    origin = (par["ox"], par["oy"], par["oz"])
    space_order = 4
    nbl = 20

    print(f"Model shape (nx, ny, nz): {shape}")
    print(
        f"Physical size: {par['nx']*par['dx']:.1f} km x "
        f"{par['ny']*par['dy']:.1f} km x {par['nz']*par['dz']:.1f} km"
    )
    print(
        f"Water column: requested={args.water_depth_m:.1f} m, "
        f"actual={actual_water_depth_km*1000.0:.1f} m "
        f"({nwater} z cells), vp={args.water_velocity:.2f} km/s"
    )

    ##################################################################
    # Acquisition geometry
    ##################################################################
    sx_vals = np.arange(par["nsx"]) * par["dsx"] + par["osx"]
    sy_vals = np.arange(par["nsy"]) * par["dsy"] + par["osy"]
    SX, SY = np.meshgrid(sx_vals, sy_vals)
    src_x = SX.ravel().astype(np.float32)
    src_y = SY.ravel().astype(np.float32)
    src_z = np.full(src_x.shape, par["sz"], dtype=np.float32)

    rx_vals = np.arange(par["nrx"]) * par["drx"] + par["orx"]
    ry_vals = np.arange(par["nry"]) * par["dry"] + par["ory"]
    RX, RY = np.meshgrid(rx_vals, ry_vals)
    rec_x = RX.ravel().astype(np.float32)
    rec_y = RY.ravel().astype(np.float32)
    # rec_z = np.zeros(rec_x.shape, dtype=np.float32)
    rec_z = np.full(rec_x.shape, par["rz"], dtype=np.float32)

    nsrc = len(src_x)
    nrec = len(rec_x)
    print(f"Sources : {nsrc}  (expected 70)")
    print(f"Receivers: {nrec}  (expected 28,000)")

    ##################################################################
    # Velocity model
    ##################################################################
    velocity_file = os.path.join(REPO, "data/overthrust.vites")   # see data/README.md
    vp_raw = np.fromfile(velocity_file, dtype=">f4").astype(np.float32) / 1000.0
    vp_raw = vp_raw.reshape(187, 801, 801)

    vp_slice = vp_raw[:180:2, :280:2, :400:2]
    vp_overthrust = vp_slice.transpose(2, 1, 0).copy()
    del vp_raw, vp_slice

    vp_true = prepend_water_column(vp_overthrust, nwater, args.water_velocity)
    del vp_overthrust
    water_mask_1d = np.zeros(par["nz"], dtype=bool)
    if nwater > 0:
        water_mask_1d[:nwater] = True
    water_mask_3d = np.broadcast_to(water_mask_1d[None, None, :], shape)
    n_water_cells = int(water_mask_3d.sum())

    m_vmin, m_vmax = float(vp_true.min()), float(vp_true.max())
    print(f"True model: shape={vp_true.shape}, vp=[{m_vmin:.2f}, {m_vmax:.2f}] km/s")

    x_coord = np.arange(par["nx"]) * par["dx"] + par["ox"]
    y_coord = np.arange(par["ny"]) * par["dy"] + par["oy"]
    z_coord = np.arange(par["nz"]) * par["dz"] + par["oz"]

    plot_slices(
        vp_true, x_coord, y_coord, z_coord,
        vmin=m_vmin, vmax=m_vmax,
        title="True Vp",
        savepath=f"{exp_name}/overthrust_true.png",
    )

    if args.init_model is not None:
        print(f"Loading initial model from: {args.init_model}")
        vp_init = np.load(args.init_model).astype(np.float32)
        if vp_init.shape == (par["nx"], par["ny"], base_nz) and nwater > 0:
            print(
                f"Initial model has base Overthrust shape {vp_init.shape}; "
                f"prepending {nwater} water cells."
            )
            vp_init = prepend_water_column(vp_init, nwater, args.water_velocity)
        elif vp_init.shape != shape:
            raise ValueError(
                f"--init_model shape {vp_init.shape} does not match expected shape {shape}"
            )
        if nwater > 0:
            vp_init = vp_init.copy()
            vp_init[water_mask_3d] = np.float32(args.water_velocity)
        print(f"Initial model: shape={vp_init.shape}, vp=[{vp_init.min():.2f}, {vp_init.max():.2f}] km/s")
    elif args.linear_init:
        vp_init_profile = np.linspace(
            m_vmin,
            m_vmax,
            base_nz,
            dtype=np.float32,
        )
        vp_init_subsurface = np.broadcast_to(
            vp_init_profile[None, None, :],
            (par["nx"], par["ny"], base_nz),
        )
        vp_init = prepend_water_column(
            vp_init_subsurface,
            nwater,
            args.water_velocity,
        )
        del vp_init_profile, vp_init_subsurface
        print(
            "Initial model: linear depth gradient in the subsurface from "
            f"{m_vmin:.2f} to {m_vmax:.2f} km/s, followed by the water layer"
        )
    else:
        sigma_smooth = tuple(args.sigma)
        if nwater > 0:
            vp_init_subsurface = gaussian_filter(vp_true[:, :, nwater:], sigma=sigma_smooth).astype(np.float32)
            vp_init = prepend_water_column(vp_init_subsurface, nwater, args.water_velocity)
            del vp_init_subsurface
        else:
            vp_init = gaussian_filter(vp_true, sigma=sigma_smooth).astype(np.float32)
        print(f"Initial model: smoothed true model with sigmas (x, y, z)={sigma_smooth}")

    if nwater > 0:
        vp_init[water_mask_3d] = np.float32(args.water_velocity)
        print(
            f"Initial model water retention: fixed {n_water_cells} cells "
            f"({nwater}/{par['nz']} z cells) at {args.water_velocity:.2f} km/s"
        )

    init_true_top_km = float(args.init_true_top_m) / 1000.0
    if init_true_top_km > 0.0:
        true_top_1d = z_coord <= init_true_top_km
        if not np.any(true_top_1d):
            print(
                f"Initial true-top mask: requested {args.init_true_top_m:.2f} m, "
                f"but no z cells are shallower than or equal to {init_true_top_km:.4f} km; disabled."
            )
        else:
            true_top_3d = np.broadcast_to(
                true_top_1d[np.newaxis, np.newaxis, :],
                shape,
            )
            vp_init = vp_init.copy()
            vp_init[true_top_3d] = vp_true[true_top_3d]
            n_locked = int(true_top_3d.sum())
            print(
                f"Initial true-top copy: copied top {args.init_true_top_m:.2f} m "
                f"({int(true_top_1d.sum())}/{par['nz']} z cells, {n_locked} total cells)"
            )
    plot_slices(
        vp_init, x_coord, y_coord, z_coord,
        vmin=m_vmin, vmax=m_vmax,
        title="Initial Vp",
        savepath=f"{exp_name}/overthrust_init.png",
    )

    ##################################################################
    # Reference modelling engine: get dt/nt/wavelet
    ##################################################################
    print("Setting up modelling engine...")
    amod_ref = AcousticWave3D(
        shape, origin, spacing,
        src_x[:1], src_y[:1], src_z[:1],
        rec_x, rec_y, rec_z,
        par["t0"], par["tn"],
        vp=vp_true,
        src_type="Ricker", f0=par["freq"],
        space_order=space_order, nbl=nbl,
        clearcache=False,
    )
    model_ref, geometry_ref = amod_ref.model_and_geometry()
    dt_obs = float(geometry_ref.dt)
    nt = int(geometry_ref.nt)
    wav = geometry_ref.src.wavelet
    t_axis = np.arange(nt) * dt_obs
    print(f"dt = {dt_obs*1000:.4f} ms, nt = {nt}")

    ##################################################################
    # Generate/load observed data, per-source cache
    ##################################################################
    dobs_dir = f"{exp_name}/dobs_parts"
    dobs_legacy_path = f"{exp_name}/dobs.npy"
    tokenpath = os.path.join(dobs_dir, "OBS_DATA_COMPLETE.json")

    clear_dobs_cache = (
        args.clear_dobs_cache
        or os.environ.get("CLEAR_DOBS_CACHE", "0").lower() in ("1", "true", "yes", "y")
    )

    if clear_dobs_cache:
        if os.path.isdir(dobs_dir):
            print(f"CLEAR_DOBS_CACHE: removing {dobs_dir}")
            shutil.rmtree(dobs_dir)
        if os.path.exists(dobs_legacy_path):
            print(f"CLEAR_DOBS_CACHE: removing legacy {dobs_legacy_path}")
            os.remove(dobs_legacy_path)

    os.makedirs(dobs_dir, exist_ok=True)

    def source_file(source_idx):
        return os.path.join(dobs_dir, f"source_{int(source_idx):04d}.npy")

    if not os.path.exists(tokenpath):
        print("Modelling observed data for all shots, saved per source...")
        amod_gen = AcousticWave3D(
            shape, origin, spacing,
            src_x, src_y, src_z,
            rec_x, rec_y, rec_z,
            par["t0"], par["tn"],
            vp=vp_true,
            src_type="Ricker", f0=par["freq"],
            space_order=space_order, nbl=nbl,
            clearcache=True,
        )
        dobs_gen, _ = amod_gen.mod_allshots()
        dobs_gen = np.ascontiguousarray(dobs_gen, dtype=np.float32)
        print(
            f"dobs_gen shape={dobs_gen.shape}, dtype={dobs_gen.dtype}, "
            f"size={dobs_gen.nbytes/1024**3:.3f} GB",
            flush=True,
        )

        assert dobs_gen.shape[0] == nsrc
        for source_idx in range(nsrc):
            outfile = source_file(source_idx)
            np.save(outfile, dobs_gen[source_idx])
            print(f"saved source {source_idx} -> {outfile}", flush=True)

        token_dict = {
            "n_sources": int(nsrc),
            "n_receivers": int(nrec),
            "nt": int(nt),
            "model_shape": list(shape),
            "water_nz": int(nwater),
            "water_depth_m": float(actual_water_depth_km * 1000.0),
            "water_velocity": float(args.water_velocity),
            "source_depth_km": 0.0,
            "receiver_depth_km": 0.0,
            "dtype": "float32",
            "layout": "one file per source: source_XXXX.npy with shape (nt, nrec)",
            "script": os.path.abspath(__file__),
        }
        with open(tokenpath, "w") as f:
            json.dump(token_dict, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        print(f"Observed-data cache complete: {tokenpath}")
        del dobs_gen

    with open(tokenpath) as f:
        token_dict = json.load(f)
    expected_cache = {
        "n_sources": int(nsrc),
        "n_receivers": int(nrec),
        "nt": int(nt),
        "model_shape": list(shape),
        "water_nz": int(nwater),
        "water_velocity": float(args.water_velocity),
        "source_depth_km": 0.0,
        "receiver_depth_km": 0.0,
    }
    mismatches = [
        key for key, expected in expected_cache.items()
        if token_dict.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            "Observed-data cache does not match this water-column setup. "
            f"Mismatched fields: {mismatches}. Use --clear_dobs_cache or a new --exp_name."
        )

    # Verify all source files exist before proceeding
    missing = [source_file(i) for i in range(nsrc) if not os.path.exists(source_file(i))]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} observed-data file(s):\n" + "\n".join(missing[:10])
        )
    print(f"Observed-data cache verified: {nsrc} source files in {dobs_dir}")
    log_memory("after verifying dobs cache")

    ##################################################################
    # Refraction masking setup (no data loaded into RAM)
    ##################################################################
    vwater = float(args.water_velocity)
    distance = np.sqrt(
        (src_x[:, None] - rec_x[None, :]) ** 2 +
        (src_y[:, None] - rec_y[None, :]) ** 2
    )
    directtime = (distance / vwater).astype(np.float32)
    del distance

    tsmask3d = TimeSpaceMasking3D(directtime, nt, dt_obs, args.toff, nsmooth=50)
    print(
        f"Refraction mask: toff={args.toff} s, nsmooth=50 samples, "
        f"direct velocity={vwater:.2f} km/s"
    )
    log_memory("after building tsmask3d")

    # ------------------------------------------------------------------
    # Time-domain gain folded into the refraction mask.
    #
    # The gain vector ``g(t) = (t + t0)^k`` is multiplied into the mask along
    # the time axis, so the per-shot operator becomes
    #     Diagonal( (mask(t, r) * g(t))_flattened )
    # which Op then applies identically to obs (via _get_observed_embedding,
    # ``Op[i] @ b[i]``) and to mod (via TorchOperator(Op[i]).apply(x_flat)
    # inside SeisLMFeatureLoss.__call__). This avoids the line-search issues
    # we saw when the gain was applied between mask and trace standardization
    # inside the loss class — here the gain is just part of the linear
    # operator chain, with a clean autograd path.
    #
    #   k = 0.0 : no gain (default)
    #   k = 0.5 : 2D-style correction (line-source spreading, cancels 1/sqrt(t))
    #   k = 1.0 : 3D-style correction (point-source spreading, cancels 1/t)
    # ------------------------------------------------------------------
    if args.gain_power != 0.0:
        t_axis = np.arange(nt, dtype=np.float32) * np.float32(dt_obs) + np.float32(args.gain_t0)
        # Negative t can only occur when t0 < 0; clamp so we never take a
        # fractional power of a negative number.
        t_axis = np.maximum(t_axis, 0.0)
        gain_vec = np.power(t_axis, np.float32(args.gain_power)).astype(np.float32)
        # Avoid the t=0 sample being scaled to exactly zero when t0=0; copy
        # the next sample's value. The mask zeros pre-arrival samples anyway,
        # so this never affects post-mask content for refraction windows.
        if args.gain_t0 == 0.0 and nt > 1:
            gain_vec[0] = gain_vec[1]
        gmin, gmax = float(gain_vec.min()), float(gain_vec.max())
        print(
            f"Time gain: t^{args.gain_power} (t0={args.gain_t0:.3f} s) — "
            f"gain range [{gmin:.3f}, {gmax:.3f}] over [0, {(nt-1)*dt_obs:.2f}] s"
        )
    else:
        gain_vec = None
        print("Time gain: disabled (--gain_power 0.0)")

    def _build_op_for_shot(i):
        """Construct the per-shot Diagonal operator: refraction mask × time gain."""
        m = tsmask3d.mask_oneshot(i)                  # (nt, nrec) float32
        if gain_vec is not None:
            m = m * gain_vec[:, None]                 # broadcast gain over receivers
        return Diagonal(m.ravel())

    # Lazy operator list: builds each Diagonal only when indexed (saves ~25 GB).
    # When --no_mask is set, Op is None and no refraction mute (or gain) is applied.
    if args.no_mask:
        TSmaskop = None
        print("Refraction mask: DISABLED (--no_mask)")
    else:
        TSmaskop = LazyMaskOpList(
            mask_fn=_build_op_for_shot,
            n=nsrc,
        )
        print("Refraction mask: enabled")

    # Lazy data: loads one shot from disk per access (saves ~27 GB), masked with the same operator that both
    # losses apply to the modeled data.
    b_lazy = PerShotLazyData(source_file_fn=source_file, n=nsrc, nt=nt, nrec=nrec, mask_ops=TSmaskop)

    ##################################################################
    # Optional shot-record plots (load transiently, do not persist)
    ##################################################################
    ishots_plot = [0, nsrc // 2, nsrc - 1]
    iry = par["nry"] // 2

    # Load a few shots transiently just for plotting — freed immediately after
    _plot_shots = [np.load(source_file(i)).astype(np.float32) for i in ishots_plot]
    d_vmin, d_vmax = np.percentile(
        np.concatenate([s.ravel() for s in _plot_shots]), [2, 98]
    )

    fig, axs = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
    for ax, isrc_plot, shot in zip(axs, ishots_plot, _plot_shots):
        rec_slice = shot.reshape(nt, par["nry"], par["nrx"])[:, iry, :]
        ax.imshow(
            rec_slice,
            aspect="auto",
            cmap="gray",
            extent=[rx_vals[0], rx_vals[-1], t_axis[-1], t_axis[0]],
            vmin=-d_vmax,
            vmax=d_vmax,
        )
        ax.set_title(f"Shot {isrc_plot}: x={src_x[isrc_plot]:.1f}, y={src_y[isrc_plot]:.1f} km")
        ax.set_xlabel("Receiver x [km]")
    axs[0].set_ylabel("Time [s]")
    plt.tight_layout()
    plt.savefig(f"{exp_name}/shot_records.png", dpi=300, bbox_inches="tight")
    plt.close()
    del _plot_shots

    ##################################################################
    # Verify refraction mask applied to data
    ##################################################################
    if not args.no_mask:
        fig, axs = plt.subplots(2, 3, figsize=(16, 10), sharey=True)
        for col, isrc_plot in enumerate(ishots_plot):
            shot_raw    = np.load(source_file(isrc_plot)).astype(np.float32)
            shot_masked = shot_raw * tsmask3d.mask_oneshot(isrc_plot)

            raw_slice    = shot_raw.reshape(nt, par["nry"], par["nrx"])[:, iry, :]
            masked_slice = shot_masked.reshape(nt, par["nry"], par["nrx"])[:, iry, :]

            clip   = np.percentile(np.abs(raw_slice), 98)
            extent = [rx_vals[0], rx_vals[-1], t_axis[-1], t_axis[0]]

            axs[0, col].imshow(raw_slice,    aspect="auto", cmap="gray",
                               extent=extent, vmin=-clip, vmax=clip)
            axs[0, col].set_title(f"Shot {isrc_plot} — raw")
            axs[0, col].set_xlabel("Receiver x [km]")

            axs[1, col].imshow(masked_slice, aspect="auto", cmap="gray",
                               extent=extent, vmin=-clip, vmax=clip)
            axs[1, col].set_title(f"Shot {isrc_plot} — masked")
            axs[1, col].set_xlabel("Receiver x [km]")

            del shot_raw, shot_masked

        axs[0, 0].set_ylabel("Time [s]")
        axs[1, 0].set_ylabel("Time [s]")
        plt.suptitle("Refraction mask: raw (top) vs masked (bottom)", y=1.01)
        plt.tight_layout()
        plt.savefig(f"{exp_name}/mask_verification.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Mask verification plot saved to {exp_name}/mask_verification.png")

    ##################################################################
    # SeisLM model and feature loss (only when --loss feature)
    ##################################################################
    feat_loss = None
    if args.loss == "feature":

        if args.cpu:
            device = "cpu"
            devices = None
        else:
            n_gpus = torch.cuda.device_count()
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"CUDA device count: {n_gpus}")
            for i in range(n_gpus):
                print(f"  cuda:{i}: {torch.cuda.get_device_name(i)}")

            if n_gpus > 0:
                torch.cuda.set_device(0)
                device = "cuda:0"
                if args.single_gpu or n_gpus == 1:
                    devices = None
                    print("Using SeisLMFeatureLoss in single-GPU mode on cuda:0")
                else:
                    devices = [f"cuda:{i}" for i in range(n_gpus)]
                    print(f"Using SeisLMFeatureLoss multi-GPU devices: {devices}")
            else:
                device = "cpu"
                devices = None
                print("No CUDA devices visible; using CPU")

        print(f"Loading SeisLM checkpoint: {args.ckpt_path}")
        seislm_model = load_pretrained_multidim_wav2vec2(args.ckpt_path, map_location=device).eval().to(device)
        log_memory("after loading SeisLM")

        cache_obs_embeddings = not args.no_cache_obs_embeddings
        cache_obs_on_cpu = cache_obs_embeddings and (not args.cache_obs_on_gpu)
        print(
            "SeisLMFeatureLoss settings: "
            f"device={device}, devices={devices}, trace_chunk={args.trace_chunk}, "
            f"cache_obs_embeddings={cache_obs_embeddings}, cache_obs_on_cpu={cache_obs_on_cpu}, "
            f"gain_power={args.gain_power}, gain_t0={args.gain_t0}"
        )

        loss_kwargs = dict(
            seislm_model=seislm_model,
            Op=TSmaskop,
            b=b_lazy,
            nr=nrec,
            nt=nt,
            t_sub=1,
            device=device,
            pool=args.pool,
            normalize_traces=True,
            trace_chunk=args.trace_chunk,
            cache_obs_embeddings=cache_obs_embeddings,
            cache_obs_on_cpu=cache_obs_on_cpu,
            dt=dt_obs,
            resample=False,
            n_input_channels=3,
        )

        if devices is not None:
            loss_kwargs["devices"] = devices

        try:
            feat_loss = SeisLMFeatureLoss(**loss_kwargs)
        except TypeError as exc:
            if "devices" in loss_kwargs:
                print(
                    "WARNING: SeisLMFeatureLoss rejected the 'devices' argument. "
                    "Falling back to single-GPU mode. Original error:",
                    exc,
                    flush=True,
                )
                loss_kwargs.pop("devices", None)
                feat_loss = SeisLMFeatureLoss(**loss_kwargs)
            else:
                raise

        log_memory("after feature loss setup")

    ##################################################################
    # FWI setup
    ##################################################################
    # Both losses compare Op[i] d_mod with b[i] = Op[i] d_obs (b_lazy, masked when the mask is on).
    l2loss   = L2(Op=TSmaskop, b=b_lazy)
    if args.loss == "feature":
        lossfn = feat_loss
        print("Loss function: SeisLMFeatureLoss")
    else:
        lossfn = l2loss
        print("Loss function: L2")

    ainv = AcousticWave3D(
        shape, origin, spacing,
        src_x, src_y, src_z,
        rec_x, rec_y, rec_z,
        par["t0"], par["tn"],
        vprange=(m_vmin, m_vmax),
        wav=wav, f0=par["freq"],
        space_order=space_order, nbl=nbl,
        loss=lossfn,
        # clearcache=True,
        factor=4,
    )

    ##################################################################
    # Initial gradient — save and plot immediately
    ##################################################################
    print("Computing initial gradient...")
    log_memory("before initial gradient")

    grad_sigmas = tuple(args.grad_sigma)
    print(f"Gradient smoothing sigmas (x, y, z): {grad_sigmas}")
    if args.grad_mask_type == "shallow_divide":
        msk_1d = build_depth_shallow_divide(
            par["nz"],
            par["dz"],
            shallow_to_km=args.grad_shallow_divide_to_km,
            k=args.grad_shallow_divide_k,
        )
        print(
            "Gradient depth mask: shallow_divide "
            f"(z<={args.grad_shallow_divide_to_km} km divided by {args.grad_shallow_divide_k})"
        )
    elif args.grad_mask_type == "soft":
        msk_1d = build_depth_taper_soft(
            par["nz"],
            par["dz"],
            taper_to_km=args.grad_soft_taper_to_km,
            power=args.grad_soft_power,
        )
        print(
            "Gradient depth mask: soft "
            f"(taper_to={args.grad_soft_taper_to_km} km, power={args.grad_soft_power})"
        )
    else:
        raise ValueError(f"Unsupported --grad_mask_type: {args.grad_mask_type}")

    if nwater > 0:
        msk_1d[:nwater] = 0.0
        print(
            f"Gradient water mask: zeroed top {actual_water_depth_km*1000.0:.1f} m "
            f"({nwater} z cells)"
        )

    depth_mask = np.broadcast_to(msk_1d[None, None, :], shape).copy()
    print(
        "Gradient depth mask stats: "
        f"min={float(depth_mask.min()):.4g}, max={float(depth_mask.max()):.4g}, "
        f"cells_not_full={int(np.count_nonzero(depth_mask != 1.0))}/{depth_mask.size}"
    )
    np.save(f"{exp_name}/gradient_depth_mask.npy", depth_mask.astype(np.float32, copy=False))
    np.save(f"{exp_name}/water_mask.npy", water_mask_3d.astype(bool, copy=False))
    postproc = PostProcessVP(scaling=1, mask=depth_mask, sigmas=grad_sigmas)
    loss0, grad0 = ainv._loss_grad(vp_init, postprocess=postproc.apply)

    # Use |grad0|.max() so the normalisation is anchored to the true peak
    # magnitude regardless of sign.
    scaling = float(np.abs(grad0).max())

    print(f"Initial loss: {loss0:.4e}, gradient max: {scaling:.4e}")

    # Save and plot vp_init and grad0 right away so they are on disk
    # even if the inversion is interrupted later.
    np.save(f"{exp_name}/vp_init.npy", vp_init)
    np.save(f"{exp_name}/vp_true.npy", vp_true)
    np.save(f"{exp_name}/grad0.npy",   grad0)

    plot_slices(vp_true, x_coord, y_coord, z_coord, vmin=m_vmin, vmax=m_vmax,
                title="True Vp", savepath=f"{exp_name}/result_true.png")
    plot_slices(vp_init, x_coord, y_coord, z_coord, vmin=m_vmin, vmax=m_vmax,
                title="Initial Vp", savepath=f"{exp_name}/result_init.png")
    plot_slices(
        grad0 / scaling,
        x_coord, y_coord, z_coord,
        vmin=-1e-1, vmax=1e-1, cmap="seismic",
        title="Initial FWI gradient (normalised)",
        cbar_label="Normalised gradient",
        savepath=f"{exp_name}/gradient.png",
    )
    print(f"Saved: vp_init.npy, vp_true.npy, grad0.npy, result_init.png, result_true.png, gradient.png")
    log_memory("after initial gradient")

    ##################################################################
    # Inversion with L-BFGS-B
    ##################################################################
    misfit_history    = []   # data misfit at iterate
    vp_error          = []   # global relative error  ||(x-v)/v||  over the full model
    vp_error_local    = []   # windowed relative error, edges/source-region/bottom excluded

    # Build the boolean mask for the windowed model-error metric. The window
    # excludes (a) the source-region near-surface, (b) the poorly-illuminated
    # bottom, and (c) lateral edges where source coverage is one-sided. We
    # build this once before the inversion starts and reuse it on every iter.
    nx_, ny_, nz_ = shape
    dx_, dy_, dz_ = par["dx"], par["dy"], par["dz"]
    mx = max(0, int(round(args.metric_margin_x      / dx_)))
    my = max(0, int(round(args.metric_margin_y      / dy_)))
    mt = max(nwater, int(round(args.metric_margin_top / dz_)))
    mb = max(0, int(round(args.metric_margin_bottom / dz_)))
    metric_mask_3d = np.zeros(shape, dtype=bool)
    if (nx_ - 2 * mx) > 0 and (ny_ - 2 * my) > 0 and (nz_ - mt - mb) > 0:
        metric_mask_3d[mx:nx_ - mx, my:ny_ - my, mt:nz_ - mb] = True
    n_kept = int(metric_mask_3d.sum())
    n_total = int(metric_mask_3d.size)
    print(
        f"Model-error metric window: x∈[{mx*dx_:.2f},{(nx_-mx)*dx_:.2f}] km, "
        f"y∈[{my*dy_:.2f},{(ny_-my)*dy_:.2f}] km, "
        f"z∈[{mt*dz_:.2f},{(nz_-mb)*dz_:.2f}] km  "
        f"({n_kept}/{n_total} cells = {100.0*n_kept/n_total:.1f}%)"
    )
    # Save the mask so plots/analysis later can reuse the same window.
    np.save(os.path.join(exp_name, "metric_mask.npy"), metric_mask_3d)

    # Pre-flatten for the callback (fast indexing on every iter).
    _mask_flat   = metric_mask_3d.ravel()
    _vp_true_flat = vp_true.ravel()

    def _windowed_relerr(xk_flat):
        """Relative L2 error restricted to the metric window:
            sqrt(sum((xk - v)^2 / v^2)) over kept cells, then divided by
            sqrt(N_kept) so the value is comparable to a per-cell RMS rather
            than scaling with how many cells survived the window. The
            non-windowed `vp_error` (for backward compatibility) keeps the
            original norm form."""
        diff_rel = (xk_flat[_mask_flat] - _vp_true_flat[_mask_flat]) / _vp_true_flat[_mask_flat]
        return float(np.linalg.norm(diff_rel) / np.sqrt(diff_rel.size))

    # Seed both error histories with the initial model before L-BFGS-B applies
    # its first update. Callback entries then correspond to iterations 1, 2, ...
    _vp_init_flat = vp_init.ravel()
    vp_error.append(np.linalg.norm((_vp_init_flat - _vp_true_flat) / _vp_true_flat))
    vp_error_local.append(_windowed_relerr(_vp_init_flat))
    print(
        f"Initial model error: local = {vp_error_local[0]:.4e}, "
        f"global = {vp_error[0]:.4e}",
        flush=True,
    )

    # Paths for incremental checkpoints. Overwritten on every iteration so the
    # files never grow — at any moment they hold the latest iterate and the
    # full history up to that iterate. Use atomic rename so an interrupt
    # mid-save can never leave a truncated file.
    _ckpt_vp        = os.path.join(exp_name, "vp_current.npy")
    _ckpt_vp_plot   = os.path.join(exp_name, "vp_current.png")
    _ckpt_loss      = os.path.join(exp_name, "loss_history.npy")
    _ckpt_err       = os.path.join(exp_name, "vp_error_history.npy")
    _ckpt_err_local = os.path.join(exp_name, "vp_error_local_history.npy")
    _ckpt_meta      = os.path.join(exp_name, "checkpoint_status.json")

    def _atomic_save(path, arr):
        """np.save then os.replace — guarantees the file at `path` is either
        the previous version or fully the new one, never half-written.
        Uses an open file handle to avoid np.save's auto-`.npy` extension
        rewriting the destination filename."""
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, arr, allow_pickle=False)
        os.replace(tmp, path)

    def _atomic_save_json(path, obj):
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=2)
        os.replace(tmp, path)

    def fwi_callback(xk):
        misfit_history.append(float(ainv.losshistory[-1]))   # objective over all shots at the accepted iterate
        vp_error.append(np.linalg.norm((xk - _vp_true_flat) / _vp_true_flat))
        vp_error_local.append(_windowed_relerr(xk))
        it = len(misfit_history)

        print(
            f"  iter {it:3d}  "
            f"loss = {misfit_history[-1]:.4e}  "
            f"err_local = {vp_error_local[-1]:.4e}  "
            f"err_global = {vp_error[-1]:.4e}",
            flush=True,
        )

        # Persist the current iterate and the histories. On a 200x140x90 grid
        # this is ~10 MB per save — negligible relative to one FWI iteration.
        try:
            current_vp = xk.reshape(shape).astype(np.float32)
            _atomic_save(_ckpt_vp,        current_vp)
            _atomic_save(_ckpt_loss,      np.asarray(misfit_history,  dtype=np.float64))
            _atomic_save(_ckpt_err,       np.asarray(vp_error,        dtype=np.float64))
            _atomic_save(_ckpt_err_local, np.asarray(vp_error_local,  dtype=np.float64))
            tmp_plot = _ckpt_vp_plot + ".tmp.png"
            plot_slices(
                current_vp,
                x_coord, y_coord, z_coord,
                vmin=m_vmin, vmax=m_vmax,
                title=f"Current Vp (iteration {it})",
                savepath=tmp_plot,
            )
            os.replace(tmp_plot, _ckpt_vp_plot)
            _atomic_save_json(_ckpt_meta, {
                "iteration":  it,
                "loss":       float(misfit_history[-1]),
                "err_local":  float(vp_error_local[-1]),
                "err_global": float(vp_error[-1]),
                "current_model": _ckpt_vp,
                "current_model_plot": _ckpt_vp_plot,
                "timestamp":  datetime.datetime.now().isoformat(timespec="seconds"),
                "exp_name":   exp_name,
                "loss_fn":    args.loss,
            })
        except Exception as exc:
            # Never let a checkpoint write failure break the inversion.
            print(f"  [checkpoint] write failed at iter {it}: {exc}", flush=True)

    postproc_inv = PostProcessVP(scaling=scaling, mask=depth_mask, sigmas=grad_sigmas).apply
    water_mask_flat = water_mask_3d.ravel()
    bounds = [
        (float(args.water_velocity), float(args.water_velocity)) if is_water else (m_vmin, m_vmax)
        for is_water in water_mask_flat
    ]
    if nwater > 0:
        print(
            f"L-BFGS-B bounds: pinned {int(np.count_nonzero(water_mask_flat))} "
            f"water cells at {args.water_velocity:.2f} km/s"
        )

    print(f"Starting FWI (L-BFGS-B, maxiter={maxiter})...")
    nl = minimize(
        ainv.loss_grad,
        vp_init.ravel(),
        method="L-BFGS-B",
        jac=True,
        args=(None, postproc_inv),
        callback=fwi_callback,
        bounds=bounds,
        options={
            "ftol": 1e-30,
            "gtol": 1e-30,
            "maxiter": maxiter,
            "maxfun": 5000,
            # "maxcor": 20,
            "disp": True,
        },
    )

    print(f"Optimizer exit: {nl.message} (status={nl.status})", flush=True)

    ##################################################################
    # Save results
    ##################################################################
    vp_inv = nl.x.reshape(shape)
    if nwater > 0:
        vp_inv[water_mask_3d] = np.float32(args.water_velocity)

    # --- Arrays ---
    np.save(f"{exp_name}/vp_inv.npy",         vp_inv)
    np.save(f"{exp_name}/loss.npy",           np.array(misfit_history))
    np.save(f"{exp_name}/vp_error.npy",       np.array(vp_error))
    np.save(f"{exp_name}/vp_error_local.npy", np.array(vp_error_local))

    # --- Optimizer summary JSON ---
    results_dict = {
        "optimizer_message": nl.message,
        "optimizer_status": int(nl.status),
        "n_iterations": len(misfit_history),
        "n_func_evals": int(nl.nfev),
        "final_loss": float(misfit_history[-1]) if misfit_history else None,
        "initial_loss": float(loss0),
        "loss_reduction": float(loss0 / misfit_history[-1]) if misfit_history else None,
        "final_vp_error":         float(vp_error[-1])       if vp_error       else None,
        "initial_vp_error":       float(vp_error[0])        if vp_error       else None,
        "final_vp_error_local":   float(vp_error_local[-1]) if vp_error_local else None,
        "initial_vp_error_local": float(vp_error_local[0])  if vp_error_local else None,
        "metric_window_cells": n_kept,
        "metric_window_total": n_total,
        "metric_margin_x_km":      args.metric_margin_x,
        "metric_margin_y_km":      args.metric_margin_y,
        "metric_margin_top_km":    args.metric_margin_top,
        "metric_margin_bottom_km": args.metric_margin_bottom,
        "grad0_max": float(scaling),
        "vp_inv_min": float(vp_inv.min()),
        "vp_inv_max": float(vp_inv.max()),
    }
    with open(f"{exp_name}/results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results summary saved to {exp_name}/results.json")

    # --- Convergence plots: loss, local error, global error ---
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    axs[0].plot(misfit_history, "o-", markersize=3)
    axs[0].set_title(f"{args.loss.upper()} misfit")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Loss")
    axs[0].grid(True, alpha=0.3)
    axs[1].plot(vp_error_local, "o-", markersize=3, color="tab:orange",
                label="local")
    axs[1].set_title("Local model error")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Windowed relative RMS error")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    axs[2].plot(vp_error, "s--", markersize=3, color="tab:gray", alpha=0.8,
                label="global")
    axs[2].set_title("Global model error")
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel(r"$\|\mathbf{v} - \mathbf{v}_{true}\|/\|\mathbf{v}_{true}\|$")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{exp_name}/convergence.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Inverted model plots ---
    plot_slices(vp_inv, x_coord, y_coord, z_coord, vmin=m_vmin, vmax=m_vmax,
                title="Inverted Vp", savepath=f"{exp_name}/result_inv.png")

    diff = vp_inv - vp_true
    dlim = float(np.abs(diff).max())
    plot_slices(
        diff, x_coord, y_coord, z_coord,
        vmin=-dlim, vmax=dlim, cmap="seismic",
        title="Inverted - True",
        cbar_label="dVp [km/s]",
        savepath=f"{exp_name}/model_difference.png",
    )

    print(f"\n{'='*60}")
    print(f"Experiment : {exp_name}")
    print(f"Loss       : {args.loss}")
    print(f"Iterations : {len(misfit_history)}")
    print(f"Initial loss: {loss0:.4e}  →  Final loss: {misfit_history[-1]:.4e}" if misfit_history else "")
    if vp_error_local:
        print(f"Model err (local) : {vp_error_local[0]:.4e}  →  {vp_error_local[-1]:.4e}")
    if vp_error:
        print(f"Model err (global): {vp_error[0]:.4e}  →  {vp_error[-1]:.4e}")
    print(f"Saved to   : {os.path.abspath(exp_name)}/")
    print(f"{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
