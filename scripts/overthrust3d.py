#!/usr/bin/env python3
r"""
3D Acoustic FWI on the SEG/EAGE Overthrust Model with a water column,
SeisLM Feature Loss, and optional single-GPU or multi-GPU SeisLM feature
extraction.

Typical run (from the repo root):
  python scripts/Overthrust3D.py --maxiter 10 --loss l2
"""

import os
import json
import shutil
import argparse
import datetime

import numpy as np
import torch
from pylops.basicoperators import Diagonal
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.signal import filtfilt
from devito import configuration

from devitofwi.waveengine.acoustic3d import AcousticWave3D
from devitofwi.loss.l2 import L2
from devitofwi.postproc.acoustic import PostProcessVP

# matplotlib backend must be set before pyplot is imported; fm_fwi.plotting
# imports pyplot, so the use() call is placed right before those imports.
import matplotlib
matplotlib.use("Agg")
from fm_fwi.losses import SeisLMFeatureLoss, HybridLoss  # noqa: E402
from fm_fwi.plotting import plot_slices, plot_curve  # noqa: E402

configuration["log-level"] = "ERROR"

# Path helpers (work regardless of where the script is invoked from)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="3D Overthrust FWI with SeisLM feature loss."
    )
    parser.add_argument("--maxiter",    type=int,   default=10)
    parser.add_argument("--trace_chunk",type=int,   default=768)
    parser.add_argument("--exp_name",   type=str,   default=None)
    parser.add_argument("--clear_dobs_cache", action="store_true")
    parser.add_argument("--no_cache_obs_embeddings", action="store_true")
    parser.add_argument("--cache_obs_on_gpu",          action="store_true")
    parser.add_argument("--cache_obs_embeddings_to_disk", action="store_true")
    parser.add_argument("--obs_embedding_cache_dir",   type=str, default=None)
    parser.add_argument("--obs_embedding_cache_tag",   type=str, default=None)
    parser.add_argument("--single_gpu", action="store_true")
    parser.add_argument("--cpu",        action="store_true")
    parser.add_argument(
        "--ckpt_path", type=str,
        default=os.environ.get(
            "SEISLM_CKPT_PATH",
            os.path.join(_REPO_ROOT, "pretrained_seislm_base", "checkpoints",
                         "epoch=39-step=1203000.ckpt"),
        ),
    )
    parser.add_argument("--toff",           type=float, default=0.5)
    parser.add_argument("--water_depth_m",  type=float, default=500.0)
    parser.add_argument("--water_velocity", type=float, default=1.5)
    parser.add_argument("--loss", type=str, default="feature",
                        choices=["feature", "l2", "hybrid"])
    parser.add_argument("--alpha",       type=float, default=0.5)
    parser.add_argument("--alpha_start", type=float, default=None)
    parser.add_argument("--alpha_end",   type=float, default=None)
    parser.add_argument("--no_mask",     action="store_true")
    parser.add_argument("--sigma",      type=float, nargs=3, default=[10.0, 10.0, 5.0],
                        metavar=("SX", "SY", "SZ"))
    parser.add_argument("--init_model", type=str, default=None)
    parser.add_argument("--linear_init",action="store_true")
    parser.add_argument("--grad_sigma", type=float, nargs=3, default=[4.0, 4.0, 2.0],
                        metavar=("SX", "SY", "SZ"))
    return parser.parse_args()


def log_memory(tag):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                rss_gb = int(line.split()[1]) / 1024**2
                break
    print(f"[memory] {tag}: CPU RSS = {rss_gb:.2f} GB", flush=True)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc    = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"[memory] {tag}: cuda:{i} alloc={alloc:.2f} GB, reserved={reserved:.2f} GB",
                  flush=True)


class TimeSpaceMasking3D:
    def __init__(self, arrival, nt, dt, toff=0.0, nsmooth=None):
        self.arrival = arrival.astype(np.float32, copy=False)
        self.nt = int(nt); self.dt = float(dt); self.toff = float(toff)
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
    def __init__(self, mask_fn, n):
        super().__init__([None] * n)
        self._mask_fn = mask_fn; self._n = n
    def __len__(self):          return self._n
    def __getitem__(self, i):   return self._mask_fn(i)


class PerShotLazyData:
    def __init__(self, source_file_fn, n, nt, nrec):
        self._source_file_fn = source_file_fn
        self._n = n; self._nt = nt; self._nrec = nrec
    def __len__(self):   return self._n
    @property
    def size(self):      return self._n * self._nt * self._nrec
    def __getitem__(self, i):
        return np.load(self._source_file_fn(i)).astype(np.float32).ravel()


def build_depth_taper_soft(nz, dz):
    taper_depth_km = 0.50; taper_exponent = 1.5
    z = np.arange(nz) * dz
    base = np.where(z < taper_depth_km,
                    0.5 * (1 - np.cos(np.pi * z / taper_depth_km)), 1.0)
    return (base ** taper_exponent).astype(np.float32)


def prepend_water_column(vp_sub, nwater, water_velocity):
    if nwater <= 0:
        return vp_sub.astype(np.float32, copy=False)
    water = np.full((vp_sub.shape[0], vp_sub.shape[1], nwater),
                    np.float32(water_velocity), dtype=np.float32)
    return np.concatenate([water, vp_sub.astype(np.float32, copy=False)], axis=2)


def main():
    args = parse_args()
    if args.init_model is not None and args.linear_init:
        raise ValueError("--init_model and --linear_init are mutually exclusive")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    if (args.alpha_start is None) != (args.alpha_end is None):
        raise ValueError("--alpha_start and --alpha_end must be set together")

    base_nz      = 90
    dz_km        = 0.05
    nwater       = int(round(args.water_depth_m / 1000.0 / dz_km))
    actual_water = nwater * dz_km
    model_nz     = base_nz + nwater
    maxiter      = args.maxiter
    exp_name     = (args.exp_name or
                    f"SeisLM_overthrust_3D_water_{int(round(args.water_depth_m))}m"
                    f"_{args.loss}_{maxiter}iter")

    print(f"Experiment: {exp_name}")
    os.makedirs(exp_name, exist_ok=True)

    # Save params
    params = dict(
        experiment=exp_name, timestamp=datetime.datetime.now().isoformat(),
        loss=args.loss, alpha=args.alpha, alpha_start=args.alpha_start,
        alpha_end=args.alpha_end, maxiter=maxiter, trace_chunk=args.trace_chunk,
        toff=args.toff, water_depth_m=args.water_depth_m,
        water_velocity=args.water_velocity, no_mask=args.no_mask,
        sigma=args.sigma, grad_sigma=args.grad_sigma,
        init_model=args.init_model, linear_init=args.linear_init,
        ckpt_path=args.ckpt_path,
    )
    with open(os.path.join(exp_name, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # ── Model parameters ────────────────────────────────────────────────
    par = dict(
        nx=200, dx=0.05, ox=0.0,
        ny=140, dy=0.05, oy=0.0,
        nz=model_nz, dz=dz_km, oz=0.0,
        nsx=10, dsx=1.0, osx=0.5,
        nsy=7,  dsy=1.0, osy=0.5, sz=0.0,
        nrx=200, drx=0.05, orx=0.0,
        nry=140, dry=0.05, ory=0.0, rz=0.0,
        t0=0.0, tn=5.0, freq=5.0,
    )
    shape   = (par["nx"], par["ny"], par["nz"])
    spacing = (par["dx"], par["dy"], par["dz"])
    origin  = (par["ox"], par["oy"], par["oz"])
    print(f"Model shape: {shape}, water cells: {nwater}")

    # ── Acquisition ─────────────────────────────────────────────────────
    SX, SY = np.meshgrid(
        np.arange(par["nsx"])*par["dsx"]+par["osx"],
        np.arange(par["nsy"])*par["dsy"]+par["osy"])
    src_x = SX.ravel().astype(np.float32)
    src_y = SY.ravel().astype(np.float32)
    src_z = np.full(src_x.shape, par["sz"], dtype=np.float32)

    rx_vals = np.arange(par["nrx"])*par["drx"]+par["orx"]
    ry_vals = np.arange(par["nry"])*par["dry"]+par["ory"]
    RX, RY  = np.meshgrid(rx_vals, ry_vals)
    rec_x   = RX.ravel().astype(np.float32)
    rec_y   = RY.ravel().astype(np.float32)
    rec_z   = np.full(rec_x.shape, par["rz"], dtype=np.float32)

    nsrc = len(src_x); nrec = len(rec_x)
    print(f"Sources: {nsrc}, Receivers: {nrec}")

    # ── Velocity model ───────────────────────────────────────────────────
    velocity_file = os.path.join(_REPO_ROOT, "data", "overthrust.vites")
    vp_raw   = np.fromfile(velocity_file, dtype=">f4").astype(np.float32) / 1000.0
    vp_raw   = vp_raw.reshape(187, 801, 801)
    vp_slice = vp_raw[:180:2, :280:2, :400:2]
    vp_sub   = vp_slice.transpose(2, 1, 0).copy()
    del vp_raw, vp_slice

    vp_true = prepend_water_column(vp_sub, nwater, args.water_velocity)
    del vp_sub
    water_mask_1d  = np.zeros(par["nz"], dtype=bool)
    if nwater > 0:
        water_mask_1d[:nwater] = True
    water_mask_3d = np.broadcast_to(water_mask_1d[None, None, :], shape)
    m_vmin, m_vmax = float(vp_true.min()), float(vp_true.max())
    print(f"True model: shape={vp_true.shape}, vp=[{m_vmin:.2f}, {m_vmax:.2f}] km/s")

    x_coord = np.arange(par["nx"])*par["dx"]+par["ox"]
    y_coord = np.arange(par["ny"])*par["dy"]+par["oy"]
    z_coord = np.arange(par["nz"])*par["dz"]+par["oz"]

    plot_slices(vp_true, x_coord, y_coord, z_coord, vmin=m_vmin, vmax=m_vmax,
                title="True Vp", savepath=f"{exp_name}/overthrust_true.png")

    # ── Initial model ────────────────────────────────────────────────────
    if args.init_model is not None:
        vp_init = np.load(args.init_model).astype(np.float32)
        if vp_init.shape != shape:
            raise ValueError(f"--init_model shape {vp_init.shape} != {shape}")
        if nwater > 0:
            vp_init[water_mask_3d] = np.float32(args.water_velocity)
    elif args.linear_init:
        profile   = np.linspace(m_vmin, m_vmax, base_nz, dtype=np.float32)
        vp_init   = prepend_water_column(
            np.broadcast_to(profile[None, None, :], (par["nx"], par["ny"], base_nz)),
            nwater, args.water_velocity)
    else:
        sigma_smooth = tuple(args.sigma)
        if nwater > 0:
            sub = gaussian_filter(vp_true[:, :, nwater:], sigma=sigma_smooth).astype(np.float32)
            vp_init = prepend_water_column(sub, nwater, args.water_velocity)
        else:
            vp_init = gaussian_filter(vp_true, sigma=sigma_smooth).astype(np.float32)
        if nwater > 0:
            vp_init[water_mask_3d] = np.float32(args.water_velocity)

    plot_slices(vp_init, x_coord, y_coord, z_coord, vmin=m_vmin, vmax=m_vmax,
                title="Initial Vp", savepath=f"{exp_name}/overthrust_init.png")

    # ── Reference engine: get dt/nt/wavelet ─────────────────────────────
    print("Setting up modelling engine...")
    amod_ref = AcousticWave3D(
        shape, origin, spacing,
        src_x[:1], src_y[:1], src_z[:1], rec_x, rec_y, rec_z,
        par["t0"], par["tn"], vp=vp_true,
        src_type="Ricker", f0=par["freq"],
        space_order=4, nbl=20, clearcache=False)
    _, geometry_ref = amod_ref.model_and_geometry()
    dt_obs = float(geometry_ref.dt)
    nt     = int(geometry_ref.nt)
    wav    = geometry_ref.src.wavelet
    t_axis = np.arange(nt) * dt_obs
    print(f"dt = {dt_obs*1000:.4f} ms, nt = {nt}")

    # ── Observed data cache ──────────────────────────────────────────────
    dobs_dir    = f"{exp_name}/dobs_parts"
    tokenpath   = os.path.join(dobs_dir, "OBS_DATA_COMPLETE.json")

    if args.clear_dobs_cache:
        if os.path.isdir(dobs_dir):
            shutil.rmtree(dobs_dir)
    os.makedirs(dobs_dir, exist_ok=True)

    def source_file(i):
        return os.path.join(dobs_dir, f"source_{int(i):04d}.npy")

    if not os.path.exists(tokenpath):
        print("Modelling observed data for all shots...")
        amod_gen = AcousticWave3D(
            shape, origin, spacing,
            src_x, src_y, src_z, rec_x, rec_y, rec_z,
            par["t0"], par["tn"], vp=vp_true,
            src_type="Ricker", f0=par["freq"],
            space_order=4, nbl=20, clearcache=True)
        dobs_gen, _ = amod_gen.mod_allshots()
        dobs_gen = np.ascontiguousarray(dobs_gen, dtype=np.float32)
        for i in range(nsrc):
            np.save(source_file(i), dobs_gen[i])
            print(f"saved source {i}", flush=True)
        token = dict(n_sources=nsrc, n_receivers=nrec, nt=nt,
                     model_shape=list(shape), water_nz=nwater,
                     water_velocity=float(args.water_velocity),
                     source_depth_km=0.0, receiver_depth_km=0.0)
        with open(tokenpath, "w") as f:
            json.dump(token, f, indent=2)
        del dobs_gen

    missing = [source_file(i) for i in range(nsrc) if not os.path.exists(source_file(i))]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} source file(s): {missing[:5]}")
    print(f"Observed data cache verified: {nsrc} files in {dobs_dir}")
    log_memory("after dobs")

    # ── Refraction masking ───────────────────────────────────────────────
    vwater   = float(args.water_velocity)
    distance = np.sqrt((src_x[:,None]-rec_x[None,:])**2 + (src_y[:,None]-rec_y[None,:])**2)
    directtime = (distance / vwater).astype(np.float32)
    tsmask3d   = TimeSpaceMasking3D(directtime, nt, dt_obs, args.toff, nsmooth=50)

    def _build_op(i):
        return Diagonal(tsmask3d.mask_oneshot(i).ravel())

    TSmaskop = None if args.no_mask else LazyMaskOpList(_build_op, nsrc)
    b_lazy   = PerShotLazyData(source_file, nsrc, nt, nrec)

    # ── SeisLM feature loss ──────────────────────────────────────────────
    feat_loss = None
    if args.loss in ("feature", "hybrid"):
        from fm_fwi.model.pretrained import load_pretrained_multidim_wav2vec2

        if args.cpu:
            device = "cpu"; devices = None
        else:
            n_gpus = torch.cuda.device_count()
            device  = "cuda:0" if n_gpus > 0 else "cpu"
            devices = ([f"cuda:{i}" for i in range(n_gpus)]
                       if n_gpus > 1 and not args.single_gpu else None)

        print(f"Loading SeisLM checkpoint: {args.ckpt_path}")
        seislm_model = load_pretrained_multidim_wav2vec2(args.ckpt_path, map_location=device)
        seislm_model = seislm_model.to(device).eval()
        log_memory("after loading SeisLM")

        cache_obs_embeddings = not args.no_cache_obs_embeddings
        obs_cache_dir = (args.obs_embedding_cache_dir or
                         os.path.join(exp_name, "obs_embedding_cache"))
        obs_cache_tag = (args.obs_embedding_cache_tag or
                         f"mask={not args.no_mask}_toff={args.toff}_water={actual_water:.3f}km")

        loss_kwargs = dict(
            seislm_model=seislm_model, Op=TSmaskop, b=b_lazy,
            nr=nrec, nt=nt, t_sub=1, device=device,
            pool="mean", normalize_traces=True, trace_chunk=args.trace_chunk,
            cache_obs_embeddings=cache_obs_embeddings,
            cache_obs_on_cpu=(cache_obs_embeddings and not args.cache_obs_on_gpu),
            cache_obs_embeddings_to_disk=bool(args.cache_obs_embeddings_to_disk),
            obs_embedding_cache_dir=(obs_cache_dir if args.cache_obs_embeddings_to_disk else None),
            obs_embedding_cache_tag=obs_cache_tag,
            dt=dt_obs, resample=False, n_input_channels=3,
        )
        if devices is not None:
            loss_kwargs["devices"] = devices
        try:
            feat_loss = SeisLMFeatureLoss(**loss_kwargs)
        except TypeError:
            loss_kwargs.pop("devices", None)
            feat_loss = SeisLMFeatureLoss(**loss_kwargs)

    # ── Loss function ────────────────────────────────────────────────────
    l2loss = L2(Op=TSmaskop, b=b_lazy)
    if args.loss == "hybrid":
        if args.alpha_start is not None:
            a0, a1 = float(args.alpha_start), float(args.alpha_end)
            denom = max(1, maxiter - 1)
            alpha_value = lambda it: float(np.clip(a0 + (a1-a0)*it/denom, 0.0, 1.0))
        else:
            alpha_value = float(args.alpha)
        lossfn = HybridLoss(l2_loss=l2loss, seislm_loss=feat_loss, alpha=alpha_value)
    elif args.loss == "feature":
        lossfn = feat_loss
    else:
        lossfn = l2loss
    print(f"Loss function: {args.loss}")

    # ── Inversion engine ─────────────────────────────────────────────────
    ainv = AcousticWave3D(
        shape, origin, spacing,
        src_x, src_y, src_z, rec_x, rec_y, rec_z,
        par["t0"], par["tn"],
        vprange=(m_vmin, m_vmax), wav=wav, f0=par["freq"],
        space_order=4, nbl=20, loss=lossfn, factor=4)

    # ── Initial gradient ─────────────────────────────────────────────────
    print("Computing initial gradient...")
    msk_1d = build_depth_taper_soft(par["nz"], par["dz"])
    if nwater > 0:
        msk_1d[:nwater] = 0.0
    depth_mask = np.broadcast_to(msk_1d[None, None, :], shape).copy()

    grad_sigmas = tuple(args.grad_sigma)
    postproc    = PostProcessVP(scaling=1, mask=depth_mask, sigmas=grad_sigmas)
    loss0, grad0 = ainv._loss_grad(vp_init, postprocess=postproc.apply)
    scaling      = float(np.abs(grad0).max())
    print(f"Initial loss: {loss0:.4e}, gradient max: {scaling:.4e}")

    np.save(f"{exp_name}/vp_init.npy",  vp_init)
    np.save(f"{exp_name}/vp_true.npy",  vp_true)
    np.save(f"{exp_name}/grad0.npy",    grad0)
    plot_slices(grad0/scaling, x_coord, y_coord, z_coord,
                vmin=-1e-1, vmax=1e-1, cmap="seismic",
                title="Initial gradient (normalised)", cbar_label="Normalised gradient",
                savepath=f"{exp_name}/gradient.png")
    log_memory("after initial gradient")

    # ── L-BFGS-B ────────────────────────────────────────────────────────
    misfit_history = []
    vp_error       = []
    _vp_true_flat  = vp_true.ravel()
    vp_error.append(float(np.linalg.norm(
        (vp_init.ravel()-_vp_true_flat)/_vp_true_flat) / np.sqrt(_vp_true_flat.size)))

    _ckpt_vp   = os.path.join(exp_name, "vp_current.npy")
    _ckpt_loss = os.path.join(exp_name, "loss_history.npy")
    _ckpt_err  = os.path.join(exp_name, "vp_error_history.npy")

    def _atomic_save(path, arr):
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, arr, allow_pickle=False)
        os.replace(tmp, path)

    def fwi_callback(xk):
        misfit_history.append(ainv.losshistory[-1])
        err = float(np.linalg.norm((xk-_vp_true_flat)/_vp_true_flat) / np.sqrt(xk.size))
        vp_error.append(err)
        it = len(misfit_history)
        if hasattr(lossfn, "step"):
            lossfn.step()
        print(f"  iter {it:3d}  loss = {misfit_history[-1]:.4e}  err = {err:.4e}", flush=True)
        try:
            _atomic_save(_ckpt_vp,   xk.reshape(shape).astype(np.float32))
            _atomic_save(_ckpt_loss, np.asarray(misfit_history, dtype=np.float64))
            _atomic_save(_ckpt_err,  np.asarray(vp_error,       dtype=np.float64))
        except Exception as exc:
            print(f"  [checkpoint] write failed at iter {it}: {exc}", flush=True)

    water_flat = water_mask_3d.ravel()
    bounds = [
        (float(args.water_velocity), float(args.water_velocity)) if w else (m_vmin, m_vmax)
        for w in water_flat
    ]
    postproc_inv = PostProcessVP(scaling=scaling, mask=depth_mask, sigmas=grad_sigmas).apply

    print(f"Starting FWI (L-BFGS-B, maxiter={maxiter})...")
    nl = minimize(
        ainv.loss_grad, vp_init.ravel(), method="L-BFGS-B", jac=True,
        args=(None, postproc_inv), callback=fwi_callback, bounds=bounds,
        options=dict(ftol=1e-30, gtol=1e-30, maxiter=maxiter, maxfun=5000, disp=True))
    print(f"Optimizer exit: {nl.message}", flush=True)

    # ── Save results ─────────────────────────────────────────────────────
    vp_inv = nl.x.reshape(shape)
    if nwater > 0:
        vp_inv[water_mask_3d] = np.float32(args.water_velocity)

    np.save(f"{exp_name}/vp_inv.npy",   vp_inv)
    np.save(f"{exp_name}/loss.npy",     np.array(misfit_history))
    np.save(f"{exp_name}/vp_error.npy", np.array(vp_error))

    plot_curve(np.array(misfit_history), f"{args.loss.upper()} misfit",
               "Loss", f"{exp_name}/loss_plot.png")
    plot_curve(np.array(vp_error), "Model error",
               "Relative RMS error", f"{exp_name}/vp_error_plot.png")

    plot_slices(vp_inv, x_coord, y_coord, z_coord, vmin=m_vmin, vmax=m_vmax,
                title="Inverted Vp", savepath=f"{exp_name}/result_inv.png")

    print(f"\nDone. Results in {os.path.abspath(exp_name)}/")


if __name__ == "__main__":
    main()
