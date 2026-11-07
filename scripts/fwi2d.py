"""2D Overthrust FWI driver (paper Figures 9-13 and 18, Tables 2-3, Supporting Information).

Script form of the original 2D Overthrust notebook: stage 1 with the early-arrival mask
and the chosen objective, then stage 2 with unmasked L2 on all data. Options cover the
revision experiments (pooling, baselines, encoder, layer, schedules, restarts, noise,
resampling, initial model, stopping tolerance). Each run writes <out>/<name>/run.json
(config, provenance, optimizer messages, histories, timings) plus the model arrays,
named as in the notebook.

Example (5-iteration check):
    python scripts/fwi2d.py --loss l2 --name check_l2 --maxiter1 5 --maxiter2 0
"""

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import minimize
from scipy.signal import butter, hilbert, sosfilt
from devito import configuration
from pylops.basicoperators import Diagonal, Identity
from devitofwi.waveengine.acoustic import AcousticWave2D
from devitofwi.preproc.masking import TimeSpaceMasking
from devitofwi.postproc.acoustic import PostProcessVP
from devitofwi.loss.dtw import SoftDTW

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from fm_fwi.losses import SeisLMFeatureLoss, L2, HybridLoss  # noqa: E402
from fm_fwi import baselines as bl  # noqa: E402
from fm_fwi.model.pretrained import (  # noqa: E402
    load_pretrained_multidim_wav2vec2, load_random_multidim_wav2vec2)

configuration["log-level"] = "ERROR"
LOSSES = ["l2", "feature", "hybrid", "envelope", "softdtw", "w2", "multiscale", "stdl2"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="run folder name under --out")
    p.add_argument("--out", default=os.path.join(REPO, "runs"))
    p.add_argument("--loss", choices=LOSSES, default="feature")
    p.add_argument("--acq", choices=["surface", "obc", "sparse", "sparse3d", "s1000_r125", "s1290_r50"], default="surface",
                   help="sparse3d: the 3D survey's density, 13 shots every 1.0 km and receivers every 50 m; "
                        "s1000_r125 / s1290_r50: shots every 1.0 / 1.29 km with receivers every 125 / 50 m")
    p.add_argument("--dx", type=float, default=0.025,
                   help="grid spacing in km (dx = dz); coarser than 25 m resamples the true model bilinearly, "
                        "receivers on every grid node, 30 shots unchanged; gradient smoothing stays in cells")
    # initial model (W7)
    p.add_argument("--init", choices=["linear", "linear3d", "smooth"], default="linear",
                   help="linear depth gradient (paper), the 3D example's linear start (true min to max velocity "
                        "over the subsurface, below the water) or Gaussian-smoothed true model")
    p.add_argument("--init-sigma", type=float, nargs=2, default=[50.0, 40.0])
    p.add_argument("--init-scale", type=float, default=1.0, help="multiply subsurface velocities (bias)")
    # SeisLM options (W1, W3, W6)
    p.add_argument("--pool", choices=["none", "mean"], default="none")
    p.add_argument("--encoder", choices=["pretrained", "random"], default="pretrained")
    p.add_argument("--seed", type=int, default=0, help="random-encoder seed")
    p.add_argument("--layer", type=int, default=None, help="0-based transformer layer (default: last)")
    p.add_argument("--resample", action="store_true", help="resample traces to 100 Hz before encoding")
    p.add_argument("--bf16", action="store_true", help="bfloat16 encoder")
    p.add_argument("--no-tf32", action="store_true", help="disable TF32 in cuDNN and matmul (exact fp32)")
    p.add_argument("--trace-chunk", type=int, default=128)
    p.add_argument("--ckpt", default=os.path.join(
        REPO, "pretrained_seislm_base/checkpoints/epoch=39-step=1203000.ckpt"))
    # hybrid schedule and L-BFGS restarts (W4)
    p.add_argument("--schedule", default="linear:100",
                   help="alpha (L2 weight) schedule: linear:N | step:N | cosine:N")
    p.add_argument("--alpha-hold", type=int, default=0, help="hold alpha constant for N iterations")
    p.add_argument("--restart-every", type=int, default=0, help="restart L-BFGS every N iterations")
    p.add_argument("--hybrid-scale", choices=["loss", "grad"], default="loss",
                   help="loss: each term divided by its summed loss at m0 (paper); grad: the L2 scale is "
                        "then multiplied so both normalized gradients have the same norm at m0 (R4.6)")
    p.add_argument("--trace-terms", type=int, default=0,
                   help="hybrid: every N accepted iterations (and at m0) evaluate each term's loss and "
                        "gradient separately and log them (R1.1, R4.6); costs two extra gradients each time")
    # baselines (W2)
    p.add_argument("--gamma", type=float, default=10.0, help="soft-DTW gamma")
    p.add_argument("--dtw-tsub", type=int, default=4)
    p.add_argument("--w2-shift", type=float, default=1.1)
    p.add_argument("--env-power", type=float, default=2.0,
                   help="envelope power p: 2 squared envelope (Wu et al. 2014), 1 envelope (Yuan et al. 2015)")
    p.add_argument("--bands", type=float, nargs="+", default=[2.0, 3.0, 4.0, 0.0],
                   help="multiscale low-pass cut-offs in Hz (0 = full band), one L-BFGS run each")
    p.add_argument("--band-iters", type=int, default=25)
    # noise (W5)
    p.add_argument("--noise-snr", type=float, default=None, help="SNR in dB; none = noise-free")
    p.add_argument("--noise-type", choices=["band", "coherent", "matched", "white", "flat"], default="band",
                   help="matched = noise with the data's own spectrum, white = flat to Nyquist "
                        "(--noise-band ignored for both); flat = flat inside --noise-band, zero outside")
    p.add_argument("--noise-band", type=float, nargs=2, default=None,
                   help="Hz; default 1-12 (band), 1-11 (flat) or 0.5-3 (coherent)")
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--noise-premute", type=float, default=None,
                   help="zero the noise before the first arrival, picked where |d| first exceeds this "
                        "fraction of the trace max (e.g. 1e-3); none = no mute")
    # source wavelet (W5): wrong wavelet in the inversion; data without low frequencies
    p.add_argument("--inv-freq", type=float, default=None,
                   help="invert with a Ricker of this peak frequency (Hz); the data keep the 5 Hz Ricker")
    p.add_argument("--inv-align", action="store_true",
                   help="with --inv-freq: centre the inversion Ricker at the data wavelet's peak time (1/freq), so only "
                        "the frequency differs; without it devitofwi delays each Ricker by 1/its own frequency")
    p.add_argument("--inv-phase", type=float, default=None,
                   help="degrees: rotate the phase of the inversion wavelet by this constant angle (analytic signal)")
    p.add_argument("--lowcut", type=float, default=None,
                   help="Hz: high-pass the source wavelet (causal 4th-order Butterworth) for the observed data "
                        "and the inversion alike, so neither has energy below it")
    # optimizer
    p.add_argument("--toff", type=float, default=0.4)
    p.add_argument("--tmax", type=float, default=None,
                   help="record length in s (default 3.996, the notebook's 1000 samples at 4 ms)")
    p.add_argument("--dt", type=float, default=None,
                   help="fixed modelling time step in s for every engine (must satisfy the CFL condition); "
                        "default: Devito's stability limit for the true model. E.g. 0.003774 = the 3D example")
    p.add_argument("--maxiter1", type=int, default=100)
    p.add_argument("--maxiter2", type=int, default=100, help="0 skips stage 2")
    p.add_argument("--stage", choices=["both", "1", "2"], default="both",
                   help="1: stage 1 only; 2: stage 2 only, from the saved stage-1 model of the existing "
                        "run <out>/<name> (same flags), e.g. stage 1 on a GPU node and stage 2 on a CPU node")
    p.add_argument("--grad-sigma2", type=float, nargs=2, default=[4.0, 4.0],
                   help="stage-2 gradient smoothing sigmas (x, z) in cells; 0 0 = no smoothing (diagnostic, R4.7). "
                        "Stage 1 always uses (4, 4)")
    p.add_argument("--maxfun1", type=int, default=50000)
    p.add_argument("--maxfun2", type=int, default=5000)
    p.add_argument("--maxcor1", type=int, default=20)
    p.add_argument("--ftol-rel", type=float, default=None,
                   help="stop when an accepted step lowers the loss by less than c*f(m0); "
                        "default: none (SciPy ftol=gtol=1e-30)")
    return p.parse_args(argv)


def alpha_schedule(spec, hold):
    kind, n = spec.split(":")
    n = float(n)
    f = {"linear": lambda it: min(1.0, it / n),
         "step": lambda it: float(it >= n),
         "cosine": lambda it: 0.5 * (1.0 - np.cos(np.pi * min(1.0, it / n)))}[kind]
    return (lambda it: f(hold * (it // hold))) if hold else f


def provenance():
    def run(*cmd):
        try:
            return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    import devito
    import scipy
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    cpu = ""
    if os.path.exists("/proc/cpuinfo"):
        cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")
    import devitofwi
    md5 = lambda f: hashlib.md5(open(f, "rb").read()).hexdigest()
    sources = ["scripts/fwi2d.py", "fm_fwi/baselines.py", "fm_fwi/losses.py",
               "fm_fwi/model/pretrained.py"]
    dfwi = os.path.dirname(devitofwi.__file__)
    commit = run("git", "rev-parse", "HEAD")
    git = {} if not commit else {"git_commit": commit,
                                 "git_dirty_md5": hashlib.md5(run("git", "diff", "HEAD").encode()).hexdigest(),
                                 "git_status": run("git", "status", "--porcelain", "--untracked-files=no")}
    return {
        **git,
        "source_md5": {f: md5(os.path.join(REPO, f)) for f in sources},
        "devitofwi": dfwi, "devitofwi_acoustic_md5": md5(os.path.join(dfwi, "waveengine/acoustic.py")),
        "host": socket.gethostname(), "cpu": cpu, "gpu": gpu,
        "omp_threads": os.environ.get("OMP_NUM_THREADS"), "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(), "torch": torch.__version__, "devito": devito.__version__,
        "numpy": np.__version__, "scipy": scipy.__version__,
    }


def velocity_error(x, vp_true):
    """RMS relative velocity error over the whole model, water included (as in the paper)."""
    return float(np.linalg.norm((x - vp_true.ravel()) / vp_true.ravel()) / np.sqrt(vp_true.size))


def run_lbfgs(ainv, x0, segments, vp_true, bounds, maxfun, maxcor, ftol_rel, on_iter=None):
    """L-BFGS-B over segments; each segment is a fresh L-BFGS run (memory reset).

    segments: list of (maxiter, setup, label); setup(x) runs at the segment start and
    returns (scaling, postprocess, f0): the PostProcessVP gradient scaling and the scaled
    loss f(m0) used by ftol_rel.
    """
    rec = {"loss": [], "err": [velocity_error(x0, vp_true)], "eval_s": [], "segments": []}
    x = x0.copy()
    stop = {"tol": None, "prev": None, "hit": False}

    def fun(xk, postproc):
        t0 = time.perf_counter()
        out = ainv.loss_grad(xk, None, postproc)
        rec["eval_s"].append(time.perf_counter() - t0)
        return out

    def callback(xk):
        rec["loss"].append(float(ainv.losshistory[-1]))
        rec["err"].append(velocity_error(xk, vp_true))
        extra = on_iter(len(rec["loss"]), xk) if on_iter else ""
        print(f"  iter {len(rec['loss']):3d}  loss = {rec['loss'][-1]:.4e}  err = {rec['err'][-1]:.4e}{extra}",
              flush=True)
        prev, stop["prev"] = stop["prev"], rec["loss"][-1]
        if stop["tol"] is not None and prev - rec["loss"][-1] <= stop["tol"]:
            stop["hit"] = True
            raise StopIteration  # L-BFGS-B ends with "STOP: CALLBACK" (SciPy >= 1.11)

    t_start = time.perf_counter()
    for maxiter, setup, label in segments:
        scaling, postproc, f0 = setup(x)
        ftol = 1e-30
        # Stopping control (R4.4): absolute decrease of the loss the optimizer sees (already
        # divided by the gradient scaling) below c * f(m0), checked after each accepted step.
        stop.update(tol=None if ftol_rel is None else ftol_rel * f0, prev=f0)
        nev0 = len(rec["eval_s"])
        res = minimize(fun, x, args=(postproc,), method="L-BFGS-B", jac=True, callback=callback,
                       bounds=bounds, options={"ftol": ftol, "gtol": 1e-30, "maxiter": maxiter,
                                               "maxfun": maxfun, "maxcor": maxcor, "disp": True})
        x = res.x
        rec["segments"].append({"label": label, "maxiter": maxiter, "scaling": float(scaling), "ftol": ftol,
                                "f0": f0, "stop_tol": stop["tol"], "stopped_by_ftol_rel": stop["hit"],
                                "message": str(res.message), "status": int(res.status),
                                "nit": int(res.nit), "nfev": int(res.nfev),
                                "evals_logged": len(rec["eval_s"]) - nev0})
        print(f"  segment {label} done: {res.message} (nit={res.nit}, nfev={res.nfev})", flush=True)
    rec["wall_s"] = time.perf_counter() - t_start
    rec["nit"] = sum(s["nit"] for s in rec["segments"])
    rec["nfev"] = sum(s["nfev"] for s in rec["segments"])
    rec["mean_eval_s"] = float(np.mean(rec["eval_s"])) if rec["eval_s"] else None
    return x, rec


def build_problem(args):
    """Model, acquisition, initial model, observed data (with optional noise) and mask,
    as in notebooks/SeisLM_Overthrust2D.ipynb. Returns a dict of everything built."""
    # ---- model and acquisition (notebook cells 5-9) ----
    water_velocity, dz = 1.5, getattr(args, "dx", 0.025)        # getattr: shift_diagnostics builds a bare namespace
    nwater = int(round(0.5 / dz))
    nx, nzb = int(round(12.5 / dz)) + 1, int(round(4.65 / dz)) + 1   # 501 x 187 at 25 m, the native Overthrust grid
    par = {"nx": nx, "dx": dz, "ox": 0, "nz": nzb + nwater, "dz": dz, "oz": 0,
           "ns": 30, "ds": 0.430, "os": 0, "sz": 0, "nr": nx, "dr": dz, "or": 0, "rz": 0.0,
           "nt": 1000, "dt": 0.004, "ot": 0, "freq": 5}
    if args.acq == "obc":
        par["rz"] = 0.5
    elif args.acq == "sparse":
        par.update(ns=10, ds=1.29, nr=101, dr=0.125)
    elif args.acq == "sparse3d":
        par.update(ns=13, ds=1.0, nr=251, dr=0.05)
    elif args.acq == "s1000_r125":
        par.update(ns=13, ds=1.0, nr=101, dr=0.125)
    elif args.acq == "s1290_r50":
        par.update(ns=10, ds=1.29, nr=251, dr=0.05)
    shape, spacing, origin = (par["nx"], par["nz"]), (par["dx"], par["dz"]), (par["ox"], par["oz"])
    space_order, nbl = 4, 20
    z = np.arange(par["nz"]) * par["dz"] + par["oz"]
    tmax = getattr(args, "tmax", None) or (par["nt"] - 1) * par["dt"] + par["ot"]
    x_s = np.zeros((par["ns"], 2))
    x_s[:, 0] = np.arange(par["ns"]) * par["ds"] + par["os"]
    x_s[:, 1] = par["sz"]
    x_r = np.zeros((par["nr"], 2))
    x_r[:, 0] = np.arange(par["nr"]) * par["dr"] + par["or"]
    x_r[:, 1] = par["rz"]

    vp = np.fromfile(os.path.join(REPO, "data/overthrust.vites"), dtype=">f4").astype(np.float32) / 1000
    vp = vp.reshape((187, 801, 801))
    sec = vp[:, 455, :501].T.astype(np.float32)                    # (501, 187) at 25 m
    if (nx, nzb) != sec.shape:                                     # coarser grid: bilinear resampling of the 25 m section
        r = dz / 0.025
        ix, iz = np.meshgrid(np.arange(nx) * r, np.arange(nzb) * r, indexing="ij")
        sec = map_coordinates(sec, [ix, iz], order=1, mode="nearest").astype(np.float32)
    water_layer = np.full((par["nx"], nwater), water_velocity, dtype=np.float32)
    vp_true = np.concatenate([water_layer, sec], axis=1)
    water = np.zeros(shape, dtype=bool)
    water[:, :nwater] = True
    vmin, vmax = vp_true.min(), vp_true.max()

    if args.init == "linear":
        v_top, v_bottom = float(vp_true[:, 0].mean()), float(vp_true[:, -1].mean())
        vp_init = np.tile((v_top + (v_bottom - v_top) * (z / z[-1]))[None, :], (par["nx"], 1)).astype(np.float32)
    elif args.init == "linear3d":   # as 3D_Overthrust_lbfgs_water.py --linear_init
        sub = np.tile(np.linspace(vmin, vmax, nzb, dtype=np.float32)[None, :], (par["nx"], 1))
        vp_init = np.concatenate([water_layer, sub], axis=1)
    else:
        sub = gaussian_filter(vp_true[:, nwater:], sigma=args.init_sigma)
        vp_init = np.concatenate([water_layer, sub.astype(np.float32)], axis=1)
    vp_init[~water] = np.clip(vp_init[~water] * args.init_scale, vmin, vmax)
    vp_init[water] = water_velocity
    grad_mask = np.ones(shape, dtype=np.float32)
    grad_mask[water] = 0.0
    bounds = [(float(water_velocity),) * 2 if w else (float(vmin), float(vmax)) for w in water.ravel()]

    # ---- observed data and mask (cells 11-15) ----
    amod = AcousticWave2D(shape, origin, spacing, x_s[:, 0], x_s[:, 1], x_r[:, 0], x_r[:, 1],
                          0., tmax, vp=vp_true, src_type="Ricker", f0=par["freq"],
                          space_order=space_order, nbl=nbl, dt=getattr(args, "dt", None), clearcache=False)
    model, geometry = amod.model_and_geometry()
    # getattr: shift_diagnostics.py builds the problem from a namespace without these flags
    hp = lambda w: w if not getattr(args, "lowcut", None) else sosfilt(
        butter(4, args.lowcut, "highpass", fs=1.0 / float(geometry.dt), output="sos"), w).astype(w.dtype)
    inv_wav = w0 = geometry.src.wavelet
    if getattr(args, "lowcut", None):        # devitofwi re-creates the source from .wav in every modelling call
        amod.wav = inv_wav = hp(w0)          # without the flags the observed data keep devitofwi's own Ricker
    if getattr(args, "inv_freq", None):      # a Ricker at inv_freq on the same time axis (same model, same dt)
        inv_wav = hp(AcousticWave2D(shape, origin, spacing, x_s[:, 0], x_s[:, 1], x_r[:, 0], x_r[:, 1],
                                    0., tmax, vp=vp_true, src_type="Ricker", f0=args.inv_freq, space_order=space_order,
                                    nbl=nbl, dt=getattr(args, "dt", None), clearcache=False).model_and_geometry()[1].src.wavelet)
        if getattr(args, "inv_align", False):  # devito's Ricker on the same time axis, centred at the data's peak time
            ricker = lambda f: ((1 - 2 * (np.pi * f * (t - 1 / par["freq"])) ** 2)
                                * np.exp(-(np.pi * f * (t - 1 / par["freq"])) ** 2)).astype(w0.dtype)
            t = np.arange(len(w0)) * float(geometry.dt)
            print(f"inv-align: max |ricker({par['freq']}) - data wavelet| = {np.abs(ricker(par['freq']) - w0).max():.2e}")
            inv_wav = hp(ricker(args.inv_freq))
    if getattr(args, "inv_phase", None):     # constant phase rotation, same amplitude spectrum and centre time
        a = hilbert(inv_wav)
        inv_wav = (np.real(a) * np.cos(np.deg2rad(args.inv_phase))
                   - np.imag(a) * np.sin(np.deg2rad(args.inv_phase))).astype(w0.dtype)
    dobs, _ = amod.mod_allshots()
    dobs = np.asarray(dobs)
    nt, dt = geometry.nt, float(geometry.dt)
    if args.noise_snr is not None:
        band = args.noise_band or {"band": [1.0, 12.0], "flat": [1.0, 11.0]}.get(args.noise_type, [0.5, 3.0])
        dobs = bl.add_noise(dobs, args.noise_snr, dt, kind=args.noise_type, band=band,
                            seed=args.noise_seed, premute=args.noise_premute).astype(dobs.dtype)
    distance = np.sqrt((x_s[:, 0][:, None] - x_r[:, 0][None, :]) ** 2 +
                       (x_s[:, 1][:, None] - x_r[:, 1][None, :]) ** 2)
    tsmask = TimeSpaceMasking(distance / water_velocity, nt, geometry.dt, args.toff, nsmooth=50)
    masks = [tsmask.mask[i].T for i in range(par["ns"])]
    TSmaskop = [Diagonal(m.ravel()) for m in masks]
    dobs_masked = np.stack([(TSmaskop[i] @ dobs[i].ravel()).reshape(nt, par["nr"]) for i in range(par["ns"])])
    b = dobs_masked.reshape(par["ns"], -1)
    return {k: v for k, v in locals().items() if k != "args"}


def main():
    args = parse_args()
    run_dir = os.path.join(args.out, args.name)
    if args.stage == "2":
        meta = json.load(open(os.path.join(run_dir, "run.json")))
        # Stage-2-only flags may differ (a stage-1-only run has --maxiter2 0); they are kept in stage2_config.
        # Flags added after the stage-1 run count as their default, which reproduces the older behaviour.
        dflt = vars(parse_args(["--name", args.name]))
        diff = {k: (meta["config"].get(k, dflt[k]), v) for k, v in vars(args).items()
                if k not in ("stage", "out", "ckpt", "maxiter2", "maxfun2", "grad_sigma2") and meta["config"].get(k, dflt[k]) != v}
        if "stage1" not in meta or "stage2" in meta or diff:
            raise SystemExit(f"{run_dir}: need a finished stage 1, no stage 2 yet and the same flags; diff={diff}")
        meta["stage2_config"], meta["stage2_provenance"] = vars(args), provenance()
    else:
        os.makedirs(run_dir, exist_ok=False)
        meta = {"config": vars(args), "provenance": provenance()}
    save_json = lambda: json.dump(meta, open(os.path.join(run_dir, "run.json"), "w"), indent=1)
    save_json()
    print(json.dumps(meta, indent=1), flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.no_tf32:
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False

    P = build_problem(args)
    (par, shape, spacing, origin, space_order, nbl, tmax, x_s, x_r, vp_true, water, vmin, vmax, vp_init, grad_mask, bounds, geometry, dobs, nt, dt, masks, TSmaskop, dobs_masked, b) = (P[k] for k in ('par', 'shape', 'spacing', 'origin', 'space_order', 'nbl', 'tmax', 'x_s', 'x_r', 'vp_true', 'water', 'vmin', 'vmax', 'vp_init', 'grad_mask', 'bounds', 'geometry', 'dobs', 'nt', 'dt', 'masks', 'TSmaskop', 'dobs_masked', 'b'))
    inv_wav = P["inv_wav"]
    if args.stage != "2":
        np.save(os.path.join(run_dir, "vp_init.npy"), vp_init)
        if args.noise_snr is not None:
            np.save(os.path.join(run_dir, "dobs_noisy.npy"), dobs)

    def make_postproc(eng, x, store, sigmas=(4, 4)):
        """PostProcessVP with scaling max|g| at x (the paper's gradient normalization); sigmas None = no smoothing."""
        loss0, g = eng._loss_grad(x.reshape(shape), postprocess=PostProcessVP(
            scaling=1, mask=grad_mask, sigmas=sigmas).apply)
        store.setdefault("grad0", g)
        store.setdefault("loss0", float(loss0))
        s = float(np.abs(g).max())
        return s, PostProcessVP(scaling=s, mask=grad_mask, sigmas=sigmas).apply, float(loss0) / s


    def stage1():
        # ---- stage-1 objective (cells 18-21) ----
        band = {"fc": None}
        trace_losses = {
            "envelope": lambda s, o: bl.envelope_l2(s, o, p=args.env_power),
            "stdl2": bl.standardized_l2,
            "w2": lambda s, o: bl.wasserstein2(s, o, dt=dt, shift=args.w2_shift),
            "multiscale": lambda s, o: bl.lowpass_l2(s, o, dt, band["fc"]),
        }
        hybrid = None
        if args.loss == "l2":
            loss_fn = L2(TSmaskop, b)
        elif args.loss in trace_losses:
            loss_fn = bl.TraceMisfit(trace_losses[args.loss], b, nt, par["nr"], mask=masks)
        elif args.loss == "softdtw":
            # Same global amplitude scale on Op and b so gamma works on ~unit-RMS data (C5).
            scale = 1.0 / (np.sqrt((dobs_masked ** 2).mean()) + 1e-30)
            meta["softdtw_scale"] = float(scale)
            bl.fix_tslearn_softdtw()                   # tslearn 0.9.0's PyTorch gradient is wrong (fm_fwi/baselines.py)
            meta["softdtw_grad"] = "fixed (D padded, R float64)"
            loss_fn = SoftDTW(Op=[Diagonal(scale * m.ravel()) for m in masks], b=scale * b,
                              nr=par["nr"], nt=nt, gamma=args.gamma, normalize=True,
                              t_sub=args.dtw_tsub, device=device)
        else:
            if args.encoder == "pretrained":
                model_fm = load_pretrained_multidim_wav2vec2(args.ckpt, map_location=device)
            else:
                model_fm = load_random_multidim_wav2vec2(args.ckpt, map_location=device, seed=args.seed)
            model_fm = model_fm.to(device, dtype=torch.bfloat16 if args.bf16 else torch.float32).eval()
            feat = SeisLMFeatureLoss(
                seislm_model=model_fm, Op=TSmaskop, b=b, nr=par["nr"], nt=nt, t_sub=1, device=device,
                pool=args.pool, normalize_traces=True, trace_chunk=args.trace_chunk,
                cache_obs_embeddings=True, cache_obs_on_cpu=True, dt=dt, resample=args.resample,
                fs_target=100.0, feature_layers=args.layer)
            loss_fn = feat
            if args.loss == "hybrid":
                loss_fn = hybrid = HybridLoss(l2_loss=L2(TSmaskop, b), seislm_loss=feat,
                                              alpha=alpha_schedule(args.schedule, args.alpha_hold))

        ainv = AcousticWave2D(shape, origin, spacing, x_s[:, 0], x_s[:, 1], x_r[:, 0], x_r[:, 1],
                              0., tmax, vprange=(vmin, vmax), wav=inv_wav, f0=par["freq"],
                              space_order=space_order, nbl=nbl, dt=getattr(args, "dt", None), loss=loss_fn)
        if hybrid is not None:
            # Scales = each term summed over all shots at m0 (notebook cell 21). The data at m0
            # come from a forward-only engine, because devitofwi's _loss_grad(computegrad=False)
            # fails on an unset gradient.
            d0, _ = AcousticWave2D(shape, origin, spacing, x_s[:, 0], x_s[:, 1], x_r[:, 0], x_r[:, 1],
                                   0., tmax, vp=vp_init, src_type="Ricker", f0=par["freq"],
                                   wav=None if inv_wav is P["w0"] else inv_wav, space_order=space_order, nbl=nbl, dt=getattr(args, "dt", None), clearcache=False).mod_allshots()
            hybrid.begin_scale_accumulation()
            for i in range(par["ns"]):
                hybrid.loss(np.asarray(d0[i]).ravel(), i, calibrating=True)
            hybrid.finalize_scales()
            meta["hybrid_scales"] = [hybrid._scale_l2, hybrid._scale_seislm]

        # Per-term tracing (R1.1, R4.6): each term on its own engine, same postprocessing as the
        # inversion but without the global max|g| scaling, which is common to both terms.
        terms = []
        if hybrid is not None and (args.trace_terms or args.hybrid_scale == "grad"):
            eng = {k: AcousticWave2D(shape, origin, spacing, x_s[:, 0], x_s[:, 1], x_r[:, 0], x_r[:, 1],
                                     0., tmax, vprange=(vmin, vmax), wav=inv_wav, f0=par["freq"],
                                     space_order=space_order, nbl=nbl, dt=getattr(args, "dt", None), loss=f)
                   for k, f in (("l2", L2(TSmaskop, b)), ("feat", feat))}
            pp = PostProcessVP(scaling=1, mask=grad_mask, sigmas=(4, 4)).apply

            def trace_terms(it, x):
                a = hybrid.alpha
                t = {"iter": it, "alpha": a}
                g = {}
                for k, S in (("l2", hybrid._scale_l2), ("feat", hybrid._scale_seislm)):
                    f, gk = eng[k]._loss_grad(x.reshape(shape), postprocess=pp)
                    g[k] = gk.ravel() / S
                    t[f"{k}_raw"], t[f"{k}_norm"], t[f"{k}_gnorm"] = float(f), float(f) / S, float(np.linalg.norm(g[k]))
                t["gnorm_ratio"] = t["l2_gnorm"] / t["feat_gnorm"]       # L2 / SeisLM, before alpha
                t["share_l2"] = a * t["l2_gnorm"] / (a * t["l2_gnorm"] + (1 - a) * t["feat_gnorm"])
                t["cos"] = float(g["l2"] @ g["feat"] / (np.linalg.norm(g["l2"]) * np.linalg.norm(g["feat"])))
                terms.append(t)
                print(f"  terms it {it}: alpha={a:.3f} L2/S={t['l2_norm']:.3e} F/S={t['feat_norm']:.3e} "
                      f"|gL2|/|gF|={t['gnorm_ratio']:.2f} L2 share={t['share_l2']:.2f} cos={t['cos']:.3f}",
                      flush=True)
                return t

            t0 = trace_terms(0, vp_init.ravel())
            if args.hybrid_scale == "grad":
                # Fixed scales again, only the L2 one changes: equal normalized gradient norms at m0.
                hybrid.set_scales(hybrid._scale_l2 * t0["gnorm_ratio"], hybrid._scale_seislm)
                meta["hybrid_scales_grad"] = [hybrid._scale_l2, hybrid._scale_seislm]
                terms.clear()
                trace_terms(0, vp_init.ravel())

        first = {}

        fixed = []

        def fixed_setup(x):
            # Restarts keep the m0 scaling (and hybrid scales), so a restart changes only the
            # L-BFGS memory (R5.3).
            if not fixed:
                fixed.append(make_postproc(ainv, x, first))
            return fixed[0]

        def band_setup(fc):
            # Multiscale: switch the band, then rescale at the current model, as is standard.
            def setup(x):
                band["fc"] = fc or None
                return make_postproc(ainv, x, first)
            return setup

        if args.loss == "multiscale":
            segments = [(args.band_iters, band_setup(fc), f"fc={fc}") for fc in args.bands]
        else:
            n = args.restart_every or args.maxiter1
            sizes = [n] * (args.maxiter1 // n) + ([args.maxiter1 % n] if args.maxiter1 % n else [])
            segments = [(k, fixed_setup, f"seg{j}") for j, k in enumerate(sizes)]

        def on_iter(it, x):
            if hybrid is None:
                return ""
            a = hybrid.alpha
            hybrid.step()
            if args.trace_terms and it % args.trace_terms == 0:
                trace_terms(it, x)   # at the alpha of the next iteration
            return f"  alpha={a:.4f}"

        x_cur, rec1 = run_lbfgs(ainv, vp_init.ravel().copy(), segments, vp_true, bounds,
                                args.maxfun1, args.maxcor1, args.ftol_rel, on_iter)
        rec1["loss0_raw"] = first.get("loss0")
        if terms:
            rec1["terms"] = terms
        rec1["losshistory_all"] = [float(v) for v in ainv.losshistory]
        vp_inv = x_cur.reshape(shape)
        np.save(os.path.join(run_dir, "vp_inv.npy"), vp_inv)
        np.save(os.path.join(run_dir, "grad.npy"), first["grad0"])
        np.save(os.path.join(run_dir, "loss.npy"), np.array(rec1["loss"]))
        np.save(os.path.join(run_dir, "loss_hist.npy"), np.array(ainv.losshistory))
        np.save(os.path.join(run_dir, "err.npy"), np.array(rec1["err"]))
        meta["stage1"] = rec1
        save_json()
        return vp_inv

    vp_inv = stage1() if args.stage != "2" else np.load(os.path.join(run_dir, "vp_inv.npy"))

    # ---- stage 2: unmasked L2 on all data (cells 29-34), identical for every method ----
    if args.maxiter2 > 0 and args.stage != "1":
        ainv2 = AcousticWave2D(shape, origin, spacing, x_s[:, 0], x_s[:, 1], x_r[:, 0], x_r[:, 1],
                               0., tmax, vprange=(vmin, vmax), wav=inv_wav, f0=par["freq"],
                               space_order=space_order, nbl=nbl, dt=getattr(args, "dt", None),
                               loss=L2(Identity(int(np.prod(dobs.shape[1:]))), dobs.reshape(par["ns"], -1)))
        st2 = {}

        def setup2(x):
            sig2 = tuple(args.grad_sigma2) if any(args.grad_sigma2) else None
            out = make_postproc(ainv2, x, st2, sigmas=sig2)
            np.save(os.path.join(run_dir, "grad_refl.npy"), st2["grad0"])
            return out

        # SciPy default maxcor (10) in stage 2, as in the notebook.
        x2, rec2 = run_lbfgs(ainv2, vp_inv.ravel(), [(args.maxiter2, setup2, "stage2")], vp_true, bounds,
                             args.maxfun2, 10, args.ftol_rel)
        rec2["losshistory_all"] = [float(v) for v in ainv2.losshistory]
        np.save(os.path.join(run_dir, "vp_inv_refl.npy"), x2.reshape(shape))
        np.save(os.path.join(run_dir, "loss_refl.npy"), np.array(rec2["loss"]))
        np.save(os.path.join(run_dir, "loss_hist_refl.npy"), np.array(ainv2.losshistory))
        np.save(os.path.join(run_dir, "err_refl.npy"), np.array(rec2["err"]))
        meta["stage2"] = rec2
        save_json()
    print(f"Done: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
