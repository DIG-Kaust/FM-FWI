"""Marmousi two-stage FWI (paper Figs 5-8), the script form of the original Marmousi notebook (cells 5-37).

Stage 1: 100 L-BFGS-B iterations of the chosen loss on data muted after the direct arrival + 0.35 s.
Stage 2: 100 L-BFGS-B iterations of L2 on the full data, from the stage-1 model.
Same settings as the notebook: no bounds, default maxcor, ftol = gtol = 1e-30, gradient smoothing (4, 4),
pool="none", normalize_traces, trace_chunk 32, hybrid scales summed over all shots at m0, alpha = min(1, it/100).
Outputs use the notebook's file names (vp_inv, err, loss_hist, *_refl, grad, grad_refl).

  python scripts/marm2d.py --loss feature --out runs/marm --name marm_feature --stage 1
  python scripts/marm2d.py --loss feature --out runs/marm --name marm_feature --stage 2
"""
import argparse, json, os, socket, sys, time
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from devito import configuration
from pylops.basicoperators import Diagonal, Identity
from devitofwi.waveengine.acoustic import AcousticWave2D
from devitofwi.preproc.masking import TimeSpaceMasking
from devitofwi.postproc.acoustic import create_mask_value, PostProcessVP

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from fm_fwi.model.pretrained import load_pretrained_multidim_wav2vec2  # noqa: E402
from fm_fwi.losses import L2, HybridLoss, SeisLMFeatureLoss  # noqa: E402
from fm_fwi.tee import Tee  # noqa: E402

configuration['log-level'] = 'ERROR'
p = argparse.ArgumentParser()
p.add_argument("--loss", choices=("l2", "feature", "hybrid"), required=True)
p.add_argument("--out", required=True)
p.add_argument("--name", required=True)
p.add_argument("--stage", choices=("1", "2", "both"), default="both")
p.add_argument("--maxiter1", type=int, default=100)
p.add_argument("--maxiter2", type=int, default=100)
p.add_argument("--obs-cache-gpu", action="store_true", help="keep observed embeddings on the GPU (July: CPU)")
p.add_argument("--trace-chunk", type=int, default=32, help="traces per SeisLM batch (July: 32); batching only")
p.add_argument("--ckpt", default=os.path.join(REPO, "pretrained_seislm_base/checkpoints/epoch=39-step=1203000.ckpt"))
args = p.parse_args()
run_dir = os.path.join(args.out, args.name)
os.makedirs(run_dir, exist_ok=True)
meta_path = os.path.join(run_dir, f"run_stage{args.stage}.json")
meta = {"config": vars(args), "host": socket.gethostname(), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

# Cell 5-11: model, acquisition, starting model
par = {'nx': 601, 'dx': 0.015, 'ox': 0, 'nz': 221, 'dz': 0.015, 'oz': 0,
       'ns': 30, 'ds': 0.300, 'os': 0., 'sz': 0, 'nr': 300, 'dr': 0.030, 'or': 0, 'rz': 0,
       'nt': 3000, 'dt': 0.002, 'ot': 0, 'freq': 7}
shape, spacing, origin = (par['nx'], par['nz']), (par['dx'], par['dz']), (par['ox'], par['oz'])
space_order, nbl, vwater, toff, sm1, sm2 = 4, 20, 1.5, 0.35, 60, 40
tmax = (np.arange(par['nt']) * par['dt'] + par['ot'])[-1]
x_s = np.zeros((par['ns'], 2)); x_s[:, 0] = np.arange(par['ns']) * par['ds'] + par['os']; x_s[:, 1] = par['sz']
x_r = np.zeros((par['nr'], 2)); x_r[:, 0] = np.arange(par['nr']) * par['dr'] + par['or']; x_r[:, 1] = par['rz']
vp_true = np.fromfile(os.path.join(REPO, 'data/Marm.bin'), np.float32).reshape(par['nz'], par['nx']).T
msk = create_mask_value(vp_true, 1.52)
vp_init = gaussian_filter(vp_true, sigma=[sm1, sm2]) * msk
vp_init[vp_init == 0] = 1.5
rel_err = lambda v: float(np.linalg.norm((v.reshape(-1) - vp_true.reshape(-1)) / vp_true.reshape(-1))
                          / np.sqrt(vp_true.size))

# Cell 13-19: observed data and direct-arrival mute
amod = AcousticWave2D(shape, origin, spacing, x_s[:, 0], x_s[:, 1], x_r[:, 0], x_r[:, 1], 0., tmax, vp=vp_true,
                      src_type="Ricker", f0=par['freq'], space_order=space_order, nbl=nbl, clearcache=False)
model, geometry = amod.model_and_geometry()
dobs, _ = amod.mod_allshots()
wav = geometry.src.wavelet


def fwi_callback(xk, loss_fn, vp_error, misfit_history, hybrid=None):   # cell 27
    misfit_history.append(loss_fn._loss_value)
    vp_error.append(rel_err(xk))
    extra = ""
    if hybrid is not None:
        extra = f"  Outer iter {hybrid._iter}: alpha={hybrid.alpha:.4f}"
        hybrid.step()
    print(f"  iter {len(misfit_history):3d}  loss = {misfit_history[-1]:.4e}  err = {vp_error[-1]:.4e}{extra}  [{time.strftime('%H:%M:%S')}]",
          flush=True)


def lbfgs(ainv, x0, postproc, maxiter, maxfun, loss_fn, vp_error, misfit, hybrid, log_mode, header=None):
    with open(os.path.join(run_dir, f"minimize_log_{args.loss}.txt"), log_mode, buffering=1) as log:
        if header:
            log.write(header)
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            nl = minimize(ainv.loss_grad, x0.ravel(), method="L-BFGS-B", jac=True, args=(None, postproc),
                          callback=lambda x: fwi_callback(x, loss_fn, vp_error, misfit, hybrid),
                          options={"ftol": 1e-30, "gtol": 1e-30, "maxiter": maxiter, "maxfun": maxfun,
                                   "disp": True})
            print(nl)
    return nl


if args.stage in ("1", "both"):
    t0 = time.time()
    distance = np.sqrt((x_s[:, 0][:, None] - x_r[:, 0][None, :]) ** 2 + (x_s[:, 1][:, None] - x_r[:, 1][None, :]) ** 2)
    tsmask = TimeSpaceMasking(distance / vwater, geometry.nt, geometry.dt, toff, nsmooth=50)
    TSmaskop = [Diagonal(tsmask.mask[i].T.ravel()) for i in range(par['ns'])]
    dobs_masked = np.hstack([(TSmaskop[i] @ dobs[i].ravel())[None, :] for i in range(par['ns'])])
    b = dobs_masked.reshape(par['ns'], -1)

    # Cell 21-22: loss
    hybrid = None
    if args.loss in ("feature", "hybrid"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        seislm = load_pretrained_multidim_wav2vec2(args.ckpt, map_location=device).to(device).eval()
        feat = SeisLMFeatureLoss(seislm_model=seislm, Op=TSmaskop, b=b, nr=par['nr'], nt=geometry.nt, t_sub=1,
                                 device=device, pool="none", normalize_traces=True, trace_chunk=args.trace_chunk,
                                 cache_obs_embeddings=True, cache_obs_on_cpu=not args.obs_cache_gpu)
    if args.loss == "l2":
        loss_fn = L2(TSmaskop, b)
    elif args.loss == "feature":
        loss_fn = feat
    else:
        loss_fn = hybrid = HybridLoss(l2_loss=L2(TSmaskop, b), seislm_loss=feat, alpha=lambda it: min(1.0, it / 100.0))

    # Cell 23
    ainv = AcousticWave2D(shape, origin, spacing, x_s[:, 0], x_s[:, 1], x_r[:, 0], x_r[:, 1], 0., tmax,
                          vprange=(vp_true.min(), vp_true.max()), wav=wav, f0=par['freq'],
                          space_order=space_order, nbl=nbl, loss=loss_fn)

    # Cell 24: hybrid scales = each term summed over all shots at m0, from the inversion engine itself as in the
    # notebook. _loss_grad(computegrad=False) needs devitofwi 662ee6d or later (Devito-fwi PR #27).
    # (A forward-only engine with its own time step, as fwi2d.py uses, does not reproduce these scales for Marmousi.)
    if hybrid is not None:
        hybrid.begin_scale_accumulation()
        ainv._loss_grad(vp_init, postprocess=None, computeloss=True, computegrad=False)
        hybrid.finalize_scales()
        meta["hybrid_scales"] = [hybrid._scale_l2, hybrid._scale_seislm]

    # Cell 25: first gradient and its scaling
    loss0, direction = ainv._loss_grad(vp_init, postprocess=PostProcessVP(scaling=1, mask=msk, sigmas=(4, 4)).apply)
    scaling = float(np.abs(direction).max())
    print(f"Scaling factor for the gradient: {scaling:.3e}\nLoss at the initial model: {loss0:.3e}", flush=True)
    np.save(os.path.join(run_dir, "grad.npy"), direction)
    meta.update(loss0=float(loss0), scaling1=scaling, J_over_maxg=float(loss0) / scaling)

    # Cell 28-31
    vp_error, misfit = [rel_err(vp_init)], []
    nl = lbfgs(ainv, vp_init, PostProcessVP(scaling=scaling, mask=msk, sigmas=(4, 4)).apply, args.maxiter1, 50000,
               loss_fn, vp_error, misfit, hybrid, "w")
    vp_inv = nl.x.reshape(shape)
    np.save(os.path.join(run_dir, "loss.npy"), np.array(misfit))
    np.save(os.path.join(run_dir, "loss_hist.npy"), np.array(ainv.losshistory))
    np.save(os.path.join(run_dir, "err.npy"), np.array(vp_error))
    np.save(os.path.join(run_dir, "vp_inv.npy"), vp_inv)
    meta["stage1"] = {"nit": int(nl.nit), "nfev": int(nl.nfev), "message": str(nl.message), "err0": vp_error[0],
                      "err": vp_error[-1], "seconds": time.time() - t0}
    json.dump(meta, open(meta_path, "w"), indent=1)

if args.stage in ("2", "both"):
    t0 = time.time()
    vp_inv = np.load(os.path.join(run_dir, "vp_inv.npy"))
    # Cell 32-37: L2 on the full data from the stage-1 model
    l2loss = L2(Identity(int(np.prod(dobs.shape[1:]))), dobs.reshape(par['ns'], -1))
    ainv2 = AcousticWave2D(shape, origin, spacing, x_s[:, 0], x_s[:, 1], x_r[:, 0], x_r[:, 1], 0., tmax,
                           vprange=(vp_true.min(), vp_true.max()), wav=wav, f0=par['freq'],
                           space_order=space_order, nbl=nbl, loss=l2loss)
    loss0, direction = ainv2._loss_grad(vp_inv, postprocess=PostProcessVP(scaling=1, mask=msk, sigmas=(4, 4)).apply)
    scaling = float(np.abs(direction).max())
    np.save(os.path.join(run_dir, "grad_refl.npy"), direction)
    vp_error2, misfit2 = [], []
    bar = "=" * 80
    nl2 = lbfgs(ainv2, vp_inv, PostProcessVP(scaling=scaling, mask=msk, sigmas=(4, 4)).apply, args.maxiter2, 5000,
                l2loss, vp_error2, misfit2, None, "a", f"\n{bar}\nInversion with reflection starts\n{bar}\n\n")
    np.save(os.path.join(run_dir, "loss_refl.npy"), np.array(misfit2))
    np.save(os.path.join(run_dir, "loss_hist_refl.npy"), np.array(ainv2.losshistory))
    np.save(os.path.join(run_dir, "err_refl.npy"), np.array(vp_error2))
    np.save(os.path.join(run_dir, "vp_inv_refl.npy"), nl2.x.reshape(shape))
    meta["stage2"] = {"nit": int(nl2.nit), "nfev": int(nl2.nfev), "message": str(nl2.message), "scaling2": scaling,
                      "err_start": rel_err(vp_inv), "err": vp_error2[-1] if vp_error2 else None,
                      "seconds": time.time() - t0}
    json.dump(meta, open(meta_path, "w"), indent=1)
print("done", flush=True)
