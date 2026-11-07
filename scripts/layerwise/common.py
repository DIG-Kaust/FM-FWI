"""Shared utilities for the SeisLM-FWI interpretability analysis suite.

All experiments reuse the FM-FWI stack (fm_fwi + devitofwi). Heavy artifacts
(observed data, geometry) are cached under runs/diagnostics/layerwise/cache/, so that
layerwise_sensitivity.py needs Devito only on its first run.

Conventions
-----------
- Traces are float32, shape [B, T] (batch of receiver traces).
- Encoders are callables: encode(traces_bt, layer=None) -> [B, D] pooled
  features (pool="mean", matching the FWI experiments) or [B, L, D] when
  pool="none" (matching Eq. 5 of the paper).
"""

import os
import numpy as np
import torch

from fm_fwi.model.pretrained import load_pretrained_multidim_wav2vec2
from fm_fwi.model.multidim_wav2vec2 import MultiDimWav2Vec2ForPreTraining

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.join(os.environ.get("FMFWI_RUNS", os.path.join(REPO, "runs")), "diagnostics")   # outputs
CACHE_DIR = os.path.join(HERE, "layerwise", "cache")
DEFAULT_CKPT = os.path.join(REPO, "pretrained_seislm_base", "checkpoints", "epoch=39-step=1203000.ckpt")

# ---------------------------------------------------------------------------
# Marmousi problem definition (as scripts/marm2d.py)
# ---------------------------------------------------------------------------

PAR = {"nx": 601, "dx": 0.015, "ox": 0,
       "nz": 221, "dz": 0.015, "oz": 0,
       "ns": 30, "ds": 0.300, "os": 0.0, "sz": 0,
       "nr": 300, "dr": 0.030, "or": 0, "rz": 0,
       "nt": 3000, "dt": 0.002, "ot": 0,
       "freq": 7}
SM1, SM2 = 60, 40          # smoothing used for the initial model
VWATER, TOFF = 1.5, 0.35   # direct-arrival mask parameters


def marmousi_paths():
    return os.path.join(REPO, "data", "Marm.bin")


def load_marmousi_models():
    """Return (vp_true, vp_init, msk, x, z) on the FWI grid."""
    from scipy.ndimage import gaussian_filter
    from devitofwi.postproc.acoustic import create_mask_value

    vp_true = np.fromfile(marmousi_paths(), np.float32)
    vp_true = vp_true.reshape(PAR["nz"], PAR["nx"]).T
    msk = create_mask_value(vp_true, 1.52)
    vp_init = gaussian_filter(vp_true, sigma=[SM1, SM2])
    vp_init = vp_init * msk
    vp_init[vp_init == 0] = 1.5
    x = np.arange(PAR["nx"]) * PAR["dx"]
    z = np.arange(PAR["nz"]) * PAR["dz"]
    return vp_true, vp_init, msk, x, z


def acquisition():
    x_s = np.zeros((PAR["ns"], 2))
    x_s[:, 0] = np.arange(PAR["ns"]) * PAR["ds"] + PAR["os"]
    x_s[:, 1] = PAR["sz"]
    x_r = np.zeros((PAR["nr"], 2))
    x_r[:, 0] = np.arange(PAR["nr"]) * PAR["dr"] + PAR["or"]
    x_r[:, 1] = PAR["rz"]
    return x_s, x_r


def make_engine(vp, loss=None, vprange=None, wav=None):
    """AcousticWave2D engine for a given velocity model (Devito required)."""
    from devitofwi.waveengine.acoustic import AcousticWave2D

    x_s, x_r = acquisition()
    shape = (PAR["nx"], PAR["nz"])
    spacing = (PAR["dx"], PAR["dz"])
    origin = (PAR["ox"], PAR["oz"])
    tmax = (PAR["nt"] - 1) * PAR["dt"]
    kwargs = dict(src_type="Ricker", f0=PAR["freq"],
                  space_order=4, nbl=20, clearcache=False)
    if loss is not None:
        kwargs["loss"] = loss
    if vprange is not None:
        return AcousticWave2D(shape, origin, spacing,
                              x_s[:, 0], x_s[:, 1], x_r[:, 0], x_r[:, 1],
                              0.0, tmax, vprange=vprange, wav=wav, **kwargs)
    return AcousticWave2D(shape, origin, spacing,
                          x_s[:, 0], x_s[:, 1], x_r[:, 0], x_r[:, 1],
                          0.0, tmax, vp=vp, **kwargs)


def direct_mask(geometry_nt, geometry_dt):
    """TimeSpaceMasking + per-shot Diagonal operators (as in the notebook)."""
    from devitofwi.preproc.masking import TimeSpaceMasking
    from pylops.basicoperators import Diagonal

    x_s, x_r = acquisition()
    distance = np.sqrt((x_s[:, 0][:, None] - x_r[:, 0][None, :]) ** 2
                       + (x_s[:, 1][:, None] - x_r[:, 1][None, :]) ** 2)
    directtime = distance / VWATER
    tsmask = TimeSpaceMasking(directtime, geometry_nt, geometry_dt, TOFF,
                              nsmooth=50)
    ops = [Diagonal(tsmask.mask[i].T.ravel()) for i in range(PAR["ns"])]
    return tsmask, ops, directtime


def observed_data(use_cache=True):
    """Model (or load cached) observed Marmousi data.

    Returns
    -------
    dobs : [ns, nt_geom, nr], dobs_masked : [ns, nt_geom, nr],
    nt_geom : int, dt_geom : float
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    f_obs = os.path.join(CACHE_DIR, "marmousi_dobs.npz")
    if use_cache and os.path.exists(f_obs):
        d = np.load(f_obs)
        return d["dobs"], d["dobs_masked"], int(d["nt_geom"]), float(d["dt_geom"])

    vp_true, _, _, _, _ = load_marmousi_models()
    amod = make_engine(vp_true)
    _, geometry = amod.model_and_geometry()
    dobs, _ = amod.mod_allshots()
    dobs = np.asarray(dobs)
    nt_geom, dt_geom = geometry.nt, geometry.dt

    tsmask, _, _ = direct_mask(nt_geom, dt_geom)
    dobs_masked = np.stack([dobs[i] * tsmask.mask[i].T for i in range(PAR["ns"])])
    np.savez(f_obs, dobs=dobs, dobs_masked=dobs_masked,
             nt_geom=nt_geom, dt_geom=dt_geom)
    return dobs, dobs_masked, nt_geom, float(dt_geom)


def model_shots_on_obs_grid(vp, nt_obs, dt_obs, ishots=None):
    """Forward-model vp and return shots on the OBSERVED time grid.

    Devito derives the simulation time step from the CFL condition of the
    given velocity model, so different trial models yield different nt.
    Modeled traces are linearly interpolated onto the observed grid
    (nt_obs, dt_obs) so that mask operators and losses built on the observed
    geometry apply. Returns [len(ishots), nt_obs, nr] float32.
    """
    amod = make_engine(vp)
    _, geometry = amod.model_and_geometry()
    dt_mod = float(geometry.dt)
    dmod, _ = amod.mod_allshots()
    dmod = np.asarray(dmod)
    if ishots is None:
        ishots = np.arange(dmod.shape[0])
    dmod = dmod[np.asarray(ishots)]
    nt_mod = dmod.shape[1]
    if nt_mod == nt_obs and np.isclose(dt_mod, float(dt_obs)):
        return dmod.astype(np.float32)
    t_mod = np.arange(nt_mod) * dt_mod
    t_obs = np.arange(nt_obs) * float(dt_obs)
    out = np.empty((dmod.shape[0], nt_obs, dmod.shape[2]), dtype=np.float32)
    for i in range(dmod.shape[0]):
        for r in range(dmod.shape[2]):
            out[i, :, r] = np.interp(t_obs, t_mod, dmod[i, :, r])
    return out


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

def load_random_multidim_wav2vec2(ckpt_path, map_location="cpu", seed=0):
    """Same architecture as the pretrained SeisLM, randomly initialized.

    The checkpoint is used ONLY to recover the config; no weights are loaded.
    """
    checkpoint = torch.load(ckpt_path, map_location=map_location,
                            weights_only=False)
    config = checkpoint["hyper_parameters"]["config"]
    torch.manual_seed(seed)
    model = MultiDimWav2Vec2ForPreTraining(config.model_config)
    return model.eval()


def get_seislm(kind, ckpt_path=DEFAULT_CKPT, device="cpu", seed=0):
    """kind in {"pretrained", "random"} -> frozen eval-mode model on device."""
    if kind == "pretrained":
        model = load_pretrained_multidim_wav2vec2(ckpt_path,
                                                  map_location=device)
    elif kind == "random":
        model = load_random_multidim_wav2vec2(ckpt_path,
                                              map_location=device, seed=seed)
    else:
        raise ValueError(f"unknown encoder kind '{kind}'")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def standardize(traces_bt, eps=1e-6):
    """Per-trace standardization, [B, T] numpy or torch -> torch [B, T]."""
    x = torch.as_tensor(traces_bt, dtype=torch.float32)
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, unbiased=False, keepdim=True).clamp_min(eps)
    return (x - mean) / std


@torch.no_grad()
def encode(model, traces_bt, layer=None, pool="mean", device="cpu",
           chunk=32, n_channels=3, normalize=True):
    """SeisLM features for a batch of traces.

    Parameters
    ----------
    layer : None (final hidden state) or int, 0-based transformer layer index
        (0 = output of first transformer block), matching
        SeisLMFeatureLoss.feature_layers.
    pool : "mean" -> [B, D];  "none" -> [B, L, D].

    Notes
    -----
    layer=-1 returns the pre-transformer (convolutional/projection) features,
    i.e. hidden_states[0].
    """
    x = standardize(traces_bt) if normalize else torch.as_tensor(
        traces_bt, dtype=torch.float32)
    outs = []
    for s in range(0, x.shape[0], chunk):
        xb = x[s:s + chunk].unsqueeze(1).repeat(1, n_channels, 1).to(device)
        need_hidden = layer is not None
        out = model.wav2vec2(xb, output_hidden_states=need_hidden)
        if layer is None:
            z = out.last_hidden_state
        else:
            z = out.hidden_states[layer + 1]  # hidden_states[0] = embeddings
        if pool == "mean":
            z = z.mean(dim=1)
        outs.append(z.cpu())
    return torch.cat(outs, dim=0)


def feature_distance(model, a_bt, b_bt, **kw):
    """Mean squared distance between features of two trace batches (scalar)."""
    za = encode(model, a_bt, **kw)
    zb = encode(model, b_bt, **kw)
    return float(((za - zb) ** 2).mean().item())


def num_layers(model):
    return len(model.wav2vec2.encoder.layers)


# ---------------------------------------------------------------------------
# Trace perturbations (all operate on [B, T] numpy arrays)
# ---------------------------------------------------------------------------

def time_shift(traces, shift_samples):
    """Non-circular (zero-padded) fractional time shift via spline
    interpolation. Preferred: no wrap-around, so no spurious exact
    invariances of pooled features under whole-token circular rolls."""
    from scipy.ndimage import shift as ndshift
    return ndshift(traces, (0, shift_samples), order=3, mode="constant",
                   cval=0.0).astype(np.float32)


def time_shift_periodic(traces, shift_samples):
    """Fourier-domain (fractional) time shift, periodic. WARNING: circular
    shifts by whole-token multiples (stride-4 samples) leave mean-pooled
    encoder features exactly unchanged; use time_shift for sensitivity
    studies."""
    T = traces.shape[-1]
    freqs = np.fft.rfftfreq(T)
    phase = np.exp(-2j * np.pi * freqs * shift_samples)
    return np.fft.irfft(np.fft.rfft(traces, axis=-1) * phase, n=T,
                        axis=-1).astype(np.float32)


def amplitude_scale(traces, factor):
    return (traces * factor).astype(np.float32)


def time_varying_gain(traces, beta, dt):
    """Exponential gain exp(beta * t). NOT removed by per-trace
    standardization (unlike a constant amplitude factor)."""
    t = np.arange(traces.shape[-1]) * dt
    return (traces * np.exp(beta * t)[None, :]).astype(np.float32)


def polarity_flip(traces):
    return (-traces).astype(np.float32)


def phase_rotation(traces, angle_deg):
    """Constant phase rotation via Hilbert transform."""
    from scipy.signal import hilbert
    ang = np.deg2rad(angle_deg)
    analytic = hilbert(traces, axis=-1)
    return (np.cos(ang) * traces
            - np.sin(ang) * np.imag(analytic)).astype(np.float32)


def bandpass(traces, dt, fmin=None, fmax=None, order=4):
    from scipy.signal import butter, sosfiltfilt
    fs = 1.0 / dt
    if fmin is not None and fmax is not None:
        sos = butter(order, [fmin, fmax], btype="band", fs=fs, output="sos")
    elif fmax is not None:
        sos = butter(order, fmax, btype="low", fs=fs, output="sos")
    else:
        sos = butter(order, fmin, btype="high", fs=fs, output="sos")
    return sosfiltfilt(sos, traces, axis=-1).astype(np.float32)


def add_noise(traces, snr_db, rng):
    p_sig = np.mean(traces ** 2, axis=-1, keepdims=True)
    p_noise = p_sig / (10 ** (snr_db / 10.0))
    noise = rng.standard_normal(traces.shape) * np.sqrt(p_noise)
    return (traces + noise).astype(np.float32)


def ricker(t, f0, t0):
    a = (np.pi * f0 * (t - t0)) ** 2
    return ((1 - 2 * a) * np.exp(-a)).astype(np.float32)


# ---------------------------------------------------------------------------
# Attention extraction (hook-based; output_attentions is not supported by the
# SDPA attention in fm_fwi, so we recompute softmax(q k^T / sqrt(d)) from the
# layer's own q/k projections)
# ---------------------------------------------------------------------------

@torch.no_grad()
def attention_maps(model, traces_bt, device="cpu", n_channels=3):
    """Return {layer_idx: [B, n_heads, L, L]} attention probabilities."""
    from fm_fwi.model import position_embedding

    maps = {}
    hooks = []

    def make_hook(idx):
        def hook(module, args, kwargs, output):
            hs = args[0] if args else kwargs["hidden_states"]
            b, t, _ = hs.shape
            q = module._shape(module.q_proj(hs), -1, b)
            k = module._shape(module.k_proj(hs), -1, b)
            if module.rotary_pos_embed:
                fc = module.freqs_cis.to(hs.device)[:t]
                q, k = position_embedding.apply_rotary_emb(
                    q.transpose(1, 2), k.transpose(1, 2), freqs_cis=fc)
                q, k = q.transpose(1, 2), k.transpose(1, 2)
            w = torch.softmax((q @ k.transpose(-1, -2)) * module.scaling,
                              dim=-1)
            maps[idx] = w.detach().cpu()
        return hook

    for i, lyr in enumerate(model.wav2vec2.encoder.layers):
        hooks.append(lyr.attention.register_forward_hook(
            make_hook(i), with_kwargs=True))
    try:
        x = standardize(traces_bt).unsqueeze(1).repeat(1, n_channels, 1)
        model.wav2vec2(x.to(device))
    finally:
        for h in hooks:
            h.remove()
    return maps
