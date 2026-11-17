"""1D two-Ricker time-shift experiment of the paper (Fig 3).

3 Hz Ricker wavelets at 1/3 and 2/3 of a 4107-sample trace, dt 3 ms, each shifted independently
over +-2.5 dominant periods in steps of 4 samples (137 x 137 grid), three-channel copy of the trace,
standardized per trace as in the inversion loss. Saves, for every grid point, the L2 misfit and the
squared feature distances to the unshifted trace of the pretrained and a randomly initialized (seed 0)
encoder: every layer, full sequence and time-averaged ([layers, 2, grid points]).

    python scripts/boa1d.py --out runs/diagnostics/boa1d      # GPU; then paper_figs/fig3_twoshift.py
"""

import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO, "pretrained_seislm_base/checkpoints/epoch=39-step=1203000.ckpt")
F0, DT, NT, STRIDE = 3.0, 0.003, 4107, 4
C1, C2 = NT // 3, 2 * NT // 3


def ricker(center):
    u = (np.pi * F0 * (np.arange(NT) - center) * DT) ** 2
    return (1 - 2 * u) * np.exp(-u)


def shift(x, s):
    out = np.zeros_like(x)
    if s >= 0:
        out[..., s:] = x[..., :x.shape[-1] - s]
    else:
        out[..., :s] = x[..., -s:]
    return out


def traces():
    w1, w2 = ricker(C1), ricker(C2)
    ref = w1 + w2
    smax = int((2.5 / F0) / DT) // STRIDE * STRIDE  # symmetric grid that contains zero shift
    sh = np.arange(-smax, smax + 1, STRIDE)
    grid = (np.stack([shift(w1, s) for s in sh])[:, None] + np.stack([shift(w2, s) for s in sh])[None])
    return ref, sh, grid.reshape(-1, NT)


def compute():
    import torch
    sys.path.insert(0, REPO)
    from fm_fwi.model.pretrained import load_pretrained_multidim_wav2vec2, load_random_multidim_wav2vec2
    dev = "cuda"
    ref, sh, grid = traces()
    t0 = time.time()
    print(f"{len(sh)} shifts, {len(grid)} grid traces", flush=True)
    res = {"shifts_s": sh * DT, "dt": DT}

    def prep(x):
        x = torch.as_tensor(x, dtype=torch.float32, device=dev)[:, None].repeat(1, 3, 1)
        return (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True, unbiased=False).clamp_min(1e-10)

    @torch.no_grad()
    def dists(model, x, chunk=64):
        """Per-layer [full, mean-pooled] squared distances of every trace in x to ref: [L, 2, N]."""
        enc = lambda t: model.wav2vec2(prep(t), output_hidden_states=True).hidden_states
        zr = enc(ref[None])
        out = []
        for s in range(0, len(x), chunk):
            z = enc(x[s:s + chunk])
            out.append(torch.stack([torch.stack([((a - b) ** 2).mean((1, 2)), ((a.mean(1) - b.mean(1)) ** 2).mean(1)])
                                    for a, b in zip(z, zr)]).cpu())
        return torch.cat(out, -1).numpy()

    for name, load in (("pretrained", lambda: load_pretrained_multidim_wav2vec2(CKPT, map_location=dev)),
                       ("random0", lambda: load_random_multidim_wav2vec2(CKPT, map_location=dev, seed=0))):
        m = load().to(dev).eval()
        res[f"{name}_std_grid"] = dists(m, grid)    # [L, 2, N*N]
        print(name, f"done {time.time() - t0:.0f} s", flush=True)
        del m
        torch.cuda.empty_cache()
    res["bl_l2_grid"] = 0.5 * ((grid - ref) ** 2).sum(-1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "runs/diagnostics/boa1d"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "boa1d.npz"), **compute())
    print("saved", args.out)
