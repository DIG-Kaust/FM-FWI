"""Time-shift sensitivity of the feature distance, layer by layer (paper Figure 4, SI Figure S7).

Repeats the paper's 1D two-Ricker time-shift experiment using features from
each transformer block (and the pre-transformer embedding), pretrained vs.
random. Shows WHERE along the depth the cycle-skipping-robust behavior
emerges, justifying the use of the last hidden state. No Devito required.

    python scripts/layerwise/layerwise_sensitivity.py --source marmousi --pool none   # and --pool mean

Writes runs/diagnostics/layerwise/layerwise_<source>_pool<pool>.npz (inputs of fig4_layerwise.py, figS7_layer_mean.py).
"""

import argparse
import os
import numpy as np
import torch

import common


def toy_traces(nt, dt, f0=3.0, shift_s=0.0):
    """Two-Ricker toy trace with both events shifted by shift_s seconds.

    Events are REGENERATED at shifted arrival times (as in the paper's 1D
    experiment) rather than circularly shifting the trace: circular rolls by
    whole-token multiples leave mean-pooled features exactly unchanged, which
    produces artifactual periodic zeros in the sensitivity curves.
    """
    t = np.arange(nt) * dt
    ref = (common.ricker(t, f0, 0.35 * t[-1] + shift_s)
           + common.ricker(t, f0, 0.6 * t[-1] + shift_s))
    return ref[None, :].astype(np.float32)          # [1, nt]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["toy", "marmousi"], default="toy")
    ap.add_argument("--f0", type=float, default=3.0, help="toy Ricker freq")
    ap.add_argument("--n-traces", type=int, default=32)
    ap.add_argument("--pool", choices=["mean", "none"], default="mean",
                    help="mean matches the FWI loss; none matches Eq. 5 "
                         "of the paper (unpooled feature matrices)")
    ap.add_argument("--ckpt", default=common.DEFAULT_CKPT)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp = os.path.join(common.HERE, "layerwise")
    os.makedirs(exp, exist_ok=True)

    if args.source == "toy":
        nt, dt = common.PAR["nt"], common.PAR["dt"]
        ref = toy_traces(nt, dt, args.f0)
        period = 1.0 / args.f0
        def shifted(sh):
            return toy_traces(nt, dt, args.f0, shift_s=sh)
    else:
        _, dobs_masked, nt, dt = common.observed_data()
        gather = dobs_masked[common.PAR["ns"] // 2]
        idx = np.linspace(0, common.PAR["nr"] - 1, args.n_traces).astype(int)
        ref = gather[:, idx].T.astype(np.float32)
        period = 1.0 / common.PAR["freq"]
        def shifted(sh):
            return common.time_shift(ref, sh / dt)   # non-circular

    shifts_s = np.linspace(-2.5 * period, 2.5 * period, 51)
    perts = [shifted(sh) for sh in shifts_s]

    results = {}
    for kind in ("pretrained", "random"):
        model = common.get_seislm(kind, args.ckpt, device)
        layers = [-1] + list(range(common.num_layers(model)))  # -1 = conv/proj
        curves = np.zeros((len(layers), len(shifts_s)))
        for li, layer in enumerate(layers):
            for si, pert in enumerate(perts):
                curves[li, si] = common.feature_distance(
                    model, ref, pert, layer=layer, pool=args.pool,
                    device=device)
            curves[li] /= curves[li].max() or 1.0
            print(f"{kind} layer {layer}: done", flush=True)
        results[kind] = (layers, curves)
        del model

    # data-space L2 reference curve
    l2c = np.array([float(np.mean(
        (common.standardize(ref).numpy()
         - common.standardize(pert).numpy()) ** 2))
        for pert in perts])
    l2c /= l2c.max()

    tag = f"{args.source}_pool{args.pool}"
    np.savez(f"{exp}/layerwise_{tag}.npz", shifts_s=shifts_s, l2=l2c,
             **{f"{k}_curves": v[1] for k, v in results.items()},
             layers=np.array(results["pretrained"][0]))

    import matplotlib.pyplot as plt
    from fm_fwi.plotting import apply_plot_style
    apply_plot_style()
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, kind in zip(axs, ("pretrained", "random")):
        layers, curves = results[kind]
        cmap = plt.cm.viridis(np.linspace(0, 1, len(layers)))
        for li, layer in enumerate(layers):
            lbl = "conv/proj" if layer == -1 else f"layer {layer}"
            ax.plot(shifts_s / period, curves[li], color=cmap[li], label=lbl)
        ax.plot(shifts_s / period, l2c, "k:", lw=2, label="L2 (data)")
        ax.set_title(kind)
        ax.set_xlabel("time shift [dominant periods]")
        ax.legend(fontsize=8, ncol=2)
    axs[0].set_ylabel("normalized feature distance")
    fig.suptitle(f"source={args.source}, pool={args.pool}")
    fig.savefig(f"{exp}/layerwise_{tag}.png", dpi=300, bbox_inches="tight")
    print("done ->", exp)


if __name__ == "__main__":
    main()
