# Background accuracy of stage-1 models (SI Table S3): RMS relative error between the stage-1 model and the true
# model after smoothing both with a Gaussian of 500 m standard deviation, below the water layer (2D) or in the 3D
# metric window. Reproduces the values logged on 29 Oct (25 m: 0.076 / 0.084; 37.5 m: 0.021 / 0.058).
# Usage: python background_accuracy.py   (reads runs/<name>/{vp_inv.npy,run.json} and runs/3d/<name>/vp_current.npy)
import json, os
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

H = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("FMFWI_RUNS", os.path.join(H, "../runs"))
VP = np.fromfile(os.path.join(H, "../data/overthrust.vites"), dtype=">f4").astype(np.float32).reshape(187, 801, 801) / 1000


def true2d(dz):
    nwater = int(round(0.5 / dz))
    nx, nzb = int(round(12.5 / dz)) + 1, int(round(4.65 / dz)) + 1
    sec = VP[:, 455, :501].T.astype(np.float32)                    # as overthrust2d.build_problem
    if (nx, nzb) != sec.shape:
        r = dz / 0.025
        ix, iz = np.meshgrid(np.arange(nx) * r, np.arange(nzb) * r, indexing="ij")
        sec = map_coordinates(sec, [ix, iz], order=1, mode="nearest").astype(np.float32)
    return np.concatenate([np.full((nx, nwater), 1.5, np.float32), sec], axis=1), nwater


def bg2d(name):
    for f in (os.path.join(R, name),):
        if os.path.exists(os.path.join(f, "vp_inv.npy")) and os.path.exists(os.path.join(f, "run.json")):
            dz = json.load(open(os.path.join(f, "run.json")))["config"].get("dx", 0.025)
            t, nw = true2d(dz)
            a, b = gaussian_filter(np.load(os.path.join(f, "vp_inv.npy")), 0.5 / dz), gaussian_filter(t, 0.5 / dz)
            return float(np.sqrt(np.mean(((a - b) / b)[:, nw:] ** 2)))
    return None


def bg3d(run):
    t = np.load(os.path.join(R, "3d", run, "vp_true.npy"))
    m = np.load(os.path.join(R, "3d", run, "vp_current.npy"))
    a, b = gaussian_filter(m, 10), gaussian_filter(t, 10)            # 50 m grid: 500 m = 10 cells
    w = np.zeros(t.shape, bool)                                     # metric window: 0.5 km lateral, 0.1 km top, 0.5 km bottom
    w[10:-10, 10:-10, 2:-10] = True
    return float(np.sqrt(np.mean(((a - b) / b)[w] ** 2)))


if __name__ == "__main__":
    pairs = [("2D example", "ref_feature", "pool_mean"), ("2D example, run 2", "ref_feature_r2", "pool_mean_r2"),
             ("2D example, run 3", "ref_feature_r3", "pool_mean_r3"), ("Grid 37.5 m", "grid37_feature", "grid37_pool_mean"),
             ("Grid 50 m", "grid50_feature", "grid50_pool_mean"), ("Shots 1 km, receivers 50 m", "survey3d_feature", "survey3d_pool_mean"),
             ("Shots 1 km, receivers 125 m", "s1000r125_feature", "s1000r125_pool_mean"),
             ("Shots 1.29 km, receivers 50 m", "s1290r50_feature", "s1290r50_pool_mean"),
             ("Sparse, run 1", "surveysparse_feature", "surveysparse_pool_mean"),
             ("Sparse, run 2", "surveysparse_feature_r2", "surveysparse_pool_mean_r2"),
             ("Sparse, run 3", "surveysparse_feature_r3", "surveysparse_pool_mean_r3"),
             ("Mask offset 0.2 s", "toff02_feature", "toff02_pool_mean"), ("Mask offset 0.6 s", "toff06_feature", "toff06_pool_mean"),
             ("Mask offset 0.8 s", "toff08_feature", "toff08_pool_mean"), ("3D sampling", "match3d_feature", "match3d_pool_mean"),
             ("3D start", "init3d_feature", "init3d_pool_mean"), ("3D start, 3D sampling", "init3d_match3d_feature", "init3d_match3d_pool_mean")]
    print(f"{'3D example':32s} full {bg3d('overthrust_feat_water_linear_init_pool_none100'):.3f}"
          f"  mean {bg3d('overthrust_feat_linear_init'):.3f}")
    for lab, f, m in pairs:
        a, b = bg2d(f), bg2d(m)
        print(f"{lab:32s} full {a if a is None else round(a, 3)}  mean {b if b is None else round(b, 3)}")
