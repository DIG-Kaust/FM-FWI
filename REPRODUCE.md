# Reproducing the paper

Commands for every result of the revised manuscript (JGR: Machine Learning and Computation, 2026JH001637). Run them
from the repository root after installing the environment (README). Each 2D run writes `runs/<name>/run.json`
(configuration, software versions, optimizer messages, losses and errors per accepted iteration) and the models
(`vp_inv.npy` after stage 1, `vp_inv_refl.npy` after stage 2). The figure scripts read `runs/`, or the folder in
`$FMFWI_RUNS`, and take the output folder as their argument, e.g. `python scripts/paper_figs/fig10_11_overthrust2d.py figs/`.

`overthrust2d.py` runs stage 1 (100 L-BFGS-B iterations of the stated objective on the masked data) and stage 2 (100
iterations of L2 on the full data) in one call; `--stage 1` or `--stage 2` runs one stage (stage 2 reads the stage-1
model from the same folder). GPU computations of the feature loss are not bitwise reproducible, so repeated runs agree
to within the run-to-run spread reported in the paper (Text S7), not exactly.

## Figure index

Each figure of the paper and its Supporting Information, the script that draws it and the PNG it writes (the file name
used in the manuscript source). `nohybrid` draws the main-text version (L2 and SeisLM); the same script without it
draws the SI version with the hybrid. Commands and the runs each script reads are in the sections below.

| Figure | Script (`scripts/paper_figs/`) | Output |
|---|---|---|
| 1, 2 (SeisLM architecture, workflow) | none: drawings | |
| 3 | `fig3_twoshift.py <boa1d.npz> <out.png> std sqrt` | `fig_3_BoA_2D.png` |
| 4 | `fig4_layerwise.py` (see the note below) | `fig_layerwise_timeshift.png` |
| 5, 6 | `fig5_6_marmousi_side.py` | `fig_4_marm.png`, `fig_5_grad.png` |
| 7 | `fig7_marmousi_convergence.py <out> nohybrid` | `fig_6_perf.png` |
| 8 | `fig8_marmousi_models.py <out> nohybrid` | `fig_7_marm_inv.png` |
| 9 (2D Overthrust model) | none: unchanged from the submitted paper | `fig_2d_ovth2.png` |
| 10, 11 | `fig10_11_overthrust2d.py <out> nohybrid` | `fig10_proposed.png`, `fig11_proposed.png` |
| 12 | `fig12_baselines.py` | `fig_baselines.png` (and `figS5_baseline_models.png`) |
| 13 | `fig13_noise.py` | `fig_noise.png` |
| 14, 15 (3D Overthrust model and slices) | none: unchanged from the submitted paper | `fig_3d_ovth.png`, `fig_3d_ovth_slices.png` |
| 16, 17 | `fig16_17_overthrust3d.py` | `fig_16_ovth_3d_perf.png`, `fig_3d_ovth_inv_lin.png` |
| 18 | `fig18_pooling.py` | `fig_pooling.png` |
| S1, S2 | `figS1_S2_encoder.py` | `figS1_encoder_convergence.png`, `figS2_encoder_models.png` |
| S3 | `figS3_init.py` | `figS3_init_tolerance.png` |
| S4 | `figS4_sampling.py` | `figS4_sampling_window.png` |
| S5 | `fig12_baselines.py` | `figS5_baseline_models.png` |
| S6 | `figS6_smooth_baselines.py` | `figS6_smooth_baselines.png` |
| S7, S8 | `fig7_marmousi_convergence.py <out>`, `fig8_marmousi_models.py <out>` | `fig_6_perf.png`, `fig_7_marm_inv.png` (paper: `figS_hybrid_marm_{perf,inv}.png`) |
| S9, S10 | `fig10_11_overthrust2d.py <out>` | `fig10_proposed.png`, `fig11_proposed.png` (paper: `figS_hybrid_ovth_{perf,inv}.png`) |
| S11 | `figS6_hybrid_terms.py` | `figS6_hybrid_terms.png` |
| S12 | `figS7_layer_mean.py` | `figS7_layer_mean.png` |
| S13 | `figS8_pooling_3d_models.py` | `figS8_pooling_3d_models.png` |
| S14 | `figS9_noise_models.py` | `figS9_noise_models.png` |

From the run outputs of the paper, every scripted figure above re-renders the published PNG pixel for pixel, except:
- Figure 4: the plotting script of the submitted figure was not kept. `fig4_layerwise.py` draws the same curves from
  the same data in a close style.
- Figures 5-8 and S7-S8: the scripts read the outputs of `marm2d.py`. The paper's Marmousi panels come from the
  original notebook runs that `marm2d.py` reproduces to within GPU run-to-run variation (see Marmousi below), so
  the script versions match their content but not their pixels. Figure S8 was assembled from the notebook renders
  of each model, and Figure 8 is Figure S8 with the hybrid row cropped out. Figures 5-6 re-render the published
  PNGs pixel for pixel when `grad.npy` comes from the notebook runs.

## Diagnostics (Figures 3, 4, 18a, S4, S12; panel (a) of Figure 12)

```
python scripts/boa1d.py --out runs/diagnostics/boa1d                 # Figure 3 data
python scripts/paper_figs/fig3_twoshift.py runs/diagnostics/boa1d/boa1d.npz figs/fig_3_BoA_2D.png std sqrt
python scripts/layerwise/layerwise_sensitivity.py --source marmousi --pool none   # Figure 4 and Figure S12
python scripts/layerwise/layerwise_sensitivity.py --source marmousi --pool mean
python scripts/paper_figs/fig4_layerwise.py figs/                # Figure 4 ("mean" adds the time-averaged curve)
python scripts/shift_diagnostics.py rigid --out runs/diagnostics      # Figure 18a
python scripts/shift_diagnostics.py sampling --out runs/diagnostics   # Figure S4
python scripts/shift_diagnostics.py window --out runs/diagnostics
python scripts/paper_figs/figS4_sampling.py figs/                # Figure S4
python scripts/shift_diagnostics.py ricker --out runs/diagnostics     # Figure 12a
```

## Marmousi (Figures 5-8)

```
for loss in l2 feature hybrid; do python scripts/marm2d.py --loss $loss --out runs/marm --name marm_${loss}_rerun; done
python scripts/paper_figs/fig5_6_marmousi_side.py figs/                  # Figures 5-6 (model, first gradients)
python scripts/paper_figs/fig7_marmousi_convergence.py figs/ nohybrid    # Figure 7
python scripts/paper_figs/fig8_marmousi_models.py figs/ nohybrid         # Figure 8
python scripts/paper_figs/fig7_marmousi_convergence.py figs/si/          # Figure S7 (with the hybrid)
python scripts/paper_figs/fig8_marmousi_models.py figs/si/               # Figure S8
```
`marm2d.py` is the script form of the original notebook: same model, acquisition, 7-Hz Ricker, mask (direct arrival
+ 0.35 s), L-BFGS-B settings and hybrid schedule. Figure 8 of the paper shows the models of the July 2026 notebook
runs; the script reproduces them to within GPU run-to-run variation (stage-2 error 0.0797 against 0.0790 for SeisLM).
The L2 panels and curves come from an earlier run of the same setup (April 2026, identical first iteration). The L2
inversions stop when the line search no longer finds a decrease, and that point moves between runs (stage 1 stopped
after 56 iterations in that run and after 71 in a July rerun), so a rerun gives a similar but not identical L2 curve.

## 2D Overthrust (Figures 9-11)

```
python scripts/overthrust2d.py --out runs --name ref_l2 --loss l2
python scripts/overthrust2d.py --out runs --name ref_feature --loss feature
python scripts/overthrust2d.py --out runs --name ref_hybrid --loss hybrid
python scripts/paper_figs/fig10_11_overthrust2d.py figs/ nohybrid   # Figures 10-11
python scripts/paper_figs/fig10_11_overthrust2d.py figs/si/          # Figures S9-S10 (with the hybrid)
```

## Established misfits (Figure 12, Table 2, Text S4, Figures S5-S6)

```
python scripts/overthrust2d.py --out runs --name bl_envelope_p1 --loss envelope --env-power 1.0
python scripts/overthrust2d.py --out runs --name bl_envelope_p1_sm20 --loss envelope --init smooth --init-sigma 20.0 16.0 --env-power 1.0
python scripts/overthrust2d.py --out runs --name bl_envelope_p1_sm50 --loss envelope --init smooth --env-power 1.0
python scripts/overthrust2d.py --out runs --name bl_multiscale_pad --loss multiscale
python scripts/overthrust2d.py --out runs --name bl_multiscale_pad_sm50 --loss multiscale --init smooth
python scripts/overthrust2d.py --out runs --name bl_stdl2 --loss stdl2
python scripts/overthrust2d.py --out runs --name bl_stdl2_sm50 --loss stdl2 --init smooth
python scripts/overthrust2d.py --out runs --name bl_w2_exact --loss w2 --w2-shift 0.75
python scripts/overthrust2d.py --out runs --name bl_w2_exact_sm50 --loss w2 --init smooth --w2-shift 0.75
python scripts/overthrust2d.py --out runs --name sdtw_g30_linear --loss softdtw --gamma 30.0
python scripts/overthrust2d.py --out runs --name sdtw_g30_sm20 --loss softdtw --init smooth --init-sigma 20.0 16.0 --gamma 30.0
python scripts/overthrust2d.py --out runs --name sdtw_g30_sm50 --loss softdtw --init smooth --gamma 30.0
python scripts/paper_figs/fig12_baselines.py figs/              # Figure 12 and Figure S5
python scripts/paper_figs/figS6_smooth_baselines.py figs/       # Figure S6
```
Figure S6 also reads `init_sm50_l2` and `init_sm50_feature` (Text S2 below). `--init smooth` without `--init-sigma`
is the 1.25-km smoothing (sigma 50, 40 cells).

## Noise (Figure 13, Table 3, Text S10, Figure S14)

```
python scripts/overthrust2d.py --out runs --name nmute_flat10_feature --loss feature --noise-snr 10.0 --noise-type flat --noise-premute 0.001
python scripts/overthrust2d.py --out runs --name nmute_flat10_l2 --loss l2 --noise-snr 10.0 --noise-type flat --noise-premute 0.001
python scripts/overthrust2d.py --out runs --name nmute_flat20_feature --loss feature --noise-snr 20.0 --noise-type flat --noise-premute 0.001
python scripts/overthrust2d.py --out runs --name nmute_flat20_l2 --loss l2 --noise-snr 20.0 --noise-type flat --noise-premute 0.001
python scripts/overthrust2d.py --out runs --name nmute_flat5_feature --loss feature --noise-snr 5.0 --noise-type flat --noise-premute 0.001
python scripts/overthrust2d.py --out runs --name nmute_flat5_l2 --loss l2 --noise-snr 5.0 --noise-type flat --noise-premute 0.001
python scripts/overthrust2d.py --out runs --name nmute_white20_feature --loss feature --noise-snr 20.0 --noise-type white --noise-premute 0.001
python scripts/overthrust2d.py --out runs --name nmute_white20_l2 --loss l2 --noise-snr 20.0 --noise-type white --noise-premute 0.001
python scripts/overthrust2d.py --out runs --name noise_flat10_feature --loss feature --noise-snr 10.0 --noise-type flat
python scripts/overthrust2d.py --out runs --name noise_flat10_l2 --loss l2 --noise-snr 10.0 --noise-type flat
python scripts/overthrust2d.py --out runs --name noise_flat20_feature --loss feature --noise-snr 20.0 --noise-type flat
python scripts/overthrust2d.py --out runs --name noise_flat20_l2 --loss l2 --noise-snr 20.0 --noise-type flat
python scripts/overthrust2d.py --out runs --name noise_flat5_feature --loss feature --noise-snr 5.0 --noise-type flat
python scripts/overthrust2d.py --out runs --name noise_flat5_l2 --loss l2 --noise-snr 5.0 --noise-type flat
python scripts/overthrust2d.py --out runs --name noise_white20_feature --loss feature --noise-snr 20.0 --noise-type white
python scripts/overthrust2d.py --out runs --name noise_white20_l2 --loss l2 --noise-snr 20.0 --noise-type white
python scripts/overthrust2d.py --out runs --name noise_white10_feature --loss feature --noise-snr 10.0 --noise-type white
python scripts/overthrust2d.py --out runs --name nmute_white10_feature_e --loss feature --noise-snr 10.0 --noise-type white --noise-premute 0.001
python scripts/overthrust2d.py --out runs --name noise_white10_l2 --loss l2 --noise-snr 10.0 --noise-type white
python scripts/overthrust2d.py --out runs --name nmute_white10_l2 --loss l2 --noise-snr 10.0 --noise-type white --noise-premute 0.001
python scripts/paper_figs/fig13_noise.py figs/
python scripts/paper_figs/figS9_noise_models.py figs/
```

## 3D Overthrust (Figures 14-17)

The 3D driver needs a GPU for the feature loss; the L2 runs need only CPUs. Stage 2 is two L-BFGS-B runs of 100 and
200 iterations (the second restarts from the model of the first).
```
python scripts/overthrust3d.py --loss feature --linear_init --maxiter 100 --toff 0.5 --exp_name overthrust_feat_linear_init
python scripts/overthrust3d.py --loss l2 --no_mask --maxiter 100 --init_model runs/3d/overthrust_feat_linear_init/vp_inv.npy --exp_name overthrust_feat_linear_init_refl
python scripts/overthrust3d.py --loss l2 --no_mask --maxiter 200 --init_model runs/3d/overthrust_feat_linear_init_refl/vp_inv.npy --exp_name overthrust_feat_linear_init_refl2
python scripts/overthrust3d.py --loss l2 --linear_init --maxiter 100 --toff 0.5 --exp_name overthrust_l2mask_linear_init
python scripts/overthrust3d.py --loss l2 --no_mask --maxiter 100 --init_model runs/3d/overthrust_l2mask_linear_init/vp_inv.npy --exp_name overthrust_l2mask_linear_init_refl
python scripts/overthrust3d.py --loss l2 --no_mask --maxiter 200 --init_model runs/3d/overthrust_l2mask_linear_init_refl/vp_inv.npy --exp_name overthrust_l2mask_linear_init_refl2
python scripts/paper_figs/fig16_17_overthrust3d.py figs/
```
Figures 16-17 show the L2 workflow `overthrust_l2mask_linear_init*` (stage 1 masks both the observed and the
modeled data) and the SeisLM workflow `overthrust_feat_linear_init*`. Every 3D run uses `--sigma 30 30 20 --grad_sigma 4 4 2 --water_depth_m 500 --water_velocity 1.5` (the defaults).

## Full versus time-averaged features (Figure 18, Text S6, Table S3, Figures S12-S13)

```
python scripts/overthrust2d.py --out runs --name pool_mean --loss feature --pool mean
python scripts/overthrust2d.py --out runs --name pool_mean_r2 --loss feature --pool mean
python scripts/overthrust2d.py --out runs --name pool_mean_r3 --loss feature --pool mean
python scripts/overthrust2d.py --out runs --name ref_feature_r2 --loss feature
python scripts/overthrust2d.py --out runs --name ref_feature_r3 --loss feature
python scripts/overthrust2d.py --out runs --name grid37_feature --loss feature --dx 0.0375
python scripts/overthrust2d.py --out runs --name grid37_pool_mean --loss feature --dx 0.0375 --pool mean
python scripts/overthrust2d.py --out runs --name grid50_feature --loss feature --dx 0.05
python scripts/overthrust2d.py --out runs --name grid50_pool_mean --loss feature --dx 0.05 --pool mean
python scripts/overthrust2d.py --out runs --name init3d_feature --loss feature --init linear3d
python scripts/overthrust2d.py --out runs --name init3d_match3d_feature --loss feature --acq sparse3d --dx 0.05 --init linear3d --toff 0.5 --tmax 5.0 --dt 0.0037735849
python scripts/overthrust2d.py --out runs --name init3d_match3d_pool_mean --loss feature --acq sparse3d --dx 0.05 --init linear3d --pool mean --toff 0.5 --tmax 5.0 --dt 0.0037735849
python scripts/overthrust2d.py --out runs --name init3d_pool_mean --loss feature --init linear3d --pool mean
python scripts/overthrust2d.py --out runs --name match3d_feature --loss feature --acq sparse3d --dx 0.05 --toff 0.5 --tmax 5.0 --dt 0.0037735849
python scripts/overthrust2d.py --out runs --name match3d_pool_mean --loss feature --acq sparse3d --dx 0.05 --pool mean --toff 0.5 --tmax 5.0 --dt 0.0037735849
python scripts/overthrust2d.py --out runs --name s1000r125_feature --loss feature --acq s1000_r125
python scripts/overthrust2d.py --out runs --name s1000r125_pool_mean --loss feature --acq s1000_r125 --pool mean
python scripts/overthrust2d.py --out runs --name s1290r50_feature --loss feature --acq s1290_r50
python scripts/overthrust2d.py --out runs --name s1290r50_pool_mean --loss feature --acq s1290_r50 --pool mean
python scripts/overthrust2d.py --out runs --name survey3d_feature --loss feature --acq sparse3d
python scripts/overthrust2d.py --out runs --name survey3d_pool_mean --loss feature --acq sparse3d --pool mean
python scripts/overthrust2d.py --out runs --name toff02_feature --loss feature --toff 0.2
python scripts/overthrust2d.py --out runs --name toff02_pool_mean --loss feature --pool mean --toff 0.2
python scripts/overthrust2d.py --out runs --name toff06_feature --loss feature --toff 0.6
python scripts/overthrust2d.py --out runs --name toff06_pool_mean --loss feature --pool mean --toff 0.6
python scripts/overthrust2d.py --out runs --name toff08_feature --loss feature --toff 0.8
python scripts/overthrust2d.py --out runs --name toff08_pool_mean --loss feature --pool mean --toff 0.8
python scripts/overthrust2d.py --out runs --name surveysparse_feature --loss feature --acq sparse
python scripts/overthrust2d.py --out runs --name surveysparse_feature_r2 --loss feature --acq sparse
python scripts/overthrust2d.py --out runs --name surveysparse_feature_r3 --loss feature --acq sparse
python scripts/overthrust2d.py --out runs --name surveysparse_pool_mean --loss feature --acq sparse --pool mean
python scripts/overthrust2d.py --out runs --name surveysparse_pool_mean_r2 --loss feature --acq sparse --pool mean
python scripts/overthrust2d.py --out runs --name surveysparse_pool_mean_r3 --loss feature --acq sparse --pool mean
python scripts/overthrust3d.py --loss feature --pool none --no_cache_obs_embeddings --trace_chunk 512 --sigma 10 10 5 --linear_init --maxiter 100 --toff 0.5 --exp_name overthrust_feat_water_linear_init_pool_none100
python scripts/overthrust3d.py --loss l2 --no_mask --maxiter 300 --init_model runs/3d/overthrust_feat_water_linear_init_pool_none100/vp_inv.npy --exp_name s2_300
python scripts/background_accuracy.py                           # background errors of Table S3
python scripts/paper_figs/fig18_pooling.py figs/
python scripts/paper_figs/figS7_layer_mean.py figs/            # Figure S12
python scripts/paper_figs/figS8_pooling_3d_models.py figs/     # Figure S13
```

## Supporting Information

Text S1, Table S1, Figures S1-S2 (encoder and feature block):
```
python scripts/overthrust2d.py --out runs --name rand_s0 --loss feature --encoder random
python scripts/overthrust2d.py --out runs --name rand_s1 --loss feature --encoder random --seed 1
python scripts/overthrust2d.py --out runs --name rand_s2 --loss feature --encoder random --seed 2
python scripts/overthrust2d.py --out runs --name layer0 --loss feature --layer 0
python scripts/overthrust2d.py --out runs --name layer3 --loss feature --layer 3
python scripts/paper_figs/figS1_S2_encoder.py figs/
```
Text S2, Table S2, Figure S3 (starting model):
```
python scripts/overthrust2d.py --out runs --name init_s0.85_feature_b --loss feature --init-scale 0.85
python scripts/overthrust2d.py --out runs --name init_s0.85_l2 --loss l2 --init-scale 0.85
python scripts/overthrust2d.py --out runs --name init_s0.90_feature --loss feature --init-scale 0.9
python scripts/overthrust2d.py --out runs --name init_s0.90_l2 --loss l2 --init-scale 0.9
python scripts/overthrust2d.py --out runs --name init_s0.95_feature --loss feature --init-scale 0.95
python scripts/overthrust2d.py --out runs --name init_s0.95_l2 --loss l2 --init-scale 0.95
python scripts/overthrust2d.py --out runs --name init_s1.05_feature --loss feature --init-scale 1.05
python scripts/overthrust2d.py --out runs --name init_s1.05_l2 --loss l2 --init-scale 1.05
python scripts/overthrust2d.py --out runs --name init_s1.10_feature --loss feature --init-scale 1.1
python scripts/overthrust2d.py --out runs --name init_s1.10_l2 --loss l2 --init-scale 1.1
python scripts/overthrust2d.py --out runs --name init_s1.15_feature_b --loss feature --init-scale 1.15
python scripts/overthrust2d.py --out runs --name init_s1.15_l2 --loss l2 --init-scale 1.15
python scripts/overthrust2d.py --out runs --name init_sm100_feature --loss feature --init smooth --init-sigma 100.0 80.0
python scripts/overthrust2d.py --out runs --name init_sm100_l2 --loss l2 --init smooth --init-sigma 100.0 80.0
python scripts/overthrust2d.py --out runs --name init_sm20_feature --loss feature --init smooth --init-sigma 20.0 16.0
python scripts/overthrust2d.py --out runs --name init_sm20_l2 --loss l2 --init smooth --init-sigma 20.0 16.0
python scripts/overthrust2d.py --out runs --name init_sm50_feature --loss feature --init smooth
python scripts/overthrust2d.py --out runs --name init_sm50_l2 --loss l2 --init smooth
python scripts/paper_figs/figS3_init.py figs/
```
Text S3 (resampling to 100 Hz):
```
python scripts/overthrust2d.py --out runs --name resample100 --loss feature --resample
```
Text S5, Figure S11 (terms of the hybrid loss; Figures S7-S10 are drawn with the Marmousi and 2D Overthrust figures):
```
python scripts/overthrust2d.py --out runs --name hyb_trace_c --loss hybrid --trace-terms 5 --maxiter2 0
python scripts/paper_figs/figS6_hybrid_terms.py figs/
```
Text S7, Table S4 (hybrid schedules and repeats):
```
python scripts/overthrust2d.py --out runs --name hyb_cos100 --loss hybrid --schedule cosine:100
python scripts/overthrust2d.py --out runs --name hyb_hold25 --loss hybrid --alpha-hold 25
python scripts/overthrust2d.py --out runs --name hyb_hold25_r2 --loss hybrid --alpha-hold 25
python scripts/overthrust2d.py --out runs --name hyb_hold25_r3 --loss hybrid --alpha-hold 25
python scripts/overthrust2d.py --out runs --name hyb_hold25_restart --loss hybrid --alpha-hold 25 --restart-every 25
python scripts/overthrust2d.py --out runs --name hyb_hold25_restart_r2 --loss hybrid --alpha-hold 25 --restart-every 25
python scripts/overthrust2d.py --out runs --name hyb_hold25_restart_r3 --loss hybrid --alpha-hold 25 --restart-every 25
python scripts/overthrust2d.py --out runs --name hyb_lin150 --loss hybrid --schedule linear:150
python scripts/overthrust2d.py --out runs --name hyb_lin50 --loss hybrid --schedule linear:50
python scripts/overthrust2d.py --out runs --name hyb_step50 --loss hybrid --schedule step:50
python scripts/overthrust2d.py --out runs --name hyb_step50_r2 --loss hybrid --schedule step:50
python scripts/overthrust2d.py --out runs --name hyb_step50_r3 --loss hybrid --schedule step:50
python scripts/overthrust2d.py --out runs --name hyb_gradscale_c --loss hybrid --hybrid-scale grad
python scripts/overthrust2d.py --out runs --name ref_hybrid_r2 --loss hybrid
python scripts/overthrust2d.py --out runs --name ref_hybrid_r3 --loss hybrid
```
Text S8, Table S5 (source wavelet, low frequencies):
```
python scripts/overthrust2d.py --out runs --name lowcut3_feature --loss feature --lowcut 3.0
python scripts/overthrust2d.py --out runs --name lowcut3_l2 --loss l2 --lowcut 3.0
python scripts/overthrust2d.py --out runs --name lowcut4_feature --loss feature --lowcut 4.0
python scripts/overthrust2d.py --out runs --name lowcut4_l2 --loss l2 --lowcut 4.0
python scripts/overthrust2d.py --out runs --name sm50_lowcut4_feature --loss feature --init smooth --lowcut 4.0
python scripts/overthrust2d.py --out runs --name sm50_lowcut4_l2 --loss l2 --init smooth --lowcut 4.0
python scripts/overthrust2d.py --out runs --name sm50_wavA55_feature --loss feature --init smooth --inv-freq 5.5 --inv-align
python scripts/overthrust2d.py --out runs --name sm50_wavA55_l2 --loss l2 --init smooth --inv-freq 5.5 --inv-align
python scripts/overthrust2d.py --out runs --name sm50_wavP30_feature --loss feature --init smooth --inv-phase 30.0
python scripts/overthrust2d.py --out runs --name sm50_wavP30_l2 --loss l2 --init smooth --inv-phase 30.0
python scripts/overthrust2d.py --out runs --name wavA55_feature --loss feature --inv-freq 5.5 --inv-align
python scripts/overthrust2d.py --out runs --name wavA55_l2 --loss l2 --inv-freq 5.5 --inv-align
python scripts/overthrust2d.py --out runs --name wavP30_feature --loss feature --inv-phase 30.0
python scripts/overthrust2d.py --out runs --name wavP30_l2 --loss l2 --inv-phase 30.0
```
Text S9, Table S6 (noise from the smoothed start):
```
python scripts/overthrust2d.py --out runs --name sm50_flat10_feature --loss feature --init smooth --noise-snr 10.0 --noise-type flat
python scripts/overthrust2d.py --out runs --name sm50_flat10_l2 --loss l2 --init smooth --noise-snr 10.0 --noise-type flat
python scripts/overthrust2d.py --out runs --name sm50_flat5_feature --loss feature --init smooth --noise-snr 5.0 --noise-type flat
python scripts/overthrust2d.py --out runs --name sm50_flat5_l2 --loss l2 --init smooth --noise-snr 5.0 --noise-type flat
python scripts/overthrust2d.py --out runs --name sm50_nmute_flat10_l2 --loss l2 --init smooth --noise-snr 10.0 --noise-type flat --noise-premute 0.001
python scripts/overthrust2d.py --out runs --name sm50_nmute_flat5_l2 --loss l2 --init smooth --noise-snr 5.0 --noise-type flat --noise-premute 0.001
python scripts/overthrust2d.py --out runs --name sm50_nmute_flat10_feature --loss feature --init smooth --noise-snr 10.0 --noise-type flat --noise-premute 0.001
python scripts/overthrust2d.py --out runs --name sm50_nmute_flat5_feature_c --loss feature --init smooth --noise-snr 5.0 --noise-type flat --noise-premute 0.001
```
