# Data

`data/Marm.bin` is tracked. The Overthrust model and the SeisLM checkpoint are not (too large); place them as follows
before running the drivers in `scripts/`:

- **SeisLM pretrained checkpoint** `pretrained_seislm_base/checkpoints/epoch=39-step=1203000.ckpt` (at the repository
  root, next to `data/`): download it from the link in the
  [seisLM README](https://github.com/liutianlin0121/seisLM#pretrained-models).
- **Marmousi velocity model** `data/Marm.bin`: 221 x 601 float32 values in km/s (the 601 lateral samples vary
  fastest), 15 m spacing; the original Marmousi model (Brougois et al., 1990, not Marmousi2), included in the
  repository.
- **SEG/EAGE Overthrust model** `data/overthrust.vites`: from the SEG/EAGE Salt and Overthrust Models dataset
  (https://wiki.seg.org/wiki/SEG/EAGE_Salt_and_Overthrust_Models), file
  `salt_and_overthrust_models/3-D_Overthrust_Model_Disk1/3D-Velocity-Grid/overthrust.vites` of the archive
  (187 x 801 x 801 big-endian float32 values in m/s). The drivers cut the 2D section and the 3D subvolume from it.
