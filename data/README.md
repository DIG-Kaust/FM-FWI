# Data

This folder is not tracked in git (the files involved are too large). Place
the following here before running the notebooks:

- **SeisLM pretrained checkpoint**: download from the link in the
  [seisLM README](https://github.com/liutianlin0121/seisLM#pretrained-models)
  and place it under `pretrained_seislm_base/checkpoints/`.
- **Marmousi velocity model** (`Marm.bin`): a standard FWI benchmark model,
  publicly available from multiple sources (e.g. the SEG Open Data
  repository).
- **Overthrust velocity model** (`overthrust.vites`): from the SEG/EAGE
  Salt and Overthrust Models dataset
  (https://wiki.seg.org/wiki/SEG/EAGE_Salt_and_Overthrust_Models).
  Inside the downloaded archive, the file is at:
  `salt_and_overthrust_models/3-D_Overthrust_Model_Disk1/3D-Velocity-Grid/overthrust.vites`

The notebooks expect these under `../data/` relative to `notebooks/`,
matching the paths set in each notebook's parameter cell — adjust those
paths if you place the files elsewhere.
