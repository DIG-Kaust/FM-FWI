"""Load a pretrained MultiDimWav2Vec2 model from a seisLM Lightning checkpoint.

This avoids depending on `lightning`/`seisbench` (training-only concerns)
just to load weights for inference: a seisLM `.ckpt` file is a plain
`torch.save`d dict with the config under `hyper_parameters["config"]` and the
model weights under `state_dict`, prefixed with `"model."`.
"""

import torch

from fm_fwi.model.multidim_wav2vec2 import MultiDimWav2Vec2ForPreTraining

_STATE_DICT_PREFIX = "model."


def load_pretrained_multidim_wav2vec2(
  ckpt_path: str,
  map_location: str = "cpu",
) -> MultiDimWav2Vec2ForPreTraining:
  """Load a MultiDimWav2Vec2ForPreTraining model from a seisLM checkpoint."""
  checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
  config = checkpoint["hyper_parameters"]["config"]

  model = MultiDimWav2Vec2ForPreTraining(config.model_config)
  state_dict = {
    key[len(_STATE_DICT_PREFIX):]: value
    for key, value in checkpoint["state_dict"].items()
    if key.startswith(_STATE_DICT_PREFIX)
  }
  model.load_state_dict(state_dict, strict=True)
  model.eval()
  return model
