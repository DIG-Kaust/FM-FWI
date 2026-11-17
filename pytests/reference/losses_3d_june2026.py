# Reference copy, not part of the package: the SeisLM loss module that produced the 3D results of the paper
# (losses.py, md5 41c8bf950761d7002a82091a8cca5caf). Used only by
# pytests/test_losses.py to show that fm_fwi.losses gives the same loss and gradient on the 3D settings.
__all__ = ["SeisLMFeatureLoss", "HybridLoss"]

import hashlib
import json
import os
import numpy as np
from fractions import Fraction
from pylops import TorchOperator
from devitofwi.nonlinear import NonlinearOperator

try:
    import torch
    import torch.nn.functional as F
except Exception:
    print("torch not available, install torch to use SeisLMFeatureLoss...")

try:
    from scipy.signal import resample_poly as scipy_resample_poly
except Exception:
    print("scipy not available, install scipy to use resampling in SeisLMFeatureLoss...")


# class SeisLMFeatureLoss:
#     r"""SeisLM feature-space loss for seismic traces (chunked, OOM-safe).

#     SoftDTW-like wrapper with __call__/grad interface.

#     Current MVP behavior:
#     - Treat receiver traces as batch dimension: [B, T] with B = nr
#     - Duplicate single-component trace to 3 channels: [B, 3, T]
#     - Compare SeisLM embeddings using MSE after temporal mean pooling
#     - Optional receiver-chunking to avoid GPU OOM
#     - Optional resampling to match SeisBench training sampling rate (100 Hz)
#     - Optional multi-layer feature comparison with per-layer weighting

#     Parameters
#     ----------
#     seislm_model : torch.nn.Module
#         SeisLM pretraining model (e.g., `lit_model.model`) exposing `.wav2vec2(...)`.
#     Op : pylops.LinearOperator or list, optional
#         If provided, modeled data are computed as Op*x.
#         If None, `x` is assumed to be the modeled shot gather itself (flattened nt*nr).
#     b : numpy.ndarray
#         Observed data array. Each `b[i]` must be flattenable to (nt, nr).
#     nr : int
#         Number of receivers.
#     nt : int
#         Number of time samples.
#     t_sub : int, optional
#         Time subsampling factor. Default 1.
#     device : str, optional
#         Device for SeisLM forward ("cpu" or "cuda"). Default "cpu".
#     pool : str, optional
#         Embedding pooling over token dimension:
#           - "mean" (default): compare [B, D]
#           - "none": compare full [B, T', D]
#     normalize_traces : bool, optional
#         If True, per-trace standardization over time. Default True.
#     eps : float, optional
#         Small epsilon for standardization. Default 1e-6.
#     cache_obs_embeddings : bool, optional
#         Cache observed embeddings per shot index. Default True.
#     cache_obs_on_cpu : bool, optional
#         Store cached observed embeddings on CPU to save GPU memory. Default True.
#     trace_chunk : int or None, optional
#         Number of receiver traces per SeisLM forward pass. If None, uses all receivers
#         at once (fastest, highest memory). Set e.g. 8/16/32 to avoid OOM.
#     dt : float or None, optional
#         Time sampling interval of the input data in seconds (e.g. 1.46e-3).
#         Required when resample=True. Default None.
#     resample : bool, optional
#         If True, resample traces from 1/dt Hz to fs_target Hz before encoding,
#         and resample the gradient back to the original time dimension before returning.
#         Resampling is performed in numpy before building the autograd graph,
#         so the gradient path is preserved. Output sizes are enforced exactly to
#         avoid rounding errors from the rational approximation of the resampling ratio.
#         Requires dt to be set. Default False.
#     fs_target : float, optional
#         Target sampling rate in Hz for resampling. Should match the sampling rate
#         of the SeisBench training data. Default 100.0.
#     resample_denom_limit : int, optional
#         Maximum denominator for the rational approximation of the resampling ratio
#         passed to resample_poly. Default 1000.
#     feature_layers : int, list of int, or None, optional
#         Which transformer hidden layers to use for feature comparison.
#         - None (default): use only the final layer (last_hidden_state), same as
#           original behavior.
#         - int: use a single hidden layer, e.g. feature_layers=6.
#         - list of int: use multiple hidden layers, e.g. feature_layers=[4, 8, 12].
#           Layer indices are 0-based (0 = first transformer layer output).
#           Loss is computed per layer and averaged (or weighted if layer_weights given).
#     layer_weights : list of float or None, optional
#         Per-layer weights for the multi-layer loss. Must have the same length as
#         feature_layers if feature_layers is a list. If None, all layers are weighted
#         equally. Weights are normalized to sum to 1 internally. Default None.

#     Notes
#     -----
#     When resample=True and Op is not None, self.x is reassigned to the resampled
#     modeled data leaf [nr, nt_rs], so the gradient is treated the same as the
#     Op is None case: resampled back to [nr, nt] and transposed to [nt, nr].
#     When Op is None, the gradient is correctly returned with respect to the original
#     (non-resampled) shot gather shape (nt, nr).

#     The SeisLM wav2vec2 model has 12 transformer layers. Layer indices range from
#     0 (shallowest) to 11 (deepest). The final layer output (last_hidden_state) is
#     equivalent to layer index 11.
#     """

#     def __init__(
#         self,
#         seislm_model,
#         Op,
#         b,
#         nr,
#         nt,
#         t_sub=1,
#         device="cpu",
#         pool="mean",
#         normalize_traces=True,
#         eps=1e-6,
#         cache_obs_embeddings=True,
#         cache_obs_on_cpu=True,
#         trace_chunk=None,
#         dt=None,
#         resample=False,
#         fs_target=100.0,
#         resample_denom_limit=1000,
#         feature_layers=None,
#         layer_weights=None,
#         n_input_channels=3,
#     ):
#         self.n_input_channels = n_input_channels
#         self.Op = Op
#         self.b = b
#         self.nr, self.nt = nr, nt
#         self.t_sub = t_sub
#         self.device = device
#         self.pool = pool
#         self.normalize_traces = normalize_traces
#         self.eps = eps
#         self.cache_obs_embeddings = cache_obs_embeddings
#         self.cache_obs_on_cpu = cache_obs_on_cpu
#         self.trace_chunk = trace_chunk

#         # Resampling
#         self.resample = resample
#         self.fs_target = fs_target
#         if resample:
#             if dt is None:
#                 raise ValueError("dt must be provided when resample=True.")
#             fs_input = 1.0 / float(dt)
#             ratio = Fraction(float(fs_target / fs_input)).limit_denominator(resample_denom_limit)
#             self._resample_up = ratio.numerator
#             self._resample_down = ratio.denominator

#             # Pre-compute exact resampled length by doing a dummy resample on a single
#             # trace of length nt. This avoids off-by-a-few-samples rounding errors
#             # when enforcing exact sizes in _resample_shot and _resample_grad_back.
#             dummy = np.zeros((1, nt), dtype=np.float32)
#             self._nt_rs = scipy_resample_poly(
#                 dummy, self._resample_up, self._resample_down, axis=-1
#             ).shape[-1]
#         else:
#             self._resample_up = None
#             self._resample_down = None
#             self._nt_rs = None

#         # Feature layers
#         # Normalize feature_layers to a list (or None for last_hidden_state only)
#         if feature_layers is None:
#             self.feature_layers = None
#             self.layer_weights = [1.0]          # single weight for last_hidden_state
#         elif isinstance(feature_layers, int):
#             self.feature_layers = [feature_layers]
#         else:
#             self.feature_layers = list(feature_layers)

#         # Normalize layer weights to sum to 1 (only when feature_layers is not None)
#         if self.feature_layers is not None:
#             n_layers = len(self.feature_layers)
#             if layer_weights is not None:
#                 if len(layer_weights) != n_layers:
#                     raise ValueError(
#                         f"layer_weights length ({len(layer_weights)}) must match "
#                         f"feature_layers length ({n_layers})."
#                     )
#                 w = np.array(layer_weights, dtype=np.float32)
#                 self.layer_weights = (w / w.sum()).tolist()
#             else:
#                 self.layer_weights = [1.0 / n_layers] * n_layers

#         # SeisLM model (usually lit_model.model)
#         self.seislm = seislm_model.to(device).eval()
#         for p in self.seislm.parameters():
#             p.requires_grad_(False)

#         # Cache model dtype/device so inputs always match (fixes float64/float32 mismatch)
#         self.model_param = next(self.seislm.parameters())
#         self.model_dtype = self.model_param.dtype
#         self.model_device = self.model_param.device

#         # Internal state
#         self.x = None                   # leaf tensor wrt which grad is returned
#         self.f = None                   # scalar loss tensor from last __call__ (non-chunk mode)
#         self._loss_value = None         # Python float loss value from last __call__
#         self._obs_cache = {}            # shot index -> list of detached embedding tensors
#         self._grad_cache = None         # numpy gradient cache for chunked mode

#     # ------------------------------------------------------------------
#     # Helpers
#     # ------------------------------------------------------------------
#     def clear_cache(self):
#         """Clear cached observed embeddings."""
#         self._obs_cache.clear()

#     def _resample_shot(self, shot_nr_nt_np):
#         """Resample a [nr_chunk, nt] numpy array along the time axis (forward).

#         Output is cropped or zero-padded to exactly self._nt_rs samples to handle
#         rounding errors from the rational approximation of the resampling ratio.

#         Parameters
#         ----------
#         shot_nr_nt_np : np.ndarray
#             Shape [nr_chunk, nt] at original sampling rate.

#         Returns
#         -------
#         np.ndarray
#             Shape [nr_chunk, nt_rs] at fs_target, exactly self._nt_rs samples.
#         """
#         g = scipy_resample_poly(
#             shot_nr_nt_np, self._resample_up, self._resample_down, axis=-1
#         ).astype(np.float32)

#         # Enforce exact output length
#         nt_out = g.shape[-1]
#         if nt_out > self._nt_rs:
#             g = g[:, :self._nt_rs]
#         elif nt_out < self._nt_rs:
#             g = np.pad(g, ((0, 0), (0, self._nt_rs - nt_out)))
#         return g

#     def _resample_grad_back(self, g_nr_nt_rs):
#         """Resample gradient from [nr, nt_rs] back to [nr, nt] (inverse resampling).

#         Output is cropped or zero-padded to exactly self.nt samples to handle
#         rounding errors from the rational approximation of the resampling ratio.

#         Parameters
#         ----------
#         g_nr_nt_rs : np.ndarray
#             Shape [nr, nt_rs] — gradient in resampled time domain.

#         Returns
#         -------
#         np.ndarray
#             Shape [nr, nt] — gradient in original time domain, exactly self.nt samples.
#         """
#         g = scipy_resample_poly(
#             g_nr_nt_rs, self._resample_down, self._resample_up, axis=-1
#         ).astype(np.float32)

#         # Enforce exact output length
#         nt_out = g.shape[-1]
#         if nt_out > self.nt:
#             g = g[:, :self.nt]
#         elif nt_out < self.nt:
#             g = np.pad(g, ((0, 0), (0, self.nt - nt_out)))
#         return g

#     def _trace_standardize(self, x):
#         """Per-trace standardization over time. x: [B, C, T]."""
#         mean = x.mean(dim=-1, keepdim=True)
#         std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.eps)
#         return (x - mean) / std

#     def _prepare_batch_3c(self, shot_nr_nt):
#         """Convert [nr_chunk, nt] -> [nr_chunk, 3, nt_sub] and normalize if requested.

#         Note: resampling is NOT performed here to preserve the autograd graph.
#         Resampling is applied upstream in numpy before creating leaf tensors.
#         """
#         x = shot_nr_nt[:, :: self.t_sub]          # [B, nt_sub]
#         x = x.unsqueeze(1).repeat(1, self.n_input_channels, 1)  # [B, C, T]
#         x = x.to(torch.float32)
#         if self.normalize_traces:
#             x = self._trace_standardize(x)
#         return x

#     def _pool(self, z):
#         """Pool embeddings over token dimension.

#         Parameters
#         ----------
#         z : torch.Tensor
#             Shape [B, T', D].

#         Returns
#         -------
#         torch.Tensor
#             Shape [B, D] if pool='mean', or [B, T', D] if pool='none'.
#         """
#         if self.pool == "mean":
#             return z.mean(dim=1)
#         elif self.pool == "none":
#             return z
#         else:
#             raise ValueError(f"Unsupported pool='{self.pool}'. Use 'mean' or 'none'.")

#     def _encode(self, x_bct):
#         """SeisLM encoder forward. Returns list of embeddings, one per requested layer.

#         Parameters
#         ----------
#         x_bct : torch.Tensor
#             Shape [B, C, T].

#         Returns
#         -------
#         list of torch.Tensor
#             If feature_layers is None: single-element list containing last_hidden_state,
#             pooled to [B, D] or left as [B, T', D] depending on self.pool.
#             If feature_layers is set: list of pooled embeddings, one per requested layer,
#             in the same order as self.feature_layers.
#         """
#         x_bct = x_bct.to(device=self.model_device, dtype=self.model_dtype)

#         # Request hidden states only if needed
#         need_hidden = self.feature_layers is not None
#         out = self.seislm.wav2vec2(x_bct, output_hidden_states=need_hidden)

#         if not need_hidden:
#             # Original behavior: last hidden state only
#             z = out.last_hidden_state                  # [B, T', D]
#             return [self._pool(z)]

#         # out.hidden_states is a tuple of length n_layers + 1
#         # index 0 = embedding layer output, indices 1..n_layers = transformer layer outputs
#         # layer_idx 0 -> hidden_states[1], layer_idx 1 -> hidden_states[2], etc.
#         hidden_states = out.hidden_states
#         embeddings = []
#         for layer_idx in self.feature_layers:
#             z = hidden_states[layer_idx + 1]           # [B, T', D]
#             embeddings.append(self._pool(z))
#         return embeddings

#     def _compute_layer_loss(self, z_mod_list, z_obs_list):
#         """Compute weighted MSE loss across all requested layers.

#         Parameters
#         ----------
#         z_mod_list : list of torch.Tensor
#             Modeled embeddings per layer.
#         z_obs_list : list of torch.Tensor
#             Observed embeddings per layer.

#         Returns
#         -------
#         torch.Tensor
#             Scalar weighted loss.
#         """
#         total = torch.tensor(0.0, device=self.model_device, dtype=self.model_dtype)
#         for z_mod, z_obs, w in zip(z_mod_list, z_obs_list, self.layer_weights):
#             total = total + w * F.mse_loss(z_mod, z_obs)
#         return total

#     def _iter_trace_chunks(self, n):
#         """Yield (start, end) chunks over receiver dimension."""
#         if self.trace_chunk is None or self.trace_chunk <= 0 or self.trace_chunk >= n:
#             yield 0, n
#         else:
#             for s in range(0, n, self.trace_chunk):
#                 e = min(s + self.trace_chunk, n)
#                 yield s, e

#     def _get_observed_embedding(self, i):
#         """Compute (or get cached) observed embedding for shot i.

#         Returns
#         -------
#         list of torch.Tensor
#             One tensor per requested layer, each [nr, D] or [nr, T', D].
#         """
#         if self.cache_obs_embeddings and i in self._obs_cache:
#             return self._obs_cache[i]

#         # b[i] flattened -> (nt, nr) -> transpose -> [nr, nt] numpy
#         b_shot_np = self.b[i].reshape(self.nt, self.nr).T.astype(np.float32)  # [nr, nt]

#         # Resample in numpy before converting to torch (no graph needed for observed)
#         if self.resample:
#             b_shot_np = self._resample_shot(b_shot_np)                         # [nr, nt_rs]

#         b_shot = torch.from_numpy(b_shot_np)                                   # [nr, nt(_rs)]

#         # Accumulate per-layer embeddings across chunks: list of lists
#         n_layers = len(self.feature_layers) if self.feature_layers is not None else 1
#         zs_per_layer = [[] for _ in range(n_layers)]

#         with torch.no_grad():
#             for s, e in self._iter_trace_chunks(self.nr):
#                 b_chunk = self._prepare_batch_3c(b_shot[s:e, :])               # [Bchunk, 3, T]
#                 z_list = self._encode(b_chunk)                                  # list of [Bchunk, D]
#                 for li, z_chunk in enumerate(z_list):
#                     z_chunk = z_chunk.detach()
#                     if self.cache_obs_on_cpu:
#                         z_chunk = z_chunk.cpu()
#                     zs_per_layer[li].append(z_chunk)

#         # Concatenate chunks for each layer: list of [nr, D]
#         z_obs_all = [torch.cat(zs, dim=0) for zs in zs_per_layer]

#         if self.cache_obs_embeddings:
#             self._obs_cache[i] = z_obs_all
#         return z_obs_all

#     def _postprocess_grad(self, g, x):
#         """Resample gradient back to original time domain and apply dtype/layout fixes.

#         Parameters
#         ----------
#         g : np.ndarray
#             Raw gradient from autograd:
#               - Op is None or (Op is not None and resample=True): shape [nr, nt_rs]
#               - Op is not None and resample=False: flat shape matching x
#         x : np.ndarray
#             Original input, used to match output dtype.

#         Returns
#         -------
#         np.ndarray
#             If Op is None or resample=True: shape [nt, nr] in original time domain.
#             If Op is not None and resample=False: flat shape matching x.
#         """
#         # When resample=True and Op is not None, self.x was reassigned to the resampled
#         # y_shot leaf, so gradient shape is [nr, nt_rs] and needs same treatment as Op is None.
#         needs_spatial_postprocess = (self.Op is None) or (self.resample and self.Op is not None)

#         if needs_spatial_postprocess:
#             if self.resample:
#                 g = self._resample_grad_back(g)    # [nr, nt_rs] -> [nr, nt]
#             g = g.T                                # [nr, nt] -> [nt, nr]

#         if hasattr(x, "dtype"):
#             g = g.astype(x.dtype, copy=False)

#         return g

#     # ------------------------------------------------------------------
#     # Public API (SoftDTW-like)
#     # ------------------------------------------------------------------
#     def __call__(self, x, i):
#         """Compute SeisLM feature loss for shot i.

#         If trace_chunk is set, the method performs chunked forward/backward and caches the
#         gradient internally to avoid GPU OOM. Then grad(...) simply returns the cache.

#         Parameters
#         ----------
#         x : numpy.ndarray
#             If Op is None: flattened modeled shot gather of shape (nt*nr,)
#             If Op is not None: optimization variable to be mapped by Op
#         i : int
#             Shot index.

#         Returns
#         -------
#         float
#             Scalar loss value.
#         """
#         # Reset previous state
#         self.f = None
#         self.x = None
#         self._loss_value = None
#         self._grad_cache = None

#         # Observed embeddings (fixed): list of [nr, D] per layer
#         z_obs_all = self._get_observed_embedding(i)

#         # Build modeled data tensor / leaf variable
#         if self.Op is not None:
#             self.x = torch.from_numpy(x).to(torch.float32).requires_grad_()
#             Op_i = self.Op[i] if isinstance(self.Op, list) else self.Op
#             y = TorchOperator(Op_i).apply(self.x)
#             y_shot = y.reshape(self.nt, self.nr).T              # [nr, nt] with grad

#             if self.resample:
#                 y_shot_np = y_shot.detach().cpu().numpy()
#                 y_shot_np = self._resample_shot(y_shot_np)
#                 y_shot = torch.from_numpy(y_shot_np).requires_grad_()
#                 self.x = y_shot
#         else:
#             x_shot_np = x.reshape(self.nt, self.nr).T.astype(np.float32)

#             if self.resample:
#                 x_shot_np = self._resample_shot(x_shot_np)

#             x_shot = torch.from_numpy(x_shot_np)
#             self.x = x_shot.to(device=self.model_device,
#                                dtype=torch.float32).requires_grad_()
#             y_shot = self.x

#         # Fast path: no chunking (higher memory)
#         if self.trace_chunk is None or self.trace_chunk <= 0 or self.trace_chunk >= self.nr:
#             z_obs_dev = [z.to(device=self.model_device, dtype=self.model_dtype)
#                          for z in z_obs_all]

#             y_batch = self._prepare_batch_3c(y_shot)
#             z_mod_list = self._encode(y_batch)
#             self.f = self._compute_layer_loss(z_mod_list, z_obs_dev)
#             self._loss_value = float(self.f.detach().cpu().item())
#             return self._loss_value

#         # --------------------------------------------------------------
#         # Chunked path: stream forward/backward per trace-chunk (OOM-safe)
#         # --------------------------------------------------------------
#         if self.x.grad is not None:
#             self.x.grad.zero_()

#         total_loss = 0.0
#         total_count = 0

#         for s, e in self._iter_trace_chunks(self.nr):
#             bsz = e - s

#             y_chunk = y_shot[s:e, :]
#             y_batch = self._prepare_batch_3c(y_chunk)
#             z_mod_list = self._encode(y_batch)

#             z_obs_chunk_list = [
#                 z_obs_all[li][s:e].to(device=self.model_device, dtype=self.model_dtype)
#                 for li in range(len(z_obs_all))
#             ]

#             loss_chunk = self._compute_layer_loss(z_mod_list, z_obs_chunk_list)

#             total_loss += float(loss_chunk.detach().cpu().item()) * bsz
#             total_count += bsz

#             (loss_chunk * (bsz / float(self.nr))).backward()

#             del y_chunk, y_batch, z_mod_list, z_obs_chunk_list, loss_chunk

#         self._loss_value = total_loss / max(total_count, 1)

#         g = self.x.grad.detach().cpu().numpy()
#         g = self._postprocess_grad(g, x)
#         self._grad_cache = g

#         self.f = None
#         return self._loss_value

#     def grad(self, x, i):
#         """Return gradient from last __call__.

#         In chunked mode, gradient is already computed/cached in __call__.
#         In non-chunked mode, this triggers backward().

#         Returns
#         -------
#         numpy.ndarray
#             If Op is None or resample=True:
#                 gradient wrt modeled shot gather, shape (nt, nr)
#                 (matching SoftDTW-style convention), in original time domain.
#             If Op is not None and resample=False:
#                 gradient wrt optimization variable x (same flat shape as x).
#         """
#         if self._grad_cache is not None:
#             return self._grad_cache

#         if self.f is None or self.x is None:
#             raise RuntimeError(
#                 "Call the loss first via `f = loss(x, i)` before calling `grad(x, i)`.")

#         self.f.backward()
#         g = self.x.grad.detach().cpu().numpy()
#         g = self._postprocess_grad(g, x)

#         return g


class HybridLoss(NonlinearOperator):
    r"""Hybrid L2 + SeisLM feature-space loss.

    Computes the combined loss:

    .. math::
        \mathcal{L} = \alpha \cdot \frac{\mathcal{L}_{L2}}{\mathcal{L}_{L2}^{(0)}} +
        (1 - \alpha) \cdot \frac{\mathcal{L}_{SeisLM}}{\mathcal{L}_{SeisLM}^{(0)}}

    where :math:`\mathcal{L}^{(0)}` denotes the value at the first evaluation,
    used to normalize both losses to the same scale.

    Parameters
    ----------
    l2_loss : L2 or L2Torch
        Instantiated L2 loss object.
    seislm_loss : SeisLMFeatureLoss
        Instantiated SeisLM feature loss object.
    alpha : float or callable, optional
        Blending weight in [0, 1], or a callable taking the current iteration
        index and returning a float in [0, 1]. Default 0.5.

    Notes
    -----
    ``loss()`` must always be called before ``grad()`` for a given shot.
    Scales are set automatically on the first call to ``loss()``.
    Call ``step()`` after each outer FWI iteration to advance the schedule.
    """

    def __init__(self, l2_loss, seislm_loss, alpha=0.5):
        self.l2_loss     = l2_loss
        self.seislm_loss = seislm_loss
        self._alpha      = alpha
        self._iter       = 0
        self._scale_l2     = None   # set on first loss() call
        self._scale_seislm = None   # set on first loss() call
        self._loss_value   = None   # scalar loss from last loss() call
        super().__init__(size=l2_loss.size, dtype=l2_loss.dtype)

    @property
    def alpha(self):
        """Current alpha value, resolved from schedule if callable."""
        a = self._alpha(self._iter) if callable(self._alpha) else float(self._alpha)
        if not 0.0 <= a <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {a} at iteration {self._iter}")
        return a

    def step(self):
        """Advance the iteration counter by one. Call after each FWI iteration."""
        self._iter += 1

    def loss(self, x, i):
        f_l2     = self.l2_loss.loss(x, i)
        f_seislm = self.seislm_loss(x, i)

        # Auto-calibrate on first evaluation using first loss values
        if self._scale_l2 is None:
            self._scale_l2     = f_l2     if f_l2     > 0 else 1.0
            self._scale_seislm = f_seislm if f_seislm > 0 else 1.0
            print(f"HybridLoss calibrated:")
            print(f"  L2     scale = {self._scale_l2:.6e}")
            print(f"  SeisLM scale = {self._scale_seislm:.6e}")

        a = self.alpha
        self._loss_value = float(a * (f_l2 / self._scale_l2) + (1.0 - a) * (f_seislm / self._scale_seislm))
        return self._loss_value

    def grad(self, x, i):
        g_l2     = self.l2_loss.grad(x, i)
        g_seislm = self.seislm_loss.grad(x, i)
        a = self.alpha
        # Divide by same scale factors used in loss() to keep loss/grad consistent
        return a * (g_l2 / self._scale_l2) + (1.0 - a) * (g_seislm.ravel() / self._scale_seislm)


import copy
import threading
from concurrent.futures import ThreadPoolExecutor


class SeisLMFeatureLoss:
    r"""SeisLM feature-space loss for seismic traces (chunked, OOM-safe).

    SoftDTW-like wrapper with __call__/grad interface.

    Current MVP behavior:
    - Treat receiver traces as batch dimension: [B, T] with B = nr
    - Duplicate single-component trace to 3 channels: [B, 3, T]
    - Compare SeisLM embeddings using MSE after temporal mean pooling
    - Optional receiver-chunking to avoid GPU OOM
    - Optional resampling to match SeisBench training sampling rate (100 Hz)
    - Optional multi-layer feature comparison with per-layer weighting

    Parameters
    ----------
    seislm_model : torch.nn.Module
        SeisLM pretraining model (e.g., `lit_model.model`) exposing `.wav2vec2(...)`.
    Op : pylops.LinearOperator or list, optional
        If provided, modeled data are computed as Op*x AND the same operator is
        applied to each observed shot ``b[i]`` before encoding, so both sides
        of the feature-space comparison see consistently masked data.
        If None, ``x`` is assumed to be the modeled shot gather itself
        (flattened nt*nr) and ``b[i]`` is used as-is.
    b : numpy.ndarray
        Observed data array. Each `b[i]` must be flattenable to (nt, nr).
    nr : int
        Number of receivers.
    nt : int
        Number of time samples.
    t_sub : int, optional
        Time subsampling factor. Default 1.
    device : str, optional
        Device for SeisLM forward ("cpu" or "cuda"). Default "cpu".
    pool : str, optional
        Embedding pooling over token dimension:
          - "mean" (default): compare [B, D]
          - "none": compare full [B, T', D]
    normalize_traces : bool, optional
        If True, per-trace standardization over time. Default True.
    eps : float, optional
        Small epsilon for standardization. Default 1e-6.
    cache_obs_embeddings : bool, optional
        Cache observed embeddings per shot index. Default True.
    cache_obs_on_cpu : bool, optional
        Store cached observed embeddings on CPU to save GPU memory. Default True.
    cache_obs_embeddings_to_disk : bool, optional
        If True, persist observed embeddings to ``obs_embedding_cache_dir`` and
        reuse them across Python runs. The in-memory cache is still checked first
        when ``cache_obs_embeddings=True``. Default False.
    obs_embedding_cache_dir : str or None, optional
        Directory used for disk-backed observed embedding cache files. Required
        when ``cache_obs_embeddings_to_disk=True``. Default None.
    obs_embedding_cache_tag : str or None, optional
        User tag included in the disk cache key. Use this to separate runs with
        different masks/operators, checkpoints, or experiments. Default None.
    trace_chunk : int or None, optional
        Number of receiver traces per SeisLM forward pass. If None, uses all receivers
        at once (fastest, highest memory). Set e.g. 8/16/32 to avoid OOM.
    dt : float or None, optional
        Time sampling interval of the input data in seconds (e.g. 1.46e-3).
        Required when resample=True. Default None.
    resample : bool, optional
        If True, resample traces from 1/dt Hz to fs_target Hz before encoding,
        and resample the gradient back to the original time dimension before returning.
        Resampling is performed in numpy before building the autograd graph,
        so the gradient path is preserved. Output sizes are enforced exactly to
        avoid rounding errors from the rational approximation of the resampling ratio.
        Requires dt to be set. Default False.
    fs_target : float, optional
        Target sampling rate in Hz for resampling. Should match the sampling rate
        of the SeisBench training data. Default 100.0.
    resample_denom_limit : int, optional
        Maximum denominator for the rational approximation of the resampling ratio
        passed to resample_poly. Default 1000.
    devices : list of str or None, optional
        List of device strings for multi-GPU parallelism, e.g. ``["cuda:0", "cuda:1"]``.
        When provided, the SeisLM model is replicated on each device and trace chunks
        are distributed across GPUs using a thread pool (one thread per GPU).
        Each GPU processes its assigned chunks sequentially; all GPUs run in parallel,
        giving an ~N× speedup for the encoding bottleneck where N = len(devices).
        Only effective for the chunked path (trace_chunk is not None).
        Gradient correctness is preserved via a detach + vector-Jacobian product (VJP).
        When ``None`` (default), falls back to the single ``device`` argument.
        Multi-GPU is only supported when ``resample=False``; if ``resample=True`` the
        code silently falls back to single-GPU chunked processing.
    feature_layers : int, list of int, or None, optional
        Which transformer hidden layers to use for feature comparison.
        - None (default): use only the final layer (last_hidden_state), same as
          original behavior.
        - int: use a single hidden layer, e.g. feature_layers=6.
        - list of int: use multiple hidden layers, e.g. feature_layers=[4, 8, 12].
          Layer indices are 0-based (0 = first transformer layer output).
          Loss is computed per layer and averaged (or weighted if layer_weights given).
    layer_weights : list of float or None, optional
        Per-layer weights for the multi-layer loss. Must have the same length as
        feature_layers if feature_layers is a list. If None, all layers are weighted
        equally. Weights are normalized to sum to 1 internally. Default None.
    debug_plot : bool, optional
        If True, dump diagnostic PNGs of the [B, C, T] tensors that actually
        enter SeisLM, for both the observed and modelled sides. The plots are
        produced *after* masking via Op, *after* optional resampling, and
        *after* per-trace standardization — i.e. they show exactly what the
        encoder sees. Any time-domain gain folded into Op (e.g. for
        geometrical-spreading correction) is therefore visible in the plots.
        Default False.
    debug_plot_dir : str or None, optional
        Output directory for debug PNGs. If None, files are written to the
        current working directory. The directory is created if missing.
    debug_plot_shots : iterable of int or None, optional
        Shot indices to plot. If None (default), only the first shot index
        encountered on each side is plotted. Useful to limit I/O when running
        full inversions.
    debug_plot_n_traces : int, optional
        Number of receiver traces to overlay per panel (chosen via linspace
        across the receiver axis). Default 8.
    debug_adjsrc : bool, optional
        If True, save the per-shot adjoint sources (dL/d_modelled) as .npy +
        a diagnostic PNG, for the shots listed in ``debug_adjsrc_shots``. Two
        files are written per shot: one for the post-mask adjoint source
        (``self.x.grad``, what Devito's adjoint solve receives) and one for
        the pre-mask version (``y_shot_conn.grad`` or its multi-GPU
        equivalent, what SeisLM's features want before the operator
        attenuates them). Default False.
    debug_adjsrc_dir : str or None, optional
        Output directory for adjoint-source files. If None, files go to CWD.
        The directory is created if missing.
    debug_adjsrc_shots : iterable of int or None, optional
        Shot indices to save. If None, only the first shot index encountered
        is saved.
    debug_adjsrc_n_traces : int, optional
        Number of receiver traces to overlay per diagnostic PNG. Default 8.

    Notes
    -----
    When resample=True and Op is not None, self.x is reassigned to the resampled
    modeled data leaf [nr, nt_rs], so the gradient is treated the same as the
    Op is None case: resampled back to [nr, nt] and transposed to [nt, nr].
    When Op is None, the gradient is correctly returned with respect to the original
    (non-resampled) shot gather shape (nt, nr).

    The SeisLM wav2vec2 model has 12 transformer layers. Layer indices range from
    0 (shallowest) to 11 (deepest). The final layer output (last_hidden_state) is
    equivalent to layer index 11.
    """

    def __init__(
        self,
        seislm_model,
        Op,
        b,
        nr,
        nt,
        t_sub=1,
        device="cpu",
        devices=None,
        pool="mean",
        normalize_traces=True,
        eps=1e-6,
        cache_obs_embeddings=True,
        cache_obs_on_cpu=True,
        cache_obs_embeddings_to_disk=False,
        obs_embedding_cache_dir=None,
        obs_embedding_cache_tag=None,
        trace_chunk=None,
        dt=None,
        resample=False,
        fs_target=100.0,
        resample_denom_limit=1000,
        feature_layers=None,
        layer_weights=None,
        n_input_channels=3,
        log_fn=None,
        debug_plot=False,
        debug_plot_dir=None,
        debug_plot_shots=None,
        debug_plot_n_traces=8,
        debug_adjsrc=False,
        debug_adjsrc_dir=None,
        debug_adjsrc_shots=None,
        debug_adjsrc_n_traces=8,
    ):
        self.n_input_channels = n_input_channels
        self.Op = Op
        self.b = b
        # Debug plotting: dump the [B, C, T] tensors that actually enter SeisLM
        # for both the observed and modelled sides, so the user can verify that
        # masking, resampling and trace standardization look right end-to-end.
        # debug_plot          : master switch (default False).
        # debug_plot_dir      : output directory for PNGs. If None, uses CWD.
        # debug_plot_shots    : iterable of shot indices to plot (e.g. [0, 35, 69]).
        #                       If None, only the first shot index encountered
        #                       per side is plotted.
        # debug_plot_n_traces : how many traces to overlay per panel.
        self.debug_plot = bool(debug_plot)
        self.debug_plot_dir = debug_plot_dir
        self.debug_plot_shots = (
            set(int(i) for i in debug_plot_shots)
            if debug_plot_shots is not None else None
        )
        self.debug_plot_n_traces = int(debug_plot_n_traces)
        self._debug_plotted = set()       # (shot_idx, side) keys already dumped

        # Adjoint-source debug: save dL/dx_modelled (post-mask, what Devito
        # back-propagates) and dL/dy (pre-mask, what SeisLM features want)
        # as .npy + diagnostic PNG. Mirrors the input-debug machinery.
        self.debug_adjsrc = bool(debug_adjsrc)
        self.debug_adjsrc_dir = debug_adjsrc_dir
        self.debug_adjsrc_shots = (
            set(int(i) for i in debug_adjsrc_shots)
            if debug_adjsrc_shots is not None else None
        )
        self.debug_adjsrc_n_traces = int(debug_adjsrc_n_traces)
        self._adjsrc_saved = set()        # shot indices already saved

        # Set per-shot during __call__ when debug capture is active. Cleared
        # after grad() consumes them to avoid retaining tensors.
        self._adjsrc_capture_active = False
        self._adjsrc_pre_mask_grad = None     # torch.Tensor [nr, nt] or None
        self._adjsrc_current_shot = None      # int or None
        self.nr, self.nt = nr, nt
        self.t_sub = t_sub
        self.device = device
        self.pool = pool
        self.normalize_traces = normalize_traces
        self.eps = eps
        self.cache_obs_embeddings = cache_obs_embeddings
        self.cache_obs_on_cpu = cache_obs_on_cpu
        self.cache_obs_embeddings_to_disk = bool(cache_obs_embeddings_to_disk)
        self.obs_embedding_cache_dir = obs_embedding_cache_dir
        self.obs_embedding_cache_tag = obs_embedding_cache_tag
        if self.cache_obs_embeddings_to_disk and self.obs_embedding_cache_dir is None:
            raise ValueError(
                "obs_embedding_cache_dir must be provided when "
                "cache_obs_embeddings_to_disk=True."
            )
        self.trace_chunk = trace_chunk
        self.log_fn = log_fn if log_fn is not None else (lambda tag: None)

        # Resampling
        self.dt = None if dt is None else float(dt)
        self.resample = resample
        self.fs_target = fs_target
        if resample:
            if dt is None:
                raise ValueError("dt must be provided when resample=True.")
            fs_input = 1.0 / float(dt)
            ratio = Fraction(float(fs_target / fs_input)).limit_denominator(resample_denom_limit)
            self._resample_up = ratio.numerator
            self._resample_down = ratio.denominator

            # Pre-compute exact resampled length by doing a dummy resample on a single
            # trace of length nt. This avoids off-by-a-few-samples rounding errors
            # when enforcing exact sizes in _resample_shot and _resample_grad_back.
            dummy = np.zeros((1, nt), dtype=np.float32)
            self._nt_rs = scipy_resample_poly(
                dummy, self._resample_up, self._resample_down, axis=-1
            ).shape[-1]
        else:
            self._resample_up = None
            self._resample_down = None
            self._nt_rs = None

        # Feature layers
        # Normalize feature_layers to a list (or None for last_hidden_state only)
        if feature_layers is None:
            self.feature_layers = None
            self.layer_weights = [1.0]          # single weight for last_hidden_state
        elif isinstance(feature_layers, int):
            self.feature_layers = [feature_layers]
        else:
            self.feature_layers = list(feature_layers)

        # Normalize layer weights to sum to 1 (only when feature_layers is not None)
        if self.feature_layers is not None:
            n_layers = len(self.feature_layers)
            if layer_weights is not None:
                if len(layer_weights) != n_layers:
                    raise ValueError(
                        f"layer_weights length ({len(layer_weights)}) must match "
                        f"feature_layers length ({n_layers})."
                    )
                w = np.array(layer_weights, dtype=np.float32)
                self.layer_weights = (w / w.sum()).tolist()
            else:
                self.layer_weights = [1.0 / n_layers] * n_layers

        # Resolve device list: multi-GPU when devices is given, else single device
        if devices is not None and len(devices) >= 1:
            self._devices = [torch.device(d) for d in devices]
        else:
            self._devices = [torch.device(device)]

        # Primary model on devices[0]
        self.seislm = seislm_model.to(self._devices[0]).eval()
        for p in self.seislm.parameters():
            p.requires_grad_(False)

        # Additional replicas for multi-GPU (one per extra device)
        self._seislm_replicas = [self.seislm]
        for dev in self._devices[1:]:
            replica = copy.deepcopy(seislm_model).to(dev).eval()
            for p in replica.parameters():
                p.requires_grad_(False)
            self._seislm_replicas.append(replica)

        # Per-GPU locks to serialize CUDA ops within each device
        self._gpu_locks = [threading.Lock() for _ in self._devices]

        # Cache model dtype/device so inputs always match (fixes float64/float32 mismatch)
        self.model_param = next(self.seislm.parameters())
        self.model_dtype = self.model_param.dtype
        self.model_device = self._devices[0]

        # Internal state
        self.x = None                   # leaf tensor wrt which grad is returned
        self.f = None                   # scalar loss tensor from last __call__ (non-chunk mode)
        self._loss_value = None         # Python float loss value from last __call__
        self._obs_cache = {}            # shot index -> list of detached embedding tensors
        self._obs_disk_cache_key = None  # lazily computed settings hash
        self._grad_cache = None         # numpy gradient cache for chunked mode

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def clear_cache(self):
        """Clear cached observed embeddings."""
        self._obs_cache.clear()

    def _obs_cache_metadata(self):
        """Metadata that affects observed SeisLM embeddings."""
        return {
            "version": 1,
            "tag": self.obs_embedding_cache_tag,
            "model_class": self.seislm.__class__.__module__ + "." + self.seislm.__class__.__name__,
            "model_dtype": str(self.model_dtype),
            "has_op": self.Op is not None,
            "nr": int(self.nr),
            "nt": int(self.nt),
            "t_sub": int(self.t_sub),
            "pool": self.pool,
            "normalize_traces": bool(self.normalize_traces),
            "eps": float(self.eps),
            "dt": None if self.dt is None else float(self.dt),
            "resample": bool(self.resample),
            "fs_target": float(self.fs_target),
            "resample_up": self._resample_up,
            "resample_down": self._resample_down,
            "nt_resampled": self._nt_rs,
            "feature_layers": self.feature_layers,
            "layer_weights": [float(w) for w in self.layer_weights],
            "n_input_channels": int(self.n_input_channels),
        }

    def _obs_cache_key(self):
        """Stable hash for disk cache files under current preprocessing settings."""
        if self._obs_disk_cache_key is None:
            payload = json.dumps(
                self._obs_cache_metadata(), sort_keys=True, separators=(",", ":")
            )
            self._obs_disk_cache_key = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
        return self._obs_disk_cache_key

    def _obs_disk_cache_path(self, i):
        fname = f"obs_embed_{self._obs_cache_key()}_shot{int(i):06d}.pt"
        return os.path.join(self.obs_embedding_cache_dir, fname)

    def _place_observed_embeddings(self, z_obs_all):
        """Place cached observed embeddings according to cache_obs_on_cpu."""
        if self.cache_obs_on_cpu:
            return [z.detach().cpu() for z in z_obs_all]
        return [
            z.detach().to(device=self.model_device, dtype=self.model_dtype)
            for z in z_obs_all
        ]

    def _load_observed_embedding_from_disk(self, i):
        if not self.cache_obs_embeddings_to_disk:
            return None

        path = self._obs_disk_cache_path(i)
        if not os.path.exists(path):
            return None

        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            if payload.get("metadata") != self._obs_cache_metadata():
                return None
            z_obs_all = payload["embeddings"]
            if not isinstance(z_obs_all, list):
                return None
            return self._place_observed_embeddings(z_obs_all)
        except Exception as exc:
            print(
                f"[SeisLMFeatureLoss] could not load observed embedding cache "
                f"{path}: {exc}. Recomputing.",
                flush=True,
            )
            return None

    def _save_observed_embedding_to_disk(self, i, z_obs_all):
        if not self.cache_obs_embeddings_to_disk:
            return

        path = self._obs_disk_cache_path(i)
        tmp_path = f"{path}.tmp.{os.getpid()}"
        try:
            os.makedirs(self.obs_embedding_cache_dir, exist_ok=True)
            payload = {
                "metadata": self._obs_cache_metadata(),
                "shot_index": int(i),
                "embeddings": [z.detach().cpu() for z in z_obs_all],
            }
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
        except Exception as exc:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            print(
                f"[SeisLMFeatureLoss] could not save observed embedding cache "
                f"{path}: {exc}",
                flush=True,
            )

    def _resample_shot(self, shot_nr_nt_np):
        """Resample a [nr_chunk, nt] numpy array along the time axis (forward).

        Output is cropped or zero-padded to exactly self._nt_rs samples to handle
        rounding errors from the rational approximation of the resampling ratio.

        Parameters
        ----------
        shot_nr_nt_np : np.ndarray
            Shape [nr_chunk, nt] at original sampling rate.

        Returns
        -------
        np.ndarray
            Shape [nr_chunk, nt_rs] at fs_target, exactly self._nt_rs samples.
        """
        g = scipy_resample_poly(
            shot_nr_nt_np, self._resample_up, self._resample_down, axis=-1
        ).astype(np.float32)

        # Enforce exact output length
        nt_out = g.shape[-1]
        if nt_out > self._nt_rs:
            g = g[:, :self._nt_rs]
        elif nt_out < self._nt_rs:
            g = np.pad(g, ((0, 0), (0, self._nt_rs - nt_out)))
        return g

    def _resample_grad_back(self, g_nr_nt_rs):
        """Resample gradient from [nr, nt_rs] back to [nr, nt] (inverse resampling).

        Output is cropped or zero-padded to exactly self.nt samples to handle
        rounding errors from the rational approximation of the resampling ratio.

        Parameters
        ----------
        g_nr_nt_rs : np.ndarray
            Shape [nr, nt_rs] — gradient in resampled time domain.

        Returns
        -------
        np.ndarray
            Shape [nr, nt] — gradient in original time domain, exactly self.nt samples.
        """
        g = scipy_resample_poly(
            g_nr_nt_rs, self._resample_down, self._resample_up, axis=-1
        ).astype(np.float32)

        # Enforce exact output length
        nt_out = g.shape[-1]
        if nt_out > self.nt:
            g = g[:, :self.nt]
        elif nt_out < self.nt:
            g = np.pad(g, ((0, 0), (0, self.nt - nt_out)))
        return g

    def _trace_standardize(self, x):
        """Per-trace standardization over time. x: [B, C, T]."""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.eps)
        return (x - mean) / std

    def _prepare_batch_3c(self, shot_nr_nt):
        """Convert [nr_chunk, nt] -> [nr_chunk, 3, nt_sub] and normalize if requested.

        Note: resampling is NOT performed here to preserve the autograd graph.
        Resampling is applied upstream in numpy before creating leaf tensors.
        Any time-domain gain (e.g. t^k geometrical-spreading correction) should
        be folded into ``Op`` upstream so it is applied identically to obs and
        modelled sides through the operator chain.
        """
        x = shot_nr_nt[:, :: self.t_sub]          # [B, nt_sub]
        x = x.to(torch.float32)
        x = x.unsqueeze(1).repeat(1, self.n_input_channels, 1)  # [B, C, T]
        if self.normalize_traces:
            x = self._trace_standardize(x)
        return x

    def _pool(self, z):
        """Pool embeddings over token dimension.

        Parameters
        ----------
        z : torch.Tensor
            Shape [B, T', D].

        Returns
        -------
        torch.Tensor
            Shape [B, D] if pool='mean', or [B, T', D] if pool='none'.
        """
        if self.pool == "mean":
            return z.mean(dim=1)
        elif self.pool == "none":
            return z
        else:
            raise ValueError(f"Unsupported pool='{self.pool}'. Use 'mean' or 'none'.")

    def _encode(self, x_bct, device_idx=0):
        """SeisLM encoder forward on the replica at *device_idx*.

        Parameters
        ----------
        x_bct : torch.Tensor
            Shape [B, C, T].  Moved to the target device internally.
        device_idx : int, optional
            Index into ``self._devices`` / ``self._seislm_replicas``. Default 0.

        Returns
        -------
        list of torch.Tensor
            If feature_layers is None: single-element list containing last_hidden_state,
            pooled to [B, D] or left as [B, T', D] depending on self.pool.
            If feature_layers is set: list of pooled embeddings, one per requested layer,
            in the same order as self.feature_layers.
        """
        model  = self._seislm_replicas[device_idx]
        device = self._devices[device_idx]
        x_bct  = x_bct.to(device=device, dtype=self.model_dtype)

        # Request hidden states only if needed
        need_hidden = self.feature_layers is not None
        out = model.wav2vec2(x_bct, output_hidden_states=need_hidden)

        if not need_hidden:
            # Original behavior: last hidden state only
            z = out.last_hidden_state                  # [B, T', D]
            return [self._pool(z)]

        # out.hidden_states is a tuple of length n_layers + 1
        # index 0 = embedding layer output, indices 1..n_layers = transformer layer outputs
        # layer_idx 0 -> hidden_states[1], layer_idx 1 -> hidden_states[2], etc.
        hidden_states = out.hidden_states
        embeddings = []
        for layer_idx in self.feature_layers:
            z = hidden_states[layer_idx + 1]           # [B, T', D]
            embeddings.append(self._pool(z))
        return embeddings

    def _compute_layer_loss(self, z_mod_list, z_obs_list):
        """Compute weighted MSE loss across all requested layers.

        Parameters
        ----------
        z_mod_list : list of torch.Tensor
            Modeled embeddings per layer.
        z_obs_list : list of torch.Tensor
            Observed embeddings per layer (must be on same device as z_mod_list).

        Returns
        -------
        torch.Tensor
            Scalar weighted loss on the same device as the input embeddings.
        """
        # Infer device/dtype from the first modeled embedding (device-agnostic)
        dev   = z_mod_list[0].device
        dtype = z_mod_list[0].dtype
        total = torch.tensor(0.0, device=dev, dtype=dtype)
        for z_mod, z_obs, w in zip(z_mod_list, z_obs_list, self.layer_weights):
            total = total + w * F.mse_loss(z_mod, z_obs)
        return total

    def _iter_trace_chunks(self, n):
        """Yield (start, end) chunks over receiver dimension."""
        if self.trace_chunk is None or self.trace_chunk <= 0 or self.trace_chunk >= n:
            yield 0, n
        else:
            for s in range(0, n, self.trace_chunk):
                e = min(s + self.trace_chunk, n)
                yield s, e

    def _should_debug_plot(self, shot_idx, side):
        """Decide whether to produce a debug plot for (shot_idx, side).

        side : "obs" or "mod".
        """
        if not self.debug_plot:
            return False
        key = (int(shot_idx), str(side))
        if key in self._debug_plotted:
            return False
        if self.debug_plot_shots is not None:
            if int(shot_idx) not in self.debug_plot_shots:
                return False
        else:
            # No explicit list: only plot the first shot we see on each side.
            already_on_side = any(s == side for (_, s) in self._debug_plotted)
            if already_on_side:
                return False
        return True

    def _debug_plot_input(self, x_bct, shot_idx, side, trace_offset=0):
        """Save a PNG of what actually enters SeisLM for one shot.

        Parameters
        ----------
        x_bct : torch.Tensor
            Shape ``[B, C, T]``, the tensor about to be fed to SeisLM. This is
            already post-Op masking (so any time-domain gain folded into Op is
            visible here), post-resample, and post-per-trace standardization.
        shot_idx : int
            Shot index for filename and title.
        side : str
            "obs" or "mod" — used in filename/title to disambiguate.
        trace_offset : int, optional
            Starting receiver index of this batch within the full ``[0, nr)``
            range, used so that overlaid trace labels are absolute receiver
            indices when plotting only the first chunk of a chunked pass.
        """
        # Lazy import: keep the loss class importable in environments without
        # matplotlib (e.g. some HPC nodes).
        try:
            import os as _os
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"[SeisLMFeatureLoss debug] matplotlib unavailable: {exc}",
                  flush=True)
            return

        outdir = self.debug_plot_dir if self.debug_plot_dir is not None else "."
        try:
            _os.makedirs(outdir, exist_ok=True)
        except Exception as exc:
            print(f"[SeisLMFeatureLoss debug] cannot create {outdir}: {exc}",
                  flush=True)
            return

        # Move to CPU + detach, take channel 0 (channels are duplicates of
        # the same single-component trace, so any channel is representative).
        x_np = x_bct.detach().to("cpu", dtype=torch.float32).numpy()    # [B, C, T]
        if x_np.ndim != 3:
            print(f"[SeisLMFeatureLoss debug] unexpected ndim={x_np.ndim}, skipping",
                  flush=True)
            return
        B, C, T = x_np.shape
        x_b_t = x_np[:, 0, :]                                            # [B, T]

        n_overlay = max(1, min(self.debug_plot_n_traces, B))
        idxs = np.linspace(0, B - 1, n_overlay).astype(int)

        fig, axs = plt.subplots(
            1, 2, figsize=(13, 4.6),
            gridspec_kw={"width_ratios": [1.0, 1.6]},
        )

        # Panel 1: imshow of the full [B, T] slab to see masking/structure
        clip = float(np.percentile(np.abs(x_b_t), 99)) if x_b_t.size else 1.0
        if clip <= 0:
            clip = 1.0
        ax0 = axs[0]
        im = ax0.imshow(
            x_b_t, aspect="auto", cmap="gray",
            vmin=-clip, vmax=clip,
            extent=[0, T, B + trace_offset, trace_offset],
        )
        ax0.set_xlabel("Time sample")
        ax0.set_ylabel("Receiver index")
        ax0.set_title(f"{side} input to SeisLM — shot {shot_idx}\n[B, T]={B}x{T}")
        fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

        # Panel 2: overlaid traces at picked indices
        ax1 = axs[1]
        for k, ridx in enumerate(idxs):
            ax1.plot(x_b_t[ridx], lw=0.8,
                     label=f"r={ridx + trace_offset}")
        ax1.axhline(0, color="k", lw=0.4, alpha=0.4)
        ax1.set_xlabel("Time sample")
        ax1.set_ylabel("Standardized amplitude")
        ax1.set_title(
            f"{n_overlay} traces (post-Op, "
            f"resample={'on' if self.resample else 'off'}, "
            f"normalize={'on' if self.normalize_traces else 'off'})"
        )
        ax1.legend(fontsize=7, ncol=2, loc="upper right")
        ax1.grid(True, alpha=0.3)

        fig.suptitle(
            f"SeisLMFeatureLoss debug — side={side}, shot={shot_idx}",
            fontsize=11, y=1.02,
        )
        fig.tight_layout()
        fname = _os.path.join(
            outdir, f"seislm_input_{side}_shot{int(shot_idx):04d}.png"
        )
        try:
            fig.savefig(fname, dpi=130, bbox_inches="tight")
            print(f"[SeisLMFeatureLoss debug] wrote {fname}", flush=True)
        except Exception as exc:
            print(f"[SeisLMFeatureLoss debug] failed to write {fname}: {exc}",
                  flush=True)
        finally:
            plt.close(fig)

        self._debug_plotted.add((int(shot_idx), str(side)))

    # ------------------------------------------------------------------
    # Adjoint-source debug capture
    # ------------------------------------------------------------------
    def _should_debug_adjsrc(self, shot_idx):
        """Decide whether to save adjoint sources for this shot."""
        if not self.debug_adjsrc:
            return False
        i = int(shot_idx)
        if i in self._adjsrc_saved:
            return False
        if self.debug_adjsrc_shots is not None:
            if i not in self.debug_adjsrc_shots:
                return False
        else:
            # No explicit list: only save the first shot we see.
            if len(self._adjsrc_saved) > 0:
                return False
        return True

    def _save_adjsrc(self, arr_nt_nr, shot_idx, kind):
        """Save adjoint source for one shot.

        Writes
            <dir>/adjsrc_<kind>_shot<NNNN>.npy   (raw (nt, nr) float32 array)
            <dir>/adjsrc_<kind>_shot<NNNN>.png   (2-panel diagnostic)

        Parameters
        ----------
        arr_nt_nr : numpy.ndarray, shape (nt, nr)
            The adjoint source on the wavefield-at-receivers grid.
        shot_idx : int
            Shot index (used in filenames and titles).
        kind : str
            "post_mask" — dL/dx_modelled (what Devito back-propagates), or
            "pre_mask"  — dL/dy (what SeisLM features want, pre-operator).
        """
        try:
            import os as _os
            outdir = self.debug_adjsrc_dir if self.debug_adjsrc_dir is not None else "."
            _os.makedirs(outdir, exist_ok=True)
        except Exception as exc:
            print(f"[SeisLMFeatureLoss adjsrc] cannot create output dir: {exc}",
                  flush=True)
            return

        arr = np.ascontiguousarray(arr_nt_nr, dtype=np.float32)
        base = f"adjsrc_{kind}_shot{int(shot_idx):04d}"

        npy_path = _os.path.join(outdir, base + ".npy")
        try:
            with open(npy_path, "wb") as fh:
                np.save(fh, arr, allow_pickle=False)
            print(f"[SeisLMFeatureLoss adjsrc] wrote {npy_path}", flush=True)
        except Exception as exc:
            print(f"[SeisLMFeatureLoss adjsrc] failed to write {npy_path}: {exc}",
                  flush=True)
            return

        # Diagnostic PNG. Same 2-panel layout as the input-debug plot:
        # (left) imshow of the (nt, nr) slab; (right) overlaid traces.
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"[SeisLMFeatureLoss adjsrc] matplotlib unavailable: {exc}",
                  flush=True)
            return

        nt, nr = arr.shape
        n_overlay = max(1, min(self.debug_adjsrc_n_traces, nr))
        idxs = np.linspace(0, nr - 1, n_overlay).astype(int)

        clip = float(np.percentile(np.abs(arr), 99)) if arr.size else 1.0
        if clip <= 0:
            clip = 1.0

        fig, axs = plt.subplots(
            1, 2, figsize=(13, 4.6),
            gridspec_kw={"width_ratios": [1.0, 1.6]},
        )
        ax0 = axs[0]
        # arr is (nt, nr); show as (receiver, time) via transpose so the
        # layout matches the input-debug plots.
        im = ax0.imshow(
            arr.T, aspect="auto", cmap="seismic",
            vmin=-clip, vmax=clip,
            extent=[0, nt, nr, 0],
        )
        ax0.set_xlabel("Time sample")
        ax0.set_ylabel("Receiver index")
        ax0.set_title(f"adjoint source ({kind}) — shot {shot_idx}\n[nt, nr]={nt}x{nr}")
        fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

        ax1 = axs[1]
        for ridx in idxs:
            ax1.plot(arr[:, ridx], lw=0.8, label=f"r={ridx}")
        ax1.axhline(0, color="k", lw=0.4, alpha=0.4)
        ax1.set_xlabel("Time sample")
        ax1.set_ylabel("Adjoint amplitude")
        ax1.set_title(f"{n_overlay} traces — adjsrc ({kind})")
        ax1.legend(fontsize=7, ncol=2, loc="upper right")
        ax1.grid(True, alpha=0.3)

        fig.suptitle(
            f"SeisLMFeatureLoss adjoint source — shot {shot_idx}, kind={kind}",
            fontsize=11, y=1.02,
        )
        fig.tight_layout()
        png_path = _os.path.join(outdir, base + ".png")
        try:
            fig.savefig(png_path, dpi=130, bbox_inches="tight")
            print(f"[SeisLMFeatureLoss adjsrc] wrote {png_path}", flush=True)
        except Exception as exc:
            print(f"[SeisLMFeatureLoss adjsrc] failed to write {png_path}: {exc}",
                  flush=True)
        finally:
            plt.close(fig)

    def _get_observed_embedding(self, i):
        """Compute (or get cached) observed embedding for shot i.

        Returns
        -------
        list of torch.Tensor
            One tensor per requested layer, each [nr, D] or [nr, T', D].
        """
        if self.cache_obs_embeddings and i in self._obs_cache:
            return self._obs_cache[i]

        z_obs_all = self._load_observed_embedding_from_disk(i)
        if z_obs_all is not None:
            if self.cache_obs_embeddings:
                self._obs_cache[i] = z_obs_all
            return z_obs_all

        # ------------------------------------------------------------------
        # Apply the *same* operator to the observed data that the modelled
        # side gets in __call__:
        #     modelled side : y_flat = Op_i @ x_flat       (Torch path)
        #     observed side : b_flat = Op_i @ self.b[i]    (NumPy path, here)
        # If Op is None, the observed data is used as-is. This keeps the two
        # sides comparable end-to-end — without it, masked modelled data
        # would be compared against unmasked observed data and the SeisLM
        # feature loss would push toward a target that cannot be reached by
        # any velocity model.
        # ------------------------------------------------------------------
        b_flat = np.asarray(self.b[i]).ravel()
        if self.Op is not None:
            Op_i = self.Op[i] if hasattr(self.Op, "__getitem__") else self.Op
            b_flat = Op_i @ b_flat                                         # masked, still flat

        # b_flat -> (nt, nr) -> transpose -> [nr, nt] numpy (mirrors modelled side)
        b_shot_np = b_flat.reshape(self.nt, self.nr).T.astype(np.float32)  # [nr, nt]

        # Resample in numpy before converting to torch (no graph needed for observed)
        if self.resample:
            b_shot_np = self._resample_shot(b_shot_np)                     # [nr, nt_rs]

        b_shot = torch.from_numpy(b_shot_np)                               # [nr, nt(_rs)]

        # Accumulate per-layer embeddings across chunks: list of lists
        n_layers = len(self.feature_layers) if self.feature_layers is not None else 1
        zs_per_layer = [[] for _ in range(n_layers)]

        plot_this_shot = self._should_debug_plot(i, "obs")

        with torch.no_grad():
            for chunk_k, (s, e) in enumerate(self._iter_trace_chunks(self.nr)):
                b_chunk = self._prepare_batch_3c(b_shot[s:e, :])           # [Bchunk, 3, T]

                # Debug: dump the very tensor that's about to enter SeisLM,
                # for the first chunk of the chosen shot only.
                if plot_this_shot and chunk_k == 0:
                    self._debug_plot_input(b_chunk, shot_idx=i, side="obs",
                                           trace_offset=s)

                z_list = self._encode(b_chunk, device_idx=0)               # list of [Bchunk, D]
                for li, z_chunk in enumerate(z_list):
                    z_chunk = z_chunk.detach()
                    if self.cache_obs_on_cpu:
                        z_chunk = z_chunk.cpu()
                    zs_per_layer[li].append(z_chunk)

        # Concatenate chunks for each layer: list of [nr, D]
        z_obs_all = [torch.cat(zs, dim=0) for zs in zs_per_layer]

        self._save_observed_embedding_to_disk(i, z_obs_all)

        if self.cache_obs_embeddings:
            self._obs_cache[i] = z_obs_all
        return z_obs_all

    def _postprocess_grad(self, g, x):
        """Resample gradient back to original time domain and apply dtype/layout fixes.

        Parameters
        ----------
        g : np.ndarray
            Raw gradient from autograd:
              - Op is None or (Op is not None and resample=True): shape [nr, nt_rs]
              - Op is not None and resample=False: flat shape matching x
        x : np.ndarray
            Original input, used to match output dtype.

        Returns
        -------
        np.ndarray
            If Op is None or resample=True: shape [nt, nr] in original time domain.
            If Op is not None and resample=False: flat shape matching x.
        """
        # When resample=True and Op is not None, self.x was reassigned to the resampled
        # y_shot leaf, so gradient shape is [nr, nt_rs] and needs same treatment as Op is None.
        needs_spatial_postprocess = (self.Op is None) or (self.resample and self.Op is not None)

        if needs_spatial_postprocess:
            if self.resample:
                g = self._resample_grad_back(g)    # [nr, nt_rs] -> [nr, nt]
            g = g.T                                # [nr, nt] -> [nt, nr]

        if hasattr(x, "dtype"):
            g = g.astype(x.dtype, copy=False)

        return g

    # ------------------------------------------------------------------
    # Public API (SoftDTW-like)
    # ------------------------------------------------------------------
    def __call__(self, x, i):
        """Compute SeisLM feature loss for shot i.

        If trace_chunk is set, the method performs chunked forward/backward and caches the
        gradient internally to avoid GPU OOM.  When multiple devices are configured the
        chunks are distributed across GPUs using a thread pool (one thread per GPU), giving
        an approximately N-times speedup for N GPUs.  Multi-GPU is only active when
        ``resample=False``; otherwise falls back to single-GPU sequential chunking.

        Parameters
        ----------
        x : numpy.ndarray
            If Op is None: flattened modeled shot gather of shape (nt*nr,)
            If Op is not None: optimization variable to be mapped by Op
        i : int
            Shot index.

        Returns
        -------
        float
            Scalar loss value.
        """
        # import resource, torch
        # rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        # if torch.cuda.is_available():
        #     gpu = torch.cuda.memory_allocated(0) / 1024**3
        #     print(f"[shot {i}] RSS={rss:.1f} GB  GPU={gpu:.2f} GB", flush=True)
        self.log_fn(f"shot {i}")
        
        # Explicitly free previous shot's graph before building new one
        if self.f is not None:
            self.f = None
        if self.x is not None:
            self.x = None
        self._loss_value = None
        self._grad_cache = None
        # Reset adjoint-source capture state for the new shot.
        self._adjsrc_capture_active = self._should_debug_adjsrc(i)
        self._adjsrc_pre_mask_grad = None
        self._adjsrc_current_shot = int(i) if self._adjsrc_capture_active else None
        # force torch to release the old graph now
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Observed embeddings (fixed): list of [nr, D] per layer
        z_obs_all = self._get_observed_embedding(i)

        # ----------------------------------------------------------------
        # Build modeled data variables
        #   y_shot        – tensor used by the fast path (may have grad_fn)
        #   y_shot_leaf   – detached CPU leaf used by the multi-GPU chunked path
        #   y_shot_conn   – connected to self.x via Op (for VJP after chunking)
        # ----------------------------------------------------------------
        y_shot_conn = None

        if self.Op is not None:
            x_flat = torch.from_numpy(x).to(torch.float32).requires_grad_()
            self.x  = x_flat
            # Op_i    = self.Op[i] if isinstance(self.Op, list) else self.Op
            Op_i = self.Op[i] if hasattr(self.Op, '__getitem__') else self.Op

            y       = TorchOperator(Op_i).apply(x_flat)
            y_shot_conn = y.reshape(self.nt, self.nr).T      # [nr, nt], has grad_fn
            del Op_i

            # If we're capturing the adjoint source for this shot, retain the
            # gradient on y_shot_conn so we can read dL/dy after backward().
            # Multi-GPU path captures dL/dy directly from `grad_leaf` and
            # ignores this; harmless to retain in either case.
            if self._adjsrc_capture_active:
                y_shot_conn.retain_grad()
                # Stash for grad() to read .grad off after backward().
                self._adjsrc_y_shot_conn = y_shot_conn
            else:
                self._adjsrc_y_shot_conn = None

            if self.resample:
                y_shot_np   = self._resample_shot(y_shot_conn.detach().cpu().numpy())
                y_shot      = torch.from_numpy(y_shot_np).requires_grad_()
                self.x      = y_shot                         # reassign (original behaviour)
                y_shot_leaf = y_shot
            else:
                y_shot      = y_shot_conn                    # used by fast path
                y_shot_leaf = y_shot_conn.detach().requires_grad_()   # CPU leaf for chunked
        else:
            x_shot_np = x.reshape(self.nt, self.nr).T.astype(np.float32)
            if self.resample:
                x_shot_np = self._resample_shot(x_shot_np)
            x_shot    = torch.from_numpy(x_shot_np)
            self.x    = x_shot.to(device=self.model_device, dtype=torch.float32).requires_grad_()
            y_shot      = self.x
            y_shot_leaf = y_shot                             # IS self.x, already a leaf

        # ----------------------------------------------------------------
        # Fast path: no chunking (single GPU, higher memory)
        # ----------------------------------------------------------------
        if self.trace_chunk is None or self.trace_chunk <= 0 or self.trace_chunk >= self.nr:
            z_obs_dev = [z.to(device=self.model_device, dtype=self.model_dtype)
                         for z in z_obs_all]
            y_batch   = self._prepare_batch_3c(y_shot)
            if self._should_debug_plot(i, "mod"):
                self._debug_plot_input(y_batch, shot_idx=i, side="mod",
                                       trace_offset=0)
            z_mod_list = self._encode(y_batch, device_idx=0)
            self.f    = self._compute_layer_loss(z_mod_list, z_obs_dev)
            self._loss_value = float(self.f.detach().cpu().item())
            # Free intermediate tensors before returning
            del y_batch, z_mod_list, z_obs_dev
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return self._loss_value

        # ----------------------------------------------------------------
        # Chunked path
        # Multi-GPU parallel when: len(devices) > 1 AND resample=False
        # Falls back to single-GPU sequential otherwise (original behaviour).
        # ----------------------------------------------------------------
        use_multi_gpu = (len(self._devices) > 1) and (not self.resample)

        # Debug plot for the modelled side: build the same [B, C, T] tensor
        # the encoder sees on the first chunk (no encoder forward is run here,
        # so this is essentially free). y_shot is the masked, possibly-resampled
        # [nr, nt(_rs)] tensor; y_shot_leaf has the same content for the
        # multi-GPU path.
        if self._should_debug_plot(i, "mod"):
            with torch.no_grad():
                first_end = self.trace_chunk if (
                    self.trace_chunk and 0 < self.trace_chunk < self.nr
                ) else self.nr
                y_for_plot = (y_shot_leaf if use_multi_gpu else y_shot)[:first_end, :]
                y_batch_dbg = self._prepare_batch_3c(y_for_plot)
                self._debug_plot_input(y_batch_dbg, shot_idx=i, side="mod",
                                       trace_offset=0)
                del y_batch_dbg

        if use_multi_gpu:
            self._call_chunked_multigpu(y_shot_leaf, y_shot_conn, z_obs_all, x)
        else:
            self._call_chunked_singlegpu(y_shot, z_obs_all, x)

        return self._loss_value

    # ------------------------------------------------------------------
    # Chunked helpers
    # ------------------------------------------------------------------
    def _call_chunked_singlegpu(self, y_shot, z_obs_all, x):
        """Original sequential single-GPU chunked forward+backward."""
        if self.x.grad is not None:
            self.x.grad.zero_()

        total_loss  = 0.0
        total_count = 0

        for s, e in self._iter_trace_chunks(self.nr):
            bsz = e - s

            y_chunk = y_shot[s:e, :]
            y_batch = self._prepare_batch_3c(y_chunk)
            z_mod_list = self._encode(y_batch, device_idx=0)

            z_obs_chunk_list = [
                z_obs_all[li][s:e].to(device=self.model_device, dtype=self.model_dtype)
                for li in range(len(z_obs_all))
            ]

            loss_chunk = self._compute_layer_loss(z_mod_list, z_obs_chunk_list)

            total_loss  += float(loss_chunk.detach().cpu().item()) * bsz
            total_count += bsz

            (loss_chunk * (bsz / float(self.nr))).backward()

            del y_chunk, y_batch, z_mod_list, z_obs_chunk_list, loss_chunk

        self._loss_value = total_loss / max(total_count, 1)

        g = self.x.grad.detach().cpu().numpy()

        # Adjoint-source capture (single-GPU chunked path).
        if self._adjsrc_capture_active:
            yc = getattr(self, "_adjsrc_y_shot_conn", None)
            if yc is not None and yc.grad is not None:
                self._adjsrc_y_grad = yc.grad.detach().cpu()
            self._dispatch_adjsrc_save(g)
            self._adjsrc_y_shot_conn = None

        g = self._postprocess_grad(g, x)
        self._grad_cache = g
        self.f = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _call_chunked_multigpu(self, y_shot_leaf, y_shot_conn, z_obs_all, x):
        """Parallel multi-GPU chunked forward+backward.

        Chunks are distributed round-robin across self._devices. One worker
        thread per GPU processes its assigned chunks sequentially while all
        GPUs run in parallel.

        Gradient correctness is maintained by computing dL/dy for each local
        chunk independently, assembling the full [nr, nt] dL/dy on CPU, then
        applying a single VJP through y_shot_conn when Op is present.
        """
        n_devices   = len(self._devices)
        chunks      = list(self._iter_trace_chunks(self.nr))
        chunks_by_device = [[] for _ in range(n_devices)]
        for ci, chunk in enumerate(chunks):
            chunks_by_device[ci % n_devices].append(chunk)

        def _process_device(device_idx, device_chunks):
            device = self._devices[device_idx]
            local_loss = 0.0
            local_count = 0
            local_grads = []

            with self._gpu_locks[device_idx]:
                for s, e in device_chunks:
                    bsz = e - s

                    # Local leaf: return the chunk gradient explicitly instead
                    # of accumulating into one shared CPU leaf from all threads.
                    y_chunk = (
                        y_shot_leaf[s:e, :]
                        .detach()
                        .to(device=device, dtype=self.model_dtype)
                        .requires_grad_(True)
                    )
                    y_batch = self._prepare_batch_3c(y_chunk)
                    z_mod_list = self._encode(y_batch, device_idx=device_idx)

                    z_obs_chunk_list = [
                        z_obs_all[li][s:e].to(device=device, dtype=self.model_dtype)
                        for li in range(len(z_obs_all))
                    ]

                    loss_chunk = self._compute_layer_loss(z_mod_list, z_obs_chunk_list)
                    local_loss += float(loss_chunk.detach().cpu().item()) * bsz
                    local_count += bsz

                    grad_chunk = torch.autograd.grad(
                        loss_chunk * (bsz / float(self.nr)), y_chunk
                    )[0]
                    local_grads.append((s, e, grad_chunk.detach().cpu()))

                    del y_chunk, y_batch, z_mod_list, z_obs_chunk_list, loss_chunk, grad_chunk

                if device.type == "cuda":
                    torch.cuda.synchronize(device)

            return local_loss, local_count, local_grads

        total_loss = 0.0
        total_count = 0
        grad_leaf = torch.zeros_like(y_shot_leaf, device="cpu", dtype=torch.float32)

        with ThreadPoolExecutor(max_workers=n_devices) as executor:
            futures = [
                executor.submit(_process_device, device_idx, device_chunks)
                for device_idx, device_chunks in enumerate(chunks_by_device)
                if device_chunks
            ]
            for fut in futures:
                local_loss, local_count, local_grads = fut.result()
                total_loss += local_loss
                total_count += local_count
                for s, e, grad_chunk in local_grads:
                    grad_leaf[s:e, :] = grad_chunk

        self._loss_value = total_loss / max(total_count, 1)

        # If we're capturing the adjoint source for this shot, grad_leaf holds
        # dL/dy on the (resampled or not) [nr, nt(_rs)] grid -- this is the
        # pre-mask adjoint source as seen by SeisLM features. Stash a CPU copy
        # for grad() to save (after backward populates self.x.grad too).
        if self._adjsrc_capture_active:
            self._adjsrc_pre_mask_grad = grad_leaf.detach().cpu().clone()

        if y_shot_conn is not None:
            # VJP: propagate the assembled dL/dy back through Op to self.x.
            # y_shot_conn = (Op @ x_flat).reshape(nt, nr).T, connected to self.x.
            grad_tensor = grad_leaf.to(y_shot_conn.device)
            y_shot_conn.backward(grad_tensor)

            # self.x == x_flat (the flat Op-input leaf), grad is now filled.
            g = self.x.grad.detach().cpu().numpy()
            # _postprocess_grad for (Op not None, resample=False): dtype cast only.
            if hasattr(x, "dtype"):
                g = g.astype(x.dtype, copy=False)
        else:
            # Direct-shot mode: assembled dL/dy is already wrt the modeled shot gather.
            g = grad_leaf.numpy()
            g = self._postprocess_grad(g, x)

        # Adjoint-source capture (multi-GPU chunked path). Pre-mask was stashed
        # above from grad_leaf; post-mask is the g we just assembled (flat for
        # Op-not-None, (nt, nr) for Op-None).
        if self._adjsrc_capture_active:
            self._dispatch_adjsrc_save(g)

        self._grad_cache = g
        self.f = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _dispatch_adjsrc_save(self, post_mask_grad_flat):
        """Save adjoint sources for the current shot if capture is active.

        Called from each of the three code paths (fast, chunked single-GPU,
        chunked multi-GPU) right after self.x.grad is populated.

        Parameters
        ----------
        post_mask_grad_flat : numpy.ndarray
            The post-mask adjoint source `self.x.grad`. For Op-not-None paths
            this is flat (nt*nr,); for Op-None paths it's already (nt, nr).
            This is what Devito's adjoint solve receives.
        """
        if not self._adjsrc_capture_active or self._adjsrc_current_shot is None:
            return

        shot = self._adjsrc_current_shot

        # Post-mask: dL/dx_modelled (what Devito receives)
        try:
            arr = np.asarray(post_mask_grad_flat).reshape(self.nt, self.nr).astype(np.float32)
            self._save_adjsrc(arr, shot, kind="post_mask")
        except Exception as exc:
            print(f"[SeisLMFeatureLoss adjsrc] post_mask save failed: {exc}",
                  flush=True)

        # Pre-mask: dL/dy, the adjoint source SeisLM features want before Op
        # attenuates them. Available via grad_leaf (multi-GPU) or via the
        # retain_grad'd y_shot_conn (fast / single-GPU chunked).
        pre_mask_arr = None
        if self._adjsrc_pre_mask_grad is not None:
            # Multi-GPU path stashed grad_leaf in __call__ as a torch.Tensor
            # of shape [nr, nt(_rs)].
            pmg = self._adjsrc_pre_mask_grad
            try:
                pre_mask_arr = pmg.numpy().T.astype(np.float32, copy=True)  # -> (nt, nr)
            except Exception as exc:
                print(f"[SeisLMFeatureLoss adjsrc] multi-gpu pre_mask conv failed: {exc}",
                      flush=True)
        elif getattr(self, "_adjsrc_y_grad", None) is not None:
            # Fast / single-GPU chunked: y_shot_conn.grad was retained.
            yg = self._adjsrc_y_grad   # shape [nr, nt]
            try:
                pre_mask_arr = yg.detach().cpu().numpy().T.astype(np.float32, copy=True)
            except Exception as exc:
                print(f"[SeisLMFeatureLoss adjsrc] fast/single-gpu pre_mask conv failed: {exc}",
                      flush=True)

        if pre_mask_arr is not None:
            try:
                self._save_adjsrc(pre_mask_arr, shot, kind="pre_mask")
            except Exception as exc:
                print(f"[SeisLMFeatureLoss adjsrc] pre_mask save failed: {exc}",
                      flush=True)

        # Mark as saved; clear the stashed tensors so we don't pin memory.
        self._adjsrc_saved.add(shot)
        self._adjsrc_capture_active = False
        self._adjsrc_pre_mask_grad = None
        self._adjsrc_current_shot = None
        if hasattr(self, "_adjsrc_y_grad"):
            self._adjsrc_y_grad = None

    def grad(self, x, i):
        """Return gradient from last __call__.

        In chunked mode, gradient is already computed/cached in __call__.
        In non-chunked mode, this triggers backward().

        Returns
        -------
        numpy.ndarray
            If Op is None or resample=True:
                gradient wrt modeled shot gather, shape (nt, nr)
                (matching SoftDTW-style convention), in original time domain.
            If Op is not None and resample=False:
                gradient wrt optimization variable x (same flat shape as x).
        """
        if self._grad_cache is not None:
            return self._grad_cache

        if self.f is None or self.x is None:
            raise RuntimeError(
                "Call the loss first via `f = loss(x, i)` before calling `grad(x, i)`.")

        self.f.backward()
        g = self.x.grad.detach().cpu().numpy()

        # Adjoint-source capture (fast path): pre-mask is on y_shot_conn.grad
        # (retained), post-mask is the flat self.x.grad we just read.
        if self._adjsrc_capture_active:
            yc = getattr(self, "_adjsrc_y_shot_conn", None)
            if yc is not None and yc.grad is not None:
                self._adjsrc_y_grad = yc.grad.detach().cpu()
            self._dispatch_adjsrc_save(g)
            self._adjsrc_y_shot_conn = None

        g = self._postprocess_grad(g, x)

        return g
