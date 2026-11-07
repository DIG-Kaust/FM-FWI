__all__ = ["SeisLMFeatureLoss", "L2", "HybridLoss"]

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

import copy
import threading
from concurrent.futures import ThreadPoolExecutor


class SeisLMFeatureLoss:
    """SeisLM feature-space loss for seismic traces (chunked).

    Treats receiver traces as batch [B, T], duplicates to [B, C, T], and
    compares SeisLM embeddings using MSE after temporal mean pooling.

    Parameters
    ----------
    seislm_model : torch.nn.Module
        Pretrained SeisLM model used for feature extraction, exposing `.wav2vec2(...)`.
    Op : pylops.LinearOperator, optional
        Linear operator.
    b : numpy.ndarray
        Observed shot gathers. b[i] must be flattenable to (nt, nr) for shot index i.
    nr : int
        Number of receivers (traces) per shot.
    nt : int
        Number of time samples per trace.
    t_sub : int, optional
        Subsampling factor along the time axis applied before encoding. Default 1.
    device : str, optional
        Device for SeisLM inference, e.g. "cpu" or "cuda:0". Default "cpu".
    devices : list of str or None, optional
        Multi-GPU device list, e.g. ["cuda:0", "cuda:1"]. Trace chunks are distributed
        across GPUs in parallel for ~N× encoding speedup. Only active when resample=False.
        Falls back to single-device when None.
    pool : str, optional
        Pooling applied to embeddings over the token dimension.
        "mean" (default) reduces [B, T', D] -> [B, D]; "none" keeps [B, T', D].
    normalize_traces : bool, optional
        If True, standardize each trace to zero mean and unit std before encoding. Default True.
    eps : float, optional
        Stability epsilon for trace standardization to avoid division by zero. Default 1e-6.
    cache_obs_embeddings : bool, optional
        Cache observed embeddings by shot index to avoid recomputing them across FWI
        iterations. Default True.
    cache_obs_on_cpu : bool, optional
        Store cached observed embeddings on CPU to reduce GPU memory usage. Default True.
    trace_chunk : int or None, optional
        Number of receiver traces per SeisLM forward pass. Set to a small value (e.g. 8, 16,
        32) to avoid GPU OOM. If None, all receivers are processed in a single pass. Default None.
    dt : float or None, optional
        Time sampling interval of the input data in seconds. Required when
        resample=True. Default None.
    resample : bool, optional
        If True, resample traces from 1/dt Hz to fs_target Hz before encoding, and resample
        the gradient back to the original time dimension. Default False.
    fs_target : float, optional
        Target sampling rate in Hz for resampling. Should match the SeisBench training
        sampling rate. Default 100.0.
    resample_denom_limit : int, optional
        Maximum denominator for the rational approximation of the resampling ratio passed to
        resample_poly. Default 1000.
    feature_layers : int, list of int, or None, optional
        Transformer hidden layers to use for feature comparison. None uses only the final
        layer (last_hidden_state). Indices are 0-based (0 = first transformer layer output);
        wav2vec2 has 12 layers. Default None.
    layer_weights : list of float or None, optional
        Per-layer weights for the multi-layer MSE loss. Must match the length of
        feature_layers. Normalized to sum to 1 internally. If None, all layers are
        weighted equally. Default None.
    n_input_channels : int, optional
        Number of input channels expected by the model. The single seismic trace is
        duplicated this many times to form the [B, C, T] input. Default 3.
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
        trace_chunk=None,
        dt=None,
        resample=False,
        fs_target=100.0,
        resample_denom_limit=1000,
        feature_layers=None,
        layer_weights=None,
        n_input_channels=3,
    ):
        self.n_input_channels = n_input_channels
        self.Op = Op
        self.b = b
        self.nr, self.nt = nr, nt
        self.t_sub = t_sub
        self.device = device
        self.pool = pool
        self.normalize_traces = normalize_traces
        self.eps = eps
        self.cache_obs_embeddings = cache_obs_embeddings
        self.cache_obs_on_cpu = cache_obs_on_cpu
        self.trace_chunk = trace_chunk

        self.resample = resample
        self.fs_target = fs_target
        if resample:
            if dt is None:
                raise ValueError("dt must be provided when resample=True.")
            fs_input = 1.0 / float(dt)
            ratio = Fraction(float(fs_target / fs_input)).limit_denominator(resample_denom_limit)
            self._resample_up = ratio.numerator
            self._resample_down = ratio.denominator
            # Dummy resample to get exact resampled length, avoiding off-by-one from rational approx
            dummy = np.zeros((1, nt), dtype=np.float32)
            self._nt_rs = scipy_resample_poly(
                dummy, self._resample_up, self._resample_down, axis=-1
            ).shape[-1]
            # Resampling as a fixed linear operator R [nt_rs, nt] (columns = resampled unit
            # impulses), applied inside autograd so the adjoint source is exact.
            eye = np.eye(nt, dtype=np.float64)
            R = scipy_resample_poly(eye, self._resample_up, self._resample_down, axis=0)[: self._nt_rs]
            R = np.pad(R, ((0, self._nt_rs - R.shape[0]), (0, 0)))
            self._R = torch.from_numpy(R.astype(np.float32))
        else:
            self._resample_up = None
            self._resample_down = None
            self._nt_rs = None
            self._R = None

        if feature_layers is None:
            self.feature_layers = None
            self.layer_weights = [1.0]
        elif isinstance(feature_layers, int):
            self.feature_layers = [feature_layers]
        else:
            self.feature_layers = list(feature_layers)

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

        if devices is not None and len(devices) >= 1:
            self._devices = [torch.device(d) for d in devices]
        else:
            self._devices = [torch.device(device)]

        self.seislm = seislm_model.to(self._devices[0]).eval()
        for p in self.seislm.parameters():
            p.requires_grad_(False)

        self._seislm_replicas = [self.seislm]
        for dev in self._devices[1:]:
            replica = copy.deepcopy(seislm_model).to(dev).eval()
            for p in replica.parameters():
                p.requires_grad_(False)
            self._seislm_replicas.append(replica)

        self._gpu_locks = [threading.Lock() for _ in self._devices]

        self.model_param = next(self.seislm.parameters())
        self.model_dtype = self.model_param.dtype
        self.model_device = self._devices[0]

        self.x = None
        self.f = None
        self._loss_value = None
        self._obs_cache = {}
        self._grad_cache = None
        self._last_args = None

    def clear_cache(self):
        self._obs_cache.clear()

    def _resample_shot(self, shot_nr_nt_np):
        """Resample [nr_chunk, nt] -> [nr_chunk, nt_rs] along time axis."""
        g = scipy_resample_poly(
            shot_nr_nt_np, self._resample_up, self._resample_down, axis=-1
        ).astype(np.float32)
        nt_out = g.shape[-1]
        if nt_out > self._nt_rs:
            g = g[:, :self._nt_rs]
        elif nt_out < self._nt_rs:
            g = np.pad(g, ((0, 0), (0, self._nt_rs - nt_out)))
        return g

    def _trace_standardize(self, x):
        """Per-trace standardization over time. x: [B, C, T]."""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.eps)
        return (x - mean) / std

    def _prepare_batch_3c(self, shot_nr_nt):
        """Convert [nr_chunk, nt] -> [nr_chunk, C, nt_sub], normalized if requested."""
        x = shot_nr_nt[:, :: self.t_sub]
        x = x.unsqueeze(1).repeat(1, self.n_input_channels, 1)
        x = x.to(torch.float32)
        if self.normalize_traces:
            x = self._trace_standardize(x)
        return x

    def _pool(self, z):
        """Pool embeddings: [B, T', D] -> [B, D] (mean) or [B, T', D] (none)."""
        if self.pool == "mean":
            return z.mean(dim=1)
        elif self.pool == "none":
            return z
        else:
            raise ValueError(f"Unsupported pool='{self.pool}'. Use 'mean' or 'none'.")

    def _encode(self, x_bct, device_idx=0):
        """Run SeisLM encoder on replica at device_idx. Returns list of pooled embeddings per layer."""
        model  = self._seislm_replicas[device_idx]
        device = self._devices[device_idx]
        x_bct  = x_bct.to(device=device, dtype=self.model_dtype)

        need_hidden = self.feature_layers is not None
        out = model.wav2vec2(x_bct, output_hidden_states=need_hidden)

        if not need_hidden:
            return [self._pool(out.last_hidden_state)]

        # hidden_states[0] = embedding layer; [1..n] = transformer layers
        hidden_states = out.hidden_states
        return [self._pool(hidden_states[layer_idx + 1]) for layer_idx in self.feature_layers]

    def _compute_layer_loss(self, z_mod_list, z_obs_list):
        """Weighted MSE loss across all requested layers."""
        dev   = z_mod_list[0].device
        dtype = z_mod_list[0].dtype
        total = torch.tensor(0.0, device=dev, dtype=dtype)
        for z_mod, z_obs, w in zip(z_mod_list, z_obs_list, self.layer_weights):
            total = total + w * F.mse_loss(z_mod, z_obs)
        return total

    def _iter_trace_chunks(self, n):
        if self.trace_chunk is None or self.trace_chunk <= 0 or self.trace_chunk >= n:
            yield 0, n
        else:
            for s in range(0, n, self.trace_chunk):
                yield s, min(s + self.trace_chunk, n)

    def _get_observed_embedding(self, i):
        """Compute (or return cached) observed embeddings for shot i."""
        if self.cache_obs_embeddings and i in self._obs_cache:
            return self._obs_cache[i]

        b_shot_np = self.b[i].reshape(self.nt, self.nr).T.astype(np.float32)
        if self.resample:
            b_shot_np = self._resample_shot(b_shot_np)
        b_shot = torch.from_numpy(b_shot_np)

        n_layers = len(self.feature_layers) if self.feature_layers is not None else 1
        zs_per_layer = [[] for _ in range(n_layers)]

        with torch.no_grad():
            for s, e in self._iter_trace_chunks(self.nr):
                b_chunk = self._prepare_batch_3c(b_shot[s:e, :])
                z_list = self._encode(b_chunk, device_idx=0)
                for li, z_chunk in enumerate(z_list):
                    z_chunk = z_chunk.detach()
                    if self.cache_obs_on_cpu:
                        z_chunk = z_chunk.cpu()
                    zs_per_layer[li].append(z_chunk)

        z_obs_all = [torch.cat(zs, dim=0) for zs in zs_per_layer]
        if self.cache_obs_embeddings:
            self._obs_cache[i] = z_obs_all
        return z_obs_all

    def _postprocess_grad(self, g, x):
        """Cast dtype; when Op is None, self.x is the shot leaf [nr, nt], so transpose to [nt, nr].

        Resampling is part of the autograd graph (self._R), so no inverse resampling is needed.
        """
        if self.Op is None:
            g = g.T
        if hasattr(x, "dtype"):
            g = g.astype(x.dtype, copy=False)
        return g

    def _loss_only(self, x, i, z_obs_all):
        """Loss value without autograd graph or backward (e.g. calibration)."""
        if self.Op is not None:
            Op_i = self.Op[i] if isinstance(self.Op, list) else self.Op
            y_np = np.asarray(Op_i @ x)
        else:
            y_np = x
        y_np = y_np.reshape(self.nt, self.nr).T.astype(np.float32)
        if self.resample:
            y_np = self._resample_shot(y_np)
        y_shot = torch.from_numpy(y_np)

        total_loss, total_count = 0.0, 0
        with torch.no_grad():
            for s, e in self._iter_trace_chunks(self.nr):
                z_mod_list = self._encode(self._prepare_batch_3c(y_shot[s:e, :]), device_idx=0)
                z_obs_chunk_list = [
                    z_obs_all[li][s:e].to(device=self.model_device, dtype=self.model_dtype)
                    for li in range(len(z_obs_all))
                ]
                loss_chunk = self._compute_layer_loss(z_mod_list, z_obs_chunk_list)
                total_loss += float(loss_chunk.cpu().item()) * (e - s)
                total_count += e - s
        return total_loss / max(total_count, 1)

    def __call__(self, x, i, compute_grad=True):
        """Compute SeisLM feature loss for shot i. Returns scalar float.

        With compute_grad=False, only the loss value is computed (no autograd
        graph, no backward); a later grad(x, i) recomputes as needed.
        """
        self.f = None
        self.x = None
        self._loss_value = None
        self._grad_cache = None
        self._last_args = (i, np.array(x, copy=True)) if compute_grad else None

        z_obs_all = self._get_observed_embedding(i)

        if not compute_grad:
            self._loss_value = self._loss_only(x, i, z_obs_all)
            return self._loss_value

        y_shot_conn = None

        if self.Op is not None:
            x_flat = torch.from_numpy(x).to(torch.float32).requires_grad_()
            self.x  = x_flat
            Op_i    = self.Op[i] if isinstance(self.Op, list) else self.Op
            y       = TorchOperator(Op_i).apply(x_flat)
            y_shot_conn = y.reshape(self.nt, self.nr).T

            if self.resample:
                y_shot_conn = y_shot_conn @ self._R.T.to(y_shot_conn.dtype)
                y_shot      = y_shot_conn
                y_shot_leaf = y_shot_conn.detach().requires_grad_()
            else:
                y_shot      = y_shot_conn
                y_shot_leaf = y_shot_conn.detach().requires_grad_()
        else:
            x_shot_np = x.reshape(self.nt, self.nr).T.astype(np.float32)
            x_shot    = torch.from_numpy(x_shot_np)
            self.x    = x_shot.to(device=self.model_device, dtype=torch.float32).requires_grad_()
            y_shot      = self.x
            y_shot_leaf = self.x
            if self.resample:
                y_shot_conn = self.x @ self._R.T.to(device=self.model_device, dtype=self.x.dtype)
                y_shot      = y_shot_conn
                y_shot_leaf = y_shot_conn.detach().requires_grad_()

        if self.trace_chunk is None or self.trace_chunk <= 0 or self.trace_chunk >= self.nr:
            z_obs_dev  = [z.to(device=self.model_device, dtype=self.model_dtype) for z in z_obs_all]
            z_mod_list = self._encode(self._prepare_batch_3c(y_shot), device_idx=0)
            self.f     = self._compute_layer_loss(z_mod_list, z_obs_dev)
            self._loss_value = float(self.f.detach().cpu().item())
            return self._loss_value

        # With resampling, chunks backpropagate into a leaf and one VJP goes through R (and Op),
        # because the matmul graph can be traversed only once.
        use_multi_gpu = (len(self._devices) > 1) or self.resample
        if use_multi_gpu:
            self._call_chunked_multigpu(y_shot_leaf, y_shot_conn, z_obs_all, x)
        else:
            self._call_chunked_singlegpu(y_shot, z_obs_all, x)

        return self._loss_value

    def _call_chunked_singlegpu(self, y_shot, z_obs_all, x):
        if self.x.grad is not None:
            self.x.grad.zero_()

        total_loss  = 0.0
        total_count = 0

        for s, e in self._iter_trace_chunks(self.nr):
            bsz        = e - s
            y_chunk    = y_shot[s:e, :]
            z_mod_list = self._encode(self._prepare_batch_3c(y_chunk), device_idx=0)
            z_obs_chunk_list = [
                z_obs_all[li][s:e].to(device=self.model_device, dtype=self.model_dtype)
                for li in range(len(z_obs_all))
            ]
            loss_chunk  = self._compute_layer_loss(z_mod_list, z_obs_chunk_list)
            total_loss  += float(loss_chunk.detach().cpu().item()) * bsz
            total_count += bsz
            (loss_chunk * (bsz / float(self.nr))).backward()
            del y_chunk, z_mod_list, z_obs_chunk_list, loss_chunk

        self._loss_value = total_loss / max(total_count, 1)
        self._grad_cache = self._postprocess_grad(self.x.grad.detach().cpu().numpy(), x)
        self.f = None

    def _call_chunked_multigpu(self, y_shot_leaf, y_shot_conn, z_obs_all, x):
        """Parallel multi-GPU chunked forward+backward.

        Chunks are distributed round-robin across devices, serialized per GPU via locks.
        Gradient correctness: y_shot_leaf accumulates grads from all threads; a single
        VJP then propagates through y_shot_conn back to self.x.
        """
        if y_shot_leaf.grad is not None:
            y_shot_leaf.grad.zero_()

        n_devices   = len(self._devices)
        chunks      = list(self._iter_trace_chunks(self.nr))
        total_loss  = [0.0]
        total_count = [0]
        _acc_lock   = threading.Lock()

        def _process_chunk(ci, s, e):
            device_idx = ci % n_devices
            device     = self._devices[device_idx]
            bsz        = e - s

            with self._gpu_locks[device_idx]:
                y_chunk    = y_shot_leaf[s:e, :]
                z_mod_list = self._encode(self._prepare_batch_3c(y_chunk), device_idx=device_idx)
                z_obs_chunk_list = [
                    z_obs_all[li][s:e].to(device=device, dtype=self.model_dtype)
                    for li in range(len(z_obs_all))
                ]
                loss_chunk = self._compute_layer_loss(z_mod_list, z_obs_chunk_list)
                loss_val   = float(loss_chunk.detach().cpu().item()) * bsz
                (loss_chunk * (bsz / float(self.nr))).backward()
                del y_chunk, z_mod_list, z_obs_chunk_list, loss_chunk

            with _acc_lock:
                total_loss[0]  += loss_val
                total_count[0] += bsz

        with ThreadPoolExecutor(max_workers=n_devices) as executor:
            for fut in [executor.submit(_process_chunk, ci, s, e) for ci, (s, e) in enumerate(chunks)]:
                fut.result()

        self._loss_value = total_loss[0] / max(total_count[0], 1)

        # VJP: propagate y_shot_leaf.grad through y_shot_conn back to self.x (x_flat leaf)
        if y_shot_conn is not None:
            y_shot_conn.backward(y_shot_leaf.grad.detach().to(y_shot_conn.device))
        self._grad_cache = self._postprocess_grad(self.x.grad.detach().cpu().numpy(), x)
        self.f = None

    def grad(self, x, i):
        """Return gradient from last __call__.

        In chunked mode the gradient is already cached. In non-chunked mode
        this triggers backward().

        Returns
        -------
        numpy.ndarray
            - **Op is None**: adjoint source (gradient of the loss wrt the modeled shot
              gather), shape (nt, nr).
            - **Op is not None, resample=False**: FWI gradient (gradient of the loss wrt
              the optimization variable x), flat shape matching x.
            - **Op is not None, resample=True**: same as resample=False; resampling is a
              linear operator inside the autograd graph, so the chain rule through the
              resampling and Op is exact.
        """
        if (
            self._last_args is None
            or self._last_args[0] != i
            or not np.array_equal(self._last_args[1], x)
        ):
            # Stale state: the last __call__ was for a different (x, i).
            # Recompute so the gradient matches the arguments actually requested.
            self(x, i)

        if self._grad_cache is not None:
            return self._grad_cache

        if self.f is None or self.x is None:
            raise RuntimeError(
                "Call the loss first via `f = loss(x, i)` before calling `grad(x, i)`.")

        self.f.backward()
        self._grad_cache = self._postprocess_grad(self.x.grad.detach().cpu().numpy(), x)
        return self._grad_cache



class L2(NonlinearOperator):
    r"""L2 Norm.

    Computes the :math:`\ell_2` norm defined as: :math:`\ell_2(\mathbf{x}) =
    \frac{\sigma}{2} ||\mathbf{Op}\mathbf{x} - \mathbf{b}||_2^2`

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`, optional
        Linear operator
    b : :obj:`numpy.ndarray`, optional
        Data vector
    sigma : :obj:`int`, optional
        Multiplicative coefficient of L2 norm
    size : :obj:`int`, optional
        Size of the input vector (needed only when both ``Op`` and
        ``b`` are ``None``)
    """

    def __init__(self, Op=None, b=None, sigma=1., size=None, dtype="float32"):
        self.Op = Op
        self.b = b
        self.sigma = sigma
        if size is None:
            if b is None:
                size = Op.shape[1]
            else:
                size = b.size
        self._loss_value = None
        super().__init__(size, dtype)

    def loss(self, x, i):
        if self.Op is not None:
            Op = self.Op[i] if isinstance(self.Op, list) else self.Op

        if self.Op is not None and self.b is not None:
            f = (self.sigma / 2.) * (np.linalg.norm(Op @ x - self.b[i]) ** 2)
        elif self.b is not None:
            f = (self.sigma / 2.) * (np.linalg.norm(x - self.b[i]) ** 2)
        elif self.Op is not None:
            f = (self.sigma / 2.) * (np.linalg.norm(Op @ x) ** 2)
        else:
            f = (self.sigma / 2.) * (np.linalg.norm(x) ** 2)
        self._loss_value = float(f)
        return self._loss_value

    def grad(self, x, i):
        if self.Op is not None:
            Op = self.Op[i] if isinstance(self.Op, list) else self.Op

        if self.Op is not None and self.b is not None:
            g = self.sigma * (Op.H @ (Op @ x - self.b[i]))
        elif self.b is not None:
            g = self.sigma * (x - self.b[i])
        elif self.Op is not None:
            g = self.sigma * (Op.H @ Op @ x)
        else:
            g = self.sigma * x
        return g


class HybridLoss(NonlinearOperator):
    r"""Hybrid L2 + SeisLM feature-space loss.

    .. math::
        L = alpha * L_L2 / S_L2 + (1 - alpha) * L_SeisLM / S_SeisLM

    The scale factors ``S_L2`` and ``S_SeisLM`` should be calibrated from the
    summed initial losses over all shots, then kept fixed during inversion.

    Parameters
    ----------
    l2_loss : L2 or L2Torch
    seislm_loss : SeisLMFeatureLoss
    alpha : float or callable, optional
        Blending weight in [0, 1], or callable(iter_index) -> float. Default 0.5.
    scale_l2, scale_seislm : float or None, optional
        Fixed scale factors for the two loss terms. If omitted, call
        ``begin_scale_accumulation()``, evaluate all initial shots with
        ``loss(..., calibrating=True)``, and then call ``finalize_scales()``
        before computing gradients.

    Notes
    -----
    ``loss()`` must always be called before ``grad()`` for a given shot.
    Call ``step()`` after each outer FWI iteration to advance the schedule.
    """

    def __init__(self, l2_loss, seislm_loss, alpha=0.5, scale_l2=None, scale_seislm=None):
        self.l2_loss     = l2_loss
        self.seislm_loss = seislm_loss
        self._alpha      = alpha
        self._iter       = 0
        self._scale_l2     = self._sanitize_scale(scale_l2) if scale_l2 is not None else None
        self._scale_seislm = self._sanitize_scale(scale_seislm) if scale_seislm is not None else None
        self._loss_value   = None
        self._calibrating = False
        self._calib_l2 = 0.0
        self._calib_seislm = 0.0
        self._calib_count = 0
        super().__init__(size=l2_loss.size, dtype=l2_loss.dtype)

    @staticmethod
    def _sanitize_scale(value):
        value = float(value)
        return value if value > 0.0 else 1.0

    @property
    def alpha(self):
        a = self._alpha(self._iter) if callable(self._alpha) else float(self._alpha)
        if not 0.0 <= a <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {a} at iteration {self._iter}")
        return a

    @property
    def scales_ready(self):
        return self._scale_l2 is not None and self._scale_seislm is not None

    def set_scales(self, scale_l2, scale_seislm):
        """Set fixed scale factors, usually summed initial losses over all shots."""
        self._scale_l2 = self._sanitize_scale(scale_l2)
        self._scale_seislm = self._sanitize_scale(scale_seislm)
        self._calibrating = False
        print("HybridLoss calibrated from all requested shots:")
        print(f"  L2     scale = {self._scale_l2:.6e}")
        print(f"  SeisLM scale = {self._scale_seislm:.6e}")

    def begin_scale_accumulation(self):
        """Start accumulating raw initial losses for all-shot calibration."""
        self._scale_l2 = None
        self._scale_seislm = None
        self._calibrating = True
        self._calib_l2 = 0.0
        self._calib_seislm = 0.0
        self._calib_count = 0

    def finalize_scales(self):
        """Finalize scale factors from accumulated raw initial losses."""
        if self._calib_count == 0:
            raise RuntimeError("No losses were accumulated for HybridLoss calibration.")
        self.set_scales(self._calib_l2, self._calib_seislm)

    def step(self):
        """Advance iteration counter by one. Call after each FWI iteration."""
        self._iter += 1

    def loss(self, x, i, calibrating=False):
        calibrating = calibrating or self._calibrating
        f_l2     = self.l2_loss.loss(x, i)
        f_seislm = self.seislm_loss(x, i, compute_grad=not calibrating)

        if calibrating:
            self._calib_l2 += float(f_l2)
            self._calib_seislm += float(f_seislm)
            self._calib_count += 1
            self._loss_value = float(f_l2 + f_seislm)
            return self._loss_value

        if not self.scales_ready:
            raise RuntimeError(
                "HybridLoss scales are not calibrated. Call set_scales(...) with "
                "summed initial losses over all shots, or run begin_scale_accumulation(), "
                "evaluate all initial shots with loss(..., calibrating=True), then "
                "call finalize_scales() before computing gradients."
            )

        a = self.alpha
        self._loss_value = float(
            a * (f_l2 / self._scale_l2) + (1.0 - a) * (f_seislm / self._scale_seislm)
        )
        return self._loss_value

    def grad(self, x, i):
        if not self.scales_ready:
            raise RuntimeError("HybridLoss scales must be calibrated before grad() is called.")
        a = self.alpha
        return (
            a * (self.l2_loss.grad(x, i) / self._scale_l2)
            + (1.0 - a) * (self.seislm_loss.grad(x, i).ravel() / self._scale_seislm)
        )

