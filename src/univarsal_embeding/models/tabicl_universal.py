"""Universal TabICL Framework with Dynamic Random Orthogonal Projections and LoRA Adaptation."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _patch_mem_get_info_for_indexless_cuda_device() -> None:
    """Work around a bug in the third-party `tabicl` library: its internal
    InferenceManager (tabicl/model/inference.py) falls back to
    `torch.device("cuda")` (no index) when estimating safe inference batch
    sizes, which newer PyTorch's `torch.cuda.mem_get_info` rejects with
    "Expected a torch.device with a specified index or an integer". This
    only surfaces in eval()/inference-mode forward passes (e.g. the 16-way
    random-projection ensemble), never in training. Idempotent - safe to
    call multiple times.
    """
    if getattr(torch.cuda.mem_get_info, "_patched_for_indexless_cuda", False):
        return
    _orig_mem_get_info = torch.cuda.mem_get_info

    def _patched_mem_get_info(device=None):
        if isinstance(device, torch.device) and device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return _orig_mem_get_info(device)

    _patched_mem_get_info._patched_for_indexless_cuda = True
    torch.cuda.mem_get_info = _patched_mem_get_info


_patch_mem_get_info_for_indexless_cuda_device()


def sample_orthogonal_matrix(
    in_dim: int,
    target_dim: int,
    device: torch.device,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Samples a random orthogonal projection matrix using QR decomposition on the Stiefel manifold.

    Returns a matrix W of shape (in_dim, target_dim) such that W^T W = I (orthonormal columns).
    IMPORTANT: Requires in_dim >= target_dim. Cannot project to higher dimensions.
    """
    if in_dim < target_dim:
        raise ValueError(
            f"Cannot project from in_dim={in_dim} to target_dim={target_dim}. "
            f"Orthogonal projection requires in_dim >= target_dim. "
            f"(All foundation encoders have dimension >= 256)"
        )

    if in_dim == target_dim:
        return torch.eye(target_dim, device=device)

    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        raw = torch.randn(in_dim, target_dim, generator=g, device=device)
    else:
        raw = torch.randn(in_dim, target_dim, device=device)

    q, _ = torch.linalg.qr(raw)
    return q[:, :target_dim]


class UniversalRandomProjection(nn.Module):
    """Universal Random Orthogonal Projection mapping variable encoder dims to canonical TabICL dim."""

    def __init__(
        self,
        in_dim: int,
        target_dim: int = 256,
        seed: int = 42,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.target_dim = target_dim

        # Deterministic initialization via sample_orthogonal_matrix
        W = sample_orthogonal_matrix(in_dim, target_dim, device=torch.device("cpu"), seed=seed)
        self.W = nn.Parameter(W, requires_grad=learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.W)


class LoRALinear(nn.Module):
    """Parameter-Efficient Low-Rank Adaptation (LoRA) on top of frozen PyTorch Linear layer."""

    def __init__(self, base_linear: nn.Linear, r: int = 16, lora_alpha: float = 32.0) -> None:
        super().__init__()
        self.base_linear = base_linear
        for p in self.base_linear.parameters():
            p.requires_grad = False

        self.r = r
        self.scaling = lora_alpha / r
        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base_linear.weight

    @property
    def bias(self):
        return self.base_linear.bias

    @property
    def in_features(self):
        return self.base_linear.in_features

    @property
    def out_features(self):
        return self.base_linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out


class UniversalTabICL(nn.Module):
    """Universal TabICL wrapper equipping TabICL with Dynamic Random Projections and LoRA Adaptation."""

    def __init__(
        self,
        checkpoint_repo: str = "jingang/TabICL",
        checkpoint_file: str = "tabicl-classifier-v2-20260212.ckpt",
        target_dim: int = 256,
        lora_r: int = 16,
        lora_alpha: float = 32.0,
        enable_lora: bool = True,
        device: str = "cpu",
        recompute: bool = True,
    ) -> None:
        super().__init__()
        self.target_dim = target_dim
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.enable_lora = enable_lora
        self.device = device
        self._projections: Dict[int, UniversalRandomProjection] = {}

        # Load pretrained TabICL
        from huggingface_hub import hf_hub_download
        from tabicl import TabICL

        ckpt_path = hf_hub_download(repo_id=checkpoint_repo, filename=checkpoint_file)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        config = dict(checkpoint["config"])
        # Gradient checkpointing (recompute activations during backward instead
        # of caching them): a pure memory/compute tradeoff, no data is dropped.
        # Needed to fit full, unclipped in-context training sets (some tasks
        # have 6000+ training rows) in GPU memory - see also the bf16 autocast
        # in forward_episodic/forward_episodic_ensemble below.
        config["recompute"] = recompute
        self.tabicl = TabICL(**config)
        self.tabicl.load_state_dict(checkpoint["state_dict"])

        # Freeze all base parameters
        for p in self.tabicl.parameters():
            p.requires_grad = False

        if enable_lora:
            self._inject_lora()

        self.to(device)

    def _inject_lora(self) -> None:
        """Inject LoRA into attention linear layers of TabICL."""
        count = 0
        for name, module in self.tabicl.named_modules():
            for child_name, child_module in module.named_children():
                if isinstance(child_module, nn.Linear) and any(
                    k in child_name.lower() for k in ["linear1", "linear2", "out_proj", "in_linear"]
                ):
                    setattr(module, child_name, LoRALinear(child_module, r=self.lora_r, lora_alpha=self.lora_alpha))
                    count += 1
        logger.info(f"Injected LoRA (r={self.lora_r}) into {count} TabICL linear projections.")

    def get_projection(self, in_dim: int, learnable_rp: bool = False) -> UniversalRandomProjection:
        if in_dim not in self._projections:
            proj = UniversalRandomProjection(
                in_dim=in_dim, target_dim=self.target_dim, seed=42, learnable=learnable_rp
            ).to(self.device)
            self._projections[in_dim] = proj
            self.add_module(f"proj_{in_dim}", proj)
        return self._projections[in_dim]

    def forward_episodic(
        self,
        X_ctx: torch.Tensor,
        y_ctx: torch.Tensor,
        X_query: torch.Tensor,
        dynamic_rp: bool = True,
        projection_seed: Optional[int] = None,
        learnable_rp: bool = False,
    ) -> torch.Tensor:
        """Runs episodic forward pass:
        
        Args:
            X_ctx: (N_ctx, D_raw) context features.
            y_ctx: (N_ctx,) context integer labels.
            X_query: (N_q, D_raw) query features.
            dynamic_rp: If True, samples a fresh random orthogonal projection matrix for this episode.
            projection_seed: Optional seed for reproducible projection during evaluation.
            learnable_rp: If True, uses the learned projection parameters.
        Returns:
            Logits for query samples: (N_q, C).
        """
        in_dim = X_ctx.shape[1]

        # 1. Random Orthogonal Projection Adapter
        if dynamic_rp and not learnable_rp:
            W = sample_orthogonal_matrix(in_dim, self.target_dim, device=X_ctx.device, seed=projection_seed)
            X_ctx_proj = torch.matmul(X_ctx, W)
            X_q_proj = torch.matmul(X_query, W)
        else:
            proj = self.get_projection(in_dim, learnable_rp=learnable_rp)
            X_ctx_proj = proj(X_ctx)
            X_q_proj = proj(X_query)

        # 2. Support-set standardization
        mean = X_ctx_proj.mean(dim=0, keepdim=True)
        std = X_ctx_proj.std(dim=0, keepdim=True).clamp_min(1e-5)
        X_ctx_std = (X_ctx_proj - mean) / std
        X_q_std = (X_q_proj - mean) / std

        # 3. Concatenate context + query along sample dimension: (1, N_ctx + N_q, D_tab)
        X_all = torch.cat([X_ctx_std, X_q_std], dim=0).unsqueeze(0)
        n_classes = len(torch.unique(y_ctx))
        y_ctx_2d = y_ctx.unsqueeze(0) if y_ctx.ndim == 1 else y_ctx

        if X_all.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.tabicl(X_all, y_ctx_2d, return_logits=True)[0, :, :n_classes]
        else:
            logits = self.tabicl(X_all, y_ctx_2d, return_logits=True)[0, :, :n_classes]
        return logits

    def forward_episodic_ensemble(
        self,
        X_ctx: torch.Tensor,
        y_ctx: torch.Tensor,
        X_query: torch.Tensor,
        num_projections: int = 16,
        base_seed: int = 42,
    ) -> torch.Tensor:
        """Evaluates episodic forward pass with Random Projection Ensembling (RPE).
        Averages softmax probabilities over multiple orthogonal projections to reduce variance.
        """
        all_probs = []
        # Inference-only: disable autograd so each of the 16 projection passes
        # releases its activations immediately instead of accumulating a
        # combined graph, which would otherwise multiply peak memory ~16x.
        with torch.no_grad():
            for i in range(num_projections):
                proj_seed = base_seed + i * 10007
                logits = self.forward_episodic(
                    X_ctx, y_ctx, X_query, dynamic_rp=True, projection_seed=proj_seed
                )
                # Softmax in float32: normalizing bf16 logits leaves row sums
                # off by up to ~0.2%, which sklearn's multi-class AUROC
                # validator rejects outright.
                probs = torch.softmax(logits.float(), dim=-1)
                all_probs.append(probs)

        # NumPy has no bfloat16 dtype; callers typically do .cpu().numpy() on
        # this result, so return float32 regardless of the autocast dtype
        # used internally during the ensemble passes.
        mean_probs = torch.stack(all_probs, dim=0).mean(dim=0).float()
        return mean_probs
