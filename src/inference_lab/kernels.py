from __future__ import annotations

import torch


def rms_norm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(variance + eps).to(x.dtype)
    return normalized * weight


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """Use the Triton kernel when available, otherwise the tested PyTorch reference."""
    try:
        from .triton_rmsnorm import triton_rms_norm
    except (ImportError, ModuleNotFoundError):
        return rms_norm_reference(x, weight, eps)
    if not x.is_cuda:
        return rms_norm_reference(x, weight, eps)
    return triton_rms_norm(x, weight, eps)
