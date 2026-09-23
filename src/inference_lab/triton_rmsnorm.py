from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(x_ptr, w_ptr, out_ptr, stride, width: tl.constexpr, eps: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, block)
    mask = offsets < width
    x = tl.load(x_ptr + row * stride + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / width
    inv_rms = tl.rsqrt(variance + eps)
    weight = tl.load(w_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + row * stride + offsets, x * inv_rms * weight, mask=mask)


def triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    if x.shape[-1] != weight.numel():
        raise ValueError("weight must match the final input dimension")
    contiguous = x.contiguous()
    output = torch.empty_like(contiguous)
    rows = contiguous.numel() // contiguous.shape[-1]
    width = contiguous.shape[-1]
    block = triton.next_power_of_2(width)
    _rms_norm_kernel[(rows,)](
        contiguous,
        weight,
        output,
        contiguous.stride(-2),
        width=width,
        eps=eps,
        block=block,
    )
    return output
