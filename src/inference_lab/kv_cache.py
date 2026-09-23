from __future__ import annotations

import math
from dataclasses import dataclass


DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int4": 0.5,
}


def kv_cache_bytes(
    *,
    layers: int,
    kv_heads: int,
    head_dim: int,
    sequence_length: int,
    batch_size: int = 1,
    dtype: str = "float16",
) -> int:
    """Return theoretical dense KV-cache storage for keys and values."""
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported dtype: {dtype}")
    dimensions = (layers, kv_heads, head_dim, sequence_length, batch_size)
    if any(value < 1 for value in dimensions):
        raise ValueError("all cache dimensions must be positive")
    elements = 2 * math.prod(dimensions)
    return int(elements * DTYPE_BYTES[dtype])


@dataclass
class Allocation:
    request_id: str
    tokens: int
    blocks: int


class PagedKVAllocator:
    """Small deterministic simulator for fixed-size paged KV allocation."""

    def __init__(self, total_blocks: int, block_size: int = 16):
        if total_blocks < 1 or block_size < 1:
            raise ValueError("total_blocks and block_size must be positive")
        self.total_blocks = total_blocks
        self.block_size = block_size
        self._allocations: dict[str, Allocation] = {}

    @property
    def used_blocks(self) -> int:
        return sum(item.blocks for item in self._allocations.values())

    @property
    def free_blocks(self) -> int:
        return self.total_blocks - self.used_blocks

    def reserve(self, request_id: str, tokens: int) -> Allocation:
        if request_id in self._allocations:
            raise ValueError(f"request already allocated: {request_id}")
        if tokens < 1:
            raise ValueError("tokens must be positive")
        blocks = math.ceil(tokens / self.block_size)
        if blocks > self.free_blocks:
            raise MemoryError(
                f"need {blocks} blocks but only {self.free_blocks} remain"
            )
        allocation = Allocation(request_id, tokens, blocks)
        self._allocations[request_id] = allocation
        return allocation

    def release(self, request_id: str) -> Allocation:
        try:
            return self._allocations.pop(request_id)
        except KeyError as exc:
            raise KeyError(f"unknown request: {request_id}") from exc
