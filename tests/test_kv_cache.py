import pytest

from inference_lab.kv_cache import PagedKVAllocator, kv_cache_bytes


def test_kv_cache_size_counts_keys_and_values():
    assert kv_cache_bytes(
        layers=2,
        kv_heads=4,
        head_dim=8,
        sequence_length=16,
        batch_size=1,
        dtype="float16",
    ) == 4096


def test_paged_allocator_reuses_released_blocks():
    allocator = PagedKVAllocator(total_blocks=4, block_size=16)
    first = allocator.reserve("a", 17)
    second = allocator.reserve("b", 16)
    assert first.blocks == 2
    assert second.blocks == 1
    assert allocator.free_blocks == 1
    allocator.release("a")
    allocator.reserve("c", 32)
    assert allocator.used_blocks == 3


def test_paged_allocator_rejects_overcommit():
    allocator = PagedKVAllocator(total_blocks=1, block_size=16)
    with pytest.raises(MemoryError):
        allocator.reserve("too-large", 17)
