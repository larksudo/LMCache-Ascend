# SPDX-License-Identifier: Apache-2.0
"""Direct correctness tests for the block-level MP-mode AscendC kernel.

These tests intentionally bypass the MP server and its CPU staging buffers.  They
exercise the device operator itself: the interleaved per-layer ``[K, V, ...]``
pointer table, its phase-1 multi-object launch loop, prefix-block skipping, and
an engine tensor whose dim-0 block stride contains padding.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


def _load_c_ops() -> object:
    """Load the compiled extension without importing the MP service stack."""
    package_dir = Path(__file__).resolve().parents[2] / "lmcache_ascend"
    candidates = sorted(package_dir.glob("c_ops*.so"))
    if not candidates:
        raise RuntimeError(f"No compiled c_ops extension found in {package_dir}")

    spec = importlib.util.spec_from_file_location("c_ops", candidates[0])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {candidates[0]}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lmc_ops = _load_c_ops()


def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _shape_desc(
    *,
    nl: int,
    nb: int,
    bs: int,
    nh: int,
    hs: int,
    element_size: int,
    block_stride_elems: int,
) -> object:
    desc = lmc_ops.PageBufferShapeDesc()
    desc.kv_size = 2
    desc.nl = nl
    desc.nb = nb
    desc.bs = bs
    desc.nh = nh
    desc.hs = hs
    desc.element_size = element_size
    desc.block_stride_elems = block_stride_elems
    return desc


def _make_paged_tensor(
    *,
    nb: int,
    bs: int,
    nh: int,
    hs: int,
    padded: bool,
    offset: int,
    device: torch.device,
) -> torch.Tensor:
    """Create a distinct fp16 paged tensor, optionally with dim-0 padding."""
    if padded:
        storage = torch.empty((nb, bs + 1, nh, hs), dtype=torch.float16, device=device)
        tensor = storage[:, :bs]
    else:
        tensor = torch.empty((nb, bs, nh, hs), dtype=torch.float16, device=device)
    values = torch.arange(
        tensor.numel(), dtype=torch.float32, device=device
    ).reshape_as(tensor)
    tensor.copy_((values + offset).to(torch.float16))
    return tensor


@pytest.mark.skipif(not _npu_available(), reason="Ascend NPU required")
@pytest.mark.parametrize("padded", [False, True])
def test_multi_layer_block_transfer_round_trip_with_prefix_skip(padded: bool) -> None:
    """D2H then H2D restores only non-skipped blocks for two MP objects."""
    device = torch.device("npu:0")
    nl, nb, bs, nh, hs = 2, 8, 4, 2, 8
    chunk = 2 * bs
    blocks_per_object = chunk // bs
    num_objects = 2
    hidden = nh * hs

    # Each layer contributes an interleaved K/V entry, exactly as vLLM-Ascend
    # SEPARATE_KV does. The selected block IDs are deliberately non-contiguous.
    paged: list[tuple[torch.Tensor, torch.Tensor]] = []
    ptrs: list[int] = []
    for layer in range(nl):
        key = _make_paged_tensor(
            nb=nb,
            bs=bs,
            nh=nh,
            hs=hs,
            padded=padded,
            offset=10_000 * layer,
            device=device,
        )
        value = _make_paged_tensor(
            nb=nb,
            bs=bs,
            nh=nh,
            hs=hs,
            padded=padded,
            offset=10_000 * layer + 5_000,
            device=device,
        )
        paged.append((key, value))
        ptrs.extend((key.data_ptr(), value.data_ptr()))

    pointer_table = torch.tensor(ptrs, dtype=torch.int64, device=device)
    block_ids = torch.tensor([1, 3, 4, 6], dtype=torch.int64, device=device)
    stride = (bs + 1 if padded else bs) * hidden
    desc = _shape_desc(
        nl=nl,
        nb=nb,
        bs=bs,
        nh=nh,
        hs=hs,
        element_size=torch.empty((), dtype=torch.float16).element_size(),
        block_stride_elems=stride if padded else 0,
    )
    objects = [
        torch.zeros((2, nl, chunk, hidden), dtype=torch.float16, device=device)
        for _ in range(num_objects)
    ]

    # Store engine -> LMCache. Skip the first block in *each* object.
    lmc_ops.multi_layer_block_kv_transfer(
        pointer_table,
        [obj.data_ptr() for obj in objects],
        block_ids,
        device,
        lmc_ops.TransferDirection.D2H,
        desc,
        chunk,
        lmc_ops.EngineKVFormat.NL_X_TWO_X_NB_BS_NH_HS,
        1,
    )
    torch.npu.synchronize()

    block_ids_cpu = block_ids.cpu().tolist()
    for object_idx, obj in enumerate(objects):
        assert torch.count_nonzero(obj[:, :, :bs]) == 0
        for local_block in range(1, blocks_per_object):
            engine_block = block_ids_cpu[object_idx * blocks_per_object + local_block]
            token_slice = slice(local_block * bs, (local_block + 1) * bs)
            for layer, (key, value) in enumerate(paged):
                torch.testing.assert_close(
                    obj[0, layer, token_slice].reshape_as(key[engine_block]),
                    key[engine_block],
                )
                torch.testing.assert_close(
                    obj[1, layer, token_slice].reshape_as(value[engine_block]),
                    value[engine_block],
                )

    # Clear the engine then restore. Prefix blocks remain clear; the remaining
    # block in each object is restored byte-for-byte from the 2LTD object.
    for key, value in paged:
        key.zero_()
        value.zero_()
    lmc_ops.multi_layer_block_kv_transfer(
        pointer_table,
        [obj.data_ptr() for obj in objects],
        block_ids,
        device,
        lmc_ops.TransferDirection.H2D,
        desc,
        chunk,
        lmc_ops.EngineKVFormat.NL_X_TWO_X_NB_BS_NH_HS,
        1,
    )
    torch.npu.synchronize()

    for object_idx in range(num_objects):
        skipped_block = block_ids_cpu[object_idx * blocks_per_object]
        restored_block = block_ids_cpu[object_idx * blocks_per_object + 1]
        for layer, (key, value) in enumerate(paged):
            assert torch.count_nonzero(key[skipped_block]) == 0
            assert torch.count_nonzero(value[skipped_block]) == 0
            torch.testing.assert_close(
                key[restored_block],
                objects[object_idx][0, layer, bs:].reshape_as(key[0]),
            )
            torch.testing.assert_close(
                value[restored_block],
                objects[object_idx][1, layer, bs:].reshape_as(value[0]),
            )
