# SPDX-License-Identifier: Apache-2.0
"""Fused NPU gather/scatter for the multiprocess non-GPU transfer path.

The upstream multiprocess (MP) ``DataTransferContext`` moves KV cache between
a worker's paged NPU memory and CPU chunks via the generic, per-layer PyTorch
helpers :func:`gather_paged_kv_to_cpu` / :func:`scatter_cpu_to_paged_kv`
(``lmcache/v1/multiprocess/transfer_context/base.py``).  Those helpers are
O(num_chunks * num_layers) kernel launches with several intermediate
allocations each — they never reach the fused Ascend transfer kernel that the
in-process connector already uses.

This module monkey-patches those two callables (on both the ``base`` and the
``worker_transfer`` namespaces) so that, **for SEPARATE_KV caches on 910B/C
NPU**, all chunks in a sub-batch share a **single**
:func:`lmcache_ascend.c_ops.multi_layer_kv_transfer` call between the paged KV
and an NPU staging buffer (one concatenated slot-mapping over every active
block), after which each chunk's gathered tokens are D2H/H2D-copied from its
staging slice via ordinary torch ``copy_``. The host leg therefore works with
any host buffer (SHM views or freshly allocated CPU tensors), and the number of
device launches collapses from one-per-chunk to one-per-sub-batch. Sub-batching
bounds the staging buffer (``_STAGING_CAP_BYTES``, default 1 GiB); the staging
buffer must stay contiguous and exactly token-sized because the kernel indexes
it by raw pointer arithmetic.

When the worker SHM pool is page-locked -- which it is once the
:class:`~lmcache.v1.platform.npu.pin_memory.NpuPinMemoryBackend` is
active (AscendCL ``aclrtHostRegister``) -- the per-chunk ``copy_(non_blocking)``
is genuinely async, so a whole sub-batch is one device kernel plus N pipelined
async copies awaited by a single stream sync. Without pinning torch falls back
to a synchronous copy, still correctly ordered before that sync.

Unsupported cases (CPU/other devices, 310P, SGLang, DSA_C8/MULTI_PLANE) fall
through to the original upstream implementation unchanged.

For MLA_KV / DSA_KV the fused kernel reuses the in-process connector's plane
layout (``npu_connectors.py`` ``_V2KVTransferMixin``): the per-plane widths
differ (``kv_lora_rank`` vs ``qk_rope_head_dim`` vs ``dsa_head_dim``), so the
planes are concatenated into one flat staging block ``[1, L, tokens, sum]``
rather than two equal slots. The SHM server, however, allocates chunk buffers
from a single upstream-negotiated ``hidden_dim_size`` + ``use_mla`` flag, and
``compute_kv_layout`` reports the K-plane product
``num_kv_heads * kv_lora_rank`` for Ascend ``(K, V)`` tuples — wrong in both
the leading and trailing dims. ``compute_kv_layout`` is therefore patched (see
:func:`_compute_kv_layout_wrapper`) so the server allocates the rank-3
``[num_layers, tokens, kv_lora_rank+qk_rope_head_dim(+dsa)]`` buffer the kernel
consumes; the gather/scatter then zero-copy ``view_as`` it to the kernel's
``[1, ...]`` staging shape. DSA_C8_KV / MULTI_PLANE_KV need a separate
multi-plane kernel and remain unsupported here.

In addition, ``EngineDrivenTransferContext.submit_store`` / ``submit_retrieve``
are patched so the fused path runs on a dedicated NPU stream
(``_NPUTransferDescriptor.transfer_stream``): the whole-device
``torch_dev.synchronize()`` that orders the gather against the model forward is
replaced with ``transfer_stream.wait_stream(torch.npu.current_stream())``
(mirrors the in-process connector at npu_connectors.py:1126). The "forward-
completion event" the engine passes in is not used: the deployed vLLM connector
resolves it to a CPU-runner ``_EventPlaceholder`` with no ``.wait`` method, so
ordering via the current (forward) stream is the robust choice; the two
pre-commit syncs become stream-scoped ``transfer_stream.synchronize()``. The
per-chunk D2H/H2D copies are issued ``non_blocking`` on that stream so N host
syncs collapse to one. Unsupported layouts (and CPU/310P workers) fall back to the
original upstream methods unchanged.

Finally, ``install_overrides`` also wires the **LMCache-driven** MP path (the
block-level object-group transfer of the MP-mode design doc section 6.1) when
the compiled block kernels are present: it rebinds the upstream
``lmcache_driven_transfer`` module's ``transfer_kv_per_object_group`` to the
Ascend port and registers the NPU KV-wrapper factory, so
``LMCACHE_MP_TRANSFER_MODE=lmcache_driven`` (or the ``lmcache.mp.mp_transfer_mode``
extra config) selects the server-driven block-level path while ``auto`` /
``engine_driven`` keep the token-level fused path above unchanged. See
:func:`_install_lmcache_driven_overrides`.

Heavy dependencies (``c_ops``, the NPU connector helpers) are imported lazily
so the module and its pure-Python helpers stay importable on hosts without a
built extension — this keeps the slot-mapping and fallback logic unit-testable.
"""

# Standard
from collections.abc import Sequence
from typing import Any, NamedTuple, Optional

# Third Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.futures import MessagingFuture
import torch

# First Party
from lmcache_ascend.v1.kv_format import KVCacheFormat

logger = init_logger(__name__)

# Formats whose chunk-shape contract is compatible with the fused kernel.
# MLA_KV / DSA_KV concatenate their differing-width planes into one flat
# staging block and need the layout-negotiation patch below; DSA_C8_KV /
# MULTI_PLANE_KV use a separate multi-plane kernel and stay unsupported.
_SUPPORTED_FORMATS: tuple[KVCacheFormat, ...] = (
    KVCacheFormat.SEPARATE_KV,
    KVCacheFormat.MLA_KV,
    KVCacheFormat.DSA_KV,
)

# Reusable per-worker transfer descriptors, keyed by the full data-pointer
# signature of the paged KV tensors. In production ``kv_caches`` is registered
# once per worker and lives for the engine's lifetime, so the signature is
# stable and the descriptor is reused across store/retrieve calls. Keying on
# the full signature (rather than ``id(kv_caches)``) avoids stale hits when a
# Python object id or a device address is reused — e.g. tests that build and
# destroy KV caches back-to-back in one process.
_descriptor_cache: dict[tuple[int, ...], "_NPUTransferDescriptor"] = {}


def _descriptor_signature(kv_caches: dict[str, object]) -> tuple[int, ...]:
    """Tuple of every paged-tensor data pointer, in layer-then-plane order."""
    sig: list[int] = []
    for value in kv_caches.values():
        if isinstance(value, (tuple, list)):
            for tensor in value:
                sig.append(tensor.data_ptr())  # type: ignore[union-attr]
        else:
            sig.append(value.data_ptr())  # type: ignore[union-attr]
    return tuple(sig)


def _first_layer_tensor(layers: list[object]) -> torch.Tensor:
    """Return the representative tensor of the first per-layer entry."""
    first = layers[0]
    if isinstance(first, (tuple, list)):
        return first[0]  # type: ignore[no-any-return]
    return first  # type: ignore[no-any-return]


class _PlaneGeometry(NamedTuple):
    """Per-format plane widths and staging layout for one paged-KV layer.

    Pure data (no device access) so the derivation is unit-testable on non-NPU
    hosts. Mirrors the in-process connector's ``v2_staging_hidden_dim``
    (npu_connectors.py:646-653) and ``get_shape`` (npu_connectors.py:2397-2411).
    """

    kv_lora_rank: int
    qk_rope_head_dim: int
    dsa_head_dim: int
    use_mla: bool
    staging_kv_lead: int
    hidden: int


def _derive_plane_geometry(
    kv_format: KVCacheFormat, first: Sequence[torch.Tensor]
) -> _PlaneGeometry:
    """Derive plane widths and staging layout for ``kv_format``.

    SEPARATE_KV stores K and V as two equal-sized planes
    (``[2, L, tokens, nh*hs]``); MLA/DSA plane widths differ, so the kernel
    concatenates them into one flat block (``[1, L, tokens, sum]``) and carves
    the planes by offset (MLAPolicy/DSAPolicy).

    Args:
        kv_format: Detected Ascend :class:`KVCacheFormat`.
        first: The first layer's per-plane tensors
            ``(k_cache, v_cache[, dsa_k_cache])``.

    Returns:
        The derived :class:`_PlaneGeometry`.

    Raises:
        ValueError: If ``kv_format`` is not a supported tuple format.
    """
    k0 = first[0]
    if kv_format == KVCacheFormat.SEPARATE_KV:
        return _PlaneGeometry(
            kv_lora_rank=0,
            qk_rope_head_dim=0,
            dsa_head_dim=0,
            use_mla=False,
            staging_kv_lead=2,
            hidden=int(k0.shape[2]) * int(k0.shape[3]),
        )
    if kv_format == KVCacheFormat.MLA_KV:
        kv_lora_rank = int(k0.shape[-1])
        qk_rope_head_dim = int(first[1].shape[-1])
        return _PlaneGeometry(
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            dsa_head_dim=0,
            use_mla=True,
            staging_kv_lead=1,
            hidden=kv_lora_rank + qk_rope_head_dim,
        )
    if kv_format == KVCacheFormat.DSA_KV:
        kv_lora_rank = int(k0.shape[-1])
        qk_rope_head_dim = int(first[1].shape[-1])
        dsa_head_dim = int(first[2].shape[-1])
        return _PlaneGeometry(
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            dsa_head_dim=dsa_head_dim,
            use_mla=True,
            staging_kv_lead=1,
            hidden=kv_lora_rank + qk_rope_head_dim + dsa_head_dim,
        )
    raise ValueError(f"Unsupported KV format for plane geometry: {kv_format.name}")


class _NPUTransferDescriptor:
    """Precomputed kernel inputs for one worker's paged KV cache.

    Built once per ``kv_caches`` registration and reused across store/retrieve
    calls so the device-resident pointer table and the staging buffer are not
    rebuilt per chunk.

    Attributes:
        device: NPU device of the paged KV tensors.
        kv_format: Detected Ascend :class:`KVCacheFormat`.
        ptr_table: Flat int64 NPU tensor of interleaved per-layer (K, V) data
            pointers in the order ``_pointers_for_entry`` produces
            (``[k0, v0, k1, v1, ...]``).
        block_size: vLLM block size (tokens per block).
        page_buffer_size: ``num_blocks * block_size`` (slots per layer).
        num_layers: Number of KV layers.
        hidden: Contiguous-buffer trailing hidden dim: ``num_heads * head_size``
            for SEPARATE_KV, or the summed plane widths
            (``kv_lora_rank + qk_rope_head_dim (+ dsa_head_dim)``) for MLA/DSA.
        staging_kv_lead: Leading dim of the staging buffer — ``2`` for
            SEPARATE_KV (two equal K/V slots), ``1`` for MLA/DSA (one flat
            concatenated block; the kernel carves the planes by offset).
        use_mla: MLA/DSA flag passed to the kernel (``True`` for MLA/DSA).
        kv_lora_rank / qk_rope_head_dim / dsa_head_dim: MLA/DSA plane widths
            (all 0 for SEPARATE_KV).
        dtype: Element dtype of the paged KV / contiguous buffer.
        transfer_stream: Dedicated NPU stream for the paged<->staging DMA and
            the D2H/H2D leg, so MP transfer does not contend with model
            kernels on the default stream.
    """

    def __init__(self, layers: list[object]) -> None:
        # First Party — lazy: avoids importing the connector / c_ops at module
        # import time so this module stays unit-testable on non-NPU hosts.
        from lmcache_ascend.v1.npu_connector.npu_connectors import _pointers_for_entry

        first = layers[0]
        if not isinstance(first, (tuple, list)):
            # Bare-tensor entries (SGLang NPU 5-D, MERGED, ...) are not handled
            # here; let the caller fall back to the upstream path.
            raise ValueError("NPU fused gather/scatter requires per-layer tuples")

        ref = _first_layer_tensor(layers)
        self.device: torch.device = ref.device
        self.dtype: torch.dtype = ref.dtype
        self.kv_format: KVCacheFormat = KVCacheFormat.detect(layers)

        if self.kv_format not in _SUPPORTED_FORMATS:
            raise ValueError(
                f"NPU fused gather/scatter does not support {self.kv_format.name}"
            )

        k0 = first[0]
        self.block_size: int = int(k0.shape[1])
        num_blocks = int(k0.shape[0])
        self.page_buffer_size: int = num_blocks * self.block_size
        self.num_layers: int = len(layers)

        # Per-format plane geometry (see _derive_plane_geometry). SEPARATE_KV
        # stores K and V as two equal-sized planes ([2, L, tokens, nh*hs]);
        # MLA/DSA plane widths differ, so the kernel concatenates them into one
        # flat block ([1, L, tokens, sum]) and carves the planes by offset.
        geo = _derive_plane_geometry(self.kv_format, first)
        self.kv_lora_rank: int = geo.kv_lora_rank
        self.qk_rope_head_dim: int = geo.qk_rope_head_dim
        self.dsa_head_dim: int = geo.dsa_head_dim
        self.use_mla: bool = geo.use_mla
        self.staging_kv_lead: int = geo.staging_kv_lead
        self.hidden: int = geo.hidden

        # Interleaved [k0, v0, k1, v1, ...] device-resident pointer table.
        ptrs: list[int] = []
        for entry in layers:
            ptrs.extend(_pointers_for_entry(entry, self.kv_format))
        cpu_ptrs = torch.tensor(ptrs, dtype=torch.int64)
        self.ptr_table: torch.Tensor = torch.empty(
            cpu_ptrs.shape, dtype=torch.int64, device=self.device
        )
        self.ptr_table.copy_(cpu_ptrs)

        # LMC-A: dedicated stream for the paged<->staging DMA + the D2H/H2D
        # leg so KV transfer no longer contends with model kernels on the
        # default NPU stream. Created eagerly (mirrors the channel
        # ``transport_stream`` at hccl_channel.py:104); the descriptor is only
        # built lazily once a supported NPU layout is registered, so the device
        # is live by this point.
        self.transfer_stream: torch.npu.Stream = torch.npu.Stream(
            device=self.device
        )

        self._staging: Optional[torch.Tensor] = None

    @property
    def plane_extras(self) -> tuple[int, int, int, int]:
        """The (k, v, dsa, scale) hidden-dim extras passed to the kernel.

        For SEPARATE_KV all four are 0 (the kernel derives ``hidden`` from
        ``size(-1)``); for MLA/DSA they carry the per-plane widths the kernel
        uses to carve the flat staging block.
        """
        return (self.kv_lora_rank, self.qk_rope_head_dim, self.dsa_head_dim, 0)

    def staging_for(self, kv_lead: int, tokens: int) -> torch.Tensor:
        """Return an NPU staging buffer ``[kv_lead, L, tokens, H]``.

        The buffer is reallocated when ``tokens`` changes. The transfer kernel
        indexes it by raw pointer arithmetic (layer ``L`` at
        ``L * tokens * hidden``), so it must be a **contiguous** buffer of exactly
        the requested token count -- a prefix slice of a larger buffer would be
        non-contiguous across layers and corrupt the transfer. Repeated batched
        calls therefore incur at most one reallocation when the chunk count (and
        thus ``tokens``) changes between sub-batches.
        """
        shape = torch.Size([kv_lead, self.num_layers, tokens, self.hidden])
        if self._staging is None or self._staging.shape != shape:
            self._staging = torch.empty(shape, dtype=self.dtype, device=self.device)
        return self._staging


def _build_descriptor(kv_caches: dict[str, object]) -> Optional[_NPUTransferDescriptor]:
    """Build a descriptor for ``kv_caches`` if it is a supported NPU layout.

    Returns ``None`` for non-NPU devices, 310P, or unsupported formats so the
    caller can route to the upstream implementation.
    """
    # Third Party / First Party — lazy (see _NPUTransferDescriptor.__init__).
    from lmcache.v1.gpu_connector.utils import get_device
    from lmcache_ascend.v1.npu_connector.npu_connectors import is_310p

    values = list(kv_caches.values())
    if not values:
        return None
    try:
        device = get_device(values)  # type: ignore[arg-type]
    except (AttributeError, IndexError):
        return None
    if device.type != "npu" or is_310p():
        return None
    try:
        return _NPUTransferDescriptor(values)
    except ValueError:
        return None


def _get_descriptor(
    kv_caches: dict[str, object],
) -> Optional[_NPUTransferDescriptor]:
    """Return the cached descriptor for ``kv_caches``, building it if needed."""
    sig = _descriptor_signature(kv_caches)
    cached = _descriptor_cache.get(sig)
    if cached is not None:
        return cached
    desc = _build_descriptor(kv_caches)
    if desc is not None:
        _descriptor_cache[sig] = desc
    return desc


#: Cached per-block offset tables ``torch.arange(block_size)``, keyed by block
#: size, so the batched slot-mapping build reuses one small constant tensor
#: instead of rebuilding it per sub-batch.
_offsets_cache: dict[int, torch.Tensor] = {}

#: Upper bound on the batched staging buffer (bytes). Bounding it keeps a large
#: multi-chunk store/retrieve from allocating one device buffer for every token
#: at once; transfers larger than this are split into sub-batches, each still a
#: single kernel launch.
_STAGING_CAP_BYTES = 1 << 30  # 1 GiB

#: Feature flag for the LMCache-driven block-level MP path (design doc 6.1).
#: ``None`` until the first probe; resolved lazily so this module stays
#: importable on hosts without the Ascend extension. Delegates to
#: :data:`lmcache_ascend.v1.multiprocess.npu_driven_transfer._HAS_NATIVE_OBJECT_GROUP_TRANSFER`
#: (upstream's identity check against ``python_ops_fallback`` — ``_patch_ops``
#: merges the raising fallback into the Ascend ``c_ops`` when the block
#: kernels are absent, so a bare ``hasattr`` probe would mis-fire).
_HAS_NATIVE_OBJECT_GROUP_TRANSFER: Optional[bool] = None


def _native_object_group_transfer_available() -> bool:
    """Resolve :data:`_HAS_NATIVE_OBJECT_GROUP_TRANSFER` once, then cache."""
    global _HAS_NATIVE_OBJECT_GROUP_TRANSFER
    if _HAS_NATIVE_OBJECT_GROUP_TRANSFER is None:
        # First Party — lazy: npu_driven_transfer is importable on any host
        # (its own c_ops import is guarded).
        from lmcache_ascend.v1.multiprocess.npu_driven_transfer import (
            _HAS_NATIVE_OBJECT_GROUP_TRANSFER as flag,
        )

        _HAS_NATIVE_OBJECT_GROUP_TRANSFER = bool(flag)
    return _HAS_NATIVE_OBJECT_GROUP_TRANSFER


def _dtype_elem_size(dtype: torch.dtype) -> int:
    """Element size in bytes for ``dtype`` (wraps ``torch.empty`` for any dtype)."""
    return torch.empty((), dtype=dtype).element_size()


def _max_tokens_per_subbatch(desc: "_NPUTransferDescriptor") -> int:
    """Max tokens a single batched staging buffer may hold under the cap.

    Staging is ``[kv_lead, num_layers, tokens, hidden]``; bound its byte size by
    :data:`_STAGING_CAP_BYTES`. Always returns at least one chunk's worth so a
    single oversized chunk still transfers (in one launch) rather than failing.
    """
    per_token = desc.staging_kv_lead * desc.num_layers * desc.hidden * _dtype_elem_size(
        desc.dtype
    )
    if per_token <= 0:
        return 1
    return max(desc.block_size, _STAGING_CAP_BYTES // per_token)


def _build_slot_mapping(
    block_ids: list[int], block_size: int, device: torch.device
) -> torch.Tensor:
    """Build a dense ``[num_tokens]`` slot-mapping ``block_id * block_size + j``.

    Accepts the full set of block ids for one batched transfer (one or many
    chunks) and produces the mapping in a single vectorized op. The kernel does
    not handle ``-1`` sentinels, so the mapping must be dense over the active
    tokens -- which it is here since ``block_ids`` holds only the vLLM block ids
    that back the active tokens.

    Args:
        block_ids: The vLLM block ids backing the tokens (in transfer order).
        block_size: Tokens per block (vLLM block size).
        device: Device on which to place the resulting mapping.

    Returns:
        An ``int64`` device tensor of length ``len(block_ids) * block_size``.

    Raises:
        ValueError: If ``block_ids`` is empty.
    """
    if not block_ids:
        raise ValueError("block_ids must be non-empty")
    bids = torch.tensor(block_ids, dtype=torch.int64)
    offsets = _offsets_cache.get(block_size)
    if offsets is None:
        offsets = torch.arange(block_size, dtype=torch.int64)
        _offsets_cache[block_size] = offsets
    slot_cpu = (bids[:, None] * block_size + offsets[None, :]).reshape(-1)
    slot = torch.empty(slot_cpu.shape, dtype=torch.int64, device=device)
    slot.copy_(slot_cpu, non_blocking=True)
    return slot


def _npu_gather_paged_kv_to_cpu(
    desc: _NPUTransferDescriptor,
    block_ids: list[int],
    blocks_per_chunk: int,
    out: Optional[list[torch.Tensor]],
    chunk_indices: Optional[list[int]],
) -> list[torch.Tensor]:
    """Gather paged NPU KV into CPU chunks via the fused kernel, batched.

    All needed chunks in a sub-batch share **one** paged->staging kernel launch
    (one concatenated slot-mapping over every active block), after which each
    chunk's gathered tokens are D2H-copied from its staging slice into its CPU
    slot. Sub-batching bounds the staging buffer to :data:`_STAGING_CAP_BYTES`.

    Honours the upstream ``out`` / ``chunk_indices`` contract: when ``out`` is
    provided (SHM path) each gathered chunk is written in place into
    ``out[out_idx]``; otherwise freshly allocated pinned CPU tensors are
    returned. With the SHM pool pinned (NpuPinMemoryBackend) the per-chunk D2H
    copies are async, so the whole sub-batch is one device kernel + N pipelined
    async copies awaited by the caller's single ``transfer_stream.synchronize()``.
    """
    # First Party — lazy.
    import lmcache_ascend.c_ops as lmc_ops

    num_chunks = len(block_ids) // blocks_per_chunk
    needed = list(chunk_indices) if chunk_indices is not None else list(range(num_chunks))
    chunks: list[torch.Tensor] = [] if out is None else out
    k1, k2, k3, k4 = desc.plane_extras

    tokens_per_chunk = blocks_per_chunk * desc.block_size
    max_tokens = _max_tokens_per_subbatch(desc)
    chunks_per_sub = max(1, max_tokens // tokens_per_chunk)

    # LMC-A: one paged->staging kernel launch per sub-batch, then per-chunk async
    # D2H from disjoint staging slices. All on the dedicated transfer stream so
    # the transfer does not contend with model kernels on the default stream;
    # completion is awaited once by the caller (submit_store, via
    # ``transfer_stream.synchronize()``) before commit.
    with torch.npu.stream(desc.transfer_stream):
        for sub_start in range(0, len(needed), chunks_per_sub):
            sub = needed[sub_start : sub_start + chunks_per_sub]
            sub_block_ids: list[int] = []
            for chunk_idx in sub:
                sub_block_ids.extend(
                    block_ids[
                        chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk
                    ]
                )
            if not sub_block_ids:
                continue
            slot_mapping = _build_slot_mapping(sub_block_ids, desc.block_size, desc.device)
            staging = desc.staging_for(kv_lead=desc.staging_kv_lead, tokens=len(sub_block_ids) * desc.block_size)

            # Paged KV -> NPU staging (device-to-device; no host memory involved).
            lmc_ops.multi_layer_kv_transfer(
                key_value=staging,
                key_value_ptrs=desc.ptr_table,
                slot_mapping=slot_mapping,
                paged_memory_device=desc.device,
                page_buffer_size=desc.page_buffer_size,
                direction=True,  # from_gpu: paged -> staging
                use_mla=desc.use_mla,
                kvcache_format_raw=desc.kv_format.value,
                k_hidden_dims=k1,
                v_hidden_dims=k2,
                dsa_hidden_dims=k3,
                dsa_c8_scale_plane_bytes=k4,
                paged_kv_block_size=desc.block_size,
            )

            # Per-chunk D2H from disjoint staging slices. ``non_blocking=True``
            # is async when the host dst is pinned (SHM pool pinned by
            # NpuPinMemoryBackend, or the freshly-allocated pinned CPU buffer);
            # otherwise torch falls back to a synchronous copy, still correctly
            # ordered before the caller's single stream sync.
            for k, _chunk_idx in enumerate(sub):
                t0 = k * tokens_per_chunk
                slc = staging[:, :, t0 : t0 + tokens_per_chunk]
                out_idx = sub_start + k
                if out is not None:
                    # The SHM slot's nominal shape is the server-negotiated shape
                    # (rank-4 [2, L, tokens, H] for SEPARATE_KV, rank-3
                    # [L, tokens, hidden] for MLA/DSA); ``view_as`` zero-copies
                    # it to the slice's staging shape, so one path covers all
                    # formats.
                    out[out_idx].view_as(slc).copy_(slc, non_blocking=True)
                else:
                    dst = torch.empty(
                        slc.shape, dtype=desc.dtype, device="cpu", pin_memory=True
                    )
                    dst.copy_(slc, non_blocking=True)
                    chunks.append(dst)

    return chunks


def _npu_scatter_cpu_to_paged_kv(
    desc: _NPUTransferDescriptor,
    block_ids: list[int],
    chunks: list[torch.Tensor],
    blocks_per_chunk: int,
    skip_first_n_tokens: int,
) -> None:
    """Scatter CPU chunks into paged NPU KV via the fused kernel, batched.

    Block-aligned ``skip_first_n_tokens`` handling mirrors the upstream helper.
    All effective (post-skip) tokens across chunks in a sub-batch share **one**
    staging->paged kernel launch: each chunk's effective slice is H2D-copied
    (async, when the SHM pool is pinned) into its disjoint offset of one
    contiguous staging buffer, then one ``multi_layer_kv_transfer`` scatters the
    whole sub-batch. Sub-batching bounds the staging buffer to
    :data:`_STAGING_CAP_BYTES`.
    """
    # First Party — lazy.
    import lmcache_ascend.c_ops as lmc_ops

    num_chunks = len(block_ids) // blocks_per_chunk
    k1, k2, k3, k4 = desc.plane_extras

    # First pass: derive each chunk's effective (post-skip) block ids + host
    # slice (pure metadata + view, no device work). ``eff_block_ids`` is dense
    # over the active tokens, as the kernel requires.
    plan: list[tuple[list[int], torch.Tensor, int]] = []  # (eff_block_ids, src_slice, eff_tokens)
    for chunk_idx in range(min(num_chunks, len(chunks))):
        chunk_block_ids = list(
            block_ids[
                chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk
            ]
        )
        chunk_start = chunk_idx * blocks_per_chunk * desc.block_size
        chunk_end = chunk_start + len(chunk_block_ids) * desc.block_size
        effective_start = max(chunk_start, skip_first_n_tokens)
        if effective_start >= chunk_end:
            continue

        skip_blocks = (effective_start - chunk_start) // desc.block_size
        skip_tokens = skip_blocks * desc.block_size
        eff_block_ids = chunk_block_ids[skip_blocks:]

        src = chunks[chunk_idx]
        # LMC-A: MLA/DSA SHM slots are rank-3 [L, tokens, hidden] (the server's
        # MLA branch); canonicalise to the kernel's [1, L, tokens, hidden] view
        # before slicing tokens. SEPARATE_KV slots are already rank-4
        # [2, L, tokens, H] and are left unchanged.
        if desc.staging_kv_lead == 1:
            src = src.reshape(desc.staging_kv_lead, desc.num_layers, -1, desc.hidden)
        src_slice = src[:, :, skip_tokens:] if skip_tokens else src
        plan.append((eff_block_ids, src_slice, len(eff_block_ids) * desc.block_size))

    if not plan:
        return

    max_tokens = _max_tokens_per_subbatch(desc)

    # LMC-A: one staging->paged kernel launch per sub-batch on the dedicated
    # transfer stream (mirrors gather). Per-chunk H2D into disjoint staging
    # offsets is async when the SHM pool is pinned; the caller's single
    # ``transfer_stream.synchronize()`` (submit_retrieve) awaits completion
    # before the SHM slot is released.
    with torch.npu.stream(desc.transfer_stream):
        i = 0
        while i < len(plan):
            sub_block_ids: list[int] = []
            sub_slices: list[tuple[torch.Tensor, int]] = []  # (src_slice, eff_tokens)
            sub_tokens = 0
            while i < len(plan):
                eff_block_ids, src_slice, eff_tokens = plan[i]
                if sub_slices and sub_tokens + eff_tokens > max_tokens:
                    break  # would exceed the cap; flush this sub-batch
                sub_block_ids.extend(eff_block_ids)
                sub_slices.append((src_slice, eff_tokens))
                sub_tokens += eff_tokens
                i += 1
            if not sub_block_ids:
                continue

            staging = desc.staging_for(
                kv_lead=desc.staging_kv_lead, tokens=sub_tokens
            )
            off = 0
            for src_slice, eff_tokens in sub_slices:
                # H2D the host slice into its disjoint staging offset. The dst
                # view is a strided region of the contiguous staging buffer;
                # ``copy_`` handles the non-contiguity (a sliced host chunk is
                # not contiguous across layers, as in the per-chunk path).
                staging[:, :, off : off + eff_tokens].copy_(src_slice, non_blocking=True)
                off += eff_tokens

            slot_mapping = _build_slot_mapping(sub_block_ids, desc.block_size, desc.device)
            lmc_ops.multi_layer_kv_transfer(
                key_value=staging,
                key_value_ptrs=desc.ptr_table,
                slot_mapping=slot_mapping,
                paged_memory_device=desc.device,
                page_buffer_size=desc.page_buffer_size,
                direction=False,  # to_gpu: staging -> paged
                use_mla=desc.use_mla,
                kvcache_format_raw=desc.kv_format.value,
                k_hidden_dims=k1,
                v_hidden_dims=k2,
                dsa_hidden_dims=k3,
                dsa_c8_scale_plane_bytes=k4,
                paged_kv_block_size=desc.block_size,
            )


# --- Override installation -------------------------------------------------

_orig_gather: Optional[object] = None
_orig_scatter: Optional[object] = None
# LMC-A: originals of EngineDrivenTransferContext.submit_store / submit_retrieve,
# saved so the NPU-aware wrappers below can fall back to them for non-NPU /
# unsupported-layout workers.
_orig_submit_store: Optional[object] = None
_orig_submit_retrieve: Optional[object] = None
# LMC-A: original of compute_kv_layout, saved so the NPU-aware wrapper below
# can correct the Ascend MLA/DSA chunk-shape contract and pass every other case
# through unchanged.
_orig_compute_kv_layout: Optional[object] = None
# LMC-A: original of the upstream LMCache-driven transfer entry point,
# saved so the block-level NPU port can be installed idempotently (and so
# tests / future fallbacks can restore the upstream behaviour).
_orig_lmcache_driven_transfer_fn: Optional[object] = None


def _install_lmcache_driven_overrides() -> None:
    """Wire the LMCache-driven block-level MP path for NPU (design doc 6.1).

    Two registrations behind the ``_HAS_NATIVE_OBJECT_GROUP_TRANSFER``
    feature flag; without the compiled block kernels both are skipped and
    upstream behaves exactly as today (``auto`` / ``engine_driven`` keep the
    token-level fused path; ``lmcache_driven`` raises upstream's clear
    "no KV-cache wrapper factory" error):

    * **Server side** — upstream ``LMCacheDrivenTransferModule``'s STORE /
      RETRIEVE handlers resolve ``transfer_kv_per_object_group`` from their
      module globals at call time, so rebinding it in the
      ``lmcache_driven_transfer`` namespace redirects every call to the
      Ascend port (block-level plan fast path + per-batch fallback). The
      port resolves ``lmcache_ascend.c_ops`` by its own name, immune to the
      ``sys.modules["lmcache.c_ops"]`` swap ordering (see
      :mod:`npu_driven_transfer`'s module docstring).
    * **Worker side** — register the NPU KV-wrapper factory
      (:class:`AscendIPCWrapper`) in the platform registry so
      ``create_transfer_context(mode=lmcache_driven)`` can build
      ``LMCacheDrivenTransferContext`` for NPU kv_caches
      (``wrap_one_kv_cache`` → ``get_kv_wrapper_factory("npu")``). Selection
      stays upstream's contract: ``LMCACHE_MP_TRANSFER_MODE=lmcache_driven``
      or the ``lmcache.mp.mp_transfer_mode`` extra config; ``auto`` (the
      default) keeps routing NPU to the engine-driven token-level path.
    """
    if not _native_object_group_transfer_available():
        logger.info(
            "LMCache-driven NPU block transfer not installed: "
            "lmcache_ascend.c_ops lacks execute_object_group_transfer"
        )
        return

    # Third Party
    from lmcache.v1.platform import _registry as platform_registry

    # First Party
    from lmcache_ascend.v1.multiprocess import npu_driven_transfer
    from lmcache_ascend.v1.multiprocess.custom_types import AscendIPCWrapper

    # Third Party
    import lmcache.v1.multiprocess.modules.lmcache_driven_transfer as ldt

    global _orig_lmcache_driven_transfer_fn
    if _orig_lmcache_driven_transfer_fn is None:
        _orig_lmcache_driven_transfer_fn = ldt.transfer_kv_per_object_group
    ldt.transfer_kv_per_object_group = (  # type: ignore[assignment]
        npu_driven_transfer.transfer_kv_per_object_group
    )

    # Idempotent: register_kv_wrapper overwrites with the same class.
    platform_registry.register_kv_wrapper("npu", AscendIPCWrapper)

    logger.info(
        "Installed NPU block-level LMCache-driven MP transfer "
        "(enable via LMCACHE_MP_TRANSFER_MODE=lmcache_driven)"
    )


def _gather_wrapper(
    kv_caches: dict[str, object],
    block_ids: list[int],
    blocks_per_chunk: int,
    layout_hints: Optional[object] = None,
    engine_kv_format: Optional[object] = None,
    out: Optional[list[torch.Tensor]] = None,
    chunk_indices: Optional[list[int]] = None,
) -> list[torch.Tensor]:
    """Dispatch gather to the fused NPU path when applicable, else upstream."""
    desc = _get_descriptor(kv_caches)
    if desc is not None:
        return _npu_gather_paged_kv_to_cpu(
            desc, block_ids, blocks_per_chunk, out, chunk_indices
        )
    assert _orig_gather is not None
    return _orig_gather(  # type: ignore[misc]
        kv_caches,
        block_ids,
        blocks_per_chunk,
        layout_hints=layout_hints,
        engine_kv_format=engine_kv_format,
        out=out,
        chunk_indices=chunk_indices,
    )


def _scatter_wrapper(
    kv_caches: dict[str, object],
    block_ids: list[int],
    chunks: list[torch.Tensor],
    blocks_per_chunk: int,
    skip_first_n_tokens: int = 0,
    layout_hints: Optional[object] = None,
    engine_kv_format: Optional[object] = None,
) -> None:
    """Dispatch scatter to the fused NPU path when applicable, else upstream."""
    desc = _get_descriptor(kv_caches)
    if desc is not None:
        _npu_scatter_cpu_to_paged_kv(
            desc, block_ids, chunks, blocks_per_chunk, skip_first_n_tokens
        )
        return
    assert _orig_scatter is not None
    _orig_scatter(  # type: ignore[misc]
        kv_caches,
        block_ids,
        chunks,
        blocks_per_chunk,
        skip_first_n_tokens=skip_first_n_tokens,
        layout_hints=layout_hints,
        engine_kv_format=engine_kv_format,
    )


def _flatten_single_group_block_ids(block_ids: list[list[int]]) -> list[int]:
    """Flatten the single per-group block-id list for engine-driven transfer.

    Mirrors upstream ``_single_group_block_ids`` (worker_transfer.py) without
    importing that private helper. The NPU fused path, like upstream's
    engine-driven path, handles one KV cache group only; multi-group transfers
    are rejected here.
    """
    if len(block_ids) != 1:
        raise RuntimeError(
            "engine-driven transfer does not support hybrid KV cache groups"
        )
    return block_ids[0]


def _submit_store_wrapper(
    self,
    _request_id: object,
    key: object,
    instance_id: int,
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[list[int]],
    _event: object,
    blocks_in_chunk: int,
) -> MessagingFuture:
    """NPU-aware ``EngineDrivenTransferContext.submit_store``.

    Falls back to the upstream method when ``kv_caches`` is not a supported NPU
    layout (``_get_descriptor`` returns ``None``). Otherwise it replaces
    upstream's whole-device pre-gather sync (worker_transfer.py:429) with a
    non-blocking ``transfer_stream.wait_stream(torch.npu.current_stream())``
    (mirrors the in-process connector at npu_connectors.py:1126). The passed
    ``_event`` is intentionally unused: the deployed vLLM connector resolves the
    "forward-completion event" to a CPU-runner ``_EventPlaceholder`` (which has
    no ``.wait``), so ordering via the current (forward) stream is both robust
    and exactly what the in-process path already does. The fused gather then
    runs on the transfer stream, and the post-gather whole-device sync (:448)
    becomes a single ``transfer_stream.synchronize()`` before commit.
    """
    if self._engine_driven_context is None:
        raise RuntimeError(
            "Engine-driven transfer context is not registered. "
            "Call register() before submit_store()."
        )
    desc = _get_descriptor(kv_caches)
    if desc is None:
        assert _orig_submit_store is not None
        return _orig_submit_store(  # type: ignore[misc]
            self,
            _request_id,
            key,
            instance_id,
            kv_caches,
            block_ids,
            _event,
            blocks_in_chunk,
        )

    # Order the transfer stream against the forward that wrote the paged KV
    # (which ran on the current/compute stream), replacing the whole-device
    # pre-gather sync (worker_transfer.py:429). The passed _event is a vLLM
    # _EventPlaceholder and is intentionally unused here.
    desc.transfer_stream.wait_stream(torch.npu.current_stream())
    result = self._engine_driven_context.prepare_store(key, instance_id)
    out_buffers, chunk_indices = result if result is not None else (None, None)
    # All chunks already in cache — nothing to gather or commit.
    if chunk_indices is not None and len(chunk_indices) == 0:
        future: MessagingFuture[bool] = MessagingFuture()
        future.set_result(True)
        return future
    cpu_chunks = _gather_wrapper(
        kv_caches,
        _flatten_single_group_block_ids(block_ids),
        blocks_in_chunk,
        layout_hints=self._layout_hints,
        engine_kv_format=self._engine_kv_format,
        out=out_buffers,
        chunk_indices=chunk_indices,
    )
    # Complete the async D2H on the transfer stream before commit, replacing
    # the whole-device sync at worker_transfer.py:448.
    desc.transfer_stream.synchronize()
    ok = self._engine_driven_context.commit_store(key, instance_id, cpu_chunks)

    future = MessagingFuture()
    future.set_result(ok)
    return future


def _submit_retrieve_wrapper(
    self,
    _request_id: object,
    key: object,
    instance_id: int,
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[list[int]],
    _event: object,
    blocks_in_chunk: int,
    skip_first_n_tokens: int = 0,
) -> MessagingFuture:
    """NPU-aware ``EngineDrivenTransferContext.submit_retrieve``.

    Falls back to the upstream method for unsupported layouts. Otherwise it runs
    the fused scatter on the transfer stream and replaces the whole-device
    post-scatter sync (worker_transfer.py:490) with a single
    ``transfer_stream.synchronize()`` before the SHM slot is released
    (``commit_retrieve``). ``event`` is unused here, matching upstream.
    """
    if self._engine_driven_context is None:
        raise RuntimeError(
            "Engine-driven transfer context is not registered. "
            "Call register() before submit_retrieve()."
        )
    desc = _get_descriptor(kv_caches)
    if desc is None:
        assert _orig_submit_retrieve is not None
        return _orig_submit_retrieve(  # type: ignore[misc]
            self,
            _request_id,
            key,
            instance_id,
            kv_caches,
            block_ids,
            _event,
            blocks_in_chunk,
            skip_first_n_tokens=skip_first_n_tokens,
        )

    src_buffers = self._engine_driven_context.prepare_retrieve(key, instance_id)
    ok = src_buffers is not None
    if src_buffers is not None:
        try:
            _scatter_wrapper(
                kv_caches,
                _flatten_single_group_block_ids(block_ids),
                src_buffers,
                blocks_in_chunk,
                skip_first_n_tokens=skip_first_n_tokens,
                layout_hints=self._layout_hints,
                engine_kv_format=self._engine_kv_format,
            )
        except (RuntimeError, ValueError, TypeError, IndexError):
            logger.exception("Failed to scatter retrieved CPU context chunks")
            ok = False
        # Ensure the scatter's device writes are complete before releasing the
        # SHM slot, replacing the whole-device sync at worker_transfer.py:490.
        desc.transfer_stream.synchronize()
    self._engine_driven_context.commit_retrieve(key, instance_id)

    future: MessagingFuture[bool] = MessagingFuture()
    future.set_result(ok)
    return future


def _compute_kv_layout_wrapper(
    kv_caches: dict[str, object],
    layout_hints: Optional[object] = None,
) -> "tuple[int, int, int, str, Any]":
    """NPU-aware ``compute_kv_layout`` for the engine-driven SHM shape contract.

    Upstream reports ``num_kv_heads * kv_lora_rank`` (the K-plane product) as
    the chunk hidden dim for Ascend ``(K, V)`` tuples, and a non-MLA format, so
    the SHM server would allocate ``[2, L, tokens, num_kv_heads*kv_lora_rank]``
    — wrong for the fused MLA/DSA kernel, which needs the planes concatenated
    as ``[num_layers, tokens, kv_lora_rank+qk_rope_head_dim(+dsa)]``. For a
    supported NPU MLA/DSA layout this returns the summed hidden dim (from the
    descriptor, the single source of truth) and an MLA format flag
    (``NL_X_NB_BS_HS``, the value :func:`is_mla` accepts) so the server's
    ``use_mla`` branch allocates the matching rank-3 buffer; the gather/scatter
    then ``view_as`` it to the kernel's ``[1, ...]`` staging shape. Every other
    case (SEPARATE_KV, CPU, CUDA, unsupported) is passed through unchanged.

    The format flag is a signalling mechanism for the server's shape branch
    only — the actual kernel call is driven by the descriptor's real Ascend
    ``KVCacheFormat`` (MLA_KV/DSA_KV), so the stored ``_engine_kv_format`` is
    never used to interpret Ascend tensors.
    """
    assert _orig_compute_kv_layout is not None
    result = _orig_compute_kv_layout(  # type: ignore[misc]
        kv_caches, layout_hints=layout_hints
    )
    desc = _get_descriptor(kv_caches)
    if desc is not None and desc.kv_format in (
        KVCacheFormat.MLA_KV,
        KVCacheFormat.DSA_KV,
    ):
        # First Party — lazy: resolve the MLA format flag is_mla() accepts.
        import lmcache_ascend.c_ops as lmc_ops

        block_size, num_layers, _hidden, dtype_str, _fmt = result
        return (
            block_size,
            num_layers,
            desc.hidden,
            dtype_str,
            lmc_ops.EngineKVFormat.NL_X_NB_BS_HS,
        )
    return result


def install_overrides() -> None:
    """Replace the upstream gather/scatter + submit callables with NPU dispatchers.

    Idempotent.  Patches both ``base`` (definition site) and ``worker_transfer``
    (import binding used by ``DataTransferContext`` at call time) for the
    gather/scatter names, because the upstream adapter imports them by value.

    Additionally patches ``EngineDrivenTransferContext.submit_store`` /
    ``submit_retrieve`` so the NPU path orders the fused transfer on a dedicated
    stream via the forward-completion event instead of upstream's whole-device
    ``torch_dev.synchronize()`` calls. Non-NPU / unsupported-layout workers fall
    back to the saved originals unchanged.

    Finally patches ``compute_kv_layout`` (on ``base`` and ``worker_transfer``)
    so the SHM server allocates the rank-3 buffer the fused MLA/DSA kernel
    consumes; see :func:`_compute_kv_layout_wrapper`, and wires the
    LMCache-driven block-level MP path via
    :func:`_install_lmcache_driven_overrides` (no-op without the compiled block
    kernels).
    """
    global _orig_gather
    global _orig_scatter
    global _orig_submit_store
    global _orig_submit_retrieve
    global _orig_compute_kv_layout

    # Third Party
    import lmcache.v1.multiprocess.transfer_context.base as base
    import lmcache.v1.multiprocess.transfer_context.worker_transfer as wt

    if _orig_gather is None:
        _orig_gather = base.gather_paged_kv_to_cpu
        _orig_scatter = base.scatter_cpu_to_paged_kv

    base.gather_paged_kv_to_cpu = _gather_wrapper  # type: ignore[assignment]
    base.scatter_cpu_to_paged_kv = _scatter_wrapper  # type: ignore[assignment]
    wt.gather_paged_kv_to_cpu = _gather_wrapper  # type: ignore[assignment]
    wt.scatter_cpu_to_paged_kv = _scatter_wrapper  # type: ignore[assignment]

    if _orig_submit_store is None:
        _orig_submit_store = wt.EngineDrivenTransferContext.submit_store
        _orig_submit_retrieve = wt.EngineDrivenTransferContext.submit_retrieve
    wt.EngineDrivenTransferContext.submit_store = (  # type: ignore[assignment]
        _submit_store_wrapper
    )
    wt.EngineDrivenTransferContext.submit_retrieve = (  # type: ignore[assignment]
        _submit_retrieve_wrapper
    )

    # LMC-A: patch compute_kv_layout so the SHM server allocates the rank-3
    # buffer the fused MLA/DSA kernel consumes (see _compute_kv_layout_wrapper).
    # Both base (definition) and wt (the binding register() resolves) are
    # patched, matching the gather/scatter handling above.
    if _orig_compute_kv_layout is None:
        _orig_compute_kv_layout = base.compute_kv_layout
    base.compute_kv_layout = _compute_kv_layout_wrapper  # type: ignore[assignment]
    wt.compute_kv_layout = _compute_kv_layout_wrapper  # type: ignore[assignment]

    # LMC-A: wire the LMCache-driven block-level MP path (design doc 6.1).
    # No-op without the compiled block kernels, so this stays safe on hosts
    # where the extension lacks execute_object_group_transfer.
    _install_lmcache_driven_overrides()

    logger.info(
        "Installed NPU fused gather/scatter + transfer-stream submit override "
        "for MP non-GPU transfer (supported formats: %s)",
        ", ".join(f.name for f in _SUPPORTED_FORMATS),
    )
    logger.info(
        "Installed NPU MLA/DSA chunk-shape negotiation patch "
        "(compute_kv_layout) for MP engine-driven transfer"
    )
