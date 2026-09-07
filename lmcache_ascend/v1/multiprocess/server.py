# SPDX-License-Identifier: Apache-2.0
"""NPU cache-context backend and MP cache-server entry point.

Implements the upstream ``BaseCacheContext`` MP interface surface (MP-mode
design doc section 3.4) for vLLM-Ascend workers:

* :class:`NPUCacheContext` -- the ``device_type = "npu"`` backend, built from
  the per-layer ``(K, V)`` tuple layout (``SEPARATE_KV``, engine format
  ``NL_X_TWO_X_NB_BS_NH_HS``).  :func:`register_npu_cache_context` installs it
  into the platform cache-context table, so upstream
  ``LMCacheDrivenTransferModule.register_kv_cache`` → ``create_cache_context``
  dispatches to it unchanged (layout-registry / liveness / reaping bookkeeping
  all comes from upstream; no request-handler monkey-patching).
* :class:`_TempNPUBuffer` -- port of upstream ``_TempGPUBuffer``
  (platform/cuda/cache_context.py): one flat uint8 NPU staging buffer carved
  into (batch, object group, kernel group) slots.
* Module ``__main__`` -- the LMCache-Ascend cache-server entry point (the
  upstream ``lmcache.v1.multiprocess.server`` CLI plus backend registration).

First-phase scope (MP-mode design doc): FullAttention ``SEPARATE_KV`` on
910B/C.  Anything else raises ``NotImplementedError`` at construction time,
before any tensor work, so the worker sees the failure at registration.
"""

# Standard
from collections.abc import Sequence
from typing import Any, Optional
import array

# Third Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.multiprocess.custom_types import KVCache
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.server import parse_args, run_cache_server
from lmcache.v1.platform.base_cache_context import BaseCacheContext
import torch

# First Party
from lmcache_ascend.v1.kv_format import KVCacheFormat
from lmcache_ascend.v1.kv_layer_groups import build_kv_layer_groups
import lmcache_ascend.c_ops as lmc_ops

logger = init_logger(__name__)


def unwrap_kv_cache_tensors(kv_caches: KVCache) -> list[torch.Tensor]:
    """Materialize the per-layer KV entries from their IPC wrappers."""
    unwrapped_tensors = []
    for ipc_wrapper in kv_caches:
        tensor = ipc_wrapper.to_tensor()
        unwrapped_tensors.append(tensor)
    return unwrapped_tensors


def list_to_gpu_tensor(lis: list[int], device: torch.device) -> torch.Tensor:
    """Upload an int64 pointer list to *device* as a 1-D tensor."""
    return torch.frombuffer(array.array("q", lis), dtype=torch.long).to(
        device, non_blocking=True
    )


class _NPUExternalStream:
    """Minimal stand-in for the ``cupy.cuda.ExternalStream`` contract.

    Upstream's store/retrieve plumbing reads only ``stream.ptr``
    (``submit_callback_to_stream`` / ``event_bus.publish_on_stream``); both
    recorders' non-CUDA fallbacks ignore the value (NPU has no
    stream-ordered host-callback support), so the raw handle is best-effort.
    ``launch_host_func`` mirrors the CPU ``StubStream`` semantics (run
    synchronously, swallow errors).
    """

    def __init__(self, npu_stream: Any) -> None:
        self._npu_stream = npu_stream

    @property
    def ptr(self) -> int:
        for attr in ("npu_stream", "cuda_stream"):
            value = getattr(self._npu_stream, attr, None)
            if isinstance(value, int):
                return value
        return 0

    def launch_host_func(self, callback: Any, arg: Any = None) -> None:
        try:
            callback(arg)
        except Exception as e:  # noqa: BLE001
            logger.warning("launch_host_func callback raised: %s", e)

    def synchronize(self) -> None:
        self._npu_stream.synchronize()


class _TempNPUBuffer:
    """
    Manages the temporary NPU buffer for NPUCacheContext.

    Port of upstream ``_TempGPUBuffer`` (platform/cuda/cache_context.py);
    only the backing device differs. The logical layout of the temp NPU
    buffer is (batch size, object group, kernel group).

    Here is an example of batch size = 4, with 2 object groups,
    and 2 kernel groups per object group:
    [
        batch 0:
            - object group 0: kernel group 0 | kernel group 1 | ...
            - object group 1: kernel group 2 | kernel group 3 | ...

        batch 1:
            - object group 0: kernel group 0 | kernel group 1 | ...
            - object group 1: kernel group 2 | kernel group 3 | ...

        batch 2:
            - object group 0: kernel group 0 | kernel group 1 | ...
            - object group 1: kernel group 2 | kernel group 3 | ...

        batch 3:
            - object group 0: kernel group 0 | kernel group 1 | ...
            - object group 1: kernel group 2 | kernel group 3 | ...
    ]

    During the multi-layer copy kernel launch, we will do it at kernel
    group level, which means we will have:
    ```
    gpu_buffers = [
        get_temp_kernel_group_buffer(batch_idx, kernel_group_idx)
        for batch_idx in range(batch_size)
    ]
    ```

    During the lmcache_memcpy_async launch, we will do it at the object group
    level, which will be:
    ```
    for i in range(batch_size):
        gpu_buffer = get_temp_object_group_buffer(batch_idx, object_group_idx)
        lmcache_memcpy_async(...)
    ```
    """

    def __init__(
        self,
        kv_layer_groups_manager: KVLayerGroupsManager,
        lmcache_tokens_per_chunk: int,
        device: torch.device,
        max_batch_size: int = 4,
    ) -> None:
        self._kv_groups_manager = kv_layer_groups_manager
        self._lmcache_tokens_per_chunk = lmcache_tokens_per_chunk
        self._max_batch_size = max_batch_size

        self._temp_buffer = torch.empty(
            self._get_size_for_single_batch() * max_batch_size,
            dtype=torch.uint8,
            device=device,
        )

        # Offset map: (batch_idx, object_group_idx, kernel_group_idx) ->
        # (byte offset in the temp buffer, size of the buffer in bytes)
        self._offset_map: dict[tuple[int, int, int], tuple[int, int]] = {}

        # (batch_idx, kernel_group_idx) -> (byte offset for the kernel group,
        # size of the buffer in bytes).
        self._offset_map_kernel_group_only: dict[tuple[int, int], tuple[int, int]] = {}

        # (batch_idx, object_group_idx) -> (byte offset for the object group,
        # size of the buffer in bytes)
        self._offset_map_object_group_only: dict[tuple[int, int], tuple[int, int]] = {}

        offset = 0
        for batch_idx in range(max_batch_size):
            for object_group_idx in range(self._kv_groups_manager.num_object_groups):
                object_group_size = 0
                object_group_start_offset = offset

                for kernel_group_idx in self._kv_groups_manager.object_groups[
                    object_group_idx
                ].kernel_group_indices:
                    key = (batch_idx, object_group_idx, kernel_group_idx)
                    key2 = (batch_idx, kernel_group_idx)

                    size = self._get_size_for_kernel_group(kernel_group_idx)
                    self._offset_map[key] = (offset, size)
                    self._offset_map_kernel_group_only[key2] = (offset, size)

                    offset += size
                    object_group_size += size

                key3 = (batch_idx, object_group_idx)
                self._offset_map_object_group_only[key3] = (
                    object_group_start_offset,
                    object_group_size,
                )

        # Shape/dtype cache for kernel groups
        self._shape_cache_kernel_group: dict[int, tuple[torch.Size, torch.dtype]] = {}
        for kernel_group_idx in range(self._kv_groups_manager.num_kernel_groups):
            shape = self._get_shape_for_kernel_group(
                self._lmcache_tokens_per_chunk, kernel_group_idx
            )
            group = self._kv_groups_manager.kernel_groups[kernel_group_idx]
            dtype = group.dtype
            self._shape_cache_kernel_group[kernel_group_idx] = (shape, dtype)

    # Public APIs
    @property
    def max_batch_size(self) -> int:
        """Maximum number of chunks (batch slots) the buffer holds."""
        return self._max_batch_size

    @property
    def buffer(self) -> torch.Tensor:
        """The flat staging tensor."""
        return self._temp_buffer

    def get_temp_kernel_group_buffer(
        self, batch_idx: int, kernel_group_idx: int
    ) -> torch.Tensor:
        """
        Returns the temp NPU buffer for the given batch index and kernel group
        index. The returned buffer is with the correct shape and dtype for the
        kernel group.

        Args:
            batch_idx: Index of the batch (0 <= batch_idx < max_batch_size)
            kernel_group_idx: Index of the kernel group.

        Returns:
            The temp NPU buffer for the given batch index and kernel group
            index.

        Raises:
            ValueError: If the batch_idx or kernel_group_idx is out of range.
        """
        key = (batch_idx, kernel_group_idx)
        if key not in self._offset_map_kernel_group_only:
            raise ValueError(
                f"Invalid batch_idx {batch_idx} or kernel_group_idx {kernel_group_idx}"
            )

        offset, size = self._offset_map_kernel_group_only[key]
        shape, dtype = self._shape_cache_kernel_group[kernel_group_idx]
        return self._temp_buffer[offset : offset + size].view(dtype).view(shape)

    def get_temp_object_group_buffer(
        self, batch_idx: int, object_group_idx: int
    ) -> torch.Tensor:
        """
        Returns the temp NPU buffer for the given batch index and object group
        index. The returned buffer is a flat uint8 raw tensor.

        Args:
            batch_idx: Index of the batch (0 <= batch_idx < max_batch_size)
            object_group_idx: Index of the object group.

        Returns:
            The temp NPU buffer for the given batch index and object group
            index.
        """
        key = (batch_idx, object_group_idx)
        if key not in self._offset_map_object_group_only:
            raise ValueError(
                f"Invalid batch_idx {batch_idx} or object_group_idx {object_group_idx}"
            )

        offset, size = self._offset_map_object_group_only[key]
        return self._temp_buffer[offset : offset + size]

    def get_kernel_group_shape_dtype(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> tuple[torch.Size, torch.dtype]:
        """
        Returns the shape and dtype for the given kernel group index and
        number of tokens.

        Will be exported by NPUCacheContext and used to construct the
        MemoryLayoutDesc

        Args:
            num_tokens: Number of tokens. Must be a whole number of lmcache
                chunk size.
            kernel_group_idx: Index of the kernel group.

        Returns:
            The shape and dtype for the given kernel group index and
            number of tokens.
        """
        _, dtype = self._shape_cache_kernel_group[kernel_group_idx]
        shape = self._get_shape_for_kernel_group(num_tokens, kernel_group_idx)

        return shape, dtype

    def get_cache_size_per_token(self) -> int:
        """
        Returns the cache size per token (in bytes), summed across all kernel
        groups.
        """
        return self._get_size_for_single_batch() // self._lmcache_tokens_per_chunk

    # Helper functions
    def _get_shape_for_kernel_group(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> torch.Size:
        """
        Returns the shape of the temp NPU buffer for the given kernel group
        index

        Args:
            num_tokens: Number of tokens
            kernel_group_idx: Index of the kernel group.

        Returns:
            The shape of the temp NPU buffer for the given kernel group index

        Raises:
            ValueError: If ``num_tokens`` is not a whole number of LMCache
                chunks.
        """
        if num_tokens % self._lmcache_tokens_per_chunk != 0:
            raise ValueError(
                f"num_tokens ({num_tokens}) must be a multiple of "
                f"lmcache_tokens_per_chunk ({self._lmcache_tokens_per_chunk})"
            )

        group = self._kv_groups_manager.kernel_groups[kernel_group_idx]
        sd = group.shape_desc

        num_chunks = num_tokens // self._lmcache_tokens_per_chunk
        num_slots = (
            self._kv_groups_manager.get_slots_per_chunk_in_sw(kernel_group_idx)
            * num_chunks
        )

        return torch.Size(
            (sd.kv_size, group.num_layers, num_slots, group.hidden_dim_size)
        )

    def _get_size_for_kernel_group(self, kernel_group_idx: int) -> int:
        """
        Returns the size in bytes of the temp NPU buffer for the given kernel
        group index

        **Assumes the size is lmcache_tokens_per_chunk

        Will only be called during initialization
        """
        shape = self._get_shape_for_kernel_group(
            self._lmcache_tokens_per_chunk, kernel_group_idx
        )
        kernel_group = self._kv_groups_manager.kernel_groups[kernel_group_idx]
        dtype = kernel_group.dtype
        return shape.numel() * dtype.itemsize

    def _get_size_for_object_group(self, object_group_idx: int) -> int:
        """
        Returns the size in bytes of the temp NPU buffer for the given object
        group index

        **Assumes the size is lmcache_tokens_per_chunk

        Will only be called during initialization
        """
        object_group = self._kv_groups_manager.object_groups[object_group_idx]
        return sum(
            self._get_size_for_kernel_group(kernel_group_idx)
            for kernel_group_idx in object_group.kernel_group_indices
        )

    def _get_size_for_single_batch(self) -> int:
        """
        Returns the size in bytes of the temp NPU buffer for a single batch
        (i.e., a single chunk)

        **Assumes the size is lmcache_tokens_per_chunk
        """
        return sum(
            self._get_size_for_object_group(object_group_idx)
            for object_group_idx in range(self._kv_groups_manager.num_object_groups)
        )


def _first_layer_tensor(layers: list) -> torch.Tensor:
    """Return the representative tensor of the first per-layer entry."""
    first = layers[0]
    if isinstance(first, (tuple, list)):
        return first[0]
    return first


class NPUCacheContext(BaseCacheContext):
    """
    Manages the shape and pointers to vLLM-Ascend NPU KV cache tensors.

    Mirrors upstream ``GPUCacheContext`` (platform/cuda/cache_context.py):
    same ``BaseCacheContext`` interface, same ``__init__`` signature (the
    platform factory calls every argument positionally).  The layout
    discovery differs: instead of upstream's GPUKVFormat-keyed accessors, the
    Ascend ``KVLayerGroupsManager`` body (:func:`build_kv_layer_groups`)
    classifies the per-layer tuples directly, and every kernel group is
    stamped with ``EngineKVFormat.NL_X_TWO_X_NB_BS_NH_HS`` (the engine format
    matching ``SEPARATE_KV``, MP-mode design doc section 1.2).
    """

    device_type = "npu"

    def __init__(
        self,
        kv_caches: KVCache,
        lmcache_tokens_per_chunk: int = 256,
        layout_hints: Optional[LayoutHints] = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,  # noqa: ARG002
        separate_object_groups: bool = True,
        full_sw_kv: bool = False,
    ):
        # First Party — lazy: keeps this module importable without the NPU
        # connector's heavier import graph.
        from lmcache_ascend.v1.npu_connector.npu_connectors import (
            _pointers_for_entry,
            is_310p,
        )

        unwrapped = unwrap_kv_cache_tensors(kv_caches)
        if not unwrapped:
            raise ValueError("NPUCacheContext requires a non-empty kv_caches list")

        kv_format = KVCacheFormat.detect(unwrapped)
        first = unwrapped[0]
        if kv_format != KVCacheFormat.SEPARATE_KV or not (
            isinstance(first, (tuple, list))
            and len(first) == 2
            and all(isinstance(t, torch.Tensor) and t.ndim == 4 for t in first)
            and first[0].shape == first[1].shape
        ):
            raise NotImplementedError(
                "NPUCacheContext (first phase) only supports the vLLM-Ascend "
                "per-layer (K, V) tuple layout [nb, bs, nh, hs] "
                f"(SEPARATE_KV); detected {kv_format.name}"
            )
        if is_310p():
            raise NotImplementedError(
                "NPUCacheContext (first phase) targets 910B/C; 310P is not "
                "supported"
            )

        # engine_group_infos drive upstream's mixed-format grouping; the
        # first-phase Ascend path groups by direct tensor introspection, so
        # they are accepted but unused.
        if engine_group_infos:
            logger.debug(
                "NPUCacheContext ignores %d engine group infos (first phase)",
                len(engine_group_infos),
            )

        ref = _first_layer_tensor(unwrapped)
        self.device_: torch.device = ref.device
        num_blocks = int(ref.shape[0])
        num_layers_val = len(unwrapped)

        kv_layer_groups_manager = KVLayerGroupsManager.__new__(KVLayerGroupsManager)
        build_kv_layer_groups(
            kv_layer_groups_manager,
            unwrapped,
            kv_format=kv_format,
            num_blocks=num_blocks,
            is_310p=False,
            layout_hints=layout_hints,
            lmcache_logical_chunk_size=lmcache_tokens_per_chunk,
        )
        # build_kv_layer_groups (the in-process connector's builder) does not
        # stamp the fields the upstream manager helpers read; fill them here.
        # First phase is full attention: tokens_per_block follows the
        # compress ratio (1 by default), sw stays -1 (no windowing).
        kv_layer_groups_manager._separate_object_groups = bool(separate_object_groups)
        kv_layer_groups_manager._full_sw_kv = False
        for group in kv_layer_groups_manager.kernel_groups:
            group.engine_kv_format = lmc_ops.EngineKVFormat.NL_X_TWO_X_NB_BS_NH_HS
            group.tokens_per_block = group.shape_desc.bs * int(
                getattr(group, "compress_ratio", 1)
            )
        if full_sw_kv:
            kv_layer_groups_manager.enable_full_sw_kv()

        # Pre-allocated NPU buffer for block IDs (up to 1M elements).
        # The caller copies block_ids into this buffer before launching the
        # block-level kernel. Single-thread assumption: no lock needed.
        _MAX_BLOCK_IDS = 1 << 20
        block_ids_buffer = torch.empty(
            _MAX_BLOCK_IDS, dtype=torch.long, device=self.device_
        )

        super().__init__(
            kv_caches=unwrapped,
            device=self.device_,
            num_layers=num_layers_val,
            kv_layer_groups_manager=kv_layer_groups_manager,
            block_ids_buffer=block_ids_buffer,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
        )

        # Interleaved [K_0, V_0, K_1, V_1, ...] pointer table per kernel
        # group, device-resident; the context holds the tensors so the
        # pointers outlive the wrappers (MP-mode design doc section 9).
        self.group_kv_pointers_: list[torch.Tensor] = []
        for group in self.kv_layer_groups_manager_.kernel_groups:
            ptrs: list[int] = []
            for layer_idx in group.layer_indices:
                ptrs.extend(_pointers_for_entry(unwrapped[layer_idx], kv_format))
            self.group_kv_pointers_.append(list_to_gpu_tensor(ptrs, self.device_))

        # Temporary NPU buffer for transfers — a single flat uint8 buffer
        self._temp_buffer = _TempNPUBuffer(
            kv_layer_groups_manager=self.kv_layer_groups_manager_,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
            device=self.device_,
            max_batch_size=4,
        )

        # NPU stream (no GDS registration: no GPUDirect Storage on NPU)
        self.npu_stream_ = torch_dev.Stream(device=self.device_)
        self.cupy_stream_: _NPUExternalStream = _NPUExternalStream(self.npu_stream_)

        logger.info(
            "Initialized NPU stream on device %s for NPUCacheContext",
            str(self.device_),
        )

    def close(self) -> None:
        """Release device-specific resources (none: no GDS on NPU)."""
        logger.debug("Closing NPUCacheContext on device %s", str(self.device_))

    @property
    def stream(self) -> Any:
        """
        Returns the NPU stream for KV cache operations
        """
        return self.npu_stream_

    @property
    def cupy_stream(self) -> _NPUExternalStream:
        return self.cupy_stream_

    def get_kernel_group_kv_pointers(self, kernel_group_idx: int) -> torch.Tensor:
        """Returns the pre-computed NPU tensor of KV cache pointers for the
        given kernel group index.
        """
        return self.group_kv_pointers_[kernel_group_idx]

    def get_temp_kernel_group_buffer(
        self, batch_idx: int, kernel_group_idx: int
    ) -> torch.Tensor:
        """Returns the temporary NPU buffer for the given batch index and
        kernel group index, with the correct shape and dtype for the kernel
        group.

        Args:
            batch_idx: Index of the batch (0 <= batch_idx < max_batch_size)
            kernel_group_idx: Index of the kernel group.

        Returns:
            The temp NPU buffer for the given batch index and kernel group
            index.
        """
        return self._temp_buffer.get_temp_kernel_group_buffer(
            batch_idx, kernel_group_idx
        )

    @property
    def max_batch_size(self) -> int:
        """Maximum number of chunks processed concurrently in one batch."""
        return self._temp_buffer.max_batch_size

    def get_temp_object_group_buffer(
        self, batch_idx: int, object_group_idx: int
    ) -> torch.Tensor:
        """Returns the temporary NPU buffer for the given batch index and
        object group index, as a flat uint8 tensor.

        Args:
            batch_idx: Index of the batch (0 <= batch_idx < max_batch_size)
            object_group_idx: Index of the object group.

        Returns:
            The temp NPU buffer for the given batch index and object group
            index.
        """
        return self._temp_buffer.get_temp_object_group_buffer(
            batch_idx, object_group_idx
        )

    def get_kernel_group_shape_dtype(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> tuple[torch.Size, torch.dtype]:
        """Returns the shape and dtype for the given kernel group index and
        number of tokens.
        Will be exported by NPUCacheContext and used to construct the
        MemoryLayoutDesc

        Args:
            num_tokens: Number of tokens. Must be a whole number of lmcache
                chunk size.
            kernel_group_idx: Index of the kernel group.

        Returns:
            The shape and dtype for the given kernel group index and number
            of tokens.
        """
        return self._temp_buffer.get_kernel_group_shape_dtype(
            num_tokens, kernel_group_idx
        )

    def cache_size_per_token(self) -> int:
        """
        Returns the cache size per *logical* token (in bytes), summed
        across all groups.
        """
        return self._temp_buffer.get_cache_size_per_token()


def register_npu_cache_context() -> None:
    """Register :class:`NPUCacheContext` as the ``"npu"`` platform backend.

    Idempotent.  After this call, upstream
    ``LMCacheDrivenTransferModule.register_kv_cache`` →
    ``create_cache_context`` builds an :class:`NPUCacheContext` for any
    kv_caches on ``torch.device("npu")`` — exactly like the CUDA backend —
    so all upstream registration bookkeeping (layout registry, liveness,
    reaping) applies unmodified.
    """
    # First Party
    from lmcache.v1.platform import cache_context as platform_cache_context

    backends = platform_cache_context.snapshot_backends()
    if backends.get("npu") is NPUCacheContext:
        return
    backends["npu"] = NPUCacheContext
    platform_cache_context.restore_backends(backends)
    logger.info("Registered NPUCacheContext as the 'npu' platform backend")


if __name__ == "__main__":
    # First Party
    from lmcache.v1.distributed.config import parse_args_to_config
    from lmcache.v1.mp_observability.config import (
        parse_args_to_observability_config,
    )
    from lmcache.v1.multiprocess.config import parse_args_to_mp_server_config

    args = parse_args()
    register_npu_cache_context()
    run_cache_server(
        mp_config=parse_args_to_mp_server_config(args),
        storage_manager_config=parse_args_to_config(args),
        obs_config=parse_args_to_observability_config(args),
    )
