/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MULTI_LAYER_BLOCK_MEM_KERNELS_H
#define MULTI_LAYER_BLOCK_MEM_KERNELS_H

#include "../types.h"

namespace kvcache_ops {

// Host-side launcher for the block-level multi-layer KV transfer kernel
// (device entries multi_layer_block_transfer_kernel_<TYPE>_<SLOTTYPE> below).
//
// Phase 1: one LMCache memory object per launch (the C++ boundary in
// csrc/mp_mem_kernels.cpp loops over the object batch), engine side is
// SEPARATE_KV — per-layer (K, V) paged tensors selected through the
// interleaved [K_0, V_0, K_1, V_1, ...] device pointer table.
//
// Params:
//   paged_buffer_ptrs      device base of the interleaved int64 pointer table
//   lmcache_obj            device address of the single LMCache object (2LTD)
//   engine_block_ids       device int64 array, [num_blocks_per_object] entries
//   num_blocks_per_object  blocks in this object (== lmcache_chunk_size / bs)
//   skip_prefix_n_blocks   leading blocks to skip (upstream same-name param)
//   nl / bs / nh / hs      PageBufferShapeDesc fields (kv_size fixed to 2)
//   block_stride_elems     engine per-block dim-0 stride, 0 = tight
//   lmcache_chunk_size     tokens per LMCache object (upstream same-name param)
//   lmcache_to_engine      true: H2D (LMCache -> engine), false: D2H
void multi_layer_block_transfer_kernel(
    AscendType type, uint32_t blockDim, void *stream,
    uint8_t *paged_buffer_ptrs, uint8_t *lmcache_obj, uint8_t *engine_block_ids,
    int32_t num_blocks_per_object, int32_t skip_prefix_n_blocks,
    int32_t nl, int32_t bs, int32_t nh, int32_t hs,
    int32_t block_stride_elems, int32_t lmcache_chunk_size,
    bool lmcache_to_engine);

} // namespace kvcache_ops

#endif // MULTI_LAYER_BLOCK_MEM_KERNELS_H
