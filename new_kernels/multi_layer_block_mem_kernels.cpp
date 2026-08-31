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

// Block-level multi-layer KV transfer between the engine's paged buffers and
// LMCache contiguous memory objects, for the MP (multiprocess) mode.
//
// Mirrors the upstream CUDA kernel multi_layer_block_transfer_kernel
// (LMCache csrc/mp_mem_kernels.cu): one work item moves one full block of one
// (layer, k_or_v) plane as a contiguous burst, instead of expanding block ids
// to a token-level slot mapping. The CUDA grid (kv_size, NB, NL) maps onto a
// grid-stride loop over the AIV cores.
//
// Phase 1 (see ascend_mp_mode_implementation_plan.md 4.5): one LMCache object
// per launch — the C++ boundary loops over the object batch and slices
// block_ids by i * num_blocks_per_object. Phase 2 restores the fused
// 4-object launch (obj_idx = flat_block_idx / num_blocks_per_object).
//
// Layouts (SEPARATE_KV == upstream EngineKVFormat::NL_X_TWO_X_NB_BS_NH_HS):
//   engine  : per-layer (K, V) paged tensors [NB, BS, NH, HS]; the plane is
//             selected through the interleaved [K_0, V_0, ...] pointer table.
//   lmcache : 2LTD [K/V, NL, chunk, NH * HS] contiguous object.

#include "multi_layer_block_mem_kernels.h"
#include "multi_layer_mem_kernels.h"
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Device kernel
// ---------------------------------------------------------------------------

template <typename scalar_t, typename slot_t>
class MultiLayerBlockTransfer {
    using local_scalar_t = AscendC::LocalTensor<scalar_t>;

public:
    __aicore__ inline MultiLayerBlockTransfer() {}

    __aicore__ inline void init(
        __gm__ uint8_t *paged_buffer_ptrs, GM_ADDR lmcache_obj,
        __gm__ uint8_t *engine_block_ids, const int32_t num_blocks_per_object,
        const int32_t skip_prefix_n_blocks, const int32_t nl, const int32_t bs,
        const int32_t nh, const int32_t hs, const int32_t block_stride_elems,
        const int32_t lmcache_chunk_size, const bool lmcache_to_engine,
        AscendC::TPipe *pipe)
    {
        paged_buffer_ptrs_ = paged_buffer_ptrs;
        lmcache_obj_ = lmcache_obj;
        engine_block_ids_ = reinterpret_cast<__gm__ slot_t *>(engine_block_ids);
        num_blocks_per_object_ = num_blocks_per_object;
        skip_prefix_n_blocks_ = skip_prefix_n_blocks;
        nl_ = nl;
        bs_ = bs;
        lmcache_chunk_size_ = lmcache_chunk_size;
        lmcache_to_engine_ = lmcache_to_engine;

        scalars_per_token_ = static_cast<int64_t>(nh) * hs;

        // Engine-side per-block dim-0 stride in scalar_t elements: the padded
        // block_stride_elems when set, the tight bs * nh * hs otherwise
        // (upstream PageBufferShapeDesc::scalars_per_block() semantics).
        engine_block_stride_ = block_stride_elems > 0
            ? static_cast<int64_t>(block_stride_elems)
            : static_cast<int64_t>(bs) * scalars_per_token_;

        // Contiguous data extent per block (trailing padding is never moved).
        data_elems_per_block_ = static_cast<int64_t>(bs) * scalars_per_token_;

        // --- UB segmentation (design doc 4.2) ---
        // Bound one segment to half of a conservative UB budget so the
        // depth-2 double-buffered queue fits; the same bound covers the
        // single-DataCopy burst length. Host-side checks guarantee
        // token_bytes <= budget / 2, so tokens_per_seg_ >= 1 here.
        const int64_t ubBudgetBytes = 128 * 1024;
        const int64_t tokenBytes =
            scalars_per_token_ * static_cast<int64_t>(sizeof(scalar_t));
        int64_t fit = ubBudgetBytes / (2 * tokenBytes);
        if (fit < 1) {
            fit = 1;
        }
        if (fit > bs) {
            fit = bs;
        }
        tokens_per_seg_ = static_cast<int32_t>(fit);

        pipe_ = pipe;
        pipe_->InitBuffer(block_que_, 2, tokens_per_seg_ * tokenBytes);
    }

    __aicore__ inline void process()
    {
        // Phase 1: single object per launch, so total_blocks is this object's
        // block count (design doc 4.1); kv_size is fixed to 2 (SEPARATE_KV).
        const int32_t total_blocks = num_blocks_per_object_;
        const int32_t total_work = nl_ * 2 * total_blocks;
        const int32_t core_num = AscendC::GetBlockNum();

        for (int32_t w = AscendC::GetBlockIdx(); w < total_work; w += core_num) {
            // Dimension decomposition mirrors the CUDA grid
            // (z, y, x) = (layer, k_or_v, flat_block), upstream L233-236.
            const int32_t layer_idx = w / (2 * total_blocks);
            const int32_t remainder = w % (2 * total_blocks);
            const int32_t k_or_v = remainder / total_blocks;
            const int32_t flat_block_idx = remainder % total_blocks;

            // Prefix blocks idle-return without reading the ids array
            // (upstream L239: the array is indexed from 0 including prefix).
            if (flat_block_idx < skip_prefix_n_blocks_) {
                continue;
            }

            const int64_t engine_block_idx =
                static_cast<int64_t>(engine_block_ids_[flat_block_idx]);
            process_single_block(layer_idx, k_or_v, flat_block_idx,
                                 engine_block_idx);
        }
    }

private:
    __aicore__ inline void process_single_block(
        const int32_t layer_idx, const int32_t k_or_v,
        const int32_t block_idx_in_object, const int64_t engine_block_idx)
    {
        // === Engine side: per-layer (K, V) paged tensors, interleaved ptr
        // table (upstream calculate_engine_global_offset for format 11) ===
        __gm__ uint8_t *paged_layer_base =
            kvcache_ops::GetLayerBasePtr<kvcache_ops::KVCacheFormat::SEPARATE_KV>(
                paged_buffer_ptrs_, layer_idx, k_or_v);
        __gm__ scalar_t *engine_ptr =
            reinterpret_cast<__gm__ scalar_t *>(paged_layer_base) +
            engine_block_idx * engine_block_stride_;

        // === LMCache side: 2LTD (upstream calculate_lmcache_global_offset) ===
        __gm__ scalar_t *lmcache_ptr =
            reinterpret_cast<__gm__ scalar_t *>(lmcache_obj_) +
            static_cast<int64_t>(k_or_v) * nl_ * lmcache_chunk_size_ *
                scalars_per_token_ +
            static_cast<int64_t>(layer_idx) * lmcache_chunk_size_ *
                scalars_per_token_ +
            static_cast<int64_t>(block_idx_in_object) * bs_ * scalars_per_token_;

        AscendC::GlobalTensor<scalar_t> engine_global;
        AscendC::GlobalTensor<scalar_t> lmcache_global;
        engine_global.SetGlobalBuffer(engine_ptr, data_elems_per_block_);
        lmcache_global.SetGlobalBuffer(lmcache_ptr, data_elems_per_block_);

        // Token-segmented, double-buffered contiguous move: both sides are
        // fully contiguous within the block, so each segment is one GM->UB
        // burst followed by one UB->GM burst (design doc 4.2).
        const int32_t num_segs = (bs_ + tokens_per_seg_ - 1) / tokens_per_seg_;
        for (int32_t seg = 0; seg < num_segs; ++seg) {
            const int64_t seg_start_token = seg * tokens_per_seg_;
            int64_t seg_tokens = static_cast<int64_t>(bs_) - seg_start_token;
            if (seg_tokens > tokens_per_seg_) {
                seg_tokens = tokens_per_seg_;
            }
            const int64_t seg_elems = seg_tokens * scalars_per_token_;
            const int64_t seg_elem_off = seg_start_token * scalars_per_token_;

            local_scalar_t buf = block_que_.template AllocTensor<scalar_t>();
            if (lmcache_to_engine_) {
                AscendC::DataCopy(buf, lmcache_global[seg_elem_off], seg_elems);
                block_que_.EnQue(buf);
                buf = block_que_.template DeQue<scalar_t>();
                AscendC::DataCopy(engine_global[seg_elem_off], buf, seg_elems);
            } else {
                AscendC::DataCopy(buf, engine_global[seg_elem_off], seg_elems);
                block_que_.EnQue(buf);
                buf = block_que_.template DeQue<scalar_t>();
                AscendC::DataCopy(lmcache_global[seg_elem_off], buf, seg_elems);
            }
            block_que_.FreeTensor(buf);
        }
    }

    AscendC::TPipe *pipe_;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 2>
        block_que_;

    __gm__ uint8_t *paged_buffer_ptrs_;
    GM_ADDR lmcache_obj_;
    __gm__ slot_t *engine_block_ids_;

    int32_t num_blocks_per_object_;
    int32_t skip_prefix_n_blocks_;
    int32_t nl_;
    int32_t bs_;
    int32_t lmcache_chunk_size_;
    bool lmcache_to_engine_;

    int64_t scalars_per_token_;
    int64_t engine_block_stride_;
    int64_t data_elems_per_block_;
    int32_t tokens_per_seg_;
};

#define MULTI_LAYER_BLOCK_TRANSFER_KERNEL_NAME(TYPE, SLOTTYPE) \
    multi_layer_block_transfer_kernel_##TYPE##_##SLOTTYPE

#define MULTI_LAYER_BLOCK_TRANSFER_KERNEL_DECLARE(TYPE, SLOTTYPE)              \
    extern "C" __global__ __aicore__ void                                      \
    MULTI_LAYER_BLOCK_TRANSFER_KERNEL_NAME(TYPE, SLOTTYPE)(                    \
        __gm__ uint8_t *paged_buffer_ptrs, GM_ADDR lmcache_obj,                \
        __gm__ uint8_t *engine_block_ids, const int32_t num_blocks_per_object, \
        const int32_t skip_prefix_n_blocks, const int32_t nl,                  \
        const int32_t bs, const int32_t nh, const int32_t hs,                  \
        const int32_t block_stride_elems, const int32_t lmcache_chunk_size,    \
        const bool lmcache_to_engine)                                          \
    {                                                                          \
        AscendC::TPipe pipe;                                                   \
        MultiLayerBlockTransfer<TYPE, SLOTTYPE> op{};                          \
        op.init(paged_buffer_ptrs, lmcache_obj, engine_block_ids,              \
                num_blocks_per_object, skip_prefix_n_blocks, nl, bs, nh, hs,   \
                block_stride_elems, lmcache_chunk_size, lmcache_to_engine,     \
                &pipe);                                                        \
        op.process();                                                          \
    }

// Declare supported kernel entries on the device side (slot_t is int64,
// mirroring the upstream block_ids dtype).
MULTI_LAYER_BLOCK_TRANSFER_KERNEL_DECLARE(half, int64_t)
MULTI_LAYER_BLOCK_TRANSFER_KERNEL_DECLARE(int8_t, int64_t)
MULTI_LAYER_BLOCK_TRANSFER_KERNEL_DECLARE(float, int64_t)
#if (__CCE_AICORE__ >= 220)
MULTI_LAYER_BLOCK_TRANSFER_KERNEL_DECLARE(bfloat16_t, int64_t)
#endif

namespace kvcache_ops {

void multi_layer_block_transfer_kernel(
    AscendType type, uint32_t blockDim, void *stream, uint8_t *paged_buffer_ptrs,
    uint8_t *lmcache_obj, uint8_t *engine_block_ids,
    int32_t num_blocks_per_object, int32_t skip_prefix_n_blocks,
    int32_t nl, int32_t bs, int32_t nh, int32_t hs,
    int32_t block_stride_elems, int32_t lmcache_chunk_size,
    bool lmcache_to_engine)
{
    switch (type) {
        case AscendType::FP16:
            MULTI_LAYER_BLOCK_TRANSFER_KERNEL_NAME(half, int64_t)
                <<<blockDim, nullptr, stream>>>(
                    paged_buffer_ptrs, lmcache_obj, engine_block_ids,
                    num_blocks_per_object, skip_prefix_n_blocks, nl, bs, nh, hs,
                    block_stride_elems, lmcache_chunk_size, lmcache_to_engine);
            break;
        case AscendType::FP32:
            MULTI_LAYER_BLOCK_TRANSFER_KERNEL_NAME(float, int64_t)
                <<<blockDim, nullptr, stream>>>(
                    paged_buffer_ptrs, lmcache_obj, engine_block_ids,
                    num_blocks_per_object, skip_prefix_n_blocks, nl, bs, nh, hs,
                    block_stride_elems, lmcache_chunk_size, lmcache_to_engine);
            break;
        case AscendType::INT8:
            MULTI_LAYER_BLOCK_TRANSFER_KERNEL_NAME(int8_t, int64_t)
                <<<blockDim, nullptr, stream>>>(
                    paged_buffer_ptrs, lmcache_obj, engine_block_ids,
                    num_blocks_per_object, skip_prefix_n_blocks, nl, bs, nh, hs,
                    block_stride_elems, lmcache_chunk_size, lmcache_to_engine);
            break;
#if (ASCEND_AICORE_ARCH >= 220)
        case AscendType::BF16:
            MULTI_LAYER_BLOCK_TRANSFER_KERNEL_NAME(bfloat16_t, int64_t)
                <<<blockDim, nullptr, stream>>>(
                    paged_buffer_ptrs, lmcache_obj, engine_block_ids,
                    num_blocks_per_object, skip_prefix_n_blocks, nl, bs, nh, hs,
                    block_stride_elems, lmcache_chunk_size, lmcache_to_engine);
            break;
#else
        case AscendType::BF16:
            // Pure byte move: fall back to the 2-byte half kernel on SoCs
            // without the bf16 expansion.
            MULTI_LAYER_BLOCK_TRANSFER_KERNEL_NAME(half, int64_t)
                <<<blockDim, nullptr, stream>>>(
                    paged_buffer_ptrs, lmcache_obj, engine_block_ids,
                    num_blocks_per_object, skip_prefix_n_blocks, nl, bs, nh, hs,
                    block_stride_elems, lmcache_chunk_size, lmcache_to_engine);
            break;
#endif
        default:
            ASCENDC_REPORT_NOT_SUPPORT(
                false, std::to_string(static_cast<int>(type)) + " is not supported.")
            throw std::runtime_error(
                "Scalar type: " + std::to_string(static_cast<int>(type)) +
                " not supported. This should not have happened.");
    }
}

} // namespace kvcache_ops
