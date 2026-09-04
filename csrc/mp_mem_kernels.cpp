// SPDX-License-Identifier: Apache-2.0

// C++ host layer for the block-level MP-mode KV transfer (mirrors upstream
// LMCache csrc/mp_mem_kernels.cu).

#include "mp_mem_kernels.h"

#include "kernels/multi_layer/multi_layer_block_mem_kernels.h"
#include "utils.h"

#include <algorithm>

namespace {

// DataCopy granularity on the GM side: segment base addresses and segment
// byte lengths must both be multiples of 32B.
constexpr int64_t kGmAlignBytes = 32;
// UB budget assumed by the device kernel: the depth-2 queue must fit two
// token segments, so a single token's bytes may not exceed half of it.
constexpr int64_t kUbBudgetBytes = 128 * 1024;

// The kernel is a pure byte mover, so fp16 and bf16 share the 2-byte
// instantiation (mirrors the upstream uint16/uint32/uint4 granularity
// dispatch in multi_layer_block_kv_transfer).
kvcache_ops::AscendType ascend_type_from_element_size(int element_size) {
  switch (element_size) {
    case 1:
      return kvcache_ops::AscendType::INT8;
    case 2:
      return kvcache_ops::AscendType::FP16;
    case 4:
      return kvcache_ops::AscendType::FP32;
    default:
      TORCH_CHECK(false, "Unsupported element_size: ", element_size,
                  " (expected 1, 2 or 4)");
  }
  return kvcache_ops::AscendType::FP16;  // unreachable
}

// Shared validation for both entry points. All TORCH_CHECKs fire before any
// stream work is enqueued; returns blocks per object.
int validate_block_transfer(const PageBufferShapeDesc& shape_desc,
                            int64_t total_blocks, int num_objects,
                            int lmcache_chunk_size,
                            EngineKVFormat engine_kv_format,
                            int skip_prefix_n_blocks) {
  TORCH_CHECK(
      engine_kv_format == EngineKVFormat::NL_X_TWO_X_NB_BS_NH_HS,
      "LMCache-Ascend block-level MP transfer currently supports only "
      "EngineKVFormat::NL_X_TWO_X_NB_BS_NH_HS (SEPARATE_KV), got ",
      static_cast<int>(engine_kv_format));
  TORCH_CHECK(shape_desc.kv_size == 2, "SEPARATE_KV requires kv_size == 2, ",
              "got ", shape_desc.kv_size);
  TORCH_CHECK(skip_prefix_n_blocks >= 0, "skip_prefix_n_blocks must be >= 0, ",
              "got ", skip_prefix_n_blocks);

  TORCH_CHECK(num_objects >= 1 && num_objects <= 4,
              "Expected 1-4 LMCache objects, got ", num_objects);
  TORCH_CHECK(total_blocks % num_objects == 0, "block_ids length (",
              total_blocks, ") must be divisible by num_objects (",
              num_objects, ")");
  const int num_blocks_per_object =
      static_cast<int>(total_blocks / num_objects);

  TORCH_CHECK(num_blocks_per_object * shape_desc.bs == lmcache_chunk_size,
              "blocks_per_object * block_size (",
              num_blocks_per_object * shape_desc.bs,
              ") must equal lmcache_chunk_size (", lmcache_chunk_size, ")");
  TORCH_CHECK(skip_prefix_n_blocks <= num_blocks_per_object,
              "skip_prefix_n_blocks (", skip_prefix_n_blocks,
              ") cannot exceed blocks per object (", num_blocks_per_object,
              ")");

  // --- DataCopy alignment boundary (design doc 4.2) ---
  // A 32B-multiple token granularity keeps every segment base address and
  // segment length aligned on both sides; the engine-side block stride must
  // be 32B-aligned as well (block base addresses).
  const int64_t token_bytes = static_cast<int64_t>(shape_desc.nh) *
                              shape_desc.hs * shape_desc.element_size;
  TORCH_CHECK(token_bytes > 0, "nh * hs * element_size must be positive");
  TORCH_CHECK(token_bytes % kGmAlignBytes == 0,
              "scalars_per_token * element_size (", token_bytes,
              " bytes) must be a multiple of ", kGmAlignBytes,
              " for DataCopy alignment");
  const int64_t engine_block_stride =
      shape_desc.block_stride_elems > 0
          ? static_cast<int64_t>(shape_desc.block_stride_elems)
          : static_cast<int64_t>(shape_desc.bs) * shape_desc.nh *
                shape_desc.hs;
  TORCH_CHECK(engine_block_stride * shape_desc.element_size % kGmAlignBytes ==
                  0,
              "engine block stride (",
              engine_block_stride * shape_desc.element_size,
              " bytes) must be a multiple of ", kGmAlignBytes,
              " for DataCopy alignment");

  // --- UB capacity boundary (design doc 4.2) ---
  // The device kernel segments a block by tokens; one token's bytes must fit
  // in half of the UB budget (depth-2 queue).
  TORCH_CHECK(token_bytes <= kUbBudgetBytes / 2, "token bytes (", token_bytes,
              ") exceed the per-segment UB budget (", kUbBudgetBytes / 2,
              "); token-level segmentation cannot fit");

  return num_blocks_per_object;
}

// Phase-1 launch loop: one object + one block_ids slice per kernel launch
// (design doc 4.5). blockDim is clamped to the work-item count so tiny
// transfers do not spin idle cores.
void launch_block_transfer_objects(
    kvcache_ops::AscendType type, uint32_t aiv_num, void* stream,
    uint8_t* paged_buffer_ptrs,
    const std::vector<int64_t>& lmcache_objects_ptrs, int64_t* block_ids_base,
    int64_t total_blocks, int num_blocks_per_object,
    const PageBufferShapeDesc& shape_desc, int lmcache_chunk_size,
    int skip_prefix_n_blocks, bool lmcache_to_engine) {
  const int64_t total_work = static_cast<int64_t>(shape_desc.nl) *
                             shape_desc.kv_size * total_blocks;
  const uint32_t blockDim =
      static_cast<uint32_t>(std::min<int64_t>(aiv_num, total_work));
  for (int i = 0; i < static_cast<int>(lmcache_objects_ptrs.size()); ++i) {
    uint8_t* engine_block_ids = reinterpret_cast<uint8_t*>(
        block_ids_base + static_cast<int64_t>(i) * num_blocks_per_object);
    kvcache_ops::multi_layer_block_transfer_kernel(
        type, blockDim, stream, paged_buffer_ptrs,
        reinterpret_cast<uint8_t*>(lmcache_objects_ptrs[i]),
        engine_block_ids, num_blocks_per_object, skip_prefix_n_blocks,
        shape_desc.nl, shape_desc.bs, shape_desc.nh, shape_desc.hs,
        shape_desc.block_stride_elems, lmcache_chunk_size, lmcache_to_engine);
  }
}

}  // namespace

void multi_layer_block_kv_transfer(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<int64_t> lmcache_objects_ptrs, const torch::Tensor& block_ids,
    const torch::Device& device, TransferDirection direction,
    PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    EngineKVFormat engine_kv_format, int skip_prefix_n_blocks) {
  // --- Validation ---
  const int num_objects = static_cast<int>(lmcache_objects_ptrs.size());
  const int64_t total_blocks = block_ids.size(0);
  const int num_blocks_per_object = validate_block_transfer(
      shape_desc, total_blocks, num_objects, lmcache_chunk_size,
      engine_kv_format, skip_prefix_n_blocks);

  TORCH_CHECK(paged_buffer_ptrs_tensor.scalar_type() == at::kLong,
              "paged_buffer_ptrs_tensor must be int64");
  TORCH_CHECK(paged_buffer_ptrs_tensor.is_privateuseone(),
              "paged_buffer_ptrs_tensor must live on the NPU");
  // data_ptr<int64_t>() itself checks block_ids' dtype.
  TORCH_CHECK(block_ids.is_privateuseone(), "block_ids must live on the NPU");

  const kvcache_ops::AscendType type =
      ascend_type_from_element_size(shape_desc.element_size);
  const bool lmcache_to_engine = (direction == TransferDirection::H2D);

  uint8_t* paged_buffer_ptrs =
      static_cast<uint8_t*>(paged_buffer_ptrs_tensor.data_ptr());
  int64_t* block_ids_base = block_ids.data_ptr<int64_t>();

  const c10::OptionalDeviceGuard device_guard(device);
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  at_npu::native::OpCommand cmd;
  cmd.Name("multi_layer_block_transfer_kernel");
  cmd.SetCustomHandler([type, stream, paged_buffer_ptrs, lmcache_objects_ptrs,
                        block_ids_base, total_blocks, num_blocks_per_object,
                        shape_desc, lmcache_chunk_size, skip_prefix_n_blocks,
                        lmcache_to_engine]() -> int {
    const char* socName = aclrtGetSocName();
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendCManager::GetInstance(socName);
    const uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
    launch_block_transfer_objects(
        type, aiv_num, stream, paged_buffer_ptrs, lmcache_objects_ptrs,
        block_ids_base, total_blocks, num_blocks_per_object, shape_desc,
        lmcache_chunk_size, skip_prefix_n_blocks, lmcache_to_engine);
    return 0;
  });
  cmd.Run();
}

void execute_object_group_transfer(
    TransferDirection direction, const torch::Device& device,
    size_t host_buffer_alignment,
    const std::vector<KernelGroupSpec>& kernel_group_specs,
    const std::vector<BatchStep>& batch_steps) {
  // Set the device guard once for the whole plan so every staging copy and
  // kernel launch below is enqueued on this device's current stream, in
  // order (mirrors upstream execute_object_group_transfer).
  const c10::OptionalDeviceGuard device_guard(device);
  const bool is_h2d = (direction == TransferDirection::H2D);
  TORCH_CHECK(device.is_privateuseone(), "device must be an NPU device");

  // --- Validate the whole plan up front, before any stream work ---
  // Bounds-check every launch's block_ids slice before the kernel
  // dereferences it on device: an out-of-range offset/length would
  // otherwise be a silent out-of-bounds device read, not a clean error.
  for (const auto& step : batch_steps) {
    for (const auto& launch : step.launches) {
      TORCH_CHECK(launch.group_idx >= 0 &&
                      launch.group_idx <
                          static_cast<int>(kernel_group_specs.size()),
                  "LaunchVar.group_idx out of range: ", launch.group_idx);
      const KernelGroupSpec& group = kernel_group_specs[launch.group_idx];
      TORCH_CHECK(launch.num_objects >= 1 &&
                      launch.num_objects <=
                          static_cast<int>(group.lmcache_objects_ptrs.size()),
                  "LaunchVar.num_objects (", launch.num_objects,
                  ") exceeds available temp buffers (",
                  group.lmcache_objects_ptrs.size(), ")");
      TORCH_CHECK(launch.block_ids_offset >= 0,
                  "LaunchVar.block_ids_offset must be non-negative, got ",
                  launch.block_ids_offset);
      TORCH_CHECK(launch.total_blocks >= 0,
                  "LaunchVar.total_blocks must be non-negative, got ",
                  launch.total_blocks);
      TORCH_CHECK(launch.block_ids_offset + launch.total_blocks <=
                      group.block_ids_capacity,
                  "LaunchVar block_ids slice [", launch.block_ids_offset, ", ",
                  launch.block_ids_offset + launch.total_blocks,
                  ") exceeds block_ids capacity ", group.block_ids_capacity);
      // Full per-launch validation (format, shape, alignment) through the
      // shared checker so the plan path rejects bad launches exactly like
      // the direct entry point would.
      validate_block_transfer(group.shape_desc, launch.total_blocks,
                              launch.num_objects, group.lmcache_chunk_size,
                              group.engine_kv_format,
                              launch.skip_prefix_n_blocks);
    }
  }

  at_npu::native::OpCommand cmd;
  cmd.Name("multi_layer_block_transfer_kernel");
  cmd.SetCustomHandler([&]() -> int {
    const char* socName = aclrtGetSocName();
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendCManager::GetInstance(socName);
    const uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
    const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    const auto do_staging = [&](const std::vector<StagingCopy>& staging) {
      for (const auto& copy : staging) {
        lmcache_memcpy_async(copy.dest, copy.src, copy.nbytes, direction,
                             copy.host_offset, host_buffer_alignment);
      }
    };

    for (const auto& step : batch_steps) {
      // H2D stages CPU->NPU temp buffers before the kernel reads them; D2H
      // stages NPU->CPU after the kernel writes them. The per-step ordering
      // must be preserved because temp buffers are reused across steps.
      if (is_h2d) {
        do_staging(step.staging);
      }
      for (const auto& launch : step.launches) {
        const KernelGroupSpec& group = kernel_group_specs[launch.group_idx];
        std::vector<int64_t> lmcache_objects_ptrs(
            group.lmcache_objects_ptrs.begin(),
            group.lmcache_objects_ptrs.begin() + launch.num_objects);
        int64_t* block_ids_base = reinterpret_cast<int64_t*>(
            group.block_ids_base +
            static_cast<uintptr_t>(launch.block_ids_offset) * sizeof(int64_t));
        launch_block_transfer_objects(
            ascend_type_from_element_size(group.shape_desc.element_size),
            aiv_num, stream,
            reinterpret_cast<uint8_t*>(group.paged_buffer_ptrs),
            lmcache_objects_ptrs, block_ids_base, launch.total_blocks,
            static_cast<int>(launch.total_blocks / launch.num_objects),
            group.shape_desc, group.lmcache_chunk_size,
            launch.skip_prefix_n_blocks, is_h2d);
      }
      if (!is_h2d) {
        do_staging(step.staging);
      }
    }
    return 0;
  });
  cmd.Run();
}

void lmcache_memcpy_async(uintptr_t dest, uintptr_t src, size_t nbytes,
                          TransferDirection direction,
                          size_t host_buffer_offset,
                          size_t host_buffer_alignments) {
  // Check that host_buffer_alignments is power of two
  TORCH_CHECK((host_buffer_alignments & (host_buffer_alignments - 1)) == 0,
              "host_buffer_alignments must be power of two");

  size_t offset = 0;
  const size_t mask = host_buffer_alignments - 1;
  const aclrtMemcpyKind kind = (direction == TransferDirection::H2D)
                                    ? ACL_MEMCPY_HOST_TO_DEVICE
                                    : ACL_MEMCPY_DEVICE_TO_HOST;
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  // Split the copy at the host buffer's alignment boundaries: each chunk
  // stays inside one aligned region (port of upstream lmcache_memcpy_async).
  while (offset < nbytes) {
    const size_t aligned_area_end =
        ((offset + host_buffer_offset) & ~mask) + host_buffer_alignments;
    const size_t real_end =
        std::min<size_t>(host_buffer_offset + nbytes, aligned_area_end);
    const size_t max_nbytes = real_end - offset - host_buffer_offset;

    const aclError ret = aclrtMemcpyAsync(
        reinterpret_cast<void*>(dest + offset),
        reinterpret_cast<const void*>(src + offset), max_nbytes, kind,
        stream);
    TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtMemcpyAsync failed: ret=", ret);

    offset += max_nbytes;
  }
}
