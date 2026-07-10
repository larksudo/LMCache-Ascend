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

#ifndef SINGLE_LAYER_MEM_KERNELS_V2_H
#define SINGLE_LAYER_MEM_KERNELS_V2_H

#include "kernel_operator.h"

constexpr int32_t ASCEND_BLOCK_LEN = 32;

template <typename scalar_t>
struct MergedPolicy {
    AscendC::GlobalTensor<scalar_t> vllmKVGlobal;
    int64_t blockStride;
    int64_t valueOffset;
    int32_t headDims;
    int32_t numHeads;
    int32_t blockSize;

    __aicore__ inline void Init(GM_ADDR kvPtr, int64_t stride, int64_t vOffset, int64_t bufSize,
                                int32_t nHeads, int32_t hDims, int32_t bSize) {
        vllmKVGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(kvPtr), bufSize);
        blockStride = stride;
        valueOffset = vOffset;
        numHeads = nHeads;
        headDims = hDims;
        blockSize = bSize;
    }

    __aicore__ inline void Copy2Local(const AscendC::LocalTensor<scalar_t>& localTensor, 
                                      int64_t blockIdx, int64_t blockOffset, 
                                      int32_t localKIdx, int32_t localVIdx,
                                      int32_t /*localDsaIdx*/) {
        int64_t kIdx = blockIdx * blockStride + blockOffset * numHeads * headDims;
        int64_t vIdx = kIdx + valueOffset;
        int32_t len = numHeads * headDims;

        AscendC::DataCopy(localTensor[localKIdx], vllmKVGlobal[kIdx], len);
        AscendC::DataCopy(localTensor[localVIdx], vllmKVGlobal[vIdx], len);
    }

    __aicore__ inline void Copy2Global(const AscendC::LocalTensor<scalar_t>& localTensor, 
                                       int64_t blockIdx, int64_t blockOffset, 
                                       int32_t localKIdx, int32_t localVIdx,
                                       int32_t /*localDsaIdx*/) {
        int64_t kIdx = blockIdx * blockStride + blockOffset * numHeads * headDims;
        int64_t vIdx = kIdx + valueOffset;
        int32_t len = numHeads * headDims;

        AscendC::DataCopy(vllmKVGlobal[kIdx], localTensor[localKIdx], len);
        AscendC::DataCopy(vllmKVGlobal[vIdx], localTensor[localVIdx], len);
    }
};

template <typename scalar_t>
struct SeparatePolicy {
    AscendC::GlobalTensor<scalar_t> vllmKeyGlobal;
    AscendC::GlobalTensor<scalar_t> vllmValueGlobal;
    AscendC::GlobalTensor<scalar_t> vllmDsaKeyGlobal;  // DSA 3rd tensor (nullptr if not DSA)
    int64_t keyBlockStride;
    int64_t valueBlockStride;
    int64_t dsaKeyBlockStride;
    int32_t headDims;   // K head dims (for MLA: kv_lora_rank)
    int32_t vHeadDims;  // V head dims (for MLA: qk_rope_head_dim; same as headDims for non-MLA)
    int32_t dsaHeadDims; // DSA extra head dims (0 if not DSA)
    int32_t numHeads;
    int32_t blockSize;

    __aicore__ inline void Init(GM_ADDR kPtr, GM_ADDR vPtr, GM_ADDR dsaPtr,
                                int64_t kStride, int64_t vStride, int64_t dsaStride,
                                int64_t kSize, int64_t vSize, int64_t dsaSize,
                                int32_t nHeads, int32_t hDims, int32_t vHDims, int32_t dsaHDims,
                                int32_t bSize) {
        vllmKeyGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(kPtr), kSize);
        vllmValueGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(vPtr), vSize);
        if (dsaPtr != nullptr) {
            vllmDsaKeyGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(dsaPtr), dsaSize);
        }
        keyBlockStride = kStride;
        valueBlockStride = vStride;
        dsaKeyBlockStride = dsaStride;
        numHeads = nHeads;
        headDims = hDims;
        vHeadDims = vHDims;
        dsaHeadDims = dsaHDims;
        blockSize = bSize;
    }

    __aicore__ inline void Copy2Local(const AscendC::LocalTensor<scalar_t>& localTensor, 
                                      int64_t blockIdx, int64_t blockOffset, 
                                      int32_t localKIdx, int32_t localVIdx, int32_t localDsaIdx) {
        int32_t kLen = numHeads * headDims;
        int32_t vLen = numHeads * vHeadDims;
        int64_t kIdx = blockIdx * keyBlockStride + blockOffset * kLen;
        int64_t vIdx = blockIdx * valueBlockStride + blockOffset * vLen;

        AscendC::DataCopy(localTensor[localKIdx], vllmKeyGlobal[kIdx], kLen);
        AscendC::DataCopy(localTensor[localVIdx], vllmValueGlobal[vIdx], vLen);
        if (dsaHeadDims > 0) {
            int32_t dsaLen = numHeads * dsaHeadDims;
            int64_t dsaIdx = blockIdx * dsaKeyBlockStride + blockOffset * dsaLen;
            AscendC::DataCopy(localTensor[localDsaIdx], vllmDsaKeyGlobal[dsaIdx], dsaLen);
        }
    }

    __aicore__ inline void Copy2Global(const AscendC::LocalTensor<scalar_t>& localTensor, 
                                       int64_t blockIdx, int64_t blockOffset, 
                                       int32_t localKIdx, int32_t localVIdx, int32_t localDsaIdx) {
        int32_t kLen = numHeads * headDims;
        int32_t vLen = numHeads * vHeadDims;
        int64_t kIdx = blockIdx * keyBlockStride + blockOffset * kLen;
        int64_t vIdx = blockIdx * valueBlockStride + blockOffset * vLen;

        AscendC::DataCopy(vllmKeyGlobal[kIdx], localTensor[localKIdx], kLen);
        AscendC::DataCopy(vllmValueGlobal[vIdx], localTensor[localVIdx], vLen);
        if (dsaHeadDims > 0) {
            int32_t dsaLen = numHeads * dsaHeadDims;
            int64_t dsaIdx = blockIdx * dsaKeyBlockStride + blockOffset * dsaLen;
            AscendC::DataCopy(vllmDsaKeyGlobal[dsaIdx], localTensor[localDsaIdx], dsaLen);
        }
    }
};

template <typename scalar_t, typename slot_t, typename PolicyT> 
class SingleLayerPagedKVCopyProcessor {
    using local_scalar_t = AscendC::LocalTensor<scalar_t>;

public:
    __aicore__ inline SingleLayerPagedKVCopyProcessor() {}

    // Accessor to initialize the specific policy
    __aicore__ inline PolicyT& GetPolicy() {
        return policy_;
    }

    // Common Initialization for shared resources (LMC, Pipe, Queue)
    __aicore__ inline void InitCommon(GM_ADDR lmcKeyValueCachePtr, 
                                      GM_ADDR slotMappingPtr, 
                                      const int64_t lmcTokenStride, const int64_t lmcValueOffset, 
                                      const int64_t lmcDsaOffset, const int64_t lmcBufferSize,
                                      const int32_t maxTokensPerLoop, const int32_t numHeads, const int32_t headDims, 
                                      const int32_t vHeadDims, const int32_t dsaHeadDims,
                                      const int32_t numTokens, const int32_t blockSize, const bool page2L, const bool lmcTokensMajor, 
                                      AscendC::TPipe *pipe)
    {
        this->pipe_ = pipe;
        this->numHeads_ = numHeads;
        this->numTokens_ = numTokens;
        this->blockSize_ = blockSize;
        this->page2L_ = page2L;
        this->headDims_ = headDims;
        this->vHeadDims_ = vHeadDims;
        this->dsaHeadDims_ = dsaHeadDims;
        this->lmcTokenStride_ = lmcTokenStride;
        this->lmcValueOffset_ = lmcValueOffset;
        this->lmcDsaOffset_ = lmcDsaOffset;
        this->lmcTokensMajor_ = lmcTokensMajor;
        
        // Fixed constant as per original implementation
        this->numKvs_ = 2; 

        this->lmcBufferGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(lmcKeyValueCachePtr), lmcBufferSize);

        // local buffer: per token = K(headDims) + V(vHeadDims) + DSA(dsaHeadDims)
        uint64_t localTokenBufferSize = maxTokensPerLoop * 
            (static_cast<uint64_t>(this->numHeads_) * (this->headDims_ + this->vHeadDims_ + this->dsaHeadDims_)) * sizeof(scalar_t);
        this->pipe_->InitBuffer(this->tokenQue_, 2, localTokenBufferSize);
    }

    __aicore__ inline void process(__gm__ uint8_t *slotmappings, int32_t tokenIdx, int32_t actualTokensPerLoop) {
        if (this->page2L_) {
            this->runCopyPage2L(slotmappings, tokenIdx, actualTokensPerLoop);
        } else {
            this->runCopyL2Page(slotmappings, tokenIdx, actualTokensPerLoop);
        }
    }

private:
    // VLLM (Global) -> LMC (Global)
    __aicore__ inline void runCopyPage2L(__gm__ uint8_t *slotmappings, int32_t tokenIdx, int32_t actualTokensPerLoop) {
        // alloc local buffer per tokens
        local_scalar_t tokensBufferTensor = this->tokenQue_.template AllocTensor<scalar_t>();
        __gm__ slot_t *slotmappingPtr = reinterpret_cast<__gm__ slot_t*>(slotmappings);

        // [Produce]: Read from VLLM to Local UB
        int64_t slot, blockIdx, blockOffset;
        int64_t localTokenBuffKIdx, localTokenBuffVIdx, localTokenBuffDsaIdx;
        int64_t realTokenIdx;

        int32_t kLen = this->numHeads_ * this->headDims_;
        int32_t vLen = this->numHeads_ * this->vHeadDims_;
        int32_t dsaLen = this->numHeads_ * this->dsaHeadDims_;
        int32_t tokenLen = kLen + vLen + dsaLen;

        for (int32_t innerTokenIdx = 0; innerTokenIdx < actualTokensPerLoop; innerTokenIdx++) {
            realTokenIdx = tokenIdx + innerTokenIdx;
            slot = static_cast<int64_t>(slotmappingPtr[realTokenIdx]);
            
            if (slot == -1) {
                continue;
            }

            blockIdx = slot / this->blockSize_;
            blockOffset = slot % this->blockSize_;

            // local buffer layout per token: [K(headDims) | V(vHeadDims) | DSA(dsaHeadDims)]
            localTokenBuffKIdx = innerTokenIdx * tokenLen;
            localTokenBuffVIdx = localTokenBuffKIdx + kLen;
            localTokenBuffDsaIdx = localTokenBuffVIdx + vLen;
            
            policy_.Copy2Local(tokensBufferTensor, blockIdx, blockOffset,
                               localTokenBuffKIdx, localTokenBuffVIdx, localTokenBuffDsaIdx);
        }

        // [Sync]: Wait for Data to arrive in UB
        this->tokenQue_.EnQue(tokensBufferTensor);
        tokensBufferTensor = this->tokenQue_.template DeQue<scalar_t>();

        // [Consume]: Write from Local UB to LMC
        CopyLocalToLmc(tokensBufferTensor, tokenIdx, actualTokensPerLoop);
        
        this->tokenQue_.FreeTensor(tokensBufferTensor);
    }

    // LMC (Global) -> VLLM (Global)
    __aicore__ inline void runCopyL2Page(__gm__ uint8_t *slotmappings, int32_t tokenIdx, int32_t actualTokensPerLoop) {
        // alloc local buffer per tokens
        local_scalar_t tokensBufferTensor = this->tokenQue_.template AllocTensor<scalar_t>();
        __gm__ slot_t *slotmappingPtr = reinterpret_cast<__gm__ slot_t*>(slotmappings);

        // [Produce]: Read from LMC to Local UB
        CopyLmcToLocal(tokensBufferTensor, tokenIdx, actualTokensPerLoop);

        // [Sync]: Wait for Data to arrive in UB (CRITICAL FIX)
        this->tokenQue_.EnQue(tokensBufferTensor);
        tokensBufferTensor = this->tokenQue_.template DeQue<scalar_t>();

        // [Consume]: Write from Local UB to VLLM
        int64_t slot, blockIdx, blockOffset;
        int64_t localTokenBuffKIdx, localTokenBuffVIdx, localTokenBuffDsaIdx;
        int64_t realTokenIdx;

        int32_t kLen = this->numHeads_ * this->headDims_;
        int32_t vLen = this->numHeads_ * this->vHeadDims_;
        int32_t dsaLen = this->numHeads_ * this->dsaHeadDims_;
        int32_t tokenLen = kLen + vLen + dsaLen;

        for (int32_t innerTokenIdx = 0; innerTokenIdx < actualTokensPerLoop; innerTokenIdx++) {
            realTokenIdx = tokenIdx + innerTokenIdx;
            slot = static_cast<int64_t>(slotmappingPtr[realTokenIdx]);
            
            if (slot == -1) {
                continue;
            }

            blockIdx = slot / this->blockSize_;
            blockOffset = slot % this->blockSize_;

            localTokenBuffKIdx = static_cast<int64_t>(innerTokenIdx) * tokenLen;
            localTokenBuffVIdx = localTokenBuffKIdx + kLen;
            localTokenBuffDsaIdx = localTokenBuffVIdx + vLen;

            policy_.Copy2Global(tokensBufferTensor, blockIdx, blockOffset,
                                localTokenBuffKIdx, localTokenBuffVIdx, localTokenBuffDsaIdx);
        }

        this->tokenQue_.FreeTensor(tokensBufferTensor);
    }

    __aicore__ inline void CopyLocalToLmc(local_scalar_t& tokensBufferTensor, int32_t tokenIdx, int32_t actualTokensPerLoop) {
        int32_t kLen = this->numHeads_ * this->headDims_;
        int32_t vLen = this->numHeads_ * this->vHeadDims_;
        int32_t dsaLen = this->numHeads_ * this->dsaHeadDims_;
        int64_t perCacheKBlockLen = (kLen * sizeof(scalar_t)) / ASCEND_BLOCK_LEN;
        int64_t perCacheVBlockLen = (vLen * sizeof(scalar_t)) / ASCEND_BLOCK_LEN;
        int64_t perCacheDsaBlockLen = (dsaLen * sizeof(scalar_t)) / ASCEND_BLOCK_LEN;
        int64_t lmcTokenKOffset = tokenIdx * this->lmcTokenStride_;
        
        AscendC::DataCopyParams tokenCopyParams;
        tokenCopyParams.blockCount = actualTokensPerLoop;

        if (this->lmcTokensMajor_ || this->numKvs_ == 1) {
            // MLA/DSA/token-major: local [K|V|DSA] per token, lmc [K|V|DSA] per token
            // K and V (and DSA) have different lengths, copy separately
            // K
            tokenCopyParams.blockLen = perCacheKBlockLen;
            tokenCopyParams.srcStride = (vLen + dsaLen) * sizeof(scalar_t) / ASCEND_BLOCK_LEN;
            tokenCopyParams.dstStride = (this->lmcTokenStride_ - kLen) * sizeof(scalar_t) / ASCEND_BLOCK_LEN;
            AscendC::DataCopy(this->lmcBufferGlobal_[lmcTokenKOffset], tokensBufferTensor, tokenCopyParams);
            
            // V
            int64_t lmcTokenVOffset = lmcTokenKOffset + this->lmcValueOffset_;
            tokenCopyParams.blockLen = perCacheVBlockLen;
            tokenCopyParams.srcStride = (kLen + dsaLen) * sizeof(scalar_t) / ASCEND_BLOCK_LEN;
            tokenCopyParams.dstStride = (this->lmcTokenStride_ - vLen) * sizeof(scalar_t) / ASCEND_BLOCK_LEN;
            AscendC::DataCopy(this->lmcBufferGlobal_[lmcTokenVOffset], tokensBufferTensor[kLen], tokenCopyParams);
            
            // DSA (only if dsaHeadDims > 0)
            if (dsaLen > 0) {
                int64_t lmcTokenDsaOffset = lmcTokenKOffset + this->lmcDsaOffset_;
                tokenCopyParams.blockLen = perCacheDsaBlockLen;
                tokenCopyParams.srcStride = (kLen + vLen) * sizeof(scalar_t) / ASCEND_BLOCK_LEN;
                tokenCopyParams.dstStride = (this->lmcTokenStride_ - dsaLen) * sizeof(scalar_t) / ASCEND_BLOCK_LEN;
                AscendC::DataCopy(this->lmcBufferGlobal_[lmcTokenDsaOffset], tokensBufferTensor[kLen + vLen], tokenCopyParams);
            }
        } else {
            // twoMajor: local [K|V] -> lmc [K_plane | V_plane]
            tokenCopyParams.blockLen = perCacheKBlockLen;
            tokenCopyParams.srcStride = perCacheVBlockLen;
            tokenCopyParams.dstStride = 0;
            
            // Copy K
            AscendC::DataCopy(this->lmcBufferGlobal_[lmcTokenKOffset], tokensBufferTensor, tokenCopyParams);
            
            // Copy V
            int64_t lmcTokenVOffset = lmcTokenKOffset + this->lmcValueOffset_;
            tokenCopyParams.blockLen = perCacheVBlockLen;
            tokenCopyParams.srcStride = perCacheKBlockLen;
            AscendC::DataCopy(this->lmcBufferGlobal_[lmcTokenVOffset], tokensBufferTensor[kLen], tokenCopyParams);
        }
    }

    __aicore__ inline void CopyLmcToLocal(local_scalar_t& tokensBufferTensor, int32_t tokenIdx, int32_t actualTokensPerLoop) {
        int32_t kLen = this->numHeads_ * this->headDims_;
        int32_t vLen = this->numHeads_ * this->vHeadDims_;
        int32_t dsaLen = this->numHeads_ * this->dsaHeadDims_;
        int64_t perCacheKBlockLen = (kLen * sizeof(scalar_t)) / ASCEND_BLOCK_LEN;
        int64_t perCacheVBlockLen = (vLen * sizeof(scalar_t)) / ASCEND_BLOCK_LEN;
        int64_t perCacheDsaBlockLen = (dsaLen * sizeof(scalar_t)) / ASCEND_BLOCK_LEN;
        
        AscendC::DataCopyParams tokensCopyParams;
        tokensCopyParams.blockCount = actualTokensPerLoop;
        int64_t lmcTokenIdx = tokenIdx * this->lmcTokenStride_;
        
        if (this->lmcTokensMajor_ || this->numKvs_ == 1) {
            // MLA/DSA/token-major: lmc [K|V|DSA] per token -> local [K|V|DSA] per token
            // K
            tokensCopyParams.blockLen = perCacheKBlockLen;
            tokensCopyParams.srcStride = (this->lmcTokenStride_ - kLen) * sizeof(scalar_t) / ASCEND_BLOCK_LEN;
            tokensCopyParams.dstStride = (vLen + dsaLen) * sizeof(scalar_t) / ASCEND_BLOCK_LEN;
            AscendC::DataCopy(tokensBufferTensor, this->lmcBufferGlobal_[lmcTokenIdx], tokensCopyParams);
            
            // V
            int64_t lmcTokenVIdx = lmcTokenIdx + this->lmcValueOffset_;
            tokensCopyParams.blockLen = perCacheVBlockLen;
            tokensCopyParams.srcStride = (this->lmcTokenStride_ - vLen) * sizeof(scalar_t) / ASCEND_BLOCK_LEN;
            tokensCopyParams.dstStride = (kLen + dsaLen) * sizeof(scalar_t) / ASCEND_BLOCK_LEN;
            AscendC::DataCopy(tokensBufferTensor[kLen], this->lmcBufferGlobal_[lmcTokenVIdx], tokensCopyParams);
            
            // DSA (only if dsaHeadDims > 0)
            if (dsaLen > 0) {
                int64_t lmcTokenDsaIdx = lmcTokenIdx + this->lmcDsaOffset_;
                tokensCopyParams.blockLen = perCacheDsaBlockLen;
                tokensCopyParams.srcStride = (this->lmcTokenStride_ - dsaLen) * sizeof(scalar_t) / ASCEND_BLOCK_LEN;
                tokensCopyParams.dstStride = (kLen + vLen) * sizeof(scalar_t) / ASCEND_BLOCK_LEN;
                AscendC::DataCopy(tokensBufferTensor[kLen + vLen], this->lmcBufferGlobal_[lmcTokenDsaIdx], tokensCopyParams);
            }
        } else {
            // twoMajor: lmc [K_plane | V_plane] -> local [K|V]
            tokensCopyParams.blockLen = perCacheKBlockLen;
            tokensCopyParams.srcStride = 0;
            tokensCopyParams.dstStride = perCacheVBlockLen;
            
            // Copy K
            AscendC::DataCopy(tokensBufferTensor, this->lmcBufferGlobal_[lmcTokenIdx], tokensCopyParams);
            
            // Copy V
            int64_t localVOffset = kLen;
            int64_t lmcTokenVIdx = lmcTokenIdx + this->lmcValueOffset_;
            tokensCopyParams.blockLen = perCacheVBlockLen;
            tokensCopyParams.dstStride = perCacheKBlockLen;
            AscendC::DataCopy(tokensBufferTensor[localVOffset], this->lmcBufferGlobal_[lmcTokenVIdx], tokensCopyParams);
        }
    }

private:
    AscendC::TPipe *pipe_;
    // Instance of the specific policy
    PolicyT policy_; 
    // a depth of 2
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 2> tokenQue_;

    // Depends on LMC setting whether we store in tokensMajor or not.
    // the layout would be the followings:
    // [tokens, kvs, heads*headsize] or [kvs, tokens, heads*headsize]
    // TODO: check whether should combine the two and use a loop
    AscendC::GlobalTensor<scalar_t> lmcBufferGlobal_;

    int64_t lmcTokenStride_;
    int64_t lmcValueOffset_;
    int64_t lmcDsaOffset_;   // offset of DSA data within a token row (k_hidden + v_hidden)
    int32_t blockSize_; // the size of the paged attention tokens block
    int32_t headDims_;
    int32_t vHeadDims_; // V head dims (same as headDims for non-MLA)
    int32_t dsaHeadDims_; // DSA extra head dims (0 if not DSA)
    int32_t numHeads_;
    int32_t numTokens_; // num tokens in the cache tensor chunk
    int16_t numKvs_; // 2 (K and V; MLA/DSA uses lmcTokensMajor_ path instead)
    bool page2L_; // whether the direction of copy is from page to lmc
    bool lmcTokensMajor_; // whether the lmc buffer is in tokens major i.e. [tokens, kvs, ...]
};

#endif // SINGLE_LAYER_MEM_KERNELS_V2_H
