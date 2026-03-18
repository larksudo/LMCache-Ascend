#include "kernel_operator.h"

namespace kvcache_ops {
namespace cachegen {
namespace impl {

__aicore__ inline auto ceil_n(int32_t size, uint32_t n) -> uint32_t {
    return size % n == 0 ? size : n * (1 + (size / n));
};

__aicore__ inline auto ceil_32(int32_t size) -> uint32_t {
    return ceil_n(size, 32);
};

__aicore__ inline auto big_to_small(uint32_t value) -> uint32_t {
    return ((value & 0xFF000000U) >> 24) | ((value & 0x00FF0000U) >> 8) | ((value & 0x0000FF00U) << 8) | ((value & 0x000000FFU) << 24);
}
class DecoderAsc {
public:
    __aicore__ inline DecoderAsc(
        GM_ADDR cdf_data_ptr,  // Input CDF [n_layers, n_channels, n_bins + 1], uint16
        GM_ADDR input_bytestreams_data_ptr, // Input bytesteam [steam_length], uint8
        GM_ADDR input_lengths_data_ptr, // Input lengths [n_layers, n_channels], uint64
        GM_ADDR output_data_ptr, // Output symbols [n_layers, batch_size, n_channels], uint8
        AscendC::TPipe& pipe,
        int32_t n_tokens,
        int32_t n_layers,
        int32_t n_channels,
        uint32_t n_bins);

    __aicore__ inline void decode(int layer_id, int channel_id);

    const static uint32_t CHANNELS_PER_DECODE = 32;

private:
    AscendC::GlobalTensor<uint16_t> gm_cdf;
    AscendC::GlobalTensor<uint8_t> gm_input_bytestream;
    AscendC::GlobalTensor<uint64_t> gm_input_lens;
    AscendC::GlobalTensor<uint8_t> gm_output;

    AscendC::TPipe& pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> streamInQ;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> CDFInQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQ;

    // Input dimensions
    int32_t n_tokens;
    int32_t n_layers;
    int32_t n_channels;
    uint32_t n_bins;
    int32_t chunk_size;

    int32_t cdf_buf_size; // units of elements
    int32_t cdf_buf_size_div_64;  // How many multiples of 64 that is 

    // Calcualation space - local tensors used throughout the calculation. All are relatively small ([n_bins]).
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
    AscendC::LocalTensor<float>   ub_cdf_calc_f; // [cdf_buf_size, channels]
    AscendC::LocalTensor<int32_t> ub_cdf_calc_i32; // [cdf_buf_size, channels]
    AscendC::LocalTensor<uint8_t> ub_match_mask_calc; // [cdf_buf_size, channels]
    AscendC::LocalTensor<int32_t> ub_count_buf; // [channles]
    AscendC::LocalTensor<float> ub_duplicated_count_f_buf; // [cdf_buf_size, channels]
    AscendC::LocalTensor<int32_t> ub_duplicated_count_i32_buf; // [cdf_buf_size, channles], shares buffer space with ub_duplicated_value_buf
    AscendC::LocalTensor<int32_t> ub_gather_buf; // [cdf_buf_size, channels] - gather [v1, v2, v3, ...] to [[v1, v1, ...], [v2, v2, ...], ...] 
    AscendC::LocalTensor<int32_t> ub_gather_buf_2; // [cdf_buf_size x channels * 2] - casts u16 -> u32
    uint64_t mask_n_bins[2];

    __aicore__ inline void syms_from_val(
        uint8_t* local_syms, 
        AscendC::LocalTensor<uint8_t>& output_syms,
        uint32_t token_idx);

    __aicore__ inline void read_n_bits(
        uint16_t bits_used,
        uint32_t& value,
        int& bit_idx,
        int rel_channel_id,
        int& buffer_offset,
        AscendC::LocalTensor<uint8_t>& byte_stream
);

    // Class has no known need to support move or copy operations
    DecoderAsc(const DecoderAsc&) = delete;
    DecoderAsc& operator=(const DecoderAsc&) = delete;
    DecoderAsc(DecoderAsc&&) = delete;
    DecoderAsc& operator=(DecoderAsc&&) = delete;
};

__aicore__ inline DecoderAsc::DecoderAsc(
    GM_ADDR cdf_data_ptr,
    GM_ADDR input_bytestreams_data_ptr,
    GM_ADDR input_lengths_data_ptr,
    GM_ADDR output_data_ptr,
    AscendC::TPipe& _pipe,
    int32_t n_tokens,
    int32_t n_layers,
    int32_t n_channels,
    uint32_t n_bins):
        pipe(_pipe),
        n_tokens(n_tokens),
        n_layers(n_layers),
        n_channels(n_channels),
        n_bins(n_bins),
        chunk_size((((n_tokens - 1) >> 5) + 1) << 5
)
{
    gm_cdf.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(cdf_data_ptr), n_layers * n_channels * (n_bins + 1));
    gm_input_lens.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(input_lengths_data_ptr),  n_layers * n_channels);
    auto max_len = gm_input_lens((n_layers * n_channels) - 1);
    gm_input_bytestream.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(input_bytestreams_data_ptr),  max_len);
    gm_output.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output_data_ptr), n_layers * n_channels * n_tokens);

    // Involved in `Compare` calls which have a 256 byte granularity requirement
    cdf_buf_size = ceil_n(n_bins + 1, 256 / sizeof(float));
    cdf_buf_size_div_64 = cdf_buf_size / 64; 

    pipe.InitBuffer(streamInQ, 1, CHANNELS_PER_DECODE * chunk_size * sizeof(uint8_t)); 
    pipe.InitBuffer(CDFInQ, 1, CHANNELS_PER_DECODE * cdf_buf_size * sizeof(uint16_t));
    pipe.InitBuffer(outQ, 1, CHANNELS_PER_DECODE * chunk_size * sizeof(uint8_t));

    uint32_t buff_size_aligned = 
        ceil_32(sizeof(float) * CHANNELS_PER_DECODE * cdf_buf_size) + // ub_cdf_f
        ceil_32(sizeof(float) * CHANNELS_PER_DECODE * cdf_buf_size) + // ub_cdf_calc_i32
        ceil_32(sizeof(int32_t) * CHANNELS_PER_DECODE) + // ub_count_buf
        ceil_32(sizeof(int32_t) * CHANNELS_PER_DECODE * cdf_buf_size) + // ub_duplicated_count_f_buf / ub_duplicated_count_i32_buf
        ceil_32(sizeof(int32_t) * CHANNELS_PER_DECODE * cdf_buf_size) + // ub_gather_buf
        ceil_32(sizeof(int32_t) * cdf_buf_size * CHANNELS_PER_DECODE * 2) + // ub_gather_buf_2
        ceil_32(sizeof(uint8_t) * CHANNELS_PER_DECODE * (cdf_buf_size / 8)); // ub_match_mask_calc

    pipe.InitBuffer(calcBuf, buff_size_aligned);

    int32_t offset = 0;

    ub_cdf_calc_f = calcBuf.GetWithOffset<float>(CHANNELS_PER_DECODE * cdf_buf_size, offset);
    offset += ceil_32(sizeof(float) * CHANNELS_PER_DECODE * cdf_buf_size);

    ub_cdf_calc_i32 = calcBuf.GetWithOffset<int32_t>(CHANNELS_PER_DECODE * cdf_buf_size, offset);
    AscendC::Duplicate(ub_cdf_calc_i32, 0x10000, CHANNELS_PER_DECODE * cdf_buf_size);
    offset += ceil_32(sizeof(float) * CHANNELS_PER_DECODE * cdf_buf_size);

    AscendC::Cast(ub_cdf_calc_f, ub_cdf_calc_i32, AscendC::RoundMode::CAST_NONE, CHANNELS_PER_DECODE * cdf_buf_size);

    // Must use Cast to convert between these two as the formats are reinterpretable
    ub_duplicated_count_f_buf = calcBuf.GetWithOffset<float>(cdf_buf_size, offset);
    ub_duplicated_count_i32_buf = ub_duplicated_count_f_buf.ReinterpretCast<int32_t>();
    offset += ceil_32(sizeof(int32_t) * CHANNELS_PER_DECODE * cdf_buf_size);

    ub_match_mask_calc = calcBuf.GetWithOffset<uint8_t>(CHANNELS_PER_DECODE * (cdf_buf_size / 8), offset);
    offset += ceil_32(sizeof(uint8_t) * CHANNELS_PER_DECODE * (cdf_buf_size / 8));

    ub_count_buf = calcBuf.GetWithOffset<int32_t>(CHANNELS_PER_DECODE, offset);
    offset += ceil_32(sizeof(int32_t) * CHANNELS_PER_DECODE);

    ub_gather_buf = calcBuf.GetWithOffset<int32_t>(cdf_buf_size * CHANNELS_PER_DECODE, offset);
    offset += ceil_32(sizeof(int32_t) * cdf_buf_size * CHANNELS_PER_DECODE);
    for (int rel_channel_id = 0; rel_channel_id < CHANNELS_PER_DECODE; ++rel_channel_id) {
        AscendC::Duplicate(ub_gather_buf[rel_channel_id * cdf_buf_size], (int32_t)(sizeof(int32_t) * rel_channel_id), cdf_buf_size);
    }

    ub_gather_buf_2 = calcBuf.GetWithOffset<int32_t>(cdf_buf_size * CHANNELS_PER_DECODE * 2, offset);
    offset += ceil_32(sizeof(int32_t) * cdf_buf_size * CHANNELS_PER_DECODE * 2);

    // Gather u16s into every other u16 slot of reinterpreted i32 (because of limitation in Cast). Every other element
    // needs to be zero (gathered from [0]).
    // 
    // Couple this with a mask so it only gathers while there are valid bin values. Otherwise, leave initialized
    // (uint16_max) value untouched
    for (int ii = 0; ii < CHANNELS_PER_DECODE; ++ii) {
        for (int jj = 0; jj < (n_bins) * 2; jj += 2) {
            ub_gather_buf_2.SetValue(ii * 2 * cdf_buf_size + jj, ii * 2 * cdf_buf_size + jj);
            ub_gather_buf_2.SetValue(ii * 2 * cdf_buf_size + jj + 1, 0);
        }
    }

    if (n_bins / 32 == 0) {
        mask_n_bins[0] = (static_cast<uint64_t>(1) << 2 * n_bins) - 1;
        mask_n_bins[1] = 0x0;
    } else {
        mask_n_bins[0] = 0xFFFFFFFFFFFFFFFF;
        mask_n_bins[1] = n_bins == 64 ? 0xFFFFFFFFFFFFFFFF : (static_cast<uint64_t>(1) << 2 * (n_bins - 32)) - 1;
    }

}

__aicore__ inline void DecoderAsc::decode(int layer_id, int channel_start_id) {
    AscendC::LocalTensor<uint8_t> ub_steam_input = streamInQ.AllocTensor<uint8_t>();
    AscendC::LocalTensor<uint16_t> ub_cdf_input = CDFInQ.AllocTensor<uint16_t>();

    for (int channel_id = channel_start_id; channel_id < channel_start_id + CHANNELS_PER_DECODE; ++channel_id) {
        uint32_t encoded_output_start = (layer_id * n_channels + channel_id) == 0 ? 0 : gm_input_lens(layer_id * n_channels + channel_id -1);
        uint32_t encoded_output_end = gm_input_lens(layer_id * n_channels + channel_id);

        auto stream_input_offset = chunk_size * (channel_id - channel_start_id);

        AscendC::DataCopyPad(
            ub_steam_input[stream_input_offset],
            gm_input_bytestream[encoded_output_start],
            {1, encoded_output_end - encoded_output_start, 0, 0, 0},
            {false, 0, 0, 0}
        );

        auto channel_size = (n_bins + 1);
        auto layer_size = channel_size * n_channels;
        auto offset = layer_size * layer_id + channel_size * channel_id;
        auto cdf_input_offset = cdf_buf_size * (channel_id - channel_start_id);

        const AscendC::DataCopyExtParams cdf_copyParams = {1, (uint32_t)((n_bins + 1) * sizeof(uint16_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint16_t> cdf_padParams = {false, 0, 0, 0};
        AscendC::DataCopyPad(ub_cdf_input[cdf_input_offset], gm_cdf[offset], cdf_copyParams, cdf_padParams);
    }

    streamInQ.EnQue(ub_steam_input);
    ub_steam_input = streamInQ.DeQue<uint8_t>();

    CDFInQ.EnQue(ub_cdf_input);
    ub_cdf_input = CDFInQ.DeQue<uint16_t>();

    // Extract the encode lengths
    //
    // Each symbol is encoded with a consistent number of bits, but fewer bits are allocated to more common symbols, and
    // more to less common symbols. This is determined when computing the CDF. The top byte of the "CDF" are the cdf 
    // bins, the lower bytes stores the length for that symbol.
    uint8_t lens[32 * CHANNELS_PER_DECODE] {0};
    auto ub_lens = ub_cdf_input.ReinterpretCast<uint8_t>();
    for (int rel_channel_id = 0; rel_channel_id < CHANNELS_PER_DECODE; ++rel_channel_id) {
        auto ub_offset = 2 * cdf_buf_size * rel_channel_id;
        auto l_offset = 32 * rel_channel_id;
        for (uint8_t mock_sym = 0; mock_sym < n_bins; mock_sym++) {
            lens[mock_sym + l_offset] = ub_lens(2 * (mock_sym + 1) + ub_offset);
        }
    }

    AscendC::ShiftRight(ub_cdf_input, ub_cdf_input, (uint16_t)8, CHANNELS_PER_DECODE * cdf_buf_size);
    AscendC::ShiftLeft(ub_cdf_input, ub_cdf_input, (uint16_t)8, CHANNELS_PER_DECODE * cdf_buf_size);

    AscendC::Gather(ub_cdf_calc_i32.ReinterpretCast<uint16_t>(), ub_cdf_input, ub_gather_buf_2.ReinterpretCast<uint32_t>(), 0, mask_n_bins, (CHANNELS_PER_DECODE * cdf_buf_size) / 64, 8);
    AscendC::Cast(ub_cdf_calc_f, ub_cdf_calc_i32, AscendC::RoundMode::CAST_NONE, CHANNELS_PER_DECODE * cdf_buf_size);

    AscendC::LocalTensor<uint8_t> output_syms = outQ.AllocTensor<uint8_t>();

    int buffer_offset[CHANNELS_PER_DECODE] = {0};
    uint32_t values[CHANNELS_PER_DECODE];
    int bit_idx[CHANNELS_PER_DECODE] = {0};
    uint16_t bits_used[CHANNELS_PER_DECODE] = {0};

    for (int rel_channel_id = 0; rel_channel_id < CHANNELS_PER_DECODE; ++rel_channel_id) {
        values[rel_channel_id] = big_to_small(ub_steam_input[rel_channel_id * chunk_size + buffer_offset[rel_channel_id]].ReinterpretCast<uint32_t>()(0));
        buffer_offset[rel_channel_id] += 4;
    }

    for (int token_idx = 0; token_idx < n_tokens; ++token_idx) {
        for (int rel_channel_id = 0; rel_channel_id < CHANNELS_PER_DECODE; ++rel_channel_id) {
            // Remove consumed bytes
            values[rel_channel_id] <<= bits_used[rel_channel_id];
            bit_idx[rel_channel_id] += bits_used[rel_channel_id];
            if (bit_idx[rel_channel_id] >= 16) {
                // Read the next bytes into the buffer
                bit_idx[rel_channel_id] -= 16;
                uint32_t next = ub_steam_input[rel_channel_id * chunk_size + buffer_offset[rel_channel_id]].ReinterpretCast<uint16_t>()(0);
                buffer_offset[rel_channel_id] += 2;
                values[rel_channel_id] = values[rel_channel_id] | (next & 0x000000FFU) << (8 + bit_idx[rel_channel_id]) | ((next & 0x0000FF00U) >> 8) << bit_idx[rel_channel_id];
            }

            const uint16_t count = (values[rel_channel_id] >> 16);
            ub_count_buf.SetValue(rel_channel_id, count);
        }

        // Compare the value to all CDF buckets at once. The first "hit" (found through leading zeroes) identifies the
        // symbol
        AscendC::Gather(ub_duplicated_count_i32_buf, ub_count_buf, ub_gather_buf.ReinterpretCast<uint32_t>(), 0, CHANNELS_PER_DECODE * cdf_buf_size);
        AscendC::Cast(ub_duplicated_count_f_buf, ub_duplicated_count_i32_buf, AscendC::RoundMode::CAST_NONE, CHANNELS_PER_DECODE * cdf_buf_size);
        AscendC::Compare(ub_match_mask_calc, ub_cdf_calc_f, ub_duplicated_count_f_buf, AscendC::CMPMODE::LE, CHANNELS_PER_DECODE * cdf_buf_size);
        for (int rel_channel_id = 0; rel_channel_id < CHANNELS_PER_DECODE; ++rel_channel_id) {
            auto sym  = 63 - AscendC::ScalarCountLeadingZero(ub_match_mask_calc.ReinterpretCast<uint64_t>()(rel_channel_id * cdf_buf_size_div_64));
            output_syms.SetValue(token_idx * CHANNELS_PER_DECODE + rel_channel_id, sym);
            bits_used[rel_channel_id] = lens[sym + 32 * rel_channel_id];
        }
    }

    streamInQ.FreeTensor(ub_steam_input);
    CDFInQ.FreeTensor(ub_cdf_input);

    outQ.EnQue(output_syms);
    output_syms = outQ.DeQue<uint8_t>();

    uint32_t out_layer_offset = layer_id * n_channels * n_tokens;

    AscendC::SliceInfo dstSliceInfo[] = {{
        (uint32_t)(channel_start_id), // startIndex (element)
        ((n_tokens - 1) * n_channels) + (channel_start_id + CHANNELS_PER_DECODE), // endIndex (element)
        n_channels - CHANNELS_PER_DECODE, // stride (element)
        CHANNELS_PER_DECODE / 32, // burst len (datablock a.k.a 32 byte)
        (uint32_t)(n_tokens * n_channels)  // shape value (element)
    }};

    AscendC::SliceInfo srcSliceInfo[] = {{
        0,
        n_tokens * CHANNELS_PER_DECODE,
        0,
        CHANNELS_PER_DECODE / 32,
        n_tokens * CHANNELS_PER_DECODE
    }};

    AscendC::DataCopy(gm_output[out_layer_offset], output_syms, dstSliceInfo, srcSliceInfo);
    outQ.FreeTensor(output_syms);
}

} // namespace impl
} // namespace cachegen
} // namespace kvcache_ops

extern "C" __global__ __aicore__ void decode_v2_kernel (
    GM_ADDR cdf_data_ptr,
    GM_ADDR bytestreams_data_ptr,
    GM_ADDR lengths_data_ptr,
    GM_ADDR output_data_ptr,
    const uint32_t n_bins,
    const int n_tokens,
    const int n_layers,
    const int n_channels
) {
    AscendC::TPipe pipe{};

    int32_t coreIdx = AscendC::GetBlockIdx();
    int32_t launchedCores = AscendC::GetBlockNum();
    kvcache_ops::cachegen::impl::DecoderAsc decoder {
        cdf_data_ptr,
        bytestreams_data_ptr,
        lengths_data_ptr,
        output_data_ptr,
        pipe, 
        n_tokens,
        n_layers,
        n_channels,
        n_bins};

    auto work_max_id = n_layers * n_channels / kvcache_ops::cachegen::impl::DecoderAsc::CHANNELS_PER_DECODE;

    for (auto work_id = coreIdx; work_id < work_max_id; work_id += launchedCores) {
        int layer_id = work_id % n_layers ;
        int channel_start_id = kvcache_ops::cachegen::impl::DecoderAsc::CHANNELS_PER_DECODE * (work_id / n_layers);
        decoder.decode(layer_id, channel_start_id);
    }
}

namespace kvcache_ops {
namespace cachegen {

void decode_v2(
    uint8_t* cdf_data_ptr,
    uint8_t* bytestreams_data_ptr,
    uint8_t* lengths_data_ptr,
    uint8_t* output_data_ptr,
    void* stream,
    const int n_aiv,
    const int n_bins,
    const int n_tokens,
    const int n_layers,
    const int n_channels) {

    int decode_blockDim = n_layers * (n_channels / impl::DecoderAsc::CHANNELS_PER_DECODE) < n_aiv ? n_layers * (n_channels / impl::DecoderAsc::CHANNELS_PER_DECODE) : n_aiv;

    decode_v2_kernel<<<decode_blockDim, nullptr, stream>>>(
        cdf_data_ptr,
        bytestreams_data_ptr,
        lengths_data_ptr,
        output_data_ptr,
        n_bins,
        n_tokens,
        n_layers,
        n_channels);
}
} // namespace cachegen
} // namespace kvcache_ops
