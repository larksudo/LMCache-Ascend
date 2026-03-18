#include "kernel_operator.h"
#include <cstring>
#include <stdexcept>

namespace kvcache_ops {
namespace cachegen {
namespace impl {

__aicore__ inline auto ceil_32(int32_t size) -> uint32_t {
    return size % 32 == 0 ? size : 32 * (1 + (size / 32));
};

class CdfCalulator {
public:
    __aicore__ inline CdfCalulator(
        GM_ADDR input, // Input symbols [n_layers, n_tokens, n_channels], uint8
        GM_ADDR output, // Output CDF [n_layers, n_channels, n_bins + 1], uint16
        AscendC::TPipe& pipe,
        int32_t n_tokens, // Chunked at a higher level limiting the input size
        int32_t n_layers,
        int32_t n_channels,
        uint32_t n_bins,
        float scale_factor); // Factor to convert a tally to a frequency distributed over the desired range

    __aicore__ inline void run(int layer_id, int channel_id);

private:
    AscendC::GlobalTensor<uint8_t> gm_input;
    AscendC::GlobalTensor<uint16_t> gm_output;

    AscendC::TPipe& pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQ;

    // Calcualation space - local tensors used throughout the calculation. All are relatively small ([n_bins + 1])
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
    AscendC::LocalTensor<float> ub_cdf_calc_f; // [n_bins + 1]
    AscendC::LocalTensor<int32_t> ub_cdf_calc_i32; // [n_bins + 1]
    AscendC::LocalTensor<uint16_t> ub_cdf_calc_i32_as_u16; // Shares storage with ub_cdf_calc_i32
    AscendC::LocalTensor<int32_t> ub_linear_filter; // [n_bins + 1]
    AscendC::LocalTensor<int32_t> ub_gather_every_other_filter; // [n_bins + 1]

    // Input dimensions
    int32_t n_tokens;
    int32_t n_layers;
    int32_t n_channels;
    uint32_t n_bins;

    // Rescales tally over [0, U16_MAX - n_bins) leaving headroom for a tie break. Typically (U16_MAX - n_bins) / n_tokens.
    // Calculation of the scale factor is easiest to do on the cpu where there are fewer type limitations, so taken as 
    // a class input.
    float scale_factor; 

    // Derived values - the input dimensions dictate the number of symbols that are copied in each pass etc. Stored here
    // at class scope to avoid recomputation
    int32_t copy_volume;
    int32_t full_chunk_volume;
    int32_t max_full_chunk_id;
    int32_t tail_chunk_size;
    bool has_tail_chunk;

    // Compute the tally, storing the result in the approprate Calc buffer
    __aicore__ inline void tally(int layer_id, int channel_id);

    __aicore__ inline void copy_in_enq(int layer_offset, int chunk_id, uint32_t copy_volume);

    __aicore__ inline void deq_count(int32_t* count, uint32_t& token_idx, uint32_t copy_count);

    // Convert the tally into the desired CDF, storing the result in the approprate Calc buffer
    __aicore__ inline void tally_to_cdf();

    // Class has no known need to support move or copy operations
    CdfCalulator(const CdfCalulator&) = delete;
    CdfCalulator& operator=(const CdfCalulator&) = delete;
    CdfCalulator(CdfCalulator&&) = delete;
    CdfCalulator& operator=(CdfCalulator&&) = delete;
};

__aicore__ inline CdfCalulator::CdfCalulator(
    GM_ADDR _input,
    GM_ADDR _output,
    AscendC::TPipe& _pipe,
    int32_t n_tokens,
    int32_t n_layers,
    int32_t n_channels,
    uint32_t n_bins,
    float scale_factor):
        pipe(_pipe),
        n_tokens(n_tokens),
        n_layers(n_layers),
        n_channels(n_channels),
        n_bins(n_bins),
        scale_factor(scale_factor) {

    gm_input.SetGlobalBuffer(_input, n_tokens * n_layers * n_channels);
    gm_output.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(_output), n_layers * n_channels * (n_bins + 1));

    // Inner per layer chunking variables
    copy_volume = n_channels * n_tokens;
    // TODO: Size of copy should be dictated by the UB size, rather than statically set
    full_chunk_volume = 1 << 15; // 2^14 == 16,384
    max_full_chunk_id = copy_volume >> 15;
    tail_chunk_size = copy_volume & ((1 << 15) - 1);
    has_tail_chunk = tail_chunk_size != 0;

    pipe.InitBuffer(inQ, 2, full_chunk_volume);
    pipe.InitBuffer(outQ, 1, (n_bins + 1) * sizeof(uint16_t));

    // Alloc sufficient space for various intermediate results.  Note that TBuf allocations must be 32-byte aligned
    uint32_t buff_size_aligned = 
        ceil_32(sizeof(float) * (n_bins + 1)) + // ub_cdf_calc_f
        ceil_32(sizeof(int32_t) * (n_bins + 1)) + // ub_cdf_calc_i32 & ub_cdf_calc_i32_as_u16
        ceil_32(sizeof(int32_t) * (n_bins + 1)) +  // ub_linear_filter
        ceil_32(sizeof(uint32_t) * (n_bins + 1)); // ub_gather_every_other_filter
    pipe.InitBuffer(calcBuf, buff_size_aligned);
    int32_t offset = 0;
    
    ub_cdf_calc_f = calcBuf.GetWithOffset<float>((n_bins + 1), offset);
    offset += ceil_32(sizeof(float) * (n_bins + 1));

    ub_cdf_calc_i32 = calcBuf.GetWithOffset<int32_t>((n_bins + 1), offset);
    ub_cdf_calc_i32_as_u16 = ub_cdf_calc_i32.ReinterpretCast<uint16_t>(); // [bin_0, 0, bin_1, 0]
    ub_cdf_calc_i32_as_u16.SetSize(2 * (n_bins + 1));
    offset += ceil_32(sizeof(int32_t) * (n_bins + 1));

    ub_linear_filter = calcBuf.GetWithOffset<int32_t>((n_bins + 1), offset);
    offset += ceil_32(sizeof(int32_t) * (n_bins + 1));

    ub_gather_every_other_filter = calcBuf.GetWithOffset<int32_t>((n_bins + 1), offset);
    offset += ceil_32(sizeof(int32_t) * (n_bins + 1));

    AscendC::ArithProgression<int32_t>(ub_linear_filter, 0, 1, n_bins + 1);
    AscendC::ArithProgression<int32_t>(ub_gather_every_other_filter, 0, 2 * sizeof(uint16_t), n_bins + 1);
}

__aicore__ inline void CdfCalulator::run(int layer_id, int channel_id) {
    tally(layer_id, channel_id);
    tally_to_cdf();

    AscendC::LocalTensor<uint16_t> ub_output_hist = outQ.DeQue<uint16_t>();
    AscendC::DataCopyExtParams copyParams = {1, 2 * (n_bins + 1), 0, 0, 0};
    auto channel_size = (n_bins + 1);
    auto layer_size = channel_size * n_channels;
    auto output_offset = layer_size * layer_id + channel_size * channel_id; 
    AscendC::DataCopyPad(gm_output[output_offset], ub_output_hist, copyParams);
    outQ.FreeTensor(ub_output_hist);
}

__aicore__ inline void CdfCalulator::tally(int layer_id, int channel_id) {
    // Arbitrary size of 256 exceeds the max bin size which is typically not more than 32
    int32_t count[256] = {};
    auto layer_offset = layer_id * n_channels * n_tokens;
    uint32_t token_idx = channel_id;
    for (auto chunk_id = 0; chunk_id < max_full_chunk_id; ++chunk_id) {
        copy_in_enq(layer_offset, chunk_id, full_chunk_volume);
        deq_count(count, token_idx, full_chunk_volume);
    }

    if (has_tail_chunk) {
        copy_in_enq(layer_offset, max_full_chunk_id, tail_chunk_size);
        deq_count(count, token_idx, tail_chunk_size);
    }

    // End result should be a cumulative tally in a Local Tensor, ready for further calculation
    uint32_t tally = 0;
    ub_cdf_calc_i32.SetValue(0, tally); 
    for (uint8_t hist_idx = 1; hist_idx <= n_bins; hist_idx++) {
        tally += count[hist_idx - 1];
        ub_cdf_calc_i32.SetValue(hist_idx, tally); 
    }
}

__aicore__ inline void CdfCalulator::copy_in_enq(int layer_offset, int chunk_id, uint32_t copy_volume) {
    // Input shape is [layers, tokens, channels]
    //
    // OPTIMIZATION OPPORTUNITY:
    // 
    // The various datacopy interfaces can't copy only [layers, :, channels] because of alignment and granularity 
    // constraints. Instead, this implementation copies the whole layer (in chunks) and then operates only on relevant
    // data from the channel of interest once in local memory. This incurs significant redunant data transfer (measured
    // as ~33% of execution time in some cases on the original CDF kernel) but overall CDF calculation is not the encode
    // bottleneck so this is okay. If that changes, optimize this copy and the tally calculation (below).
    AscendC::LocalTensor<uint8_t> ub_input = inQ.AllocTensor<uint8_t>();
    const AscendC::DataCopyExtParams copyParams = {1, (uint32_t)(copy_volume), 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint8_t> padParams = {false, 0, 0, 0};
    AscendC::DataCopyPad(ub_input, gm_input[layer_offset + chunk_id * full_chunk_volume], copyParams, padParams);
    inQ.EnQue(ub_input);
}

__aicore__ inline void CdfCalulator::deq_count(int32_t* count, uint32_t& token_idx, uint32_t copy_volume) {
    // OPTIMIZATION OPPORTUNITY:
    //
    // This tally is a scalar heavy approach - CDF calculation is not the encode bottleneck so this is okay. If CDF
    // computation does become a limitiation then measurements of the original CDF calculation has this accounting for
    // 60% of execution time.
    AscendC::LocalTensor<uint8_t> ub_input = inQ.DeQue<uint8_t>();
    for (; token_idx < copy_volume; token_idx += n_channels) {
        count[ub_input(token_idx)] += 1;
    }
    token_idx = token_idx - copy_volume;
    inQ.FreeTensor(ub_input);
}

__aicore__ inline void CdfCalulator::tally_to_cdf() {
    // To make use of AscendC::Muls, cast to float. The counts being cast can comfortably be exactly represented as
    // floats - this cast safe
    AscendC::Cast(ub_cdf_calc_f, ub_cdf_calc_i32, AscendC::RoundMode::CAST_NONE, n_bins + 1);
    // Covert the tally to a normalized frequency
    AscendC::Muls(ub_cdf_calc_f, ub_cdf_calc_f, scale_factor, n_bins + 1);
    AscendC::Cast(ub_cdf_calc_i32, ub_cdf_calc_f, AscendC::RoundMode::CAST_FLOOR, n_bins + 1);
    // Break ties in the histogram with a linear mask
    AscendC::Add(ub_cdf_calc_i32, ub_cdf_calc_i32, ub_linear_filter, n_bins + 1);
    AscendC::ShiftLeft(ub_cdf_calc_i32, ub_cdf_calc_i32, 8, n_bins + 1);

    AscendC::LocalTensor<uint16_t> ub_output_hist = outQ.AllocTensor<uint16_t>();
    // Cast to u16 not supported so, as a work around, gather the relevant bits from the Reinterpreted i32 
    AscendC::Gather(ub_output_hist, ub_cdf_calc_i32_as_u16, ub_gather_every_other_filter.ReinterpretCast<uint32_t>(), 0, n_bins + 1);

    // Run a mock encoding of each symbol so that the length of encodes can be smuggled in the lower bits of the CDF.
    //
    // Find the longest span that fits within a CDF bucket. Once determined, any unused bit allocation can be passed
    // to improve encode efficiency
    uint8_t lens[64] {0};
    const int max_symbol = n_bins -1;
    for (uint8_t mock_sym = 0; mock_sym < n_bins; mock_sym++) {
        const uint16_t c_low = ub_output_hist(mock_sym);
        const uint16_t c_high = mock_sym == max_symbol ? 0xFF00U : ub_output_hist(mock_sym + 1);

        // Find the first bit set in high but not in low
        uint16_t next_pos = AscendC::ScalarCountLeadingZero(static_cast<uint64_t>(c_high & (~c_low))) - 47;
        uint8_t len_base = static_cast<uint8_t>(next_pos);

        uint16_t base_encode = c_low;
        base_encode = (base_encode >> (16 - len_base)) << (16 - len_base);

        // Propose two encodings and determine which unambiguously (no matter what is appended) identifies the symbol
        uint16_t upper = base_encode + ((1 << (16 - len_base))); // 0x b1 b2 .. b_len(1) 0 0 0 0
        uint16_t lower = upper - 1; // 0x b1 b2 .. b_len(0) 1 1 1 1

        // first 1 in high not in upper. Crop here and upper is guaranteed less than high 
        uint8_t len_break_upper = AscendC::ScalarCountLeadingZero(static_cast<uint64_t>(c_high & ~upper)) - 47; 
        // first 1 in lower not in low. Crop here and lower is guaranteed greater than high
         uint8_t len_break_lower = AscendC::ScalarCountLeadingZero(static_cast<uint64_t>(lower & ~c_low)) - 47;

        if (mock_sym == 0 || len_break_lower >= lens[mock_sym - 1]) {
            // No need to append that last 1 as we know that all remaining digits (in binary) are 0 so anything appended
            // will result in a val greater than or equal to the lower bound
            len_break_lower -= 1;
        }

        if (len_break_upper < len_break_lower) {
            lens[mock_sym] = len_break_upper;
        } else {
            lens[mock_sym] = len_break_lower;
        }

        uint16_t c_high_mask;
        c_high_mask = ((1 << lens[mock_sym]) - 1) << (16 - lens[mock_sym]);
        ub_output_hist.SetValue(mock_sym + 1, (c_high & c_high_mask));
    }

    for (uint8_t mock_sym = 0; mock_sym < n_bins; mock_sym++) {
        ub_output_hist.SetValue(mock_sym + 1, ub_output_hist(mock_sym + 1) | lens[mock_sym]);
    }

    outQ.EnQue(ub_output_hist);
}
} // namespace impl
} // namespace cachegen
} // namespace kvcache_ops

extern "C" __global__ __aicore__ void calculate_cdf_kernel (
    GM_ADDR input,
    GM_ADDR output,
    const int n_tokens,
    const int n_layers,
    const int n_channels,
    const uint32_t n_bins,
    const float scale_factor) {
    AscendC::TPipe pipe{};
    kvcache_ops::cachegen::impl::CdfCalulator cdf_calulator{
        input,
        output,
        pipe,
        n_tokens,
        n_layers,
        n_channels,
        n_bins,
        scale_factor
    };

    int max_work_idx = n_layers * n_channels;

    int32_t coreIdx = AscendC::GetBlockIdx();
    int32_t launchedCores = AscendC::GetBlockNum();

    for (int work_idx = coreIdx; work_idx < max_work_idx; work_idx += launchedCores) {
        int layer_id = work_idx % n_layers;
        int channel_id = work_idx / n_layers;
        cdf_calulator.run(layer_id, channel_id);
    }
}

namespace kvcache_ops {
namespace cachegen {

void calculate_cdf(
    uint8_t* input,
    uint8_t* output,
    void* stream,
    const int n_aiv,
    const int n_bins, 
    const int n_tokens, 
    const int n_layers, 
    const int n_channels) {

    // Dedicate 8 bits to the cdf, one to signal if the most favourable encode is upper or lower aligned, and 5 bits to the encode length
    // Scale to 255 => longest encode will be 8
    float scale_factor = static_cast<float>(0x00FF - n_bins - 1) / static_cast<float>(n_tokens);

    int blockDim = n_layers * n_channels < n_aiv ? n_layers * n_channels : n_aiv;

    calculate_cdf_kernel<<<blockDim, nullptr, stream>>>(
        input,
        output,
        n_tokens,
        n_layers,
        n_channels,
        n_bins,
        scale_factor);
}
} // namespace cachegen
} // namespace kvcache_ops
