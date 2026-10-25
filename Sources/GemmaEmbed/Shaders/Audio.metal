// The gemma4a audio tower: subsampling convolutions, clipped-linear glue, local attention with
// relative keys, the GLU and depthwise convolution module. Written for correctness against the
// CPU reference first; the GEMMs and norms come from GEMM.metal and Elementwise.metal.

struct AudioConvParams {
    uint inTime;
    uint inFreq;
    uint inChannels;
    uint outTime;
    uint outFreq;
    uint outChannels;
    uint inOffset;
    uint outOffset;
    float eps;
};

struct AudioElementwiseParams {
    uint rows;
    uint width;
    uint perDimWidth;
    float scale;
    float lo;
    float hi;
};

struct AudioRowsParams {
    uint rows;
    uint width;
    uint taps;
    float scale;
};

struct AudioAttentionParams {
    uint tokens;
    uint heads;
    uint headDim;
    uint window;
    float softcap;
};

// One threadgroup per output position (freq, time) of a 3x3 stride-2 pad-1 convolution over
// `x[(inOffset + t) * inFreq + f][inChannels]`, one thread per output channel, followed by a
// LayerNorm over the channels, the norm weight and ReLU. ggml runs the im2col in fp16, so every
// input value is rounded to half first. The kernel is `[outChannels][inChannels][3][3]` fp32.
// The 1->128 layer runs here; the 128->32 layer runs as the implicit GEMM below.
kernel void audio_subsample(device const float* x [[buffer(0)]],
                            device const float* w [[buffer(1)]],
                            device const float* norm [[buffer(2)]],
                            device float* out [[buffer(3)]],
                            constant AudioConvParams& p [[buffer(4)]],
                            uint2 group [[threadgroup_position_in_grid]],
                            uint oc [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sgid [[simdgroup_index_in_threadgroup]],
                            uint simdgroups [[simdgroups_per_threadgroup]]) {
    threadgroup float partial[32];
    const uint of = group.x, ot = group.y;
    float acc = 0.0f;
    if (oc < p.outChannels) {
        device const float* wk = w + oc * p.inChannels * 9;
        for (uint kh = 0; kh < 3; kh++) {
            const int t = int(2 * ot + kh) - 1;
            if (t < 0 || t >= int(p.inTime)) continue;
            for (uint kw = 0; kw < 3; kw++) {
                const int f = int(2 * of + kw) - 1;
                if (f < 0 || f >= int(p.inFreq)) continue;
                device const float* xi = x + ((p.inOffset + uint(t)) * p.inFreq + uint(f)) * p.inChannels;
                for (uint ic = 0; ic < p.inChannels; ic++) {
                    acc += float(half(xi[ic])) * wk[ic * 9 + kh * 3 + kw];
                }
            }
        }
    }
    float s = simd_sum(oc < p.outChannels ? acc : 0.0f);
    if (lane == 0) partial[sgid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = 0.0f;
    for (uint i = 0; i < simdgroups; i++) mean += partial[i];
    mean /= float(p.outChannels);
    const float d = acc - mean;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = simd_sum(oc < p.outChannels ? d * d : 0.0f);
    if (lane == 0) partial[sgid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float variance = 0.0f;
    for (uint i = 0; i < simdgroups; i++) variance += partial[i];
    variance /= float(p.outChannels);
    if (oc >= p.outChannels) return;
    const float scale = 1.0f / precise::sqrt(variance + p.eps);
    out[((p.outOffset + ot) * p.outFreq + of) * p.outChannels + oc] = max(0.0f, d * scale * norm[oc]);
}

// The same convolution as an implicit GEMM for the 128->32 layer: Y[M, 32] = patch[M, K] * W[32, K]^T
// with M the output positions in (time, freq) order and K = 9 * inChannels in (tap, channel) order,
// so each row's K slice for one tap is a contiguous channel run of `x`. The patch tile is gathered
// into threadgroup memory on the fly (fp16-rounded, zero outside the input) instead of an im2col
// buffer, the weight tile is read from the `[outChannels][inChannels][3][3]` layout with a stride-9
// gather, and the LayerNorm, norm weight and ReLU run over the staged output tile, one output row
// per simdgroup pass. BN is fixed at 32: a tile holds whole output rows, which the norm needs.
template <int BM, int BK, int SG>
kernel void audio_subsample_gemm(device const float* x [[buffer(0)]],
                                 device const float* w [[buffer(1)]],
                                 device const float* norm [[buffer(2)]],
                                 device float* out [[buffer(3)]],
                                 constant AudioConvParams& p [[buffer(4)]],
                                 uint tg [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint sgid [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    constexpr int BN = 32;
    constexpr int LD = BK + 4;
    constexpr int THREADS = SG * 32;
    constexpr int SN = BN / SG;
    constexpr int TM = BM / 8, TN = SN / 8;
    constexpr int OUT_LD = BN + 4;
    constexpr int STAGE = (BM + BN) * LD;
    static_assert(STAGE >= BM * OUT_LD, "the output tile must fit the staging buffer");
    threadgroup float stage[STAGE];
    threadgroup float* Xs = stage;
    threadgroup float* Ws = stage + BM * LD;

    const uint M = p.outTime * p.outFreq;
    const uint K = 9 * p.inChannels;
    const uint m0 = tg * BM;

    simdgroup_float8x8 acc[TM][TN];
    #pragma clang loop unroll(full)
    for (int i = 0; i < TM; i++)
        #pragma clang loop unroll(full)
        for (int j = 0; j < TN; j++) acc[i][j] = simdgroup_float8x8(0.0f);

    for (uint k0 = 0; k0 < K; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint u = tid; u < BM * BK / 4; u += THREADS) {
            const uint row = u / (BK / 4), c4 = u % (BK / 4);
            const uint m = m0 + row, k = k0 + c4 * 4;
            const uint ot = m / p.outFreq, of = m % p.outFreq;
            const uint tap = k / p.inChannels, ic = k % p.inChannels;
            const int t = int(2 * ot + tap / 3) - 1, f = int(2 * of + tap % 3) - 1;
            float4 v = 0.0f;
            if (m < M && t >= 0 && t < int(p.inTime) && f >= 0 && f < int(p.inFreq)) {
                const float4 raw = *(device const float4*)(x + ((p.inOffset + uint(t)) * p.inFreq + uint(f)) * p.inChannels + ic);
                v = float4(half4(raw));
            }
            threadgroup float* dst = Xs + row * LD + c4 * 4;
            dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
        }
        for (uint u = tid; u < BN * BK / 8; u += THREADS) {
            const uint row = u / (BK / 8), c8 = u % (BK / 8);
            const uint k = k0 + c8 * 8;
            device const float* src = w + (row * p.inChannels + k % p.inChannels) * 9 + k / p.inChannels;
            threadgroup float* dst = Ws + row * LD + c8 * 8;
            #pragma clang loop unroll(full)
            for (int i = 0; i < 8; i++) dst[i] = src[i * 9];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 a[TM];
            simdgroup_float8x8 b[TN];
            #pragma clang loop unroll(full)
            for (int i = 0; i < TM; i++)
                simdgroup_load(a[i], Xs + i * 8 * LD + kk, LD);
            #pragma clang loop unroll(full)
            for (int j = 0; j < TN; j++)
                simdgroup_load(b[j], Ws + (sgid * SN + j * 8) * LD + kk, LD, ulong2(0, 0), true);
            #pragma clang loop unroll(full)
            for (int i = 0; i < TM; i++)
                #pragma clang loop unroll(full)
                for (int j = 0; j < TN; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (int i = 0; i < TM; i++)
        #pragma clang loop unroll(full)
        for (int j = 0; j < TN; j++)
            simdgroup_store(acc[i][j], stage + i * 8 * OUT_LD + sgid * SN + j * 8, OUT_LD);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float nw = norm[lane];
    for (uint row = sgid; row < BM; row += SG) {
        const uint m = m0 + row;
        if (m >= M) break;
        const float v = stage[row * OUT_LD + lane];
        const float mean = simd_sum(v) / float(BN);
        const float d = v - mean;
        const float variance = simd_sum(d * d) / float(BN);
        const float scale = 1.0f / precise::sqrt(variance + p.eps);
        out[(p.outOffset * p.outFreq + m) * BN + lane] = max(0.0f, d * scale * nw);
    }
}

#define AUDIO_SUBSAMPLE_GEMM(SUFFIX, BM, BK, SG) \
    template [[host_name("audio_subsample_gemm_" SUFFIX)]] kernel void audio_subsample_gemm<BM, BK, SG>( \
        device const float*, device const float*, device const float*, device float*, constant AudioConvParams&, uint, uint, uint, uint);

AUDIO_SUBSAMPLE_GEMM("32x32x32_4", 32, 32, 4)

// x = clamp(x * scale * perDim[i % perDimWidth], lo, hi), the per-dimension factor only when perDimWidth > 0.
kernel void audio_elementwise(device float* x [[buffer(0)]],
                              device const float* perDim [[buffer(1)]],
                              constant AudioElementwiseParams& p [[buffer(2)]],
                              uint i [[thread_position_in_grid]]) {
    if (i >= p.rows * p.width) return;
    float v = x[i] * p.scale;
    if (p.perDimWidth > 0) v *= perDim[i % p.perDimWidth];
    x[i] = clamp(v, p.lo, p.hi);
}

kernel void audio_silu(device float* x [[buffer(0)]],
                       constant AudioRowsParams& p [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= p.rows * p.width) return;
    const float v = x[i];
    x[i] = v / (1.0f + exp(-v));
}

// x += scale * t
kernel void audio_add(device float* x [[buffer(0)]],
                      device const float* t [[buffer(1)]],
                      constant AudioRowsParams& p [[buffer(2)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= p.rows * p.width) return;
    x[i] += p.scale * t[i];
}

kernel void audio_add_bias(device float* x [[buffer(0)]],
                           device const float* bias [[buffer(1)]],
                           constant AudioRowsParams& p [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= p.rows * p.width) return;
    x[i] += bias[i % p.width];
}

// out[t][i] = x[t][i] * sigmoid(x[t][width + i]) over rows of 2 * width.
kernel void audio_glu(device const float* x [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      constant AudioRowsParams& p [[buffer(2)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= p.rows * p.width) return;
    const uint t = i / p.width, c = i % p.width;
    const float gate = x[t * 2 * p.width + p.width + c];
    out[i] = x[t * 2 * p.width + c] * (1.0f / (1.0f + exp(-gate)));
}

// Causal depthwise convolution within a chunk: out[t][c] = sum_k x[t + k - (kernel - 1)][c] * w[c][k],
// with frames before the chunk's first row counting as zero.
kernel void audio_dwconv(device const float* x [[buffer(0)]],
                         device const float* w [[buffer(1)]],
                         device const uint* chunkStart [[buffer(2)]],
                         device float* out [[buffer(3)]],
                         constant AudioRowsParams& p [[buffer(4)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= p.rows * p.width) return;
    const uint t = i / p.width, c = i % p.width;
    const int first = int(chunkStart[t]);
    float s = 0.0f;
    for (uint k = 0; k < p.taps; k++) {
        const int src = int(t + k) - int(p.taps - 1);
        if (src >= first) s += x[src * p.width + c] * w[c * p.taps + k];
    }
    out[i] = s;
}

// Local causal attention with relative keys: query t attends to keys t - 11 ... t of its chunk, and the key at
// distance d scores q . k + q . rel[d], soft-capped as softcap * tanh(s / softcap).
// One simdgroup per (query, head): lane l owns dims 4l .. 4l + 3, every score is a simd reduction over the
// lanes, and the softmax runs redundantly on all 32 lanes. The threadgroup holds 8 consecutive queries of
// one head, so their overlapping keys share the cache lines.
#define AUDIO_ATTENTION_WINDOW 12
#define AUDIO_HEAD_DIM 128

kernel void audio_attention(device const float* q [[buffer(0)]],
                            device const float* k [[buffer(1)]],
                            device const float* v [[buffer(2)]],
                            device const float* rel [[buffer(3)]],
                            device const uint* chunkStart [[buffer(4)]],
                            device float* out [[buffer(5)]],
                            constant AudioAttentionParams& p [[buffer(6)]],
                            uint2 gid [[thread_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    const uint t = gid.x / 32, h = gid.y;
    if (t >= p.tokens || h >= p.heads) return;
    const uint width = p.heads * AUDIO_HEAD_DIM;
    const uint first = max(chunkStart[t], t + 1 - min(uint(AUDIO_ATTENTION_WINDOW), t + 1));
    const uint count = t + 1 - first;
    const uint col = h * AUDIO_HEAD_DIM + lane * 4;
    const float4 qt = *(device const float4*)(q + t * width + col);
    // scores[n] is the key at distance n; the loops walk n downwards so the sums run over keys in
    // increasing order, as the CPU reference does.
    float scores[AUDIO_ATTENTION_WINDOW];
#pragma clang loop unroll(full)
    for (int n = AUDIO_ATTENTION_WINDOW - 1; n >= 0; n--) {
        float partial = 0.0f;
        if (uint(n) < count) {
            const float4 kj = *(device const float4*)(k + (t - n) * width + col);
            const float4 rn = *(device const float4*)(rel + n * width + col);
            partial = dot(qt, kj) + dot(qt, rn);
        }
        scores[n] = simd_sum(partial);
    }
    float maxScore = -INFINITY;
#pragma clang loop unroll(full)
    for (int n = AUDIO_ATTENTION_WINDOW - 1; n >= 0; n--) {
        if (uint(n) < count) {
            scores[n] = p.softcap * precise::tanh(scores[n] / p.softcap);
            maxScore = max(maxScore, scores[n]);
        }
    }
    float sum = 0.0f;
#pragma clang loop unroll(full)
    for (int n = AUDIO_ATTENTION_WINDOW - 1; n >= 0; n--) {
        if (uint(n) < count) {
            scores[n] = exp(scores[n] - maxScore);
            sum += scores[n];
        }
    }
    float4 o = float4(0.0f);
#pragma clang loop unroll(full)
    for (int n = AUDIO_ATTENTION_WINDOW - 1; n >= 0; n--) {
        if (uint(n) < count) {
            const float w = scores[n] / sum;
            o += w * *(device const float4*)(v + (t - n) * width + col);
        }
    }
    *(device float4*)(out + t * width + col) = o;
}
