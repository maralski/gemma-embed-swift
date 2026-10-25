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

// One thread per (token, head). Query t sees keys max(chunkStart, t - window + 1) ... t; a key at
// distance d scores q . k + q . rel[d], soft-capped as softcap * tanh(s / softcap).
kernel void audio_attention(device const float* q [[buffer(0)]],
                            device const float* k [[buffer(1)]],
                            device const float* v [[buffer(2)]],
                            device const float* rel [[buffer(3)]],
                            device const uint* chunkStart [[buffer(4)]],
                            device float* out [[buffer(5)]],
                            constant AudioAttentionParams& p [[buffer(6)]],
                            uint2 gid [[thread_position_in_grid]]) {
    const uint t = gid.x, h = gid.y;
    if (t >= p.tokens || h >= p.heads) return;
    const uint width = p.heads * p.headDim;
    const uint first = max(chunkStart[t], t + 1 - min(p.window, t + 1));
    device const float* qt = q + t * width + h * p.headDim;
    float scores[32];
    float maxScore = -INFINITY;
    for (uint j = first; j <= t; j++) {
        device const float* kj = k + j * width + h * p.headDim;
        device const float* rj = rel + (t - j) * width + h * p.headDim;
        float s = 0.0f, r = 0.0f;
        for (uint i = 0; i < p.headDim; i++) {
            s += qt[i] * kj[i];
            r += qt[i] * rj[i];
        }
        const float capped = p.softcap * precise::tanh((s + r) / p.softcap);
        scores[j - first] = capped;
        maxScore = max(maxScore, capped);
    }
    float sum = 0.0f;
    for (uint j = first; j <= t; j++) {
        const float e = exp(scores[j - first] - maxScore);
        scores[j - first] = e;
        sum += e;
    }
    device float* o = out + t * width + h * p.headDim;
    for (uint i = 0; i < p.headDim; i++) o[i] = 0.0f;
    for (uint j = first; j <= t; j++) {
        const float w = scores[j - first] / sum;
        device const float* vj = v + j * width + h * p.headDim;
        for (uint i = 0; i < p.headDim; i++) o[i] += w * vj[i];
    }
}
