// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
// Y[M,N] = X[M,K] * W[N,K]^T with fp32 activations and quantized or bf16 weights.
//
// Every loop over the fragment arrays is fully unrolled so they are indexed with constants and
// stay in registers; a rolled loop puts them in memory.

struct FmtQ8 {
    // Stages 8 consecutive weights of one row starting at column k (k % 8 == 0).
    static inline void load8(device const uchar* w, uint rowBytes, uint row, uint k, thread float* out) {
        device const uchar* b = w + row * rowBytes + (k / 32) * Q8_BLOCK_BYTES;
        const float scale = f16_to_float(ushort(b[0]) | (ushort(b[1]) << 8));
        device const ushort* q = (device const ushort*)(b + 2 + (k % 32));
        for (uint i = 0; i < 4; i++) {
            const ushort v = q[i];
            out[2 * i] = scale * float(as_type<char>(uchar(v & 0xFF)));
            out[2 * i + 1] = scale * float(as_type<char>(uchar(v >> 8)));
        }
    }
};

struct FmtBF16 {
    static inline void load8(device const uchar* w, uint rowBytes, uint row, uint k, thread float* out) {
        const uint4 v = *(device const uint4*)(w + row * rowBytes + k * 2);
        out[0] = bf16_to_float(ushort(v.x & 0xFFFF)); out[1] = bf16_to_float(ushort(v.x >> 16));
        out[2] = bf16_to_float(ushort(v.y & 0xFFFF)); out[3] = bf16_to_float(ushort(v.y >> 16));
        out[4] = bf16_to_float(ushort(v.z & 0xFFFF)); out[5] = bf16_to_float(ushort(v.z >> 16));
        out[6] = bf16_to_float(ushort(v.w & 0xFFFF)); out[7] = bf16_to_float(ushort(v.w >> 16));
    }
};

struct FmtF32 {
    static inline void load8(device const uchar* w, uint rowBytes, uint row, uint k, thread float* out) {
        device const float4* q = (device const float4*)(w + row * rowBytes + k * 4);
        const float4 a = q[0], b = q[1];
        out[0] = a.x; out[1] = a.y; out[2] = a.z; out[3] = a.w;
        out[4] = b.x; out[5] = b.y; out[6] = b.z; out[7] = b.w;
    }
};

struct EpiStore {
    static inline float4 apply(float4 v, device const float* aux, uint index) { return v; }
};

// y = gelu(aux) * y, where aux holds the gate projection at the same index.
struct EpiGeglu {
    static inline float4 apply(float4 v, device const float* aux, uint index) {
        const float4 g = *(device const float4*)(aux + index);
        return float4(gelu_tanh(g.x), gelu_tanh(g.y), gelu_tanh(g.z), gelu_tanh(g.w)) * v;
    }
};

// y = gelu_quick(aux) * y, the vision tower's FFN activation.
struct EpiGegluQuick {
    static inline float4 apply(float4 v, device const float* aux, uint index) {
        const float4 g = *(device const float4*)(aux + index);
        return float4(gelu_quick(g.x), gelu_quick(g.y), gelu_quick(g.z), gelu_quick(g.w)) * v;
    }
};

// y = gelu(y) * aux, where aux holds the per-layer input at the same index.
struct EpiGeluTimes {
    static inline float4 apply(float4 v, device const float* aux, uint index) {
        const float4 a = *(device const float4*)(aux + index);
        return float4(gelu_tanh(v.x), gelu_tanh(v.y), gelu_tanh(v.z), gelu_tanh(v.w)) * a;
    }
};

template <int BM, int BN, int BK>
constexpr int gemm_stage_floats() { return (BM + BN) * (BK + 4); }

template <typename Fmt, typename Epi, int BM, int BN, int BK, int SG>
inline void gemm_tile(device const float* X, device const uchar* W, device float* Y, constant GemmParams& p,
                      device const float* aux, threadgroup float* stage, uint2 tgid, uint tid, uint sgid) {
    constexpr int LD = BK + 4;
    constexpr int THREADS = SG * 32;
    constexpr int SN = BN / SG;
    constexpr int TM = BM / 8, TN = SN / 8;
    constexpr int STAGE = gemm_stage_floats<BM, BN, BK>();
    constexpr int OUT_LD = BN + 4;
    // Output rows staged per pass: the largest power-of-two fraction of BM that fits.
    constexpr int FIT = STAGE / OUT_LD;
    constexpr int PASS_ROWS = FIT >= BM ? BM : (FIT >= BM / 2 ? BM / 2 : (FIT >= BM / 4 ? BM / 4 : BM / 8));
    constexpr int PASSES = BM / PASS_ROWS;
    threadgroup float* Xs = stage;
    threadgroup float* Ws = stage + BM * LD;

    const uint m0 = tgid.y * BM;
    const uint n0 = tgid.x * BN;
    const uint K = p.K;

    simdgroup_float8x8 acc[TM][TN];
    #pragma clang loop unroll(full)
    for (int i = 0; i < TM; i++)
        #pragma clang loop unroll(full)
        for (int j = 0; j < TN; j++) acc[i][j] = simdgroup_float8x8(0.0f);

    for (uint k0 = 0; k0 < K; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint u = tid; u < BM * BK / 4; u += THREADS) {
            const uint row = u / (BK / 4), c4 = u % (BK / 4);
            const float4 v = *(device const float4*)(X + (m0 + row) * K + k0 + c4 * 4);
            threadgroup float* dst = Xs + row * LD + c4 * 4;
            dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
        }
        for (uint u = tid; u < BN * BK / 8; u += THREADS) {
            const uint row = u / (BK / 8), c8 = u % (BK / 8);
            float vals[8];
            Fmt::load8(W, p.weightRowBytes, n0 + row, k0 + c8 * 8, vals);
            threadgroup float* dst = Ws + row * LD + c8 * 8;
            for (int i = 0; i < 8; i++) dst[i] = vals[i];
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

    for (int pass = 0; pass < PASSES; pass++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int rowBase = pass * PASS_ROWS;
        #pragma clang loop unroll(full)
        for (int i = 0; i < TM; i++) {
            const int row = i * 8 - rowBase;
            if (row < 0 || row >= PASS_ROWS) continue;
            #pragma clang loop unroll(full)
            for (int j = 0; j < TN; j++)
                simdgroup_store(acc[i][j], stage + row * OUT_LD + sgid * SN + j * 8, OUT_LD);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint u = tid; u < PASS_ROWS * BN / 4; u += THREADS) {
            const uint row = u / (BN / 4), c4 = u % (BN / 4);
            threadgroup const float* src = stage + row * OUT_LD + c4 * 4;
            const uint index = (m0 + rowBase + row) * p.N + n0 + c4 * 4;
            float4 v = float4(src[0], src[1], src[2], src[3]);
            *(device float4*)(Y + index) = Epi::apply(v, aux, index);
        }
    }
}

template <typename Fmt, typename Epi, int BM, int BN, int BK, int SG>
kernel void gemm_kernel(device const float* X [[buffer(0)]],
                        device const uchar* W [[buffer(1)]],
                        device float* Y [[buffer(2)]],
                        constant GemmParams& p [[buffer(3)]],
                        device const float* aux [[buffer(4)]],
                        uint2 tgid [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint sgid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float stage[gemm_stage_floats<BM, BN, BK>()];
    gemm_tile<Fmt, Epi, BM, BN, BK, SG>(X, W, Y, p, aux, stage, tgid, tid, sgid);
}

template <typename Fmt, int BM, int BN, int BK, int SG>
kernel void gemm3_kernel(device const float* X [[buffer(0)]],
                         device const uchar* W0 [[buffer(1)]],
                         device float* Y0 [[buffer(2)]],
                         constant GemmParams& p [[buffer(3)]],
                         device const uchar* W1 [[buffer(5)]],
                         device const uchar* W2 [[buffer(6)]],
                         device float* Y1 [[buffer(7)]],
                         device float* Y2 [[buffer(8)]],
                         uint3 tgid [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint sgid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float stage[gemm_stage_floats<BM, BN, BK>()];
    device const uchar* W = tgid.z == 0 ? W0 : (tgid.z == 1 ? W1 : W2);
    device float* Y = tgid.z == 0 ? Y0 : (tgid.z == 1 ? Y1 : Y2);
    gemm_tile<Fmt, EpiStore, BM, BN, BK, SG>(X, W, Y, p, X, stage, tgid.xy, tid, sgid);
}

#define GEMM_VARIANT(NAME, FMT, EPI, BM, BN, BK, SG) \
    template [[host_name(NAME)]] kernel void gemm_kernel<FMT, EPI, BM, BN, BK, SG>( \
        device const float*, device const uchar*, device float*, constant GemmParams&, device const float*, uint2, uint, uint);

#define GEMM3_VARIANT(NAME, FMT, BM, BN, BK, SG) \
    template [[host_name(NAME)]] kernel void gemm3_kernel<FMT, BM, BN, BK, SG>( \
        device const float*, device const uchar*, device float*, constant GemmParams&, \
        device const uchar*, device const uchar*, device float*, device float*, uint3, uint, uint);

#define GEMM_TILE(SUFFIX, BM, BN, BK, SG) \
    GEMM_VARIANT("gemm_q8_" SUFFIX, FmtQ8, EpiStore, BM, BN, BK, SG) \
    GEMM_VARIANT("gemm_q8_geglu_" SUFFIX, FmtQ8, EpiGeglu, BM, BN, BK, SG) \
    GEMM_VARIANT("gemm_q8_gelutimes_" SUFFIX, FmtQ8, EpiGeluTimes, BM, BN, BK, SG) \
    GEMM_VARIANT("gemm_q8_gegluquick_" SUFFIX, FmtQ8, EpiGegluQuick, BM, BN, BK, SG) \
    GEMM_VARIANT("gemm_bf16_" SUFFIX, FmtBF16, EpiStore, BM, BN, BK, SG) \
    GEMM_VARIANT("gemm_f32_" SUFFIX, FmtF32, EpiStore, BM, BN, BK, SG) \
    GEMM3_VARIANT("gemm3_q8_" SUFFIX, FmtQ8, BM, BN, BK, SG)

GEMM_TILE("32x32x32s4", 32, 32, 32, 4)
GEMM_TILE("32x64x32s4", 32, 64, 32, 4)
GEMM_TILE("64x32x32s4", 64, 32, 32, 4)
GEMM_TILE("32x64x16s4", 32, 64, 16, 4)
GEMM_TILE("64x64x16s4", 64, 64, 16, 4)
GEMM_TILE("64x64x16s8", 64, 64, 16, 8)
GEMM_TILE("64x64x32s8", 64, 64, 32, 8)
GEMM_TILE("64x128x16s8", 64, 128, 16, 8)
GEMM_TILE("64x64x16s2", 64, 64, 16, 2)
GEMM_TILE("16x32x16s4", 16, 32, 16, 4)
GEMM_TILE("16x32x16s2", 16, 32, 16, 2)
GEMM_TILE("64x128x16s4", 64, 128, 16, 4)

GEMM_TILE("8x16x32s2", 8, 16, 32, 2)

// Each lane owns (fm, fn) and (fm, fn + 1) in every 8x8 fragment.
template <typename Epi>
kernel void gemm_q8_register_kernel(device const float* X [[buffer(0)]],
                                    device const uchar* W [[buffer(1)]],
                                    device float* Y [[buffer(2)]],
                                    constant GemmParams& p [[buffer(3)]],
                                    device const float* aux [[buffer(4)]],
                                    uint2 tgid [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint sgid [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    const uint fm = ((lane / 4) & 4) + ((lane / 2) % 4);
    const uint fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const uint m0 = tgid.y * 8;
    const uint n0 = tgid.x * 16 + sgid * 8;
    simdgroup_float8x8 acc(0.0f);

    for (uint k0 = 0; k0 < p.K; k0 += 32) {
        device const uchar* w0 = W + (n0 + fn) * p.weightRowBytes + (k0 / 32) * Q8_BLOCK_BYTES;
        device const uchar* w1 = w0 + p.weightRowBytes;
        const float scale0 = f16_to_float(ushort(w0[0]) | (ushort(w0[1]) << 8));
        const float scale1 = f16_to_float(ushort(w1[0]) | (ushort(w1[1]) << 8));
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_float8x8 a;
            simdgroup_float8x8 b;
            const float2 x = *(device const float2*)(X + (m0 + fm) * p.K + k0 + kk + fn);
            a.thread_elements()[0] = x.x;
            a.thread_elements()[1] = x.y;
            // B is W transposed: fragment row fm is K, column fn is an output row.
            b.thread_elements()[0] = scale0 * float(as_type<char>(w0[2 + kk + fm]));
            b.thread_elements()[1] = scale1 * float(as_type<char>(w1[2 + kk + fm]));
            simdgroup_multiply_accumulate(acc, a, b, acc);
        }
    }

    threadgroup float output[8 * 20];
    simdgroup_store(acc, output + sgid * 8, 20);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        const uint row = tid / 4, c4 = tid % 4;
        threadgroup const float* src = output + row * 20 + c4 * 4;
        const uint index = (m0 + row) * p.N + tgid.x * 16 + c4 * 4;
        const float4 v = float4(src[0], src[1], src[2], src[3]);
        *(device float4*)(Y + index) = Epi::apply(v, aux, index);
    }
}

#define GEMM_Q8_REGISTER_VARIANT(NAME, EPI) \
    template [[host_name(NAME)]] kernel void gemm_q8_register_kernel<EPI>( \
        device const float*, device const uchar*, device float*, constant GemmParams&, device const float*, uint2, uint, uint, uint);

GEMM_Q8_REGISTER_VARIANT("gemm_q8_register_8x16x32s2", EpiStore)
GEMM_Q8_REGISTER_VARIANT("gemm_q8_geglu_register_8x16x32s2", EpiGeglu)
GEMM_Q8_REGISTER_VARIANT("gemm_q8_gelutimes_register_8x16x32s2", EpiGeluTimes)

template <typename Epi, int BM, int BN, int SG>
kernel void gemm_q8_batch_register_kernel(device const float* X [[buffer(0)]],
                                          device const uchar* W [[buffer(1)]],
                                          device float* Y [[buffer(2)]],
                                          constant GemmParams& p [[buffer(3)]],
                                          device const float* aux [[buffer(4)]],
                                          uint2 tgid [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]],
                                          uint sgid [[simdgroup_index_in_threadgroup]],
                                          uint lane [[thread_index_in_simdgroup]]) {
    constexpr int SN = BN / SG, TM = BM / 8, TN = SN / 8;
    constexpr int OUT_LD = BN + 4, THREADS = SG * 32;
    const uint fm = ((lane / 4) & 4) + ((lane / 2) % 4);
    const uint fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const uint m0 = tgid.y * BM;
    const uint n0 = tgid.x * BN + sgid * SN;
    simdgroup_float8x8 acc[TM][TN];
    #pragma clang loop unroll(full)
    for (int i = 0; i < TM; i++)
        #pragma clang loop unroll(full)
        for (int j = 0; j < TN; j++) acc[i][j] = simdgroup_float8x8(0.0f);

    for (uint k0 = 0; k0 < p.K; k0 += 32) {
        float2 scales[TN];
        #pragma clang loop unroll(full)
        for (int j = 0; j < TN; j++) {
            device const uchar* w0 = W + (n0 + j * 8 + fn) * p.weightRowBytes + (k0 / 32) * Q8_BLOCK_BYTES;
            device const uchar* w1 = w0 + p.weightRowBytes;
            scales[j] = float2(f16_to_float(ushort(w0[0]) | (ushort(w0[1]) << 8)),
                               f16_to_float(ushort(w1[0]) | (ushort(w1[1]) << 8)));
        }
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_float8x8 b[TN];
            #pragma clang loop unroll(full)
            for (int j = 0; j < TN; j++) {
                device const uchar* w0 = W + (n0 + j * 8 + fn) * p.weightRowBytes + (k0 / 32) * Q8_BLOCK_BYTES;
                device const uchar* w1 = w0 + p.weightRowBytes;
                b[j].thread_elements()[0] = scales[j].x * float(as_type<char>(w0[2 + kk + fm]));
                b[j].thread_elements()[1] = scales[j].y * float(as_type<char>(w1[2 + kk + fm]));
            }
            #pragma clang loop unroll(full)
            for (int i = 0; i < TM; i++) {
                simdgroup_float8x8 a;
                const float2 x = *(device const float2*)(X + (m0 + i * 8 + fm) * p.K + k0 + kk + fn);
                a.thread_elements()[0] = x.x;
                a.thread_elements()[1] = x.y;
                #pragma clang loop unroll(full)
                for (int j = 0; j < TN; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a, b[j], acc[i][j]);
            }
        }
    }

    threadgroup float output[BM * OUT_LD];
    #pragma clang loop unroll(full)
    for (int i = 0; i < TM; i++)
        #pragma clang loop unroll(full)
        for (int j = 0; j < TN; j++)
            simdgroup_store(acc[i][j], output + i * 8 * OUT_LD + sgid * SN + j * 8, OUT_LD);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint u = tid; u < BM * BN / 4; u += THREADS) {
        const uint row = u / (BN / 4), c4 = u % (BN / 4);
        threadgroup const float* src = output + row * OUT_LD + c4 * 4;
        const uint index = (m0 + row) * p.N + tgid.x * BN + c4 * 4;
        const float4 v = float4(src[0], src[1], src[2], src[3]);
        *(device float4*)(Y + index) = Epi::apply(v, aux, index);
    }
}

#define GEMM_Q8_BATCH_REGISTER_VARIANT(NAME, EPI, BM, BN, SG) \
    template [[host_name(NAME)]] kernel void gemm_q8_batch_register_kernel<EPI, BM, BN, SG>( \
        device const float*, device const uchar*, device float*, constant GemmParams&, device const float*, uint2, uint, uint, uint);

#define GEMM_Q8_BATCH_REGISTER_TILE(SUFFIX, BM, BN, SG) \
    GEMM_Q8_BATCH_REGISTER_VARIANT("gemm_q8_batch_register_" SUFFIX, EpiStore, BM, BN, SG) \
    GEMM_Q8_BATCH_REGISTER_VARIANT("gemm_q8_geglu_batch_register_" SUFFIX, EpiGeglu, BM, BN, SG) \
    GEMM_Q8_BATCH_REGISTER_VARIANT("gemm_q8_gelutimes_batch_register_" SUFFIX, EpiGeluTimes, BM, BN, SG)

GEMM_Q8_BATCH_REGISTER_TILE("32x32x32s4", 32, 32, 4)
GEMM_Q8_BATCH_REGISTER_TILE("32x64x32s4", 32, 64, 4)
GEMM_Q8_BATCH_REGISTER_TILE("32x128x32s4", 32, 128, 4)

kernel void gemm_q8_dual_geglu_register_8x16x32s2(
    device const float* X [[buffer(0)]],
    device const uchar* Wgate [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant GemmParams& p [[buffer(3)]],
    device const uchar* Wup [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint fm = ((lane / 4) & 4) + ((lane / 2) % 4);
    const uint fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const uint m0 = tgid.y * 8;
    const uint n0 = tgid.x * 16 + sgid * 8;
    simdgroup_float8x8 gate(0.0f), up(0.0f);

    for (uint k0 = 0; k0 < p.K; k0 += 32) {
        const uint offset = (n0 + fn) * p.weightRowBytes + (k0 / 32) * Q8_BLOCK_BYTES;
        device const uchar* g0 = Wgate + offset;
        device const uchar* g1 = g0 + p.weightRowBytes;
        device const uchar* u0 = Wup + offset;
        device const uchar* u1 = u0 + p.weightRowBytes;
        const float gs0 = f16_to_float(ushort(g0[0]) | (ushort(g0[1]) << 8));
        const float gs1 = f16_to_float(ushort(g1[0]) | (ushort(g1[1]) << 8));
        const float us0 = f16_to_float(ushort(u0[0]) | (ushort(u0[1]) << 8));
        const float us1 = f16_to_float(ushort(u1[0]) | (ushort(u1[1]) << 8));
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_float8x8 a, g, u;
            const float2 x = *(device const float2*)(X + (m0 + fm) * p.K + k0 + kk + fn);
            a.thread_elements()[0] = x.x;
            a.thread_elements()[1] = x.y;
            g.thread_elements()[0] = gs0 * float(as_type<char>(g0[2 + kk + fm]));
            g.thread_elements()[1] = gs1 * float(as_type<char>(g1[2 + kk + fm]));
            u.thread_elements()[0] = us0 * float(as_type<char>(u0[2 + kk + fm]));
            u.thread_elements()[1] = us1 * float(as_type<char>(u1[2 + kk + fm]));
            simdgroup_multiply_accumulate(gate, a, g, gate);
            simdgroup_multiply_accumulate(up, a, u, up);
        }
    }

    constexpr uint OUT_LD = 20, OUT_SIZE = 8 * OUT_LD;
    threadgroup float output[2 * OUT_SIZE];
    simdgroup_store(gate, output + sgid * 8, OUT_LD);
    simdgroup_store(up, output + OUT_SIZE + sgid * 8, OUT_LD);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        const uint row = tid / 4, c4 = tid % 4;
        threadgroup const float* gs = output + row * OUT_LD + c4 * 4;
        threadgroup const float* us = gs + OUT_SIZE;
        const float4 g = float4(gs[0], gs[1], gs[2], gs[3]);
        const float4 u = float4(us[0], us[1], us[2], us[3]);
        const uint index = (m0 + row) * p.N + tgid.x * 16 + c4 * 4;
        *(device float4*)(Y + index) = float4(gelu_tanh(g.x), gelu_tanh(g.y), gelu_tanh(g.z), gelu_tanh(g.w)) * u;
    }
}
