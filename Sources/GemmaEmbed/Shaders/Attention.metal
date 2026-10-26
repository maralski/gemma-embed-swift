// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
// Bidirectional flash-style attention over packed variable-length sequences.
//
// Element ownership of a simdgroup_float8x8 on Apple GPUs (measured on M3 Max): lane l holds
// row ((l >> 1) & 3) | ((l >> 4) << 2), columns (l & 1) * 2 + ((l >> 3) & 1) * 4 and the next
// one, so the four lanes that share a row differ in lane bits 0 and 3 and a row reduction is
// simd_shuffle_xor by 1 and by 8.

constant int ATT_KEYS = 32;
constant int ATT_DC = 64;
constant int ATT_LD = ATT_DC + 4;
constant int ATT_TILE_FLOATS = ATT_KEYS * ATT_LD;

// Stages rows [row0, row0 + 32) x columns [col0, col0 + 64) of a [rows][stride] matrix.
inline void stage_tile(threadgroup float* tile, device const float* src, uint stride, uint tid, uint threads) {
    for (uint u = tid; u < ATT_KEYS * ATT_DC / 4; u += threads) {
        const uint r = u / (ATT_DC / 4), c4 = u % (ATT_DC / 4);
        const float4 v = *(device const float4*)(src + r * stride + c4 * 4);
        threadgroup float* dst = tile + r * ATT_LD + c4 * 4;
        dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
    }
}

template <int ATT_DS>
kernel void attention_kernel(device const float* Q [[buffer(0)]],
                             device const float* K [[buffer(1)]],
                             device const float* V [[buffer(2)]],
                             device float* O [[buffer(3)]],
                             device const AttentionWork* work [[buffer(4)]],
                             constant AttentionParams& p [[buffer(5)]],
                             threadgroup float* tile [[threadgroup(0)]],
                             uint3 tgid [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint sgid [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    const AttentionWork w = work[tgid.x];
    const uint h = tgid.y;
    const uint slice = tgid.z;
    const uint kh = h / (p.heads / p.kvHeads);
    const uint hd = p.headDim;
    const uint d0 = slice * ATT_DS;
    const uint len = w.seqLen;
    const uint simdgroups = p.queriesPerGroup / 8;
    const uint threads = simdgroups * 32;
    const uint qRow0 = w.q0 + sgid * 8;
    const uint hw = p.halfWindow;

    device const float* qBase = Q + (w.seqStart + qRow0) * p.qStride + h * hd;
    device const float* kBase = K + w.seqStart * p.kvStride + kh * hd;
    device const float* vBase = V + w.seqStart * p.kvStride + kh * hd + d0;

    // Every loop over o[] is fully unrolled so each index is a compile-time constant and the
    // accumulators stay in registers; a single runtime index puts the whole array in memory and
    // made the kernel about twice as slow.
    simdgroup_float8x8 o[ATT_DS / 8];
    #pragma clang loop unroll(full)
    for (int c = 0; c < ATT_DS / 8; c++) o[c] = simdgroup_float8x8(0.0f);

    const uint r = ((lane >> 1) & 3) | ((lane >> 4) << 2);
    const uint c0 = (lane & 1) * 2 + ((lane >> 3) & 1) * 4;
    const int qi = int(qRow0 + r);
    float m = -INFINITY;
    float l = 0.0f;

    // Key range for the whole query block, clipped to the window; every simdgroup walks it
    // because the K and V tiles are shared.
    uint kStart = 0, kEnd = len;
    if (hw > 0) {
        const int lo = int(w.q0) - int(hw);
        const int hi = int(w.q0) + int(p.queriesPerGroup) - 1 + int(hw) + 1;
        kStart = uint(max(lo, 0));
        kEnd = min(uint(max(hi, 0)), len);
    }
    kStart = kStart / ATT_KEYS * ATT_KEYS;

    for (uint kb = kStart; kb < kEnd; kb += ATT_KEYS) {
        simdgroup_float8x8 s[4];
        for (int j = 0; j < 4; j++) s[j] = simdgroup_float8x8(0.0f);
        for (uint dc = 0; dc < hd; dc += ATT_DC) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            stage_tile(tile, kBase + kb * p.kvStride + dc, p.kvStride, tid, threads);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int dd = 0; dd < ATT_DC; dd += 8) {
                simdgroup_float8x8 a;
                simdgroup_load(a, qBase + dc + dd, p.qStride);
                for (int j = 0; j < 4; j++) {
                    simdgroup_float8x8 b;
                    simdgroup_load(b, tile + j * 8 * ATT_LD + dd, ATT_LD, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(s[j], a, b, s[j]);
                }
            }
        }
        float bmax = -INFINITY;
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; j++) {
            #pragma clang loop unroll(full)
            for (int e = 0; e < 2; e++) {
                const int kj = int(kb) + j * 8 + int(c0) + e;
                bool valid = kj < int(len);
                if (hw > 0) valid = valid && abs(qi - kj) <= int(hw);
                const float v = valid ? s[j].thread_elements()[e] * p.scale : -INFINITY;
                s[j].thread_elements()[e] = v;
                bmax = max(bmax, v);
            }
        }
        bmax = max(bmax, simd_shuffle_xor(bmax, 1));
        bmax = max(bmax, simd_shuffle_xor(bmax, 8));
        const float mNew = max(m, bmax);
        float alpha = 1.0f;
        float sum = 0.0f;
        if (mNew == -INFINITY) {
            #pragma clang loop unroll(full)
            for (int j = 0; j < 4; j++) s[j] = simdgroup_float8x8(0.0f);
        } else {
            alpha = precise::exp(m - mNew);
            #pragma clang loop unroll(full)
            for (int j = 0; j < 4; j++) {
                #pragma clang loop unroll(full)
                for (int e = 0; e < 2; e++) {
                    const float v = precise::exp(s[j].thread_elements()[e] - mNew);
                    s[j].thread_elements()[e] = v;
                    sum += v;
                }
            }
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 8);
        l = l * alpha + sum;
        m = mNew;

        if (!simd_all(alpha == 1.0f)) {
            #pragma clang loop unroll(full)
            for (int c = 0; c < ATT_DS / 8; c++) {
                o[c].thread_elements()[0] *= alpha;
                o[c].thread_elements()[1] *= alpha;
            }
        }
        #pragma clang loop unroll(full)
        for (int dc = 0; dc < ATT_DS; dc += ATT_DC) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            stage_tile(tile, vBase + kb * p.kvStride + dc, p.kvStride, tid, threads);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (int j = 0; j < 4; j++) {
                #pragma clang loop unroll(full)
                for (int dd = 0; dd < ATT_DC; dd += 8) {
                    simdgroup_float8x8 vm;
                    simdgroup_load(vm, tile + j * 8 * ATT_LD + dd, ATT_LD);
                    const int c = (dc + dd) / 8;
                    simdgroup_multiply_accumulate(o[c], s[j], vm, o[c]);
                }
            }
        }
    }

    if (qRow0 + r >= len) return;
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    device float* oRow = O + (w.seqStart + qRow0 + r) * p.oStride + h * hd + d0 + c0;
    #pragma clang loop unroll(full)
    for (int c = 0; c < ATT_DS / 8; c++) {
        *(device float2*)(oRow + c * 8) = float2(o[c].thread_elements()[0], o[c].thread_elements()[1]) * inv;
    }
}

#define ATTENTION_VARIANT(NAME, DS) \
    template [[host_name(NAME)]] kernel void attention_kernel<DS>( \
        device const float*, device const float*, device const float*, device float*, \
        device const AttentionWork*, constant AttentionParams&, threadgroup float*, uint3, uint, uint, uint);

ATTENTION_VARIANT("attention_fwd", 256)
ATTENTION_VARIANT("attention_fwd_hd512", 512)
ATTENTION_VARIANT("attention_fwd_ds128", 128)

constant int VIS_QUERIES = 64;
constant int VIS_THREADS = VIS_QUERIES / 8 * 32;
constant int VIS_KT_LD = ATT_KEYS + 4;
constant int VIS_KT_FLOATS = ATT_DC * VIS_KT_LD;
constant int VIS_STAGE_FLOAT4S_PER_THREAD = ATT_KEYS * ATT_DC / 4 / VIS_THREADS;

kernel void attention_vision(device const float* Q [[buffer(0)]],
                             device const float* K [[buffer(1)]],
                             device const float* V [[buffer(2)]],
                             device float* O [[buffer(3)]],
                             device const AttentionWork* work [[buffer(4)]],
                             constant AttentionParams& p [[buffer(5)]],
                             threadgroup float* tile [[threadgroup(0)]],
                             uint3 tgid [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint sgid [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float* kTile = tile;
    threadgroup float* vTile = tile + VIS_KT_FLOATS;
    const AttentionWork w = work[tgid.x];
    const uint h = tgid.y;
    const uint len = w.seqLen;
    const uint qRow0 = w.q0 + sgid * 8;
    device const float* qBase = Q + (w.seqStart + qRow0) * p.qStride + h * ATT_DC;
    device const float* kBase = K + w.seqStart * p.kvStride + h * ATT_DC;
    device const float* vBase = V + w.seqStart * p.kvStride + h * ATT_DC;

    simdgroup_float8x8 q[ATT_DC / 8];
    simdgroup_float8x8 o[ATT_DC / 8];
    #pragma clang loop unroll(full)
    for (int c = 0; c < ATT_DC / 8; c++) {
        simdgroup_load(q[c], qBase + c * 8, p.qStride);
        o[c] = simdgroup_float8x8(0.0f);
    }
    const uint c0 = (lane & 1) * 2 + ((lane >> 3) & 1) * 4;
    float m = -INFINITY;
    float l = 0.0f;

    for (uint kb = 0; kb < len; kb += ATT_KEYS) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (int i = 0; i < VIS_STAGE_FLOAT4S_PER_THREAD; i++) {
            const uint u = tid + i * VIS_THREADS;
            const uint key = u / (ATT_DC / 4), d = u % (ATT_DC / 4) * 4;
            const float4 kv = *(device const float4*)(kBase + (kb + key) * p.kvStride + d);
            const float4 vv = *(device const float4*)(vBase + (kb + key) * p.kvStride + d);
            kTile[(d + 0) * VIS_KT_LD + key] = kv.x;
            kTile[(d + 1) * VIS_KT_LD + key] = kv.y;
            kTile[(d + 2) * VIS_KT_LD + key] = kv.z;
            kTile[(d + 3) * VIS_KT_LD + key] = kv.w;
            threadgroup float* vDst = vTile + key * ATT_LD + d;
            vDst[0] = vv.x; vDst[1] = vv.y; vDst[2] = vv.z; vDst[3] = vv.w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 s[ATT_KEYS / 8];
        #pragma clang loop unroll(full)
        for (int j = 0; j < ATT_KEYS / 8; j++) s[j] = simdgroup_float8x8(0.0f);
        #pragma clang loop unroll(full)
        for (int dd = 0; dd < ATT_DC / 8; dd++) {
            #pragma clang loop unroll(full)
            for (int j = 0; j < ATT_KEYS / 8; j++) {
                simdgroup_float8x8 b;
                simdgroup_load(b, kTile + dd * 8 * VIS_KT_LD + j * 8, VIS_KT_LD);
                simdgroup_multiply_accumulate(s[j], q[dd], b, s[j]);
            }
        }
        float bmax = -INFINITY;
        #pragma clang loop unroll(full)
        for (int j = 0; j < ATT_KEYS / 8; j++) {
            #pragma clang loop unroll(full)
            for (int e = 0; e < 2; e++) {
                const int kj = int(kb) + j * 8 + int(c0) + e;
                const float v = kj < int(len) ? s[j].thread_elements()[e] * p.scale : -INFINITY;
                s[j].thread_elements()[e] = v;
                bmax = max(bmax, v);
            }
        }
        bmax = max(bmax, simd_shuffle_xor(bmax, 1));
        bmax = max(bmax, simd_shuffle_xor(bmax, 8));
        const float mNew = max(m, bmax);
        const float alpha = precise::exp(m - mNew);
        float sum = 0.0f;
        #pragma clang loop unroll(full)
        for (int j = 0; j < ATT_KEYS / 8; j++) {
            #pragma clang loop unroll(full)
            for (int e = 0; e < 2; e++) {
                const float v = precise::exp(s[j].thread_elements()[e] - mNew);
                s[j].thread_elements()[e] = v;
                sum += v;
            }
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 8);
        l = l * alpha + sum;
        m = mNew;
        if (!simd_all(alpha == 1.0f)) {
            #pragma clang loop unroll(full)
            for (int c = 0; c < ATT_DC / 8; c++) {
                o[c].thread_elements()[0] *= alpha;
                o[c].thread_elements()[1] *= alpha;
            }
        }
        #pragma clang loop unroll(full)
        for (int j = 0; j < ATT_KEYS / 8; j++) {
            #pragma clang loop unroll(full)
            for (int dd = 0; dd < ATT_DC / 8; dd++) {
                simdgroup_float8x8 vm;
                simdgroup_load(vm, vTile + j * 8 * ATT_LD + dd * 8, ATT_LD);
                simdgroup_multiply_accumulate(o[dd], s[j], vm, o[dd]);
            }
        }
    }

    const uint r = ((lane >> 1) & 3) | ((lane >> 4) << 2);
    if (qRow0 + r >= len) return;
    const float inv = 1.0f / l;
    device float* oRow = O + (w.seqStart + qRow0 + r) * p.oStride + h * ATT_DC + c0;
    #pragma clang loop unroll(full)
    for (int c = 0; c < ATT_DC / 8; c++) {
        *(device float2*)(oRow + c * 8) = float2(o[c].thread_elements()[0], o[c].thread_elements()[1]) * inv;
    }
}
