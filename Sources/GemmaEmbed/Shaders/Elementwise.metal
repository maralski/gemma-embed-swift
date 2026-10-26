// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
// Token embedding gather, norms, RoPE, activations, residual updates and pooling.

// Floats a simdgroup covers when each of its 32 lanes moves one float4.
constant int SIMD_FLOAT4_SPAN = 32 * 4;

kernel void embed_q8(device const uchar* table [[buffer(0)]],
                     device const int* tokens [[buffer(1)]],
                     device float* out [[buffer(2)]],
                     constant EmbedParams& p [[buffer(3)]],
                     device const float* vectors [[buffer(4)]],
                     device float* residual [[buffer(5)]],
                     uint2 gid [[thread_position_in_grid]]) {
    const uint t = gid.y;
    const uint block = gid.x;
    if (t >= p.paddedRows || block * 32 >= p.width) return;
    const uint offset = t * p.width + block * 32;
    device float* o = out + offset;
    device float* r = residual + offset;
    if (t >= p.tokens) {
        for (uint j = 0; j < 32; j++) { o[j] = 0.0f; r[j] = 0.0f; }
        return;
    }
    const int id = tokens[t];
    if (id < 0) {
        device const float* v = vectors + uint(-(id + 1)) * p.width + block * 32;
        for (uint j = 0; j < 32; j++) { const float e = v[j]; o[j] = e; r[j] = e; }
        return;
    }
    const uint rowBytes = p.width / 32 * Q8_BLOCK_BYTES;
    device const uchar* b = table + uint(id) * rowBytes + block * Q8_BLOCK_BYTES;
    const float scale = f16_to_float(ushort(b[0]) | (ushort(b[1]) << 8)) * p.scale;
    for (uint j = 0; j < 32; j++) { const float e = scale * float(as_type<char>(b[2 + j])); o[j] = e; r[j] = e; }
}

// RMS norm of each row: out = x * preScale / sqrt(mean((x * preScale)^2) + eps) * w.
kernel void rmsnorm_rows(device const float* x [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         device const float* weight [[buffer(2)]],
                         constant RowParams& p [[buffer(3)]],
                         uint tid [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tid / 32;
    if (row >= p.rows) return;
    device const float* xr = x + row * p.inStride;
    device float* o = out + row * p.outStride;
    float ss = 0.0f;
    if (p.width % SIMD_FLOAT4_SPAN == 0) {
        for (uint i = lane * 4; i < p.width; i += SIMD_FLOAT4_SPAN) {
            const float4 v = *(device const float4*)(xr + i) * p.preScale;
            ss += v.x * v.x;
            ss += v.y * v.y;
            ss += v.z * v.z;
            ss += v.w * v.w;
        }
        ss = simd_sum(ss);
        const float inv = 1.0f / precise::sqrt(ss / float(p.width) + p.eps);
        for (uint i = lane * 4; i < p.width; i += SIMD_FLOAT4_SPAN) {
            const float4 v = *(device const float4*)(xr + i) * p.preScale * inv;
            *(device float4*)(o + i) = p.hasWeight ? v * *(device const float4*)(weight + i) : v;
        }
        return;
    }
    for (uint i = lane; i < p.width; i += 32) {
        const float v = xr[i] * p.preScale;
        ss += v * v;
    }
    ss = simd_sum(ss);
    const float inv = 1.0f / precise::sqrt(ss / float(p.width) + p.eps);
    for (uint i = lane; i < p.width; i += 32) {
        const float v = xr[i] * p.preScale * inv;
        o[i] = p.hasWeight ? v * weight[i] : v;
    }
}

// NeoX RoPE in place on [tokens][heads][headDim] with a precomputed cos/sin table
// indexed by [position][pair] (two floats per entry).
kernel void rope_neox(device float* x [[buffer(0)]],
                      device const int* positions [[buffer(1)]],
                      device const float* table [[buffer(2)]],
                      constant RopeParams& p [[buffer(3)]],
                      uint3 gid [[thread_position_in_grid]]) {
    const uint pair = gid.x;
    const uint head = gid.y;
    const uint t = gid.z;
    const uint halfDim = p.headDim / 2;
    if (pair >= halfDim || head >= p.heads || t >= p.tokens) return;
    device float* v = x + t * p.stride + head * p.headDim;
    device const float* cs = table + (uint(positions[t]) * p.tableStride + pair) * 2;
    const float c = cs[0], s = cs[1];
    const float a = v[pair], b = v[pair + halfDim];
    v[pair] = a * c - b * s;
    v[pair + halfDim] = a * s + b * c;
}

template <int W>
kernel void rmsnorm_rope_kernel(device float* x [[buffer(0)]],
                                device const float* weight [[buffer(1)]],
                                device const int* positions [[buffer(2)]],
                                device const float* table [[buffer(3)]],
                                constant RowParams& n [[buffer(4)]],
                                constant RopeParams& p [[buffer(5)]],
                                uint tid [[thread_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
    constexpr int CHUNKS = W / SIMD_FLOAT4_SPAN;
    constexpr int HALF = W / 2;
    static_assert(HALF % SIMD_FLOAT4_SPAN == 0, "a pair's two values must sit in the same lane");
    const uint row = tid / 32;
    if (row >= n.rows) return;
    device float* xr = x + row * W;
    float4 v[CHUNKS];
    float ss = 0.0f;
    for (int j = 0; j < CHUNKS; j++) {
        v[j] = *(device const float4*)(xr + lane * 4 + SIMD_FLOAT4_SPAN * j);
        const float4 u = v[j] * n.preScale;
        ss += u.x * u.x;
        ss += u.y * u.y;
        ss += u.z * u.z;
        ss += u.w * u.w;
    }
    ss = simd_sum(ss);
    const float inv = 1.0f / precise::sqrt(ss / float(W) + n.eps);
    device const float* cs = table + uint(positions[row / p.heads]) * p.tableStride * 2;
    for (int j = 0; j < CHUNKS / 2; j++) {
        const uint pair = lane * 4 + SIMD_FLOAT4_SPAN * j;
        const float4 a = v[j] * n.preScale * inv * *(device const float4*)(weight + pair);
        const float4 b = v[j + CHUNKS / 2] * n.preScale * inv * *(device const float4*)(weight + pair + HALF);
        const float4 cs0 = *(device const float4*)(cs + pair * 2);
        const float4 cs1 = *(device const float4*)(cs + pair * 2 + 4);
        const float4 c = float4(cs0.x, cs0.z, cs1.x, cs1.z);
        const float4 s = float4(cs0.y, cs0.w, cs1.y, cs1.w);
        *(device float4*)(xr + pair) = a * c - b * s;
        *(device float4*)(xr + pair + HALF) = a * s + b * c;
    }
}

template [[host_name("rmsnorm_rope_256")]] kernel void rmsnorm_rope_kernel<256>(
    device float*, device const float*, device const int*, device const float*, constant RowParams&, constant RopeParams&, uint, uint);
template [[host_name("rmsnorm_rope_512")]] kernel void rmsnorm_rope_kernel<512>(
    device float*, device const float*, device const int*, device const float*, constant RowParams&, constant RopeParams&, uint, uint);

// 2D NeoX RoPE in place on [tokens][heads][headDim]: the first half of every head rotates by the
// patch column px, the second half by the patch row py, each as a NeoX rotation of headDim / 2
// dims. The table holds headDim / 4 pairs per position.
kernel void rope_2d(device float* x [[buffer(0)]],
                    device const int* px [[buffer(1)]],
                    device const int* py [[buffer(2)]],
                    device const float* table [[buffer(3)]],
                    constant RopeParams& p [[buffer(4)]],
                    uint3 gid [[thread_position_in_grid]]) {
    const uint quarter = p.headDim / 4;
    const uint pair = gid.x % quarter;
    const uint axis = gid.x / quarter;
    const uint head = gid.y;
    const uint t = gid.z;
    if (axis >= 2 || head >= p.heads || t >= p.tokens) return;
    device float* v = x + t * p.stride + head * p.headDim + axis * (p.headDim / 2);
    const int pos = axis == 0 ? px[t] : py[t];
    device const float* cs = table + (uint(pos) * p.tableStride + pair) * 2;
    const float c = cs[0], s = cs[1];
    const float a = v[pair], b = v[pair + quarter];
    v[pair] = a * c - b * s;
    v[pair + quarter] = a * s + b * c;
}

kernel void rmsnorm_rope_2d_64(device float* x [[buffer(0)]],
                               device const float* weight [[buffer(1)]],
                               device const int* px [[buffer(2)]],
                               device const int* py [[buffer(3)]],
                               device const float* table [[buffer(4)]],
                               constant RowParams& n [[buffer(5)]],
                               constant RopeParams& p [[buffer(6)]],
                               uint tid [[thread_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint W = 64, QUARTER = W / 4;
    const uint row = tid / 32;
    if (row >= n.rows) return;
    device float* xr = x + row * W;
    const float x0 = xr[lane], x1 = xr[lane + 32];
    float ss = 0.0f;
    const float v0 = x0 * n.preScale;
    ss += v0 * v0;
    const float v1 = x1 * n.preScale;
    ss += v1 * v1;
    ss = simd_sum(ss);
    const float inv = 1.0f / precise::sqrt(ss / float(W) + n.eps);
    const float u0 = x0 * n.preScale * inv * weight[lane];
    const float u1 = x1 * n.preScale * inv * weight[lane + 32];
    const float w0 = simd_shuffle_xor(u0, QUARTER);
    const float w1 = simd_shuffle_xor(u1, QUARTER);
    const uint t = row / p.heads;
    const uint pair = lane % QUARTER;
    const float2 c0 = *(device const float2*)(table + (uint(px[t]) * p.tableStride + pair) * 2);
    const float2 c1 = *(device const float2*)(table + (uint(py[t]) * p.tableStride + pair) * 2);
    if (lane < QUARTER) {
        xr[lane] = u0 * c0.x - w0 * c0.y;
        xr[lane + 32] = u1 * c1.x - w1 * c1.y;
    } else {
        xr[lane] = w0 * c0.y + u0 * c0.x;
        xr[lane + 32] = w1 * c1.y + u1 * c1.x;
    }
}

// h[t] = (h[t] + table[px[t]]) + table[positionCount + py[t]]: the learned column and row
// embeddings of every patch, added in the CPU reference's order.
kernel void add_position_rows(device float* h [[buffer(0)]],
                              device const float* table [[buffer(1)]],
                              device const int* px [[buffer(2)]],
                              device const int* py [[buffer(3)]],
                              constant PositionParams& p [[buffer(4)]],
                              uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x;
    const uint t = gid.y;
    if (i >= p.width || t >= p.rows) return;
    device float* row = h + t * p.width;
    row[i] = (row[i] + table[uint(px[t]) * p.width + i]) + table[(p.positionCount + uint(py[t])) * p.width + i];
}

// Merges every merge x merge window of an image's patch grid into one token row:
// pooled[tokenStart + oy * gx / merge + ox] = sum of the window's patch rows * scale, where scale
// is sqrt(width) / merge^2. Grid z selects the image, y the token, x the channel.
kernel void pool_patches(device const float* x [[buffer(0)]],
                         device const PoolGrid* grids [[buffer(1)]],
                         device float* pooled [[buffer(2)]],
                         constant PoolPatchParams& p [[buffer(3)]],
                         uint3 gid [[thread_position_in_grid]]) {
    const PoolGrid g = grids[gid.z];
    const uint m = p.merge;
    const uint tx = g.gridX / m;
    const uint token = gid.y;
    const uint i = gid.x;
    if (i >= p.width || token >= tx * (g.gridY / m)) return;
    const uint ox = token % tx, oy = token / tx;
    float acc = 0.0f;
    for (uint ky = 0; ky < m; ky++)
        for (uint kx = 0; kx < m; kx++)
            acc += x[(g.patchStart + (oy * m + ky) * g.gridX + ox * m + kx) * p.width + i];
    pooled[(g.tokenStart + token) * p.width + i] = acc * p.scale;
}

// pooled[s] = mean over the sequence's rows of x. One threadgroup per sequence.
kernel void mean_pool(device const float* x [[buffer(0)]],
                      device const Sequence* seqs [[buffer(1)]],
                      device float* pooled [[buffer(2)]],
                      constant PoolParams& p [[buffer(3)]],
                      uint s [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint tgSize [[threads_per_threadgroup]]) {
    const Sequence seq = seqs[s];
    for (uint c = tid; c < p.width; c += tgSize) {
        float acc = 0.0f;
        for (uint r = 0; r < seq.length; r++) acc += x[(seq.start + r) * p.stride + c];
        pooled[s * p.width + c] = acc / float(seq.length);
    }
}

// x = (x + rmsnorm(t) * w) * scale, one simdgroup per row. Fuses the post-norm of a block
// output with its residual add.
kernel void rmsnorm_residual(device float* x [[buffer(0)]],
                             device const float* t [[buffer(1)]],
                             device const float* weight [[buffer(2)]],
                             constant RowParams& p [[buffer(3)]],
                             uint tid [[thread_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tid / 32;
    if (row >= p.rows) return;
    device const float* tr = t + row * p.inStride;
    device float* xr = x + row * p.outStride;
    float ss = 0.0f;
    if (p.width % SIMD_FLOAT4_SPAN == 0) {
        for (uint i = lane * 4; i < p.width; i += SIMD_FLOAT4_SPAN) {
            const float4 v = *(device const float4*)(tr + i);
            ss += v.x * v.x;
            ss += v.y * v.y;
            ss += v.z * v.z;
            ss += v.w * v.w;
        }
        ss = simd_sum(ss);
        const float inv = 1.0f / precise::sqrt(ss / float(p.width) + p.eps);
        for (uint i = lane * 4; i < p.width; i += SIMD_FLOAT4_SPAN) {
            device float4* xo = (device float4*)(xr + i);
            *xo = (*xo + *(device const float4*)(tr + i) * inv * *(device const float4*)(weight + i)) * p.preScale;
        }
        return;
    }
    for (uint i = lane; i < p.width; i += 32) {
        const float v = tr[i];
        ss += v * v;
    }
    ss = simd_sum(ss);
    const float inv = 1.0f / precise::sqrt(ss / float(p.width) + p.eps);
    for (uint i = lane; i < p.width; i += 32) {
        xr[i] = (xr[i] + tr[i] * inv * weight[i]) * p.preScale;
    }
}


template <int W>
kernel void rmsnorm_residual_norm_kernel(device float* x [[buffer(0)]],
                                         device const float* t [[buffer(1)]],
                                         device const float* weight [[buffer(2)]],
                                         constant RowParams& r [[buffer(3)]],
                                         device float* out [[buffer(4)]],
                                         device const float* outWeight [[buffer(5)]],
                                         constant RowParams& n [[buffer(6)]],
                                         uint tid [[thread_position_in_grid]],
                                         uint lane [[thread_index_in_simdgroup]]) {
    constexpr int CHUNKS = W / SIMD_FLOAT4_SPAN;
    static_assert(W % SIMD_FLOAT4_SPAN == 0, "rows must take the float4 paths");
    const uint row = tid / 32;
    if (row >= r.rows) return;
    device const float* tr = t + row * W;
    device float* xr = x + row * W;
    device float* o = out + row * W;
    float4 tv[CHUNKS];
    float ss = 0.0f;
#pragma clang loop unroll(full)
    for (int j = 0; j < CHUNKS; j++) {
        tv[j] = *(device const float4*)(tr + lane * 4 + SIMD_FLOAT4_SPAN * j);
        ss += tv[j].x * tv[j].x;
        ss += tv[j].y * tv[j].y;
        ss += tv[j].z * tv[j].z;
        ss += tv[j].w * tv[j].w;
    }
    ss = simd_sum(ss);
    const float inv = 1.0f / precise::sqrt(ss / float(W) + r.eps);
    float4 xv[CHUNKS];
    float ss2 = 0.0f;
#pragma clang loop unroll(full)
    for (int j = 0; j < CHUNKS; j++) {
        const uint i = lane * 4 + SIMD_FLOAT4_SPAN * j;
        device float4* xo = (device float4*)(xr + i);
        xv[j] = (*xo + tv[j] * inv * *(device const float4*)(weight + i)) * r.preScale;
        *xo = xv[j];
        const float4 v = xv[j] * n.preScale;
        ss2 += v.x * v.x;
        ss2 += v.y * v.y;
        ss2 += v.z * v.z;
        ss2 += v.w * v.w;
    }
    ss2 = simd_sum(ss2);
    const float inv2 = 1.0f / precise::sqrt(ss2 / float(W) + n.eps);
#pragma clang loop unroll(full)
    for (int j = 0; j < CHUNKS; j++) {
        const uint i = lane * 4 + SIMD_FLOAT4_SPAN * j;
        *(device float4*)(o + i) = xv[j] * n.preScale * inv2 * *(device const float4*)(outWeight + i);
    }
}

template [[host_name("rmsnorm_residual_norm_768")]] kernel void rmsnorm_residual_norm_kernel<768>(
    device float*, device const float*, device const float*, constant RowParams&, device float*, device const float*, constant RowParams&, uint, uint);
template [[host_name("rmsnorm_residual_norm_512")]] kernel void rmsnorm_residual_norm_kernel<512>(
    device float*, device const float*, device const float*, constant RowParams&, device float*, device const float*, constant RowParams&, uint, uint);
