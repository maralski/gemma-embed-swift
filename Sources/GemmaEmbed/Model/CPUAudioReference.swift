// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Accelerate
import Foundation

/// A plain fp32 forward pass of the gemma4a audio tower on the CPU, the oracle for the Metal
/// audio encoder. It follows llama.cpp's `clip_graph_gemma4a::build` node by node and, like
/// `CPUReference`, favours obviousness over speed.
final class CPUAudioReference {
    let config: AudioConfig
    let weights: AudioWeights
    private var cache: [String: [Float]] = [:]
    private let lock = NSLock()

    init(config: AudioConfig, weights: AudioWeights) {
        self.config = config
        self.weights = weights
    }

    /// Dequantized copy of a tensor, cached by name.
    func matrix(_ t: WeightTensor) -> [Float] {
        lock.lock()
        defer { lock.unlock() }
        if let m = cache[t.info.name] { return m }
        let m = t.dequantized()
        cache[t.info.name] = m
        return m
    }

    /// Dequantized and transposed copy `[columns][rows]` of a matrix, cached by name.
    func transposed(_ t: WeightTensor) -> [Float] {
        let key = t.info.name + "^T"
        lock.lock()
        defer { lock.unlock() }
        if let m = cache[key] { return m }
        let out = CPUReference.transposeMatrix(t.dequantized(), rows: t.rows, cols: t.columns)
        cache[key] = out
        return out
    }

    static func silu(_ x: Float) -> Float { x / (1 + expf(-x)) }
    static func sigmoid(_ x: Float) -> Float { 1 / (1 + expf(-x)) }
    static func roundToHalf(_ x: Float) -> Float { Float(Float16(x)) }

    static func clamp(_ x: [Float], _ lo: Float, _ hi: Float) -> [Float] {
        if lo == -.greatestFiniteMagnitude && hi == .greatestFiniteMagnitude { return x }
        return x.map { min(max($0, lo), hi) }
    }

    /// `clamp(x) @ W^T (+ bias)`, clamped: llama.cpp's `build_mm` over a `Gemma4ClippableLinear`.
    func linear(_ x: [Float], rows: Int, _ l: ClippedLinear) -> [Float] {
        let input = CPUAudioReference.clamp(x, l.inputMin, l.inputMax)
        var y = CPUReference.matmul(input, rows: rows, cols: l.weight.columns, transposedWeight: transposed(l.weight), outputs: l.weight.rows)
        y = CPUAudioReference.clamp(y, l.outputMin, l.outputMax)
        if let bias = l.bias {
            let b = matrix(bias)
            let n = l.weight.rows
            for r in 0..<rows { for i in 0..<n { y[r * n + i] += b[i] } }
        }
        return y
    }

    /// ggml's `conv_2d` with kernel 3, stride 2 and padding 1 over `x` laid out `[time][freq][channel]`,
    /// followed by `ggml_norm` over the channels, the norm weight and ReLU, in the same layout.
    /// ggml runs the im2col in fp16, so the input is rounded to fp16 first.
    static func subsample(_ x: [Float], time: Int, freq: Int, channels: Int, weight: [Float], outChannels: Int,
                          norm: [Float], eps: Float) -> (values: [Float], time: Int, freq: Int) {
        let outTime = (time - 1) / 2 + 1, outFreq = (freq - 1) / 2 + 1
        let taps = 9 * channels
        var patch = [Float](repeating: 0, count: taps)
        var out = [Float](repeating: 0, count: outTime * outFreq * outChannels)
        let wT = CPUReference.transposeMatrix(weight, rows: outChannels, cols: taps)
        var column = [Float](repeating: 0, count: outChannels)
        for ot in 0..<outTime {
            for of in 0..<outFreq {
                for kh in 0..<3 {
                    let t = 2 * ot + kh - 1
                    for kw in 0..<3 {
                        let f = 2 * of + kw - 1
                        for c in 0..<channels {
                            let inside = t >= 0 && t < time && f >= 0 && f < freq
                            patch[c * 9 + kh * 3 + kw] = inside ? roundToHalf(x[(t * freq + f) * channels + c]) : 0
                        }
                    }
                }
                column = CPUReference.matmul(patch, rows: 1, cols: taps, transposedWeight: wT, outputs: outChannels)
                var mean: Float = 0
                for v in column { mean += v }
                mean /= Float(outChannels)
                var variance: Float = 0
                for v in column { variance += (v - mean) * (v - mean) }
                variance /= Float(outChannels)
                let scale = 1 / (variance + eps).squareRoot()
                let o = (ot * outFreq + of) * outChannels
                for c in 0..<outChannels { out[o + c] = max(0, (column[c] - mean) * scale * norm[c]) }
            }
        }
        return (out, outTime, outFreq)
    }

    /// `pos_emb` as llama.cpp fills it: row `r` is the sinusoid of position `relativePositions - 1 - r`,
    /// sines over the first half of the width and cosines over the second.
    static func sinusoid(position: Int, width: Int) -> [Float] {
        let timescales = width / 2
        let increment = logf(10000) / Float(max(timescales - 1, 1))
        var out = [Float](repeating: 0, count: width)
        for i in 0..<timescales {
            let scaled = Float(position) * expf(-Float(i) * increment)
            out[i] = sinf(scaled)
            out[i + timescales] = cosf(scaled)
        }
        return out
    }

    /// The projected relative-position keys of a layer, `[distance][hidden]` for distances 0 to 12.
    func relativeKeys(_ l: AudioLayerWeights) -> [Float] {
        let d = config.info.hiddenSize
        let table = (0..<config.relativePositions).flatMap { CPUAudioReference.sinusoid(position: $0, width: d) }
        return linear(table, rows: config.relativePositions, l.kRel)
    }

    /// Local causal attention: query `t` sees keys `t - 11 ... t` of its chunk, each scored by content
    /// and by the query against the key of their distance, soft-capped at 50 (`50 tanh(s / 50)`).
    /// `q`, `k`, `v` are `[tokens][heads][headDim]`; returns the same layout.
    static func attention(q: [Float], k: [Float], v: [Float], relativeKeys: [Float], tokens: Int, heads: Int, headDim hd: Int,
                          window: Int, softcap: Float) -> [Float] {
        var out = [Float](repeating: 0, count: tokens * heads * hd)
        let width = heads * hd
        var scores = [Float](repeating: 0, count: window)
        for h in 0..<heads {
            for t in 0..<tokens {
                let first = max(0, t - window + 1)
                let qo = t * width + h * hd
                var maxScore = -Float.infinity
                for j in first...t {
                    let ko = j * width + h * hd
                    let ro = (t - j) * width + h * hd
                    var s: Float = 0
                    for i in 0..<hd { s += q[qo + i] * k[ko + i] }
                    var r: Float = 0
                    for i in 0..<hd { r += q[qo + i] * relativeKeys[ro + i] }
                    let capped = softcap * tanhf((s + r) / softcap)
                    scores[j - first] = capped
                    maxScore = max(maxScore, capped)
                }
                var sum: Float = 0
                for j in first...t {
                    let e = expf(scores[j - first] - maxScore)
                    scores[j - first] = e
                    sum += e
                }
                for j in first...t {
                    let p = scores[j - first] / sum
                    let vo = j * width + h * hd
                    for i in 0..<hd { out[qo + i] += p * v[vo + i] }
                }
            }
        }
        return out
    }

    /// `ggml_ssm_conv` after a left pad of 4: `y[t][c] = sum_k x[t + k - 4][c] * w[c][k]`.
    static func depthwiseCausal(_ x: [Float], tokens: Int, width: Int, taps w: [Float], kernel: Int) -> [Float] {
        var out = [Float](repeating: 0, count: tokens * width)
        for t in 0..<tokens {
            for c in 0..<width {
                var s: Float = 0
                for k in 0..<kernel {
                    let src = t + k - (kernel - 1)
                    if src >= 0 { s += x[src * width + c] * w[c * kernel + k] }
                }
                out[t * width + c] = s
            }
        }
        return out
    }

    /// Runs the audio tower over one chunk and returns `chunk.tokenCount` rows of `projectionDim`
    /// values, the rows that stand in for the chunk in the text sequence. `trace` receives named
    /// intermediate tensors (see `CPUAudioReferenceTests` for the node each name maps to).
    func forward(_ chunk: MelChunk, trace: ((String, [Float]) -> Void)? = nil) -> [Float] {
        let d = config.info.hiddenSize
        let eps = config.normEpsilon
        let heads = config.info.headCount, hd = config.headDim
        precondition(chunk.melBins == config.info.melBins && chunk.values.count == chunk.frames * chunk.melBins)
        trace?("mel", chunk.values)

        // Subsampling: [time][freq][1] -> [time/2][freq/2][128] -> [time/4][freq/4][32].
        var act = chunk.values
        var time = chunk.frames, freq = chunk.melBins, channels = 1
        for (i, conv) in weights.subsampling.enumerated() {
            let outChannels = config.subsamplingChannels[i]
            (act, time, freq) = CPUAudioReference.subsample(act, time: time, freq: freq, channels: channels, weight: matrix(conv.weight),
                                                            outChannels: outChannels, norm: matrix(conv.norm), eps: eps)
            channels = outChannels
            trace?("subsample-\(i)", act)
        }
        let T = time
        precondition(T == chunk.tokenCount && freq * channels == config.subsamplingOutputWidth)
        // The flattened row is `freq * channels + channel`, which is the [time][freq][channel] layout already.
        var x = linear(act, rows: T, weights.inputProjection)
        trace?("encoder_in", x)

        for (l, w) in weights.layers.enumerated() {
            var h = CPUReference.rmsNorm(x, width: d, weight: matrix(w.ffnNorm), eps: eps)
            trace?("ffn1_normed-\(l)", h)
            var up = linear(h, rows: T, w.ffnUp)
            trace?("ffn1_up-\(l)", up)
            for i in 0..<up.count { up[i] = CPUAudioReference.silu(up[i]) }
            var down = linear(up, rows: T, w.ffnDown)
            trace?("ffn1_down-\(l)", down)
            down = CPUReference.rmsNorm(down, width: d, weight: matrix(w.ffnPostNorm), eps: eps)
            trace?("ffn1_post_normed-\(l)", down)
            for i in 0..<x.count { x[i] += config.feedForwardResidualScale * down[i] }
            trace?("after_ffn1-\(l)", x)

            h = CPUReference.rmsNorm(x, width: d, weight: matrix(w.attnPreNorm), eps: eps)
            trace?("attn_normed-\(l)", h)
            var q = linear(h, rows: T, w.wq)
            var k = linear(h, rows: T, w.wk)
            let v = linear(h, rows: T, w.wv)
            trace?("v-\(l)", v)
            let perDim = matrix(w.perDimScale)
            for t in 0..<T {
                for hh in 0..<heads {
                    for i in 0..<hd { q[(t * heads + hh) * hd + i] *= config.queryScale * perDim[i] }
                }
            }
            for i in 0..<k.count { k[i] *= config.keyScale }
            trace?("q_scaled-\(l)", q)
            trace?("k_scaled-\(l)", k)
            let rel = relativeKeys(w)
            trace?("k_rel-\(l)", rel)
            let attn = CPUAudioReference.attention(q: q, k: k, v: v, relativeKeys: rel, tokens: T, heads: heads, headDim: hd,
                                                   window: config.attentionWindow, softcap: config.softcap)
            trace?("attn-\(l)", attn)
            var o = linear(attn, rows: T, w.wo)
            trace?("attn_out-\(l)", o)
            o = CPUReference.rmsNorm(o, width: d, weight: matrix(w.attnPostNorm), eps: eps)
            for i in 0..<x.count { x[i] += o[i] }
            trace?("after_attn-\(l)", x)

            h = CPUReference.rmsNorm(x, width: d, weight: matrix(w.convPreNorm), eps: eps)
            trace?("conv_normed-\(l)", h)
            let pw1 = linear(h, rows: T, w.convPointwise1)
            trace?("conv_pw1-\(l)", pw1)
            var glu = [Float](repeating: 0, count: T * d)
            for t in 0..<T {
                for i in 0..<d { glu[t * d + i] = pw1[t * 2 * d + i] * CPUAudioReference.sigmoid(pw1[t * 2 * d + d + i]) }
            }
            trace?("conv_glu-\(l)", glu)
            var dw = CPUAudioReference.depthwiseCausal(glu, tokens: T, width: d, taps: matrix(w.convDepthwise), kernel: config.depthwiseKernel)
            trace?("conv_dw-\(l)", dw)
            dw = CPUReference.rmsNorm(dw, width: d, weight: matrix(w.convDepthwiseNorm), eps: eps)
            for i in 0..<dw.count { dw[i] = CPUAudioReference.silu(dw[i]) }
            trace?("conv_silu-\(l)", dw)
            let pw2 = linear(dw, rows: T, w.convPointwise2)
            trace?("conv_pw2-\(l)", pw2)
            for i in 0..<x.count { x[i] += pw2[i] }
            trace?("after_conv-\(l)", x)

            h = CPUReference.rmsNorm(x, width: d, weight: matrix(w.ffnNorm1), eps: eps)
            trace?("ffn2_normed-\(l)", h)
            up = linear(h, rows: T, w.ffnUp1)
            trace?("ffn2_up-\(l)", up)
            for i in 0..<up.count { up[i] = CPUAudioReference.silu(up[i]) }
            down = linear(up, rows: T, w.ffnDown1)
            trace?("ffn2_down-\(l)", down)
            down = CPUReference.rmsNorm(down, width: d, weight: matrix(w.ffnPostNorm1), eps: eps)
            for i in 0..<x.count { x[i] += config.feedForwardResidualScale * down[i] }
            trace?("after_ffn2-\(l)", x)

            x = CPUReference.rmsNorm(x, width: d, weight: matrix(w.outputNorm), eps: eps)
            trace?("layer_out-\(l)", x)
        }

        let out = linear(x, rows: T, weights.outputProjection)
        trace?("out_proj", out)
        let normed = CPUReference.rmsNorm(out, width: weights.outputProjection.weight.rows, weight: nil, eps: eps)
        let projected = linear(normed, rows: T, weights.projection)
        trace?("projected", projected)
        return projected
    }
}
