import Accelerate
import Foundation

/// A plain fp32 forward pass on the CPU. It exists to validate the Metal kernels and the
/// whole GPU forward, so it favours obviousness over speed; only the matrix products use BLAS.
final class CPUReference {
    let config: ModelConfig
    let weights: ModelWeights
    private var cache: [String: [Float]] = [:]
    private let lock = NSLock()

    init(config: ModelConfig, weights: ModelWeights) {
        self.config = config
        self.weights = weights
    }

    private var ropeTables: [Int: RopeTable] = [:]

    func ropeTable(for layer: LayerConfig) -> RopeTable {
        lock.lock()
        defer { lock.unlock() }
        if let t = ropeTables[layer.headDim] { return t }
        let t = RopeTable(headDim: layer.headDim, base: layer.ropeBase, positions: config.info.maxTokens)
        ropeTables[layer.headDim] = t
        return t
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

    static func transposeMatrix(_ m: [Float], rows: Int, cols: Int) -> [Float] {
        var out = [Float](repeating: 0, count: m.count)
        m.withUnsafeBufferPointer { mp in
            out.withUnsafeMutableBufferPointer { op in
                vDSP_mtrans(mp.baseAddress!, 1, op.baseAddress!, 1, vDSP_Length(cols), vDSP_Length(rows))
            }
        }
        return out
    }

    /// Dequantized and transposed copy `[columns][rows]` of a matrix, cached by name.
    func transposed(_ t: WeightTensor) -> [Float] {
        let key = t.info.name + "^T"
        lock.lock()
        defer { lock.unlock() }
        if let m = cache[key] { return m }
        let m = cache[t.info.name] ?? t.dequantized()
        let out = CPUReference.transposeMatrix(m, rows: t.rows, cols: t.columns)
        cache[key] = out
        return out
    }

    /// `y[M,N] = x[M,K] * wT[K,N]`, where `wT` is the transposed weight matrix.
    static func matmul(_ x: [Float], rows m: Int, cols k: Int, transposedWeight wT: [Float], outputs n: Int) -> [Float] {
        precondition(x.count == m * k && wT.count == n * k)
        var y = [Float](repeating: 0, count: m * n)
        x.withUnsafeBufferPointer { xp in
            wT.withUnsafeBufferPointer { wp in
                y.withUnsafeMutableBufferPointer { yp in
                    vDSP_mmul(xp.baseAddress!, 1, wp.baseAddress!, 1, yp.baseAddress!, 1, vDSP_Length(m), vDSP_Length(n), vDSP_Length(k))
                }
            }
        }
        return y
    }

    /// RMS norm of every `width`-wide row of `x`, scaled by `weight` when given.
    static func rmsNorm(_ x: [Float], width: Int, weight: [Float]?, eps: Float) -> [Float] {
        precondition(x.count % width == 0)
        var out = x
        let rows = x.count / width
        for r in 0..<rows {
            var ss: Float = 0
            for i in 0..<width { let v = x[r * width + i]; ss += v * v }
            let inv = 1 / (ss / Float(width) + eps).squareRoot()
            for i in 0..<width {
                out[r * width + i] = x[r * width + i] * inv * (weight?[i] ?? 1)
            }
        }
        return out
    }

    static func geluTanh(_ x: Float) -> Float {
        let c: Float = 0.79788456080286535587989211986876
        return 0.5 * x * (1 + tanh(c * x * (1 + 0.044715 * x * x)))
    }

    /// NeoX RoPE over `[tokens, heads, headDim]` with one position per token.
    static func rope(_ x: inout [Float], tokens: Int, heads: Int, table: RopeTable, positions: [Int32]) {
        let headDim = table.headDim
        let half = headDim / 2
        for t in 0..<tokens {
            let pos = Int(positions[t])
            for h in 0..<heads {
                let o = (t * heads + h) * headDim
                for i in 0..<half {
                    let (c, s) = table.factors(position: pos, pair: i)
                    let a = x[o + i], b = x[o + i + half]
                    x[o + i] = a * c - b * s
                    x[o + i + half] = a * s + b * c
                }
            }
        }
    }

    /// Bidirectional attention over one sequence with an optional symmetric window.
    /// `q` is `[tokens, heads, hd]`, `k` and `v` are `[tokens, kvHeads, hd]`; returns `[tokens, heads, hd]`.
    static func attention(q: [Float], k: [Float], v: [Float], tokens: Int, heads: Int, kvHeads: Int, headDim hd: Int,
                          scale: Float, window: Int?) -> [Float] {
        var out = [Float](repeating: 0, count: tokens * heads * hd)
        let group = heads / kvHeads
        func headMatrix(_ x: [Float], head: Int, headCount: Int) -> [Float] {
            var m = [Float](repeating: 0, count: tokens * hd)
            for t in 0..<tokens {
                for d in 0..<hd { m[t * hd + d] = x[(t * headCount + head) * hd + d] }
            }
            return m
        }
        var scores = [Float](repeating: 0, count: tokens * tokens)
        for h in 0..<heads {
            let qh = headMatrix(q, head: h, headCount: heads)
            let kh = headMatrix(k, head: h / group, headCount: kvHeads)
            let vh = headMatrix(v, head: h / group, headCount: kvHeads)
            let kT = transposeMatrix(kh, rows: tokens, cols: hd)
            qh.withUnsafeBufferPointer { qp in
                kT.withUnsafeBufferPointer { kp in
                    scores.withUnsafeMutableBufferPointer { sp in
                        vDSP_mmul(qp.baseAddress!, 1, kp.baseAddress!, 1, sp.baseAddress!, 1, vDSP_Length(tokens), vDSP_Length(tokens), vDSP_Length(hd))
                    }
                }
            }
            for i in 0..<tokens {
                var maxScore = -Float.infinity
                for j in 0..<tokens {
                    if let w = window, abs(i - j) > w / 2 {
                        scores[i * tokens + j] = -.infinity
                    } else {
                        scores[i * tokens + j] *= scale
                        maxScore = max(maxScore, scores[i * tokens + j])
                    }
                }
                var sum: Float = 0
                for j in 0..<tokens {
                    let s = scores[i * tokens + j]
                    let e = s == -.infinity ? 0 : expf(s - maxScore)
                    scores[i * tokens + j] = e
                    sum += e
                }
                let inv = 1 / sum
                for j in 0..<tokens { scores[i * tokens + j] *= inv }
            }
            var oh = [Float](repeating: 0, count: tokens * hd)
            scores.withUnsafeBufferPointer { sp in
                vh.withUnsafeBufferPointer { vp in
                    oh.withUnsafeMutableBufferPointer { op in
                        vDSP_mmul(sp.baseAddress!, 1, vp.baseAddress!, 1, op.baseAddress!, 1, vDSP_Length(tokens), vDSP_Length(hd), vDSP_Length(tokens))
                    }
                }
            }
            for t in 0..<tokens {
                for d in 0..<hd { out[(t * heads + h) * hd + d] = oh[t * hd + d] }
            }
        }
        return out
    }

    /// Runs one token sequence and returns the mean-pooled, unnormalised output vector.
    /// `trace` receives named intermediate tensors using llama.cpp's names.
    func forward(tokens: [Int32], trace: ((String, [Float]) -> Void)? = nil) -> [Float] {
        forward(rows: tokens.map(InputRow.token), trace: trace)
    }

    /// Runs one sequence of token and vector rows and returns the mean-pooled, unnormalised output vector.
    func forward(rows: [InputRow], trace: ((String, [Float]) -> Void)? = nil) -> [Float] {
        let info = config.info
        let d = info.hiddenSize
        let T = rows.count
        let eps = info.normEpsilon
        let positions = (0..<T).map(Int32.init)

        var x = [Float](repeating: 0, count: T * d)
        x.withUnsafeMutableBufferPointer { xp in
            for (t, row) in rows.enumerated() {
                let dst = xp.baseAddress! + t * d
                switch row {
                case .token(let id):
                    weights.tokenEmbedding.dequantizeRow(Int(id), into: dst)
                    for i in 0..<d { dst[i] *= config.embeddingScale }
                case .vector(let v):
                    precondition(v.count == d, "vector row is \(v.count) wide, the residual stream \(d)")
                    for i in 0..<d { dst[i] = v[i] }
                }
            }
        }
        trace?("inp_scaled", x)

        let plDim = info.perLayerInputSize
        var perLayer = CPUReference.matmul(x, rows: T, cols: d, transposedWeight: transposed(weights.perLayerProj), outputs: plDim * info.layerCount)
        for i in 0..<perLayer.count { perLayer[i] *= config.perLayerScale }
        perLayer = CPUReference.rmsNorm(perLayer, width: plDim, weight: matrix(weights.perLayerProjNorm), eps: eps)
        trace?("inp_per_layer", perLayer)

        for layer in config.layers {
            let l = layer.index
            let w = weights.layers[l]
            let heads = info.headCount, kvHeads = layer.kvHeadCount, hd = layer.headDim

            let h = CPUReference.rmsNorm(x, width: d, weight: matrix(w.attnNorm), eps: eps)
            trace?("attn_norm-\(l)", h)
            var q = CPUReference.matmul(h, rows: T, cols: d, transposedWeight: transposed(w.wq), outputs: heads * hd)
            var k = CPUReference.matmul(h, rows: T, cols: d, transposedWeight: transposed(w.wk), outputs: kvHeads * hd)
            var v = CPUReference.matmul(h, rows: T, cols: d, transposedWeight: transposed(w.wv), outputs: kvHeads * hd)
            trace?("Qcur-\(l)", q)
            trace?("Kcur-\(l)", k)
            trace?("Vcur-\(l)", v)
            q = CPUReference.rmsNorm(q, width: hd, weight: matrix(w.qNorm), eps: eps)
            k = CPUReference.rmsNorm(k, width: hd, weight: matrix(w.kNorm), eps: eps)
            v = CPUReference.rmsNorm(v, width: hd, weight: nil, eps: eps)
            trace?("Qcur_normed-\(l)", q)
            trace?("Kcur_normed-\(l)", k)
            trace?("Vcur_normed-\(l)", v)
            let table = ropeTable(for: layer)
            CPUReference.rope(&q, tokens: T, heads: heads, table: table, positions: positions)
            CPUReference.rope(&k, tokens: T, heads: kvHeads, table: table, positions: positions)
            trace?("Qcur_pos-\(l)", q)
            trace?("Kcur_pos-\(l)", k)
            let attn = CPUReference.attention(q: q, k: k, v: v, tokens: T, heads: heads, kvHeads: kvHeads, headDim: hd,
                                              scale: 1, window: layer.isSlidingWindow ? info.slidingWindow : nil)
            trace?("kqv_out-\(l)", attn)
            var o = CPUReference.matmul(attn, rows: T, cols: heads * hd, transposedWeight: transposed(w.wo), outputs: d)
            o = CPUReference.rmsNorm(o, width: d, weight: matrix(w.postAttnNorm), eps: eps)
            trace?("attn_post_norm-\(l)", o)
            for i in 0..<x.count { x[i] += o[i] }
            trace?("attn_out-\(l)", x)

            let f = CPUReference.rmsNorm(x, width: d, weight: matrix(w.ffnNorm), eps: eps)
            trace?("ffn_norm-\(l)", f)
            let gate = CPUReference.matmul(f, rows: T, cols: d, transposedWeight: transposed(w.wGate), outputs: info.feedForwardSize)
            let up = CPUReference.matmul(f, rows: T, cols: d, transposedWeight: transposed(w.wUp), outputs: info.feedForwardSize)
            var act = [Float](repeating: 0, count: gate.count)
            for i in 0..<act.count { act[i] = CPUReference.geluTanh(gate[i]) * up[i] }
            trace?("ffn_geglu-\(l)", act)
            var down = CPUReference.matmul(act, rows: T, cols: info.feedForwardSize, transposedWeight: transposed(w.wDown), outputs: d)
            trace?("ffn_out-\(l)", down)
            down = CPUReference.rmsNorm(down, width: d, weight: matrix(w.postFfnNorm), eps: eps)
            trace?("ffn_post_norm-\(l)", down)
            for i in 0..<x.count { x[i] += down[i] }
            trace?("pe_in-\(l)", x)

            var g = CPUReference.matmul(x, rows: T, cols: d, transposedWeight: transposed(w.inpGate), outputs: plDim)
            for t in 0..<T {
                for i in 0..<plDim {
                    let idx = t * plDim + i
                    g[idx] = CPUReference.geluTanh(g[idx]) * perLayer[(t * info.layerCount + l) * plDim + i]
                }
            }
            var p = CPUReference.matmul(g, rows: T, cols: plDim, transposedWeight: transposed(w.proj), outputs: d)
            p = CPUReference.rmsNorm(p, width: d, weight: matrix(w.postNorm), eps: eps)
            trace?("per_layer_embd_out-\(l)", p)
            for i in 0..<x.count { x[i] = (x[i] + p[i]) * w.outScale }
            trace?("l_out-\(l)", x)
        }

        let normed = CPUReference.rmsNorm(x, width: d, weight: matrix(weights.outputNorm), eps: eps)
        trace?("result_norm", normed)
        let projected = CPUReference.matmul(normed, rows: T, cols: d, transposedWeight: transposed(weights.output), outputs: info.outputDimensions)
        trace?("result_embd", projected)
        var pooled = [Float](repeating: 0, count: info.outputDimensions)
        for t in 0..<T {
            for i in 0..<info.outputDimensions { pooled[i] += projected[t * info.outputDimensions + i] }
        }
        for i in 0..<pooled.count { pooled[i] /= Float(T) }
        return pooled
    }
}
