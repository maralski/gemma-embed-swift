// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Accelerate
import Foundation

/// A plain fp32 forward pass of the gemma4v vision tower on the CPU, the oracle for the Metal
/// image encoder. Like `CPUReference` it favours obviousness over speed.
final class CPUVisionReference {
    let config: VisionConfig
    let weights: VisionWeights
    private var cache: [String: [Float]] = [:]
    private let lock = NSLock()

    init(config: VisionConfig, weights: VisionWeights) {
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

    static func geluQuick(_ x: Float) -> Float {
        x / (1 + expf(-1.702 * x))
    }

    /// 2D NeoX RoPE over `[patches, heads, headDim]`: the first half of each head rotates by the
    /// patch column, the second half by the patch row, each as a NeoX rotation of `headDim / 2` dims.
    static func rope2D(_ x: inout [Float], heads: Int, headDim: Int, table: RopeTable, patchX: [Int32], patchY: [Int32]) {
        let half = headDim / 2
        precondition(table.headDim == half)
        let quarter = half / 2
        for t in 0..<patchX.count {
            for (offset, pos) in [(0, Int(patchX[t])), (half, Int(patchY[t]))] {
                for h in 0..<heads {
                    let o = (t * heads + h) * headDim + offset
                    for i in 0..<quarter {
                        let (c, s) = table.factors(position: pos, pair: i)
                        let a = x[o + i], b = x[o + i + quarter]
                        x[o + i] = a * c - b * s
                        x[o + i + quarter] = a * s + b * c
                    }
                }
            }
        }
    }

    /// Averages every `merge` by `merge` window of a `gridX` by `gridY` grid of `width`-wide rows
    /// and scales by `sqrt(width)`; token `oy * (gridX / merge) + ox`.
    static func pool(_ x: [Float], gridX: Int, gridY: Int, merge m: Int, width d: Int) -> [Float] {
        let tx = gridX / m
        var pooled = [Float](repeating: 0, count: tx * (gridY / m) * d)
        let poolScale = Float(d).squareRoot() / Float(m * m)
        for oy in 0..<(gridY / m) {
            for ox in 0..<tx {
                let o = (oy * tx + ox) * d
                for ky in 0..<m {
                    for kx in 0..<m {
                        let src = ((oy * m + ky) * gridX + ox * m + kx) * d
                        for i in 0..<d { pooled[o + i] += x[src + i] }
                    }
                }
                for i in 0..<d { pooled[o + i] *= poolScale }
            }
        }
        return pooled
    }

    /// Runs the vision tower and returns `tokenCount` rows of `projectionDim` values, the
    /// vectors that stand in for the image in the text sequence.
    /// `trace` receives named intermediate tensors using llama.cpp's names.
    func forward(_ image: PreparedImage, trace: ((String, [Float]) -> Void)? = nil) -> [Float] {
        let d = config.info.hiddenSize
        let P = image.patchCount
        let eps = config.normEpsilon
        let heads = config.headCount, hd = config.headDim
        let ff = config.feedForwardSize
        precondition(image.patches.count == P * config.patchArea)
        precondition(max(image.gridX, image.gridY) <= weights.positionCount)

        var x = CPUReference.matmul(image.patches, rows: P, cols: config.patchArea,
                                    transposedWeight: transposed(weights.patchEmbedding), outputs: d)
        trace?("inp", x)
        x.withUnsafeMutableBufferPointer { xp in
            var row = [Float](repeating: 0, count: d)
            for t in 0..<P {
                weights.positionTable.dequantizeRow(Int(image.patchX[t]), into: &row)
                for i in 0..<d { xp[t * d + i] += row[i] }
                weights.positionTable.dequantizeRow(weights.positionCount + Int(image.patchY[t]), into: &row)
                for i in 0..<d { xp[t * d + i] += row[i] }
            }
        }
        trace?("pos_embd", x)

        let table = RopeTable(headDim: hd / 2, base: config.ropeTheta, positions: max(image.gridX, image.gridY))
        for (l, w) in weights.layers.enumerated() {
            let a = CPUReference.rmsNorm(x, width: d, weight: matrix(w.ln1), eps: eps)
            trace?("layer_inp_normed-\(l)", a)
            var q = CPUReference.matmul(a, rows: P, cols: d, transposedWeight: transposed(w.wq), outputs: d)
            var k = CPUReference.matmul(a, rows: P, cols: d, transposedWeight: transposed(w.wk), outputs: d)
            var v = CPUReference.matmul(a, rows: P, cols: d, transposedWeight: transposed(w.wv), outputs: d)
            trace?("Vcur-\(l)", v)
            q = CPUReference.rmsNorm(q, width: hd, weight: matrix(w.qNorm), eps: eps)
            k = CPUReference.rmsNorm(k, width: hd, weight: matrix(w.kNorm), eps: eps)
            trace?("Qcur-\(l)", q)
            trace?("Kcur-\(l)", k)
            CPUVisionReference.rope2D(&q, heads: heads, headDim: hd, table: table, patchX: image.patchX, patchY: image.patchY)
            CPUVisionReference.rope2D(&k, heads: heads, headDim: hd, table: table, patchX: image.patchX, patchY: image.patchY)
            trace?("Qcur_pos-\(l)", q)
            trace?("Kcur_pos-\(l)", k)
            v = CPUReference.rmsNorm(v, width: hd, weight: nil, eps: eps)
            trace?("Vcur_normed-\(l)", v)
            let attn = CPUReference.attention(q: q, k: k, v: v, tokens: P, heads: heads, kvHeads: heads, headDim: hd,
                                              scale: 1, window: nil)
            trace?("kqv_out-\(l)", attn)
            var o = CPUReference.matmul(attn, rows: P, cols: d, transposedWeight: transposed(w.wo), outputs: d)
            trace?("attn_out-\(l)", o)
            o = CPUReference.rmsNorm(o, width: d, weight: matrix(w.postAttnNorm), eps: eps)
            trace?("attn_post_normed-\(l)", o)
            for i in 0..<x.count { x[i] += o[i] }
            trace?("ffn_inp-\(l)", x)

            let f = CPUReference.rmsNorm(x, width: d, weight: matrix(w.ln2), eps: eps)
            trace?("ffn_inp_normed-\(l)", f)
            let gate = CPUReference.matmul(f, rows: P, cols: d, transposedWeight: transposed(w.wGate), outputs: ff)
            let up = CPUReference.matmul(f, rows: P, cols: d, transposedWeight: transposed(w.wUp), outputs: ff)
            trace?("ffn_gate-\(l)", gate)
            trace?("ffn_up-\(l)", up)
            var act = [Float](repeating: 0, count: gate.count)
            for i in 0..<act.count { act[i] = CPUVisionReference.geluQuick(gate[i]) * up[i] }
            trace?("ffn_geglu_quick-\(l)", act)
            var down = CPUReference.matmul(act, rows: P, cols: ff, transposedWeight: transposed(w.wDown), outputs: d)
            trace?("ffn_out-\(l)", down)
            down = CPUReference.rmsNorm(down, width: d, weight: matrix(w.postFfnNorm), eps: eps)
            trace?("ffn_post_normed-\(l)", down)
            for i in 0..<x.count { x[i] += down[i] }
            trace?("layer_out-\(l)", x)
        }

        let T = image.tokenCount
        let pooled = CPUVisionReference.pool(x, gridX: image.gridX, gridY: image.gridY, merge: image.mergeSize, width: d)
        trace?("pooled", pooled)
        let normed = CPUReference.rmsNorm(pooled, width: d, weight: nil, eps: eps)
        let projected = CPUReference.matmul(normed, rows: T, cols: d, transposedWeight: transposed(weights.projection),
                                            outputs: config.info.projectionDim)
        trace?("projected", projected)
        return projected
    }
}
