import Foundation
import Metal
import XCTest
@testable import GemmaEmbed

/// Deterministic pseudo-random floats for kernel inputs.
struct SeededRandom {
    var state: UInt64

    init(seed: UInt64) { state = seed &+ 0x9E37_79B9_7F4A_7C15 }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in [-scale, scale).
    mutating func floats(_ n: Int, scale: Float = 1) -> [Float] {
        (0..<n).map { _ in (Float(next() >> 40) / Float(1 << 24) * 2 - 1) * scale }
    }
}

final class KernelTests: XCTestCase {
    final class Env {
        let context: MetalContext
        let config: ModelConfig
        let weights: ModelWeights
        let forward: MetalForward
        let reference: CPUReference

        init(context: MetalContext, config: ModelConfig, weights: ModelWeights, forward: MetalForward, reference: CPUReference) {
            self.context = context
            self.config = config
            self.weights = weights
            self.forward = forward
            self.reference = reference
        }
    }

    nonisolated(unsafe) static var env: Env?

    override func setUpWithError() throws {
        if Self.env == nil {
            let file = try GGUFFile(url: try TestSupport.requireWeights())
            let context = try MetalContext()
            let config = try ModelConfig(gguf: file, deviceName: context.device.name, fileSHA256: nil)
            let weights = try ModelWeights(gguf: file, config: config)
            let forward = try MetalForward(context: context, config: config, weights: weights)
            Self.env = Env(context: context, config: config, weights: weights, forward: forward, reference: CPUReference(config: config, weights: weights))
        }
    }

    final class VisionEnv {
        let config: VisionConfig
        let weights: VisionWeights
        let tower: VisionTower
        let reference: CPUVisionReference

        init(config: VisionConfig, weights: VisionWeights, tower: VisionTower, reference: CPUVisionReference) {
            self.config = config
            self.weights = weights
            self.tower = tower
            self.reference = reference
        }
    }

    nonisolated(unsafe) static var visionEnv: VisionEnv?

    func vision() throws -> VisionEnv {
        if let v = Self.visionEnv { return v }
        let env = try XCTUnwrap(Self.env)
        let file = try GGUFFile(url: try TestSupport.requireImageEncoder())
        let config = try VisionConfig(gguf: file, fileSHA256: nil)
        let weights = try VisionWeights(gguf: file, config: config)
        let tower = try VisionTower(context: env.context, config: config, weights: weights,
                                    markers: try ImageMarkers(tokenizer: Tokenizer(gguf: env.weights.file)))
        let v = VisionEnv(config: config, weights: weights, tower: tower, reference: CPUVisionReference(config: config, weights: weights))
        Self.visionEnv = v
        return v
    }

    /// Encodes `body` in one command buffer; `weights` is the buffer the tensors live in, the text model's by default.
    func run(weights: MTLBuffer? = nil, _ body: (Kernels) throws -> Void) throws {
        let env = try XCTUnwrap(Self.env)
        let cb = try XCTUnwrap(env.context.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        let kernels = Kernels(forward: env.forward, encoder: enc, weights: weights)
        enc.useResource(kernels.weights, usage: .read)
        try body(kernels)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
    }

    func buffer(_ values: [Float], label: String = "test") throws -> MTLBuffer {
        try XCTUnwrap(Self.env).context.makeBuffer(array: values, label: label)
    }

    func assertClose(_ got: [Float], _ want: [Float], tolerance: Float = 1e-4, _ message: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(got.count, want.count, message, file: file, line: line)
        let rel = TestSupport.maxRelDiff(got, want)
        print("kernel parity \(message): max relative diff \(rel)")
        XCTAssertLessThan(rel, tolerance, "\(message): max relative diff \(rel)", file: file, line: line)
    }

    func testShadersCompileAndAllPipelinesBuild() throws {
        let env = try XCTUnwrap(Self.env)
        try env.context.warmUpPipelines()
        XCTAssertGreaterThan(env.context.library.functionNames.count, 10)
    }

    func testGemmAllModelShapesAndTiles() throws {
        let env = try XCTUnwrap(Self.env)
        var rng = SeededRandom(seed: 1)
        let l0 = env.weights.layers[0], l5 = env.weights.layers[5]
        let cases: [(WeightTensor, Int)] = [
            (l0.wq, 0), (l0.wk, 0), (l0.wo, 0), (l0.wGate, 0), (l0.wDown, 0), (l0.inpGate, 0), (l0.proj, 0),
            (l5.wq, 0), (l5.wk, 0), (l5.wo, 0),
            (env.weights.output, 0), (env.weights.perLayerProj, 7 * 512),
        ]
        for (w, rowOffset) in cases {
            let K = w.columns
            let N = rowOffset == 0 ? w.rows : 512
            let full = env.reference.matrix(w)
            let wSlice = Array(full[(rowOffset * K)..<((rowOffset + N) * K)])
            let wT = CPUReference.transposeMatrix(wSlice, rows: N, cols: K)
            for M in [64, 192] {
                let x = rng.floats(M * K, scale: 2)
                let want = CPUReference.matmul(x, rows: M, cols: K, transposedWeight: wT, outputs: N)
                let xb = try buffer(x)
                for tile in GemmTile.compiled {
                    let yb = try buffer([Float](repeating: 0, count: M * N))
                    try run { k in
                        try k.gemm(xb, weight: w, weightRowOffset: rowOffset, out: yb, rows: M, outputs: N, inputs: K, tile: tile)
                    }
                    assertClose(yb.array(of: Float.self, count: M * N), want, "\(w.info.name) M=\(M) tile \(tile.suffix)")
                }
            }
        }
    }

    func testGemmGegluEpilogue() throws {
        let env = try XCTUnwrap(Self.env)
        var rng = SeededRandom(seed: 7)
        let w = env.weights.layers[2].wUp
        let M = 64, K = w.columns, N = w.rows
        let x = rng.floats(M * K)
        let gate = rng.floats(M * N, scale: 4)
        let wT = CPUReference.transposeMatrix(env.reference.matrix(w), rows: N, cols: K)
        let up = CPUReference.matmul(x, rows: M, cols: K, transposedWeight: wT, outputs: N)
        let want = zip(gate, up).map { CPUReference.geluTanh($0) * $1 }
        let xb = try buffer(x), gb = try buffer(gate)
        for tile in GemmTile.compiled {
            let yb = try buffer([Float](repeating: 0, count: M * N))
            try run { k in try k.gemm(xb, weight: w, out: yb, rows: M, outputs: N, inputs: K, tile: tile, epilogue: .geglu(gate: gb)) }
            assertClose(yb.array(of: Float.self, count: M * N), want, "geglu epilogue tile \(tile.suffix)")
            let zb = try buffer([Float](repeating: 0, count: M * N))
            try run { k in try k.gemm(xb, weight: w, out: zb, rows: M, outputs: N, inputs: K, tile: tile, epilogue: .geluTimes(factor: gb)) }
            let wantTimes = zip(gate, up).map { CPUReference.geluTanh($1) * $0 }
            assertClose(zb.array(of: Float.self, count: M * N), wantTimes, "gelu-times epilogue tile \(tile.suffix)")
        }
    }

    func testGemmRowResultIndependentOfTile() throws {
        let env = try XCTUnwrap(Self.env)
        var rng = SeededRandom(seed: 2)
        let w = env.weights.layers[3].wUp
        let M = 128, K = w.columns, N = w.rows
        let x = rng.floats(M * K)
        let xb = try buffer(x)
        var results: [[Float]] = []
        for tile in GemmTile.compiled {
            let yb = try buffer([Float](repeating: 0, count: M * N))
            try run { k in try k.gemm(xb, weight: w, out: yb, rows: M, outputs: N, inputs: K, tile: tile) }
            results.append(yb.array(of: Float.self, count: M * N))
        }
        for r in results.dropFirst() {
            XCTAssertEqual(r, results[0], "tile shapes must produce bit-identical results")
        }
    }

    func testGemm3MatchesThreeGemmsAtEveryTile() throws {
        let env = try XCTUnwrap(Self.env)
        var rng = SeededRandom(seed: 12)
        let ws = (env.weights.layers[0].wk, env.weights.layers[0].wv, env.weights.layers[1].wk)
        let M = 128, K = ws.0.columns, N = ws.0.rows
        let xb = try buffer(rng.floats(M * K))
        var want: [[Float]] = []
        for w in [ws.0, ws.1, ws.2] {
            let yb = try buffer([Float](repeating: 0, count: M * N))
            try run { k in try k.gemm(xb, weight: w, out: yb, rows: M, outputs: N, inputs: K) }
            want.append(yb.array(of: Float.self, count: M * N))
        }
        for tile in GemmTile.compiled {
            let ys = try (buffer([Float](repeating: 0, count: M * N)), buffer([Float](repeating: 0, count: M * N)),
                          buffer([Float](repeating: 0, count: M * N)))
            try run { k in try k.gemm3(xb, weights: ws, outs: ys, rows: M, outputs: N, inputs: K, tile: tile) }
            for (i, y) in [ys.0, ys.1, ys.2].enumerated() {
                XCTAssertEqual(y.array(of: Float.self, count: M * N), want[i], "gemm3 output \(i) tile \(tile.suffix)")
            }
        }
    }

    func testRmsNorm() throws {
        let env = try XCTUnwrap(Self.env)
        var rng = SeededRandom(seed: 3)
        let eps = env.config.info.normEpsilon
        for (width, weight, preScale) in [(512, Optional(env.weights.layers[0].attnNorm), Float(1)),
                                          (256, Optional(env.weights.layers[0].qNorm), 1),
                                          (512, Optional(env.weights.layers[5].kNorm), 1),
                                          (512, nil, 1),
                                          (512, Optional(env.weights.perLayerProjNorm), env.config.perLayerScale)] {
            let rows = 37
            let x = rng.floats(rows * width, scale: 5)
            let want = CPUReference.rmsNorm(x.map { $0 * preScale }, width: width, weight: weight.map { env.reference.matrix($0) }, eps: eps)
            let xb = try buffer(x)
            let ob = try buffer([Float](repeating: 0, count: rows * width))
            try run { k in try k.rmsNorm(xb, out: ob, weight: weight, rows: rows, width: width, eps: eps, preScale: preScale) }
            assertClose(ob.array(of: Float.self, count: rows * width), want, "rmsnorm width \(width) weight \(weight?.info.name ?? "none")")
            try run { k in try k.rmsNorm(xb, out: xb, weight: weight, rows: rows, width: width, eps: eps, preScale: preScale) }
            assertClose(xb.array(of: Float.self, count: rows * width), want, "in-place rmsnorm width \(width)")
        }
    }

    func testRmsNormResidual() throws {
        let env = try XCTUnwrap(Self.env)
        var rng = SeededRandom(seed: 8)
        let eps = env.config.info.normEpsilon
        let weight = env.weights.layers[4].postNorm
        let rows = 45, width = 512
        let x = rng.floats(rows * width, scale: 3), t = rng.floats(rows * width, scale: 5)
        let normed = CPUReference.rmsNorm(t, width: width, weight: env.reference.matrix(weight), eps: eps)
        let want = zip(x, normed).map { ($0 + $1) * 0.8 }
        let xb = try buffer(x), tb = try buffer(t)
        try run { k in try k.rmsNormResidual(xb, add: tb, weight: weight, rows: rows, width: width, eps: eps, scale: 0.8) }
        assertClose(xb.array(of: Float.self, count: rows * width), want, "rmsnorm_residual")
    }

    func testRmsNormResidualNormAtTextWidthMatchesResidualThenNorm() throws {
        let env = try XCTUnwrap(Self.env)
        var rng = SeededRandom(seed: 19)
        let eps = env.config.info.normEpsilon, d = env.config.info.hiddenSize
        let l = env.weights.layers[4]
        let rows = 229
        let x = rng.floats(rows * d, scale: 3), t = rng.floats(rows * d, scale: 5)
        for scale: Float in [1, l.outScale] {
            let xb = try buffer(x), tb = try buffer(t), hb = try buffer([Float](repeating: 7, count: rows * d))
            let sxb = try buffer(x), shb = try buffer([Float](repeating: 7, count: rows * d))
            try run { k in
                try k.rmsNormResidual(sxb, add: tb, weight: l.postNorm, rows: rows, width: d, eps: eps, scale: scale)
                try k.rmsNorm(sxb, out: shb, weight: env.weights.layers[5].attnNorm, rows: rows, width: d, eps: eps)
            }
            try run { k in
                try k.rmsNormResidualNorm(xb, add: tb, weight: l.postNorm, out: hb, outWeight: env.weights.layers[5].attnNorm,
                                          rows: rows, width: d, eps: eps, scale: scale)
            }
            XCTAssertEqual(xb.array(of: Float.self, count: rows * d), sxb.array(of: Float.self, count: rows * d),
                           "scale \(scale): the fused residual update must equal rmsnorm_residual bit for bit")
            XCTAssertEqual(hb.array(of: Float.self, count: rows * d), shb.array(of: Float.self, count: rows * d),
                           "scale \(scale): the fused norm must equal rmsnorm_rows over the updated rows bit for bit")
        }
    }

    func testRope() throws {
        let env = try XCTUnwrap(Self.env)
        var rng = SeededRandom(seed: 4)
        for layer in [env.config.layers[0], env.config.layers[5]] {
            let table = env.reference.ropeTable(for: layer)
            let tokens = 300
            let positions = (0..<tokens).map { Int32($0 * 27 % 8192) }
            for heads in [1, 2, 4] {
                let x = rng.floats(tokens * heads * layer.headDim)
                var want = x
                CPUReference.rope(&want, tokens: tokens, heads: heads, table: table, positions: positions)
                let xb = try buffer(x)
                let pb = try env.context.makeBuffer(array: positions, label: "positions")
                let tb = try env.context.makeBuffer(array: table.values, label: "table")
                try run { k in try k.rope(xb, positions: pb, table: tb, pairs: table.pairs, tokens: tokens, heads: heads, headDim: layer.headDim) }
                assertClose(xb.array(of: Float.self, count: x.count), want, "rope hd \(layer.headDim) heads \(heads)")
            }
        }
    }

    func testRmsNormRopeMatchesNormThenRope() throws {
        let env = try XCTUnwrap(Self.env)
        var rng = SeededRandom(seed: 9)
        let eps = env.config.info.normEpsilon
        for layer in [env.config.layers[0], env.config.layers[5]] {
            let table = env.reference.ropeTable(for: layer)
            let weight = env.weights.layers[layer.index].kNorm
            let tokens = 131, heads = layer.kvHeadCount, hd = layer.headDim
            let positions = (0..<tokens).map { Int32($0 * 13 % 2048) }
            let x = rng.floats(tokens * heads * hd, scale: 4)
            let xb = try buffer(x), sb = try buffer(x)
            let pb = try env.context.makeBuffer(array: positions, label: "positions")
            let tb = try env.context.makeBuffer(array: table.values, label: "table")
            try run { k in
                try k.rmsNorm(sb, out: sb, weight: weight, rows: tokens * heads, width: hd, eps: eps)
                try k.rope(sb, positions: pb, table: tb, pairs: table.pairs, tokens: tokens, heads: heads, headDim: hd)
            }
            try run { k in try k.rmsNormRope(xb, weight: weight, positions: pb, table: tb, pairs: table.pairs, tokens: tokens, heads: heads, headDim: hd, eps: eps) }
            XCTAssertEqual(xb.array(of: Float.self, count: x.count), sb.array(of: Float.self, count: x.count),
                           "fused norm and rope must equal the separate kernels bit for bit, hd \(hd)")
            var want = CPUReference.rmsNorm(x, width: hd, weight: env.reference.matrix(weight), eps: eps)
            CPUReference.rope(&want, tokens: tokens, heads: heads, table: table, positions: positions)
            assertClose(xb.array(of: Float.self, count: x.count), want, "rmsnorm_rope hd \(hd)")
        }
    }

    func testEmbedding() throws {
        let env = try XCTUnwrap(Self.env)
        let ids: [Int32] = [2, 8071, 236787, 3927, 1354, 1109, 7609, 1, 0, 262143]
        let padded = 64
        let d = env.config.info.hiddenSize
        let tb = try env.context.makeBuffer(array: ids, label: "ids")
        let ob = try buffer([Float](repeating: 1, count: padded * d))
        let rb = try buffer([Float](repeating: 0, count: padded * d))
        try run { k in try k.embed(table: env.weights.tokenEmbedding, tokens: tb, out: ob, residual: rb, tokenCount: ids.count, paddedRows: padded, scale: env.config.embeddingScale) }
        let got = ob.array(of: Float.self, count: padded * d)
        var want = [Float](repeating: 0, count: padded * d)
        want.withUnsafeMutableBufferPointer { wp in
            for (t, id) in ids.enumerated() { env.weights.tokenEmbedding.dequantizeRow(Int(id), into: wp.baseAddress! + t * d) }
        }
        for i in 0..<(ids.count * d) { want[i] *= env.config.embeddingScale }
        assertClose(got, want, "embed")
        XCTAssertEqual(Array(got[(ids.count * d)...]), [Float](repeating: 0, count: (padded - ids.count) * d), "padding rows must be zero")
    }

    /// Negative ids copy rows of the vectors buffer unscaled, the image rows of a text sequence.
    func testEmbeddingWithVectorRows() throws {
        let env = try XCTUnwrap(Self.env)
        var rng = SeededRandom(seed: 11)
        let d = env.config.info.hiddenSize
        let vectors = rng.floats(4 * d, scale: 3)
        let ids: [Int32] = [2, 255999, -1, -2, -3, -4, 258882, 1, -3, 8071]
        let padded = 64
        let tb = try env.context.makeBuffer(array: ids, label: "ids")
        let vb = try buffer(vectors)
        let ob = try buffer([Float](repeating: 1, count: padded * d))
        let rb = try buffer([Float](repeating: 2, count: padded * d))
        try run { k in
            try k.embed(table: env.weights.tokenEmbedding, tokens: tb, vectors: vb, out: ob, residual: rb, tokenCount: ids.count, paddedRows: padded,
                        scale: env.config.embeddingScale)
        }
        let got = ob.array(of: Float.self, count: padded * d)
        XCTAssertEqual(rb.array(of: Float.self, count: padded * d), got, "the residual copy must equal the embedding rows, padding included")
        var want = [Float](repeating: 0, count: padded * d)
        want.withUnsafeMutableBufferPointer { wp in
            for (t, id) in ids.enumerated() {
                if id < 0 {
                    for i in 0..<d { wp[t * d + i] = vectors[Int(-(id + 1)) * d + i] }
                } else {
                    env.weights.tokenEmbedding.dequantizeRow(Int(id), into: wp.baseAddress! + t * d)
                    for i in 0..<d { wp[t * d + i] *= env.config.embeddingScale }
                }
            }
        }
        assertClose(got, want, "embed with vector rows")
        for (t, id) in ids.enumerated() where id < 0 {
            XCTAssertEqual(Array(got[(t * d)..<((t + 1) * d)]), Array(vectors[(Int(-(id + 1)) * d)..<((Int(-(id + 1)) + 1) * d)]),
                           "vector row \(t) must be copied exactly, unscaled")
        }
        XCTAssertEqual(Array(got[(ids.count * d)...]), [Float](repeating: 0, count: (padded - ids.count) * d), "padding rows must be zero")
    }

    func testMeanPool() throws {
        let env = try XCTUnwrap(Self.env)
        var rng = SeededRandom(seed: 6)
        let width = 768
        let lengths = [1, 7, 33, 100]
        let total = lengths.reduce(0, +)
        let x = rng.floats(total * width)
        var seqs: [SequenceRange] = []
        var start = 0
        for l in lengths { seqs.append(SequenceRange(start: UInt32(start), length: UInt32(l))); start += l }
        let xb = try buffer(x)
        let sb = try env.context.makeBuffer(array: seqs, label: "seqs")
        let ob = try buffer([Float](repeating: 0, count: lengths.count * width))
        try run { k in try k.meanPool(xb, sequences: sb, count: lengths.count, out: ob, width: width) }
        let got = ob.array(of: Float.self, count: lengths.count * width)
        var want = [Float](repeating: 0, count: lengths.count * width)
        for (s, seq) in seqs.enumerated() {
            for r in 0..<Int(seq.length) {
                for c in 0..<width { want[s * width + c] += x[(Int(seq.start) + r) * width + c] }
            }
            for c in 0..<width { want[s * width + c] /= Float(seq.length) }
        }
        assertClose(got, want, "mean_pool")
    }

    /// Attention over packed sequences must equal per-sequence CPU attention and must not leak
    /// across sequence boundaries, for full and sliding-window layers.
    func testAttentionPackedSequences() throws {
        let env = try XCTUnwrap(Self.env)
        for (heads, kvHeads, hd, halfWindow) in [(4, 2, 256, 0), (4, 2, 256, 512), (4, 1, 512, 0), (4, 1, 512, 512)] {
            for lengths in [[1, 7, 64], [1024, 5], [2048], [2048, 3, 1024, 7]] {
                try checkAttention(env: env, lengths: lengths, heads: heads, kvHeads: kvHeads, headDim: hd, halfWindow: halfWindow, seed: UInt64(hd + halfWindow + lengths.count))
            }
        }
    }

    func testAttentionQueryGroupSizes() throws {
        let env = try XCTUnwrap(Self.env)
        let saved = env.forward.schedule
        defer { env.forward.schedule = saved }
        for queries in KernelSchedule.attentionQueryCandidates {
            env.forward.schedule.attentionQueries = queries
            try checkAttention(env: env, lengths: [100, 13, 1], heads: 4, kvHeads: 2, headDim: 256, halfWindow: 512, seed: 99)
            try checkAttention(env: env, lengths: [1100], heads: 4, kvHeads: 1, headDim: 512, halfWindow: 512, seed: 100)
            try checkAttention(env: env, lengths: [630, 9, 81], heads: 12, kvHeads: 12, headDim: 64, halfWindow: 0, seed: 101)
        }
    }

    func testAttentionSliceWidthsAreBitIdentical() throws {
        let env = try XCTUnwrap(Self.env)
        for (kvHeads, hd, halfWindow) in [(2, 256, 512), (1, 512, 512), (1, 512, 0)] {
            var rng = SeededRandom(seed: UInt64(hd + kvHeads))
            var batch = PackedBatch()
            for l in [229, 3, 700] { batch.append([Int32](repeating: 0, count: l)) }
            let rows = MetalForward.paddedRows(batch.tokenCount) + MetalForward.slackRows
            let scale = 1 / Float(hd).squareRoot()
            let qb = try buffer(rng.floats(rows * 4 * hd, scale: 2 * scale))
            let kb = try buffer(rng.floats(rows * kvHeads * hd, scale: 2 * scale))
            let vb = try buffer(rng.floats(rows * kvHeads * hd))
            let work = batch.attentionWork(queriesPerGroup: env.forward.schedule.attentionQueries)
            let wb = try env.context.makeBuffer(array: work, label: "work")
            func output(sliceWidth: Int) throws -> [Float] {
                let ob = try buffer([Float](repeating: .nan, count: rows * 4 * hd))
                try run { kern in
                    try kern.attention(q: qb, k: kb, v: vb, out: ob, work: wb, workItems: work.count, heads: 4, kvHeads: kvHeads, headDim: hd,
                                       halfWindow: halfWindow, scale: 1, sliceWidth: sliceWidth)
                }
                return ob.array(of: Float.self, count: batch.tokenCount * 4 * hd)
            }
            let full = try output(sliceWidth: hd)
            XCTAssertFalse(full.contains { $0.isNaN })
            var width = hd / 2
            while width >= 128 {
                let sliced = try output(sliceWidth: width)
                let differing = zip(full, sliced).filter { $0.bitPattern != $1.bitPattern }.count
                XCTAssertEqual(differing, 0, "hd \(hd) window \(halfWindow): slices of \(width) differ from the whole head")
                width /= 2
            }
        }
    }

    func testAttentionSliceWidthFollowsTheGrid() {
        XCTAssertEqual(KernelSchedule.attentionSliceWidth(headDim: 64, threadgroups: 1), 64)
        XCTAssertEqual(KernelSchedule.attentionSliceWidth(headDim: 256, threadgroups: 32), 128)
        XCTAssertEqual(KernelSchedule.attentionSliceWidth(headDim: 512, threadgroups: 32), 128)
        XCTAssertEqual(KernelSchedule.attentionSliceWidth(headDim: 512, threadgroups: 64), 256)
        XCTAssertEqual(KernelSchedule.attentionSliceWidth(headDim: 256, threadgroups: 128), 256)
        XCTAssertEqual(KernelSchedule.attentionSliceWidth(headDim: 512, threadgroups: 128), 512)
    }

    /// The vision tower's attention: 12 heads of 64, no window, each image its own sequence.
    func testAttentionHeadDim64PackedImages() throws {
        let env = try XCTUnwrap(Self.env)
        for lengths in [[1170, 2304, 80], [1, 7, 64]] {
            try checkAttention(env: env, lengths: lengths, heads: 12, kvHeads: 12, headDim: 64, halfWindow: 0, seed: UInt64(64 + lengths.count))
        }
    }

    private func checkAttention(env: Env, lengths: [Int], heads: Int, kvHeads: Int, headDim hd: Int, halfWindow: Int, seed: UInt64) throws {
        var rng = SeededRandom(seed: seed)
        var batch = PackedBatch()
        for l in lengths { batch.append([Int32](repeating: 0, count: l)) }
        let total = batch.tokenCount
        let rows = MetalForward.paddedRows(total) + MetalForward.slackRows
        // Scores stay O(1) with unit-scale rows of length hd scaled by 1/sqrt(hd), like normed q and k.
        let scale = 1 / Float(hd).squareRoot()
        let q = rng.floats(total * heads * hd, scale: 2 * scale)
        let k = rng.floats(total * kvHeads * hd, scale: 2 * scale)
        let v = rng.floats(total * kvHeads * hd)
        let qb = try buffer(q + [Float](repeating: 0, count: (rows - total) * heads * hd))
        let kb = try buffer(k + [Float](repeating: 0, count: (rows - total) * kvHeads * hd))
        let vb = try buffer(v + [Float](repeating: 0, count: (rows - total) * kvHeads * hd))
        let ob = try buffer([Float](repeating: .nan, count: rows * heads * hd))
        let work = batch.attentionWork(queriesPerGroup: env.forward.schedule.attentionQueries(headDim: hd))
        let wb = try env.context.makeBuffer(array: work, label: "work")
        try run { kern in
            try kern.attention(q: qb, k: kb, v: vb, out: ob, work: wb, workItems: work.count, heads: heads, kvHeads: kvHeads, headDim: hd,
                               halfWindow: halfWindow, scale: 1)
        }
        let got = ob.array(of: Float.self, count: total * heads * hd)
        for seq in batch.sequences {
            let s = Int(seq.start), n = Int(seq.length)
            let want = CPUReference.attention(q: Array(q[(s * heads * hd)..<((s + n) * heads * hd)]),
                                              k: Array(k[(s * kvHeads * hd)..<((s + n) * kvHeads * hd)]),
                                              v: Array(v[(s * kvHeads * hd)..<((s + n) * kvHeads * hd)]),
                                              tokens: n, heads: heads, kvHeads: kvHeads, headDim: hd, scale: 1,
                                              window: halfWindow > 0 ? halfWindow * 2 : nil)
            let gotSeq = Array(got[(s * heads * hd)..<((s + n) * heads * hd)])
            XCTAssertFalse(gotSeq.contains { $0.isNaN }, "attention wrote NaN for lengths \(lengths) hd \(hd)")
            assertClose(gotSeq, want, "attention lengths \(lengths) seq len \(n) hd \(hd) kv \(kvHeads) window \(halfWindow)")
        }
    }

    func testGemmVisionShapesAndTiles() throws {
        let v = try vision()
        var rng = SeededRandom(seed: 12)
        let l0 = v.weights.layers[0], l9 = v.weights.layers[9]
        for w in [v.weights.patchEmbedding, l0.wq, l0.wo, l9.wGate, l9.wDown, v.weights.projection] {
            let K = w.columns, N = w.rows
            let wT = v.reference.transposed(w)
            for M in [64, 192] {
                // Patches are fp16-rounded pixels in [-1, 1]; the other inputs are normed rows.
                let x = rng.floats(M * K, scale: 2).map { w.type == .f32 ? Float(Float16($0 / 2)) : $0 }
                let want = CPUReference.matmul(x, rows: M, cols: K, transposedWeight: wT, outputs: N)
                let xb = try buffer(x)
                for tile in GemmTile.compiled {
                    let yb = try buffer([Float](repeating: 0, count: M * N))
                    try run(weights: v.tower.weightBuffer) { k in try k.gemm(xb, weight: w, out: yb, rows: M, outputs: N, inputs: K, tile: tile) }
                    assertClose(yb.array(of: Float.self, count: M * N), want, "\(w.info.name) (\(w.type)) M=\(M) tile \(tile.suffix)")
                }
            }
        }
    }

    func testGemmGegluQuickEpilogue() throws {
        let v = try vision()
        var rng = SeededRandom(seed: 13)
        let w = v.weights.layers[4].wUp
        let M = 64, K = w.columns, N = w.rows
        let x = rng.floats(M * K)
        let gate = rng.floats(M * N, scale: 6)
        let up = CPUReference.matmul(x, rows: M, cols: K, transposedWeight: v.reference.transposed(w), outputs: N)
        let want = zip(gate, up).map { CPUVisionReference.geluQuick($0) * $1 }
        let xb = try buffer(x), gb = try buffer(gate)
        for tile in GemmTile.compiled {
            let yb = try buffer([Float](repeating: 0, count: M * N))
            try run(weights: v.tower.weightBuffer) { k in
                try k.gemm(xb, weight: w, out: yb, rows: M, outputs: N, inputs: K, tile: tile, epilogue: .gegluQuick(gate: gb))
            }
            assertClose(yb.array(of: Float.self, count: M * N), want, "gelu_quick epilogue tile \(tile.suffix)")
        }
    }

    func testVisionRmsNorms() throws {
        let v = try vision()
        var rng = SeededRandom(seed: 14)
        let eps = v.config.normEpsilon
        let l3 = v.weights.layers[3]
        for (width, weight) in [(64, Optional(l3.qNorm)), (64, Optional(l3.kNorm)), (64, nil), (768, Optional(l3.ln1)), (768, nil)] {
            let rows = 12 * 37
            let x = rng.floats(rows * width, scale: 5)
            let want = CPUReference.rmsNorm(x, width: width, weight: weight.map { v.reference.matrix($0) }, eps: eps)
            let xb = try buffer(x)
            try run(weights: v.tower.weightBuffer) { k in try k.rmsNorm(xb, out: xb, weight: weight, rows: rows, width: width, eps: eps) }
            assertClose(xb.array(of: Float.self, count: rows * width), want, "vision rmsnorm width \(width) weight \(weight?.info.name ?? "none")")
        }
    }

    func testAddPositionRows() throws {
        let v = try vision()
        let env = try XCTUnwrap(Self.env)
        var rng = SeededRandom(seed: 15)
        let d = v.config.info.hiddenSize
        let count = v.weights.positionCount
        let rows = 300
        let h = rng.floats(rows * d, scale: 2)
        let px = (0..<rows).map { Int32(($0 * 37) % count) }
        let py = (0..<rows).map { Int32(count - 1 - ($0 * 11) % count) }
        var want = h
        var row = [Float](repeating: 0, count: d)
        for t in 0..<rows {
            v.weights.positionTable.dequantizeRow(Int(px[t]), into: &row)
            for i in 0..<d { want[t * d + i] += row[i] }
            v.weights.positionTable.dequantizeRow(count + Int(py[t]), into: &row)
            for i in 0..<d { want[t * d + i] += row[i] }
        }
        let hb = try buffer(h)
        let xb = try env.context.makeBuffer(array: px, label: "px"), yb = try env.context.makeBuffer(array: py, label: "py")
        try run(weights: v.tower.weightBuffer) { k in
            try k.addPositionRows(hb, table: v.weights.positionTable, positionCount: count, patchX: xb, patchY: yb, rows: rows)
        }
        assertClose(hb.array(of: Float.self, count: rows * d), want, "add_position_rows")
    }

    func testRope2D() throws {
        let v = try vision()
        let env = try XCTUnwrap(Self.env)
        var rng = SeededRandom(seed: 16)
        let heads = v.config.headCount, hd = v.config.headDim
        let table = RopeTable(headDim: hd / 2, base: v.config.ropeTheta, positions: v.weights.positionCount)
        for (gx, gy) in [(48, 48), (39, 30), (3, 333)] {
            let P = gx * gy
            let px = (0..<P).map { Int32($0 % gx) }, py = (0..<P).map { Int32($0 / gx) }
            let x = rng.floats(P * heads * hd)
            var want = x
            CPUVisionReference.rope2D(&want, heads: heads, headDim: hd, table: table, patchX: px, patchY: py)
            let xb = try buffer(x)
            let pxb = try env.context.makeBuffer(array: px, label: "px"), pyb = try env.context.makeBuffer(array: py, label: "py")
            let tb = try env.context.makeBuffer(array: table.values, label: "rope2d")
            try run { k in try k.rope2D(xb, patchX: pxb, patchY: pyb, table: tb, tokens: P, heads: heads, headDim: hd) }
            assertClose(xb.array(of: Float.self, count: x.count), want, "rope_2d grid \(gx)x\(gy)")
        }
    }

    func testRmsNormRope2DMatchesNormThenRope2D() throws {
        let v = try vision()
        let env = try XCTUnwrap(Self.env)
        var rng = SeededRandom(seed: 17)
        let heads = v.config.headCount, hd = v.config.headDim, eps = v.config.normEpsilon
        let table = RopeTable(headDim: hd / 2, base: v.config.ropeTheta, positions: v.weights.positionCount)
        let tb = try env.context.makeBuffer(array: table.values, label: "rope2d")
        for (gx, gy, weight) in [(48, 48, v.weights.layers[2].qNorm), (39, 30, v.weights.layers[7].kNorm), (3, 333, v.weights.layers[0].qNorm)] {
            let P = gx * gy
            let px = (0..<P).map { Int32($0 % gx) }, py = (0..<P).map { Int32($0 / gx) }
            let x = rng.floats(P * heads * hd, scale: 4)
            let xb = try buffer(x), sb = try buffer(x)
            let pxb = try env.context.makeBuffer(array: px, label: "px"), pyb = try env.context.makeBuffer(array: py, label: "py")
            try run(weights: v.tower.weightBuffer) { k in
                try k.rmsNorm(sb, out: sb, weight: weight, rows: P * heads, width: hd, eps: eps)
                try k.rope2D(sb, patchX: pxb, patchY: pyb, table: tb, tokens: P, heads: heads, headDim: hd)
            }
            try run(weights: v.tower.weightBuffer) { k in
                try k.rmsNormRope2D(xb, weight: weight, patchX: pxb, patchY: pyb, table: tb, tokens: P, heads: heads, headDim: hd, eps: eps)
            }
            XCTAssertEqual(xb.array(of: Float.self, count: x.count), sb.array(of: Float.self, count: x.count),
                           "fused norm and 2D rope must equal the separate kernels bit for bit, grid \(gx)x\(gy)")
            var want = CPUReference.rmsNorm(x, width: hd, weight: v.reference.matrix(weight), eps: eps)
            CPUVisionReference.rope2D(&want, heads: heads, headDim: hd, table: table, patchX: px, patchY: py)
            assertClose(xb.array(of: Float.self, count: x.count), want, "rmsnorm_rope_2d grid \(gx)x\(gy)")
        }
    }

    func testRmsNormResidualNormMatchesResidualThenNorm() throws {
        let v = try vision()
        var rng = SeededRandom(seed: 18)
        let eps = v.config.normEpsilon, d = v.config.info.hiddenSize
        let l = v.weights.layers[5]
        let rows = 2025
        let x = rng.floats(rows * d, scale: 3), t = rng.floats(rows * d, scale: 5)
        let xb = try buffer(x), tb = try buffer(t), hb = try buffer([Float](repeating: 7, count: rows * d))
        let sxb = try buffer(x), shb = try buffer([Float](repeating: 7, count: rows * d))
        try run(weights: v.tower.weightBuffer) { k in
            try k.rmsNormResidual(sxb, add: tb, weight: l.postAttnNorm, rows: rows, width: d, eps: eps)
            try k.rmsNorm(sxb, out: shb, weight: l.ln2, rows: rows, width: d, eps: eps)
        }
        try run(weights: v.tower.weightBuffer) { k in
            try k.rmsNormResidualNorm(xb, add: tb, weight: l.postAttnNorm, out: hb, outWeight: l.ln2, rows: rows, width: d, eps: eps)
        }
        XCTAssertEqual(xb.array(of: Float.self, count: rows * d), sxb.array(of: Float.self, count: rows * d),
                       "the fused residual update must equal rmsnorm_residual bit for bit")
        XCTAssertEqual(hb.array(of: Float.self, count: rows * d), shb.array(of: Float.self, count: rows * d),
                       "the fused norm must equal rmsnorm_rows over the updated rows bit for bit")
        let updated = zip(x, CPUReference.rmsNorm(t, width: d, weight: v.reference.matrix(l.postAttnNorm), eps: eps)).map { $0 + $1 }
        assertClose(hb.array(of: Float.self, count: rows * d), CPUReference.rmsNorm(updated, width: d, weight: v.reference.matrix(l.ln2), eps: eps),
                    "rmsnorm_residual_norm")
    }

    /// Several images in one dispatch, each pooled from its own patch rows into its own token rows.
    func testPoolPatches() throws {
        let v = try vision()
        let env = try XCTUnwrap(Self.env)
        var rng = SeededRandom(seed: 17)
        let d = v.config.info.hiddenSize, m = v.config.mergeSize
        let grids = [(39, 30), (48, 48), (3, 3), (111, 9), (144, 144)]
        var x: [Float] = [], want: [Float] = [], descriptors: [PoolGrid] = []
        var patches = 0, tokens = 0
        for (gx, gy) in grids {
            let image = rng.floats(gx * gy * d, scale: 4)
            descriptors.append(PoolGrid(patchStart: UInt32(patches), tokenStart: UInt32(tokens), gridX: UInt32(gx), gridY: UInt32(gy)))
            x += image
            want += CPUVisionReference.pool(image, gridX: gx, gridY: gy, merge: m, width: d)
            patches += gx * gy
            tokens += (gx / m) * (gy / m)
        }
        let xb = try buffer(x)
        let gb = try env.context.makeBuffer(array: descriptors, label: "grids")
        let ob = try buffer([Float](repeating: .nan, count: tokens * d))
        let maxTokens = grids.map { ($0.0 / m) * ($0.1 / m) }.max()!
        try run { k in try k.poolPatches(xb, grids: gb, images: grids.count, maxTokens: maxTokens, out: ob, width: d, merge: m) }
        assertClose(ob.array(of: Float.self, count: tokens * d), want, "pool_patches grids \(grids)")
    }
}
