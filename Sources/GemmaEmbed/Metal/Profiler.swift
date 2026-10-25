import Foundation
import Metal

/// Timing instruments for the kernels, used by the CLI's `--profile` mode and by autotune work.
public enum Profiler {
    public struct Sample: Sendable {
        public let name: String
        public let seconds: Double
        public let note: String
    }

    /// Times each kernel class over the shapes of a real packed batch, by encoding `reps`
    /// dispatches of that kernel alone in one command buffer and reading GPU timestamps.
    public static func profile(model: EmbeddingModel, texts: [String], reps: Int = 10) throws -> [Sample] {
        let forward = model.forward
        let config = forward.config
        let weights = forward.weights
        var batch = PackedBatch()
        for t in texts { batch.append(model.tokenizer.encode(TaskPrefix.document.prefix + t)) }
        let N = batch.tokenCount
        let padded = MetalForward.paddedRows(N)
        let ws = try forward.workspace(forRows: padded)
        ws.tokens.write(batch.tokens)
        ws.positions.write(batch.positions)
        ws.sequences.write(batch.sequences)
        let work = batch.attentionWork(queriesPerGroup: forward.schedule.attentionQueries)
        ws.work.write(work)
        let info = config.info
        let d = info.hiddenSize
        let eps = info.normEpsilon
        var samples: [Sample] = []

        func time(_ name: String, note: String, _ body: (Kernels) throws -> Void) throws {
            guard let cb = forward.context.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return }
            enc.useResource(forward.weightBuffer, usage: .read)
            let k = Kernels(forward: forward, encoder: enc)
            for _ in 0..<reps { try body(k) }
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            samples.append(Sample(name: name, seconds: (cb.gpuEndTime - cb.gpuStartTime) / Double(reps), note: note))
        }

        var tiny = PackedBatch()
        tiny.append(model.tokenizer.encode("warm up"))
        for i in 0..<3 {
            let t0 = Date()
            _ = try forward.run(tiny)
            let t1 = Date()
            let gpuTiny = try forward.timeForward(tiny)
            samples.append(Sample(name: "tiny batch run \(i)", seconds: t1.timeIntervalSince(t0), note: String(format: "gpu %.1f ms", gpuTiny * 1000)))
        }
        let wall0 = Date()
        _ = try forward.run(batch)
        let wall = Date().timeIntervalSince(wall0)
        let gpu = try forward.timeForward(batch)
        samples.append(Sample(name: "whole forward (wall)", seconds: wall, note: "\(N) tokens, \(batch.sequences.count) sequences"))
        samples.append(Sample(name: "whole forward (gpu)", seconds: gpu, note: "\(N) tokens"))

        let l0 = weights.layers[0], l5 = weights.layers[5]
        try time("gemm q8 512->1024 (q, sliding)", note: "20 sliding layers") { k in try k.gemm(ws.h, weight: l0.wq, out: ws.q, rows: padded, outputs: 1024, inputs: d) }
        try time("gemm q8 512->512 (k/v/gate/proj)", note: "x6 per layer") { k in try k.gemm(ws.h, weight: l0.wk, out: ws.k, rows: padded, outputs: 512, inputs: d) }
        try time("gemm q8 1024->512 (o, sliding)", note: "") { k in try k.gemm(ws.attn, weight: l0.wo, out: ws.tmp, rows: padded, outputs: d, inputs: 1024) }
        try time("gemm q8 512->2048 (ffn gate/up)", note: "x2 per layer") { k in try k.gemm(ws.h, weight: l0.wGate, out: ws.gate, rows: padded, outputs: 2048, inputs: d) }
        try time("gemm q8 2048->512 (ffn down)", note: "") { k in try k.gemm(ws.gate, weight: l0.wDown, out: ws.tmp, rows: padded, outputs: d, inputs: 2048) }
        try time("gemm q8 512->2048 (q, global)", note: "x4 layers") { k in try k.gemm(ws.h, weight: l5.wq, out: ws.q, rows: padded, outputs: 2048, inputs: d) }
        try time("gemm bf16 512->512 (per-layer proj)", note: "") { k in try k.gemm(ws.x0, weight: weights.perLayerProj, weightRowOffset: 0, out: ws.tmp, rows: padded, outputs: 512, inputs: d) }
        try time("rmsnorm 512", note: "x1 per layer") { k in try k.rmsNorm(ws.x, out: ws.h, weight: l0.attnNorm, rows: N, width: d, eps: eps) }
        try time("rmsnorm heads 256", note: "") { k in try k.rmsNorm(ws.q, out: ws.q, weight: l0.qNorm, rows: N * 4, width: 256, eps: eps) }
        let rope = forward.ropeTable(headDim: 256)
        try time("rope q 4x256", note: "") { k in try k.rope(ws.q, positions: ws.positions, table: rope.buffer, pairs: rope.table.pairs, tokens: N, heads: 4, headDim: 256) }
        try time("rmsnorm+rope q 4x256", note: "fused") { k in try k.rmsNormRope(ws.q, weight: l0.qNorm, positions: ws.positions, table: rope.buffer, pairs: rope.table.pairs, tokens: N, heads: 4, headDim: 256, eps: eps) }
        // The attention kernel's time depends on the score distribution, so it runs on the forward's own q, k and v.
        for s in try stageAttention(forward: forward, batch: batch) {
            try time(s.name, note: s.note) { k in
                try k.attention(q: s.q, k: s.k, v: s.v, out: ws.attn, work: ws.work, workItems: work.count, heads: info.headCount,
                                kvHeads: s.layer.kvHeadCount, headDim: s.layer.headDim,
                                halfWindow: s.layer.isSlidingWindow ? info.slidingWindow / 2 : 0, scale: 1)
            }
        }
        try time("mean pool", note: "") { k in try k.meanPool(ws.out, sequences: ws.sequences, count: batch.sequences.count, out: ws.pooled, width: 768) }
        try time("embed", note: "x0 and x") { k in try k.embed(table: weights.tokenEmbedding, tokens: ws.tokens, out: ws.x0, residual: ws.x, tokenCount: N, paddedRows: padded, scale: 1) }
        return samples
    }

    /// One layer's attention inputs from a forward over the profiled batch, in buffers no other timing writes.
    struct StagedAttention {
        let name: String
        let note: String
        let layer: LayerConfig
        let q: MTLBuffer
        let k: MTLBuffer
        let v: MTLBuffer
    }

    /// The attention inputs of the first sliding layer and the first global layer in a forward over `batch`.
    static func stageAttention(forward: MetalForward, batch: PackedBatch) throws -> [StagedAttention] {
        let layers = forward.config.layers
        return try [true, false].compactMap { sliding in
            let kind = layers.filter { $0.isSlidingWindow == sliding }
            guard let layer = kind.first else { return nil }
            let (q, k, v) = try forward.attentionInputs(batch, layer: layer.index)
            return StagedAttention(name: "attention \(sliding ? "sliding" : "global") hd\(layer.headDim)", note: "\(kind.count) layers",
                                   layer: layer, q: q, k: k, v: v)
        }
    }
}

extension Profiler {
    /// Times each vision tower kernel class over a real batch of images, plus the whole image
    /// forward and the vision tower alone.
    public static func profileImages(model: EmbeddingModel, urls: [URL], maxImageTokens: Int = ImageEmbedOptions.defaultMaxImageTokens,
                                     reps: Int = 5) throws -> [Sample] {
        let forward = model.forward
        guard let tower = forward.vision, let config = model.visionConfig else { throw GemmaEmbedError.imageEncoderNotLoaded }
        let preprocessor = ImagePreprocessor(config: config)
        let images = try urls.map { try preprocessor.prepare(.url($0), maxImageTokens: maxImageTokens) }
        let batch = VisionBatch(images: images)
        let P = batch.patchCount, T = batch.tokenCount
        let patchRows = MetalForward.paddedRows(P)
        let d = config.info.hiddenSize, ff = config.feedForwardSize
        let heads = config.headCount, hd = config.headDim, eps = config.normEpsilon
        let layers = tower.weights.layers.count
        let w = tower.weights.layers[0]
        var samples: [Sample] = []

        var whole: [Double] = []
        for _ in 0..<(reps + 1) { whole.append(try forward.timeForward(images: images)) }
        let (ws, workItems) = try tower.upload(batch, context: forward.context)
        let vectors = try forward.workspace(forRows: MetalForward.paddedRows(T + 4 * images.count)).vectors

        func time(_ name: String, note: String, _ body: (Kernels) throws -> Void) throws {
            guard let cb = forward.context.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return }
            enc.useResource(tower.weightBuffer, usage: .read)
            let k = Kernels(forward: forward, encoder: enc, weights: tower.weightBuffer)
            for _ in 0..<reps { try body(k) }
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            samples.append(Sample(name: name, seconds: (cb.gpuEndTime - cb.gpuStartTime) / Double(reps), note: note))
        }

        samples.append(Sample(name: "image forward (gpu)", seconds: whole.dropFirst().min()!,
                              note: "\(images.count) images, \(P) patches, \(T) tokens"))
        try time("vision tower", note: "\(layers) layers") { k in
            try tower.encode(batch, workItems: workItems, workspace: ws, forward: forward, encoder: k.encoder, vectors: vectors)
        }
        try time("gemm patch embedding", note: "once") { k in
            try k.gemm(ws.patches, weight: tower.weights.patchEmbedding, out: ws.x, rows: patchRows, outputs: d, inputs: config.patchArea)
        }
        try time("gemm \(d)->\(d) (q/k/v/o)", note: "o, x1 per layer") { k in
            try k.gemm(ws.h, weight: w.wq, out: ws.q, rows: patchRows, outputs: d, inputs: d)
        }
        try time("gemm \(d)->3x\(d) (q, k, v in one)", note: "x1 per layer") { k in
            try k.gemm3(ws.h, weights: (w.wq, w.wk, w.wv), outs: (ws.q, ws.k, ws.v), rows: patchRows, outputs: d, inputs: d)
        }
        try time("gemm \(d)->\(ff) (gate)", note: "x1 per layer") { k in
            try k.gemm(ws.h, weight: w.wGate, out: ws.gate, rows: patchRows, outputs: ff, inputs: d)
        }
        try time("gemm \(d)->\(ff) (up, gelu-quick)", note: "x1 per layer") { k in
            try k.gemm(ws.h, weight: w.wUp, out: ws.up, rows: patchRows, outputs: ff, inputs: d, epilogue: .gegluQuick(gate: ws.gate))
        }
        try time("gemm \(ff)->\(d) (down)", note: "x1 per layer") { k in
            try k.gemm(ws.up, weight: w.wDown, out: ws.tmp, rows: patchRows, outputs: d, inputs: ff)
        }
        try time("rmsnorm \(d)", note: "x2 per layer") { k in try k.rmsNorm(ws.x, out: ws.h, weight: w.ln1, rows: P, width: d, eps: eps) }
        try time("rmsnorm heads \(hd)", note: "x3 per layer") { k in
            try k.rmsNorm(ws.q, out: ws.q, weight: w.qNorm, rows: P * heads, width: hd, eps: eps)
        }
        try time("rope 2d", note: "x2 per layer") { k in
            try k.rope2D(ws.q, patchX: ws.patchX, patchY: ws.patchY, table: tower.rope.buffer, tokens: P, heads: heads, headDim: hd)
        }
        try time("rmsnorm+rope 2d q \(heads)x\(hd)", note: "fused, x2 per layer") { k in
            try k.rmsNormRope2D(ws.q, weight: w.qNorm, patchX: ws.patchX, patchY: ws.patchY, table: tower.rope.buffer, tokens: P, heads: heads, headDim: hd, eps: eps)
        }
        try time("attention hd\(hd)", note: "x1 per layer") { k in
            try k.attention(q: ws.q, k: ws.k, v: ws.v, out: ws.attn, work: ws.work, workItems: workItems,
                            heads: heads, kvHeads: heads, headDim: hd, halfWindow: 0, scale: 1)
        }
        try time("rmsnorm residual \(d)", note: "x2 per layer") { k in
            try k.rmsNormResidual(ws.x, add: ws.tmp, weight: w.postAttnNorm, rows: P, width: d, eps: eps)
        }
        try time("rmsnorm residual+norm \(d)", note: "fused, x2 per layer") { k in
            try k.rmsNormResidualNorm(ws.x, add: ws.tmp, weight: w.postAttnNorm, out: ws.h, outWeight: w.ln2, rows: P, width: d, eps: eps)
        }
        try time("pool, norm and projection", note: "once") { k in
            try k.poolPatches(ws.x, grids: ws.grids, images: batch.images.count, maxTokens: batch.maxTokensPerImage, out: ws.pooled,
                              width: d, merge: config.mergeSize)
            try k.rmsNorm(ws.pooled, out: ws.pooled, weight: nil, rows: T, width: d, eps: eps)
            try k.gemm(ws.pooled, weight: tower.weights.projection, out: vectors, rows: MetalForward.paddedRows(T),
                       outputs: config.info.projectionDim, inputs: d)
        }
        return samples
    }
}
