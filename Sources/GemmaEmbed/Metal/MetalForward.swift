// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// A batch of sequences packed back to back, with no padding between them. A token id below
/// zero, `-(r + 1)`, stands for row `r` of the image vectors buffer, entered unscaled.
struct PackedBatch {
    var tokens: [Int32] = []
    var positions: [Int32] = []
    var sequences: [SequenceRange] = []
    /// Image sequences reference rows `0..<vectorRowCount` of the vectors buffer.
    private(set) var vectorRowCount = 0

    var tokenCount: Int { tokens.count }

    mutating func append(_ ids: [Int32]) {
        sequences.append(SequenceRange(start: UInt32(tokens.count), length: UInt32(ids.count)))
        tokens.append(contentsOf: ids)
        positions.append(contentsOf: (0..<ids.count).map(Int32.init))
    }

    /// The ids of the next `count` unreferenced rows of the vectors buffer, now referenced.
    mutating func reserveVectorRows(_ count: Int) -> [Int32] {
        defer { vectorRowCount += count }
        return (vectorRowCount..<(vectorRowCount + count)).map { -Int32($0) - 1 }
    }

    /// Appends `[bos, begin, v..., end, eos]`, where the `vectorRows` v rows are the next
    /// unreferenced rows of the vectors buffer.
    mutating func appendSpanSequence(markers: SpanMarkers, vectorRows: Int) {
        append([markers.bos, markers.begin] + reserveVectorRows(vectorRows) + [markers.end, markers.eos])
    }

    /// Leaves rows below `row` unreferenced, so a second tower can start its rows on a padded boundary.
    mutating func skipVectorRows(to row: Int) {
        vectorRowCount = max(vectorRowCount, row)
    }

    func attentionWork(queriesPerGroup: Int) -> [AttentionWork] {
        AttentionWork.items(for: sequences, queriesPerGroup: queriesPerGroup)
    }
}

extension AttentionWork {
    /// One work item per block of `queriesPerGroup` query rows of every sequence.
    static func items(for sequences: [SequenceRange], queriesPerGroup: Int) -> [AttentionWork] {
        var work: [AttentionWork] = []
        for s in sequences {
            var q0 = 0
            while q0 < Int(s.length) {
                work.append(AttentionWork(seqStart: s.start, seqLen: s.length, q0: UInt32(q0)))
                q0 += queriesPerGroup
            }
        }
        return work
    }
}

/// Runs the whole forward pass on the GPU for one packed batch.
final class MetalForward: @unchecked Sendable {
    let context: MetalContext
    let config: ModelConfig
    let weights: ModelWeights
    /// One buffer over the whole mapped weight file; tensors are offsets into it.
    let weightBuffer: MTLBuffer
    var schedule: KernelSchedule
    var dispatchType: MTLDispatchType = .concurrent
    /// The image encoder, when one was loaded with the model.
    let vision: VisionTower?
    /// The audio encoder, when one was loaded with the model.
    let audio: AudioTower?
    private let ropeTables: [Int: (table: RopeTable, buffer: MTLBuffer)]
    private var workspace: Workspace?

    /// Rows are padded to this multiple so every GEMM tile and attention block stays in bounds.
    static let rowAlignment = 64
    static let slackRows = (KernelSchedule.attentionQueryCandidates + [KernelSchedule.visionAttentionQueries]).max()!

    init(context: MetalContext, config: ModelConfig, weights: ModelWeights, schedule: KernelSchedule = .default,
         vision: VisionTower? = nil, audio: AudioTower? = nil) throws {
        if let vision, vision.config.info.projectionDim != config.info.hiddenSize {
            throw GemmaEmbedError.invalidGGUF("image encoder projects to \(vision.config.info.projectionDim) dimensions, the text model is \(config.info.hiddenSize) wide")
        }
        if let audio, audio.config.info.projectionDim != config.info.hiddenSize {
            throw GemmaEmbedError.invalidGGUF("audio encoder projects to \(audio.config.info.projectionDim) dimensions, the text model is \(config.info.hiddenSize) wide")
        }
        self.context = context
        self.config = config
        self.weights = weights
        self.schedule = schedule
        self.vision = vision
        self.audio = audio
        let mapping = weights.file.mapping
        guard let wb = context.device.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(mutating: mapping.base),
                                                 length: mapping.mappedLength, options: .storageModeShared, deallocator: nil) else {
            throw GemmaEmbedError.gpuFailure("cannot wrap the weight mapping in a Metal buffer")
        }
        wb.label = "weights"
        self.weightBuffer = wb
        var tables: [Int: (RopeTable, MTLBuffer)] = [:]
        for layer in config.layers where tables[layer.headDim] == nil {
            let table = RopeTable(headDim: layer.headDim, base: layer.ropeBase, positions: config.info.maxTokens)
            tables[layer.headDim] = (table, try context.makeBuffer(array: table.values, label: "rope\(layer.headDim)"))
        }
        self.ropeTables = tables
    }

    /// Activation buffers for up to `capacity` padded rows.
    final class Workspace {
        let capacity: Int
        let x0: MTLBuffer
        let x: MTLBuffer
        let h: MTLBuffer
        let q: MTLBuffer
        let k: MTLBuffer
        let v: MTLBuffer
        let attn: MTLBuffer
        let tmp: MTLBuffer
        let tmp2: MTLBuffer
        let gate: MTLBuffer
        let up: MTLBuffer
        let ple: MTLBuffer
        let out: MTLBuffer
        let tokens: MTLBuffer
        let positions: MTLBuffer
        let sequences: MTLBuffer
        let work: MTLBuffer
        let pooled: MTLBuffer
        /// Rows the encoders hand the text tower: image rows from 0, audio rows from the next
        /// padded row boundary, each tower writing whole padded GEMM tiles.
        let vectors: MTLBuffer

        init(context: MetalContext, config: ModelConfig, capacity: Int) throws {
            self.capacity = capacity
            let rows = capacity + MetalForward.slackRows
            let d = config.info.hiddenSize
            let qDim = config.layers.map { $0.headDim * config.info.headCount }.max()!
            let kvDim = config.layers.map { $0.headDim * $0.kvHeadCount }.max()!
            let ff = config.info.feedForwardSize
            func floats(_ n: Int, _ label: String) throws -> MTLBuffer { try context.makeBuffer(bytes: n * 4, label: label) }
            x0 = try floats(rows * d, "x0")
            x = try floats(rows * d, "x")
            h = try floats(rows * d, "h")
            q = try floats(rows * qDim, "q")
            k = try floats(rows * kvDim, "k")
            v = try floats(rows * kvDim, "v")
            attn = try floats(rows * qDim, "attn")
            tmp = try floats(rows * d, "tmp")
            tmp2 = try floats(rows * d, "tmp2")
            gate = try floats(rows * ff, "gate")
            up = try floats(rows * ff, "up")
            ple = try floats(rows * config.info.perLayerInputSize, "ple")
            out = try floats(rows * config.info.outputDimensions, "out")
            tokens = try context.makeBuffer(bytes: rows * 4, label: "tokens")
            positions = try context.makeBuffer(bytes: rows * 4, label: "positions")
            sequences = try context.makeBuffer(bytes: max(capacity, 1) * MemoryLayout<SequenceRange>.stride, label: "sequences")
            work = try context.makeBuffer(bytes: max(capacity, 1) * MemoryLayout<AttentionWork>.stride, label: "attentionWork")
            pooled = try floats(max(capacity, 1) * config.info.outputDimensions, "pooled")
            vectors = try floats((capacity + 2 * MetalForward.rowAlignment) * d, "vectors")
        }
    }

    func ropeTable(headDim: Int) -> (table: RopeTable, buffer: MTLBuffer) {
        ropeTables[headDim]!
    }

    func workspace(forRows rows: Int) throws -> Workspace {
        if let ws = workspace, ws.capacity >= rows { return ws }
        let ws = try Workspace(context: context, config: config, capacity: rows)
        workspace = ws
        return ws
    }

    static func paddedRows(_ tokens: Int) -> Int {
        max(rowAlignment, (tokens + rowAlignment - 1) / rowAlignment * rowAlignment)
    }

    /// Pooled, unnormalised output vectors, one per sequence in the batch. The batch's vector rows
    /// refer to the vision tower's output over `images`, in order, run in the same command buffer.
    func run(_ batch: PackedBatch, images: VisionBatch? = nil, audio: AudioBatch? = nil) throws -> [[Float]] {
        let count = batch.sequences.count
        guard count > 0 else { return [] }
        let pass = try execute(batch, images: images, audio: audio, trace: "run")
        return pass.workspace.pooled.rows(count, width: config.info.outputDimensions)
    }

    /// GPU time of one forward over the batch, in seconds, with no readback.
    func timeForward(_ batch: PackedBatch, images: VisionBatch? = nil, audio: AudioBatch? = nil) throws -> Double {
        try execute(batch, images: images, audio: audio, trace: nil).gpuSeconds
    }

    /// Pooled, unnormalised output vectors, one per image: the vision tower over every image,
    /// then the text tower over each image's `[bos, <|image>, vectors..., <image|>, eos]`, in one
    /// command buffer.
    func run(images: [PreparedImage]) throws -> [[Float]] {
        guard !images.isEmpty else { return [] }
        let (vision, text) = try imageBatches(images)
        let pass = try execute(text, images: vision, audio: nil, trace: "run images")
        return pass.workspace.pooled.rows(images.count, width: config.info.outputDimensions)
    }

    /// GPU time of one image forward over `images`, vision and text towers, in seconds, with no readback.
    func timeForward(images: [PreparedImage]) throws -> Double {
        let (vision, text) = try imageBatches(images)
        return try execute(text, images: vision, audio: nil, trace: nil).gpuSeconds
    }

    /// The vision tower's output for each image, `tokenCount` rows of the text width, flattened.
    func imageVectors(_ images: [PreparedImage]) throws -> [[Float]] {
        guard !images.isEmpty else { return [] }
        let (vision, text) = try imageBatches(images)
        let pass = try execute(text, images: vision, audio: nil, trace: nil)
        let width = config.info.hiddenSize
        let all = pass.workspace.vectors.array(of: Float.self, count: vision.tokenCount * width)
        return zip(images, vision.grids).map { image, grid in
            let start = Int(grid.tokenStart) * width
            return Array(all[start..<(start + image.tokenCount * width)])
        }
    }

    /// Pooled, unnormalised output vectors, one per sound: the audio tower over every chunk, then
    /// the text tower over each sound's `[bos, <|audio>, rows..., <audio|>, eos]`, in one command buffer.
    func run(audio audios: [PreparedAudio]) throws -> [[Float]] {
        guard !audios.isEmpty else { return [] }
        let (batch, text) = try audioBatches(audios)
        let pass = try execute(text, images: nil, audio: batch, trace: "run audio")
        return pass.workspace.pooled.rows(audios.count, width: config.info.outputDimensions)
    }

    /// GPU time of one audio forward over `audios`, audio and text towers, in seconds, with no readback.
    func timeForward(audio audios: [PreparedAudio]) throws -> Double {
        let (batch, text) = try audioBatches(audios)
        return try execute(text, images: nil, audio: batch, trace: nil).gpuSeconds
    }

    /// The audio tower's output for each sound, `tokenCount` rows of the text width, flattened.
    func audioVectors(_ audios: [PreparedAudio]) throws -> [[Float]] {
        guard !audios.isEmpty else { return [] }
        let (batch, text) = try audioBatches(audios)
        let pass = try execute(text, images: nil, audio: batch, trace: nil)
        let width = config.info.hiddenSize
        let all = pass.workspace.vectors.array(of: Float.self, count: batch.tokenCount * width)
        return batch.tokenRanges.map { Array(all[($0.lowerBound * width)..<($0.upperBound * width)]) }
    }

    private func audioBatches(_ audios: [PreparedAudio]) throws -> (AudioBatch, PackedBatch) {
        guard let audio else { throw GemmaEmbedError.audioEncoderNotLoaded }
        let batch = AudioBatch(audios: audios)
        var text = PackedBatch()
        for a in audios {
            precondition(a.tokenCount + 4 <= config.info.maxTokens, "audio of \(a.tokenCount) tokens exceeds the text context")
            text.appendSpanSequence(markers: audio.markers, vectorRows: a.tokenCount)
        }
        return (batch, text)
    }

    /// The q, k and v that layer `layer`'s attention reads in a forward over `batch`, after their
    /// norms and rope, copied out of the workspace. The forward stops after that attention.
    func attentionInputs(_ batch: PackedBatch, layer: Int) throws -> (q: MTLBuffer, k: MTLBuffer, v: MTLBuffer) {
        let ws = try execute(batch, images: nil, audio: nil, trace: nil, stopAfterAttention: layer).workspace
        func copy(_ b: MTLBuffer) throws -> MTLBuffer {
            try context.makeBuffer(copying: UnsafeRawBufferPointer(start: b.contents(), count: b.length), label: "\(b.label ?? "") layer \(layer)")
        }
        return (try copy(ws.q), try copy(ws.k), try copy(ws.v))
    }

    private func imageBatches(_ images: [PreparedImage]) throws -> (VisionBatch, PackedBatch) {
        guard let vision else { throw GemmaEmbedError.imageEncoderNotLoaded }
        let batch = VisionBatch(images: images)
        var text = PackedBatch()
        for image in images {
            precondition(image.tokenCount + 4 <= config.info.maxTokens, "image of \(image.tokenCount) tokens exceeds the text context")
            text.appendSpanSequence(markers: vision.markers, vectorRows: image.tokenCount)
        }
        return (batch, text)
    }

    private struct Pass {
        let workspace: Workspace
        let gpuSeconds: Double
    }

    /// Where a batch's audio rows start in the vectors buffer: after the image rows, on a padded boundary.
    static func audioVectorRowOffset(imageRows: Int) -> Int {
        imageRows == 0 ? 0 : paddedRows(imageRows)
    }

    /// Uploads and encodes the vision tower over `images` and the audio tower over `audio` when
    /// given, then the text tower over `batch`, whose vector rows refer to the towers' output
    /// (image rows first, audio rows from `audioVectorRowOffset`), and waits for the GPU.
    private func execute(_ batch: PackedBatch, images: VisionBatch?, audio audioBatch: AudioBatch?, trace label: String?,
                         stopAfterAttention: Int? = nil) throws -> Pass {
        let imageRows = images?.tokenCount ?? 0
        let audioRows = audioBatch?.tokenCount ?? 0
        let audioOffset = MetalForward.audioVectorRowOffset(imageRows: imageRows)
        precondition(batch.vectorRowCount == (audioRows == 0 ? imageRows : audioOffset + audioRows))
        var visionPass: (tower: VisionTower, batch: VisionBatch, workspace: VisionWorkspace, workItems: Int)?
        if let images {
            guard let vision else { throw GemmaEmbedError.imageEncoderNotLoaded }
            let (vws, items) = try vision.upload(images, context: context)
            visionPass = (vision, images, vws, items)
        }
        var audioPass: (tower: AudioTower, batch: AudioBatch, workspace: AudioWorkspace)?
        if let audioBatch {
            guard let audio else { throw GemmaEmbedError.audioEncoderNotLoaded }
            audioPass = (audio, audioBatch, try audio.upload(audioBatch, context: context))
        }
        let padded = MetalForward.paddedRows(batch.tokenCount)
        let ws = try workspace(forRows: padded)
        precondition(ws.vectors.length >= (audioOffset + MetalForward.paddedRows(audioRows)) * config.info.hiddenSize * 4)
        ws.tokens.write(batch.tokens)
        ws.positions.write(batch.positions)
        ws.sequences.write(batch.sequences)
        let work = batch.attentionWork(queriesPerGroup: schedule.attentionQueries)
        ws.work.write(work)

        guard let cb = context.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder(dispatchType: dispatchType) else {
            throw GemmaEmbedError.gpuFailure("cannot create a command buffer")
        }
        enc.useResource(weightBuffer, usage: .read)
        let tEncode = Date()
        if let v = visionPass {
            enc.useResource(v.tower.weightBuffer, usage: .read)
            try v.tower.encode(v.batch, workItems: v.workItems, workspace: v.workspace, forward: self, encoder: enc, vectors: ws.vectors)
        }
        if let a = audioPass {
            enc.useResource(a.tower.weightBuffer, usage: .read)
            try a.tower.encode(a.batch, workspace: a.workspace, forward: self, encoder: enc, vectors: ws.vectors, vectorRowOffset: audioOffset)
        }
        try encode(batch: batch, workItems: work.count, paddedRows: padded, workspace: ws, vectors: (visionPass != nil || audioPass != nil) ? ws.vectors : nil,
                   stopAfterAttention: stopAfterAttention, encoder: enc)
        enc.endEncoding()
        let tCommit = Date()
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw GemmaEmbedError.gpuFailure(String(describing: error)) }
        if let label, ProcessInfo.processInfo.environment["GEMMA_EMBED_TRACE"] != nil {
            print(String(format: "\(label): encode %.1f ms, commit+wait %.1f ms, gpu %.1f ms, scheduled-start lag %.1f ms", tCommit.timeIntervalSince(tEncode) * 1000, Date().timeIntervalSince(tCommit) * 1000, (cb.gpuEndTime - cb.gpuStartTime) * 1000, (cb.gpuStartTime - cb.kernelStartTime) * 1000))
        }
        return Pass(workspace: ws, gpuSeconds: cb.gpuEndTime - cb.gpuStartTime)
    }

    /// With `stopAfterAttention`, encodes no further than that layer's attention, so the workspace's
    /// q, k and v hold that layer's attention inputs.
    private func encode(batch: PackedBatch, workItems: Int, paddedRows: Int, workspace ws: Workspace, vectors: MTLBuffer?,
                        stopAfterAttention: Int?, encoder enc: MTLComputeCommandEncoder) throws {
        let info = config.info
        let d = info.hiddenSize
        let N = batch.tokenCount
        let eps = info.normEpsilon
        let k = Kernels(forward: self, encoder: enc)

        try k.embed(table: weights.tokenEmbedding, tokens: ws.tokens, vectors: vectors, out: ws.x0, residual: ws.x, tokenCount: N, paddedRows: paddedRows,
                    scale: config.embeddingScale)

        try k.rmsNorm(ws.x, out: ws.h, weight: weights.layers[0].attnNorm, rows: N, width: d, eps: eps)
        for layer in config.layers {
            let w = weights.layers[layer.index]
            let heads = info.headCount, kvHeads = layer.kvHeadCount, hd = layer.headDim
            let qDim = heads * hd, kvDim = kvHeads * hd

            let pl = info.perLayerInputSize
            try k.together {
                try k.gemm(ws.h, weight: w.wq, out: ws.q, rows: paddedRows, outputs: qDim, inputs: d)
                try k.gemm(ws.h, weight: w.wk, out: ws.k, rows: paddedRows, outputs: kvDim, inputs: d)
                try k.gemm(ws.h, weight: w.wv, out: ws.v, rows: paddedRows, outputs: kvDim, inputs: d)
                try k.gemm(ws.x0, weight: weights.perLayerProj, weightRowOffset: layer.index * pl, out: ws.tmp, rows: paddedRows, outputs: pl, inputs: d)
            }
            let rope = ropeTables[hd]!
            try k.together {
                try k.rmsNormRope(ws.q, weight: w.qNorm, positions: ws.positions, table: rope.buffer, pairs: rope.table.pairs, tokens: N, heads: heads, headDim: hd, eps: eps)
                try k.rmsNormRope(ws.k, weight: w.kNorm, positions: ws.positions, table: rope.buffer, pairs: rope.table.pairs, tokens: N, heads: kvHeads, headDim: hd, eps: eps)
                try k.rmsNorm(ws.v, out: ws.v, weight: nil, rows: N * kvHeads, width: hd, eps: eps)
                try k.rmsNorm(ws.tmp, out: ws.ple, weight: weights.perLayerProjNorm, rows: N, width: pl, eps: eps, preScale: config.perLayerScale)
            }
            try k.attention(q: ws.q, k: ws.k, v: ws.v, out: ws.attn, work: ws.work, workItems: workItems,
                            heads: heads, kvHeads: kvHeads, headDim: hd,
                            halfWindow: layer.isSlidingWindow ? info.slidingWindow / 2 : 0, scale: 1)
            if layer.index == stopAfterAttention { return }
            try k.gemm(ws.attn, weight: w.wo, out: ws.tmp, rows: paddedRows, outputs: d, inputs: qDim)
            try k.rmsNormResidualNorm(ws.x, add: ws.tmp, weight: w.postAttnNorm, out: ws.h, outWeight: w.ffnNorm, rows: N, width: d, eps: eps)

            try k.gemm(ws.h, weight: w.wGate, out: ws.gate, rows: paddedRows, outputs: info.feedForwardSize, inputs: d)
            try k.gemm(ws.h, weight: w.wUp, out: ws.up, rows: paddedRows, outputs: info.feedForwardSize, inputs: d, epilogue: .geglu(gate: ws.gate))
            try k.gemm(ws.up, weight: w.wDown, out: ws.tmp, rows: paddedRows, outputs: d, inputs: info.feedForwardSize)
            try k.rmsNormResidual(ws.x, add: ws.tmp, weight: w.postFfnNorm, rows: N, width: d, eps: eps)

            try k.gemm(ws.x, weight: w.inpGate, out: ws.tmp, rows: paddedRows, outputs: pl, inputs: d, epilogue: .geluTimes(factor: ws.ple))
            try k.gemm(ws.tmp, weight: w.proj, out: ws.tmp2, rows: paddedRows, outputs: d, inputs: pl)
            let nextNorm = layer.index + 1 < config.layers.count ? weights.layers[layer.index + 1].attnNorm : weights.outputNorm
            try k.rmsNormResidualNorm(ws.x, add: ws.tmp2, weight: w.postNorm, out: ws.h, outWeight: nextNorm, rows: N, width: d, eps: eps,
                                      scale: w.outScale)
        }

        try k.gemm(ws.h, weight: weights.output, out: ws.out, rows: paddedRows, outputs: info.outputDimensions, inputs: d)
        try k.meanPool(ws.out, sequences: ws.sequences, count: batch.sequences.count, out: ws.pooled, width: info.outputDimensions)
    }
}

/// Encodes one kernel dispatch per call on a compute encoder.
struct Kernels {
    let forward: MetalForward
    let encoder: MTLComputeCommandEncoder
    /// The buffer every weight tensor's `fileOffset` points into: the text model file by
    /// default, the image encoder file for the vision tower.
    let weights: MTLBuffer

    init(forward: MetalForward, encoder: MTLComputeCommandEncoder, weights: MTLBuffer? = nil) {
        self.forward = forward
        self.encoder = encoder
        self.weights = weights ?? forward.weightBuffer
    }

    private var context: MetalContext { forward.context }

    final class Grouping { var depth = 0 }
    private let grouping = Grouping()

    private func dispatch(threads: MTLSize, perGroup: MTLSize) {
        encoder.dispatchThreads(threads, threadsPerThreadgroup: perGroup)
        barrier()
    }

    private func dispatch(groups: MTLSize, perGroup: MTLSize) {
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: perGroup)
        barrier()
    }

    func dispatchThreads(_ threads: MTLSize, perGroup: MTLSize) { dispatch(threads: threads, perGroup: perGroup) }
    func dispatchGroups(_ groups: MTLSize, perGroup: MTLSize) { dispatch(groups: groups, perGroup: perGroup) }
    func bindPipeline(_ name: String) throws -> MTLComputePipelineState { try bind(name) }
    func setParams<T: BitwiseCopyable>(_ value: T, index: Int) { setBytes(value, index: index) }

    private func barrier() {
        if encoder.dispatchType == .concurrent && grouping.depth == 0 { encoder.memoryBarrier(scope: .buffers) }
    }

    func together(_ body: () throws -> Void) rethrows {
        grouping.depth += 1
        defer {
            grouping.depth -= 1
            barrier()
        }
        try body()
    }

    private func bind(_ name: String) throws -> MTLComputePipelineState {
        let p = try context.pipeline(name)
        encoder.setComputePipelineState(p)
        return p
    }

    private func setBytes<T: BitwiseCopyable>(_ value: T, index: Int) {
        withUnsafeBytes(of: value) { encoder.setBytes($0.baseAddress!, length: $0.count, index: index) }
    }

    func embed(table: WeightTensor, tokens: MTLBuffer, vectors: MTLBuffer? = nil, out: MTLBuffer, residual: MTLBuffer,
               tokenCount: Int, paddedRows: Int, scale: Float) throws {
        _ = try bind("embed_q8")
        encoder.setBuffer(weights, offset: table.fileOffset, index: 0)
        encoder.setBuffer(tokens, offset: 0, index: 1)
        encoder.setBuffer(out, offset: 0, index: 2)
        setBytes(EmbedParams(tokens: UInt32(tokenCount), paddedRows: UInt32(paddedRows), width: UInt32(table.columns), scale: scale), index: 3)
        encoder.setBuffer(vectors ?? out, offset: 0, index: 4)
        encoder.setBuffer(residual, offset: 0, index: 5)
        dispatch(threads: MTLSize(width: table.columns / 32, height: paddedRows, depth: 1), perGroup: MTLSize(width: 16, height: 16, depth: 1))
    }

    func rmsNorm(_ x: MTLBuffer, out: MTLBuffer, weight: WeightTensor?, rows: Int, width: Int, eps: Float, preScale: Float = 1) throws {
        guard rows > 0 else { return }
        _ = try bind("rmsnorm_rows")
        encoder.setBuffer(x, offset: 0, index: 0)
        encoder.setBuffer(out, offset: 0, index: 1)
        encoder.setBuffer(weights, offset: weight?.fileOffset ?? 0, index: 2)
        setBytes(RowParams(rows: UInt32(rows), width: UInt32(width), inStride: UInt32(width), outStride: UInt32(width),
                           eps: eps, preScale: preScale, hasWeight: weight == nil ? 0 : 1), index: 3)
        dispatch(threads: MTLSize(width: rows * 32, height: 1, depth: 1), perGroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    enum Epilogue {
        case store
        /// `out = gelu(gate) * out`
        case geglu(gate: MTLBuffer)
        /// `out = gelu(out) * factor`
        case geluTimes(factor: MTLBuffer)
        /// `out = gelu_quick(gate) * out`
        case gegluQuick(gate: MTLBuffer)
    }

    /// `out = x * weight^T`, optionally followed by an elementwise epilogue.
    func gemm(_ x: MTLBuffer, weight: WeightTensor, weightRowOffset: Int = 0, out: MTLBuffer, outRowOffset: Int = 0, rows: Int, outputs: Int, inputs: Int,
              tile: GemmTile? = nil, epilogue: Epilogue = .store) throws {
        let tile = tile ?? forward.schedule.gemmTile(rows: rows, outputs: outputs, inputs: inputs)
        precondition(rows % tile.rows == 0 && outputs % tile.columns == 0 && inputs % tile.depth == 0 && weight.columns == inputs)
        let format: String
        switch weight.type {
        case .q8_0: format = "q8"
        case .bf16: format = "bf16"
        case .f32: format = "f32"
        case .f16: throw GemmaEmbedError.unsupportedTensorType(weight.info.name, weight.type.rawValue)
        }
        let epilogueName: String
        let aux: MTLBuffer
        switch epilogue {
        case .store: epilogueName = ""; aux = x
        case .geglu(let gate): epilogueName = "geglu_"; aux = gate
        case .geluTimes(let factor): epilogueName = "gelutimes_"; aux = factor
        case .gegluQuick(let gate): epilogueName = "gegluquick_"; aux = gate
        }
        precondition(epilogueName.isEmpty || weight.type == .q8_0)
        _ = try bind("gemm_\(format)_\(epilogueName)\(tile.suffix)")
        encoder.setBuffer(x, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: weight.fileOffset + weightRowOffset * weight.rowBytes, index: 1)
        encoder.setBuffer(out, offset: outRowOffset * outputs * 4, index: 2)
        setBytes(GemmParams(M: UInt32(rows), N: UInt32(outputs), K: UInt32(inputs), weightRowBytes: UInt32(weight.rowBytes)), index: 3)
        encoder.setBuffer(aux, offset: 0, index: 4)
        dispatch(groups: MTLSize(width: outputs / tile.columns, height: rows / tile.rows, depth: 1), perGroup: MTLSize(width: tile.threads, height: 1, depth: 1))
    }

    func gemm3(_ x: MTLBuffer, weights w: (WeightTensor, WeightTensor, WeightTensor), outs: (MTLBuffer, MTLBuffer, MTLBuffer),
               rows: Int, outputs: Int, inputs: Int, tile: GemmTile? = nil) throws {
        let combinedOutputs = 3 * outputs
        let tile = tile ?? forward.schedule.gemmTile(rows: rows, outputs: combinedOutputs)
        precondition(rows % tile.rows == 0 && outputs % tile.columns == 0 && inputs % tile.depth == 0)
        for t in [w.0, w.1, w.2] {
            precondition(t.type == .q8_0 && t.columns == inputs && t.rows == outputs)
        }
        _ = try bind("gemm3_q8_\(tile.suffix)")
        encoder.setBuffer(x, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: w.0.fileOffset, index: 1)
        encoder.setBuffer(outs.0, offset: 0, index: 2)
        setBytes(GemmParams(M: UInt32(rows), N: UInt32(outputs), K: UInt32(inputs), weightRowBytes: UInt32(w.0.rowBytes)), index: 3)
        encoder.setBuffer(weights, offset: w.1.fileOffset, index: 5)
        encoder.setBuffer(weights, offset: w.2.fileOffset, index: 6)
        encoder.setBuffer(outs.1, offset: 0, index: 7)
        encoder.setBuffer(outs.2, offset: 0, index: 8)
        dispatch(groups: MTLSize(width: outputs / tile.columns, height: rows / tile.rows, depth: 3), perGroup: MTLSize(width: tile.threads, height: 1, depth: 1))
    }

    /// `x = (x + rmsnorm(t) * weight) * scale`
    func rmsNormResidual(_ x: MTLBuffer, add t: MTLBuffer, weight: WeightTensor, rows: Int, width: Int, eps: Float, scale: Float = 1) throws {
        guard rows > 0 else { return }
        _ = try bind("rmsnorm_residual")
        encoder.setBuffer(x, offset: 0, index: 0)
        encoder.setBuffer(t, offset: 0, index: 1)
        encoder.setBuffer(weights, offset: weight.fileOffset, index: 2)
        setBytes(RowParams(rows: UInt32(rows), width: UInt32(width), inStride: UInt32(width), outStride: UInt32(width),
                           eps: eps, preScale: scale, hasWeight: 1), index: 3)
        dispatch(threads: MTLSize(width: rows * 32, height: 1, depth: 1), perGroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func rmsNormResidualNorm(_ x: MTLBuffer, add t: MTLBuffer, weight: WeightTensor, out: MTLBuffer, outWeight: WeightTensor,
                             rows: Int, width: Int, eps: Float, scale: Float = 1) throws {
        guard rows > 0 else { return }
        precondition(width == 512 || width == 768)
        _ = try bind("rmsnorm_residual_norm_\(width)")
        let params = RowParams(rows: UInt32(rows), width: UInt32(width), inStride: UInt32(width), outStride: UInt32(width),
                               eps: eps, preScale: 1, hasWeight: 1)
        encoder.setBuffer(x, offset: 0, index: 0)
        encoder.setBuffer(t, offset: 0, index: 1)
        encoder.setBuffer(weights, offset: weight.fileOffset, index: 2)
        var residual = params
        residual.preScale = scale
        setBytes(residual, index: 3)
        encoder.setBuffer(out, offset: 0, index: 4)
        encoder.setBuffer(weights, offset: outWeight.fileOffset, index: 5)
        setBytes(params, index: 6)
        dispatch(threads: MTLSize(width: rows * 32, height: 1, depth: 1), perGroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func rope(_ x: MTLBuffer, positions: MTLBuffer, table: MTLBuffer, pairs: Int, tokens: Int, heads: Int, headDim: Int) throws {
        guard tokens > 0 else { return }
        _ = try bind("rope_neox")
        encoder.setBuffer(x, offset: 0, index: 0)
        encoder.setBuffer(positions, offset: 0, index: 1)
        encoder.setBuffer(table, offset: 0, index: 2)
        setBytes(RopeParams(tokens: UInt32(tokens), heads: UInt32(heads), headDim: UInt32(headDim), stride: UInt32(heads * headDim), tableStride: UInt32(pairs)), index: 3)
        dispatch(threads: MTLSize(width: headDim / 2, height: heads, depth: tokens), perGroup: MTLSize(width: min(headDim / 2, 64), height: 1, depth: 4))
    }

    func rmsNormRope(_ x: MTLBuffer, weight: WeightTensor, positions: MTLBuffer, table: MTLBuffer, pairs: Int, tokens: Int, heads: Int, headDim: Int, eps: Float) throws {
        guard tokens > 0 else { return }
        precondition(headDim == 256 || headDim == 512)
        _ = try bind("rmsnorm_rope_\(headDim)")
        encoder.setBuffer(x, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: weight.fileOffset, index: 1)
        encoder.setBuffer(positions, offset: 0, index: 2)
        encoder.setBuffer(table, offset: 0, index: 3)
        setBytes(RowParams(rows: UInt32(tokens * heads), width: UInt32(headDim), inStride: UInt32(headDim), outStride: UInt32(headDim),
                           eps: eps, preScale: 1, hasWeight: 1), index: 4)
        setBytes(RopeParams(tokens: UInt32(tokens), heads: UInt32(heads), headDim: UInt32(headDim), stride: UInt32(heads * headDim), tableStride: UInt32(pairs)), index: 5)
        dispatch(threads: MTLSize(width: tokens * heads * 32, height: 1, depth: 1), perGroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func attention(q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, out: MTLBuffer, work: MTLBuffer, workItems: Int,
                   heads: Int, kvHeads: Int, headDim: Int, halfWindow: Int, scale: Float, sliceWidth: Int? = nil) throws {
        guard workItems > 0 else { return }
        let queries = forward.schedule.attentionQueries(headDim: headDim)
        let sliceWidth = sliceWidth ?? KernelSchedule.attentionSliceWidth(headDim: headDim, threadgroups: workItems * heads)
        let name = [64: "attention_vision", 128: "attention_fwd_ds128", 256: "attention_fwd", 512: "attention_fwd_hd512"][sliceWidth]!
        precondition(headDim % sliceWidth == 0)
        precondition(headDim != 64 || (kvHeads == heads && halfWindow == 0), "the vision attention kernel has no grouped K/V heads and no window")
        _ = try bind(name)
        encoder.setBuffer(q, offset: 0, index: 0)
        encoder.setBuffer(k, offset: 0, index: 1)
        encoder.setBuffer(v, offset: 0, index: 2)
        encoder.setBuffer(out, offset: 0, index: 3)
        encoder.setBuffer(work, offset: 0, index: 4)
        setBytes(AttentionParams(heads: UInt32(heads), kvHeads: UInt32(kvHeads), headDim: UInt32(headDim),
                                 qStride: UInt32(heads * headDim), kvStride: UInt32(kvHeads * headDim), oStride: UInt32(heads * headDim),
                                 halfWindow: UInt32(halfWindow), queriesPerGroup: UInt32(queries), scale: scale), index: 5)
        let simdgroups = queries / 8
        let tileBytes = sliceWidth == 64 ? KernelSchedule.visionAttentionTileBytes
            : sliceWidth >= 256 ? KernelSchedule.attentionOccupancyCapBytes : KernelSchedule.attentionTileBytes
        encoder.setThreadgroupMemoryLength(tileBytes, index: 0)
        let slices = headDim / sliceWidth
        dispatch(groups: MTLSize(width: workItems, height: heads, depth: slices), perGroup: MTLSize(width: simdgroups * 32, height: 1, depth: 1))
    }

    /// `rope_neox` over each half of every head: the first half by `patchX`, the second by `patchY`.
    /// `table` has `headDim / 4` pairs per position.
    func rope2D(_ x: MTLBuffer, patchX: MTLBuffer, patchY: MTLBuffer, table: MTLBuffer, tokens: Int, heads: Int, headDim: Int) throws {
        guard tokens > 0 else { return }
        _ = try bind("rope_2d")
        encoder.setBuffer(x, offset: 0, index: 0)
        encoder.setBuffer(patchX, offset: 0, index: 1)
        encoder.setBuffer(patchY, offset: 0, index: 2)
        encoder.setBuffer(table, offset: 0, index: 3)
        setBytes(RopeParams(tokens: UInt32(tokens), heads: UInt32(heads), headDim: UInt32(headDim), stride: UInt32(heads * headDim),
                            tableStride: UInt32(headDim / 4)), index: 4)
        dispatch(threads: MTLSize(width: headDim / 2, height: heads, depth: tokens), perGroup: MTLSize(width: headDim / 2, height: 1, depth: 4))
    }

    func rmsNormRope2D(_ x: MTLBuffer, weight: WeightTensor, patchX: MTLBuffer, patchY: MTLBuffer, table: MTLBuffer,
                       tokens: Int, heads: Int, headDim: Int, eps: Float) throws {
        guard tokens > 0 else { return }
        precondition(headDim == 64)
        _ = try bind("rmsnorm_rope_2d_64")
        encoder.setBuffer(x, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: weight.fileOffset, index: 1)
        encoder.setBuffer(patchX, offset: 0, index: 2)
        encoder.setBuffer(patchY, offset: 0, index: 3)
        encoder.setBuffer(table, offset: 0, index: 4)
        setBytes(RowParams(rows: UInt32(tokens * heads), width: UInt32(headDim), inStride: UInt32(headDim), outStride: UInt32(headDim),
                           eps: eps, preScale: 1, hasWeight: 1), index: 5)
        setBytes(RopeParams(tokens: UInt32(tokens), heads: UInt32(heads), headDim: UInt32(headDim), stride: UInt32(heads * headDim),
                            tableStride: UInt32(headDim / 4)), index: 6)
        dispatch(threads: MTLSize(width: tokens * heads * 32, height: 1, depth: 1), perGroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// `x[t] += table[patchX[t]] + table[positionCount + patchY[t]]` over `rows` rows as wide as the table.
    func addPositionRows(_ x: MTLBuffer, table: WeightTensor, positionCount: Int, patchX: MTLBuffer, patchY: MTLBuffer, rows: Int) throws {
        guard rows > 0 else { return }
        precondition(table.type == .f32 && table.rows == 2 * positionCount)
        _ = try bind("add_position_rows")
        encoder.setBuffer(x, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: table.fileOffset, index: 1)
        encoder.setBuffer(patchX, offset: 0, index: 2)
        encoder.setBuffer(patchY, offset: 0, index: 3)
        setBytes(PositionParams(rows: UInt32(rows), width: UInt32(table.columns), positionCount: UInt32(positionCount)), index: 4)
        dispatch(threads: MTLSize(width: table.columns, height: rows, depth: 1), perGroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Pools every `merge` by `merge` window of each image's patch rows into one token row, scaled by `sqrt(width)`.
    func poolPatches(_ x: MTLBuffer, grids: MTLBuffer, images: Int, maxTokens: Int, out: MTLBuffer, width: Int, merge: Int) throws {
        guard images > 0, maxTokens > 0 else { return }
        _ = try bind("pool_patches")
        encoder.setBuffer(x, offset: 0, index: 0)
        encoder.setBuffer(grids, offset: 0, index: 1)
        encoder.setBuffer(out, offset: 0, index: 2)
        setBytes(PoolPatchParams(width: UInt32(width), merge: UInt32(merge), scale: Float(width).squareRoot() / Float(merge * merge)), index: 3)
        dispatch(threads: MTLSize(width: width, height: maxTokens, depth: images), perGroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func meanPool(_ x: MTLBuffer, sequences: MTLBuffer, count: Int, out: MTLBuffer, width: Int) throws {
        _ = try bind("mean_pool")
        encoder.setBuffer(x, offset: 0, index: 0)
        encoder.setBuffer(sequences, offset: 0, index: 1)
        encoder.setBuffer(out, offset: 0, index: 2)
        setBytes(PoolParams(width: UInt32(width), stride: UInt32(width)), index: 3)
        dispatch(groups: MTLSize(width: count, height: 1, depth: 1), perGroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}
