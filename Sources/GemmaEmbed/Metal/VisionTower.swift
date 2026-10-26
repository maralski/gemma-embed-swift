// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// Images packed back to back as patch rows for the vision tower, and the token rows their
/// pooled vectors fill, in the same order.
struct VisionBatch {
    let images: [PreparedImage]
    /// The patch rows of each image; attention stays within one.
    let sequences: [SequenceRange]
    let grids: [PoolGrid]
    let patchCount: Int
    let tokenCount: Int
    let maxTokensPerImage: Int

    init(images: [PreparedImage]) {
        var sequences: [SequenceRange] = []
        var grids: [PoolGrid] = []
        var patches = 0, tokens = 0
        for image in images {
            sequences.append(SequenceRange(start: UInt32(patches), length: UInt32(image.patchCount)))
            grids.append(PoolGrid(patchStart: UInt32(patches), tokenStart: UInt32(tokens), gridX: UInt32(image.gridX), gridY: UInt32(image.gridY)))
            patches += image.patchCount
            tokens += image.tokenCount
        }
        self.images = images
        self.sequences = sequences
        self.grids = grids
        patchCount = patches
        tokenCount = tokens
        maxTokensPerImage = images.map(\.tokenCount).max() ?? 0
    }
}

/// Activation buffers of the vision tower for up to `patchCapacity` padded patch rows and
/// `tokenCapacity` padded token rows.
final class VisionWorkspace {
    let patchCapacity: Int
    let tokenCapacity: Int
    let patches: MTLBuffer
    let x: MTLBuffer
    let h: MTLBuffer
    let q: MTLBuffer
    let k: MTLBuffer
    let v: MTLBuffer
    let attn: MTLBuffer
    let tmp: MTLBuffer
    let gate: MTLBuffer
    let up: MTLBuffer
    let pooled: MTLBuffer
    let patchX: MTLBuffer
    let patchY: MTLBuffer
    let sequences: MTLBuffer
    let work: MTLBuffer
    let grids: MTLBuffer

    init(context: MetalContext, config: VisionConfig, patchRows: Int, tokenRows: Int) throws {
        patchCapacity = patchRows
        tokenCapacity = tokenRows
        let rows = patchRows + MetalForward.slackRows
        let d = config.info.hiddenSize
        func floats(_ n: Int, _ label: String) throws -> MTLBuffer { try context.makeBuffer(bytes: n * 4, label: label) }
        patches = try floats(rows * config.patchArea, "patches")
        x = try floats(rows * d, "vx")
        h = try floats(rows * d, "vh")
        q = try floats(rows * d, "vq")
        k = try floats(rows * d, "vk")
        v = try floats(rows * d, "vv")
        attn = try floats(rows * d, "vattn")
        tmp = try floats(rows * d, "vtmp")
        gate = try floats(rows * config.feedForwardSize, "vgate")
        up = try floats(rows * config.feedForwardSize, "vup")
        pooled = try floats(tokenRows * d, "vpooled")
        patchX = try context.makeBuffer(bytes: rows * 4, label: "patchX")
        patchY = try context.makeBuffer(bytes: rows * 4, label: "patchY")
        // Every image has at least one patch and every work item at least one query row.
        sequences = try context.makeBuffer(bytes: patchRows * MemoryLayout<SequenceRange>.stride, label: "imageSequences")
        work = try context.makeBuffer(bytes: patchRows * MemoryLayout<AttentionWork>.stride, label: "imageAttentionWork")
        grids = try context.makeBuffer(bytes: patchRows * MemoryLayout<PoolGrid>.stride, label: "poolGrids")
    }
}

/// The gemma4v image encoder on the GPU: its weights, its 2D RoPE table, the activation
/// buffers of the last batch, and the markers that frame its vectors in a text sequence.
final class VisionTower: @unchecked Sendable {
    let config: VisionConfig
    let weights: VisionWeights
    let markers: SpanMarkers
    /// One buffer over the whole mapped image encoder file; tensors are offsets into it.
    let weightBuffer: MTLBuffer
    let rope: (table: RopeTable, buffer: MTLBuffer)
    private var workspace: VisionWorkspace?

    init(context: MetalContext, config: VisionConfig, weights: VisionWeights, markers: SpanMarkers) throws {
        self.config = config
        self.weights = weights
        self.markers = markers
        let mapping = weights.file.mapping
        guard let wb = context.device.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(mutating: mapping.base),
                                                 length: mapping.mappedLength, options: .storageModeShared, deallocator: nil) else {
            throw GemmaEmbedError.gpuFailure("cannot wrap the image encoder mapping in a Metal buffer")
        }
        wb.label = "visionWeights"
        weightBuffer = wb
        // Patch columns and rows index the position tables, so the tables bound both.
        let table = RopeTable(headDim: config.headDim / 2, base: config.ropeTheta, positions: weights.positionCount)
        rope = (table, try context.makeBuffer(array: table.values, label: "rope2d"))
    }

    /// Writes the batch's patches and descriptors into a workspace large enough for it, and
    /// returns the workspace with the number of attention work items.
    func upload(_ batch: VisionBatch, context: MetalContext) throws -> (VisionWorkspace, Int) {
        let patchRows = MetalForward.paddedRows(batch.patchCount)
        let tokenRows = MetalForward.paddedRows(batch.tokenCount)
        let ws: VisionWorkspace
        if let current = workspace, current.patchCapacity >= patchRows, current.tokenCapacity >= tokenRows {
            ws = current
        } else {
            ws = try VisionWorkspace(context: context, config: config, patchRows: max(patchRows, workspace?.patchCapacity ?? 0),
                                     tokenRows: max(tokenRows, workspace?.tokenCapacity ?? 0))
            workspace = ws
        }
        let area = config.patchArea
        for (image, seq) in zip(batch.images, batch.sequences) {
            precondition(image.mergeSize == config.mergeSize && image.patches.count == image.patchCount * area)
            precondition(max(image.gridX, image.gridY) <= weights.positionCount)
            let row = Int(seq.start)
            ws.patches.write(image.patches, offset: row * area * 4)
            ws.patchX.write(image.patchX, offset: row * 4)
            ws.patchY.write(image.patchY, offset: row * 4)
        }
        // Padding rows go through every GEMM; zeroing them keeps stale values out of the batch.
        memset(ws.patches.contents() + batch.patchCount * area * 4, 0, (patchRows - batch.patchCount) * area * 4)
        ws.sequences.write(batch.sequences)
        ws.grids.write(batch.grids)
        let work = AttentionWork.items(for: batch.sequences, queriesPerGroup: KernelSchedule.visionAttentionQueries)
        ws.work.write(work)
        return (ws, work.count)
    }

    /// Encodes the tower over `batch`, writing its `tokenCount` projected rows into `vectors` from row 0.
    func encode(_ batch: VisionBatch, workItems: Int, workspace ws: VisionWorkspace, forward: MetalForward,
                encoder: MTLComputeCommandEncoder, vectors: MTLBuffer) throws {
        let k = Kernels(forward: forward, encoder: encoder, weights: weightBuffer)
        let d = config.info.hiddenSize
        let ff = config.feedForwardSize
        let heads = config.headCount, hd = config.headDim
        let eps = config.normEpsilon
        let P = batch.patchCount, T = batch.tokenCount
        let patchRows = MetalForward.paddedRows(P)

        try k.gemm(ws.patches, weight: weights.patchEmbedding, out: ws.x, rows: patchRows, outputs: d, inputs: config.patchArea)
        try k.addPositionRows(ws.x, table: weights.positionTable, positionCount: weights.positionCount, patchX: ws.patchX, patchY: ws.patchY, rows: P)

        try k.rmsNorm(ws.x, out: ws.h, weight: weights.layers[0].ln1, rows: P, width: d, eps: eps)
        for (index, w) in weights.layers.enumerated() {
            try k.gemm3(ws.h, weights: (w.wq, w.wk, w.wv), outs: (ws.q, ws.k, ws.v), rows: patchRows, outputs: d, inputs: d)
            try k.together {
                try k.rmsNormRope2D(ws.q, weight: w.qNorm, patchX: ws.patchX, patchY: ws.patchY, table: rope.buffer, tokens: P, heads: heads, headDim: hd, eps: eps)
                try k.rmsNormRope2D(ws.k, weight: w.kNorm, patchX: ws.patchX, patchY: ws.patchY, table: rope.buffer, tokens: P, heads: heads, headDim: hd, eps: eps)
                try k.rmsNorm(ws.v, out: ws.v, weight: nil, rows: P * heads, width: hd, eps: eps)
            }
            try k.attention(q: ws.q, k: ws.k, v: ws.v, out: ws.attn, work: ws.work, workItems: workItems,
                            heads: heads, kvHeads: heads, headDim: hd, halfWindow: 0, scale: 1)
            try k.gemm(ws.attn, weight: w.wo, out: ws.tmp, rows: patchRows, outputs: d, inputs: d)
            try k.rmsNormResidualNorm(ws.x, add: ws.tmp, weight: w.postAttnNorm, out: ws.h, outWeight: w.ln2, rows: P, width: d, eps: eps)

            try k.gemm(ws.h, weight: w.wGate, out: ws.gate, rows: patchRows, outputs: ff, inputs: d)
            try k.gemm(ws.h, weight: w.wUp, out: ws.up, rows: patchRows, outputs: ff, inputs: d, epilogue: .gegluQuick(gate: ws.gate))
            try k.gemm(ws.up, weight: w.wDown, out: ws.tmp, rows: patchRows, outputs: d, inputs: ff)
            if index + 1 < weights.layers.count {
                try k.rmsNormResidualNorm(ws.x, add: ws.tmp, weight: w.postFfnNorm, out: ws.h, outWeight: weights.layers[index + 1].ln1,
                                          rows: P, width: d, eps: eps)
            } else {
                try k.rmsNormResidual(ws.x, add: ws.tmp, weight: w.postFfnNorm, rows: P, width: d, eps: eps)
            }
        }

        try k.poolPatches(ws.x, grids: ws.grids, images: batch.images.count, maxTokens: batch.maxTokensPerImage, out: ws.pooled,
                          width: d, merge: config.mergeSize)
        try k.rmsNorm(ws.pooled, out: ws.pooled, weight: nil, rows: T, width: d, eps: eps)
        try k.gemm(ws.pooled, weight: weights.projection, out: vectors, rows: MetalForward.paddedRows(T),
                   outputs: config.info.projectionDim, inputs: d)
    }
}
