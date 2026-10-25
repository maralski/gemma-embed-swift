import Foundation
import Metal

/// Where one chunk's rows sit in an audio batch: its mel frames, the rows after the first
/// convolution and its tokens, all packed back to back across the batch's chunks.
struct AudioChunkPlacement {
    let frameStart: Int
    let frames: Int
    let halfStart: Int
    let halfFrames: Int
    let tokenStart: Int
    let tokens: Int
}

/// Sounds packed back to back for the audio tower, chunk after chunk; each sound's tokens are the
/// consecutive tokens of its chunks.
struct AudioBatch {
    let audios: [PreparedAudio]
    let chunks: [AudioChunkPlacement]
    /// Token range of each sound.
    let tokenRanges: [Range<Int>]
    let frameCount: Int
    let halfFrameCount: Int
    let tokenCount: Int
    let melBins: Int

    init(audios: [PreparedAudio]) {
        var chunks: [AudioChunkPlacement] = []
        var ranges: [Range<Int>] = []
        var frames = 0, halves = 0, tokens = 0
        for audio in audios {
            let start = tokens
            for chunk in audio.chunks {
                let half = (chunk.frames - 1) / 2 + 1
                chunks.append(AudioChunkPlacement(frameStart: frames, frames: chunk.frames, halfStart: halves, halfFrames: half,
                                                  tokenStart: tokens, tokens: chunk.tokenCount))
                frames += chunk.frames
                halves += half
                tokens += chunk.tokenCount
            }
            ranges.append(start..<tokens)
        }
        self.audios = audios
        self.chunks = chunks
        tokenRanges = ranges
        frameCount = frames
        halfFrameCount = halves
        tokenCount = tokens
        melBins = audios.first?.chunks.first?.melBins ?? 0
    }
}

/// Activation buffers of the audio tower for up to `frameCapacity` mel frames and `tokenCapacity`
/// padded token rows.
final class AudioWorkspace {
    let frameCapacity: Int
    let tokenCapacity: Int
    let mel: MTLBuffer
    /// Output of the first convolution, `[halfFrames][melBins / 2][128]`.
    let half: MTLBuffer
    /// Output of the second convolution flattened to `[tokens][32 * 32]`, the Conformer's input.
    let flat: MTLBuffer
    let x: MTLBuffer
    let h: MTLBuffer
    let q: MTLBuffer
    let k: MTLBuffer
    let v: MTLBuffer
    let attn: MTLBuffer
    let tmp: MTLBuffer
    let tmp2: MTLBuffer
    let up: MTLBuffer
    let wide: MTLBuffer
    let out: MTLBuffer
    /// First token row of every token's chunk, for the causal window and the depthwise convolution.
    let chunkStart: MTLBuffer

    init(context: MetalContext, config: AudioConfig, frames: Int, tokenRows: Int) throws {
        frameCapacity = frames
        tokenCapacity = tokenRows
        let rows = tokenRows + MetalForward.slackRows
        let d = config.info.hiddenSize
        let halfFrames = (frames - 1) / 2 + 1
        func floats(_ n: Int, _ label: String) throws -> MTLBuffer { try context.makeBuffer(bytes: n * 4, label: label) }
        mel = try floats(frames * config.info.melBins, "amel")
        half = try floats(halfFrames * (config.info.melBins / 2) * config.subsamplingChannels[0], "ahalf")
        flat = try floats(rows * config.subsamplingOutputWidth, "aflat")
        x = try floats(rows * d, "ax")
        h = try floats(rows * d, "ah")
        q = try floats(rows * d, "aq")
        k = try floats(rows * d, "ak")
        v = try floats(rows * d, "av")
        attn = try floats(rows * d, "aattn")
        tmp = try floats(rows * d, "atmp")
        tmp2 = try floats(rows * d, "atmp2")
        up = try floats(rows * config.info.feedForwardSize, "aup")
        wide = try floats(rows * 2 * d, "awide")
        out = try floats(rows * config.outputWidth, "aout")
        chunkStart = try context.makeBuffer(bytes: rows * 4, label: "achunkStart")
    }
}

/// The gemma4a audio encoder on the GPU: its weights, the projected relative-position keys of
/// every layer, the activation buffers of the last batch, and the markers that frame its rows.
final class AudioTower: @unchecked Sendable {
    let config: AudioConfig
    let weights: AudioWeights
    let markers: SpanMarkers
    /// One buffer over the whole mapped encoder file; tensors are offsets into it.
    let weightBuffer: MTLBuffer
    /// `[layer][distance][hidden]`: `W_krel` applied to the sinusoid of every distance, from the CPU reference.
    let relativeKeys: MTLBuffer
    /// Clamp bounds of every clipped linear, two floats each, indexed by `ClippedLinear` name.
    private var workspace: AudioWorkspace?

    init(context: MetalContext, config: AudioConfig, weights: AudioWeights, markers: SpanMarkers) throws {
        self.config = config
        self.weights = weights
        self.markers = markers
        let mapping = weights.file.mapping
        guard let wb = context.device.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(mutating: mapping.base),
                                                 length: mapping.mappedLength, options: .storageModeShared, deallocator: nil) else {
            throw GemmaEmbedError.gpuFailure("cannot wrap the audio encoder mapping in a Metal buffer")
        }
        wb.label = "audioWeights"
        weightBuffer = wb
        let reference = CPUAudioReference(config: config, weights: weights)
        relativeKeys = try context.makeBuffer(array: weights.layers.flatMap { reference.relativeKeys($0) }, label: "audioRelativeKeys")
        for layer in weights.layers {
            // q, k and v share one clamped input and run as one grouped GEMM, so their bounds must agree.
            precondition(layer.wq.inputMin == layer.wk.inputMin && layer.wq.inputMin == layer.wv.inputMin
                         && layer.wq.inputMax == layer.wk.inputMax && layer.wq.inputMax == layer.wv.inputMax
                         && layer.wq.outputMin == layer.wk.outputMin && layer.wq.outputMin == layer.wv.outputMin
                         && layer.wq.outputMax == layer.wk.outputMax && layer.wq.outputMax == layer.wv.outputMax,
                         "the q, k and v clamp bounds of a layer differ; the grouped projection assumes they agree")
        }
    }

    /// Writes the batch's mel frames and chunk table into a workspace large enough for it.
    func upload(_ batch: AudioBatch, context: MetalContext) throws -> AudioWorkspace {
        let frames = max(batch.frameCount, 1)
        let tokenRows = MetalForward.paddedRows(batch.tokenCount)
        let ws: AudioWorkspace
        if let current = workspace, current.frameCapacity >= frames, current.tokenCapacity >= tokenRows {
            ws = current
        } else {
            ws = try AudioWorkspace(context: context, config: config, frames: max(frames, workspace?.frameCapacity ?? 0),
                                    tokenRows: max(tokenRows, workspace?.tokenCapacity ?? 0))
            workspace = ws
        }
        precondition(batch.melBins == config.info.melBins)
        var chunkStart = [UInt32](repeating: 0, count: tokenRows)
        for (audio, chunks) in zip(batch.audios, batch.chunkPlacements()) {
            for (chunk, placement) in zip(audio.chunks, chunks) {
                ws.mel.write(chunk.values, offset: placement.frameStart * config.info.melBins * 4)
                for t in placement.tokenStart..<(placement.tokenStart + placement.tokens) { chunkStart[t] = UInt32(placement.tokenStart) }
            }
        }
        ws.chunkStart.write(chunkStart)
        // Padding rows go through every GEMM; zeroing them keeps stale values out of the batch.
        let width = config.subsamplingOutputWidth
        memset(ws.flat.contents() + batch.tokenCount * width * 4, 0, (tokenRows - batch.tokenCount) * width * 4)
        return ws
    }

    /// Encodes the tower over `batch`, writing its `tokenCount` projected rows into `vectors`
    /// from row `vectorRowOffset`.
    func encode(_ batch: AudioBatch, workspace ws: AudioWorkspace, forward: MetalForward, encoder: MTLComputeCommandEncoder,
                vectors: MTLBuffer, vectorRowOffset: Int) throws {
        let k = Kernels(forward: forward, encoder: encoder, weights: weightBuffer)
        let d = config.info.hiddenSize
        let heads = config.info.headCount, hd = config.headDim
        let eps = config.normEpsilon
        let T = batch.tokenCount
        let rows = MetalForward.paddedRows(T)
        let mel = config.info.melBins

        for placement in batch.chunks {
            let halfFreq = (mel - 1) / 2 + 1
            try k.audioSubsample(ws.mel, weight: weights.subsampling[0].weight, norm: weights.subsampling[0].norm, out: ws.half,
                                 inTime: placement.frames, inFreq: mel, inChannels: 1, inOffset: placement.frameStart,
                                 outChannels: config.subsamplingChannels[0], outOffset: placement.halfStart, eps: eps)
            try k.audioSubsample(ws.half, weight: weights.subsampling[1].weight, norm: weights.subsampling[1].norm, out: ws.flat,
                                 inTime: placement.halfFrames, inFreq: halfFreq, inChannels: config.subsamplingChannels[0], inOffset: placement.halfStart,
                                 outChannels: config.subsamplingChannels[1], outOffset: placement.tokenStart, eps: eps)
        }
        try linear(k, ws.flat, weights.inputProjection, out: ws.x, rows: rows, tokens: T)

        for (index, w) in weights.layers.enumerated() {
            try halfStepFeedForward(k, ws, norm: w.ffnNorm, up: w.ffnUp, down: w.ffnDown, postNorm: w.ffnPostNorm, rows: rows, tokens: T)

            try k.rmsNorm(ws.x, out: ws.h, weight: w.attnPreNorm, rows: T, width: d, eps: eps)
            try k.audioElementwise(ws.h, rows: T, width: d, lo: w.wq.inputMin, hi: w.wq.inputMax)
            try k.gemm3(ws.h, weights: (w.wq.weight, w.wk.weight, w.wv.weight), outs: (ws.q, ws.k, ws.v), rows: rows, outputs: d, inputs: d)
            try k.together {
                try k.audioElementwise(ws.q, rows: T, width: d, lo: w.wq.outputMin, hi: w.wq.outputMax)
                try k.audioElementwise(ws.k, rows: T, width: d, lo: w.wk.outputMin, hi: w.wk.outputMax)
                try k.audioElementwise(ws.v, rows: T, width: d, lo: w.wv.outputMin, hi: w.wv.outputMax)
            }
            try k.together {
                try k.audioElementwise(ws.q, rows: T, width: d, scale: config.queryScale, perDim: w.perDimScale, perDimWidth: hd)
                try k.audioElementwise(ws.k, rows: T, width: d, scale: config.keyScale)
            }
            try k.audioAttention(q: ws.q, k: ws.k, v: ws.v, relativeKeys: relativeKeys, layer: index, chunkStart: ws.chunkStart, out: ws.attn,
                                 tokens: T, heads: heads, headDim: hd, window: config.attentionWindow, softcap: config.softcap)
            try linear(k, ws.attn, w.wo, out: ws.tmp, rows: rows, tokens: T)
            try k.rmsNormResidual(ws.x, add: ws.tmp, weight: w.attnPostNorm, rows: T, width: d, eps: eps)

            try k.rmsNorm(ws.x, out: ws.h, weight: w.convPreNorm, rows: T, width: d, eps: eps)
            try linear(k, ws.h, w.convPointwise1, out: ws.wide, rows: rows, tokens: T)
            try k.audioGLU(ws.wide, out: ws.tmp, rows: T, width: d)
            try k.audioDepthwise(ws.tmp, taps: w.convDepthwise, chunkStart: ws.chunkStart, out: ws.tmp2, rows: T, width: d, kernel: config.depthwiseKernel)
            try k.rmsNorm(ws.tmp2, out: ws.tmp2, weight: w.convDepthwiseNorm, rows: T, width: d, eps: eps)
            try k.audioSiLU(ws.tmp2, rows: T, width: d)
            try linear(k, ws.tmp2, w.convPointwise2, out: ws.tmp, rows: rows, tokens: T)
            try k.audioAdd(ws.x, ws.tmp, rows: T, width: d, scale: 1)

            try halfStepFeedForward(k, ws, norm: w.ffnNorm1, up: w.ffnUp1, down: w.ffnDown1, postNorm: w.ffnPostNorm1, rows: rows, tokens: T)
            try k.rmsNorm(ws.x, out: ws.x, weight: w.outputNorm, rows: T, width: d, eps: eps)
        }

        try linear(k, ws.x, weights.outputProjection, out: ws.out, rows: rows, tokens: T)
        try k.rmsNorm(ws.out, out: ws.out, weight: nil, rows: T, width: config.outputWidth, eps: eps)
        try linear(k, ws.out, weights.projection, out: vectors, outRowOffset: vectorRowOffset, rows: rows, tokens: T)
    }

    /// `x = x + 0.5 * rmsnorm(down(silu(up(rmsnorm(x)))))`, each linear clipped.
    private func halfStepFeedForward(_ k: Kernels, _ ws: AudioWorkspace, norm: WeightTensor, up: ClippedLinear, down: ClippedLinear,
                                     postNorm: WeightTensor, rows: Int, tokens T: Int) throws {
        let d = config.info.hiddenSize
        try k.rmsNorm(ws.x, out: ws.h, weight: norm, rows: T, width: d, eps: config.normEpsilon)
        try linear(k, ws.h, up, out: ws.up, rows: rows, tokens: T)
        try k.audioSiLU(ws.up, rows: T, width: up.weight.rows)
        try linear(k, ws.up, down, out: ws.tmp, rows: rows, tokens: T)
        try k.rmsNorm(ws.tmp, out: ws.tmp2, weight: postNorm, rows: T, width: d, eps: config.normEpsilon)
        try k.audioAdd(ws.x, ws.tmp2, rows: T, width: d, scale: config.feedForwardResidualScale)
    }

    /// A clipped linear: clamps `x` in place when the layer clamps its input, multiplies, clamps
    /// the output and adds the bias. `x` is consumed by this linear alone.
    private func linear(_ k: Kernels, _ x: MTLBuffer, _ l: ClippedLinear, out: MTLBuffer, outRowOffset: Int = 0, rows: Int, tokens T: Int) throws {
        let inputs = l.weight.columns, outputs = l.weight.rows
        if l.clampsInput { try k.audioElementwise(x, rows: T, width: inputs, lo: l.inputMin, hi: l.inputMax) }
        try k.gemm(x, weight: l.weight, out: out, outRowOffset: outRowOffset, rows: rows, outputs: outputs, inputs: inputs)
        let outOffset = outRowOffset * outputs * 4
        if l.clampsOutput { try k.audioElementwise(out, offset: outOffset, rows: T, width: outputs, lo: l.outputMin, hi: l.outputMax) }
        if let bias = l.bias { try k.audioAddBias(out, offset: outOffset, bias: bias, rows: T, width: outputs) }
    }
}

extension AudioBatch {
    /// The placements grouped by sound, in order.
    func chunkPlacements() -> [[AudioChunkPlacement]] {
        var result: [[AudioChunkPlacement]] = []
        var next = 0
        for audio in audios {
            result.append(Array(chunks[next..<(next + audio.chunks.count)]))
            next += audio.chunks.count
        }
        return result
    }
}

extension Kernels {
    private func elementCount(_ rows: Int, _ width: Int) -> MTLSize { MTLSize(width: rows * width, height: 1, depth: 1) }
    private var elementGroup: MTLSize { MTLSize(width: 256, height: 1, depth: 1) }

    func audioSubsample(_ x: MTLBuffer, weight: WeightTensor, norm: WeightTensor, out: MTLBuffer, inTime: Int, inFreq: Int, inChannels: Int,
                        inOffset: Int, outChannels: Int, outOffset: Int, eps: Float) throws {
        precondition(weight.type == .f32 && norm.type == .f32 && outChannels % 32 == 0 && outChannels <= 1024)
        let outTime = (inTime - 1) / 2 + 1, outFreq = (inFreq - 1) / 2 + 1
        _ = try bindPipeline("audio_subsample")
        encoder.setBuffer(x, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: weight.fileOffset, index: 1)
        encoder.setBuffer(weights, offset: norm.fileOffset, index: 2)
        encoder.setBuffer(out, offset: 0, index: 3)
        setParams(AudioConvParams(inTime: UInt32(inTime), inFreq: UInt32(inFreq), inChannels: UInt32(inChannels), outTime: UInt32(outTime),
                                  outFreq: UInt32(outFreq), outChannels: UInt32(outChannels), inOffset: UInt32(inOffset), outOffset: UInt32(outOffset), eps: eps), index: 4)
        dispatchGroups(MTLSize(width: outFreq, height: outTime, depth: 1), perGroup: MTLSize(width: outChannels, height: 1, depth: 1))
    }

    /// `x = clamp(x * scale * perDim[i % perDimWidth], lo, hi)` over `rows` rows of `width`.
    func audioElementwise(_ x: MTLBuffer, offset: Int = 0, rows: Int, width: Int, scale: Float = 1, perDim: WeightTensor? = nil, perDimWidth: Int = 0,
                          lo: Float = -.greatestFiniteMagnitude, hi: Float = .greatestFiniteMagnitude) throws {
        guard rows > 0 else { return }
        _ = try bindPipeline("audio_elementwise")
        encoder.setBuffer(x, offset: offset, index: 0)
        encoder.setBuffer(weights, offset: perDim?.fileOffset ?? 0, index: 1)
        setParams(AudioElementwiseParams(rows: UInt32(rows), width: UInt32(width), perDimWidth: UInt32(perDim == nil ? 0 : perDimWidth),
                                         scale: scale, lo: lo, hi: hi), index: 2)
        dispatchThreads(elementCount(rows, width), perGroup: elementGroup)
    }

    func audioSiLU(_ x: MTLBuffer, rows: Int, width: Int) throws {
        guard rows > 0 else { return }
        _ = try bindPipeline("audio_silu")
        encoder.setBuffer(x, offset: 0, index: 0)
        setParams(AudioRowsParams(rows: UInt32(rows), width: UInt32(width), taps: 0, scale: 1), index: 1)
        dispatchThreads(elementCount(rows, width), perGroup: elementGroup)
    }

    /// `x += scale * t`
    func audioAdd(_ x: MTLBuffer, _ t: MTLBuffer, rows: Int, width: Int, scale: Float) throws {
        guard rows > 0 else { return }
        _ = try bindPipeline("audio_add")
        encoder.setBuffer(x, offset: 0, index: 0)
        encoder.setBuffer(t, offset: 0, index: 1)
        setParams(AudioRowsParams(rows: UInt32(rows), width: UInt32(width), taps: 0, scale: scale), index: 2)
        dispatchThreads(elementCount(rows, width), perGroup: elementGroup)
    }

    func audioAddBias(_ x: MTLBuffer, offset: Int = 0, bias: WeightTensor, rows: Int, width: Int) throws {
        guard rows > 0 else { return }
        precondition(bias.type == .f32 && bias.columns == width)
        _ = try bindPipeline("audio_add_bias")
        encoder.setBuffer(x, offset: offset, index: 0)
        encoder.setBuffer(weights, offset: bias.fileOffset, index: 1)
        setParams(AudioRowsParams(rows: UInt32(rows), width: UInt32(width), taps: 0, scale: 1), index: 2)
        dispatchThreads(elementCount(rows, width), perGroup: elementGroup)
    }

    /// `out[t][i] = x[t][i] * sigmoid(x[t][width + i])`
    func audioGLU(_ x: MTLBuffer, out: MTLBuffer, rows: Int, width: Int) throws {
        guard rows > 0 else { return }
        _ = try bindPipeline("audio_glu")
        encoder.setBuffer(x, offset: 0, index: 0)
        encoder.setBuffer(out, offset: 0, index: 1)
        setParams(AudioRowsParams(rows: UInt32(rows), width: UInt32(width), taps: 0, scale: 1), index: 2)
        dispatchThreads(elementCount(rows, width), perGroup: elementGroup)
    }

    func audioDepthwise(_ x: MTLBuffer, taps: WeightTensor, chunkStart: MTLBuffer, out: MTLBuffer, rows: Int, width: Int, kernel: Int) throws {
        guard rows > 0 else { return }
        precondition(taps.type == .f32 && taps.columns == kernel && taps.rows == width)
        _ = try bindPipeline("audio_dwconv")
        encoder.setBuffer(x, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: taps.fileOffset, index: 1)
        encoder.setBuffer(chunkStart, offset: 0, index: 2)
        encoder.setBuffer(out, offset: 0, index: 3)
        setParams(AudioRowsParams(rows: UInt32(rows), width: UInt32(width), taps: UInt32(kernel), scale: 1), index: 4)
        dispatchThreads(elementCount(rows, width), perGroup: elementGroup)
    }

    func audioAttention(q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, relativeKeys: MTLBuffer, layer: Int, chunkStart: MTLBuffer, out: MTLBuffer,
                        tokens: Int, heads: Int, headDim: Int, window: Int, softcap: Float) throws {
        guard tokens > 0 else { return }
        precondition(window <= 32)
        _ = try bindPipeline("audio_attention")
        encoder.setBuffer(q, offset: 0, index: 0)
        encoder.setBuffer(k, offset: 0, index: 1)
        encoder.setBuffer(v, offset: 0, index: 2)
        encoder.setBuffer(relativeKeys, offset: layer * (window + 1) * heads * headDim * 4, index: 3)
        encoder.setBuffer(chunkStart, offset: 0, index: 4)
        encoder.setBuffer(out, offset: 0, index: 5)
        setParams(AudioAttentionParams(tokens: UInt32(tokens), heads: UInt32(heads), headDim: UInt32(headDim), window: UInt32(window), softcap: softcap), index: 6)
        dispatchThreads(MTLSize(width: tokens, height: heads, depth: 1), perGroup: MTLSize(width: 32, height: heads, depth: 1))
    }
}
