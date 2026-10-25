import Foundation

/// A linear layer whose input and output llama.cpp clamps (`Gemma4ClippableLinear`): the four
/// bounds sit beside the weight as `.input_min`, `.input_max`, `.output_min` and `.output_max`
/// scalars, and default to the float range when absent.
struct ClippedLinear: Sendable {
    let weight: WeightTensor
    let bias: WeightTensor?
    let inputMin: Float
    let inputMax: Float
    let outputMin: Float
    let outputMax: Float

    var clampsInput: Bool { inputMin > -.greatestFiniteMagnitude || inputMax < .greatestFiniteMagnitude }
    var clampsOutput: Bool { outputMin > -.greatestFiniteMagnitude || outputMax < .greatestFiniteMagnitude }
}

struct AudioLayerWeights: Sendable {
    let ffnNorm: WeightTensor
    let ffnUp: ClippedLinear
    let ffnDown: ClippedLinear
    let ffnPostNorm: WeightTensor
    let attnPreNorm: WeightTensor
    let wq: ClippedLinear
    let wk: ClippedLinear
    let wv: ClippedLinear
    let wo: ClippedLinear
    /// Projects the sinusoidal relative positions to per-head keys.
    let kRel: ClippedLinear
    /// `[headDim]`, applied to the scaled queries as stored (the converter folded the softplus).
    let perDimScale: WeightTensor
    let attnPostNorm: WeightTensor
    /// The norm before the convolution module. The GGUF stores it as `conv_norm`: llama.cpp's
    /// loader reads `conv_norm` into its pre-norm slot and `norm_conv` into the post-depthwise slot.
    let convPreNorm: WeightTensor
    let convPointwise1: ClippedLinear
    /// `[hidden][5]` depthwise taps, tap 4 on the current frame.
    let convDepthwise: WeightTensor
    /// The RMS norm after the depthwise convolution (`norm_conv` in the GGUF).
    let convDepthwiseNorm: WeightTensor
    let convPointwise2: ClippedLinear
    let ffnNorm1: WeightTensor
    let ffnUp1: ClippedLinear
    let ffnDown1: ClippedLinear
    let ffnPostNorm1: WeightTensor
    let outputNorm: WeightTensor
}

/// The audio tensors of the mmproj file (`a.*`, `mm.a.*`), shape-checked against the configuration.
struct AudioWeights: Sendable {
    /// Keeps the mapping alive for as long as any tensor pointer is in use.
    let file: GGUFFile
    /// Two subsampling convolutions, each `[outChannels][inChannels][3][3]` fp32 (GGUF `[3, 3, in, out]`), with a LayerNorm weight per channel.
    let subsampling: [(weight: WeightTensor, norm: WeightTensor)]
    /// `a.input_projection.weight`, `[hidden][32 * 32]` fp32.
    let inputProjection: ClippedLinear
    let layers: [AudioLayerWeights]
    /// `a.pre_encode.out`, `[1536][hidden]` with a bias.
    let outputProjection: ClippedLinear
    /// `mm.a.input_projection.weight`, `[projectionDim][1536]`.
    let projection: ClippedLinear

    init(gguf: GGUFFile, config: AudioConfig) throws {
        func info(_ name: String, _ shape: [Int]?, types: [GGUFTensorType]) throws -> GGUFTensorInfo {
            guard let info = gguf.tensor(named: name) else { throw GemmaEmbedError.missingTensor(name) }
            if let shape, info.shape != shape { throw GemmaEmbedError.tensorShapeMismatch(name, expected: shape, found: info.shape) }
            guard types.contains(info.type) else { throw GemmaEmbedError.unsupportedTensorType(name, info.type.rawValue) }
            return info
        }
        func view(_ info: GGUFTensorInfo, as shape: [Int]) -> WeightTensor {
            let viewed = GGUFTensorInfo(name: info.name, shape: shape, type: info.type, offset: info.offset)
            precondition(viewed.elementCount == info.elementCount)
            return WeightTensor(info: viewed, base: gguf.bytes(of: info).baseAddress!, fileOffset: gguf.fileOffset(of: info))
        }
        func tensor(_ name: String, _ shape: [Int], types: [GGUFTensorType] = [.q8_0, .f32, .bf16, .f16]) throws -> WeightTensor {
            let i = try info(name, shape, types: types)
            return view(i, as: i.shape)
        }
        func scalar(_ name: String, default value: Float) throws -> Float {
            guard gguf.tensor(named: name) != nil else { return value }
            return try tensor(name, [1], types: [.f32]).scalar()
        }
        /// `base.weight` of shape `[inputs, outputs]` with its optional bias and clamp scalars.
        func linear(_ base: String, inputs: Int, outputs: Int, types: [GGUFTensorType] = [.q8_0, .f32, .bf16, .f16]) throws -> ClippedLinear {
            ClippedLinear(weight: try tensor("\(base).weight", [inputs, outputs], types: types),
                          bias: gguf.tensor(named: "\(base).bias") == nil ? nil : try tensor("\(base).bias", [outputs], types: [.f32]),
                          inputMin: try scalar("\(base).input_min", default: -.greatestFiniteMagnitude),
                          inputMax: try scalar("\(base).input_max", default: .greatestFiniteMagnitude),
                          outputMin: try scalar("\(base).output_min", default: -.greatestFiniteMagnitude),
                          outputMax: try scalar("\(base).output_max", default: .greatestFiniteMagnitude))
        }
        self.file = gguf
        let d = config.info.hiddenSize
        let ff = config.info.feedForwardSize
        let k = config.subsamplingKernel
        var subsampling: [(weight: WeightTensor, norm: WeightTensor)] = []
        var inChannels = 1
        for (i, outChannels) in config.subsamplingChannels.enumerated() {
            let w = try info("a.conv1d.\(i).weight", [k, k, inChannels, outChannels], types: [.f32])
            subsampling.append((view(w, as: [k * k * inChannels, outChannels]), try tensor("a.conv1d.\(i).norm.weight", [outChannels], types: [.f32])))
            inChannels = outChannels
        }
        self.subsampling = subsampling
        inputProjection = try linear("a.input_projection", inputs: config.subsamplingOutputWidth, outputs: d)
        let outWidth = try info("a.pre_encode.out.weight", nil, types: [.q8_0, .f32, .bf16, .f16]).shape
        guard outWidth.count == 2, outWidth[0] == d else {
            throw GemmaEmbedError.invalidGGUF("a.pre_encode.out.weight has shape \(outWidth), expected [\(d), n]")
        }
        outputProjection = try linear("a.pre_encode.out", inputs: d, outputs: outWidth[1])
        projection = try linear("mm.a.input_projection", inputs: outWidth[1], outputs: config.info.projectionDim)
        layers = try (0..<config.info.layerCount).map { i in
            let b = "a.blk.\(i)"
            return AudioLayerWeights(
                ffnNorm: try tensor("\(b).ffn_norm.weight", [d], types: [.f32]),
                ffnUp: try linear("\(b).ffn_up", inputs: d, outputs: ff),
                ffnDown: try linear("\(b).ffn_down", inputs: ff, outputs: d),
                ffnPostNorm: try tensor("\(b).ffn_post_norm.weight", [d], types: [.f32]),
                attnPreNorm: try tensor("\(b).attn_pre_norm.weight", [d], types: [.f32]),
                wq: try linear("\(b).attn_q", inputs: d, outputs: d),
                wk: try linear("\(b).attn_k", inputs: d, outputs: d),
                wv: try linear("\(b).attn_v", inputs: d, outputs: d),
                wo: try linear("\(b).attn_out", inputs: d, outputs: d),
                kRel: try linear("\(b).attn_k_rel", inputs: d, outputs: d),
                perDimScale: try tensor("\(b).per_dim_scale.weight", [config.headDim], types: [.f32]),
                attnPostNorm: try tensor("\(b).attn_post_norm.weight", [d], types: [.f32]),
                convPreNorm: try tensor("\(b).conv_norm.weight", [d], types: [.f32]),
                convPointwise1: try linear("\(b).conv_pw1", inputs: d, outputs: 2 * d),
                convDepthwise: try tensor("\(b).conv_dw.weight", [config.depthwiseKernel, d], types: [.f32]),
                convDepthwiseNorm: try tensor("\(b).norm_conv.weight", [d], types: [.f32]),
                convPointwise2: try linear("\(b).conv_pw2", inputs: d, outputs: d),
                ffnNorm1: try tensor("\(b).ffn_norm_1.weight", [d], types: [.f32]),
                ffnUp1: try linear("\(b).ffn_up_1", inputs: d, outputs: ff),
                ffnDown1: try linear("\(b).ffn_down_1", inputs: ff, outputs: d),
                ffnPostNorm1: try tensor("\(b).ffn_post_norm_1.weight", [d], types: [.f32]),
                outputNorm: try tensor("\(b).ln2.weight", [d], types: [.f32]))
        }
    }
}
