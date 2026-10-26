// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Static facts about a loaded model.
public struct ModelInfo: Sendable {
    public let architecture: String
    public let name: String
    public let layerCount: Int
    /// Width of the residual stream.
    public let hiddenSize: Int
    /// Width of the final embedding vector.
    public let outputDimensions: Int
    public let feedForwardSize: Int
    public let perLayerInputSize: Int
    public let headCount: Int
    public let vocabularySize: Int
    public let slidingWindow: Int
    /// Longest sequence the model supports; longer inputs are truncated.
    public let maxTokens: Int
    public let normEpsilon: Float
    /// Storage type of the projection weights, for example `Q8_0`.
    public let quantization: String
    /// SHA256 of the weight file, when computed at load.
    public let fileSHA256: String?
    public let deviceName: String
    public let fileSize: Int
    /// The image encoder loaded with the model, or nil when `LoadOptions.imageEncoder` was not set.
    public let imageEncoder: ImageEncoderInfo?
    /// The audio encoder loaded with the model, or nil when `LoadOptions.audioEncoder` was not set.
    public let audioEncoder: AudioEncoderInfo?
}

struct LayerConfig: Sendable {
    let index: Int
    let isSlidingWindow: Bool
    let kvHeadCount: Int
    let headDim: Int
    let ropeBase: Float
}

/// Everything the forward pass needs to know about shapes, derived from GGUF metadata.
struct ModelConfig: Sendable {
    let info: ModelInfo
    let layers: [LayerConfig]
    /// Scale applied to token embeddings: sqrt(hiddenSize).
    let embeddingScale: Float
    /// Scale applied to the per-layer projection: 1 / sqrt(hiddenSize).
    let perLayerScale: Float

    static let supportedArchitecture = "gemma-embedding2"
    static let supportedMaxTokens = 8192

    init(gguf: GGUFFile, deviceName: String, fileSHA256: String?, imageEncoder: ImageEncoderInfo? = nil,
         audioEncoder: AudioEncoderInfo? = nil) throws {
        let arch = gguf.string("general.architecture") ?? ""
        guard arch == ModelConfig.supportedArchitecture else { throw GemmaEmbedError.unsupportedArchitecture(arch) }
        func int(_ key: String) throws -> Int {
            guard let v = gguf.int("\(arch).\(key)") else { throw GemmaEmbedError.missingMetadata("\(arch).\(key)") }
            return v
        }
        func float(_ key: String) throws -> Float {
            guard let v = gguf.float("\(arch).\(key)") else { throw GemmaEmbedError.missingMetadata("\(arch).\(key)") }
            return v
        }
        let layerCount = try int("block_count")
        let hidden = try int("embedding_length")
        let headCount = try int("attention.head_count")
        guard let kvHeads = gguf.metadata["\(arch).attention.head_count_kv"]?.int32ArrayValue, kvHeads.count == layerCount else {
            throw GemmaEmbedError.missingMetadata("\(arch).attention.head_count_kv")
        }
        guard let pattern = gguf.metadata["\(arch).attention.sliding_window_pattern"]?.boolArrayValue, pattern.count == layerCount else {
            throw GemmaEmbedError.missingMetadata("\(arch).attention.sliding_window_pattern")
        }
        let headDimGlobal = try int("attention.key_length")
        let headDimSliding = try int("attention.key_length_swa")
        guard try int("attention.value_length") == headDimGlobal, try int("attention.value_length_swa") == headDimSliding else {
            throw GemmaEmbedError.invalidGGUF("key and value head sizes differ")
        }
        let ropeGlobal = try float("rope.freq_base")
        let ropeSliding = gguf.float("\(arch).rope.freq_base_swa") ?? ropeGlobal
        guard let vocab = gguf.metadata["tokenizer.ggml.tokens"]?.stringArrayValue?.count else { throw GemmaEmbedError.missingMetadata("tokenizer.ggml.tokens") }
        let quant = gguf.tensor(named: "blk.0.attn_q.weight")?.type.name ?? "unknown"
        self.info = ModelInfo(
            architecture: arch,
            name: gguf.string("general.name") ?? arch,
            layerCount: layerCount,
            hiddenSize: hidden,
            outputDimensions: try int("embedding_length_out"),
            feedForwardSize: try int("feed_forward_length"),
            perLayerInputSize: try int("embedding_length_per_layer_input"),
            headCount: headCount,
            vocabularySize: vocab,
            slidingWindow: try int("attention.sliding_window"),
            maxTokens: ModelConfig.supportedMaxTokens,
            normEpsilon: try float("attention.layer_norm_rms_epsilon"),
            quantization: quant,
            fileSHA256: fileSHA256,
            deviceName: deviceName,
            fileSize: gguf.fileSize,
            imageEncoder: imageEncoder,
            audioEncoder: audioEncoder)
        self.layers = (0..<layerCount).map { i in
            LayerConfig(index: i,
                        isSlidingWindow: pattern[i],
                        kvHeadCount: Int(kvHeads[i]),
                        headDim: pattern[i] ? headDimSliding : headDimGlobal,
                        ropeBase: pattern[i] ? ropeSliding : ropeGlobal)
        }
        self.embeddingScale = Float(hidden).squareRoot()
        self.perLayerScale = 1 / Float(hidden).squareRoot()
    }
}
