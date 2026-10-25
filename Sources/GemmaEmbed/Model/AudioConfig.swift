import Foundation

/// Static facts about a loaded audio encoder.
public struct AudioEncoderInfo: Sendable {
    public let layerCount: Int
    /// Width of the Conformer residual stream.
    public let hiddenSize: Int
    public let headCount: Int
    public let feedForwardSize: Int
    /// Mel filters per 10 ms frame.
    public let melBins: Int
    /// Width of the vectors handed to the text tower.
    public let projectionDim: Int
    public let fileSize: Int
    /// SHA256 of the encoder file, when computed at load.
    public let fileSHA256: String?
}

/// Shapes and constants of a gemma4a audio encoder. The GGUF carries the sizes under
/// `clip.audio.*`; the rest is hard-coded in llama.cpp (`clip.cpp` and `models/gemma4a.cpp`).
struct AudioConfig: Sendable {
    let info: AudioEncoderInfo

    /// llama.cpp hard-codes 1e-6 because the GGUF's `layer_norm_epsilon` was written wrong by the converter.
    let normEpsilon: Float = 1e-6
    /// Every query attends to itself and the 11 frames before it.
    let attentionWindow = 12
    /// Relative positions the sinusoidal table covers: distances 0 to `attentionWindow`.
    let relativePositions = 13
    let softcap: Float = 50
    /// Half-step feed-forward residual weight.
    let feedForwardResidualScale: Float = 0.5
    let depthwiseKernel = 5
    let subsamplingChannels = [128, 32]
    let subsamplingKernel = 3

    var headDim: Int { info.hiddenSize / info.headCount }
    /// `(1 / sqrt(headDim)) / ln 2` on the queries and `softplus(1) / ln 2` on the keys.
    var queryScale: Float { (1 / Float(headDim).squareRoot()) / logf(2) }
    var keyScale: Float { logf(1 + expf(1)) / logf(2) }
    /// Width of the flattened subsampling output that enters the Conformer: 32 frequency rows of 32 channels.
    var subsamplingOutputWidth: Int { (info.melBins / 4) * subsamplingChannels[1] }
    /// Width of `a.pre_encode.out`, the projection the multimodal embedder reads.
    let outputWidth: Int

    static let supportedArchitecture = "clip"
    static let supportedProjector = "gemma4a"

    init(gguf: GGUFFile, fileSHA256: String?) throws {
        let arch = gguf.string("general.architecture") ?? ""
        guard arch == AudioConfig.supportedArchitecture else {
            throw GemmaEmbedError.invalidGGUF("audio encoder architecture is '\(arch)', expected \(AudioConfig.supportedArchitecture)")
        }
        guard gguf.metadata["clip.has_audio_encoder"]?.boolValue == true else {
            throw GemmaEmbedError.invalidGGUF("the mmproj file has no audio encoder (clip.has_audio_encoder)")
        }
        let projector = gguf.string("clip.audio.projector_type") ?? ""
        guard projector == AudioConfig.supportedProjector else {
            throw GemmaEmbedError.invalidGGUF("audio projector is '\(projector)', expected \(AudioConfig.supportedProjector)")
        }
        func int(_ key: String) throws -> Int {
            guard let v = gguf.int("clip.audio.\(key)") else { throw GemmaEmbedError.missingMetadata("clip.audio.\(key)") }
            return v
        }
        let hidden = try int("embedding_length")
        let heads = try int("attention.head_count")
        guard heads > 0, hidden % heads == 0 else { throw GemmaEmbedError.invalidGGUF("audio width \(hidden) does not split into \(heads) heads") }
        let mel = try int("num_mel_bins")
        guard mel % 4 == 0 else { throw GemmaEmbedError.invalidGGUF("\(mel) mel bins do not survive two stride-2 convolutions evenly") }
        guard let outShape = gguf.tensor(named: "a.pre_encode.out.weight")?.shape, outShape.count == 2, outShape[0] == hidden else {
            throw GemmaEmbedError.invalidGGUF("a.pre_encode.out.weight is missing or not [\(hidden), n]")
        }
        outputWidth = outShape[1]
        info = AudioEncoderInfo(
            layerCount: try int("block_count"),
            hiddenSize: hidden,
            headCount: heads,
            feedForwardSize: try int("feed_forward_length"),
            melBins: mel,
            projectionDim: try int("projection_dim"),
            fileSize: gguf.fileSize,
            fileSHA256: fileSHA256)
    }
}
