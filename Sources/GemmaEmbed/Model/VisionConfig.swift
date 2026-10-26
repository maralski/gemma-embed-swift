// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Static facts about a loaded image encoder.
public struct ImageEncoderInfo: Sendable {
    public let layerCount: Int
    /// Width of the vision tower's residual stream.
    public let hiddenSize: Int
    /// Side of the square pixel patch that becomes one vision position.
    public let patchSize: Int
    /// Width of the vectors handed to the text tower.
    public let projectionDim: Int
    public let fileSize: Int
    /// SHA256 of the image encoder file, when computed at load.
    public let fileSHA256: String?
}

/// Shapes and constants of a gemma4v image encoder, derived from the mmproj GGUF metadata.
struct VisionConfig: Sendable {
    let info: ImageEncoderInfo
    let headCount: Int
    let feedForwardSize: Int
    let normEpsilon: Float
    let imageMean: [Float]
    let imageStd: [Float]

    // llama.cpp hard codes these for gemma4v; the GGUF carries no keys for them.
    let mergeSize = 3
    let ropeTheta: Float = 100
    let minImageTokens = 70
    let nativeMaxImageTokens = 1120

    var headDim: Int { info.hiddenSize / headCount }
    /// Elements in one flattened patch: patchSize^2 pixels of 3 channels.
    var patchArea: Int { info.patchSize * info.patchSize * 3 }
    /// Image sides are multiples of this, so the patch grid divides into merge windows.
    var alignment: Int { info.patchSize * mergeSize }
    var pixelsPerToken: Int { alignment * alignment }

    static let supportedArchitecture = "clip"
    static let supportedProjector = "gemma4v"

    init(gguf: GGUFFile, fileSHA256: String?) throws {
        let arch = gguf.string("general.architecture") ?? ""
        guard arch == VisionConfig.supportedArchitecture else {
            throw GemmaEmbedError.invalidGGUF("image encoder architecture is '\(arch)', expected \(VisionConfig.supportedArchitecture)")
        }
        let projector = gguf.string("clip.vision.projector_type") ?? ""
        guard projector == VisionConfig.supportedProjector else {
            throw GemmaEmbedError.invalidGGUF("image projector is '\(projector)', expected \(VisionConfig.supportedProjector)")
        }
        func int(_ key: String) throws -> Int {
            guard let v = gguf.int("clip.vision.\(key)") else { throw GemmaEmbedError.missingMetadata("clip.vision.\(key)") }
            return v
        }
        func floats(_ key: String) throws -> [Float] {
            guard let v = gguf.metadata["clip.vision.\(key)"]?.float32ArrayValue, v.count == 3 else {
                throw GemmaEmbedError.missingMetadata("clip.vision.\(key)")
            }
            return v
        }
        guard let eps = gguf.float("clip.vision.attention.layer_norm_epsilon") else {
            throw GemmaEmbedError.missingMetadata("clip.vision.attention.layer_norm_epsilon")
        }
        let hidden = try int("embedding_length")
        let heads = try int("attention.head_count")
        guard heads > 0, hidden % heads == 0 else { throw GemmaEmbedError.invalidGGUF("vision width \(hidden) does not split into \(heads) heads") }
        self.info = ImageEncoderInfo(
            layerCount: try int("block_count"),
            hiddenSize: hidden,
            patchSize: try int("patch_size"),
            projectionDim: try int("projection_dim"),
            fileSize: gguf.fileSize,
            fileSHA256: fileSHA256)
        self.headCount = heads
        self.feedForwardSize = try int("feed_forward_length")
        self.normEpsilon = eps
        self.imageMean = try floats("image_mean")
        self.imageStd = try floats("image_std")
    }
}
