import Foundation

/// Errors raised by GemmaEmbed.
public enum GemmaEmbedError: Error, CustomStringConvertible, Sendable {
    case fileNotReadable(String, String)
    case invalidGGUF(String)
    case unsupportedTensorType(String, UInt32)
    case unsupportedArchitecture(String)
    case missingTensor(String)
    case missingMetadata(String)
    case tensorShapeMismatch(String, expected: [Int], found: [Int])
    case noMetalDevice
    case shaderCompilation(String)
    case shaderSourceMissing(tried: [String])
    case invalidOptions(String)
    case gpuFailure(String)
    case imageDecoding(String)
    case imageEncoderNotLoaded
    case cancelled

    public var description: String {
        switch self {
        case .fileNotReadable(let path, let reason): return "cannot read \(path): \(reason)"
        case .invalidGGUF(let reason): return "invalid GGUF file: \(reason)"
        case .unsupportedTensorType(let name, let raw): return "tensor \(name) uses unsupported storage type \(raw)"
        case .unsupportedArchitecture(let arch): return "unsupported architecture '\(arch)'; GemmaEmbed supports gemma-embedding2"
        case .missingTensor(let name): return "weight file lacks tensor \(name)"
        case .missingMetadata(let key): return "weight file lacks metadata key \(key)"
        case .tensorShapeMismatch(let name, let expected, let found): return "tensor \(name) has shape \(found), expected \(expected)"
        case .noMetalDevice: return "no Metal device is available"
        case .shaderCompilation(let log): return "Metal shader compilation failed: \(log)"
        case .shaderSourceMissing(let tried):
            return "Metal shader sources not found; looked for Shaders/ in " + tried.joined(separator: ", ")
        case .invalidOptions(let reason): return "invalid options: \(reason)"
        case .gpuFailure(let reason): return "GPU execution failed: \(reason)"
        case .imageDecoding(let reason): return "cannot decode image: \(reason)"
        case .imageEncoderNotLoaded: return "no image encoder is loaded; pass the mmproj GGUF when loading the model"
        case .cancelled: return "embedding was cancelled"
        }
    }
}
