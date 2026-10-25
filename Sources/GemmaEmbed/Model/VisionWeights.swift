import Foundation

struct VisionLayerWeights: Sendable {
    let ln1: WeightTensor
    let wq: WeightTensor
    let wk: WeightTensor
    let wv: WeightTensor
    let wo: WeightTensor
    let qNorm: WeightTensor
    let kNorm: WeightTensor
    let postAttnNorm: WeightTensor
    let ln2: WeightTensor
    let wGate: WeightTensor
    let wUp: WeightTensor
    let wDown: WeightTensor
    let postFfnNorm: WeightTensor
}

/// The vision tensors of a gemma4v mmproj file, shape-checked against the configuration.
/// The audio tower (`a.*`, `mm.a.*`) in the same file is not referenced.
struct VisionWeights: Sendable {
    /// Keeps the mapping alive for as long as any tensor pointer is in use.
    let file: GGUFFile
    /// `v.patch_embd.weight` viewed as `[hidden][patchArea]`, each row indexed `c * p^2 + ky * p + kx`.
    let patchEmbedding: WeightTensor
    /// `v.position_embd.weight` viewed as `[2 * positionCount][hidden]`: the x table, then the y table.
    let positionTable: WeightTensor
    /// Rows in each of the two position tables.
    let positionCount: Int
    let layers: [VisionLayerWeights]
    /// `mm.input_projection.weight`, `[projectionDim][hidden]`.
    let projection: WeightTensor

    init(gguf: GGUFFile, config: VisionConfig) throws {
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
        self.file = gguf
        let d = config.info.hiddenSize
        let p = config.info.patchSize
        let ff = config.feedForwardSize
        let hd = config.headDim

        let patch = try info("v.patch_embd.weight", [p, p, 3, d], types: [.f32, .f16])
        patchEmbedding = view(patch, as: [config.patchArea, d])
        let position = try info("v.position_embd.weight", nil, types: [.f32])
        guard position.shape.count == 3, position.shape[0] == d, position.shape[2] == 2, position.shape[1] > 0 else {
            throw GemmaEmbedError.invalidGGUF("\(position.name) has shape \(position.shape), expected [\(d), n, 2]")
        }
        positionCount = position.shape[1]
        positionTable = view(position, as: [d, 2 * positionCount])
        projection = try tensor("mm.input_projection.weight", [d, config.info.projectionDim])
        layers = try (0..<config.info.layerCount).map { i in
            VisionLayerWeights(
                ln1: try tensor("v.blk.\(i).ln1.weight", [d], types: [.f32]),
                wq: try tensor("v.blk.\(i).attn_q.weight", [d, d]),
                wk: try tensor("v.blk.\(i).attn_k.weight", [d, d]),
                wv: try tensor("v.blk.\(i).attn_v.weight", [d, d]),
                wo: try tensor("v.blk.\(i).attn_out.weight", [d, d]),
                qNorm: try tensor("v.blk.\(i).attn_q_norm.weight", [hd], types: [.f32]),
                kNorm: try tensor("v.blk.\(i).attn_k_norm.weight", [hd], types: [.f32]),
                postAttnNorm: try tensor("v.blk.\(i).attn_post_norm.weight", [d], types: [.f32]),
                ln2: try tensor("v.blk.\(i).ln2.weight", [d], types: [.f32]),
                wGate: try tensor("v.blk.\(i).ffn_gate.weight", [d, ff]),
                wUp: try tensor("v.blk.\(i).ffn_up.weight", [d, ff]),
                wDown: try tensor("v.blk.\(i).ffn_down.weight", [ff, d]),
                postFfnNorm: try tensor("v.blk.\(i).ffn_post_norm.weight", [d], types: [.f32]))
        }
    }
}
