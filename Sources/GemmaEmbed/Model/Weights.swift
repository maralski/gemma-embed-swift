// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// A tensor inside the mapped weight file, with element access that widens storage to fp32.
struct WeightTensor: @unchecked Sendable {
    let info: GGUFTensorInfo
    let base: UnsafeRawPointer
    /// Absolute byte offset from the start of the file (and of the Metal buffer wrapping it).
    let fileOffset: Int

    /// Elements along the contiguous dimension.
    var columns: Int { info.shape[0] }
    /// Number of rows (product of the remaining dimensions).
    var rows: Int { info.elementCount / columns }
    var rowBytes: Int { columns / info.type.blockSize * info.type.blockBytes }
    var type: GGUFTensorType { info.type }

    func dequantizeRow(_ row: Int, into out: UnsafeMutablePointer<Float>) {
        let p = base + row * rowBytes
        let n = columns
        switch info.type {
        case .f32:
            memcpy(out, p, n * 4)
        case .f16:
            for i in 0..<n {
                out[i] = Float(Float16(bitPattern: p.loadUnaligned(fromByteOffset: 2 * i, as: UInt16.self)))
            }
        case .bf16:
            for i in 0..<n {
                out[i] = Float(bitPattern: UInt32(p.loadUnaligned(fromByteOffset: 2 * i, as: UInt16.self)) << 16)
            }
        case .q8_0:
            let blocks = n / 32
            for b in 0..<blocks {
                let bp = p + b * 34
                let scale = Float(Float16(bitPattern: bp.loadUnaligned(as: UInt16.self)))
                for j in 0..<32 {
                    out[b * 32 + j] = scale * Float(bp.load(fromByteOffset: 2 + j, as: Int8.self))
                }
            }
        }
    }

    /// The whole tensor as row-major fp32 `[rows][columns]`.
    func dequantized() -> [Float] {
        let rows = self.rows, cols = self.columns
        return [Float](unsafeUninitializedCapacity: rows * cols) { buf, n in
            for r in 0..<rows { dequantizeRow(r, into: buf.baseAddress! + r * cols) }
            n = rows * cols
        }
    }

    func scalar() -> Float {
        var v: Float = 0
        dequantizeRow(0, into: &v)
        return v
    }
}

struct LayerWeights: Sendable {
    let attnNorm: WeightTensor
    let wq: WeightTensor
    let wk: WeightTensor
    let wv: WeightTensor
    let wo: WeightTensor
    let qNorm: WeightTensor
    let kNorm: WeightTensor
    let postAttnNorm: WeightTensor
    let ffnNorm: WeightTensor
    let wGate: WeightTensor
    let wUp: WeightTensor
    let wDown: WeightTensor
    let postFfnNorm: WeightTensor
    let inpGate: WeightTensor
    let proj: WeightTensor
    let postNorm: WeightTensor
    let outScale: Float
}

/// All tensors of a gemma-embedding2 model, shape-checked against the configuration.
struct ModelWeights: Sendable {
    /// Keeps the mapping alive for as long as any tensor pointer is in use.
    let file: GGUFFile
    let tokenEmbedding: WeightTensor
    let perLayerProj: WeightTensor
    let perLayerProjNorm: WeightTensor
    let outputNorm: WeightTensor
    let output: WeightTensor
    let layers: [LayerWeights]

    init(gguf: GGUFFile, config: ModelConfig) throws {
        func tensor(_ name: String, _ shape: [Int], types: [GGUFTensorType] = [.q8_0, .f32, .bf16, .f16]) throws -> WeightTensor {
            guard let info = gguf.tensor(named: name) else { throw GemmaEmbedError.missingTensor(name) }
            guard info.shape == shape else { throw GemmaEmbedError.tensorShapeMismatch(name, expected: shape, found: info.shape) }
            guard types.contains(info.type) else { throw GemmaEmbedError.unsupportedTensorType(name, info.type.rawValue) }
            return WeightTensor(info: info, base: gguf.bytes(of: info).baseAddress!, fileOffset: gguf.fileOffset(of: info))
        }
        self.file = gguf
        let d = config.info.hiddenSize
        let ff = config.info.feedForwardSize
        let pl = config.info.perLayerInputSize
        tokenEmbedding = try tensor("token_embd.weight", [d, config.info.vocabularySize])
        perLayerProj = try tensor("per_layer_model_proj.weight", [d, pl * config.info.layerCount])
        perLayerProjNorm = try tensor("per_layer_proj_norm.weight", [pl], types: [.f32])
        outputNorm = try tensor("output_norm.weight", [d], types: [.f32])
        output = try tensor("output.weight", [d, config.info.outputDimensions])
        layers = try config.layers.map { layer in
            let i = layer.index
            let hd = layer.headDim
            let qDim = config.info.headCount * hd
            let kvDim = layer.kvHeadCount * hd
            return LayerWeights(
                attnNorm: try tensor("blk.\(i).attn_norm.weight", [d], types: [.f32]),
                wq: try tensor("blk.\(i).attn_q.weight", [d, qDim]),
                wk: try tensor("blk.\(i).attn_k.weight", [d, kvDim]),
                wv: try tensor("blk.\(i).attn_v.weight", [d, kvDim]),
                wo: try tensor("blk.\(i).attn_output.weight", [qDim, d]),
                qNorm: try tensor("blk.\(i).attn_q_norm.weight", [hd], types: [.f32]),
                kNorm: try tensor("blk.\(i).attn_k_norm.weight", [hd], types: [.f32]),
                postAttnNorm: try tensor("blk.\(i).post_attention_norm.weight", [d], types: [.f32]),
                ffnNorm: try tensor("blk.\(i).ffn_norm.weight", [d], types: [.f32]),
                wGate: try tensor("blk.\(i).ffn_gate.weight", [d, ff]),
                wUp: try tensor("blk.\(i).ffn_up.weight", [d, ff]),
                wDown: try tensor("blk.\(i).ffn_down.weight", [ff, d]),
                postFfnNorm: try tensor("blk.\(i).post_ffw_norm.weight", [d], types: [.f32]),
                inpGate: try tensor("blk.\(i).inp_gate.weight", [d, pl]),
                proj: try tensor("blk.\(i).proj.weight", [pl, d]),
                postNorm: try tensor("blk.\(i).post_norm.weight", [d], types: [.f32]),
                outScale: try tensor("blk.\(i).layer_output_scale.weight", [1], types: [.f32]).scalar())
        }
    }
}
