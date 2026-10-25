import Foundation
import XCTest
@testable import GemmaEmbed

final class CPUAudioReferenceTests: XCTestCase {
    struct Audio {
        let config: AudioConfig
        let frontEnd: AudioFrontEnd
        let reference: CPUAudioReference
    }

    nonisolated(unsafe) static var sharedAudio: Audio?

    func audio() throws -> Audio {
        if let a = Self.sharedAudio { return a }
        let file = try GGUFFile(url: try TestSupport.requireImageEncoder())
        let config = try AudioConfig(gguf: file, fileSHA256: nil)
        let a = Audio(config: config, frontEnd: AudioFrontEnd(melBins: config.info.melBins),
                      reference: CPUAudioReference(config: config, weights: try AudioWeights(gguf: file, config: config)))
        Self.sharedAudio = a
        return a
    }

    /// Where each traced tensor sits in llama.cpp's graph. gemma4a names almost no nodes, so the
    /// dump is keyed by evaluation index: the subsampling takes nodes 0 to 35, every Conformer
    /// layer 112 nodes from 36, and the output projections follow the last layer. The op at each
    /// index is checked, so a shift in llama.cpp's graph fails loudly rather than comparing the
    /// wrong tensors.
    enum Graph {
        static let layerBase = 36
        static let layerNodes = 112
        /// Traced name, node index and op before the layers.
        static let head: [(String, Int, String)] = [("mel", 1, "CONT"), ("subsample-0", 15, "UNARY"), ("subsample-1", 29, "UNARY"), ("encoder_in", 35, "CLAMP")]
        /// Traced name without its layer suffix, offset within the layer and op.
        static let layer: [(String, Int, String)] = [
            ("ffn1_normed", 1, "MUL"), ("ffn1_up", 4, "CLAMP"), ("ffn1_down", 8, "CLAMP"), ("ffn1_post_normed", 10, "MUL"),
            ("after_ffn1", 12, "ADD"), ("attn_normed", 14, "MUL"), ("v", 17, "CLAMP"), ("k_scaled", 28, "SCALE"),
            ("q_scaled", 40, "MUL"), ("k_rel", 41, "MUL_MAT"), ("attn", 69, "CONT"), ("attn_out", 72, "CLAMP"),
            ("after_attn", 75, "ADD"), ("conv_normed", 77, "MUL"), ("conv_pw1", 80, "CLAMP"), ("conv_glu", 85, "MUL"),
            ("conv_dw", 89, "SSM_CONV"), ("conv_silu", 92, "UNARY"), ("conv_pw2", 95, "CLAMP"), ("after_conv", 96, "ADD"),
            ("ffn2_normed", 98, "MUL"), ("ffn2_up", 101, "CLAMP"), ("ffn2_down", 105, "CLAMP"), ("after_ffn2", 109, "ADD"),
            ("layer_out", 111, "MUL"),
        ]
        /// Traced name, offset after the last layer and op.
        static let tail: [(String, Int, String)] = [("out_proj", 3, "ADD"), ("projected", 7, "CLAMP")]
        /// Traced tensors whose memory layout differs from the dump's, compared by sum only.
        static let sumOnly: Set<String> = ["subsample", "k_rel"]

        static func nodeIndex(_ name: String, layers: Int) -> (index: Int, op: String)? {
            if let h = head.first(where: { $0.0 == name }) { return (h.1, h.2) }
            if let t = tail.first(where: { $0.0 == name }) { return (layerBase + layerNodes * layers + t.1, t.2) }
            guard let dash = name.lastIndex(of: "-"), let l = Int(name[name.index(after: dash)...]) else { return nil }
            let stem = String(name[..<dash])
            guard let entry = layer.first(where: { $0.0 == stem }) else { return nil }
            return (layerBase + layerNodes * l + entry.1, entry.2)
        }
    }

    /// Every traced tensor against llama.cpp's node by sum, and by value where the dump saved
    /// the node, over every dumped clip and chunk; the front end's own mel is the input, so this
    /// also covers its fp16 rounding at the first convolution.
    func testNodesMatchLlamaCpp() throws {
        let a = try audio()
        let names = TestSupport.audioDumpNames()
        guard !names.isEmpty else { throw XCTSkip("no audio dumps under \(TestSupport.audioDumpsRoot)") }
        let layers = a.config.info.layerCount
        var worstSum = (name: "", error: 0.0)
        var worstFull = (name: "", cosine: Float(1), relDiff: Float(0))
        var sums = 0, fulls = 0
        for name in names {
            let dump = try TestSupport.requireAudioDump(name)
            let prepared = try a.frontEnd.prepare(try dump.samples())
            XCTAssertEqual(prepared.chunks.count, dump.chunks.count, name)
            for (c, (chunk, mel)) in zip(dump.chunks, prepared.chunks).enumerated() {
                let start = Date()
                let rows = a.reference.forward(mel) { traced, values in
                    guard let (offset, op) = Graph.nodeIndex(traced, layers: layers) else { XCTFail("\(traced) has no node mapping"); return }
                    guard let node = dump.node(chunk.firstNode + offset) else { XCTFail("\(name) chunk \(c): no node \(chunk.firstNode + offset) for \(traced)"); return }
                    XCTAssertEqual(node.op, op, "\(name) chunk \(c) \(traced) at node \(node.index)")
                    XCTAssertEqual(node.shape.reduce(1, *), values.count, "\(name) chunk \(c) \(traced) element count")
                    let sum = values.reduce(0.0) { $0 + Double($1) }
                    let error = abs(sum - node.sum) / max(node.absSum, 1e-6)
                    if error > worstSum.error { worstSum = ("\(name) chunk \(c) \(traced)", error) }
                    XCTAssertLessThan(error, 1e-3, "\(name) chunk \(c) \(traced): sum \(sum) vs llama.cpp \(node.sum)")
                    sums += 1
                    let stem = traced.lastIndex(of: "-").map { String(traced[..<$0]) } ?? traced
                    guard let file = node.file, !Graph.sumOnly.contains(stem) else { return }
                    guard let want = try? dump.floats(file) else { return }
                    let cosine = TestSupport.cosine(values, want)
                    let rel = TestSupport.maxRelDiff(values, want)
                    if cosine < worstFull.cosine { worstFull = ("\(name) chunk \(c) \(traced)", cosine, rel) }
                    XCTAssertGreaterThan(cosine, 0.9999, "\(name) chunk \(c) \(traced)")
                    XCTAssertLessThan(rel, 2e-2, "\(name) chunk \(c) \(traced)")
                    fulls += 1
                }
                let seconds = Date().timeIntervalSince(start)
                let want = try dump.outputRows(chunk: c)
                XCTAssertEqual(rows.count, want.count, "\(name) chunk \(c) output rows")
                let width = a.config.info.projectionDim
                // A few rows of a 750-token chunk drift to cosine 0.999 against llama.cpp's Metal
                // backend while the median row stays at 0.9999996 and the pooled rows at 0.9999998.
                // llama.cpp's CPU backend, which quantises activations for Q8_0 matmuls, sits at row
                // cosine 0.86 against this reference and pools to 0.99999, so the spread is the
                // oracle's activation precision; the text tower pools the rows.
                var minCosine: Float = 1
                var pooled = [Float](repeating: 0, count: width), pooledWant = pooled
                for r in 0..<(want.count / width) {
                    let mine = Array(rows[r * width..<(r + 1) * width]), theirs = Array(want[r * width..<(r + 1) * width])
                    minCosine = min(minCosine, TestSupport.cosine(mine, theirs))
                    for i in 0..<width { pooled[i] += mine[i]; pooledWant[i] += theirs[i] }
                }
                let pooledCosine = TestSupport.cosine(pooled, pooledWant)
                print("\(name) chunk \(c): \(mel.frames) frames, \(mel.tokenCount) tokens in \(String(format: "%.1f", seconds)) s; output rows vs libmtmd min cosine \(minCosine), pooled \(pooledCosine)")
                XCTAssertGreaterThan(minCosine, 0.998, "\(name) chunk \(c) output rows")
                XCTAssertGreaterThan(pooledCosine, 0.99999, "\(name) chunk \(c) pooled output rows")
            }
        }
        print("audio nodes vs llama.cpp: \(sums) sums (worst \(worstSum.name) \(worstSum.error)), \(fulls) full tensors (worst \(worstFull.name) cosine \(worstFull.cosine) rel diff \(worstFull.relDiff))")
    }
}
