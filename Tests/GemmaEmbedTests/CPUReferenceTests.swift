import Foundation
import XCTest
@testable import GemmaEmbed

final class CPUReferenceTests: XCTestCase {
    nonisolated(unsafe) static var shared: (config: ModelConfig, weights: ModelWeights, tokenizer: Tokenizer, reference: CPUReference)?

    override func setUpWithError() throws {
        if Self.shared == nil {
            let file = try GGUFFile(url: try TestSupport.requireWeights())
            let config = try ModelConfig(gguf: file, deviceName: "cpu", fileSHA256: nil)
            let weights = try ModelWeights(gguf: file, config: config)
            Self.shared = (config, weights, try Tokenizer(gguf: file), CPUReference(config: config, weights: weights))
        }
    }

    static func normalised(_ v: [Float]) -> [Float] {
        let n = v.reduce(0) { $0 + $1 * $1 }.squareRoot()
        return v.map { $0 / n }
    }

    func testConfig() throws {
        let s = try XCTUnwrap(Self.shared)
        XCTAssertEqual(s.config.info.layerCount, 24)
        XCTAssertEqual(s.config.layers.filter(\.isSlidingWindow).count, 20)
        XCTAssertEqual(s.config.layers[5].headDim, 512)
        XCTAssertEqual(s.config.layers[5].kvHeadCount, 1)
        XCTAssertEqual(s.config.layers[5].ropeBase, 1_000_000)
        XCTAssertEqual(s.config.layers[0].headDim, 256)
        XCTAssertEqual(s.config.layers[0].kvHeadCount, 2)
        XCTAssertEqual(s.config.layers[0].ropeBase, 10_000)
        XCTAssertEqual(s.config.info.maxTokens, 8192)
        XCTAssertEqual(s.config.info.quantization, "Q8_0")
    }

    /// Per-tensor sums from llama.cpp's debug callback decide where any divergence starts.
    func testActivationsMatchLlamaCpp() throws {
        let s = try XCTUnwrap(Self.shared)
        let rows = try TestSupport.fixtureJSONLines("activations.jsonl")
        let tokens = try TestSupport.tokenFixtures()
        var byText: [Int: [(name: String, sum: Double)]] = [:]
        var inputs: [Int: [Int32]] = [:]
        for row in rows {
            let i = row["i"] as! Int
            if row["record"] as? String == "meta" {
                inputs[i] = (row["token_ids"] as! [Int]).map(Int32.init)
                continue
            }
            guard let name = row["name"] as? String, let sum = row["sum"] as? Double else { continue }
            byText[i, default: []].append((name, sum))
        }
        XCTAssertEqual(byText.count, 3)
        for (i, expected) in byText.sorted(by: { $0.key < $1.key }) {
            let ids = try XCTUnwrap(inputs[i])
            XCTAssertEqual(ids, tokens[i].tokens)
            var sums: [String: Double] = [:]
            _ = s.reference.forward(tokens: ids) { name, values in
                sums[name] = values.reduce(0.0) { $0 + Double($1) }
            }
            var checked = 0
            for (name, want) in expected where sums[name] != nil {
                let got = sums[name]!
                let tolerance = max(0.02 * abs(want), 0.5)
                XCTAssertEqual(got, want, accuracy: tolerance, "text \(i) tensor \(name): got \(got), llama.cpp \(want)")
                checked += 1
            }
            XCTAssertGreaterThan(checked, 200, "text \(i): too few tensors compared")
        }
    }

    func testEmbeddingsMatchReference() throws {
        let s = try XCTUnwrap(Self.shared)
        let tokens = try TestSupport.tokenFixtures()
        let embeddings = try TestSupport.embeddingFixtures()
        for i in [20, 0, 35] {
            let ids = s.tokenizer.encode(tokens[i].input)
            XCTAssertEqual(ids, tokens[i].tokens)
            let got = Self.normalised(s.reference.forward(tokens: ids))
            let want = embeddings[i].embedding
            XCTAssertEqual(got.count, 768)
            let cos = TestSupport.cosine(got, want)
            let maxDiff = TestSupport.maxAbsDiff(got, want)
            print("cpu reference text \(i): tokens \(ids.count) cosine \(cos) maxAbsDiff \(maxDiff)")
            XCTAssertGreaterThan(cos, 0.9995, "text \(i) cosine \(cos)")
            XCTAssertLessThan(maxDiff, 2e-3, "text \(i) max abs diff \(maxDiff)")
        }
    }
}
