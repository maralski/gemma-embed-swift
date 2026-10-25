import Foundation
import XCTest
@testable import GemmaEmbed

final class CPUVisionReferenceTests: XCTestCase {
    struct Vision {
        let config: VisionConfig
        let preprocessor: ImagePreprocessor
        let reference: CPUVisionReference
    }

    nonisolated(unsafe) static var sharedVision: Vision?

    static let fullDumps = ["09-tiny-noise", "04-desert-dunes", "07-shapes"]

    func vision() throws -> Vision {
        if let v = Self.sharedVision { return v }
        let file = try GGUFFile(url: try TestSupport.requireImageEncoder())
        let config = try VisionConfig(gguf: file, fileSHA256: nil)
        let v = Vision(config: config, preprocessor: ImagePreprocessor(config: config),
                       reference: CPUVisionReference(config: config, weights: try VisionWeights(gguf: file, config: config)))
        Self.sharedVision = v
        return v
    }

    /// Rotating llama.cpp's own `Qcur` must give its `Qcur_pos`: this isolates the 2D RoPE
    /// pairing from every upstream difference. (The dumps keep `Kcur` only after the rotation.)
    func testRope2DMatchesLlamaCpp() throws {
        let v = try vision()
        let dump = try TestSupport.requireVisionDump("09-tiny-noise")
        let image = v.preprocessor.prepare(try dump.decodedRGB(), maxImageTokens: 256)
        let table = RopeTable(headDim: 32, base: v.config.ropeTheta, positions: max(image.gridX, image.gridY))
        var x = try dump.values("Qcur-0")
        CPUVisionReference.rope2D(&x, heads: 12, headDim: 64, table: table, patchX: image.patchX, patchY: image.patchY)
        let want = try dump.values("Qcur_pos-0")
        let maxDiff = TestSupport.maxAbsDiff(x, want)
        print("rope2d Qcur_pos-0 from llama.cpp's Qcur-0: max abs diff \(maxDiff)")
        XCTAssertLessThan(maxDiff, 1e-4)
    }

    /// Every node llama.cpp dumped is compared by its sum, and the nodes saved in full by cosine,
    /// with the vision tower fed stb_image's bytes so the decoder is out of the comparison.
    func testNodesMatchLlamaCpp() throws {
        let v = try vision()
        for name in Self.fullDumps {
            let dump = try TestSupport.requireVisionDump(name)
            let image = v.preprocessor.prepare(try dump.decodedRGB(), maxImageTokens: 256)
            var sums: [String: Double] = [:]
            var full: [(String, Float, Float)] = []
            let start = Date()
            let rows = v.reference.forward(image) { node, values in
                sums[node] = values.reduce(0.0) { $0 + Double($1) }
                guard let file = dump.node(node)?.file, let want = try? dump.floats(file) else { return }
                XCTAssertEqual(values.count, want.count, "\(name) \(node)")
                full.append((node, TestSupport.cosine(values, want), TestSupport.maxRelDiff(values, want)))
            }
            let seconds = Date().timeIntervalSince(start)
            var checked = 0
            var worst = (name: "", error: 0.0)
            for node in dump.nodes {
                guard let got = sums[node.name], dump.node(node.name)?.index == node.index else { continue }
                // Sums cancel, so the error is measured against the sum of magnitudes.
                let error = abs(got - node.sum) / max(node.absSum, 1e-6)
                if error > worst.error { worst = (node.name, error) }
                XCTAssertLessThan(error, 1e-3, "\(name) \(node.name): sum \(got), llama.cpp \(node.sum), abs sum \(node.absSum)")
                checked += 1
            }
            XCTAssertGreaterThan(checked, 16 * 18, "\(name): too few nodes compared")
            for (node, cosine, rel) in full {
                print("vision \(name) \(node): cosine \(cosine) max rel diff \(rel)")
                XCTAssertGreaterThan(cosine, 0.9995, "\(name) \(node)")
            }
            XCTAssertEqual(full.count, 19, "\(name): saved nodes compared")

            let want = try dump.floats("output_embd.f32")
            let width = v.config.info.projectionDim
            XCTAssertEqual(rows.count, image.tokenCount * width)
            XCTAssertEqual(rows.count, want.count)
            var minCosine: Float = 1
            for t in 0..<image.tokenCount {
                let r = t * width..<(t + 1) * width
                let cosine = TestSupport.cosine(Array(rows[r]), Array(want[r]))
                if cosine.isNaN || cosine < minCosine { minCosine = cosine }
            }
            print("vision \(name): \(image.patchCount) patches, \(image.tokenCount) tokens, \(checked) node sums, worst sum error \(worst.error) at \(worst.name), min row cosine \(minCosine), max abs diff \(TestSupport.maxAbsDiff(rows, want)), \(String(format: "%.1f", seconds)) s")
            XCTAssertGreaterThan(minCosine, 0.9995, name)
        }
    }

    struct Text {
        let reference: CPUReference
        let markers: ImageMarkers
    }

    nonisolated(unsafe) static var sharedText: Text?

    func text() throws -> Text {
        if let t = Self.sharedText { return t }
        let file = try GGUFFile(url: try TestSupport.requireWeights())
        let config = try ModelConfig(gguf: file, deviceName: "cpu", fileSHA256: nil)
        let t = Text(reference: CPUReference(config: config, weights: try ModelWeights(gguf: file, config: config)),
                     markers: try ImageMarkers(tokenizer: Tokenizer(gguf: file)))
        Self.sharedText = t
        return t
    }

    func testImageMarkers() throws {
        let markers = try text().markers
        XCTAssertEqual(markers, try ImageMarkers(tokenizer: Tokenizer(gguf: GGUFFile(url: TestSupport.requireWeights()))))
        XCTAssertEqual([markers.bos, markers.begin, markers.end, markers.eos], [2, 255999, 258882, 1])
        let rows = markers.sequence(vectors: [1, 2, 3, 4, 5, 6], width: 3)
        XCTAssertEqual(rows, [.token(2), .token(255999), .vector([1, 2, 3]), .vector([4, 5, 6]), .token(258882), .token(1)])
    }

    /// The whole pipeline against llama-server: preprocessing, the vision tower, then the text
    /// tower over `[bos, <|image>, vectors..., <image|>, eos]`, mean pooled and normalised,
    /// starting from the file as GemmaEmbed decodes it.
    func testImageEmbeddingsMatchLlamaServer() throws {
        let v = try vision()
        let t = try text()
        let fixtures = try TestSupport.imageEmbeddingFixtures()
        let images = ["synthetic/09-tiny-noise.png", "scenes/04-desert-dunes.jpg", "synthetic/07-shapes.jpg", "scenes/00-green-hills.jpg",
                      "synthetic/02-gradient-h.png"]
        func embed(_ image: PreparedImage) -> (rows: Int, embedding: [Float]) {
            let rows = t.markers.sequence(vectors: v.reference.forward(image), width: v.config.info.projectionDim)
            return (rows.count, CPUReferenceTests.normalised(t.reference.forward(rows: rows)))
        }
        for file in images {
            let fixture = try XCTUnwrap(fixtures[file], file)
            let image = v.preprocessor.prepare(try ImageDecoder.decode(.url(TestSupport.fixture("images/" + file))), maxImageTokens: 256)
            let (rows, got) = embed(image)
            XCTAssertEqual(rows, fixture.promptTokens, file)
            let cosine = TestSupport.cosine(got, fixture.embedding)
            print("image embedding \(file): \(image.width)x\(image.height), \(rows) rows, cosine \(cosine), max abs diff \(TestSupport.maxAbsDiff(got, fixture.embedding))")
            XCTAssertGreaterThan(cosine, 0.9995, file)
        }
    }
}
