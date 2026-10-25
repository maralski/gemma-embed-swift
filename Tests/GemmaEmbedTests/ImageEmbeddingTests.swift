import CoreGraphics
import Foundation
import ImageIO
import XCTest
@testable import GemmaEmbed

final class ImageEmbeddingTests: XCTestCase {
    nonisolated(unsafe) static var model: EmbeddingModel?

    override func setUp() async throws {
        if Self.model == nil {
            Self.model = try await EmbeddingModel.load(from: try TestSupport.requireWeights(),
                                                       options: LoadOptions(imageEncoder: try TestSupport.requireImageEncoder()))
        }
    }

    func norm(_ v: [Float]) -> Float { v.reduce(0) { $0 + $1 * $1 }.squareRoot() }

    struct ManifestEntry {
        let file: String
        let kind: String
        let source: String?
        let variant: String?
        let url: URL
    }

    /// Every manifest image present in the bundle or `models/fixtures/images`, in manifest order.
    static func manifest() throws -> [ManifestEntry] {
        let data = try Data(contentsOf: TestSupport.fixture("images/manifest.json"))
        let rows = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [[String: Any]])
        let urls = Dictionary(uniqueKeysWithValues: try MetalImageForwardTests.fixtureImages().map { ($0.file, $0.url) })
        return rows.compactMap { row in
            let file = row["file"] as! String
            guard let url = urls[file] else { return nil }
            return ManifestEntry(file: file, kind: row["kind"] as! String, source: row["source"] as? String,
                                 variant: row["variant"] as? String, url: url)
        }
    }

    func entry(_ file: String) throws -> ManifestEntry {
        guard let e = try Self.manifest().first(where: { $0.file == file }) else { throw XCTSkip("fixture image \(file) missing") }
        return e
    }

    static func stem(_ file: String) -> String {
        ((file as NSString).lastPathComponent as NSString).deletingPathExtension
    }

    static func median(_ values: [Float]) -> Float {
        let s = values.sorted()
        return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
    }

    /// The bar is llama-server's own vectors: GemmaEmbed must rank the captions as they do, and with
    /// these scenes both rank each picture's own caption above the other nine.
    func testPicturesRankCaptionsAsLlamaServerDoes() async throws {
        let model = try XCTUnwrap(Self.model)
        let captions = try TestSupport.fixtureJSONLines("images/captions.jsonl")
        XCTAssertEqual(captions.count, 10)
        let imageFixtures = try TestSupport.imageEmbeddingFixtures()
        var queries: [Embedding] = []
        var referenceQueries: [[Float]] = []
        for row in captions {
            let text = row["text"] as! String
            for (task, key) in [(TaskPrefix.query(nil), "query"), (TaskPrefix.document(title: nil), "document")] {
                let reference = row[key] as! [String: Any]
                let e = try await model.embed(text, options: EmbedOptions(task: task))
                XCTAssertEqual(e.tokenCount, reference["prompt_tokens"] as? Int, "\(key) \(text)")
                let cosine = TestSupport.cosine(e.vector, (reference["embedding"] as! [Double]).map(Float.init))
                XCTAssertGreaterThan(cosine, 0.9995, "\(key) \(text) against llama-server")
                if key == "query" {
                    queries.append(e)
                    referenceQueries.append((reference["embedding"] as! [Double]).map(Float.init))
                }
            }
        }
        let described = captions.enumerated().compactMap { i, row in (row["describes"] as? String).map { (caption: i, file: $0) } }
        XCTAssertEqual(described.count, 5)
        let pictures = try await model.embed(images: try described.map { .url(try entry($0.file).url) })

        let similarity = pictures.map { p in queries.map { model.similarity(p, $0) } }
        let reference = try described.map { d in
            let picture = try XCTUnwrap(imageFixtures[d.file], d.file).embedding
            return referenceQueries.map { TestSupport.cosine(picture, $0) }
        }
        let worst = zip(similarity, reference).map { TestSupport.maxAbsDiff($0, $1) }.max()!
        print("cross-modal: largest similarity difference from llama-server's vectors \(worst)")
        XCTAssertLessThan(worst, 0.002)

        func rank(_ row: [Float], of caption: Int) -> Int { row.filter { $0 > row[caption] }.count + 1 }
        let ranks = zip(similarity, described).map { rank($0, of: $1.caption) }
        let referenceRanks = zip(reference, described).map { rank($0, of: $1.caption) }
        for ((d, r), row) in zip(zip(described, ranks), similarity) {
            print("cross-modal \(d.file): own caption ranks \(r) of 10 at \(String(format: "%.3f", row[d.caption])), best \(String(format: "%.3f", row.max()!))")
        }
        let firsts = ranks.filter { $0 == 1 }.count
        var table = "| caption | " + described.map { Self.stem($0.file) }.joined(separator: " | ") + " |\n"
        table += "|---|" + String(repeating: "---:|", count: described.count) + "\n"
        for (c, row) in captions.enumerated() {
            let cells = similarity.indices.map { p -> String in
                let value = String(format: "%.3f", similarity[p][c])
                let best = similarity[p].indices.max { similarity[p][$0] < similarity[p][$1] } == c
                return best ? "**\(value)**" : value
            }
            table += "| \(row["text"] as! String) | " + cells.joined(separator: " | ") + " |\n"
        }
        print("cross-modal similarity, picture columns against caption rows (query task), best caption per picture in bold:\n" + table)
        print("cross-modal: \(firsts) of 5 pictures rank their own caption first; ranks \(ranks), llama-server's vectors \(referenceRanks)")
        XCTAssertEqual(ranks, referenceRanks)
        XCTAssertEqual(ranks, Array(repeating: 1, count: described.count), "pictures that prefer another caption to their own")
    }

    func testVariantsFindTheirOriginalAsNearestNeighbour() async throws {
        let model = try XCTUnwrap(Self.model)
        let entries = try Self.manifest()
        XCTAssertGreaterThanOrEqual(entries.count, 27)
        let fixtures = try TestSupport.imageEmbeddingFixtures()
        let embeddings = try await model.embed(images: entries.map { .url($0.url) })
        XCTAssertEqual(embeddings.count, entries.count)
        for (e, embedding) in zip(entries, embeddings) {
            let fixture = try XCTUnwrap(fixtures[e.file], e.file)
            XCTAssertEqual(embedding.tokenCount, fixture.promptTokens, e.file)
            XCTAssertGreaterThan(TestSupport.cosine(embedding.vector, fixture.embedding), 0.9995, "\(e.file) against llama-server")
        }
        let index = Dictionary(uniqueKeysWithValues: entries.enumerated().map { ($1.file, $0) })
        func cosine(_ a: Int, _ b: Int) -> Float { model.similarity(embeddings[a], embeddings[b]) }

        var byVariant: [String: [(file: String, cosine: Float)]] = [:]
        for (i, e) in entries.enumerated() {
            guard let source = e.source, let variant = e.variant, let original = index[source] else { continue }
            byVariant[variant, default: []].append((e.file, cosine(original, i)))
        }
        var table = "| variant | pairs | min | median | max |\n|---|---:|---:|---:|---:|\n"
        for (variant, pairs) in byVariant.sorted(by: { $0.key < $1.key }) {
            let values = pairs.map(\.cosine)
            table += String(format: "| %@ | %d | %.3f | %.3f | %.3f |\n", variant, pairs.count, values.min()!, Self.median(values), values.max()!)
        }
        let variantPairs = byVariant.values.flatMap { $0 }
        let closest = variantPairs.min { $0.cosine < $1.cosine }!

        func family(_ e: ManifestEntry) -> String { e.source ?? e.file }
        var unrelated: (a: String, b: String, cosine: Float) = ("", "", -1)
        for i in entries.indices {
            for j in entries.indices where j > i && family(entries[i]) != family(entries[j]) {
                let c = cosine(i, j)
                if c > unrelated.cosine { unrelated = (entries[i].file, entries[j].file, c) }
            }
        }
        print("near duplicates, cosine between each scene and its variant:\n" + table)
        print(String(format: "near duplicates: lowest variant pair %.4f (%@), highest unrelated pair %.4f (%@ and %@), %d variant pairs over %d images",
                     closest.cosine, closest.file, unrelated.cosine, unrelated.a, unrelated.b, variantPairs.count, entries.count))
        XCTAssertGreaterThanOrEqual(variantPairs.count, 9)
        // The harshest variant (the coral reef at JPEG quality 40, 0.896) still sits above the closest
        // unrelated pair (solid grey against the grey gradient, 0.889), in llama-server's vectors too.
        XCTAssertGreaterThan(closest.cosine, unrelated.cosine, "a threshold separating variants from unrelated images")
        var nearest = 0
        for (i, e) in entries.enumerated() {
            guard let source = e.source, let original = index[source] else { continue }
            let rival = entries.indices.filter { $0 != i && $0 != original }.max { cosine(i, $0) < cosine(i, $1) }!
            let margin = cosine(i, original) - cosine(i, rival)
            print(String(format: "near duplicates: %@ original %.4f, nearest other %.4f (%@), margin %+.4f",
                         e.file, cosine(i, original), cosine(i, rival), entries[rival].file, margin))
            if margin > 0 { nearest += 1 }
        }
        XCTAssertEqual(nearest, variantPairs.count, "variants whose nearest neighbour is their original")
    }

    func testMatryoshkaAndNormalisation() async throws {
        let model = try XCTUnwrap(Self.model)
        let image = ImageInput.url(try entry("scenes/04-desert-dunes.jpg").url)
        let full = try await model.embed(image: image, options: ImageEmbedOptions(normalise: false))
        XCTAssertEqual(full.vector.count, 768)
        XCTAssertNotEqual(norm(full.vector), 1, accuracy: 1e-3)
        for dims in EmbedOptions.allowedDimensions {
            let e = try await model.embed(image: image, options: ImageEmbedOptions(dimensions: dims))
            XCTAssertEqual(e.vector.count, dims)
            XCTAssertEqual(norm(e.vector), 1, accuracy: 1e-5)
            XCTAssertFalse(e.truncated)
            let expected = Array(full.vector.prefix(dims))
            let n = norm(expected)
            for i in 0..<dims { XCTAssertEqual(e.vector[i], expected[i] / n, accuracy: 1e-5) }
        }
    }

    func testDataAndCGImageInputsMatchTheFile() async throws {
        let model = try XCTUnwrap(Self.model)
        for file in ["synthetic/07-shapes.jpg", "synthetic/08-checker.png", "scenes/03-snowy-peaks.jpg"] {
            let url = try entry(file).url
            let data = try Data(contentsOf: url)
            let source = try XCTUnwrap(CGImageSourceCreateWithData(data as CFData, nil))
            let cgImage = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
            let out = try await model.embed(images: [.url(url), .data(data), .cgImage(cgImage)])
            XCTAssertEqual(out[1].vector, out[0].vector, "\(file) as data")
            let diff = TestSupport.maxAbsDiff(out[2].vector, out[0].vector)
            print("\(file) as CGImage against the file: max abs diff \(diff), cosine \(TestSupport.cosine(out[2].vector, out[0].vector))")
            if file.hasSuffix(".png") {
                XCTAssertEqual(out[2].vector, out[0].vector, "\(file) as CGImage")
            } else {
                // ImageIO decodes JPEG a level or two away from stb_image (Architecture.md).
                XCTAssertGreaterThan(TestSupport.cosine(out[2].vector, out[0].vector), 0.99, "\(file) as CGImage")
            }
        }
    }

    func testBatchingIsIndependentAndReportsProgress() async throws {
        let model = try XCTUnwrap(Self.model)
        let entries = try Self.manifest().filter { $0.kind != "variant" }
        let images = entries.map { ImageInput.url($0.url) }
        XCTAssertGreaterThanOrEqual(images.count, 15)
        let autoProgress = ProgressRecorder(), threeProgress = ProgressRecorder()
        let auto = try await model.embed(images: images) { autoProgress.record($0) }
        let one = try await model.embed(images: images, options: ImageEmbedOptions(batchSize: 1))
        let three = try await model.embed(images: images, options: ImageEmbedOptions(batchSize: 3)) { threeProgress.record($0) }
        for i in images.indices {
            XCTAssertLessThan(TestSupport.maxAbsDiff(auto[i].vector, one[i].vector), 1e-5, entries[i].file)
            XCTAssertLessThan(TestSupport.maxAbsDiff(auto[i].vector, three[i].vector), 1e-5, entries[i].file)
        }
        let rows = auto.reduce(0) { $0 + $1.tokenCount }
        // A merge window is 3 by 3 patches.
        let patches = auto.map { ($0.tokenCount - EmbeddingModel.imageFrameRows) * 9 }
        let autoBatches = EmbeddingModel.batches(patches.map { [$0] }, batchSize: nil, budgets: [ImageEmbedOptions.patchBudget]).count
        XCTAssertGreaterThan(autoBatches, 1)
        for (events, batches) in [(autoProgress.events, autoBatches), (threeProgress.events, (images.count + 2) / 3)] {
            XCTAssertEqual(events.count, batches)
            XCTAssertEqual(events.map(\.completedTexts), events.map(\.completedTexts).sorted())
            XCTAssertEqual(events.last?.completedTexts, images.count)
            XCTAssertEqual(events.last?.totalTexts, images.count)
            XCTAssertEqual(events.last?.completedTokens, rows)
        }
    }

    func testCancellationBetweenBatches() async throws {
        let model = try XCTUnwrap(Self.model)
        let url = try entry("synthetic/00-solid-red.png").url
        let task = Task {
            try await model.embed(images: Array(repeating: .url(url), count: 12), options: ImageEmbedOptions(batchSize: 1))
        }
        task.cancel()
        do {
            _ = try await task.value
        } catch is CancellationError {
            return
        }
        XCTFail("a cancelled embed must throw CancellationError")
    }

    func testInvalidOptionsAreRejected() async throws {
        let model = try XCTUnwrap(Self.model)
        let image = ImageInput.url(try entry("synthetic/09-tiny-noise.png").url)
        for options in [ImageEmbedOptions(maxImageTokens: 0), ImageEmbedOptions(maxImageTokens: 1121),
                        ImageEmbedOptions(dimensions: 100), ImageEmbedOptions(batchSize: 0)] {
            do {
                _ = try await model.embed(image: image, options: options)
                XCTFail("\(options) must be rejected")
            } catch GemmaEmbedError.invalidOptions { }
        }
        for cap in [1, 1120] {
            let e = try await model.embed(image: image, options: ImageEmbedOptions(maxImageTokens: cap))
            XCTAssertEqual(e.vector.count, 768, "maxImageTokens \(cap)")
        }
        XCTAssertEqual(ImageEmbedOptions().maxImageTokens, 256)
    }

    func testUnreadableImageNamesItsIndexAndFile() async throws {
        let model = try XCTUnwrap(Self.model)
        let good = try entry("synthetic/00-solid-red.png").url
        let bad = FileManager.default.temporaryDirectory.appendingPathComponent("not-an-image-\(UUID().uuidString).jpg")
        try Data("plain text, not a picture".utf8).write(to: bad)
        defer { try? FileManager.default.removeItem(at: bad) }
        do {
            _ = try await model.embed(images: [.url(good), .url(bad)])
            XCTFail("a text file must not embed as an image")
        } catch GemmaEmbedError.imageDecoding(let reason) {
            XCTAssertTrue(reason.contains("image 1"), reason)
            XCTAssertTrue(reason.contains(bad.path), reason)
        }
        do {
            _ = try await model.embed(images: [.data(Data("nope".utf8))])
            XCTFail("bytes that are no image must not embed")
        } catch GemmaEmbedError.imageDecoding(let reason) {
            XCTAssertTrue(reason.hasPrefix("image 0:"), reason)
        }
    }

    func testInfoReportsTheImageEncoder() throws {
        let model = try XCTUnwrap(Self.model)
        let info = try XCTUnwrap(model.info.imageEncoder)
        XCTAssertEqual(info.layerCount, 16)
        XCTAssertEqual(info.hiddenSize, 768)
        XCTAssertEqual(info.patchSize, 16)
        XCTAssertEqual(info.projectionDim, 512)
        XCTAssertGreaterThan(info.fileSize, 0)
        XCTAssertNotNil(model.loadTimings["imageEncoder"])
    }

    func testWithoutTheImageEncoderImagesAreRefused() async throws {
        let model = try await EmbeddingModel.load(from: try TestSupport.requireWeights())
        XCTAssertNil(model.info.imageEncoder)
        XCTAssertNil(model.loadTimings["imageEncoder"])
        do {
            _ = try await model.embed(image: .url(try entry("synthetic/00-solid-red.png").url))
            XCTFail("a model loaded without the image encoder must refuse images")
        } catch GemmaEmbedError.imageEncoderNotLoaded { }
    }
}
