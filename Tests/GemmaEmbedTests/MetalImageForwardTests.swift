// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import XCTest
@testable import GemmaEmbed

final class MetalImageForwardTests: XCTestCase {
    final class Env {
        let forward: MetalForward
        let vision: VisionTower
        let preprocessor: ImagePreprocessor
        let reference: CPUVisionReference

        init(forward: MetalForward, vision: VisionTower, preprocessor: ImagePreprocessor, reference: CPUVisionReference) {
            self.forward = forward
            self.vision = vision
            self.preprocessor = preprocessor
            self.reference = reference
        }
    }

    nonisolated(unsafe) static var env: Env?

    override func setUpWithError() throws {
        if Self.env == nil {
            let file = try GGUFFile(url: try TestSupport.requireWeights())
            let visionFile = try GGUFFile(url: try TestSupport.requireImageEncoder())
            let context = try MetalContext()
            let config = try ModelConfig(gguf: file, deviceName: context.device.name, fileSHA256: nil)
            let visionConfig = try VisionConfig(gguf: visionFile, fileSHA256: nil)
            let visionWeights = try VisionWeights(gguf: visionFile, config: visionConfig)
            let vision = try VisionTower(context: context, config: visionConfig, weights: visionWeights,
                                         markers: try SpanMarkers.image(tokenizer: Tokenizer(gguf: file)))
            let forward = try MetalForward(context: context, config: config, weights: try ModelWeights(gguf: file, config: config), vision: vision)
            Self.env = Env(forward: forward, vision: vision, preprocessor: ImagePreprocessor(config: visionConfig),
                           reference: CPUVisionReference(config: visionConfig, weights: visionWeights))
        }
    }

    static let externalFixtures = TestSupport.modelsDirectory.appendingPathComponent("fixtures/images")

    struct FixtureImage {
        let file: String
        let kind: String
        let url: URL
        var isJPEG: Bool { file.hasSuffix(".jpg") }
    }

    /// Every manifest image found in the test bundle or, failing that, in `models/fixtures/images`.
    static func fixtureImages() throws -> [FixtureImage] {
        let data = try Data(contentsOf: TestSupport.fixture("images/manifest.json"))
        let manifest = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [[String: Any]])
        return manifest.compactMap { entry in
            let file = entry["file"] as! String
            let external = externalFixtures.appendingPathComponent(file)
            guard let url = (try? TestSupport.fixture("images/" + file)) ??
                    (FileManager.default.isReadableFile(atPath: external.path) ? external : nil) else { return nil }
            return FixtureImage(file: file, kind: entry["kind"] as! String, url: url)
        }
    }

    static func normalised(_ v: [Float]) -> [Float] {
        let n = v.reduce(0) { $0 + $1 * $1 }.squareRoot()
        return v.map { $0 / n }
    }

    func prepare(_ image: FixtureImage, maxImageTokens: Int = 256) throws -> PreparedImage {
        try XCTUnwrap(Self.env).preprocessor.prepare(.url(image.url), maxImageTokens: maxImageTokens)
    }

    /// The vision tower alone, fed stb_image's bytes, against the CPU reference and llama.cpp's own output rows.
    func testVisionTowerMatchesCPUReferenceAndLlamaCpp() throws {
        let env = try XCTUnwrap(Self.env)
        let dumps = try CPUVisionReferenceTests.fullDumps.map { try TestSupport.requireVisionDump($0) }
        let images = try dumps.map { env.preprocessor.prepare(try $0.decodedRGB(), maxImageTokens: 256) }
        let got = try env.forward.imageVectors(images)
        let width = env.vision.config.info.projectionDim
        XCTAssertEqual(got.count, images.count)
        for ((dump, image), rows) in zip(zip(dumps, images), got) {
            XCTAssertEqual(rows.count, image.tokenCount * width, dump.name)
            let cpu = env.reference.forward(image)
            let llama = try dump.floats("output_embd.f32")
            let slices = (0..<image.tokenCount).map { t in t * width..<(t + 1) * width }
            let vsCPU = slices.map { TestSupport.cosine(Array(rows[$0]), Array(cpu[$0])) }
            let vsLlama = slices.map { TestSupport.cosine(Array(rows[$0]), Array(llama[$0])) }
            print("gpu vision \(dump.name): \(image.tokenCount) rows, min row cosine vs CPU \(vsCPU.min()!) (max abs diff \(TestSupport.maxAbsDiff(rows, cpu))), vs llama.cpp \(vsLlama.min()!) (max abs diff \(TestSupport.maxAbsDiff(rows, llama)))")
            // Written as `!(c > bound)` so a NaN row counts as a failure.
            XCTAssertEqual(vsCPU.filter { !($0 > 0.99999) }.count, 0, "\(dump.name) rows below 0.99999 vs CPU reference")
            XCTAssertEqual(vsLlama.filter { !($0 > 0.9995) }.count, 0, "\(dump.name) rows below 0.9995 vs llama.cpp")
        }
    }

    /// Every fixture image through GemmaEmbed's own decode, and every one with an stb_image decode
    /// on disk through stb's bytes as well, which isolates the GPU path from the decoder.
    func testAllFixtureImagesMatchLlamaServer() throws {
        let env = try XCTUnwrap(Self.env)
        let fixtures = try TestSupport.imageEmbeddingFixtures()
        let images = try Self.fixtureImages()
        XCTAssertGreaterThanOrEqual(images.count, 27)
        var cases: [(image: FixtureImage, decoder: String, prepared: PreparedImage)] = []
        for image in images {
            cases.append((image, "GemmaEmbed", try prepare(image)))
            let stem = ((image.file as NSString).lastPathComponent as NSString).deletingPathExtension
            if image.isJPEG, let dump = TestSupport.decodedRGBDump(stem), dump.imageURL.path.hasSuffix(image.file) {
                cases.append((image, "stb_image", env.preprocessor.prepare(try dump.decodedRGB(), maxImageTokens: 256)))
            }
        }
        var minByKind: [String: Float] = [:]
        for start in stride(from: 0, to: cases.count, by: 8) {
            let chunk = Array(cases[start..<min(start + 8, cases.count)])
            let outputs = try env.forward.run(images: chunk.map(\.prepared))
            XCTAssertEqual(outputs.count, chunk.count)
            for (c, out) in zip(chunk, outputs) {
                let fixture = try XCTUnwrap(fixtures[c.image.file], c.image.file)
                XCTAssertEqual(c.prepared.tokenCount + 4, fixture.promptTokens, c.image.file)
                let cosine = TestSupport.cosine(Self.normalised(out), fixture.embedding)
                let kind = "\(c.image.kind) \(c.image.isJPEG ? "jpeg" : "png") via \(c.decoder)"
                minByKind[kind] = min(minByKind[kind] ?? 1, cosine)
                print("gpu image embedding \(c.image.file) via \(c.decoder): \(c.prepared.width)x\(c.prepared.height), \(c.prepared.tokenCount + 4) rows, cosine \(cosine)")
                XCTAssertGreaterThan(cosine, 0.9995, "\(c.image.file) via \(c.decoder)")
            }
        }
        for (kind, cosine) in minByKind.sorted(by: { $0.key < $1.key }) { print("gpu image embedding min cosine \(kind): \(cosine)") }
        print("gpu image embeddings compared: \(images.count) images, \(cases.count) decodes")
    }

    func testMaxImageTokens1120MatchesLlamaServer() throws {
        let env = try XCTUnwrap(Self.env)
        let rows = try TestSupport.fixtureJSONLines("images/embeddings-1120.jsonl")
        let byFile = Dictionary(uniqueKeysWithValues: try Self.fixtureImages().map { ($0.file, $0) })
        var chosen: [(FixtureImage, [Float], Int)] = []
        for row in rows {
            XCTAssertEqual(row["max_image_tokens"] as? Int, 1120)
            let image = try XCTUnwrap(byFile[row["file"] as! String], "fixture \(row["file"]!) missing")
            chosen.append((image, (row["embedding"] as! [Double]).map(Float.init), row["prompt_tokens"] as! Int))
        }
        let prepared = try chosen.map { try prepare($0.0, maxImageTokens: 1120) }
        let outputs = try env.forward.run(images: prepared)
        for ((image, want, promptTokens), (p, out)) in zip(chosen, zip(prepared, outputs)) {
            XCTAssertEqual(p.tokenCount + 4, promptTokens, image.file)
            let cosine = TestSupport.cosine(Self.normalised(out), want)
            print("gpu image embedding at 1120 tokens \(image.file): \(p.width)x\(p.height), \(p.tokenCount + 4) rows, cosine \(cosine)")
            XCTAssertGreaterThan(cosine, 0.9995, image.file)
        }
    }

    /// Eight images of different sizes, the first and last of which is the subject.
    func mixedBatch() throws -> (subject: PreparedImage, others: [PreparedImage], files: [String]) {
        let images = Dictionary(uniqueKeysWithValues: try Self.fixtureImages().map { ($0.file, $0) })
        let files = ["synthetic/09-tiny-noise.png", "scenes/00-green-hills.jpg", "synthetic/00-solid-red.png", "scenes/04-desert-dunes.jpg",
                     "synthetic/01-solid-grey.png", "synthetic/03-gradient-rgb.jpg", "synthetic/06-text-times.jpg", "synthetic/02-gradient-h.png"]
        let prepared = try files.map { try prepare(try XCTUnwrap(images[$0], $0)) }
        XCTAssertEqual(Set(prepared.map { "\($0.width)x\($0.height)" }).count, 8, "the batch must mix eight sizes")
        return (prepared[0], Array(prepared.dropFirst()), files)
    }

    func testBatchIndependence() throws {
        let env = try XCTUnwrap(Self.env)
        let (subject, others, _) = try mixedBatch()
        let alone = Self.normalised(try env.forward.run(images: [subject])[0])
        let first = Self.normalised(try env.forward.run(images: [subject] + others)[0])
        let last = Self.normalised(try env.forward.run(images: others + [subject]).last!)
        let diffFirst = TestSupport.maxAbsDiff(alone, first), diffLast = TestSupport.maxAbsDiff(alone, last)
        print("image batch independence: alone vs first of 8 max abs diff \(diffFirst), alone vs last of 8 \(diffLast)")
        XCTAssertLessThan(diffFirst, 1e-5)
        XCTAssertLessThan(diffLast, 1e-5)
    }

    func testImageBatchIndependenceAcrossGemmTiles() throws {
        let env = try XCTUnwrap(Self.env)
        let images = Dictionary(uniqueKeysWithValues: try Self.fixtureImages().map { ($0.file, $0) })
        let files = ["scenes/00-green-hills.jpg", "synthetic/06-text-times.jpg", "scenes/04-desert-dunes.jpg"]
        let prepared = try files.map { try prepare(try XCTUnwrap(images[$0], $0)) }
        let schedule = env.forward.schedule
        let width = env.vision.config.info.hiddenSize
        let batchRows = MetalForward.paddedRows(prepared.map(\.patchCount).reduce(0, +))
        for image in prepared {
            XCTAssertNotEqual(schedule.gemmTile(rows: MetalForward.paddedRows(image.patchCount), outputs: width),
                              schedule.gemmTile(rows: batchRows, outputs: width))
        }
        let together = try env.forward.run(images: prepared)
        for (i, image) in prepared.enumerated() {
            let alone = try env.forward.run(images: [image])[0]
            let differing = zip(alone, together[i]).filter { $0.bitPattern != $1.bitPattern }.count
            XCTAssertEqual(differing, 0, "\(files[i]) must embed the same alone as in a batch; max abs diff \(TestSupport.maxAbsDiff(alone, together[i]))")
        }
    }

    func testConcurrentEncoderMatchesSerialOverRepeatedRuns() throws {
        let env = try XCTUnwrap(Self.env)
        let images = Dictionary(uniqueKeysWithValues: try Self.fixtureImages().map { ($0.file, $0) })
        let noise = try prepare(try XCTUnwrap(images["synthetic/09-tiny-noise.png"]))
        let red = try prepare(try XCTUnwrap(images["synthetic/00-solid-red.png"]))
        let texts = try TestSupport.tokenFixtures().map(\.tokens).sorted { $0.count < $1.count }
        func mixed(_ before: ArraySlice<[Int32]>, _ picture: PreparedImage, _ after: ArraySlice<[Int32]>) -> (PackedBatch, VisionBatch?) {
            var batch = PackedBatch()
            for t in before { batch.append(t) }
            batch.appendSpanSequence(markers: env.vision.markers, vectorRows: picture.tokenCount)
            for t in after { batch.append(t) }
            return (batch, VisionBatch(images: [picture]))
        }
        func text(_ tokens: [Int32]) -> (PackedBatch, VisionBatch?) {
            var batch = PackedBatch()
            batch.append(tokens)
            return (batch, nil)
        }
        let alternatingBatches: [(name: String, batches: [(PackedBatch, VisionBatch?)])] = [
            ("mixed", [mixed(texts.prefix(3), noise, texts.suffix(2)), mixed(texts[10..<12], red, texts.suffix(1))]),
            ("one short text", [text(texts[0]), text(texts[1])]),
        ]
        defer { env.forward.dispatchType = .concurrent }
        for c in alternatingBatches {
            env.forward.dispatchType = .serial
            let serial = try c.batches.map { try env.forward.run($0.0, images: $0.1) }
            XCTAssertNotEqual(serial[0].joined().map(\.bitPattern), serial[1].joined().map(\.bitPattern), "\(c.name): the two batches must differ")
            env.forward.dispatchType = .concurrent
            var differingRuns = 0
            for i in 0..<50 {
                let out = try env.forward.run(c.batches[i % 2].0, images: c.batches[i % 2].1)
                if zip(out.joined(), serial[i % 2].joined()).contains(where: { $0.bitPattern != $1.bitPattern }) { differingRuns += 1 }
            }
            XCTAssertEqual(differingRuns, 0, "\(c.name): runs on the concurrent encoder that differ from the serial encoder")
        }
    }

    func testBatchReturnsOneVectorPerImageInOrder() throws {
        let env = try XCTUnwrap(Self.env)
        let fixtures = try TestSupport.imageEmbeddingFixtures()
        let (subject, others, files) = try mixedBatch()
        let batch = [subject] + others
        let outputs = try env.forward.run(images: batch).map(Self.normalised)
        XCTAssertEqual(outputs.count, 8)
        let wants = try files.map { try XCTUnwrap(fixtures[$0], $0).embedding }
        for (i, out) in outputs.enumerated() {
            let best = wants.indices.max { TestSupport.cosine(out, wants[$0]) < TestSupport.cosine(out, wants[$1]) }!
            XCTAssertEqual(best, i, "output \(i) (\(files[i])) best matches \(files[best])")
        }

        _ = try env.forward.timeForward(images: batch)
        let eight = (0..<5).map { _ in try! env.forward.timeForward(images: batch) }.sorted()[2]
        let one = (0..<5).map { _ in try! env.forward.timeForward(images: [batch[3]]) }.sorted()[2]
        print(String(format: "image forward gpu: batch of 8 (%d patches) %.1f ms, one image (%d patches) %.1f ms",
                     batch.map(\.patchCount).reduce(0, +), eight * 1000, batch[3].patchCount, one * 1000))
    }
}
