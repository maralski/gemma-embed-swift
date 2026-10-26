// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import XCTest
@testable import GemmaEmbed

final class EmbeddingModelTests: XCTestCase {
    nonisolated(unsafe) static var model: EmbeddingModel?

    override func setUp() async throws {
        if Self.model == nil {
            Self.model = try await EmbeddingModel.load(from: try TestSupport.requireWeights())
        }
    }

    func norm(_ v: [Float]) -> Float { v.reduce(0) { $0 + $1 * $1 }.squareRoot() }

    func testInfoAndDevice() async throws {
        let model = try XCTUnwrap(Self.model)
        XCTAssertEqual(model.info.architecture, "gemma-embedding2")
        XCTAssertEqual(model.info.outputDimensions, 768)
        XCTAssertEqual(model.info.maxTokens, 8192)
        XCTAssertFalse(model.device.name.isEmpty)
        XCTAssertEqual(model.tokenizer.bosId, 2)
    }

    func testEmbedMatchesFixtureAndAppliesPrefix() async throws {
        let model = try XCTUnwrap(Self.model)
        let fixtures = try TestSupport.tokenFixtures()
        let references = try TestSupport.embeddingFixtures()
        let query = fixtures[21]
        XCTAssertTrue(query.input.hasPrefix(TaskPrefix.query.prefix))
        let text = String(query.input.dropFirst(TaskPrefix.query.prefix.count))
        let e = try await model.embed(text, options: EmbedOptions(task: .query))
        XCTAssertEqual(e.tokenCount, query.tokens.count)
        XCTAssertFalse(e.truncated)
        XCTAssertEqual(norm(e.vector), 1, accuracy: 1e-5)
        XCTAssertGreaterThan(TestSupport.cosine(e.vector, references[21].embedding), 0.9995)
        let doc = fixtures[0]
        let docText = String(doc.input.dropFirst(TaskPrefix.document.prefix.count))
        let d = try await model.embed(docText)
        XCTAssertGreaterThan(TestSupport.cosine(d.vector, references[0].embedding), 0.9995)
        XCTAssertEqual(TaskPrefix.document(title: "Hello").prefix, "title: Hello | text: ")
        XCTAssertEqual(TaskPrefix.query("question answering").prefix, "task: question answering | query: ")
        XCTAssertEqual(TaskPrefix.none.prefix, "")
    }

    func testMatryoshkaAndNormalisation() async throws {
        let model = try XCTUnwrap(Self.model)
        let text = "Matryoshka dolls nest inside each other."
        let full = try await model.embed(text, options: EmbedOptions(normalise: false))
        XCTAssertEqual(full.vector.count, 768)
        XCTAssertNotEqual(norm(full.vector), 1, accuracy: 1e-3)
        for dims in EmbedOptions.allowedDimensions {
            let e = try await model.embed(text, options: EmbedOptions(dimensions: dims))
            XCTAssertEqual(e.vector.count, dims)
            XCTAssertEqual(norm(e.vector), 1, accuracy: 1e-5)
            let expected = Array(full.vector.prefix(dims))
            let n = norm(expected)
            for i in 0..<dims { XCTAssertEqual(e.vector[i], expected[i] / n, accuracy: 1e-5) }
        }
        do {
            _ = try await model.embed(text, options: EmbedOptions(dimensions: 100))
            XCTFail("dimensions 100 must be rejected")
        } catch GemmaEmbedError.invalidOptions { }
    }

    func testTruncationKeepsEOS() async throws {
        let model = try XCTUnwrap(Self.model)
        let text = Array(repeating: "a long sentence that keeps going", count: 50).joined(separator: " ")
        let e = try await model.embed(text, options: EmbedOptions(maxTokens: 16))
        XCTAssertTrue(e.truncated)
        XCTAssertEqual(e.tokenCount, 16)
        let ids = Array(model.tokenizer.encode(TaskPrefix.document.prefix + text).prefix(15)) + [model.tokenizer.eosId]
        let direct = try await model.embedTokens([ids])[0]
        let n = norm(direct)
        for i in 0..<768 { XCTAssertEqual(e.vector[i], direct[i] / n, accuracy: 1e-5) }
        let short = try await model.embed("short", options: EmbedOptions(maxTokens: 16))
        XCTAssertFalse(short.truncated)
    }

    func testBatchingIsIndependentAndReportsProgress() async throws {
        let model = try XCTUnwrap(Self.model)
        let texts = (0..<20).map { "Document number \($0) talks about " + Array(repeating: "topic \($0 % 3)", count: $0 * 3 + 1).joined(separator: " ") }
        let progress = ProgressRecorder()
        let auto = try await model.embed(texts) { progress.record($0) }
        let one = try await model.embed(texts, options: EmbedOptions(batchSize: 1))
        let three = try await model.embed(texts, options: EmbedOptions(batchSize: 3))
        for i in 0..<texts.count {
            XCTAssertLessThan(TestSupport.maxAbsDiff(auto[i].vector, one[i].vector), 1e-5)
            XCTAssertLessThan(TestSupport.maxAbsDiff(auto[i].vector, three[i].vector), 1e-5)
        }
        let events = progress.events
        XCTAssertFalse(events.isEmpty)
        XCTAssertEqual(events.last?.completedTexts, texts.count)
        XCTAssertEqual(events.last?.totalTexts, texts.count)
        XCTAssertEqual(events.last?.completedTokens, auto.reduce(0) { $0 + $1.tokenCount })
        XCTAssertGreaterThan(model.similarity(auto[3], auto[6]), model.similarity(auto[3], auto[7]))
    }

    func testBatchPlanning() {
        XCTAssertEqual(EmbeddingModel.batches([[5], [5], [5], [5]], batchSize: nil, budgets: [10]), [0..<2, 2..<4])
        XCTAssertEqual(EmbeddingModel.batches([[50], [5]], batchSize: nil, budgets: [10]), [0..<1, 1..<2])
        XCTAssertEqual(EmbeddingModel.batches([[1], [1], [1]], batchSize: 2, budgets: [10]), [0..<2, 2..<3])
        XCTAssertEqual(EmbeddingModel.batches([], batchSize: nil, budgets: [10]), [])
        // Each budget cuts on its own: a batch ignoring the second cost would hold all three.
        XCTAssertEqual(EmbeddingModel.batches([[1, 3], [1, 3], [1, 1]], batchSize: nil, budgets: [10, 6]), [0..<2, 2..<3])
        XCTAssertEqual(EmbeddingModel.batches([[6, 0], [6, 0]], batchSize: nil, budgets: [10, 6]), [0..<1, 1..<2])
    }

    func testCancellationBetweenBatches() async throws {
        let model = try XCTUnwrap(Self.model)
        let texts = (0..<12).map { "cancel me \($0)" }
        let task = Task {
            try await model.embed(texts, options: EmbedOptions(batchSize: 1))
        }
        task.cancel()
        do {
            _ = try await task.value
        } catch is CancellationError {
            return
        }
        XCTFail("a cancelled embed must throw CancellationError")
    }

    func testAutotuneCacheRoundTrip() throws {
        let model = try XCTUnwrap(Self.model)
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("gemma-autotune-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let tuner = Autotune(cacheDirectory: dir)
        var cache = tuner.loadCache()
        XCTAssertTrue(cache.isEmpty)
        let schedule = KernelSchedule(gemmTile: GemmTile.candidates[1], attentionQueries: 16)
        cache["key"] = Autotune.CacheEntry(schedule: schedule, defaultSeconds: 1, tunedSeconds: 0.9, date: Date())
        tuner.saveCache(cache)
        XCTAssertEqual(tuner.loadCache()["key"]?.schedule, schedule)
        XCTAssertEqual(model.schedule, .default)
    }
}

final class ProgressRecorder: @unchecked Sendable {
    private var stored: [EmbeddingProgress] = []
    private let lock = NSLock()

    func record(_ p: EmbeddingProgress) {
        lock.lock()
        stored.append(p)
        lock.unlock()
    }

    var events: [EmbeddingProgress] {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }
}
