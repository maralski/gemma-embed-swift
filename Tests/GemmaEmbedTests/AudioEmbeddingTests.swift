// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import XCTest
@testable import GemmaEmbed

final class AudioEmbeddingTests: XCTestCase {
    nonisolated(unsafe) static var model: EmbeddingModel?

    override func setUp() async throws {
        if Self.model == nil {
            Self.model = try await EmbeddingModel.load(from: try TestSupport.requireWeights(),
                                                       options: LoadOptions(audioEncoder: try TestSupport.requireImageEncoder()))
        }
    }

    func fixture(_ file: String) throws -> TestSupport.AudioFixture {
        guard let f = try TestSupport.audioFixtures().first(where: { $0.file == file }) else { throw XCTSkip("fixture clip \(file) missing") }
        return f
    }

    static let speechDirectory = TestSupport.modelsDirectory.appendingPathComponent("audio/speech")

    /// Every fixture clip through the public API in one call, against llama-server's vectors.
    func testAllFixtureClipsMatchLlamaServer() async throws {
        let model = try XCTUnwrap(Self.model)
        let fixtures = try TestSupport.audioFixtures()
        let reference = try TestSupport.audioEmbeddingFixtures()
        let embeddings = try await model.embed(audios: fixtures.map { .url($0.url) })
        var worst = (file: "", cosine: Float(1))
        for (fixture, e) in zip(fixtures, embeddings) {
            let want = try XCTUnwrap(reference[fixture.file], fixture.file)
            XCTAssertEqual(e.tokenCount, want.promptTokens, "\(fixture.file) tokens")
            XCTAssertFalse(e.truncated)
            let cosine = TestSupport.cosine(e.vector, want.embedding)
            if cosine < worst.cosine { worst = (fixture.file, cosine) }
            XCTAssertGreaterThan(cosine, 0.9999, fixture.file)
        }
        print("embed(audios:) vs llama-server over \(fixtures.count) clips: min cosine \(worst.cosine) (\(worst.file))")
    }

    /// Spoken sentences from `scripts/make_speech_samples.sh` against their transcripts and five
    /// off-topic texts: every clip must rank its own transcript first, with the ranks llama-server's
    /// own vectors give, and the similarities must agree with llama-server's within 0.002.
    func testSpeechRanksTranscriptsAsLlamaServerDoes() async throws {
        let model = try XCTUnwrap(Self.model)
        let dir = Self.speechDirectory
        let clipsURL = dir.appendingPathComponent("embeddings.jsonl"), queriesURL = dir.appendingPathComponent("queries.jsonl")
        guard FileManager.default.isReadableFile(atPath: clipsURL.path), FileManager.default.isReadableFile(atPath: queriesURL.path) else {
            throw XCTSkip("no speech samples under \(dir.path); run scripts/make_speech_samples.sh and scripts/oracle_audio.py --speech")
        }
        func rows(_ url: URL) throws -> [[String: Any]] {
            try String(contentsOf: url, encoding: .utf8).split(separator: "\n").map { try XCTUnwrap(JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any]) }
        }
        let clipRows = try rows(clipsURL).filter { FileManager.default.isReadableFile(atPath: dir.appendingPathComponent($0["file"] as! String).path) }
        let queryRows = try rows(queriesURL)
        guard clipRows.count >= 3 else { throw XCTSkip("fewer than three speech clips present") }

        let clips = try await model.embed(audios: clipRows.map { .url(dir.appendingPathComponent($0["file"] as! String)) })
        for (row, e) in zip(clipRows, clips) {
            let want = (row["embedding"] as! [Double]).map(Float.init)
            XCTAssertEqual(e.tokenCount, row["prompt_tokens"] as? Int, row["file"] as! String)
            XCTAssertGreaterThan(TestSupport.cosine(e.vector, want), 0.9999, row["file"] as! String)
        }
        var queries: [Embedding] = [], referenceQueries: [[Float]] = []
        for row in queryRows {
            let reference = row["query"] as! [String: Any]
            let e = try await model.embed(row["text"] as! String, options: EmbedOptions(task: .query))
            XCTAssertGreaterThan(TestSupport.cosine(e.vector, (reference["embedding"] as! [Double]).map(Float.init)), 0.9995, row["text"] as! String)
            queries.append(e)
            referenceQueries.append((reference["embedding"] as! [Double]).map(Float.init))
        }
        let similarity = clips.map { c in queries.map { model.similarity(c, $0) } }
        let reference = clipRows.map { row in referenceQueries.map { TestSupport.cosine((row["embedding"] as! [Double]).map(Float.init), $0) } }
        let worst = zip(similarity, reference).map { TestSupport.maxAbsDiff($0, $1) }.max()!
        print("speech cross-modal: largest similarity difference from llama-server's vectors \(worst)")
        XCTAssertLessThan(worst, 0.002)
        func rank(_ row: [Float], of query: Int) -> Int { row.filter { $0 > row[query] }.count + 1 }
        for (c, row) in clipRows.enumerated() {
            let own = try XCTUnwrap(queryRows.firstIndex { $0["describes"] as? String == row["file"] as? String }, "no transcript query for \(row["file"]!)")
            let r = rank(similarity[c], of: own), want = rank(reference[c], of: own)
            let margin = similarity[c][own] - similarity[c].enumerated().filter { $0.offset != own }.map(\.element).max()!
            print("speech \(row["file"]!): own transcript ranks \(r) of \(queries.count) (llama-server \(want)), own \(String(format: "%.3f", similarity[c][own])), margin \(String(format: "%+.3f", margin))")
            XCTAssertEqual(r, want, row["file"] as! String)
            XCTAssertEqual(r, 1, "\(row["file"]!) prefers another text to its own transcript")
        }
    }

    /// The same clip as a file, as its bytes and as decoded samples must embed to the same bits.
    func testDataAndSamplesInputsMatchTheFile() async throws {
        let model = try XCTUnwrap(Self.model)
        for file in ["formats/04-stereo.wav", "formats/02-rate-44100.wav", "formats/07-pcm24.wav"] {
            let url = try fixture(file).url
            let data = try Data(contentsOf: url)
            let decoded = try WAVDecoder.decode(data)
            let out = try await model.embed(audios: [.url(url), .data(data), .samples(decoded.samples, sampleRate: decoded.sampleRate, channels: decoded.channels)])
            XCTAssertEqual(out[1].vector, out[0].vector, "\(file) as data")
            XCTAssertEqual(out[2].vector, out[0].vector, "\(file) as samples")
            XCTAssertEqual(out[1].tokenCount, out[0].tokenCount)
        }
    }

    func testBatchingIsIndependentAndReportsProgress() async throws {
        let model = try XCTUnwrap(Self.model)
        let fixtures = try TestSupport.audioFixtures().filter { $0.kind != "format" }
        let clips = fixtures.map { AudioInput.url($0.url) }
        XCTAssertGreaterThanOrEqual(clips.count, 15)
        let autoProgress = ProgressRecorder(), threeProgress = ProgressRecorder()
        let auto = try await model.embed(audios: clips) { autoProgress.record($0) }
        let one = try await model.embed(audios: clips, options: AudioEmbedOptions(batchSize: 1))
        let three = try await model.embed(audios: clips, options: AudioEmbedOptions(batchSize: 3)) { threeProgress.record($0) }
        for i in clips.indices {
            XCTAssertEqual(auto[i].vector, one[i].vector, fixtures[i].file)
            XCTAssertEqual(auto[i].vector, three[i].vector, fixtures[i].file)
        }
        let rows = auto.reduce(0) { $0 + $1.tokenCount }
        let frontEnd = AudioFrontEnd(melBins: 128)
        let frames = try fixtures.map { try frontEnd.prepare(try AudioDecoder.decode(.url($0.url))).frameCount }
        let autoBatches = EmbeddingModel.batches(frames.map { [$0] }, batchSize: nil, budgets: [AudioEmbedOptions.frameBudget]).count
        for (events, batches) in [(autoProgress.events, autoBatches), (threeProgress.events, (clips.count + 2) / 3)] {
            XCTAssertEqual(events.count, batches)
            XCTAssertEqual(events.map(\.completedTexts), events.map(\.completedTexts).sorted())
            XCTAssertEqual(events.last?.completedTexts, clips.count)
            XCTAssertEqual(events.last?.totalTexts, clips.count)
            XCTAssertEqual(events.last?.completedTokens, rows)
        }
    }

    func testCancellationBetweenBatches() async throws {
        let model = try XCTUnwrap(Self.model)
        let url = try fixture("signals/08-short-tone.wav").url
        let task = Task {
            try await model.embed(audios: Array(repeating: .url(url), count: 12), options: AudioEmbedOptions(batchSize: 1))
        }
        task.cancel()
        do {
            _ = try await task.value
        } catch is CancellationError {
            return
        }
        XCTFail("a cancelled embed must throw CancellationError")
    }

    func testInvalidOptionsAndOverlongClipsAreRejected() async throws {
        let model = try XCTUnwrap(Self.model)
        let clip = AudioInput.url(try fixture("signals/08-short-tone.wav").url)
        for options in [AudioEmbedOptions(dimensions: 100), AudioEmbedOptions(batchSize: 0), AudioEmbedOptions(maxTokens: 4), AudioEmbedOptions(maxTokens: 9000)] {
            do {
                _ = try await model.embed(audio: clip, options: options)
                XCTFail("\(options) must be rejected")
            } catch GemmaEmbedError.invalidOptions {}
        }
        // The 35 s clip needs 879 rows.
        let long = AudioInput.url(try fixture("long/01-35s.wav").url)
        do {
            _ = try await model.embed(audio: long, options: AudioEmbedOptions(maxTokens: 878))
            XCTFail("a clip over maxTokens must be rejected")
        } catch GemmaEmbedError.invalidOptions(let reason) {
            XCTAssertTrue(reason.contains("879"), reason)
        }
        let exact = try await model.embed(audio: long, options: AudioEmbedOptions(maxTokens: 879))
        XCTAssertEqual(exact.tokenCount, 879)
        do {
            _ = try await model.embed(audio: .samples([0.1], sampleRate: 16000))
            XCTFail("a one-sample clip has no frame and must be refused")
        } catch GemmaEmbedError.audioDecoding {}
        let tiny = try await model.embed(audio: .samples([0.1, 0.2], sampleRate: 16000))
        XCTAssertEqual(tiny.tokenCount, 5)
    }

    func testMatryoshkaAndNormalisation() async throws {
        let model = try XCTUnwrap(Self.model)
        let clip = AudioInput.url(try fixture("tones/00-sine-440.wav").url)
        let full = try await model.embed(audio: clip)
        let raw = try await model.embed(audio: clip, options: AudioEmbedOptions(normalise: false))
        XCTAssertEqual(full.vector.reduce(0) { $0 + $1 * $1 }.squareRoot(), 1, accuracy: 1e-4)
        XCTAssertGreaterThan(TestSupport.cosine(full.vector, raw.vector), 0.99999)
        XCTAssertNotEqual(raw.vector.reduce(0) { $0 + $1 * $1 }.squareRoot(), 1, accuracy: 1e-3)
        for dims in [512, 256, 128] {
            let e = try await model.embed(audio: clip, options: AudioEmbedOptions(dimensions: dims))
            XCTAssertEqual(e.vector.count, dims)
            XCTAssertEqual(e.vector.reduce(0) { $0 + $1 * $1 }.squareRoot(), 1, accuracy: 1e-4)
            XCTAssertGreaterThan(TestSupport.cosine(e.vector, Array(full.vector.prefix(dims))), 0.99999)
        }
    }

    func testUnreadableClipNamesItsIndexAndFile() async throws {
        let model = try XCTUnwrap(Self.model)
        let good = AudioInput.url(try fixture("signals/08-short-tone.wav").url)
        let missing = URL(fileURLWithPath: "/nonexistent/clip.wav")
        do {
            _ = try await model.embed(audios: [good, .url(missing)])
            XCTFail("a missing file must throw")
        } catch GemmaEmbedError.audioDecoding(let reason) {
            XCTAssertTrue(reason.contains("audio 1"), reason)
            XCTAssertTrue(reason.contains(missing.path), reason)
        }
        do {
            _ = try await model.embed(audio: .data(Data("not audio at all".utf8)))
            XCTFail("junk bytes must throw")
        } catch GemmaEmbedError.audioDecoding {}
    }

    func testInfoReportsTheAudioEncoder() throws {
        let model = try XCTUnwrap(Self.model)
        let encoder = try XCTUnwrap(model.info.audioEncoder)
        XCTAssertEqual(encoder.layerCount, 12)
        XCTAssertEqual(encoder.hiddenSize, 1024)
        XCTAssertEqual(encoder.headCount, 8)
        XCTAssertEqual(encoder.melBins, 128)
        XCTAssertEqual(encoder.projectionDim, 512)
        XCTAssertNil(model.info.imageEncoder)
    }

    func testWithoutTheAudioEncoderClipsAreRefused() async throws {
        let model = try await EmbeddingModel.load(from: try TestSupport.requireWeights())
        XCTAssertNil(model.info.audioEncoder)
        do {
            _ = try await model.embed(audio: .url(try fixture("signals/08-short-tone.wav").url))
            XCTFail("a clip was embedded without the audio encoder")
        } catch GemmaEmbedError.audioEncoderNotLoaded {}
    }

    /// Loading both towers from one file maps it once and serves both modalities.
    func testBothEncodersFromOneFile() async throws {
        let mmproj = try TestSupport.requireImageEncoder()
        let model = try await EmbeddingModel.load(from: try TestSupport.requireWeights(), options: LoadOptions(imageEncoder: mmproj, audioEncoder: mmproj))
        XCTAssertNotNil(model.info.imageEncoder)
        XCTAssertNotNil(model.info.audioEncoder)
        let clip = try await model.embed(audio: .url(try fixture("signals/08-short-tone.wav").url))
        let same = try await XCTUnwrap(Self.model).embed(audio: .url(try fixture("signals/08-short-tone.wav").url))
        XCTAssertEqual(clip.vector, same.vector)
    }
}
