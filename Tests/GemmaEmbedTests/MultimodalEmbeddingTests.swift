import Foundation
import XCTest
@testable import GemmaEmbed

final class MultimodalEmbeddingTests: XCTestCase {
    nonisolated(unsafe) static var model: EmbeddingModel?

    override func setUp() async throws {
        if Self.model == nil {
            let mmproj = try TestSupport.requireImageEncoder()
            Self.model = try await EmbeddingModel.load(from: try TestSupport.requireWeights(),
                                                       options: LoadOptions(imageEncoder: mmproj, audioEncoder: mmproj))
        }
    }

    func clip(_ file: String) throws -> AudioInput {
        guard let fixture = try TestSupport.audioFixtures().first(where: { $0.file == file }) else { throw XCTSkip("fixture clip \(file) missing") }
        return .url(fixture.url)
    }

    func image(_ file: String) throws -> ImageInput {
        guard let fixture = try MetalImageForwardTests.fixtureImages().first(where: { $0.file == file }) else {
            throw XCTSkip("fixture image \(file) missing")
        }
        return .url(fixture.url)
    }

    func maxDifference(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).map { abs($0 - $1) }.max() ?? .infinity
    }

    func assertSame(_ a: Embedding, _ b: Embedding, _ label: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a.tokenCount, b.tokenCount, label, file: file, line: line)
        XCTAssertEqual(a.vector.count, b.vector.count, label, file: file, line: line)
        XCTAssertGreaterThan(TestSupport.cosine(a.vector, b.vector), 0.99999, label, file: file, line: line)
        XCTAssertLessThan(maxDifference(a.vector, b.vector), 1e-4, label, file: file, line: line)
    }

    static let text = "A blue computer sits on a wooden desk next to a window."

    /// Catches a text-only input laid out differently from `embed(_:)`: a missing BOS or EOS,
    /// the prefix tokenized apart from the text, or special tokens added twice.
    func testTextOnlyInputMatchesTextEmbedding() async throws {
        let model = try XCTUnwrap(Self.model)
        for task in [TaskPrefix.document, .query] {
            let mixed = try await model.embed(MultimodalInput([.text(Self.text)]), options: MultimodalEmbedOptions(task: task))
            let text = try await model.embed(Self.text, options: EmbedOptions(task: task))
            assertSame(mixed, text, "\(task)")
        }
    }

    /// Catches an image-only input laid out differently from `embed(images:)`, or image rows
    /// entering scaled like token rows.
    func testImageOnlyInputMatchesImageEmbedding() async throws {
        let model = try XCTUnwrap(Self.model)
        for file in ["scenes/00-green-hills.jpg", "synthetic/07-shapes.jpg"] {
            let picture = try image(file)
            let mixed = try await model.embed(MultimodalInput([.image(picture)]), options: MultimodalEmbedOptions(task: .none))
            let alone = try await model.embed(image: picture)
            assertSame(mixed, alone, file)
        }
    }

    /// Catches a mixed input that drops its text or its image, and image rows that point at
    /// another input's vectors when several inputs with images share one GPU batch.
    func testTextWithImageReadsBothAndIsBatchIndependent() async throws {
        let model = try XCTUnwrap(Self.model)
        let hills = try image("scenes/00-green-hills.jpg"), shapes = try image("synthetic/07-shapes.jpg")
        let input = MultimodalInput(text: Self.text, images: [hills])
        let mixed = try await model.embed(input)
        let text = try await model.embed(Self.text)
        let picture = try await model.embed(image: hills)

        let textTokens = model.tokenizer.encode(TaskPrefix.document.prefix + Self.text, addSpecialTokens: false).count
        let imageTokens = picture.tokenCount - EmbeddingModel.imageFrameRows
        XCTAssertEqual(mixed.tokenCount, 1 + textTokens + 1 + imageTokens + 1 + 1)
        XCTAssertFalse(mixed.truncated)
        XCTAssertEqual(mixed.vector.reduce(0) { $0 + $1 * $1 }.squareRoot(), 1, accuracy: 1e-4)
        XCTAssertLessThan(TestSupport.cosine(mixed.vector, text.vector), 0.999)
        XCTAssertLessThan(TestSupport.cosine(mixed.vector, picture.vector), 0.999)

        let inputs = [MultimodalInput(text: "shapes on a white background", images: [shapes]),
                      MultimodalInput([.text("a short caption")]),
                      input,
                      MultimodalInput([.image(shapes), .text("between"), .image(hills)])]
        let batched = try await model.embed(inputs, options: MultimodalEmbedOptions(batchSize: inputs.count))
        for (i, other) in inputs.enumerated() {
            let solo = try await model.embed(other)
            XCTAssertEqual(batched[i].tokenCount, solo.tokenCount, "input \(i)")
            XCTAssertGreaterThan(TestSupport.cosine(batched[i].vector, solo.vector), 0.9999, "input \(i)")
        }
    }

    /// Catches the part order being lost on the way to the model.
    func testOrderReachesTheModel() async throws {
        let model = try XCTUnwrap(Self.model)
        let hills = try image("scenes/00-green-hills.jpg")
        let textFirst = try await model.embed(MultimodalInput([.text(Self.text), .image(hills)]))
        let imageFirst = try await model.embed(MultimodalInput([.image(hills), .text(Self.text)]))
        XCTAssertLessThan(TestSupport.cosine(textFirst.vector, imageFirst.vector), 0.9999)
    }

    /// Catches truncation that cuts the image, overshoots `maxTokens`, or drops all the text when
    /// some of it fits; and an input whose image alone overflows getting through.
    func testTruncationKeepsTheImage() async throws {
        let model = try XCTUnwrap(Self.model)
        let hills = try image("scenes/00-green-hills.jpg")
        let picture = try await model.embed(image: hills)
        let long = String(repeating: Self.text + " ", count: 100)
        let maxTokens = picture.tokenCount + 32
        let cut = try await model.embed(MultimodalInput(text: long, images: [hills]), options: MultimodalEmbedOptions(maxTokens: maxTokens))
        XCTAssertTrue(cut.truncated)
        XCTAssertEqual(cut.tokenCount, maxTokens)
        XCTAssertLessThan(TestSupport.cosine(cut.vector, picture.vector), 0.999)

        do {
            _ = try await model.embed(MultimodalInput(text: long, images: [hills]),
                                      options: MultimodalEmbedOptions(maxTokens: picture.tokenCount - 1))
            XCTFail("an image that alone exceeds maxTokens was embedded")
        } catch GemmaEmbedError.invalidOptions {}
    }

    /// Catches text-only inputs requiring the image encoder, and an image silently dropped without it.
    /// Catches a clip-only input laid out differently from `embed(audios:)`, or audio rows framed
    /// with the image tokens.
    func testAudioOnlyInputMatchesAudioEmbedding() async throws {
        let model = try XCTUnwrap(Self.model)
        for file in ["signals/10-tone-noise.wav", "long/01-35s.wav"] {
            let sound = try clip(file)
            let mixed = try await model.embed(MultimodalInput([.audio(sound)]), options: MultimodalEmbedOptions(task: .none, maxTokens: 1000))
            let alone = try await model.embed(audio: sound)
            assertSame(mixed, alone, file)
        }
    }

    /// Catches audio rows pointing at image rows or another input's clip when text, images and
    /// clips share one GPU batch, and a clip's rows not counting toward the token limit.
    func testTextImageAndAudioTogetherAreBatchIndependent() async throws {
        let model = try XCTUnwrap(Self.model)
        let hills = try image("scenes/00-green-hills.jpg"), shapes = try image("synthetic/07-shapes.jpg")
        let tone = try clip("signals/08-short-tone.wav"), chirp = try clip("signals/09-short-chirp.wav")
        let input = MultimodalInput([.text("a tone, then a picture"), .audio(tone), .image(hills), .text("and a chirp"), .audio(chirp)])
        let mixed = try await model.embed(input)
        let text = try await model.embed("a tone, then a picture and a chirp")
        let soundAlone = try await model.embed(audio: tone)
        let picture = try await model.embed(image: hills)
        let textTokens = model.tokenizer.encode(TaskPrefix.document.prefix + "a tone, then a picture", addSpecialTokens: false).count
            + model.tokenizer.encode("and a chirp", addSpecialTokens: false).count
        let chirpRows = try await model.embed(audio: chirp).tokenCount - EmbeddingModel.audioFrameRows
        XCTAssertEqual(mixed.tokenCount, 2 + textTokens + (soundAlone.tokenCount - 2) + (picture.tokenCount - 2) + chirpRows + 2)
        XCTAssertLessThan(TestSupport.cosine(mixed.vector, text.vector), 0.999)
        XCTAssertLessThan(TestSupport.cosine(mixed.vector, soundAlone.vector), 0.999)

        let inputs = [MultimodalInput([.audio(chirp)]),
                      input,
                      MultimodalInput([.image(shapes), .text("between"), .audio(tone)]),
                      MultimodalInput([.text("a short caption")]),
                      MultimodalInput([.audio(tone), .audio(chirp)])]
        let batched = try await model.embed(inputs, options: MultimodalEmbedOptions(batchSize: inputs.count))
        for (i, other) in inputs.enumerated() {
            let solo = try await model.embed(other)
            XCTAssertEqual(batched[i].tokenCount, solo.tokenCount, "input \(i)")
            XCTAssertGreaterThan(TestSupport.cosine(batched[i].vector, solo.vector), 0.9999, "input \(i)")
        }
        do {
            _ = try await model.embed(input, options: MultimodalEmbedOptions(maxTokens: 60))
            XCTFail("an input whose clips and image alone overflow must be refused")
        } catch GemmaEmbedError.invalidOptions {}
    }

    func testWithoutAudioEncoder() async throws {
        let model = try await EmbeddingModel.load(from: try TestSupport.requireWeights(),
                                                   options: LoadOptions(imageEncoder: try TestSupport.requireImageEncoder()))
        do {
            _ = try await model.embed(MultimodalInput([.text(Self.text), .audio(try clip("signals/08-short-tone.wav"))]))
            XCTFail("a clip was embedded without the audio encoder")
        } catch GemmaEmbedError.audioEncoderNotLoaded {}
    }

    func testWithoutImageEncoder() async throws {
        let model = try await EmbeddingModel.load(from: try TestSupport.requireWeights())
        let mixed = try await model.embed(MultimodalInput([.text(Self.text)]))
        let text = try await model.embed(Self.text)
        assertSame(mixed, text, "text only")
        do {
            _ = try await model.embed(MultimodalInput(text: Self.text, images: [try image("scenes/00-green-hills.jpg")]))
            XCTFail("an image was embedded without the image encoder")
        } catch GemmaEmbedError.imageEncoderNotLoaded {}
    }
}
