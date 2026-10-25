import Foundation
import XCTest
@testable import GemmaEmbed

/// The sequence layout of a mixed input, without weights: one id per character stands in for the tokenizer.
final class MultimodalSequenceTests: XCTestCase {
    let markers = SpanMarkers(bos: 2, begin: 900, end: 901, eos: 1)
    let image = MultimodalInput.Part.image(.data(Data()))

    func ids(_ text: String) -> [Int32] { text.unicodeScalars.map { Int32($0.value) } }

    func sequence(_ prefix: String, _ parts: [MultimodalInput.Part], rows: [Int], maxTokens: Int = 100) -> MultimodalSequence? {
        MultimodalSequence(prefix: prefix, parts: parts, imageRows: rows, maxTokens: maxTokens, markers: markers, encode: ids)
    }

    /// Catches the prefix landing anywhere but in front of the first text, a missing or misplaced
    /// frame token, parts out of order, and image rows numbered from zero for each image, which
    /// would make the second image read the first one's vectors.
    func testLaysOutPartsInOrderInsideImageFrames() throws {
        let s = try XCTUnwrap(sequence("P:", [.text("ab"), image, .text("c"), image], rows: [2, 3]))
        XCTAssertEqual(s.ids, [2] + ids("P:ab") + [900, -1, -2, 901] + ids("c") + [900, -3, -4, -5, 901, 1])
        XCTAssertFalse(s.truncated)
    }

    /// Catches the prefix being dropped, or joined to a text after the image, when the input opens with an image.
    func testPrefixLeadsAnInputThatOpensWithAnImage() throws {
        let s = try XCTUnwrap(sequence("P", [image, .text("a")], rows: [1]))
        XCTAssertEqual(s.ids, [2] + ids("P") + [900, -1, 901] + ids("a") + [1])
    }

    /// Catches cutting the first text before the last, cutting from the front of a text, and
    /// cutting into the image frames or the EOS.
    func testTruncationCutsTextFromTheEndAndKeepsFrames() throws {
        let s = try XCTUnwrap(sequence("", [.text("abcd"), image, .text("xyz")], rows: [2], maxTokens: 9))
        XCTAssertEqual(s.ids, [2] + ids("abc") + [900, -1, -2, 901, 1])
        XCTAssertTrue(s.truncated)
    }

    /// Catches an off-by-one at the limit: images plus frame that fill `maxTokens` exactly are
    /// kept with all their text cut, one more row is refused.
    func testImagesThatAloneOverflowAreRefused() throws {
        let fits = try XCTUnwrap(sequence("P", [.text("a"), image], rows: [5], maxTokens: 9))
        XCTAssertEqual(fits.ids, [2, 900, -1, -2, -3, -4, -5, 901, 1])
        XCTAssertTrue(fits.truncated)
        XCTAssertNil(sequence("P", [.text("a"), image], rows: [5], maxTokens: 8))
    }

    /// Catches an empty input losing its prefix, its BOS or its EOS.
    func testEmptyInputIsThePrefixAlone() throws {
        XCTAssertEqual(try XCTUnwrap(sequence("P", [], rows: [])).ids, [2] + ids("P") + [1])
        XCTAssertEqual(try XCTUnwrap(sequence("", [], rows: [])).ids, [2, 1])
    }
}
