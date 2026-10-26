// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import XCTest
@testable import GemmaEmbed

/// The sequence layout of a mixed input, without weights: one id per character stands in for the tokenizer.
final class MultimodalSequenceTests: XCTestCase {
    let markers = SpanMarkers(bos: 2, begin: 900, end: 901, eos: 1)
    let audioMarkers = SpanMarkers(bos: 2, begin: 800, end: 801, eos: 1)
    let image = MultimodalInput.Part.image(.data(Data()))
    let audio = MultimodalInput.Part.audio(.data(Data()))

    func ids(_ text: String) -> [Int32] { text.unicodeScalars.map { Int32($0.value) } }

    /// Image spans unless the part at that position is a clip.
    func sequence(_ prefix: String, _ parts: [MultimodalInput.Part], rows: [Int], maxTokens: Int = 100) -> MultimodalSequence? {
        var rows = rows.makeIterator()
        let spans = parts.compactMap { part -> EncoderSpan? in
            switch part {
            case .text: return nil
            case .image: return EncoderSpan(rows: rows.next()!, markers: markers)
            case .audio: return EncoderSpan(rows: rows.next()!, markers: audioMarkers)
            }
        }
        return MultimodalSequence(prefix: prefix, parts: parts, spans: spans, maxTokens: maxTokens, bos: 2, eos: 1, encode: ids)
    }

    /// Catches a clip framed with the image tokens, or its rows not continuing the image rows' numbering.
    func testAudioSpansUseTheirOwnFrameAndShareRowNumbering() throws {
        let s = try XCTUnwrap(sequence("P:", [.text("a"), image, audio, .text("b")], rows: [2, 3]))
        XCTAssertEqual(s.ids, [2] + ids("P:a") + [900, -1, -2, 901, 800, -3, -4, -5, 801] + ids("b") + [1])
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
