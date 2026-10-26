// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import XCTest
@testable import GemmaEmbed

final class TokenizerTests: XCTestCase {
    nonisolated(unsafe) static var tokenizer: Tokenizer?

    override func setUpWithError() throws {
        if Self.tokenizer == nil {
            let file = try GGUFFile(url: try TestSupport.requireWeights())
            Self.tokenizer = try Tokenizer(gguf: file)
        }
    }

    func testFixturesMatchExactly() throws {
        let tokenizer = try XCTUnwrap(Self.tokenizer)
        let fixtures = try TestSupport.tokenFixtures()
        XCTAssertEqual(fixtures.count, 40)
        for fixture in fixtures {
            let got = tokenizer.encode(fixture.input)
            XCTAssertEqual(got, fixture.tokens, "fixture \(fixture.index): \(fixture.input.prefix(60))")
        }
    }

    func testSpecialIds() throws {
        let tokenizer = try XCTUnwrap(Self.tokenizer)
        XCTAssertEqual(tokenizer.bosId, 2)
        XCTAssertEqual(tokenizer.eosId, 1)
        XCTAssertEqual(tokenizer.padId, 0)
        XCTAssertEqual(tokenizer.encode("", addSpecialTokens: true), [2, 1])
        XCTAssertEqual(tokenizer.encode("", addSpecialTokens: false), [])
        XCTAssertEqual(tokenizer.encode("hi", addSpecialTokens: false).count, 1)
    }

    func testControlTextIsNotParsedByDefault() throws {
        let tokenizer = try XCTUnwrap(Self.tokenizer)
        let plain = tokenizer.encode("<bos>hello", addSpecialTokens: false)
        XCTAssertFalse(plain.contains(2))
        let parsed = tokenizer.encode("<bos>hello", addSpecialTokens: false, parseSpecialTokens: true)
        XCTAssertEqual(parsed.first, 2)
    }

    func testRoundTrip() throws {
        let tokenizer = try XCTUnwrap(Self.tokenizer)
        let samples = [
            "hello world",
            "The quick brown fox jumps over the lazy dog.",
            "  leading and trailing spaces  ",
            "tabs\tand\nnewlines\n\n\nand more",
            "numbers 1234567890 and symbols !@#$%^&*()_+-=[]{}|;':\",./<>?",
            "Ünïcödé: naïve café résumé",
            "日本語のテキストと中文和한국어",
            "emoji 👍🏽🚀 and 🇦🇺 flags",
            "mixed العربية and हिन्दी text",
            "zero\u{200B}width and \u{FEFF}bom",
            String(repeating: "a", count: 3000),
        ]
        var rng = SeededRandom(seed: 43)
        var random: [String] = []
        for _ in 0..<20 {
            let count = Int.random(in: 1...40, using: &rng)
            var s = ""
            for _ in 0..<count {
                let scalar = UInt32.random(in: 0x20...0x2FFF, using: &rng)
                // U+2581 is SentencePiece's word-boundary marker; it decodes to a space here and in llama.cpp.
                guard scalar != 0x2581 else { continue }
                if let u = Unicode.Scalar(scalar) { s.unicodeScalars.append(u) }
            }
            random.append(s)
        }
        for s in samples + random {
            let ids = tokenizer.encode(s, addSpecialTokens: false)
            XCTAssertEqual(tokenizer.decode(ids), s, "round trip failed for \(s.debugDescription)")
            let withSpecial = tokenizer.encode(s)
            XCTAssertEqual(tokenizer.decode(withSpecial), s)
        }
    }
}

extension SeededRandom: RandomNumberGenerator {}
