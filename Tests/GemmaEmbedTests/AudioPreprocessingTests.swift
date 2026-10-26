// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import XCTest
@testable import GemmaEmbed

final class AudioPreprocessingTests: XCTestCase {
    func testFrameAndTokenCounts() {
        // llama-server's prompt_tokens for the fixtures, minus the four frame tokens.
        XCTAssertEqual(AudioFrontEnd.frameCount(samples: 4800), 29)
        XCTAssertEqual(AudioFrontEnd.tokenCount(samples: 4800), 8)
        XCTAssertEqual(AudioFrontEnd.tokenCount(samples: 16000), 25)
        XCTAssertEqual(AudioFrontEnd.tokenCount(samples: 480000), 750)
        XCTAssertEqual(AudioFrontEnd.tokenCount(samples: 560000), 875)
        XCTAssertEqual(AudioFrontEnd.frameCount(samples: 160), 1)
        XCTAssertEqual(AudioFrontEnd.frameCount(samples: AudioFrontEnd.minimumSamples), 1)
        XCTAssertEqual(AudioFrontEnd.frameCount(samples: 1), 0)
    }

    func testTooShortClipIsRefused() {
        let frontEnd = AudioFrontEnd(melBins: 128)
        XCTAssertThrowsError(try frontEnd.prepare([Float](repeating: 0.1, count: 1)))
        XCTAssertThrowsError(try frontEnd.prepare([]))
        XCTAssertNoThrow(try frontEnd.prepare([Float](repeating: 0.1, count: AudioFrontEnd.minimumSamples)))
    }

    func testChunking() throws {
        let frontEnd = AudioFrontEnd(melBins: 128)
        let prepared = try frontEnd.prepare([Float](repeating: 0.01, count: 480000 + 16000))
        XCTAssertEqual(prepared.chunks.map(\.frames), [2999, 99])
        XCTAssertEqual(prepared.chunks.map(\.tokenCount), [750, 25])
        XCTAssertEqual(prepared.tokenCount, 775)
    }

    /// The log-mel frames of every dumped clip against the input node of llama.cpp's graph,
    /// chunk by chunk: the samples are miniaudio's own, so this isolates the front end.
    func testLogMelMatchesLlamaCpp() throws {
        let names = TestSupport.audioDumpNames()
        guard !names.isEmpty else { throw XCTSkip("no audio dumps under \(TestSupport.audioDumpsRoot)") }
        let frontEnd = AudioFrontEnd(melBins: 128)
        var worst: Float = 0
        var checked = 0
        for name in names {
            let dump = try TestSupport.requireAudioDump(name)
            let prepared = try frontEnd.prepare(try dump.samples())
            XCTAssertEqual(prepared.chunks.count, dump.chunks.count, name)
            for (chunk, mel) in zip(dump.chunks, prepared.chunks) {
                let node = try XCTUnwrap(dump.node(chunk.firstNode + 1), "\(name) has no mel node")
                XCTAssertEqual(node.op, "CONT", name)
                XCTAssertEqual(node.shape, [128, mel.frames, 1, 1], "\(name) mel shape")
                XCTAssertEqual(mel.tokenCount, chunk.tokenCount, name)
                guard node.file != nil else { continue }
                let want = try dump.values(chunk.firstNode + 1)
                // fp32 FFT roundoff shows under the log where a mel sum is small: a bin at the 0.001
                // floor (log -6.9) moves by parts in a thousand, a bin with energy by 3e-5.
                var worstHigh: Float = 0, worstMid: Float = 0, worstLow: Float = 0
                for i in 0..<want.count {
                    let d = abs(mel.values[i] - want[i])
                    if want[i] > -2 { worstHigh = max(worstHigh, d) } else if want[i] > -6 { worstMid = max(worstMid, d) } else { worstLow = max(worstLow, d) }
                }
                XCTAssertFalse(worstLow.isNaN || worstMid.isNaN || worstHigh.isNaN, name)
                XCTAssertLessThan(worstHigh, 1e-4, "\(name) log-mel vs llama.cpp, bins above -2")
                XCTAssertLessThan(worstMid, 2e-3, "\(name) log-mel vs llama.cpp, bins between -6 and -2")
                XCTAssertLessThan(worstLow, 5e-3, "\(name) log-mel vs llama.cpp, bins below -6")
                print("\(name): log-mel max abs diff \(worstHigh) above -2, \(worstMid) between -6 and -2, \(worstLow) below -6")
                worst = max(worst, worstLow)
                checked += 1
            }
        }
        print("log-mel vs llama.cpp over \(checked) chunks of \(names.count) clips: max abs diff \(worst)")
        XCTAssertGreaterThan(checked, 0)
    }
}
