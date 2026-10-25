import Foundation
import XCTest
@testable import GemmaEmbed

final class AudioDecoderTests: XCTestCase {
    // MARK: WAV bytes built in the test

    enum SampleFormat {
        case u8, s16, s24, s32, f32, f64

        var bits: Int {
            switch self {
            case .u8: return 8
            case .s16: return 16
            case .s24: return 24
            case .s32: return 32
            case .f32: return 32
            case .f64: return 64
            }
        }

        var tag: UInt16 {
            switch self {
            case .f32, .f64: return WAVDecoder.formatIEEEFloat
            default: return WAVDecoder.formatPCM
            }
        }
    }

    static func le<T: FixedWidthInteger>(_ value: T) -> [UInt8] {
        withUnsafeBytes(of: value.littleEndian) { Array($0) }
    }

    static func chunk(_ id: String, _ payload: [UInt8]) -> [UInt8] {
        Array(id.utf8) + le(UInt32(payload.count)) + payload + (payload.count % 2 == 1 ? [0] : [])
    }

    /// A RIFF/WAVE file around `samples`, which holds the raw little-endian sample bytes.
    static func wav(_ format: SampleFormat, channels: Int, rate: Int, samples: [UInt8], extensible: Bool = false,
                    chunksBeforeData: [[UInt8]] = [], declaredDataSize: Int? = nil, fmtAfterData: Bool = false) -> Data {
        let blockAlign = channels * format.bits / 8
        var fmt = le(extensible ? WAVDecoder.formatExtensible : format.tag) + le(UInt16(channels)) + le(UInt32(rate))
            + le(UInt32(rate * blockAlign)) + le(UInt16(blockAlign)) + le(UInt16(format.bits))
        if extensible {
            fmt += le(UInt16(22)) + le(UInt16(format.bits)) + le(UInt32((1 << channels) - 1)) + le(format.tag)
                + [0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71]
        }
        var data = Array("data".utf8) + le(UInt32(declaredDataSize ?? samples.count)) + samples
        if samples.count % 2 == 1, declaredDataSize == nil { data.append(0) }
        var body = fmtAfterData ? [] : chunk("fmt ", fmt)
        for c in chunksBeforeData { body += c }
        body += data
        if fmtAfterData { body += chunk("fmt ", fmt) }
        return Data(Array("RIFF".utf8) + le(UInt32(4 + body.count)) + Array("WAVE".utf8) + body)
    }

    static func bytes<T: FixedWidthInteger>(_ values: [T]) -> [UInt8] { values.flatMap { le($0) } }
    static func bytes24(_ values: [Int32]) -> [UInt8] { values.flatMap { Array(le($0)[0..<3]) } }
    static func floatBytes(_ values: [Float]) -> [UInt8] { values.flatMap { le($0.bitPattern) } }
    static func doubleBytes(_ values: [Double]) -> [UInt8] { values.flatMap { le($0.bitPattern) } }

    func testPCM8BitIsUnsignedWithDrWavScaling() throws {
        let got = try WAVDecoder.decode(Self.wav(.u8, channels: 1, rate: 16000, samples: [0, 1, 127, 128, 255]))
        XCTAssertEqual(got.channels, 1)
        XCTAssertEqual(got.sampleRate, 16000)
        XCTAssertEqual(got.samples, [-1.0, -0.9921568632125854, -0.0039215087890625, 0.003921627998352051, 1.0])
    }

    func testPCM16Bit() throws {
        let got = try WAVDecoder.decode(Self.wav(.s16, channels: 1, rate: 44100, samples: Self.bytes([Int16(32767), -32768, 1, -1, 12345])))
        XCTAssertEqual(got.sampleRate, 44100)
        XCTAssertEqual(got.samples, [0.999969482421875, -1.0, 3.0517578125e-05, -3.0517578125e-05, 0.376739501953125])
    }

    func testPCM24Bit() throws {
        let got = try WAVDecoder.decode(Self.wav(.s24, channels: 1, rate: 48000, samples: Self.bytes24([8388607, -8388608, 1, -1, -1234567])))
        XCTAssertEqual(got.samples, [0.9999998807907104, -1.0, 1.1920928955078125e-07, -1.1920928955078125e-07, -0.14717185497283936])
    }

    func testPCM32Bit() throws {
        let got = try WAVDecoder.decode(Self.wav(.s32, channels: 1, rate: 8000, samples: Self.bytes([Int32(2147483647), -2147483648, 1, -1, 123456789])))
        XCTAssertEqual(got.samples, [1.0, -1.0, 4.656612873077393e-10, -4.656612873077393e-10, 0.057489048689603806])
    }

    func testIEEEFloat32IsCopiedBitForBit() throws {
        let values: [Float] = [0.1, -0.333333343, 1.5, -2.0, Float(bitPattern: 0x3F800001), Float(bitPattern: 0x00000001)]
        let got = try WAVDecoder.decode(Self.wav(.f32, channels: 1, rate: 16000, samples: Self.floatBytes(values)))
        XCTAssertEqual(got.samples.map(\.bitPattern), values.map(\.bitPattern))
    }

    func testIEEEFloat64RoundsToFloat() throws {
        let values: [Double] = [0.1, -0.3333333333333333, 1.0 / 3.0, 0.7, -1e-50]
        let got = try WAVDecoder.decode(Self.wav(.f64, channels: 2, rate: 22050, samples: Self.doubleBytes(values + [0.0])))
        XCTAssertEqual(got.channels, 2)
        XCTAssertEqual(got.samples, (values + [0.0]).map { Float($0) })
    }

    func testExtensibleHeaderUsesTheSubFormat() throws {
        let s24 = try WAVDecoder.decode(Self.wav(.s24, channels: 6, rate: 44100, samples: Self.bytes24(Array(repeating: 4194304, count: 12)), extensible: true))
        XCTAssertEqual(s24.channels, 6)
        XCTAssertEqual(s24.sampleRate, 44100)
        XCTAssertEqual(s24.samples, Array(repeating: 0.5, count: 12))
        let f32 = try WAVDecoder.decode(Self.wav(.f32, channels: 1, rate: 16000, samples: Self.floatBytes([0.25, -0.75]), extensible: true))
        XCTAssertEqual(f32.samples, [0.25, -0.75])
    }

    func testUnknownAndOddSizedChunksAreSkipped() throws {
        let list = Self.chunk("LIST", Array("INFOISFT".utf8) + Self.le(UInt32(7)) + Array("gemma\0\0".utf8))
        let odd = Self.chunk("junk", [1, 2, 3])
        let fact = Self.chunk("fact", Self.le(UInt32(3)))
        let bext = Self.chunk("bext", [UInt8](repeating: 0xAB, count: 5))
        let samples = Self.bytes([Int16(100), 200, -300])
        let got = try WAVDecoder.decode(Self.wav(.s16, channels: 1, rate: 16000, samples: samples, chunksBeforeData: [list, odd, fact, bext]))
        XCTAssertEqual(got.samples, [100, 200, -300].map { Float($0) * 0.000030517578125 })
    }

    /// dr_wav clamps a data chunk that claims more bytes than the file holds, then drops a trailing partial frame.
    func testOverrunningDataChunkIsClampedToTheFile() throws {
        let samples = Self.bytes([Int16(1), 2, 3, 4, 5]) + [0x07]
        let got = try WAVDecoder.decode(Self.wav(.s16, channels: 2, rate: 16000, samples: samples, declaredDataSize: 1_000_000))
        XCTAssertEqual(got.frameCount, 2)
        XCTAssertEqual(got.samples, [1, 2, 3, 4].map { Float($0) * 0.000030517578125 })
    }

    func testMalformedFilesAreRejectedWithAReason() {
        func reason(_ data: Data) -> String? {
            do {
                _ = try WAVDecoder.decode(data)
                return nil
            } catch GemmaEmbedError.audioDecoding(let reason) {
                return reason
            } catch {
                return "unexpected \(error)"
            }
        }
        XCTAssertEqual(reason(Data("RIFX\0\0\0\0WAVE".utf8)), "WAV: not a RIFF file")
        XCTAssertEqual(reason(Data("RIFF\0\0\0\0AVI ".utf8)), "WAV: not a WAVE file")
        XCTAssertEqual(reason(Self.wav(.s16, channels: 1, rate: 16000, samples: Self.bytes([Int16(1)]), fmtAfterData: true)), "WAV: no fmt chunk before the data chunk")
        let header: [UInt8] = Array("RIFF".utf8) + Self.le(UInt32(4)) + Array("WAVE".utf8)
        XCTAssertEqual(reason(Data(header + Self.chunk("fmt ", [UInt8](repeating: 0, count: 16)))), "WAV: no data chunk")
        var adpcm = [UInt8](Self.wav(.s16, channels: 1, rate: 16000, samples: Self.bytes([Int16(1), 2])))
        adpcm[20] = 0x02
        XCTAssertEqual(reason(Data(adpcm)), "WAV: format tag 0x2 is not PCM or IEEE float")
        var zeroChannels = [UInt8](Self.wav(.s16, channels: 1, rate: 16000, samples: Self.bytes([Int16(1), 2])))
        zeroChannels[22] = 0
        XCTAssertEqual(reason(Data(zeroChannels)), "WAV: channel count 0 is out of range")
        let shortFmt: [UInt8] = Array("fmt ".utf8) + Self.le(UInt32(16)) + [1, 0]
        XCTAssertEqual(reason(Data(header + shortFmt)), "WAV: fmt chunk is truncated")
    }

    // MARK: MiniaudioConverter on hand-built input

    static func noise(_ count: Int, seed: UInt32 = 1) -> [Float] {
        var state = seed
        return (0..<count).map { _ in
            state = 1664525 &* state &+ 1013904223
            return Float(Int32(bitPattern: state)) / 2147483648.0
        }
    }

    func testMono16kPassesThroughBitIdentical() {
        let samples = Self.noise(5003)
        let out = MiniaudioConverter.toMono16k(DecodedAudio(samples: samples, channels: 1, sampleRate: 16000))
        XCTAssertEqual(out.map(\.bitPattern), samples.map(\.bitPattern))
    }

    func testIdenticalStereoChannelsDownmixToThatChannel() {
        let mono = Self.noise(4001)
        let stereo = mono.flatMap { [$0, $0] }
        let out = MiniaudioConverter.toMono16k(DecodedAudio(samples: stereo, channels: 2, sampleRate: 16000))
        XCTAssertEqual(out.map(\.bitPattern), mono.map(\.bitPattern))
    }

    func testDownmixAveragesChannelsInOrder() {
        let out = MiniaudioConverter.downmixToMono([0.5, -0.25, 1.0, 0.75, 0.75, 0.75], channels: 3)
        XCTAssertEqual(out, [(0.5 - 0.25 + 1.0) / 3, (0.75 + 0.75 + 0.75) / 3])
    }

    /// `ma_calculate_frame_count_after_resampling` for the check files, which all have odd frame counts.
    func testResampledLengthRule() {
        func count(_ rate: UInt32, _ frames: UInt64) -> UInt64 {
            MiniaudioConverter.frameCountAfterResampling(sampleRateOut: 16000, sampleRateIn: rate, frameCountIn: frames)
        }
        XCTAssertEqual(count(16000, 5003), 5003)
        XCTAssertEqual(count(44100, 13231), 4800)
        XCTAssertEqual(count(8000, 2401), 4802)
        XCTAssertEqual(count(48000, 14401), 4801)
        XCTAssertEqual(count(11025, 3308), 4800)
        XCTAssertEqual(count(22050, 6616), 4800)
        XCTAssertEqual(count(96000, 28801), 4801)
        XCTAssertEqual(count(32000, 9601), 4801)
        XCTAssertEqual(count(44100, 0), 0)
    }

    func testResamplerProducesTheDeclaredLengthAndNoNaN() {
        for rate in [8000, 11025, 22050, 32000, 44100, 48000, 96000] {
            let frames = rate * 3 / 10 + 1
            let out = MiniaudioConverter.toMono16k(DecodedAudio(samples: Self.noise(frames), channels: 1, sampleRate: rate))
            let want = MiniaudioConverter.frameCountAfterResampling(sampleRateOut: 16000, sampleRateIn: UInt32(rate), frameCountIn: UInt64(frames))
            XCTAssertEqual(UInt64(out.count), want, "\(rate) Hz")
            XCTAssertFalse(out.contains { $0.isNaN }, "\(rate) Hz")
            XCTAssertLessThan(out.map(abs).max() ?? 0, 2, "\(rate) Hz")
        }
    }

    func testEveryInputFormDecodesAlike() throws {
        let data = Self.wav(.s16, channels: 2, rate: 22050, samples: Self.bytes(Self.noise(3002).map { Int16($0 * 32767) }))
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("gemma-embed-test-\(UUID().uuidString).wav")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let fromURL = try AudioDecoder.decode(.url(url))
        XCTAssertEqual(try AudioDecoder.decode(.data(data)), fromURL)
        let decoded = try WAVDecoder.decode(data)
        XCTAssertEqual(try AudioDecoder.decode(.samples(decoded.samples, sampleRate: decoded.sampleRate, channels: decoded.channels)), fromURL)
        XCTAssertEqual(fromURL.count, 1089)
        XCTAssertThrowsError(try AudioDecoder.decode(.data(Data("not audio at all".utf8)))) { error in
            guard case GemmaEmbedError.audioDecoding = error else { return XCTFail("unexpected error \(error)") }
        }
        XCTAssertThrowsError(try AudioDecoder.decode(.url(URL(fileURLWithPath: "/nonexistent/clip.wav")))) { error in
            XCTAssertEqual("\(error)", "cannot decode audio: /nonexistent/clip.wav is not readable")
        }
        XCTAssertThrowsError(try AudioDecoder.decode(.samples([0.1, 0.2, 0.3], sampleRate: 16000, channels: 2)))
    }

    // MARK: Parity with miniaudio, from models/audio/dumps

    static let defaultDumpsPath = TestSupport.modelsDirectory.appendingPathComponent("audio/dumps").path

    struct DecodeDump {
        let name: String
        let audioURL: URL
        let samples: [Float]
        /// Set when the samples are not miniaudio's decode of `audioURL` itself but of the file
        /// it was encoded from (llama.cpp cannot read M4A).
        let oracle: String?
    }

    /// Every dump directory with `samples.f32` whose audio file still exists; skips when there are none.
    static func decodeDumps() throws -> [DecodeDump] {
        let root = ProcessInfo.processInfo.environment["GEMMA_EMBED_AUDIO_DUMPS"] ?? defaultDumpsPath
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: root) else {
            throw XCTSkip("no audio dumps at \(root); run scripts/make_decode_checks.py or set GEMMA_EMBED_AUDIO_DUMPS")
        }
        var dumps: [DecodeDump] = []
        for name in names.sorted() {
            let directory = URL(fileURLWithPath: root).appendingPathComponent(name)
            let samplesURL = directory.appendingPathComponent("samples.f32")
            let metaURL = directory.appendingPathComponent("meta.json")
            guard FileManager.default.isReadableFile(atPath: samplesURL.path), FileManager.default.isReadableFile(atPath: metaURL.path) else { continue }
            let meta = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: metaURL)) as? [String: Any])
            let audioPath = try XCTUnwrap(meta["audio"] as? String)
            guard FileManager.default.isReadableFile(atPath: audioPath) else { continue }
            let samples = try Data(contentsOf: samplesURL).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            XCTAssertEqual(samples.count, meta["n_samples"] as? Int, "\(name): samples.f32 disagrees with meta.json")
            dumps.append(DecodeDump(name: name, audioURL: URL(fileURLWithPath: audioPath), samples: samples, oracle: meta["oracle"] as? String))
        }
        if dumps.isEmpty {
            throw XCTSkip("no audio dumps with samples.f32 and a readable audio file under \(root)")
        }
        return dumps
    }

    /// WAV decodes are bit-identical to miniaudio's for every bit depth, channel count and rate.
    func testWAVDecodeMatchesMiniaudio() throws {
        let dumps = try Self.decodeDumps().filter { $0.audioURL.pathExtension.lowercased() == "wav" }
        if dumps.isEmpty { throw XCTSkip("no WAV dumps") }
        for dump in dumps {
            let got = try AudioDecoder.decode(.url(dump.audioURL))
            XCTAssertEqual(got.count, dump.samples.count, "\(dump.name): length")
            guard got.count == dump.samples.count else { continue }
            let diff = TestSupport.maxAbsDiff(got, dump.samples)
            let differing = zip(got, dump.samples).filter { $0.bitPattern != $1.bitPattern }.count
            print("decode \(dump.name): \(got.count) samples, max abs diff \(diff), differing samples \(differing)")
            XCTAssertEqual(differing, 0, "\(dump.name): max abs diff \(diff)")
        }
    }

    /// MP3 goes through dr_mp3 in llama.cpp and AudioToolbox here, so the bound is loose; M4A has
    /// no llama.cpp decode at all, so it is measured against the miniaudio output of the WAV it
    /// was encoded from, which also carries the AAC codec's own loss and delay. `.data` must give
    /// the same samples as `.url`, since AudioToolbox has no file extension to go by there.
    func testCompressedDecodeIsCloseToMiniaudio() throws {
        let dumps = try Self.decodeDumps().filter { $0.audioURL.pathExtension.lowercased() != "wav" }
        if dumps.isEmpty { throw XCTSkip("no MP3 or M4A dumps") }
        for dump in dumps {
            let got = try AudioDecoder.decode(.url(dump.audioURL))
            XCTAssertEqual(try AudioDecoder.decode(.data(Data(contentsOf: dump.audioURL))), got, "\(dump.name): .data differs from .url")
            let (lag, diff) = Self.bestAlignment(got, dump.samples)
            print("decode \(dump.name): \(got.count) samples vs \(dump.samples.count)" + (dump.oracle.map { " (oracle: \($0))" } ?? "")
                  + ", best lag \(lag), max abs diff at that lag \(diff)")
            XCTAssertLessThan(abs(got.count - dump.samples.count), 2400, "\(dump.name): length")
            if dump.oracle == nil {
                XCTAssertLessThanOrEqual(abs(lag), 1200, dump.name)
                XCTAssertLessThan(diff, 1e-3, dump.name)
            } else {
                XCTAssertLessThan(diff, 0.25, dump.name)
            }
        }
    }

    /// The shift of `got` against `want` (within a second) that minimises the max abs diff over the overlap.
    static func bestAlignment(_ got: [Float], _ want: [Float]) -> (lag: Int, diff: Float) {
        var best = (lag: 0, diff: Float.infinity)
        for lag in -2400...2400 {
            var m: Float = 0
            var count = 0
            for i in stride(from: 0, to: want.count, by: 1) {
                let j = i + lag
                guard j >= 0, j < got.count else { continue }
                m = max(m, abs(got[j] - want[i]))
                count += 1
                if m >= best.diff { break }
            }
            if count > want.count / 2, m < best.diff { best = (lag, m) }
        }
        return best
    }
}
