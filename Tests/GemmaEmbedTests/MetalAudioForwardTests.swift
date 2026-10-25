import Foundation
import XCTest
@testable import GemmaEmbed

final class MetalAudioForwardTests: XCTestCase {
    final class Env {
        let forward: MetalForward
        let audio: AudioTower
        let frontEnd: AudioFrontEnd
        let reference: CPUAudioReference

        init(forward: MetalForward, audio: AudioTower, frontEnd: AudioFrontEnd, reference: CPUAudioReference) {
            self.forward = forward
            self.audio = audio
            self.frontEnd = frontEnd
            self.reference = reference
        }
    }

    nonisolated(unsafe) static var env: Env?

    override func setUpWithError() throws {
        if Self.env == nil {
            let file = try GGUFFile(url: try TestSupport.requireWeights())
            let encoderFile = try GGUFFile(url: try TestSupport.requireImageEncoder())
            let context = try MetalContext()
            let config = try ModelConfig(gguf: file, deviceName: context.device.name, fileSHA256: nil)
            let audioConfig = try AudioConfig(gguf: encoderFile, fileSHA256: nil)
            let audioWeights = try AudioWeights(gguf: encoderFile, config: audioConfig)
            let audio = try AudioTower(context: context, config: audioConfig, weights: audioWeights,
                                       markers: try SpanMarkers.audio(tokenizer: Tokenizer(gguf: file)))
            let forward = try MetalForward(context: context, config: config, weights: try ModelWeights(gguf: file, config: config), audio: audio)
            Self.env = Env(forward: forward, audio: audio, frontEnd: AudioFrontEnd(melBins: audioConfig.info.melBins),
                           reference: CPUAudioReference(config: audioConfig, weights: audioWeights))
        }
    }

    static func normalised(_ v: [Float]) -> [Float] {
        let n = v.reduce(0) { $0 + $1 * $1 }.squareRoot()
        return v.map { $0 / n }
    }

    func prepare(_ url: URL) throws -> PreparedAudio {
        try XCTUnwrap(Self.env).frontEnd.prepare(try AudioDecoder.decode(.url(url)))
    }

    /// The audio tower alone over the dumped clips packed in one batch, against the CPU reference
    /// row by row and llama.cpp's own output rows, both fed miniaudio's samples.
    func testAudioTowerMatchesCPUReferenceAndLlamaCpp() throws {
        let env = try XCTUnwrap(Self.env)
        let names = TestSupport.audioDumpNames().filter { $0 != "long-35s" }
        guard !names.isEmpty else { throw XCTSkip("no audio dumps under \(TestSupport.audioDumpsRoot)") }
        let dumps = try names.map { try TestSupport.requireAudioDump($0) }
        let audios = try dumps.map { try env.frontEnd.prepare(try $0.samples()) }
        let got = try env.forward.audioVectors(audios)
        let width = env.audio.config.info.projectionDim
        var worstCPU: Float = 1, worstDiff: Float = 0, worstLlama: Float = 1
        for ((dump, audio), rows) in zip(zip(dumps, audios), got) {
            XCTAssertEqual(rows.count, audio.tokenCount * width, dump.name)
            let cpu = audio.chunks.flatMap { env.reference.forward($0) }
            let llama = try (0..<dump.chunks.count).flatMap { try dump.outputRows(chunk: $0) }
            XCTAssertEqual(cpu.count, rows.count, dump.name)
            XCTAssertEqual(llama.count, rows.count, dump.name)
            for r in 0..<audio.tokenCount {
                let mine = Array(rows[r * width..<(r + 1) * width])
                let c = TestSupport.cosine(mine, Array(cpu[r * width..<(r + 1) * width]))
                let l = TestSupport.cosine(mine, Array(llama[r * width..<(r + 1) * width]))
                worstCPU = min(worstCPU, c)
                worstLlama = min(worstLlama, l)
                worstDiff = max(worstDiff, TestSupport.maxAbsDiff(mine, Array(cpu[r * width..<(r + 1) * width])))
            }
        }
        print("audio tower vs CPU reference over \(names.count) clips: min row cosine \(worstCPU), max abs diff \(worstDiff); vs libmtmd rows min cosine \(worstLlama)")
        XCTAssertGreaterThan(worstCPU, 0.99999)
        XCTAssertLessThan(worstDiff, 2e-3)
        XCTAssertGreaterThan(worstLlama, 0.999)
    }

    /// Every fixture clip through decoding, the front end, the audio tower and the text tower,
    /// against llama-server's vector for the same file.
    func testAllFixtureClipsMatchLlamaServer() throws {
        let env = try XCTUnwrap(Self.env)
        _ = try TestSupport.requireWeights()
        let fixtures = try TestSupport.audioFixtures()
        let reference = try TestSupport.audioEmbeddingFixtures()
        var worst = (file: "", cosine: Float(1), diff: Float(0))
        var byKind: [String: Float] = [:]
        for fixture in fixtures {
            let want = try XCTUnwrap(reference[fixture.file], "\(fixture.file) has no llama-server vector")
            let audio = try prepare(fixture.url)
            XCTAssertEqual(audio.tokenCount + 4, want.promptTokens, "\(fixture.file) token count")
            let got = Self.normalised(try env.forward.run(audio: [audio])[0])
            let cosine = TestSupport.cosine(got, want.embedding)
            let diff = TestSupport.maxAbsDiff(got, want.embedding)
            byKind[fixture.kind] = min(byKind[fixture.kind] ?? 1, cosine)
            if cosine < worst.cosine { worst = (fixture.file, cosine, diff) }
            XCTAssertGreaterThan(cosine, 0.9999, fixture.file)
        }
        print("audio embeddings vs llama-server over \(fixtures.count) clips: min cosine \(worst.cosine) (\(worst.file), max abs diff \(worst.diff)); by kind \(byKind)")
    }

    /// A clip embeds to the same bits alone and inside a batch, first or last, beside clips of other lengths.
    func testBatchIndependence() throws {
        let env = try XCTUnwrap(Self.env)
        let fixtures = try TestSupport.audioFixtures()
        let picks = ["signals/08-short-tone.wav", "signals/11-mixture.wav", "tones/00-sine-440.wav", "long/01-35s.wav", "formats/04-stereo.wav"]
        let audios = try picks.map { pick in try prepare(try XCTUnwrap(fixtures.first { $0.file == pick }).url) }
        let alone = try audios.map { try env.forward.run(audio: [$0])[0] }
        let together = try env.forward.run(audio: audios)
        let reversed = try env.forward.run(audio: Array(audios.reversed()))
        for i in 0..<audios.count {
            XCTAssertEqual(alone[i], together[i], "\(picks[i]) in a batch of \(audios.count)")
            XCTAssertEqual(alone[i], reversed[audios.count - 1 - i], "\(picks[i]) in the reversed batch")
        }
    }
}
