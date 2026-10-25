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
        let names = TestSupport.audioDumpNames()
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
        // Rows have a norm near 28, so 5e-3 is 2e-4 of a row.
        XCTAssertLessThan(worstDiff, 5e-3)
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

    /// Each subsampling layer against the CPU reference on the dumped clips: the first layer on the
    /// mel input, the second on the CPU's first-layer output (its own arithmetic alone) and on the
    /// GPU's, where the fp16 rounding of the first layer's output flips a few values by a half ulp.
    func testSubsamplingLayersMatchCPUReference() throws {
        let env = try XCTUnwrap(Self.env)
        let names = TestSupport.audioDumpNames()
        guard !names.isEmpty else { throw XCTSkip("no audio dumps under \(TestSupport.audioDumpsRoot)") }
        let audios = try names.map { try env.frontEnd.prepare(try TestSupport.requireAudioDump($0).samples()) }
        let batch = AudioBatch(audios: audios)
        let tower = env.audio, config = tower.config, weights = tower.weights
        let mel = config.info.melBins, halfFreq = (mel - 1) / 2 + 1, eps = config.normEpsilon
        let ws = try tower.upload(batch, context: env.forward.context)
        let w0 = env.reference.matrix(weights.subsampling[0].weight), n0 = env.reference.matrix(weights.subsampling[0].norm)
        let w1 = env.reference.matrix(weights.subsampling[1].weight), n1 = env.reference.matrix(weights.subsampling[1].norm)
        func run(_ body: (Kernels) throws -> Void) throws {
            let cb = try XCTUnwrap(env.forward.context.queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            let k = Kernels(forward: env.forward, encoder: enc, weights: tower.weightBuffer)
            enc.useResource(tower.weightBuffer, usage: .read)
            try body(k)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)
        }
        var worst = (layer0: Float(0), layer1: Float(0), layer1Cosine: Float(1))
        for (audio, placements) in zip(audios, batch.chunkPlacements()) {
            for (chunk, placement) in zip(audio.chunks, placements) {
                let (cpu0, halfT, _) = CPUAudioReference.subsample(chunk.values, time: chunk.frames, freq: mel, channels: 1, weight: w0,
                                                                    outChannels: 128, norm: n0, eps: eps)
                let (cpu1, _, _) = CPUAudioReference.subsample(cpu0, time: halfT, freq: halfFreq, channels: 128, weight: w1,
                                                               outChannels: 32, norm: n1, eps: eps)
                XCTAssertEqual(halfT, placement.halfFrames)
                func layer0(_ k: Kernels) throws {
                    try k.audioSubsample(ws.mel, weight: weights.subsampling[0].weight, norm: weights.subsampling[0].norm, out: ws.half,
                                         inTime: placement.frames, inFreq: mel, inChannels: 1, inOffset: placement.frameStart,
                                         outChannels: 128, outOffset: placement.halfStart, eps: eps)
                }
                func layer1(_ k: Kernels) throws {
                    try k.audioSubsample(ws.half, weight: weights.subsampling[1].weight, norm: weights.subsampling[1].norm, out: ws.flat,
                                         inTime: placement.halfFrames, inFreq: halfFreq, inChannels: 128, inOffset: placement.halfStart,
                                         outChannels: 32, outOffset: placement.tokenStart, eps: eps)
                }
                let halfOffset = placement.halfStart * halfFreq * 128 * 4, flatOffset = placement.tokenStart * 1024 * 4
                try run { k in try layer0(k); try layer1(k) }
                let gpu0 = ws.half.array(of: Float.self, count: cpu0.count, offset: halfOffset)
                let gpu1 = ws.flat.array(of: Float.self, count: cpu1.count, offset: flatOffset)
                worst.layer0 = max(worst.layer0, TestSupport.maxAbsDiff(gpu0, cpu0))
                worst.layer1Cosine = min(worst.layer1Cosine, TestSupport.cosine(gpu1, cpu1))
                ws.half.write(cpu0, offset: halfOffset)
                try run { k in try layer1(k) }
                worst.layer1 = max(worst.layer1, TestSupport.maxAbsDiff(ws.flat.array(of: Float.self, count: cpu1.count, offset: flatOffset), cpu1))
            }
        }
        print("subsampling vs CPU reference over \(names.count) clips: layer 0 max abs diff \(worst.layer0), layer 1 on the CPU's input \(worst.layer1), layer 1 on the GPU's input min cosine \(worst.layer1Cosine)")
        // Measured 1.2e-5, 3.2e-5 and 0.99999976 on values of unit scale (LayerNorm output).
        XCTAssertLessThan(worst.layer0, 1e-4)
        XCTAssertLessThan(worst.layer1, 2e-4)
        XCTAssertGreaterThan(worst.layer1Cosine, 0.99999)
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
