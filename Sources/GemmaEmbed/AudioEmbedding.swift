import Foundation

extension EmbeddingModel {
    /// Rows the text tower adds around a clip's rows: `bos`, `<|audio>`, `<audio|>` and `eos`.
    static let audioFrameRows = 4

    /// Embeds one clip.
    public func embed(audio: AudioInput, options: AudioEmbedOptions = .init()) async throws -> Embedding {
        try await embed(audios: [audio], options: options)[0]
    }

    /// Embeds many clips into the space the text embeddings live in. Clips are decoded and turned
    /// into log-mel frames on the CPU several at a time, then batched on the GPU; `progress` fires
    /// after every batch. Cancelling the surrounding task stops the work between batches. Throws
    /// `audioEncoderNotLoaded` unless the model was loaded with `LoadOptions.audioEncoder`,
    /// `audioDecoding` naming the clip's index (and path) when one cannot be read, and
    /// `invalidOptions` when a clip needs more rows than `maxTokens`.
    public func embed(audios: [AudioInput], options: AudioEmbedOptions = .init(),
                      progress: (@Sendable (EmbeddingProgress) -> Void)? = nil) async throws -> [Embedding] {
        var results: [Embedding] = []
        results.reserveCapacity(audios.count)
        var completedTokens = 0
        try await forEachAudioBatch(audios, options: options) { batch in
            let vectors = try await onGPU { try $0.run(audio: batch) }
            for (audio, vector) in zip(batch, vectors) {
                let rows = audio.tokenCount + EmbeddingModel.audioFrameRows
                results.append(Embedding(vector: finish(vector, dimensions: options.dimensions, normalise: options.normalise),
                                         tokenCount: rows, truncated: false))
                completedTokens += rows
            }
            progress?(EmbeddingProgress(completedTexts: results.count, totalTexts: audios.count, completedTokens: completedTokens))
        }
        return results
    }

    /// GPU seconds of the audio forward, audio and text towers, over `audios` in the batches
    /// `embed(audios:)` runs for `options`; decoding and the log-mel front end are not counted.
    public func audioForwardSeconds(_ audios: [AudioInput], options: AudioEmbedOptions = .init()) async throws -> Double {
        try await audioForwardBatchSeconds(audios, options: options).reduce(0, +)
    }

    /// GPU seconds of each batch of the audio forward over `audios`, in the order `embed(audios:)` runs them.
    public func audioForwardBatchSeconds(_ audios: [AudioInput], options: AudioEmbedOptions = .init()) async throws -> [Double] {
        var seconds: [Double] = []
        try await forEachAudioBatch(audios, options: options) { batch in
            seconds.append(try await onGPU { try $0.timeForward(audio: batch) })
        }
        return seconds
    }

    /// Prepares `inputs` and hands them to `body` in order, in the GPU batches `options` asks for.
    func forEachAudioBatch(_ inputs: [AudioInput], options: AudioEmbedOptions,
                           _ body: ([PreparedAudio]) async throws -> Void) async throws {
        guard let audioConfig else { throw GemmaEmbedError.audioEncoderNotLoaded }
        guard EmbedOptions.allowedDimensions.contains(options.dimensions) else {
            throw GemmaEmbedError.invalidOptions("dimensions must be one of \(EmbedOptions.allowedDimensions)")
        }
        let maxRows = options.maxTokens ?? info.maxTokens
        guard maxRows > EmbeddingModel.audioFrameRows && maxRows <= info.maxTokens else {
            throw GemmaEmbedError.invalidOptions("maxTokens must be between \(EmbeddingModel.audioFrameRows + 1) and \(info.maxTokens)")
        }
        if let b = options.batchSize, b < 1 { throw GemmaEmbedError.invalidOptions("batchSize must be at least 1") }

        let frontEnd = AudioFrontEnd(melBins: audioConfig.info.melBins)
        try await EmbeddingModel.forEachBatch(count: inputs.count, batchSize: options.batchSize, budgets: [AudioEmbedOptions.frameBudget],
                                              cost: { [$0.frameCount] }, prepare: { range in
            try await prepare(range.map { ("audio \($0)", inputs[$0]) }, with: frontEnd, maxRows: maxRows)
        }, body)
    }

    /// Decodes and analyses `audios` in parallel off the caller's thread, keeping their order.
    /// A decoding error names the clip by its `name`.
    func prepare(_ audios: [(name: String, input: AudioInput)], with frontEnd: AudioFrontEnd, maxRows: Int) async throws -> [PreparedAudio] {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let slots = Slots<Result<PreparedAudio, Error>>(count: audios.count)
                DispatchQueue.concurrentPerform(iterations: audios.count) { i in
                    let (name, input) = audios[i]
                    slots.buffer[i] = Result {
                        do {
                            let samples = try AudioDecoder.decode(input)
                            let audio = try frontEnd.prepare(samples)
                            let rows = audio.tokenCount + EmbeddingModel.audioFrameRows
                            guard rows <= maxRows else {
                                let seconds = Double(samples.count) / Double(AudioFrontEnd.sampleRate)
                                throw GemmaEmbedError.invalidOptions(String(format: "%@ is %.1f s and needs %d rows; maxTokens allows %d (about %.1f s)",
                                                                            name, seconds, rows, maxRows, Double(maxRows - EmbeddingModel.audioFrameRows) * 0.04))
                            }
                            return audio
                        } catch GemmaEmbedError.audioDecoding(let reason) {
                            if case .url(let url) = input, !reason.hasPrefix(url.path) { throw GemmaEmbedError.audioDecoding("\(name), \(url.path): \(reason)") }
                            throw GemmaEmbedError.audioDecoding("\(name): \(reason)")
                        }
                    }
                }
                continuation.resume(with: Result { try slots.buffer.map { try $0!.get() } })
            }
        }
    }
}
