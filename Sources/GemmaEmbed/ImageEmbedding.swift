import Foundation

extension EmbeddingModel {
    /// Rows the text tower adds around an image's vectors: `bos`, `<|image>`, `<image|>` and `eos`.
    static let imageFrameRows = 4

    /// Embeds one image.
    public func embed(image: ImageInput, options: ImageEmbedOptions = .init()) async throws -> Embedding {
        try await embed(images: [image], options: options)[0]
    }

    /// Embeds many images into the space the text embeddings live in. Images are decoded and
    /// resized on the CPU several at a time, then batched on the GPU; `progress` fires after every
    /// batch. Cancelling the surrounding task stops the work between batches. Throws
    /// `imageEncoderNotLoaded` unless the model was loaded with `LoadOptions.imageEncoder`, and
    /// `imageDecoding` naming the image's index (and path) when one cannot be read.
    public func embed(images: [ImageInput], options: ImageEmbedOptions = .init(),
                      progress: (@Sendable (EmbeddingProgress) -> Void)? = nil) async throws -> [Embedding] {
        var results: [Embedding] = []
        results.reserveCapacity(images.count)
        var completedTokens = 0
        try await forEachImageBatch(images, options: options) { batch in
            let vectors = try await onGPU { try $0.run(images: batch) }
            for (image, vector) in zip(batch, vectors) {
                let rows = image.tokenCount + EmbeddingModel.imageFrameRows
                results.append(Embedding(vector: finish(vector, dimensions: options.dimensions, normalise: options.normalise),
                                         tokenCount: rows, truncated: false))
                completedTokens += rows
            }
            progress?(EmbeddingProgress(completedTexts: results.count, totalTexts: images.count, completedTokens: completedTokens))
        }
        return results
    }

    /// GPU seconds of the image forward, vision and text towers, over `images` in the batches
    /// `embed(images:)` runs for `options`; decoding and resizing are not counted.
    /// Used by the CLI's benchmark mode.
    public func imageForwardSeconds(_ images: [ImageInput], options: ImageEmbedOptions = .init()) async throws -> Double {
        try await imageForwardBatchSeconds(images, options: options).reduce(0, +)
    }

    /// GPU seconds of each batch of the image forward over `images`, in the order `embed(images:)`
    /// runs them for `options`.
    public func imageForwardBatchSeconds(_ images: [ImageInput], options: ImageEmbedOptions = .init()) async throws -> [Double] {
        var seconds: [Double] = []
        try await forEachImageBatch(images, options: options) { batch in
            seconds.append(try await onGPU { try $0.timeForward(images: batch) })
        }
        return seconds
    }

    /// Prepares `inputs` and hands them to `body` in order, in the GPU batches `options` asks for.
    func forEachImageBatch(_ inputs: [ImageInput], options: ImageEmbedOptions,
                           _ body: ([PreparedImage]) async throws -> Void) async throws {
        guard let visionConfig else { throw GemmaEmbedError.imageEncoderNotLoaded }
        guard EmbedOptions.allowedDimensions.contains(options.dimensions) else {
            throw GemmaEmbedError.invalidOptions("dimensions must be one of \(EmbedOptions.allowedDimensions)")
        }
        guard options.maxImageTokens >= 1 && options.maxImageTokens <= visionConfig.nativeMaxImageTokens else {
            throw GemmaEmbedError.invalidOptions("maxImageTokens must be between 1 and \(visionConfig.nativeMaxImageTokens)")
        }
        if let b = options.batchSize, b < 1 { throw GemmaEmbedError.invalidOptions("batchSize must be at least 1") }

        let preprocessor = ImagePreprocessor(config: visionConfig)
        try await EmbeddingModel.forEachBatch(count: inputs.count, batchSize: options.batchSize, budgets: [ImageEmbedOptions.patchBudget],
                                              cost: { [$0.patchCount] }, prepare: { range in
            try await prepare(range.map { ("image \($0)", inputs[$0]) }, with: preprocessor, maxImageTokens: options.maxImageTokens)
        }, body)
    }

    /// Decodes and resizes `images` in parallel off the caller's thread, keeping their order.
    /// A decoding error names the image by its `name`.
    func prepare(_ images: [(name: String, input: ImageInput)], with preprocessor: ImagePreprocessor,
                 maxImageTokens: Int) async throws -> [PreparedImage] {
        let maxRows = info.maxTokens
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let slots = Slots<Result<PreparedImage, Error>>(count: images.count)
                DispatchQueue.concurrentPerform(iterations: images.count) { i in
                    let (name, input) = images[i]
                    slots.buffer[i] = Result {
                        do {
                            let image = try preprocessor.prepare(input, maxImageTokens: maxImageTokens)
                            let rows = image.tokenCount + EmbeddingModel.imageFrameRows
                            guard rows <= maxRows else {
                                throw GemmaEmbedError.invalidOptions("\(name) needs \(rows) rows, the text model takes \(maxRows)")
                            }
                            return image
                        } catch GemmaEmbedError.imageDecoding(let reason) {
                            if case .url(let url) = input { throw GemmaEmbedError.imageDecoding("\(name), \(url.path): \(reason)") }
                            throw GemmaEmbedError.imageDecoding("\(name): \(reason)")
                        }
                    }
                }
                continuation.resume(with: Result { try slots.buffer.map { try $0!.get() } })
            }
        }
    }
}

/// One value per index, each written by exactly one `concurrentPerform` iteration.
final class Slots<T>: @unchecked Sendable {
    let buffer: UnsafeMutableBufferPointer<T?>

    init(count: Int) {
        buffer = .allocate(capacity: count)
        buffer.initialize(repeating: nil)
    }

    deinit {
        _ = buffer.deinitialize()
        buffer.deallocate()
    }
}
