import Foundation

/// One input to the text tower: text and images in the order they appear. EmbeddingGemma 2
/// reads an image where its `<|image|>` placeholder sits in the text; the parts carry that order.
public struct MultimodalInput: Sendable {
    public enum Part: Sendable {
        case text(String)
        case image(ImageInput)
    }

    public var parts: [Part]

    public init(_ parts: [Part]) {
        self.parts = parts
    }

    /// The text followed by the images.
    public init(text: String, images: [ImageInput]) {
        self.init([.text(text)] + images.map { .image($0) })
    }
}

/// The text-tower sequence of one `MultimodalInput`, as llama.cpp's mtmd lays it out for gemma4v:
/// `[bos] tok(prefix + text) [<|image>] v... [<image|>] tok(text) ... [eos]`.
struct MultimodalSequence {
    /// Token ids, and `-(r + 1)` for row `r` of the input's image rows, counted across its images in order.
    let ids: [Int32]
    let truncated: Bool

    /// Lays out `parts` with `imageRows[k]` rows for the k-th image. The prefix joins the first
    /// text, or leads on its own before an image. Text is cut from the end until the sequence fits
    /// `maxTokens`; nil when the images and the frame alone do not fit.
    init?(prefix: String, parts: [MultimodalInput.Part], imageRows: [Int], maxTokens: Int, markers: SpanMarkers,
          encode: (String) -> [Int32]) {
        var parts = parts
        if case .text(let first)? = parts.first {
            parts[0] = .text(prefix + first)
        } else {
            parts.insert(.text(prefix), at: 0)
        }
        var texts = parts.map { part -> [Int32]? in
            if case .text(let text) = part { return encode(text) }
            return nil
        }
        precondition(texts.filter { $0 == nil }.count == imageRows.count)
        let fixed = 2 + imageRows.reduce(0) { $0 + $1 + 2 }
        guard fixed <= maxTokens else { return nil }
        var excess = fixed + texts.reduce(0) { $0 + ($1?.count ?? 0) } - maxTokens
        truncated = excess > 0
        for i in texts.indices.reversed() where excess > 0 {
            guard let text = texts[i] else { continue }
            let cut = min(excess, text.count)
            texts[i] = Array(text.dropLast(cut))
            excess -= cut
        }
        var ids = [markers.bos]
        var images = imageRows.makeIterator()
        var row: Int32 = 0
        for text in texts {
            if let text {
                ids += text
            } else {
                let count = Int32(images.next()!)
                ids.append(markers.begin)
                ids += (row..<(row + count)).map { -$0 - 1 }
                ids.append(markers.end)
                row += count
            }
        }
        ids.append(markers.eos)
        self.ids = ids
    }
}

extension EmbeddingModel {
    /// Embeds one input of text and images.
    public func embed(_ input: MultimodalInput, options: MultimodalEmbedOptions = .init()) async throws -> Embedding {
        try await embed([input], options: options)[0]
    }

    /// Embeds many inputs of text and images, each in one forward pass that reads its images
    /// where they sit among its texts. Images are decoded and resized on the CPU several at a
    /// time; `progress` fires after every GPU batch. Cancelling the surrounding task stops the
    /// work between batches. Throws `imageEncoderNotLoaded` when an input has an image and the
    /// model was loaded without `LoadOptions.imageEncoder`, and `imageDecoding` naming the input's
    /// index and the image's part when one cannot be read.
    public func embed(_ inputs: [MultimodalInput], options: MultimodalEmbedOptions = .init(),
                      progress: (@Sendable (EmbeddingProgress) -> Void)? = nil) async throws -> [Embedding] {
        guard EmbedOptions.allowedDimensions.contains(options.dimensions) else {
            throw GemmaEmbedError.invalidOptions("dimensions must be one of \(EmbedOptions.allowedDimensions)")
        }
        guard options.maxTokens >= 2 && options.maxTokens <= info.maxTokens else {
            throw GemmaEmbedError.invalidOptions("maxTokens must be between 2 and \(info.maxTokens)")
        }
        if let b = options.batchSize, b < 1 { throw GemmaEmbedError.invalidOptions("batchSize must be at least 1") }
        let imageParts = inputs.map { input in
            input.parts.enumerated().compactMap { p, part -> (part: Int, image: ImageInput)? in
                if case .image(let image) = part { return (p, image) }
                return nil
            }
        }
        let preprocessor = visionConfig.map(ImagePreprocessor.init(config:))
        if let visionConfig {
            guard options.maxImageTokens >= 1 && options.maxImageTokens <= visionConfig.nativeMaxImageTokens else {
                throw GemmaEmbedError.invalidOptions("maxImageTokens must be between 1 and \(visionConfig.nativeMaxImageTokens)")
            }
        } else if imageParts.contains(where: { !$0.isEmpty }) {
            throw GemmaEmbedError.imageEncoderNotLoaded
        }
        let markers = try SpanMarkers.image(tokenizer: tokenizer)

        var results: [Embedding] = []
        results.reserveCapacity(inputs.count)
        var completedTokens = 0
        try await EmbeddingModel.forEachBatch(count: inputs.count, batchSize: options.batchSize,
                                              budgets: [batchTokenBudget, ImageEmbedOptions.patchBudget],
                                              cost: { [$0.sequence.ids.count, $0.images.reduce(0) { $0 + $1.patchCount }] },
                                              prepare: { range -> [(sequence: MultimodalSequence, images: [PreparedImage])] in
            let named = range.flatMap { i in imageParts[i].map { ("input \(i), part \($0.part)", $0.image) } }
            let prepared = named.isEmpty ? [] : try await prepare(named, with: preprocessor!, maxImageTokens: options.maxImageTokens)
            var next = 0
            return try range.map { i in
                let own = Array(prepared[next..<(next + imageParts[i].count)])
                next += own.count
                guard let sequence = MultimodalSequence(prefix: options.task.prefix, parts: inputs[i].parts, imageRows: own.map(\.tokenCount),
                                                        maxTokens: options.maxTokens, markers: markers,
                                                        encode: { tokenizer.encode($0, addSpecialTokens: false) }) else {
                    throw GemmaEmbedError.invalidOptions("input \(i): its images alone need more than maxTokens (\(options.maxTokens)) rows")
                }
                return (sequence, own)
            }
        }) { batch in
            var packed = PackedBatch()
            for item in batch {
                let rows = packed.reserveVectorRows(item.images.reduce(0) { $0 + $1.tokenCount })
                packed.append(item.sequence.ids.map { $0 < 0 ? rows[Int(-$0) - 1] : $0 })
            }
            let images = batch.flatMap(\.images)
            let vision = images.isEmpty ? nil : VisionBatch(images: images)
            let vectors = try await onGPU { [packed] in try $0.run(packed, images: vision) }
            for (item, vector) in zip(batch, vectors) {
                results.append(Embedding(vector: finish(vector, dimensions: options.dimensions, normalise: options.normalise),
                                         tokenCount: item.sequence.ids.count, truncated: item.sequence.truncated))
            }
            completedTokens += packed.tokenCount
            progress?(EmbeddingProgress(completedTexts: results.count, totalTexts: inputs.count, completedTokens: completedTokens))
        }
        return results
    }
}
