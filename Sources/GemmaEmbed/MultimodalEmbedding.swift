import Foundation

/// One input to the text tower: text, images and audio in the order they appear. EmbeddingGemma 2
/// reads an image or a clip where its placeholder sits in the text; the parts carry that order.
public struct MultimodalInput: Sendable {
    public enum Part: Sendable {
        case text(String)
        case image(ImageInput)
        case audio(AudioInput)
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

/// An image's or a clip's rows inside a mixed input: how many, and the tokens that frame them.
struct EncoderSpan: Sendable {
    let rows: Int
    let markers: SpanMarkers
}

/// The text-tower sequence of one `MultimodalInput`, as llama.cpp's mtmd lays it out:
/// `[bos] tok(prefix + text) [<|image>] v... [<image|>] tok(text) [<|audio>] a... [<audio|>] ... [eos]`.
struct MultimodalSequence: Sendable {
    /// Token ids, and `-(r + 1)` for row `r` of the input's encoder rows, counted across its spans in order.
    let ids: [Int32]
    let truncated: Bool

    /// Lays out `parts` with `spans[k]` for the k-th image or clip between `bos` and `eos`. The
    /// prefix joins the first text, or leads on its own before an image or clip. Text is cut from
    /// the end until the sequence fits `maxTokens`; nil when the spans and their frames alone do not fit.
    init?(prefix: String, parts: [MultimodalInput.Part], spans: [EncoderSpan], maxTokens: Int, bos: Int32, eos: Int32,
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
        precondition(texts.filter { $0 == nil }.count == spans.count)
        let fixed = 2 + spans.reduce(0) { $0 + $1.rows + 2 }
        guard fixed <= maxTokens else { return nil }
        var excess = fixed + texts.reduce(0) { $0 + ($1?.count ?? 0) } - maxTokens
        truncated = excess > 0
        for i in texts.indices.reversed() where excess > 0 {
            guard let text = texts[i] else { continue }
            let cut = min(excess, text.count)
            texts[i] = Array(text.dropLast(cut))
            excess -= cut
        }
        var ids = [bos]
        var remaining = spans.makeIterator()
        var row: Int32 = 0
        for text in texts {
            if let text {
                ids += text
            } else {
                let span = remaining.next()!
                ids.append(span.markers.begin)
                ids += (row..<(row + Int32(span.rows))).map { -$0 - 1 }
                ids.append(span.markers.end)
                row += Int32(span.rows)
            }
        }
        ids.append(eos)
        self.ids = ids
    }
}

extension EmbeddingModel {
    /// Embeds one input of text, images and audio.
    public func embed(_ input: MultimodalInput, options: MultimodalEmbedOptions = .init()) async throws -> Embedding {
        try await embed([input], options: options)[0]
    }

    /// Embeds many inputs of text, images and audio, each in one forward pass that reads its
    /// images and clips where they sit among its texts. Images and clips are decoded on the CPU
    /// several at a time; `progress` fires after every GPU batch. Cancelling the surrounding task
    /// stops the work between batches. Throws `imageEncoderNotLoaded` or `audioEncoderNotLoaded`
    /// when an input needs an encoder the model was loaded without, and `imageDecoding` or
    /// `audioDecoding` naming the input's index and the part when one cannot be read.
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
        let audioParts = inputs.map { input in
            input.parts.enumerated().compactMap { p, part -> (part: Int, audio: AudioInput)? in
                if case .audio(let audio) = part { return (p, audio) }
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
        let frontEnd = audioConfig.map { AudioFrontEnd(melBins: $0.info.melBins) }
        if frontEnd == nil && audioParts.contains(where: { !$0.isEmpty }) {
            throw GemmaEmbedError.audioEncoderNotLoaded
        }
        let imageMarkers = try SpanMarkers.image(tokenizer: tokenizer)
        let audioMarkers = try SpanMarkers.audio(tokenizer: tokenizer)

        struct Item: Sendable {
            let sequence: MultimodalSequence
            let images: [PreparedImage]
            let audios: [PreparedAudio]
            /// Whether each span, in part order, is an image (else a clip).
            let spanIsImage: [Bool]
        }
        var results: [Embedding] = []
        results.reserveCapacity(inputs.count)
        var completedTokens = 0
        try await EmbeddingModel.forEachBatch(count: inputs.count, batchSize: options.batchSize,
                                              budgets: [batchTokenBudget, ImageEmbedOptions.patchBudget, AudioEmbedOptions.frameBudget],
                                              cost: { [$0.sequence.ids.count, $0.images.reduce(0) { $0 + $1.patchCount }, $0.audios.reduce(0) { $0 + $1.frameCount }] },
                                              prepare: { range -> [Item] in
            let namedImages = range.flatMap { i in imageParts[i].map { ("input \(i), part \($0.part)", $0.image) } }
            let images = namedImages.isEmpty ? [] : try await self.prepare(namedImages, with: preprocessor!, maxImageTokens: options.maxImageTokens)
            let namedAudios = range.flatMap { i in audioParts[i].map { ("input \(i), part \($0.part)", $0.audio) } }
            let audios = namedAudios.isEmpty ? [] : try await self.prepare(namedAudios, with: frontEnd!, maxRows: options.maxTokens)
            var nextImage = 0, nextAudio = 0
            return try range.map { i in
                let ownImages = Array(images[nextImage..<(nextImage + imageParts[i].count)])
                nextImage += ownImages.count
                let ownAudios = Array(audios[nextAudio..<(nextAudio + audioParts[i].count)])
                nextAudio += ownAudios.count
                var imageIterator = ownImages.makeIterator(), audioIterator = ownAudios.makeIterator()
                var spans: [EncoderSpan] = [], spanIsImage: [Bool] = []
                for part in inputs[i].parts {
                    switch part {
                    case .text: continue
                    case .image: spans.append(EncoderSpan(rows: imageIterator.next()!.tokenCount, markers: imageMarkers)); spanIsImage.append(true)
                    case .audio: spans.append(EncoderSpan(rows: audioIterator.next()!.tokenCount, markers: audioMarkers)); spanIsImage.append(false)
                    }
                }
                guard let sequence = MultimodalSequence(prefix: options.task.prefix, parts: inputs[i].parts, spans: spans, maxTokens: options.maxTokens,
                                                        bos: self.tokenizer.bosId, eos: self.tokenizer.eosId, encode: { self.tokenizer.encode($0, addSpecialTokens: false) }) else {
                    throw GemmaEmbedError.invalidOptions("input \(i): its images and clips alone need more than maxTokens (\(options.maxTokens)) rows")
                }
                return Item(sequence: sequence, images: ownImages, audios: ownAudios, spanIsImage: spanIsImage)
            }
        }) { batch in
            // Image rows fill the vectors buffer from row 0, audio rows from the next padded boundary.
            var packed = PackedBatch()
            let imageRowIds = batch.map { packed.reserveVectorRows($0.images.reduce(0) { $0 + $1.tokenCount }) }
            let imageRows = imageRowIds.reduce(0) { $0 + $1.count }
            if batch.contains(where: { !$0.audios.isEmpty }) { packed.skipVectorRows(to: MetalForward.audioVectorRowOffset(imageRows: imageRows)) }
            let audioRowIds = batch.map { packed.reserveVectorRows($0.audios.reduce(0) { $0 + $1.tokenCount }) }
            for (b, item) in batch.enumerated() {
                var images = imageRowIds[b].makeIterator(), audios = audioRowIds[b].makeIterator()
                var rowMap: [Int32] = []
                var imageIndex = 0, audioIndex = 0
                for isImage in item.spanIsImage {
                    if isImage {
                        for _ in 0..<item.images[imageIndex].tokenCount { rowMap.append(images.next()!) }
                        imageIndex += 1
                    } else {
                        for _ in 0..<item.audios[audioIndex].tokenCount { rowMap.append(audios.next()!) }
                        audioIndex += 1
                    }
                }
                packed.append(item.sequence.ids.map { $0 < 0 ? rowMap[Int(-$0) - 1] : $0 })
            }
            let images = batch.flatMap(\.images)
            let audios = batch.flatMap(\.audios)
            let vision = images.isEmpty ? nil : VisionBatch(images: images)
            let audio = audios.isEmpty ? nil : AudioBatch(audios: audios)
            let vectors = try await onGPU { [packed] in try $0.run(packed, images: vision, audio: audio) }
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
