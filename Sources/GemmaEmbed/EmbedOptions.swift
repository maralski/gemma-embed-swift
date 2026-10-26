// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// The instruction prefix EmbeddingGemma expects in front of the text, per the model card.
public enum TaskPrefix: Sendable, Equatable {
    /// A document to be retrieved: `title: none | text: ` (or the given title).
    case document(title: String? = nil)
    /// A search query: `task: search result | query: `, or a custom task description.
    case query(String? = nil)
    case clustering
    case classification
    /// Sentence similarity: `task: sentence similarity | query: `.
    case similarity
    /// No prefix; the text is embedded as is.
    case none

    public static let document = TaskPrefix.document(title: nil)
    public static let query = TaskPrefix.query(nil)

    /// The exact prefix string prepended to the text.
    public var prefix: String {
        switch self {
        case .document(let title): return "title: \(title ?? "none") | text: "
        case .query(let task): return "task: \(task ?? "search result") | query: "
        case .clustering: return "task: clustering | query: "
        case .classification: return "task: classification | query: "
        case .similarity: return "task: sentence similarity | query: "
        case .none: return ""
        }
    }
}

/// Options for one `embed` call.
public struct EmbedOptions: Sendable {
    public var task: TaskPrefix
    /// Longest token sequence per text, including BOS and EOS. Longer texts are truncated
    /// to this many tokens and still end with EOS. At most `ModelInfo.maxTokens`.
    public var maxTokens: Int
    /// Output width: 768, 512, 256 or 128 (Matryoshka truncation followed by renormalisation).
    public var dimensions: Int
    /// L2-normalise the output so dot products are cosine similarities.
    public var normalise: Bool
    /// Texts per GPU batch; `nil` packs texts by token count automatically.
    public var batchSize: Int?

    public init(task: TaskPrefix = .document, maxTokens: Int = 2048, dimensions: Int = 768, normalise: Bool = true, batchSize: Int? = nil) {
        self.task = task
        self.maxTokens = maxTokens
        self.dimensions = dimensions
        self.normalise = normalise
        self.batchSize = batchSize
    }

    public static let allowedDimensions = [768, 512, 256, 128]
}

/// Options for one `embed(images:)` call.
public struct ImageEmbedOptions: Sendable {
    /// Output width: 768, 512, 256 or 128, as for text.
    public var dimensions: Int
    /// L2-normalise the output so dot products are cosine similarities.
    public var normalise: Bool
    /// Most vision tokens per image, between 1 and 1120 (llama.cpp's native limit). Larger images
    /// are scaled down to fit; smaller ones are scaled up to at least 70 tokens. The default, 256,
    /// is what `llama-server` uses with its default batch size. The same image embeds differently
    /// at different caps (cosine 0.96 between 256 and 1120 on a large photo), so embed a whole
    /// corpus, and the queries against it, with one cap.
    public var maxImageTokens: Int
    /// Images per GPU batch; `nil` packs images up to `patchBudget` patches per batch, and always
    /// at least one image. Larger batches run slower per image on an M3 Max (see the README), so
    /// the default is one full-size image, or several small ones, per batch.
    public var batchSize: Int?

    public init(dimensions: Int = 768, normalise: Bool = true, maxImageTokens: Int = ImageEmbedOptions.defaultMaxImageTokens,
                batchSize: Int? = nil) {
        self.dimensions = dimensions
        self.normalise = normalise
        self.maxImageTokens = maxImageTokens
        self.batchSize = batchSize
    }

    public static let defaultMaxImageTokens = 256
    /// Patch rows per GPU batch when `batchSize` is nil: one image at the default cap (2304
    /// patches), or several smaller ones. Measured on an M3 Max, GPU time per image grows with
    /// the batch (6.3 s for 30 images one at a time, 7.7 s eight at a time).
    public static let patchBudget = 4096
}

/// Options for one `embed(audios:)` call.
public struct AudioEmbedOptions: Sendable {
    /// Output width: 768, 512, 256 or 128, as for text.
    public var dimensions: Int
    /// L2-normalise the output so dot products are cosine similarities.
    public var normalise: Bool
    /// Longest sequence per clip, counting BOS, EOS, the two frame tokens and one row per 40 ms
    /// of audio; nil means the model's limit (8192 rows, about 5 minutes 27 seconds). A clip that
    /// needs more rows is refused with `invalidOptions`, since llama.cpp does not cut audio either.
    public var maxTokens: Int?
    /// Clips per GPU batch; `nil` packs clips up to `frameBudget` mel frames per batch, and always
    /// at least one clip.
    public var batchSize: Int?

    public init(dimensions: Int = 768, normalise: Bool = true, maxTokens: Int? = nil, batchSize: Int? = nil) {
        self.dimensions = dimensions
        self.normalise = normalise
        self.maxTokens = maxTokens
        self.batchSize = batchSize
    }

    /// Mel frames (10 ms each) per GPU batch when `batchSize` is nil: two minutes of audio. The
    /// first convolution's output takes 32 KB per frame, so a batch holds about 400 MB of it.
    public static let frameBudget = 12000
}

/// Options for one `embed(_: MultimodalInput)` call.
public struct MultimodalEmbedOptions: Sendable {
    /// Prepended to the input's first text, as the model card prescribes for text only; it leads
    /// on its own when the input opens with an image.
    public var task: TaskPrefix
    /// Longest sequence per input, counting BOS, EOS, the frame tokens and every image's and
    /// clip's rows. Text is cut from the end (the last text part first) until the input fits; the
    /// image and audio frames stay whole. An input whose images and clips alone exceed it is
    /// refused with `invalidOptions`. At most `ModelInfo.maxTokens`.
    public var maxTokens: Int
    /// Output width: 768, 512, 256 or 128, as for text.
    public var dimensions: Int
    /// L2-normalise the output so dot products are cosine similarities.
    public var normalise: Bool
    /// Most vision tokens per image, as `ImageEmbedOptions.maxImageTokens`.
    public var maxImageTokens: Int
    /// Inputs per GPU batch; `nil` packs inputs up to `LoadOptions.batchTokenBudget` rows and
    /// `ImageEmbedOptions.patchBudget` image patches per batch, and always at least one input.
    public var batchSize: Int?

    public init(task: TaskPrefix = .document, maxTokens: Int = 2048, dimensions: Int = 768, normalise: Bool = true,
                maxImageTokens: Int = ImageEmbedOptions.defaultMaxImageTokens, batchSize: Int? = nil) {
        self.task = task
        self.maxTokens = maxTokens
        self.dimensions = dimensions
        self.normalise = normalise
        self.maxImageTokens = maxImageTokens
        self.batchSize = batchSize
    }
}

/// One embedded text, image, clip or mixed input.
public struct Embedding: Sendable {
    public let vector: [Float]
    /// Rows the text tower saw: tokens including BOS and EOS after any truncation, plus for each
    /// image or clip its encoder rows and the two that frame them.
    public let tokenCount: Int
    /// True when the text exceeded `maxTokens` and was cut.
    public let truncated: Bool

    public init(vector: [Float], tokenCount: Int, truncated: Bool) {
        self.vector = vector
        self.tokenCount = tokenCount
        self.truncated = truncated
    }
}

/// Progress of a long `embed` call, reported after every GPU batch. For images, clips and mixed
/// inputs, texts count inputs and tokens count the rows of `Embedding.tokenCount`.
public struct EmbeddingProgress: Sendable {
    public let completedTexts: Int
    public let totalTexts: Int
    public let completedTokens: Int
}

/// Options for `EmbeddingModel.load`.
public struct LoadOptions: Sendable {
    /// Run the kernel autotuner on first load and cache its schedule per device and model.
    public var autotune: Bool
    /// Where autotune results live; defaults to `Application Support/GemmaEmbed`.
    public var autotuneCacheDirectory: URL?
    /// Use this schedule instead of the default or the autotuned one.
    public var schedule: KernelSchedule?
    /// Compute the SHA256 of the weight file (about 0.2 s for 300 MB).
    public var computeFileHash: Bool
    /// Token budget per GPU batch when `EmbedOptions.batchSize` is nil.
    public var batchTokenBudget: Int
    /// The image encoder GGUF (`mmproj-embeddinggemma-2-Q8_0.gguf`) to load alongside the text
    /// model, which `embed(images:)` needs. When nil the file is never opened. The encoder is a
    /// load option rather than something added to a loaded model, so a model never changes
    /// after `load` and needs no lock.
    public var imageEncoder: URL?
    /// The audio encoder GGUF, which `embed(audios:)` needs. It is the same mmproj file as the
    /// image encoder, which holds both towers; pass it to one option, the other or both, and only
    /// the towers asked for are loaded. The file is mapped once when both name it.
    public var audioEncoder: URL?

    public init(autotune: Bool = false, autotuneCacheDirectory: URL? = nil, schedule: KernelSchedule? = nil,
                computeFileHash: Bool = false, batchTokenBudget: Int = 8192, imageEncoder: URL? = nil, audioEncoder: URL? = nil) {
        self.autotune = autotune
        self.autotuneCacheDirectory = autotuneCacheDirectory
        self.schedule = schedule
        self.computeFileHash = computeFileHash
        self.batchTokenBudget = batchTokenBudget
        self.imageEncoder = imageEncoder
        self.audioEncoder = audioEncoder
    }
}

/// Facts about the Metal device the model runs on.
public struct DeviceInfo: Sendable {
    public let name: String
    public let maxBufferLength: Int
    public let recommendedMaxWorkingSetSize: Int
    public let hasUnifiedMemory: Bool
}
