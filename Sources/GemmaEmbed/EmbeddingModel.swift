import CryptoKit
import Foundation
import Metal

/// An EmbeddingGemma 2 text encoder, and optionally its image encoder, running on the local GPU.
///
/// ```swift
/// let model = try await EmbeddingModel.load(from: weightsURL)
/// let docs = try await model.embed(["first text", "second text"])
/// let query = try await model.embed("what is the first text about?", options: .init(task: .query))
/// let score = model.similarity(query, docs[0])
/// ```
///
/// Images and texts share one embedding space once the image encoder is loaded:
///
/// ```swift
/// let model = try await EmbeddingModel.load(from: weightsURL, options: .init(imageEncoder: mmprojURL))
/// let picture = try await model.embed(image: .url(photoURL))
/// let caption = try await model.embed("a vineyard at sunset", options: .init(task: .query))
/// let score = model.similarity(picture, caption)
/// ```
public final class EmbeddingModel: @unchecked Sendable {
    public let info: ModelInfo
    public let tokenizer: Tokenizer
    public let device: DeviceInfo
    /// The kernel configuration in use (defaults, an explicit override, or the autotuned one).
    public var schedule: KernelSchedule { forward.schedule }
    /// Seconds spent in `load`, split by phase.
    public let loadTimings: [String: Double]

    let forward: MetalForward
    let visionConfig: VisionConfig?
    let batchTokenBudget: Int
    private let gpuQueue = DispatchQueue(label: "GemmaEmbed.forward", qos: .userInitiated)

    /// Opens a GGUF weight file, validates it, maps it for the GPU and warms every kernel up.
    public static func load(from url: URL, options: LoadOptions = .init()) async throws -> EmbeddingModel {
        try await Task.detached(priority: .userInitiated) {
            try EmbeddingModel(url: url, options: options)
        }.value
    }

    init(url: URL, options: LoadOptions) throws {
        var timings: [String: Double] = [:]
        var t = Date()
        func lap(_ name: String) {
            timings[name, default: 0] += Date().timeIntervalSince(t)
            t = Date()
        }
        let file = try GGUFFile(url: url)
        lap("parse")
        let context = try MetalContext()
        lap("compileShaders")
        let hash = options.computeFileHash ? EmbeddingModel.sha256(of: file) : nil
        lap("hash")
        var image: (config: VisionConfig, weights: VisionWeights)?
        if let encoderURL = options.imageEncoder {
            let encoderFile = try GGUFFile(url: encoderURL)
            let encoderConfig = try VisionConfig(gguf: encoderFile, fileSHA256: options.computeFileHash ? EmbeddingModel.sha256(of: encoderFile) : nil)
            image = (encoderConfig, try VisionWeights(gguf: encoderFile, config: encoderConfig))
            encoderFile.prefault()
            lap("imageEncoder")
        }
        let config = try ModelConfig(gguf: file, deviceName: context.device.name, fileSHA256: hash, imageEncoder: image?.config.info)
        let weights = try ModelWeights(gguf: file, config: config)
        self.tokenizer = try Tokenizer(gguf: file)
        lap("tokenizer")
        file.prefault()
        lap("prefault")
        var vision: VisionTower?
        if let image {
            vision = try VisionTower(context: context, config: image.config, weights: image.weights, markers: ImageMarkers(tokenizer: tokenizer))
            lap("imageEncoder")
        }
        let forward = try MetalForward(context: context, config: config, weights: weights, schedule: options.schedule ?? .default, vision: vision)
        try context.warmUpPipelines()
        lap("pipelines")
        self.forward = forward
        self.visionConfig = image?.config
        self.info = config.info
        self.device = DeviceInfo(name: context.device.name, maxBufferLength: context.device.maxBufferLength,
                                 recommendedMaxWorkingSetSize: Int(context.device.recommendedMaxWorkingSetSize),
                                 hasUnifiedMemory: context.device.hasUnifiedMemory)
        self.batchTokenBudget = max(64, options.batchTokenBudget)
        // One forward over a small batch wires the weights and fills the workspace before the first request.
        var warm = PackedBatch()
        warm.append(tokenizer.encode("warm up"))
        _ = try forward.run(warm)
        lap("warmForward")
        if options.schedule == nil && options.autotune {
            let tuner = Autotune(cacheDirectory: options.autotuneCacheDirectory)
            forward.schedule = try tuner.schedule(for: forward, tokenizer: tokenizer, fileHash: hash ?? EmbeddingModel.sha256(of: file))
            lap("autotune")
        }
        self.loadTimings = timings
    }

    static func sha256(of file: GGUFFile) -> String {
        var hasher = SHA256()
        let base = file.mapping.base
        let size = file.mapping.size
        var offset = 0
        let chunk = 64 << 20
        while offset < size {
            let n = min(chunk, size - offset)
            hasher.update(bufferPointer: UnsafeRawBufferPointer(start: base + offset, count: n))
            offset += n
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Embeds one text.
    public func embed(_ text: String, options: EmbedOptions = .init()) async throws -> Embedding {
        try await embed([text], options: options)[0]
    }

    /// Embeds many texts, batching them on the GPU. `progress` fires after every batch.
    /// Cancelling the surrounding task stops the work between batches.
    public func embed(_ texts: [String], options: EmbedOptions = .init(),
                      progress: (@Sendable (EmbeddingProgress) -> Void)? = nil) async throws -> [Embedding] {
        guard EmbedOptions.allowedDimensions.contains(options.dimensions) else {
            throw GemmaEmbedError.invalidOptions("dimensions must be one of \(EmbedOptions.allowedDimensions)")
        }
        guard options.maxTokens >= 2 && options.maxTokens <= info.maxTokens else {
            throw GemmaEmbedError.invalidOptions("maxTokens must be between 2 and \(info.maxTokens)")
        }
        if let b = options.batchSize, b < 1 { throw GemmaEmbedError.invalidOptions("batchSize must be at least 1") }
        guard !texts.isEmpty else { return [] }

        let prepared = texts.map { prepare($0, options: options) }
        let plan = EmbeddingModel.batches(prepared.map { [$0.tokens.count] }, batchSize: options.batchSize, budgets: [batchTokenBudget])
        var results = [Embedding?](repeating: nil, count: texts.count)
        var completedTexts = 0
        var completedTokens = 0
        for range in plan {
            try Task.checkCancellation()
            var batch = PackedBatch()
            for i in range { batch.append(prepared[i].tokens) }
            let vectors = try await runBatch(batch)
            for (j, i) in range.enumerated() {
                results[i] = Embedding(vector: finish(vectors[j], dimensions: options.dimensions, normalise: options.normalise),
                                       tokenCount: prepared[i].tokens.count, truncated: prepared[i].truncated)
            }
            completedTexts += range.count
            completedTokens += batch.tokenCount
            progress?(EmbeddingProgress(completedTexts: completedTexts, totalTexts: texts.count, completedTokens: completedTokens))
        }
        return results.map { $0! }
    }

    /// Cosine similarity when both embeddings are normalised, otherwise the dot product.
    public func similarity(_ a: Embedding, _ b: Embedding) -> Float {
        var dot: Float = 0
        for i in 0..<min(a.vector.count, b.vector.count) { dot += a.vector[i] * b.vector[i] }
        return dot
    }

    struct PreparedText {
        let tokens: [Int32]
        let truncated: Bool
    }

    func prepare(_ text: String, options: EmbedOptions) -> PreparedText {
        var ids = tokenizer.encode(options.task.prefix + text)
        if ids.count > options.maxTokens {
            ids = Array(ids.prefix(options.maxTokens - 1)) + [tokenizer.eosId]
            return PreparedText(tokens: ids, truncated: true)
        }
        return PreparedText(tokens: ids, truncated: false)
    }

    /// Splits items into consecutive batches of at least one item, by count when `batchSize` is
    /// given, otherwise before the item that would take any of the batch's summed costs past its budget.
    static func batches(_ costs: [[Int]], batchSize: Int?, budgets: [Int]) -> [Range<Int>] {
        var result: [Range<Int>] = []
        var start = 0
        var totals = budgets.map { _ in 0 }
        for (i, cost) in costs.enumerated() {
            let full = batchSize.map { i - start >= $0 } ?? zip(zip(totals, cost), budgets).contains { $0.0 + $0.1 > $1 }
            if i > start && full {
                result.append(start..<i)
                start = i
                totals = budgets.map { _ in 0 }
            }
            totals = zip(totals, cost).map(+)
        }
        if start < costs.count { result.append(start..<costs.count) }
        return result
    }

    /// Prepares `count` items a chunk at a time and hands them to `body` in order, in the batches
    /// `batches` cuts. Preparation runs a chunk ahead, so a batch is cut only once the item after
    /// it is known and the batches are the ones `batches` plans over every item.
    static func forEachBatch<Item>(count: Int, batchSize: Int?, budgets: [Int], cost: (Item) -> [Int],
                                   prepare: (Range<Int>) async throws -> [Item],
                                   _ body: ([Item]) async throws -> Void) async throws {
        let chunk = max(batchSize ?? 1, ProcessInfo.processInfo.activeProcessorCount)
        var pending: [Item] = []
        var next = 0
        while next < count || !pending.isEmpty {
            try Task.checkCancellation()
            var plan = batches(pending.map(cost), batchSize: batchSize, budgets: budgets)
            while next < count && plan.count < 2 {
                let range = next..<min(count, next + chunk)
                pending += try await prepare(range)
                next = range.upperBound
                plan = batches(pending.map(cost), batchSize: batchSize, budgets: budgets)
            }
            try await body(Array(pending[plan[0]]))
            pending.removeFirst(plan[0].count)
        }
    }

    func finish(_ vector: [Float], dimensions: Int, normalise: Bool) -> [Float] {
        var v = Array(vector.prefix(dimensions))
        if normalise {
            let norm = v.reduce(0) { $0 + $1 * $1 }.squareRoot()
            if norm > 0 { for i in 0..<v.count { v[i] /= norm } }
        }
        return v
    }

    func runBatch(_ batch: PackedBatch) async throws -> [[Float]] {
        try await onGPU { try $0.run(batch) }
    }

    /// Runs `work` on the queue that owns the GPU workspaces.
    func onGPU<T: Sendable>(_ work: @escaping @Sendable (MetalForward) throws -> T) async throws -> T {
        let forward = self.forward
        return try await withCheckedThrowingContinuation { continuation in
            gpuQueue.async {
                continuation.resume(with: Result { try work(forward) })
            }
        }
    }

    /// Runs the forward over already-tokenized sequences and returns pooled, unnormalised vectors.
    /// Used by the CLI's parity and benchmark modes.
    public func embedTokens(_ sequences: [[Int32]]) async throws -> [[Float]] {
        var batch = PackedBatch()
        for s in sequences { batch.append(s) }
        return try await runBatch(batch)
    }
}
