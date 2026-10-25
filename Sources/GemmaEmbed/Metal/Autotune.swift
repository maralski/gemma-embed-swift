import Foundation

/// Picks GEMM and attention configurations by timing whole forwards on real text.
///
/// Candidates must beat the default by 5 percent on a packed batch of real sequences, and the
/// winning schedule is confirmed by a second whole-forward measurement. Results are cached per
/// device, kernel source digest and model hash so later loads measure nothing.
public struct Autotune: Sendable {
    public let cacheDirectory: URL

    public init(cacheDirectory: URL? = nil) {
        self.cacheDirectory = cacheDirectory ?? Autotune.defaultCacheDirectory
    }

    public static var defaultCacheDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return base.appendingPathComponent("GemmaEmbed", isDirectory: true)
    }

    public var cacheURL: URL { cacheDirectory.appendingPathComponent("autotune.json") }

    struct CacheEntry: Codable {
        var schedule: KernelSchedule
        var defaultSeconds: Double
        var tunedSeconds: Double
        var date: Date
    }

    /// Returns the cached schedule for the key, or tunes and caches one.
    func schedule(for forward: MetalForward, tokenizer: Tokenizer, fileHash: String) throws -> KernelSchedule {
        let key = "\(forward.context.device.name)|\(forward.context.sourceDigest)|\(fileHash)|v1"
        var cache = loadCache()
        if let entry = cache[key] { return entry.schedule }
        let result = try tune(forward: forward, tokenizer: tokenizer)
        cache[key] = CacheEntry(schedule: result.schedule, defaultSeconds: result.defaultSeconds, tunedSeconds: result.tunedSeconds, date: Date())
        saveCache(cache)
        return result.schedule
    }

    public struct Result: Sendable {
        public let schedule: KernelSchedule
        public let defaultSeconds: Double
        public let tunedSeconds: Double
        public let measurements: [String: Double]
    }

    /// Real text with the document prefix, repeated to a few thousand tokens of mixed lengths.
    static func sampleBatch(tokenizer: Tokenizer) -> PackedBatch {
        let sentences = [
            "The central bank left its policy rate unchanged for a third consecutive meeting, citing slower inflation.",
            "A new open-weight language model was released today with a permissive license and a 32K context window.",
            "Engineers retain their best people with growth, autonomy and recognition rather than perks.",
            "Rain is expected across the region overnight, with gusts up to 60 km/h along the coast.",
        ]
        var batch = PackedBatch()
        var i = 0
        while batch.tokenCount < 6000 {
            let repeats = [1, 3, 8, 20, 40][i % 5]
            let text = TaskPrefix.document.prefix + Array(repeating: sentences[i % sentences.count], count: repeats).joined(separator: " ")
            batch.append(tokenizer.encode(text))
            i += 1
        }
        return batch
    }

    func tune(forward: MetalForward, tokenizer: Tokenizer) throws -> Result {
        let batch = Autotune.sampleBatch(tokenizer: tokenizer)
        let saved = forward.schedule
        defer { forward.schedule = saved }
        var measurements: [String: Double] = [:]

        func time(_ schedule: KernelSchedule) throws -> Double {
            forward.schedule = schedule
            _ = try forward.timeForward(batch)
            var samples: [Double] = []
            for _ in 0..<5 { samples.append(try forward.timeForward(batch)) }
            return samples.sorted()[samples.count / 2]
        }

        let defaultSeconds = try time(.default)
        measurements["default"] = defaultSeconds
        var best = KernelSchedule.default
        var bestSeconds = defaultSeconds
        for tile in GemmTile.candidates where tile != best.gemmTile {
            var s = best
            s.gemmTile = tile
            let t = try time(s)
            measurements["gemm \(tile.suffix)"] = t
            if t < bestSeconds * 0.95 { best = s; bestSeconds = t }
        }
        for queries in KernelSchedule.attentionQueryCandidates where queries != best.attentionQueries {
            var s = best
            s.attentionQueries = queries
            let t = try time(s)
            measurements["attention q\(queries)"] = t
            if t < bestSeconds * 0.95 { best = s; bestSeconds = t }
        }
        // Whole-forward confirmation: the winner must still beat the default on a fresh measurement.
        if best != .default {
            let again = try time(best)
            let defaultAgain = try time(.default)
            measurements["confirm tuned"] = again
            measurements["confirm default"] = defaultAgain
            if again >= defaultAgain * 0.98 {
                best = .default
                bestSeconds = defaultAgain
            } else {
                bestSeconds = again
            }
        }
        return Result(schedule: best, defaultSeconds: defaultSeconds, tunedSeconds: bestSeconds, measurements: measurements)
    }

    func loadCache() -> [String: CacheEntry] {
        guard let data = try? Data(contentsOf: cacheURL) else { return [:] }
        return (try? JSONDecoder().decode([String: CacheEntry].self, from: data)) ?? [:]
    }

    func saveCache(_ cache: [String: CacheEntry]) {
        try? FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(cache) { try? data.write(to: cacheURL, options: .atomic) }
    }

    /// Tunes a loaded model and returns the measurements without touching the cache.
    public static func measure(model: EmbeddingModel) throws -> Result {
        try Autotune().tune(forward: model.forward, tokenizer: model.tokenizer)
    }
}
