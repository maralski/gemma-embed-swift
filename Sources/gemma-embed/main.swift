import Foundation
import GemmaEmbed

let usage = """
usage: gemma-embed [options] [FILE]
       gemma-embed --images [options] FILE... | DIR
       gemma-embed --audio [options] FILE... | DIR
  Embeds one text per line from FILE or stdin, or the given images or sound files, and prints one JSON
  object per line.

options:
  --model PATH          GGUF weight file (default: $GEMMA_EMBED_WEIGHTS)
  --task NAME           document | query | clustering | classification | similarity | none (default: document)
  --dimensions N        768 | 512 | 256 | 128 (default: 768)
  --max-tokens N        truncate texts to N tokens including BOS and EOS (default: 2048)
  --batch-tokens N      token budget per GPU batch (default: 8192)
  --autotune            run or reuse the per-device kernel autotune
  --bench FILE          time a throughput run over FILE: a feed JSON with "posts" ({"gist","text"})
                        or a text file with one document per line; prints posts per second
  --runs N              repetitions for --bench (default: 3)
  --parity DIR          check tokens.jsonl and embeddings.jsonl in DIR; exit 1 below cosine 0.9995
  --tune-report         print autotune measurements for this device and exit
  --profile FILE        time each kernel class over the texts in FILE (same formats as --bench)
  --profile-images DIR  time each vision tower kernel class over the first --batch-size images of DIR (default 1)
  --info                print model and device facts and exit

image options:
  --image-encoder PATH  image encoder GGUF, the mmproj file (default: $GEMMA_EMBED_IMAGE_WEIGHTS)
  --images              embed the image files given, or every image in a directory given (jpg, jpeg,
                        png, heic, tif, tiff, gif, bmp, webp, sorted by name); honours --dimensions
  --max-image-tokens N  most vision tokens per image, 1 to 1120 (default: 256); one value per corpus,
                        since a different cap embeds the same image differently
  --batch-size N|auto   images per GPU batch, or auto for by patch count (default: by patch count;
                        --bench-images: by patch count, 8 and 1)
  --parity-images DIR   check the images of manifest.json against embeddings.jsonl in DIR;
                        exit 1 below cosine 0.9995
  --bench-images DIR    time embedding every image in DIR, decoding included, and the GPU part alone;
                        prints images per second for each batch size

audio options:
  --audio-encoder PATH  audio encoder GGUF, the same mmproj file (default: $GEMMA_EMBED_AUDIO_WEIGHTS,
                        else the image encoder path)
  --audio               embed the sound files given, or every sound file in a directory given (wav, mp3,
                        m4a, aac, aiff, aif, caf, flac, sorted by name); honours --dimensions and, when
                        given, --max-tokens (default: the model's 8192 rows, about 5 minutes 27 seconds)
  --batch-size N|auto   clips per GPU batch, or auto for by mel frames (default: by mel frames;
                        --bench-audio: by mel frames, 8 and 1)
  --parity-audio DIR    check the clips of manifest.json against embeddings.jsonl in DIR;
                        exit 1 below cosine 0.9995
  --bench-audio DIR     time embedding every clip in DIR, decoding included, and the GPU part alone;
                        prints audio seconds per wall second for each batch size
  --profile-audio DIR   time each audio tower kernel class over the first --batch-size clips of DIR (default 1)
"""

struct Arguments {
    var model: String? = ProcessInfo.processInfo.environment["GEMMA_EMBED_WEIGHTS"]
    var task = "document"
    var dimensions = 768
    var maxTokens = 2048
    var maxTokensGiven = false
    var batchTokens = 8192
    var autotune = false
    var bench: String?
    var runs = 3
    var parity: String?
    var tuneReport = false
    var profile: String?
    var profileImages: String?
    var info = false
    var files: [String] = []
    var imageEncoder: String? = ProcessInfo.processInfo.environment["GEMMA_EMBED_IMAGE_WEIGHTS"]
    var images = false
    var maxImageTokens = ImageEmbedOptions.defaultMaxImageTokens
    enum BatchChoice { case auto, fixed(Int) }
    var batch: BatchChoice?
    var batchSize: Int? {
        if case .fixed(let n) = batch { return n }
        return nil
    }
    var parityImages: String?
    var benchImages: String?
    var audioEncoder: String? = ProcessInfo.processInfo.environment["GEMMA_EMBED_AUDIO_WEIGHTS"]
    var audio = false
    var parityAudio: String?
    var benchAudio: String?
    var profileAudio: String?

    init(_ args: [String]) throws {
        var i = 0
        func value() throws -> String {
            i += 1
            guard i < args.count else { throw CLIError("missing value for \(args[i - 1])") }
            return args[i]
        }
        while i < args.count {
            switch args[i] {
            case "--model": model = try value()
            case "--task": task = try value()
            case "--dimensions": dimensions = Int(try value()) ?? 0
            case "--max-tokens": maxTokens = Int(try value()) ?? 0; maxTokensGiven = true
            case "--batch-tokens": batchTokens = Int(try value()) ?? 0
            case "--autotune": autotune = true
            case "--bench": bench = try value()
            case "--runs": runs = Int(try value()) ?? 3
            case "--parity": parity = try value()
            case "--tune-report": tuneReport = true
            case "--profile": profile = try value()
            case "--profile-images": profileImages = try value()
            case "--info": info = true
            case "--image-encoder": imageEncoder = try value()
            case "--images": images = true
            case "--max-image-tokens": maxImageTokens = Int(try value()) ?? 0
            case "--batch-size":
                let v = try value()
                batch = v == "auto" ? .auto : .fixed(Int(v) ?? 0)
            case "--parity-images": parityImages = try value()
            case "--bench-images": benchImages = try value()
            case "--audio-encoder": audioEncoder = try value()
            case "--audio": audio = true
            case "--parity-audio": parityAudio = try value()
            case "--bench-audio": benchAudio = try value()
            case "--profile-audio": profileAudio = try value()
            case "-h", "--help": print(usage); exit(0)
            default:
                if args[i].hasPrefix("-") { throw CLIError("unknown option \(args[i])") }
                files.append(args[i])
            }
            i += 1
        }
    }

    var imageMode: Bool { images || parityImages != nil || benchImages != nil || profileImages != nil }
    var audioMode: Bool { audio || parityAudio != nil || benchAudio != nil || profileAudio != nil }

    var imageOptions: ImageEmbedOptions {
        ImageEmbedOptions(dimensions: dimensions, maxImageTokens: maxImageTokens, batchSize: batchSize)
    }

    var audioOptions: AudioEmbedOptions {
        AudioEmbedOptions(dimensions: dimensions, maxTokens: maxTokensGiven ? maxTokens : nil, batchSize: batchSize)
    }

    var taskPrefix: TaskPrefix {
        get throws {
            switch task {
            case "document": return .document
            case "query": return .query
            case "clustering": return .clustering
            case "classification": return .classification
            case "similarity": return .similarity
            case "none": return .none
            default: throw CLIError("unknown task \(task)")
            }
        }
    }
}

struct CLIError: Error, CustomStringConvertible {
    let description: String
    init(_ d: String) { description = d }
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data(("gemma-embed: " + message + "\n").utf8))
    exit(1)
}

func loadModel(_ a: Arguments) async throws -> EmbeddingModel {
    guard let path = a.model else { throw CLIError("no weight file: pass --model or set GEMMA_EMBED_WEIGHTS") }
    if a.imageMode && a.imageEncoder == nil {
        throw CLIError("no image encoder: pass --image-encoder or set GEMMA_EMBED_IMAGE_WEIGHTS")
    }
    let audioPath = a.audioEncoder ?? a.imageEncoder
    if a.audioMode && audioPath == nil {
        throw CLIError("no audio encoder: pass --audio-encoder or set GEMMA_EMBED_AUDIO_WEIGHTS (the mmproj file)")
    }
    let encoder = a.imageMode || a.info ? a.imageEncoder.map { URL(fileURLWithPath: $0) } : nil
    let audioEncoder = a.audioMode || a.info ? audioPath.map { URL(fileURLWithPath: $0) } : nil
    let t0 = Date()
    let model = try await EmbeddingModel.load(from: URL(fileURLWithPath: path),
                                              options: LoadOptions(autotune: a.autotune, batchTokenBudget: a.batchTokens, imageEncoder: encoder,
                                                                   audioEncoder: audioEncoder))
    let phases = model.loadTimings.sorted { $0.value > $1.value }.map { "\($0.key) \(String(format: "%.2f", $0.value))s" }.joined(separator: ", ")
    FileHandle.standardError.write(Data(String(format: "loaded %@ on %@ in %.2fs (%@)\n", model.info.name, model.device.name, Date().timeIntervalSince(t0), phases).utf8))
    return model
}

func readLines(_ a: Arguments) throws -> [String] {
    guard a.files.count <= 1 else { throw CLIError("text mode reads one FILE; pass --images or --audio to embed files") }
    let data: Data
    if let f = a.files.first {
        data = try Data(contentsOf: URL(fileURLWithPath: f))
    } else {
        data = FileHandle.standardInput.readDataToEndOfFile()
    }
    return String(decoding: data, as: UTF8.self).split(separator: "\n", omittingEmptySubsequences: false).map(String.init).filter { !$0.isEmpty }
}

func jsonLine(_ object: [String: Any]) -> String {
    let data = try! JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    return String(decoding: data, as: UTF8.self)
}

func benchTexts(_ path: String) throws -> [String] {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any], let posts = json["posts"] as? [[String: Any]] {
        return posts.map { p in
            let gist = p["gist"] as? String ?? ""
            let text = String((p["text"] as? String ?? "").prefix(2000))
            return (gist + "\n\n" + text).trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }
    return String(decoding: data, as: UTF8.self).split(separator: "\n").map(String.init).filter { !$0.isEmpty }
}

func runBench(_ a: Arguments, model: EmbeddingModel, path: String) async throws {
    let texts = try benchTexts(path)
    let options = EmbedOptions(task: try a.taskPrefix, maxTokens: a.maxTokens, dimensions: a.dimensions)
    let warm = try await model.embed(Array(texts.prefix(8)), options: options)
    _ = warm
    var seconds: [Double] = []
    var tokens = 0
    for _ in 0..<max(1, a.runs) {
        let t0 = Date()
        let out = try await model.embed(texts, options: options)
        seconds.append(Date().timeIntervalSince(t0))
        tokens = out.reduce(0) { $0 + $1.tokenCount }
    }
    let best = seconds.min()!
    let median = seconds.sorted()[seconds.count / 2]
    print(jsonLine([
        "texts": texts.count, "tokens": tokens, "runs": seconds.count,
        "seconds_median": median, "seconds_best": best,
        "texts_per_second_median": Double(texts.count) / median,
        "tokens_per_second_median": Double(tokens) / median,
        "schedule": "gemm \(model.schedule.gemmTile.rows)x\(model.schedule.gemmTile.columns)x\(model.schedule.gemmTile.depth), attention q\(model.schedule.attentionQueries)",
        "device": model.device.name,
    ]))
}

func runParity(model: EmbeddingModel, dir: String) async throws -> Bool {
    let base = URL(fileURLWithPath: dir)
    func rows(_ name: String) throws -> [[String: Any]] {
        let text = try String(contentsOf: base.appendingPathComponent(name), encoding: .utf8)
        return try text.split(separator: "\n").map { try JSONSerialization.jsonObject(with: Data($0.utf8)) as! [String: Any] }
    }
    let tokenRows = try rows("tokens.jsonl")
    let embeddingRows = try rows("embeddings.jsonl")
    var tokenMismatches = 0
    var sequences: [[Int32]] = []
    for r in tokenRows {
        let want = (r["token_ids"] as! [Int]).map(Int32.init)
        let got = model.tokenizer.encode(r["input"] as! String)
        if got != want { tokenMismatches += 1 }
        sequences.append(want)
    }
    let vectors = try await model.embedTokens(sequences)
    var minCos: Float = 1
    var maxDiff: Float = 0
    for (i, r) in embeddingRows.enumerated() {
        let want = (r["embedding"] as! [Double]).map(Float.init)
        let raw = vectors[i]
        let norm = raw.reduce(0) { $0 + $1 * $1 }.squareRoot()
        let got = raw.map { $0 / norm }
        var dot: Float = 0, diff: Float = 0
        for j in 0..<want.count { dot += got[j] * want[j]; diff = max(diff, abs(got[j] - want[j])) }
        minCos = min(minCos, dot)
        maxDiff = max(maxDiff, diff)
    }
    print(jsonLine(["texts": tokenRows.count, "token_mismatches": tokenMismatches, "min_cosine": Double(minCos), "max_abs_diff": Double(maxDiff)]))
    return tokenMismatches == 0 && minCos > 0.9995
}

let imageExtensions: Set<String> = ["jpg", "jpeg", "png", "heic", "tif", "tiff", "gif", "bmp", "webp"]

/// The files given, with each directory replaced by its image files sorted by name.
func imageFiles(_ paths: [String]) throws -> [URL] {
    let urls = try paths.flatMap { path -> [URL] in
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory) else { throw CLIError("\(path) does not exist") }
        guard isDirectory.boolValue else { return [URL(fileURLWithPath: path)] }
        return try FileManager.default.contentsOfDirectory(atPath: path)
            .filter { imageExtensions.contains(($0 as NSString).pathExtension.lowercased()) }
            .sorted()
            .map { URL(fileURLWithPath: path).appendingPathComponent($0) }
    }
    guard !urls.isEmpty else { throw CLIError("no image files in \(paths.joined(separator: " "))") }
    return urls
}

func runImages(_ a: Arguments, model: EmbeddingModel) async throws {
    let urls = try imageFiles(a.files)
    let embeddings = try await model.embed(images: urls.map { .url($0) }, options: a.imageOptions)
    for (i, (url, e)) in zip(urls, embeddings).enumerated() {
        print(jsonLine(["index": i, "file": url.path, "tokens": e.tokenCount, "embedding": e.vector.map(Double.init)]))
    }
}

func runParityImages(_ a: Arguments, model: EmbeddingModel, dir: String) async throws -> Bool {
    let base = URL(fileURLWithPath: dir)
    let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: base.appendingPathComponent("manifest.json"))) as! [[String: Any]]
    let text = try String(contentsOf: base.appendingPathComponent("embeddings.jsonl"), encoding: .utf8)
    var references: [String: [String: Any]] = [:]
    for line in text.split(separator: "\n") {
        let row = try JSONSerialization.jsonObject(with: Data(line.utf8)) as! [String: Any]
        references[row["file"] as! String] = row
    }
    let files = manifest.map { $0["file"] as! String }.filter {
        references[$0] != nil && FileManager.default.isReadableFile(atPath: base.appendingPathComponent($0).path)
    }
    guard !files.isEmpty else { throw CLIError("no manifest image of \(dir) is present with a reference embedding") }
    let embeddings = try await model.embed(images: files.map { .url(base.appendingPathComponent($0)) }, options: a.imageOptions)
    var minCos: Float = 1, maxDiff: Float = 0, worst = files[0], tokenMismatches = 0
    for (file, e) in zip(files, embeddings) {
        let row = references[file]!
        let want = (row["embedding"] as! [Double]).map(Float.init)
        if e.tokenCount != row["prompt_tokens"] as? Int { tokenMismatches += 1 }
        var dot: Float = 0, diff: Float = 0
        for j in 0..<want.count { dot += e.vector[j] * want[j]; diff = max(diff, abs(e.vector[j] - want[j])) }
        if !(dot >= minCos) { minCos = dot; worst = file }
        maxDiff = max(maxDiff, diff)
    }
    print(jsonLine(["images": files.count, "token_mismatches": tokenMismatches, "min_cosine": Double(minCos),
                    "max_abs_diff": Double(maxDiff), "worst_file": worst, "max_image_tokens": a.maxImageTokens]))
    return tokenMismatches == 0 && minCos > 0.9995
}

let audioExtensions: Set<String> = ["wav", "mp3", "m4a", "aac", "aiff", "aif", "caf", "flac"]

/// The files given, with each directory replaced by its sound files sorted by name.
func audioFiles(_ paths: [String]) throws -> [URL] {
    let urls = try paths.flatMap { path -> [URL] in
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory) else { throw CLIError("\(path) does not exist") }
        guard isDirectory.boolValue else { return [URL(fileURLWithPath: path)] }
        return try FileManager.default.contentsOfDirectory(atPath: path)
            .filter { audioExtensions.contains(($0 as NSString).pathExtension.lowercased()) }
            .sorted()
            .map { URL(fileURLWithPath: path).appendingPathComponent($0) }
    }
    guard !urls.isEmpty else { throw CLIError("no sound files in \(paths.joined(separator: " "))") }
    return urls
}

/// Seconds of 16 kHz audio a clip's rows stand for: 40 ms per row after the two stride-2 convolutions.
func audioSeconds(rows: Int) -> Double { Double(rows - 4) * 0.04 }

func runAudio(_ a: Arguments, model: EmbeddingModel) async throws {
    let urls = try audioFiles(a.files)
    let embeddings = try await model.embed(audios: urls.map { .url($0) }, options: a.audioOptions)
    for (i, (url, e)) in zip(urls, embeddings).enumerated() {
        print(jsonLine(["index": i, "file": url.path, "tokens": e.tokenCount, "seconds": audioSeconds(rows: e.tokenCount), "embedding": e.vector.map(Double.init)]))
    }
}

func runParityAudio(_ a: Arguments, model: EmbeddingModel, dir: String) async throws -> Bool {
    let base = URL(fileURLWithPath: dir)
    let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: base.appendingPathComponent("manifest.json"))) as! [[String: Any]]
    let text = try String(contentsOf: base.appendingPathComponent("embeddings.jsonl"), encoding: .utf8)
    var references: [String: [String: Any]] = [:]
    for line in text.split(separator: "\n") {
        let row = try JSONSerialization.jsonObject(with: Data(line.utf8)) as! [String: Any]
        references[row["file"] as! String] = row
    }
    let files = manifest.map { $0["file"] as! String }.filter {
        references[$0] != nil && FileManager.default.isReadableFile(atPath: base.appendingPathComponent($0).path)
    }
    guard !files.isEmpty else { throw CLIError("no manifest clip of \(dir) is present with a reference embedding") }
    let embeddings = try await model.embed(audios: files.map { .url(base.appendingPathComponent($0)) }, options: a.audioOptions)
    var minCos: Float = 1, maxDiff: Float = 0, worst = files[0], tokenMismatches = 0
    for (file, e) in zip(files, embeddings) {
        let row = references[file]!
        let want = (row["embedding"] as! [Double]).map(Float.init)
        if e.tokenCount != row["prompt_tokens"] as? Int { tokenMismatches += 1 }
        var dot: Float = 0, diff: Float = 0
        for j in 0..<want.count { dot += e.vector[j] * want[j]; diff = max(diff, abs(e.vector[j] - want[j])) }
        if !(dot >= minCos) { minCos = dot; worst = file }
        maxDiff = max(maxDiff, diff)
    }
    print(jsonLine(["clips": files.count, "token_mismatches": tokenMismatches, "min_cosine": Double(minCos),
                    "max_abs_diff": Double(maxDiff), "worst_file": worst]))
    return tokenMismatches == 0 && minCos > 0.9995
}

func runBenchAudio(_ a: Arguments, model: EmbeddingModel, dir: String) async throws {
    let clips = try audioFiles([dir]).map { AudioInput.url($0) }
    _ = try await model.embed(audios: Array(clips.prefix(8)), options: AudioEmbedOptions(batchSize: 8))
    let sizes: [Int?]
    switch a.batch {
    case .auto: sizes = [nil]
    case .fixed(let n): sizes = [n]
    case nil: sizes = [nil, 8, 1]
    }
    for size in sizes {
        let options = AudioEmbedOptions(dimensions: a.dimensions, maxTokens: a.maxTokensGiven ? a.maxTokens : nil, batchSize: size)
        var seconds: [Double] = [], gpuSeconds: [Double] = []
        var rows = 0, audioLength = 0.0
        for _ in 0..<max(1, a.runs) {
            let t0 = Date()
            let out = try await model.embed(audios: clips, options: options)
            seconds.append(Date().timeIntervalSince(t0))
            rows = out.reduce(0) { $0 + $1.tokenCount }
            audioLength = out.reduce(0) { $0 + audioSeconds(rows: $1.tokenCount) }
        }
        var batchRuns: [[Double]] = []
        for _ in 0..<max(1, a.runs) {
            let batches = try await model.audioForwardBatchSeconds(clips, options: options)
            batchRuns.append(batches)
            gpuSeconds.append(batches.reduce(0, +))
        }
        let median = seconds.sorted()[seconds.count / 2], best = seconds.min()!
        let gpuMedian = gpuSeconds.sorted()[gpuSeconds.count / 2]
        let gpuMinSum = sumOfBatchMinimums(batchRuns)
        print(jsonLine([
            "clips": clips.count, "batch_size": size.map { $0 as Any } ?? "auto", "runs": seconds.count, "rows": rows, "audio_seconds": audioLength,
            "seconds_median": median, "seconds_best": best,
            "clips_per_second_median": Double(clips.count) / median, "clips_per_second_best": Double(clips.count) / best,
            "audio_seconds_per_second_median": audioLength / median, "audio_seconds_per_second_best": audioLength / best,
            "gpu_seconds_median": gpuMedian, "gpu_audio_seconds_per_second_median": audioLength / gpuMedian,
            "gpu_seconds_minsum": gpuMinSum, "gpu_batches": batchRuns[0].count,
            "audio_encoder_load_seconds": model.loadTimings["audioEncoder"] ?? 0, "device": model.device.name,
        ]))
    }
}

/// Each batch's fastest run, summed over the batches; GPU work from outside the process slows some
/// runs of a batch and not others, so this tracks the uncontended time.
func sumOfBatchMinimums(_ runs: [[Double]]) -> Double {
    runs[0].indices.reduce(0.0) { total, i in total + runs.map { $0[i] }.min()! }
}

func runBenchImages(_ a: Arguments, model: EmbeddingModel, dir: String) async throws {
    let images = try imageFiles([dir]).map { ImageInput.url($0) }
    _ = try await model.embed(images: Array(images.prefix(8)), options: ImageEmbedOptions(maxImageTokens: a.maxImageTokens, batchSize: 8))
    let sizes: [Int?]
    switch a.batch {
    case .auto: sizes = [nil]
    case .fixed(let n): sizes = [n]
    case nil: sizes = [nil, 8, 1]
    }
    for size in sizes {
        let options = ImageEmbedOptions(dimensions: a.dimensions, maxImageTokens: a.maxImageTokens, batchSize: size)
        var seconds: [Double] = [], gpuSeconds: [Double] = []
        var rows = 0
        for _ in 0..<max(1, a.runs) {
            let t0 = Date()
            let out = try await model.embed(images: images, options: options)
            seconds.append(Date().timeIntervalSince(t0))
            rows = out.reduce(0) { $0 + $1.tokenCount }
        }
        var batchRuns: [[Double]] = []
        for _ in 0..<max(1, a.runs) {
            let batches = try await model.imageForwardBatchSeconds(images, options: options)
            batchRuns.append(batches)
            gpuSeconds.append(batches.reduce(0, +))
        }
        let median = seconds.sorted()[seconds.count / 2], best = seconds.min()!
        let gpuMedian = gpuSeconds.sorted()[gpuSeconds.count / 2]
        let gpuMinSum = sumOfBatchMinimums(batchRuns)
        print(jsonLine([
            "images": images.count, "batch_size": size.map { $0 as Any } ?? "auto", "runs": seconds.count, "rows": rows, "max_image_tokens": a.maxImageTokens,
            "seconds_median": median, "seconds_best": best,
            "images_per_second_median": Double(images.count) / median, "images_per_second_best": Double(images.count) / best,
            "gpu_seconds_median": gpuMedian, "gpu_images_per_second_median": Double(images.count) / gpuMedian,
            "gpu_seconds_minsum": gpuMinSum, "gpu_batches": batchRuns[0].count,
            "image_encoder_load_seconds": model.loadTimings["imageEncoder"] ?? 0, "device": model.device.name,
        ]))
    }
}

let arguments: Arguments
do {
    arguments = try Arguments(Array(CommandLine.arguments.dropFirst()))
} catch {
    fail("\(error)\n\(usage)")
}

Task.detached {
    var code: Int32 = 0
    do {
        let model = try await loadModel(arguments)
        if arguments.info {
            let i = model.info
            print(jsonLine(["architecture": i.architecture, "name": i.name, "layers": i.layerCount, "hidden": i.hiddenSize,
                            "output_dimensions": i.outputDimensions, "vocabulary": i.vocabularySize, "max_tokens": i.maxTokens,
                            "quantization": i.quantization, "device": model.device.name, "file_bytes": i.fileSize,
                            "sliding_window": i.slidingWindow,
                            "image_encoder": i.imageEncoder.map { e -> Any in
                                ["layers": e.layerCount, "hidden": e.hiddenSize, "patch_size": e.patchSize,
                                 "projection_dim": e.projectionDim, "file_bytes": e.fileSize,
                                 "default_max_image_tokens": ImageEmbedOptions.defaultMaxImageTokens]
                            } ?? NSNull(),
                            "audio_encoder": i.audioEncoder.map { e -> Any in
                                ["layers": e.layerCount, "hidden": e.hiddenSize, "heads": e.headCount, "mel_bins": e.melBins,
                                 "projection_dim": e.projectionDim, "file_bytes": e.fileSize, "sample_rate": 16000, "chunk_seconds": 30]
                            } ?? NSNull()]))
        } else if arguments.tuneReport {
            let result = try Autotune.measure(model: model)
            for (k, v) in result.measurements.sorted(by: { $0.value < $1.value }) { print(String(format: "%-24@ %.4fs", k, v)) }
            print("chosen: gemm \(result.schedule.gemmTile.rows)x\(result.schedule.gemmTile.columns)x\(result.schedule.gemmTile.depth), attention q\(result.schedule.attentionQueries)")
        } else if let path = arguments.profile {
            let texts = try benchTexts(path)
            for s in try Profiler.profile(model: model, texts: texts) {
                print(String(format: "%-40@ %9.3f ms  %@", s.name, s.seconds * 1000, s.note))
            }
        } else if let dir = arguments.profileImages {
            let urls = Array(try imageFiles([dir]).prefix(arguments.batchSize ?? 1))
            for s in try Profiler.profileImages(model: model, urls: urls, maxImageTokens: arguments.maxImageTokens) {
                print(String(format: "%-40@ %9.3f ms  %@", s.name, s.seconds * 1000, s.note))
            }
        } else if let dir = arguments.profileAudio {
            let urls = Array(try audioFiles([dir]).prefix(arguments.batchSize ?? 1))
            for s in try Profiler.profileAudio(model: model, urls: urls) {
                print(String(format: "%-40@ %9.3f ms  %@", s.name, s.seconds * 1000, s.note))
            }
        } else if let path = arguments.bench {
            try await runBench(arguments, model: model, path: path)
        } else if let dir = arguments.parity {
            code = try await runParity(model: model, dir: dir) ? 0 : 1
        } else if let dir = arguments.parityImages {
            code = try await runParityImages(arguments, model: model, dir: dir) ? 0 : 1
        } else if let dir = arguments.benchImages {
            try await runBenchImages(arguments, model: model, dir: dir)
        } else if let dir = arguments.parityAudio {
            code = try await runParityAudio(arguments, model: model, dir: dir) ? 0 : 1
        } else if let dir = arguments.benchAudio {
            try await runBenchAudio(arguments, model: model, dir: dir)
        } else if arguments.audio {
            try await runAudio(arguments, model: model)
        } else if arguments.images {
            try await runImages(arguments, model: model)
        } else {
            let texts = try readLines(arguments)
            let options = EmbedOptions(task: try arguments.taskPrefix, maxTokens: arguments.maxTokens, dimensions: arguments.dimensions)
            let embeddings = try await model.embed(texts, options: options)
            for (i, e) in embeddings.enumerated() {
                print(jsonLine(["index": i, "tokens": e.tokenCount, "truncated": e.truncated, "embedding": e.vector.map(Double.init)]))
            }
        }
    } catch {
        FileHandle.standardError.write(Data("gemma-embed: \(error)\n".utf8))
        code = 1
    }
    fflush(stdout)
    exit(code)
}
dispatchMain()
