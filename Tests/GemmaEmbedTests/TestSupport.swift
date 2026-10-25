import Foundation
import XCTest
@testable import GemmaEmbed

enum TestSupport {
    /// `models/` at the package root, gitignored: weights and llama.cpp dumps live there unless an env var points elsewhere.
    static let modelsDirectory = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("models")

    static let defaultWeightsPath = modelsDirectory.appendingPathComponent("embeddinggemma-2-Q8_0.gguf").path

    static var weightsURL: URL? {
        let path = ProcessInfo.processInfo.environment["GEMMA_EMBED_WEIGHTS"] ?? defaultWeightsPath
        return FileManager.default.isReadableFile(atPath: path) ? URL(fileURLWithPath: path) : nil
    }

    static func requireWeights() throws -> URL {
        guard let url = weightsURL else {
            throw XCTSkip("weights not found; set GEMMA_EMBED_WEIGHTS to the embeddinggemma-2-Q8_0.gguf path")
        }
        return url
    }

    static let defaultImageEncoderPath = modelsDirectory.appendingPathComponent("mmproj-embeddinggemma-2-Q8_0.gguf").path

    static var imageEncoderURL: URL? {
        let path = ProcessInfo.processInfo.environment["GEMMA_EMBED_IMAGE_WEIGHTS"] ?? defaultImageEncoderPath
        return FileManager.default.isReadableFile(atPath: path) ? URL(fileURLWithPath: path) : nil
    }

    static func requireImageEncoder() throws -> URL {
        guard let url = imageEncoderURL else {
            throw XCTSkip("image encoder not found; set GEMMA_EMBED_IMAGE_WEIGHTS to the mmproj-embeddinggemma-2-Q8_0.gguf path")
        }
        return url
    }

    /// A file under `Fixtures`; `name` may be a relative path such as `images/manifest.json`.
    static func fixture(_ name: String) throws -> URL {
        let path = name as NSString
        let directory = path.deletingLastPathComponent
        let subdirectory = directory.isEmpty ? "Fixtures" : "Fixtures/" + directory
        guard let url = Bundle.module.url(forResource: path.lastPathComponent, withExtension: nil, subdirectory: subdirectory) else {
            throw XCTSkip("fixture \(name) missing")
        }
        return url
    }

    static func fixtureJSONLines(_ name: String) throws -> [[String: Any]] {
        let text = try String(contentsOf: fixture(name), encoding: .utf8)
        return try text.split(separator: "\n").map { line in
            try XCTUnwrap(JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any])
        }
    }

    struct TokenFixture {
        let index: Int
        let input: String
        let tokens: [Int32]
    }

    static func tokenFixtures() throws -> [TokenFixture] {
        try fixtureJSONLines("tokens.jsonl").map { row in
            TokenFixture(index: row["i"] as! Int, input: row["input"] as! String, tokens: (row["token_ids"] as! [Int]).map(Int32.init))
        }
    }

    struct EmbeddingFixture {
        let index: Int
        let tokenCount: Int
        let embedding: [Float]
    }

    static func embeddingFixtures() throws -> [EmbeddingFixture] {
        try fixtureJSONLines("embeddings.jsonl").map { row in
            EmbeddingFixture(index: row["i"] as! Int, tokenCount: row["prompt_tokens"] as! Int, embedding: (row["embedding"] as! [Double]).map(Float.init))
        }
    }

    static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0..<a.count { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (na.squareRoot() * nb.squareRoot())
    }

    /// NaN when either side holds a NaN, so a broken kernel cannot pass as a zero difference.
    static func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        var m: Float = 0
        for i in 0..<a.count {
            let d = abs(a[i] - b[i])
            if d.isNaN { return .nan }
            m = max(m, d)
        }
        return m
    }

    static func maxRelDiff(_ got: [Float], _ want: [Float]) -> Float {
        var m: Float = 0
        var scale: Float = 0
        for v in want { scale = max(scale, abs(v)) }
        for i in 0..<got.count {
            let d = abs(got[i] - want[i])
            if d.isNaN { return .nan }
            m = max(m, d)
        }
        return m / max(scale, 1e-30)
    }

    struct ImageEmbeddingFixture {
        let file: String
        let promptTokens: Int
        let embedding: [Float]
    }

    /// llama-server embeddings of the image fixtures, keyed by path relative to `Fixtures/images`.
    static func imageEmbeddingFixtures() throws -> [String: ImageEmbeddingFixture] {
        let rows = try fixtureJSONLines("images/embeddings.jsonl").map { row in
            ImageEmbeddingFixture(file: row["file"] as! String, promptTokens: row["prompt_tokens"] as! Int,
                                  embedding: (row["embedding"] as! [Double]).map(Float.init))
        }
        return Dictionary(uniqueKeysWithValues: rows.map { ($0.file, $0) })
    }

    static let defaultVisionDumpsPath = modelsDirectory.appendingPathComponent("dumps").path

    /// One image's vision graph as dumped from llama.cpp's libmtmd through `cb_eval`.
    struct VisionDump {
        struct Node {
            let index: Int
            let name: String
            let op: String
            /// ggml `ne`, fastest dimension first.
            let shape: [Int]
            let sum: Double
            let absSum: Double
            /// Raw fp32 file next to `nodes.jsonl`, for the nodes that were saved in full.
            let file: String?
        }

        let name: String
        let directory: URL
        /// The original image file the dump was made from.
        let imageURL: URL
        let decodedWidth: Int
        let decodedHeight: Int
        let imageTokenCount: Int
        let nodes: [Node]

        /// The last node of that name: llama.cpp names a view after its source, so names can repeat.
        func node(_ name: String) -> Node? {
            nodes.last { $0.name == name }
        }

        func floats(_ file: String) throws -> [Float] {
            let data = try Data(contentsOf: directory.appendingPathComponent(file))
            return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }

        /// The saved values of a node, in ggml layout (fastest dimension first).
        func values(_ name: String) throws -> [Float] {
            let node = try XCTUnwrap(node(name), "dump \(self.name) has no node \(name)")
            let file = try XCTUnwrap(node.file, "dump \(self.name) did not save \(name)")
            let values = try floats(file)
            XCTAssertEqual(values.count, node.shape.reduce(1, *), "\(self.name) \(name)")
            return values
        }

        /// stb_image's decode of the original file.
        func decodedRGB() throws -> RGBImage {
            let bytes = try Data(contentsOf: directory.appendingPathComponent("decoded_rgb.bin"))
            return RGBImage(width: decodedWidth, height: decodedHeight, pixels: [UInt8](bytes))
        }
    }

    static let defaultDecodedRGBDumpsPath = modelsDirectory.appendingPathComponent("dumps-rgb").path

    /// stb_image's decode of a fixture image and llama.cpp's vision output for it, when present.
    static func decodedRGBDump(_ name: String) -> VisionDump? {
        try? requireVisionDump(name, root: ProcessInfo.processInfo.environment["GEMMA_EMBED_DECODED_RGB_DUMPS"] ?? defaultDecodedRGBDumpsPath)
    }

    static func requireVisionDump(_ name: String, root: String? = nil) throws -> VisionDump {
        let root = root ?? ProcessInfo.processInfo.environment["GEMMA_EMBED_VISION_DUMPS"] ?? defaultVisionDumpsPath
        let directory = URL(fileURLWithPath: root).appendingPathComponent(name)
        let metaURL = directory.appendingPathComponent("meta.json")
        guard FileManager.default.isReadableFile(atPath: metaURL.path) else {
            throw XCTSkip("vision dump \(name) not found; set GEMMA_EMBED_VISION_DUMPS to the dumps directory")
        }
        let meta = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: metaURL)) as? [String: Any])
        var nodes: [VisionDump.Node] = []
        if let text = try? String(contentsOf: directory.appendingPathComponent("nodes.jsonl"), encoding: .utf8) {
            for line in text.split(separator: "\n") {
                let row = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any])
                guard let index = row["i"] as? Int, let sum = row["sum"] as? Double else { continue }
                let file = row["file"] as? String
                nodes.append(VisionDump.Node(index: index, name: row["name"] as! String, op: row["op"] as! String,
                                             shape: row["ne"] as! [Int], sum: sum, absSum: row["abs_sum"] as? Double ?? 0,
                                             file: file?.isEmpty == false ? file : nil))
            }
        }
        return VisionDump(name: name, directory: directory, imageURL: URL(fileURLWithPath: meta["image"] as! String),
                          decodedWidth: meta["decoded_nx"] as! Int, decodedHeight: meta["decoded_ny"] as! Int,
                          imageTokenCount: meta["n_image_tokens"] as! Int, nodes: nodes)
    }
}

extension TestSupport {
    static let defaultAudioDumpsPath = modelsDirectory.appendingPathComponent("audio/dumps").path

    /// One clip's audio graph as dumped from llama.cpp's libmtmd through `cb_eval`
    /// (`scripts/audio_dump.cpp`): miniaudio's samples, the chunk layout, every node and the
    /// 512-wide output rows of every chunk.
    struct AudioDump {
        struct Chunk {
            let tokenCount: Int
            /// Node indices of this chunk's graph: `firstNode ..< endNode`.
            let firstNode: Int
            let endNode: Int
        }

        let name: String
        let directory: URL
        let audioURL: URL
        let sampleCount: Int
        let chunks: [Chunk]
        let nodes: [VisionDump.Node]
        /// Text tokens around the audio: `[bos, <|audio>]` before and `[<audio|>, eos]` after.
        let tokensBefore: [Int32]
        let tokensAfter: [Int32]

        /// Node `index` of the dump; indices run across chunks.
        func node(_ index: Int) -> VisionDump.Node? {
            nodes.first { $0.index == index }
        }

        func floats(_ file: String) throws -> [Float] {
            let data = try Data(contentsOf: directory.appendingPathComponent(file))
            return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }

        /// The saved values of node `index`, in ggml layout (fastest dimension first).
        func values(_ index: Int) throws -> [Float] {
            let node = try XCTUnwrap(node(index), "dump \(name) has no node \(index)")
            let file = try XCTUnwrap(node.file, "dump \(name) did not save node \(index)")
            let values = try floats(file)
            XCTAssertEqual(values.count, node.shape.reduce(1, *), "\(name) node \(index)")
            return values
        }

        /// miniaudio's decode of the clip: mono float32 at 16 kHz.
        func samples() throws -> [Float] {
            let s = try floats("samples.f32")
            XCTAssertEqual(s.count, sampleCount)
            return s
        }

        /// The audio tower's output for chunk `c`: `tokenCount` rows of 512.
        func outputRows(chunk c: Int) throws -> [Float] {
            try floats("output_embd-\(c).f32")
        }
    }

    static var audioDumpsRoot: String {
        ProcessInfo.processInfo.environment["GEMMA_EMBED_AUDIO_DUMPS"] ?? defaultAudioDumpsPath
    }

    /// The dumps present under the audio dumps directory, by name, excluding the decode-only ones.
    static func audioDumpNames() -> [String] {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: audioDumpsRoot)) ?? []
        return names.filter { !$0.hasPrefix("decode-") && FileManager.default.fileExists(atPath: "\(audioDumpsRoot)/\($0)/nodes.jsonl") }.sorted()
    }

    static func requireAudioDump(_ name: String) throws -> AudioDump {
        let directory = URL(fileURLWithPath: audioDumpsRoot).appendingPathComponent(name)
        let metaURL = directory.appendingPathComponent("meta.json")
        guard FileManager.default.isReadableFile(atPath: metaURL.path) else {
            throw XCTSkip("audio dump \(name) not found; set GEMMA_EMBED_AUDIO_DUMPS to the dumps directory")
        }
        let meta = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: metaURL)) as? [String: Any])
        var nodes: [VisionDump.Node] = []
        if let text = try? String(contentsOf: directory.appendingPathComponent("nodes.jsonl"), encoding: .utf8) {
            for line in text.split(separator: "\n") {
                let row = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any])
                guard let index = row["i"] as? Int, let sum = row["sum"] as? Double else { continue }
                let file = row["file"] as? String
                nodes.append(VisionDump.Node(index: index, name: row["name"] as! String, op: row["op"] as! String,
                                             shape: row["ne"] as! [Int], sum: sum, absSum: row["abs_sum"] as? Double ?? 0,
                                             file: file?.isEmpty == false ? file : nil))
            }
        }
        var chunks: [AudioDump.Chunk] = []
        var before: [Int32] = [], after: [Int32] = []
        for entry in meta["chunks"] as! [[String: Any]] {
            if entry["type"] as? String == "audio" {
                chunks.append(AudioDump.Chunk(tokenCount: entry["n_tokens"] as! Int, firstNode: entry["first_node"] as! Int, endNode: entry["end_node"] as! Int))
            } else {
                let tokens = (entry["tokens"] as! [Int]).map(Int32.init)
                if chunks.isEmpty { before += tokens } else { after += tokens }
            }
        }
        return AudioDump(name: name, directory: directory, audioURL: URL(fileURLWithPath: meta["audio"] as! String),
                         sampleCount: meta["n_samples"] as! Int, chunks: chunks, nodes: nodes, tokensBefore: before, tokensAfter: after)
    }
}

extension TestSupport {
    struct AudioFixture {
        let file: String
        let kind: String
        let seconds: Double
        let sampleRate: Int
        let channels: Int
        let format: String
        let url: URL
    }

    /// Every clip in `Fixtures/audio/manifest.json`.
    static func audioFixtures() throws -> [AudioFixture] {
        let data = try Data(contentsOf: fixture("audio/manifest.json"))
        let manifest = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [[String: Any]])
        return try manifest.map { entry in
            let file = entry["file"] as! String
            return AudioFixture(file: file, kind: entry["kind"] as! String, seconds: entry["seconds"] as! Double,
                                sampleRate: entry["sample_rate"] as! Int, channels: entry["channels"] as! Int,
                                format: entry["format"] as! String, url: try fixture("audio/" + file))
        }
    }

    struct AudioEmbeddingFixture {
        let file: String
        let promptTokens: Int
        let embedding: [Float]
    }

    /// llama-server embeddings of the audio fixtures, keyed by path relative to `Fixtures/audio`.
    static func audioEmbeddingFixtures() throws -> [String: AudioEmbeddingFixture] {
        let rows = try fixtureJSONLines("audio/embeddings.jsonl").map { row in
            AudioEmbeddingFixture(file: row["file"] as! String, promptTokens: row["prompt_tokens"] as! Int,
                                  embedding: (row["embedding"] as! [Double]).map(Float.init))
        }
        return Dictionary(uniqueKeysWithValues: rows.map { ($0.file, $0) })
    }
}
