import Foundation
import XCTest
@testable import GemmaEmbed

final class GGUFReaderTests: XCTestCase {
    func testTensorDirectoryMatchesFixture() throws {
        let url = try TestSupport.requireWeights()
        let file = try GGUFFile(url: url)
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: TestSupport.fixture("model.json"))) as! [String: Any]
        let expected = json["tensors"] as! [[String: Any]]
        XCTAssertEqual(file.tensors.count, expected.count)
        XCTAssertEqual(file.tensors.count, json["tensor_count"] as! Int)
        XCTAssertEqual(file.metadata.count, json["kv_count"] as! Int)
        XCTAssertEqual(file.fileSize, json["bytes"] as! Int)
        for (got, want) in zip(file.tensors, expected) {
            XCTAssertEqual(got.name, want["name"] as! String)
            XCTAssertEqual(got.shape, want["shape_ne"] as! [Int], got.name)
            XCTAssertEqual(got.type.name, want["type"] as! String, got.name)
            XCTAssertEqual(got.offset, want["offset"] as! Int, got.name)
        }
        let counts = Dictionary(grouping: file.tensors, by: { $0.type.name }).mapValues(\.count)
        XCTAssertEqual(counts, ["Q8_0": 218, "F32": 194, "BF16": 1])
        XCTAssertEqual(file.alignment, 32)
        XCTAssertEqual(file.dataOffset % 32, 0)
        let last = file.tensors.max(by: { $0.offset < $1.offset })!
        XCTAssertEqual(file.dataOffset + last.offset + last.byteCount, file.fileSize)
    }

    func testMetadataMatchesFixture() throws {
        let url = try TestSupport.requireWeights()
        let file = try GGUFFile(url: url)
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: TestSupport.fixture("model.json"))) as! [String: Any]
        let meta = json["metadata"] as! [String: [String: Any]]
        XCTAssertEqual(file.string("general.architecture"), "gemma-embedding2")
        XCTAssertEqual(file.int("gemma-embedding2.block_count"), 24)
        XCTAssertEqual(file.int("gemma-embedding2.embedding_length"), 512)
        XCTAssertEqual(file.int("gemma-embedding2.embedding_length_out"), 768)
        XCTAssertEqual(file.int("gemma-embedding2.feed_forward_length"), 2048)
        XCTAssertEqual(file.int("gemma-embedding2.attention.head_count"), 4)
        XCTAssertEqual(file.int("gemma-embedding2.attention.sliding_window"), 1024)
        XCTAssertEqual(file.float("gemma-embedding2.rope.freq_base"), 1_000_000)
        XCTAssertEqual(file.float("gemma-embedding2.rope.freq_base_swa"), 10_000)
        XCTAssertEqual(file.bool("gemma-embedding2.attention.causal"), false)
        XCTAssertEqual(file.metadata["gemma-embedding2.attention.head_count_kv"]?.int32ArrayValue,
                       (meta["gemma-embedding2.attention.head_count_kv"]!["items"] as! [Int]).map(Int32.init))
        XCTAssertEqual(file.metadata["gemma-embedding2.attention.sliding_window_pattern"]?.boolArrayValue,
                       meta["gemma-embedding2.attention.sliding_window_pattern"]!["items"] as? [Bool])
        XCTAssertEqual(file.string("tokenizer.ggml.model"), "gemma4")
        XCTAssertEqual(file.int("tokenizer.ggml.bos_token_id"), 2)
        XCTAssertEqual(file.int("tokenizer.ggml.eos_token_id"), 1)
        XCTAssertEqual(file.int("tokenizer.ggml.padding_token_id"), 0)
        XCTAssertEqual(file.bool("tokenizer.ggml.add_bos_token"), true)
        XCTAssertEqual(file.bool("tokenizer.ggml.add_eos_token"), true)
        let tokens = try XCTUnwrap(file.metadata["tokenizer.ggml.tokens"]?.stringArrayValue)
        XCTAssertEqual(tokens.count, 262144)
        XCTAssertEqual(Array(tokens[0..<5]), ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>"])
        let merges = try XCTUnwrap(file.metadata["tokenizer.ggml.merges"]?.stringArrayValue)
        XCTAssertEqual(merges.count, meta["tokenizer.ggml.merges"]!["len"] as! Int)
        let types = try XCTUnwrap(file.metadata["tokenizer.ggml.token_type"]?.int32ArrayValue)
        XCTAssertEqual(types.count, 262144)
        for key in meta.keys {
            XCTAssertNotNil(file.metadata[key], "missing key \(key)")
        }
    }

    func testRejectsNonGGUF() throws {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("not-a-gguf-\(UUID().uuidString)")
        try Data(repeating: 0x41, count: 4096).write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }
        XCTAssertThrowsError(try GGUFFile(url: tmp)) { error in
            guard case GemmaEmbedError.invalidGGUF = error else { return XCTFail("unexpected error \(error)") }
        }
    }
}
