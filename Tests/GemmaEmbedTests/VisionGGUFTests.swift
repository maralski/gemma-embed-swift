import Foundation
import XCTest
@testable import GemmaEmbed

final class VisionGGUFTests: XCTestCase {
    func fixture() throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: TestSupport.fixture("mmproj.json"))) as? [String: Any])
    }

    func testTensorDirectoryMatchesFixture() throws {
        let file = try GGUFFile(url: try TestSupport.requireImageEncoder())
        let json = try fixture()
        let expected = json["tensors"] as! [[String: Any]]
        XCTAssertEqual(Int(file.version), json["version"] as! Int)
        XCTAssertEqual(file.tensors.count, expected.count)
        XCTAssertEqual(file.tensors.count, json["tensor_count"] as! Int)
        XCTAssertEqual(file.metadata.count, json["kv_count"] as! Int)
        XCTAssertEqual(file.fileSize, json["bytes"] as! Int)
        XCTAssertEqual(file.alignment, json["alignment"] as! Int)
        XCTAssertEqual(file.dataOffset, json["data_offset"] as! Int)
        for (got, want) in zip(file.tensors, expected) {
            XCTAssertEqual(got.name, want["name"] as! String)
            XCTAssertEqual(got.shape, want["shape_ne"] as! [Int], got.name)
            XCTAssertEqual(got.type.name, want["type"] as! String, got.name)
            XCTAssertEqual(got.offset, want["offset"] as! Int, got.name)
        }
        let last = file.tensors.max(by: { $0.offset < $1.offset })!
        XCTAssertEqual(file.dataOffset + last.offset + last.byteCount, file.fileSize)
    }

    func testMetadataMatchesFixture() throws {
        let file = try GGUFFile(url: try TestSupport.requireImageEncoder())
        let meta = try fixture()["metadata"] as! [String: [String: Any]]
        XCTAssertEqual(Set(file.metadata.keys), Set(meta.keys))
        for (key, entry) in meta {
            let got = try XCTUnwrap(file.metadata[key], key)
            switch entry["type"] as! String {
            case "str": XCTAssertEqual(got.stringValue, entry["value"] as? String, key)
            case "bool": XCTAssertEqual(got.boolValue, entry["value"] as? Bool, key)
            case "u32", "i32", "u64", "i64": XCTAssertEqual(got.intValue, entry["value"] as? Int, key)
            case "f32": XCTAssertEqual(got.floatValue, (entry["value"] as! NSNumber).floatValue, key)
            case "arr[f32]": XCTAssertEqual(got.float32ArrayValue, (entry["items"] as! [NSNumber]).map(\.floatValue), key)
            case "arr[str]": XCTAssertEqual(got.stringArrayValue, entry["items"] as? [String], key)
            case let type: XCTFail("\(key): fixture type \(type) not compared")
            }
        }
    }

    func testVisionConfig() throws {
        let file = try GGUFFile(url: try TestSupport.requireImageEncoder())
        let config = try VisionConfig(gguf: file, fileSHA256: nil)
        XCTAssertEqual(config.info.layerCount, 16)
        XCTAssertEqual(config.info.hiddenSize, 768)
        XCTAssertEqual(config.info.patchSize, 16)
        XCTAssertEqual(config.info.projectionDim, 512)
        XCTAssertEqual(config.info.fileSize, 554_821_024)
        XCTAssertEqual(config.headCount, 12)
        XCTAssertEqual(config.headDim, 64)
        XCTAssertEqual(config.feedForwardSize, 3072)
        XCTAssertEqual(config.normEpsilon, 1e-6)
        XCTAssertEqual(config.imageMean, [0, 0, 0])
        XCTAssertEqual(config.imageStd, [1, 1, 1])
        XCTAssertEqual(config.patchArea, 768)
        XCTAssertEqual(config.alignment, 48)
        XCTAssertEqual(config.pixelsPerToken, 2304)
    }

    func testRejectsTheTextModel() throws {
        let file = try GGUFFile(url: try TestSupport.requireWeights())
        XCTAssertThrowsError(try VisionConfig(gguf: file, fileSHA256: nil)) { error in
            guard case GemmaEmbedError.invalidGGUF = error else { return XCTFail("unexpected error \(error)") }
        }
    }

    func testWeightsLoadWithVisionTensorsOnly() throws {
        let file = try GGUFFile(url: try TestSupport.requireImageEncoder())
        let config = try VisionConfig(gguf: file, fileSHA256: nil)
        let weights = try VisionWeights(gguf: file, config: config)
        XCTAssertEqual(weights.layers.count, 16)
        XCTAssertEqual(weights.patchEmbedding.rows, 768)
        XCTAssertEqual(weights.patchEmbedding.columns, 768)
        XCTAssertEqual(weights.patchEmbedding.type, .f32)
        XCTAssertEqual(weights.positionCount, 10240)
        XCTAssertEqual(weights.positionTable.rows, 20480)
        XCTAssertEqual(weights.positionTable.columns, 768)
        XCTAssertEqual(weights.projection.rows, 512)
        XCTAssertEqual(weights.projection.columns, 768)
        XCTAssertEqual(weights.projection.type, .q8_0)

        // The patch kernel is [kx, ky, c, oc] in ggml order, so row oc of the view is element c*256 + ky*16 + kx.
        let raw = file.bytes(of: try XCTUnwrap(file.tensor(named: "v.patch_embd.weight"))).bindMemory(to: Float.self)
        var row = [Float](repeating: 0, count: 768)
        weights.patchEmbedding.dequantizeRow(5, into: &row)
        XCTAssertEqual(row[2 * 256 + 7 * 16 + 3], raw[5 * 768 + 2 * 256 + 7 * 16 + 3])
        let yRaw = file.bytes(of: try XCTUnwrap(file.tensor(named: "v.position_embd.weight"))).bindMemory(to: Float.self)
        weights.positionTable.dequantizeRow(10240 + 9, into: &row)
        XCTAssertEqual(row, Array(yRaw[(10240 + 9) * 768..<(10240 + 10) * 768]))

        let referenced = Set([weights.patchEmbedding, weights.positionTable, weights.projection].map(\.info.name)
                             + weights.layers.flatMap { l in
                                 [l.ln1, l.wq, l.wk, l.wv, l.wo, l.qNorm, l.kNorm, l.postAttnNorm, l.ln2, l.wGate, l.wUp, l.wDown, l.postFfnNorm].map(\.info.name)
                             })
        let vision = Set(file.tensors.map(\.name).filter { $0.hasPrefix("v.") || $0 == "mm.input_projection.weight" })
        XCTAssertEqual(referenced.subtracting(vision).sorted(), [], "referenced but not vision tensors")
        XCTAssertEqual(vision.subtracting(referenced).sorted(), [], "vision tensors not referenced")
        XCTAssertEqual(referenced.count, 211)
        XCTAssertTrue(referenced.allSatisfy { !$0.hasPrefix("a.") && !$0.hasPrefix("mm.a.") })
    }
}
