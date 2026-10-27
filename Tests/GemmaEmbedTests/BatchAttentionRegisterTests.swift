// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import XCTest
@testable import GemmaEmbed

final class BatchAttentionRegisterPolicyTests: XCTestCase {
    func testBatchAttentionRegisterSelectionKeepsShortAndMediaPaths() {
        for mode in TextAttentionOperandLoading.allCases {
            XCTAssertEqual(mode.selected(tokenCount: 8192, hasVectors: false, headDim: 512), mode)
            XCTAssertEqual(mode.selected(tokenCount: 512, hasVectors: false, headDim: 256), mode)
            XCTAssertEqual(mode.selected(tokenCount: 511, hasVectors: false, headDim: 512), .staged)
            XCTAssertEqual(mode.selected(tokenCount: 8192, hasVectors: true, headDim: 512), .staged)
            XCTAssertEqual(mode.selected(tokenCount: 8192, hasVectors: false, headDim: 64), .staged)
        }
    }
}

extension MetalForwardTests {
    private func batchAttentionRegisterInputs(_ env: Env) throws -> [(name: String, batch: PackedBatch, sequences: [[Int32]])] {
        let body = try TestSupport.tokenFixtures().flatMap { $0.tokens.dropFirst().dropLast() }
        let shapes: [(String, [Int])] = [
            ("64x8", Array(repeating: 64, count: 8)),
            ("64x32", Array(repeating: 64, count: 32)),
            ("256x8", Array(repeating: 256, count: 8)),
            ("1024x8", Array(repeating: 1024, count: 8)),
            ("mixed", [1, 63, 256, 17, 1024]),
        ]
        var inputs: [(name: String, batch: PackedBatch, sequences: [[Int32]])] = []
        for (name, lengths) in shapes {
            var batch = PackedBatch()
            var sequences: [[Int32]] = []
            for (index, length) in lengths.enumerated() {
                var tokens: [Int32] = [env.tokenizer.bosId]
                if length > 1 {
                    for position in 0..<(length - 2) {
                        tokens.append(body[(index * 127 + position) % body.count])
                    }
                    tokens.append(env.tokenizer.eosId)
                }
                batch.append(tokens)
                sequences.append(tokens)
            }
            inputs.append((name, batch, sequences))
        }
        return inputs
    }

    private func assertBatchAttentionRegisterBitsEqual(_ actual: [Float], _ expected: [Float], _ message: String,
                                                       file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.count, expected.count, message, file: file, line: line)
        let differing = zip(actual, expected).reduce(0) { $0 + ($1.0.bitPattern == $1.1.bitPattern ? 0 : 1) }
        XCTAssertEqual(differing, 0, message, file: file, line: line)
    }

    func testBatchAttentionRegisterKeysMatchStagedForwardInputsExactly() throws {
        let env = try XCTUnwrap(Self.env)
        let forward = env.forward
        let saved = forward.textAttentionOperandLoading
        defer { forward.textAttentionOperandLoading = saved }
        forward.textAttentionOperandLoading = .staged
        for (name, batch, _) in try batchAttentionRegisterInputs(env) {
            let work = batch.attentionWork(queriesPerGroup: forward.schedule.attentionQueries)
            let workBuffer = try forward.context.makeBuffer(array: work, label: "attention register work")
            for staged in try Profiler.stageAttention(forward: forward, batch: batch) {
                let layer = staged.layer
                let heads = forward.config.info.headCount
                let count = batch.tokenCount * heads * layer.headDim
                func output(_ mode: TextAttentionOperandLoading) throws -> [Float] {
                    let rows = MetalForward.paddedRows(batch.tokenCount) + MetalForward.slackRows
                    let out = try forward.context.makeBuffer(array: [Float](repeating: .nan, count: rows * heads * layer.headDim),
                                                             label: "attention register output")
                    let cb = try XCTUnwrap(forward.context.queue.makeCommandBuffer())
                    let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
                    let kernels = Kernels(forward: forward, encoder: enc)
                    try kernels.attention(q: staged.q, k: staged.k, v: staged.v, out: out, work: workBuffer,
                                          workItems: work.count, heads: heads, kvHeads: layer.kvHeadCount,
                                          headDim: layer.headDim,
                                          halfWindow: layer.isSlidingWindow ? forward.config.info.slidingWindow / 2 : 0,
                                          scale: 1, operandLoading: mode)
                    enc.endEncoding()
                    cb.commit()
                    cb.waitUntilCompleted()
                    XCTAssertNil(cb.error)
                    return out.array(of: Float.self, count: count)
                }
                let expected = try output(.staged)
                XCTAssertFalse(expected.contains { !$0.isFinite }, "\(name), \(staged.name)")
                let actual = try output(.registerKeys)
                assertBatchAttentionRegisterBitsEqual(actual, expected, "\(name), \(staged.name), register keys")
            }
        }
    }

    func testBatchAttentionRegisterKeysAndAdaptiveGemmMatchStagedForwardExactly() throws {
        let env = try XCTUnwrap(Self.env)
        let forward = env.forward
        let saved = forward.textAttentionOperandLoading
        let savedGemm = forward.batchRegisterGemmPolicy
        defer {
            forward.textAttentionOperandLoading = saved
            forward.batchRegisterGemmPolicy = savedGemm
        }
        let width = forward.config.info.outputDimensions
        for (name, batch, sequences) in try batchAttentionRegisterInputs(env) {
            forward.textAttentionOperandLoading = .staged
            forward.batchRegisterGemmPolicy = .staged
            let expected = try forward.run(batch)
            let count = batch.tokenCount * width
            let expectedRows = try forward.workspace(forRows: batch.tokenCount).out.array(of: Float.self, count: count)
            forward.textAttentionOperandLoading = .registerKeys
            forward.batchRegisterGemmPolicy = .automatic
            let actual = try forward.run(batch)
            let actualRows = try forward.workspace(forRows: batch.tokenCount).out.array(of: Float.self, count: count)
            XCTAssertEqual(actual.count, expected.count)
            for index in expected.indices {
                assertBatchAttentionRegisterBitsEqual(actual[index], expected[index], "\(name), combined, pooled vector \(index)")
                assertBatchAttentionRegisterBitsEqual(Self.normalised(actual[index]), Self.normalised(expected[index]),
                                                      "\(name), combined, normalised vector \(index)")
            }
            assertBatchAttentionRegisterBitsEqual(actualRows, expectedRows, "\(name), combined, token outputs")
            if name == "1024x8" || name == "mixed" {
                var rowOffset = 0
                for (index, tokens) in sequences.enumerated() {
                    var alone = PackedBatch()
                    alone.append(tokens)
                    let individual = try forward.run(alone)[0]
                    let individualRows = try forward.workspace(forRows: tokens.count).out.array(of: Float.self, count: tokens.count * width)
                    assertBatchAttentionRegisterBitsEqual(individual, actual[index], "\(name), combined, individual vector \(index)")
                    assertBatchAttentionRegisterBitsEqual(individualRows, Array(actualRows[rowOffset * width..<(rowOffset + tokens.count) * width]),
                                                          "\(name), combined, individual token rows \(index)")
                    rowOffset += tokens.count
                }
            }
        }
    }
}
