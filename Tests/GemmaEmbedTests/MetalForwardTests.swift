// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import XCTest
@testable import GemmaEmbed

final class MetalForwardTests: XCTestCase {
    final class Env {
        let forward: MetalForward
        let tokenizer: Tokenizer
        init(forward: MetalForward, tokenizer: Tokenizer) {
            self.forward = forward
            self.tokenizer = tokenizer
        }
    }

    nonisolated(unsafe) static var env: Env?

    override func setUpWithError() throws {
        if Self.env == nil {
            let file = try GGUFFile(url: try TestSupport.requireWeights())
            let context = try MetalContext()
            let config = try ModelConfig(gguf: file, deviceName: context.device.name, fileSHA256: nil)
            let weights = try ModelWeights(gguf: file, config: config)
            Self.env = Env(forward: try MetalForward(context: context, config: config, weights: weights), tokenizer: try Tokenizer(gguf: file))
        }
    }

    static func normalised(_ v: [Float]) -> [Float] {
        let n = v.reduce(0) { $0 + $1 * $1 }.squareRoot()
        return v.map { $0 / n }
    }

    func testAllFixturesMatchReferenceEmbeddings() throws {
        let env = try XCTUnwrap(Self.env)
        let tokens = try TestSupport.tokenFixtures()
        let embeddings = try TestSupport.embeddingFixtures()
        var batch = PackedBatch()
        for t in tokens { batch.append(t.tokens) }
        let outputs = try env.forward.run(batch)
        XCTAssertEqual(outputs.count, 40)
        var worstCos: Float = 1, worstDiff: Float = 0
        for (i, out) in outputs.enumerated() {
            let got = Self.normalised(out)
            let cos = TestSupport.cosine(got, embeddings[i].embedding)
            let diff = TestSupport.maxAbsDiff(got, embeddings[i].embedding)
            worstCos = min(worstCos, cos)
            worstDiff = max(worstDiff, diff)
            XCTAssertGreaterThan(cos, 0.9995, "text \(i) cosine \(cos)")
        }
        print("metal forward parity over 40 fixtures: min cosine \(worstCos), max abs diff \(worstDiff)")
    }

    func testBatchIndependence() throws {
        let env = try XCTUnwrap(Self.env)
        let tokens = try TestSupport.tokenFixtures()
        let subject = tokens[35].tokens
        var alone = PackedBatch()
        alone.append(subject)
        let single = try env.forward.run(alone)[0]
        var batch = PackedBatch()
        for i in 0..<32 { batch.append(tokens[(i * 7) % 40].tokens) }
        batch.append(subject)
        let inBatch = try env.forward.run(batch).last!
        let diff = TestSupport.maxAbsDiff(Self.normalised(single), Self.normalised(inBatch))
        print("batch independence max abs diff \(diff)")
        XCTAssertLessThan(diff, 1e-5)
        var reordered = PackedBatch()
        reordered.append(subject)
        for i in 0..<32 { reordered.append(tokens[(i * 3) % 40].tokens) }
        let first = try env.forward.run(reordered)[0]
        XCTAssertLessThan(TestSupport.maxAbsDiff(Self.normalised(single), Self.normalised(first)), 1e-5)
    }

    func testBatchIndependenceAcrossGemmTiles() throws {
        let env = try XCTUnwrap(Self.env)
        let tokens = try TestSupport.tokenFixtures()
        let subject = tokens[35].tokens
        var alone = PackedBatch()
        alone.append(subject)
        var batch = PackedBatch()
        var i = 0
        while batch.tokenCount < 8192 - subject.count {
            batch.append(tokens[(i * 7) % 40].tokens)
            i += 1
        }
        batch.append(subject)
        let schedule = env.forward.schedule
        let width = env.forward.config.info.hiddenSize
        XCTAssertNotEqual(schedule.gemmTile(rows: MetalForward.paddedRows(alone.tokenCount), outputs: width),
                          schedule.gemmTile(rows: MetalForward.paddedRows(batch.tokenCount), outputs: width))
        let single = try env.forward.run(alone)[0]
        let inBatch = try env.forward.run(batch).last!
        XCTAssertEqual(single.count, inBatch.count)
        let differing = zip(single, inBatch).filter { $0.bitPattern != $1.bitPattern }.count
        XCTAssertEqual(differing, 0, "a text's vector must not depend on the tile its batch size picks; max abs diff \(TestSupport.maxAbsDiff(single, inBatch))")
    }

    func testSmallTextBatchesMatchLargerBatchesAcrossTileBoundaries() throws {
        let env = try XCTUnwrap(Self.env)
        let body = try TestSupport.tokenFixtures().flatMap { $0.tokens.dropFirst().dropLast() }
        let savedSchedule = env.forward.schedule
        defer { env.forward.schedule = savedSchedule }
        for queries in KernelSchedule.attentionQueryCandidates {
            env.forward.schedule.attentionQueries = queries
            for count in [2, 13, 15, 16, 17, 31, 32, 33, 63, 64, 65] {
                let subject = [env.tokenizer.bosId] + Array(body.prefix(count - 2)) + [env.tokenizer.eosId]
                var packed = PackedBatch()
                packed.append(Array(body.prefix(19)))
                packed.append(subject)
                packed.append(Array(body.prefix(97)))
                let inBatch = try env.forward.run(packed)[1]
                var alone = PackedBatch()
                alone.append(subject)
                let single = try env.forward.run(alone)[0]
                XCTAssertEqual(single.map(\.bitPattern), inBatch.map(\.bitPattern),
                               "\(count) tokens and \(queries) attention queries must match after reusing a larger workspace")
            }
        }
    }

    func testRegisterQ8GemmSmallPackedBatchesMatchFallback() throws {
        let env = try XCTUnwrap(Self.env)
        let body = try TestSupport.tokenFixtures().flatMap { $0.tokens.dropFirst().dropLast() }
        for lengths in [[5, 8], [8, 8], [8, 9]] {
            var small = PackedBatch()
            for count in lengths {
                small.append([env.tokenizer.bosId] + Array(body.prefix(count - 2)) + [env.tokenizer.eosId])
            }
            var large = small
            large.append(Array(body.prefix(97)))
            let expected = try env.forward.run(large)
            let actual = try env.forward.run(small)
            for index in lengths.indices {
                XCTAssertEqual(actual[index].map(\.bitPattern), expected[index].map(\.bitPattern),
                               "packed lengths \(lengths), sequence \(index)")
            }
        }
    }

    func testFullBatchesAndAutotuneUseTheScheduleGemmTile() {
        let schedule = KernelSchedule(gemmTile: GemmTile(rows: 64, columns: 128, depth: 16, simdgroups: 8))
        for rows in [MetalForward.paddedRows(6000), 8192] {
            for outputs in [512, 768, 1024, 2048] {
                XCTAssertEqual(schedule.gemmTile(rows: rows, outputs: outputs), schedule.gemmTile, "rows \(rows) outputs \(outputs)")
            }
        }
    }

    func testMatchesCPUReferenceOnLongInput() throws {
        let env = try XCTUnwrap(Self.env)
        let tokens = try TestSupport.tokenFixtures()
        var ids: [Int32] = [env.tokenizer.bosId]
        while ids.count < 1500 {
            for t in tokens where ids.count < 1500 { ids.append(contentsOf: t.tokens.dropFirst().dropLast()) }
        }
        ids.append(env.tokenizer.eosId)
        var batch = PackedBatch()
        batch.append(ids)
        let got = Self.normalised(try env.forward.run(batch)[0])
        let want = Self.normalised(CPUReference(config: env.forward.config, weights: env.forward.weights).forward(tokens: ids))
        let cos = TestSupport.cosine(got, want)
        let diff = TestSupport.maxAbsDiff(got, want)
        print("long input (\(ids.count) tokens, sliding windows active) vs CPU reference: cosine \(cos) max abs diff \(diff)")
        XCTAssertGreaterThan(cos, 0.99999)
        XCTAssertLessThan(diff, 1e-4)
    }
}

extension MetalForwardTests {
    /// Texts of 1054, 1629 and 3222 tokens exercise the symmetric sliding window against llama.cpp.
    func testLongTextsMatchLlamaCppBeyondTheWindow() throws {
        let env = try XCTUnwrap(Self.env)
        let rows = try TestSupport.fixtureJSONLines("long_embeddings.jsonl")
        XCTAssertEqual(rows.count, 3)
        var batch = PackedBatch()
        for row in rows {
            let ids = env.tokenizer.encode((row["prefix"] as! String) + (row["text"] as! String))
            XCTAssertEqual(ids.count, row["prompt_tokens"] as! Int)
            batch.append(ids)
        }
        let outputs = try env.forward.run(batch)
        for (row, out) in zip(rows, outputs) {
            let want = (row["embedding"] as! [Double]).map(Float.init)
            let cos = TestSupport.cosine(Self.normalised(out), want)
            print("long text \(row["prompt_tokens"]!) tokens vs llama.cpp: cosine \(cos)")
            XCTAssertGreaterThan(cos, 0.9995)
        }
    }
}

extension MetalForwardTests {
    /// The profiler times attention on the q, k and v the forward produces, not on what earlier timings left in the workspace.
    func testProfilerAttentionInputsMatchCPUReference() throws {
        let env = try XCTUnwrap(Self.env)
        let tokens = try TestSupport.tokenFixtures()
        let subject = tokens[35].tokens
        var batch = PackedBatch()
        batch.append(tokens[3].tokens)
        batch.append(subject)
        let start = Int(batch.sequences[1].start)
        var traced: [String: [Float]] = [:]
        _ = CPUReference(config: env.forward.config, weights: env.forward.weights).forward(tokens: subject) { traced[$0] = $1 }

        let staged = try Profiler.stageAttention(forward: env.forward, batch: batch)
        XCTAssertEqual(staged.map(\.name), ["attention sliding hd256", "attention global hd512"])
        let heads = env.forward.config.info.headCount
        for s in staged {
            let l = s.layer.index
            for (name, buffer, rowWidth, cpuName) in [("q", s.q, heads * s.layer.headDim, "Qcur_pos-\(l)"),
                                                      ("k", s.k, s.layer.kvHeadCount * s.layer.headDim, "Kcur_pos-\(l)"),
                                                      ("v", s.v, s.layer.kvHeadCount * s.layer.headDim, "Vcur_normed-\(l)")] {
                let want = try XCTUnwrap(traced[cpuName])
                XCTAssertEqual(want.count, subject.count * rowWidth)
                let got = buffer.array(of: Float.self, count: subject.count * rowWidth, offset: start * rowWidth * 4)
                let diff = TestSupport.maxAbsDiff(got, want)
                print("\(s.name) layer \(l) \(name) vs CPU reference: max abs diff \(diff)")
                XCTAssertLessThan(diff, 1e-3, "\(s.name) \(name)")
            }
        }
    }
}
