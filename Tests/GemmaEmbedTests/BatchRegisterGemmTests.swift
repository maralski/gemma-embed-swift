// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Metal
import XCTest
@testable import GemmaEmbed

final class BatchRegisterGemmPolicyTests: XCTestCase {
    func testBatchRegisterGemmAutomaticPolicyHonorsCustomSchedulesAndFixedOverrides() {
        let custom = GemmTile(rows: 32, columns: 64, depth: 16, simdgroups: 4)
        let boundaries: [(Int, BatchRegisterGemmTile?)] = [
            (63, nil), (64, .rows32Columns32), (512, .rows32Columns32),
            (513, .rows32Columns64), (2048, .rows32Columns64), (2049, .rows32Columns128),
            (6000, .rows32Columns128), (8192, .rows32Columns128),
        ]
        for (rows, expected) in boundaries {
            for queries in [16, 32, 64] {
                let schedule = KernelSchedule(attentionQueries: queries)
                XCTAssertEqual(BatchRegisterGemmPolicy.automatic.tile(rows: rows, scheduledTile: schedule.gemmTile), expected)
            }
            XCTAssertNil(BatchRegisterGemmPolicy.automatic.tile(rows: rows, scheduledTile: custom))
            XCTAssertNil(BatchRegisterGemmPolicy.staged.tile(rows: rows, scheduledTile: custom))
        }
        XCTAssertEqual(BatchRegisterGemmPolicy.fixed(.rows32Columns128).tile(rows: 6000, scheduledTile: custom), .rows32Columns128)
        XCTAssertEqual(BatchRegisterGemmPolicy.fixed(.rows32Columns64).tile(rows: 6000, scheduledTile: .default), .rows32Columns64)
    }
}

extension KernelTests {
    func testBatchRegisterGemmCustomScheduleDispatchMatchesExplicitStagedBits() throws {
        let env = try XCTUnwrap(Self.env)
        let forward = env.forward
        let savedPolicy = forward.batchRegisterGemmPolicy
        let savedSchedule = forward.schedule
        defer {
            forward.batchRegisterGemmPolicy = savedPolicy
            forward.schedule = savedSchedule
        }
        let custom = GemmTile(rows: 32, columns: 64, depth: 16, simdgroups: 4)
        forward.schedule = KernelSchedule(gemmTile: custom)
        forward.batchRegisterGemmPolicy = .automatic
        let weight = env.weights.layers[0].wq
        let rows = MetalForward.paddedRows(6000)
        let count = (rows + 2) * weight.rows
        var rng = SeededRandom(seed: 947)
        let x = try buffer(rng.floats(rows * weight.columns, scale: 2))
        let actual = try buffer([Float](repeating: -777, count: count))
        let expected = try buffer([Float](repeating: -777, count: count))
        try run { general in
            let text = Kernels(forward: forward, encoder: general.encoder, domain: .pureText)
            XCTAssertNil(text.batchRegisterTile(weight: weight, rows: rows, outputs: weight.rows, inputs: weight.columns,
                                                requestedTile: nil, epilogue: .store))
            XCTAssertEqual(forward.schedule.gemmTile(rows: rows, outputs: weight.rows, inputs: weight.columns), custom)
            try text.gemm(x, weight: weight, out: actual, outRowOffset: 1,
                          rows: rows, outputs: weight.rows, inputs: weight.columns)
            try general.gemm(x, weight: weight, out: expected, outRowOffset: 1,
                             rows: rows, outputs: weight.rows, inputs: weight.columns, tile: custom)
        }
        XCTAssertEqual(actual.array(of: UInt32.self, count: count), expected.array(of: UInt32.self, count: count))
        XCTAssertEqual(actual.array(of: Float.self, count: weight.rows), [Float](repeating: -777, count: weight.rows))
        XCTAssertEqual(actual.array(of: Float.self, count: weight.rows, offset: (rows + 1) * weight.rows * 4),
                       [Float](repeating: -777, count: weight.rows))
    }

    private func batchRegisterShapes() throws -> [(weight: WeightTensor, offset: Int, outputs: Int)] {
        let env = try XCTUnwrap(Self.env)
        let l0 = env.weights.layers[0], l5 = env.weights.layers[5]
        let candidates: [(WeightTensor, Int)] = [
            (l0.wq, 0), (l0.wk, 0), (l0.wv, 0), (l0.wo, 0), (l0.wGate, 0), (l0.wUp, 0),
            (l0.wDown, 0), (l0.inpGate, 0), (l0.proj, 0),
            (l5.wq, 0), (l5.wk, 0), (l5.wo, 0), (env.weights.output, 0),
            (env.weights.perLayerProj, 7 * env.config.info.perLayerInputSize),
        ]
        var seen = Set<String>()
        return candidates.compactMap { weight, offset in
            let outputs = offset == 0 ? weight.rows : env.config.info.perLayerInputSize
            let key = "\(weight.type)-\(outputs)-\(weight.columns)-\(offset)"
            return seen.insert(key).inserted ? (weight, offset, outputs) : nil
        }
    }

    func testBatchRegisterGemmAllTextShapesAndRowBoundariesExactly() throws {
        let env = try XCTUnwrap(Self.env)
        let saved = env.forward.batchRegisterGemmPolicy
        defer { env.forward.batchRegisterGemmPolicy = saved }
        var rng = SeededRandom(seed: 913)
        for variant in BatchRegisterGemmTile.allCases {
            env.forward.batchRegisterGemmPolicy = .fixed(variant)
            for (weight, offset, outputs) in try batchRegisterShapes() {
                for rows in [63, 64, 65, 511, 512, 513, 2048, 2049, 8192] {
                    let padded = MetalForward.paddedRows(rows)
                    let x = try buffer(rng.floats(padded * weight.columns, scale: 2))
                    try assertBatchRegisterGemm(x, weight: weight, weightRowOffset: offset, rows: rows,
                                                outputs: outputs, variant: variant, epilogue: .store)
                }
            }
        }
    }

    func testBatchRegisterGemmAutomaticRowBoundariesExactly() throws {
        let env = try XCTUnwrap(Self.env)
        let saved = env.forward.batchRegisterGemmPolicy
        defer { env.forward.batchRegisterGemmPolicy = saved }
        env.forward.batchRegisterGemmPolicy = .automatic
        let boundaries: [(Int, BatchRegisterGemmTile?)] = [
            (63, nil), (64, .rows32Columns32), (511, .rows32Columns32), (512, .rows32Columns32),
            (513, .rows32Columns64), (2048, .rows32Columns64),
            (2049, .rows32Columns128), (8192, .rows32Columns128),
        ]
        var rng = SeededRandom(seed: 915)
        for (weight, offset, outputs) in try batchRegisterShapes() {
            for (rows, expectedTile) in boundaries {
                let x = try buffer(rng.floats(MetalForward.paddedRows(rows) * weight.columns, scale: 2))
                try assertBatchRegisterGemm(x, weight: weight, weightRowOffset: offset, rows: rows,
                                           outputs: outputs, variant: expectedTile, epilogue: .store)
            }
        }
    }

    func testBatchRegisterGemmFloat4EpiloguesAndOffsetsExactly() throws {
        let env = try XCTUnwrap(Self.env)
        let saved = env.forward.batchRegisterGemmPolicy
        defer { env.forward.batchRegisterGemmPolicy = saved }
        var rng = SeededRandom(seed: 914)
        for variant in BatchRegisterGemmTile.allCases {
            env.forward.batchRegisterGemmPolicy = .fixed(variant)
            for (weight, offset, outputs) in try batchRegisterShapes() where weight.type == .q8_0 {
                let rows = 65
                let padded = MetalForward.paddedRows(rows)
                let x = try buffer(rng.floats(padded * weight.columns, scale: 2))
                let aux = try buffer(rng.floats(padded * outputs, scale: 4))
                for epilogue in [Kernels.Epilogue.geglu(gate: aux), .geluTimes(factor: aux)] {
                    try assertBatchRegisterGemm(x, weight: weight, weightRowOffset: offset, rows: rows,
                                                outputs: outputs, variant: variant, epilogue: epilogue)
                }
            }
        }
    }

    private func assertBatchRegisterGemm(_ x: MTLBuffer, weight: WeightTensor, weightRowOffset: Int,
                                         rows: Int, outputs: Int, variant: BatchRegisterGemmTile?,
                                         epilogue: Kernels.Epilogue, file: StaticString = #filePath, line: UInt = #line) throws {
        let env = try XCTUnwrap(Self.env)
        let expectedSelection = weight.type == .q8_0 && rows >= 64 ? variant : nil
        let tile = expectedSelection?.tile ?? env.forward.schedule.gemmTile(rows: rows, outputs: outputs, inputs: weight.columns)
        let padded = (rows + tile.rows - 1) / tile.rows * tile.rows
        let referenceTile = GemmTile.compiled.contains(tile) ? tile
            : GemmTile(rows: 32, columns: 64, depth: 16, simdgroups: 4)
        XCTAssertEqual(padded % referenceTile.rows, 0, file: file, line: line)
        let count = (padded + 2) * outputs
        let candidate = try buffer([Float](repeating: -777, count: count))
        let staged = try buffer([Float](repeating: -777, count: count))
        try run { general in
            let text = Kernels(forward: env.forward, encoder: general.encoder, domain: .pureText)
            XCTAssertEqual(text.batchRegisterTile(weight: weight, rows: rows, outputs: outputs, inputs: weight.columns,
                                                  requestedTile: nil, epilogue: epilogue), expectedSelection,
                           "kernel parity must execute the expected register tile", file: file, line: line)
            try text.gemm(x, weight: weight, weightRowOffset: weightRowOffset, out: candidate, outRowOffset: 1,
                          rows: rows, outputs: outputs, inputs: weight.columns, epilogue: epilogue)
            try general.gemm(x, weight: weight, weightRowOffset: weightRowOffset, out: staged, outRowOffset: 1,
                             rows: padded, outputs: outputs, inputs: weight.columns, tile: referenceTile, epilogue: epilogue)
        }
        let actual = candidate.array(of: UInt32.self, count: count)
        let expected = staged.array(of: UInt32.self, count: count)
        let firstDifference = actual.indices.first { actual[$0] != expected[$0] }
        XCTAssertNil(firstDifference, "\(weight.info.name), rows \(rows), tile \(tile.suffix), \(epilogue)", file: file, line: line)
        XCTAssertEqual(candidate.array(of: Float.self, count: outputs), [Float](repeating: -777, count: outputs), file: file, line: line)
        XCTAssertEqual(candidate.array(of: Float.self, count: outputs, offset: (padded + 1) * outputs * 4),
                       [Float](repeating: -777, count: outputs), file: file, line: line)
    }

    func testBatchRegisterGemmRequiresExplicitPureTextEligibility() throws {
        let env = try XCTUnwrap(Self.env)
        let weight = env.weights.layers[0].wq
        let saved = env.forward.batchRegisterGemmPolicy
        defer { env.forward.batchRegisterGemmPolicy = saved }
        env.forward.batchRegisterGemmPolicy = .automatic
        let x = try buffer([Float](repeating: 0.25, count: 64 * weight.columns))
        let candidate = try buffer([Float](repeating: -777, count: 64 * weight.rows))
        let staged = try buffer([Float](repeating: -777, count: 64 * weight.rows))
        try run { general in
            let text = Kernels(forward: env.forward, encoder: general.encoder, domain: .pureText)
            func selection(_ k: Kernels, rows: Int = 64, requested: GemmTile? = nil,
                           epilogue: Kernels.Epilogue = .store) -> BatchRegisterGemmTile? {
                k.batchRegisterTile(weight: weight, rows: rows, outputs: weight.rows, inputs: weight.columns,
                                    requestedTile: requested, epilogue: epilogue)
            }
            XCTAssertEqual(selection(text), .rows32Columns32)
            XCTAssertNil(selection(general), "shared text weights alone do not make a tower or multimodal pass eligible")
            let otherWeights = try buffer([0])
            let other = Kernels(forward: env.forward, encoder: general.encoder, weights: otherWeights, domain: .pureText)
            XCTAssertNil(selection(other))
            for rows in [0, 1, 16, 17, 63] { XCTAssertNil(selection(text, rows: rows)) }
            for requested in GemmTile.compiled { XCTAssertNil(selection(text, requested: requested)) }
            for type in [GGUFTensorType.bf16, .f32] {
                let info = GGUFTensorInfo(name: weight.info.name, shape: weight.info.shape, type: type, offset: weight.info.offset)
                let nonQ8 = WeightTensor(info: info, base: weight.base, fileOffset: weight.fileOffset)
                XCTAssertNil(text.batchRegisterTile(weight: nonQ8, rows: 64, outputs: nonQ8.rows, inputs: nonQ8.columns,
                                                    requestedTile: nil, epilogue: .store))
            }
            XCTAssertNil(text.batchRegisterTile(weight: weight, rows: 64, outputs: 31, inputs: weight.columns,
                                                requestedTile: nil, epilogue: .store))
            XCTAssertNil(text.batchRegisterTile(weight: weight, rows: 64, outputs: weight.rows, inputs: 31,
                                                requestedTile: nil, epilogue: .store))
            XCTAssertNil(selection(text, epilogue: .gegluQuick(gate: candidate)))
            try text.gemm(x, weight: weight, out: candidate, rows: 64, outputs: weight.rows, inputs: weight.columns)
            try general.gemm(x, weight: weight, out: staged, rows: 64, outputs: weight.rows, inputs: weight.columns)
            env.forward.batchRegisterGemmPolicy = .staged
            XCTAssertNil(selection(text))
        }
        XCTAssertEqual(candidate.array(of: UInt32.self, count: 64 * weight.rows),
                       staged.array(of: UInt32.self, count: 64 * weight.rows))
    }
}

extension MetalForwardTests {
    func testBatchRegisterGemmPackedBatchInvarianceAndStagedParity() throws {
        let env = try XCTUnwrap(Self.env)
        let forward = env.forward
        let saved = forward.batchRegisterGemmPolicy
        defer { forward.batchRegisterGemmPolicy = saved }
        let fixtures = try TestSupport.tokenFixtures()
        let body = fixtures.flatMap { $0.tokens.dropFirst().dropLast() }
        let subject = Array(fixtures[35].tokens.prefix(64))
        var alone = PackedBatch()
        alone.append(subject)
        forward.batchRegisterGemmPolicy = .staged
        let single = try forward.run(alone)[0]
        for count in [64, 65, 511, 512, 513, 2048, 2049, 8192] {
            var batch = PackedBatch()
            var remaining = count - subject.count
            while remaining > 0 {
                let length = min(remaining, 127)
                batch.append(Array(body.prefix(length)))
                remaining -= length
            }
            batch.append(subject)
            XCTAssertEqual(batch.tokenCount, count)
            forward.batchRegisterGemmPolicy = .staged
            let expected = try forward.run(batch)
            let width = forward.config.info.outputDimensions
            let expectedRows = try forward.workspace(forRows: count).out.array(of: UInt32.self, count: count * width)
            forward.batchRegisterGemmPolicy = .automatic
            let actual = try forward.run(batch)
            let actualRows = try forward.workspace(forRows: count).out.array(of: UInt32.self, count: count * width)
            XCTAssertEqual(actual.map { $0.map(\.bitPattern) }, expected.map { $0.map(\.bitPattern) },
                           "automatic packed batch with \(count) rows")
            XCTAssertEqual(actualRows, expectedRows, "automatic raw token outputs for \(count) rows")
            XCTAssertEqual(actual.map { Self.normalised($0).map(\.bitPattern) },
                           expected.map { Self.normalised($0).map(\.bitPattern) },
                           "automatic normalised pooled outputs for \(count) rows")
            XCTAssertEqual(actual.last!.map(\.bitPattern), single.map(\.bitPattern),
                           "subject output after \(count - subject.count) unrelated packed tokens")
        }
    }

    func testBatchRegisterGemmAutomatic8192PackMatchesIndividual1024AndMixedLengthsExactly() throws {
        let env = try XCTUnwrap(Self.env)
        let forward = env.forward
        let saved = forward.batchRegisterGemmPolicy
        defer { forward.batchRegisterGemmPolicy = saved }
        forward.batchRegisterGemmPolicy = .automatic
        let body = try TestSupport.tokenFixtures().flatMap { $0.tokens.dropFirst().dropLast() }
        let width = forward.config.info.outputDimensions
        for lengths in [[Int](repeating: 1024, count: 8), [13, 64, 512, 1024, 2048, 4096, 435]] {
            var sequences: [[Int32]] = []
            for (index, length) in lengths.enumerated() {
                var tokens: [Int32] = []
                for position in 0..<length {
                    tokens.append(body[(index * 127 + position) % body.count])
                }
                sequences.append(tokens)
            }
            var batch = PackedBatch()
            for tokens in sequences { batch.append(tokens) }
            XCTAssertEqual(batch.tokenCount, 8192)
            let packed = try forward.run(batch)
            let packedRows = try forward.workspace(forRows: batch.tokenCount).out.array(of: UInt32.self, count: batch.tokenCount * width)
            var rowOffset = 0
            for (index, tokens) in sequences.enumerated() {
                var alone = PackedBatch()
                alone.append(tokens)
                let individual = try forward.run(alone)[0]
                let individualRows = try forward.workspace(forRows: tokens.count).out.array(of: UInt32.self, count: tokens.count * width)
                XCTAssertEqual(individual.map(\.bitPattern), packed[index].map(\.bitPattern),
                               "pooled output of \(tokens.count) tokens in an 8192-token pack")
                XCTAssertEqual(individualRows, Array(packedRows[rowOffset * width..<(rowOffset + tokens.count) * width]),
                               "raw token outputs of \(tokens.count) tokens in an 8192-token pack")
                rowOffset += tokens.count
            }
        }
    }
}
