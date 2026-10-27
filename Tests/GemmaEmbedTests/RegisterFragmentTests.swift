// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Metal
import XCTest
@testable import GemmaEmbed

final class RegisterFragmentTests: XCTestCase {
    func testRegisterFragmentOwnershipMatchesMetalLoadAndStore() throws {
        let context = try MetalContext()
        let source = """
        #include <metal_stdlib>
        #include <metal_simdgroup_matrix>
        using namespace metal;
        kernel void fragment_layout(device const float* input [[buffer(0)]],
                                    device float* loaded [[buffer(1)]],
                                    device float* stored [[buffer(2)]],
                                    uint lane [[thread_index_in_simdgroup]]) {
            const uint fm = ((lane / 4) & 4) + ((lane / 2) % 4);
            const uint fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
            simdgroup_float8x8 a;
            simdgroup_load(a, input, 8);
            loaded[lane * 2] = a.thread_elements()[0];
            loaded[lane * 2 + 1] = a.thread_elements()[1];
            simdgroup_float8x8 b;
            b.thread_elements()[0] = float(fm * 8 + fn);
            b.thread_elements()[1] = float(fm * 8 + fn + 1);
            simdgroup_store(b, stored, 8);
        }
        """
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        let library = try context.device.makeLibrary(source: source, options: options)
        let pipeline = try context.device.makeComputePipelineState(function: XCTUnwrap(library.makeFunction(name: "fragment_layout")))
        XCTAssertEqual(pipeline.threadExecutionWidth, 32)
        let identity = (0..<64).map { Float($0) }
        let input = try context.makeBuffer(array: identity, label: "fragment input")
        let loaded = try context.makeBuffer(array: [Float](repeating: -1, count: 64), label: "fragment elements")
        let stored = try context.makeBuffer(array: [Float](repeating: -1, count: 64), label: "fragment output")
        let cb = try XCTUnwrap(context.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        enc.setComputePipelineState(pipeline)
        for (index, buffer) in [input, loaded, stored].enumerated() { enc.setBuffer(buffer, offset: 0, index: index) }
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
        let ownership = (0..<32).flatMap { lane -> [Float] in
            let row = ((lane / 4) & 4) + ((lane / 2) % 4)
            let column = ((lane / 4) & 2) * 2 + (lane % 2) * 2
            return [Float(row * 8 + column), Float(row * 8 + column + 1)]
        }
        XCTAssertEqual(ownership.sorted(), identity, "every cell has exactly one owner")
        XCTAssertEqual(loaded.array(of: Float.self, count: 64), ownership, "Metal's fragment load must match the register mapping")
        XCTAssertEqual(stored.array(of: Float.self, count: 64), identity, "Metal's fragment store must match the register mapping")
    }
}
