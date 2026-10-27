// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import CryptoKit
import Foundation
import Metal

/// Owns the Metal device, the compiled shader library and every pipeline state.
///
/// The shaders ship as `.metal` sources and compile at load time, because SwiftPM does not
/// build a metallib from the command line. Compile errors therefore surface here, as
/// `GemmaEmbedError.shaderCompilation`.
final class MetalContext: @unchecked Sendable {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary
    /// SHA256 of the concatenated shader source, part of the autotune cache key.
    let sourceDigest: String
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private let lock = NSLock()

    static let shaderFiles = ["Common.metal", "Elementwise.metal", "GEMM.metal", "Attention.metal", "Audio.metal"]

    init(device: MTLDevice? = nil) throws {
        guard let device = device ?? MTLCreateSystemDefaultDevice() else { throw GemmaEmbedError.noMetalDevice }
        self.device = device
        guard let queue = device.makeCommandQueue() else { throw GemmaEmbedError.noMetalDevice }
        self.queue = queue
        let source = MetalContext.shaderSource()
        self.sourceDigest = SHA256.hash(data: Data(source.utf8)).map { String(format: "%02x", $0) }.joined()
        let options = MTLCompileOptions()
        // Safe math keeps the -inf masking and exp semantics the attention kernel relies on.
        if #available(macOS 15.0, *) {
            options.mathMode = .safe
        } else {
            options.setValue(false, forKey: "fastMathEnabled")
        }
        options.languageVersion = .version3_1
        do {
            self.library = try device.makeLibrary(source: source, options: options)
        } catch {
            throw GemmaEmbedError.shaderCompilation(String(describing: error))
        }
        try validateFragmentLayout()
    }

    private func validateFragmentLayout() throws {
        let probe = try pipeline("validate_float8x8_fragment_layout")
        guard probe.threadExecutionWidth == 32 else {
            throw GemmaEmbedError.gpuFailure("unsupported SIMD layout: float8x8 fragments require a 32-thread SIMD group, got width \(probe.threadExecutionWidth)")
        }
        let expected: [Float] = [
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
            32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63,
        ]
        let output = try makeBuffer(array: [Float](repeating: -1, count: expected.count), label: "fragment layout validation")
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw GemmaEmbedError.gpuFailure("cannot encode float8x8 fragment layout validation")
        }
        enc.setComputePipelineState(probe)
        enc.setBuffer(output, offset: 0, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error {
            throw GemmaEmbedError.gpuFailure("float8x8 fragment layout validation failed: \(error)")
        }
        guard cb.status == .completed else {
            throw GemmaEmbedError.gpuFailure("float8x8 fragment layout validation did not complete")
        }
        let actual = output.array(of: Float.self, count: expected.count)
        for i in expected.indices where actual[i] != expected[i] {
            throw GemmaEmbedError.gpuFailure("unsupported float8x8 fragment ownership: index \(i) expected \(expected[i]), got \(actual[i])")
        }
    }

    static let resourceBundleName = "GemmaEmbed_GemmaEmbed.bundle"

    /// Reads the shaders from the first resource bundle copy found, else uses the copy compiled in.
    static func shaderSource() -> String {
        (try? shaderSource(inBundles: shaderBundleCandidates())) ?? concatenated { EmbeddedShaders.sources[$0] ?? "" }
    }

    /// Where a copy of the resource bundle can live, in search order. SwiftPM's `Bundle.module` is
    /// not consulted: it looks at the root of an .app, where codesign refuses loose files, then at
    /// the absolute `.build` path of the checkout, and it traps instead of throwing.
    static func shaderBundleCandidates(main: Bundle = .main, library: Bundle = Bundle(for: MetalContext.self)) -> [URL] {
        let urls = [
            main.resourceURL,
            library.resourceURL,
            library.bundleURL.deletingLastPathComponent(),
            main.bundleURL,
        ].compactMap { $0?.appendingPathComponent(resourceBundleName).standardizedFileURL }
        var seen = Set<URL>()
        return urls.filter { seen.insert($0).inserted }
    }

    static func shaderSource(inBundles candidates: [URL]) throws -> String {
        for bundle in candidates {
            let dir = bundle.appendingPathComponent("Shaders")
            if let source = try? concatenated({ try String(contentsOf: dir.appendingPathComponent($0), encoding: .utf8) }) {
                return source
            }
        }
        throw GemmaEmbedError.shaderSourceMissing(tried: candidates.map(\.path))
    }

    private static func concatenated(_ text: (String) throws -> String) rethrows -> String {
        try shaderFiles.map { try text($0) + "\n" }.joined()
    }

    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        lock.lock()
        if let p = pipelines[name] { lock.unlock(); return p }
        lock.unlock()
        guard let function = library.makeFunction(name: name) else {
            throw GemmaEmbedError.shaderCompilation("missing kernel \(name)")
        }
        let p: MTLComputePipelineState
        do {
            p = try device.makeComputePipelineState(function: function)
        } catch {
            throw GemmaEmbedError.shaderCompilation("pipeline \(name): \(error)")
        }
        lock.lock()
        pipelines[name] = p
        lock.unlock()
        return p
    }

    /// Builds every pipeline the forward pass can use, in parallel.
    func warmUpPipelines() throws {
        let names = library.functionNames
        let failure = ErrorBox()
        DispatchQueue.concurrentPerform(iterations: names.count) { i in
            do {
                _ = try pipeline(names[i])
            } catch {
                failure.set(error)
            }
        }
        if let error = failure.get() { throw error }
    }

    func makeBuffer(bytes: Int, label: String) throws -> MTLBuffer {
        guard let b = device.makeBuffer(length: max(bytes, 16), options: .storageModeShared) else {
            throw GemmaEmbedError.gpuFailure("cannot allocate \(bytes) bytes for \(label)")
        }
        b.label = label
        memset(b.contents(), 0, b.length)
        return b
    }

    func makeBuffer<T>(array: [T], label: String) throws -> MTLBuffer {
        try array.withUnsafeBytes { try makeBuffer(copying: $0, label: label) }
    }

    func makeBuffer(copying bytes: UnsafeRawBufferPointer, label: String) throws -> MTLBuffer {
        let b = try makeBuffer(bytes: bytes.count, label: label)
        if let base = bytes.baseAddress { memcpy(b.contents(), base, bytes.count) }
        return b
    }
}

final class ErrorBox: @unchecked Sendable {
    private var error: Error?
    private let lock = NSLock()

    func set(_ e: Error) {
        lock.lock()
        if error == nil { error = e }
        lock.unlock()
    }

    func get() -> Error? {
        lock.lock()
        defer { lock.unlock() }
        return error
    }
}

extension MTLBuffer {
    /// Reads `count` elements of `T` starting at byte offset `offset`.
    func array<T>(of _: T.Type, count: Int, offset: Int = 0) -> [T] {
        let p = (contents() + offset).bindMemory(to: T.self, capacity: count)
        return Array(UnsafeBufferPointer(start: p, count: count))
    }

    /// The first `count` rows of a buffer of `width`-wide float rows.
    func rows(_ count: Int, width: Int) -> [[Float]] {
        let all = array(of: Float.self, count: count * width)
        return (0..<count).map { Array(all[($0 * width)..<(($0 + 1) * width)]) }
    }

    func write<T>(_ values: [T], offset: Int = 0) {
        values.withUnsafeBytes { write(bytes: $0, offset: offset) }
    }

    func write(bytes: UnsafeRawBufferPointer, offset: Int) {
        guard let base = bytes.baseAddress, bytes.count > 0 else { return }
        precondition(offset + bytes.count <= length)
        memcpy(contents() + offset, base, bytes.count)
    }
}
