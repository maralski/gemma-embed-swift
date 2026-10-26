// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import XCTest
@testable import GemmaEmbed

final class ShaderSourceTests: XCTestCase {
    static let shaderDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Sources/GemmaEmbed/Shaders")

    func temporaryDirectory() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("ShaderSourceTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        return dir
    }

    func testFindsShadersInAppContentsResources() throws {
        let app = try temporaryDirectory().appendingPathComponent("Foo.app")
        let contents = app.appendingPathComponent("Contents")
        let bundle = contents.appendingPathComponent("Resources/GemmaEmbed_GemmaEmbed.bundle")
        try FileManager.default.createDirectory(at: contents.appendingPathComponent("MacOS"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: Self.shaderDir, to: bundle.appendingPathComponent("Shaders"))
        let plist: [String: String] = ["CFBundleExecutable": "Foo", "CFBundleIdentifier": "test.foo", "CFBundlePackageType": "APPL"]
        try PropertyListSerialization.data(fromPropertyList: plist, format: .xml, options: 0)
            .write(to: contents.appendingPathComponent("Info.plist"))
        let marker = "// copy inside Foo.app\n"
        let common = bundle.appendingPathComponent("Shaders/Common.metal")
        try (marker + String(contentsOf: common, encoding: .utf8)).write(to: common, atomically: true, encoding: .utf8)

        let appBundle = try XCTUnwrap(Bundle(url: app))
        let candidates = MetalContext.shaderBundleCandidates(main: appBundle, library: appBundle)
        XCTAssertEqual(candidates.first?.path, bundle.standardizedFileURL.path)

        let source = try MetalContext.shaderSource(inBundles: candidates)
        XCTAssertTrue(source.hasPrefix(marker))
        let expected = try MetalContext.shaderFiles
            .map { try String(contentsOf: Self.shaderDir.appendingPathComponent($0), encoding: .utf8) + "\n" }
            .joined()
        XCTAssertEqual(source, marker + expected)
    }

    func testMissingShadersErrorNamesEveryPathTried() throws {
        let root = try temporaryDirectory()
        let candidates = ["A.app/Contents/Resources", "B"].map {
            root.appendingPathComponent($0).appendingPathComponent(MetalContext.resourceBundleName)
        }
        XCTAssertThrowsError(try MetalContext.shaderSource(inBundles: candidates)) { error in
            guard case GemmaEmbedError.shaderSourceMissing(let tried) = error else {
                return XCTFail("unexpected error \(error)")
            }
            XCTAssertEqual(tried, candidates.map(\.path))
            let message = String(describing: error)
            for path in tried { XCTAssertTrue(message.contains(path), message) }
        }
    }

    func testEmbeddedShadersMatchTheMetalFiles() throws {
        let onDisk = try FileManager.default.contentsOfDirectory(atPath: Self.shaderDir.path).filter { $0.hasSuffix(".metal") }
        XCTAssertEqual(Set(onDisk), Set(MetalContext.shaderFiles))
        XCTAssertEqual(Set(EmbeddedShaders.sources.keys), Set(MetalContext.shaderFiles))
        for name in onDisk {
            let file = try String(contentsOf: Self.shaderDir.appendingPathComponent(name), encoding: .utf8)
            XCTAssertEqual(EmbeddedShaders.sources[name], file, "\(name) is stale: run scripts/embed_shaders.py")
        }
    }
}
