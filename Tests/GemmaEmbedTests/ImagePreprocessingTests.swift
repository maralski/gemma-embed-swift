// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import CoreGraphics
import Foundation
import XCTest
@testable import GemmaEmbed

final class ImagePreprocessingTests: XCTestCase {
    static let dumps = ["00-green-hills", "02-gradient-h", "04-desert-dunes", "06-lavender-field-cropped-10-top-left", "07-shapes", "09-tiny-noise"]

    nonisolated(unsafe) static var sharedPreprocessor: ImagePreprocessor?

    func preprocessor() throws -> ImagePreprocessor {
        if let p = Self.sharedPreprocessor { return p }
        let file = try GGUFFile(url: try TestSupport.requireImageEncoder())
        let p = ImagePreprocessor(config: try VisionConfig(gguf: file, fileSHA256: nil))
        Self.sharedPreprocessor = p
        return p
    }

    /// The bytes behind `inp_raw_scaled` (planar `[c][y][x]`, `2 * byte / 255 - 1`), interleaved again.
    static func bytes(ofScaled raw: [Float], width: Int, height: Int) -> RGBImage {
        var pixels = [UInt8](repeating: 0, count: width * height * 3)
        for c in 0..<3 {
            for i in 0..<width * height {
                pixels[i * 3 + c] = UInt8(((raw[c * width * height + i] + 1) * 255 / 2).rounded())
            }
        }
        return RGBImage(width: width, height: height, pixels: pixels)
    }

    /// PNGs (through ImageIO) and JPEGs (through the stb_image port) decode to stb_image's bytes exactly.
    func testDecodeMatchesStbImage() throws {
        for name in Self.dumps {
            let dump = try TestSupport.requireVisionDump(name)
            let want = try dump.decodedRGB()
            let got = try ImageDecoder.decode(.url(dump.imageURL))
            XCTAssertEqual(got.width, want.width, name)
            XCTAssertEqual(got.height, want.height, name)
            var maxDiff = 0, differing = 0
            for (a, b) in zip(got.pixels, want.pixels) {
                maxDiff = max(maxDiff, abs(Int(a) - Int(b)))
                if a != b { differing += 1 }
            }
            print("decode \(name): max byte diff \(maxDiff), differing bytes \(differing)")
            XCTAssertEqual(maxDiff, 0, "\(name): \(dump.imageURL.pathExtension) decode differs from stb_image")
        }
    }

    func testEveryInputFormDecodesAlike() throws {
        let url = try TestSupport.fixture("images/synthetic/09-tiny-noise.png")
        let fromURL = try ImageDecoder.decode(.url(url))
        XCTAssertEqual(try ImageDecoder.decode(.data(Data(contentsOf: url))), fromURL)
        XCTAssertEqual(try ImageDecoder.decode(.cgImage(ImageDecoder.cgImage(.url(url)))), fromURL)
        XCTAssertThrowsError(try ImageDecoder.decode(.data(Data("not an image".utf8)))) { error in
            guard case GemmaEmbedError.imageDecoding = error else { return XCTFail("unexpected error \(error)") }
        }
        XCTAssertThrowsError(try ImageDecoder.decode(.url(URL(fileURLWithPath: "/nonexistent/image.png"))))
    }

    func testGreyImagesReplicateTheirSamples() throws {
        let width = 7, height = 3
        let grey = (0..<width * height).map { UInt8($0 * 12) }
        let provider = CGDataProvider(data: Data(grey) as CFData)!
        let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: width,
                            space: CGColorSpace(name: CGColorSpace.genericGrayGamma2_2)!, bitmapInfo: CGBitmapInfo(rawValue: 0),
                            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        let got = try ImageDecoder.decode(.cgImage(image))
        XCTAssertEqual(got.pixels, grey.flatMap { [$0, $0, $0] })
    }

    /// Sizes llama.cpp chose for the dumped images: downscale, in range, upscale and a letterboxed aspect.
    func testTargetSizeBranches() {
        func target(_ w: Int, _ h: Int) -> [Int] {
            let t = ImageResizer.targetSize(width: w, height: h, alignment: 48, minPixels: 70 * 2304, maxPixels: 256 * 2304)
            return [t.width, t.height]
        }
        XCTAssertEqual(target(1600, 1600), [768, 768])
        XCTAssertEqual(target(1728, 1728), [768, 768])
        XCTAssertEqual(target(640, 640), [624, 624])
        XCTAssertEqual(target(800, 400), [816, 384])
        XCTAssertEqual(target(200, 150), [480, 384])
    }

    func testTargetSizeMatchesLlamaCpp() throws {
        let pre = try preprocessor()
        for name in Self.dumps {
            let dump = try TestSupport.requireVisionDump(name)
            let raw = try XCTUnwrap(dump.node("inp_raw_scaled"))
            let target = pre.targetSize(width: dump.decodedWidth, height: dump.decodedHeight, maxImageTokens: 256)
            XCTAssertEqual([target.width, target.height, 3], Array(raw.shape.prefix(3)), name)
            XCTAssertEqual((target.width / 48) * (target.height / 48), dump.imageTokenCount, name)
        }
    }

    /// The resize runs on stb_image's bytes so that the comparison isolates it from the decoder.
    func testResizeMatchesLlamaCppExactly() throws {
        let pre = try preprocessor()
        for name in Self.dumps {
            let dump = try TestSupport.requireVisionDump(name)
            let node = try XCTUnwrap(dump.node("inp_raw_scaled"))
            let raw = try dump.values("inp_raw_scaled")
            let (w, h) = (node.shape[0], node.shape[1])
            let want = Self.bytes(ofScaled: raw, width: w, height: h)
            let got = pre.resize(try dump.decodedRGB(), maxImageTokens: 256)
            XCTAssertEqual(got.width, w, name)
            XCTAssertEqual(got.height, h, name)
            var maxDiff = 0, differing = 0
            for (a, b) in zip(got.pixels, want.pixels) {
                maxDiff = max(maxDiff, abs(Int(a) - Int(b)))
                if a != b { differing += 1 }
            }
            print("resize \(name): \(dump.decodedWidth)x\(dump.decodedHeight) -> \(w)x\(h), max byte diff \(maxDiff), differing bytes \(differing)")
            XCTAssertEqual(maxDiff, 0, name)

            var scaledMismatches = 0
            for c in 0..<3 {
                for i in 0..<w * h where 2 * (Float(got.pixels[i * 3 + c]) / 255) - 1 != raw[c * w * h + i] { scaledMismatches += 1 }
            }
            XCTAssertEqual(scaledMismatches, 0, "\(name): normalised values differ from inp_raw_scaled")
        }
    }

    /// Patch `py * gx + px`, element `c * 256 + ky * 16 + kx`, is the fp16 rounding of the pixel in `inp_raw_scaled`.
    func testPatchLayoutAndFp16Rounding() throws {
        let pre = try preprocessor()
        for name in ["09-tiny-noise", "02-gradient-h"] {
            let dump = try TestSupport.requireVisionDump(name)
            let node = try XCTUnwrap(dump.node("inp_raw_scaled"))
            let raw = try dump.values("inp_raw_scaled")
            let (w, h) = (node.shape[0], node.shape[1])
            let image = pre.prepare(try dump.decodedRGB(), maxImageTokens: 256)
            XCTAssertEqual(image.gridX, w / 16)
            XCTAssertEqual(image.gridY, h / 16)
            XCTAssertEqual(image.patchCount, image.gridX * image.gridY)
            XCTAssertEqual(image.tokenCount, dump.imageTokenCount)
            XCTAssertEqual(image.patches.count, image.patchCount * 768)
            var mismatches = 0, rounded = 0
            for py in 0..<image.gridY {
                for px in 0..<image.gridX {
                    let i = py * image.gridX + px
                    XCTAssertEqual(image.patchX[i], Int32(px))
                    XCTAssertEqual(image.patchY[i], Int32(py))
                    for c in 0..<3 {
                        for ky in 0..<16 {
                            for kx in 0..<16 {
                                let v = raw[c * w * h + (py * 16 + ky) * w + px * 16 + kx]
                                let got = image.patches[i * 768 + c * 256 + ky * 16 + kx]
                                if got != Float(Float16(v)) { mismatches += 1 }
                                if got != v { rounded += 1 }
                            }
                        }
                    }
                }
            }
            print("patches \(name): \(image.patchCount) patches, \(rounded) values changed by the fp16 rounding")
            XCTAssertEqual(mismatches, 0, name)
            XCTAssertGreaterThan(rounded, 0, "\(name): expected some values to change under fp16 rounding")
        }
    }
}
