// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import CoreGraphics
import Foundation

/// An image to embed.
public enum ImageInput: @unchecked Sendable {
    /// A file in any format ImageIO reads (JPEG, PNG, HEIC, TIFF, GIF and others).
    case url(URL)
    /// Encoded image bytes in any format ImageIO reads.
    case data(Data)
    /// An image already decoded; its samples are used in its own colour space.
    case cgImage(CGImage)
}

/// An 8-bit RGB raster, interleaved and row-major: `pixels[(y * width + x) * 3 + c]`.
struct RGBImage: Sendable, Equatable {
    let width: Int
    let height: Int
    var pixels: [UInt8]

    init(width: Int, height: Int, pixels: [UInt8]) {
        precondition(width > 0 && height > 0 && pixels.count == width * height * 3)
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    /// A black image.
    init(width: Int, height: Int) {
        self.init(width: width, height: height, pixels: [UInt8](repeating: 0, count: width * height * 3))
    }
}
