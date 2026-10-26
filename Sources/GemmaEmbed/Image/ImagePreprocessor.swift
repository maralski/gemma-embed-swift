// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// An image resized, normalised and cut into patches: the input of the vision tower.
struct PreparedImage: Sendable {
    /// Size of the resized image in pixels.
    let width: Int
    let height: Int
    /// Patch grid: `width / patchSize` columns by `height / patchSize` rows.
    let gridX: Int
    let gridY: Int
    /// Side of the square window of patches pooled into one token.
    let mergeSize: Int
    /// `patchCount` rows of `patchArea` values, row `py * gridX + px`, element `c * p^2 + ky * p + kx`.
    /// Each value is `2 * byte / 255 - 1` rounded to fp16, because ggml runs the patch im2col in fp16.
    let patches: [Float]
    /// Column and row of every patch, for the position tables and the 2D RoPE.
    let patchX: [Int32]
    let patchY: [Int32]

    var patchCount: Int { gridX * gridY }
    var tokenGridX: Int { gridX / mergeSize }
    var tokenGridY: Int { gridY / mergeSize }
    /// Vectors the image contributes to the text sequence, token `oy * tokenGridX + ox`.
    var tokenCount: Int { tokenGridX * tokenGridY }

    /// Patchifies an already resized image whose sides are multiples of `patchSize * mergeSize`.
    init(resized image: RGBImage, patchSize p: Int, mergeSize: Int, mean: [Float], std: [Float]) {
        precondition(image.width % (p * mergeSize) == 0 && image.height % (p * mergeSize) == 0)
        width = image.width
        height = image.height
        gridX = width / p
        gridY = height / p
        self.mergeSize = mergeSize
        let area = p * p * 3
        let count = gridX * gridY
        var patches = [Float](repeating: 0, count: count * area)
        var patchX = [Int32](repeating: 0, count: count)
        var patchY = [Int32](repeating: 0, count: count)
        for py in 0..<gridY {
            for px in 0..<gridX {
                let i = py * gridX + px
                patchX[i] = Int32(px)
                patchY[i] = Int32(py)
                for c in 0..<3 {
                    for ky in 0..<p {
                        for kx in 0..<p {
                            let byte = image.pixels[((py * p + ky) * width + px * p + kx) * 3 + c]
                            let f = (Float(byte) / 255 - mean[c]) / std[c]
                            patches[i * area + c * p * p + ky * p + kx] = Float(Float16(2 * f - 1))
                        }
                    }
                }
            }
        }
        self.patches = patches
        self.patchX = patchX
        self.patchY = patchY
    }
}

/// Turns any `ImageInput` into a `PreparedImage` exactly as llama.cpp's mtmd does for gemma4v.
struct ImagePreprocessor: Sendable {
    let patchSize: Int
    let mergeSize: Int
    let minImageTokens: Int
    let mean: [Float]
    let std: [Float]

    init(config: VisionConfig) {
        patchSize = config.info.patchSize
        mergeSize = config.mergeSize
        minImageTokens = config.minImageTokens
        mean = config.imageMean
        std = config.imageStd
    }

    var alignment: Int { patchSize * mergeSize }

    func prepare(_ input: ImageInput, maxImageTokens: Int) throws -> PreparedImage {
        prepare(try ImageDecoder.decode(input), maxImageTokens: maxImageTokens)
    }

    func prepare(_ image: RGBImage, maxImageTokens: Int) -> PreparedImage {
        let resized = resize(image, maxImageTokens: maxImageTokens)
        return PreparedImage(resized: resized, patchSize: patchSize, mergeSize: mergeSize, mean: mean, std: std)
    }

    /// The decoded image resized to the size the vision tower sees.
    func resize(_ image: RGBImage, maxImageTokens: Int) -> RGBImage {
        let target = targetSize(width: image.width, height: image.height, maxImageTokens: maxImageTokens)
        return ImageResizer.fit(image, width: target.width, height: target.height)
    }

    func targetSize(width: Int, height: Int, maxImageTokens: Int) -> (width: Int, height: Int) {
        let pixelsPerToken = alignment * alignment
        return ImageResizer.targetSize(width: width, height: height, alignment: alignment,
                                       minPixels: minImageTokens * pixelsPerToken,
                                       maxPixels: maxImageTokens * pixelsPerToken)
    }
}
