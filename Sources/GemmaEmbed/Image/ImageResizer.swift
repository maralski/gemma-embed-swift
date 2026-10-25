import Foundation

/// The geometry and resampling of llama.cpp's `img_tool` (`tools/mtmd/mtmd-image.cpp`), ported
/// operation for operation so the resized bytes are identical. Float32 and Double are used
/// exactly where the C++ uses `float` and `double`.
enum ImageResizer {
    /// `calc_size_preserved_ratio`: the size an image of `width` by `height` is resized to, each side
    /// a multiple of `alignment`, with the pixel count pulled into `minPixels...maxPixels`.
    static func targetSize(width: Int, height: Int, alignment: Int, minPixels: Int, maxPixels: Int) -> (width: Int, height: Int) {
        precondition(width > 0 && height > 0 && alignment > 0)
        let f = Float(alignment)
        func roundBy(_ x: Float) -> Int { Int((x / f).rounded()) * alignment }
        func ceilBy(_ x: Float) -> Int { Int((x / f).rounded(.up)) * alignment }
        func floorBy(_ x: Float) -> Int { Int((x / f).rounded(.down)) * alignment }
        var w = max(alignment, roundBy(Float(width)))
        var h = max(alignment, roundBy(Float(height)))
        if maxPixels > 0 && w * h > maxPixels {
            let beta = (Float(height) * Float(width) / Float(maxPixels)).squareRoot()
            h = max(alignment, floorBy(Float(height) / beta))
            w = max(alignment, floorBy(Float(width) / beta))
        } else if minPixels > 0 && w * h < minPixels {
            let beta = (Float(minPixels) / (Float(height) * Float(width))).squareRoot()
            h = ceilBy(Float(height) * beta)
            w = ceilBy(Float(width) * beta)
        }
        return (w, h)
    }

    /// `img_tool::resize` with `PAD_CEIL` and a black pad: the image is scaled to fit inside
    /// `width` by `height` keeping its aspect ratio, then centred on a black canvas.
    static func fit(_ image: RGBImage, width: Int, height: Int) -> RGBImage {
        if image.width == width && image.height == height { return image }
        let scale = min(Float(width) / Float(image.width), Float(height) / Float(image.height))
        let fittedWidth = min(Int((Float(image.width) * scale).rounded(.up)), width)
        let fittedHeight = min(Int((Float(image.height) * scale).rounded(.up)), height)
        let fitted = resizeBicubic(image, width: fittedWidth, height: fittedHeight)
        if fittedWidth == width && fittedHeight == height { return fitted }
        var canvas = RGBImage(width: width, height: height)
        let offsetX = (width - fittedWidth) / 2
        let offsetY = (height - fittedHeight) / 2
        for y in 0..<fittedHeight {
            let src = y * fittedWidth * 3
            let dst = ((y + offsetY) * width + offsetX) * 3
            canvas.pixels[dst..<dst + fittedWidth * 3] = fitted.pixels[src..<src + fittedWidth * 3]
        }
        return canvas
    }

    /// `resize_pillow` with the bicubic filter: Pillow's separable resampling with `a = -0.5`,
    /// 22-bit fixed-point weights, a horizontal pass then a vertical pass, each rounded to 8 bits.
    static func resizeBicubic(_ image: RGBImage, width: Int, height: Int) -> RGBImage {
        precondition(width > 0 && width <= 65536 && height > 0 && height <= 65536)
        var out = image
        if width != image.width {
            let k = Kernel(inSize: image.width, outSize: width)
            out = resampleHorizontal(out, width: width, kernel: k)
        }
        if height != image.height {
            let k = Kernel(inSize: image.height, outSize: height)
            out = resampleVertical(out, height: height, kernel: k)
        }
        return out
    }

    private static let precisionBits: Int32 = 32 - 8 - 2

    private static func clip8(_ v: Int32) -> UInt8 {
        UInt8(clamping: v >> precisionBits)
    }

    /// Fixed-point filter taps for every output pixel of one dimension (`precompute_weights`).
    struct Kernel {
        let size: Int
        /// First input pixel of each output pixel.
        var start: [Int]
        /// Number of input pixels of each output pixel.
        var count: [Int]
        /// `size` taps per output pixel.
        var weights: [Int32]

        init(inSize: Int, outSize: Int) {
            let filterSupport = 2.0
            let scale = Double(inSize) / Double(outSize)
            let filterScale = max(scale, 1.0)
            let support = filterSupport * filterScale
            size = Int(support.rounded(.up)) * 2 + 1
            start = [Int](repeating: 0, count: outSize)
            count = [Int](repeating: 0, count: outSize)
            var pre = [Double](repeating: 0, count: outSize * size)
            for xx in 0..<outSize {
                let center = (Double(xx) + 0.5) * scale
                var ww = 0.0
                let ss = 1.0 / filterScale
                // Int(_:) truncates toward zero, like the C++ cast.
                let xmin = max(0, Int(center - support + 0.5))
                let xmax = min(inSize, Int(center + support + 0.5)) - xmin
                for x in 0..<xmax {
                    let w = Kernel.bicubic((Double(x + xmin) - center + 0.5) * ss)
                    pre[xx * size + x] = w
                    ww += w
                }
                if ww != 0 {
                    for x in 0..<xmax { pre[xx * size + x] /= ww }
                }
                start[xx] = xmin
                count[xx] = xmax
            }
            let fxpScale = Double(1 << Int(ImageResizer.precisionBits))
            // Pillow adds or subtracts 0.5 and truncates toward zero.
            weights = pre.map { Int32($0 * fxpScale + ($0 < 0 ? -0.5 : 0.5)) }
        }

        static func bicubic(_ x: Double) -> Double {
            let a = -0.5
            let x = abs(x)
            if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
            if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
            return 0
        }
    }

    private static func resampleHorizontal(_ image: RGBImage, width outWidth: Int, kernel k: Kernel) -> RGBImage {
        var out = [UInt8](repeating: 0, count: outWidth * image.height * 3)
        let inWidth = image.width
        image.pixels.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for yy in 0..<image.height {
                    let srcRow = yy * inWidth * 3
                    let dstRow = yy * outWidth * 3
                    for xx in 0..<outWidth {
                        var s0: Int32 = 1 << (precisionBits - 1)
                        var s1 = s0, s2 = s0
                        let taps = xx * k.size
                        var p = srcRow + k.start[xx] * 3
                        for x in 0..<k.count[xx] {
                            let w = k.weights[taps + x]
                            s0 &+= Int32(src[p]) &* w
                            s1 &+= Int32(src[p + 1]) &* w
                            s2 &+= Int32(src[p + 2]) &* w
                            p += 3
                        }
                        dst[dstRow + xx * 3] = clip8(s0)
                        dst[dstRow + xx * 3 + 1] = clip8(s1)
                        dst[dstRow + xx * 3 + 2] = clip8(s2)
                    }
                }
            }
        }
        return RGBImage(width: outWidth, height: image.height, pixels: out)
    }

    private static func resampleVertical(_ image: RGBImage, height outHeight: Int, kernel k: Kernel) -> RGBImage {
        let rowElements = image.width * 3
        var out = [UInt8](repeating: 0, count: rowElements * outHeight)
        var acc = [Int32](repeating: 0, count: rowElements)
        image.pixels.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                acc.withUnsafeMutableBufferPointer { acc in
                    for yy in 0..<outHeight {
                        for i in 0..<rowElements { acc[i] = 1 << (precisionBits - 1) }
                        let taps = yy * k.size
                        for y in 0..<k.count[yy] {
                            let w = k.weights[taps + y]
                            let srcRow = (k.start[yy] + y) * rowElements
                            for i in 0..<rowElements { acc[i] &+= Int32(src[srcRow + i]) &* w }
                        }
                        let dstRow = yy * rowElements
                        for i in 0..<rowElements { dst[dstRow + i] = clip8(acc[i]) }
                    }
                }
            }
        }
        return RGBImage(width: image.width, height: outHeight, pixels: out)
    }
}
