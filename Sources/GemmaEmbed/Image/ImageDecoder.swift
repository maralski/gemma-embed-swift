import CoreGraphics
import Foundation
import ImageIO

/// Decodes an `ImageInput` to 8-bit RGB without colour management, the way llama.cpp's
/// stb_image does: samples keep the values stored in the file and any embedded profile is
/// ignored. EXIF orientation is not applied, matching stb_image.
///
/// JPEG bytes (from a URL or data) go through `JPEGDecoder`, a port of stb_image's decoder, so
/// they decode to llama.cpp's bytes exactly; everything else, and `.cgImage`, goes through ImageIO.
/// A JPEG that stb_image does not implement (12-bit, lossless, hierarchical or arithmetic coded,
/// or with its height in a DNL marker) falls back to ImageIO so it still embeds, though llama.cpp
/// itself would refuse it. A corrupt JPEG fails with stb_image's reason.
enum ImageDecoder {
    static func decode(_ input: ImageInput) throws -> RGBImage {
        switch input {
        case .cgImage(let image):
            return try decode(image)
        case .url(let url):
            guard FileManager.default.isReadableFile(atPath: url.path),
                  let data = try? Data(contentsOf: url, options: .mappedIfSafe) else {
                throw GemmaEmbedError.imageDecoding("\(url.path) is not readable")
            }
            return try decode(data)
        case .data(let data):
            return try decode(data)
        }
    }

    private static func decode(_ data: Data) throws -> RGBImage {
        if data.starts(with: [0xFF, 0xD8]) {
            do {
                return try JPEGDecoder.decode(data)
            } catch let rejection as JPEGDecoder.Rejection where rejection.isUnsupportedEncoding {
                // ImageIO below.
            } catch let rejection as JPEGDecoder.Rejection {
                throw GemmaEmbedError.imageDecoding(rejection.reason)
            }
        }
        return try decode(cgImage(.data(data)))
    }

    static func cgImage(_ input: ImageInput) throws -> CGImage {
        let source: CGImageSource?
        switch input {
        case .cgImage(let image):
            return image
        case .url(let url):
            guard FileManager.default.isReadableFile(atPath: url.path) else {
                throw GemmaEmbedError.imageDecoding("\(url.path) is not readable")
            }
            source = CGImageSourceCreateWithURL(url as CFURL, nil)
        case .data(let data):
            source = CGImageSourceCreateWithData(data as CFData, nil)
        }
        guard let source, CGImageSourceGetCount(source) > 0,
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw GemmaEmbedError.imageDecoding("not an image format ImageIO can read")
        }
        return image
    }

    static func decode(_ image: CGImage) throws -> RGBImage {
        let width = image.width, height = image.height
        guard width > 0, height > 0 else { throw GemmaEmbedError.imageDecoding("image has no pixels") }
        let space = image.colorSpace
        if space?.model == .monochrome, let space {
            return try decodeGrey(image, space: space)
        }
        let opaque = [.none, .noneSkipFirst, .noneSkipLast].contains(image.alphaInfo)
        let alpha: CGImageAlphaInfo = opaque ? .noneSkipLast : .premultipliedLast
        let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!
        let candidates = [rgbSpace(of: space), sRGB].compactMap { $0 }
        for target in candidates {
            guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                                          space: target, bitmapInfo: alpha.rawValue),
                  let data = context.data else { continue }
            context.interpolationQuality = .none
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            let rgba = data.bindMemory(to: UInt8.self, capacity: width * height * 4)
            var pixels = [UInt8](repeating: 0, count: width * height * 3)
            for i in 0..<width * height {
                let a = Int(rgba[i * 4 + 3])
                for c in 0..<3 {
                    let v = Int(rgba[i * 4 + c])
                    // stb_image keeps the stored colour of translucent pixels; undo the premultiplication.
                    pixels[i * 3 + c] = opaque || a == 255 || a == 0 ? UInt8(v) : UInt8(min(255, (v * 255 + a / 2) / a))
                }
            }
            return RGBImage(width: width, height: height, pixels: pixels)
        }
        throw GemmaEmbedError.imageDecoding("cannot create an 8-bit RGB bitmap context")
    }

    /// The image's own RGB space when drawing into it is an identity copy; nil when the
    /// samples must be converted (no colour space, CMYK, Lab and the like).
    private static func rgbSpace(of space: CGColorSpace?) -> CGColorSpace? {
        guard let space else { return nil }
        switch space.model {
        case .rgb: return space
        case .indexed: return space.baseColorSpace.flatMap { $0.model == .rgb ? $0 : nil }
        default: return nil
        }
    }

    /// Grey images are drawn in their own grey space and replicated to three channels, as stb_image does.
    private static func decodeGrey(_ image: CGImage, space: CGColorSpace) throws -> RGBImage {
        let width = image.width, height = image.height
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width,
                                      space: space, bitmapInfo: CGImageAlphaInfo.none.rawValue),
              let data = context.data else {
            throw GemmaEmbedError.imageDecoding("cannot create an 8-bit grey bitmap context")
        }
        context.interpolationQuality = .none
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        let grey = data.bindMemory(to: UInt8.self, capacity: width * height)
        var pixels = [UInt8](repeating: 0, count: width * height * 3)
        for i in 0..<width * height {
            pixels[i * 3] = grey[i]
            pixels[i * 3 + 1] = grey[i]
            pixels[i * 3 + 2] = grey[i]
        }
        return RGBImage(width: width, height: height, pixels: pixels)
    }
}
