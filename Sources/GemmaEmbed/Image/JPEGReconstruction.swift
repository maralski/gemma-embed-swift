// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// From coefficients to RGB: stb_image's IDCT, upsamplers and colour conversion.
extension JPEGDecoder.Parser {
    var colourTransform: JPEGDecoder.ColourTransform {
        switch components.count {
        case 1: return .grey
        case 3: return rgbIDs == 3 || (adobeTransform == 0 && !jfif) ? .rgb : .yCbCr
        default:
            switch adobeTransform {
            case 0: return .cmyk
            case 2: return .ycck
            default: return .yCbCr
            }
        }
    }

    /// `load_jpeg_image` for three requested components: upsample each plane row by row and convert to RGB.
    func reconstruct() throws -> RGBImage {
        if 3 * width * height + 1 > Int(Int32.max) { throw JPEGDecoder.Rejection.corrupt("outofmem") }
        var resamplers = components.map { JPEGDecoder.Resampler(component: $0, hMax: hMax, vMax: vMax, imageWidth: width) }
        let lines = components.map { _ in JPEGDecoder.Buffer<UInt8>(count: width + 3, repeating: 0) }
        let transform = colourTransform
        var rows = [UnsafePointer<UInt8>](repeating: UnsafePointer(lines[0].base), count: components.count)
        let pixels = [UInt8](unsafeUninitializedCapacity: width * height * 3) { output, initialized in
            let out = output.baseAddress!
            for j in 0..<height {
                for k in components.indices {
                    rows[k] = resamplers[k].nextRow(into: lines[k].base, from: components[k])
                }
                let o = out + 3 * width * j
                switch transform {
                case .grey:
                    for i in 0..<width {
                        let g = rows[0][i]
                        o[3 * i] = g
                        o[3 * i + 1] = g
                        o[3 * i + 2] = g
                    }
                case .rgb:
                    for i in 0..<width {
                        o[3 * i] = rows[0][i]
                        o[3 * i + 1] = rows[1][i]
                        o[3 * i + 2] = rows[2][i]
                    }
                case .yCbCr:
                    JPEGDecoder.convertYCbCr(o, rows[0], rows[1], rows[2], count: width)
                case .cmyk:
                    for i in 0..<width {
                        let m = rows[3][i]
                        o[3 * i] = JPEGDecoder.blinn(rows[0][i], m)
                        o[3 * i + 1] = JPEGDecoder.blinn(rows[1][i], m)
                        o[3 * i + 2] = JPEGDecoder.blinn(rows[2][i], m)
                    }
                case .ycck:
                    JPEGDecoder.convertYCbCr(o, rows[0], rows[1], rows[2], count: width)
                    for i in 0..<width {
                        let m = rows[3][i]
                        o[3 * i] = JPEGDecoder.blinn(255 - o[3 * i], m)
                        o[3 * i + 1] = JPEGDecoder.blinn(255 - o[3 * i + 1], m)
                        o[3 * i + 2] = JPEGDecoder.blinn(255 - o[3 * i + 2], m)
                    }
                }
            }
            initialized = width * height * 3
        }
        return RGBImage(width: width, height: height, pixels: pixels)
    }
}

extension JPEGDecoder {
    /// How the decoded components become RGB, settled by the component count and the JFIF and Adobe markers.
    enum ColourTransform {
        case grey
        case rgb
        case yCbCr
        case cmyk
        case ycck
    }

    /// stb's per-component row upsampler state (`stbi__resample`).
    struct Resampler {
        enum Kind {
            case none
            case vertical2
            case horizontal2
            case both2
            case nearest
        }

        let kind: Kind
        let hs: Int
        let vs: Int
        let lowResWidth: Int
        var ystep: Int
        var ypos = 0
        /// Offsets into the plane of the two source rows (`line0`, `line1`).
        var line0 = 0
        var line1 = 0

        init(component: Component, hMax: Int, vMax: Int, imageWidth: Int) {
            hs = hMax / component.h
            vs = vMax / component.v
            ystep = vs >> 1
            lowResWidth = (imageWidth + hs - 1) / hs
            switch (hs, vs) {
            case (1, 1): kind = .none
            case (1, 2): kind = .vertical2
            case (2, 1): kind = .horizontal2
            case (2, 2): kind = .both2
            default: kind = .nearest
            }
        }

        mutating func nextRow(into line: UnsafeMutablePointer<UInt8>, from component: Component) -> UnsafePointer<UInt8> {
            let plane = component.samples.base
            let bottom = ystep >= vs >> 1
            let near = UnsafePointer(plane + (bottom ? line1 : line0))
            let far = UnsafePointer(plane + (bottom ? line0 : line1))
            let row: UnsafePointer<UInt8>
            switch kind {
            case .none: row = near
            case .vertical2: row = JPEGDecoder.resampleVertical2(line, near, far, lowResWidth)
            case .horizontal2: row = JPEGDecoder.resampleHorizontal2(line, near, lowResWidth)
            case .both2: row = JPEGDecoder.resampleBoth2(line, near, far, lowResWidth)
            case .nearest: row = JPEGDecoder.resampleNearest(line, near, lowResWidth, hs)
            }
            ystep += 1
            if ystep >= vs {
                ystep = 0
                line0 = line1
                ypos += 1
                if ypos < component.height { line1 += component.stride }
            }
            return row
        }
    }

    // MARK: Kernels

    /// `stbi__clamp`.
    @inline(__always)
    static func clamp(_ x: Int32) -> UInt8 {
        UInt32(bitPattern: x) > 255 ? (x < 0 ? 0 : 255) : UInt8(truncatingIfNeeded: x)
    }

    /// `STBI__IDCT_1D` (jidctint's DCT_ISLOW), constants scaled by 4096 as `stbi__f2f` rounds them.
    @inline(__always)
    static func idct1D(_ s0: Int32, _ s1: Int32, _ s2: Int32, _ s3: Int32, _ s4: Int32, _ s5: Int32, _ s6: Int32, _ s7: Int32)
        -> (x0: Int32, x1: Int32, x2: Int32, x3: Int32, t0: Int32, t1: Int32, t2: Int32, t3: Int32) {
        var p2 = s2, p3 = s6
        var p1 = (p2 &+ p3) &* 2217
        var t2 = p1 &+ p3 &* -7567
        var t3 = p1 &+ p2 &* 3135
        p2 = s0
        p3 = s4
        var t0 = (p2 &+ p3) &* 4096
        var t1 = (p2 &- p3) &* 4096
        let x0 = t0 &+ t3, x3 = t0 &- t3, x1 = t1 &+ t2, x2 = t1 &- t2
        t0 = s7
        t1 = s5
        t2 = s3
        t3 = s1
        p3 = t0 &+ t2
        var p4 = t1 &+ t3
        p1 = t0 &+ t3
        p2 = t1 &+ t2
        let p5 = (p3 &+ p4) &* 4816
        t0 = t0 &* 1223
        t1 = t1 &* 8410
        t2 = t2 &* 12586
        t3 = t3 &* 6149
        p1 = p5 &+ p1 &* -3685
        p2 = p5 &+ p2 &* -10497
        p3 = p3 &* -8034
        p4 = p4 &* -1597
        t3 = t3 &+ (p1 &+ p4)
        t2 = t2 &+ (p2 &+ p3)
        t1 = t1 &+ (p2 &+ p4)
        t0 = t0 &+ (p1 &+ p3)
        return (x0, x1, x2, x3, t0, t1, t2, t3)
    }

    /// `stbi__idct_block`: columns keep two extra bits with a 512 bias, rows round at bit 17 and add 128.
    static func idct(_ data: UnsafePointer<Int16>, into out: UnsafeMutablePointer<UInt8>, stride: Int, scratch v: UnsafeMutablePointer<Int32>) {
        for i in 0..<8 {
            let d = data + i
            if d[8] == 0 && d[16] == 0 && d[24] == 0 && d[32] == 0 && d[40] == 0 && d[48] == 0 && d[56] == 0 {
                let dc = Int32(d[0]) &* 4
                for r in 0..<8 { v[i + 8 * r] = dc }
            } else {
                var e = idct1D(Int32(d[0]), Int32(d[8]), Int32(d[16]), Int32(d[24]), Int32(d[32]), Int32(d[40]), Int32(d[48]), Int32(d[56]))
                e.x0 &+= 512
                e.x1 &+= 512
                e.x2 &+= 512
                e.x3 &+= 512
                v[i] = (e.x0 &+ e.t3) &>> 10
                v[i + 56] = (e.x0 &- e.t3) &>> 10
                v[i + 8] = (e.x1 &+ e.t2) &>> 10
                v[i + 48] = (e.x1 &- e.t2) &>> 10
                v[i + 16] = (e.x2 &+ e.t1) &>> 10
                v[i + 40] = (e.x2 &- e.t1) &>> 10
                v[i + 24] = (e.x3 &+ e.t0) &>> 10
                v[i + 32] = (e.x3 &- e.t0) &>> 10
            }
        }
        let bias: Int32 = 65536 + (128 << 17)
        for i in 0..<8 {
            let r = v + 8 * i
            let o = out + stride * i
            var e = idct1D(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7])
            e.x0 &+= bias
            e.x1 &+= bias
            e.x2 &+= bias
            e.x3 &+= bias
            o[0] = clamp((e.x0 &+ e.t3) &>> 17)
            o[7] = clamp((e.x0 &- e.t3) &>> 17)
            o[1] = clamp((e.x1 &+ e.t2) &>> 17)
            o[6] = clamp((e.x1 &- e.t2) &>> 17)
            o[2] = clamp((e.x2 &+ e.t1) &>> 17)
            o[5] = clamp((e.x2 &- e.t1) &>> 17)
            o[3] = clamp((e.x3 &+ e.t0) &>> 17)
            o[4] = clamp((e.x3 &- e.t0) &>> 17)
        }
    }

    /// `stbi__resample_row_v_2`.
    static func resampleVertical2(_ out: UnsafeMutablePointer<UInt8>, _ near: UnsafePointer<UInt8>, _ far: UnsafePointer<UInt8>,
                                  _ w: Int) -> UnsafePointer<UInt8> {
        for i in 0..<w {
            out[i] = UInt8(truncatingIfNeeded: (3 * Int(near[i]) + Int(far[i]) + 2) >> 2)
        }
        return UnsafePointer(out)
    }

    /// `stbi__resample_row_h_2`.
    static func resampleHorizontal2(_ out: UnsafeMutablePointer<UInt8>, _ input: UnsafePointer<UInt8>, _ w: Int) -> UnsafePointer<UInt8> {
        if w == 1 {
            out[0] = input[0]
            out[1] = input[0]
            return UnsafePointer(out)
        }
        out[0] = input[0]
        out[1] = UInt8(truncatingIfNeeded: (Int(input[0]) * 3 + Int(input[1]) + 2) >> 2)
        var i = 1
        while i < w - 1 {
            let n = 3 * Int(input[i]) + 2
            out[i * 2] = UInt8(truncatingIfNeeded: (n + Int(input[i - 1])) >> 2)
            out[i * 2 + 1] = UInt8(truncatingIfNeeded: (n + Int(input[i + 1])) >> 2)
            i += 1
        }
        out[i * 2] = UInt8(truncatingIfNeeded: (Int(input[w - 2]) * 3 + Int(input[w - 1]) + 2) >> 2)
        out[i * 2 + 1] = input[w - 1]
        return UnsafePointer(out)
    }

    /// `stbi__resample_row_hv_2`: the vertical pass into `t`, then the horizontal pass over it.
    static func resampleBoth2(_ out: UnsafeMutablePointer<UInt8>, _ near: UnsafePointer<UInt8>, _ far: UnsafePointer<UInt8>,
                              _ w: Int) -> UnsafePointer<UInt8> {
        if w == 1 {
            let v = UInt8(truncatingIfNeeded: (3 * Int(near[0]) + Int(far[0]) + 2) >> 2)
            out[0] = v
            out[1] = v
            return UnsafePointer(out)
        }
        var t1 = 3 * Int(near[0]) + Int(far[0])
        out[0] = UInt8(truncatingIfNeeded: (t1 + 2) >> 2)
        for i in 1..<w {
            let t0 = t1
            t1 = 3 * Int(near[i]) + Int(far[i])
            out[i * 2 - 1] = UInt8(truncatingIfNeeded: (3 * t0 + t1 + 8) >> 4)
            out[i * 2] = UInt8(truncatingIfNeeded: (3 * t1 + t0 + 8) >> 4)
        }
        out[w * 2 - 1] = UInt8(truncatingIfNeeded: (t1 + 2) >> 2)
        return UnsafePointer(out)
    }

    /// `stbi__resample_row_generic`: nearest neighbour across, the near row only.
    static func resampleNearest(_ out: UnsafeMutablePointer<UInt8>, _ near: UnsafePointer<UInt8>, _ w: Int, _ hs: Int) -> UnsafePointer<UInt8> {
        for i in 0..<w {
            for j in 0..<hs { out[i * hs + j] = near[i] }
        }
        return UnsafePointer(out)
    }

    /// `stbi__YCbCr_to_RGB_row` with stb's `stbi__float2fixed` constants: 1.402, -0.71414, -0.34414
    /// (whose product keeps only its top 16 bits) and 1.772, each scaled by 4096 and shifted by 8.
    static func convertYCbCr(_ out: UnsafeMutablePointer<UInt8>, _ y: UnsafePointer<UInt8>, _ cb: UnsafePointer<UInt8>,
                             _ cr: UnsafePointer<UInt8>, count: Int) {
        let highHalf = Int32(bitPattern: 0xFFFF_0000)
        for i in 0..<count {
            let yFixed = (Int32(y[i]) << 20) &+ (1 << 19)
            let r = Int32(cr[i]) - 128
            let b = Int32(cb[i]) - 128
            let red = (yFixed &+ r &* 1_470_208) &>> 20
            let green = (yFixed &+ r &* -748_800 &+ ((b &* -360_960) & highHalf)) &>> 20
            let blue = (yFixed &+ b &* 1_858_048) &>> 20
            out[3 * i] = clamp(red)
            out[3 * i + 1] = clamp(green)
            out[3 * i + 2] = clamp(blue)
        }
    }

    /// `stbi__blinn_8x8`: `x * y / 255`, rounded.
    @inline(__always)
    static func blinn(_ x: UInt8, _ y: UInt8) -> UInt8 {
        let t = UInt32(x) * UInt32(y) + 128
        return UInt8(truncatingIfNeeded: (t + (t >> 8)) >> 8)
    }
}
