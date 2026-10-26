// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
// Usage: swift scripts/make_fixture_images.swift <all images dir> [<package images dir>]
//   e.g. swift scripts/make_fixture_images.swift models/fixtures/images Tests/GemmaEmbedTests/Fixtures/images
// (about three minutes interpreted; swiftc -O builds a binary that takes seconds, with the same output)
// Draws every image fixture from code, so all of them can be redistributed: 10 procedural photo-like
// scenes (scenes/), 10 synthetic CoreGraphics images (synthetic/), 10 near-duplicate variants of the
// scenes (variants/), the JPEG encodings stb_image supports (jpeg-variants/) and two it refuses
// (jpeg-unsupported/), plus manifest.json for the first three sets. jpeg-variants/ and
// jpeg-unsupported/ need cjpeg (libjpeg-turbo 3) and magick (ImageMagick 7) on PATH.
// With a package directory, it also copies manifest.json and every image under 250 KB there; the
// larger ones stay in the first directory, where the tests look for them (models/fixtures/images).
// The reference vectors come from scripts/oracle_images.py and the llama.cpp dumps from
// scripts/vision_dump.cpp.
import CoreGraphics
import CoreText
import Foundation
import ImageIO
import UniformTypeIdentifiers

let root = URL(fileURLWithPath: CommandLine.arguments[1])
let package = CommandLine.arguments.count > 2 ? URL(fileURLWithPath: CommandLine.arguments[2]) : nil
let fm = FileManager.default
for d in ["scenes", "synthetic", "variants", "jpeg-variants", "jpeg-unsupported"] {
    try? fm.createDirectory(at: root.appendingPathComponent(d), withIntermediateDirectories: true)
}
let srgb = CGColorSpace(name: CGColorSpace.sRGB)!

func context(_ w: Int, _ h: Int) -> CGContext {
    CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: srgb,
              bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
}

func write(_ image: CGImage, to url: URL, type: UTType, quality: Double? = nil) {
    let dest = CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil)!
    var props: [CFString: Any] = [:]
    if let q = quality { props[kCGImageDestinationLossyCompressionQuality] = q }
    CGImageDestinationAddImage(dest, image, props as CFDictionary)
    precondition(CGImageDestinationFinalize(dest), "write failed \(url.path)")
}

func load(_ url: URL) -> CGImage {
    let src = CGImageSourceCreateWithURL(url as CFURL, nil)!
    return CGImageSourceCreateImageAtIndex(src, 0, nil)!
}

var manifest: [[String: Any]] = []
func record(_ file: String, _ kind: String, _ note: String, source: String? = nil, variant: String? = nil) {
    var m: [String: Any] = ["file": file, "kind": kind, "note": note]
    if let s = source { m["source"] = s }
    if let v = variant { m["variant"] = v }
    manifest.append(m)
}

func color(_ c: [Double], _ alpha: Double = 1) -> CGColor {
    CGColor(colorSpace: srgb, components: c.map { CGFloat($0) } + [CGFloat(alpha)])!
}

// Synthetic images.
func synthetic(_ index: Int, _ name: String, _ w: Int, _ h: Int, type: UTType = .jpeg, quality: Double = 0.92, _ note: String, _ draw: (CGContext) -> Void) {
    let ctx = context(w, h)
    draw(ctx)
    let ext = type == .png ? "png" : "jpg"
    let file = String(format: "synthetic/%02d-%@.%@", index, name, ext)
    write(ctx.makeImage()!, to: root.appendingPathComponent(file), type: type, quality: type == .png ? nil : quality)
    record(file, "synthetic", note)
}

synthetic(0, "solid-red", 640, 480, type: .png, "solid rgb(220,30,30), png") { c in
    c.setFillColor(CGColor(colorSpace: srgb, components: [220/255, 30/255, 30/255, 1])!)
    c.fill(CGRect(x: 0, y: 0, width: 640, height: 480))
}
synthetic(1, "solid-grey", 512, 512, type: .png, "solid rgb(128,128,128), png, 512 square") { c in
    c.setFillColor(CGColor(colorSpace: srgb, components: [128/255, 128/255, 128/255, 1])!)
    c.fill(CGRect(x: 0, y: 0, width: 512, height: 512))
}
synthetic(2, "gradient-h", 800, 400, type: .png, "horizontal black to white gradient, png") { c in
    let g = CGGradient(colorsSpace: srgb, colors: [CGColor(colorSpace: srgb, components: [0, 0, 0, 1])!, CGColor(colorSpace: srgb, components: [1, 1, 1, 1])!] as CFArray, locations: [0, 1])!
    c.drawLinearGradient(g, start: CGPoint(x: 0, y: 0), end: CGPoint(x: 800, y: 0), options: [])
}
synthetic(3, "gradient-rgb", 600, 900, "vertical red to green to blue gradient, portrait jpeg") { c in
    let colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]].map { CGColor(colorSpace: srgb, components: $0.map { CGFloat($0) })! }
    let g = CGGradient(colorsSpace: srgb, colors: colors as CFArray, locations: [0, 0.5, 1])!
    c.drawLinearGradient(g, start: CGPoint(x: 0, y: 0), end: CGPoint(x: 0, y: 900), options: [])
}
func text(_ c: CGContext, _ s: String, font: String, size: CGFloat, at p: CGPoint, color: [CGFloat]) {
    let attrs: [NSAttributedString.Key: Any] = [kCTFontAttributeName as NSAttributedString.Key: CTFontCreateWithName(font as CFString, size, nil), kCTForegroundColorAttributeName as NSAttributedString.Key: CGColor(colorSpace: srgb, components: color)!]
    let line = CTLineCreateWithAttributedString(NSAttributedString(string: s, attributes: attrs))
    c.textPosition = p
    CTLineDraw(line, c)
}
synthetic(4, "text-helvetica", 1024, 512, "white page with the words 'quarterly revenue report' in Helvetica") { c in
    c.setFillColor(CGColor(colorSpace: srgb, components: [1, 1, 1, 1])!); c.fill(CGRect(x: 0, y: 0, width: 1024, height: 512))
    text(c, "Quarterly revenue report", font: "Helvetica-Bold", size: 72, at: CGPoint(x: 60, y: 300), color: [0.1, 0.1, 0.1, 1])
    text(c, "Rates were left unchanged for a third meeting.", font: "Helvetica", size: 36, at: CGPoint(x: 60, y: 200), color: [0.2, 0.2, 0.3, 1])
}
synthetic(5, "text-menlo", 768, 768, "dark terminal with green monospaced code in Menlo") { c in
    c.setFillColor(CGColor(colorSpace: srgb, components: [0.05, 0.05, 0.08, 1])!); c.fill(CGRect(x: 0, y: 0, width: 768, height: 768))
    for (i, s) in ["func embed(images: [ImageInput]) async throws", "    let vectors = try await forward.run(batch)", "    return vectors.map(Embedding.init)", "}"].enumerated() {
        text(c, s, font: "Menlo", size: 28, at: CGPoint(x: 40, y: 650 - CGFloat(i) * 50), color: [0.4, 1, 0.5, 1])
    }
}
synthetic(6, "text-times", 1200, 300, "a headline in Times New Roman on cream") { c in
    c.setFillColor(CGColor(colorSpace: srgb, components: [0.98, 0.96, 0.9, 1])!); c.fill(CGRect(x: 0, y: 0, width: 1200, height: 300))
    text(c, "Central bank holds interest rates", font: "TimesNewRomanPS-BoldMT", size: 80, at: CGPoint(x: 40, y: 110), color: [0, 0, 0, 1])
}
synthetic(7, "shapes", 640, 640, "blue circle, yellow square and red triangle on white") { c in
    c.setFillColor(CGColor(colorSpace: srgb, components: [1, 1, 1, 1])!); c.fill(CGRect(x: 0, y: 0, width: 640, height: 640))
    c.setFillColor(CGColor(colorSpace: srgb, components: [0.1, 0.3, 0.9, 1])!); c.fillEllipse(in: CGRect(x: 60, y: 340, width: 240, height: 240))
    c.setFillColor(CGColor(colorSpace: srgb, components: [1, 0.85, 0.1, 1])!); c.fill(CGRect(x: 360, y: 360, width: 220, height: 220))
    c.setFillColor(CGColor(colorSpace: srgb, components: [0.9, 0.1, 0.1, 1])!)
    c.move(to: CGPoint(x: 320, y: 60)); c.addLine(to: CGPoint(x: 560, y: 60)); c.addLine(to: CGPoint(x: 440, y: 280)); c.closePath(); c.fillPath()
}
synthetic(8, "checker", 1000, 700, type: .png, "black and white checkerboard of 50 px squares, png") { c in
    for y in 0..<14 { for x in 0..<20 {
        c.setFillColor(CGColor(colorSpace: srgb, components: (x + y) % 2 == 0 ? [0, 0, 0, 1] : [1, 1, 1, 1])!)
        c.fill(CGRect(x: x * 50, y: y * 50, width: 50, height: 50))
    } }
}
synthetic(9, "tiny-noise", 200, 150, type: .png, "small 200x150 deterministic colour noise, png (below the minimum pixel budget, gets upscaled)") { c in
    var state: UInt64 = 42
    for y in 0..<15 { for x in 0..<20 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        let r = CGFloat((state >> 33) & 255) / 255, g = CGFloat((state >> 41) & 255) / 255, b = CGFloat((state >> 49) & 255) / 255
        c.setFillColor(CGColor(colorSpace: srgb, components: [r, g, b, 1])!)
        c.fill(CGRect(x: x * 10, y: y * 10, width: 10, height: 10))
    } }
}

// Procedural scenes: photo-like pictures with distinct subjects and palettes, sized like the
// photos the tests were written against (1600 square for the downscale path, 640 square for the
// in-range path, 1920 square for the largest decode).

struct RNG {
    var state: UInt64
    mutating func next() -> Double {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return Double((z ^ (z >> 31)) >> 11) / Double(1 << 53)
    }
    mutating func range(_ a: Double, _ b: Double) -> Double { a + (b - a) * next() }
}

func lattice(_ x: Int, _ y: Int, _ seed: UInt64) -> Double {
    var h = UInt64(bitPattern: Int64(x)) &* 0x9E3779B97F4A7C15 ^ UInt64(bitPattern: Int64(y)) &* 0xC2B2AE3D27D4EB4F ^ seed &* 0x165667B19E3779F9
    h = (h ^ (h >> 29)) &* 0xBF58476D1CE4E5B9
    h ^= h >> 32
    return Double(h & 0xFFFFFF) / Double(0xFFFFFF)
}

func valueNoise(_ x: Double, _ y: Double, _ seed: UInt64) -> Double {
    let xi = Int(floor(x)), yi = Int(floor(y))
    let fx = x - floor(x), fy = y - floor(y)
    let sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy)
    let a = lattice(xi, yi, seed), b = lattice(xi + 1, yi, seed), c = lattice(xi, yi + 1, seed), d = lattice(xi + 1, yi + 1, seed)
    return (a + (b - a) * sx) * (1 - sy) + (c + (d - c) * sx) * sy
}

/// Fractal value noise in [0, 1].
func fbm(_ x: Double, _ y: Double, octaves: Int = 5, seed: UInt64) -> Double {
    var sum = 0.0, amp = 0.5, freq = 1.0, norm = 0.0
    for o in 0..<octaves {
        sum += amp * valueNoise(x * freq, y * freq, seed &+ UInt64(o))
        norm += amp
        amp *= 0.5
        freq *= 2.03
    }
    return sum / norm
}

/// A scene in unit coordinates: `u` runs left to right and `v` bottom to top, both 0 to 1 across
/// the width, so a scene keeps its shapes at any size (lengths in `v` scale with the width too).
final class Canvas {
    let c: CGContext
    let w: Double, h: Double
    var rng: RNG
    let seed: UInt64

    init(_ width: Int, _ height: Int, seed: UInt64) {
        c = context(width, height)
        w = Double(width); h = Double(height)
        rng = RNG(state: seed)
        self.seed = seed
    }

    func p(_ u: Double, _ v: Double) -> CGPoint { CGPoint(x: u * w, y: v * h) }

    func verticalGradient(_ stops: [(Double, [Double])], from v0: Double = 0, to v1: Double = 1) {
        let g = CGGradient(colorsSpace: srgb, colors: stops.map { color($0.1) } as CFArray, locations: stops.map { CGFloat($0.0) })!
        c.drawLinearGradient(g, start: p(0, v0), end: p(0, v1), options: [.drawsBeforeStartLocation, .drawsAfterEndLocation])
    }

    func glow(_ u: Double, _ v: Double, radius: Double, _ rgb: [Double], alpha: Double) {
        let g = CGGradient(colorsSpace: srgb, colors: [color(rgb, alpha), color(rgb, 0)] as CFArray, locations: [0, 1])!
        c.drawRadialGradient(g, startCenter: p(u, v), startRadius: 0, endCenter: p(u, v), endRadius: radius * w, options: [])
    }

    func ellipse(_ u: Double, _ v: Double, _ ru: Double, _ rv: Double, _ rgb: [Double], alpha: Double = 1) {
        c.setFillColor(color(rgb, alpha))
        c.fillEllipse(in: CGRect(x: (u - ru) * w, y: v * h - rv * w, width: 2 * ru * w, height: 2 * rv * w))
    }

    func polygon(_ points: [(Double, Double)], _ rgb: [Double], alpha: Double = 1) {
        c.setFillColor(color(rgb, alpha))
        c.move(to: p(points[0].0, points[0].1))
        for q in points.dropFirst() { c.addLine(to: p(q.0, q.1)) }
        c.closePath()
        c.fillPath()
    }

    func line(_ points: [(Double, Double)], width: Double, _ rgb: [Double], alpha: Double = 1) {
        c.setStrokeColor(color(rgb, alpha))
        c.setLineWidth(width * w)
        c.setLineCap(.round)
        c.setLineJoin(.round)
        c.move(to: p(points[0].0, points[0].1))
        for q in points.dropFirst() { c.addLine(to: p(q.0, q.1)) }
        c.strokePath()
    }

    /// The land under `top(u)`, filled with a vertical gradient from `low` at the bottom to `high` at its crest.
    func ridge(_ top: (Double) -> Double, low: [Double], high: [Double]) {
        var pts: [(Double, Double)] = [(0, -0.01)]
        for i in 0...400 { let u = Double(i) / 400; pts.append((u, top(u))) }
        pts.append((1, -0.01))
        c.saveGState()
        c.move(to: p(pts[0].0, pts[0].1))
        for q in pts.dropFirst() { c.addLine(to: p(q.0, q.1)) }
        c.closePath()
        c.clip()
        let crest = pts.map(\.1).max()!
        verticalGradient([(0, low), (1, high)], from: 0, to: crest)
        c.restoreGState()
    }

    func text(_ s: String, font: String, size: Double, at u: Double, _ v: Double, _ rgb: [Double]) {
        let attrs: [NSAttributedString.Key: Any] = [kCTFontAttributeName as NSAttributedString.Key: CTFontCreateWithName(font as CFString, size * w, nil),
                                                    kCTForegroundColorAttributeName as NSAttributedString.Key: color(rgb)]
        let line = CTLineCreateWithAttributedString(NSAttributedString(string: s, attributes: attrs))
        c.textPosition = p(u, v)
        CTLineDraw(line, c)
    }

    /// Per-pixel pass over what is drawn so far: `f(u, v, &rgb)` with channels in 0...1.
    func shade(_ f: (Double, Double, inout (Double, Double, Double)) -> Void) {
        let px = c.data!.assumingMemoryBound(to: UInt8.self)
        let width = Int(w), height = Int(h)
        for row in 0..<height {
            let v = 1 - (Double(row) + 0.5) / h
            for col in 0..<width {
                let i = (row * width + col) * 4
                var rgb = (Double(px[i]) / 255, Double(px[i + 1]) / 255, Double(px[i + 2]) / 255)
                f((Double(col) + 0.5) / w, v, &rgb)
                px[i] = UInt8(max(0, min(255, (rgb.0 * 255).rounded())))
                px[i + 1] = UInt8(max(0, min(255, (rgb.1 * 255).rounded())))
                px[i + 2] = UInt8(max(0, min(255, (rgb.2 * 255).rounded())))
            }
        }
    }

    /// Lens softness: two passes of a separable box blur of `radius` pixels, edges clamped.
    func soften(radius: Int) {
        let px = c.data!.assumingMemoryBound(to: UInt8.self)
        let width = Int(w), height = Int(h)
        var line = [Int](repeating: 0, count: max(width, height) * 3)
        func pass(count: Int, lines: Int, at index: (Int, Int) -> Int) {
            for l in 0..<lines {
                for k in 0..<count { for ch in 0..<3 { line[k * 3 + ch] = Int(px[index(l, k) + ch]) } }
                for k in 0..<count {
                    for ch in 0..<3 {
                        var sum = 0
                        for d in -radius...radius { sum += line[min(count - 1, max(0, k + d)) * 3 + ch] }
                        px[index(l, k) + ch] = UInt8((sum + radius) / (2 * radius + 1))
                    }
                }
            }
        }
        for _ in 0..<2 {
            pass(count: width, lines: height) { row, col in (row * width + col) * 4 }
            pass(count: height, lines: width) { col, row in (row * width + col) * 4 }
        }
    }

    /// Photographic texture: fractal brightness variation where `amount(u, v)` is above zero, then sensor grain.
    func texture(scale: Double, grain: Double, _ amount: @escaping (Double, Double) -> Double) {
        let s = seed
        var noise = RNG(state: seed ^ 0xA5A5)
        let aspect = h / w
        shade { u, v, rgb in
            let a = amount(u, v)
            var k = 1.0
            if a > 0 { k += a * (fbm(u * scale, v * scale * aspect, octaves: 3, seed: s) - 0.5) * 2 }
            let n = (noise.next() - 0.5) * grain
            rgb = (rgb.0 * k + n, rgb.1 * k + n, rgb.2 * k + n)
        }
    }
}

struct Scene {
    let name: String
    let width: Int
    let height: Int
    let note: String
    let draw: (Canvas) -> Void
}

let scenes: [Scene] = [
    Scene(name: "green-hills", width: 1600, height: 1600,
          note: "rolling green hills striped with vineyard rows, a red barn and trees under a blue sky with white clouds") { s in
        s.verticalGradient([(0, [0.78, 0.88, 0.97]), (0.55, [0.62, 0.79, 0.95]), (1, [0.22, 0.47, 0.84])], from: 0.45, to: 1)
        for _ in 0..<7 {
            let cu = s.rng.range(0.05, 0.95), cv = s.rng.range(0.66, 0.92)
            for _ in 0..<14 {
                s.glow(cu + s.rng.range(-0.08, 0.08), cv + s.rng.range(-0.015, 0.025), radius: s.rng.range(0.03, 0.07), [1, 1, 1], alpha: 0.75)
            }
        }
        s.ridge({ u in 0.52 + 0.04 * sin(u * 5.1 + 0.4) + 0.02 * sin(u * 13 + 1) }, low: [0.36, 0.55, 0.47], high: [0.52, 0.66, 0.62])
        s.ridge({ u in 0.42 + 0.06 * sin(u * 3.3 + 2.1) + 0.015 * sin(u * 11) }, low: [0.25, 0.5, 0.18], high: [0.42, 0.66, 0.26])
        let near: (Double) -> Double = { u in 0.3 + 0.08 * sin(u * 2.4 + 0.9) + 0.02 * sin(u * 7.7) }
        s.ridge(near, low: [0.3, 0.55, 0.12], high: [0.5, 0.75, 0.22])
        for row in 1..<16 {
            let drop = Double(row) * Double(row) * 0.0013
            s.line((0...100).map { i in let u = Double(i) / 100; return (u, near(u) - drop) }, width: 0.004 + Double(row) * 0.0006, [0.2, 0.4, 0.1], alpha: 0.7)
        }
        for _ in 0..<9 {
            let u = s.rng.range(0.02, 0.98), base = 0.42 + 0.06 * sin(u * 3.3 + 2.1) + 0.015 * sin(u * 11)
            s.line([(u, base - 0.01), (u, base + 0.012)], width: 0.004, [0.3, 0.2, 0.1])
            s.ellipse(u, base + 0.025, 0.016, 0.02, [0.12, 0.33, 0.1])
        }
        let bu = 0.68, bv = near(0.68) - 0.03
        s.polygon([(bu - 0.05, bv), (bu + 0.05, bv), (bu + 0.05, bv + 0.05), (bu, bv + 0.085), (bu - 0.05, bv + 0.05)], [0.68, 0.13, 0.1])
        s.polygon([(bu - 0.056, bv + 0.048), (bu, bv + 0.09), (bu + 0.056, bv + 0.048), (bu + 0.056, bv + 0.054), (bu, bv + 0.096), (bu - 0.056, bv + 0.054)], [0.95, 0.95, 0.92])
        s.polygon([(bu - 0.015, bv), (bu + 0.015, bv), (bu + 0.015, bv + 0.035), (bu - 0.015, bv + 0.035)], [0.35, 0.08, 0.06])
        s.texture(scale: 60, grain: 0.002) { _, v in v < 0.4 ? 0.04 : 0 }
    },
    Scene(name: "ocean-sunset", width: 1024, height: 1024,
          note: "the sun setting over a calm ocean, orange and purple sky, the sun's reflection on the water and a palm tree silhouette") { s in
        s.verticalGradient([(0, [1, 0.78, 0.35]), (0.25, [0.98, 0.52, 0.22]), (0.6, [0.75, 0.3, 0.42]), (1, [0.3, 0.16, 0.42])], from: 0.42, to: 1)
        s.glow(0.55, 0.5, radius: 0.35, [1, 0.85, 0.5], alpha: 0.8)
        s.ellipse(0.55, 0.49, 0.075, 0.075, [1, 0.95, 0.75])
        for _ in 0..<5 {
            let v = s.rng.range(0.6, 0.85), u = s.rng.range(0.1, 0.8)
            s.ellipse(u, v, s.rng.range(0.12, 0.25), 0.008, [0.55, 0.22, 0.35], alpha: 0.6)
        }
        s.c.saveGState()
        s.c.clip(to: CGRect(x: 0, y: 0, width: s.w, height: 0.42 * s.h))
        s.verticalGradient([(0, [0.08, 0.06, 0.2]), (1, [0.42, 0.22, 0.38])], from: 0, to: 0.42)
        for _ in 0..<900 {
            let v = s.rng.range(0, 0.42), spread = 0.04 + (0.42 - v) * 0.35
            let u = 0.55 + s.rng.range(-spread, spread)
            let bright = 1 - abs(u - 0.55) / spread
            s.line([(u - 0.01 - 0.02 * bright, v), (u + 0.01 + 0.02 * bright, v)], width: 0.002 + 0.002 * bright, [1, 0.6 + 0.35 * bright, 0.3 + 0.4 * bright], alpha: 0.35 + 0.5 * bright)
        }
        s.c.restoreGState()
        let trunk = (0...20).map { i -> (Double, Double) in let t = Double(i) / 20; return (0.12 + 0.1 * t * t, -0.02 + 0.72 * t) }
        s.line(trunk, width: 0.022, [0.06, 0.03, 0.06])
        for k in 0..<9 {
            let angle = Double(k) / 9 * 2 * .pi + 0.3, top = trunk.last!
            let frond = (0...12).map { i -> (Double, Double) in
                let t = Double(i) / 12
                return (top.0 + cos(angle) * 0.22 * t, top.1 + sin(angle) * 0.12 * t - 0.12 * t * t)
            }
            s.line(frond, width: 0.012, [0.06, 0.03, 0.06])
            for i in stride(from: 2, to: 12, by: 2) {
                let f = frond[i]
                s.line([f, (f.0 + 0.02 * cos(angle + 1), f.1 - 0.035)], width: 0.005, [0.06, 0.03, 0.06])
                s.line([f, (f.0 + 0.02 * cos(angle - 1), f.1 - 0.035)], width: 0.005, [0.06, 0.03, 0.06])
            }
        }
        for _ in 0..<5 {
            let u = s.rng.range(0.6, 0.9), v = s.rng.range(0.7, 0.85)
            s.line([(u - 0.02, v + 0.008), (u, v), (u + 0.02, v + 0.008)], width: 0.004, [0.12, 0.06, 0.12])
        }
        s.texture(scale: 40, grain: 0.002) { _, v in v < 0.42 ? 0.06 : 0.015 }
    },
    Scene(name: "coral-reef", width: 800, height: 800,
          note: "an underwater coral reef with orange and red corals, yellow and striped tropical fish and sun rays through blue water") { s in
        s.verticalGradient([(0, [0.02, 0.2, 0.38]), (0.6, [0.05, 0.45, 0.65]), (1, [0.3, 0.75, 0.85])])
        for _ in 0..<7 {
            let u = s.rng.range(0.1, 0.9), spread = s.rng.range(0.03, 0.07)
            s.polygon([(u - spread * 0.3, 1.01), (u + spread * 0.3, 1.01), (u + spread + 0.15, -0.01), (u - spread + 0.1, -0.01)], [0.8, 0.95, 1], alpha: 0.08)
        }
        s.ridge({ u in 0.14 + 0.03 * sin(u * 6 + 1) }, low: [0.62, 0.55, 0.38], high: [0.82, 0.74, 0.52])
        func coral(_ u: Double, _ v: Double, _ angle: Double, _ length: Double, _ width: Double, _ depth: Int, _ rgb: [Double]) {
            let end = (u + cos(angle) * length, v + sin(angle) * length)
            s.line([(u, v), end], width: width, rgb)
            if depth > 0 {
                for turn in [-0.45, 0.4] {
                    coral(end.0, end.1, angle + turn + s.rng.range(-0.15, 0.15), length * 0.72, width * 0.75, depth - 1, rgb)
                }
            } else {
                s.ellipse(end.0, end.1, width, width, rgb.map { min(1, $0 * 1.25) })
            }
        }
        let corals: [[Double]] = [[0.95, 0.42, 0.2], [0.85, 0.15, 0.25], [0.98, 0.6, 0.7], [0.6, 0.25, 0.65], [0.95, 0.55, 0.15]]
        for k in 0..<9 {
            let u = 0.06 + Double(k) * 0.11 + s.rng.range(-0.03, 0.03)
            coral(u, 0.12 + s.rng.range(0, 0.06), .pi / 2 + s.rng.range(-0.2, 0.2), s.rng.range(0.07, 0.11), 0.018, 4, corals[k % corals.count])
        }
        for k in 0..<6 {
            let u = s.rng.range(0.05, 0.95)
            s.ellipse(u, 0.13, 0.05, 0.035, k % 2 == 0 ? [0.45, 0.6, 0.3] : [0.75, 0.6, 0.3])
        }
        func fish(_ u: Double, _ v: Double, _ size: Double, _ facing: Double, _ body: [Double], stripes: Bool) {
            s.polygon([(u - facing * size * 0.9, v), (u - facing * size * 1.5, v + size * 0.45), (u - facing * size * 1.5, v - size * 0.45)], body)
            s.ellipse(u, v, size, size * 0.5, body)
            if stripes {
                for k in [-0.35, 0.15] { s.line([(u + k * size, v + size * 0.45), (u + k * size, v - size * 0.45)], width: size * 0.18, [1, 1, 1]) }
            }
            s.ellipse(u + facing * size * 0.6, v + size * 0.12, size * 0.1, size * 0.1, [0.05, 0.05, 0.05])
        }
        for k in 0..<11 {
            let palette: [[Double]] = [[1, 0.85, 0.1], [1, 0.5, 0.05], [0.2, 0.4, 0.95], [1, 0.85, 0.1]]
            fish(s.rng.range(0.1, 0.9), s.rng.range(0.32, 0.85), s.rng.range(0.03, 0.06), k % 2 == 0 ? 1 : -1, palette[k % 4], stripes: k % 4 == 1)
        }
        for _ in 0..<40 {
            let r = s.rng.range(0.003, 0.009)
            s.c.setStrokeColor(color([0.9, 0.97, 1], 0.6)); s.c.setLineWidth(0.0015 * s.w)
            s.c.strokeEllipse(in: CGRect(x: s.rng.range(0.55, 0.75) * s.w, y: s.rng.range(0.3, 1) * s.h, width: 2 * r * s.w, height: 2 * r * s.w))
        }
        s.texture(scale: 50, grain: 0.002) { _, v in v < 0.2 ? 0.125 : 0.03 }
    },
    Scene(name: "snowy-peaks", width: 1280, height: 1280,
          note: "snow-capped grey mountain peaks above a pine forest, reflected in a still alpine lake") { s in
        s.verticalGradient([(0, [0.85, 0.9, 0.95]), (1, [0.35, 0.58, 0.85])], from: 0.5, to: 1)
        func peaks(_ seed: Double) -> (Double) -> Double {
            { u in
                let a = abs(sin(u * 4.2 + seed)), b = abs(sin(u * 9.7 + seed * 2))
                return 0.52 + 0.25 * pow(a, 3) + 0.07 * pow(b, 2) + 0.012 * sin(u * 61 + seed)
            }
        }
        let back = peaks(1.3), front = peaks(0.2)
        s.ridge({ back($0) + 0.04 }, low: [0.55, 0.6, 0.7], high: [0.7, 0.74, 0.82])
        s.ridge(front, low: [0.32, 0.34, 0.38], high: [0.5, 0.52, 0.56])
        s.c.saveGState()
        var snow: [(Double, Double)] = []
        for i in 0...400 { let u = Double(i) / 400; snow.append((u, front(u))) }
        for i in stride(from: 400, through: 0, by: -1) {
            let u = Double(i) / 400
            snow.append((u, min(front(u), max(0.66 + 0.015 * sin(u * 37), front(u) - 0.08 - 0.02 * sin(u * 23)))))
        }
        s.polygon(snow, [0.96, 0.97, 1])
        s.c.restoreGState()
        s.ridge({ u in 0.5 + 0.01 * sin(u * 30) }, low: [0.1, 0.2, 0.12], high: [0.14, 0.28, 0.16])
        for i in 0..<160 {
            let u = Double(i) / 160 + s.rng.range(-0.004, 0.004), base = s.rng.range(0.46, 0.5), height = s.rng.range(0.03, 0.08)
            s.polygon([(u - height * 0.22, base), (u + height * 0.22, base), (u, base + height)], [0.05, s.rng.range(0.15, 0.25), 0.1])
        }
        s.c.saveGState()
        s.c.clip(to: CGRect(x: 0, y: 0, width: s.w, height: 0.46 * s.h))
        s.verticalGradient([(0, [0.2, 0.32, 0.45]), (1, [0.45, 0.6, 0.75])], from: 0, to: 0.46)
        s.c.translateBy(x: 0, y: 0.92 * s.h)
        s.c.scaleBy(x: 1, y: -1)
        s.ridge({ back($0) + 0.04 }, low: [0.3, 0.4, 0.55], high: [0.42, 0.5, 0.62])
        s.ridge(front, low: [0.2, 0.25, 0.35], high: [0.35, 0.4, 0.5])
        s.polygon(snow, [0.6, 0.68, 0.8])
        s.ridge({ u in 0.5 + 0.01 * sin(u * 30) }, low: [0.05, 0.12, 0.12], high: [0.08, 0.16, 0.15])
        s.c.restoreGState()
        for _ in 0..<300 {
            let v = s.rng.range(0, 0.45), u = s.rng.range(0, 1)
            s.line([(u, v), (u + s.rng.range(0.03, 0.12), v)], width: 0.0015, [0.75, 0.82, 0.92], alpha: 0.25)
        }
        s.texture(scale: 70, grain: 0.002) { _, v in v > 0.5 ? 0.09 : 0.025 }
    },
    Scene(name: "desert-dunes", width: 640, height: 640,
          note: "orange sand dunes with wind ripples, a green saguaro cactus and a hazy pale sky") { s in
        s.verticalGradient([(0, [0.98, 0.9, 0.75]), (1, [0.55, 0.75, 0.92])], from: 0.55, to: 1)
        s.glow(0.8, 0.88, radius: 0.18, [1, 1, 0.92], alpha: 0.9)
        s.ridge({ u in 0.58 + 0.06 * sin(u * 3 + 1) }, low: [0.82, 0.52, 0.28], high: [0.95, 0.7, 0.45])
        s.ridge({ u in 0.45 + 0.1 * sin(u * 2.2 + 3.5) }, low: [0.72, 0.36, 0.14], high: [0.96, 0.6, 0.3])
        s.ridge({ u in 0.28 + 0.12 * pow(abs(sin(u * 2.6 + 0.3)), 1.5) }, low: [0.62, 0.28, 0.1], high: [0.98, 0.58, 0.25])
        let cu = 0.3, base = 0.2
        s.line([(cu, base), (cu, base + 0.32)], width: 0.045, [0.22, 0.42, 0.2])
        s.line([(cu, base + 0.12), (cu - 0.08, base + 0.12), (cu - 0.08, base + 0.24)], width: 0.03, [0.22, 0.42, 0.2])
        s.line([(cu, base + 0.17), (cu + 0.07, base + 0.17), (cu + 0.07, base + 0.27)], width: 0.03, [0.22, 0.42, 0.2])
        s.line([(cu + 0.008, base + 0.02), (cu + 0.008, base + 0.31)], width: 0.006, [0.3, 0.55, 0.28])
        s.ellipse(cu, base - 0.005, 0.06, 0.01, [0.45, 0.2, 0.08], alpha: 0.5)
        s.shade { u, v, rgb in
            guard v < 0.6 else { return }
            let k = 1 + 0.07 * sin(v * 180 + 9 * sin(u * 7) + 4 * sin(u * 23))
            rgb = (rgb.0 * k, rgb.1 * k, rgb.2 * k)
        }
        s.texture(scale: 25, grain: 0.002) { _, v in v < 0.6 ? 0.05 : 0.01 }
    },
    Scene(name: "cherry-blossom", width: 1024, height: 1024,
          note: "pink cherry blossoms on dark branches against a pale blue spring sky, petals falling") { s in
        s.verticalGradient([(0, [0.86, 0.9, 0.96]), (1, [0.6, 0.76, 0.93])])
        for _ in 0..<25 { s.ellipse(s.rng.range(0, 1), s.rng.range(0, 1), 0.05, 0.05, [1, 0.85, 0.9], alpha: 0.25) }
        var tips: [(Double, Double)] = []
        func branch(_ u: Double, _ v: Double, _ angle: Double, _ length: Double, _ width: Double, _ depth: Int) {
            let mid = (u + cos(angle) * length * 0.5 + s.rng.range(-0.02, 0.02), v + sin(angle) * length * 0.5 + s.rng.range(-0.02, 0.02))
            let end = (u + cos(angle) * length, v + sin(angle) * length)
            s.line([(u, v), mid, end], width: width, [0.22, 0.13, 0.1])
            tips.append(mid)
            tips.append(end)
            guard depth > 0 else { return }
            for turn in [-0.5, 0.45] {
                branch(end.0, end.1, angle + turn + s.rng.range(-0.2, 0.2), length * s.rng.range(0.6, 0.8), width * 0.65, depth - 1)
            }
        }
        branch(-0.02, 0.15, 0.55, 0.32, 0.04, 5)
        branch(1.02, 0.9, .pi + 0.35, 0.25, 0.03, 4)
        func blossom(_ u: Double, _ v: Double, _ r: Double, _ rgb: [Double]) {
            let turn = s.rng.range(0, 1.3)
            for k in 0..<5 {
                let a = Double(k) / 5 * 2 * .pi + turn
                s.ellipse(u + cos(a) * r * 0.9, v + sin(a) * r * 0.9, r * 0.75, r * 0.75, rgb)
            }
            s.ellipse(u, v, r * 0.35, r * 0.35, [0.85, 0.3, 0.45])
        }
        for t in tips {
            for _ in 0..<5 {
                let pink = s.rng.range(0, 1)
                blossom(t.0 + s.rng.range(-0.04, 0.04), t.1 + s.rng.range(-0.04, 0.04), s.rng.range(0.008, 0.016), [1, 0.7 + 0.2 * pink, 0.8 + 0.12 * pink])
            }
        }
        for _ in 0..<60 { s.ellipse(s.rng.range(0, 1), s.rng.range(0, 1), 0.006, 0.004, [1, 0.75, 0.85], alpha: 0.9) }
        s.texture(scale: 30, grain: 0.002) { _, _ in 0.02 }
    },
    Scene(name: "lavender-field", width: 1920, height: 1920,
          note: "rows of purple lavender running to the horizon, a lone tree and a stone farmhouse under a warm evening sky") { s in
        s.verticalGradient([(0, [0.98, 0.8, 0.6]), (0.4, [0.9, 0.7, 0.75]), (1, [0.45, 0.5, 0.78])], from: 0.55, to: 1)
        s.ridge({ u in 0.56 + 0.015 * sin(u * 4) }, low: [0.42, 0.45, 0.55], high: [0.5, 0.5, 0.62])
        s.ridge({ _ in 0.55 }, low: [0.3, 0.4, 0.2], high: [0.45, 0.5, 0.3])
        let vanish = (0.5, 0.56)
        for k in -30...30 {
            let foot = 0.5 + Double(k) * 0.07
            let left = (foot - 0.02, -0.01), right = (foot + 0.02, -0.01)
            s.polygon([left, right, (vanish.0 + 0.001, vanish.1 - 0.005), (vanish.0 - 0.001, vanish.1 - 0.005)], [0.48, 0.32, 0.68])
        }
        for _ in 0..<5000 {
            let v = s.rng.range(0, 0.54), k = (s.rng.range(-30, 30)).rounded()
            let t = v / 0.56, foot = 0.5 + k * 0.07
            let u = foot + (vanish.0 - foot) * t + s.rng.range(-0.012, 0.012) * (1 - t)
            s.line([(u, v), (u + s.rng.range(-0.002, 0.002), v + 0.02 * (1 - t))], width: 0.003 * (1 - t) + 0.0005, [s.rng.range(0.5, 0.65), 0.35, s.rng.range(0.75, 0.88)], alpha: 0.8)
        }
        s.line([(0.78, 0.55), (0.78, 0.6)], width: 0.006, [0.2, 0.15, 0.12])
        for _ in 0..<12 { s.ellipse(0.78 + s.rng.range(-0.03, 0.03), 0.615 + s.rng.range(-0.015, 0.02), 0.022, 0.018, [0.18, 0.28, 0.14]) }
        s.polygon([(0.2, 0.55), (0.27, 0.55), (0.27, 0.578), (0.2, 0.578)], [0.85, 0.78, 0.65])
        s.polygon([(0.195, 0.578), (0.275, 0.578), (0.235, 0.597)], [0.7, 0.35, 0.25])
        s.texture(scale: 90, grain: 0.002) { _, v in v < 0.56 ? 0.1 : 0.015 }
    },
    Scene(name: "night-sky", width: 720, height: 720,
          note: "a starry night sky with the Milky Way and a crescent moon above black mountains and a lit cabin") { s in
        s.verticalGradient([(0, [0.08, 0.12, 0.28]), (1, [0.01, 0.02, 0.08])], from: 0.2, to: 1)
        let seed = s.seed
        s.shade { u, v, rgb in
            let band = exp(-pow((v - 0.35 - u * 0.55) * 4.5, 2))
            let cloud = band * fbm(u * 6, v * 6, seed: seed) * 0.45
            rgb = (rgb.0 + cloud * 0.8, rgb.1 + cloud * 0.75, rgb.2 + cloud)
        }
        for _ in 0..<700 {
            let r = pow(s.rng.next(), 6) * 0.004 + 0.0008
            s.ellipse(s.rng.range(0, 1), s.rng.range(0.25, 1), r, r, [1, 1, s.rng.range(0.8, 1)], alpha: s.rng.range(0.5, 1))
        }
        s.glow(0.76, 0.8, radius: 0.15, [0.9, 0.92, 1], alpha: 0.35)
        s.ellipse(0.76, 0.8, 0.065, 0.065, [0.97, 0.95, 0.85])
        s.ellipse(0.795, 0.815, 0.06, 0.06, [0.02, 0.04, 0.12])
        s.ridge({ u in 0.3 + 0.1 * abs(sin(u * 3.5 + 0.5)) + 0.02 * sin(u * 23) }, low: [0.02, 0.02, 0.04], high: [0.05, 0.06, 0.1])
        s.ridge({ u in 0.18 + 0.04 * sin(u * 5 + 2) }, low: [0, 0, 0], high: [0.02, 0.02, 0.03])
        s.polygon([(0.3, 0.17), (0.38, 0.17), (0.38, 0.215), (0.34, 0.245), (0.3, 0.215)], [0.03, 0.03, 0.04])
        s.polygon([(0.315, 0.18), (0.335, 0.18), (0.335, 0.2), (0.315, 0.2)], [1, 0.78, 0.35])
        s.glow(0.325, 0.19, radius: 0.04, [1, 0.7, 0.3], alpha: 0.4)
        s.texture(scale: 50, grain: 0.002) { _, _ in 0 }
    },
    Scene(name: "sunflower-field", width: 1024, height: 1024,
          note: "a field of bright yellow sunflowers with brown centres and green leaves under a summer sky") { s in
        s.verticalGradient([(0, [0.75, 0.88, 0.98]), (1, [0.25, 0.55, 0.92])], from: 0.6, to: 1)
        for _ in 0..<4 { s.ellipse(s.rng.range(0.1, 0.9), s.rng.range(0.8, 0.95), 0.09, 0.025, [1, 1, 1], alpha: 0.7) }
        s.ridge({ u in 0.62 + 0.01 * sin(u * 9) }, low: [0.25, 0.42, 0.12], high: [0.35, 0.5, 0.18])
        func sunflower(_ u: Double, _ v: Double, _ r: Double) {
            s.line([(u, v), (u + r * 0.2, v - 0.5)], width: r * 0.18, [0.25, 0.45, 0.12])
            s.ellipse(u + r * 0.8, v - r * 1.6, r * 0.7, r * 0.3, [0.2, 0.5, 0.15])
            let turn = s.rng.range(0, 1)
            for k in 0..<18 {
                let a = Double(k) / 18 * 2 * .pi + turn
                s.ellipse(u + cos(a) * r * 0.95, v + sin(a) * r * 0.95, r * 0.42, r * 0.42, [1, s.rng.range(0.72, 0.85), 0.05])
            }
            s.ellipse(u, v, r * 0.6, r * 0.6, [0.35, 0.2, 0.08])
            s.ellipse(u, v, r * 0.35, r * 0.35, [0.22, 0.12, 0.05])
        }
        var rows: [(Double, Double, Double)] = []
        for row in 0..<9 {
            let v = 0.6 - Double(row) * 0.075, r = 0.012 + Double(row) * 0.009
            var u = s.rng.range(-0.02, 0.04)
            while u < 1.05 { rows.append((u, v + s.rng.range(-0.02, 0.02), r)); u += r * 2.6 + s.rng.range(0, r) }
        }
        for f in rows { sunflower(f.0, f.1, f.2) }
        s.texture(scale: 40, grain: 0.002) { _, v in v < 0.62 ? 0.06 : 0.015 }
    },
    Scene(name: "city-skyline", width: 960, height: 960,
          note: "a grey city skyline at dusk with lit office windows, a neon HOTEL sign and the towers reflected in a river") { s in
        s.verticalGradient([(0, [0.75, 0.75, 0.78]), (0.5, [0.55, 0.57, 0.63]), (1, [0.3, 0.32, 0.4])], from: 0.35, to: 1)
        var towers: [(Double, Double, Double, [Double])] = []
        var u = -0.02
        while u < 1 {
            let width = s.rng.range(0.05, 0.11), top = s.rng.range(0.5, 0.9), grey = s.rng.range(0.25, 0.5)
            towers.append((u, width, top, [grey, grey, grey * 1.08]))
            u += width + s.rng.range(0.003, 0.02)
        }
        func drawTowers(lit: Bool) {
            for (u, width, top, rgb) in towers {
                s.polygon([(u, 0.35), (u + width, 0.35), (u + width, top), (u, top)], rgb)
                s.line([(u + width * 0.5, top), (u + width * 0.5, top + 0.03)], width: 0.002, rgb)
                var wv = 0.37
                while wv < top - 0.02 {
                    var wu = u + 0.008
                    while wu < u + width - 0.012 {
                        let on = s.rng.next() < 0.45
                        s.polygon([(wu, wv), (wu + 0.007, wv), (wu + 0.007, wv + 0.01), (wu, wv + 0.01)], on && lit ? [1, 0.85, 0.5] : [rgb[0] * 0.7, rgb[1] * 0.7, rgb[2] * 0.75])
                        wu += 0.013
                    }
                    wv += 0.018
                }
            }
        }
        drawTowers(lit: true)
        s.text("HOTEL", font: "Helvetica-Bold", size: 0.035, at: towers[3].0 + 0.003, 0.4, [1, 0.25, 0.4])
        s.c.saveGState()
        s.c.clip(to: CGRect(x: 0, y: 0, width: s.w, height: 0.35 * s.h))
        s.verticalGradient([(0, [0.18, 0.2, 0.25]), (1, [0.4, 0.42, 0.48])], from: 0, to: 0.35)
        s.c.translateBy(x: 0, y: 0.7 * s.h)
        s.c.scaleBy(x: 1, y: -1)
        s.c.setAlpha(0.45)
        drawTowers(lit: true)
        s.c.restoreGState()
        for _ in 0..<250 {
            let v = s.rng.range(0, 0.34), u = s.rng.range(0, 1)
            s.line([(u, v), (u + s.rng.range(0.02, 0.08), v)], width: 0.0015, [0.15, 0.17, 0.22], alpha: 0.5)
        }
        s.texture(scale: 60, grain: 0.002) { _, _ in 0.025 }
    },
]

for (i, scene) in scenes.enumerated() {
    let canvas = Canvas(scene.width, scene.height, seed: UInt64(1000 + i))
    scene.draw(canvas)
    canvas.soften(radius: max(1, scene.width / 500))
    let file = String(format: "scenes/%02d-%@.jpg", i, scene.name)
    write(canvas.c.makeImage()!, to: root.appendingPathComponent(file), type: .jpeg, quality: 0.92)
    record(file, "scene", "procedural scene drawn by this script: " + scene.note)
}

// Near-duplicate variants of the scenes.
let variants: [(String, String)] = [("resized-0.5", "resized to half width and height"), ("cropped-10", "central crop keeping 90 percent of each side"),
                                    ("jpeg-q40", "re-encoded as JPEG quality 40"), ("flipped", "flipped horizontally"), ("colour-shift", "warmer colour cast (red +12, blue -12 per channel)"),
                                    ("resized-0.33", "resized to a third"), ("cropped-10-top-left", "top-left crop keeping 90 percent of each side"),
                                    ("jpeg-q40", "re-encoded as JPEG quality 40"), ("flipped", "flipped horizontally"), ("colour-shift", "warmer colour cast (red +12, blue -12 per channel)")]
func sceneFile(_ i: Int) -> String { String(format: "scenes/%02d-%@.jpg", i, scenes[i].name) }
for (i, scene) in scenes.enumerated() {
    let src = load(root.appendingPathComponent(sceneFile(i)))
    let (variant, note) = variants[i]
    let w = src.width, h = src.height
    var out: CGImage
    var quality = 0.92
    switch variant {
    case "resized-0.5", "resized-0.33":
        let f = variant == "resized-0.5" ? 0.5 : 1.0 / 3.0
        let nw = Int(Double(w) * f), nh = Int(Double(h) * f)
        let c = context(nw, nh); c.interpolationQuality = .high; c.draw(src, in: CGRect(x: 0, y: 0, width: nw, height: nh)); out = c.makeImage()!
    case "cropped-10":
        let cw = w * 9 / 10, ch = h * 9 / 10
        out = src.cropping(to: CGRect(x: (w - cw) / 2, y: (h - ch) / 2, width: cw, height: ch))!
    case "cropped-10-top-left":
        out = src.cropping(to: CGRect(x: 0, y: 0, width: w * 9 / 10, height: h * 9 / 10))!
    case "jpeg-q40":
        out = src; quality = 0.4
    case "flipped":
        let c = context(w, h); c.translateBy(x: CGFloat(w), y: 0); c.scaleBy(x: -1, y: 1); c.draw(src, in: CGRect(x: 0, y: 0, width: w, height: h)); out = c.makeImage()!
    case "colour-shift":
        let c = context(w, h); c.draw(src, in: CGRect(x: 0, y: 0, width: w, height: h))
        let p = c.data!.assumingMemoryBound(to: UInt8.self)
        for i in stride(from: 0, to: w * h * 4, by: 4) { p[i] = UInt8(min(255, Int(p[i]) + 12)); p[i + 2] = UInt8(max(0, Int(p[i + 2]) - 12)) }
        out = c.makeImage()!
    default: fatalError()
    }
    let file = String(format: "variants/%02d-%@-%@.jpg", i, scene.name, variant)
    write(out, to: root.appendingPathComponent(file), type: .jpeg, quality: quality)
    record(file, "variant", note, source: sceneFile(i), variant: variant)
}

// JPEG encodings for the decoder tests, from libjpeg-turbo's cjpeg (baseline without restart
// markers unless asked, the sampling factors, progressive, grey, 12-bit, arithmetic coding) and
// ImageMagick (four-component YCCK, an odd size).
func run(_ args: [String]) {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    p.arguments = args
    try! p.run()
    p.waitUntilExit()
    precondition(p.terminationStatus == 0, "failed: \(args.joined(separator: " "))")
}
let scratch = fm.temporaryDirectory.appendingPathComponent("make-fixture-images-\(getpid())")
try! fm.createDirectory(at: scratch, withIntermediateDirectories: true)
defer { try? fm.removeItem(at: scratch) }
func ppm(_ file: String) -> String {
    let out = scratch.appendingPathComponent((file as NSString).lastPathComponent + ".ppm").path
    run(["djpeg", "-outfile", out, root.appendingPathComponent(file).path])
    return out
}
let dunes = ppm(sceneFile(4)), headline = ppm("synthetic/06-text-times.jpg"), blossomThird = ppm("variants/05-\(scenes[5].name)-resized-0.33.jpg")
let cjpegFiles: [(String, [String], String)] = [
    ("jpeg-variants/411-\(scenes[4].name).jpg", ["-sample", "4x1"], dunes),
    ("jpeg-variants/422-\(scenes[4].name).jpg", ["-sample", "2x1"], dunes),
    ("jpeg-variants/440-\(scenes[4].name).jpg", ["-sample", "1x2"], dunes),
    ("jpeg-variants/444-\(scenes[4].name).jpg", ["-sample", "1x1"], dunes),
    ("jpeg-variants/grey-\(scenes[4].name).jpg", ["-grayscale"], dunes),
    ("jpeg-variants/progressive-420-\(scenes[4].name).jpg", ["-progressive", "-sample", "2x2"], dunes),
    ("jpeg-variants/progressive-444-\(scenes[4].name).jpg", ["-progressive", "-sample", "1x1"], dunes),
    ("jpeg-variants/restart-\(scenes[4].name).jpg", ["-restart", "1"], dunes),
    ("jpeg-variants/420-odd-text-times.jpg", [], headline),
    ("jpeg-unsupported/12-bit-\(scenes[5].name).jpg", ["-precision", "12"], blossomThird),
    ("jpeg-unsupported/arithmetic-\(scenes[5].name).jpg", ["-arithmetic"], blossomThird),
]
for (file, flags, input) in cjpegFiles {
    run(["cjpeg", "-quality", "85"] + flags + ["-outfile", root.appendingPathComponent(file).path, input])
}
run(["magick", root.appendingPathComponent(sceneFile(2)).path, "-colorspace", "CMYK", "-quality", "80",
     root.appendingPathComponent("jpeg-variants/cmyk-\(scenes[2].name).jpg").path])
run(["magick", root.appendingPathComponent(sceneFile(0)).path, "-resize", "1203x797!", "-quality", "75",
     root.appendingPathComponent("jpeg-variants/odd-size-\(scenes[0].name).jpg").path])

let data = try! JSONSerialization.data(withJSONObject: manifest.sorted { ($0["file"] as! String) < ($1["file"] as! String) }, options: [.prettyPrinted, .sortedKeys])
try! data.write(to: root.appendingPathComponent("manifest.json"))
print("wrote \(manifest.count) manifest entries")

if let package {
    let limit = 250_000
    let images = (manifest.map { $0["file"] as! String } + cjpegFiles.map(\.0) +
                  ["jpeg-variants/cmyk-\(scenes[2].name).jpg", "jpeg-variants/odd-size-\(scenes[0].name).jpg"]).sorted()
    for file in ["manifest.json"] + images {
        let from = root.appendingPathComponent(file), to = package.appendingPathComponent(file)
        let size = (try! fm.attributesOfItem(atPath: from.path)[.size] as! NSNumber).intValue
        try? fm.removeItem(at: to)
        guard file == "manifest.json" || size < limit else { print("local only (\(size) bytes): \(file)"); continue }
        try! fm.createDirectory(at: to.deletingLastPathComponent(), withIntermediateDirectories: true)
        try! fm.copyItem(at: from, to: to)
    }
}
