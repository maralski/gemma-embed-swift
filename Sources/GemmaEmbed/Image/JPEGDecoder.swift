import Foundation

/// A Swift port of stb_image 2.30's JPEG decoder as llama.cpp's libmtmd compiles it (no SIMD
/// kernels), producing the bytes of `stbi_load_from_memory(data, count, &x, &y, &n, 3)`.
///
/// Every integer step follows stb: the Huffman fast tables, the IDCT's fixed-point constants
/// and biases, the fancy upsamplers' rounding and the fixed-point YCbCr conversion. stb also
/// keeps what it decoded when a file ends early or carries an unexpected marker after the
/// frame header, and so does this port.
enum JPEGDecoder {
    /// Why stb_image refuses a file, in its own words.
    struct Rejection: Error, CustomStringConvertible {
        let reason: String
        /// A valid JPEG that stb_image does not implement (12-bit, lossless, hierarchical or
        /// arithmetic coded, or with its height deferred to a DNL marker), as opposed to a corrupt one.
        let isUnsupportedEncoding: Bool

        static func corrupt(_ reason: String) -> Rejection { Rejection(reason: reason, isUnsupportedEncoding: false) }
        static func unsupported(_ reason: String) -> Rejection { Rejection(reason: reason, isUnsupportedEncoding: true) }

        var description: String { reason }
    }

    static func decode(_ data: Data) throws -> RGBImage {
        try data.withUnsafeBytes { raw in
            var parser = Parser(bytes: raw.bindMemory(to: UInt8.self))
            try parser.decodeImage()
            return try parser.reconstruct()
        }
    }

    /// Where the coefficient at position `k` of the zigzag order sits in the row-major block;
    /// the 15 trailing entries let corrupt runs index past the end, as stb's table does.
    static let dezigzag: [UInt8] = [
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
        63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
    ]

    static let fastBits = 9
    static let noMarker: UInt8 = 0xFF

    /// Zeroed storage that the hot loops address through a pointer, freed with its owner.
    final class Buffer<Element> {
        let base: UnsafeMutablePointer<Element>
        let count: Int

        init(count: Int, repeating value: Element) {
            base = .allocate(capacity: max(count, 1))
            base.initialize(repeating: value, count: count)
            self.count = count
        }

        init(_ values: [Element]) {
            count = values.count
            base = .allocate(capacity: max(count, 1))
            _ = UnsafeMutableBufferPointer(start: base, count: count).initialize(from: values)
        }

        deinit {
            base.deinitialize(count: count)
            base.deallocate()
        }
    }

    /// One Huffman table with stb's 9-bit lookup. A table the file never defines stays zeroed,
    /// as in stb's zero-filled decoder state.
    final class HuffmanTable {
        /// Symbol index for each 9-bit prefix, or 255 when the code is longer.
        let fast = Buffer<UInt8>(count: 1 << JPEGDecoder.fastBits, repeating: 0)
        let code = Buffer<UInt16>(count: 256, repeating: 0)
        let values = Buffer<UInt8>(count: 256, repeating: 0)
        let size = Buffer<UInt8>(count: 257, repeating: 0)
        /// Largest code plus one for each length, shifted to 16 bits.
        let maxcode = Buffer<UInt32>(count: 18, repeating: 0)
        /// Symbol index minus code for each length.
        let delta = Buffer<Int32>(count: 17, repeating: 0)
        /// AC only: run, magnitude and combined length of short codes decoded in one lookup.
        let fastAC = Buffer<Int16>(count: 1 << JPEGDecoder.fastBits, repeating: 0)

        /// `stbi__build_huffman`: `counts[i]` codes of length `i + 1`.
        func build(counts: [Int]) throws {
            let size = size.base, code = code.base, fast = fast.base
            var k = 0
            for i in 0..<16 {
                for _ in 0..<counts[i] {
                    size[k] = UInt8(i + 1)
                    k += 1
                    if k >= 257 { throw Rejection.corrupt("bad size list") }
                }
            }
            size[k] = 0

            var next: UInt32 = 0
            k = 0
            for j in 1...16 {
                delta.base[j] = Int32(truncatingIfNeeded: k - Int(next))
                if Int(size[k]) == j {
                    while Int(size[k]) == j {
                        code[k] = UInt16(truncatingIfNeeded: next)
                        k += 1
                        next &+= 1
                    }
                    if next &- 1 >= UInt32(1) << UInt32(j) { throw Rejection.corrupt("bad code lengths") }
                }
                maxcode.base[j] = next << UInt32(16 - j)
                next <<= 1
            }
            maxcode.base[17] = 0xFFFF_FFFF

            fast.update(repeating: 255, count: 1 << JPEGDecoder.fastBits)
            for i in 0..<k {
                let s = Int(size[i])
                if s <= JPEGDecoder.fastBits {
                    let c = Int(code[i]) << (JPEGDecoder.fastBits - s)
                    for j in 0..<(1 << (JPEGDecoder.fastBits - s)) {
                        fast[c + j] = UInt8(i)
                    }
                }
            }
        }

        /// `stbi__build_fast_ac`, run once the values are read.
        func buildFastAC() {
            let bits = JPEGDecoder.fastBits
            for i in 0..<(1 << bits) {
                let index = fast.base[i]
                fastAC.base[i] = 0
                guard index < 255 else { continue }
                let rs = Int(values.base[Int(index)])
                let run = (rs >> 4) & 15
                let magnitudeBits = rs & 15
                let length = Int(size.base[Int(index)])
                guard magnitudeBits != 0, length + magnitudeBits <= bits else { continue }
                var k = ((i << length) & ((1 << bits) - 1)) >> (bits - magnitudeBits)
                if k < 1 << (magnitudeBits - 1) { k += 1 - (1 << magnitudeBits) }
                if k >= -128 && k <= 127 {
                    fastAC.base[i] = Int16(k * 256 + run * 16 + length + magnitudeBits)
                }
            }
        }
    }

    /// One colour component of the frame and the planes it decodes into.
    struct Component {
        let id: UInt8
        let h: Int
        let v: Int
        let quantTable: Int
        var dcTable = 0
        var acTable = 0
        var dcPrediction: Int32 = 0
        /// Samples this component covers (stb's `x`, `y`).
        let width: Int
        let height: Int
        /// The plane, padded to whole MCUs (stb's `w2`, `h2`).
        let stride: Int
        let samples: Buffer<UInt8>
        /// Progressive only: 64 coefficients per block, `stride / 8` blocks per row.
        let coefficients: Buffer<Int16>?

        var blocksPerRow: Int { stride / 8 }
    }

    /// The bytes after the marker segments, read as `stbi__get8` does: zero past the end.
    struct ByteStream {
        let bytes: UnsafeBufferPointer<UInt8>
        var position = 0

        var atEnd: Bool { position >= bytes.count }

        @inline(__always)
        mutating func byte() -> UInt8 {
            guard position < bytes.count else { return 0 }
            let b = bytes[position]
            position += 1
            return b
        }

        mutating func uint16() -> Int {
            let high = Int(byte())
            return (high << 8) + Int(byte())
        }

        mutating func skip(_ n: Int) {
            if n == 0 { return }
            position = n < 0 ? bytes.count : position + n
        }
    }

    struct Parser {
        var stream: ByteStream

        var codeBuffer: UInt32 = 0
        var codeBits = 0
        var marker = JPEGDecoder.noMarker
        var noMore = false

        let huffmanDC = (0..<4).map { _ in HuffmanTable() }
        let huffmanAC = (0..<4).map { _ in HuffmanTable() }
        let dequant = Buffer<UInt16>(count: 4 * 64, repeating: 0)
        let zigzag = Buffer<UInt8>(JPEGDecoder.dezigzag)
        let block = Buffer<Int16>(count: 64, repeating: 0)
        let idctScratch = Buffer<Int32>(count: 64, repeating: 0)

        var components: [Component] = []
        var width = 0
        var height = 0
        var hMax = 1
        var vMax = 1
        var mcuColumns = 0
        var mcuRows = 0
        var progressive = false
        var rgbIDs = 0
        var jfif = false
        var adobeTransform = -1

        var scanOrder: [Int] = []
        var spectralStart = 0
        var spectralEnd = 0
        var successiveHigh = 0
        var successiveLow = 0
        var eobRun = 0
        var restartInterval = 0
        var todo = 0

        init(bytes: UnsafeBufferPointer<UInt8>) {
            stream = ByteStream(bytes: bytes)
        }

        // MARK: Markers

        /// `stbi__get_marker`: the marker the entropy decoder stopped at, else the next one in the stream.
        mutating func nextMarker() -> UInt8 {
            if marker != JPEGDecoder.noMarker {
                defer { marker = JPEGDecoder.noMarker }
                return marker
            }
            var x = stream.byte()
            if x != 0xFF { return JPEGDecoder.noMarker }
            while x == 0xFF { x = stream.byte() }
            return x
        }

        static func isRestart(_ m: UInt8) -> Bool { m >= 0xD0 && m <= 0xD7 }
        static func isFrame(_ m: UInt8) -> Bool { m == 0xC0 || m == 0xC1 || m == 0xC2 }

        /// Frame types stb_image reports as an unknown marker: lossless, hierarchical and
        /// arithmetic coded frames, and the arithmetic conditioning table.
        static func isUnsupportedFrame(_ m: UInt8) -> Bool {
            switch m {
            case 0xC3, 0xC5...0xC7, 0xC9...0xCF: return true
            default: return false
            }
        }

        /// `stbi__decode_jpeg_image`.
        mutating func decodeImage() throws {
            restartInterval = 0
            jfif = false
            adobeTransform = -1
            marker = JPEGDecoder.noMarker
            if nextMarker() != 0xD8 { throw Rejection.corrupt("no SOI") }
            var m = nextMarker()
            while !Self.isFrame(m) {
                try processMarker(m)
                m = nextMarker()
                while m == JPEGDecoder.noMarker {
                    if stream.atEnd { throw Rejection.corrupt("no SOF") }
                    m = nextMarker()
                }
            }
            progressive = m == 0xC2
            try readFrameHeader()

            m = nextMarker()
            while m != 0xD9 {
                if m == 0xDA {
                    try readScanHeader()
                    try decodeScan()
                    if marker == JPEGDecoder.noMarker { marker = skipJunkAtEnd() }
                    m = nextMarker()
                    if Self.isRestart(m) { m = nextMarker() }
                } else if m == 0xDC {
                    let length = stream.uint16()
                    let lines = stream.uint16()
                    if length != 4 { throw Rejection.corrupt("bad DNL len") }
                    if lines != height { throw Rejection.corrupt("bad DNL height") }
                    m = nextMarker()
                } else {
                    do {
                        try processMarker(m)
                    } catch {
                        // stb returns what it has decoded and skips the progressive finish.
                        return
                    }
                    m = nextMarker()
                }
            }
            if progressive { finishProgressive() }
        }

        /// `stbi__process_marker`: tables, restart interval, APPn and comments.
        mutating func processMarker(_ m: UInt8) throws {
            switch m {
            case JPEGDecoder.noMarker:
                throw Rejection.corrupt("expected marker")

            case 0xDD:
                if stream.uint16() != 4 { throw Rejection.corrupt("bad DRI len") }
                restartInterval = stream.uint16()

            case 0xDB:
                var length = stream.uint16() - 2
                while length > 0 {
                    let q = Int(stream.byte())
                    let precision = q >> 4, table = q & 15
                    if precision != 0 && precision != 1 { throw Rejection.corrupt("bad DQT type") }
                    if table > 3 { throw Rejection.corrupt("bad DQT table") }
                    for i in 0..<64 {
                        let value = precision != 0 ? UInt16(truncatingIfNeeded: stream.uint16()) : UInt16(stream.byte())
                        dequant.base[table * 64 + Int(zigzag.base[i])] = value
                    }
                    length -= precision != 0 ? 129 : 65
                }
                // stb fails here without naming a reason.
                if length != 0 { throw Rejection.corrupt("Corrupt JPEG") }

            case 0xC4:
                var length = stream.uint16() - 2
                while length > 0 {
                    let q = Int(stream.byte())
                    let tableClass = q >> 4, index = q & 15
                    if tableClass > 1 || index > 3 { throw Rejection.corrupt("bad DHT header") }
                    var counts = [Int](repeating: 0, count: 16)
                    for i in 0..<16 { counts[i] = Int(stream.byte()) }
                    let n = counts.reduce(0, +)
                    if n > 256 { throw Rejection.corrupt("bad DHT header") }
                    length -= 17
                    let table = tableClass == 0 ? huffmanDC[index] : huffmanAC[index]
                    try table.build(counts: counts)
                    for i in 0..<n { table.values.base[i] = stream.byte() }
                    if tableClass != 0 { table.buildFastAC() }
                    length -= n
                }
                if length != 0 { throw Rejection.corrupt("Corrupt JPEG") }

            case 0xE0...0xEF, 0xFE:
                var length = stream.uint16()
                if length < 2 { throw Rejection.corrupt(m == 0xFE ? "bad COM len" : "bad APP len") }
                length -= 2
                if m == 0xE0 && length >= 5 {
                    if readTag([0x4A, 0x46, 0x49, 0x46, 0]) { jfif = true }
                    length -= 5
                } else if m == 0xEE && length >= 12 {
                    let ok = readTag([0x41, 0x64, 0x6F, 0x62, 0x65, 0])
                    length -= 6
                    if ok {
                        _ = stream.byte()
                        _ = stream.uint16()
                        _ = stream.uint16()
                        adobeTransform = Int(stream.byte())
                        length -= 6
                    }
                }
                stream.skip(length)

            default:
                if Self.isUnsupportedFrame(m) { throw Rejection.unsupported("unknown marker") }
                throw Rejection.corrupt("unknown marker")
            }
        }

        /// Reads `tag.count` bytes, all of them, and reports whether they spell `tag`.
        mutating func readTag(_ tag: [UInt8]) -> Bool {
            var ok = true
            for expected in tag where stream.byte() != expected { ok = false }
            return ok
        }

        /// `stbi__process_frame_header`, which also allocates the planes.
        mutating func readFrameHeader() throws {
            let length = stream.uint16()
            if length < 11 { throw Rejection.corrupt("bad SOF len") }
            if stream.byte() != 8 { throw Rejection.unsupported("only 8-bit") }
            height = stream.uint16()
            if height == 0 { throw Rejection.unsupported("no header height") }
            width = stream.uint16()
            if width == 0 { throw Rejection.corrupt("0 width") }
            let count = Int(stream.byte())
            if count != 3 && count != 1 && count != 4 { throw Rejection.corrupt("bad component count") }
            if length != 8 + 3 * count { throw Rejection.corrupt("bad SOF len") }

            rgbIDs = 0
            var specs: [(id: UInt8, h: Int, v: Int, tq: Int)] = []
            for i in 0..<count {
                let id = stream.byte()
                if count == 3 && id == [0x52, 0x47, 0x42][i] { rgbIDs += 1 }
                let q = Int(stream.byte())
                let h = q >> 4
                if h == 0 || h > 4 { throw Rejection.corrupt("bad H") }
                let v = q & 15
                if v == 0 || v > 4 { throw Rejection.corrupt("bad V") }
                let tq = Int(stream.byte())
                if tq > 3 { throw Rejection.corrupt("bad TQ") }
                specs.append((id, h, v, tq))
            }

            if width * height > Int(Int32.max) || width * height * count > Int(Int32.max) {
                throw Rejection.corrupt("too large")
            }
            hMax = specs.map(\.h).max()!
            vMax = specs.map(\.v).max()!
            for spec in specs {
                if hMax % spec.h != 0 { throw Rejection.corrupt("bad H") }
                if vMax % spec.v != 0 { throw Rejection.corrupt("bad V") }
            }
            mcuColumns = (width + hMax * 8 - 1) / (hMax * 8)
            mcuRows = (height + vMax * 8 - 1) / (vMax * 8)

            components = try specs.map { spec in
                let stride = mcuColumns * spec.h * 8
                let rows = mcuRows * spec.v * 8
                if stride * rows + 15 > Int(Int32.max) { throw Rejection.corrupt("outofmem") }
                if progressive && stride * rows * 2 + 15 > Int(Int32.max) { throw Rejection.corrupt("outofmem") }
                return Component(id: spec.id, h: spec.h, v: spec.v, quantTable: spec.tq,
                                 width: (width * spec.h + hMax - 1) / hMax, height: (height * spec.v + vMax - 1) / vMax,
                                 stride: stride, samples: Buffer(count: stride * rows, repeating: 0),
                                 coefficients: progressive ? Buffer(count: stride * rows, repeating: 0) : nil)
            }
        }

        /// `stbi__process_scan_header`.
        mutating func readScanHeader() throws {
            let length = stream.uint16()
            let count = Int(stream.byte())
            if count < 1 || count > 4 || count > components.count { throw Rejection.corrupt("bad SOS component count") }
            if length != 6 + 2 * count { throw Rejection.corrupt("bad SOS len") }
            scanOrder = []
            for _ in 0..<count {
                let id = stream.byte()
                let q = Int(stream.byte())
                // stb fails on an unknown component id without naming a reason.
                guard let which = components.firstIndex(where: { $0.id == id }) else { throw Rejection.corrupt("Corrupt JPEG") }
                if q >> 4 > 3 { throw Rejection.corrupt("bad DC huff") }
                if q & 15 > 3 { throw Rejection.corrupt("bad AC huff") }
                components[which].dcTable = q >> 4
                components[which].acTable = q & 15
                scanOrder.append(which)
            }
            spectralStart = Int(stream.byte())
            spectralEnd = Int(stream.byte())
            let approximation = Int(stream.byte())
            successiveHigh = approximation >> 4
            successiveLow = approximation & 15
            if progressive {
                if spectralStart > 63 || spectralEnd > 63 || spectralStart > spectralEnd || successiveHigh > 13 || successiveLow > 13 {
                    throw Rejection.corrupt("bad SOS")
                }
            } else {
                if spectralStart != 0 { throw Rejection.corrupt("bad SOS") }
                if successiveHigh != 0 || successiveLow != 0 { throw Rejection.corrupt("bad SOS") }
                spectralEnd = 63
            }
        }

        /// `stbi__skip_jpeg_junk_at_end`: the next real marker after a scan, if any.
        mutating func skipJunkAtEnd() -> UInt8 {
            while !stream.atEnd {
                var x = stream.byte()
                while x == 0xFF {
                    if stream.atEnd { return JPEGDecoder.noMarker }
                    x = stream.byte()
                    if x != 0x00 && x != 0xFF { return x }
                }
            }
            return JPEGDecoder.noMarker
        }

        // MARK: Entropy-coded data

        /// `stbi__grow_buffer_unsafe`: fill the bit buffer past 24 bits, unstuffing FF00 and
        /// stopping at a marker, after which it feeds zeros.
        mutating func fillBits() {
            repeat {
                let b: UInt32 = noMore ? 0 : UInt32(stream.byte())
                if b == 0xFF {
                    var c = stream.byte()
                    while c == 0xFF { c = stream.byte() }
                    if c != 0 {
                        marker = c
                        noMore = true
                        return
                    }
                }
                codeBuffer |= b &<< UInt32(truncatingIfNeeded: 24 - codeBits)
                codeBits += 8
            } while codeBits <= 24
        }

        /// `stbi__jpeg_huff_decode`: the next symbol, or -1 for a bad code.
        @inline(__always)
        mutating func decodeSymbol(_ table: HuffmanTable) -> Int {
            if codeBits < 16 { fillBits() }
            let c = Int(codeBuffer >> UInt32(32 - JPEGDecoder.fastBits)) & ((1 << JPEGDecoder.fastBits) - 1)
            let fastIndex = Int(table.fast.base[c])
            if fastIndex < 255 {
                let s = Int(table.size.base[fastIndex])
                if s > codeBits { return -1 }
                codeBuffer <<= UInt32(s)
                codeBits -= s
                return Int(table.values.base[fastIndex])
            }
            let temp = codeBuffer >> 16
            var k = JPEGDecoder.fastBits + 1
            while k < 17 && temp >= table.maxcode.base[k] { k += 1 }
            if k == 17 {
                codeBits -= 16
                return -1
            }
            if k > codeBits { return -1 }
            let index = Int((codeBuffer >> UInt32(32 - k)) & ((1 << UInt32(k)) - 1)) + Int(table.delta.base[k])
            if index < 0 || index >= 256 { return -1 }
            codeBits -= k
            codeBuffer <<= UInt32(k)
            return Int(table.values.base[index])
        }

        /// `stbi__extend_receive`: `n` bits as a signed value.
        @inline(__always)
        mutating func receiveExtend(_ n: Int) -> Int {
            if codeBits < n { fillBits() }
            if codeBits < n { return 0 }
            let negative = codeBuffer >> 31 == 0
            let mask = (UInt32(1) << UInt32(n)) - 1
            let rotated = (codeBuffer << UInt32(n)) | (codeBuffer >> UInt32(32 - n))
            codeBuffer = rotated & ~mask
            codeBits -= n
            let k = Int(rotated & mask)
            return negative ? k + 1 - (1 << n) : k
        }

        /// `stbi__jpeg_get_bits`.
        mutating func bits(_ n: Int) -> Int {
            if codeBits < n { fillBits() }
            if codeBits < n { return 0 }
            let mask = (UInt32(1) << UInt32(n)) - 1
            let rotated = (codeBuffer << UInt32(n)) | (codeBuffer >> UInt32(32 - n))
            codeBuffer = rotated & ~mask
            codeBits -= n
            return Int(rotated & mask)
        }

        /// `stbi__jpeg_get_bit`.
        @inline(__always)
        mutating func bit() -> Bool {
            if codeBits < 1 { fillBits() }
            if codeBits < 1 { return false }
            let k = codeBuffer
            codeBuffer <<= 1
            codeBits -= 1
            return k & 0x8000_0000 != 0
        }

        /// `stbi__jpeg_reset`, at the start of a scan and after each restart marker.
        mutating func resetEntropy() {
            codeBits = 0
            codeBuffer = 0
            noMore = false
            for i in components.indices { components[i].dcPrediction = 0 }
            marker = JPEGDecoder.noMarker
            todo = restartInterval != 0 ? restartInterval : 0x7FFF_FFFF
            eobRun = 0
        }

        /// Counts down the restart interval after an MCU; false ends the scan, as stb does when
        /// the expected restart marker is missing.
        @inline(__always)
        mutating func continueAfterMCU() -> Bool {
            todo -= 1
            guard todo <= 0 else { return true }
            if codeBits < 24 { fillBits() }
            if !Self.isRestart(marker) { return false }
            resetEntropy()
            return true
        }

        /// `stbi__parse_entropy_coded_data`.
        mutating func decodeScan() throws {
            resetEntropy()
            if scanOrder.count == 1 {
                let n = scanOrder[0]
                let component = components[n]
                let blocksX = (component.width + 7) >> 3, blocksY = (component.height + 7) >> 3
                for j in 0..<blocksY {
                    for i in 0..<blocksX {
                        if let coefficients = component.coefficients {
                            let data = coefficients.base + 64 * (i + j * component.blocksPerRow)
                            if spectralStart == 0 {
                                try decodeProgressiveDC(data, component: n)
                            } else {
                                try decodeProgressiveAC(data, table: huffmanAC[component.acTable])
                            }
                        } else {
                            try decodeBlock(component: n)
                            JPEGDecoder.idct(block.base, into: component.samples.base + component.stride * j * 8 + i * 8,
                                             stride: component.stride, scratch: idctScratch.base)
                        }
                        if !continueAfterMCU() { return }
                    }
                }
            } else {
                for j in 0..<mcuRows {
                    for i in 0..<mcuColumns {
                        for n in scanOrder {
                            let component = components[n]
                            for y in 0..<component.v {
                                for x in 0..<component.h {
                                    let blockX = i * component.h + x, blockY = j * component.v + y
                                    if let coefficients = component.coefficients {
                                        try decodeProgressiveDC(coefficients.base + 64 * (blockX + blockY * component.blocksPerRow), component: n)
                                    } else {
                                        try decodeBlock(component: n)
                                        JPEGDecoder.idct(block.base, into: component.samples.base + component.stride * blockY * 8 + blockX * 8,
                                                         stride: component.stride, scratch: idctScratch.base)
                                    }
                                }
                            }
                        }
                        if !continueAfterMCU() { return }
                    }
                }
            }
        }

        /// The new DC prediction, as stb validates it.
        static func predict(_ prediction: Int32, _ diff: Int) throws -> Int32 {
            let (sum, overflow) = prediction.addingReportingOverflow(Int32(truncatingIfNeeded: diff))
            if overflow { throw Rejection.corrupt("bad delta") }
            return sum
        }

        /// `stbi__mul2shorts_valid`.
        static func productFitsInShort(_ a: Int, _ b: Int) -> Bool {
            if b == 0 || b == -1 { return true }
            if (a >= 0) == (b >= 0) { return a <= Int(Int16.max) / b }
            if b < 0 { return a <= Int(Int16.min) / b }
            return a >= Int(Int16.min) / b
        }

        /// `stbi__jpeg_decode_block`: one baseline block, dequantised, into `block`.
        mutating func decodeBlock(component n: Int) throws {
            let dcTable = huffmanDC[components[n].dcTable]
            let acTable = huffmanAC[components[n].acTable]
            let dequant = dequant.base + 64 * components[n].quantTable
            let fastAC = acTable.fastAC.base
            let zigzag = zigzag.base
            let data = block.base

            if codeBits < 16 { fillBits() }
            let t = decodeSymbol(dcTable)
            if t < 0 || t > 15 { throw Rejection.corrupt("bad huffman code") }
            data.update(repeating: 0, count: 64)
            let diff = t != 0 ? receiveExtend(t) : 0
            let dc = try Self.predict(components[n].dcPrediction, diff)
            components[n].dcPrediction = dc
            if !Self.productFitsInShort(Int(dc), Int(dequant[0])) { throw Rejection.corrupt("can't merge dc and ac") }
            data[0] = Int16(truncatingIfNeeded: Int(dc) * Int(dequant[0]))

            var k = 1
            repeat {
                if codeBits < 16 { fillBits() }
                let c = Int(codeBuffer >> UInt32(32 - JPEGDecoder.fastBits)) & ((1 << JPEGDecoder.fastBits) - 1)
                let r = Int(fastAC[c])
                if r != 0 {
                    k += (r >> 4) & 15
                    let s = r & 15
                    if s > codeBits { throw Rejection.corrupt("bad huffman code") }
                    codeBuffer <<= UInt32(s)
                    codeBits -= s
                    let zig = Int(zigzag[k])
                    k += 1
                    data[zig] = Int16(truncatingIfNeeded: (r >> 8) * Int(dequant[zig]))
                } else {
                    let rs = decodeSymbol(acTable)
                    if rs < 0 { throw Rejection.corrupt("bad huffman code") }
                    let s = rs & 15
                    if s == 0 {
                        if rs != 0xF0 { break }
                        k += 16
                    } else {
                        k += rs >> 4
                        let zig = Int(zigzag[k])
                        k += 1
                        data[zig] = Int16(truncatingIfNeeded: receiveExtend(s) * Int(dequant[zig]))
                    }
                }
            } while k < 64
        }

        /// `stbi__jpeg_decode_block_prog_dc`.
        mutating func decodeProgressiveDC(_ data: UnsafeMutablePointer<Int16>, component n: Int) throws {
            if spectralEnd != 0 { throw Rejection.corrupt("can't merge dc and ac") }
            if codeBits < 16 { fillBits() }
            if successiveHigh == 0 {
                data.update(repeating: 0, count: 64)
                let t = decodeSymbol(huffmanDC[components[n].dcTable])
                if t < 0 || t > 15 { throw Rejection.corrupt("can't merge dc and ac") }
                let diff = t != 0 ? receiveExtend(t) : 0
                let dc = try Self.predict(components[n].dcPrediction, diff)
                components[n].dcPrediction = dc
                if !Self.productFitsInShort(Int(dc), 1 << successiveLow) { throw Rejection.corrupt("can't merge dc and ac") }
                data[0] = Int16(truncatingIfNeeded: Int(dc) * (1 << successiveLow))
            } else if bit() {
                data[0] &+= Int16(truncatingIfNeeded: 1 << successiveLow)
            }
        }

        /// `stbi__jpeg_decode_block_prog_ac`: first and refinement passes over one band.
        mutating func decodeProgressiveAC(_ data: UnsafeMutablePointer<Int16>, table: HuffmanTable) throws {
            if spectralStart == 0 { throw Rejection.corrupt("can't merge dc and ac") }
            let zigzag = zigzag.base
            if successiveHigh == 0 {
                let shift = successiveLow
                if eobRun != 0 {
                    eobRun -= 1
                    return
                }
                let fastAC = table.fastAC.base
                var k = spectralStart
                repeat {
                    if codeBits < 16 { fillBits() }
                    let c = Int(codeBuffer >> UInt32(32 - JPEGDecoder.fastBits)) & ((1 << JPEGDecoder.fastBits) - 1)
                    let r = Int(fastAC[c])
                    if r != 0 {
                        k += (r >> 4) & 15
                        let s = r & 15
                        if s > codeBits { throw Rejection.corrupt("bad huffman code") }
                        codeBuffer <<= UInt32(s)
                        codeBits -= s
                        let zig = Int(zigzag[k])
                        k += 1
                        data[zig] = Int16(truncatingIfNeeded: (r >> 8) * (1 << shift))
                    } else {
                        let rs = decodeSymbol(table)
                        if rs < 0 { throw Rejection.corrupt("bad huffman code") }
                        let s = rs & 15
                        let r = rs >> 4
                        if s == 0 {
                            if r < 15 {
                                eobRun = 1 << r
                                if r != 0 { eobRun += bits(r) }
                                eobRun -= 1
                                break
                            }
                            k += 16
                        } else {
                            k += r
                            let zig = Int(zigzag[k])
                            k += 1
                            data[zig] = Int16(truncatingIfNeeded: receiveExtend(s) * (1 << shift))
                        }
                    }
                } while k <= spectralEnd
            } else {
                let bitValue = Int16(truncatingIfNeeded: 1 << successiveLow)
                if eobRun != 0 {
                    eobRun -= 1
                    for k in spectralStart...spectralEnd {
                        refine(data + Int(zigzag[k]), bitValue)
                    }
                } else {
                    var k = spectralStart
                    repeat {
                        let rs = decodeSymbol(table)
                        if rs < 0 { throw Rejection.corrupt("bad huffman code") }
                        var s = rs & 15
                        var r = rs >> 4
                        if s == 0 {
                            if r < 15 {
                                eobRun = (1 << r) - 1
                                if r != 0 { eobRun += bits(r) }
                                r = 64
                            }
                        } else {
                            if s != 1 { throw Rejection.corrupt("bad huffman code") }
                            s = bit() ? Int(bitValue) : -Int(bitValue)
                        }
                        while k <= spectralEnd {
                            let p = data + Int(zigzag[k])
                            k += 1
                            if p.pointee != 0 {
                                refine(p, bitValue)
                            } else {
                                if r == 0 {
                                    p.pointee = Int16(truncatingIfNeeded: s)
                                    break
                                }
                                r -= 1
                            }
                        }
                    } while k <= spectralEnd
                }
            }
        }

        /// A correction bit for a coefficient already nonzero.
        @inline(__always)
        mutating func refine(_ p: UnsafeMutablePointer<Int16>, _ bitValue: Int16) {
            guard p.pointee != 0, bit(), p.pointee & bitValue == 0 else { return }
            if p.pointee > 0 { p.pointee &+= bitValue } else { p.pointee &-= bitValue }
        }

        /// `stbi__jpeg_finish`: dequantise and transform the progressive coefficients.
        mutating func finishProgressive() {
            for component in components {
                guard let coefficients = component.coefficients else { continue }
                let dequant = dequant.base + 64 * component.quantTable
                for j in 0..<((component.height + 7) >> 3) {
                    for i in 0..<((component.width + 7) >> 3) {
                        let data = coefficients.base + 64 * (i + j * component.blocksPerRow)
                        for k in 0..<64 { data[k] = Int16(truncatingIfNeeded: Int(data[k]) * Int(dequant[k])) }
                        JPEGDecoder.idct(data, into: component.samples.base + component.stride * j * 8 + i * 8,
                                         stride: component.stride, scratch: idctScratch.base)
                    }
                }
            }
        }
    }
}
