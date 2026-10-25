import Foundation
import XCTest
@testable import GemmaEmbed

/// The JPEG port against stb_image's own output, dumped from llama.cpp's build of it.
final class JPEGDecoderTests: XCTestCase {
    static let defaultDumpsPath = TestSupport.defaultDecodedRGBDumpsPath

    /// The package's JPEG fixtures and the 1600x1600 and odd-sized extras: baseline 4:2:0 with a
    /// restart marker after every MCU row (the fixtures) or none (the `jv-` files).
    static let baseline420 = [
        "00-green-hills", "02-coral-reef", "03-snowy-peaks", "04-desert-dunes", "05-cherry-blossom", "07-night-sky", "08-sunflower-field", "09-city-skyline",
        "03-gradient-rgb", "04-text-helvetica", "05-text-menlo", "06-text-times", "07-shapes",
        "00-green-hills-resized-0.5", "01-ocean-sunset-cropped-10", "02-coral-reef-jpeg-q40", "03-snowy-peaks-flipped",
        "04-desert-dunes-colour-shift", "05-cherry-blossom-resized-0.33", "07-night-sky-jpeg-q40", "08-sunflower-field-flipped",
        "09-city-skyline-colour-shift", "01-ocean-sunset", "06-lavender-field", "06-lavender-field-cropped-10-top-left",
        "jv-420-odd-text-times", "jv-odd-size-green-hills",
    ]

    struct Dump {
        let name: String
        let image: URL
        let width: Int
        let height: Int
        let rgb: [UInt8]
    }

    /// stb_image's decode of one image, and the image: from the package's fixtures when it ships
    /// there, else at the path the dump recorded.
    static func requireDump(_ name: String) throws -> Dump {
        let root = ProcessInfo.processInfo.environment["GEMMA_EMBED_DECODED_RGB_DUMPS"] ?? defaultDumpsPath
        let directory = URL(fileURLWithPath: root).appendingPathComponent(name)
        let metaURL = directory.appendingPathComponent("meta.json")
        guard FileManager.default.isReadableFile(atPath: metaURL.path) else {
            throw XCTSkip("RGB dump \(name) not found; set GEMMA_EMBED_DECODED_RGB_DUMPS to the dumps-rgb directory")
        }
        let meta = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: metaURL)) as? [String: Any])
        let recorded = meta["image"] as! String
        let relative = try XCTUnwrap(recorded.components(separatedBy: "fixtures/images/").last)
        let image = (try? TestSupport.fixture("images/" + relative)) ?? URL(fileURLWithPath: recorded)
        return Dump(name: name, image: image, width: meta["decoded_nx"] as! Int, height: meta["decoded_ny"] as! Int,
                    rgb: [UInt8](try Data(contentsOf: directory.appendingPathComponent("decoded_rgb.bin"))))
    }

    func assertDecodesExactly(_ names: [String], file: StaticString = #filePath, line: UInt = #line) throws {
        var compared = 0
        for name in names {
            guard let dump = try? Self.requireDump(name) else { continue }
            let got = try JPEGDecoder.decode(Data(contentsOf: dump.image))
            XCTAssertEqual([got.width, got.height], [dump.width, dump.height], name, file: file, line: line)
            XCTAssertEqual(got.pixels.count, dump.rgb.count, name, file: file, line: line)
            var differing = 0
            var first: Int?
            for i in 0..<min(got.pixels.count, dump.rgb.count) where got.pixels[i] != dump.rgb[i] {
                differing += 1
                if first == nil { first = i }
            }
            if let i = first {
                let pixel = i / 3
                XCTFail("\(name): \(differing) bytes differ from stb_image; first at byte \(i) (x \(pixel % dump.width), y \(pixel / dump.width), channel \(i % 3)): got \(got.pixels[i]), stb \(dump.rgb[i])",
                        file: file, line: line)
            } else {
                print("jpeg \(name): \(dump.width)x\(dump.height) matches stb_image byte for byte")
            }
            compared += 1
        }
        if compared == 0 { throw XCTSkip("no RGB dumps found for \(names)") }
        XCTAssertEqual(compared, names.count, "dumps compared", file: file, line: line)
    }

    func test1Baseline420() throws {
        try assertDecodesExactly(Self.baseline420)
    }

    func test2Baseline444() throws {
        try assertDecodesExactly(["jv-444-desert-dunes"])
    }

    /// 4:2:2 and 4:4:0 take stb's horizontal and vertical fancy upsamplers, 4:1:1 the nearest-neighbour one.
    func test3OtherSubsampling() throws {
        try assertDecodesExactly(["jv-422-desert-dunes", "jv-440-desert-dunes", "jv-411-desert-dunes"])
    }

    func test4Grey() throws {
        try assertDecodesExactly(["jv-grey-desert-dunes"])
    }

    /// ImageMagick's four-component JPEG carries Adobe transform 2 (YCCK), which stb converts
    /// with the YCbCr kernel and then multiplies the inverse by K.
    func test5FourComponents() throws {
        try assertDecodesExactly(["jv-cmyk-coral-reef"])
    }

    func test6Progressive() throws {
        try assertDecodesExactly(["jv-progressive-420-desert-dunes", "jv-progressive-444-desert-dunes"])
    }

    func test7RestartMarkers() throws {
        try assertDecodesExactly(["jv-restart-desert-dunes"])
    }

    func testDecodeTime() throws {
        let data = try Data(contentsOf: TestSupport.fixture("images/scenes/00-green-hills.jpg"))
        var best = Double.infinity
        for _ in 0..<10 {
            let start = DispatchTime.now().uptimeNanoseconds
            let image = try JPEGDecoder.decode(data)
            best = min(best, Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9)
            XCTAssertEqual([image.width, image.height], [1600, 1600])
        }
        print(String(format: "jpeg decode 00-green-hills.jpg 1600x1600 baseline 4:2:0: best of 10 %.1f ms", best * 1000))
        #if !DEBUG
        XCTAssertLessThan(best, 0.1)
        #endif
    }

    /// The fixture bytes with `edit` applied to the frame header (`sof` is the offset of its FF marker byte).
    static func patched(_ name: String, _ edit: (inout [UInt8], _ sof: Int) -> Void) throws -> Data {
        var bytes = [UInt8](try Data(contentsOf: TestSupport.fixture(name)))
        let sof = try XCTUnwrap((0..<bytes.count - 1).first { bytes[$0] == 0xFF && bytes[$0 + 1] == 0xC0 })
        edit(&bytes, sof)
        return Data(bytes)
    }

    func assertRejected(_ data: Data, reason: String, unsupported: Bool, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try JPEGDecoder.decode(data), file: file, line: line) { error in
            guard let rejection = error as? JPEGDecoder.Rejection else { return XCTFail("unexpected error \(error)", file: file, line: line) }
            XCTAssertEqual(rejection.reason, reason, file: file, line: line)
            XCTAssertEqual(rejection.isUnsupportedEncoding, unsupported, file: file, line: line)
        }
    }

    /// Files stb_image refuses fail with its reason; the encodings it does not implement are marked
    /// so `ImageDecoder` can hand them to ImageIO, and ImageIO does decode them.
    func testRejectionsFollowStbImage() throws {
        let arithmetic = try Data(contentsOf: TestSupport.fixture("images/jpeg-unsupported/arithmetic-cherry-blossom.jpg"))
        let twelveBit = try Data(contentsOf: TestSupport.fixture("images/jpeg-unsupported/12-bit-cherry-blossom.jpg"))
        assertRejected(arithmetic, reason: "unknown marker", unsupported: true)
        assertRejected(twelveBit, reason: "only 8-bit", unsupported: true)
        for data in [arithmetic, twelveBit] {
            let image = try ImageDecoder.decode(.data(data))
            XCTAssertEqual([image.width, image.height], [341, 341])
        }

        let fixture = "images/variants/05-cherry-blossom-resized-0.33.jpg"
        assertRejected(try Self.patched(fixture) { bytes, sof in bytes[sof + 5] = 0; bytes[sof + 6] = 0 },
                       reason: "no header height", unsupported: true)
        assertRejected(try Self.patched(fixture) { bytes, sof in bytes[sof + 1] = 0xC3 }, reason: "unknown marker", unsupported: true)
        assertRejected(try Self.patched(fixture) { bytes, sof in bytes[sof + 9] = 2 }, reason: "bad component count", unsupported: false)
        assertRejected(try Self.patched(fixture) { bytes, sof in bytes[sof + 11] = 0x52 }, reason: "bad H", unsupported: false)
        assertRejected(Data([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x01]), reason: "bad APP len", unsupported: false)
        assertRejected(Data([0xFF, 0xD8]), reason: "expected marker", unsupported: false)
        XCTAssertThrowsError(try ImageDecoder.decode(.data(Data([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x01])))) { error in
            guard case GemmaEmbedError.imageDecoding("bad APP len") = error else { return XCTFail("unexpected error \(error)") }
        }
    }

    /// stb keeps what it decoded when the data stops early: the rows past the end come from zero bits.
    func testTruncatedFileKeepsDecodedRows() throws {
        let dump = try Self.requireDump("05-cherry-blossom-resized-0.33")
        let data = try Data(contentsOf: dump.image)
        let image = try JPEGDecoder.decode(data.prefix(data.count / 2))
        XCTAssertEqual([image.width, image.height], [dump.width, dump.height])
        let firstRows = 3 * dump.width * 16
        XCTAssertEqual(Array(image.pixels.prefix(firstRows)), Array(dump.rgb.prefix(firstRows)))
        XCTAssertNotEqual(image.pixels, dump.rgb)
    }
}
