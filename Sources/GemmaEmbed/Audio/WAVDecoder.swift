// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// PCM samples as decoded from a file: interleaved floats in the range -1 to 1, before any
/// downmix or resampling.
struct DecodedAudio: Equatable, Sendable {
    let samples: [Float]
    let channels: Int
    let sampleRate: Int

    var frameCount: Int { channels == 0 ? 0 : samples.count / channels }
}

/// A RIFF/WAVE parser that produces the float32 samples miniaudio's dr_wav produces when a
/// decoder asks it for f32, so a WAV embeds exactly as it does through llama.cpp.
///
/// Behaviour copied from `ma_dr_wav_init__internal` and `ma_dr_wav_read_pcm_frames_f32`: the
/// chunk walk stops at `data`, so `fmt ` must come first; unknown chunks are skipped with their
/// odd-size padding; a `data` size that overruns the file is clamped to the bytes present; the
/// frame count is the clamped size divided by the frame size, rounded down. Integer samples are
/// converted with dr_wav's constants (`ma_dr_wav_u8_to_f32` and friends), 32-bit floats are
/// copied bit for bit and 64-bit floats are rounded to float.
enum WAVDecoder {
    static let formatPCM: UInt16 = 0x0001
    static let formatIEEEFloat: UInt16 = 0x0003
    static let formatExtensible: UInt16 = 0xFFFE

    static func decode(_ data: Data) throws -> DecodedAudio {
        var reader = Reader(bytes: [UInt8](data))
        guard reader.fourCC() == "RIFF" else { throw reject("not a RIFF file") }
        _ = reader.u32()
        guard reader.fourCC() == "WAVE" else { throw reject("not a WAVE file") }

        var fmt: Format?
        var dataRange: Range<Int>?
        // dr_wav stops at the data chunk and ends the walk when a skip runs past the file.
        chunks: while let header = reader.chunkHeader() {
            let size = Int(header.size)
            let padding = Int(header.size & 1)
            switch header.id {
            case "fmt ":
                fmt = try readFormat(&reader, chunkSize: size)
                guard reader.skip(padding) else { break chunks }
            case "data":
                dataRange = reader.offset..<reader.offset + size
                break chunks
            case "fact":
                guard reader.skip(4) else { throw reject("fact chunk is truncated") }
                guard reader.skip(size - 4 + padding) else { break chunks }
            default:
                guard reader.skip(size + padding) else { break chunks }
            }
        }

        guard let fmt else { throw reject("no fmt chunk before the data chunk") }
        guard let dataRange else { throw reject("no data chunk") }
        guard fmt.sampleRate > 0, fmt.sampleRate <= 384_000 else { throw reject("sample rate \(fmt.sampleRate) is out of range") }
        guard fmt.channels > 0, fmt.channels <= 256 else { throw reject("channel count \(fmt.channels) is out of range") }
        guard fmt.bitsPerSample > 0, fmt.bitsPerSample <= 64 else { throw reject("\(fmt.bitsPerSample) bits per sample is out of range") }
        guard fmt.blockAlign > 0 else { throw reject("block align is zero") }

        let channels = Int(fmt.channels)
        let bytesPerFrame = fmt.bitsPerSample % 8 == 0 ? Int(fmt.bitsPerSample) * channels / 8 : Int(fmt.blockAlign)
        guard bytesPerFrame > 0 else { throw reject("frame size is zero") }
        var dataSize = min(dataRange.count, reader.bytes.count - dataRange.lowerBound)
        dataSize -= dataSize % bytesPerFrame
        let frames = dataSize / bytesPerFrame
        let payload = reader.bytes[dataRange.lowerBound..<dataRange.lowerBound + dataSize]

        let samples: [Float]
        switch fmt.translatedTag {
        case formatPCM:
            let bytesPerSample = bytesPerFrame / channels
            guard bytesPerSample > 0, bytesPerFrame % channels == 0 else {
                throw reject("\(fmt.bitsPerSample)-bit PCM frames do not split evenly over \(channels) channels")
            }
            samples = pcmToFloat(payload, count: frames * channels, bytesPerSample: bytesPerSample)
        case formatIEEEFloat:
            samples = ieeeToFloat(payload, count: frames * channels, bytesPerSample: bytesPerFrame / channels, bitsPerSample: Int(fmt.bitsPerSample))
        default:
            throw reject("format tag 0x\(String(fmt.translatedTag, radix: 16)) is not PCM or IEEE float")
        }
        return DecodedAudio(samples: samples, channels: channels, sampleRate: Int(fmt.sampleRate))
    }

    struct Format {
        var tag: UInt16 = 0
        var channels: UInt16 = 0
        var sampleRate: UInt32 = 0
        var blockAlign: UInt16 = 0
        var bitsPerSample: UInt16 = 0
        var subFormatTag: UInt16 = 0

        /// `WAVE_FORMAT_EXTENSIBLE` defers to the first two bytes of its sub-format GUID.
        var translatedTag: UInt16 { tag == WAVDecoder.formatExtensible ? subFormatTag : tag }
    }

    private static func readFormat(_ reader: inout Reader, chunkSize: Int) throws -> Format {
        guard reader.remaining >= 16 else { throw reject("fmt chunk is truncated") }
        var fmt = Format()
        fmt.tag = reader.u16()
        fmt.channels = reader.u16()
        fmt.sampleRate = reader.u32()
        _ = reader.u32()
        fmt.blockAlign = reader.u16()
        fmt.bitsPerSample = reader.u16()
        var consumed = 16
        if chunkSize > 16 {
            guard reader.remaining >= 2 else { throw reject("fmt chunk is truncated") }
            let extensionSize = Int(reader.u16())
            consumed = 18
            if extensionSize > 0 {
                if fmt.tag == formatExtensible {
                    guard extensionSize == 22 else { throw reject("extensible fmt chunk has a \(extensionSize)-byte extension, expected 22") }
                    guard reader.remaining >= 22 else { throw reject("fmt chunk is truncated") }
                    _ = reader.u16()
                    _ = reader.u32()
                    fmt.subFormatTag = reader.u16()
                    _ = reader.skip(14)
                } else {
                    guard reader.skip(extensionSize) else { throw reject("fmt chunk is truncated") }
                }
                consumed += extensionSize
            }
            guard reader.skip(chunkSize - consumed) else { throw reject("fmt chunk is truncated") }
        }
        return fmt
    }

    private static func pcmToFloat(_ bytes: ArraySlice<UInt8>, count: Int, bytesPerSample: Int) -> [Float] {
        var out = [Float](repeating: 0, count: count)
        let base = bytes.startIndex
        switch bytesPerSample {
        case 1:
            for i in 0..<count {
                var x = Float(bytes[base + i])
                x = x * 0.00784313725490196078
                x = x - 1
                out[i] = x
            }
        case 2:
            for i in 0..<count {
                let v = Int16(bitPattern: UInt16(bytes[base + 2 * i]) | UInt16(bytes[base + 2 * i + 1]) << 8)
                out[i] = Float(v) * 0.000030517578125
            }
        case 3:
            for i in 0..<count {
                let a = UInt32(bytes[base + 3 * i]) << 8
                let b = UInt32(bytes[base + 3 * i + 1]) << 16
                let c = UInt32(bytes[base + 3 * i + 2]) << 24
                let x = Double(Int32(bitPattern: a | b | c) >> 8)
                out[i] = Float(x * 0.00000011920928955078125)
            }
        case 4:
            for i in 0..<count {
                var raw: UInt32 = 0
                for k in 0..<4 { raw |= UInt32(bytes[base + 4 * i + k]) << (8 * UInt32(k)) }
                out[i] = Float(Double(Int32(bitPattern: raw)) / 2147483648.0)
            }
        case 5...8:
            for i in 0..<count {
                var sample: UInt64 = 0
                var shift = UInt64(8 - bytesPerSample) * 8
                for k in 0..<bytesPerSample {
                    sample |= UInt64(bytes[base + bytesPerSample * i + k]) << shift
                    shift += 8
                }
                out[i] = Float(Double(Int64(bitPattern: sample)) / 9223372036854775807.0)
            }
        default:
            break
        }
        return out
    }

    private static func ieeeToFloat(_ bytes: ArraySlice<UInt8>, count: Int, bytesPerSample: Int, bitsPerSample: Int) -> [Float] {
        var out = [Float](repeating: 0, count: count)
        let base = bytes.startIndex
        if bitsPerSample == 32 {
            for i in 0..<count {
                var raw: UInt32 = 0
                for k in 0..<4 { raw |= UInt32(bytes[base + 4 * i + k]) << (8 * UInt32(k)) }
                out[i] = Float(bitPattern: raw)
            }
        } else if bytesPerSample == 8 {
            for i in 0..<count {
                var raw: UInt64 = 0
                for k in 0..<8 { raw |= UInt64(bytes[base + 8 * i + k]) << (8 * UInt64(k)) }
                out[i] = Float(Double(bitPattern: raw))
            }
        }
        return out
    }

    private static func reject(_ reason: String) -> GemmaEmbedError {
        .audioDecoding("WAV: " + reason)
    }

    struct Reader {
        let bytes: [UInt8]
        var offset = 0

        var remaining: Int { bytes.count - offset }

        mutating func u16() -> UInt16 {
            defer { offset += 2 }
            return UInt16(bytes[offset]) | UInt16(bytes[offset + 1]) << 8
        }

        mutating func u32() -> UInt32 {
            defer { offset += 4 }
            return UInt32(bytes[offset]) | UInt32(bytes[offset + 1]) << 8 | UInt32(bytes[offset + 2]) << 16 | UInt32(bytes[offset + 3]) << 24
        }

        mutating func fourCC() -> String? {
            guard remaining >= 4 else { return nil }
            defer { offset += 4 }
            return String(decoding: bytes[offset..<offset + 4], as: UTF8.self)
        }

        /// Nil when the file ends inside the header, which ends the chunk walk as in dr_wav.
        mutating func chunkHeader() -> (id: String, size: UInt32)? {
            guard remaining >= 8, let id = fourCC() else { return nil }
            return (id, u32())
        }

        /// Moves the cursor, backwards too; false when that leaves the file, as dr_wav's seek fails.
        mutating func skip(_ delta: Int) -> Bool {
            let target = offset + delta
            guard target >= 0, target <= bytes.count else { return false }
            offset = target
            return true
        }
    }
}
