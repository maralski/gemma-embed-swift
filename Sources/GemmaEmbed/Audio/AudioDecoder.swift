import AVFoundation
import Foundation

/// Decodes an `AudioInput` to the mono 16 kHz float samples the audio encoder consumes, the
/// way llama.cpp's miniaudio decoder does.
///
/// WAV bytes go through `WAVDecoder`, a port of dr_wav's f32 read, so they match llama.cpp
/// exactly. Anything else goes through AVFoundation at the file's own sample rate and channel
/// count (no AVFoundation conversion), so the downmix and resample are still miniaudio's; only
/// the codec differs from llama.cpp's dr_mp3 or dr_flac. `.samples` are taken as given.
/// Everything then passes through `MiniaudioConverter`.
enum AudioDecoder {
    static let targetSampleRate = Int(MiniaudioConverter.targetSampleRate)

    static func decode(_ input: AudioInput) throws -> [Float] {
        switch input {
        case .url(let url):
            guard FileManager.default.isReadableFile(atPath: url.path),
                  let data = try? Data(contentsOf: url, options: .mappedIfSafe) else {
                throw GemmaEmbedError.audioDecoding("\(url.path) is not readable")
            }
            if isWAV(data) {
                return try convert(try decodeWAV(data, path: url.path))
            }
            return try convert(try decodeWithAVFoundation(url: url, path: url.path))
        case .data(let data):
            if isWAV(data) {
                return try convert(try decodeWAV(data, path: nil))
            }
            return try convert(try decodeWithAVFoundation(data: data))
        case .samples(let samples, let sampleRate, let channels):
            guard channels > 0 else { throw GemmaEmbedError.audioDecoding("channel count must be positive") }
            guard sampleRate > 0 else { throw GemmaEmbedError.audioDecoding("sample rate must be positive") }
            guard samples.count % channels == 0 else {
                throw GemmaEmbedError.audioDecoding("\(samples.count) samples do not split over \(channels) channels")
            }
            return try convert(DecodedAudio(samples: samples, channels: channels, sampleRate: sampleRate))
        }
    }

    /// The same check llama.cpp's `is_audio_file` makes for WAV.
    static func isWAV(_ data: Data) -> Bool {
        data.count >= 12 && data.starts(with: Array("RIFF".utf8)) && data[data.startIndex + 8..<data.startIndex + 12].elementsEqual(Array("WAVE".utf8))
    }

    private static func convert(_ audio: DecodedAudio) throws -> [Float] {
        guard audio.frameCount > 0 else { throw GemmaEmbedError.audioDecoding("audio has no samples") }
        return MiniaudioConverter.toMono16k(audio)
    }

    private static func decodeWAV(_ data: Data, path: String?) throws -> DecodedAudio {
        do {
            return try WAVDecoder.decode(data)
        } catch GemmaEmbedError.audioDecoding(let reason) {
            throw GemmaEmbedError.audioDecoding(path.map { "\($0): \(reason)" } ?? reason)
        }
    }

    private static func decodeWithAVFoundation(data: Data) throws -> DecodedAudio {
        // AVAudioFile reads files only; AudioToolbox sniffs the container from the bytes, so the
        // extension can be anything.
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("gemma-embed-\(UUID().uuidString).audio")
        do {
            try data.write(to: url)
        } catch {
            throw GemmaEmbedError.audioDecoding("cannot write the audio bytes to a temporary file: \(error.localizedDescription)")
        }
        defer { try? FileManager.default.removeItem(at: url) }
        return try decodeWithAVFoundation(url: url, path: nil)
    }

    /// Float32 at the file's own rate and channel count; `AVAudioFile`'s processing format is
    /// deinterleaved float, so the channels are interleaved here.
    private static func decodeWithAVFoundation(url: URL, path: String?) throws -> DecodedAudio {
        let prefix = path.map { "\($0): " } ?? ""
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            throw GemmaEmbedError.audioDecoding("\(prefix)not an audio format AVFoundation can read (\(error.localizedDescription))")
        }
        let format = file.processingFormat
        let channels = Int(format.channelCount)
        let frames = Int(file.length)
        guard channels > 0, frames > 0 else { throw GemmaEmbedError.audioDecoding("\(prefix)audio has no samples") }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames)) else {
            throw GemmaEmbedError.audioDecoding("\(prefix)cannot allocate a \(frames)-frame buffer")
        }
        do {
            try file.read(into: buffer)
        } catch {
            throw GemmaEmbedError.audioDecoding("\(prefix)AVFoundation failed to decode (\(error.localizedDescription))")
        }
        guard let planes = buffer.floatChannelData else {
            throw GemmaEmbedError.audioDecoding("\(prefix)AVFoundation did not produce float samples")
        }
        let read = Int(buffer.frameLength)
        var samples = [Float](repeating: 0, count: read * channels)
        if format.isInterleaved {
            let interleaved = UnsafeBufferPointer(start: planes[0], count: read * channels)
            samples = Array(interleaved)
        } else {
            for channel in 0..<channels {
                let plane = planes[channel]
                for frame in 0..<read {
                    samples[frame * channels + channel] = plane[frame]
                }
            }
        }
        return DecodedAudio(samples: samples, channels: channels, sampleRate: Int(format.sampleRate))
    }
}
