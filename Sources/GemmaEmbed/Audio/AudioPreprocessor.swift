import Foundation

/// One 30-second (or shorter, final) chunk of audio as log-mel frames: `frames` rows of `melBins`
/// values, row-major, the tensor the audio tower's first convolution reads.
struct MelChunk: Sendable {
    let frames: Int
    let melBins: Int
    let values: [Float]

    /// Rows the chunk contributes to the text sequence after the two stride-2 convolutions.
    var tokenCount: Int { AudioFrontEnd.tokenCount(frames: frames) }
}

/// A sound ready for the audio tower: its 16 kHz sample count and its chunks.
struct PreparedAudio: Sendable {
    let sampleCount: Int
    let chunks: [MelChunk]

    var tokenCount: Int { chunks.reduce(0) { $0 + $1.tokenCount } }
    var frameCount: Int { chunks.reduce(0) { $0 + $1.frames } }
}

/// The front end llama.cpp's `mtmd_audio_preprocessor_gemma4a` applies before the audio tower,
/// reproduced with its arithmetic: 16 kHz mono samples are cut into 30-second chunks; each chunk
/// gets 160 zeros in front, is framed every 160 samples, windowed with a periodic 320-sample Hann
/// (20 ms) zero-padded to 512, transformed with a 512-point radix-2 FFT, reduced to magnitudes,
/// mapped onto 128 HTK mel filters, floored at 0.001 and logged. The constants are hard-coded in
/// llama.cpp (`clip.cpp`, `PROJECTOR_TYPE_GEMMA4A`); only the mel count is GGUF metadata.
struct AudioFrontEnd: Sendable {
    static let sampleRate = 16000
    static let fftSize = 512
    static let windowLength = 320
    static let hop = 160
    static let chunkSamples = 30 * sampleRate
    /// llama.cpp's frame count truncates toward zero, so any clip of two or more samples has a
    /// frame; a single sample has none and is refused.
    static let minimumSamples = 2
    static let bins = fftSize / 2 + 1

    let melBins: Int
    /// Periodic Hann over the first `windowLength` taps, zero to `fftSize`.
    let hann: [Float]
    /// `sin(2 pi i / fftSize)` and `cos(2 pi i / fftSize)` for `i < fftSize`, computed in double and stored as float.
    let sinTable: [Float]
    let cosTable: [Float]
    /// `[melBins][bins]` triangular HTK filters from 0 Hz to 8 kHz, no area normalisation.
    let filters: [Float]

    init(melBins: Int) {
        self.melBins = melBins
        let n = AudioFrontEnd.fftSize
        var hann = [Float](repeating: 0, count: n)
        for i in 0..<AudioFrontEnd.windowLength {
            hann[i] = 0.5 - 0.5 * cosf((2 * Float.pi * Float(i)) / Float(AudioFrontEnd.windowLength))
        }
        self.hann = hann
        sinTable = (0..<n).map { sinf(Float((2 * Double.pi * Double($0)) / Double(n))) }
        cosTable = (0..<n).map { cosf(Float((2 * Double.pi * Double($0)) / Double(n))) }
        filters = AudioFrontEnd.melFilterbank(melBins: melBins, fftSize: n, sampleRate: AudioFrontEnd.sampleRate)
    }

    /// Frames llama.cpp takes from `samples` samples of one chunk (PyTorch's `unfold(320 + 1, 160)`
    /// over the left-padded waveform).
    static func frameCount(samples: Int) -> Int {
        (samples + hop - (windowLength + 1)) / hop + 1
    }

    /// Two convolutions of kernel 3, stride 2 and padding 1 turn `frames` frames into tokens.
    static func tokenCount(frames: Int) -> Int {
        var n = frames
        for _ in 0..<2 { n = (n - 1) / 2 + 1 }
        return n
    }

    /// Tokens a clip of `samples` 16 kHz samples produces across its chunks.
    static func tokenCount(samples: Int) -> Int {
        var total = 0
        var offset = 0
        while offset < samples {
            let n = min(chunkSamples, samples - offset)
            total += tokenCount(frames: frameCount(samples: n))
            offset += n
        }
        return total
    }

    /// `ma`'s HTK mel filterbank as `mtmd_audio_cache::fill_mel_filterbank_matrix` builds it with
    /// `use_htk`, no Slaney normalisation and scale 1: triangles between `melBins + 2` points
    /// spaced evenly in mel from 0 Hz to half the sample rate, sampled at the FFT bin centres.
    static func melFilterbank(melBins: Int, fftSize: Int, sampleRate: Int) -> [Float] {
        func hzToMel(_ f: Double) -> Double { 2595 * log10(1 + f / 700) }
        func melToHz(_ m: Double) -> Double { 700 * (pow(10, m / 2595) - 1) }
        let bins = fftSize / 2 + 1
        let binHz = Double(sampleRate) / Double(fftSize)
        let mLo = hzToMel(0), mHi = hzToMel(Double(sampleRate) / 2)
        let hz = (0..<(melBins + 2)).map { melToHz(mLo + (mHi - mLo) * (Double($0) / Double(melBins + 1))) }
        var out = [Float](repeating: 0, count: melBins * bins)
        for m in 0..<melBins {
            let left = hz[m], centre = hz[m + 1], right = hz[m + 2]
            let denomL = max(1e-30, centre - left), denomR = max(1e-30, right - centre)
            for k in 0..<bins {
                let f = Double(k) * binHz
                var w = 0.0
                if f >= left && f <= centre {
                    w = (f - left) / denomL
                } else if f > centre && f <= right {
                    w = (right - f) / denomR
                }
                out[m * bins + k] = Float(w)
            }
        }
        return out
    }

    /// Cuts `samples` into chunks and computes each chunk's log-mel frames. Throws when the clip
    /// is too short for one frame, which llama.cpp refuses too.
    func prepare(_ samples: [Float]) throws -> PreparedAudio {
        guard samples.count >= AudioFrontEnd.minimumSamples else {
            throw GemmaEmbedError.audioDecoding("the clip has \(samples.count) samples at 16 kHz; at least \(AudioFrontEnd.minimumSamples) are needed for one frame")
        }
        var chunks: [MelChunk] = []
        var offset = 0
        while offset < samples.count {
            let n = min(AudioFrontEnd.chunkSamples, samples.count - offset)
            chunks.append(logMel(samples[offset..<(offset + n)]))
            offset += n
        }
        return PreparedAudio(sampleCount: samples.count, chunks: chunks)
    }

    /// The log-mel frames of one chunk.
    func logMel(_ chunk: ArraySlice<Float>) -> MelChunk {
        let n = AudioFrontEnd.fftSize, hop = AudioFrontEnd.hop, bins = AudioFrontEnd.bins
        let frames = AudioFrontEnd.frameCount(samples: chunk.count)
        precondition(frames >= 1)
        // 160 zeros in front, then zeros behind so the last frame has a full 512 samples.
        let padLeft = AudioFrontEnd.windowLength / 2
        var padded = [Float](repeating: 0, count: (frames - 1) * hop + n)
        padded.replaceSubrange(padLeft..<(padLeft + chunk.count), with: chunk)

        var values = [Float](repeating: 0, count: frames * melBins)
        var input = [Float](repeating: 0, count: n)
        var spectrum = [Float](repeating: 0, count: 2 * n)
        var magnitude = [Float](repeating: 0, count: bins)
        for frame in 0..<frames {
            let offset = frame * hop
            for j in 0..<n { input[j] = hann[j] * padded[offset + j] }
            fft(input, count: n, out: &spectrum)
            for j in 0..<bins {
                let re = spectrum[2 * j], im = spectrum[2 * j + 1]
                magnitude[j] = (re * re + im * im).squareRoot()
            }
            for m in 0..<melBins {
                let f = m * bins
                var sum = 0.0
                var k = 0
                // llama.cpp sums four products in float before adding to its double accumulator.
                while k < bins - 3 {
                    let partial = magnitude[k] * filters[f + k] + magnitude[k + 1] * filters[f + k + 1]
                        + magnitude[k + 2] * filters[f + k + 2] + magnitude[k + 3] * filters[f + k + 3]
                    sum += Double(partial)
                    k += 4
                }
                while k < bins {
                    sum += Double(magnitude[k] * filters[f + k])
                    k += 1
                }
                values[frame * melBins + m] = Float(log(max(sum, Double(Float(0.001)))))
            }
        }
        return MelChunk(frames: frames, melBins: melBins, values: values)
    }

    /// Radix-2 Cooley-Tukey FFT of `count` real samples taken every `stride` from `x[base...]`,
    /// written as interleaved complex pairs at `out[outOffset...]`, with llama.cpp's twiddle table
    /// and butterfly order. Strided views replace llama.cpp's even/odd copies.
    private func fft(_ x: [Float], base: Int = 0, count: Int, stride: Int = 1, out: inout [Float], outOffset: Int = 0) {
        if count == 1 {
            out[outOffset] = x[base]
            out[outOffset + 1] = 0
            return
        }
        let half = count / 2
        var sub = [Float](repeating: 0, count: 2 * count)
        fft(x, base: base, count: half, stride: 2 * stride, out: &sub, outOffset: 0)
        fft(x, base: base + stride, count: half, stride: 2 * stride, out: &sub, outOffset: count)
        combine(sub, count: count, out: &out, outOffset: outOffset)
    }

    /// `sub` holds the even half's spectrum then the odd half's, each `count` floats.
    private func combine(_ sub: [Float], count: Int, out: inout [Float], outOffset: Int) {
        let half = count / 2
        let step = AudioFrontEnd.fftSize / count
        for k in 0..<half {
            let re = cosTable[k * step]
            let im = -sinTable[k * step]
            let reOdd = sub[count + 2 * k], imOdd = sub[count + 2 * k + 1]
            let evenRe = sub[2 * k], evenIm = sub[2 * k + 1]
            out[outOffset + 2 * k] = evenRe + re * reOdd - im * imOdd
            out[outOffset + 2 * k + 1] = evenIm + re * imOdd + im * reOdd
            out[outOffset + 2 * (k + half)] = evenRe - re * reOdd + im * imOdd
            out[outOffset + 2 * (k + half) + 1] = evenIm - re * imOdd - im * reOdd
        }
    }
}
