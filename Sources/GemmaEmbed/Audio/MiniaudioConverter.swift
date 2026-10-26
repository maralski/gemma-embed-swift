// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// The data conversion `ma_decoder` applies between its backend decoder and a caller that asked
/// for `f32, 1 channel, 16000 Hz`, ported from miniaudio 0.11.25 so GemmaEmbed feeds the audio
/// encoder the same samples llama.cpp does.
///
/// With the backend already producing f32 (the wav backend is asked for f32, and every other
/// path here starts from floats) the converter never converts formats. Its execution path is
/// `channels_first`: the channel converter runs before the resampler, because the output has
/// fewer channels than the input. N channels to mono takes `ma_channel_conversion_path_mono_out`
/// (the output channel map is empty, so the default mono map is used), which averages the
/// channels in order rather than applying rectangular weights. The resampler is
/// `ma_linear_resampler` with the default `lpfOrder` of 4 (two biquad stages), run on the mono
/// stream. Floating-point operations keep miniaudio's order and width: coefficients in double,
/// samples in float.
enum MiniaudioConverter {
    static let targetSampleRate: UInt32 = 16000

    /// The mono 16 kHz samples llama.cpp reads: `ma_decoder_get_length_in_pcm_frames` frames,
    /// filled with what the resampler produces and zero past the point where it runs out of
    /// input, since llama.cpp sizes its buffer from the length and reads once.
    static func toMono16k(_ audio: DecodedAudio) -> [Float] {
        let mono = audio.channels == 1 ? audio.samples : downmixToMono(audio.samples, channels: audio.channels)
        let sampleRate = UInt32(audio.sampleRate)
        if sampleRate == targetSampleRate {
            return mono
        }
        var resampler = LinearResampler(sampleRateIn: sampleRate, sampleRateOut: targetSampleRate)
        let produced = resampler.process(mono)
        let length = Int(frameCountAfterResampling(sampleRateOut: targetSampleRate, sampleRateIn: sampleRate, frameCountIn: UInt64(mono.count)))
        if produced.count >= length {
            return Array(produced[0..<length])
        }
        return produced + [Float](repeating: 0, count: length - produced.count)
    }

    /// `ma_channel_converter_process_pcm_frames__mono_out` for f32: the channels summed in order,
    /// divided by the channel count.
    static func downmixToMono(_ samples: [Float], channels: Int) -> [Float] {
        let frames = samples.count / channels
        var out = [Float](repeating: 0, count: frames)
        let divisor = Float(channels)
        for frame in 0..<frames {
            var t: Float = 0
            for channel in 0..<channels {
                t += samples[frame * channels + channel]
            }
            out[frame] = t / divisor
        }
        return out
    }

    /// `ma_calculate_frame_count_after_resampling`, with the rates as the file declares them.
    static func frameCountAfterResampling(sampleRateOut: UInt32, sampleRateIn: UInt32, frameCountIn: UInt64) -> UInt64 {
        if sampleRateIn == 0 || sampleRateOut == 0 || frameCountIn == 0 { return 0 }
        if sampleRateOut == sampleRateIn { return frameCountIn }
        let rateIn = UInt64(sampleRateIn), rateOut = UInt64(sampleRateOut)
        var outputFrameCount = (frameCountIn * rateOut) / rateIn
        let preliminaryFromFrac = (outputFrameCount * (rateIn / rateOut)) / rateOut
        let preliminary = outputFrameCount * (rateIn % rateOut) + preliminaryFromFrac
        if preliminary <= frameCountIn {
            outputFrameCount += 1
        }
        return outputFrameCount
    }

    /// `ma_linear_resampler` for one f32 channel: fixed-point time in whole input frames plus a
    /// fraction over the gcd-reduced output rate, linear interpolation between the last two input
    /// frames, and a low-pass filter applied to each input frame when downsampling or to each
    /// output frame when upsampling.
    struct LinearResampler {
        let sampleRateIn: UInt32
        let sampleRateOut: UInt32
        let inAdvanceInt: UInt32
        let inAdvanceFrac: UInt32
        var inTimeInt: UInt32 = 1
        var inTimeFrac: UInt32 = 0
        var x0: Float = 0
        var x1: Float = 0
        var lowPass: LowPassFilter

        init(sampleRateIn: UInt32, sampleRateOut: UInt32) {
            let gcf = Self.gcd(sampleRateIn, sampleRateOut)
            self.sampleRateIn = sampleRateIn / gcf
            self.sampleRateOut = sampleRateOut / gcf
            inAdvanceInt = self.sampleRateIn / self.sampleRateOut
            inAdvanceFrac = self.sampleRateIn % self.sampleRateOut
            let lpfSampleRate = max(self.sampleRateIn, self.sampleRateOut)
            let lpfCutoffFrequency = Double(min(self.sampleRateIn, self.sampleRateOut)) * 0.5 * 1.0
            lowPass = LowPassFilter(order: 4, sampleRate: lpfSampleRate, cutoffFrequency: lpfCutoffFrequency)
        }

        /// Every output frame the stream yields from `input`; stops when the next frame would need
        /// input past the end.
        mutating func process(_ input: [Float]) -> [Float] {
            let downsampling = sampleRateIn > sampleRateOut
            var out: [Float] = []
            out.reserveCapacity(Int(UInt64(input.count) * UInt64(sampleRateOut) / UInt64(sampleRateIn)) + 1)
            var next = 0
            while true {
                while inTimeInt > 0 && next < input.count {
                    x0 = x1
                    x1 = input[next]
                    if downsampling {
                        x1 = lowPass.process(x1)
                    }
                    next += 1
                    inTimeInt -= 1
                }
                if inTimeInt > 0 {
                    break
                }
                var sample = interpolate()
                if !downsampling {
                    sample = lowPass.process(sample)
                }
                out.append(sample)
                inTimeInt += inAdvanceInt
                inTimeFrac += inAdvanceFrac
                if inTimeFrac >= sampleRateOut {
                    inTimeFrac -= sampleRateOut
                    inTimeInt += 1
                }
            }
            return out
        }

        /// `ma_linear_resampler_interpolate_frame_f32` over `ma_mix_f32_fast`, three separately
        /// rounded float operations.
        private func interpolate() -> Float {
            let a = Float(inTimeFrac) / Float(sampleRateOut)
            let r0 = x1 - x0
            let r1 = r0 * a
            return x0 + r1
        }

        static func gcd(_ a: UInt32, _ b: UInt32) -> UInt32 {
            var a = a, b = b
            while b != 0 {
                let t = a
                a = b
                b = t % a
            }
            return a
        }
    }

    /// `ma_lpf` of even order: `order / 2` second-order Butterworth sections (`ma_lpf2`) in series,
    /// each a direct form 2 transposed biquad in float with coefficients computed in double.
    struct LowPassFilter {
        var stages: [Biquad]

        init(order: Int, sampleRate: UInt32, cutoffFrequency: Double) {
            precondition(order % 2 == 0, "odd orders add a first-order stage this port does not carry")
            stages = (0..<order / 2).map { stage in
                let a = Double(1 + stage * 2) * (Double.pi / Double(order * 2))
                let q = 1 / (2 * Self.cosd(a))
                return Biquad(sampleRate: sampleRate, cutoffFrequency: cutoffFrequency, q: q)
            }
        }

        /// miniaudio's `ma_cosd` is `sin(pi / 2 - x)`, which differs from `cos` in the last bit.
        static func cosd(_ x: Double) -> Double {
            sin((Double.pi * 0.5) - x)
        }

        mutating func process(_ x: Float) -> Float {
            var y = x
            for i in stages.indices {
                y = stages[i].process(y)
            }
            return y
        }
    }

    struct Biquad {
        let b0: Float, b1: Float, b2: Float, a1: Float, a2: Float
        var r1: Float = 0
        var r2: Float = 0

        /// `ma_lpf2__get_biquad_config` then `ma_biquad_reinit`'s normalisation by a0.
        init(sampleRate: UInt32, cutoffFrequency: Double, q: Double) {
            let w = 2 * Double.pi * cutoffFrequency / Double(sampleRate)
            let s = sin(w)
            let c = LowPassFilter.cosd(w)
            let a = s / (2 * q)
            let b0 = (1 - c) / 2
            let b1 = 1 - c
            let b2 = (1 - c) / 2
            let a0 = 1 + a
            let a1 = -2 * c
            let a2 = 1 - a
            self.b0 = Float(b0 / a0)
            self.b1 = Float(b1 / a0)
            self.b2 = Float(b2 / a0)
            self.a1 = Float(a1 / a0)
            self.a2 = Float(a2 / a0)
        }

        /// `ma_biquad_process_pcm_frame_f32__direct_form_2_transposed`, with the multiply-adds
        /// fused the way clang's default `-ffp-contract=on` compiles them in llama.cpp's build:
        /// `b0*x + r1` is one fma, and `b1*x - a1*y` fuses the first product into the subtraction
        /// of the separately rounded second. Unfused arithmetic lands within 8e-7 but not bit for bit.
        mutating func process(_ x: Float) -> Float {
            let y = r1.addingProduct(b0, x)
            r1 = (-(a1 * y)).addingProduct(b1, x) + r2
            r2 = (-(a2 * y)).addingProduct(b2, x)
            return y
        }
    }
}
