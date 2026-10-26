#!/usr/bin/env python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Writes the audio fixtures in Tests/GemmaEmbedTests/Fixtures/audio from code.

    python3 -I scripts/make_fixture_audio.py [<audio dir>]

Every byte is synthesised here from sines, chirps and seeded noise, so the set can be redistributed:
there are no recordings and no voices. Running it again writes identical bytes and removes any WAV
in the folder that the list below no longer names. It writes tones/, signals/, formats/ (sample
rates, channel counts and sample formats of one signal), long/ (the 30-second chunk boundary and a
two-chunk clip) and manifest.json. The generic clips are short so the folder, with the reference
vectors, stays under 4 MB. scripts/oracle_audio.py writes embeddings.jsonl from llama-server.

The llama.cpp dumps (scripts/audio_dump.cpp) are taken from the clips marked dump in the manifest.
From the package root, with the dump() helper from audio_dump.cpp's header:
    dump $A/signals/08-short-tone.wav models/audio/dumps/short-tone
    dump $A/signals/09-short-chirp.wav models/audio/dumps/short-chirp
    dump $A/signals/10-tone-noise.wav models/audio/dumps/tone-noise '^(0|1|2|3):'
    dump $A/signals/11-mixture.wav models/audio/dumps/mixture '^(0|1|2|3):'
    dump $A/long/01-35s.wav models/audio/dumps/long-35s '^(0|1|1388|1389):'
    dump $A/formats/03-rate-48000.wav models/audio/dumps/rate-48000 '^(0|1):'
    dump $A/formats/04-stereo.wav models/audio/dumps/stereo '^(0|1):'
The last three keep only the mel input and the output rows (the 35 s clip's second chunk starts at
node 1388); every dump also holds miniaudio's decoded samples, which the decoder tests read.
"""
import json
import math
import os
import struct
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Tests", "GemmaEmbedTests", "Fixtures", "audio")
TAU = 2 * math.pi


def noise(seed, n):
    # 64-bit LCG (Knuth's MMIX constants); the top 24 bits give a uniform value in [-1, 1).
    x, out = seed, []
    for _ in range(n):
        x = (6364136223846793005 * x + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        out.append((x >> 40) / 8388608.0 - 1.0)
    return out


def sine(rate, seconds, freq, amp, n=None):
    n = round(rate * seconds) if n is None else n
    return [amp * math.sin(TAU * freq * i / rate) for i in range(n)]


def glide(rate, n, freq_at, amp_at):
    """A sine whose frequency and amplitude follow functions of time, by phase accumulation."""
    phase, out = 0.0, []
    for i in range(n):
        t = i / rate
        out.append(amp_at(t) * math.sin(phase))
        phase += TAU * freq_at(t) / rate
    return out


def linear_chirp(rate, seconds, f0, f1, amp):
    n, T = round(rate * seconds), seconds
    return [amp * math.sin(TAU * (f0 * t + (f1 - f0) * t * t / (2 * T))) for t in (i / rate for i in range(n))]


def exp_chirp(rate, seconds, f0, f1, amp):
    n, T, k = round(rate * seconds), seconds, f1 / f0
    return [amp * math.sin(TAU * f0 * T / math.log(k) * (k ** (t / T) - 1)) for t in (i / rate for i in range(n))]


def mix(*signals):
    return [sum(v) for v in zip(*signals)]


def lowpass(x, alpha):
    y, out = 0.0, []
    for v in x:
        y += alpha * (v - y)
        out.append(y)
    return out


def normalise(x, peak):
    m = max(abs(v) for v in x)
    return [v * peak / m for v in x]


def burst(rate):
    """0.4 s bursts of 660 Hz then 990 Hz, each followed by 0.4 s of silence."""
    on, out = round(0.4 * rate), []
    for k in range(3):
        out += sine(rate, 0, 660 if k % 2 == 0 else 990, 0.6, on)
        if k < 2:
            out += [0.0] * on
    return out


def clicks(rate, seconds, per_second):
    n, out = round(rate * seconds), [0.0] * round(rate * seconds)
    for i in range(0, n, rate // per_second):
        out[i] = 0.8
        if i + 1 < n:
            out[i + 1] = -0.4
    return out


def mixture(rate):
    """7 s whose content changes every 1.4 s: tone, rising chirp, noise, chord, AM tone."""
    seg = round(1.4 * rate)
    am = [(0.4 + 0.3 * math.sin(TAU * 3 * i / rate)) * math.sin(TAU * 700 * i / rate) for i in range(seg)]
    return (sine(rate, 0, 330, 0.6, seg) + linear_chirp(rate, 1.4, 250, 4500, 0.5)
            + [0.4 * v for v in noise(11, seg)]
            + mix(sine(rate, 0, 392, 0.25, seg), sine(rate, 0, 494, 0.25, seg), sine(rate, 0, 587, 0.25, seg))
            + am)


def chirp_tone(rate):
    """The formats/ signal: a 300 to 3500 Hz chirp under a 440 Hz tone, below Nyquist at 8 kHz."""
    return mix(linear_chirp(rate, 0.5, 300, 3500, 0.4), sine(rate, 0.5, 440, 0.3))


def other(rate):
    """A second channel unlike chirp_tone: a falling chirp under a 660 Hz tone."""
    return mix(exp_chirp(rate, 0.5, 3000, 250, 0.4), sine(rate, 0.5, 660, 0.3))


def drifting(rate, seconds, p1, p2):
    """Two partials drifting slowly in pitch under a slow swell."""
    n = round(rate * seconds)
    a = glide(rate, n, lambda t: 220 + 110 * math.sin(TAU * t / p1), lambda t: 0.3 + 0.1 * math.sin(TAU * t / 7))
    b = glide(rate, n, lambda t: 520 + 180 * math.sin(TAU * t / p2), lambda t: 0.2 + 0.1 * math.cos(TAU * t / 5))
    return mix(a, b)


# file, kind, note, sample rate, sample format, dump, channels as functions of the rate
R16 = 16000
CLIPS = [
    ("tones/00-sine-440.wav", "tone", "440 Hz sine, 1.5 s", R16, "pcm16", False, [lambda r: sine(r, 1.5, 440, 0.6)]),
    ("tones/01-sine-1k.wav", "tone", "1 kHz sine, 1 s", R16, "pcm16", False, [lambda r: sine(r, 1, 1000, 0.6)]),
    ("tones/02-sine-110.wav", "tone", "low 110 Hz sine, 1 s", R16, "pcm16", False, [lambda r: sine(r, 1, 110, 0.6)]),
    ("tones/03-chord.wav", "tone", "C major chord (262, 330, 392 Hz), 1.5 s", R16, "pcm16", False,
     [lambda r: mix(sine(r, 1.5, 261.63, 0.25), sine(r, 1.5, 329.63, 0.25), sine(r, 1.5, 392.0, 0.25))]),
    ("tones/04-burst.wav", "tone", "0.4 s bursts of 660 and 990 Hz with 0.4 s silences between, 2 s", R16, "pcm16", False, [burst]),
    ("signals/00-chirp-up.wav", "signal", "linear chirp 200 to 6000 Hz, 1.5 s", R16, "pcm16", False,
     [lambda r: linear_chirp(r, 1.5, 200, 6000, 0.6)]),
    ("signals/01-chirp-down.wav", "signal", "exponential chirp 6000 down to 200 Hz, 1.5 s", R16, "pcm16", False,
     [lambda r: exp_chirp(r, 1.5, 6000, 200, 0.6)]),
    ("signals/02-white-noise.wav", "signal", "uniform white noise at 0.5, 1 s", R16, "pcm16", False,
     [lambda r: [0.5 * v for v in noise(1, r)]]),
    ("signals/03-lowpass-noise.wav", "signal", "white noise through a one-pole low-pass (alpha 0.05), 1 s", R16, "pcm16", False,
     [lambda r: normalise(lowpass(noise(2, r), 0.05), 0.6)]),
    ("signals/04-click-train.wav", "signal", "10 clicks per second, 1 s", R16, "pcm16", False, [lambda r: clicks(r, 1, 10)]),
    ("signals/05-am-tone.wav", "signal", "1 kHz tone amplitude-modulated at 4 Hz, depth 0.8, 1.5 s", R16, "pcm16", False,
     [lambda r: [0.4 * (1 + 0.8 * math.sin(TAU * 4 * i / r)) * math.sin(TAU * 1000 * i / r) for i in range(round(1.5 * r))]]),
    ("signals/06-fm-tone.wav", "signal", "800 Hz tone frequency-modulated at 5 Hz, 200 Hz deviation, 1.5 s", R16, "pcm16", False,
     [lambda r: [0.6 * math.sin(TAU * 800 * i / r + 40 * math.sin(TAU * 5 * i / r)) for i in range(round(1.5 * r))]]),
    ("signals/07-silence.wav", "signal", "digital silence, 1 s", R16, "pcm16", False, [lambda r: [0.0] * r]),
    ("signals/08-short-tone.wav", "signal", "523 Hz sine, 0.3 s", R16, "pcm16", True, [lambda r: sine(r, 0.3, 523.25, 0.6)]),
    ("signals/09-short-chirp.wav", "signal", "linear chirp 300 to 4000 Hz, 1 s", R16, "pcm16", True,
     [lambda r: linear_chirp(r, 1, 300, 4000, 0.6)]),
    ("signals/10-tone-noise.wav", "signal", "523 Hz sine at 0.5 plus white noise at 0.15, 2.3 s", R16, "pcm16", True,
     [lambda r: mix(sine(r, 2.3, 523.25, 0.5), [0.15 * v for v in noise(3, round(2.3 * r))])]),
    ("signals/11-mixture.wav", "signal", "content changing every 1.4 s: tone, chirp, noise, chord, AM tone, 7 s", R16, "pcm16", True,
     [mixture]),
    ("formats/00-rate-8000.wav", "format", "the chirp-plus-tone signal at 8000 Hz, 0.5 s", 8000, "pcm16", False, [chirp_tone]),
    ("formats/01-rate-22050.wav", "format", "the chirp-plus-tone signal at 22050 Hz, 0.5 s", 22050, "pcm16", False, [chirp_tone]),
    ("formats/02-rate-44100.wav", "format", "the chirp-plus-tone signal at 44100 Hz, 0.5 s", 44100, "pcm16", False, [chirp_tone]),
    ("formats/03-rate-48000.wav", "format", "the chirp-plus-tone signal at 48000 Hz, 0.5 s", 48000, "pcm16", False, [chirp_tone]),
    ("formats/04-stereo.wav", "format", "stereo: the chirp-plus-tone signal left, a falling chirp plus 660 Hz right, 0.5 s", R16, "pcm16", False,
     [chirp_tone, other]),
    ("formats/05-quad.wav", "format", "4 channels: chirp-plus-tone, falling chirp plus 660 Hz, 1 kHz sine, noise, 0.5 s", R16, "pcm16", False,
     [chirp_tone, other, lambda r: sine(r, 0.5, 1000, 0.5), lambda r: [0.4 * v for v in noise(4, r // 2)]]),
    ("formats/06-pcm8.wav", "format", "the chirp-plus-tone signal as 8-bit unsigned PCM, 0.5 s", R16, "pcm8", False, [chirp_tone]),
    ("formats/07-pcm24.wav", "format", "the chirp-plus-tone signal as 24-bit PCM, 0.5 s", R16, "pcm24", False, [chirp_tone]),
    ("formats/08-pcm32.wav", "format", "the chirp-plus-tone signal as 32-bit integer PCM, 0.5 s", R16, "pcm32", False, [chirp_tone]),
    ("formats/09-float32.wav", "format", "the chirp-plus-tone signal as 32-bit IEEE float, 0.5 s", R16, "float32", False, [chirp_tone]),
    ("long/00-30s.wav", "long", "slowly drifting tones, exactly 480000 samples: one full 30 s chunk", R16, "pcm16", False,
     [lambda r: drifting(r, 30, 13, 9)]),
    ("long/01-35s.wav", "long", "slowly drifting tones, 35 s: a full chunk and a 5 s chunk", R16, "pcm16", False,
     [lambda r: drifting(r, 35, 17, 11)]),
]

# sample format: WAVE format tag, bits per sample, encoder of one sample in [-1, 1]
FORMATS = {
    "pcm8": (1, 8, lambda v: struct.pack("<B", round(v * 127) + 128)),
    "pcm16": (1, 16, lambda v: struct.pack("<h", round(v * 32767))),
    "pcm24": (1, 24, lambda v: struct.pack("<i", round(v * 8388607))[:3]),
    "pcm32": (1, 32, lambda v: struct.pack("<i", round(v * 2147483647))),
    "float32": (3, 32, lambda v: struct.pack("<f", v)),
}


def wav_bytes(channels, rate, fmt):
    tag, bits, encode = FORMATS[fmt]
    frames = len(channels[0])
    assert all(len(c) == frames for c in channels)
    assert max(abs(v) for c in channels for v in c) <= 0.8
    data = b"".join(encode(v) for frame in zip(*channels) for v in frame)
    block = len(channels) * bits // 8
    fmt_chunk = struct.pack("<HHIIHH", tag, len(channels), rate, rate * block, block, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt_chunk)) + fmt_chunk + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body, frames


manifest, total = [], 0
for file, kind, note, rate, fmt, dump, signals in CLIPS:
    path = os.path.join(ROOT, file)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    blob, frames = wav_bytes([s(rate) for s in signals], rate, fmt)
    with open(path, "wb") as f:
        f.write(blob)
    total += len(blob)
    manifest.append({"file": file, "kind": kind, "note": note, "seconds": round(frames / rate, 6), "sample_rate": rate,
                     "channels": len(signals), "format": fmt, "dump": dump})
    print(f"{file:30} {frames / rate:7.3f} s {rate:6} Hz {len(signals)} ch {fmt:8} {len(blob):9} B{'  dump' if dump else ''}")

named = {m["file"] for m in manifest}
for d, _, files in os.walk(ROOT):
    for name in files:
        rel = os.path.relpath(os.path.join(d, name), ROOT)
        if name.endswith(".wav") and rel not in named:
            os.remove(os.path.join(d, name))
            print("removed", rel)

with open(os.path.join(ROOT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
print(f"{len(manifest)} clips, {total} bytes of WAV ({total / 1048576:.2f} MiB)")
