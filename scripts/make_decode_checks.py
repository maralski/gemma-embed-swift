#!/usr/bin/env python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Writes small WAVs covering every bit depth, channel count and sample rate the audio decoder
must match, encodes a FLAC, an MP3 and an M4A from them with ffmpeg and an AIFF and a CAF with
afconvert, and dumps miniaudio's decode of each through scripts/audio_dump.cpp, so the
AudioDecoderTests parity tests have an oracle.

    python3 scripts/make_decode_checks.py <path/to/audio_dump>

Files go to models/audio/decode-checks/, dumps to models/audio/dumps/decode-<name>/, each with
samples.f32 (miniaudio's mono 16 kHz float32 output) and meta.json (the audio path, n_samples).
llama.cpp cannot decode M4A, AIFF or CAF (its is_audio_file accepts only WAV, MP3 and FLAC), so
those dumps hold the source WAV's miniaudio output with "oracle" naming the WAV: the lossless
AIFF and CAF must then decode to the same bits, and the M4A measures how far AVFoundation's AAC
decode sits from them. Paths are recorded resolved, so a worktree's models symlink does not leak
into meta.json. Deterministic: no random module.
"""
import json
import math
import os
import shutil
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.realpath(os.path.join(ROOT, "models"))
CHECKS = os.path.join(MODELS, "audio", "decode-checks")
DUMPS = os.path.join(MODELS, "audio", "dumps")
TEXT = os.path.join(MODELS, "embeddinggemma-2-Q8_0.gguf")
MMPROJ = os.path.join(MODELS, "mmproj-embeddinggemma-2-Q8_0.gguf")
FFMPEG = "/opt/homebrew/bin/ffmpeg"
AFCONVERT = "/usr/bin/afconvert"

WAVE_FORMAT_PCM = 1
WAVE_FORMAT_IEEE_FLOAT = 3
WAVE_FORMAT_EXTENSIBLE = 0xFFFE
# Tag-prefixed GUIDs of the sub-formats, the KSDATAFORMAT_SUBTYPE_* constants.
SUBFORMAT_TAIL = bytes.fromhex("000000001000800000aa00389b71")


def signal(frames, rate, channel, channels):
    """A sum of sines whose frequencies depend on the channel plus a small deterministic noise
    term, so every channel differs and rounding paths get exercised. In the range -0.9 to 0.9."""
    out = []
    state = 12345 + 7919 * channel
    f1 = 220.0 * (1 + 0.5 * channel)
    f2 = 1375.0 + 333.0 * channel
    for i in range(frames):
        t = i / rate
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        noise = (state / 0x7FFFFFFF) * 2 - 1
        v = 0.55 * math.sin(2 * math.pi * f1 * t) + 0.25 * math.sin(2 * math.pi * f2 * t + channel) + 0.08 * noise
        out.append(max(-0.9, min(0.9, v)))
    return out


def quantize(samples, fmt):
    """Encodes floats as the sample bytes of one of: u8, s16, s24, s32, f32, f64."""
    if fmt == "u8":
        return bytes(int(round((s + 1) * 127.5)) for s in samples)
    if fmt == "s16":
        return b"".join(struct.pack("<h", int(round(s * 32767))) for s in samples)
    if fmt == "s24":
        return b"".join(struct.pack("<i", int(round(s * 8388607)))[:3] for s in samples)
    if fmt == "s32":
        return b"".join(struct.pack("<i", int(round(s * 2147483647))) for s in samples)
    if fmt == "f32":
        return b"".join(struct.pack("<f", s) for s in samples)
    if fmt == "f64":
        return b"".join(struct.pack("<d", s) for s in samples)
    raise ValueError(fmt)


BITS = {"u8": 8, "s16": 16, "s24": 24, "s32": 32, "f32": 32, "f64": 64}


def chunk(fourcc, payload):
    """A RIFF chunk with the pad byte an odd payload needs."""
    return fourcc + struct.pack("<I", len(payload)) + payload + (b"\0" if len(payload) % 2 else b"")


def wav_bytes(fmt, channels, rate, frames, extensible=False, extra_chunks=()):
    bits = BITS[fmt]
    tag = WAVE_FORMAT_IEEE_FLOAT if fmt in ("f32", "f64") else WAVE_FORMAT_PCM
    block_align = channels * bits // 8
    fmt_payload = struct.pack("<HHIIHH", WAVE_FORMAT_EXTENSIBLE if extensible else tag, channels, rate,
                              rate * block_align, block_align, bits)
    if extensible:
        channel_mask = (1 << channels) - 1
        fmt_payload += struct.pack("<HHI", 22, bits, channel_mask) + struct.pack("<H", tag) + SUBFORMAT_TAIL
    interleaved = []
    per_channel = [signal(frames, rate, c, channels) for c in range(channels)]
    for i in range(frames):
        for c in range(channels):
            interleaved.append(per_channel[c][i])
    body = chunk(b"fmt ", fmt_payload)
    for fourcc, payload in extra_chunks:
        body += chunk(fourcc, payload)
    body += chunk(b"data", quantize(interleaved, fmt))
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body


def checks():
    """(name, bytes) for every WAV. About a third of a second each, odd frame counts throughout
    so the resampled lengths exercise the rounding rule."""
    out = []
    for fmt in BITS:
        out.append((f"mono16k-{fmt}", wav_bytes(fmt, 1, 16000, 5003)))
    out.append(("stereo16k-s16", wav_bytes("s16", 2, 16000, 5003)))
    for channels in (3, 4, 6):
        out.append((f"ch{channels}-16k-s16", wav_bytes("s16", channels, 16000, 4001)))
    for rate in (8000, 11025, 22050, 32000, 44100, 48000, 96000):
        frames = rate * 3 // 10 + 1
        out.append((f"mono{rate}-s16", wav_bytes("s16", 1, rate, frames)))
        out.append((f"stereo{rate}-s16", wav_bytes("s16", 2, rate, frames)))
    out.append(("mono44100-f32", wav_bytes("f32", 1, 44100, 13231)))
    out.append(("stereo48000-s24", wav_bytes("s24", 2, 48000, 14401)))
    out.append(("odd-count-mono16k-s16", wav_bytes("s16", 1, 16000, 777)))
    list_chunk = (b"LIST", b"INFOISFT" + struct.pack("<I", 7) + b"gemma\0\0" + b"\0")
    odd_chunk = (b"junk", b"\x01\x02\x03")
    out.append(("list-odd-chunks-mono16k-s16", wav_bytes("s16", 1, 16000, 3001, extra_chunks=(list_chunk, odd_chunk))))
    out.append(("extensible-stereo44100-s24", wav_bytes("s24", 2, 44100, 9001, extensible=True)))
    out.append(("extensible-mono16k-f32", wav_bytes("f32", 1, 16000, 2001, extensible=True)))
    return out


def run(cmd):
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def dump(audio_dump, audio_path, name):
    outdir = os.path.join(DUMPS, f"decode-{name}")
    os.makedirs(outdir, exist_ok=True)
    result = run([sys.executable, os.path.join(ROOT, "scripts", "gpulock.py"), "--", audio_dump, TEXT, MMPROJ, audio_path, outdir, "^$"])
    with open(os.path.join(outdir, "log.txt"), "w") as f:
        f.write(result.stdout)
    ok = result.returncode == 0 and os.path.exists(os.path.join(outdir, "samples.f32")) and os.path.exists(os.path.join(outdir, "meta.json"))
    if not ok:
        print(f"  audio_dump failed for {name} (exit {result.returncode}); see {outdir}/log.txt", flush=True)
    return outdir, ok


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    audio_dump = os.path.abspath(sys.argv[1])
    os.makedirs(CHECKS, exist_ok=True)
    os.makedirs(DUMPS, exist_ok=True)

    wavs = []
    for name, data in checks():
        path = os.path.join(CHECKS, name + ".wav")
        with open(path, "wb") as f:
            f.write(data)
        wavs.append((name, path))
    print(f"wrote {len(wavs)} WAVs to {CHECKS}")

    source = os.path.join(CHECKS, "stereo44100-s16.wav")
    encoded = {"mp3": ["-codec:a", "libmp3lame", "-b:a", "128k"], "m4a": ["-codec:a", "aac", "-b:a", "128k"], "flac": ["-codec:a", "flac"]}
    for ext, codec in encoded.items():
        result = run([FFMPEG, "-y", "-loglevel", "error", "-i", source] + codec + [os.path.join(CHECKS, f"stereo44100.{ext}")])
        if result.returncode != 0:
            sys.exit(f"ffmpeg failed for {ext}:\n{result.stdout}")
    for ext, fmt, data in (("aiff", "AIFF", "BEI16"), ("caf", "caff", "LEI16")):
        result = run([AFCONVERT, "-f", fmt, "-d", data, source, os.path.join(CHECKS, f"stereo44100.{ext}")])
        if result.returncode != 0:
            sys.exit(f"afconvert failed for {ext}:\n{result.stdout}")

    for name, path in wavs:
        dump(audio_dump, path, name)
    for ext in ("mp3", "flac"):
        dump(audio_dump, os.path.join(CHECKS, f"stereo44100.{ext}"), f"stereo44100-{ext}")

    # llama.cpp refuses these containers; their dumps carry the source WAV's samples as the oracle.
    source_dir = os.path.join(DUMPS, "decode-stereo44100-s16")
    with open(os.path.join(source_dir, "meta.json")) as f:
        meta = json.load(f)
    for ext in ("m4a", "aiff", "caf"):
        path = os.path.join(CHECKS, f"stereo44100.{ext}")
        outdir, ok = dump(audio_dump, path, f"stereo44100-{ext}")
        if ok:
            continue
        shutil.copyfile(os.path.join(source_dir, "samples.f32"), os.path.join(outdir, "samples.f32"))
        with open(os.path.join(outdir, "meta.json"), "w") as f:
            json.dump({"audio": path, "n_samples": meta["n_samples"], "sample_rate": meta["sample_rate"], "oracle": source}, f)
        print(f"  llama.cpp refuses {ext}; {outdir} holds the source WAV's samples with \"oracle\" set", flush=True)


if __name__ == "__main__":
    main()
