#!/usr/bin/env python3
"""Regenerates the spoken audio bench corpus that scripts/ab_audio.py times.

Writes 40 clips, clip-00.wav to clip-39.wav (715 s of 16 kHz mono 16-bit PCM from macOS `say` and
`afconvert`), each beside its transcript clip-NN.txt. A fixed seed picks 1 to 24 sentences per
clip and a speaking rate of 170, 185 or 200 words per minute, so the transcripts and rates are
the same on every run. The audio bytes depend on the installed macOS voice and can differ
between macOS versions; compare transcripts, not WAV bytes, when checking a regenerated corpus.

    scripts/make_speech_bench.py [OUT_DIR]    (default: models/audio/bench)
"""
import os
import random
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SENTENCES = [
    "The committee kept its benchmark rate on hold for a third straight meeting.",
    "Heavy rain is expected along the coast tonight, with flooding possible in low areas.",
    "Our team shipped the new search feature after six weeks of testing.",
    "Add two cups of flour, a pinch of salt, and stir until the dough comes together.",
    "The match went to extra time before the home side scored the winning goal.",
    "Researchers found that the new battery keeps most of its charge after a thousand cycles.",
    "Please remember to bring your badge to the office on Monday morning.",
    "The museum's new wing displays sculpture from the last two centuries.",
    "Traffic on the bridge is moving slowly because of roadworks in the left lane.",
    "A small startup released an open model that runs on a laptop.",
    "The orchestra opened the season with a symphony by a local composer.",
    "Shares fell sharply after the company cut its forecast for next year.",
    "The hiking trail climbs steeply through pine forest to a quiet alpine lake.",
    "Doctors recommend at least seven hours of sleep for most adults.",
    "The library will extend its opening hours during the exam period.",
    "Engineers traced the outage to a misconfigured load balancer.",
]
SENTENCES_PER_CLIP = [1] * 12 + [2] * 10 + [4] * 8 + [8] * 6 + [16] * 3 + [24] * 1
RATES = [170, 185, 200]


def main():
    out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models/audio/bench"))
    os.makedirs(out, exist_ok=True)
    random.seed(7)
    # Every text is drawn before any rate; drawing them clip by clip reproduces 1 of the 40 transcripts.
    clips = [" ".join(random.choice(SENTENCES) for _ in range(n)) for n in SENTENCES_PER_CLIP]
    with tempfile.TemporaryDirectory() as tmp:
        for i, text in enumerate(clips):
            aiff = os.path.join(tmp, f"clip-{i:02d}.aiff")
            subprocess.run(["say", "-r", str(random.choice(RATES)), "-o", aiff, text], check=True)
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", aiff,
                            os.path.join(out, f"clip-{i:02d}.wav")], check=True)
            with open(os.path.join(out, f"clip-{i:02d}.txt"), "w") as f:
                f.write(text + "\n")
    print(f"{len(clips)} clips in {out}")


if __name__ == "__main__":
    main()
