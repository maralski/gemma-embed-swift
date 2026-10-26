#!/usr/bin/env python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Interleaved A/B audio throughput and quality check for two gemma-embed binaries.

Runs A and B alternately (A B, B A, A B ...) holding the GPU lock for the whole run, timing
--bench-audio over DIR in two modes per sample: production batching (--batch-size auto, by mel
frames), the metric, and one clip per batch (--batch-size 1), the latency guard. The primary
figure is GPU seconds of the audio forward (audio and text towers): each sample sums every
batch's fastest of --runs GPU passes, so work from outside the lock drops out. Wall seconds
(median of --runs full embed(audios:) passes, WAV decode and log-mel included) and audio seconds
per wall second are reported beside it, with a verdict of their own. Each verdict rests on the
median of per-pair ratios A/B, each pair from back-to-back A and B samples: it must clear 3
percent, agree with the ratio of medians, and have its interquartile range on one side of 1.

Quality runs in the same lock: both binaries' --parity-audio on the committed fixtures (B min
cosine >= 0.9999 and within 1e-6 of A's, no token mismatches), and B's vectors over DIR within
min cosine 0.99999 of A's with the same token counts.

    scripts/ab_audio.py --a /path/to/old/gemma-embed --b .build/release/gemma-embed

It takes the GPU lock itself; do not wrap it in scripts/gpulock.py, whose lock it would wait on
forever. DIR defaults to models/audio/bench (scripts/make_speech_bench.py). Prints one JSON line;
work_a and work_b (clips, audio seconds, tokens) must match each other and DIR.
"""
import argparse
import fcntl
import json
import math
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOADS = os.environ.get("GEMMA_EMBED_MODELS", os.path.join(ROOT, "models"))
DEFAULT_DIR = os.path.join(DOWNLOADS, "audio/bench")
FIXTURES = os.path.join(ROOT, "Tests/GemmaEmbedTests/Fixtures/audio")
LOCK_PATH = "/tmp/gemma-embed-gpu.lock"
PARITY_MIN_COSINE = 0.9999
PARITY_SLACK = 1e-6
VECTOR_MIN_COSINE = 0.99999
MIN_SPEEDUP = 0.03
MODES = {"auto": "auto", "single": "1"}
METRICS = {"gpu": "gpu_seconds_minsum", "wall": "seconds_median"}
WORK_FIELDS = ["clips", "audio_seconds", "tokens"]
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".aiff", ".aif", ".caf", ".flac"}


def env():
    return dict(os.environ,
                GEMMA_EMBED_WEIGHTS=os.environ.get("GEMMA_EMBED_WEIGHTS", f"{DOWNLOADS}/embeddinggemma-2-Q8_0.gguf"),
                GEMMA_EMBED_AUDIO_WEIGHTS=os.environ.get("GEMMA_EMBED_AUDIO_WEIGHTS", f"{DOWNLOADS}/mmproj-embeddinggemma-2-Q8_0.gguf"))


def run(binary, args):
    return subprocess.run([binary] + args, capture_output=True, text=True, env=env())


def last_json(stdout):
    for line in reversed(stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError("no JSON line in output")


def bench(binary, folder, mode, runs, errors):
    p = run(binary, ["--bench-audio", folder, "--batch-size", MODES[mode], "--runs", str(runs)])
    if p.returncode != 0:
        errors.append(f"{binary} {mode}: exit {p.returncode}: {p.stderr.strip()[-300:]}")
        return None
    try:
        return last_json(p.stdout)
    except ValueError as e:
        errors.append(f"{binary} {mode}: {e}")
        return None


def vectors(binary, folder, errors):
    p = run(binary, ["--audio", folder])
    if p.returncode != 0:
        errors.append(f"{binary} --audio: exit {p.returncode}: {p.stderr.strip()[-300:]}")
        return None
    return {os.path.basename(r["file"]): r for r in map(json.loads, (l for l in p.stdout.splitlines() if l.startswith("{")))}


def parity(binary, folder, errors):
    p = run(binary, ["--parity-audio", folder])
    try:
        j = last_json(p.stdout)
    except ValueError as e:
        errors.append(f"{binary} --parity-audio: {e}: {p.stderr.strip()[-300:]}")
        return None
    j["exit"] = p.returncode
    return j


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def summary(xs):
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs)}


def compare(a, b, ratios):
    """The ab_images statistics for one metric: per-side summaries, ratio of medians, paired ratios."""
    sa, sb = summary(a), summary(b)
    speedup = sa["median"] / sb["median"]
    paired = statistics.median(ratios)
    q1, _, q3 = statistics.quantiles(sorted(ratios), n=4) if len(ratios) >= 2 else (ratios[0], 0, ratios[0])
    agree = (speedup > 1) == (paired > 1)
    verdict = ("faster" if paired > 1 else "slower") if agree and abs(paired - 1) > MIN_SPEEDUP and (q1 > 1 or q3 < 1) else "no measurable difference"
    return {"a": sa, "b": sb, "speedup": round(speedup, 4), "speedup_best": round(sa["min"] / sb["min"], 4),
            "speedup_paired": round(paired, 4), "paired_iqr": [round(q1, 4), round(q3, 4)],
            "paired_ratios": [round(x, 4) for x in ratios],
            "spread": round(max((sa["max"] - sa["min"]) / sa["median"], (sb["max"] - sb["min"]) / sb["median"]), 4),
            "verdict": verdict}


def work(j):
    return (j["clips"], round(j["audio_seconds"], 2), j["rows"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="baseline binary")
    ap.add_argument("--b", required=True, help="candidate binary")
    ap.add_argument("--dir", default=DEFAULT_DIR, help="folder of sound files")
    ap.add_argument("--pairs", type=int, default=5, help="samples per side and mode")
    ap.add_argument("--modes", default="auto,single", help="comma-separated subset of auto,single")
    ap.add_argument("--runs", type=int, default=3, help="passes per sample; the GPU sample is the sum of each batch's fastest pass")
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    modes = [m for m in args.modes.split(",") if m]
    errors = []
    clips_in_dir = sum(1 for f in os.listdir(args.dir) if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS)

    lock = open(LOCK_PATH, "w")
    t_lock = time.time()
    fcntl.flock(lock, fcntl.LOCK_EX)
    waited = time.time() - t_lock
    load = os.getloadavg()[0]

    samples = {(s, m): [] for s in "AB" for m in modes}
    pairs = {m: [] for m in modes}
    for i in range(args.pairs):
        pair = {}
        for side in (["A", "B"] if i % 2 == 0 else ["B", "A"]):
            binary = args.a if side == "A" else args.b
            for mode in modes:
                j = bench(binary, args.dir, mode, args.runs, errors)
                if j is not None:
                    pair[(side, mode)] = j
                    samples[(side, mode)].append(j)
        for mode in modes:
            if ("A", mode) in pair and ("B", mode) in pair:
                pairs[mode].append((pair[("A", mode)], pair[("B", mode)]))

    result = {"label": args.label, "dir": args.dir, "a": args.a, "b": args.b, "clips_in_dir": clips_in_dir,
              "load_avg_at_start": round(load, 2), "lock_wait_seconds": round(waited, 1), "work_fields": WORK_FIELDS}
    for mode in modes:
        wa, wb = sorted({work(j) for j in samples[("A", mode)]}), sorted({work(j) for j in samples[("B", mode)]})
        r = {"work_a": wa, "work_b": wb,
             "gpu_batches_a": sorted({j["gpu_batches"] for j in samples[("A", mode)]}),
             "gpu_batches_b": sorted({j["gpu_batches"] for j in samples[("B", mode)]})}
        if pairs[mode]:
            for name, key in METRICS.items():
                r[name] = compare([j[key] for j in samples[("A", mode)]], [j[key] for j in samples[("B", mode)]],
                                  [a[key] / b[key] for a, b in pairs[mode]])
            audio = wa[0][1]
            r.update(audio_seconds_per_second_a=round(audio / r["wall"]["a"]["median"], 1),
                     audio_seconds_per_second_b=round(audio / r["wall"]["b"]["median"], 1),
                     gpu_audio_seconds_per_second_a=round(audio / r["gpu"]["a"]["median"], 1),
                     gpu_audio_seconds_per_second_b=round(audio / r["gpu"]["b"]["median"], 1))
            if wa != wb:
                errors.append(f"{mode}: work differs, A {wa} B {wb}")
            if any(w[0] != clips_in_dir for w in wa + wb):
                errors.append(f"{mode}: benched clips differ from the {clips_in_dir} sound files in {args.dir}")
        else:
            r["gpu"] = r["wall"] = {"verdict": "inconclusive"}
        result[mode] = r

    if not args.skip_quality:
        pa, pb = parity(args.a, FIXTURES, errors), parity(args.b, FIXTURES, errors)
        result.update(parity_a=pa, parity_b=pb)
        va, vb = vectors(args.a, args.dir, errors), vectors(args.b, args.dir, errors)
        if va and vb:
            if va.keys() != vb.keys() or len(va) != clips_in_dir:
                errors.append(f"vector sets differ: A {len(va)} B {len(vb)} clips, {clips_in_dir} in DIR")
            keys = [k for k in va if k in vb]
            if any(len(va[k]["embedding"]) != len(vb[k]["embedding"]) for k in keys):
                errors.append("vector lengths differ between A and B")
            result["b_vs_a"] = {"clips": len(keys),
                                "min_cosine": min(cosine(va[k]["embedding"], vb[k]["embedding"]) for k in keys),
                                "max_abs_diff": max(max(abs(x - y) for x, y in zip(va[k]["embedding"], vb[k]["embedding"])) for k in keys),
                                "tokens_a": sum(va[k]["tokens"] for k in keys), "tokens_b": sum(vb[k]["tokens"] for k in keys),
                                "token_mismatches": sum(va[k]["tokens"] != vb[k]["tokens"] for k in keys)}
        ok = (pa is not None and pb is not None and pb["exit"] == 0 and pb["token_mismatches"] == 0
              and pb["min_cosine"] >= PARITY_MIN_COSINE and pb["min_cosine"] >= pa["min_cosine"] - PARITY_SLACK
              and "b_vs_a" in result and result["b_vs_a"]["clips"] == clips_in_dir
              and result["b_vs_a"]["token_mismatches"] == 0 and result["b_vs_a"]["min_cosine"] >= VECTOR_MIN_COSINE)
        result["quality"] = "PASS" if ok else "FAIL"
        result["quality_gate"] = {"parity_min_cosine": PARITY_MIN_COSINE, "parity_slack_vs_a": PARITY_SLACK,
                                  "b_vs_a_min_cosine": VECTOR_MIN_COSINE, "token_mismatches": 0}

    result["errors"] = errors
    result["error_count"] = len(errors)
    fcntl.flock(lock, fcntl.LOCK_UN)
    print(json.dumps(result, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
