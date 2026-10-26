#!/usr/bin/env python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Interleaved A/B image throughput and quality check for two gemma-embed binaries.

Runs A and B alternately under the GPU lock, timing --bench-images over DIR twice per sample:
production batching (--batch-size auto) and one image at a time (--batch-size 1). The metric is
GPU seconds of the image forward (vision and text towers). Each sample sums every batch's fastest
of --runs GPU passes, so work from outside the GPU lock drops out; wall seconds are reported
beside it. The verdict rests on the median of per-pair ratios, each from back-to-back A and B
samples: it must clear 3 percent, agree with the ratio of medians, and have its interquartile
range on one side of 1. The candidate must keep llama-server image parity and llama.cpp text parity, and its image
vectors over DIR must match A's.

    scripts/ab_images.py --a /path/to/old/gemma-embed --b .build/release/gemma-embed --dir /path/to/images
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
DOWNLOADS = os.environ.get("GEMMA_EMBED_MODELS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
DEFAULT_DIR = os.path.join(ROOT, "models/fixtures/images/scenes")
FIXTURES = os.path.join(ROOT, "Tests/GemmaEmbedTests/Fixtures")
LOCK_PATH = "/tmp/gemma-embed-gpu.lock"
VECTOR_MIN_COSINE = 0.99999
PARITY_SLACK = 2e-6
MODES = {"auto": "auto", "single": "1"}


def env():
    return dict(os.environ,
                GEMMA_EMBED_WEIGHTS=os.environ.get("GEMMA_EMBED_WEIGHTS", f"{DOWNLOADS}/embeddinggemma-2-Q8_0.gguf"),
                GEMMA_EMBED_IMAGE_WEIGHTS=os.environ.get("GEMMA_EMBED_IMAGE_WEIGHTS", f"{DOWNLOADS}/mmproj-embeddinggemma-2-Q8_0.gguf"))


def run(binary, args):
    return subprocess.run([binary] + args, capture_output=True, text=True, env=env())


def last_json(stdout):
    for line in reversed(stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError("no JSON line in output")


def bench(binary, folder, mode, runs, errors):
    p = run(binary, ["--bench-images", folder, "--batch-size", MODES[mode], "--runs", str(runs)])
    if p.returncode != 0:
        errors.append(f"{binary} {mode}: exit {p.returncode}: {p.stderr.strip()[-300:]}")
        return None
    try:
        return last_json(p.stdout)
    except ValueError as e:
        errors.append(f"{binary} {mode}: {e}")
        return None


def vectors(binary, folder, errors):
    p = run(binary, ["--images", folder])
    if p.returncode != 0:
        errors.append(f"{binary} --images: exit {p.returncode}: {p.stderr.strip()[-300:]}")
        return None
    return {os.path.basename(r["file"]): r["embedding"] for r in map(json.loads, (l for l in p.stdout.splitlines() if l.startswith("{")))}


def parity(binary, flag, folder, errors):
    p = run(binary, [flag, folder])
    try:
        j = last_json(p.stdout)
    except ValueError as e:
        errors.append(f"{binary} {flag}: {e}: {p.stderr.strip()[-300:]}")
        return None
    j["exit"] = p.returncode
    return j


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def summary(xs):
    if not xs:
        return None
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="baseline binary")
    ap.add_argument("--b", required=True, help="candidate binary")
    ap.add_argument("--dir", default=DEFAULT_DIR, help="folder of images")
    ap.add_argument("--pairs", type=int, default=5, help="samples per side and mode")
    ap.add_argument("--modes", default="auto,single", help="comma-separated subset of auto,single")
    ap.add_argument("--runs", type=int, default=3, help="GPU passes per sample; the sample is the sum of each batch's fastest pass")
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    modes = [m for m in args.modes.split(",") if m]
    errors = []

    lock = open(LOCK_PATH, "w")
    t_lock = time.time()
    fcntl.flock(lock, fcntl.LOCK_EX)
    waited = time.time() - t_lock
    load = os.getloadavg()[0]

    gpu = {(s, m): [] for s in "AB" for m in modes}
    paired = {m: [] for m in modes}
    wall = {(s, m): [] for s in "AB" for m in modes}
    work = {(s, m): set() for s in "AB" for m in modes}
    for i in range(args.pairs):
        pair = {}
        for side in (["A", "B"] if i % 2 == 0 else ["B", "A"]):
            binary = args.a if side == "A" else args.b
            for mode in modes:
                j = bench(binary, args.dir, mode, args.runs, errors)
                if j is None:
                    continue
                if "gpu_seconds_minsum" not in j:
                    errors.append(f"{binary} prints no gpu_seconds_minsum; rebuild it from hill/vision-profile or later")
                    continue
                pair[(side, mode)] = j["gpu_seconds_minsum"]
                gpu[(side, mode)].append(j["gpu_seconds_minsum"])
                wall[(side, mode)].append(j["seconds_median"])
                work[(side, mode)].add((j["images"], j["rows"]))
        for mode in modes:
            if ("A", mode) in pair and ("B", mode) in pair:
                paired[mode].append(pair[("A", mode)] / pair[("B", mode)])

    result = {"label": args.label, "dir": args.dir, "a": args.a, "b": args.b,
              "load_avg_at_start": round(load, 2), "lock_wait_seconds": round(waited, 1)}
    for mode in modes:
        ga, gb = summary(gpu[("A", mode)]), summary(gpu[("B", mode)])
        wa, wb = summary(wall[("A", mode)]), summary(wall[("B", mode)])
        r = {"gpu_seconds_a": ga, "gpu_seconds_b": gb, "wall_seconds_a": wa, "wall_seconds_b": wb,
             "work_a": sorted(work[("A", mode)]), "work_b": sorted(work[("B", mode)])}
        if ga and gb:
            images = sorted(work[("A", mode)])[0][0]
            speedup, best = ga["median"] / gb["median"], ga["min"] / gb["min"]
            r.update(speedup=round(speedup, 4), speedup_best=round(best, 4),
                     wall_speedup=round(wa["median"] / wb["median"], 4),
                     gpu_images_per_second_a=round(images / ga["median"], 2), gpu_images_per_second_b=round(images / gb["median"], 2),
                     spread=round(max((ga["max"] - ga["min"]) / ga["median"], (gb["max"] - gb["min"]) / gb["median"]), 4))
            ratios = sorted(paired[mode])
            paired_speedup = statistics.median(ratios)
            q1, _, q3 = statistics.quantiles(ratios, n=4) if len(ratios) >= 2 else (ratios[0], 0, ratios[0])
            r.update(speedup_paired=round(paired_speedup, 4), paired_iqr=[round(q1, 4), round(q3, 4)], paired_ratios=[round(x, 4) for x in paired[mode]])
            agree = (speedup > 1) == (paired_speedup > 1)
            r["verdict"] = ("faster" if paired_speedup > 1 else "slower") if agree and abs(paired_speedup - 1) > 0.03 and (q1 > 1 or q3 < 1) else "no measurable difference"
            if work[("A", mode)] != work[("B", mode)]:
                errors.append(f"{mode}: work differs, A {sorted(work[('A', mode)])} B {sorted(work[('B', mode)])}")
        else:
            r["verdict"] = "inconclusive"
        result[mode] = r

    if not args.skip_quality:
        pia = parity(args.a, "--parity-images", os.path.join(FIXTURES, "images"), errors)
        pib = parity(args.b, "--parity-images", os.path.join(FIXTURES, "images"), errors)
        pta = parity(args.a, "--parity", FIXTURES, errors)
        ptb = parity(args.b, "--parity", FIXTURES, errors)
        result.update(image_parity_a=pia, image_parity_b=pib, text_parity_a=pta, text_parity_b=ptb)
        va, vb = vectors(args.a, args.dir, errors), vectors(args.b, args.dir, errors)
        if va and vb:
            if va.keys() != vb.keys():
                errors.append("image vector sets differ")
            cos = [cosine(va[k], vb[k]) for k in va if k in vb]
            result["b_vs_a"] = {"images": len(cos), "min_cosine": min(cos),
                                "max_abs_diff": max(max(abs(x - y) for x, y in zip(va[k], vb[k])) for k in va if k in vb)}
        ok = (pia and pib and pta and ptb and "b_vs_a" in result
              and pib["token_mismatches"] == 0 and pib["min_cosine"] >= pia["min_cosine"] - PARITY_SLACK
              and ptb["exit"] == 0 and ptb["token_mismatches"] == 0 and ptb["min_cosine"] >= pta["min_cosine"] - PARITY_SLACK
              and result["b_vs_a"]["min_cosine"] >= VECTOR_MIN_COSINE)
        result["quality"] = "PASS" if ok else "FAIL"
        result["quality_gate"] = {"image_and_text_parity": f"B min cosine within {PARITY_SLACK} of A's, 0 token mismatches",
                                  "b_vs_a_min_cosine": VECTOR_MIN_COSINE}

    result["errors"] = errors
    result["error_count"] = len(errors)
    fcntl.flock(lock, fcntl.LOCK_UN)
    print(json.dumps(result, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
