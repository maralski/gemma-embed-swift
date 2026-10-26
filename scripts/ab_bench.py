#!/usr/bin/env python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Interleaved A/B throughput and quality check for two gemma-embed binaries.

Runs A and B alternately (A B B A A B ...) so drift and GPU contention hit both sides alike,
takes an exclusive lock so two measurements never share the GPU, and gates the candidate on
quality: llama.cpp parity fixtures and a direct comparison of its vectors against A's.

    scripts/ab_bench.py --a /path/to/old/gemma-embed --b .build/release/gemma-embed --bench /path/to/posts.json

Prints one JSON line with medians, ranges, the speedup, error counts and the quality verdict.
The primary metric is GPU seconds of the timed batches (summed from the per-batch trace lines),
so CPU load from builds elsewhere on the machine does not enter it; wall seconds are reported too.
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

DOWNLOADS = os.environ.get("GEMMA_EMBED_MODELS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
DEFAULT_WEIGHTS = os.environ.get("GEMMA_EMBED_WEIGHTS", f"{DOWNLOADS}/embeddinggemma-2-Q8_0.gguf")
DEFAULT_BENCH = f"{DOWNLOADS}/posts.json"
DEFAULT_FIXTURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Tests/GemmaEmbedTests/Fixtures")
LOCK_PATH = "/tmp/gemma-embed-gpu.lock"
QUALITY_MIN_COSINE = 0.99999


def run(binary, args, env, stdin=None):
    p = subprocess.run([binary] + args, input=stdin, capture_output=True, text=True, env=env)
    return p


def last_json(stdout):
    for line in reversed(stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError("no JSON line in output")


def bench_once(binary, extra, bench, env, errors):
    """One --bench run. GPU seconds come from the per-batch trace lines (the first is the
    warm-up batch), so compiles and other CPU load on the machine do not enter the metric."""
    p = run(binary, ["--bench", bench, "--runs", "1"] + extra, dict(env, GEMMA_EMBED_TRACE="1"))
    if p.returncode != 0:
        errors.append(f"{binary}: exit {p.returncode}: {p.stderr.strip()[-300:]}")
        return None
    try:
        j = last_json(p.stdout)
    except ValueError as e:
        errors.append(f"{binary}: {e}")
        return None
    gpu_ms = []
    for line in p.stdout.splitlines():
        if line.startswith("run:") and " gpu " in line:
            gpu_ms.append(float(line.split(" gpu ")[1].split(" ms")[0]))
    if len(gpu_ms) < 2:
        errors.append(f"{binary}: {len(gpu_ms)} trace lines, expected the warm-up batch and the timed batches")
        return None
    j["gpu_seconds"] = sum(gpu_ms[1:]) / 1000.0
    j["gpu_batch_ms"] = gpu_ms[1:]
    j["batches"] = len(gpu_ms) - 1
    return j


def bench_texts(path):
    with open(path) as f:
        data = f.read()
    try:
        j = json.loads(data)
        posts = j.get("posts") if isinstance(j, dict) else None
    except json.JSONDecodeError:
        posts = None
    if posts is not None:
        texts = []
        for p in posts:
            t = ((p.get("gist") or "") + "\n\n" + (p.get("text") or "")[:2000]).strip()
            texts.append(" ".join(t.split()))
        return texts
    return [" ".join(l.split()) for l in data.splitlines() if l.strip()]


def embed_all(binary, texts, env, errors):
    p = run(binary, [], env, stdin="\n".join(texts) + "\n")
    if p.returncode != 0:
        errors.append(f"{binary} stdin embed: exit {p.returncode}: {p.stderr.strip()[-300:]}")
        return None
    rows = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    rows.sort(key=lambda r: r["index"])
    if len(rows) != len(texts):
        errors.append(f"{binary} stdin embed: {len(rows)} vectors for {len(texts)} texts")
        return None
    return [r["embedding"] for r in rows]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def parity(binary, fixtures, env, errors):
    p = run(binary, ["--parity", fixtures], env)
    try:
        j = last_json(p.stdout)
    except ValueError as e:
        errors.append(f"{binary} parity: {e}: {p.stderr.strip()[-300:]}")
        return None
    j["exit"] = p.returncode
    return j


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="baseline binary")
    ap.add_argument("--b", required=True, help="candidate binary")
    ap.add_argument("--a-args", default="", help="extra CLI args for A, space separated")
    ap.add_argument("--b-args", default="", help="extra CLI args for B, space separated")
    ap.add_argument("--bench", default=DEFAULT_BENCH)
    ap.add_argument("--pairs", type=int, default=7, help="samples per side")
    ap.add_argument("--fixtures", default=DEFAULT_FIXTURES)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    env = dict(os.environ, GEMMA_EMBED_WEIGHTS=args.weights)
    a_extra = args.a_args.split()
    b_extra = args.b_args.split()
    errors = []

    lock = open(LOCK_PATH, "w")
    t_lock = time.time()
    fcntl.flock(lock, fcntl.LOCK_EX)
    waited = time.time() - t_lock

    load = os.getloadavg()[0]
    secs = {"A": [], "B": []}
    wall = {"A": [], "B": []}
    batch_ms = {"A": [], "B": []}
    tokens = {"A": set(), "B": set()}
    for i in range(args.pairs):
        order = ["A", "B"] if i % 2 == 0 else ["B", "A"]
        for side in order:
            binary, extra = (args.a, a_extra) if side == "A" else (args.b, b_extra)
            j = bench_once(binary, extra, args.bench, env, errors)
            if j is None:
                continue
            secs[side].append(j["gpu_seconds"])
            wall[side].append(j["seconds_median"])
            batch_ms[side].append(j["gpu_batch_ms"])
            tokens[side].add(j["tokens"])

    def summary(xs):
        if not xs:
            return None
        q1, _, q3 = statistics.quantiles(xs, n=4) if len(xs) >= 2 else (min(xs), 0, max(xs))
        return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "q1": q1, "q3": q3, "n": len(xs)}

    def minsum(rows):
        # Sum over batches of the per-batch minimum across samples: contention bursts hit
        # different batches in different samples, so this tracks the uncontended time.
        if not rows or len({len(r) for r in rows}) != 1:
            return None
        return sum(min(col) for col in zip(*rows)) / 1000.0

    sa, sb = summary(secs["A"]), summary(secs["B"])
    ma, mb = minsum(batch_ms["A"]), minsum(batch_ms["B"])
    result = {
        "label": args.label,
        "bench": os.path.basename(args.bench),
        "a": args.a, "b": args.b, "a_args": args.a_args, "b_args": args.b_args,
        "gpu_seconds_a": sa, "gpu_seconds_b": sb,
        "gpu_minsum_a": ma, "gpu_minsum_b": mb,
        "wall_seconds_a": summary(wall["A"]), "wall_seconds_b": summary(wall["B"]),
        "tokens_a": sorted(tokens["A"]), "tokens_b": sorted(tokens["B"]),
        "load_avg_at_start": round(load, 2), "lock_wait_seconds": round(waited, 1),
    }
    if sa and sb:
        speedup = sa["median"] / sb["median"]
        # Interquartile spread, so one contended sample does not hide a real difference. The
        # best-of ratio is the second estimator: under intermittent GPU contention the minimum
        # tracks the uncontended time.
        spread = max((sa["q3"] - sa["q1"]) / sa["median"], (sb["q3"] - sb["q1"]) / sb["median"])
        result["speedup"] = round(speedup, 4)
        result["speedup_best"] = round(sa["min"] / sb["min"], 4)
        result["gpu_tokens_per_second_a"] = round(sorted(tokens["A"])[0] / sa["median"], 1)
        result["gpu_tokens_per_second_b"] = round(sorted(tokens["B"])[0] / sb["median"], 1)
        wa, wb = result["wall_seconds_a"], result["wall_seconds_b"]
        if wa and wb:
            result["wall_speedup"] = round(wa["median"] / wb["median"], 4)
        result["spread"] = round(spread, 4)
        estimates = [speedup, sa["min"] / sb["min"]]
        if ma and mb:
            result["speedup_minsum"] = round(ma / mb, 4)
            estimates.append(ma / mb)
        # Three estimators (median ratio, best-of ratio, min-sum ratio) must agree in direction
        # and all clear 3 percent before the harness calls a difference. On a busy machine any
        # one of them alone has been seen 9 percent off on identical binaries.
        agree = all(e > 1 for e in estimates) or all(e < 1 for e in estimates)
        if agree and min(abs(e - 1) for e in estimates) > 0.03:
            result["verdict"] = "faster" if speedup > 1 else "slower"
        else:
            result["verdict"] = "no measurable difference"
    else:
        result["verdict"] = "inconclusive"

    if not args.skip_quality:
        pa = parity(args.a, args.fixtures, env, errors)
        pb = parity(args.b, args.fixtures, env, errors)
        result["parity_a"] = pa
        result["parity_b"] = pb
        texts = bench_texts(args.bench)
        va = embed_all(args.a, texts, env, errors)
        vb = embed_all(args.b, texts, env, errors)
        if va and vb:
            cosines = [cosine(x, y) for x, y in zip(va, vb)]
            diffs = [max(abs(p - q) for p, q in zip(x, y)) for x, y in zip(va, vb)]
            result["b_vs_a"] = {"texts": len(texts), "min_cosine": min(cosines), "max_abs_diff": max(diffs)}
        quality_ok = (
            pb is not None and pb["exit"] == 0 and pb["token_mismatches"] == 0 and pb["min_cosine"] >= QUALITY_MIN_COSINE
            and "b_vs_a" in result and result["b_vs_a"]["min_cosine"] >= QUALITY_MIN_COSINE
        )
        result["quality"] = "PASS" if quality_ok else "FAIL"
        result["quality_gate"] = {"parity_min_cosine": QUALITY_MIN_COSINE, "b_vs_a_min_cosine": QUALITY_MIN_COSINE, "token_mismatches": 0}

    result["errors"] = errors
    result["error_count"] = len(errors)
    fcntl.flock(lock, fcntl.LOCK_UN)
    print(json.dumps(result, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
