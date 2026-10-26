#!/usr/bin/env python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""GemmaEmbed vs llama-server: audio throughput over a folder of clips, and audio parity.

usage: compare_llamacpp_audio.py AUDIO_DIR [ROUNDS]
Weights come from $GEMMA_EMBED_MODELS (default: models/ in the checkout). Logs go to
$TMPDIR/gemma-embed-compare-audio. Starts one llama-server (-c 8192 -b 8192 -ub 8192, since the
embedding path needs a whole clip in one physical batch), holds the GPU lock for the whole run,
alternates the engines' order each round and waits out any xctest process. llama-server takes
one clip per request (mtmd encodes one audio at a time); GemmaEmbed runs --bench-audio with
production batching (--batch-size auto) and one clip per batch. The metric is audio seconds per
wall second, decoding included on both sides; the median over rounds is reported with the range.
The last block compares every clip's vector between the two engines.
"""
import base64, fcntl, json, os, statistics, subprocess, sys, time, urllib.request, wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = f"{ROOT}/.build/release/gemma-embed"
D = os.environ.get("GEMMA_EMBED_MODELS", os.path.join(ROOT, "models"))
TEXT, MMPROJ = f"{D}/embeddinggemma-2-Q8_0.gguf", f"{D}/mmproj-embeddinggemma-2-Q8_0.gguf"
if len(sys.argv) < 2:
    sys.exit(__doc__)
AUDIO = sys.argv[1]
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
HERE = os.path.join(os.environ.get("TMPDIR", "/tmp"), "gemma-embed-compare-audio")
os.makedirs(HERE, exist_ok=True)
PORT = 18095
ENV = dict(os.environ, GEMMA_EMBED_WEIGHTS=TEXT, GEMMA_EMBED_AUDIO_WEIGHTS=MMPROJ)
errors, waits = [], []


def tests_running():
    if subprocess.run(["pgrep", "-x", "xctest"], capture_output=True).returncode == 0:
        return True
    ps = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True).stdout.splitlines()
    return any("llama-server" in c and f"--port {PORT}" not in c for c in ps if not c.startswith(("ps ", "grep")))


def wait_for_quiet_gpu(label):
    t0 = time.time()
    while tests_running():
        time.sleep(5)
    if time.time() - t0 > 1:
        waits.append(f"{label}: waited {time.time() - t0:.0f} s for xctest")


def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/embeddings", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def wait_ready():
    for _ in range(600):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if json.load(r).get("status") == "ok":
                    return
        except Exception:
            time.sleep(0.5)
    sys.exit("llama-server did not become ready")


def last_json(out):
    return json.loads([l for l in out.splitlines() if l.startswith("{")][-1])


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / ((sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5))


def clips():
    files = sorted(f for f in os.listdir(AUDIO) if f.lower().endswith(".wav"))
    out = []
    for f in files:
        with wave.open(os.path.join(AUDIO, f)) as w:
            seconds = w.getnframes() / w.getframerate()
        out.append((f, seconds))
    return out


def gemma_bench(batch):
    p = subprocess.run([BIN, "--bench-audio", AUDIO, "--batch-size", batch, "--runs", "1"], capture_output=True, text=True, env=ENV)
    if p.returncode != 0:
        errors.append(f"gemma audio {batch} exit {p.returncode}: {p.stderr[-200:]}")
        return None
    j = last_json(p.stdout)
    return {"clips": j["clips"], "audio_seconds": j["audio_seconds"], "seconds": j["seconds_median"],
            "audio_seconds_per_second": j["audio_seconds"] / j["seconds_median"]}


def llama_bench(payloads, keep=None):
    t0 = time.time()
    tokens = 0
    for f, body in payloads:
        r = post({"model": "e", "input": [body]})
        tokens += r["usage"]["prompt_tokens"]
        if keep is not None:
            keep[f] = r["data"][0]["embedding"]
    s = time.time() - t0
    total = sum(seconds for _, seconds in clips())
    return {"clips": len(payloads), "audio_seconds": total, "prompt_tokens": tokens, "seconds": s, "audio_seconds_per_second": total / s}


def main():
    lock = open("/tmp/gemma-embed-gpu.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    log = open(f"{HERE}/llama-server.log", "w")
    server = subprocess.Popen(["llama-server", "-m", TEXT, "--mmproj", MMPROJ, "--embedding", "-c", "8192", "-b", "8192", "-ub", "8192",
                               "--port", str(PORT)], stdout=log, stderr=log)
    results = {k: [] for k in ["gemma_auto", "gemma_1", "llama_1"]}
    llama_vecs = {}
    try:
        wait_ready()
        payloads = []
        for f, _ in clips():
            data = base64.b64encode(open(os.path.join(AUDIO, f), "rb").read()).decode()
            payloads.append((f, {"content": [{"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}]}))
        gemma_bench("auto")
        llama_bench(payloads[:3])
        for i in range(ROUNDS):
            order = [("gemma_auto", lambda: gemma_bench("auto")), ("gemma_1", lambda: gemma_bench("1")),
                     ("llama_1", lambda: llama_bench(payloads, llama_vecs if i == 0 else None))]
            if i % 2:
                order.reverse()
            for name, fn in order:
                wait_for_quiet_gpu(name)
                r = fn()
                if r:
                    results[name].append(r)
                    print(f"round {i} {name}: {r['audio_seconds_per_second']:.1f} audio s/s ({r['seconds']:.2f} s)", file=sys.stderr)
        p = subprocess.run([BIN, "--audio", AUDIO], capture_output=True, text=True, env=ENV)
        gemma_vecs = {os.path.basename(json.loads(l)["file"]): json.loads(l)["embedding"] for l in p.stdout.splitlines() if l.startswith("{")}
        common = sorted(set(gemma_vecs) & set(llama_vecs))
        cos = [cosine(gemma_vecs[f], llama_vecs[f]) for f in common]
        summary = {"rounds": ROUNDS, "clips": len(payloads), "audio_seconds": results["llama_1"][0]["audio_seconds"] if results["llama_1"] else None,
                   "audio_parity": {"clips": len(common), "min_cosine": min(cos), "median_cosine": statistics.median(cos),
                                    "worst": common[cos.index(min(cos))]} if cos else None,
                   "errors": errors, "waits": waits}
        for k, rows in results.items():
            v = [r["audio_seconds_per_second"] for r in rows]
            if v:
                summary[k] = {"median_audio_seconds_per_second": statistics.median(v), "min": min(v), "max": max(v),
                              "median_seconds": statistics.median(r["seconds"] for r in rows), "samples": len(v)}
        print(json.dumps(summary, indent=1))
    finally:
        server.terminate()
        server.wait()


if __name__ == "__main__":
    main()
