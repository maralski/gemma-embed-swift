#!/usr/bin/env python3
"""GemmaEmbed vs llama-server: one short query's latency and image throughput over a folder.

usage: compare_llamacpp.py IMAGE_DIR [ROUNDS]
Weights come from $GEMMA_EMBED_MODELS (default: models/ in the checkout). Logs and the query file
go to $TMPDIR/gemma-embed-compare.
Starts three llama-server instances, holds the GPU lock for the whole run, alternates the
engines' order each round, and waits out any xctest process (test runs that skip the lock).
"""
import base64, fcntl, json, math, os, statistics, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = f"{ROOT}/.build/release/gemma-embed"
D = os.environ.get("GEMMA_EMBED_MODELS", os.path.join(ROOT, "models"))
TEXT, MMPROJ = f"{D}/embeddinggemma-2-Q8_0.gguf", f"{D}/mmproj-embeddinggemma-2-Q8_0.gguf"
if len(sys.argv) < 2:
    sys.exit(__doc__)
IMAGES = sys.argv[1]
HERE = os.path.join(os.environ.get("TMPDIR", "/tmp"), "gemma-embed-compare")
os.makedirs(HERE, exist_ok=True)
QUERY = "Which central bank left interest rates unchanged at its October meeting this year"
QUERY_FILE = f"{HERE}/q.txt"
with open(QUERY_FILE, "w") as f:
    f.write(QUERY + "\n")
QUERY_PREFIX = "task: search result | query: "
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
QUERY_REPS = 20
ENV = dict(os.environ, GEMMA_EMBED_WEIGHTS=TEXT, GEMMA_EMBED_IMAGE_WEIGHTS=MMPROJ)

SERVERS = {
    "text_best": (18092, ["-m", TEXT, "--embedding", "-c", "8192", "-b", "8192", "-ub", "8192", "-np", "8"]),
    "text_np1": (18093, ["-m", TEXT, "--embedding", "-c", "8192", "-b", "8192", "-ub", "8192", "-np", "1"]),
    "image_default": (18094, ["-m", TEXT, "--mmproj", MMPROJ, "--embedding"]),
}
errors, waits = [], []


def tests_running():
    """xctest, or a llama-server that is not one of ours, means GPU work outside the lock."""
    if subprocess.run(["pgrep", "-x", "xctest"], capture_output=True).returncode == 0:
        return True
    ps = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True).stdout.splitlines()
    return any("llama-server" in c and "--port 1809" not in c for c in ps if not c.startswith(("ps ", "grep")))


def wait_for_quiet_gpu(label):
    t0 = time.time()
    while tests_running():
        time.sleep(5)
    if time.time() - t0 > 1:
        waits.append(f"{label}: waited {time.time() - t0:.0f} s for xctest")


def post(port, inputs):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/embeddings", data=json.dumps({"input": inputs}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def wait_ready(port):
    for _ in range(240):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise SystemExit(f"server on {port} never became ready")


def last_json(out):
    return next(json.loads(l) for l in reversed(out.strip().splitlines()) if l.startswith("{"))


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


def gemma_query():
    p = subprocess.run([BIN, "--task", "query", "--bench", QUERY_FILE, "--runs", str(QUERY_REPS)], capture_output=True, text=True, env=ENV)
    if p.returncode:
        errors.append(f"gemma query exit {p.returncode}")
        return None
    j = last_json(p.stdout)
    return {"median_ms": j["seconds_median"] * 1000, "best_ms": j["seconds_best"] * 1000, "tokens": j["tokens"]}


def llama_query(port):
    text = QUERY_PREFIX + QUERY
    post(port, [text])
    ms, tokens = [], None
    for _ in range(QUERY_REPS):
        t0 = time.perf_counter()
        r = post(port, [text])
        ms.append((time.perf_counter() - t0) * 1000)
        tokens = r["usage"]["prompt_tokens"]
        if len(r["data"]) != 1 or len(r["data"][0]["embedding"]) != 768:
            errors.append(f"llama query on {port}: bad response shape")
    return {"median_ms": statistics.median(ms), "best_ms": min(ms), "tokens": tokens}


def gemma_images(batch):
    p = subprocess.run([BIN, "--bench-images", IMAGES, "--batch-size", batch, "--runs", "1"], capture_output=True, text=True, env=ENV)
    if p.returncode:
        errors.append(f"gemma images {batch} exit {p.returncode}: {p.stderr[-200:]}")
        return None
    j = last_json(p.stdout)
    return {"images": j["images"], "rows": j.get("rows"), "seconds": j["seconds_median"], "images_per_second": j["images"] / j["seconds_median"]}


def image_payloads():
    out = []
    for f in sorted(os.listdir(IMAGES)):
        ext = f.lower().rsplit(".", 1)[-1]
        if ext not in ("jpg", "jpeg", "png"):
            continue
        mime = "image/png" if ext == "png" else "image/jpeg"
        data = base64.b64encode(open(os.path.join(IMAGES, f), "rb").read()).decode()
        out.append((f, {"content": [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}]}))
    return out


def llama_images(port, payloads, per_request, keep=None):
    post(port, [payloads[0][1]])
    t0 = time.perf_counter()
    tokens = 0
    for i in range(0, len(payloads), per_request):
        chunk = payloads[i:i + per_request]
        r = post(port, [p for _, p in chunk])
        tokens += r["usage"]["prompt_tokens"]
        if len(r["data"]) != len(chunk):
            errors.append(f"llama images {port}: {len(r['data'])} vectors for {len(chunk)} images")
        if keep is not None:
            for (name, _), d in zip(chunk, sorted(r["data"], key=lambda d: d["index"])):
                keep[name] = d["embedding"]
    s = time.perf_counter() - t0
    return {"images": len(payloads), "prompt_tokens": tokens, "seconds": s, "images_per_second": len(payloads) / s}


def main():
    procs = []
    for name, (port, args) in SERVERS.items():
        log = open(f"{HERE}/server-{name}.log", "w")
        procs.append(subprocess.Popen(["/opt/homebrew/bin/llama-server", "--port", str(port)] + args, stdout=log, stderr=subprocess.STDOUT))
    try:
        for port, _ in SERVERS.values():
            wait_ready(port)
        payloads = image_payloads()
        lock = open("/tmp/gemma-embed-gpu.lock", "w")
        fcntl.flock(lock, fcntl.LOCK_EX)
        load = os.getloadavg()
        keys = ["q_gemma", "q_llama_best", "q_llama_np1", "img_gemma_1", "img_gemma_auto", "img_llama_1", "img_llama_8"]
        res = {k: [] for k in keys}
        llama_vecs = {}
        for i in range(ROUNDS):
            steps = [
                ("q_gemma", gemma_query), ("q_llama_best", lambda: llama_query(18092)), ("q_llama_np1", lambda: llama_query(18093)),
                ("img_gemma_1", lambda: gemma_images("1")), ("img_gemma_auto", lambda: gemma_images("auto")),
                ("img_llama_1", lambda: llama_images(18094, payloads, 1, llama_vecs if i == 0 else None)),
                ("img_llama_8", lambda: llama_images(18094, payloads, 8)),
            ]
            if i % 2:
                steps = [s for s in steps if "llama" in s[0]] + [s for s in steps if "gemma" in s[0]]
            for key, fn in steps:
                wait_for_quiet_gpu(f"{i} {key}")
                r = fn()
                if tests_running():
                    errors.append(f"{i} {key}: xctest started during the step")
                if r:
                    res[key].append(r)
                    print(i, key, json.dumps(r), flush=True)
        fcntl.flock(lock, fcntl.LOCK_UN)

        p = subprocess.run([BIN, "--images", IMAGES], capture_output=True, text=True, env=ENV)
        gemma_vecs = {os.path.basename(r["file"]): r["embedding"] for r in map(json.loads, (l for l in p.stdout.splitlines() if l.startswith("{")))}
        common = sorted(set(gemma_vecs) & set(llama_vecs))
        cos = [cosine(gemma_vecs[k], llama_vecs[k]) for k in common]
        qv = last_json(subprocess.run([BIN, "--task", "query"], input=QUERY + "\n", capture_output=True, text=True, env=ENV).stdout)["embedding"]
        ql = post(18092, [QUERY_PREFIX + QUERY])["data"][0]["embedding"]

        summary = {"query": QUERY, "rounds": ROUNDS, "query_reps_per_round": QUERY_REPS,
                   "load_avg_at_start": load, "load_avg_at_end": os.getloadavg(),
                   "image_parity": {"images": len(common), "min_cosine": min(cos), "median_cosine": statistics.median(cos),
                                    "worst": sorted(zip(cos, common))[:5]},
                   "query_parity_cosine": cosine(qv, ql), "errors": errors, "waits": waits}
        for k, rows in res.items():
            if rows:
                field = "median_ms" if k.startswith("q_") else "images_per_second"
                xs = [r[field] for r in rows]
                summary[k] = {"metric": field, "median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs)}
        print("SUMMARY " + json.dumps(summary))
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
