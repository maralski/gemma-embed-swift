#!/usr/bin/env python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Times llama-server embeddings over the same posts gemma-embed --bench uses.

usage: bench_llamacpp.py <posts.json> [--port 8092] [--batch 32] [--runs 3]
Texts are gist + text[:2000] with the document prefix, sent 32 per request
to /v1/embeddings.
"""
import json, sys, time, urllib.request

def main():
    args = sys.argv[1:]
    path = args[0]
    port = int(args[args.index("--port") + 1]) if "--port" in args else 8092
    batch = int(args[args.index("--batch") + 1]) if "--batch" in args else 32
    runs = int(args[args.index("--runs") + 1]) if "--runs" in args else 3
    posts = json.load(open(path))["posts"]
    texts = ["title: none | text: " + ((p.get("gist") or "") + "\n\n" + (p.get("text") or "")[:2000]).strip() for p in posts]
    url = f"http://127.0.0.1:{port}/v1/embeddings"

    def post(inputs):
        req = urllib.request.Request(url, data=json.dumps({"input": inputs}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)

    post(texts[:8])
    tokens = 0
    seconds = []
    for _ in range(runs):
        t0 = time.time()
        tokens = 0
        for i in range(0, len(texts), batch):
            tokens += post(texts[i:i + batch])["usage"]["prompt_tokens"]
        seconds.append(time.time() - t0)
    median = sorted(seconds)[len(seconds) // 2]
    print(json.dumps({"texts": len(texts), "tokens": tokens, "runs": runs, "seconds_median": median, "seconds_best": min(seconds),
                      "texts_per_second_median": len(texts) / median, "tokens_per_second_median": tokens / median}))

if __name__ == "__main__":
    main()
