#!/usr/bin/env python3
"""Regenerates the text fixtures in Tests/GemmaEmbedTests/Fixtures from llama.cpp.

    scripts/gpulock.py -- python3 -I scripts/make_text_fixtures.py models/embeddinggemma-2-Q8_0.gguf

The kind, prefix and text of each row of tokens.jsonl are the source; everything else is rewritten:
  tokens.jsonl           token ids from llama-tokenize (BOS and EOS added, as llama-server's embeddings path does)
  embeddings.jsonl       L2-normalised vectors from llama-server's /v1/embeddings, one per row of tokens.jsonl
  long_embeddings.jsonl  the long posts and then the posts, newlines as spaces, joined and cut at a word end to
                         exactly 1054, 1629 and 3222 tokens, so the symmetric sliding window is active
  activations.jsonl      llama-debug's sum and corner values for every graph tensor of texts 20, 21 and 22
llama-tokenize, llama-server and llama-debug must be on PATH. The checked-in fixtures come from build 11716.
"""
import json
import os
import re
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Tests", "GemmaEmbedTests", "Fixtures")
LONG_TOKENS = [1054, 1629, 3222]
ACTIVATION_TEXTS = [20, 21, 22]
CONTEXT = 8192


def llama_version():
    out = subprocess.run(["llama-server", "--version"], capture_output=True, text=True).stderr
    m = re.search(r"version: (\S+) \(build (\d+), commit (\w+)\)", out)
    if not m:
        sys.exit("cannot read llama-server --version:\n" + out)
    return m.groups()


def tokenize(model, text):
    r = subprocess.run(["llama-tokenize", "-m", model, "--stdin", "--ids", "--log-disable"],
                       input=text.encode(), capture_output=True, check=True)
    return json.loads(r.stdout.decode().strip().splitlines()[-1])


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, model):
        port = free_port()
        self.url = f"http://127.0.0.1:{port}"
        self.alias = os.path.basename(model)
        self.proc = subprocess.Popen(["llama-server", "-m", model, "--alias", self.alias, "--embedding", "--host", "127.0.0.1",
                                      "--port", str(port), "-c", str(CONTEXT), "-b", str(CONTEXT), "-ub", str(CONTEXT), "-np", "1"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(240):
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if r.status == 200 and self.proc.poll() is None:
                        with urllib.request.urlopen(self.url + "/v1/models") as m:
                            assert json.load(m)["data"][0]["id"] == self.alias
                        return
            except OSError:
                pass
            if self.proc.poll() is not None:
                sys.exit("llama-server exited during startup")
            time.sleep(0.5)
        sys.exit("llama-server did not become healthy")

    def post(self, path, body):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)

    def token_count(self, text):
        return len(self.post("/tokenize", {"content": text, "add_special": True})["tokens"])

    def embed(self, text):
        d = self.post("/v1/embeddings", {"input": text})
        return d["usage"]["prompt_tokens"], d["data"][0]["embedding"]

    def close(self):
        self.proc.terminate()
        self.proc.wait()


def write_jsonl(name, rows, ascii_only):
    with open(os.path.join(FIXTURES, name), "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=ascii_only) + "\n")


def cut_to_tokens(server, prefix, text, target):
    """The shortest prefix of text whose input has exactly target tokens, ending at a word end where one does."""
    ends = [k for k in range(1, len(text) + 1) if k == len(text) or text[k] == " " and text[k - 1] != " "]
    lo, hi = 0, len(ends) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if server.token_count(prefix + text[:ends[mid]]) < target:
            lo = mid + 1
        else:
            hi = mid
    for end in ends[max(lo - 3, 0):lo + 4]:
        if server.token_count(prefix + text[:end]) == target:
            return text[:end]
    for end in range(ends[max(lo - 1, 0)], ends[lo] + 1):
        if server.token_count(prefix + text[:end]) == target:
            return text[:end]
    sys.exit(f"no cut gives exactly {target} tokens")


def parse_debug_log(log):
    header = re.compile(r"^common_debug_cb_eval:\s+(.+?) = \((\w+)\)\s+(\w+)\((.*)\) = \{([\d, ]+)\}\s*$")
    lines = log.splitlines()
    tensors, k = [], 0
    while k < len(lines):
        m = header.match(lines[k])
        k += 1
        if not m:
            continue
        body = []
        while not lines[k].strip().startswith("sum ="):
            body.append(lines[k].strip())
            k += 1
        total = float(lines[k].split("=")[1])
        k += 1
        corners = " ".join(body).replace("...,", "null,")
        corners = re.sub(r",\s*\]", "]", corners).replace("nan", "NaN").replace("-inf", "-Infinity").replace("inf", "Infinity")
        tensors.append({"name": m.group(1), "dtype": m.group(2), "op": m.group(3), "args": m.group(4),
                        "ne": [int(x) for x in m.group(5).split(",")], "sum": total, "corners": json.loads(corners)})
    pooled = [n for n, t in enumerate(tensors) if t["name"] == "result_embd_pooled"]
    assert len(pooled) == 2, len(pooled)  # the warmup eval, then the real one
    return tensors[pooled[0] + 1:]


def activations(model, row, tool):
    base = ["llama-debug", "-m", model, "--embedding", "-p", row["input"], "-c", "512"]
    log = subprocess.run(base + ["--tensor-filter", "."], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True).stdout
    tensors = parse_debug_log(log)
    with tempfile.TemporaryDirectory() as out:
        # --save-logits suppresses the tensor printing, so the saved ids and vector need a second run.
        subprocess.run(base + ["--save-logits", "--logits-output-dir", out], capture_output=True, check=True)
        stem = "llamacpp-" + os.path.splitext(os.path.basename(model))[0] + "-embeddings"
        prompt = open(os.path.join(out, stem + "-prompt.txt")).read()
        ids = [int(x) for x in re.search(r"token ids: (.*)", prompt).group(1).split(",")]
        vector = list(struct.unpack("<768f", open(os.path.join(out, stem + ".bin"), "rb").read()))
    assert ids == row["token_ids"], (row["i"], ids, row["token_ids"])
    meta = {"record": "meta", "i": row["i"], "input": row["input"], "token_ids": ids, "n_tensors": len(tensors),
            "tool": tool, "final_embedding_l2": vector}
    return [meta] + [{"record": "tensor", "i": row["i"], "seq": n, **t} for n, t in enumerate(tensors)]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    model = sys.argv[1]
    version, build, commit = llama_version()
    print(f"llama.cpp {version} build {build} commit {commit}")

    sources = [json.loads(line) for line in open(os.path.join(FIXTURES, "tokens.jsonl"))]
    rows = []
    for i, s in enumerate(sources):
        text_input = s["prefix"] + s["text"]
        ids = tokenize(model, text_input)
        rows.append({"i": i, "kind": s["kind"], "prefix": s["prefix"], "text": s["text"], "input": text_input, "chars": len(text_input),
                     "bos_added": ids[0] == 2, "eos_added": ids[-1] == 1, "n_tokens": len(ids), "token_ids": ids})
    write_jsonl("tokens.jsonl", rows, ascii_only=False)
    print("tokens.jsonl", [r["n_tokens"] for r in rows])

    server = Server(model)
    try:
        embeddings = []
        for r in rows:
            assert server.post("/tokenize", {"content": r["input"], "add_special": True})["tokens"] == r["token_ids"], r["i"]
            n, e = server.embed(r["input"])
            assert n == r["n_tokens"], (r["i"], n, r["n_tokens"])
            embeddings.append({"i": r["i"], "server_model": server.alias, "pooling": "model default (GGUF pooling_type)",
                               "normalized": "L2 (OpenAI endpoint)", "prompt_tokens": n, "dim": len(e), "embedding": e})
        write_jsonl("embeddings.jsonl", embeddings, ascii_only=True)
        print("embeddings.jsonl", len(embeddings), "rows")

        prefix = "title: none | text: "
        joined = " ".join(r["text"].replace("\n", " ") for kind in ["long-post", "post"] for r in rows if r["kind"] == kind)
        source = f"llama-server {version} build {build}, /v1/embeddings, -c {CONTEXT} -ub {CONTEXT}"
        longs = []
        for target in LONG_TOKENS:
            text = cut_to_tokens(server, prefix, joined, target)
            n, e = server.embed(prefix + text)
            assert n == target, (n, target)
            longs.append({"source": source, "prefix": prefix, "text": text, "prompt_tokens": n, "embedding": e})
        write_jsonl("long_embeddings.jsonl", longs, ascii_only=True)
        print("long_embeddings.jsonl", [(len(r["text"]), r["prompt_tokens"]) for r in longs])
    finally:
        server.close()

    tool = f"llama-debug b{build}-{commit} --embedding (Metal, Q8_0)"
    records = [rec for i in ACTIVATION_TEXTS for rec in activations(model, rows[i], tool)]
    with open(os.path.join(FIXTURES, "activations.jsonl"), "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=rec["record"] != "meta") + "\n")
    print("activations.jsonl", len(records), "records")


if __name__ == "__main__":
    main()
