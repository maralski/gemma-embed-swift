# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
# Usage: python3 -I scripts/oracle_images.py <fixtures/images dir> <llama-server base url> [1120]
# The server: llama-server -m embeddinggemma-2-Q8_0.gguf --mmproj mmproj-embeddinggemma-2-Q8_0.gguf --embedding --port 8093
# Writes embeddings.jsonl (one row per image in manifest.json: file, kind, prompt_tokens, 768-d L2 vector)
# and captions.jsonl (10 captions embedded as documents and as queries) from llama-server's /v1/embeddings.
# With 1120, writes embeddings-1120.jsonl for four images instead; the server must then run with
# -ub 8192 -b 8192, since llama-server caps the image tokens at 256 under its default batch size.
import json, sys, base64, urllib.request, os
R, EMB = sys.argv[1], sys.argv[2]
CAP_1120 = sys.argv[3:] == ["1120"]
def post(body):
    req = urllib.request.Request(EMB + "/v1/embeddings", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r: return json.load(r)
def embed_image(path):
    mime = "image/png" if path.endswith(".png") else "image/jpeg"
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return post({"model": "embeddinggemma-2", "input": [{"content": [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]}]})
if CAP_1120:
    with open(os.path.join(R, "embeddings-1120.jsonl"), "w") as f:
        for file in ["scenes/00-green-hills.jpg", "scenes/04-desert-dunes.jpg", "synthetic/07-shapes.jpg", "variants/00-green-hills-resized-0.5.jpg"]:
            res = embed_image(os.path.join(R, file))
            row = {"file": file, "max_image_tokens": 1120, "server_args": "-ub 8192 -b 8192", "prompt_tokens": res["usage"]["prompt_tokens"],
                   "dim": len(res["data"][0]["embedding"]), "embedding": res["data"][0]["embedding"]}
            f.write(json.dumps(row) + "\n")
            print(file, res["usage"]["prompt_tokens"])
    sys.exit(0)
manifest = json.load(open(os.path.join(R, "manifest.json")))
with open(os.path.join(R, "embeddings.jsonl"), "w") as f:
    for m in manifest:
        res = embed_image(os.path.join(R, m["file"]))
        row = {"file": m["file"], "kind": m["kind"], "note": m["note"], "source": m.get("source"), "variant": m.get("variant"),
               "prompt_tokens": res["usage"]["prompt_tokens"], "dim": len(res["data"][0]["embedding"]), "embedding": res["data"][0]["embedding"]}
        f.write(json.dumps(row) + "\n")
        print(m["file"], res["usage"]["prompt_tokens"])
captions = [
    ("scenes/00-green-hills.jpg", "rolling green hills striped with crop rows, a red barn and trees under a blue sky with white clouds"),
    ("scenes/01-ocean-sunset.jpg", "the sun setting over the ocean with an orange sky and a palm tree silhouette"),
    ("scenes/02-coral-reef.jpg", "an underwater coral reef with colourful corals and tropical fish"),
    ("scenes/05-cherry-blossom.jpg", "pink cherry blossoms on dark branches against a pale blue sky"),
    ("scenes/08-sunflower-field.jpg", "a field of bright yellow sunflowers under a blue summer sky"),
    (None, "the central bank left interest rates unchanged for a third consecutive meeting"),
    (None, "a recipe for sourdough bread with a long cold fermentation"),
    (None, "quarterly earnings call transcript for a semiconductor company"),
    (None, "how to configure a Kubernetes ingress controller with TLS"),
    (None, "a black and white photograph of a crowded railway station"),
]
with open(os.path.join(R, "captions.jsonl"), "w") as f:
    for i, (img, text) in enumerate(captions):
        row = {"i": i, "describes": img, "text": text}
        for task, prefix in [("document", "title: none | text: "), ("query", "task: search result | query: ")]:
            res = post({"model": "embeddinggemma-2", "input": [prefix + text]})
            row[task] = {"input": prefix + text, "prompt_tokens": res["usage"]["prompt_tokens"], "embedding": res["data"][0]["embedding"]}
        f.write(json.dumps(row) + "\n")
        print("caption", i, row["document"]["prompt_tokens"], row["query"]["prompt_tokens"])
