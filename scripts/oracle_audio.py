# Usage: python3 -I scripts/oracle_audio.py <fixtures/audio dir> <llama-server base url>
#        python3 -I scripts/oracle_audio.py --speech <speech dir> <llama-server base url>
# The server, under the GPU lock, from the package root:
#   python3 scripts/gpulock.py -- llama-server -m models/embeddinggemma-2-Q8_0.gguf \
#     --mmproj models/mmproj-embeddinggemma-2-Q8_0.gguf --embedding -c 8192 -b 8192 -ub 8192 --port 8094
# The embeddings path needs the whole prompt in one physical batch, and the 35 s clip is 879 tokens,
# so the default -ub 512 would refuse it. The server warns that audio is experimental.
# The first form writes embeddings.jsonl in the fixtures folder: one row per clip in manifest.json with
# prompt_tokens and the 768-d L2 vector from /v1/embeddings. The second takes NN.wav and NN.txt
# (transcript) pairs and writes embeddings.jsonl (one row per clip) and queries.jsonl (each transcript
# and five off-topic texts, embedded as documents and as queries).
# Expected prompt_tokens: llama.cpp cuts 16 kHz audio into 30 s chunks of 480000 samples; a chunk of n
# samples has (n + 160 - 321) // 160 + 1 frames (hop 160, window 320, left pad 160), and two stride-2
# stages turn f frames into (f - 1) // 2 + 1 tokens each. The prompt is bos, <|audio>, every chunk's
# tokens, <audio|>, eos, so prompt_tokens = 4 + the chunks' tokens. miniaudio resamples other rates
# first, to about round(n * 16000 / rate) samples, so a mismatch there is information, not an error.
import base64, json, os, sys, urllib.request, wave
SERVER_ARGS = "-c 8192 -b 8192 -ub 8192"
SPEECH = sys.argv[1] == "--speech"
R, EMB = sys.argv[1 + SPEECH], sys.argv[2 + SPEECH]
def post(body):
    req = urllib.request.Request(EMB + "/v1/embeddings", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r: return json.load(r)
def embed_audio(path):
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return post({"model": "embeddinggemma-2", "input": [{"content": [{"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]}]})
def expected_tokens(n):
    def chunk(m):
        f = (m + 160 - 321) // 160 + 1
        for _ in range(2): f = (f - 1) // 2 + 1
        return f
    return 4 + sum(chunk(min(480000, n - s)) for s in range(0, n, 480000))
def vector(res):
    return {"prompt_tokens": res["usage"]["prompt_tokens"], "dim": len(res["data"][0]["embedding"]), "embedding": res["data"][0]["embedding"]}
if not SPEECH:
    manifest = json.load(open(os.path.join(R, "manifest.json")))
    with open(os.path.join(R, "embeddings.jsonl"), "w") as f:
        for m in manifest:
            row = {k: m[k] for k in ["file", "kind", "note", "seconds", "sample_rate", "channels", "format"]}
            row.update(vector(embed_audio(os.path.join(R, m["file"]))), server_args=SERVER_ARGS)
            f.write(json.dumps(row) + "\n")
            n = round(m["seconds"] * 16000)
            want = expected_tokens(n)
            check = ("matches" if row["prompt_tokens"] == want else f"MISMATCH, formula {want}") if m["sample_rate"] == 16000 \
                else f"resampled, formula on {n} samples gives {want}"
            print(f'{m["file"]:30} {row["prompt_tokens"]:5}  {check}')
    sys.exit(0)
clips = sorted(f for f in os.listdir(R) if f.endswith(".wav"))
off_topic = [
    "the central bank left interest rates unchanged for a third consecutive meeting",
    "a recipe for sourdough bread with a long cold fermentation",
    "quarterly earnings call transcript for a semiconductor company",
    "how to configure a Kubernetes ingress controller with TLS",
    "a black and white photograph of a crowded railway station",
]
texts = []
with open(os.path.join(R, "embeddings.jsonl"), "w") as f:
    for clip in clips:
        transcript = open(os.path.join(R, clip[:-4] + ".txt")).read().strip()
        texts.append((clip, transcript))
        n = wave.open(os.path.join(R, clip)).getnframes()
        row = {"file": clip, "transcript": transcript}
        row.update(vector(embed_audio(os.path.join(R, clip))), server_args=SERVER_ARGS)
        f.write(json.dumps(row) + "\n")
        print(clip, row["prompt_tokens"], "formula", expected_tokens(n))
texts += [(None, t) for t in off_topic]
with open(os.path.join(R, "queries.jsonl"), "w") as f:
    for i, (clip, text) in enumerate(texts):
        row = {"i": i, "describes": clip, "text": text}
        for task, prefix in [("document", "title: none | text: "), ("query", "task: search result | query: ")]:
            res = post({"model": "embeddinggemma-2", "input": [prefix + text]})
            row[task] = {"input": prefix + text, "prompt_tokens": res["usage"]["prompt_tokens"], "embedding": res["data"][0]["embedding"]}
        row["server_args"] = SERVER_ARGS
        f.write(json.dumps(row) + "\n")
        print("query", i, row["document"]["prompt_tokens"], row["query"]["prompt_tokens"])
