# GemmaEmbed and llama.cpp

This page compares GemmaEmbed with llama.cpp for one job, EmbeddingGemma 2 text and image
embeddings on a Mac with Apple Silicon. llama.cpp is the reference that GemmaEmbed is checked
and benchmarked against. Both read the same GGUF files, `embeddinggemma-2-Q8_0.gguf` and
`mmproj-embeddinggemma-2-Q8_0.gguf`.

Every number here comes from [README.md](README.md), [CHANGELOG.md](CHANGELOG.md) or a run made
on 2026-10-25 for this page, and each table names its source. All of them come from one
machine, an Apple M3 Max (40 GPU cores, 128 GB) running macOS 26, and from llama.cpp 0.7.0-dev
build 11716 (commit `797de63b2`, the Homebrew `--HEAD` build).

## Summary

| | GemmaEmbed 0.3.0 | llama.cpp build 11716 |
|---|---|---|
| Text vectors | min cosine 0.9999988 against llama.cpp over 40 fixtures | the reference |
| Image vectors | min cosine 0.9999985 against llama-server over 30 fixtures | the reference |
| Text throughput, 285 feed posts (private corpus) | 176.4 posts per second | 118.3 posts per second with 8 slots |
| One 22-token query | 8.3 ms in process | 15.0 ms over local HTTP |
| Image throughput, 153 feed pictures (private corpus) | 8.46 images per second | 6.04 images per second |
| Audio vectors (unreleased, next version) | min cosine 0.99998933 against llama-server over 29 fixture clips | the reference |
| Audio throughput, 40 spoken clips, 715 s (unreleased) | 274.9 audio seconds per second | 153.6 audio seconds per second |
| Models | EmbeddingGemma 2 only | many model architectures, for generation and embeddings |
| Platforms | macOS 14 or later on Apple Silicon | macOS, Linux and Windows, on CPU, Metal, CUDA, Vulkan and other backends |
| Weight files | the Q8_0 files only | Q8_0, BF16, F16, F32 and the quantized GGUF formats |
| Interface | Swift API inside your process, and a CLI | HTTP server, command-line tools and a C library |
| Dependencies | the Apple SDK | a C++ toolchain and CMake to build it |

The query and image rows were measured on a loaded machine. The Speed section gives their
ranges.

## Vectors match llama.cpp

GemmaEmbed's tests compare it against vectors from llama.cpp build 11716 over the same Q8_0
files. The Q8_0 file is itself an approximation of the fp32 model. Agreement with llama.cpp means
the two engines compute the same thing, not that either is closer to the fp32 model.

| Check | Result | Source |
|---|---|---|
| Tokenizer, 40 fixture texts in five scripts, with URLs, numbers and emoji | token ids identical | README, Parity |
| Text embeddings, 40 fixtures, GemmaEmbed 0.3.0 against `llama-server --embedding` | min cosine 0.9999988, max abs diff 1.4e-4 | README, Parity |
| Text embeddings at 1054, 1629 and 3222 tokens, sliding window active | cosine 0.9999995 or better | README, Parity |
| One 22-token query, GemmaEmbed 0.3.0 against llama-server | cosine 0.9999994 | measured 2026-10-25 |
| Image embeddings, 30 fixture images at the default cap of 256 vision tokens, against `llama-server --embedding --mmproj` | min cosine 0.9999985, max abs diff 2.8e-4 | README, Image parity |
| Image embeddings, 4 fixture images at llama.cpp's native cap of 1120 | min cosine 0.9999992, max abs diff 1.4e-4 | README, Image parity |
| Image embeddings, 153 feed pictures (private corpus) at cap 256, GemmaEmbed 0.3.0 against llama-server | min cosine 0.99991, median 0.9999996 | measured 2026-10-25 |
| Image preprocessing on 6 images, and JPEG decoding on 36 JPEGs | byte-identical to llama.cpp and stb_image | README, Image parity |
| Caption ranking, 5 scenes against 10 captions | every scene ranks its own caption first, the same ranks as llama-server's own vectors | README, Cross-modal check |

A corpus embedded with one engine can be searched with queries embedded by the other. The task
prefix and the image cap must match. llama-server caps images at 256 vision tokens with its
default batch size and logs the cap when it loads. `-ub` raises it, up to 1120. GemmaEmbed's
`maxImageTokens` defaults to 256 to match. The same picture embeds at cosine 0.973 between caps
256 and 1120 (README, Image embeddings).

## Speed

### Text batches

Source: README, Performance. Measured on 2026-10-25 under a GPU lock with no other GPU work and
a load average of 3.3, five runs. The corpus is 285 social media posts of
2026-10-17, a private set that is not in the repository. It has 41,675 tokens in all, 146 tokens
per post on average, and the longest post has 604. llama.cpp gets 32 texts per request to
`/v1/embeddings`.

| Engine | Posts per second | Seconds for 285 posts | Tokens per second |
|---|---|---|---|
| GemmaEmbed 0.3.0 | 176.4 | 1.62 (1.61 to 1.66) | 25,790 |
| llama.cpp, `llama-server --embedding -c 8192 -b 8192 -ub 8192 -np 8` | 118.3 | 2.41 | 17,300 |
| llama.cpp, same flags with `-np 1` (one sequence at a time) | 59.6 | 4.78 | 8,720 |

GemmaEmbed 0.3.0 is 1.49 times faster than llama.cpp's best configuration here and 2.96 times
faster than one slot.

With `-np 8 -c 8192`, llama-server gives each slot 1024 tokens of context. That fits this corpus.
A longer text needs a larger `-c`.

### One short query

Search embeds one short query at a time, so latency matters more than throughput there. The
README gives 7.4 ms wall for one 22-token query on GemmaEmbed 0.3.0, with no llama.cpp figure.
The table below was measured for this page.

The query is "which central bank left interest rates unchanged at its October meeting this
year" with the query prefix, 22 tokens on both engines. GemmaEmbed ran
`gemma-embed --task query --bench q.txt --runs 20` and reports the median wall time of one
`embed` call: tokenizing, the GPU forward and reading the vector back. llama-server received the
same text, one request at a time, 20 times after a warm-up request. Its time is the full HTTP
round trip on 127.0.0.1, which is what a Swift app that calls llama-server waits for. The share
spent in HTTP and JSON was not measured separately.

| Engine | Milliseconds, median of 12 rounds | Range of round medians |
|---|---|---|
| GemmaEmbed 0.3.0, in process | 8.3 | 7.1 to 14.8 |
| llama-server, `-c 8192 -b 8192 -ub 8192 -np 8` | 15.0 | 12.0 to 22.3 |
| llama-server, same flags with `-np 1` | 17.9 | 12.4 to 21.2 |

GemmaEmbed was faster in all 12 rounds. Within a round, llama-server with 8 slots took 1.69
times as long as GemmaEmbed (median of the 12 per-round ratios, interquartile range 1.47 to
1.90).

The rounds alternated the engines' order under the GPU lock and paused while a test run was
going. A test run still started during 7 of the 84 timed steps. Other work kept the
load average between 4.3 and above 40 during the runs, against 3.3 for the README's numbers. The
load is the likely reason GemmaEmbed's median here is 8.3 ms and not the README's 7.4 ms, though
that was not tested.

### Images

The pictures are the README's 153 feed JPEGs, 480 to 1280 pixels wide, a private set that is
not in the repository. Both engines used the cap of 256 and processed the same 28,740 rows,
counted by `gemma-embed --bench-images` and by llama-server's `usage.prompt_tokens`.
GemmaEmbed ran `--bench-images` once per round. Its time includes decoding and resizing on the CPU. llama-server ran with its default flags plus
`--embedding --mmproj`, which gave 4 slots. It received each picture as a base64 data URL, 1 or 8
per request, and its time covers the whole folder.

| Configuration | Images per second, median of 12 rounds | Range |
|---|---|---|
| GemmaEmbed 0.3.0, one image per GPU batch | 8.46 | 4.48 to 11.04 |
| GemmaEmbed 0.3.0, batched by patch count (`--batch-size auto`) | 7.84 | 3.00 to 11.64 |
| llama-server, 1 image per request | 5.91 | 2.87 to 7.91 |
| llama-server, 8 images per request | 6.04 | 3.03 to 8.29 |

GemmaEmbed with one image per batch beat llama-server with 8 images per request in 11 of 12
rounds. The median per-round ratio is 1.32, with an interquartile range of 1.19 to 1.97. Treat
that size as approximate. Both engines decode images on the CPU inside the timed region, and the
CPU load from other work moved every figure by a factor of two or more. In the first five
rounds, llama-server with `-np 8` was no faster than with its default 4 slots (6.27 against 6.72
images per second).

### Audio

Audio embeddings are on `main` but not yet in a release. Source: README, Audio performance.
Measured on 2026-10-26 at a load average of about 4 to 6, five alternating rounds, over 40 clips
spoken by macOS `say` (715 s in all, 3.3 to 85 s each; `scripts/make_speech_bench.py` regenerates
them, they are not in the repository). Both engines decode WAV on the CPU inside the timed region.
llama-server receives each clip as base64 in its own request.

| Configuration | Audio seconds per second, median of 5 | Range |
|---|---|---|
| GemmaEmbed, default batching by mel frames | 274.9 | 271.0 to 324.7 |
| GemmaEmbed, one clip per GPU batch | 235.9 | 225.9 to 271.9 |
| llama-server build 11716, `--embedding --mmproj`, one clip per request | 153.6 | 148.6 to 160.2 |

GemmaEmbed with one clip per batch is 1.54 times faster than llama-server, and 1.79 times with its
default batching; HTTP and base64 are on llama-server's side of that gap. The vectors agree to min
cosine 0.9999994 over the same clips. llama-server logs that its audio input is experimental.

## What llama.cpp does that GemmaEmbed does not

llama.cpp is a general inference engine. GemmaEmbed runs one model family and nothing else.

- llama.cpp runs many model architectures, for text generation, chat, embeddings, reranking and
  vision.
- It runs on Linux and Windows as well as macOS, with CPU, CUDA, Vulkan and other backends
  besides Metal.
- It reads BF16, F16 and F32 weight files and the quantized GGUF formats. GemmaEmbed reads the
  Q8_0 files only. BF16 and F16 files are on its roadmap.
- `llama-server` serves an OpenAI-compatible HTTP API (`/v1/embeddings`) to clients in any
  language, from one process, with parallel slots.
- New models and GGUF format changes land in llama.cpp first. GemmaEmbed follows them.

## What GemmaEmbed offers

- A Swift API that runs inside your process. `embed` is an `async` call that returns `[[Float]]`.
  There is no server to start, no port, no JSON and no base64 for images.
- No dependencies beyond the Apple SDK, so adding it is one SwiftPM package.
- Images as a file URL, encoded bytes or a `CGImage`, including HEIC and anything else ImageIO
  reads. JPEG and PNG decode exactly as llama.cpp decodes them.
- Text and images mixed in one input (`MultimodalInput`), in the sequence that llama.cpp's
  `mtmd` builds.
- Matryoshka truncation to 512, 256 or 128 dimensions, and L2 normalisation, as `EmbedOptions`
  fields.
- The model card's task prefixes as an option (`.query`, `.document`, `.clustering` and the
  rest). With llama-server, the client prepends the prefix text itself.
- Progress after every GPU batch, and cancellation between batches through the surrounding
  `Task`.
- The same vector for a text whether it travels alone or in a batch, bit for bit (README,
  Parity).

## What it costs to integrate

Measured on 2026-10-25 from the release build of this repository and from the Homebrew
`llama.cpp` HEAD-797de63 and `ggml` 0.27.0 installs.

| | GemmaEmbed 0.3.0 | llama.cpp build 11716 |
|---|---|---|
| Add to a project | one SwiftPM dependency | build with CMake or install a package, then run `llama-server` as a separate process |
| Run-time libraries | Apple system frameworks only: Metal, Accelerate, ImageIO, CoreGraphics, CryptoKit and Foundation | `libllama`, `libllama-common`, `libmtmd`, `libllama-server-impl`, `libggml`, `libggml-base`, the ggml Metal and CPU backends, and OpenSSL |
| On disk | the `gemma-embed` executable with the library linked in, 1,507,400 bytes, plus an optional 52 KB shader bundle (the shader source is also compiled into the library) | `llama-server`, 33,472 bytes, plus 13,090,432 bytes of llama.cpp dylibs and 6,722,640 bytes of ggml libraries with the Metal backend and the M2 and M3 CPU backend |
| Model load | 0.27 s with a warm shader cache, about 1 s after a build (README, Performance) | not measured here |
| Process model | in your process, so a crash in GemmaEmbed is a crash in your app | a separate process that you start, health-check and restart |

## When to use which

Use llama.cpp when:

- You need any model other than EmbeddingGemma 2, now or later.
- You deploy on Linux, Windows or a non-Apple GPU.
- Your client is not Swift, or several processes or machines share one embedding service.
- You want a quantized file smaller than Q8_0, or BF16 weights.
- You already run llama-server for generation and want one engine to operate.
- You want the engine that new model releases target first.

Use GemmaEmbed when:

- Your app is written in Swift, runs on a Mac with Apple Silicon, and embeds EmbeddingGemma 2
  text or images.
- You want no server process and no libraries outside the Apple SDK.
- Query latency or batch throughput matters, per the measurements above.
- You want mixed text and image inputs, Matryoshka truncation, progress or cancellation from a
  Swift call.

The two engines' vectors agree to within the cosines above, so you can mix them. For example, a
Linux server can build the index with llama.cpp and a Mac app can embed queries with GemmaEmbed,
if both use the same task prefixes and the same image cap.

## Reproduce the measurements

Download the two GGUF files listed in the README, build GemmaEmbed in release, and install
llama.cpp build 11716 or later.

```
swift build -c release
export GEMMA_EMBED_WEIGHTS=/path/to/embeddinggemma-2-Q8_0.gguf
export GEMMA_EMBED_IMAGE_WEIGHTS=/path/to/mmproj-embeddinggemma-2-Q8_0.gguf
```

### Parity

`gemma-embed --parity` checks the tokenizer and the text embeddings against the llama.cpp
fixtures in `Tests/GemmaEmbedTests/Fixtures`. `--parity-images` checks the image fixtures. Both
exit non-zero below cosine 0.9995.

```
.build/release/gemma-embed --parity Tests/GemmaEmbedTests/Fixtures
.build/release/gemma-embed --parity-images Tests/GemmaEmbedTests/Fixtures/images
```

Three scripts regenerate the fixtures. Every fixture is original or drawn from code, so all of
them can be redistributed.

- `scripts/make_text_fixtures.py` keeps the texts in `tokens.jsonl` and rewrites the token ids,
  the vectors, the long texts and the activation sums from `llama-tokenize`, `llama-server` and
  `llama-debug`, which must be on `PATH`.
- `scripts/make_fixture_images.swift` draws the scenes, the synthetic images, the variants and
  the JPEG encodings into `models/fixtures/images`, and copies the images under 250 KB into the
  package. The JPEG encodings need `cjpeg` and `magick`.
- `scripts/oracle_images.py` writes the image reference vectors and the caption vectors from a
  running llama-server. With `1120` it writes the four cap-1120 vectors, and the server must then
  run with `-ub 8192 -b 8192`.

```
python3 scripts/gpulock.py -- python3 -I scripts/make_text_fixtures.py models/embeddinggemma-2-Q8_0.gguf
swift scripts/make_fixture_images.swift models/fixtures/images Tests/GemmaEmbedTests/Fixtures/images
llama-server -m models/embeddinggemma-2-Q8_0.gguf --mmproj models/mmproj-embeddinggemma-2-Q8_0.gguf --embedding --port 8093 &
python3 -I scripts/oracle_images.py models/fixtures/images http://127.0.0.1:8093
```

`oracle_images.py` needs the full image set, so it runs on `models/fixtures/images`. Copy the
`embeddings.jsonl`, `embeddings-1120.jsonl` and `captions.jsonl` it writes there into
`Tests/GemmaEmbedTests/Fixtures/images`.

### Text throughput

`gemma-embed --bench` times the posts file in process. `scripts/bench_llamacpp.py` sends the
same texts to llama-server: gist plus text cut at 2000 characters, with the document prefix, 32
texts per request to `/v1/embeddings`. It warms up with one request and times `--runs` passes
over the whole file. It prints the median and best seconds, texts per second, and tokens per
second as counted by the server's `usage.prompt_tokens`.

```
.build/release/gemma-embed --bench /path/to/posts.json --runs 3
llama-server -m /path/to/embeddinggemma-2-Q8_0.gguf --embedding -c 8192 -b 8192 -ub 8192 -np 8 --port 8092 &
python3 scripts/bench_llamacpp.py /path/to/posts.json --port 8092 --batch 32 --runs 3
```

### Query latency and images

`scripts/compare_llamacpp.py` made the query latency and image throughput tables above. It needs
the release build and `llama-server` at `/opt/homebrew/bin/llama-server`, and reads the two GGUF
files from `$GEMMA_EMBED_MODELS`, or from `models/` in the checkout when that is unset.

```
python3 scripts/compare_llamacpp.py /path/to/pictures 12
```

The arguments are the picture folder and the number of rounds (default 5). The script starts
three llama-servers: text with 8 slots, text with 1 slot, and images with default flags. It
holds the GPU lock for the whole run and swaps the engines' order every other round. Before each
step it waits while a test run or another llama-server is using the GPU, and it records any step
that a test run overlapped. Each round times the 22-token query 20 times on each engine
(GemmaEmbed through `--task query --bench`, llama-server over HTTP after a warm-up request), then
the folder with `gemma-embed --bench-images` at batch size 1 and auto, and with llama-server at 1
and 8 pictures per request. It ends with a `SUMMARY` line of JSON: the median, minimum and
maximum of each configuration, the image and query parity against llama-server, and the load
average. Logs go to `$TMPDIR/gemma-embed-compare`.

The 153 feed pictures behind the tables are a private corpus, not in the repository, so any
folder of JPEGs reproduces the method but not the figures. Each round prints GemmaEmbed's `rows`
and llama-server's summed `prompt_tokens` for the folder. If the two differ, the engines used
different image caps.

### Comparing two GemmaEmbed builds

`scripts/ab_bench.py` and `scripts/ab_images.py` do not run llama.cpp. They compare two
`gemma-embed` builds, and the README's version-to-version numbers come from them. Each runs the
two builds alternately (A B B A ...) under the GPU lock in `/tmp/gemma-embed-gpu.lock`, so drift
hits both alike.

- `ab_bench.py` times `--bench` and takes GPU seconds from the per-batch trace lines as its
  primary metric. It calls a difference only when the median, best-of and per-batch-minimum
  ratios agree in direction and all clear 3 percent.
- `ab_images.py` times `--bench-images` with batching by patch count and with one image per
  batch. It calls a difference from the median of per-pair ratios, which must clear 3 percent
  and have its interquartile range on one side of 1.
- Both then gate the candidate on quality: llama.cpp text parity, llama-server image parity for
  `ab_images.py`, and a direct comparison of the candidate's vectors against the baseline's.

`scripts/gpulock.py -- COMMAND` runs any other command under the same lock. On a shared machine,
interleave the engines and hold the lock, as these scripts do.
