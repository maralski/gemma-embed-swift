# GemmaEmbed

GemmaEmbed runs Google's EmbeddingGemma 2 text and image encoders in-process on Apple Silicon
with its own Metal kernels. It is a Swift package with no dependencies beyond the Apple SDK: no
llama.cpp, no MLX, no Core ML. You hand it the official GGUF weight files and get 768-dimensional
embeddings that match llama.cpp's to a cosine similarity of 0.99999 for text and 0.99999 for
images, in one shared space.

## Supported model

| | |
|---|---|
| Model | `embeddinggemma-2` (GGUF architecture `gemma-embedding2`), 24 layers, 271M parameters |
| Weight file | `embeddinggemma-2-Q8_0.gguf` from https://huggingface.co/ggml-org/embeddinggemma-2-GGUF |
| Size | 309,855,456 bytes |
| SHA256 | `2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135` |
| Output | 768 dimensions, Matryoshka truncation to 512, 256 or 128 |
| Input | up to 8192 tokens per text (default limit 2048), any number of texts per call |
| Image encoder (optional) | `mmproj-embeddinggemma-2-Q8_0.gguf` from the same repository, 554,821,024 bytes, SHA256 `c4a8a52691ecef40618438928bdf9e68379b854e24166f292592353db0aab64f` (GGUF projector `gemma4v`, 16 layers, 473M parameters with the unused audio tower) |
| Image input | JPEG, PNG, HEIC and anything else ImageIO reads, or a `CGImage`; any size, scaled to 70 to 1120 vision tokens (default cap 256) |
| Platform | macOS 14 or later on Apple Silicon, Swift 6 |

The forward pass is documented in [Sources/GemmaEmbed/Model/Architecture.md](Sources/GemmaEmbed/Model/Architecture.md).

## Installation

Add the package to `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/maralski/gemma-embed-swift.git", from: "0.3.0"),
],
targets: [
    .target(name: "MyApp", dependencies: [.product(name: "GemmaEmbed", package: "gemma-embed-swift")]),
]
```

In Xcode, choose File > Add Package Dependencies and enter the same URL. The weights are not
part of the package. Download `embeddinggemma-2-Q8_0.gguf` (and, for images,
`mmproj-embeddinggemma-2-Q8_0.gguf`) from the Hugging Face repository above.

For a side-by-side comparison with llama.cpp (accuracy, speed, features, when to use which),
see [COMPARISON.md](COMPARISON.md).

## Usage

```swift
import GemmaEmbed

let model = try await EmbeddingModel.load(from: URL(fileURLWithPath: "embeddinggemma-2-Q8_0.gguf"))
let documents = try await model.embed(["Rates were left unchanged.", "A new open-weight model shipped."])
let query = try await model.embed("central bank decision", options: EmbedOptions(task: .query))
let best = documents.indices.max { model.similarity(query, documents[$0]) < model.similarity(query, documents[$1]) }
```

`EmbedOptions` selects the model card's task prefix (`.document`, `.query`, `.clustering`,
`.classification`, `.similarity`, `.none`), the token limit, the output width and whether the
vector is L2-normalised. `embed` batches texts by token count, reports progress after every GPU
batch and stops between batches when the surrounding `Task` is cancelled. Every text is embedded
identically whether it travels alone or inside a batch.

## Image embeddings

Load the model with the image encoder and embed files, bytes or `CGImage`s. The vectors live in
the same space as the text embeddings, so a caption query can be scored against pictures and a
picture against pictures.

```swift
let model = try await EmbeddingModel.load(from: textURL, options: LoadOptions(imageEncoder: mmprojURL))
let pictures = try await model.embed(images: [.url(photoURL), .data(jpegBytes), .cgImage(image)])
let caption = try await model.embed("rolling green vineyard hills", options: EmbedOptions(task: .query))
let score = model.similarity(caption, pictures[0])
print(model.info.imageEncoder?.layerCount)   // 16 when loaded, nil otherwise
```

`ImageEmbedOptions` carries the same `dimensions` and `normalise` as text, `batchSize` (images
per GPU batch, by default packed by patch count) and `maxImageTokens`. The image encoder is a
`LoadOptions` field rather than something added to a loaded model, so a model never changes
after `load`; loading text alone never opens the mmproj file, and `embed(images:)` without it
throws `imageEncoderNotLoaded`.

The encoder sees each image resized so that its patch grid gives at most `maxImageTokens`
vision tokens (16 by 16 pixel patches, pooled 3 by 3), and at least 70. The default cap of 256 is
what `llama-server` applies with its default batch size; llama.cpp's native limit is 1120. The
same photo embeds differently at different caps (cosine 0.973 between 256 and 1120 on the 1600
by 1600 green hills scene), so a corpus and the queries against it must share one cap. JPEGs
decode through a Swift port of stb_image's decoder, byte for byte what llama.cpp sees; PNGs
match it too. The preprocessing and the vision tower are documented in
[Sources/GemmaEmbed/Model/Architecture.md](Sources/GemmaEmbed/Model/Architecture.md).

### Text and images together

One input can mix text and images. The model reads each image where it sits among the text, in
one forward pass, so a caption and its picture embed as one document.

```swift
let post = MultimodalInput([.text("Our new office, first day:"), .image(.url(photoURL)), .text("More soon.")])
let posts = try await model.embed([post, MultimodalInput(text: "Quarterly results", images: [.data(chartPNG)])])
let query = try await model.embed("office opening", options: EmbedOptions(task: .query))
let score = model.similarity(query, posts[0])
```

The sequence is the one llama.cpp's `mtmd` builds for EmbeddingGemma 2:
`[bos] text [<|image>] image rows [<image|>] text ... [eos]`. `MultimodalEmbedOptions.task`
prefixes the first text, or stands alone before an image that opens the input. `maxTokens`
counts every row; text is cut from the end until the input fits and images are never cut, so an
input whose images alone exceed it throws `invalidOptions`. A text-only input embeds exactly as
`embed(_:)` does and an image-only input with `task: .none` exactly as `embed(images:)` does.
Inputs with images need the image encoder; text-only inputs do not.

### Image parity

Reference vectors come from llama.cpp build 11716 (`llama-server --embedding --mmproj`) over the
same Q8_0 files, and per-node dumps of its vision graph taken through `libmtmd`'s eval callback
(`scripts/vision_dump.cpp`). Every fixture image is drawn from code by
`scripts/make_fixture_images.swift`: 10 procedural scenes (green hills, an ocean sunset, a coral
reef, snowy peaks, desert dunes, cherry blossom, a lavender field, a night sky, a sunflower field,
a city skyline), 10 synthetic CoreGraphics images (solid colours, gradients, text in three fonts,
shapes, a checkerboard, noise) and 10 near-duplicate variants of the scenes.
`scripts/oracle_images.py` writes the reference vectors from a running llama-server.

| Check | Result |
|---|---|
| mmproj GGUF, 963 tensors and 30 metadata keys | identical to the directory dump |
| Preprocessing on 6 images (decode, size choice, Pillow-style bicubic resize, padding, normalisation, patch layout) | byte-identical to llama.cpp's pixel tensor |
| JPEG decoder on 36 JPEGs (every sampling factor, progressive, grey, YCCK, restart markers) | byte-identical to stb_image |
| CPU reference vision tower vs llama.cpp per-node sums, 3 images, 292 nodes each | within 1e-3, worst 8.5e-5 |
| CPU reference vision vectors vs llama.cpp (512-wide rows) | row cosine 0.99996 or better |
| Metal kernels vs CPU reference (patch GEMM, position tables, 2D RoPE, head-64 attention, gelu_quick epilogue, 3x3 pooling, mixed-row embedding) | max relative diff 5.0e-6 |
| Metal vision tower vs CPU reference, 3 images | row cosine 0.9999993 or better, max abs diff 1.2e-4 |
| Metal image embeddings vs llama-server, all 30 images, cap 256 | min cosine 0.9999985, max abs diff 2.8e-4 |
| Metal image embeddings vs llama-server, 4 images at cap 1120 | min cosine 0.9999992, max abs diff 1.4e-4 |
| Same image alone and inside a batch of 8 mixed sizes, first or last | bit-identical |
| Text embeddings vs llama.cpp, 40 fixtures (see Parity) | min cosine 0.9999988 |

Run `swift test -c release` to reproduce (see Tests for where the tests find the image encoder,
the llama.cpp dumps and the two images too large to ship in the package), or
`gemma-embed --parity-images Tests/GemmaEmbedTests/Fixtures/images`.

### Cross-modal check

Five scenes scored against ten caption queries (`EmbedOptions(task: .query)`, cosine). Five
captions describe the scenes and five are off-topic. Every scene ranks its own caption first, by
0.125 or more over the next caption, and llama-server's own vectors give the same ranks. The
scores below come from llama-server's vectors in `Fixtures/images/captions.jsonl` and
`embeddings.jsonl`; the test requires GemmaEmbed's to agree within 0.002.

| caption | 00-green-hills | 01-ocean-sunset | 02-coral-reef | 05-cherry-blossom | 08-sunflower-field |
|---|---:|---:|---:|---:|---:|
| rolling green hills striped with crop rows, a red barn and trees under a blue sky with white clouds | **0.795** | 0.526 | 0.560 | 0.536 | 0.658 |
| the sun setting over the ocean with an orange sky and a palm tree silhouette | 0.488 | **0.750** | 0.602 | 0.521 | 0.560 |
| an underwater coral reef with colourful corals and tropical fish | 0.481 | 0.570 | **0.752** | 0.538 | 0.528 |
| pink cherry blossoms on dark branches against a pale blue sky | 0.525 | 0.567 | 0.574 | **0.787** | 0.562 |
| a field of bright yellow sunflowers under a blue summer sky | 0.601 | 0.575 | 0.595 | 0.578 | **0.784** |
| the central bank left interest rates unchanged for a third consecutive meeting | 0.528 | 0.502 | 0.506 | 0.517 | 0.528 |
| a recipe for sourdough bread with a long cold fermentation | 0.487 | 0.468 | 0.440 | 0.463 | 0.480 |
| quarterly earnings call transcript for a semiconductor company | 0.478 | 0.427 | 0.443 | 0.467 | 0.474 |
| how to configure a Kubernetes ingress controller with TLS | 0.448 | 0.450 | 0.436 | 0.431 | 0.455 |
| a black and white photograph of a crowded railway station | 0.452 | 0.446 | 0.452 | 0.467 | 0.490 |

The smallest margin is the sunflower field, whose own caption (0.784) leads the green hills
caption (0.658). Off-topic captions score 0.43 to 0.53.

### Near duplicates

Cosine between each scene and its variant, for choosing a deduplication threshold. Every variant
has its original as nearest neighbour. The lowest variant pair is 0.896 (the coral reef
re-encoded at JPEG quality 40), and the highest cosine between two unrelated images is 0.889
(`synthetic/01-solid-grey` against `synthetic/02-gradient-h`). A threshold between the two
separates them on this set, with a gap of 0.007. Among the scenes and their variants alone, the
highest unrelated cosine is 0.822 (the green hills and the sunflower field).

| variant | pairs | min | median | max |
|---|---:|---:|---:|---:|
| resized to half | 1 | 0.996 | 0.996 | 0.996 |
| resized to a third | 1 | 0.987 | 0.987 | 0.987 |
| central crop keeping 90 percent | 1 | 0.998 | 0.998 | 0.998 |
| top-left crop keeping 90 percent | 1 | 0.994 | 0.994 | 0.994 |
| JPEG quality 40 | 2 | 0.896 | 0.943 | 0.989 |
| flipped horizontally | 2 | 0.991 | 0.993 | 0.996 |
| colour shift (red +12, blue -12) | 2 | 0.994 | 0.995 | 0.995 |

### Image performance

Measured on 2026-10-25 on the same Apple M3 Max as the text numbers, over 153 social media feed
pictures of 2026-10-23 (a private corpus, not in the repository; real JPEGs, 480 to 1280 pixels
wide, 188 vision tokens on average), one picture per GPU batch as the default batching runs
them, with `gemma-embed --bench-images --batch-size 1`, the two builds alternating under a GPU
lock, three rounds each:

| Configuration | Images per second (median of 3) | Seconds for 153 images | GPU seconds |
|---|---|---|---|
| GemmaEmbed 0.3.0 | 10.40 | 14.72 | 14.48 |
| GemmaEmbed 0.2.0 | 8.49 | 18.02 | 17.89 |

Image vectors stay within 2e-6 of 0.2.0's. Image parity with llama-server is in Image parity
above (min cosine 0.9999985 over the 30 fixtures). A feed picture takes about 95 ms of GPU time,
almost all of it the vision tower; the text tower over its rows takes about 3 ms.

The rest of this section was measured on 2026-10-19 with GemmaEmbed 0.2.0, over the 30 fixture
images of that release, an earlier set since replaced (average 216 vision tokens each, 220 rows
through the text tower), with `gemma-embed --bench-images --runs 5`.
Other processes kept the CPU load average between 26 and 80 during the runs, but the GPU was
otherwise idle: `GEMMA_EMBED_TRACE=1` showed a scheduled-start lag of 0.2 ms on almost every
command buffer and 9.7 ms at worst. Wall time includes JPEG and PNG decoding and resizing on the
CPU (12 ms per 1600 by 1600 JPEG, run in parallel). GPU time is the command buffer's own
duration over the same batches.

| Configuration | Images per second (median of 5) | Seconds for 30 images | GPU seconds |
|---|---|---|---|
| GemmaEmbed 0.2.0, one image per GPU batch | 4.71 | 6.37 | 6.34 |
| GemmaEmbed 0.2.0, 2 images per batch | 4.37 | 6.87 | 6.49 |
| GemmaEmbed 0.2.0, 4 images per batch | 3.99 | 7.51 | 7.17 |
| GemmaEmbed 0.2.0, 8 images per batch | 3.28 to 3.73 (two runs) | 8.05 to 9.14 | 6.76 to 7.71 |
| llama.cpp 0.7.0-dev build 11716, `llama-server --embedding --mmproj`, 1 image per request | 3.48 | 8.62 | |
| llama.cpp, same build, 8 images per request | 4.90 | 6.13 | |

The GPU time itself grows with the batch: packing more images into one forward makes the vision
GEMMs slower, not faster, and the trend is monotonic from 1 to 8. The likely cause is the
activation buffers outgrowing the on-chip cache (a batch of 8 full-size images is 18,432 patch
rows, and its 3072-wide FFN activations take 226 MB): profiled on 0.3.0, a batch of 8 pictures
runs 17 percent slower than the sum of its kernels timed alone, while one picture runs faster
than its kernel sum. Budgets of 8192 to 32768 patches measured 4 to 6 percent slower than 4096
and cost 2 to 4 times the memory. The
default therefore packs images up to 4096 patch rows per batch, which is one image at the
default cap and several smaller ones; `ImageEmbedOptions.batchSize` overrides it.

Image encoder load time: 0.24 s on the first load after the file left the page cache (mapping
and prefaulting the 555 MB file, building the tower), 0.02 s on later loads. Loading the text
model alone does not open the file.

Reproduce:

```
export GEMMA_EMBED_WEIGHTS=/path/to/embeddinggemma-2-Q8_0.gguf
export GEMMA_EMBED_IMAGE_WEIGHTS=/path/to/mmproj-embeddinggemma-2-Q8_0.gguf
GEMMA_EMBED_TRACE=1 .build/release/gemma-embed --bench-images /path/to/images --runs 5
.build/release/gemma-embed --profile-images /path/to/images --batch-size 1
python3 scripts/ab_images.py --a /path/to/old/gemma-embed --b .build/release/gemma-embed --dir /path/to/images
```

`--profile-images` times each vision kernel class over the first `--batch-size` images, and
`--bench-images` reports `gpu_seconds_minsum`, every batch's fastest GPU pass summed.
`scripts/ab_images.py` runs two builds alternately and checks image parity, text parity and that
the candidate's vectors match the baseline's.

## Command line

```
swift build -c release
export GEMMA_EMBED_WEIGHTS=/path/to/embeddinggemma-2-Q8_0.gguf
echo "hello world" | .build/release/gemma-embed --task query --dimensions 256
.build/release/gemma-embed --parity Tests/GemmaEmbedTests/Fixtures
export GEMMA_EMBED_IMAGE_WEIGHTS=/path/to/mmproj-embeddinggemma-2-Q8_0.gguf
.build/release/gemma-embed --images photo.jpg ~/Pictures
.build/release/gemma-embed --parity-images Tests/GemmaEmbedTests/Fixtures/images
.build/release/gemma-embed --bench-images ~/Pictures --runs 3
.build/release/gemma-embed --bench-images ~/Pictures --batch-size auto
.build/release/gemma-embed --profile-images ~/Pictures --batch-size 1
.build/release/gemma-embed --bench posts.json --runs 3
.build/release/gemma-embed --profile posts.json
```

`gemma-embed` prints one JSON object per input line. `--parity` checks the tokenizer and the
embeddings against the llama.cpp fixtures and exits non-zero below cosine 0.9995. `--bench` times
a throughput run over a feed JSON file (`{"posts": [{"gist", "text"}]}`) or a plain text file with
one document per line. Image modes take the mmproj file from `--image-encoder` or
`$GEMMA_EMBED_IMAGE_WEIGHTS`: `--images FILE... | DIR` prints one JSON vector per image,
`--parity-images DIR` checks a fixtures folder (`manifest.json` and `embeddings.jsonl`) and
`--bench-images DIR` measures images per second, by default with patch-count batching, 8 and 1
images per batch (`--batch-size auto` or `N` picks one). `--help` lists the rest.

## Packaging

The shaders compile from `.metal` source when a model loads. GemmaEmbed looks for the SwiftPM
resource bundle `GemmaEmbed_GemmaEmbed.bundle` in this order:

1. The main bundle's resources: `Contents/Resources` inside an app.
2. The resources of the bundle that holds the GemmaEmbed code, then the directory next to it.
3. The main bundle's own directory, which is where `swift run` and a bare executable find it.

If no copy is found, GemmaEmbed uses the shader source compiled into the library, so a model
still loads with no bundle at all. It never reads the checkout's `.build` directory, which
SwiftPM's `Bundle.module` falls back to.

An app built outside Xcode must copy the bundle into `Contents/Resources`. Codesign refuses
loose files at the root of an `.app`. A build script can do it with this line, after it
sets `SHADERS="$BIN_DIR/GemmaEmbed_GemmaEmbed.bundle"` from `swift build -c release --show-bin-path`:

```
cp -R "$SHADERS" "$APP/Contents/Resources/"
```

After you edit a file in `Sources/GemmaEmbed/Shaders`, run `scripts/embed_shaders.py` to
regenerate `Sources/GemmaEmbed/Metal/EmbeddedShaders.swift`. `ShaderSourceTests` fails while
the two disagree.

## Tests

`swift test -c release` runs the suite. The debug configuration also passes, in about six
minutes, because the CPU reference runs unoptimised. The tests read the weights and the
llama.cpp dumps from `models/` at the package root, which is gitignored. An environment variable
overrides each path, and a test whose input is missing skips.

| Path | Contents | Override |
|---|---|---|
| `models/embeddinggemma-2-Q8_0.gguf` | the text weights | `GEMMA_EMBED_WEIGHTS` |
| `models/mmproj-embeddinggemma-2-Q8_0.gguf` | the image encoder | `GEMMA_EMBED_IMAGE_WEIGHTS` |
| `models/dumps/` | llama.cpp's vision graph for six images (`scripts/vision_dump.cpp`) | `GEMMA_EMBED_VISION_DUMPS` |
| `models/dumps-rgb/` | stb_image's decode of every JPEG fixture (`scripts/vision_dump.cpp`) | `GEMMA_EMBED_DECODED_RGB_DUMPS` |
| `models/fixtures/images/` | the full image fixture set from `scripts/make_fixture_images.swift`, including the lavender field scene and its variant, which are over 250 KB and not in the package | none |

`scripts/vision_dump.cpp` lists the commands that write both dump directories.

## Parity

Reference vectors come from llama.cpp build 11716 (`llama-server --embedding`) over the same Q8_0
file, which is itself an approximation of the fp32 model, so agreement is bounded by that oracle.

| Check | Result |
|---|---|
| Tokenizer, 40 fixture texts (five scripts, URLs, numbers, emoji) | token ids identical |
| CPU fp32 reference vs llama.cpp per-tensor activation sums, 3 texts | every named tensor within 2 percent |
| CPU fp32 reference vs llama.cpp embeddings, 3 texts | min cosine 0.9999992, max abs diff 8.5e-5 |
| Metal kernels vs CPU reference (GEMM all shapes and tiles, RMSNorm, RoPE, attention at 1 to 2048 tokens, GeGLU, pooling) | max relative diff 4.5e-6 |
| Metal forward vs llama.cpp, all 40 fixtures | min cosine 0.9999988, max abs diff 1.4e-4 |
| Metal forward vs llama.cpp at 1054, 1629 and 3222 tokens (sliding window active) | cosine 0.9999995 or better |
| Metal forward vs CPU reference at 1672 tokens | cosine 0.9999999, max abs diff 1.8e-7 |
| Same text alone and inside a batch of 33 | bit-identical |

Run `swift test -c release` to reproduce (see Tests). `scripts/make_text_fixtures.py`
regenerates the text fixtures from llama.cpp.

## Performance

Measured on 2026-10-25 on an Apple M3 Max (40 GPU cores, 128 GB) running macOS 26, Xcode 26.3,
under a GPU lock with no other GPU work and a load average of 3.3, five alternating runs per
build. The corpus is 285 social media posts of 2026-10-17, a private set that is not in the
repository (gist plus text cut at 2000 characters, document prefix), 41,675 tokens in all, 146
tokens per post on average, longest 604.

| Engine | Posts per second | Seconds for 285 posts | Tokens per second |
|---|---|---|---|
| GemmaEmbed 0.3.0 | 176.4 | 1.62 (1.61 to 1.66) | 25,790 |
| GemmaEmbed 0.2.0 | 117.7 | 2.42 (2.42 to 2.46) | 17,220 |
| llama.cpp 0.7.0-dev build 11716, `llama-server --embedding -np 8 -ub 8192`, 32 texts per request | 118.3 | 2.41 | 17,300 |
| llama.cpp, same build, `-np 1` (one sequence at a time) | 59.6 | 4.78 | 8,720 |

GemmaEmbed 0.3.0 is 1.49 times faster than llama.cpp's best configuration and 2.96 times faster
than its single-slot default, with the same parity (min cosine 0.9999988 against llama.cpp over
the 40 fixtures) and vectors within 3e-7 of 0.2.0's. A single 22-token query takes 7.4 ms wall,
down from 15.2 ms in 0.2.0. The GEMMs take most of the GPU time and run at 7.6 to 8.8 TFLOPS fp32
in a sustained microbench; attention takes about an eighth.

Measuring on a shared machine: GPU time of short command buffers reads high because the GPU runs
at a low clock, so per-kernel lines of `--profile` mislead for kernels under about 2 ms, and a
microbench of short bursts can rank GEMM tiles the wrong way round. `scripts/ab_bench.py` runs two
builds alternately and calls a difference only when the median, best-of and per-batch-minimum
ratios agree; `scripts/gpulock.py` keeps two measurements off the GPU at once.

Load time: 0.27 s when the Metal shader cache is warm (tokenizer 0.17 s), about 1 s on the first
load after a build (shader compilation 0.25 s, pipelines 0.3 s). Autotune, when enabled, adds
about 4 s on the first load and nothing afterwards.

Reproduce:

```
swift build -c release
export GEMMA_EMBED_WEIGHTS=/path/to/embeddinggemma-2-Q8_0.gguf
.build/release/gemma-embed --bench /path/to/posts.json --runs 3

llama-server -m /path/to/embeddinggemma-2-Q8_0.gguf --embedding -c 8192 -b 8192 -ub 8192 -np 8 --port 8092 &
python3 scripts/bench_llamacpp.py /path/to/posts.json --port 8092 --batch 32 --runs 3
python3 scripts/ab_bench.py --a /path/to/old/gemma-embed --b .build/release/gemma-embed --bench /path/to/posts.json
```

## Design notes

The kernels follow these design rules:

- The weight file is mapped `MAP_SHARED` and read-only, pre-touched from several threads, and
  wrapped in one `MTLBuffer` with no copy. Tensors are offsets into that buffer.
- Shaders ship as `.metal` sources and compile when the model loads (0.2 s), because SwiftPM
  does not produce a metallib from the command line. Every pipeline is built at load. See
  Packaging for where the sources are read from.
- All arithmetic is fp32. Q8_0 and BF16 are storage formats widened inside the kernels.
- GEMM uses `simdgroup_float8x8` with X and dequantized W tiles staged in threadgroup memory,
  and stages the output tile so rows leave as coalesced `float4` stores. The tile is picked by
  output size (smaller tiles for small batches, a wider one for inner dimensions above 2048), and
  every tile accumulates over K in the same order, so a row's result is bit-identical whatever
  batch it travelled in. Every loop over fragment arrays is fully unrolled: a rolled loop keeps
  the arrays in memory and runs several times slower.
- Attention is the new work: flash-style, bidirectional, over sequences packed back to back
  without padding. A threadgroup owns 8 to 64 query rows of one sequence for one head and
  streams 32-key blocks staged once in threadgroup memory; the output accumulators stay in
  registers and the online softmax runs in fp32 on the score fragments themselves. Small batches
  split each head into narrower slices to fill the GPU, and the vision tower has its own kernel
  for its 64-wide heads. Sliding-window layers mask `|i - j| > 512`, the symmetric half window
  llama.cpp uses.
- The forward runs on a concurrent compute encoder with a buffer barrier after every dispatch,
  except within small groups of independent dispatches (q, k and v, for instance), which may
  overlap. A stress test runs two batches alternately on it and checks every result against a
  serial encoder bit for bit.
- Padding is avoided entirely: texts are concatenated, every per-token kernel runs on the
  concatenation, and attention takes a work list of (sequence, query block) pairs.
- Autotune, off by default, times whole forwards on real text for the GEMM tile and the
  attention query group, requires a 5 percent win, confirms the winner against the default and
  caches the result per device, kernel digest and model hash.

## Roadmap

- iOS and visionOS targets (the code has no macOS-only dependency beyond the platform entry).
- BF16 and F16 weight files.
- The audio encoder of the EmbeddingGemma 2 family (its tensors are in the mmproj file, unused).
- Overlapping image decoding with the GPU forward (worth about 5 percent at batch 8).
- Fusing the QKV projections and the per-layer-input gate into fewer dispatches.

## Author

Peter Marelas ([@maralski](https://github.com/maralski)).

## License

Copyright 2026 Peter Marelas. Apache License 2.0, the license Google releases EmbeddingGemma 2 and the Gemma 4 models under.
See [LICENSE](LICENSE) and [NOTICE](NOTICE). The model weights are not included and are
licensed separately by Google; Google asks that deployments follow the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). Code ported
from llama.cpp and stb_image keeps its MIT notices in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).

Gemma is a trademark of Google LLC. This project is not affiliated with or endorsed by Google.
