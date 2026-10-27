# GemmaEmbed

GemmaEmbed runs Google's EmbeddingGemma 2 text, image and audio encoders in-process on Apple
Silicon with its own Metal kernels. It is a Swift package with no dependencies beyond the Apple
SDK: no llama.cpp, no MLX, no Core ML. You hand it the official GGUF weight files and get
768-dimensional embeddings that match llama.cpp's to a cosine similarity of 0.99999 for text,
0.99999 for images and 0.99998 for audio, in one shared space.

## Supported model

| | |
|---|---|
| Model | `embeddinggemma-2` (GGUF architecture `gemma-embedding2`), 24 layers, 271M parameters |
| Weight file | `embeddinggemma-2-Q8_0.gguf` from https://huggingface.co/ggml-org/embeddinggemma-2-GGUF |
| Size | 309,855,456 bytes |
| SHA256 | `2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135` |
| Output | 768 dimensions, Matryoshka truncation to 512, 256 or 128 |
| Input | up to 8192 tokens per text (default limit 2048), any number of texts per call |
| Image and audio encoders (optional) | `mmproj-embeddinggemma-2-Q8_0.gguf` from the same repository, 554,821,024 bytes, SHA256 `c4a8a52691ecef40618438928bdf9e68379b854e24166f292592353db0aab64f` (GGUF projectors `gemma4v`, 16 layers, and `gemma4a`, 12 Conformer layers; 473M parameters in all) |
| Image input | JPEG, PNG, HEIC and anything else ImageIO reads, or a `CGImage`; any size, scaled to 70 to 1120 vision tokens (default cap 256) |
| Audio input | WAV (parsed by GemmaEmbed), MP3, M4A, AIFF, CAF, FLAC (through AVFoundation) or PCM samples; any rate and channel count, resampled to 16 kHz mono; 25 rows per second, up to 5 minutes 27 seconds per clip |
| Platform | macOS 14 or later on Apple Silicon, Swift 6 |

The forward pass is documented in [Sources/GemmaEmbed/Model/Architecture.md](Sources/GemmaEmbed/Model/Architecture.md).

## Installation

Add the package to `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/maralski/gemma-embed-swift.git", from: "0.4.0"),
],
targets: [
    .target(name: "MyApp", dependencies: [.product(name: "GemmaEmbed", package: "gemma-embed-swift")]),
]
```

In Xcode, choose File > Add Package Dependencies and enter the same URL. The weights are not
part of the package. Download `embeddinggemma-2-Q8_0.gguf` (and, for images and audio,
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

### Text, images and audio together

One input can mix text, images and audio. The model reads each picture and each clip where it
sits among the text, in one forward pass, so a post with its photo and voice note embeds as one
document.

```swift
let mmproj = URL(fileURLWithPath: "mmproj-embeddinggemma-2-Q8_0.gguf")
let model = try await EmbeddingModel.load(from: textURL, options: LoadOptions(imageEncoder: mmproj, audioEncoder: mmproj))
let post = MultimodalInput([.text("Our walk on Saturday:"), .image(.url(photoURL)), .text("and what the guide said:"), .audio(.url(voiceNoteURL))])
let posts = try await model.embed([post, MultimodalInput(text: "Quarterly results", images: [.data(chartPNG)])])
let query = try await model.embed("guided hill walk", options: EmbedOptions(task: .query))
let score = model.similarity(query, posts[0])
```

The sequence is the one llama.cpp's `mtmd` builds for EmbeddingGemma 2:
`[bos] text [<|image>] image rows [<image|>] text [<|audio>] audio rows [<audio|>] ... [eos]`,
with the parts in the order given. `MultimodalEmbedOptions.task` prefixes the first text, or
stands alone before a picture or clip that opens the input. `maxTokens` counts every row; text is
cut from the end until the input fits and pictures and clips are never cut, so an input whose
pictures and clips alone exceed it throws `invalidOptions`. A text-only input embeds exactly as
`embed(_:)` does, and an image-only or audio-only input with `task: .none` exactly as
`embed(images:)` or `embed(audios:)` does. Inputs with images need the image encoder and inputs
with audio the audio encoder (one mmproj file holds both); text-only inputs need neither. Inputs
of all three kinds batch together, and each embeds to the same bits alone or in a batch.

The example's shape, a sentence, a fixture photo, a second sentence and a 3 s spoken clip (359
rows), embeds to cosine 0.9999997 against `llama-server --embedding --mmproj` given the same
parts, max abs diff 9.3e-5, on 2026-10-26.

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

The 27 October 2026 UTC hill climb compared in-process GemmaEmbed and llama.cpp on an Apple
M3 Max with 40 GPU cores and 128 GiB memory. Both used the same Q8_0 weights and image-token cap.
Each workload embeds the same eight encoded fixture images, either through eight separate
calls or one request with automatic batching. Timing includes decoding, resizing, both model
stages, synchronization, readback and normalization. File reads and model loading are excluded.

Each ratio is llama.cpp's median wall time divided by GemmaEmbed's median wall time. Higher
values favor GemmaEmbed. Each column is a separate run with seven samples per engine.

| Workload | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| 8 separate calls, cap 256 | 1.202× | 1.252× | 1.193× |
| One 8-image request, cap 256 | 1.179× | 1.217× | 1.170× |
| 8 separate calls, cap 1,120 | 1.239× | 1.273× | 1.247× |
| One 8-image request, cap 1,120 | 1.234× | 1.243× | 1.254× |

The final source is `b0e1dd8`. None of the eight image configurations earned retention, and
none of the twelve final comparisons reached the 1.50× target. These ratios describe the
existing implementation. The native automatic budget remains 4,096 patch rows.
See [COMPARISON.md](COMPARISON.md#images) for the settings and limits, and the
[recorded samples](benchmarks/hillclimb-2026-10-27.json) for all wall times.

## Audio embeddings

Load the model with the audio encoder, which lives in the same mmproj file as the image encoder,
and embed files, bytes or samples. The vectors live in the same space as the text and image
embeddings, so a spoken sentence can be scored against its transcript, and a sound against a
caption.

```swift
let model = try await EmbeddingModel.load(from: textURL, options: LoadOptions(audioEncoder: mmprojURL))
let clips = try await model.embed(audios: [.url(wavURL), .data(mp3Bytes), .samples(pcm, sampleRate: 44100, channels: 2)])
let transcript = try await model.embed("the train now departs from platform four", options: EmbedOptions(task: .query))
let score = model.similarity(transcript, clips[0])
print(model.info.audioEncoder?.layerCount)   // 12 when loaded, nil otherwise
```

`AudioEmbedOptions` carries the same `dimensions` and `normalise` as text, `batchSize` (clips per
GPU batch, by default packed by mel frames, two minutes of audio per batch) and `maxTokens`. A
clip costs one row per 40 ms plus four frame tokens; `maxTokens` defaults to the model's 8192
rows, which is 5 minutes 27 seconds, and a longer clip is refused with `invalidOptions` rather
than cut, because llama.cpp does not cut audio either. Pass the mmproj file as `imageEncoder`,
`audioEncoder` or both; the file is mapped once and only the towers asked for are built. Embedding
audio without the encoder throws `audioEncoderNotLoaded`.

The encoder sees llama.cpp's front end exactly: miniaudio's decode and resample to 16 kHz mono
(WAV bit for bit, including resampled and multichannel files; other codecs decode through
AVFoundation, so an MP3 sits within 3e-5 per sample of llama.cpp's), 128 log-mel bins every 10 ms
over a 20 ms window, and a split into 30 second chunks that the tower encodes one by one. A
`MultimodalInput` takes `.audio` parts beside `.text` and `.image` parts, framed with `<|audio>`
and `<audio|>` where they sit. The front end and the tower are documented in
[Sources/GemmaEmbed/Model/Architecture.md](Sources/GemmaEmbed/Model/Architecture.md).

### Audio parity

Reference vectors come from llama.cpp build 11716 (`llama-server --embedding --mmproj -ub 8192`)
over the same Q8_0 files, and per-node dumps of its audio graph taken through `libmtmd`'s eval
callback (`scripts/audio_dump.cpp`). Every fixture clip is synthesised by
`scripts/make_fixture_audio.py`: tones, a chord, bursts, chirps, noise, clicks, AM and FM signals,
silence, 0.3 to 7 second signals, the same signal at 8, 22.05, 44.1 and 48 kHz, in stereo and four
channels, and as 8, 24 and 32-bit integer and float32 WAV, plus a 30.0 s clip and a 35 s clip
that crosses the chunk boundary. No recordings and no voices are in the repository.
`scripts/oracle_audio.py` writes the reference vectors from a running llama-server.

| Check | Result |
|---|---|
| WAV decode and miniaudio's downmix and resample, 32 variants (bit depths, 3 to 6 channels, 8 to 96 kHz), plus FLAC, AIFF and CAF of one of them | bit-identical to miniaudio's samples |
| Log-mel front end vs llama.cpp's graph input, 8 chunks of 7 clips | bins above -2 within 3e-5; bins between -6 and -2 within 1.2e-3; bins at the 0.001 floor within 2.4e-3 |
| CPU reference audio tower vs llama.cpp per-node sums, 8 chunks, 2448 nodes | within 2.7e-4 of the absolute sum |
| CPU reference vs llama.cpp, 589 full tensors | cosine 0.999995 or better |
| Metal audio tower vs CPU reference, 7 clips including the two-chunk 35 s clip | row cosine 0.999999 or better, max abs diff 3.6e-3 on rows of norm 28 |
| Metal audio embeddings vs llama-server, all 29 fixture clips | min cosine 0.99998933 (the 35 s clip), max abs diff 7.2e-4 |
| A generated 5-minute clip (11 chunks, 7504 rows, not in the repository) vs llama-server | cosine 0.9999937 |
| Same clip alone and inside a batch of 5 mixed lengths, first or last | bit-identical |
| Text and image parity | unchanged (see Image parity and Parity) |

Run `swift test -c release` to reproduce (see Tests), or
`gemma-embed --parity-audio Tests/GemmaEmbedTests/Fixtures/audio`.

### Audio performance

The 27 October 2026 UTC hill climb compared the public audio path with llama.cpp's in-process
APIs on the same M3 Max. The primary corpus contains eight short and eight long synthetic
speech clips, spanning 3.4 to 85.1 seconds. Each sample embeds all eight clips of its length,
through eight separate calls or one request with automatic batching. Timing includes decoding,
resampling, log-mel preparation, the audio and language stages, readback and normalization.

Ratios are llama.cpp median wall time divided by GemmaEmbed median wall time, with seven
samples per engine in each of three runs.

| Workload | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| 8 short clips, separate calls | 1.481× | 1.290× | 1.392× |
| 8 short clips, one request | 1.935× | 2.000× | 1.696× |
| 8 long clips, separate calls | 1.347× | 1.328× | 1.310× |
| 8 long clips, one request | 1.430× | 1.370× | 1.367× |

Only the short-batch workload reached 1.50× in all three runs. Ten source configurations were
measured. No audio optimization was retained for release: the combined candidate passed its
correctness checks but failed text and image performance guards. The final comparison uses
the unchanged `b0e1dd8` baseline. Two further frontend ideas remained unmeasured.
See [COMPARISON.md](COMPARISON.md#audio) for the settings and limits, and the
[recorded samples](benchmarks/hillclimb-2026-10-27.json) for all wall times.

### Cross-modal check

Ten sentences spoken by macOS `say` (`scripts/make_speech_samples.sh`, not in the repository) are
scored against their ten transcripts and five off-topic texts, all embedded as queries. Every
clip ranks its own transcript first, by 0.17 or more over the next text, and llama-server's own
vectors give the same ranks; GemmaEmbed's similarities agree with llama-server's within 1e-4. Own
cosines run from 0.80 to 0.90, and the smallest margin is a recipe step whose nearest other text
is the other cooking sentence. The test skips when the speech folder is absent.

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
.build/release/gemma-embed --audio talk.wav ~/Recordings
.build/release/gemma-embed --parity-audio Tests/GemmaEmbedTests/Fixtures/audio
.build/release/gemma-embed --bench-audio ~/Recordings --runs 3
.build/release/gemma-embed --profile-audio ~/Recordings --batch-size 1
```

`gemma-embed` prints one JSON object per input line. `--parity` checks the tokenizer and the
embeddings against the llama.cpp fixtures and exits non-zero below cosine 0.9995. `--bench` times
a throughput run over a feed JSON file (`{"posts": [{"gist", "text"}]}`) or a plain text file with
one document per line. Image modes take the mmproj file from `--image-encoder` or
`$GEMMA_EMBED_IMAGE_WEIGHTS`: `--images FILE... | DIR` prints one JSON vector per image,
`--parity-images DIR` checks a fixtures folder (`manifest.json` and `embeddings.jsonl`) and
`--bench-images DIR` measures images per second, by default with patch-count batching, 8 and 1
images per batch (`--batch-size auto` or `N` picks one). Audio modes take the same mmproj file
from `--audio-encoder`, `$GEMMA_EMBED_AUDIO_WEIGHTS` or the image encoder path: `--audio FILE... |
DIR` prints one JSON vector per clip with its row count and seconds, `--parity-audio DIR` checks a
fixtures folder, `--bench-audio DIR` measures audio seconds per wall second and per GPU second,
and `--profile-audio DIR` times each audio kernel class. `--help` lists the rest.

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
| `models/audio/dumps/` | llama.cpp's audio graph for seven clips (`scripts/audio_dump.cpp`) and miniaudio's decode of the decode-check WAVs (`scripts/make_decode_checks.py`) | `GEMMA_EMBED_AUDIO_DUMPS` |
| `models/audio/speech/` | ten spoken sentences from `scripts/make_speech_samples.sh` with llama-server's vectors from `scripts/oracle_audio.py --speech` | none |

`scripts/vision_dump.cpp` and `scripts/audio_dump.cpp` list the commands that write the dump directories.

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

These results come from the 27 October 2026 UTC hill climbs against the in-process llama.cpp
C API, using release builds, resident models and full Metal offload. The machine is an Apple
M3 Max with 16 CPU cores, 40 GPU cores and 128 GiB memory. The comparator is llama.cpp
0.7.0-dev build 11716, commit `797de63b2`, with the same Q8_0 weights.

The text results below measure the code merged in [PR #2](https://github.com/maralski/gemma-embed-swift/pull/2),
following the short-text changes in [PR #1](https://github.com/maralski/gemma-embed-swift/pull/1).
These tables describe `main` at `b0e1dd8`; the `0.4.0` release predates the text changes.
Each value is llama.cpp median suite wall time divided by GemmaEmbed median suite wall time
in one seven-sample run. Higher values favor GemmaEmbed. The query suite makes twelve
individual calls with 13 to 16 tokens each; its time is not a single-query latency percentile.

| Workload | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| 12 queries, pre-tokenized | 1.548× | 1.794× | 1.445× |
| 12 queries, tokenizer included | 1.528× | 1.413× | 1.521× |
| One 64-token sequence | 1.973× | 1.964× | 2.137× |
| 8 × 64-token sequences | 1.414× | 1.375× | 1.535× |
| 32 × 64-token sequences | 1.461× | 1.317× | 1.272× |
| 8 × 256-token sequences | 1.391× | 1.306× | 1.223× |
| 8 × 1,024-token sequences | 1.256× | 1.192× | 1.115× |

Only the row labelled "tokenizer included" includes text tokenization. The other rows start
with identical token IDs and finish with normalized CPU vectors. The 8,192-token document
workload uses one native packed call and four llama.cpp packs under the fixed 2,048-token
batch setting.

The earlier short-text series reached 1.50× for pre-tokenized queries in all three runs.
The later batch series above did not meet that threshold across all four batch workloads.
All final runs remain reported. Shared-desktop drift moved absolute times, so these are
workload-specific measurements rather than a universal speed guarantee.

[COMPARISON.md](COMPARISON.md) includes both text series, image and audio results, exact
comparator settings, timing boundaries and quality checks. The
[benchmark data](benchmarks/hillclimb-2026-10-27.json) contains every final sample behind those
tables, with per-run medians and ranges in seconds. Measurements exclude model loading,
disk input reads, warmups and validation work.

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

- iOS and visionOS targets. With `.iOS(.v18)` added to the platforms, the library compiles for iOS
  unchanged (iOS 17 lacks `MTLCompileOptions.mathMode`); it has not yet run on a device or
  simulator, and visionOS has not been built.
- F16 weight files, which are refused at load today, and a check of BF16 files against llama.cpp
  (the GEMM kernels read BF16 weights, but no BF16 file has been tested).
- Fusing the text tower's q, k and v projections into one grouped GEMM, as the vision and audio
  towers already do, and the per-layer-input gate into fewer dispatches.

## Author

Peter Marelas ([@maralski](https://github.com/maralski)).

## License

Copyright 2026 Peter Marelas. Apache License 2.0, the license Google releases EmbeddingGemma 2 and the Gemma 4 models under.
See [LICENSE](LICENSE) and [NOTICE](NOTICE). The model weights are not included and are
licensed separately by Google; Google asks that deployments follow the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). Code ported
from llama.cpp and stb_image keeps its MIT notices in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).

Gemma is a trademark of Google LLC. This project is not affiliated with or endorsed by Google.
