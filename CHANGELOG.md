# Changelog

## 0.3.0 (2026-10-25)

Text and images in one input.

- `embed(_: MultimodalInput, options:)` and `embed(_: [MultimodalInput], options:progress:)`
  embed text and images in the order they appear, in one forward pass per input. The sequence is
  llama.cpp's `mtmd` layout for gemma4v: `[bos] text [<|image>] rows [<image|>] text ... [eos]`.
- `MultimodalEmbedOptions`: `task` prefixes the first text, `maxTokens` counts every row and cuts
  text from the end (images stay whole; an input whose images alone overflow is refused),
  `dimensions`, `normalise`, `maxImageTokens`, `batchSize`. Batches are cut by row and patch
  budget.
- A text-only input matches `embed(_:)` and an image-only input with no prefix matches
  `embed(images:)` exactly; text-only inputs work without the image encoder.
- Text and image APIs unchanged.

Speed, with vectors unchanged: text vectors within 3e-7 and image vectors within 2e-6 of 0.2.0's.
Measured on an Apple M3 Max on 2026-10-25:

- Text throughput 1.50 times 0.2.0's on 285 feed posts, a private corpus not in the repository
  (176 posts per second against 118), 1.49 times llama.cpp's best configuration. A 22-token
  query takes 7.4 ms wall instead of 15.2 ms.
- Image throughput 1.22 times 0.2.0's on 153 feed pictures, also private (10.4 images per second
  against 8.5).
- How: attention keeps its output accumulators in registers and runs its softmax on the score
  fragments; GEMM loops are fully unrolled and the tile is picked by output size; the vision
  tower has its own attention kernel and runs q, k and v as one grouped GEMM; head norms and RoPE,
  and residual updates and the next norm, run as single kernels; small batches split attention
  heads; the forward runs on a concurrent encoder with explicit barriers.
- Tools: `gemma-embed --profile-images`, `--batch-size auto`, `gpu_seconds_minsum` in
  `--bench-images`, `EmbeddingModel.imageForwardBatchSeconds`, and `scripts/ab_bench.py`,
  `scripts/ab_images.py` and `scripts/gpulock.py` for interleaved A/B measurement, and
  `scripts/compare_llamacpp.py` for the query latency and image throughput against llama-server.

Fixtures that can be redistributed:

- The text fixtures are 40 original texts, fictional posts among them, in place of the earlier
  set. `scripts/make_text_fixtures.py` regenerates the token ids, vectors, long texts and
  activation sums from llama.cpp.
- The image fixtures are drawn by `scripts/make_fixture_images.swift`: 10 procedural scenes
  (hills, a sunset, a coral reef, peaks, dunes, blossom, lavender, a night sky, sunflowers, a city)
  in place of the earlier pictures, with their variants and JPEG encodings.
- Reference vectors regenerated with llama.cpp build 11716. Parity: text min cosine 0.9999988 over
  40 fixtures, images min cosine 0.9999985 over 30 at cap 256 and 0.9999992 over 4 at cap 1120.
- The tests read weights, llama.cpp dumps and the larger images from `models/` at the package root
  (gitignored) unless an environment variable points elsewhere.

## 0.2.0 (2026-10-19)

Image embeddings, in the same 768-dimensional space as the text embeddings.

- `LoadOptions.imageEncoder` loads `mmproj-embeddinggemma-2-Q8_0.gguf` beside the text model;
  `ModelInfo.imageEncoder` reports it. Loading text alone never opens the file.
- `embed(images:options:progress:)` and `embed(image:options:)` over `ImageInput` (a file URL,
  encoded bytes or a `CGImage`), with `ImageEmbedOptions` (`dimensions`, `normalise`,
  `maxImageTokens`, `batchSize`). Decoding and resizing run on the CPU a batch ahead of the GPU.
- The `gemma4v` vision tower on Metal: patch projection, learned position tables, 2D NeoX RoPE,
  head-64 attention over packed images, `gelu_quick` GEMM epilogue, 3 by 3 average pooling and the
  projection into the text tower, which then runs over `[bos, <|image>, vectors, <image|>, eos]`
  with mixed token and vector rows. One command buffer per batch.
- Preprocessing identical to llama.cpp's `mtmd`: size choice, Pillow-style bicubic resize with
  black padding, fp16 rounding of the patch tensor; and a Swift port of stb_image's JPEG decoder
  so JPEG bytes match llama.cpp's exactly (ImageIO's decoder cost up to 0.004 of cosine).
- Parity with llama.cpp build 11716 on the 30 fixture images of this release, since replaced:
  min cosine 0.999996.
- CLI: `--image-encoder`, `--images`, `--max-image-tokens`, `--batch-size`, `--parity-images`,
  `--bench-images`; `--info` reports the image encoder.
- Fixtures and scripts: `scripts/gguf_dump.py`, `scripts/make_fixture_images.swift`,
  `scripts/oracle_images.py`, `scripts/vision_dump.cpp` (a `libmtmd` harness that dumps the
  vision graph nodes).
- Text API and text parity unchanged.

## 0.1.1 (2026-10-18)

- Fix: inside an app, the shaders were read only through SwiftPM's `Bundle.module`, which looks
  at the app's root and then at the absolute `.build` path of the checkout. A bundle copied into
  `Contents/Resources` was never read, and after `swift package clean` the first model load
  crashed. GemmaEmbed now searches the main bundle's resources, the library's own bundle and the
  main bundle's directory, and never reads `.build`.
- The shader source is also compiled into the library (`EmbeddedShaders.swift`, generated by
  `scripts/embed_shaders.py`), so a model loads even with no resource bundle.
- New error `GemmaEmbedError.shaderSourceMissing(tried:)` names every path searched.
- README: a Packaging section.

## 0.1.0 (2026-10-18)

First release.

- GGUF reader (v2 and v3) over a shared read-only memory mapping, with Q8_0, BF16, F16 and F32 tensor views.
- Gemma 4 style BPE tokenizer that reproduces llama.cpp token ids for the embeddinggemma-2 vocabulary.
- CPU fp32 reference forward used by the tests to validate every Metal kernel and the whole forward.
- Metal forward for `gemma-embedding2`: Q8_0 embedding gather, RMSNorm, NeoX RoPE, simdgroup-matrix
  GEMM over Q8_0 and BF16 weights with a fused GeGLU epilogue, bidirectional flash-style attention over
  packed variable-length sequences with symmetric sliding windows, fused post-norm residuals, mean pooling.
- `EmbeddingModel` API with task prefixes, Matryoshka truncation, token-budget batching, progress and
  cancellation; `gemma-embed` CLI with `--bench`, `--parity`, `--profile` and `--tune-report`.
- Optional per-device autotune over GEMM tiles and attention query groups, cached in
  `Application Support/GemmaEmbed/autotune.json`.
