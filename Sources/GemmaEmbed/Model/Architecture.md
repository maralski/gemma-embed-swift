# EmbeddingGemma 2 forward pass

Worked out from the GGUF tensor list and confirmed against llama.cpp build 11716
(`src/models/gemma-embedding2.cpp`) and its per-tensor activation dumps. The CPU reference
in `CPUReference.swift` and the Metal forward both implement exactly this.

Shapes use `T` tokens, `d = 512` hidden, `pl = 512` per-layer input width, `ff = 2048`,
`H = 4` query heads. Sliding-window layers (20 of 24) have `hd = 256`, 2 KV heads and RoPE base
1e4. Global layers (5, 11, 17, 23) have `hd = 512`, 1 KV head and RoPE base 1e6.

Norm weights in the GGUF already carry the `+1` that Gemma applies, so every RMSNorm is
`x / sqrt(mean(x^2) + 1e-6) * w`. Where no weight is named the norm is unscaled.

```
x = embed[tokens] * sqrt(d)                                         [T, d]
ple = rmsnorm_pl(x @ Wplp^T / sqrt(d), per_layer_proj_norm)         [T, 24, pl]  (norm per 512-chunk)

for each layer l:
  h = rmsnorm(x, attn_norm)
  q = h @ Wq^T  -> [T, H, hd];  k = h @ Wk^T -> [T, KV, hd];  v = h @ Wv^T -> [T, KV, hd]
  q = rmsnorm_hd(q, attn_q_norm);  k = rmsnorm_hd(k, attn_k_norm);  v = rmsnorm_hd(v, none)
  q, k = rope_neox(q, k, base_l, positions 0..T-1, all hd dims)
  a = softmax(q k^T + mask) v           scale 1.0 (QK norms make 1/sqrt(hd) unnecessary)
        mask: bidirectional; sliding layers mask |i - j| > 512 (symmetric half window)
  x = x + rmsnorm(a @ Wo^T, post_attention_norm)
  f = rmsnorm(x, ffn_norm)
  x = x + rmsnorm((gelu_tanh(f @ Wgate^T) * (f @ Wup^T)) @ Wdown^T, post_ffw_norm)
  g = gelu_tanh(x @ Winp_gate^T) * ple[:, l, :]
  x = (x + rmsnorm(g @ Wproj^T, post_norm)) * layer_output_scale[l]

y = rmsnorm(x, output_norm) @ Wout^T                                 [T, 768]
embedding = mean over tokens of y, then L2-normalised
```

Facts that were not obvious from the tensor names alone:

- There is no per-layer token embedding table (unlike Gemma 3n); the per-layer inputs come only
  from `per_layer_model_proj` applied to the scaled token embedding.
- `V` gets an unweighted RMS norm per head, `Q` and `K` get weighted ones.
- The attention score scale is 1.0.
- `layer_output_scale` is a scalar per layer applied after the per-layer-input residual.
- RoPE is NeoX style (rotate halves) over the full head dimension, with no scaling.
- Mean pooling runs over every token including BOS and EOS. Projecting per token then pooling
  equals pooling then projecting; the Metal path pools the 768-wide rows.
- `gelu_tanh(x) = 0.5 x (1 + tanh(0.7978845608 x (1 + 0.044715 x^2)))`, ggml's constants.

# Image encoder

Worked out from the `mmproj-embeddinggemma-2-Q8_0.gguf` tensor list and from llama.cpp build 11716
(`tools/mtmd/models/gemma4v.cpp`, `clip.cpp`, `mtmd-image.cpp`, `mtmd.cpp`), confirmed against
per-node dumps of its vision graph taken through the `cb_eval` callback of `libmtmd`. The GGUF
calls the projector `gemma4v`; the `a.*` audio tensors in the same file are not used.

Shapes use `P` patches per image, `dv = 768` hidden, `ffv = 3072`, `Hv = 12` heads of `hdv = 64`.

## Preprocessing (what llama.cpp's `mtmd` does, replicated exactly)

1. Decode the file to 8-bit RGB with no colour management, as llama.cpp's stb_image does
   (`stbi_load_from_memory(..., 3)`, which ignores embedded profiles and EXIF orientation).
   JPEGs go through `JPEGDecoder`, a Swift port of stb_image 2.30's scalar JPEG path (baseline
   and progressive Huffman, every sampling factor, restart intervals, JFIF and Adobe APP14,
   CMYK and YCCK), and decode to stb_image's bytes exactly. ImageIO's own JPEG decoder differs
   from stb_image by a level or two in up to a quarter of the bytes. Decoded through ImageIO
   (passed in as a `CGImage`), the 25 JPEG fixtures lose up to 0.0022 of cosine against
   llama-server (0.9978 on `scenes/09-city-skyline.jpg`); through `JPEGDecoder` the worst is
   0.999999. Other formats go through ImageIO, drawn into a bitmap of the image's own colour
   space so the samples are not converted; PNGs match stb_image exactly.
   JPEGs stb_image does not implement (12-bit, lossless, arithmetic coded) fall back to ImageIO,
   where llama.cpp would refuse them.
2. Choose the target size with `calc_size_preserved_ratio` (align 48 = patch 16 times merge 3):
   `w = max(48, round(W / 48) * 48)`, same for `h`. If `w * h > maxPixels`, scale both by
   `beta = sqrt(W * H / maxPixels)` and floor to a multiple of 48 (at least 48). Else if
   `w * h < minPixels`, scale by `beta = sqrt(minPixels / (W * H))` and ceil to a multiple of 48.
   `minPixels = 70 * 2304` (70 tokens). `maxPixels = maxImageTokens * 2304`; llama.cpp's native
   limit is 1120 tokens, but `llama-server` with its default `-ub 512` caps it at 256, and 256 is
   GemmaEmbed's default (`ImageEmbedOptions.maxImageTokens`).
3. Resize with the Pillow-compatible separable bicubic filter from `mtmd-image.cpp`
   (`resize_pillow`, `a = -0.5`, support 2, filter widened by the downscale factor, 22-bit
   fixed-point weights rounded Pillow style, horizontal pass then vertical, each pass rounding to
   8 bits). gemma4v keeps llama.cpp's default `PAD_CEIL`: the image is scaled by
   `s = min(w / W, h / H)` (fp32) to `min(ceil(W s), w)` by `min(ceil(H s), h)` and centred on a
   black `w` by `h` canvas at offset `((w - w') / 2, (h - h') / 2)`, so an aspect ratio that the
   rounding to 48 changed gets thin black bars (800x400 becomes 768x384 inside 816x384, 200x150
   becomes 480x360 inside 480x384). An image already at the target size is copied as is.
4. `f = byte / 255` (mean 0, std 1 in the GGUF), then `x = 2 f - 1` in fp32.
5. Patchify: patch `(px, py)` of a `gx = w / 16` by `gy = h / 16` grid is the 768-vector indexed
   `c * 256 + ky * 16 + kx` (channel outermost), patch index `i = py * gx + px`. ggml's conv2d runs
   its im2col in fp16, so every pixel value is rounded to fp16 before the patch projection.

## Vision tower

Every norm is an RMS norm with `eps = 1e-6`, `x / sqrt(mean(x^2) + eps) * w` (weights used as
stored). `gelu_quick(x) = x / (1 + exp(-1.702 x))`.

```
h = patches @ Wpatch^T                              [P, dv]   (Wpatch is v.patch_embd.weight [768][3][16][16] viewed as [768][768], f32, no bias)
h = h + posX[px] + posY[py]                          (v.position_embd.weight [768, 10240, 2]: rows 0..10239 are the x table, rows 10240..20479 the y table)

for each of the 16 layers:
  a = rmsnorm(h, ln1)
  q = a @ Wq^T -> [P, Hv, hdv];  k = a @ Wk^T;  v = a @ Wv^T          (768 -> 768 each, no bias)
  q = rmsnorm_hd(q, attn_q_norm);  k = rmsnorm_hd(k, attn_k_norm)      (weights of width 64)
  q, k = rope2d(q, k)      dims 0..31 rotated NeoX style by px, dims 32..63 by py,
                           theta_i = pos * 100^(-2i/32), pairs (i, i+16) within each half
  v = rmsnorm_hd(v, none)
  o = softmax(q k^T) v     scale 1.0, bidirectional over the patches of the same image only
  h = h + rmsnorm(o @ Wo^T, attn_post_norm)
  f = rmsnorm(h, ln2)
  h = h + rmsnorm((gelu_quick(f @ Wgate^T) * (f @ Wup^T)) @ Wdown^T, ffn_post_norm)

pooled = avgpool3x3(h over the gx by gy grid) * sqrt(dv)             [T, dv], T = (gx/3) * (gy/3), token index oy * (gx/3) + ox
vectors = rmsnorm(pooled, none) @ Wproj^T                            [T, 512]  (mm.input_projection.weight, Q8_0)
```

Facts that were not obvious from the tensor names alone:

- The FFN activation is `gelu_quick`, not the tanh GELU of the text tower: the GGUF has neither
  `clip.use_gelu` nor `clip.use_silu`, and llama.cpp's default for that case is `gelu_quick`.
- `n_merge = 3` and `rope_theta = 100` are hard-coded in llama.cpp for `gemma4v`; they are not
  metadata keys. `clip.vision.image_size = 224` is not used for anything.
- The learned position tables and the 2D RoPE both apply; the tables are indexed by the raw
  patch column and row.
- `V` gets an unweighted RMS norm per head, after `Q` and `K` are normed and rotated.
- The score scale is 1.0 and there is no attention mask within an image.
- There is no pre-norm or post-norm around the tower, no layer scales, no patch bias, and
  `std_bias`, `std_scale` and the `Gemma4ClippableLinear` clamps are absent for the vision
  tensors (the clamp scalars exist only for the audio tower).

## Into the text tower

`mtmd` wraps the `T` vectors as a sequence for the text model:

```
[BOS] [<|image> = 255999] v_1 ... v_T [<image|> = 258882] [EOS]
```

The four tokens are ordinary token embeddings scaled by `sqrt(512)`; the image rows enter the
residual stream unscaled (`build_inp_embd` scales only token lookups). Positions run 0 to T + 3,
the per-layer inputs are projected from these rows like any other, and the output is the mean
over all T + 4 rows of the projected hidden state, L2-normalised. Everything after the vectors is
the text forward of the previous section, which is why a text-only and an image embedding share
one space.

An input of text and images follows the same rule. `mtmd` tokenizes each text part without special
tokens, puts each image's vectors between `<|image>` and `<image|>` where its `<|image|>`
placeholder sat, and adds one BOS and one EOS around the whole, with nothing else in between:
`[BOS] text [<|image>] v... [<image|>] text ... [EOS]`, positions 0 to N - 1.

# Audio encoder

Worked out from the `a.*` and `mm.a.*` tensors of `mmproj-embeddinggemma-2-Q8_0.gguf` and from
llama.cpp build 11716 (`tools/mtmd/models/gemma4a.cpp`, `clip.cpp`, `mtmd-audio.cpp`, `mtmd.cpp`,
`mtmd-helper.cpp`), confirmed against per-node dumps of its audio graph taken through the
`cb_eval` callback of `libmtmd` (`scripts/audio_dump.cpp`). The GGUF calls the projector `gemma4a`.
The CPU reference in `CPUAudioReference.swift` and the Metal tower in `AudioTower.swift` both
implement exactly this.

Shapes use `F` log-mel frames per chunk, `T` tokens per chunk, `da = 1024` hidden, `ffa = 4096`,
`Ha = 8` heads of `hda = 128`, 12 layers. Every norm is an RMS norm with `eps = 1e-6` unless named
otherwise; llama.cpp hard-codes that value because the GGUF's `layer_norm_epsilon` was written
wrong by the converter.

## Decoding (what llama.cpp's miniaudio does, replicated exactly)

llama.cpp decodes every sound file with miniaudio into float32 mono at 16 kHz. `WAVDecoder` reads
WAV (PCM 8, 16, 24 and 32-bit, float 32 and 64, `WAVE_FORMAT_EXTENSIBLE`) with dr_wav's integer to
float conversions, `MiniaudioConverter` reproduces `ma_data_converter` (channels averaged in order
to mono first, then `ma_linear_resampler` with its two Butterworth low-pass biquads, fixed-point
fraction and `ma_calculate_frame_count_after_resampling` output length), and the decode tests
compare against miniaudio's own samples for 32 WAV variants, bit for bit, resampled rates
included. Other formats (MP3, M4A, AIFF, CAF, FLAC) go through AVFoundation at the file's own rate
and channel count and then through the same converter; only the codec differs from llama.cpp's
dr_mp3 or dr_flac (an MP3 decoded by both sits within 3e-5 per sample). `.samples` inputs are
taken as given and converted the same way.

## Front end (`mtmd_audio_preprocessor_gemma4a`)

1. Cut the samples into chunks of 30 s (480000 samples); the last chunk is whatever remains. A
   clip of a single sample has no frame and is refused; two samples give one frame.
2. Per chunk, put 160 zeros in front and enough zeros behind for the last frame, and take
   `F = (n + 160 - 321) / 160 + 1` frames of 512 samples every 160 samples (10 ms). The window is a
   periodic 320-sample Hann (20 ms) zero-padded to 512, so a frame is centred on `160 i` and sees
   10 ms either side.
3. 512-point FFT, magnitudes (`sqrt(re^2 + im^2)`, not powers), 128 triangular HTK mel filters
   from 0 Hz to 8 kHz with no area normalisation, `max(sum, 0.001)`, natural log. The tensor is
   `[F][128]`. GemmaEmbed's fp32 FFT agrees with llama.cpp's to 3e-5 in bins with energy and to
   2.4e-3 in bins at the floor, where roundoff relative to the frame's peak is magnified by the log.
4. Token count per chunk: `T = ceil(ceil(F / 2) / 2)` (two convolutions of kernel 3, stride 2,
   padding 1), so 30 s gives 2999 frames and 750 tokens, 25 tokens per second. llama.cpp splits
   at exactly 480000 samples, and a 35 s clip is 750 + 125 tokens.

## Audio tower

```
m = log_mel rounded to fp16                                           [F, 128]        (ggml's conv2d runs its im2col in fp16)
c0 = relu(layernorm_ch(conv3x3_s2_p1(m, 1 -> 128 channels)) * w)       [F/2, 64, 128]  (a.conv1d.0; LayerNorm over the channels, eps 1e-6, no bias)
c1 = relu(layernorm_ch(conv3x3_s2_p1(fp16(c0), 128 -> 32)) * w)        [T, 32, 32]     (a.conv1d.1)
x = flatten(c1) @ Winp^T                                               [T, da]         (row = freq * 32 + channel; a.input_projection, f32, no clamp)

for each of the 12 layers:                                             (every linear below is a Gemma4ClippableLinear, see Facts)
  x = x + 0.5 * rmsnorm(down(silu(up(rmsnorm(x, ffn_norm)))), ffn_post_norm)            up: da -> ffa, down: ffa -> da
  h = rmsnorm(x, attn_pre_norm)
  q = (h @ Wq^T) * (1 / sqrt(hda)) / ln 2 * per_dim_scale;  k = (h @ Wk^T) * softplus(1) / ln 2;  v = h @ Wv^T
  krel[d] = Wkrel @ sinusoid(d)   for d = 0 ... 12                     sin over dims 0..511 and cos over 512..1023 of position d, timescales 10000^(-i/511)
  score(t, j) = 50 tanh((q_t . k_j + q_t . krel[t - j]) / 50)          for j in max(chunk start, t - 11) ... t, per head
  x = x + rmsnorm(softmax_j(score) v @ Wo^T, attn_post_norm)
  h = rmsnorm(x, conv_norm)
  g = pw1(h)[:, :da] * sigmoid(pw1(h)[:, da:])                          pw1: da -> 2 da
  y[t] = sum_k g[t + k - 4] * w_dw[:, k]                               causal depthwise kernel 5 within the chunk (ggml_ssm_conv)
  x = x + pw2(silu(rmsnorm(y, norm_conv)))                             pw2: da -> da
  x = x + 0.5 * rmsnorm(down_1(silu(up_1(rmsnorm(x, ffn_norm_1)))), ffn_post_norm_1)
  x = rmsnorm(x, ln2)

o = rmsnorm(x @ Wout^T + bout, none)                                   [T, 1536]       (a.pre_encode.out, no clamp)
vectors = o @ Wproj^T                                                  [T, 512]        (mm.a.input_projection, Q8_0)
```

Facts that were not obvious from the tensor names alone:

- Every `a.blk.*` linear except `attn_k_rel` is a `Gemma4ClippableLinear`: its input is clamped to
  `[input_min, input_max]` and its output to `[output_min, output_max]`, four scalars stored beside
  the weight. The bounds are active (a layer-0 FFN input of 12.9 is cut at 12.8125). `q`, `k` and `v`
  share one set of bounds in every layer, so one clamped copy of the normed input feeds all three.
  llama.cpp registers clamps for every weight and defaults the missing ones to the float range,
  which is why the dumps show no-op `CLAMP` nodes around the unclamped projections too.
- The attention is local and causal: a frame sees itself and the 11 frames before it, within its
  chunk. llama.cpp computes it as blocked attention over 12-query blocks with a 24-key context and
  a Transformer-XL relative shift; the mask and the shift reduce to the window above with the
  relative key indexed by the distance `t - j`. `per_dim_scale` is stored after its softplus.
- The score soft cap is `50 tanh(s / 50)`, applied before the softmax, with no further scale.
- The two feed-forwards are half steps: their normed output is added with weight 0.5. The
  attention and convolution residuals are added with weight 1.
- The convolution module's norm weights are swapped in the GGUF: `conv_norm` is the norm before
  the module and `norm_conv` the norm after the depthwise convolution, as llama.cpp's loader reads
  them.
- Every layer ends with `ln2`, and the last layer's normed output is what `a.pre_encode.out`
  projects. The multimodal embedder's norm (`rms_norm` over 1536) has no weight; the GGUF has no
  `mm.a.soft_emb_norm`.
- The two convolutions have no bias; their LayerNorm (`ggml_norm`, mean and variance over the
  channels) has a weight and no bias.

## Into the text tower

`mtmd` wraps the `T` vectors of every chunk, in order, as one span for the text model:

```
[BOS] [<|audio> = 256000] v_1 ... v_T(chunk 1) ... v(chunk n) [<audio|> = 258883] [EOS]
```

The frame tokens are ordinary token embeddings scaled by `sqrt(512)`; the audio rows enter the
residual stream unscaled, as the image rows do. A 30 s clip is therefore 754 rows, and the text
model's 8192 rows hold 5 minutes 27 seconds of audio. Inside a `MultimodalInput`, every image and
clip keeps its own frame tokens where its part sits among the texts:
`[BOS] text [<|image>] v... [<image|>] text [<|audio>] a... [<audio|>] ... [EOS]`.

Chunks are independent: the front end, the convolutions' zero padding and the attention window all
restart at a chunk boundary, so a 35 s clip embeds differently from the same audio presented as one
sequence would. GemmaEmbed reproduces the split.

Parity (CPU reference and Metal against llama.cpp build 11716): over eight dumped chunks every one of
the 2448 graph node sums agrees within 2.7e-4 of the node's absolute sum, and 589 full tensors agree
to cosine 0.999995 or better. llama.cpp's Metal backend runs its Q8_0 matmuls over long sequences
with fp16 activation tiles, so a few output rows of a 750-token chunk sit at cosine 0.999 while the
median row is 0.9999996 and the mean-pooled rows are 0.9999998. Over the 29 fixture clips the final
embeddings agree with llama-server to cosine 0.99998933 or better.
