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
