// Dumps llama.cpp's vision path for one image: the stb-decoded RGB bytes, the token
// layout mtmd builds around the image, every vision graph node (name, op, shape, sum)
// and the full data of selected nodes, plus the 512-wide output embeddings.
//
// usage: vision_dump <text.gguf> <mmproj.gguf> <image> <outdir> [max_image_tokens] [dump_regex]
// build (Homebrew llama.cpp 0.7.0 build 11716 and ggml):
//   clang++ -std=c++17 -O1 -I/opt/homebrew/opt/llama.cpp/include -I/opt/homebrew/opt/ggml/include \
//     -L/opt/homebrew/opt/llama.cpp/lib -L/opt/homebrew/opt/ggml/lib -lmtmd -lllama -lggml -lggml-base \
//     -Wl,-rpath,/opt/homebrew/opt/llama.cpp/lib -Wl,-rpath,/opt/homebrew/opt/ggml/lib -o vision_dump scripts/vision_dump.cpp
// The tests read the dumps from models/ (gitignored), or from the directories that
// GEMMA_EMBED_VISION_DUMPS and GEMMA_EMBED_DECODED_RGB_DUMPS name. To make them, write every image
// with scripts/make_fixture_images.swift into models/fixtures/images, put the two GGUF files in
// models/, then from the package root (absolute paths, since meta.json records the image path):
//   M=$PWD/models; I=$M/fixtures/images
//   dump() { mkdir -p "$2" && ./vision_dump $M/embeddinggemma-2-Q8_0.gguf $M/mmproj-embeddinggemma-2-Q8_0.gguf "$1" "$2" 256 ${3:+"$3"} 2> "$2/log.txt"; }
//   for f in $I/scenes/*.jpg $I/variants/*.jpg $I/synthetic/*.jpg; do dump $f $M/dumps-rgb/$(basename $f .jpg) '^$'; done
//   for f in $I/jpeg-variants/*.jpg; do dump $f $M/dumps-rgb/jv-$(basename $f .jpg) '^$'; done
//   for f in scenes/00-green-hills.jpg synthetic/02-gradient-h.png scenes/04-desert-dunes.jpg \
//            variants/06-lavender-field-cropped-10-top-left.jpg synthetic/07-shapes.jpg synthetic/09-tiny-noise.png; do
//     n=$(basename $f); dump $I/$f $M/dumps/${n%.*}; done
// dumps-rgb/ holds stb_image's decode of each JPEG and dumps/ the vision graph of six images.
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <regex>
#include <string>
#include <vector>
#include <fstream>

struct Dump {
    std::string outdir;
    std::regex keep;
    FILE * index;
    int n = 0;
};

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    Dump * d = (Dump *) ud;
    if (ask) return true;
    if (t->type != GGML_TYPE_F32) {
        fprintf(d->index, "{\"name\":\"%s\",\"op\":\"%s\",\"type\":\"%s\",\"ne\":[%lld,%lld,%lld,%lld],\"skipped\":true}\n",
                ggml_get_name(t), ggml_op_name(t->op), ggml_type_name(t->type), (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
        return true;
    }
    std::vector<float> data(ggml_nelements(t));
    ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
    double sum = 0, asum = 0;
    for (float v : data) { sum += v; asum += v < 0 ? -v : v; }
    std::string name = ggml_get_name(t);
    bool kept = std::regex_search(name, d->keep);
    std::string file;
    if (kept) {
        file = std::to_string(d->n) + "-" + name + ".f32";
        std::ofstream f(d->outdir + "/" + file, std::ios::binary);
        f.write((const char *) data.data(), data.size() * sizeof(float));
    }
    fprintf(d->index, "{\"i\":%d,\"name\":\"%s\",\"op\":\"%s\",\"ne\":[%lld,%lld,%lld,%lld],\"sum\":%.9g,\"abs_sum\":%.9g,\"first\":[%.9g,%.9g,%.9g,%.9g],\"file\":\"%s\"}\n",
            d->n, name.c_str(), ggml_op_name(t->op), (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
            sum, asum, data.size() > 0 ? data[0] : 0.f, data.size() > 1 ? data[1] : 0.f, data.size() > 2 ? data[2] : 0.f, data.size() > 3 ? data[3] : 0.f, file.c_str());
    d->n++;
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: vision_dump text.gguf mmproj.gguf image outdir [max_tokens] [regex]\n"); return 2; }
    const char * text_path = argv[1];
    const char * mmproj_path = argv[2];
    const char * image_path = argv[3];
    std::string outdir = argv[4];
    int max_tokens = argc > 5 ? atoi(argv[5]) : 256;
    std::string keep = argc > 6 ? argv[6] : "^(inp_raw_scaled|inp|pos_embd|layer_inp_normed-0|Qcur-0|Qcur_norm_per_head-0|Qcur_pos-0|Kcur_pos-0|Vcur_normed-0|kqv_out-0|attn_out-0|attn_post_normed-0|ffn_inp-0|ffn_inp_normed-0|ffn_out-0|ffn_post_normed-0|layer_out-0|layer_out-7|layer_out-15|pooled|projected)$";

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(text_path, mp);
    if (!model) { fprintf(stderr, "cannot load text model\n"); return 1; }

    Dump dump;
    dump.outdir = outdir;
    dump.keep = std::regex(keep);
    dump.index = fopen((outdir + "/nodes.jsonl").c_str(), "w");

    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = true;
    cp.print_timings = false;
    cp.n_threads = 8;
    cp.image_max_tokens = max_tokens;
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = &dump;
    cp.warmup = false;
    mtmd_context * ctx = mtmd_init_from_file(mmproj_path, model, cp);
    if (!ctx) { fprintf(stderr, "cannot load mmproj\n"); return 1; }

    mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(ctx, image_path, false, mtmd_helper_init_opt_default());
    mtmd_bitmap * bmp = bw.bitmap;
    if (!bmp) { fprintf(stderr, "cannot decode image\n"); return 1; }
    const uint32_t nx = mtmd_bitmap_get_nx(bmp), ny = mtmd_bitmap_get_ny(bmp);
    {
        std::ofstream f(outdir + "/decoded_rgb.bin", std::ios::binary);
        f.write((const char *) mtmd_bitmap_get_data(bmp), (size_t) nx * ny * 3);
    }

    mtmd_input_text text;
    text.text = "<__media__>";
    text.text_len = strlen(text.text);
    text.add_special = true;
    text.parse_special = true;
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bitmaps[1] = { bmp };
    if (mtmd_tokenize(ctx, chunks, &text, bitmaps, 1) != 0) { fprintf(stderr, "tokenize failed\n"); return 1; }

    FILE * meta = fopen((outdir + "/meta.json").c_str(), "w");
    fprintf(meta, "{\"image\":\"%s\",\"decoded_nx\":%u,\"decoded_ny\":%u,\"max_image_tokens\":%d,\"chunks\":[", image_path, nx, ny, max_tokens);
    size_t n_image_tokens = 0;
    const int n_embd_inp = llama_model_n_embd_inp(model);
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        if (i) fprintf(meta, ",");
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n = 0;
            const llama_token * toks = mtmd_input_chunk_get_tokens_text(chunk, &n);
            fprintf(meta, "{\"type\":\"text\",\"tokens\":[");
            for (size_t k = 0; k < n; k++) fprintf(meta, "%s%d", k ? "," : "", toks[k]);
            fprintf(meta, "]}");
        } else {
            const mtmd_image_tokens * it = mtmd_input_chunk_get_tokens_image(chunk);
            n_image_tokens = mtmd_image_tokens_get_n_tokens(it);
            fprintf(meta, "{\"type\":\"image\",\"n_tokens\":%zu}", n_image_tokens);
            if (mtmd_encode_chunk(ctx, chunk) != 0) { fprintf(stderr, "encode failed\n"); return 1; }
            const float * embd = mtmd_get_output_embd(ctx);
            std::ofstream f(outdir + "/output_embd.f32", std::ios::binary);
            f.write((const char *) embd, n_image_tokens * n_embd_inp * sizeof(float));
        }
    }
    fprintf(meta, "],\"n_embd_inp\":%d,\"n_image_tokens\":%zu}\n", n_embd_inp, n_image_tokens);
    fclose(meta);
    fclose(dump.index);
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bmp);
    mtmd_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
