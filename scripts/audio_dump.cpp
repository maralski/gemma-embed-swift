// Dumps llama.cpp's audio path for one sound file: the samples miniaudio decoded and resampled
// to 16 kHz mono, the token layout mtmd builds around the audio, every audio graph node (index,
// name, op, shape, sum) with the full data of selected nodes, and the 512-wide output embeddings
// of every 30-second chunk.
//
// usage: audio_dump <text.gguf> <mmproj.gguf> <audio file> <outdir> [keep_regex|all]
// build (Homebrew llama.cpp 0.7.0 build 11716 and ggml):
//   clang++ -std=c++17 -O1 -I/opt/homebrew/opt/llama.cpp/include -I/opt/homebrew/opt/ggml/include \
//     -L/opt/homebrew/opt/llama.cpp/lib -L/opt/homebrew/opt/ggml/lib -lmtmd -lllama -lggml -lggml-base \
//     -Wl,-rpath,/opt/homebrew/opt/llama.cpp/lib -Wl,-rpath,/opt/homebrew/opt/ggml/lib -o audio_dump scripts/audio_dump.cpp
// The gemma4a graph names almost none of its nodes, so nodes are matched by index, op and shape.
// `keep_regex` selects the nodes whose data is written in full by "<index>:<op>" (for instance
// '^(0|1):' or ':MUL_MAT$'); `all` keeps every node, which suits clips under a second.
// The tests read the dumps from models/audio/dumps (gitignored) or GEMMA_EMBED_AUDIO_DUMPS. To make
// them, write the fixtures with scripts/make_fixture_audio.py, then from the package root:
//   M=$PWD/models; A=Tests/GemmaEmbedTests/Fixtures/audio
//   dump() { mkdir -p "$2" && ./audio_dump $M/embeddinggemma-2-Q8_0.gguf $M/mmproj-embeddinggemma-2-Q8_0.gguf "$1" "$2" ${3:-all} 2> "$2/log.txt"; }
//   (the list of clips is in scripts/make_fixture_audio.py, under "dumps")
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
    bool keepAll = false;
    FILE * index;
    int n = 0;
};

static std::string safe(const std::string & s) {
    std::string out;
    for (char c : s) out += (isalnum((unsigned char) c) || c == '-' || c == '_' || c == '.') ? c : '_';
    return out;
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    Dump * d = (Dump *) ud;
    if (ask) return true;
    std::string name = ggml_get_name(t);
    if (t->type != GGML_TYPE_F32) {
        fprintf(d->index, "{\"i\":%d,\"name\":\"%s\",\"op\":\"%s\",\"type\":\"%s\",\"ne\":[%lld,%lld,%lld,%lld],\"skipped\":true}\n",
                d->n, name.c_str(), ggml_op_name(t->op), ggml_type_name(t->type), (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
        d->n++;
        return true;
    }
    std::vector<float> data(ggml_nelements(t));
    ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
    double sum = 0, asum = 0;
    for (float v : data) { sum += v; asum += v < 0 ? -v : v; }
    std::string key = std::to_string(d->n) + ":" + ggml_op_name(t->op);
    bool kept = d->keepAll || std::regex_search(key, d->keep);
    std::string file;
    if (kept) {
        file = std::to_string(d->n) + "-" + ggml_op_name(t->op) + (name.empty() ? "" : "-" + safe(name)) + ".f32";
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
    if (argc < 5) { fprintf(stderr, "usage: audio_dump text.gguf mmproj.gguf audio outdir [keep_regex|all]\n"); return 2; }
    const char * text_path = argv[1];
    const char * mmproj_path = argv[2];
    const char * audio_path = argv[3];
    std::string outdir = argv[4];
    std::string keep = argc > 5 ? argv[5] : "^(0|1|2|3):";

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(text_path, mp);
    if (!model) { fprintf(stderr, "cannot load text model\n"); return 1; }

    Dump dump;
    dump.outdir = outdir;
    dump.keepAll = keep == "all";
    dump.keep = std::regex(dump.keepAll ? "" : keep);
    dump.index = fopen((outdir + "/nodes.jsonl").c_str(), "w");

    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = true;
    cp.print_timings = false;
    cp.n_threads = 8;
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = &dump;
    cp.warmup = false;
    mtmd_context * ctx = mtmd_init_from_file(mmproj_path, model, cp);
    if (!ctx) { fprintf(stderr, "cannot load mmproj\n"); return 1; }
    if (!mtmd_support_audio(ctx)) { fprintf(stderr, "mmproj has no audio encoder\n"); return 1; }

    mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(ctx, audio_path, false, mtmd_helper_init_opt_default());
    mtmd_bitmap * bmp = bw.bitmap;
    if (!bmp || !mtmd_bitmap_is_audio(bmp)) { fprintf(stderr, "cannot decode audio\n"); return 1; }
    const uint32_t n_samples = mtmd_bitmap_get_nx(bmp);
    {
        std::ofstream f(outdir + "/samples.f32", std::ios::binary);
        f.write((const char *) mtmd_bitmap_get_data(bmp), (size_t) n_samples * sizeof(float));
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
    fprintf(meta, "{\"audio\":\"%s\",\"n_samples\":%u,\"sample_rate\":%d,\"chunks\":[", audio_path, n_samples, mtmd_get_audio_sample_rate(ctx));
    size_t n_audio_tokens = 0;
    int n_audio_chunks = 0;
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
            size_t n = mtmd_input_chunk_get_n_tokens(chunk);
            fprintf(meta, "{\"type\":\"audio\",\"n_tokens\":%zu,\"first_node\":%d", n, dump.n);
            if (mtmd_encode_chunk(ctx, chunk) != 0) { fprintf(stderr, "encode failed\n"); return 1; }
            fprintf(meta, ",\"end_node\":%d}", dump.n);
            const float * embd = mtmd_get_output_embd(ctx);
            std::ofstream f(outdir + "/output_embd-" + std::to_string(n_audio_chunks) + ".f32", std::ios::binary);
            f.write((const char *) embd, n * n_embd_inp * sizeof(float));
            n_audio_tokens += n;
            n_audio_chunks++;
        }
    }
    fprintf(meta, "],\"n_embd_inp\":%d,\"n_audio_tokens\":%zu,\"n_audio_chunks\":%d}\n", n_embd_inp, n_audio_tokens, n_audio_chunks);
    fclose(meta);
    fclose(dump.index);
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bmp);
    mtmd_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
