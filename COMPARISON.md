# GemmaEmbed and llama.cpp

This page records the text, image and audio hill-climb results from 27 October 2026 UTC.
Every timed comparison calls llama.cpp in process through its C API and, for media, its
`mtmd` APIs. Both engines use the same Q8_0 model files. The numbers describe warm embedding
work on one Apple M3 Max.

The [recorded benchmark data](benchmarks/hillclimb-2026-10-27.json) contains all seven timed
samples per engine, case and run, their medians and ranges in seconds, comparator settings,
source commits and hashes of the original result files and event logs.

## Measurement method

- Hardware: Apple M3 Max, 16 CPU cores, 40 GPU cores, 128 GiB memory.
- Comparator: llama.cpp 0.7.0-dev build 11716, commit `797de63b2`, with ggml 0.27.0 and full
  Metal offload. Comparator settings were tuned before each campaign's final series and then fixed.
- Weights: `embeddinggemma-2-Q8_0.gguf`, plus `mmproj-embeddinggemma-2-Q8_0.gguf` for media.
  [README model details](README.md#supported-model) lists their sizes and SHA-256 hashes.
- Sampling: three separately declared final runs, seven observations per engine and workload
  in each run. The engines alternate order, and GPU work runs serially.
- Estimator: median llama.cpp wall seconds divided by median GemmaEmbed wall seconds within
  the same run. A value of 1.50× means 50% higher throughput for equal work, or one third less time.
- Every run remains in the tables. Ratios are not averaged across workloads or pooled across
  runs to decide whether a target passed. The target must pass in each required case and run.
- Model and context loading, input-file reads, warmups, proof checks and record writing are
  outside timing. Each timed call computes new vectors and includes synchronization, CPU
  readback and normalization.

The machine remained an active desktop. Interleaving reduces bias, but it does not remove
scheduling, power-state or background-load effects. Absolute times drifted between runs.
The data file preserves each run's seven observations and minimum and maximum times.
Separate campaign results must not be subtracted to estimate a native code improvement.

## Text

### Merged text implementation

These results measure the final batch implementation tested at `69855d6`, subsequently merged
in [PR #2](https://github.com/maralski/gemma-embed-swift/pull/2) as `b0e1dd8`.
It includes the earlier [PR #1](https://github.com/maralski/gemma-embed-swift/pull/1) changes.
The text cases have the following fixed work counts:

| Case | Calls | Vectors | Total tokens | Native packs | llama.cpp packs |
|---|---:|---:|---:|---:|---:|
| `query-b1` | 12 | 12 | 171 | 12 | 12 |
| `doc64-b1` | 1 | 1 | 64 | 1 | 1 |
| `doc64-b8` | 1 | 8 | 512 | 1 | 1 |
| `doc64-b32` | 1 | 32 | 2,048 | 1 | 1 |
| `doc256-b8` | 1 | 8 | 2,048 | 1 | 1 |
| `doc1024-b8` | 1 | 8 | 8,192 | 1 | 4 |

The twelve queries contain 13 to 16 tokens each. A query sample times all twelve separate
calls. It is a suite time, not a single-query p50. The document cases time one embedding
request. The last case has 8,192 tokens: GemmaEmbed uses one packed call, while llama.cpp
uses four packs under its fixed 2,048-token batch and microbatch limits. The full work is timed.

#### Pre-tokenized text

Both engines receive identical token IDs. Timing runs through the GPU forward and normalized
CPU output, excluding tokenization.

| Workload | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| 12 queries, called individually | 1.548× | 1.794× | 1.445× |
| One 64-token sequence | 1.973× | 1.964× | 2.137× |
| 8 × 64-token sequences | 1.414× | 1.375× | 1.535× |
| 32 × 64-token sequences | 1.461× | 1.317× | 1.272× |
| 8 × 256-token sequences | 1.391× | 1.306× | 1.223× |
| 8 × 1,024-token sequences | 1.256× | 1.192× | 1.115× |

#### Tokenizer included

These separate measurements also include each engine's text tokenization.

| Workload | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| 12 queries, called individually | 1.528× | 1.413× | 1.521× |
| One 64-token sequence | 2.082× | 1.925× | 1.954× |
| 8 × 64-token sequences | 1.318× | 1.416× | 1.275× |
| 32 × 64-token sequences | 1.403× | 1.373× | 1.259× |
| 8 × 256-token sequences | 1.366× | 1.371× | 1.340× |
| 8 × 1,024-token sequences | 1.148× | 1.160× | 1.077× |

The batch campaign's target required all four multi-document cases to reach 1.50× in every
run. It was not met. The merged changes passed separate native comparisons and regression
checks; the tables above describe their final comparison against llama.cpp.

#### Text comparator settings

Both text campaigns use 12 CPU threads for each llama.cpp thread setting, mean pooling,
8,192 context tokens, batch and microbatch sizes of 2,048, 64 sequences, unified KV storage
and full GPU offload. Models remain resident. GemmaEmbed autotuning is disabled, its text
batch-token budget is 8,192, and the benchmark uses the same normalization in both engines.

### Earlier short-text series

The first hill climb measured `39b8b3f`, the source merged in
[PR #1](https://github.com/maralski/gemma-embed-swift/pull/1). Its primary pre-tokenized query
case reached 1.50× in all three runs. The tokenizer-inclusive query case missed in the first
run. Both complete six-case series are retained here as historical measurements of that source.
They are separate from the later merged batch series above.

Pre-tokenized text:

| Workload | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| 12 queries, called individually | 1.509× | 1.565× | 1.628× |
| One 64-token sequence | 1.670× | 1.649× | 1.645× |
| 8 × 64-token sequences | 1.247× | 1.335× | 1.216× |
| 32 × 64-token sequences | 1.386× | 1.318× | 1.289× |
| 8 × 256-token sequences | 1.295× | 1.319× | 1.217× |
| 8 × 1,024-token sequences | 1.265× | 1.083× | 1.127× |

Tokenizer included:

| Workload | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| 12 queries, called individually | 1.461× | 1.562× | 1.633× |
| One 64-token sequence | 1.643× | 1.684× | 1.673× |
| 8 × 64-token sequences | 1.200× | 1.229× | 1.302× |
| 32 × 64-token sequences | 1.310× | 1.309× | 1.272× |
| 8 × 256-token sequences | 1.257× | 1.268× | 1.230× |
| 8 × 1,024-token sequences | 1.206× | 1.075× | 1.118× |

## Images

The final source is `b0e1dd8`. Eight source configurations across six mechanisms were measured;
none passed the frozen retention rule. The final series therefore measures the existing
implementation, with no image optimization adopted from this campaign.

Each case embeds the same eight encoded fixture images. "Separate calls" means eight public
singleton calls. "One request" uses automatic native batching. The cap limits vision tokens
per image; each engine's actual pixel tensor and work receipts were checked at both caps.

Timing begins with resident encoded image bytes and includes decoding, resizing, patch
preparation, the vision and language stages, synchronization, readback, normalization and
owned outputs.

| Workload | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| 8 separate calls, cap 256 | 1.202× | 1.252× | 1.193× |
| One 8-image request, cap 256 | 1.179× | 1.217× | 1.170× |
| 8 separate calls, cap 1,120 | 1.239× | 1.273× | 1.247× |
| One 8-image request, cap 1,120 | 1.234× | 1.243× | 1.254× |

All twelve image comparisons missed 1.50×. These measurements do not establish a universal
performance limit or a gain over the starting native implementation.

### Image comparator settings

| Case | Vision mode | Vision threads | Text threads | Context rows | Batch and microbatch rows | Max sequences | Text flash |
|---|---|---:|---:|---:|---:|---:|---|
| `singles-256` | compatible | 8 | 12 | 16,384 | 8,192 | 64 | off |
| `batch-256` | serial | 8 | 12 | 16,384 | 8,192 | 64 | auto |
| `singles-1120` | compatible | 4 | 8 | 16,384 | 8,192 | 64 | auto |
| `batch-1120` | compatible | 8 | 12 | 4,096 | 2,048 | 8 | auto |

Preparation uses one worker in every case. Text batch-thread count equals text thread count.
Vision batch tokens are 1,024, with vision flash set to `auto` in every case. The native
automatic patch budget is 4,096 rows. Both engines use the same cap and process all eight images.

## Audio

The final source is the unchanged `b0e1dd8` baseline. Ten configurations across eight mechanisms
were measured. The combined candidate passed 155 active tests with the same 23 known skips,
but failed one text and one image performance guard. No audio optimization was accepted.
Two frontend ideas remained held and unmeasured, with no measured-configuration credit.

The primary corpus contains sixteen synthetic speech clips. The eight short clips span
3.4 to 5.0 seconds; the eight long clips span 28.0 to 85.1 seconds. Each sample
embeds all eight clips of its length. Separate calls and one automatic-batch request are
measured independently. A separate qualification covered 67 clips.

Timing begins with resident encoded bytes and includes public decoding and resampling,
complete log-mel preparation, the audio and whole-sequence language stages, pooling,
synchronization, readback, public normalization and owned returned outputs.

| Workload | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| 8 short clips, separate calls | 1.481× | 1.290× | 1.392× |
| 8 short clips, one request | 1.935× | 2.000× | 1.696× |
| 8 long clips, separate calls | 1.347× | 1.328× | 1.310× |
| 8 long clips, one request | 1.430× | 1.370× | 1.367× |

Only the three short-batch cells pass 1.50×. The target required all twelve cells to pass,
so the audio target was not met. These ratios describe the existing baseline's advantage
over the fixed llama.cpp comparator, not an accepted audio code change.

### Audio comparator settings

The selected configuration, T10, uses four preparation workers, eight audio threads, four
text threads and four text batch threads. Language mode is `batch`. Context is 16,384 rows,
batch and microbatch sizes are 8,192, and the maximum sequence count is 64. Audio and text
flash settings are both `auto`. The same configuration is used for all four final cases.

## Quality checks

Every final comparison completed without recorded work or correctness errors. Checks cover
finite, normalized 768-dimensional outputs and matching input and work counts. The fixed
per-vector cosine thresholds against actual llama.cpp results are 0.99999 for text and
images, and 0.99998 for audio.

Native candidates also had to preserve raw and normalized output bits against their frozen
native baselines. Image validation includes actual llama.cpp pixel witnesses at both caps
and packed-input independence. Audio validation includes exact native PCM and mel features,
PCM agreement with the actual llama.cpp input, strict mel checks, raw and normalized output
bits, and independence. Proof validation runs outside the timers.

Correctness success alone did not qualify a candidate for release. Selection also required
repeatable native gains and the declared regression guards. The image and audio targets
remained unmet, and their final comparisons use unchanged native sources.

## Recorded data and reproduction limits

[hillclimb-2026-10-27.json](benchmarks/hillclimb-2026-10-27.json) is a portable extraction of the
four verified campaign result files and their final event logs. It contains all 96 case/run
measurements, with seven samples per engine: 1,344 timed observations in total. The earlier
short-text campaign is labelled separately from the merged text implementation.

The following command recomputes every published ratio from those recorded samples:

```sh
python3 -I - <<'PY'
import json
import statistics
from pathlib import Path

results = json.loads(Path("benchmarks/hillclimb-2026-10-27.json").read_text())
for campaign, record in results["campaigns"].items():
    for run in record["runs"]:
        for case, row in run["cases"].items():
            native = statistics.median(row["native"]["samples_seconds"])
            llama = statistics.median(row["llama_cpp"]["samples_seconds"])
            assert llama / native == row["llama_over_native"]
            print(campaign, run["name"], case, f"{llama / native:.6f}x")
PY
```

This command replays arithmetic, not model execution. The complete frozen campaign tools,
fixtures, proof arrays, compiler receipts and original logs were retained separately; this
repository contains the compact final measurements. Running the CLI's `--bench` options
measures your chosen inputs but does not recreate these exact cases or their timing boundaries.
The dataset pins the measured commits and comparator settings. Later software versions or a
different machine require a fresh comparison.
