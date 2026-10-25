#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Shared helpers for every GemmaEmbed kernel. Storage types (Q8_0, BF16, F16) are widened to
// fp32 at load time; every arithmetic operation runs in fp32.

inline float bf16_to_float(ushort b) {
    return as_type<float>(uint(b) << 16);
}

inline float f16_to_float(ushort h) {
    return float(as_type<half>(h));
}

inline float gelu_tanh(float x) {
    const float c = 0.79788456080286535587989211986876f;
    return 0.5f * x * (1.0f + precise::tanh(c * x * (1.0f + 0.044715f * x * x)));
}

inline float gelu_quick(float x) {
    return x / (1.0f + precise::exp(-1.702f * x));
}

// Q8_0 layout: blocks of 32 along the contiguous dimension, each block is an fp16 scale
// followed by 32 int8 values (34 bytes). A row of K elements occupies K / 32 * 34 bytes.
constant uint Q8_BLOCK_BYTES = 34;

struct RowParams {
    uint rows;
    uint width;
    uint inStride;
    uint outStride;
    float eps;
    float preScale;
    uint hasWeight;
};

struct RopeParams {
    uint tokens;
    uint heads;
    uint headDim;
    uint stride;
    uint tableStride;
};

struct EmbedParams {
    uint tokens;
    uint paddedRows;
    uint width;
    float scale;
};

struct GemmParams {
    uint M;
    uint N;
    uint K;
    uint weightRowBytes;
};

struct AttentionParams {
    uint heads;
    uint kvHeads;
    uint headDim;
    uint qStride;
    uint kvStride;
    uint oStride;
    uint halfWindow;
    uint queriesPerGroup;
    float scale;
};

struct AttentionWork {
    uint seqStart;
    uint seqLen;
    uint q0;
};

struct PoolParams {
    uint width;
    uint stride;
};

struct Sequence {
    uint start;
    uint length;
};

struct PositionParams {
    uint rows;
    uint width;
    uint positionCount;
};

// One image of a vision batch: where its patch rows and token rows start, and its patch grid.
struct PoolGrid {
    uint patchStart;
    uint tokenStart;
    uint gridX;
    uint gridY;
};

struct PoolPatchParams {
    uint width;
    uint merge;
    float scale;
};
