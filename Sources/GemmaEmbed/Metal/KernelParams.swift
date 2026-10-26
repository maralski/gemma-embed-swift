// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

// Swift mirrors of the parameter structs in Common.metal. Field order and types must match.

struct RowParams {
    var rows: UInt32
    var width: UInt32
    var inStride: UInt32
    var outStride: UInt32
    var eps: Float
    var preScale: Float
    var hasWeight: UInt32
}

struct RopeParams {
    var tokens: UInt32
    var heads: UInt32
    var headDim: UInt32
    var stride: UInt32
    var tableStride: UInt32
}

struct EmbedParams {
    var tokens: UInt32
    var paddedRows: UInt32
    var width: UInt32
    var scale: Float
}

struct GemmParams {
    var M: UInt32
    var N: UInt32
    var K: UInt32
    var weightRowBytes: UInt32
}

struct AttentionParams {
    var heads: UInt32
    var kvHeads: UInt32
    var headDim: UInt32
    var qStride: UInt32
    var kvStride: UInt32
    var oStride: UInt32
    var halfWindow: UInt32
    var queriesPerGroup: UInt32
    var scale: Float
}

struct AttentionWork {
    var seqStart: UInt32
    var seqLen: UInt32
    var q0: UInt32
}

struct PoolParams {
    var width: UInt32
    var stride: UInt32
}

struct SequenceRange {
    var start: UInt32
    var length: UInt32
}

struct PositionParams {
    var rows: UInt32
    var width: UInt32
    var positionCount: UInt32
}

struct PoolGrid {
    var patchStart: UInt32
    var tokenStart: UInt32
    var gridX: UInt32
    var gridY: UInt32
}

struct PoolPatchParams {
    var width: UInt32
    var merge: UInt32
    var scale: Float
}

/// Tile shape of the GEMM kernel; the kernel name encodes it.
public struct GemmTile: Sendable, Hashable, Codable {
    public let rows: Int
    public let columns: Int
    public let depth: Int
    /// Simdgroups per threadgroup (2, 4 or 8).
    public let simdgroups: Int

    public init(rows: Int, columns: Int, depth: Int, simdgroups: Int = 4) {
        self.rows = rows
        self.columns = columns
        self.depth = depth
        self.simdgroups = simdgroups
    }

    public static let `default` = GemmTile(rows: 64, columns: 64, depth: 16, simdgroups: 2)
    /// Every compiled tile; the tests check all of them.
    public static let compiled = [
        GemmTile(rows: 32, columns: 32, depth: 32, simdgroups: 4),
        GemmTile(rows: 32, columns: 64, depth: 32, simdgroups: 4),
        GemmTile(rows: 64, columns: 32, depth: 32, simdgroups: 4),
        GemmTile(rows: 32, columns: 64, depth: 16, simdgroups: 4),
        GemmTile(rows: 64, columns: 64, depth: 16, simdgroups: 4),
        GemmTile(rows: 64, columns: 64, depth: 16, simdgroups: 8),
        GemmTile(rows: 64, columns: 64, depth: 32, simdgroups: 8),
        GemmTile(rows: 64, columns: 128, depth: 16, simdgroups: 8),
        GemmTile(rows: 64, columns: 64, depth: 16, simdgroups: 2),
        GemmTile(rows: 16, columns: 32, depth: 16, simdgroups: 4),
        GemmTile(rows: 16, columns: 32, depth: 16, simdgroups: 2),
        GemmTile(rows: 64, columns: 128, depth: 16, simdgroups: 4),
    ]
    /// Tiles autotune considers.
    public static let candidates = [
        GemmTile(rows: 32, columns: 32, depth: 32, simdgroups: 4),
        GemmTile(rows: 32, columns: 64, depth: 32, simdgroups: 4),
        GemmTile(rows: 64, columns: 32, depth: 32, simdgroups: 4),
        GemmTile(rows: 32, columns: 64, depth: 16, simdgroups: 4),
        GemmTile(rows: 64, columns: 64, depth: 16, simdgroups: 4),
        GemmTile(rows: 64, columns: 64, depth: 16, simdgroups: 8),
        GemmTile(rows: 64, columns: 64, depth: 32, simdgroups: 8),
        GemmTile(rows: 64, columns: 128, depth: 16, simdgroups: 8),
        GemmTile(rows: 64, columns: 64, depth: 16, simdgroups: 2),
    ]

    var suffix: String { "\(rows)x\(columns)x\(depth)s\(simdgroups)" }
    var threads: Int { simdgroups * 32 }
}

/// Kernel configuration the forward pass runs with. Autotune may replace the defaults.
public struct KernelSchedule: Sendable, Hashable, Codable {
    /// Tile for GEMMs larger than every bound in `smallGemmTiles`; autotune picks it.
    public var gemmTile: GemmTile
    /// Query rows per attention threadgroup (8 per simdgroup).
    public var attentionQueries: Int

    public init(gemmTile: GemmTile = .default, attentionQueries: Int = 32) {
        self.gemmTile = gemmTile
        self.attentionQueries = attentionQueries
    }

    public static let `default` = KernelSchedule()
    public static let attentionQueryCandidates = [16, 32, 64]
    static let visionAttentionQueries = 64
    static let visionAttentionTileBytes = (64 * 36 + 32 * 68) * 4

    func attentionQueries(headDim: Int) -> Int {
        headDim == 64 ? Self.visionAttentionQueries : attentionQueries
    }
    /// Threadgroup memory the text attention variants request, above what they use, so fewer
    /// threadgroups share a GPU core.
    static let attentionOccupancyCapBytes = 17 * 1024
    static let attentionTileBytes = 32 * 68 * 4

    static let attentionSliceBelowThreadgroups = 128

    static func attentionSliceWidth(headDim: Int, threadgroups: Int) -> Int {
        guard headDim >= 256 else { return 64 }
        var width = headDim
        while width > 128 && threadgroups * (headDim / width) < attentionSliceBelowThreadgroups { width /= 2 }
        return width
    }

    static let smallGemmTiles: [(maxOutputElements: Int, tile: GemmTile)] = [
        (1 << 17, GemmTile(rows: 16, columns: 32, depth: 16, simdgroups: 4)),
        (1 << 18, GemmTile(rows: 16, columns: 32, depth: 16, simdgroups: 2)),
        (1 << 21, GemmTile(rows: 32, columns: 64, depth: 16, simdgroups: 4)),
    ]

    static let longInnerGemmTile = GemmTile(rows: 64, columns: 128, depth: 16, simdgroups: 4)
    /// The text model's longest GEMM inner dimension; only longer ones use `longInnerGemmTile`.
    static let longestTextInnerDimension = 2048
    static let longInnerMinOutputElements = 1 << 20

    func gemmTile(rows: Int, outputs: Int) -> GemmTile {
        Self.smallGemmTiles.first { rows * outputs <= $0.maxOutputElements }?.tile ?? gemmTile
    }

    func gemmTile(rows: Int, outputs: Int, inputs: Int) -> GemmTile {
        let long = Self.longInnerGemmTile
        if inputs > Self.longestTextInnerDimension && rows * outputs > Self.longInnerMinOutputElements && rows % long.rows == 0 && outputs % long.columns == 0 {
            return long
        }
        return gemmTile(rows: rows, outputs: outputs)
    }
}

// Mirrors of the structs in Audio.metal.

struct AudioConvParams {
    var inTime: UInt32
    var inFreq: UInt32
    var inChannels: UInt32
    var outTime: UInt32
    var outFreq: UInt32
    var outChannels: UInt32
    var inOffset: UInt32
    var outOffset: UInt32
    var eps: Float
}

struct AudioElementwiseParams {
    var rows: UInt32
    var width: UInt32
    var perDimWidth: UInt32
    var scale: Float
    var lo: Float
    var hi: Float
}

struct AudioRowsParams {
    var rows: UInt32
    var width: UInt32
    var taps: UInt32
    var scale: Float
}

struct AudioAttentionParams {
    var tokens: UInt32
    var heads: UInt32
    var headDim: UInt32
    var window: UInt32
    var softcap: Float
}
