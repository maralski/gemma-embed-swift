import Foundation

/// Precomputed NeoX rotation factors, `[position][pair] -> (cos, sin)`, computed in double
/// precision so the CPU reference and the Metal kernel rotate with identical factors.
struct RopeTable: Sendable {
    let headDim: Int
    let base: Float
    let positions: Int
    /// Interleaved cos, sin per `(position, pair)`.
    let values: [Float]

    var pairs: Int { headDim / 2 }

    init(headDim: Int, base: Float, positions: Int) {
        self.headDim = headDim
        self.base = base
        self.positions = positions
        let pairs = headDim / 2
        var values = [Float](repeating: 0, count: positions * pairs * 2)
        for i in 0..<pairs {
            let invFreq = pow(Double(base), -2.0 * Double(i) / Double(headDim))
            for pos in 0..<positions {
                let theta = Double(pos) * invFreq
                values[(pos * pairs + i) * 2] = Float(cos(theta))
                values[(pos * pairs + i) * 2 + 1] = Float(sin(theta))
            }
        }
        self.values = values
    }

    func factors(position: Int, pair: Int) -> (cos: Float, sin: Float) {
        let o = (position * pairs + pair) * 2
        return (values[o], values[o + 1])
    }
}
