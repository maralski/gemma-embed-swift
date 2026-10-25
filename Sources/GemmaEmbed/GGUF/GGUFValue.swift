import Foundation

/// A metadata value read from a GGUF key-value section.
public enum GGUFValue: Sendable, Equatable {
    case uint8(UInt8)
    case int8(Int8)
    case uint16(UInt16)
    case int16(Int16)
    case uint32(UInt32)
    case int32(Int32)
    case float32(Float)
    case bool(Bool)
    case string(String)
    case uint64(UInt64)
    case int64(Int64)
    case float64(Double)
    case stringArray([String])
    case float32Array([Float])
    case int32Array([Int32])
    case boolArray([Bool])
    case array([GGUFValue])

    /// The value as an integer, for any integer-typed scalar.
    public var intValue: Int? {
        switch self {
        case .uint8(let v): return Int(v)
        case .int8(let v): return Int(v)
        case .uint16(let v): return Int(v)
        case .int16(let v): return Int(v)
        case .uint32(let v): return Int(v)
        case .int32(let v): return Int(v)
        case .uint64(let v): return Int(v)
        case .int64(let v): return Int(v)
        default: return nil
        }
    }

    public var floatValue: Float? {
        switch self {
        case .float32(let v): return v
        case .float64(let v): return Float(v)
        default: return intValue.map(Float.init)
        }
    }

    public var stringValue: String? {
        if case .string(let s) = self { return s }
        return nil
    }

    public var boolValue: Bool? {
        if case .bool(let b) = self { return b }
        return nil
    }

    public var stringArrayValue: [String]? {
        if case .stringArray(let a) = self { return a }
        return nil
    }

    public var float32ArrayValue: [Float]? {
        if case .float32Array(let a) = self { return a }
        return nil
    }

    public var int32ArrayValue: [Int32]? {
        switch self {
        case .int32Array(let a): return a
        case .array(let items): return items.map { Int32($0.intValue ?? 0) }
        default: return nil
        }
    }

    public var boolArrayValue: [Bool]? {
        if case .boolArray(let a) = self { return a }
        return nil
    }
}

/// Storage type of a GGUF tensor. Raw values follow ggml's `ggml_type` enumeration.
public enum GGUFTensorType: UInt32, Sendable {
    case f32 = 0
    case f16 = 1
    case q8_0 = 8
    case bf16 = 30

    public var name: String {
        switch self {
        case .f32: return "F32"
        case .f16: return "F16"
        case .q8_0: return "Q8_0"
        case .bf16: return "BF16"
        }
    }

    /// Number of elements that share one storage block.
    public var blockSize: Int {
        switch self {
        case .q8_0: return 32
        default: return 1
        }
    }

    /// Bytes per storage block.
    public var blockBytes: Int {
        switch self {
        case .f32: return 4
        case .f16, .bf16: return 2
        case .q8_0: return 34
        }
    }
}

/// Describes one tensor in a GGUF file: its name, shape, storage type and location.
public struct GGUFTensorInfo: Sendable, Equatable {
    public let name: String
    /// Dimensions in GGUF `ne` order: the fastest-varying dimension first.
    public let shape: [Int]
    public let type: GGUFTensorType
    /// Byte offset of the tensor data from the start of the data section.
    public let offset: Int

    public var elementCount: Int { shape.reduce(1, *) }

    public var byteCount: Int {
        let blocks = elementCount / type.blockSize
        return blocks * type.blockBytes
    }
}
