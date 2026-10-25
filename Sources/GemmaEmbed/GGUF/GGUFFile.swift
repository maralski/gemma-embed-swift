import Foundation

/// A parsed GGUF file: metadata, tensor directory and a shared memory mapping of the bytes.
public final class GGUFFile: @unchecked Sendable {
    public let url: URL
    public let version: UInt32
    public let metadata: [String: GGUFValue]
    /// Tensors in file order.
    public let tensors: [GGUFTensorInfo]
    /// Offset of the aligned data section from the start of the file.
    public let dataOffset: Int
    public let alignment: Int
    let mapping: MappedFile

    private let tensorIndex: [String: Int]

    public var fileSize: Int { mapping.size }

    public init(url: URL) throws {
        self.url = url
        let mapping = try MappedFile(url: url)
        self.mapping = mapping
        var cursor = Cursor(base: mapping.base, size: mapping.size)
        let magic = try cursor.readUInt32()
        guard magic == 0x4655_4747 else { throw GemmaEmbedError.invalidGGUF("bad magic 0x\(String(magic, radix: 16))") }
        let version = try cursor.readUInt32()
        guard version == 2 || version == 3 else { throw GemmaEmbedError.invalidGGUF("unsupported version \(version)") }
        self.version = version
        let tensorCount = Int(try cursor.readUInt64())
        let kvCount = Int(try cursor.readUInt64())
        var metadata: [String: GGUFValue] = [:]
        metadata.reserveCapacity(kvCount)
        for _ in 0..<kvCount {
            let key = try cursor.readString()
            let type = try cursor.readUInt32()
            metadata[key] = try cursor.readValue(type: type)
        }
        self.metadata = metadata
        var tensors: [GGUFTensorInfo] = []
        tensors.reserveCapacity(tensorCount)
        var index: [String: Int] = [:]
        for i in 0..<tensorCount {
            let name = try cursor.readString()
            let ndim = Int(try cursor.readUInt32())
            guard ndim <= 4 else { throw GemmaEmbedError.invalidGGUF("tensor \(name) has \(ndim) dimensions") }
            var shape: [Int] = []
            for _ in 0..<ndim { shape.append(Int(try cursor.readUInt64())) }
            let rawType = try cursor.readUInt32()
            guard let type = GGUFTensorType(rawValue: rawType) else {
                throw GemmaEmbedError.unsupportedTensorType(name, rawType)
            }
            let offset = Int(try cursor.readUInt64())
            tensors.append(GGUFTensorInfo(name: name, shape: shape, type: type, offset: offset))
            index[name] = i
        }
        self.tensors = tensors
        self.tensorIndex = index
        let alignment = metadata["general.alignment"]?.intValue ?? 32
        self.alignment = alignment
        self.dataOffset = (cursor.position + alignment - 1) / alignment * alignment
        for t in tensors {
            let end = dataOffset + t.offset + t.byteCount
            guard end <= mapping.size else { throw GemmaEmbedError.invalidGGUF("tensor \(t.name) extends past the end of the file") }
        }
    }

    public func tensor(named name: String) -> GGUFTensorInfo? {
        tensorIndex[name].map { tensors[$0] }
    }

    /// Raw bytes of a tensor inside the mapping.
    public func bytes(of tensor: GGUFTensorInfo) -> UnsafeRawBufferPointer {
        UnsafeRawBufferPointer(start: mapping.base + dataOffset + tensor.offset, count: tensor.byteCount)
    }

    /// Absolute byte offset of a tensor from the start of the file.
    public func fileOffset(of tensor: GGUFTensorInfo) -> Int {
        dataOffset + tensor.offset
    }

    public func string(_ key: String) -> String? { metadata[key]?.stringValue }
    public func int(_ key: String) -> Int? { metadata[key]?.intValue }
    public func float(_ key: String) -> Float? { metadata[key]?.floatValue }
    public func bool(_ key: String) -> Bool? { metadata[key]?.boolValue }

    /// Touches every page of the mapping from several threads.
    public func prefault() { mapping.prefault() }
}

private struct Cursor {
    let base: UnsafeRawPointer
    let size: Int
    var position: Int = 0

    init(base: UnsafeRawPointer, size: Int) {
        self.base = base
        self.size = size
    }

    mutating func need(_ n: Int) throws {
        guard position + n <= size else { throw GemmaEmbedError.invalidGGUF("truncated header at byte \(position)") }
    }

    mutating func read<T>(_: T.Type) throws -> T {
        try need(MemoryLayout<T>.size)
        let v = base.loadUnaligned(fromByteOffset: position, as: T.self)
        position += MemoryLayout<T>.size
        return v
    }

    mutating func readUInt32() throws -> UInt32 { try read(UInt32.self) }
    mutating func readUInt64() throws -> UInt64 { try read(UInt64.self) }

    mutating func readString() throws -> String {
        let n = Int(try readUInt64())
        try need(n)
        let buf = UnsafeRawBufferPointer(start: base + position, count: n)
        position += n
        return String(decoding: buf, as: UTF8.self)
    }

    mutating func readValue(type: UInt32) throws -> GGUFValue {
        switch type {
        case 0: return .uint8(try read(UInt8.self))
        case 1: return .int8(try read(Int8.self))
        case 2: return .uint16(try read(UInt16.self))
        case 3: return .int16(try read(Int16.self))
        case 4: return .uint32(try read(UInt32.self))
        case 5: return .int32(try read(Int32.self))
        case 6: return .float32(try read(Float.self))
        case 7: return .bool(try read(UInt8.self) != 0)
        case 8: return .string(try readString())
        case 9:
            let elementType = try readUInt32()
            let count = Int(try readUInt64())
            switch elementType {
            case 8:
                var items: [String] = []
                items.reserveCapacity(count)
                for _ in 0..<count { items.append(try readString()) }
                return .stringArray(items)
            case 6:
                try need(count * 4)
                let items = [Float](unsafeUninitializedCapacity: count) { buf, n in
                    memcpy(buf.baseAddress!, base + position, count * 4)
                    n = count
                }
                position += count * 4
                return .float32Array(items)
            case 5:
                try need(count * 4)
                let items = [Int32](unsafeUninitializedCapacity: count) { buf, n in
                    memcpy(buf.baseAddress!, base + position, count * 4)
                    n = count
                }
                position += count * 4
                return .int32Array(items)
            case 7:
                var items: [Bool] = []
                items.reserveCapacity(count)
                for _ in 0..<count { items.append(try read(UInt8.self) != 0) }
                return .boolArray(items)
            default:
                var items: [GGUFValue] = []
                items.reserveCapacity(count)
                for _ in 0..<count { items.append(try readValue(type: elementType)) }
                return .array(items)
            }
        case 10: return .uint64(try read(UInt64.self))
        case 11: return .int64(try read(Int64.self))
        case 12: return .float64(try read(Double.self))
        default: throw GemmaEmbedError.invalidGGUF("unknown metadata value type \(type)")
        }
    }
}
