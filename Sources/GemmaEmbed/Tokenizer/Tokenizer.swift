import Foundation

/// The Gemma 4 style tokenizer carried by the GGUF file: SentencePiece-style BPE over raw
/// UTF-8 with `U+2581` for spaces, newline-run splitting and `<0xXX>` byte fallback.
///
/// The algorithm mirrors llama.cpp's `gemma4` vocabulary handling so that token ids match
/// what `llama-tokenize` produces for the same text.
public final class Tokenizer: @unchecked Sendable {
    public let bosId: Int32
    public let eosId: Int32
    public let padId: Int32
    public let unknownId: Int32?
    public let vocabularySize: Int

    private let pieces: [String]
    private let pieceBytes: [[UInt8]]
    private let types: [Int32]
    private let pieceToId: [String: Int32]
    private let merges: [MergeKey: Int32]
    private let byteToId: [Int32]
    /// Control, user-defined and unknown tokens, longest text first.
    private let specialTokens: [(id: Int32, bytes: [UInt8])]
    private let addBOS: Bool
    private let addEOS: Bool

    private static let space: [UInt8] = [0xE2, 0x96, 0x81]

    enum TokenType: Int32 {
        case normal = 1, unknown = 2, control = 3, userDefined = 4, unused = 5, byte = 6
    }

    private struct MergeKey: Hashable {
        let left: String
        let right: String
    }

    /// Builds the tokenizer from GGUF metadata.
    public convenience init(gguf: GGUFFile) throws {
        guard gguf.string("tokenizer.ggml.model") == "gemma4" else {
            throw GemmaEmbedError.unsupportedArchitecture("tokenizer model \(gguf.string("tokenizer.ggml.model") ?? "missing")")
        }
        guard let tokens = gguf.metadata["tokenizer.ggml.tokens"]?.stringArrayValue else { throw GemmaEmbedError.missingMetadata("tokenizer.ggml.tokens") }
        guard let types = gguf.metadata["tokenizer.ggml.token_type"]?.int32ArrayValue else { throw GemmaEmbedError.missingMetadata("tokenizer.ggml.token_type") }
        guard let merges = gguf.metadata["tokenizer.ggml.merges"]?.stringArrayValue else { throw GemmaEmbedError.missingMetadata("tokenizer.ggml.merges") }
        guard let bos = gguf.int("tokenizer.ggml.bos_token_id") else { throw GemmaEmbedError.missingMetadata("tokenizer.ggml.bos_token_id") }
        guard let eos = gguf.int("tokenizer.ggml.eos_token_id") else { throw GemmaEmbedError.missingMetadata("tokenizer.ggml.eos_token_id") }
        let pad = gguf.int("tokenizer.ggml.padding_token_id") ?? 0
        let unk = gguf.int("tokenizer.ggml.unknown_token_id")
        self.init(tokens: tokens, types: types, merges: merges,
                  bosId: Int32(bos), eosId: Int32(eos), padId: Int32(pad), unknownId: unk.map(Int32.init),
                  addBOS: gguf.bool("tokenizer.ggml.add_bos_token") ?? true,
                  addEOS: gguf.bool("tokenizer.ggml.add_eos_token") ?? false)
    }

    init(tokens: [String], types: [Int32], merges: [String], bosId: Int32, eosId: Int32, padId: Int32, unknownId: Int32?, addBOS: Bool, addEOS: Bool) {
        self.pieces = tokens
        self.pieceBytes = tokens.map { Array($0.utf8) }
        self.types = types
        self.bosId = bosId
        self.eosId = eosId
        self.padId = padId
        self.unknownId = unknownId
        self.addBOS = addBOS
        self.addEOS = addEOS
        self.vocabularySize = tokens.count
        var pieceToId: [String: Int32] = [:]
        pieceToId.reserveCapacity(tokens.count)
        for (i, t) in tokens.enumerated() where pieceToId[t] == nil { pieceToId[t] = Int32(i) }
        self.pieceToId = pieceToId
        var mergeMap: [MergeKey: Int32] = [:]
        mergeMap.reserveCapacity(merges.count)
        for (rank, merge) in merges.enumerated() {
            let bytes = Array(merge.utf8)
            guard bytes.count > 1, let split = bytes[1...].firstIndex(of: 0x20) else { continue }
            let key = MergeKey(left: String(decoding: bytes[..<split], as: UTF8.self),
                               right: String(decoding: bytes[(split + 1)...], as: UTF8.self))
            if mergeMap[key] == nil { mergeMap[key] = Int32(rank) }
        }
        self.merges = mergeMap
        var byteToId = [Int32](repeating: -1, count: 256)
        for b in 0..<256 {
            let name = String(format: "<0x%02X>", b)
            if let id = pieceToId[name], types[Int(id)] == TokenType.byte.rawValue { byteToId[b] = id }
        }
        self.byteToId = byteToId
        var specials: [(id: Int32, bytes: [UInt8])] = []
        for (i, t) in types.enumerated() {
            if t == TokenType.control.rawValue || t == TokenType.userDefined.rawValue || t == TokenType.unknown.rawValue {
                specials.append((Int32(i), Array(tokens[i].utf8)))
            }
        }
        specials.sort { $0.bytes.count > $1.bytes.count }
        self.specialTokens = specials
    }

    /// Encodes text into token ids.
    ///
    /// - Parameters:
    ///   - text: The input text.
    ///   - addSpecialTokens: Adds BOS and EOS as the model's metadata requests.
    ///   - parseSpecialTokens: Recognises the textual form of control tokens (for example `<bos>`)
    ///     inside `text`. Off by default so arbitrary input cannot inject control tokens.
    public func encode(_ text: String, addSpecialTokens: Bool = true, parseSpecialTokens: Bool = false) -> [Int32] {
        var output: [Int32] = []
        if addSpecialTokens && addBOS { output.append(bosId) }
        let bytes = Array(text.utf8)
        for fragment in partition(bytes, parseSpecial: parseSpecialTokens) {
            switch fragment {
            case .token(let id): output.append(id)
            case .text(let range): encodeFragment(bytes[range], into: &output)
            }
        }
        if addSpecialTokens && addEOS { output.append(eosId) }
        return output
    }

    /// Decodes token ids back to text. Control tokens are dropped; byte tokens are reassembled.
    public func decode(_ ids: [Int32]) -> String {
        var bytes: [UInt8] = []
        for id in ids {
            let i = Int(id)
            guard i >= 0 && i < pieces.count else { continue }
            switch types[i] {
            case TokenType.control.rawValue, TokenType.unknown.rawValue, TokenType.unused.rawValue:
                continue
            case TokenType.byte.rawValue:
                if let b = byteValue(ofPiece: pieces[i]) { bytes.append(b) } else { bytes.append(contentsOf: pieceBytes[i]) }
            default:
                bytes.append(contentsOf: pieceBytes[i])
            }
        }
        var out: [UInt8] = []
        out.reserveCapacity(bytes.count)
        var i = 0
        while i < bytes.count {
            if i + 2 < bytes.count && bytes[i] == 0xE2 && bytes[i + 1] == 0x96 && bytes[i + 2] == 0x81 {
                out.append(0x20)
                i += 3
            } else {
                out.append(bytes[i])
                i += 1
            }
        }
        return String(decoding: out, as: UTF8.self)
    }

    /// The text of a token id, as stored in the vocabulary.
    public func piece(_ id: Int32) -> String? {
        let i = Int(id)
        return i >= 0 && i < pieces.count ? pieces[i] : nil
    }

    public func isControl(_ id: Int32) -> Bool {
        let i = Int(id)
        return i >= 0 && i < types.count && types[i] == TokenType.control.rawValue
    }

    private func byteValue(ofPiece piece: String) -> UInt8? {
        guard piece.hasPrefix("<0x"), piece.hasSuffix(">"), piece.count == 6 else { return nil }
        return UInt8(piece.dropFirst(3).dropLast(), radix: 16)
    }

    private enum Fragment {
        case token(Int32)
        case text(Range<Int>)
    }

    private func partition(_ bytes: [UInt8], parseSpecial: Bool) -> [Fragment] {
        var fragments: [Fragment] = [.text(0..<bytes.count)]
        guard parseSpecial else { return fragments }
        for special in specialTokens where !special.bytes.isEmpty {
            var next: [Fragment] = []
            for fragment in fragments {
                guard case .text(let range) = fragment else { next.append(fragment); continue }
                var start = range.lowerBound
                while let match = find(special.bytes, in: bytes, from: start, to: range.upperBound) {
                    if match > start { next.append(.text(start..<match)) }
                    next.append(.token(special.id))
                    start = match + special.bytes.count
                }
                if start < range.upperBound { next.append(.text(start..<range.upperBound)) }
            }
            fragments = next
        }
        return fragments
    }

    private func find(_ needle: [UInt8], in hay: [UInt8], from: Int, to: Int) -> Int? {
        let n = needle.count
        guard n > 0, to - from >= n else { return nil }
        var i = from
        while i + n <= to {
            if hay[i] == needle[0] {
                var ok = true
                for j in 1..<n where hay[i + j] != needle[j] { ok = false; break }
                if ok { return i }
            }
            i += 1
        }
        return nil
    }

    private func encodeFragment(_ fragment: ArraySlice<UInt8>, into output: inout [Int32]) {
        var escaped: [UInt8] = []
        escaped.reserveCapacity(fragment.count + 16)
        for b in fragment {
            if b == 0x20 { escaped.append(contentsOf: Tokenizer.space) } else { escaped.append(b) }
        }
        var i = 0
        while i < escaped.count {
            let isNewline = escaped[i] == 0x0A
            var j = i + 1
            while j < escaped.count && (escaped[j] == 0x0A) == isNewline { j += 1 }
            encodeWord(Array(escaped[i..<j]), allNewlines: isNewline, into: &output)
            i = j
        }
    }

    private struct Symbol {
        var start: Int
        var length: Int
        var prev: Int
        var next: Int
    }

    private struct Bigram {
        let rank: Int32
        let left: Int
        let right: Int
        let leftLength: Int
        let rightLength: Int
    }

    private func encodeWord(_ word: [UInt8], allNewlines: Bool, into output: inout [Int32]) {
        var symbols: [Symbol] = []
        if allNewlines, pieceToId[String(decoding: word, as: UTF8.self)] != nil {
            symbols.append(Symbol(start: 0, length: word.count, prev: -1, next: -1))
        } else {
            var offset = 0
            while offset < word.count {
                let len = min(word.count - offset, utf8Length(word[offset]))
                let index = symbols.count
                symbols.append(Symbol(start: offset, length: len, prev: index - 1, next: offset + len == word.count ? -1 : index + 1))
                offset += len
            }
        }
        var heap = BigramHeap()
        func pushBigram(_ left: Int, _ right: Int) {
            guard left >= 0 && right >= 0 else { return }
            let l = symbols[left], r = symbols[right]
            let key = MergeKey(left: String(decoding: word[l.start..<(l.start + l.length)], as: UTF8.self),
                               right: String(decoding: word[r.start..<(r.start + r.length)], as: UTF8.self))
            guard let rank = merges[key] else { return }
            heap.push(Bigram(rank: rank, left: left, right: right, leftLength: l.length, rightLength: r.length))
        }
        if symbols.count > 1 {
            for i in 1..<symbols.count { pushBigram(i - 1, i) }
        }
        while let bigram = heap.pop() {
            let l = symbols[bigram.left], r = symbols[bigram.right]
            if l.length == 0 || r.length == 0 || l.length != bigram.leftLength || r.length != bigram.rightLength { continue }
            symbols[bigram.left].length += r.length
            symbols[bigram.right].length = 0
            symbols[bigram.left].next = r.next
            if r.next >= 0 { symbols[r.next].prev = bigram.left }
            pushBigram(symbols[bigram.left].prev, bigram.left)
            pushBigram(bigram.left, symbols[bigram.left].next)
        }
        for sym in symbols where sym.length > 0 {
            let slice = word[sym.start..<(sym.start + sym.length)]
            if let id = pieceToId[String(decoding: slice, as: UTF8.self)] {
                output.append(id)
            } else {
                for b in slice {
                    let id = byteToId[Int(b)]
                    if id >= 0 { output.append(id) }
                }
            }
        }
    }

    private func utf8Length(_ lead: UInt8) -> Int {
        if lead < 0x80 { return 1 }
        if lead & 0xE0 == 0xC0 { return 2 }
        if lead & 0xF0 == 0xE0 { return 3 }
        if lead & 0xF8 == 0xF0 { return 4 }
        return 1
    }

    /// Min-heap on (rank, left index), matching llama.cpp's merge order.
    private struct BigramHeap {
        private var items: [Bigram] = []

        private func less(_ a: Bigram, _ b: Bigram) -> Bool {
            a.rank < b.rank || (a.rank == b.rank && a.left < b.left)
        }

        mutating func push(_ item: Bigram) {
            items.append(item)
            var i = items.count - 1
            while i > 0 {
                let parent = (i - 1) / 2
                if less(items[i], items[parent]) { items.swapAt(i, parent); i = parent } else { break }
            }
        }

        mutating func pop() -> Bigram? {
            guard let top = items.first else { return nil }
            let last = items.removeLast()
            if !items.isEmpty {
                items[0] = last
                var i = 0
                while true {
                    let l = 2 * i + 1, r = l + 1
                    var m = i
                    if l < items.count && less(items[l], items[m]) { m = l }
                    if r < items.count && less(items[r], items[m]) { m = r }
                    if m == i { break }
                    items.swapAt(i, m)
                    i = m
                }
            }
            return top
        }
    }
}
