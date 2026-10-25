import Foundation

/// One row of the text tower's input sequence.
enum InputRow: Sendable, Equatable {
    /// A vocabulary id, looked up in the token embedding table and scaled by `sqrt(hidden)`.
    case token(Int32)
    /// A row already `hidden` wide, entered into the residual stream unscaled (an image or audio vector).
    case vector([Float])
}

/// The control tokens that frame an encoder's rows in the text sequence:
/// `[bos] [begin] v_1 ... v_T [end] [eos]`, with `<|image>`/`<image|>` around image rows and
/// `<|audio>`/`<audio|>` around audio rows.
struct SpanMarkers: Sendable, Equatable {
    let bos: Int32
    let begin: Int32
    let end: Int32
    let eos: Int32

    static let imageBeginText = "<|image>"
    static let imageEndText = "<image|>"
    static let audioBeginText = "<|audio>"
    static let audioEndText = "<audio|>"

    init(bos: Int32, begin: Int32, end: Int32, eos: Int32) {
        self.bos = bos
        self.begin = begin
        self.end = end
        self.eos = eos
    }

    init(tokenizer: Tokenizer, begin: String, end: String) throws {
        func single(_ text: String) throws -> Int32 {
            let ids = tokenizer.encode(text, addSpecialTokens: false, parseSpecialTokens: true)
            guard ids.count == 1 else { throw GemmaEmbedError.invalidGGUF("vocabulary has no single token for \(text)") }
            return ids[0]
        }
        self.init(bos: tokenizer.bosId, begin: try single(begin), end: try single(end), eos: tokenizer.eosId)
    }

    static func image(tokenizer: Tokenizer) throws -> SpanMarkers {
        try SpanMarkers(tokenizer: tokenizer, begin: imageBeginText, end: imageEndText)
    }

    static func audio(tokenizer: Tokenizer) throws -> SpanMarkers {
        try SpanMarkers(tokenizer: tokenizer, begin: audioBeginText, end: audioEndText)
    }

    /// The text-tower sequence for `vectors`, a flat array of rows `width` wide.
    func sequence(vectors: [Float], width: Int) -> [InputRow] {
        precondition(width > 0 && vectors.count % width == 0)
        let rows = (0..<vectors.count / width).map { InputRow.vector(Array(vectors[$0 * width..<($0 + 1) * width])) }
        return [.token(bos), .token(begin)] + rows + [.token(end), .token(eos)]
    }
}
