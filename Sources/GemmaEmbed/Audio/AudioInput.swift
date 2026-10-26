// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// A sound to embed.
public enum AudioInput: Sendable {
    /// A file: WAV is parsed by GemmaEmbed itself, anything else (MP3, M4A, AIFF, CAF, FLAC)
    /// is decoded through AVFoundation.
    case url(URL)
    /// Encoded audio bytes in any of the same formats.
    case data(Data)
    /// PCM samples, interleaved when `channels` is more than one, in the range -1 to 1.
    case samples([Float], sampleRate: Int, channels: Int = 1)
}
