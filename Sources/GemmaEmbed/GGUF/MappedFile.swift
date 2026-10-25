import Foundation

/// A read-only `MAP_SHARED` memory mapping of a whole file.
///
/// Shared mappings are deliberate: the GPU faults pages of a private mapping far
/// more slowly, and a shared read-only mapping lets Metal wrap the bytes without a copy.
final class MappedFile: @unchecked Sendable {
    let base: UnsafeRawPointer
    let size: Int
    /// Mapping length rounded up to the page size, as `MTLBuffer` no-copy wrapping requires.
    let mappedLength: Int

    init(url: URL) throws {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw GemmaEmbedError.fileNotReadable(url.path, String(cString: strerror(errno))) }
        defer { close(fd) }
        var st = stat()
        guard fstat(fd, &st) == 0 else { throw GemmaEmbedError.fileNotReadable(url.path, String(cString: strerror(errno))) }
        let size = Int(st.st_size)
        guard size > 0 else { throw GemmaEmbedError.invalidGGUF("file is empty") }
        let page = Int(getpagesize())
        let mappedLength = (size + page - 1) / page * page
        guard let ptr = mmap(nil, mappedLength, PROT_READ, MAP_SHARED, fd, 0), ptr != MAP_FAILED else {
            throw GemmaEmbedError.fileNotReadable(url.path, "mmap failed: " + String(cString: strerror(errno)))
        }
        self.base = UnsafeRawPointer(ptr)
        self.size = size
        self.mappedLength = mappedLength
    }

    /// Touches every page from several threads so the first GPU use does not pay the page faults.
    func prefault() {
        madvise(UnsafeMutableRawPointer(mutating: base), mappedLength, MADV_WILLNEED)
        let page = Int(getpagesize())
        let chunk = 64 << 20
        let chunks = (size + chunk - 1) / chunk
        DispatchQueue.concurrentPerform(iterations: chunks) { i in
            let start = i * chunk
            let end = min(size, start + chunk)
            var acc: UInt8 = 0
            var p = start
            while p < end {
                acc &+= base.load(fromByteOffset: p, as: UInt8.self)
                p += page
            }
            _ = acc
        }
    }

    deinit {
        munmap(UnsafeMutableRawPointer(mutating: base), mappedLength)
    }
}
