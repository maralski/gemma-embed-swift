#!/usr/bin/env python3
"""Writes a JSON description of a GGUF file: metadata (long arrays summarised by a hash) and the tensor directory.

usage: scripts/gguf_dump.py <in.gguf> <out.json>
Produced Tests/GemmaEmbedTests/Fixtures/mmproj.json; model.json came from the earlier prototype script.
"""
import struct, sys, json, hashlib
T = {0:"u8",1:"i8",2:"u16",3:"i16",4:"u32",5:"i32",6:"f32",7:"bool",8:"str",9:"arr",10:"u64",11:"i64",12:"f64"}
FMT = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<?",10:"<Q",11:"<q",12:"<d"}
QT = {0:"F32",1:"F16",8:"Q8_0",30:"BF16"}
src, dst = sys.argv[1], sys.argv[2]
raw = open(src, "rb").read()
pos = 0
def rd(fmt):
    global pos
    v = struct.unpack_from(fmt, raw, pos); pos += struct.calcsize(fmt); return v[0]
def rstr():
    global pos
    n = rd("<Q"); s = raw[pos:pos+n].decode("utf-8", "replace"); pos += n; return s
def rval(t):
    if t == 8: return rstr()
    if t == 9:
        et = rd("<I"); n = rd("<Q")
        return {"type": T[et], "len": n, "items": [rval(et) for _ in range(n)]}
    return rd(FMT[t])
assert raw[:4] == b"GGUF"; pos = 4
ver = rd("<I"); nt = rd("<Q"); nkv = rd("<Q")
meta = {}
for _ in range(nkv):
    k = rstr(); t = rd("<I"); v = rval(t)
    if t == 9:
        it = v["items"]
        if v["len"] > 64:
            meta[k] = {"type": "arr[%s]" % v["type"], "len": v["len"], "first": it[:8], "sha256_of_json_items": hashlib.sha256(json.dumps(it).encode()).hexdigest()}
        else:
            meta[k] = {"type": "arr[%s]" % v["type"], "len": v["len"], "items": it}
    else:
        meta[k] = {"type": T[t], "value": v}
tensors = []
for _ in range(nt):
    name = rstr(); nd = rd("<I"); dims = [rd("<Q") for _ in range(nd)]; qt = rd("<I"); off = rd("<Q")
    tensors.append({"name": name, "shape_ne": dims, "type": QT.get(qt, str(qt)), "offset": off})
align = meta.get("general.alignment", {}).get("value", 32)
json.dump({"version": ver, "tensor_count": nt, "kv_count": nkv, "bytes": len(raw), "alignment": align,
           "data_offset": (pos + align - 1) // align * align, "sha256": hashlib.sha256(raw).hexdigest(),
           "metadata": meta, "tensors": tensors}, open(dst, "w"), indent=1)
