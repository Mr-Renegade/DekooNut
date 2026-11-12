"""SARC packs (.pack.zs) for tools/build_gamedata.py: unpack, replace or add members, repack; the zstd dictionaries the
game compresses them with; and the resource size table (RESTBL) entry a grown file needs. Input bytes are never
written back; every function returns new bytes. Needs Python 3.14 (compression.zstd).

Packs (read in Bootup.Nin_Ounce_NVN2, Foliage, ResidentCommon): a zstd frame (content size, single segment, no
checksum, dictionary ID 4) of a SARC compressed with the dictionary "pack.zsdic", one of the four in Pack/ZsDic.pack.zs
(a plain zstd SARC: zs.zsdic, bcett.byml.zsdic, pack.zsdic, bkprb.zsdic; the frame header names the one it needs).

SARC v1.0, little endian: header "SARC" (0x14 bytes: header size, BOM 0xFEFF, file size, data offset, version 0x100);
"SFAT" (0xC bytes: node count, hash key 0x65) and one 16-byte node per member (name hash, 0x01000000 | name offset / 4,
data start, data end, both relative to the data offset), nodes sorted by hash; "SFNT" (8 bytes) and the names, each
NUL-terminated and padded to 4. Members are stored in node order, each at an aligned offset, and the file ends at the
last member's end. The game's packs start every member at the next multiple of 8 (bgyml, byml, asb). Each member
keeps the smallest alignment that puts it where it was, at least 8 (_alignment), so a repack of an unchanged pack is
byte-identical (checked on all three) and a member that grows moves the later ones only to the next multiple of 8.

Compression: libzstd 1.5.7 at any level does not reproduce the shipped frames byte for byte, so compress() makes a frame with the same header fields (content size, single segment,
no checksum, the same dictionary ID) at level 19. Any zstd decoder holding the dictionary reads it; the frame's window
is its content size, like the original's.

RESTBL (System/Resource/ResourceSizeTable.Product.<version>.Nin_Ounce_NVN2.rsizetable.zs, plain zstd): "RESTBL",
version 1, name size 0xA0, then a table of {crc32 of the RomFS path without .zs, u32 size} sorted by crc, then a table
of {name[0xA0], u32 size} for paths whose crc collides. The size is the memory the game reserves for the resource; for
the three packs it is the decompressed size plus 316-336 bytes. restbl_grow() adds a member's growth (rounded up to 32)
to the pack's entry, so a mod never needs more than the table allows.
"""
import struct, zlib
import compression.zstd as zstd
from compression.zstd import CompressionParameter as CP

HASH_KEY = 0x65
MAX_ALIGN = 0x10000
_dicts = None        # name -> ZstdDict
_reader = None       # object with .read(romfs_path) -> bytes (romfs_read.Romfs)


def use_romfs(reader):
    """Where the dictionaries come from: anything with read(path) (tools/gamedata/romfs_read.Romfs)."""
    global _reader, _dicts
    _reader, _dicts = reader, None


def dicts():
    global _dicts
    if _dicts is None:
        if _reader is None: raise RuntimeError("sarc.use_romfs(reader) first: the zstd dictionaries are in Pack/ZsDic.pack.zs")
        raw = zstd.decompress(_reader.read("Pack/ZsDic.pack.zs"))
        _dicts = {n: zstd.ZstdDict(b) for n, b in Sarc(raw).items()}
    return _dicts


def frame_dict_id(frame):
    """The dictionary ID in a zstd frame header (0 = none)."""
    if frame[:4] != b"\x28\xb5\x2f\xfd": raise ValueError("not a zstd frame")
    fhd = frame[4]
    did_len = (0, 1, 2, 4)[fhd & 3]
    pos = 5 + (0 if fhd & 0x20 else 1)   # window descriptor unless single segment
    return int.from_bytes(frame[pos:pos + did_len], "little") if did_len else 0


def decompress(frame):
    """(bytes, dictionary name or None) for a .zs file."""
    did = frame_dict_id(frame)
    if not did: return zstd.decompress(frame), None
    for name, d in dicts().items():
        if d.dict_id == did: return zstd.decompress(frame, zstd_dict=d), name
    raise ValueError(f"no dictionary with ID {did} in Pack/ZsDic.pack.zs")


def compress(data, dict_name=None, level=19):
    """A frame like the shipped ones: content size, single segment, no checksum, the dictionary's ID."""
    opts = {CP.compression_level: level, CP.checksum_flag: 0, CP.content_size_flag: 1, CP.dict_id_flag: 1}
    return zstd.compress(data, options=opts, zstd_dict=dicts()[dict_name] if dict_name else None)


def name_hash(name):
    h = 0
    for c in name.encode("utf-8"): h = (h * HASH_KEY + (c if c < 0x80 else c - 0x100)) & 0xFFFFFFFF
    return h


def _alignment(prev_end, start):
    """The alignment to keep for a member stored at start after a member ending at prev_end: the smallest power of two
    that puts it exactly there (so an unchanged repack is byte-identical), but at least what its offset allows up to 8
    (so a grown earlier member never leaves it at an odd offset). A member behind more padding than any alignment
    explains keeps the largest power of two dividing its offset."""
    floor = 1
    while start % (floor * 2) == 0 and floor < 8: floor *= 2
    a = 1
    while a <= MAX_ALIGN:
        if prev_end + (-prev_end % a) == start: return max(a, floor)
        a *= 2
    a = floor
    while start % (a * 2) == 0 and a < MAX_ALIGN: a *= 2
    return a


class Sarc:
    """Members in pack order: name -> bytes, plus the alignment each member had."""

    def __init__(self, data):
        if data[:4] != b"SARC": raise ValueError("not a SARC")
        hs, bom = struct.unpack_from("<HH", data, 4)
        if bom != 0xFEFF: raise ValueError("big-endian SARC not supported")
        size, doff, ver = struct.unpack_from("<IIH", data, 8)
        if hs != 0x14 or size != len(data) or ver != 0x100: raise ValueError("unexpected SARC header")
        if data[0x14:0x18] != b"SFAT": raise ValueError("no SFAT")
        n, key = struct.unpack_from("<HI", data, 0x1A)
        if key != HASH_KEY: raise ValueError(f"hash key {key:#x}")
        sfnt = 0x20 + 16 * n
        if data[sfnt:sfnt + 4] != b"SFNT": raise ValueError("no SFNT")
        names = sfnt + 8
        self.data_offset = doff
        self.members = {}   # name -> [bytes, alignment]
        prev = doff
        for i in range(n):
            h, attr, s, e = struct.unpack_from("<IIII", data, 0x20 + 16 * i)
            if attr >> 24 != 1: raise ValueError("member without a name")
            no = names + (attr & 0xFFFFFF) * 4
            name = data[no:data.index(b"\0", no)].decode("utf-8")
            if name_hash(name) != h: raise ValueError(f"{name}: hash mismatch")
            a = doff + s
            if a < prev: raise ValueError("members not stored in node order")
            self.members[name] = [data[a:doff + e], _alignment(prev, a)]
            prev = doff + e

    def items(self):
        return {k: v[0] for k, v in self.members.items()}.items()

    def __getitem__(self, name):
        return self.members[name][0]

    def __contains__(self, name):
        return name in self.members

    def set(self, name, data, align=8):
        """Replace a member (it keeps its alignment) or add one (at align, a power of two)."""
        if name in self.members: self.members[name][0] = bytes(data)
        else: self.members[name] = [bytes(data), align]

    def build(self):
        order = sorted(self.members, key=name_hash)
        if len({name_hash(n) for n in order}) != len(order): raise ValueError("two members share a name hash")
        names, noff = bytearray(), {}
        for n in order:
            noff[n] = len(names) // 4
            names += n.encode("utf-8") + b"\0"
            names += b"\0" * (-len(names) % 4)
        head = 0x20 + 16 * len(order) + 8 + len(names)
        first = self.members[order[0]][1] if order else 1
        # the shipped data offset is kept while the name table still fits before it; otherwise the next one aligned to
        # the first member's alignment (at least 8)
        doff = self.data_offset if getattr(self, "data_offset", 0) >= head else head + (-head % max(8, first))
        nodes, body = bytearray(), bytearray()
        for n in order:
            b, align = self.members[n]
            pad = -(doff + len(body)) % align
            body += b"\0" * pad
            s = len(body); body += b
            nodes += struct.pack("<IIII", name_hash(n), 0x01000000 | noff[n], s, len(body))
        out = bytearray(b"SARC" + struct.pack("<HHIIHH", 0x14, 0xFEFF, 0, doff, 0x100, 0))
        out += b"SFAT" + struct.pack("<HHI", 0xC, len(order), HASH_KEY) + nodes
        out += b"SFNT" + struct.pack("<HH", 8, 0) + names
        out += b"\0" * (doff - len(out)) + body
        struct.pack_into("<I", out, 8, len(out))
        return bytes(out)


def edit_pack(frame, member, fn):
    """A .pack.zs frame with member's bytes replaced by fn(old bytes). Returns (new frame, growth in bytes of the
    decompressed pack)."""
    raw, dname = decompress(frame)
    s = Sarc(raw)
    if member not in s: raise KeyError(f"{member} is not in this pack")
    s.set(member, fn(s[member]))
    new = s.build()
    return compress(new, dname), len(new) - len(raw)


# ---- resource size table

def _restbl_parse(raw):
    if raw[:6] != b"RESTBL": raise ValueError("not a RESTBL")
    ver, nsz, nc, nn = struct.unpack_from("<IIII", raw, 6)
    if ver != 1: raise ValueError(f"RESTBL version {ver}")
    return nsz, nc, nn


def restbl_lookup(raw, path):
    """The size entry for a RomFS path (without .zs), or None."""
    nsz, nc, nn = _restbl_parse(raw)
    o = 22 + 8 * nc
    for i in range(nn):   # collision names first, as the game would match them by name
        e = o + i * (nsz + 4)
        if raw[e:e + nsz].rstrip(b"\0").decode("utf-8") == path: return struct.unpack_from("<I", raw, e + nsz)[0]
    h = zlib.crc32(path.encode("utf-8"))
    lo, hi = 0, nc
    while lo < hi:
        mid = (lo + hi) // 2
        k = struct.unpack_from("<I", raw, 22 + 8 * mid)[0]
        if k < h: lo = mid + 1
        elif k > h: hi = mid
        else: return struct.unpack_from("<I", raw, 22 + 8 * mid + 4)[0]
    return None


def restbl_set(raw, path, size):
    """RESTBL bytes with path's entry set to size (the entry must exist)."""
    nsz, nc, nn = _restbl_parse(raw)
    out = bytearray(raw)
    o = 22 + 8 * nc
    for i in range(nn):
        e = o + i * (nsz + 4)
        if raw[e:e + nsz].rstrip(b"\0").decode("utf-8") == path:
            struct.pack_into("<I", out, e + nsz, size); return bytes(out)
    h = zlib.crc32(path.encode("utf-8"))
    for i in range(nc):   # sorted, but a linear scan is fast enough for one edit and simpler to trust
        if struct.unpack_from("<I", raw, 22 + 8 * i)[0] == h:
            struct.pack_into("<I", out, 22 + 8 * i + 4, size); return bytes(out)
    raise KeyError(f"{path} has no RESTBL entry")


def restbl_grow(raw, path, growth):
    """Add growth bytes (rounded up to 32) to path's entry; a shrink or no growth leaves the table as it is."""
    if growth <= 0: return raw
    old = restbl_lookup(raw, path)
    if old is None: raise KeyError(f"{path} has no RESTBL entry")
    return restbl_set(raw, path, old + ((growth + 31) & ~31))
