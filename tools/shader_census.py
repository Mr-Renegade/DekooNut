"""Every Switch 2 shader stage of the packed RomFS, for tools/prepare_oot.py (nvn2_inventory). Library only.

Port of MacNativeGames games/totk/scripts/prepare_shader_seed.py (inventory, bnsh_shaders, sharcb_shaders, sarc) and the
package walk of locate_shaders.py @ b75be8cf (MIT, Copyright (c) 2026 MacNativeGames contributors).

Needs Python 3.14 (compression.zstd).
"""
import compression.zstd as zstd
import concurrent.futures
import hashlib
import json
import multiprocessing
import struct

ZSTD = b'\x28\xb5\x2f\xfd'
# NVN program control section magic, u32 0x98761234
CONTROL_MAGIC = b'\x34\x12\x76\x98'


def checked(data, offset, size):
    if offset < 0 or size < 0 or offset + size > len(data):
        raise ValueError('shader container range outside file')
    return data[offset:offset + size]


def sarc(data):
    if data[:4] != b'SARC':
        raise ValueError('expected SARC')
    endian = '<' if data[6:8] == b'\xff\xfe' else '>'
    header, size, start = struct.unpack_from(endian + 'HxxII', data, 4)
    if size != len(data) or header < 20 or checked(data, header, 4) != b'SFAT':
        raise ValueError('invalid SARC header')
    count = struct.unpack_from(endian + 'H', data, header + 6)[0]
    names = header + 12 + count * 16
    if checked(data, names, 4) != b'SFNT':
        raise ValueError('invalid SARC string table')
    names += 8
    result = {}
    for i in range(count):
        _, attribute, lo, hi = struct.unpack_from(endian + '4I', checked(data, header + 12 + i * 16, 16))
        off = names + (attribute & 0xffffff) * 4
        if attribute >> 24 != 1 or not names <= off < start or hi < lo or start + lo < start:
            raise ValueError('invalid SARC member')
        name = data[off:data.index(0, off, start)].decode('utf-8')
        # Names only label members here (nothing is written under them). This game's packs have names the
        # strict file-path rule refused, such as a leading '/', so odd or repeated ones get a unique label.
        if name.startswith('/') or '\\' in name or any(p in ('', '.', '..') for p in name.split('/')) or name in result:
            name = f'{name.strip("/")}#{i}'
        result[name] = checked(data, start + lo, hi - lo)
    return result


# A Switch 2 stage is Ampere code with a compact control section: magic 0x98761234, version 0xC9 at +4, its own length
# at +0x30 (0xd8, 0xd0 in effects, 0 in agl's packages and in memory), the code area's length at +0x44, the stage byte
# at +0x56 for compute, task and mesh (5, 6, 7). Layouts after nxnative dm/gfx/tools/shader_split.py (MIT): the BNSH
# program's stage pointers (+0x08 vs, +0x28 ps, +0x30 cs, +0x40 task, +0x48 mesh) and the BAHS record scan. The runtime
# lowers these stages at program creation (runtime/nvn2_lower.cpp); `oot --lower-shaders` does the same for the stages
# found here.
NVN2_DIRS = ('Shader/', 'Effect/', 'Lib/', 'Layout2Shader/')
NVN2_KEY_CTL = 0x48
NVN2_STAGES = (('vs', 0x08), ('ps', 0x28), ('cs', 0x30), ('ts', 0x40), ('ms', 0x48))
NVN2_STAGE_BYTE = {'cs': 5, 'ts': 6, 'ms': 7}
NVN2_STAGE_ID = {'vs': 0, 'ps': 4, 'cs': 5, 'ts': 6, 'ms': 7}   # runtime/gfx.h GFX_STAGE_*


def nvn2_control(ctl):
    return len(ctl) >= NVN2_KEY_CTL and ctl[:8] == CONTROL_MAGIC + b'\xc9\0\0\0'


def nvn2_key(ctl, code):
    """The stage key: the first 0x48 control bytes with +0x30 as zero, and the code area."""
    return hashlib.sha256(ctl[:0x30] + bytes(4) + ctl[0x34:NVN2_KEY_CTL] + code).hexdigest()[:16]


def nvn2_stage_of(ctl, code):
    """The stage of a stage found without a stage pointer: the control's stage byte, else the SPH type (1 vs, 5 ps)."""
    if len(ctl) > 0x56 and ctl[0x56] in (5, 6, 7):
        return {5: 'cs', 6: 'ts', 7: 'ms'}[ctl[0x56]]
    if len(code) < 4:
        return None
    return {1: 'vs', 5: 'ps'}.get(struct.unpack_from('<I', code)[0] >> 10 & 15)


def nvn2_bnsh(data, base):
    """(stage, control, code) of every NVN 2 stage of the BNSH at data[base:]; a pointer that leads elsewhere is skipped:
    an older program ends before the task and mesh pointers."""
    if data[base + 0x60:base + 0x64] != b'grsc':
        return
    count, variations = struct.unpack_from('<IQ', data, base + 0x7c)
    for v in range(count):
        at = base + variations + v * 0x40 + 0x10
        if at + 8 > len(data):
            return
        program = struct.unpack_from('<Q', data, at)[0]
        if not program:
            continue
        for stage, rel in NVN2_STAGES:
            at = base + program + rel
            if at + 8 > len(data):
                continue
            offset = struct.unpack_from('<Q', data, at)[0]
            if not offset or base + offset + 0x20 > len(data):
                continue
            ctl_at, code_at, code_size, ctl_size = struct.unpack_from('<QQII', data, base + offset + 8)
            if not 0x48 <= ctl_size < 0x10000 or base + ctl_at + ctl_size > len(data) or base + code_at + code_size > len(data):
                continue
            ctl, code = data[base + ctl_at:base + ctl_at + ctl_size], data[base + code_at:base + code_at + code_size]
            if not nvn2_control(ctl) or struct.unpack_from('<I', ctl, 0x44)[0] != len(code):
                continue
            if stage in NVN2_STAGE_BYTE and (len(ctl) <= 0x56 or ctl[0x56] != NVN2_STAGE_BYTE[stage]):
                continue
            yield stage, ctl, code


def nvn2_bahs(data):
    """agl's packages (.sharcb): records <u32 code offset in the pool, u32 0, u32 control size, i32 control from the
    record>; the pool holds the code area."""
    if len(data) < 0x20 or data[:4] != b'BAHS':
        return
    pool_size, pool = struct.unpack_from('<II', data, 0x18)
    if pool + pool_size > len(data):
        return
    for a in range(0, pool - 16, 4):
        off, zero, size, rel = struct.unpack_from('<IIIi', data, a)
        if zero or off % 0x100 or not 0x48 <= size < 0x10000 or a + rel < 0 or a + rel + size > pool:
            continue
        ctl = data[a + rel:a + rel + size]
        if not nvn2_control(ctl):
            continue
        n = struct.unpack_from('<I', ctl, 0x44)[0]
        if off + n > pool_size:
            continue
        code = data[pool + off:pool + off + n]
        stage = nvn2_stage_of(ctl, code)
        if stage:
            yield stage, ctl, code


def nvn2_sead_primitive(data):
    """sead's primitive drawer: four u32 offsets (vertex control, pixel control, vertex code, pixel code)."""
    if len(data) < 0x10:
        return
    offsets = struct.unpack_from('<4I', data)
    for stage, c, k in (('vs', offsets[0], offsets[2]), ('ps', offsets[1], offsets[3])):
        if c < 0x10 or c + 0x48 > len(data) or not nvn2_control(data[c:c + 0x48]):
            return
        ctl = data[c:c + max(struct.unpack_from('<I', data, c + 0x30)[0], 0x48)]
        code = data[k:k + struct.unpack_from('<I', ctl, 0x44)[0]]
        if nvn2_stage_of(ctl, code) == stage:
            yield stage, ctl, code


_nvn2_dictionaries = {}


def _nvn2_decode(raw):
    info = zstd.get_frame_info(raw)
    if info.decompressed_size is None or info.decompressed_size > 1 << 30:
        raise ValueError('unbounded archive')
    return zstd.decompress(raw, zstd_dict=_nvn2_dictionaries[info.dictionary_id] if info.dictionary_id else None)


def nvn2_ndsr(data):
    """agl's upscaler pack (agl_ndsr.pck): 'NDSR', 16-byte records from +0x10 up to a zero one: control offset, control
    length << 4 | type, code offset, type. A record that does not point at a control section holds data, and data
    entries can shift the next shader by 8."""
    def valid(c):
        return c + 0x48 <= len(data) and nvn2_control(data[c:c + 0x48])

    def score(o):
        return sum(1 for c, _, _, _ in struct.iter_unpack('<4I', data[o:o + 256]) if valid(c))

    o = max((0x10, 0x14), key=score)
    while o + 16 <= len(data):
        c, cl, k, tail = struct.unpack_from('<4I', data, o)
        if not (c or cl or k or tail):
            return
        if not valid(c):
            o += 8
            continue
        o += 16
        ctl = data[c:c + max(cl >> 4, 0x48)]
        code = data[k:k + struct.unpack_from('<I', ctl, 0x44)[0]]
        stage = nvn2_stage_of(ctl, code)
        if stage and len(code) == struct.unpack_from('<I', ctl, 0x44)[0]:
            yield stage, ctl, code


def _nvn2_visit(name, data, found, depth=0):
    if data[:4] == b'SARC' and depth < 5:
        for member, raw in sarc(data).items():
            _nvn2_visit(f'{name}/{member}', _nvn2_decode(raw) if raw[:4] == ZSTD else raw, found, depth + 1)
        return
    stages = list(nvn2_ndsr(data)) if data[:4] == b'NDSR' else list(nvn2_bahs(data)) or list(nvn2_sead_primitive(data))
    pos = 0
    while (pos := data.find(b'BNSH\0\0\0\0', pos)) >= 0:
        stages += nvn2_bnsh(data, pos)
        pos += max(struct.unpack_from('<I', data, pos + 28)[0], 8) if pos + 32 <= len(data) else 8
    for stage, ctl, code in stages:
        found.setdefault(nvn2_key(ctl, code), (stage, ctl, code, name))


def _nvn2_init(romfs_bin, dictionaries):
    global _nvn2_stream
    _nvn2_stream = open(romfs_bin, 'rb')
    for raw in dictionaries:
        d = zstd.ZstdDict(raw)
        _nvn2_dictionaries[d.dict_id] = d


def _nvn2_file(row):
    name, offset, size = row
    _nvn2_stream.seek(offset)
    raw = _nvn2_stream.read(size)
    found = {}
    if raw[:4] == ZSTD:
        raw = _nvn2_decode(raw)
    _nvn2_visit(name, raw, found)
    return found


def nvn2_inventory(romfs, workers=6, progress=None):
    """Every unique NVN 2 stage of a packed RomFS: key -> (stage, control, code, source). romfs holds romfs.bin and
    index.json (tools/romfs_index.py). Decompression dominates, so the files go to `workers` processes."""
    index = json.loads((romfs / 'index.json').read_text(encoding='utf-8'))
    if index.get('status') != 'pass':
        raise ValueError('RomFS has not passed integrity verification')
    rows = [(r['path'], r['offset'], r['size']) for r in index['files'] if r['path'].startswith(NVN2_DIRS)]
    dic = next(r for r in index['files'] if r['path'] == 'Pack/ZsDic.pack.zs')
    with (romfs / 'romfs.bin').open('rb') as stream:
        stream.seek(dic['offset'])
        dictionaries = list(sarc(zstd.decompress(stream.read(dic['size']))).values())
    rows.sort(key=lambda r: -r[2])   # the big archives first, so no worker is left with one at the end
    shaders, done = {}, 0
    with concurrent.futures.ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('fork'),
                                                initializer=_nvn2_init,
                                                initargs=(str(romfs / 'romfs.bin'), dictionaries)) as pool:
        for found in pool.map(_nvn2_file, rows, chunksize=4):
            for key, stage in found.items():
                shaders.setdefault(key, stage)
            done += 1
            if progress and done % 500 == 0:
                progress(done, len(rows), len(shaders))
    return shaders
