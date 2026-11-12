#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Write the game's Deku Nut item icon as a PNG, read from the player's own game files (for the install program).

    python3.14 tools/game_icon.py --romfs <folder with romfs.bin + index.json> --out <file.png>
    python3.14 tools/game_icon.py --dir <extracted RomFS folder> --out <file.png>
    python3.14 tools/game_icon.py --nup <game>.nup [--package update] --out <file.png>

The icon is RomFS UI/Tex/Icon/Item_DekuNut.bntx.zs: a zstd frame (a dictionary from Pack/ZsDic.pack.zs if the frame
names one) of a BNTX holding one ASTC texture in block-linear storage. The header is read here; the block-linear copy
and the ASTC decode are the runtime's (runtime/tex_deswizzle.c, runtime/tex_astc.c), built with clang on first use
into build/setup/icon/oot_icon by way of tools/icon/oot_icon.c. The PNG is game data: never commit it.

Python 3.14 (compression.zstd) and clang; otherwise the standard library.
"""
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / 'runtime'
BUILD = ROOT / 'build' / 'setup' / 'icon'
ICON = 'UI/Tex/Icon/Item_DekuNut.bntx.zs'
SOURCES = [ROOT / 'tools' / 'icon' / 'oot_icon.c', RUNTIME / 'tex_astc.c', RUNTIME / 'tex_deswizzle.c',
           RUNTIME / 'nvn_tex.c']
HEADERS = [RUNTIME / n for n in ('tex.h', 'nvn_tex.h', 'nvn_obj.h', 'nvn.h', 'cfg.h')]
# tex.h includes vk.h for two type names only; the helper sees this instead of the graphics headers.
VK_STUB = '#pragma once\n#include <stdint.h>\ntypedef int32_t VkFormat;\ntypedef uint32_t VkFormatFeatureFlags;\n'
ASTC = [(4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (8, 5), (8, 6), (8, 8), (10, 5), (10, 6), (10, 8), (10, 10),
        (12, 10), (12, 12)]
ASTC_FIRST = 0x2D        # BNTX format type of ASTC 4x4; the others follow in the order above
UNORM, SRGB = 1, 6       # BNTX format data types

sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'tools' / 'gamedata'))
import extract_nup as en   # noqa: E402


class IconError(Exception):
    pass


class Folder:
    def __init__(self, root):
        self.root = Path(root)

    def read(self, path):
        p = self.root.joinpath(*path.split('/'))
        if not p.is_file(): raise IconError(f'{p} not found')
        return p.read_bytes()


class NupFiles:
    """RomFS files of one package of a NUP container, read one at a time; the base package fills in files the chosen
    package does not carry."""

    def __init__(self, path, package):
        self.nup = en.Nup(path)
        self.zstd = en.Zstd()
        pkgs = [en.pick(self.nup, package)]
        if pkgs[0].kind != 0:
            pkgs += [p for p in self.nup.packages if p.kind == 0 and p.title_id & ~0xFFF == pkgs[0].title_id & ~0xFFF]
        self.files = {}
        for pkg in reversed(pkgs):
            self.files.update(('/'.join(parts), idx) for parts, idx in self.nup.package_files(pkg, 'romfs'))

    def read(self, path):
        if path not in self.files: raise IconError(f'{path} is not in the container')
        with tempfile.TemporaryDirectory(prefix='oot-icon-') as tmp:
            dest = Path(tmp) / 'file'
            en.extract_file(self.nup, self.zstd, self.files[path], dest, True)
            return dest.read_bytes()


def bntx_level0(data):
    """(width, height, block w, block h, block height log2, srgb, storage bytes) of the first texture's level 0."""
    def u16(o): return struct.unpack_from('<H', data, o)[0]
    def u32(o): return struct.unpack_from('<I', data, o)[0]
    def u64(o): return struct.unpack_from('<Q', data, o)[0]
    if len(data) < 0x40 or data[:4] != b'BNTX' or u16(0xC) != 0xFEFF: raise IconError('not a little-endian BNTX')
    if u32(0x24) < 1: raise IconError('the BNTX holds no texture')
    t = u64(u64(0x28))
    if t + 0x90 > len(data) or data[t:t + 4] != b'BRTI': raise IconError('no texture info (BRTI) in the BNTX')
    tile, mips, samples, fmt = u16(t + 0x12), u16(t + 0x16), u32(t + 0x18), u32(t + 0x1C)
    w, h, depth, layers, th = u32(t + 0x24), u32(t + 0x28), u32(t + 0x2C), u32(t + 0x30), u32(t + 0x34)
    if tile != 0: raise IconError(f'texture tile mode {tile}, expected block-linear (0)')
    if samples != 1 or depth != 1 or layers != 1: raise IconError('not a plain 2D texture')
    k = (fmt >> 8) - ASTC_FIRST
    if not 0 <= k < len(ASTC) or fmt & 0xFF not in (UNORM, SRGB): raise IconError(f'format {fmt:#x} is not ASTC')
    if not 0 < w <= 16384 or not 0 < h <= 16384 or th > 5: raise IconError('bad texture size')
    bw, bh = ASTC[k]
    row_bytes, rows = (w + bw - 1) // bw * 16, (h + bh - 1) // bh
    size = (row_bytes + 63) // 64 * (512 << th) * -(-rows // (8 << th))
    start = u64(u64(t + 0x70))
    end = u64(u64(t + 0x70) + 8) if mips > 1 else start + u32(t + 0x50)
    if end - start < size or end > len(data): raise IconError('the texture data is shorter than its size')
    return w, h, bw, bh, th, fmt & 0xFF == SRGB, data[start:start + size]


def png(w, h, rgba):
    def chunk(kind, body):
        return struct.pack('>I', len(body)) + kind + body + struct.pack('>I', zlib.crc32(kind + body))
    stride = w * 4
    raw = b''.join(b'\0' + rgba[y * stride:(y + 1) * stride] for y in range(h))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b''))


def shrink(w, h, rgba, size):
    """The picture as size x size RGBA, each pixel the mean of the source area it covers (alpha-weighted colour)."""
    out = bytearray(size * size * 4)
    for y in range(size):
        y0, y1 = y * h // size, max((y + 1) * h // size, y * h // size + 1)
        for x in range(size):
            x0, x1 = x * w // size, max((x + 1) * w // size, x * w // size + 1)
            r = g = b = a = 0
            for yy in range(y0, y1):
                row = rgba[(yy * w + x0) * 4:(yy * w + x1) * 4]
                for i in range(0, len(row), 4):
                    al = row[i + 3]
                    r += row[i] * al; g += row[i + 1] * al; b += row[i + 2] * al; a += al
            n = (y1 - y0) * (x1 - x0)
            o = (y * size + x) * 4
            if a:
                out[o:o + 4] = bytes((r // a, g // a, b // a, a // n))
    return bytes(out)


def ico(w, h, rgba, size=256):
    """A Windows icon file holding the picture as one PNG."""
    body = png(size, size, shrink(w, h, rgba, size))
    return struct.pack('<HHH', 0, 1, 1) + struct.pack('<BBBBHHII', 0, 0, 0, 0, 1, 32, len(body), 22) + body


def helper():
    """build/setup/icon/oot_icon, rebuilt when a source, a header or this file is newer."""
    exe = BUILD / ('oot_icon.exe' if os.name == 'nt' else 'oot_icon')
    deps = SOURCES + HEADERS + [Path(__file__)]
    if exe.is_file() and exe.stat().st_mtime >= max(p.stat().st_mtime for p in deps):
        return exe
    cc = next((c for c in (os.environ.get('CC'), 'clang') if c and shutil.which(c)), None)
    if cc is None: raise IconError('clang not found: the icon helper is built with clang (or set CC)')
    BUILD.mkdir(parents=True, exist_ok=True)
    stub, overlay = BUILD / 'vk_stub.h', BUILD / 'vfs.json'
    stub.write_text(VK_STUB)
    # a file-system overlay puts the stub where runtime/tex.h looks for vk.h; the runtime folder is not touched
    overlay.write_text(json.dumps({'version': 0, 'case-sensitive': 'false', 'roots': [
        {'name': RUNTIME.as_posix(), 'type': 'directory',
         'contents': [{'name': 'vk.h', 'type': 'file', 'external-contents': stub.as_posix()}]}]}))
    tmp = exe.with_name(exe.stem + '.tmp' + exe.suffix)
    cmd = [cc, '-O2', '-fno-strict-aliasing', '-w', '-ivfsoverlay', str(overlay), '-I', str(RUNTIME),
           *map(str, SOURCES), '-o', str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        first = (r.stderr.strip().splitlines() or ['no output'])[0]
        raise IconError(f'building the icon helper failed: {first}')
    os.replace(tmp, exe)
    return exe


def decode(w, h, bw, bh, th, srgb, storage):
    exe = helper()
    with tempfile.TemporaryDirectory(prefix='oot-icon-') as tmp:
        src, dst = Path(tmp) / 'level0.bin', Path(tmp) / 'rgba.bin'
        src.write_bytes(storage)
        r = subprocess.run([str(exe), str(src), str(dst), str(w), str(h), str(bw), str(bh), str(th),
                            '1' if srgb else '0'], capture_output=True, text=True)
        if r.returncode: raise IconError((r.stderr.strip().splitlines() or ['the icon helper failed'])[0])
        rgba = dst.read_bytes()
    if len(rgba) != w * h * 4: raise IconError('the icon helper wrote the wrong size')
    return rgba


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--romfs', type=Path, help='folder with romfs.bin and index.json (tools/pack_romfs.py)')
    src.add_argument('--dir', type=Path, help='extracted RomFS folder')
    src.add_argument('--nup', type=Path, help='the game\'s .nup container')
    ap.add_argument('--package', default='update', help='with --nup: update (default), base or an index')
    ap.add_argument('--out', type=Path, required=True, help='PNG to write')
    ap.add_argument('--ico', type=Path, help='also write the picture as a 256x256 Windows icon')
    args = ap.parse_args(argv)
    try:
        try:
            import sarc
        except ImportError:
            raise IconError('needs Python 3.14 (compression.zstd)')
        if args.romfs:
            from romfs_read import Romfs
            if not (args.romfs / 'index.json').is_file(): raise IconError(f'{args.romfs}: no index.json')
            reader = Romfs(args.romfs)
            if ICON not in reader.files: raise IconError(f'{ICON} is not in {args.romfs}')
        elif args.dir:
            reader = Folder(args.dir)
        else:
            reader = NupFiles(args.nup, args.package)
        sarc.use_romfs(reader)
        try:
            bntx, _ = sarc.decompress(reader.read(ICON))
        except (ValueError, sarc.zstd.ZstdError) as e:
            raise IconError(f'{ICON}: {e}')
        try:
            w, h, *rest = bntx_level0(bntx)
        except struct.error:
            raise IconError('the BNTX is truncated')
        rgba = decode(w, h, *rest)
        data = png(w, h, rgba)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(data)
        if args.ico:
            args.ico.write_bytes(ico(w, h, rgba))
    except (IconError, en.NupError, OSError, KeyError, ValueError, struct.error) as e:
        print(f'game_icon: {e}', file=sys.stderr)
        return 1
    print(f'{args.out} {w}x{h} ({len(data)} bytes)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
