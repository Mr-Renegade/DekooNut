"""Convert ZBIC NSOs (flag bit 7, firmware 22.0.0+) of an ExeFS folder to plain NSOs the lifter reads.

    python3 tools/nso_plain.py <exefs_in> <exefs_out>

Every file is copied; an NSO0 with bit 7 set has its segments decoded by build/zbic/zbic_dec (tools/zbic/zbic_dec.c)
and stored as they are: flag bits 0-2 and 7 are cleared, the compressed sizes become the segment sizes, the hash bits
and hashes stay. Each decoded segment is checked against the SHA-256 in the NSO header (offset 0xA0), which is the
loader's own check, so a wrong decode cannot pass.
"""
import hashlib
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEC = ROOT / 'build' / 'zbic' / ('zbic_dec.exe' if sys.platform == 'win32' else 'zbic_dec')
HDR = 0x100
ZBIC_BIT = 1 << 7


def convert(src: Path, dst: Path) -> str:
    raw = src.read_bytes()
    if raw[:4] != b'NSO0':
        shutil.copyfile(src, dst)
        return 'copied (not an NSO)'
    flags = struct.unpack_from('<I', raw, 0xC)[0]
    if not flags & ZBIC_BIT:
        shutil.copyfile(src, dst)
        return 'copied (LZ4/stored)'
    segs = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(3):
            file_off, mem_off, size = struct.unpack_from('<III', raw, 0x10 + i * 0x10)
            comp = struct.unpack_from('<I', raw, 0x60 + i * 4)[0]
            data = raw[file_off:file_off + comp]
            if flags & (1 << i):
                a, b = Path(tmp) / f'{i}.in', Path(tmp) / f'{i}.out'
                a.write_bytes(data)
                r = subprocess.run([str(DEC), str(a), str(size), str(b)], capture_output=True, text=True)
                if r.returncode:
                    raise SystemExit(f'{src.name} segment {i}: {r.stderr.strip() or "decode failed"}')
                data = b.read_bytes()
            if flags & (1 << (3 + i)):
                want = raw[0xA0 + 32 * i:0xC0 + 32 * i]
                if hashlib.sha256(data).digest() != want:
                    raise SystemExit(f'{src.name} segment {i}: SHA-256 differs from the NSO header')
            segs.append((mem_off, data))
    out = bytearray(raw[:HDR])
    body = bytearray()
    for i, (mem_off, data) in enumerate(segs):
        struct.pack_into('<III', out, 0x10 + i * 0x10, HDR + len(body), mem_off, len(data))
        struct.pack_into('<I', out, 0x60 + i * 4, len(data))
        body += data
    struct.pack_into('<I', out, 0xC, flags & ~(0x7 | ZBIC_BIT))
    dst.write_bytes(bytes(out) + bytes(body))
    return 'ZBIC -> plain, ' + ', '.join(f'seg{i} {len(d):#x} sha256 ok' for i, (_, d) in enumerate(segs))


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if not DEC.exists():
        raise SystemExit(f'{DEC} missing: run tools/build_zbic.py first')
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.iterdir()):
        if p.is_file():
            print(f'{p.name:10} {convert(p, dst / p.name)}')


if __name__ == '__main__':
    main()
