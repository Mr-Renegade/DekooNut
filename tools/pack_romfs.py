"""Pack an extracted RomFS directory into the flat image and index.json the runtime reads (runtime/romfs.c).

The runtime needs one image file and the offset and size of each file in it (tools/romfs_index.py), so the files are
concatenated in path order, each aligned to 16 bytes. Dot files the dump tool leaves (".listing") are skipped. The
output is game data: it goes under build/.

    python3 tools/pack_romfs.py --dir <dump>/romfs --out build/romfs-oot111

Standard library only.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ALIGN = 16
CHUNK = 8 << 20


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dir', type=Path, required=True, help='extracted RomFS root')
    ap.add_argument('--out', type=Path, required=True, help='folder for romfs.bin and index.json')
    args = ap.parse_args()
    root = args.dir.resolve()
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in filenames:
            if not name.startswith('.'):
                p = Path(dirpath) / name
                files.append((p.relative_to(root).as_posix(), p))
    files.sort(key=lambda e: e[0].encode())
    args.out.mkdir(parents=True, exist_ok=True)
    image = args.out / 'romfs.bin'
    sha = hashlib.sha256()
    entries, off, t0 = [], 0, time.time()
    with open(image, 'wb') as dst:
        for i, (rel, p) in enumerate(files):
            pad = -off % ALIGN
            if pad:
                dst.write(b'\0' * pad)
                sha.update(b'\0' * pad)
                off += pad
            size = 0
            with open(p, 'rb') as src:
                while chunk := src.read(CHUNK):
                    dst.write(chunk)
                    sha.update(chunk)
                    size += len(chunk)
            entries.append({'path': rel, 'offset': off, 'size': size})   # no leading '/', as extract_romfs.py
            off += size
            if i % 20000 == 0:
                print(f'{i}/{len(files)} files, {off / 2**30:.1f} GiB', file=sys.stderr)
    # 'pass' is what tools/romfs_index.py requires: the image was written whole and its SHA-256 is recorded. There is
    # no NCA hash tree to check the files against, which 'source' says.
    manifest = {'status': 'pass', 'source': 'packed from a loose RomFS folder; no NCA hash tree checked',
                'romfs_size': off, 'romfs_sha256': sha.hexdigest(), 'files': entries}
    (args.out / 'index.json').write_text(json.dumps(manifest), encoding='utf-8')
    print(f'{len(entries)} files, {off / 2**30:.2f} GiB, {time.time() - t0:.0f} s -> {image}')


if __name__ == '__main__':
    main()
