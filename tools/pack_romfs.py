"""Pack a RomFS into the flat image and index.json the runtime reads (runtime/romfs.c).

The runtime needs one image file and the offset and size of each file in it (tools/romfs_index.py), so the files are
concatenated in path order, each aligned to 16 bytes. Dot files the dump tool leaves (".listing") are skipped. The
output is game data: it goes under build/.

    python3 tools/pack_romfs.py --dir <dump>/romfs --out build/romfs-oot111
    python3.14 tools/pack_romfs.py --nup <game>.nup --out build/romfs-oot111

--dir takes an extracted RomFS folder. --nup reads the container itself: the base package's RomFS with the newest
update's files laid over it, the same files `tools/extract_nup.py --overlay-update` writes, without the copy on disk;
each file is checked against the SHA-256 in the container. Both give the same image.

Standard library only (--nup: what tools/extract_nup.py needs for zstd).
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


def dir_files(root):
    """[(RomFS path, size, chunks())] of a loose folder."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in filenames:
            if not name.startswith('.'):
                p = Path(dirpath) / name
                files.append((p.relative_to(root).as_posix(), p))

    def chunks(p):
        with open(p, 'rb') as src:
            while chunk := src.read(CHUNK):
                yield chunk

    return [(rel, p.stat().st_size, lambda p=p: chunks(p)) for rel, p in files]


def nup_files(path):
    """The same list from a container: base, then the newest update over it."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import extract_nup as nx
    nup, zstd = nx.Nup(path), nx.Zstd()
    chosen = {}
    for which in ('base', 'update'):
        try:
            pkg = nx.pick(nup, which)
        except nx.NupError:
            continue
        for parts, idx in nup.package_files(pkg, 'romfs'):
            if not parts[-1].startswith('.'):
                chosen['/'.join(parts)] = idx
    if not chosen:
        raise SystemExit(f'{path}: no RomFS in the container')

    def chunks(idx):
        name, size, ext_off, ext_size, sha, codec = nup.files[idx]
        h, got = hashlib.sha256(), 0
        if size:
            nup.f.seek(ext_off)
            if codec == nx.CODEC_RAW:
                left = size
                while left:
                    chunk = nup.f.read(min(CHUNK, left))
                    if not chunk:
                        raise SystemExit(f'{name}: the container ends inside the file')
                    left -= len(chunk)
                    h.update(chunk)
                    got += len(chunk)
                    yield chunk
            else:
                for chunk in zstd.stream(nup.f, ext_size):
                    h.update(chunk)
                    got += len(chunk)
                    yield chunk
        if got != size or h.digest() != sha:
            raise SystemExit(f'{name}: does not match the container record (a damaged file: copy it again)')

    return [(rel, nup.files[idx][1], lambda idx=idx: chunks(idx)) for rel, idx in chosen.items()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--dir', type=Path, help='extracted RomFS root')
    src.add_argument('--nup', type=Path, help='NUP container: base with the newest update over it')
    ap.add_argument('--out', type=Path, required=True, help='folder for romfs.bin and index.json')
    args = ap.parse_args()
    files = dir_files(args.dir.resolve()) if args.dir else nup_files(args.nup)
    files.sort(key=lambda e: e[0].encode())
    total = sum(size for _, size, _ in files)
    args.out.mkdir(parents=True, exist_ok=True)
    image = args.out / 'romfs.bin'
    (args.out / 'index.json').unlink(missing_ok=True)   # no index beside an unfinished image
    sha = hashlib.sha256()
    entries, off, t0, told = [], 0, time.time(), 0.0
    with open(image, 'wb') as dst:
        for i, (rel, want, chunks) in enumerate(files):
            pad = -off % ALIGN
            if pad:
                dst.write(b'\0' * pad)
                sha.update(b'\0' * pad)
                off += pad
            size = 0
            for chunk in chunks():
                dst.write(chunk)
                sha.update(chunk)
                size += len(chunk)
            if size != want:
                raise SystemExit(f'{rel}: {size} bytes read, {want} expected (the source changed?)')
            entries.append({'path': rel, 'offset': off, 'size': size})   # no leading '/'
            off += size
            if time.time() - told >= 1:
                told = time.time()
                print(f'{i + 1}/{len(files)} files, {off / 2**30:.2f} of {total / 2**30:.2f} GiB', file=sys.stderr,
                      flush=True)
    # 'pass' is what tools/romfs_index.py requires: the image was written whole and its SHA-256 is recorded.
    source = ('packed from a NUP container; each file checked against its SHA-256 there' if args.nup else
              'packed from a loose RomFS folder; no NCA hash tree checked')
    manifest = {'status': 'pass', 'source': source,
                'romfs_size': off, 'romfs_sha256': sha.hexdigest(), 'files': entries}
    (args.out / 'index.json').write_text(json.dumps(manifest), encoding='utf-8')
    print(f'{len(entries)} files, {off / 2**30:.2f} GiB, {time.time() - t0:.0f} s -> {image}')


if __name__ == '__main__':
    main()
