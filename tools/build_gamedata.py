#!/usr/bin/env python3
"""Build a game-data mod pack from the player's own RomFS, read-only.

    python3 tools/build_gamedata.py <pack> <mod>[:<arg>] ... [--romfs DIR] [--out DIR]
    python3 tools/build_gamedata.py hs625 dynres:0.625 heap:208

Reads <romfs>/romfs.bin and its index.json (default build/romfs-oot111, written by tools/pack_romfs.py) and never
writes there. Each mod is tools/gamedata/mods/<mod>.py: TARGETS (RomFS paths) and apply(path, data, arg) -> new bytes,
applied in the order given, so two mods may change one file. Only the files that changed go to
<out>/<pack>/<RomFS path>, with <out>/<pack>/manifest.tsv (path, size, SHA-256, the original's SHA-256):
runtime/romfs.c serves them in place of the originals when OOT_ROMFS_MODS=<out>/<pack>, and refuses a pack whose
originals differ. The same mods on the same RomFS give the same bytes. Output default: build/gamedata (ignored).
Needs Python 3.14 (compression.zstd) for .zs files.
"""
import argparse, hashlib, importlib.util, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "gamedata"))
from romfs_read import Romfs   # noqa: E402


def load_mod(name):
    p = ROOT / "tools" / "gamedata" / "mods" / f"{name}.py"
    if not p.exists(): sys.exit(f"no mod {name} ({p})")
    spec = importlib.util.spec_from_file_location(f"mod_{name}", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pack"); ap.add_argument("mods", nargs="+", help="<mod>[:<arg>]")
    ap.add_argument("--romfs", type=Path, default=ROOT / "build" / "romfs-oot111")
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "gamedata")
    a = ap.parse_args()
    if not a.pack.replace("-", "").replace("_", "").isalnum(): sys.exit("pack names: letters, digits, - and _")
    r = Romfs(a.romfs)
    import sarc; sarc.use_romfs(r)   # the packs' zstd dictionaries
    files = {}   # path -> (original bytes, current bytes)
    for spec in a.mods:
        name, _, arg = spec.partition(":")
        m = load_mod(name)
        for path in m.TARGETS:
            if path not in r.files: sys.exit(f"{name}: {path} is not in this RomFS")
            orig, cur = files.get(path) or (r.read(path),) * 2
            new = m.apply(path, cur, arg)
            print(f"{name}{':' + arg if arg else ''}: {path}: {'changed' if new != cur else 'unchanged'}")
            for line in getattr(m, "LOG", []): print(f"  {line}")
            if hasattr(m, "LOG"): m.LOG.clear()
            files[path] = (orig, new)
    out = a.out / a.pack
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    rows = []
    for path in sorted(files):
        orig, new = files[path]
        if new == orig: continue
        dst = out / path; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(new)
        rows.append(f"{path}\t{len(new)}\t{hashlib.sha256(new).hexdigest()}\t{hashlib.sha256(orig).hexdigest()}")
    (out / "manifest.tsv").write_text(f"# {a.pack}: {' '.join(a.mods)} (tools/build_gamedata.py; RomFS {r.sha256})\n"
                                      + "\n".join(rows) + "\n")
    print(f"{out}: {len(rows)} file(s)")


if __name__ == "__main__":
    main()
