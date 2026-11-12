#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build build/zbic/zbic_dec, the ZBIC segment decoder of tools/nso_plain.py, from pinned, read-only upstream sources.

    python3 tools/build_zbic.py [--cc clang]
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "https://github.com/hexkyz/zstd.git"
COMMIT = "02a8f10c6f4acfcfba2e31e034fbae9f07938f96"
SRC = ROOT / "refs" / "zbic-zstd"
MAIN = ROOT / "tools" / "zbic" / "zbic_dec.c"
OUT = ROOT / "build" / "zbic" / ("zbic_dec.exe" if sys.platform == "win32" else "zbic_dec")


def git(*args, cwd=None):
    return subprocess.run(["git", *map(str, args)], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def source():
    if not SRC.exists():
        # Fetched beside the final folder and moved into place only when complete.
        partial = SRC.with_name(SRC.name + ".partial")
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        git("init", "-q", cwd=partial)
        git("remote", "add", "origin", URL, cwd=partial)
        git("sparse-checkout", "set", "lib", cwd=partial)
        git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT, cwd=partial)
        git("checkout", "-q", "--detach", COMMIT, cwd=partial)
        partial.rename(SRC)
    if git("rev-parse", "HEAD", cwd=SRC) != COMMIT or git("status", "--porcelain", cwd=SRC):
        raise SystemExit(f"{SRC} must be clean at {COMMIT}; it was left unchanged")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cc", help="C compiler (default: clang, else cc, from PATH)")
    args = parser.parse_args()
    cc = args.cc or shutil.which("clang") or shutil.which("cc")
    if not cc:
        parser.error("no C compiler found: pass --cc or put clang on PATH")
    source()
    lib = SRC / "lib"
    sources = sorted((lib / "common").glob("*.c")) + sorted((lib / "decompress").glob("*.c"))
    inputs = [MAIN, Path(__file__), lib / "zstd.h", *sources, *(lib / "common").glob("*.h"),
              *(lib / "decompress").glob("*.h")]
    if OUT.exists() and OUT.stat().st_mtime > max(p.stat().st_mtime for p in inputs):
        print(f"{OUT.relative_to(ROOT)} is up to date")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # ZSTD_ZBIC_SUPPORT switches zstd.h to the ZBIC frame magic and the FSE table reader to BIC coding; the
    # assembly Huffman loop is left out so a plain compiler command builds it on every platform.
    subprocess.run([cc, "-O2", "-DZSTD_ZBIC_SUPPORT=1", "-DZSTD_DISABLE_ASM=1", f"-I{lib}", f"-I{lib / 'common'}",
                    MAIN, *sources, "-o", OUT], check=True)
    print(f"built {OUT.relative_to(ROOT)} with {cc} from {URL} @ {COMMIT[:7]}")


if __name__ == "__main__":
    main()
