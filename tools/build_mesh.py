#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the standalone task/mesh translator from pinned, read-only upstream sources."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {
    "lichii-mesh": ("https://github.com/zortos293/Lichii.git", "9b1a32dbb160d2d55ad1409961e87ec4a1256711"),
    "mesh-fmt": ("https://github.com/fmtlib/fmt.git", "407c905e45ad75fc29bf0f9bb7c5c2fd3475976f"),
    "mesh-sirit": ("https://github.com/eden-emulator/sirit.git", "b870b062998244231a4f08004d3b25151732c5c5"),
    "mesh-spirv": ("https://github.com/KhronosGroup/SPIRV-Headers.git", "04f10f650d514df88b76d25e83db360142c7b174"),
}
BOOST_SHA256 = "9de758db755e8330a01d995b0a24d09798048400ac25c03fc5ea9be364b13c93"


def run(*args):
    subprocess.run([str(arg) for arg in args], cwd=ROOT, check=True)


def source(name, url, commit):
    directory = ROOT / "refs" / name
    if not directory.exists():
        # The same bytes on every system: the patches of tools/mesh are made against them.
        run("git", "clone", "-c", "core.autocrlf=false", "--filter=blob:none", "--no-checkout", url, directory)
        if name == "lichii-mesh":
            run("git", "-C", directory, "sparse-checkout", "set", "src/common", "src/shader_recompiler")
        run("git", "-C", directory, "fetch", "--depth", "1", "origin", commit)
        run("git", "-C", directory, "checkout", "--detach", commit)
    current = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(directory), "status", "--porcelain"], text=True).strip()
    if current != commit or dirty:
        raise RuntimeError(f"{directory} must be clean at {commit}; existing reference sources were left unchanged")


def boost():
    cache = ROOT / "build/mesh-deps"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "boost_1_89_0.tar.gz"
    if not archive.exists():
        temporary = archive.with_suffix(".download")
        with urllib.request.urlopen("https://archives.boost.io/release/1.89.0/source/boost_1_89_0.tar.gz") as src:
            with temporary.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        temporary.replace(archive)
    with archive.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != BOOST_SHA256:
        raise RuntimeError("Boost archive hash mismatch")
    headers = cache / "boost_1_89_0"
    if not (headers / "boost/version.hpp").exists() or not (headers / "LICENSE_1_0.txt").exists():
        with tarfile.open(archive) as tar:
            selected = [entry for entry in tar if entry.name.startswith("boost_1_89_0/boost/") or
                        entry.name == "boost_1_89_0/LICENSE_1_0.txt"]
            for entry in selected:
                if not (cache / entry.name).resolve().is_relative_to(headers.resolve()):
                    raise RuntimeError("Boost archive path escapes its cache")
            tar.extractall(cache, members=selected, filter="data")
    return headers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolchain", type=Path,
                        help="Windows: the folder tools/win_deps.py fills (llvm-mingw/bin or llvm/bin, and bin/ninja)")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--out", type=Path, default=ROOT / "build/mesh", help="build directory (default build/mesh)")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    for name, (url, commit) in SOURCES.items():
        source(name, url, commit)
    boost_headers = boost()
    llvm = args.toolchain / "llvm-mingw" if args.toolchain and (args.toolchain / "llvm-mingw").is_dir() else \
        args.toolchain / "llvm" if args.toolchain else None
    clang = llvm / "bin/clang++.exe" if llvm else shutil.which("clang++")
    ninja = args.toolchain / "bin/ninja.exe" if args.toolchain else shutil.which("ninja")
    if not clang or not ninja:
        parser.error("provide --toolchain or put clang++ and ninja on PATH")
    out = args.out.resolve()
    run("cmake", "-S", ROOT / "tools/mesh", "-B", out, "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_CXX_COMPILER={clang}", f"-DCMAKE_MAKE_PROGRAM={ninja}",
        f"-DBOOST_HEADERS={boost_headers}", f"-DSPIRV_HEADERS={ROOT / 'refs/mesh-spirv/include'}",
        # Windows: the C++ runtime inside the program, so no library has to sit beside it.
        *(["-DCMAKE_EXE_LINKER_FLAGS=-static"] if args.toolchain else []))
    run("cmake", "--build", out, "-j", args.jobs)


if __name__ == "__main__":
    main()
