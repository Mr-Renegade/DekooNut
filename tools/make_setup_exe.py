#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pack setup as one file for Windows: build/release/DekooNut Setup.exe.

    python3.14 tools/make_setup_exe.py --cross DIR      on Linux; DIR = a Linux-hosted llvm-mingw (x86_64 UCRT)
    python tools\\make_setup_exe.py                      on Windows, with the toolchain setup fetched

The file is setup/win/launcher.c with a zip at its end: this checkout's tracked files, the setup window already
built (setup/gui) and SDL3.dll. Started, it unpacks the zip into DekooNut\\src beside itself and runs setup.bat
there, which fetches everything else. It holds nothing of the game.

On Linux the Windows libraries come from tools/win_deps.py (into build/release/win-deps; needs Python 3.14).
"""
import argparse
import hashlib
import io
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'build' / 'release'
NAME = 'DekooNut Setup.exe'
PRIVATE = ('docs/AUDIT.md', 'docs/audit/', 'docs/DEKOONUT-LOG.md')   # working notes, not part of a release
IMGUI = ROOT / 'runtime' / 'third_party' / 'imgui'
GUI = ROOT / 'setup' / 'gui'
WINDOW = [GUI / 'main.cpp', GUI / 'imgui' / 'imgui_impl_sdl3.cpp', GUI / 'imgui' / 'imgui_impl_sdlrenderer3.cpp',
          *(IMGUI / f'{n}.cpp' for n in ('imgui', 'imgui_draw', 'imgui_tables', 'imgui_widgets'))]


def run(*cmd):
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--cross', type=Path, help='Linux: llvm-mingw directory with bin/x86_64-w64-mingw32-clang')
    args = ap.parse_args()
    if os.name == 'nt':
        tc = ROOT / 'toolchain'
        cc, cxx, windres = (tc / 'llvm-mingw' / 'bin' / n for n in ('clang.exe', 'clang++.exe', 'windres.exe'))
        deps = tc / 'deps-mingw'
    else:
        if not args.cross:
            sys.exit('--cross DIR is needed on Linux')
        cc, cxx, windres = (args.cross / 'bin' / f'x86_64-w64-mingw32-{n}' for n in ('clang', 'clang++', 'windres'))
        deps = OUT / 'win-deps' / 'deps-mingw'
        if not (deps / 'include' / 'SDL3' / 'SDL.h').is_file():
            run(sys.executable, ROOT / 'tools' / 'win_deps.py', '--dest', OUT / 'win-deps')
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        sys.exit('commit first: the file is made from HEAD')
    work = OUT / 'exe'
    work.mkdir(parents=True, exist_ok=True)

    window = work / 'dekoonut-setup.exe'
    # No path of the building PC in the programs (asserts carry their file's name).
    maps = [f'-ffile-prefix-map={ROOT}=.', f'-ffile-prefix-map={cc.parent.parent}=llvm-mingw']
    run(cxx, '-std=c++17', '-O2', *maps, '-I', IMGUI, '-I', GUI, '-I', deps / 'include', *WINDOW, '-o', window,
        deps / 'lib' / 'libSDL3.dll.a', '-static', '-mwindows')
    res = work / 'launcher.res'
    run(windres, ROOT / 'setup' / 'win' / 'launcher.rc', '-O', 'coff', '-o', res)
    stub = work / 'launcher.exe'
    run(cc, '-O2', *maps, '-municode', '-mwindows', ROOT / 'setup' / 'win' / 'launcher.c', res, '-o', stub, '-static')

    # The checkout's files, then the window's program last (see launcher.c on its time).
    src = subprocess.check_output(['git', '-C', str(ROOT), 'archive', '--format=tar', 'HEAD'])
    out = OUT / NAME
    out.write_bytes(stub.read_bytes())
    # Mode 'a' on a file that is no zip appends one whose offsets count from the file's start: every zip reader takes it.
    with zipfile.ZipFile(out, 'a', zipfile.ZIP_DEFLATED, compresslevel=9) as z, tarfile.open(fileobj=io.BytesIO(src)) as t:
        for member in t:
            if member.isfile() and not member.name.startswith(PRIVATE):
                z.writestr(member.name, t.extractfile(member).read())
        z.write(deps / 'bin' / 'SDL3.dll', 'build/setup/gui/SDL3.dll')
        z.write(window, 'build/setup/gui/dekoonut-setup.exe')
    print(f'{out}: {out.stat().st_size / 1e6:.1f} MB, sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
