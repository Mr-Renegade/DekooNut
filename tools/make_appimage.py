#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pack setup as one file for Linux: build/release/DekooNut-Setup-x86_64.AppImage.

    python3 tools/make_appimage.py

The file holds this checkout's tracked files (as a tar), the setup window already built, and the SDL3 library the
window uses. Started, it unpacks the files into ~/.local/share/dekoonut/src (DEKOONUT_HOME changes the folder) and
runs setup.sh there: the same set-up as from a clone, without git and without compiling the window first. It holds
nothing of the game; the build tools are fetched by setup as always.

Needs: git, the window's build (clang, SDL3), and appimagetool, which is downloaded into build/release/tools.
"""
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'build' / 'release'
TOOL_URL = 'https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage'
NAME = 'DekooNut-Setup-x86_64.AppImage'
PRIVATE = ('docs/AUDIT.md', 'docs/audit', 'docs/DEKOONUT-LOG.md')   # working notes, not part of a release

APPRUN = '''#!/bin/sh
# DekooNut setup as one file: its sources and the tools it fetches live in a folder of yours.
here=$(dirname "$(readlink -f "$0")")
home=${DEKOONUT_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/dekoonut}
src=$home/src
want=$(cat "$here/payload/id")
if [ "$(cat "$src/.release-id" 2>/dev/null)" != "$want" ]; then
    mkdir -p "$src/build/setup/gui" || exit 1
    tar -xf "$here/payload/src.tar" -C "$src" || exit 1
    cp "$here/payload/dekoonut-setup" "$here/payload/libSDL3.so.0" "$src/build/setup/gui/" || exit 1
    touch "$src/build/setup/gui/dekoonut-setup"   # newer than its sources: setup does not build it again
    echo "$want" > "$src/.release-id"
fi
exec "$src/setup.sh" "$@"
'''
DESKTOP = '''[Desktop Entry]
Type=Application
Name=DekooNut Setup
Comment=Builds The Legend of Zelda: Ocarina of Time for this PC from your own copy
Exec=AppRun
Icon=dekoonut-setup
Categories=Game;
Terminal=false
'''


def run(*cmd, **kw):
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def main():
    sys.path.insert(0, str(ROOT))
    from setup.gui import launch
    window = launch.build()
    if not window:
        sys.exit('the setup window could not be built')
    sdl = None
    for line in subprocess.check_output(['ldd', str(window)], text=True).splitlines():
        if 'libSDL3.so.0' in line and '=>' in line:
            sdl = Path(line.split('=>')[1].split()[0])
    if not sdl or not sdl.is_file():
        sys.exit('libSDL3.so.0 was not found beside the window\'s program or in the system')
    head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        sys.exit('commit first: the file is made from HEAD')
    version = (ROOT / 'VERSION').read_text().strip()

    app = OUT / 'AppDir'
    shutil.rmtree(app, ignore_errors=True)
    (app / 'payload').mkdir(parents=True)
    tar = app / 'payload' / 'src.tar'
    run('git', '-C', ROOT, 'archive', '--format=tar', '-o', tar, 'HEAD')
    run('tar', '--delete', '-f', tar, *PRIVATE)
    if str(ROOT).encode() in window.read_bytes():
        sys.exit(f'{window} holds this checkout\'s path: delete build/setup/gui and run again')
    shutil.copy2(window, app / 'payload' / 'dekoonut-setup')
    shutil.copy2(sdl.resolve(), app / 'payload' / 'libSDL3.so.0')
    (app / 'payload' / 'id').write_text(f'{version}-{head}-{hashlib.sha256(tar.read_bytes()).hexdigest()[:12]}\n')
    (app / 'AppRun').write_text(APPRUN)
    (app / 'AppRun').chmod(0o755)
    (app / 'dekoonut-setup.desktop').write_text(DESKTOP)
    shutil.copy2(ROOT / 'assets' / 'icon.png', app / 'dekoonut-setup.png')
    shutil.copy2(ROOT / 'assets' / 'icon.png', app / '.DirIcon')

    tool = OUT / 'tools' / 'appimagetool-x86_64.AppImage'
    if not tool.is_file():
        tool.parent.mkdir(parents=True, exist_ok=True)
        print(f'downloading {TOOL_URL}')
        urllib.request.urlretrieve(TOOL_URL, tool)
        tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    out = OUT / NAME
    run(tool, '--appimage-extract-and-run', app, out, env=dict(os.environ, ARCH='x86_64', VERSION=version))
    print(f'{out}: {out.stat().st_size / 1e6:.1f} MB, sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
