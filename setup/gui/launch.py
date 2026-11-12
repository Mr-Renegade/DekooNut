# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the setup window when its sources changed (a few seconds) and start it.

The window is setup/gui/main.cpp with Dear ImGui over SDL3, compiled into build/setup/gui/. Without a compiler or
SDL3 it cannot exist yet: then the terminal (or a desktop dialog) says what to install.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .. import ROOT, fetch, host

HERE = ROOT / 'setup' / 'gui'
IMGUI = ROOT / 'runtime' / 'third_party' / 'imgui'
OUT = ROOT / 'build' / 'setup' / 'gui'
EXE = OUT / ('dekoonut-setup' + host.EXE)
SOURCES = [HERE / 'main.cpp', HERE / 'imgui' / 'imgui_impl_sdl3.cpp', HERE / 'imgui' / 'imgui_impl_sdlrenderer3.cpp',
           *(IMGUI / f'{n}.cpp' for n in ('imgui', 'imgui_draw', 'imgui_tables', 'imgui_widgets'))]


def say(text, command=''):
    """To the terminal, and to a desktop dialog when setup was started without one."""
    print(text + (f'\n\n    {command}\n' if command else ''), file=sys.stderr)
    if not command and 'Missing' not in text:
        return
    if sys.stderr and sys.stderr.isatty():
        return
    body = text + (f'\n\n{command}' if command else '')
    if host.WIN:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, body, 'DekooNut setup', 0x10)
        return
    for tool in (['zenity', '--error', '--no-wrap', '--title=DekooNut setup', f'--text={body}'],
                 ['kdialog', '--title', 'DekooNut setup', '--error', body]):
        if shutil.which(tool[0]):
            subprocess.run(tool)
            return


def build(say=print):
    """The window's program, built when a source is newer. None (after saying why) when it cannot be built."""
    inputs = SOURCES + list(HERE.glob('imgui/*.h')) + list(IMGUI.glob('*.h')) + [Path(__file__)]   # its flags are here
    if EXE.is_file() and EXE.stat().st_mtime >= max(p.stat().st_mtime for p in inputs):
        return EXE
    env = host.tool_env()
    if host.WIN:
        flags = windows_tools(say)
        if flags is None:
            return None
    else:
        need = [(what, pkg) for what, pkg in host.missing_packages() if what in ('clang++', 'clang', 'SDL3/SDL.h')]
        if need:
            say('The setup window needs a compiler and SDL3 before it can open. Missing: ' +
                ', '.join(what for what, _ in need) + '.\nInstall them, then start setup again:',
                host.install_command(host.missing_packages()))
            return None
        # $ORIGIN: an SDL3 library beside the program is taken first (the one-file release puts one there).
        flags = ([], ['-lSDL3', '-lm', '-Wl,-rpath,$ORIGIN'] + (['-fuse-ld=lld'] if host.which('ld.lld', env) else []))
    OUT.mkdir(parents=True, exist_ok=True)
    print('building the setup window ...', file=sys.stderr)
    clang = host.which('clang++', host.tool_env())
    objects = []
    jobs = []
    for src in SOURCES:
        obj = OUT / (src.stem + '.o')
        objects.append(obj)
        if obj.is_file() and obj.stat().st_mtime >= max(p.stat().st_mtime for p in [src] + inputs[len(SOURCES):]):
            continue
        # -ffile-prefix-map: no path of this PC in the program (asserts carry their file's name).
        jobs.append(subprocess.Popen([clang, '-std=c++17', '-O2', f'-ffile-prefix-map={ROOT}=.', '-c', str(src), '-o',
                                      str(obj), '-I', str(IMGUI), '-I', str(HERE), *flags[0]], env=host.tool_env()))
    if any(job.wait() for job in jobs):
        say('The setup window could not be compiled; the lines above say why.')
        return None
    if subprocess.run([clang, *map(str, objects), '-o', str(EXE), *flags[1]], env=host.tool_env()).returncode:
        say('The setup window could not be linked; the lines above say why.')
        return None
    if host.WIN:
        shutil.copy2(ROOT / 'toolchain' / 'deps-mingw' / 'bin' / 'SDL3.dll', OUT / 'SDL3.dll')
    return EXE


def windows_tools(say):
    """(compile flags, link flags) on Windows, after fetching the compiler and SDL3 when they are not there yet."""
    missing = host.missing_packages()
    if missing:
        say('Setup needs these first. Missing: ' + ', '.join(what for what, _ in missing) +
            '.\nRun this in a terminal, then start setup again:', host.install_command(missing))
        return None
    if not fetch.win_ready():
        print('fetching the compiler and the libraries into toolchain\\ (once, about 400 MB) ...', file=sys.stderr)
        subprocess.run([host.python314(), str(ROOT / 'tools' / 'win_deps.py')])
        if not fetch.win_ready():
            say('The build tools could not be fetched; the lines above say why.')
            return None
    deps = ROOT / 'toolchain' / 'deps-mingw'
    # -static for the C++ runtime; SDL3 stays a DLL, so its import library is named as a file.
    return (['-I', str(deps / 'include')], [str(deps / 'lib' / 'libSDL3.dll.a'), '-static', '-mwindows'])


def main(argv):
    exe = build(say)
    if not exe:
        return 1
    os.chdir(ROOT)
    return subprocess.run([str(exe), '--root', str(ROOT), '--python', sys.executable, *argv],
                          env=host.tool_env()).returncode
