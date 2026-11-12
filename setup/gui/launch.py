# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the setup window when its sources changed (a few seconds) and start it.

The window is setup/gui/main.cpp with Dear ImGui over SDL3, compiled into build/setup/gui/. Without a compiler or
SDL3 it cannot exist yet: then the terminal (or a desktop dialog) says what to install.
"""
import os
import shutil
import subprocess
import sys

from .. import ROOT, host

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
    if sys.stderr.isatty():
        return
    body = text + (f'\n\n{command}' if command else '')
    for tool in (['zenity', '--error', '--no-wrap', '--title=DekooNut setup', f'--text={body}'],
                 ['kdialog', '--title', 'DekooNut setup', '--error', body]):
        if shutil.which(tool[0]):
            subprocess.run(tool)
            return


def build(say=print):
    """The window's program, built when a source is newer. None (after saying why) when it cannot be built."""
    inputs = SOURCES + list(HERE.glob('imgui/*.h')) + list(IMGUI.glob('*.h'))
    if EXE.is_file() and EXE.stat().st_mtime >= max(p.stat().st_mtime for p in inputs):
        return EXE
    env = host.tool_env()
    need = [(what, pkg) for what, pkg in host.missing_packages() if what in ('clang++', 'clang', 'SDL3/SDL.h')]
    if need:
        say('The setup window needs a compiler and SDL3 before it can open. Missing: ' +
            ', '.join(what for what, _ in need) + '.\nInstall them, then start setup again:',
            host.install_command(host.missing_packages()))
        return None
    OUT.mkdir(parents=True, exist_ok=True)
    print('building the setup window ...', file=sys.stderr)
    objects = []
    jobs = []
    for src in SOURCES:
        obj = OUT / (src.stem + '.o')
        objects.append(obj)
        if obj.is_file() and obj.stat().st_mtime >= max(p.stat().st_mtime for p in [src] + inputs[len(SOURCES):]):
            continue
        jobs.append(subprocess.Popen([host.which('clang++', env), '-std=c++17', '-O2', '-c', str(src), '-o', str(obj),
                                      '-I', str(IMGUI), '-I', str(HERE), '-Wno-c99-extensions'], env=env))
    if any(job.wait() for job in jobs):
        say('The setup window could not be compiled; the lines above say why.')
        return None
    link = [host.which('clang++', env), *map(str, objects), '-o', str(EXE), '-lSDL3', '-lm']
    if host.which('ld.lld', env):
        link.append('-fuse-ld=lld')
    if subprocess.run(link, env=env).returncode:
        say('The setup window could not be linked; the lines above say why.')
        return None
    return EXE


def main(argv):
    exe = build(say)
    if not exe:
        return 1
    os.chdir(ROOT)
    return subprocess.run([str(exe), '--root', str(ROOT), '--python', sys.executable, *argv],
                          env=host.tool_env()).returncode
