# SPDX-License-Identifier: GPL-3.0-or-later
"""What this PC has: CPU, GPU, memory, disk and the build tools. Nothing here changes the PC.

System packages (a compiler, the libraries' headers) are named with the one command that installs them; the tools
setup can get without root (fetch.py) are not asked of the player.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import ROOT

GiB = 1 << 30
WIN = os.name == 'nt'
EXE = '.exe' if WIN else ''

# Programs on PATH: name -> apt package
PROGRAMS = {'clang': 'clang', 'clang++': 'clang', 'ld.lld': 'lld', 'git': 'git'}
# Headers the game and the window compile against: include -> apt package
HEADERS = {'vulkan/vulkan.h': 'libvulkan-dev', 'SDL3/SDL.h': 'libsdl3-dev', 'opus/opus.h': 'libopus-dev',
           'vpx/vpx_decoder.h': 'libvpx-dev', 'dav1d/dav1d.h': 'libdav1d-dev', 'zstd.h': 'libzstd-dev',
           'zlib.h': 'zlib1g-dev'}
STDCXX = 'g++'   # brings libstdc++'s headers and its static library, which the game links


def tool_env():
    """The environment of every tool setup runs: the fetched tools first on PATH."""
    env = dict(os.environ)
    first = [ROOT / '.venv' / 'bin', ROOT / 'toolchain' / 'dotnet', ROOT / 'toolchain' / 'uv']
    env['PATH'] = os.pathsep.join([str(p) for p in first if p.is_dir()] + [env.get('PATH', '')])
    if (ROOT / 'toolchain' / 'dotnet' / ('dotnet' + EXE)).is_file():
        env['DOTNET_ROOT'] = str(ROOT / 'toolchain' / 'dotnet')
    env['DOTNET_CLI_TELEMETRY_OPTOUT'] = '1'
    env['DOTNET_NOLOGO'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def which(name, env=None):
    return shutil.which(name, path=(env or tool_env())['PATH'])


def _has_header(header, clang, env):
    if clang:
        r = subprocess.run([clang, '-E', '-x', 'c', '-', '-o', os.devnull], input=f'#include <{header}>\n',
                           text=True, capture_output=True, env=env)
        return r.returncode == 0
    dirs = ['/usr/include', '/usr/local/include'] + [d for d in env.get('CPATH', '').split(os.pathsep) if d]
    return any((Path(d) / header).is_file() for d in dirs)


def _has_static_stdcxx(env):
    clang = which('clang++', env)
    if not clang:
        return False
    r = subprocess.run([clang, '-print-file-name=libstdc++.a'], text=True, capture_output=True, env=env)
    return os.path.isabs(r.stdout.strip())


def missing_packages():
    """[(what, apt package)] for everything that needs the system's package manager."""
    env = tool_env()
    out = [(name, pkg) for name, pkg in PROGRAMS.items() if not which(name, env)]
    clang = which('clang', env)
    out += [(header, pkg) for header, pkg in HEADERS.items() if not _has_header(header, clang, env)]
    if clang and not _has_static_stdcxx(env):
        out.append(('libstdc++ (static)', STDCXX))
    return out


def install_command(missing):
    packages = sorted({pkg for _, pkg in missing})
    if shutil.which('apt-get') or shutil.which('apt'):
        return 'sudo apt install ' + ' '.join(packages)
    return ''   # another distribution: the names of what is missing have to do


def cpu_flags():
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('flags'):
                return set(line.split(':', 1)[1].split())
    except OSError:
        pass
    return set()


def cpu_ok():
    """The build uses -march=x86-64-v3."""
    flags = cpu_flags()
    return not flags or {'avx2', 'bmi2', 'fma', 'movbe'} <= flags


def memory_bytes():
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (ValueError, OSError):
        return 0


def gpus():
    """[(name, VRAM MiB, driver)] of the NVIDIA GPUs; empty when the driver's tool is missing."""
    smi = shutil.which('nvidia-smi')
    if not smi:
        return []
    r = subprocess.run([smi, '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'],
                       text=True, capture_output=True)
    out = []
    for line in r.stdout.splitlines() if r.returncode == 0 else []:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) == 3:
            out.append((parts[0], int(parts[1]) if parts[1].isdigit() else 0, parts[2]))
    return out


def free_bytes(path):
    """Free space on the disk that holds path (which need not exist yet)."""
    p = Path(path).resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


def python314():
    """A Python with compression.zstd, for the tools that read the game's compressed files."""
    if sys.version_info >= (3, 14):
        return sys.executable
    venv = ROOT / '.venv-py314' / 'bin' / 'python'
    return str(venv) if venv.is_file() else shutil.which('python3.14')


def python312():
    """The generator's interpreter: the pinned packages of tools/requirements.txt."""
    venv = ROOT / '.venv' / ('Scripts' if WIN else 'bin') / ('python' + EXE)
    return str(venv) if venv.is_file() else None


def jobs(requested=None):
    return requested or max(1, min(os.cpu_count() or 4, 16))
