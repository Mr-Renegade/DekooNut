# SPDX-License-Identifier: GPL-3.0-or-later
"""The tools setup gets without root, into the ignored toolchain/ and .venv folders of this checkout:

    uv              toolchain/uv        makes the Python environments below
    Python 3.12     .venv               the generator's interpreter with tools/requirements.txt, plus ninja and cmake
                                        when the system has none
    Python 3.14     .venv-py314         only when the system's python3 is older
    .NET 10 SDK     toolchain/dotnet    builds the shader translator; skipped when `dotnet` already has an SDK 10
    DLSS SDK        toolchain/dlss      NVIDIA's SDK at a pinned tag: headers, the loader library, the model library
                                        (Linux; a Windows build has no DLSS yet)
    Windows only    toolchain/llvm-mingw, deps-mingw, bin: the compiler and the libraries (tools/win_deps.py)

Each is skipped when it is there. Nothing is installed outside this checkout.
"""
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from . import ROOT, host
from .runner import SetupError

UV_URL = 'https://github.com/astral-sh/uv/releases/latest/download/' + (
    'uv-x86_64-pc-windows-msvc.zip' if host.WIN else 'uv-x86_64-unknown-linux-gnu.tar.gz')
DOTNET_SCRIPT = 'https://dot.net/v1/dotnet-install.' + ('ps1' if host.WIN else 'sh')
DOTNET_CHANNEL = '10.0'
GIT_URL = 'https://github.com/git-for-windows/git/releases/download/v2.56.1.windows.1/MinGit-2.56.1-64-bit.zip'
GIT_SHA256 = '4a92850b6e917da4fd010669b614b55dccf5b0ec333a534cc4be4700dfefdafd'
DLSS_URL = 'https://github.com/NVIDIA/DLSS.git'
DLSS_TAG = 'v310.9.16'
DLSS_COMMIT = '9981c6350dc58c0ce21f024fd65e31e851c31689'
DLSS_DIR = ROOT / 'toolchain' / 'dlss' / DLSS_TAG


def download(ctx, url, dest, what):
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part')
    try:
        with urllib.request.urlopen(url, timeout=60) as src, open(part, 'wb') as out:
            total, done = int(src.headers.get('Content-Length') or 0), 0
            while block := src.read(1 << 16):
                out.write(block)
                done += len(block)
                ctx.progress(f'Downloading {what}', done, total)
    except OSError as e:
        part.unlink(missing_ok=True)
        raise SetupError(f'{what} could not be downloaded.', f'{url}\n{e}\nCheck the internet connection and '
                         'run setup again.') from None
    part.replace(dest)


def uv(ctx):
    found = host.which('uv')
    if found:
        return found
    folder = ROOT / 'toolchain' / 'uv'
    archive = folder / UV_URL.rsplit('/', 1)[1]
    download(ctx, UV_URL, archive, 'uv')
    if host.WIN:
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if Path(name).name in ('uv.exe', 'uvx.exe'):
                    (folder / Path(name).name).write_bytes(z.read(name))
    else:
        with tarfile.open(archive) as tar:
            for member in tar:
                if member.isfile() and Path(member.name).name in ('uv', 'uvx'):
                    member.name = Path(member.name).name
                    tar.extract(member, folder, filter='data')
    archive.unlink()
    if not (folder / ('uv' + host.EXE)).is_file():
        raise SetupError('The uv archive did not hold the uv program.', UV_URL)
    return str(folder / ('uv' + host.EXE))


def venv312(ctx, run):
    """.venv with the generator's packages; ninja and cmake too when the system has none."""
    need = ['-r', str(ROOT / 'tools' / 'requirements.txt')]
    extra = [name for name in ('ninja', 'cmake') if not shutil.which(name) and not
             (ROOT / 'toolchain' / 'bin' / (name + host.EXE)).is_file()]
    marker = ROOT / '.venv' / '.setup-packages'
    want = (ROOT / 'tools' / 'requirements.txt').read_text() + ' '.join(extra)
    if host.python312() and marker.is_file() and marker.read_text() == want:
        return
    tool = uv(ctx)
    ctx.progress('Python 3.12 for the generator', force=True)
    if not host.python312():
        run(ctx, [tool, 'venv', '--python', '3.12', str(ROOT / '.venv')])
    run(ctx, [tool, 'pip', 'install', '--python', host.python312(), *need, *extra])
    marker.write_text(want)


def venv314(ctx, run):
    if host.python314():
        return
    tool = uv(ctx)
    ctx.progress('Python 3.14 for the game-file tools', force=True)
    run(ctx, [tool, 'venv', '--python', '3.14', str(ROOT / '.venv-py314')])
    if not host.python314():
        raise SetupError('Python 3.14 could not be set up.', 'The log has the details.')


def has_dotnet():
    dotnet = host.which('dotnet')
    if not dotnet:
        return False
    r = subprocess.run([dotnet, '--list-sdks'], text=True, capture_output=True, env=host.tool_env())
    return any(line.startswith(DOTNET_CHANNEL.split('.')[0] + '.') for line in r.stdout.splitlines())


def dotnet(ctx, run):
    if has_dotnet():
        return
    folder = ROOT / 'toolchain' / 'dotnet'
    script = ROOT / 'toolchain' / DOTNET_SCRIPT.rsplit('/', 1)[1]
    download(ctx, DOTNET_SCRIPT, script, 'the .NET install script')
    ctx.progress('Downloading the .NET SDK (about 220 MB)', force=True)
    if host.WIN:
        run(ctx, ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script), '-Channel',
                  DOTNET_CHANNEL, '-InstallDir', str(folder), '-NoPath'])
    else:
        run(ctx, ['bash', str(script), '--channel', DOTNET_CHANNEL, '--install-dir', str(folder), '--no-path'])
    if not has_dotnet():
        raise SetupError('The .NET SDK could not be set up.', 'The log has the details.')


def dlss_ready():
    return ((DLSS_DIR / 'include' / 'nvsdk_ngx_vk.h').is_file() and
            (DLSS_DIR / 'lib' / 'Linux_x86_64' / 'libnvsdk_ngx.a').is_file() and bool(dlss_model()))


def dlss_model():
    """The model library the game loads at run time from its dlss folder."""
    found = sorted((DLSS_DIR / 'lib' / 'Linux_x86_64' / 'rel').glob('libnvidia-ngx-dlss.so.*'))
    return found[-1] if found else None


def dlss(ctx, run):
    """NVIDIA's SDK from NVIDIA's own repository, under its licence (LICENSE.txt in the folder). Never passed on."""
    if dlss_ready():
        return
    ctx.progress('Fetching the NVIDIA DLSS SDK', force=True)
    if DLSS_DIR.exists():
        shutil.rmtree(DLSS_DIR)
    DLSS_DIR.parent.mkdir(parents=True, exist_ok=True)
    git = ['git', '-C', str(DLSS_DIR)]
    run(ctx, ['git', 'clone', '--filter=blob:none', '--no-checkout', DLSS_URL, str(DLSS_DIR)])
    run(ctx, git + ['sparse-checkout', 'set', '--no-cone', '/LICENSE.txt', '/include/', '/lib/Linux_x86_64/'])
    run(ctx, git + ['checkout', '--detach', DLSS_COMMIT])
    if not dlss_ready():
        raise SetupError('The NVIDIA DLSS SDK is incomplete after the download.', str(DLSS_DIR))


def win_git(ctx):
    """git for the steps that fetch pinned sources, when the PC has none: a portable copy in toolchain/git."""
    if host.which('git'):
        return
    import hashlib
    folder = ROOT / 'toolchain' / 'git'
    archive = ROOT / 'toolchain' / 'mingit.zip'
    download(ctx, GIT_URL, archive, 'git')
    if hashlib.sha256(archive.read_bytes()).hexdigest() != GIT_SHA256:
        archive.unlink()
        raise SetupError('The git download does not match its checksum.', GIT_URL)
    with zipfile.ZipFile(archive) as z:
        z.extractall(folder)
    archive.unlink()
    if not host.which('git'):
        raise SetupError('git could not be set up.', str(folder))


def win_ready():
    tc = ROOT / 'toolchain'
    return ((tc / 'llvm-mingw' / 'bin' / 'clang.exe').is_file() and (tc / 'bin' / 'ninja.exe').is_file() and
            (tc / 'deps-mingw' / 'include' / 'SDL3' / 'SDL.h').is_file())


def win_tools(ctx, run):
    """The compiler and the libraries of a Windows build: pinned downloads, unpacked into toolchain/."""
    if win_ready():
        return
    ctx.progress('Downloading the compiler and the libraries', force=True)
    run(ctx, [host.python314(), str(ROOT / 'tools' / 'win_deps.py')])
    if not win_ready():
        raise SetupError('The Windows build tools are incomplete after the download.', str(ROOT / 'toolchain'))


def all_ready():
    extra_ok = all(host.which(name) for name in ('ninja', 'cmake'))
    system = (win_ready() and bool(host.which('git'))) if host.WIN else dlss_ready()
    return bool(host.python312() and host.python314() and has_dotnet() and system and extra_ok)


if __name__ == '__main__':
    sys.exit('run through: python3 -m setup')
