#!/usr/bin/env python3
"""Fetch the Windows build tools into toolchain/: what tools/build_game.py --toolchain and --deps expect.

    python tools/win_deps.py [--dest DIR]        (default: toolchain/ of this checkout)

Nothing is installed: every file is a pinned download (sha256 checked, kept in <dest>/dl/), unpacked into
    <dest>/llvm-mingw/    clang, lld and the mingw-w64 UCRT runtime (x86_64)
    <dest>/deps-mingw/    include/, lib/ and bin/ of SDL3, Opus, libvpx, dav1d, zstd, the Vulkan headers and loader
                          import library, and VulkanMemoryAllocator
    <dest>/bin/ninja.exe
The library packages are MSYS2's UCRT64 builds, taken as archives. Needs Python 3.14 (tarfile reads .tar.zst).
Runs on any OS, so the folder can be prepared on another machine and copied.
"""
import argparse, hashlib, shutil, sys, tarfile, urllib.request, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MSYS2 = 'https://repo.msys2.org/mingw/ucrt64/mingw-w64-ucrt-x86_64-'
SDL = '3.4.18'
LLVM = '20261020'
# name: (url, sha256)
PINS = {
    'llvm-mingw': (f'https://github.com/mstorsjo/llvm-mingw/releases/download/{LLVM}/llvm-mingw-{LLVM}-ucrt-x86_64.zip',
                   'fd5462b7f5d93b2ff981fcd3f9c7e427e47f3203c64ea91c27bcc444eebd74cd'),
    'sdl3-dll': (f'https://github.com/libsdl-org/SDL/releases/download/release-{SDL}/SDL3-{SDL}-win32-x64.zip',
                 '75c2c0fc74e7d1206aaedc22893a35887332f4f85f5c8b74fddc10d177985f2c'),
    'sdl3-devel': (f'https://github.com/libsdl-org/SDL/releases/download/release-{SDL}/SDL3-devel-{SDL}-mingw.tar.gz',
                   '049b9dd711e3fb3f8d84feb998dfec54a75e17cf539c4a80ff4d058d839771cf'),
    'ninja': ('https://github.com/ninja-build/ninja/releases/download/v1.13.2/ninja-win.zip',
              '07fc8261b42b20e71d1720b39068c2e14ffcee6396b76fb7a795fb460b78dc65'),
    'opus': (MSYS2 + 'opus-1.6.1-2-any.pkg.tar.zst', 'bb114bcf26225f50f874497ebfd6b8224fff2a20bda3f8ee45cd07ce5f02657b'),
    'libvpx': (MSYS2 + 'libvpx-1.17.0-1-any.pkg.tar.zst', '8ff6919dbb88ea26ce28d22fd81d6f0d6e200cf56a4dd81ffc2d33124fe98024'),
    'dav1d': (MSYS2 + 'dav1d-1.5.4-1-any.pkg.tar.zst', 'ccf26c6f29fd36f5692d57fd5e0fe3becec492e0b875fd5e04f47a9582eb8b37'),
    'vulkan-headers': (MSYS2 + 'vulkan-headers-1~1.4.363.0-1-any.pkg.tar.zst',
                       'be602b2657cace531316e78b1eda4087e8f7f84ebb877f08fba198a2db06406a'),
    'vulkan-loader': (MSYS2 + 'vulkan-loader-1~1.4.363.0-1-any.pkg.tar.zst',
                      '065076fe13d39ab97eea4740596426d5ae741b5044e48e1c9495d04835c92dbe'),
    'vma': (MSYS2 + 'vulkan-memory-allocator-3.4.0-1-any.pkg.tar.zst',
            '44c81162337915afecda3185f66fd98268a54454580b2717080c9dfef57fe781'),
    'zstd': (MSYS2 + 'zstd-1.5.7-2-any.pkg.tar.zst', 'dbdb8427280046a2b41697780aa4c52983b708082b0da4755951dc3bea96ca89'),
}
# package: (include entries, lib files, bin files) below its ucrt64/ folder
PARTS = {
    'opus': (['opus'], ['libopus.dll.a'], ['libopus-0.dll']),
    'libvpx': (['vpx'], ['libvpx.a'], []),
    'dav1d': (['dav1d'], ['libdav1d.a'], []),
    'vulkan-headers': (['vulkan', 'vk_video'], [], []),
    'vulkan-loader': ([], ['libvulkan-1.dll.a'], []),
    'vma': (['vma'], [], []),
    'zstd': (['zstd.h', 'zstd_errors.h'], ['libzstd.a'], []),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


STEP = [0]


def fetch(dl, name):
    url, want = PINS[name]
    path = dl / url.rsplit('/', 1)[1].replace('~', '_')
    STEP[0] += 1
    print(f'win_deps: {STEP[0]}/{len(PINS)} {name}' + ('' if path.exists() else f': downloading {url}'), flush=True)
    if not path.exists():
        tmp = path.with_suffix(path.suffix + '.part')
        req = urllib.request.Request(url, headers={'User-Agent': 'win_deps'})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, 'wb') as f: shutil.copyfileobj(r, f, 1 << 20)
        tmp.replace(path)
    ok = path.with_suffix(path.suffix + '.ok')   # checked once: a second run only looks at the marker
    if ok.exists() and ok.read_text() == want: return path
    got = sha256(path)
    if got != want:
        path.unlink()
        sys.exit(f'win_deps: {path.name}: sha256 {got}, pinned {want}; deleted, run again')
    ok.write_text(want)
    return path


def unpack(archive, work):
    out = work / archive.name
    if (out / '.done').exists(): return out
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    if archive.suffix == '.zip':
        with zipfile.ZipFile(archive) as z: z.extractall(out)
    else:
        with tarfile.open(archive) as t: t.extractall(out, filter='data')
    (out / '.done').touch()
    return out


def copy(src, dst):
    if src.is_dir(): shutil.copytree(src, dst, dirs_exist_ok=True)
    else: dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dest', type=Path, default=ROOT / 'toolchain')
    a = ap.parse_args()
    dest = a.dest.resolve(); dl = dest / 'dl'; work = dest / 'dl' / 'unpacked'
    dl.mkdir(parents=True, exist_ok=True)
    llvm = dest / 'llvm-mingw'
    archive = fetch(dl, 'llvm-mingw')
    if not (llvm / 'bin' / 'clang.exe').exists():
        print('win_deps: unpacking llvm-mingw', flush=True)
        src = unpack(archive, work) / f'llvm-mingw-{LLVM}-ucrt-x86_64'
        if llvm.exists(): shutil.rmtree(llvm)
        shutil.move(str(src), str(llvm))
        shutil.rmtree(work / archive.name, ignore_errors=True)
    deps = dest / 'deps-mingw'
    done = deps / '.done'
    if done.exists() and done.read_text() == repr(PINS) and (dest / 'bin' / 'ninja.exe').exists():
        print(f'win_deps: up to date: {llvm}, {deps}')
        return
    for sub in ('include', 'lib', 'bin'): (deps / sub).mkdir(parents=True, exist_ok=True)
    sdl = unpack(fetch(dl, 'sdl3-devel'), work) / f'SDL3-{SDL}' / 'x86_64-w64-mingw32'
    copy(sdl / 'include', deps / 'include')
    copy(sdl / 'lib' / 'libSDL3.dll.a', deps / 'lib' / 'libSDL3.dll.a')
    copy(unpack(fetch(dl, 'sdl3-dll'), work) / 'SDL3.dll', deps / 'bin' / 'SDL3.dll')
    for pkg, (incs, libs, bins) in PARTS.items():
        root = unpack(fetch(dl, pkg), work) / 'ucrt64'
        for n in incs: copy(root / 'include' / n, deps / 'include' / n)
        for n in libs: copy(root / 'lib' / n, deps / 'lib' / n)
        for n in bins: copy(root / 'bin' / n, deps / 'bin' / n)
    copy(unpack(fetch(dl, 'ninja'), work) / 'ninja.exe', dest / 'bin' / 'ninja.exe')
    done.write_text(repr(PINS))
    print(f'win_deps: {llvm}\nwin_deps: {deps}\nwin_deps: {dest / "bin" / "ninja.exe"}')


if __name__ == '__main__':
    main()
