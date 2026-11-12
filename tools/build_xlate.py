"""Build the shader translator: build/xlate/oot_xlate.exe (Ryujinx 1.3.2 + patches, NativeAOT, SPIR-V for Vulkan).

Copies tools/vendor/ryujinx to build/xlate/ryujinx, applies the selected tools/xlate/patches there (tools/vendor stays
byte-identical), then runs `dotnet publish`. The default applies patches 0001-0013 and 0015-0023, preserving the fast floating
min/max/clamp operations. Patch 0015 makes runtime storage-candidate resolution available only to explicit version-1
NVN 2 compute requests; missing/version-0 requests retain the legacy pass. --defined-float-minmax also applies 0014's integer selection and numeric constant folding;
this adds shader ALU work. Needs the .NET 10 SDK; on Windows also the MSVC build tools and vswhere.exe. On Linux it builds build/xlate/oot_xlate for linux-x64, which
NativeAOT links with the system's clang (no MSVC); NativeAOT builds for the OS it runs on only.

Next to the exe it writes oot_xlate.id: the translator identity, a sha256 over everything the build reads (this script,
Program.cs, the csproj, the selected patches, the vendor tree). runtime/gfx_shader.c puts it into every shader cache key, so a
translator change re-keys the cache without a hand-made KEY_VERSION bump. --id-only writes it for the exe already built
(the caller vouches that the exe matches the sources and must use the same --defined-float-minmax setting).

  python tools/build_xlate.py [--defined-float-minmax] [--id-only] [--out DIR]

--out builds into DIR instead of build/xlate (a task build dir: oot.exe takes the translator next to it).
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "tools" / "vendor" / "ryujinx"
XLATE = ROOT / "tools" / "xlate"
BUILD = ROOT / "build" / "xlate"
WIN = os.name == "nt"
EXE = "oot_xlate.exe" if WIN else "oot_xlate"
VSWHERE_DIR = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer"
DEFINED_FLOAT_MINMAX_PATCH = "0014-numeric-float-minmax-and-saturation.patch"


def tree_hashes(base: Path) -> dict[str, str]:
    """Hash every source file; src/*/obj and src/*/bin are build output (an IDE restore writes them, the vendor
    .gitignore ignores them and prepare_sources() drops them), so they are left out."""
    def build_output(rel: Path) -> bool:
        return len(rel.parts) > 2 and rel.parts[0] == "src" and rel.parts[2] in ("obj", "bin")

    return {
        p.relative_to(base).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(base.rglob("*"))
        if p.is_file() and not build_output(p.relative_to(base))
    }


def selected_patches(*, defined_float_minmax: bool = False) -> list[Path]:
    patches = sorted((XLATE / "patches").glob("*.patch"))
    if len(patches) != 35:
        sys.exit(f"expected 35 patches, found {len(patches)}")
    optional = XLATE / "patches" / DEFINED_FLOAT_MINMAX_PATCH
    if optional not in patches:
        sys.exit(f"missing optional patch: {DEFINED_FLOAT_MINMAX_PATCH}")
    return patches if defined_float_minmax else [p for p in patches if p != optional]


def prepare_sources(*, defined_float_minmax: bool = False) -> Path:
    src = BUILD / "ryujinx"
    if BUILD.exists():
        shutil.rmtree(BUILD / "ryujinx", ignore_errors=True)
    shutil.copytree(VENDOR, src)
    for junk in list(src.glob("src/*/obj")) + list(src.glob("src/*/bin")):
        shutil.rmtree(junk)
    for patch in selected_patches(defined_float_minmax=defined_float_minmax):
        # git apply works outside a repository when given a directory; --check first so a failure names the patch.
        for extra in (["--check"], []):
            r = subprocess.run(
                ["git", "apply", *extra, "--whitespace=nowarn", "--directory", str(src.relative_to(BUILD)), str(patch)],
                cwd=BUILD, capture_output=True, text=True,
                # Inside a repository (a git worktree with a real build/) git apply skips ignored paths without a word.
                env={**os.environ, "GIT_CEILING_DIRECTORIES": str(BUILD.parent)},
            )
            if r.returncode != 0:
                sys.exit(f"patch does not apply: {patch.name}\n{r.stdout}{r.stderr}")
        print(f"applied {patch.name}")
    apply_fmz_experiment(src)
    return src


def apply_fmz_experiment(src: Path) -> None:
    """Maxwell's FMZ rule on every FP32 multiply (0 x anything = 0), emitted behind the translator's fix bit 2
    (patch 0036): plain IEEE unless that bit is cleared."""
    path = src / "src" / "Ryujinx.Graphics.Shader" / "Translation" / "EmitterContextInsts.cs"
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    changes = [
        ('''            return context.Add(fpType | Instruction.FusedMultiplyAdd, Local(), a, b, c);
''', '''            if (fpType == Instruction.FP32 && !XlateFix.On(XlateFix.NoFmz))
            {
                Operand fmzZero = context.Add(Instruction.LogicalOr, Local(), context.FPCompareEqual(a, ConstF(0.0f)), context.FPCompareEqual(b, ConstF(0.0f)));
                Operand fmzResult = context.Add(fpType | Instruction.FusedMultiplyAdd, Local(), a, b, c);
                return context.ConditionalSelect(fmzZero, c, fmzResult);
            }

            return context.Add(fpType | Instruction.FusedMultiplyAdd, Local(), a, b, c);
'''),
        ('''            return context.Add(fpType | Instruction.Multiply, Local(), a, b);
''', '''            if (fpType == Instruction.FP32 && !XlateFix.On(XlateFix.NoFmz))
            {
                Operand fmzZero = context.Add(Instruction.LogicalOr, Local(), context.FPCompareEqual(a, ConstF(0.0f)), context.FPCompareEqual(b, ConstF(0.0f)));
                Operand fmzResult = context.Add(fpType | Instruction.Multiply, Local(), a, b);
                return context.ConditionalSelect(fmzZero, ConstF(0.0f), fmzResult);
            }

            return context.Add(fpType | Instruction.Multiply, Local(), a, b);
'''),
    ]
    for old, new in changes:
        if text.count(old) != 1:
            sys.exit(f"FMZ experiment does not apply: {old.strip()}")
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8", newline="\n")
    print("applied the FMZ experiment")


def identity(*, defined_float_minmax: bool = False) -> str:
    """sha256 over the translator's inputs, in a fixed order: path and content hash of each file."""
    h = hashlib.sha256()
    files = [Path(__file__).resolve(), XLATE / "Program.cs", XLATE / "oot_xlate.csproj",
             *selected_patches(defined_float_minmax=defined_float_minmax)]
    for f in files:
        h.update(f"{f.name}\0{hashlib.sha256(f.read_bytes()).hexdigest()}\n".encode())
    for rel, digest in tree_hashes(VENDOR).items():
        h.update(f"vendor/{rel}\0{digest}\n".encode())
    return h.hexdigest()


def write_identity(exe: Path, *, defined_float_minmax: bool = False) -> None:
    ident = identity(defined_float_minmax=defined_float_minmax)
    exe.with_suffix(".id").write_text(ident + "\n", encoding="ascii")
    print(f"translator identity {ident[:16]} -> {exe.with_suffix('.id')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--id-only", action="store_true", help="write oot_xlate.id for the exe already built")
    parser.add_argument("--out", type=Path, help="build directory instead of build/xlate")
    parser.add_argument("--defined-float-minmax", action="store_true",
                        help="opt in to numeric NaN/signed-zero min/max/SAT via integer selection (more shader ALU work)")
    parser.add_argument("--llvm", type=Path, help="Linux: an LLVM directory whose bin/clang and lld link the exe")
    parser.add_argument("--sysroot", type=Path, help="Linux: link against this sysroot's glibc instead of the host's "
                                                     "(a portable build)")
    a = parser.parse_args()
    if a.out:
        global BUILD
        BUILD = a.out.resolve()
    if a.id_only:
        write_identity(BUILD / EXE, defined_float_minmax=a.defined_float_minmax)
        return
    started = time.time()
    dotnet = shutil.which("dotnet")
    if not dotnet:
        sys.exit("missing: dotnet (.NET 10 SDK)")
    env = dict(os.environ)
    if WIN:
        if not (VSWHERE_DIR / "vswhere.exe").is_file() and not shutil.which("vswhere"):
            sys.exit(f"missing: vswhere.exe (expected in {VSWHERE_DIR}); NativeAOT cannot link without it")
        env["PATH"] = str(VSWHERE_DIR) + os.pathsep + env["PATH"]

    src = prepare_sources(defined_float_minmax=a.defined_float_minmax)
    work = BUILD / "xlate"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    for name in ("Program.cs", "oot_xlate.csproj"):
        shutil.copy2(XLATE / name, work / name)
    out = BUILD / "publish"
    shutil.rmtree(out, ignore_errors=True)
    cmd = [
        dotnet, "publish", str(work / "oot_xlate.csproj"), "-c", "Release", "-r", "win-x64" if WIN else "linux-x64",
        "-p:PublishAot=true", "-p:InvariantGlobalization=true", f"-p:RyuSrc={src}", "-o", str(out),
    ]
    if a.llvm:
        env["PATH"] = str(a.llvm.resolve() / "bin") + os.pathsep + env["PATH"]
        cmd += ["-p:CppCompilerAndLinker=clang", "-p:LinkerFlavor=lld"]
    if a.sysroot:
        cmd.append(f"-p:SysRoot={a.sysroot.resolve()}")
    if subprocess.run(cmd, env=env).returncode != 0:
        sys.exit("dotnet publish failed")
    exe = BUILD / EXE
    shutil.copy2(out / EXE, exe)
    write_identity(exe, defined_float_minmax=a.defined_float_minmax)
    print(f"built {exe} ({exe.stat().st_size / 1e6:.1f} MB) in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
