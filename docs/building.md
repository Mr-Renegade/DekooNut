# Building DekooNut

The full steps from your own copy of the game to a running build. The short form is in the README.

## What you supply

- Your own legally obtained copy of *The Legend of Zelda: Ocarina of Time* for Nintendo Switch 2, update 1.1.1
  (title 0400486024DEC000), dumped from your own console.
- The repository holds no game data, and none may be added. `tools/check_content.py` refuses game files, keys and
  generated game code; it runs as the pre-commit hook (`git config core.hooksPath .githooks`).
- The form the tools take:
  - a NUP container (`.nup`), which `tools/extract_nup.py` unpacks into an ExeFS folder and a RomFS folder; or
  - those two folders as loose files (`main`, `main.npdm`, `rtld`, `sdk`, `subsdk0` in the ExeFS; the RomFS tree).
- The ExeFS must match `tools/data/expected-oot111.json` (SHA-256 of each file after `tools/nso_plain.py`).
  `tools/build_game.py` bakes these hashes into the game, which checks the files at start.
- Everything made from the dump stays under `build/` and `generated/`, which git ignores.

## Host requirements

- Linux x86-64 (the build uses `-march=x86-64-v3`: an AVX2 CPU).
- clang and lld (C11 and C++20), ninja, git. `llvm-ar` only for `--make-runtime-lib`.
- Libraries with headers: Vulkan, SDL3, Opus, libvpx, dav1d, zstd, and VulkanMemoryAllocator's `vk_mem_alloc.h`
  (fetched into `toolchain/vma` when the system has none). libstdc++ is linked statically.
- Python 3.12 for the generator (`recomp/`) with `tools/requirements.txt` (numpy and lz4; capstone and unicorn for the
  tests): `uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r tools/requirements.txt`.
- Python 3.14 for `tools/extract_nup.py`, `tools/prepare_oot.py` and `tools/build_gamedata.py` (they use
  `compression.zstd`; standard library only). `tools/prepare_oot.sh` looks for `.venv-py314/bin/python`, else
  `python3.14`.
- The .NET 10 SDK, for the shader translator (`tools/build_xlate.py`; NativeAOT links it with the system's clang).
- cmake, for the mesh shader compiler (`tools/build_mesh.py`). It downloads Boost 1.89.0 headers and clones its
  sources (Lichii, fmt, sirit, SPIRV-Headers) at pinned commits into `refs/`.
- A GPU with Vulkan 1.3 (the default pick is the first discrete GPU with Vulkan 1.3).
- The NVIDIA DLSS SDK (`include/nvsdk_ngx_vk.h`, `lib/Linux_x86_64/libnvsdk_ngx.a`), passed with `--dlss-sdk DIR`
  or `OOT_DLSS_SDK`. It is never part of the repository. `tools/build_game.py` stops when it is missing: a build
  without it is not supported. At run time DLSS needs the SDK's `libnvidia-ngx-dlss.so.<version>` in a `dlss` folder
  next to `oot` and an NVIDIA GPU; without them the game logs why and uses its other upscale path.
- glslc and spirv-val, only when you edit a shader under `runtime/shaders/` (`python3 tools/build_shaders.py`
  rewrites the committed headers; `tools/build_game.py` refuses to build when one no longer matches its source).
- A Linux desktop with systemd for `tools/prepare_oot.sh` (it runs under `systemd-run --user` with a memory cap).

## Setup

`./setup.sh` does everything below for an install folder of its own, in a window or from a terminal:

    ./setup.sh                                          the window
    ./setup.sh --game <game>.nup --dest <folder>        the same from a terminal (--game also takes a folder
                                                        holding exefs/ and romfs/)
    ./setup.sh --dest <folder>                          go on after a stop, or rebuild after `git pull`
    ./setup.sh --check                                  only say what this PC lacks

- It asks for the system packages it cannot get itself with one command (on Ubuntu: clang, lld, git and the
  development files of Vulkan, SDL3, Opus, libvpx, dav1d, zstd and zlib). The rest it fetches into this checkout,
  without root: uv and the two Python environments, ninja and cmake when the system has none, the .NET SDK
  (`toolchain/dotnet`) and the NVIDIA DLSS SDK at a pinned tag from NVIDIA's repository (`toolchain/dlss`, under
  NVIDIA's licence).
- The install folder holds the game as it runs (`oot`, the translator, the mesh shader compiler and its `mesh-cache/`,
  `exefs/`, `romfs/`, `dlss/`, the game-data pack of the chosen quality level, `oot.ini`, `profile/` with the saves,
  the shader caches), the launcher
  `dekoonut` and `dekoonut.png`: the game's own Deku Nut icon, read from your game data
  (`tools/game_icon.py`). The applications menu gets an entry "DekooNut" (`--no-menu-entry` leaves it out). About
  36 GB.
- Steps are recorded in `oot-install.json`; a step whose inputs did not change is skipped, so a stopped setup goes
  on where it was, and after an update of this repository only what changed is rebuilt (the generated C and the
  objects stay in the folder's `.setup/`). Logs: `.setup/logs/`.
- The window (`setup/gui/`, compiled on first start into `build/setup/gui/`) is a front end over
  `./setup.sh --json ...`, which prints one JSON event per line (`setup/events.py`).
- Quality levels: `setup/quality.py` (a game-data pack, the launcher's environment and `oot.ini` keys per level).
  Setup suggests one from the graphics card's video memory and the PC's memory; `--quality` or the window choose
  another.
- On the PC it was written on (16 threads, RTX 4080 SUPER, NVMe) a first install takes about 19 minutes: 43 s for the
  game data, 37 s to recompile, 4 minutes to compile the game, 13 minutes to prepare the shaders. A second run on
  the same folder with nothing changed takes seconds.
- Tests without game files: `python3 tests/setup_test.py` (the bookkeeping), and `tests/setup_fake.py`, a stand-in
  backend that plays a build for the window (`build/setup/gui/dekoonut-setup --root . --backend "python3
  tests/setup_fake.py"`).

## Build

### 1. Extract the dump

From a `.nup`:

    python3.14 tools/extract_nup.py <game>.nup --list
    python3.14 tools/extract_nup.py <game>.nup --package base --overlay-update --out build/extracted/oot111-dump

`--list` shows the packages in the container: `base` is the game as released, `update` is update 1.1.1. The build takes
the base package with the update laid over it (`--overlay-update`): the update's five ExeFS files, and the base RomFS
with the update's files replacing and adding to it (186,808 files).

This writes `build/extracted/oot111-dump/exefs` and `.../romfs`, each file checked against the SHA-256 in the container.

Convert the ExeFS to plain NSOs (the game's `main` is ZBIC-compressed):

    python3 tools/build_zbic.py
    python3 tools/nso_plain.py build/extracted/oot111-dump/exefs build/extracted/oot111/exefs

Every file is copied; ZBIC segments are decoded by `build/zbic/zbic_dec` and checked against the hashes in the NSO
header. `tools/build_zbic.py` builds the decoder: `tools/zbic/zbic_dec.c` with a zstd that reads ZBIC frames, fetched
at a pinned commit into `refs/zbic-zstd`.

Pack the RomFS into one image:

    python3 tools/pack_romfs.py --dir build/extracted/oot111-dump/romfs --out build/romfs-oot111

This writes `build/romfs-oot111/romfs.bin` and `index.json` (offset, size and path of each file).
`python3.14 tools/pack_romfs.py --nup <game>.nup --out build/romfs-oot111` writes the same image straight from the
container, without the extracted RomFS on disk (then extract with `--only exefs`).

### 2. Generate the C

    .venv/bin/python recomp/gen.py --exefs build/extracted/oot111/exefs --out generated/oot111 \
        --patches recomp/patches-oot111.txt

This lifts `main`, `subsdk0` and `sdk` (the default `--modules`) into `generated/oot111/`: `layout.h`, `layout.json`,
`patches.h`, and per module the C chunks, `<mod>_table.c` and `manifest.json`. Workers are sized by CPU count and free
memory (`--jobs N` to set them).

### 3. Build the shader translator

    python3 tools/build_xlate.py

This copies `tools/vendor/ryujinx` to `build/xlate/ryujinx`, applies `tools/xlate/patches`, runs `dotnet publish`
and writes `build/xlate/oot_xlate` and `build/xlate/oot_xlate.id` (its identity, part of every shader cache key).
A rebuilt translator changes every cache key: run step 6 again before the next start.

### 4. Build the mesh shader compiler

    python3 tools/build_mesh.py

This clones its pinned sources into `refs/` (`lichii-mesh`, `mesh-fmt`, `mesh-sirit`, `mesh-spirv`), downloads the
Boost headers into `build/mesh-deps`, and builds `build/mesh/oot_mesh_xlate` with cmake and ninja. The game finds it
there from `build/oot111/` (`../mesh/oot_mesh_xlate`) and compiles task and mesh shaders with it in the background,
caching them in `build/mesh-cache`. Linux only.

### 5. Build the game

    python3 tools/build_game.py --version oot111 --expected tools/data/expected-oot111.json \
        --exefs build/extracted/oot111/exefs --romfs build/romfs-oot111 --out build/oot111 \
        --profile <profile dir> --dlss-sdk <DLSS SDK dir>

This compiles `generated/oot111` and every `runtime/` source with clang through `build/oot111/build.ninja` and links
`build/oot111/oot`. It also writes `modules.c` (layout, function tables, expected hashes, the default ExeFS, RomFS and
profile paths), `romfs-index.bin` and `romfs_config.h`. A rerun recompiles only what changed. `--jobs N` sets the
parallel compiles (default: by memory and CPU count). `--toolchain DIR` takes clang and lld from DIR/bin, `--sysroot
DIR` builds against another glibc, `--deps DIR` links Opus, libvpx and zstd statically from DIR; `--frame-pointers`
keeps frame pointers in the runtime objects.

### 6. Prepare the shaders

    tools/prepare_oot.sh build/oot111

This runs `tools/prepare_oot.py` (Python 3.14) with the translator `build/xlate/oot_xlate` and the RomFS
`build/romfs-oot111` (`XLATE=` and `ROMFS=` change them), 6 threads, under a 14G memory cap (`MEMORY=`). Steps, each
skipped when its result is current: census (every Switch 2 shader stage of the RomFS into `seed-nvn2.bin`), lowering
(`oot --lower-shaders`), plan (`seed-answers.bin`), and build (`oot --prepare-shaders`: translations into
`build/oot111/shader-cache/` and pipelines into the driver's disk cache). It can be stopped and run again. Run it after
every translator rebuild.

Optional: `python3 tools/pack_shader_cache.py build/oot111/shader-cache` folds the loose cache files into
`shader-cache.pack` (the game closed).

### 7. Optional: game-data packs

    python3.14 tools/build_gamedata.py hs625 dynres:0.625 heap:208
    python3.14 tools/build_gamedata.py hs750 dynres:0.75 heap:208

Each reads `build/romfs-oot111` (never writes there) and writes the changed files with a `manifest.tsv` into
`build/gamedata/<pack>/`. `dynres:<scale>` holds the internal 3D resolution at that share of the 1706x960 game image;
`heap:<MiB>` grows the GPU texture heap by that much. The other mods: `shadowfade:<start>/<end>`, `lod:<coef>`.

## Run

    cd build/oot111
    OOT_PROFILE_DIR=<profile dir> OOT_XLATE=../xlate/oot_xlate ./oot

- `OOT_PROFILE_DIR`: saves and settings, any directory of yours. Without it: a `profile` folder next to `oot`, else the `--profile` baked in
  at build time. `OOT_EXEFS` and `OOT_ROMFS` (or `--exefs`, `--romfs`, `--profile` on the command line) do the same
  for the game files.
- `OOT_XLATE`: the translator. Without it: `oot_xlate` next to `oot`, else `build/xlate/oot_xlate` of the checkout.
- With a game-data pack: `OOT_ROMFS_MODS=../gamedata/hs625 OOT_GFX_SCALE=0` (or `hs750`). The game serves the pack's
  files in place of the originals and refuses a pack built from another RomFS. `OOT_GFX_SCALE=0` makes the game draw
  at its full display size.
- Settings: `oot.ini` next to `oot` (copy `oot.ini.example`; each line shows the default). An `OOT_<KEY>` environment
  variable overrides the key of the same name. Many settings also change in the in-game menu.
- The first start translates the shaders step 6 did not prepare, and draws a mesh shader only once
  `oot_mesh_xlate` has compiled it.

## Layout of the repository

| path | what |
|---|---|
| `recomp/` | the recompiler: lifts the game's AArch64 code to C (`gen.py`), with the patch table for this game |
| `runtime/` | the runtime linked with the generated C: kernel, services, NVN front end, Vulkan replay, menu |
| `runtime/shaders/` | the runtime's own shaders; their compiled form is committed as `runtime/*_spv.h` |
| `runtime/third_party/` | adapted third-party code that is compiled into the game |
| `tools/` | extraction, build, shader translator (`xlate/`, `vendor/`), mesh shader compiler (`mesh/`), game-data packs |
| `tests/` | tests of the tools and differential tests of the recompiler's lifter |
| `oot.ini.example` | every setting with its default |

Everything derived from the game (`build/`, `generated/`) and fetched references (`refs/`, `toolchain/`) is ignored
by git.
