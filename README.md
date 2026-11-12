<p align="center"><img src="assets/icon.png" alt="DekooNut" width="128"></p>

<h1 align="center">DekooNut</h1>

A native PC build of *The Legend of Zelda: Ocarina of Time* for Nintendo Switch 2 (update 1.1.1). The game's own
code is recompiled to C ahead of time and linked with a runtime that stands in for the console: memory, threads,
system services, files, audio, input, and the graphics API, which is recorded and replayed on Vulkan.

Everything happens on your PC, from your own copy of the game.

> [!IMPORTANT]
> This repository and its releases **do not contain the game, any game file, keys or a built game**, and do not explain
> how to get them. **You need your own legally obtained copy of the game.** Everything the build creates (the
> generated code, the `oot` program, the shader caches, your saves) comes from your copy and stays on your PC. Never
> share those files, and never add them here: `tools/check_content.py`, which also runs as the pre-commit hook,
> refuses game files, keys and generated game code.

- The repository holds no keys and no tool that decrypts game files or gets around copy protection.
- A build is for the PC it was made on. Do not pass builds on: a build contains code generated from your game, and
  it links NVIDIA's DLSS library, whose licence does not allow it to be redistributed with this program.
- Questions about where to get the game are not answered here; such requests are closed.
- The software is provided as it is, without warranty of any kind. Use it at your own risk.

DekooNut is an unofficial project. It is not affiliated with, endorsed by or supported by Nintendo. The Legend of
Zelda and Ocarina of Time are trademarks of Nintendo.

Linux x86-64 with an NVIDIA GPU is what the build is played on: 4K output through DLSS at 60 fps. A Windows build is
new: it compiles and its set-up's first steps run there, but it has not been played through yet.

## What you need

- Your own legally obtained copy of the game with update 1.1.1, dumped from your own console: a `.nup` file, or the
  `exefs` and `romfs` folders.
- A 64-bit Linux PC with an NVIDIA graphics card (8 GB of video memory or more), 16 GB of memory, a CPU with AVX2
  and about 40 GB of free disk space. The build is tested on Ubuntu 26.04.
- An internet connection for the set-up: it fetches the build tools.

## Install and play

Download `DekooNut-Setup-x86_64.AppImage` from the [releases](https://github.com/Mr-Renegade/DekooNut/releases), make
it executable and start it (it keeps its files in `~/.local/share/dekoonut`), or use a clone:

    git clone https://github.com/Mr-Renegade/DekooNut.git
    cd DekooNut
    ./setup.sh

1. The window opens. Choose your game file, the folder to install into and a quality level; the level suggested
   for your graphics card is already chosen.
2. Press **Build the game**. If the PC lacks system packages (a compiler and the development files of a few
   libraries), the window names them and installs them when you press **Install them now**; it asks for your
   password. Everything else (Python environments, the .NET SDK, the NVIDIA DLSS SDK) it fetches into the checkout
   by itself, without touching the system.
3. Wait. The game data is copied, the game's code is recompiled and compiled, and the shaders are prepared for your
   graphics card: about 20 minutes on a fast PC, longer on a slow one. It can be stopped and goes on where it was.
4. Press **Play**. DekooNut is also in your applications menu from now on, with the game's own icon, and
   `<install folder>/dekoonut` starts it from a terminal.

The same from a terminal, without the window:

    ./setup.sh --check                                   what this PC lacks, and the level it would get
    ./setup.sh --install-packages                        install the system packages (sudo)
    ./setup.sh --game <game>.nup --dest <folder>         build
    ./setup.sh --dest <folder> --quality balanced        change the level of an install

**Quality levels.** Default is what the game is played and tested with (4K output through DLSS at 60 fps on a 16 GB
card). Balanced and Light are for 8 GB cards such as an RTX 3070: [docs/quality-levels.md](docs/quality-levels.md)
has the numbers. A level is a few lines in the install's `oot.ini`, which you can also edit by hand
(`oot.ini.example` lists every setting); many settings change in the in-game menu as well.

**Updating.** `git pull`, then `./setup.sh --dest <install folder>` (or the window): only what changed is rebuilt.
Your saves are in the install folder's `profile` and stay.

**Windows.** `setup.bat` is the same set-up for Windows; it fetches everything it needs, Python and git included.
Its first steps run there; a complete install on Windows has not been done yet, and a Windows build has no DLSS for
now.

Building by hand, step by step, and what each step writes: [docs/building.md](docs/building.md).

## More

- [docs/building.md](docs/building.md): requirements, every build step, running, the layout of the repository.
- [docs/how-it-works.md](docs/how-it-works.md): how the runtime works, by topic.
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md): third-party code, its sources and licences.

## Licence

Copyright (c) 2026 Mr-Renegade and contributors.

DekooNut as a whole is under the [GNU General Public License, version 3 or later](LICENSE): it includes GPL code.
Files adapted from MIT-licensed projects keep that licence ([LICENSE-MIT](LICENSE-MIT)) and their copyright notice at
the top of the file. Every piece of third-party code, its source, revision and licence is listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The licences grant nothing for Nintendo's game.

The icon in `assets/` is a placeholder picture made for this repository; it is not taken from the game.
