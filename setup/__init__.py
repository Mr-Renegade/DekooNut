# SPDX-License-Identifier: GPL-3.0-or-later
"""DekooNut setup: builds a playable folder from the player's own game files.

    ./setup.sh                                         the window
    ./setup.sh --game FILE.nup --dest DIR [--quality LEVEL]
    ./setup.sh --dest DIR                              resume, or rebuild after an update of this repository
    ./setup.sh --check                                 only say what this PC lacks

Modules: cli (the command line and the terminal output), events (progress as text or JSON lines, the log file),
host (what the PC has: CPU, GPU, disk, build tools), fetch (the tools setup gets without root), steps (the build
steps), runner (which steps run: resume and update are one rule), state (oot-install.json), quality (the levels),
gui/ (the window, a front end over `--json`). Standard library only.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = (ROOT / 'VERSION').read_text(encoding='utf-8').strip() if (ROOT / 'VERSION').is_file() else '0'
