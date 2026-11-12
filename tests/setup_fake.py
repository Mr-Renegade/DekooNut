# SPDX-License-Identifier: GPL-3.0-or-later
"""A stand-in for `python3 -m setup` that plays a build without building: for the window's tests and screenshots.

    python3 tests/setup_fake.py [--scenario ok|missing|fail] [--speed N] --info | --json --game X --dest D

It prints the events of setup/events.py with made-up amounts. `ok` also sends the repository's picture as the
game's icon. Nothing is read or written.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from setup.events import Reporter   # noqa: E402
from setup.steps import all_steps   # noqa: E402

SECONDS = {'check': 0.6, 'tools': 1.5, 'icon': 0.8, 'exefs': 0.8, 'romfs': 4, 'generate': 3, 'translator': 2.5,
           'mesh': 2, 'game': 6, 'shaders': 5, 'gamedata': 0.6, 'finish': 0.4}
TEXT = {'romfs': '{d:.1f} of 24.5 GB', 'generate': 'Writing C', 'game': 'Compiling', 'mesh': 'Compiling',
        'shaders': 'Compiling for this graphics card', 'tools': 'Downloading the .NET SDK (about 220 MB)'}
COMMAND = 'sudo apt install clang lld libdav1d-dev libopus-dev libsdl3-dev libvpx-dev libvulkan-dev libzstd-dev'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='ok')
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--info', action='store_true')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--game', default='')
    ap.add_argument('--dest', default='')
    ap.add_argument('--quality')
    ap.add_argument('--display')
    a = ap.parse_args()
    missing = a.scenario == 'missing'
    if a.info:
        print(json.dumps({
            'version': '0.1.0', 'game': str(Path.home() / 'Games' / 'my-game.nup'),
            'dest': str(Path.home() / 'Games' / 'DekooNut'), 'installed': False, 'quality': 'light',
            'suggested': 'light', 'suggested_why': 'NVIDIA GeForce RTX 3070, 8 GB of video memory, a 2560x1440 display',
            'levels': [{'id': 'default', 'name': 'Default', 'about': 'The settings the game is played and tested '
                        'with: 4K output through DLSS at 60 fps.'},
                       {'id': 'light', 'name': 'Light', 'about': 'A made-up second level, to show the choice: 1440p output, a smaller 3D picture.'}],
            'missing': ['clang', 'ld.lld', 'SDL3/SDL.h', 'opus/opus.h'] if missing else [],
            'install_command': COMMAND if missing else '', 'gpu': 'NVIDIA GeForce RTX 3070'}))
        return 0
    rep = Reporter(as_json=True)
    steps = all_steps()
    rep.start('0.1.0', a.dest, a.dest + '/.setup/logs/setup.log', steps)
    for s in steps:
        rep.step(s.id, 'started')
        if s.id == 'check' and missing:
            rep.error('This PC lacks build tools that have to be installed first.',
                      'Missing: clang, ld.lld, SDL3/SDL.h, opus/opus.h\nRun this in a terminal, then start setup '
                      'again:', COMMAND, step=s.id)
            rep.emit('end', result='failed', seconds=round(rep.elapsed(), 1))
            return 1
        if s.id == 'game' and a.scenario == 'fail':
            rep.error('The game build failed (exit code 1).', 'clang: error: linker command failed', step=s.id)
            rep.emit('end', result='failed', seconds=round(rep.elapsed(), 1))
            return 1
        total = SECONDS[s.id] / a.speed
        t0 = time.monotonic()
        n = 0
        while (dt := time.monotonic() - t0) < total:
            frac = dt / total
            if s.id in TEXT:
                rep.progress(s.id, TEXT[s.id].format(d=frac * 24.5), int(frac * 1000), 1000)
            n += 1
            rep.log(f'[{n}/{int(total * 20)}] {s.id}: line {n} of the tool output')
            time.sleep(0.05)
        if s.id == 'icon':
            rep.icon(ROOT / 'assets' / 'icon.png')
        rep.step(s.id, 'done', seconds=round(total, 1))
    rep.emit('end', result='ok', seconds=round(rep.elapsed(), 1), play=a.dest + '/dekoonut')
    return 0


if __name__ == '__main__':
    sys.exit(main())
