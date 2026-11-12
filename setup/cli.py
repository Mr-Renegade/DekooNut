# SPDX-License-Identifier: GPL-3.0-or-later
"""The command line of setup.

    setup                                   open the window
    setup --game FILE.nup|DIR --dest DIR    build a playable folder (DIR: a folder holding exefs/ and romfs/)
    setup --dest DIR                        resume, or rebuild after this repository changed
    setup --check                           say what this PC lacks, change nothing
    setup --info                            one JSON object for the window: version, quality levels, last choices

Exit code: 0 done, 1 failed, 2 bad command line, 3 cancelled (Ctrl+C or SIGTERM).
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import ROOT, VERSION, host, quality
from . import state as statefile
from .events import Reporter
from .runner import Context, run_steps
from .steps import LAUNCHER, all_steps

EXIT = {'ok': 0, 'failed': 1, 'cancelled': 3}
REMEMBER = Path(os.environ.get('XDG_CONFIG_HOME') or Path.home() / '.config') / 'dekoonut' / 'setup.json'


def remembered():
    try:
        return json.loads(REMEMBER.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def remember(game, dest, level, desktop=False):
    try:
        REMEMBER.parent.mkdir(parents=True, exist_ok=True)
        REMEMBER.write_text(json.dumps({'game': game, 'dest': dest, 'quality': level, 'desktop': bool(desktop)}),
                            encoding='utf-8')
    except OSError:
        pass


def info(display):
    last = remembered()
    dest = last.get('dest') or str(Path.home() / 'Games' / 'DekooNut')
    missing = host.missing_packages()
    cards = host.gpus()
    size = tuple(int(n) for n in display.lower().split('x')) if display else None
    suggested, why = quality.suggest(cards[0][0] if cards else '', cards[0][1] if cards else 0, size,
                                     host.memory_bytes())
    print(json.dumps({
        'version': VERSION,
        'game': last.get('game', ''),
        'dest': dest,
        'installed': (Path(dest) / LAUNCHER).is_file(),
        'quality': last.get('quality') if last.get('quality') in quality.LEVELS else suggested,
        'suggested': suggested, 'suggested_why': why,
        'levels': [{'id': k, 'name': v['name'], 'about': v['about']} for k, v in quality.LEVELS.items()],
        'missing': [what for what, _ in missing],
        'install_command': host.install_command(missing) if missing else '',
        'can_install': bool(host.install_argv(missing, graphical=True)),
        'gpu': cards[0][0] if cards else '',
        'desktop': bool(last.get('desktop')),   # the window's tick box as it was left
    }, ensure_ascii=False))
    return 0


def check():
    missing = host.missing_packages()
    cards = host.gpus()
    print(f'CPU: {"ok" if host.cpu_ok() else "no AVX2: the game cannot run on it"}')
    print(f'Memory: {host.memory_bytes() / (1 << 30):.0f} GB')
    print('Graphics card: ' + (', '.join(f'{n} ({m / 1024:.0f} GB, driver {d})' for n, m, d in cards) or
                               'no NVIDIA card found'))
    suggested, why = quality.suggest(cards[0][0] if cards else '', cards[0][1] if cards else 0,
                                     ram=host.memory_bytes())
    print(f'Suggested quality level: {suggested} ({why})')
    if not missing:
        print('Build tools: everything setup cannot get by itself is installed.')
        return 0
    print('Missing: ' + ', '.join(what for what, _ in missing))
    command = host.install_command(missing)
    if command:
        print(f'\n    {command}\n\n(or: ./setup.sh --install-packages, which runs that line)')
    return 1


def install_packages(as_json):
    """Install the missing system packages: sudo in a terminal, the desktop's password prompt for the window."""
    from .runner import Context, SetupError, Step, run_steps
    from .steps import run

    class Packages(Step):
        id, title, always = 'packages', 'Installing the system packages', True

        def run(self, ctx):
            missing = host.missing_packages()
            argv = host.install_argv(missing, graphical=as_json)
            if not missing:
                return
            if not argv:
                raise SetupError('Setup cannot install these itself on this system.',
                                 'Missing: ' + ', '.join(what for what, _ in missing), host.install_command(missing))
            if as_json:
                ctx.progress('Waiting for your password, then installing', force=True)
                run(ctx, argv, env=dict(os.environ), what='The package installation')
            elif subprocess.run(argv).returncode:
                raise SetupError('The package installation failed.', '', host.install_command(missing))
            left = host.missing_packages()
            if left:
                raise SetupError('Some build tools are still missing.', ', '.join(what for what, _ in left),
                                 host.install_command(left))

    rep = Reporter(as_json=as_json)
    steps = [Packages()]
    rep.start(VERSION, '', '', steps)
    ctx = Context(Path(os.devnull), rep, None)
    ctx.save = lambda: None
    result = run_steps(steps, ctx)
    rep.emit('end', result=result, seconds=round(rep.elapsed(), 1))
    return EXIT[result]


def cancel(signum, frame):
    raise KeyboardInterrupt


def watch_cancel(ctx):
    """A file <dest>/.setup/cancel asks for a stop: how the window cancels where a signal would not end the tools
    setup started (Windows)."""
    from .steps import stop
    flag = ctx.work / 'cancel'
    while True:
        time.sleep(0.5)
        if flag.is_file():
            flag.unlink(missing_ok=True)
            ctx.cancelled = True
            child = ctx.child
            if child:
                stop(child)
            return


def lock(dest):
    """One setup per install folder; the OS drops the lock when the process ends."""
    try:
        import fcntl
    except ImportError:
        return None
    (dest / '.setup').mkdir(parents=True, exist_ok=True)
    held = open(dest / '.setup' / 'lock', 'w')
    try:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return held


def main(argv=None):
    ap = argparse.ArgumentParser(prog='setup', description='Builds DekooNut from your own copy of the game.')
    ap.add_argument('--game', help='your game: a .nup file, or a folder holding exefs/ and romfs/')
    ap.add_argument('--dest', type=Path, help='install folder')
    ap.add_argument('--quality', choices=sorted(quality.LEVELS),
                    help='quality level (default: the one suggested for this PC; --check names it)')
    ap.add_argument('--jobs', type=int, help='parallel compiles (default: by CPU count and memory)')
    ap.add_argument('--no-menu-entry', action='store_true', help='do not add DekooNut to the applications menu')
    ap.add_argument('--desktop-entry', action='store_true', help='also put a DekooNut shortcut on the desktop (Windows)')
    ap.add_argument('--json', action='store_true', help='progress as JSON lines (what the window reads)')
    ap.add_argument('--check', action='store_true', help='only say what this PC lacks')
    ap.add_argument('--install-packages', action='store_true',
                    help='install the system packages --check names (asks for the password)')
    ap.add_argument('--info', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--display', help=argparse.SUPPRESS)   # WxH, from the window, for the suggested level
    ap.add_argument('--gui', action='store_true', help='open the window (the default without --dest)')
    args, rest = ap.parse_known_args(argv)
    if args.info:
        return info(args.display)
    if args.check:
        return check()
    if args.install_packages:
        return install_packages(args.json)
    if args.gui or not args.dest:
        if args.game and not args.dest:
            ap.error('--dest is needed with --game')
        from .gui import launch
        given = [x for flag, value in (('--game', args.game), ('--dest', args.dest)) if value for x in (flag, str(value))]
        return launch.main(given + rest)
    if rest:
        ap.error('unknown arguments: ' + ' '.join(rest))

    for name in ('SIGTERM', 'SIGHUP'):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), cancel)
    dest = args.dest.expanduser().resolve()
    log = dest / '.setup' / 'logs' / time.strftime('setup-%Y%m%d-%H%M%S.log')
    try:
        state = statefile.load(dest)
    except statefile.StateError as e:
        rep = Reporter(as_json=args.json)
        rep.error(str(e))
        rep.emit('end', result='failed', seconds=0)
        return 1
    rep = Reporter(as_json=args.json, log_path=log)
    steps = all_steps()
    rep.start(VERSION, dest, log, steps)
    held = lock(dest)
    if held is False:
        rep.error('Another setup is working on this install folder.', str(dest))
        result = 'failed'
    else:
        ctx = Context(dest, rep, args, state=state)
        (ctx.work / 'cancel').unlink(missing_ok=True)
        threading.Thread(target=watch_cancel, args=(ctx,), daemon=True).start()
        try:
            result = run_steps(steps, ctx)
        except KeyboardInterrupt:   # a second Ctrl+C while the first is handled
            result = 'cancelled'
        if result == 'ok':
            remember(ctx.state.get('game', ''), str(dest), ctx.state.get('quality', ''), args.desktop_entry)
    rep.emit('end', result=result, seconds=round(rep.elapsed(), 1),
             **({'play': str(dest / LAUNCHER)} if result == 'ok' else {}))
    rep.close()
    return EXIT[result]


if __name__ == '__main__':
    sys.exit(main())
