# SPDX-License-Identifier: GPL-3.0-or-later
"""Setup's bookkeeping with made-up steps, no game files: which steps run on a first install, a resume, an update
and after a failure; the JSON events; the quality suggestion; the launcher and menu entry text.

    python3 tests/setup_test.py
"""
import io
import json
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from setup import quality, steps   # noqa: E402
from setup import state as statefile   # noqa: E402
from setup.events import Reporter   # noqa: E402
from setup.runner import Context, SetupError, Step, run_steps   # noqa: E402

RAN = []


class Fake(Step):
    def __init__(self, id, key='k', always=False, fail=None, output=True):
        self.id, self.title, self._key, self.always, self.fail, self.output = id, id.title(), key, always, fail, output

    def key(self, ctx):
        return self._key

    def current(self, ctx, record):
        return self.output

    def run(self, ctx):
        RAN.append(self.id)
        if self.fail == 'error':
            raise SetupError('It broke.', 'detail', 'fix --it')
        if self.fail == 'cancel':
            raise KeyboardInterrupt
        if self.fail == 'bug':
            raise ValueError('boom')
        return {'note': self.id}


def run(dest, plan):
    RAN.clear()
    out = io.StringIO()
    rep = Reporter(as_json=True, out=out)
    ctx = Context(dest, rep, types.SimpleNamespace(), state=statefile.load(dest))
    rep.start('test', dest, dest / 'log', plan)
    result = run_steps(plan, ctx)
    return result, list(RAN), [json.loads(line) for line in out.getvalue().splitlines()]


def check(what, got, want):
    if got != want:
        raise SystemExit(f'FAIL {what}: got {got!r}, want {want!r}')
    print(f'ok   {what}')


def main():
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / 'install'
        plan = lambda **kw: [Fake('check', always=True), Fake('a', **kw.get('a', {})), Fake('b', **kw.get('b', {})),
                             Fake('finish', always=True)]
        result, ran, events = run(dest, plan())
        check('first install runs every step', (result, ran), ('ok', ['check', 'a', 'b', 'finish']))
        check('events: start, then a started and a done per step',
              [e['type'] for e in events], ['start'] + ['step'] * 8)
        check('the state records the steps that are not `always`',
              sorted(statefile.load(dest)['steps']), ['a', 'b'])
        check('a second run skips what is current', run(dest, plan())[:2], ('ok', ['check', 'finish']))
        check('a changed key runs that step again', run(dest, plan(b={'key': 'k2'}))[:2],
              ('ok', ['check', 'b', 'finish']))
        check('a missing output runs that step again', run(dest, plan(a={'output': False}, b={'key': 'k2'}))[:2],
              ('ok', ['check', 'a', 'finish']))
        result, ran, events = run(dest, plan(a={'key': 'k3', 'fail': 'error'}, b={'key': 'k3'}))
        check('a failure stops the run', (result, ran), ('failed', ['check', 'a']))
        error = [e for e in events if e['type'] == 'error'][0]
        check('the error event carries text, detail, command and step',
              (error['text'], error['detail'], error['command'], error['step']), ('It broke.', 'detail', 'fix --it', 'a'))
        check('the next run goes on from the failed step', run(dest, plan(a={'key': 'k3'}, b={'key': 'k3'}))[:2],
              ('ok', ['check', 'a', 'b', 'finish']))
        check('a cancel is reported as one', run(dest, plan(b={'key': 'k4', 'fail': 'cancel'}))[0], 'cancelled')
        check('an internal error fails the run, it does not escape', run(dest, plan(b={'key': 'k5', 'fail': 'bug'}))[0],
              'failed')

        # the launcher and the menu entry
        steps.MENU_ENTRY = Path(tmp) / 'applications' / 'dekoonut.desktop'
        ctx = Context(dest, Reporter(as_json=True, out=io.StringIO()), types.SimpleNamespace(no_menu_entry=False))
        ctx.state['quality'] = quality.DEFAULT
        steps.Finish().run(ctx)
        launcher = (dest / 'dekoonut').read_text()
        level = quality.level(None)
        check('the launcher starts oot from its own folder', 'exec ./oot "$@"' in launcher and 'cd "$here"' in launcher, True)
        check('the launcher names the level\'s pack', f'gamedata/{level["pack"]}' in launcher, True)
        check('the launcher shows no overlay', 'MANGOHUD' in launcher or 'PERF_OVERLAY' in launcher, False)
        entry = steps.MENU_ENTRY.read_text()
        check('the menu entry starts the launcher with the install\'s icon',
              f'Exec="{dest / "dekoonut"}"' in entry and f'Icon={dest / "dekoonut.png"}' in entry, True)
        steps.MENU_ENTRY.write_text('[Desktop Entry]\nName=Someone else\n')
        steps.Finish().run(ctx)
        check('a menu entry setup did not write is left alone', 'Someone else' in steps.MENU_ENTRY.read_text(), True)

    name, why = quality.suggest('Card', 16 * 1024, (3840, 2160))
    check('a strong card gets the first level', name, next(iter(quality.LEVELS)))
    check('the reason names the card and the display', 'Card' in why and '3840x2160' in why, True)
    check('a weak card gets the lightest level', quality.suggest('Card', 2 * 1024)[0], list(quality.LEVELS)[-1])
    check('no card: the default level', quality.suggest('', 0)[0], quality.DEFAULT)
    print('all passed')


if __name__ == '__main__':
    main()
