# SPDX-License-Identifier: GPL-3.0-or-later
"""Which steps run, and the bookkeeping around them: a first install, a resume and an update are one rule.

Each step has a key: a fingerprint of everything its output depends on, which includes the keys of the steps it
reads from. A step is skipped when oot-install.json records it with the same key and its output is still in place
(Step.current, cheap). `always` steps run every time. A failed or cancelled run leaves the finished steps recorded.
"""
import datetime
import hashlib
import json
import threading
import traceback

from . import state as statefile


class SetupError(Exception):
    """A failure the player can act on: one sentence, optional detail lines and a command that fixes it."""

    def __init__(self, text, detail='', command=''):
        super().__init__(text)
        self.text, self.detail, self.command = text, detail, command


class Step:
    id = ''
    title = ''
    always = False

    def key(self, ctx):
        return ''

    def current(self, ctx, record):
        return True

    def run(self, ctx):
        """Do the work. Returns fields to record with the step."""
        return {}


def digest(*parts):
    h = hashlib.sha256()
    for part in parts:
        h.update(json.dumps(part, sort_keys=True, ensure_ascii=False, default=str).encode() + b'\0')
    return h.hexdigest()[:32]


class Context:
    def __init__(self, dest, reporter, options, state=None):
        self.dest, self.reporter, self.options = dest, reporter, options
        self.work = dest / '.setup'
        self.state = state or statefile.empty()
        self.keys, self.steps = {}, []
        self.step = None            # the id of the running step, for progress events
        self.child = None           # the running tool, for a cancel
        self.cancelled = False      # set by cli.py's watcher when the window asked for a stop through a file

    def record(self, step_id):
        return self.state['steps'].get(step_id)

    def save(self):
        self.dest.mkdir(parents=True, exist_ok=True)
        statefile.save(self.dest, self.state)

    def progress(self, text, done=None, total=None, force=False):
        if self.cancelled and threading.current_thread() is threading.main_thread():
            raise KeyboardInterrupt
        self.reporter.progress(self.step, text, done, total, force)


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def run_steps(steps, ctx):
    """Returns 'ok', 'failed' or 'cancelled'; the events say why."""
    rep = ctx.reporter
    ctx.steps = steps
    try:
        for s in steps:
            # A key is computed when its turn comes: it may hash what the steps before it wrote.
            ctx.keys[s.id] = key = s.key(ctx)
            rec = ctx.record(s.id)
            if not s.always and rec and rec.get('key') == key and s.current(ctx, rec):
                rep.step(s.id, 'skipped', reason='current')
                continue
            ctx.step = s.id
            rep.step(s.id, 'started')
            t0 = rep.elapsed()
            fields = s.run(ctx) or {}
            seconds = round(rep.elapsed() - t0, 1)
            if not s.always:
                ctx.state['steps'][s.id] = {'key': key, 'finished': now_utc(), 'seconds': seconds, **fields}
            ctx.save()
            rep.step(s.id, 'done', seconds=seconds)
            ctx.step = None
        return 'ok'
    except SetupError as e:
        rep.error(e.text, e.detail, e.command, step=ctx.step)
        return 'failed'
    except statefile.StateError as e:
        rep.error(str(e), step=ctx.step)
        return 'failed'
    except KeyboardInterrupt:
        rep.log(f'cancelled during {ctx.step}')
        return 'cancelled'
    except Exception as e:
        rep.log(traceback.format_exc())
        rep.error(f'Setup stopped on an internal error ({type(e).__name__}: {e}).',
                  'The log has the details.', step=ctx.step)
        return 'failed'
