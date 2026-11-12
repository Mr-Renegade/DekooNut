# SPDX-License-Identifier: GPL-3.0-or-later
"""Progress of a setup run: one JSON object per line for the window (--json), or plain lines for a terminal.
Everything also goes to the log file.

Events ("type", "t" seconds since the start, and):
    start     version, dest, log, steps [{id, title}]
    step      step, state started | done (seconds) | skipped (reason)
    progress  step, text, and done/total when the amount is known
    log       line (a tool's own output)
    icon      path (the picture the window shows)
    warning   text
    error     text, detail, command (a line to run, when one fixes it), step
    end       result ok | failed | cancelled, seconds, play (the launcher, when ok)
"""
import json
import sys
import threading
import time

PROGRESS_EVERY = 0.25   # seconds between two progress events of one step


class Reporter:
    def __init__(self, as_json=False, log_path=None, out=None):
        self.as_json, self.out = as_json, out or sys.stdout
        self.t0 = time.monotonic()
        self.lock = threading.Lock()
        self.titles, self.order = {}, []
        self.last_progress = 0.0
        self.live = False            # a progress line is on the terminal and not yet ended
        self.tty = not as_json and self.out.isatty()
        self.file = None
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.file = open(log_path, 'a', encoding='utf-8', newline='\n')

    def elapsed(self):
        return time.monotonic() - self.t0

    def close(self):
        if self.file:
            self.file.close()
            self.file = None

    # ---- output

    def _file(self, text):
        if self.file:
            self.file.write(f'[{self.elapsed():8.1f}] {text}\n')
            self.file.flush()

    def _line(self, text, live=False):
        """A terminal line; a live one is overwritten by the next."""
        if self.tty and self.live:
            self.out.write('\r\033[K')
        if live and self.tty:
            self.out.write(text[:max(20, _columns() - 1)])
            self.live = True
        else:
            self.out.write(text + '\n')
            self.live = False
        self.out.flush()

    def emit(self, type, **fields):
        with self.lock:
            if self.as_json:
                self.out.write(json.dumps({'type': type, 't': round(self.elapsed(), 2), **fields},
                                          ensure_ascii=False) + '\n')
                self.out.flush()
            else:
                self._text(type, fields)

    def _text(self, type, f):
        if type == 'start':
            self._line(f'DekooNut setup {f["version"]}  ->  {f["dest"]}')
            self._line(f'log: {f["log"]}')
        elif type == 'step':
            n, title = self.order.index(f['step']) + 1, self.titles.get(f['step'], f['step'])
            head = f'[{n:2}/{len(self.order)}] {title}'
            if f['state'] == 'started':
                self._line(head + ' ...', live=True)
            elif f['state'] == 'done':
                self._line(f'{head}: done ({_duration(f["seconds"])})')
            else:
                self._line(f'{head}: already done')
        elif type == 'progress':
            n, title = self.order.index(f['step']) + 1, self.titles.get(f['step'], f['step'])
            amount = f' {100 * f["done"] // f["total"]:3}%' if f.get('total') else ''
            if self.tty:
                self._line(f'[{n:2}/{len(self.order)}] {title}:{amount} {f["text"]}', live=True)
        elif type == 'warning':
            self._line(f'warning: {f["text"]}')
        elif type == 'error':
            self._line(f'error: {f["text"]}')
            for line in (f.get('detail') or '').splitlines():
                self._line(f'  {line}')
            if f.get('command'):
                self._line(f'\n    {f["command"]}\n')
        elif type == 'end':
            word = {'ok': 'Done', 'failed': 'Setup failed', 'cancelled': 'Cancelled'}[f['result']]
            self._line(f'{word} after {_duration(f["seconds"])}.')
            if f.get('play'):
                self._line(f'Play: {f["play"]}')

    # ---- what the steps call

    def start(self, version, dest, log, steps):
        self.order = [s.id for s in steps]
        self.titles = {s.id: s.title for s in steps}
        self._file(f'setup {version} dest {dest}')
        self.emit('start', version=version, dest=str(dest), log=str(log),
                  steps=[{'id': s.id, 'title': s.title} for s in steps])

    def step(self, step, state, **fields):
        self._file(f'step {step} {state} {fields or ""}')
        self.emit('step', step=step, state=state, **fields)

    def progress(self, step, text, done=None, total=None, force=False):
        now = time.monotonic()
        if not force and now - self.last_progress < PROGRESS_EVERY:
            return
        self.last_progress = now
        fields = {'step': step, 'text': text}
        if total:
            fields.update(done=done, total=total)
        self.emit('progress', **fields)

    def log(self, line):
        """A tool's output line: to the file and the window, not to the terminal (it would bury the steps)."""
        self._file(line)
        if self.as_json:
            self.emit('log', line=line)

    def warning(self, text):
        self._file(f'warning: {text}')
        self.emit('warning', text=text)

    def error(self, text, detail='', command='', step=None):
        self._file(f'error: {text}\n{detail}\n{command}')
        self.emit('error', text=text, detail=detail, command=command, step=step)

    def icon(self, path):
        self.emit('icon', path=str(path))


def _columns():
    import shutil
    return shutil.get_terminal_size((100, 24)).columns


def _duration(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f'{seconds} s'
    if seconds < 3600:
        return f'{seconds // 60} min {seconds % 60:02} s'
    return f'{seconds // 3600} h {seconds % 3600 // 60:02} min'
