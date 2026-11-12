# SPDX-License-Identifier: GPL-3.0-or-later
"""oot-install.json in the install folder: what built it and which steps finished.

    {"format": 1, "version": "...", "game": "<path given>", "quality": "<level>",
     "steps": {"<step>": {"key": "<input fingerprint>", "finished": "<UTC time>", "seconds": 12.3, ...}}}

A step whose recorded key equals the current one and whose output is still there is not run again (runner.py).
The file is replaced atomically after each step.
"""
import json
import os
from pathlib import Path

NAME = 'oot-install.json'
FORMAT = 1


class StateError(Exception):
    pass


def empty():
    return {'format': FORMAT, 'steps': {}}


def load(dest):
    """The state of dest, or None when the folder has none."""
    path = Path(dest) / NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        raise StateError(f'{path} cannot be read') from None
    if not isinstance(data, dict) or not isinstance(data.get('steps'), dict):
        raise StateError(f'{path} cannot be read')
    if data.get('format', 0) > FORMAT:
        raise StateError(f'{path} was written by a newer setup')
    return data


def save(dest, data):
    path = Path(dest) / NAME
    tmp = path.with_name(NAME + '.tmp')
    with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
