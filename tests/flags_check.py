"""Build and run tests/flags_check.c: runtime/cpu.h's function-local flag helpers against its packed flag helpers.

Usage: flags_check.py
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests' / 'diff'))
from common import WORK, MSYS_BIN, compiler  # noqa: E402


def main():
    work = WORK / 'flags'
    work.mkdir(parents=True, exist_ok=True)
    exe = work / ('flags_check.exe' if os.name == 'nt' else 'flags_check')
    env = dict(os.environ)
    if os.name == 'nt' and MSYS_BIN.exists():
        env['PATH'] = str(MSYS_BIN) + os.pathsep + env.get('PATH', '')
    for opt in ('-O0', '-O1'):
        subprocess.check_call([compiler(), opt, '-fno-strict-aliasing', '-I', str(ROOT / 'runtime'),
                               str(ROOT / 'tests' / 'flags_check.c'), '-o', str(exe)], env=env)
        print(opt, end=': ', flush=True)
        if subprocess.run([str(exe)], env=env).returncode:
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
