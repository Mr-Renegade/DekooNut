"""Differential test of the instructions OoT Recomp added to the portable lifter, against Unicorn as ARM64 "max".

Needs no game files. The Cortex-A57 model of tests/diff/simd.py rejects these (ARMv8.2 and later), so Unicorn runs as
UC_CPU_ARM64_MAX:

  * SDOT / UDOT (vector, 4S from 16B), Q = 0 and 1, including Rd == Rn and Rd == Rm;
  * FNEG / FABS (vector, half precision), Q = 0 and 1.

PACGA has no Unicorn oracle worth having (the result is a keyed MAC), so it is checked for its contract instead: the
low 32 bits are zero, and the same inputs give the same tag.

Usage: newinsn.py [--cases K] [--seed S]
"""
import argparse
import functools
import os
import platform
import random
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'recomp'))

import unicorn as uc
from unicorn import arm64_const as A

from common import WORK, BUF, BUF_SIZE, ROOT, Emu, build_harness, diff_state, dis, run_harness
from lifter import Lifter

print = functools.partial(print, flush=True)
CODE, CODE_SIZE = 0x7210000000, 0x10000


def words():
    ws = []
    regs = [(0, 1, 2), (4, 0, 3), (5, 5, 6), (7, 8, 7), (9, 9, 9), (30, 31, 1), (2, 3, 4)]
    for Q in (0, 1):
        for U in (0, 1):
            for d, n, m in regs:  # UDOT (U = 1) / SDOT (U = 0)
                ws.append(0x0E809400 | Q << 30 | U << 29 | m << 16 | n << 5 | d)
        for U in (0, 1):  # FABS (U = 0) / FNEG (U = 1)
            for d, n, _ in regs:
                ws.append(0x0EF8F800 | Q << 30 | U << 29 | n << 5 | d)
    return ws


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--cases', type=int, default=200)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--cflags', default='-march=x86-64-v3' if platform.machine().lower() in ('amd64', 'x86_64') else '')
    args = ap.parse_args()
    r = random.Random(args.seed)
    t0 = time.time()
    lifter = Lifter(0, 'c')
    pick = words()
    pacga = [0x9AD13262, 0x9AC03020]  # PACGA X2, X19, X17 and PACGA X0, X1, X0
    allw = pick + pacga
    code_bytes = struct.pack(f'<{len(allw)}I', *allw) + bytes(0x100)
    emu = Emu([(CODE, bytes(CODE_SIZE))], model=A.UC_CPU_ARM64_MAX)
    bodies = []
    for i, w in enumerate(allw):
        ins = lifter.lift(w, CODE + 4 * i)
        if ins.kind == 'trap':
            print(f'not lifted: {w:08x} {dis(w)}')
            return 1
        bodies.append(ins.code)
    work = WORK / 'newinsn'
    work.mkdir(parents=True, exist_ok=True)
    exe = build_harness(work, bodies, 4, args.cflags)
    emu.mu.mem_write(CODE, code_bytes)
    emu.mu.ctl_remove_cache(CODE, CODE + CODE_SIZE)
    buf0 = bytes(r.getrandbits(8) for _ in range(4096)) * (BUF_SIZE // 4096)
    emu.mu.mem_write(BUF, buf0)
    code_path = work / 'code.bin'
    code_path.write_bytes(code_bytes)

    cases, expect = [], []
    for k, w in enumerate(allw):
        for _ in range(args.cases):
            v = [r.getrandbits(128) for _ in range(32)]
            x = [r.getrandbits(64) for _ in range(31)]
            case = (k, 0, x, BUF + 0x8000, r.getrandbits(4) << 28, v, False)
            if k < len(pick):
                try:
                    ox, osp, onz, opc, ov = emu.run(CODE + 4 * k, 0, x, BUF + 0x8000, case[4], v)
                except uc.UcError as e:
                    print(f'unicorn rejects {w:08x} {dis(w)}: {e}')
                    return 1
                expect.append((ox, osp, onz, opc, ov, 0))
            cases.append(case)
    recs, err = run_harness(exe, work, buf0, cases, [(CODE, code_path)])
    if err or len(recs) != len(cases):
        print(f'harness failed: {err}')
        return 1

    nfail, fails = 0, {}
    for i, (case, got) in enumerate(zip(cases, recs)):
        k = case[0]
        if k >= len(pick):
            continue
        d = diff_state(got, expect[i], check_pc=False, check_mem=False)
        if d:
            nfail += 1
            fails.setdefault(allw[k], d[:3])

    # PACGA contract
    bad_pac = 0
    seen = {}
    for case, got in zip(cases, recs):
        k = case[0]
        if k < len(pick):
            continue
        w = allw[k]
        rd, rn, rm = w & 31, (w >> 5) & 31, (w >> 16) & 31
        x = case[2]
        key = (w, 0 if rn == 31 else x[rn], 0 if rm == 31 else x[rm])
        tag = got[0][rd] if rd != 31 else 0
        if rd != 31 and tag & 0xFFFFFFFF:
            bad_pac += 1
        if seen.setdefault(key, tag) != tag:
            bad_pac += 1

    print(f'{len(pick)} SDOT/UDOT/FNEG/FABS encodings x {args.cases} cases: {nfail} failing cases, '
          f'{len(fails)} failing encodings; PACGA contract violations: {bad_pac} ({time.time() - t0:.0f}s)')
    for w, d in list(fails.items())[:20]:
        print(f'  {w:08x} {dis(w):36} {"; ".join(d)}')
    return 1 if fails or bad_pac else 0


if __name__ == '__main__':
    sys.exit(main())
