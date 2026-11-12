"""Differential test of exclusives and atomics against Unicorn (max CPU model, which has the ARMv8.1 LSE atomics).

Single steps cannot cover these: a store exclusive only succeeds after a load exclusive, and the A57 model rejects
LSE. So this test enumerates every variant and runs short sequences, with the monitor cleared before each case:

- LDXR/LDAXR then STXR/STLXR to the same address (succeeds), all sizes;
- LDXP/LDAXP then STXP/STLXP to the same address (succeeds) or 16 bytes further (fails), W and X pairs;
- STXR and STXP alone (fail without a monitor);
- LDADD, LDCLR, LDEOR, LDSET, LDSMAX, LDSMIN, LDUMAX, LDUMIN and SWP, every size and ordering;
- CAS and CASP, every size and ordering, with the compare value equal to memory in half of the cases.

Needs no game files. Usage: atomic.py [--cases K] [--seed S] [--jobs J]
"""
import argparse
import os
import platform
import random
import struct
import sys
import time
import zlib

import unicorn as uc
from unicorn import arm64_const as A

from common import WORK, BUF, BUF_SIZE, ROOT, Emu, build_harness, diff_state, dis, rand_fpcr, rand_v, rand_x, run_harness
from lifter import Lifter

BUF_MID = BUF + 0x10000
CODE = 0x7220000000
SLOT = 16  # bytes of code per test
CLREX = 0xD5033F5F


def lse(size, a, r, rs, o3, opc, rn, rt):
    return 0x38200000 | size << 30 | a << 23 | r << 22 | rs << 16 | o3 << 15 | opc << 12 | rn << 5 | rt


def cas(size, l, rs, o0, rn, rt):
    return 0x08A07C00 | size << 30 | l << 22 | rs << 16 | o0 << 15 | rn << 5 | rt


def casp(sz, l, rs, o0, rn, rt):
    return 0x08207C00 | sz << 30 | l << 22 | rs << 16 | o0 << 15 | rn << 5 | rt


def ldxp(sz, o0, rt2, rn, rt):
    return 0x887F0000 | sz << 30 | o0 << 15 | rt2 << 10 | rn << 5 | rt


def stxp(sz, o0, rs, rt2, rn, rt):
    return 0x88200000 | sz << 30 | rs << 16 | o0 << 15 | rt2 << 10 | rn << 5 | rt


def ldxr(size, o0, rn, rt):
    return 0x085F7C00 | size << 30 | o0 << 15 | rn << 5 | rt


def stxr(size, o0, rs, rn, rt):
    return 0x08007C00 | size << 30 | rs << 16 | o0 << 15 | rn << 5 | rt


def regs(r, n, exclude=()):
    """n distinct general registers 0-30 outside exclude."""
    return r.sample([i for i in range(31) if i not in exclude], n)


def tests(r):
    """[(words, rn, kind)]: kind 'cas'/'casp' marks compare-and-swap tests (state gets a matching compare value)."""
    out = []
    for size in range(4):
        for a in (0, 1):
            for rel in (0, 1):
                for o3, opc in [(0, k) for k in range(8)] + [(1, 0)]:
                    rn = r.choice(regs(r, 1) + [31])
                    rs, rt = r.randrange(32), r.randrange(32)
                    out.append(([lse(size, a, rel, rs, o3, opc, rn, rt)], rn, 'lse'))
                rn = r.choice(regs(r, 1) + [31])
                rs, rt = regs(r, 2, {rn})
                out.append(([cas(size, a, rs, rel, rn, rt)], rn, 'cas'))
        # LDAPR{B,H} (ARMv8.3 RCpc, Switch 2 code): A=1, R=0, Rs=XZR, o3=1, opc=4
        rn = r.choice(regs(r, 1) + [31])
        out.append(([lse(size, 1, 0, 31, 1, 4, rn, r.randrange(32))], rn, 'lse'))
    for sz in (0, 1):
        for l in (0, 1):
            for o0 in (0, 1):
                rn = r.choice(regs(r, 1) + [31])
                rs, rt = (2 * i for i in r.sample(range(15), 2))
                if rn in (rs, rs + 1):
                    rs = 30 if rn < 30 else 0  # Rs+1 = XZR or away from Rn
                out.append(([casp(sz, l, rs, o0, rn, rt)], rn, 'casp'))
    for size in range(4):
        for o0 in (0, 1):
            rn = r.choice(regs(r, 1) + [31])
            rt, rs, rt_st = regs(r, 3, {rn})
            out.append(([ldxr(size, o0, rn, rt), stxr(size, o0, rs, rn, rt_st)], rn, 'excl'))
            out.append(([stxr(size, o0, rs, rn, rt_st)], rn, 'excl'))
    for sz in (0, 1):
        for o0 in (0, 1):
            rn = r.choice(regs(r, 1) + [31])
            rt, rt2, rs, d1, d2 = regs(r, 5, {rn})
            out.append(([ldxp(sz, o0, rt2, rn, rt)], rn, 'excl'))
            out.append(([ldxp(sz, o0, rt2, rn, rt), stxp(sz, o0, rs, d2, rn, d1)], rn, 'excl'))
            out.append(([stxp(sz, o0, rs, d2, rn, d1)], rn, 'excl'))
            # A store exclusive to another address fails; Rn is advanced between the two by an ADD.
            if rn != 31:
                add = 0x91004000 | rn << 5 | rn  # add xn, xn, #16
                out.append(([ldxp(sz, o0, rt2, rn, rt), add, stxp(sz, o0, rs, d2, rn, d1)], rn, 'excl'))
    return out


def unicorn_wrong(w):
    """LDSMAX/LDSMIN on bytes, halfwords and words: Unicorn 2.1 (QEMU 5.0) compares them unsigned."""
    return (w & 0x3F200C00) == 0x38200000 and (w >> 30) < 3 and not (w >> 15) & 1 and (w >> 12) & 7 in (4, 5)


def model_signed_minmax(w, x, sp, nzcv, v, buf0):
    """Reference for unicorn_wrong(w) words, from the ARM pseudocode: returns the expected state and memory CRC."""
    nb = 1 << (w >> 30)
    rs, rn, rt = (w >> 16) & 31, (w >> 5) & 31, w & 31
    addr = sp if rn == 31 else x[rn]
    off = addr - BUF
    old = int.from_bytes(buf0[off:off + nb], 'little')
    operand = (x[rs] if rs != 31 else 0) & ((1 << 8 * nb) - 1)
    sign = lambda u: u - (1 << 8 * nb) if u >> (8 * nb - 1) else u
    pick = max if (w >> 12) & 7 == 4 else min
    new = old if pick(sign(old), sign(operand)) == sign(old) else operand
    mem = bytearray(buf0)
    mem[off:off + nb] = new.to_bytes(nb, 'little')
    ox = list(x[:31])
    if rt != 31:
        ox[rt] = old
    return (ox, sp, nzcv, 0, list(v), zlib.crc32(bytes(mem)))


def make_case(r, words, rn, kind, buf0):
    addr = BUF_MID + r.randrange(0, 0x100, 16)
    x = [rand_x(r) for _ in range(31)]
    sp = addr if rn == 31 else BUF_MID + r.randrange(0, 0x100, 16)
    if rn != 31:
        x[rn] = addr
    w = words[0]
    if kind in ('cas', 'casp') and r.random() < 0.5:
        off = addr - BUF
        rs = (w >> 16) & 31
        if kind == 'cas':
            nb = 1 << (w >> 30)
            if rs != 31:
                x[rs] = int.from_bytes(buf0[off:off + nb], 'little')
        else:
            nb = 8 if (w >> 30) & 1 else 4
            for i in (0, 1):
                if rs + i != 31:
                    x[rs + i] = int.from_bytes(buf0[off + i * nb:off + (i + 1) * nb], 'little')
    return rand_fpcr(r), x, sp, r.getrandbits(4) << 28, [rand_v(r) for _ in range(32)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--cases', type=int, default=64)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 4)
    ap.add_argument('--cflags', default='-march=x86-64-v3' if platform.machine().lower() in ('amd64', 'x86_64') else '')
    args = ap.parse_args()
    r = random.Random(args.seed)
    t0 = time.time()

    ts = tests(r)
    code = [CLREX, 0, 0, 0]
    bodies = []
    lifter = Lifter(0, 'c')
    for i, (words, _, _) in enumerate(ts):
        pc = CODE + SLOT * (i + 1)
        code += words + [0] * (SLOT // 4 - len(words))
        bodies.append(' '.join(lifter.lift(w, pc + 4 * j).code for j, w in enumerate(words)))
    code_bytes = struct.pack(f'<{len(code)}I', *code)
    work = WORK / 'atomic'
    work.mkdir(parents=True, exist_ok=True)
    code_path = work / 'code.bin'
    code_path.write_bytes(code_bytes)
    exe = build_harness(work, bodies, args.jobs, args.cflags)

    emu = Emu([(CODE, code_bytes)], A.UC_CPU_ARM64_MAX)
    mu = emu.mu
    buf0 = bytes(r.getrandbits(8) for _ in range(4096)) * (BUF_SIZE // 4096)
    mu.mem_write(BUF, buf0)
    cases, expect = [], []
    for k, (words, rn, kind) in enumerate(ts):
        pc = CODE + SLOT * (k + 1)
        for _ in range(args.cases):
            fpcr, x, sp, nzcv, v = make_case(r, words, rn, kind, buf0)
            if unicorn_wrong(words[0]):
                cases.append((k, fpcr, x, sp, nzcv, v, True))
                expect.append(model_signed_minmax(words[0], x, sp, nzcv, v, buf0))
                continue
            mu.emu_start(CODE, CODE + 4, count=1)  # clear the exclusive monitor
            try:
                ox, osp, onz, opc, ov = emu.run(pc, fpcr, x, sp, nzcv, v, count=len(words))
            except uc.UcError as e:
                raise SystemExit(f'unicorn rejected {" ; ".join(dis(w) for w in words)}: {e}')
            mh = zlib.crc32(bytes(mu.mem_read(BUF, BUF_SIZE)))
            mu.mem_write(BUF, buf0)
            cases.append((k, fpcr, x, sp, nzcv, v, True))
            expect.append((ox, osp, onz, opc, ov, mh))

    recs, err = run_harness(exe, work, buf0, cases, [(CODE, code_path)])
    if err:
        k = cases[len(recs)][0] if len(recs) < len(cases) else None
        print(f'harness failed: {err}' + (f', stopped at {" ; ".join(dis(w) for w in ts[k][0])}' if k is not None else ''))
    fails, nfail = {}, 0
    for case, want, got in zip(cases, expect, recs):
        d = diff_state(got, want, check_pc=False, check_mem=True)
        if d:
            nfail += 1
            fails.setdefault(case[0], d[:4])
    print(f'{len(ts)} tests, {len(cases)} cases: {nfail} failing cases, {len(fails)} failing tests '
          f'({time.time() - t0:.0f}s)')
    for k, d in fails.items():
        print(f'  {" ; ".join(dis(w) for w in ts[k][0]):70} {"; ".join(d)}')
    return 1 if fails or err else 0


if __name__ == '__main__':
    sys.exit(main())
