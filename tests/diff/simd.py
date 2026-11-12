"""Differential test of the portable FP/SIMD lowering (lifter simd='c') against Unicorn as a Cortex-A57.

Needs no game files. Enumerates every opcode bit pattern of the FP/SIMD data-processing space (bits 27..25 = 111),
the FP/SIMD load/store space and CRC32 with random register fields, and compares which encodings the lifter and
Unicorn accept. Then a stratified sample of encodings both accept is compiled and run with random interesting
operands (NaNs, infinities, denormals, +-0, saturation edges, random FPCR.DN/FZ/RMode/AHP): X0-X30, SP, NZCV, V0-V31
and, for stores, memory must match. FPSR is not compared; the lowering does not model the cumulative exception bits
or QC.

Adapted from LAS Recomp tools/test_simd.py @ da35b83 (MIT, Copyright (c) 2026 Peiikai).

Usage: simd.py [--encodings N] [--cases K] [--seed S] [--jobs J] [--no-validity] [--filter HEX/MASK]
"""
import argparse
import functools
import os
import platform
import random
import struct
import sys
import time
import zlib
from collections import Counter, defaultdict

import unicorn as uc
from unicorn import arm64_const as A

from common import WORK, BUF, BUF_SIZE, ROOT, UC_INVALID, Emu, build_harness, diff_state, dis, mnemonic, rand_fpcr, \
    rand_v, rand_x, run_harness
from lifter import Lifter, Unsupported

print = functools.partial(print, flush=True)
BASE_AREA = BUF + 0x8000
CODE, CODE_SIZE = 0x7210000000, 0x400000


def is_ldst(w):
    return (w >> 25) & 1 == 0 and (w >> 27) & 1 == 1


def is_store(w):
    if not is_ldst(w):
        return False
    if (w & 0x3B000000) == 0x18000000:  # literal
        return False
    return not (w >> 22) & 1  # L bit, same position for structures, pairs and single registers


def is_literal(w):
    return (w & 0x3F000000) == 0x1C000000


def class_key(w):
    """Opcode signature: register fields (Rd, Rn, Rm) cleared."""
    if is_ldst(w):
        return w & 0xFFE0FC00 if (w & 0xBE000000) == 0x0C000000 else w & 0xFFFFFC00
    return w & 0xFFE0FC00


def candidates(r):
    """Every opcode bit pattern of the FP/SIMD data-processing and load/store spaces, random registers."""
    out = []
    for p in range(1 << 19):
        hi, mid = p >> 15, p & 0x7FFF
        rr = r.getrandbits(10)
        out.append((hi << 28) | (0b111 << 25) | (mid << 10) | rr)  # data processing
        out.append((hi << 28) | (1 << 27) | (1 << 26) | (mid << 10) | rr)  # loads/stores
    for sf in (0, 1):
        for C in (0, 1):
            for sz in range(4):
                for _ in range(4):
                    out.append((sf << 31) | (0b0011010110 << 21) | (r.getrandbits(5) << 16) | (0b010 << 13) |
                               (C << 12) | (sz << 10) | r.getrandbits(10))
    return out


def validity(lifter, ws, emu):
    """Map word -> (lifter accepts, Unicorn accepts); words are placed at CODE + 4i."""
    mu = emu.mu
    res = {}
    step = CODE_SIZE // 4
    for base in range(0, len(ws), step):
        chunk = ws[base:base + step]
        mu.mem_write(CODE, struct.pack(f'<{len(chunk)}I', *chunk))
        mu.ctl_remove_cache(CODE, CODE + CODE_SIZE)
        state = list(zip(emu.xregs, [BASE_AREA] * 31)) + [(A.UC_ARM64_REG_SP, BASE_AREA)]
        for i, w in enumerate(chunk):
            pc = CODE + 4 * i
            try:
                lifter.lift(w, pc)
                lok = True
            except Unsupported:
                lok = False
            if i % 32 == 0:
                mu.reg_write_batch(state)
            try:
                mu.emu_start(pc, pc + 4, count=1)
                uok = True
            except uc.UcError as e:
                uok = e.errno not in UC_INVALID
            res[w] = (lok, uok)
    return res


def make_case(r, w):
    fpcr = rand_fpcr(r)
    v = [rand_v(r) for _ in range(32)]
    nzcv = r.getrandbits(4) << 28
    sp = BASE_AREA + r.randrange(0, 0x100, 16)
    if is_ldst(w):
        x = [BASE_AREA + r.randrange(0, 0x100) for _ in range(31)]
        rm = (w >> 16) & 31
        if rm < 31 and rm != (w >> 5) & 31:
            reg_off = (w & 0x3B200C00) == 0x38200800  # register offset
            post_reg = (w & 0xBF800000) in (0x0C800000, 0x0D800000)
            if reg_off or post_reg:
                x[rm] = r.randrange(0, 0x100)
    else:
        x = [rand_x(r) for _ in range(31)]
    return fpcr, x, sp, nzcv, v


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--encodings', type=int, default=20000)
    ap.add_argument('--cases', type=int, default=8)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 4)
    ap.add_argument('--no-validity', action='store_true', help='skip the full validity enumeration')
    ap.add_argument('--filter', help='only encodings with (w & MASK) == VAL, as VAL/MASK in hex')
    ap.add_argument('--show', type=int, default=40)
    ap.add_argument('--cflags', default='-march=x86-64-v3' if platform.machine().lower() in ('amd64', 'x86_64') else '')
    args = ap.parse_args()
    r = random.Random(args.seed)
    t0 = time.time()
    lifter = Lifter(0, 'c')
    emu = Emu([(CODE, bytes(CODE_SIZE))])
    cands = candidates(r)
    if args.filter:
        val, mask = (int(x, 16) for x in args.filter.split('/'))
        cands = [w for w in cands if (w & mask) == val]
    if args.no_validity:
        val = validity(lifter, r.sample(cands, min(len(cands), args.encodings * 4)), emu)
    else:
        val = validity(lifter, cands, emu)
    agree = Counter()
    only = defaultdict(list)
    for w, (lok, uok) in val.items():
        agree[(lok, uok)] += 1
        if lok != uok:
            only['lifter' if lok else 'unicorn'].append(w)
    print(f'validity: {len(val)} encodings, both valid {agree[(True, True)]}, both invalid {agree[(False, False)]}, '
          f'lifter-only {agree[(True, False)]}, unicorn-only {agree[(False, True)]} ({time.time() - t0:.0f}s)')
    for kind in ('lifter', 'unicorn'):
        ws = only[kind]
        if ws:
            print(f'  {kind}-only by mnemonic: {dict(Counter(mnemonic(w) for w in ws).most_common(30))}')
            for w in ws[:args.show // 2]:
                print(f'    {w:08x} {dis(w)}')

    # Stratified sample of valid encodings: one per opcode signature first.
    valid = [w for w, (lok, uok) in val.items() if lok and uok]
    r.shuffle(valid)
    seen, pick = set(), []
    for w in valid:
        k = class_key(w)
        if k not in seen:
            seen.add(k)
            pick.append(w)
    picked = set(pick)
    pick = (pick + [w for w in valid if w not in picked])[:args.encodings]
    # Placed at CODE + 4i; literal loads get an offset into the code region.
    for i, w in enumerate(pick):
        pc = CODE + 4 * i
        if is_literal(w):
            tgt = CODE + r.randrange(0, (len(pick) * 4 + 0x1000) & ~3, 4)
            pick[i] = (w & ~(0x7FFFF << 5)) | ((((tgt - pc) >> 2) & 0x7FFFF) << 5)
    code_bytes = struct.pack(f'<{len(pick)}I', *pick) + bytes(0x1000 + 64)
    bodies = [lifter.lift(w, CODE + 4 * i).code for i, w in enumerate(pick)]
    work = WORK / 'simd'
    work.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    exe = build_harness(work, bodies, args.jobs, args.cflags)
    print(f'compiled {len(bodies)} tests ({time.time() - t1:.0f}s)')

    mu = emu.mu
    mu.mem_write(CODE, code_bytes)
    mu.ctl_remove_cache(CODE, CODE + CODE_SIZE)
    buf0 = bytes(r.getrandbits(8) for _ in range(4096)) * (BUF_SIZE // 4096)
    mu.mem_write(BUF, buf0)
    code_path = work / 'code.bin'
    code_path.write_bytes(code_bytes)
    cases, expect = [], []
    t2 = time.time()
    for k, w in enumerate(pick):
        pc = CODE + 4 * k
        st = is_store(w)
        for _ in range(args.cases):
            fpcr, x, sp, nzcv, v = make_case(r, w)
            try:
                ox, osp, onz, opc, ov = emu.run(pc, fpcr, x, sp, nzcv, v)
            except uc.UcError:
                if st:
                    mu.mem_write(BUF, buf0)
                continue
            mh = 0
            if st:
                mh = zlib.crc32(bytes(mu.mem_read(BUF, BUF_SIZE)))
                mu.mem_write(BUF, buf0)
            cases.append((k, fpcr, x, sp, nzcv, v, st))
            expect.append((ox, osp, onz, opc, ov, mh))
    print(f'unicorn: {len(cases)} cases ({time.time() - t2:.0f}s)')

    t3 = time.time()
    recs, err = run_harness(exe, work, buf0, cases, [(CODE, code_path)])
    print(f'harness: {len(recs)} of {len(cases)} cases ({time.time() - t3:.0f}s)')
    if err:
        k = cases[len(recs)][0] if len(recs) < len(cases) else None
        print(f'harness failed: {err}' + (f', stopped at {pick[k]:08x} {dis(pick[k])}' if k is not None else ''))

    fails, nfail = {}, 0
    for case, want, got in zip(cases, expect, recs):
        k, v, st = case[0], case[5], case[6]
        d = diff_state(got, want, check_pc=False, check_mem=st)
        if d:
            nfail += 1
            w = pick[k]
            if w not in fails:
                ins = [f'v{j}={v[j]:032x}' for j in sorted({(w >> 5) & 31, (w >> 16) & 31, w & 31})]
                fails[w] = (case[1], d[:3], ins)
    print(f'{len(pick)} encodings, {len(cases)} cases: {nfail} failing cases, {len(fails)} failing encodings '
          f'({time.time() - t0:.0f}s total)')
    if fails:
        print('  failing by mnemonic:', dict(Counter(mnemonic(w) for w in fails).most_common(40)))
    for w, (fpcr, d, ins) in list(fails.items())[:args.show]:
        print(f'  {w:08x} {dis(w):40} fpcr={fpcr:x} {"; ".join(d)}')
        print(f'      {" ".join(ins)}')
    return 1 if fails or err else 0


if __name__ == '__main__':
    sys.exit(main())
