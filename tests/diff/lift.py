"""Differential test of the lifter on the game's own instructions: lifted C against Unicorn, one instruction at a time.

Samples distinct instruction words from the .text of each ExeFS module, one per opcode signature first, lifts each
with the portable FP/SIMD lowering, runs it from random register state, and compares X0-X30, SP, NZCV, PC (for
branches), V0-V31 and the scratch buffer (for loads and stores) with Unicorn as a Cortex-A57.

Skipped: system instructions and exceptions, whose single step cannot be compared, and exclusives and ARMv8.1
atomics, which atomic.py covers. Encodings the lifter rejects are counted per module and mnemonic; --coverage lifts
every distinct word instead of testing.

Adapted from LAS Recomp tools/test_lift.py @ da35b83 (MIT, Copyright (c) 2026 Peiikai).

Usage: lift.py [--exefs DIR] [--modules main,sdk,subsdk0] [--encodings N] [--cases K] [--seed S] [--jobs J]
       lift.py --coverage
"""
import argparse
import os
import platform
import random
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import unicorn as uc

from common import WORK, BUF, BUF_SIZE, ROOT, Emu, build_harness, diff_state, dis, mnemonic, rand_fpcr, rand_v, rand_x, \
    run_harness
from lifter import Lifter, Unsupported, bits
from nso import Nso

BUF_MID = BUF + 0x10000
# Test load addresses, below 2^39 and clear of BUF.
BASES = {'main': 0x7000000000, 'subsdk0': 0x7008000000, 'sdk': 0x7010000000}


def is_mem(w):
    return bits(w, 25, 1) == 0 and bits(w, 27, 1) == 1


def classify(w):
    """'skip' for instructions whose single-step behaviour cannot be compared."""
    if (w & 0xFFC00000) == 0xD5000000:  # system: hints, barriers, MRS/MSR, cache maintenance
        return 'skip'
    if (w & 0x3F000000) == 0x08000000 and (bits(w, 23, 1) == 0 or bits(w, 21, 1)):
        return 'skip'  # exclusives need a paired load; CAS is ARMv8.1; both are covered by atomic.py
    if (w & 0x3F200C00) == 0x38200000:  # LSE atomics, ARMv8.1: atomic.py
        return 'skip'
    if (w & 0xFF000000) == 0xD4000000:  # SVC, BRK, HLT
        return 'skip'
    if (w & 0xFFFF0000) == 0:  # UDF
        return 'skip'
    return 'ok'


def class_keys(words):
    """Opcode signature per word: register fields (Rd, Rn, Rm) cleared."""
    return words & np.uint32(0xFFE0FC00)


def to_test_body(ins, pc):
    if ins.kind == 'op':
        return ins.code
    nxt = f'0x{pc + 4:x}ull'
    if ins.kind == 'b':
        return f'c->pc = 0x{ins.target:x}ull;'
    if ins.kind == 'bl':
        return f'c->x[30] = {nxt}; c->pc = 0x{ins.target:x}ull;'
    if ins.kind == 'bcond':
        return f'c->pc = ({ins.cond}) ? 0x{ins.target:x}ull : {nxt};'
    if ins.kind == 'br':
        return f'c->pc = {ins.target};'
    if ins.kind == 'blr':
        return f'{{ uint64_t t = {ins.target}; c->x[30] = {nxt}; c->pc = t; }}'
    if ins.kind == 'ret':
        return 'c->pc = c->x[30];'
    raise Unsupported(ins.kind)


def pick_module(name, nso, base, count, r, unsupported):
    """Distinct words of .text, one per opcode signature first, lifted. Returns [(word, pc, body)]."""
    lo, size = nso.text
    words = np.frombuffer(bytes(nso.image[lo:lo + size]), dtype='<u4')
    uniq, first = np.unique(words, return_index=True)
    keys = class_keys(uniq)
    order = list(range(len(uniq)))
    r.shuffle(order)
    seen, primary, rest = set(), [], []
    for i in order:
        k = int(keys[i])
        (rest if k in seen else primary).append(i)
        seen.add(k)
    lifter = Lifter(base, 'c')
    picks = []
    for i in primary + rest:
        if len(picks) >= count:
            break
        w = int(uniq[i])
        if classify(w) == 'skip':
            continue
        pc = base + lo + int(first[i]) * 4
        try:
            body = to_test_body(lifter.lift(w, pc), pc)
        except Unsupported:
            unsupported[(name, mnemonic(w))] += 1
            continue
        picks.append((w, pc, body))
    print(f'{name}: {len(words)} words, {len(uniq)} distinct, {len(seen)} signatures, {len(picks)} picked')
    return picks


def coverage(name, nso, base):
    """Lift every distinct word of .text. Returns {kind mnemonic: [distinct words, occurrences]} of words the lifter
    rejects (Unsupported) or turns into an undefined-instruction trap (rt_udf)."""
    lo, size = nso.text
    words = np.frombuffer(bytes(nso.image[lo:lo + size]), dtype='<u4')
    uniq, first, counts = np.unique(words, return_index=True, return_counts=True)
    lifter = Lifter(base, 'c')
    rejected = {}
    for w, i, n in zip(uniq.tolist(), first.tolist(), counts.tolist()):
        try:
            ins = lifter.lift(w, base + lo + i * 4)
            if ins.kind != 'trap' or not ins.code.startswith('rt_udf'):
                continue
            key = f'udf {mnemonic(w)}'
        except Unsupported:
            key = f'unsupported {mnemonic(w)}'
        e = rejected.setdefault(key, [0, 0])
        e[0] += 1
        e[1] += n
    total = sum(e[1] for e in rejected.values())
    print(f'{name}: {len(words)} words, {len(uniq)} distinct, {total} rejected '
          f'({sum(e[0] for e in rejected.values())} distinct)')
    for mn, (d, n) in sorted(rejected.items(), key=lambda kv: -kv[1][1]):
        print(f'  {mn:24} {n:8} occurrences {d:6} distinct')
    return rejected


def make_case(r, w):
    if is_mem(w):
        x = [BUF_MID + r.randrange(0, 0x100, 8) for _ in range(31)]
        if (bits(w, 21, 1) and bits(w, 10, 2) == 2) or (w & 0xBFA00000) == 0x0C800000:  # register offset
            rm = bits(w, 16, 5)
            if rm != bits(w, 5, 5) and rm < 31:
                x[rm] = r.randrange(0, 0x100)
    else:
        x = [rand_x(r) for _ in range(31)]
    sp = BUF_MID + r.randrange(0, 0x100, 16)
    nzcv = r.getrandbits(4) << 28
    v = [rand_v(r) for _ in range(32)]
    fpcr = rand_fpcr(r)
    return fpcr, x, sp, nzcv, v


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--exefs', type=Path, default=ROOT / 'build' / 'extracted' / 'oot111' / 'exefs')
    ap.add_argument('--modules', default='main,sdk,subsdk0')
    ap.add_argument('--encodings', type=int, default=20000, help='per module')
    ap.add_argument('--cases', type=int, default=3, help='random states per encoding')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 4)
    ap.add_argument('--show', type=int, default=40)
    ap.add_argument('--cflags', default='-march=x86-64-v3' if platform.machine().lower() in ('amd64', 'x86_64') else '')
    ap.add_argument('--coverage', action='store_true',
                    help='only lift every distinct word and list the rejected ones (capstone "?" = undecodable data)')
    args = ap.parse_args()
    r = random.Random(args.seed)
    t0 = time.time()
    names = args.modules.split(',')
    if args.coverage:
        for name in names:
            coverage(name, Nso(str(args.exefs / name)), BASES[name])
        print(f'({time.time() - t0:.0f}s)')
        return 0

    work = WORK / 'lift'
    work.mkdir(parents=True, exist_ok=True)
    images, picks, unsupported = [], [], Counter()
    for name in names:
        nso = Nso(str(args.exefs / name))
        base = BASES[name]
        img = work / f'{name}.img'
        img.write_bytes(bytes(nso.image))
        images.append((base, img, nso.image))
        picks += [(name,) + p for p in pick_module(name, nso, base, args.encodings, r, unsupported)]
    print(f'lifted {len(picks)} encodings ({time.time() - t0:.0f}s)')

    t1 = time.time()
    exe = build_harness(work, [p[3] for p in picks], args.jobs, args.cflags)
    print(f'compiled ({time.time() - t1:.0f}s)')

    emu = Emu([(base, data) for base, _, data in images])
    mu = emu.mu
    buf0 = bytes(r.getrandbits(8) for _ in range(4096)) * (BUF_SIZE // 4096)
    mu.mem_write(BUF, buf0)
    cases, expect = [], []
    t2 = time.time()
    for k, (name, w, pc, _) in enumerate(picks):
        mem = is_mem(w)
        for _ in range(args.cases):
            fpcr, x, sp, nzcv, v = make_case(r, w)
            try:
                ox, osp, onz, opc, ov = emu.run(pc, fpcr, x, sp, nzcv, v)
            except uc.UcError:
                if mem:
                    mu.mem_write(BUF, buf0)
                continue
            mh = 0
            if mem:
                mh = zlib.crc32(bytes(mu.mem_read(BUF, BUF_SIZE)))
                mu.mem_write(BUF, buf0)
            cases.append((k, fpcr, x, sp, nzcv, v, mem))
            expect.append((ox, osp, onz, opc, ov, mh))
    print(f'unicorn: {len(cases)} cases ({time.time() - t2:.0f}s)')

    t3 = time.time()
    recs, err = run_harness(exe, work, buf0, cases, [(base, img) for base, img, _ in images])
    print(f'harness: {len(recs)} of {len(cases)} cases ({time.time() - t3:.0f}s)')
    if err:
        k = cases[len(recs)][0] if len(recs) < len(cases) else None
        where = f', stopped at {picks[k][1]:08x} {dis(picks[k][1])}' if k is not None else ''
        print(f'harness failed: {err}{where}')

    fails, nfail = {}, 0
    for case, want, got in zip(cases, expect, recs):
        k = case[0]
        name, w, pc, _ = picks[k]
        d = diff_state(got, want, check_pc=got[3] != 0, check_mem=case[6])
        if d:
            nfail += 1
            fails.setdefault(w, (name, pc, d[:4]))

    print('unsupported encodings (distinct words) by module and mnemonic:')
    for (name, mn), n in sorted(unsupported.items(), key=lambda kv: (kv[0][0], -kv[1])):
        print(f'  {name:8} {mn:10} {n}')
    print(f'{len(picks)} encodings, {len(cases)} cases: {nfail} failing cases, {len(fails)} failing encodings '
          f'({time.time() - t0:.0f}s total)')
    if fails:
        print('  failing by mnemonic:', dict(Counter(mnemonic(w) for w in fails).most_common(40)))
    for w, (name, pc, d) in list(fails.items())[:args.show]:
        print(f'  {name:8} {w:08x} {dis(w, pc):40} {"; ".join(d)}')
    return 1 if fails or err else 0


if __name__ == '__main__':
    sys.exit(main())
