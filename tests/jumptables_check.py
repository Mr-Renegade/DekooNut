"""Check of recomp/jumptables.py on every `br` in the game's modules against an emulation oracle.

The oracle is an emulated jump-table method: find `cmp #N; b.hi/b.hs` before the br (or a `b.ls/b.lo` to the
dispatch block), replay ADRP/ADR/ADD-imm from the function entry, then single-step Unicorn from the bounds check to
the br once per index. Functions are [seed, next seed) over the module's seeds plus BL targets.

Reports per module: br sites, resolved by the resolver / by the oracle, agreement, `is_computed` sites left
unresolved, tables whose targets leave the function, an independent check of hoisted table bases (hoisted_base),
PLT stubs (`br x17` after the GOT load) that must stay a plain call, and resolver time. A disagreement is counted
as 'oracle wrong' only with independent evidence (oracle_wrong). Samples of every disagreement, unverified base and
unresolved computed site are printed with disassembly. Output also goes to build/logs/jumptables-check.log, and one
row per table-like br to build/logs/jumptables-<module>.tsv.

The oracle is adapted from LAS Recomp tools/gen.py @ da35b83 (MIT, Copyright (c) 2026 Peiikai).

Exits with 1 if any disagreement is unexplained, a hoisted base is not verified, or a PLT stub counts as computed.

Usage: jumptables_check.py [--exefs DIR] [--modules main,sdk,subsdk0] [--samples N] [--no-oracle]
"""
import argparse
import bisect
import sys
import time
from collections import Counter
from pathlib import Path

import capstone
import numpy as np
import unicorn as uc
from unicorn import arm64_const as A

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'recomp'))

import jumptables  # noqa: E402
import layout  # noqa: E402
from lifter import bits, sext  # noqa: E402
from module import Module  # noqa: E402

SCRATCH_MEM = 0x7F00000000
LOG = ROOT / 'build' / 'logs' / 'jumptables-check.log'

_md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)


class Oracle:
    """The Unicorn jump-table resolver."""

    def __init__(self, mod):
        self.m = mod
        mu = uc.Uc(uc.UC_ARCH_ARM64, uc.UC_MODE_ARM)
        mu.mem_map(mod.base, len(mod.image))
        mu.mem_write(mod.base, mod.image)
        mu.mem_map(SCRATCH_MEM, 0x100000)
        mu.reg_write(A.UC_ARM64_REG_CPACR_EL1, 0x300000)
        self.mu = mu
        self.set_at = self.regs = None
        self.start = self.cmp = None   # the first instruction of the bounds-check block, the bounds cmp
        self.guard = None              # _taken: the b.ls/b.lo into the dispatch block (set_at)
        self.count = None              # indices the bounds check lets through
        self.exclude = frozenset()     # indices the emulation skips (oracle_wrong: excluded on every way in)
        self.watch = self.seen = None  # (address, register): record the register when the emulation reaches address

    def word(self, a):
        return self.m.word(a)

    def kind(self, a):
        return self.m.lift(a).kind

    def jump_table(self, br_pc, entry, end):
        """Targets, or None. Leaves the index set point and the registers set there in self.set_at, self.regs."""
        self.set_at = self.regs = self.start = self.cmp = self.guard = self.count = None
        r = self._fallthrough(br_pc, entry, end)
        if r is None:
            r = self._taken(br_pc, entry, end)
        return r

    def _index_regs(self, start, at, idx):
        def copy_src(w):
            if (w & 0x7F800000) == 0x12000000:  # AND (imm)
                return bits(w, 5, 5)
            if (w & 0x7FE0FFE0) == 0x2A0003E0:  # MOV (ORR with zr)
                return bits(w, 16, 5)
            if (w & 0x7F800000) in (0x53000000, 0x13000000):  # UBFM/SBFM
                return bits(w, 5, 5)
            return None
        regs = {idx}
        changed = True
        while changed:
            changed = False
            for a in range(start, at, 4):
                w = self.word(a)
                rd, src = w & 31, copy_src(w)
                if src is None or src >= 31 or rd >= 31:
                    continue
                for x, y in ((rd, src), (src, rd)):
                    if x in regs and y not in regs:
                        regs.add(y)
                        changed = True
        return sorted(regs)

    def _reset(self, entry, start):
        mu = self.mu
        for r in range(29):
            mu.reg_write(A.UC_ARM64_REG_X0 + r, SCRATCH_MEM + 0x80000)
        mu.reg_write(A.UC_ARM64_REG_SP, SCRATCH_MEM + 0x80000)
        a = entry
        while a < start:
            w = self.word(a)
            if (w & 0x1F000000) == 0x10000000 or (w & 0xFF800000) == 0x91000000:
                mu.emu_start(a, 0, count=1)
            a += 4

    def _step_to(self, frm, to):
        mu, pc = self.mu, frm
        for _ in range(64):
            if self.watch and pc == self.watch[0] and self.seen is None:
                self.seen = self._read(self.watch[1])
            if pc == to:
                return True
            mu.emu_start(pc, 0, count=1)
            pc = mu.reg_read(A.UC_ARM64_REG_PC)
        return False

    def _read(self, rn):
        return self.mu.reg_read(A.UC_ARM64_REG_X0 + rn if rn < 29 else
                                (A.UC_ARM64_REG_X29 if rn == 29 else A.UC_ARM64_REG_X30))

    def _emulate(self, br_pc, entry, end, start, regs, count, set_at):
        rn = bits(self.word(br_pc), 5, 5)
        targets = []
        try:
            for i in range(count):
                if i in self.exclude:
                    continue
                self._reset(entry, start)
                if set_at != start and not self._step_to(start, set_at):
                    return None
                for r in regs:
                    self.mu.reg_write(A.UC_ARM64_REG_X0 + r if r < 29 else A.UC_ARM64_REG_X29, i)
                if not self._step_to(set_at, br_pc):
                    return None
                t = self._read(rn)
                if not (entry <= t < end) or t % 4:
                    return None
                targets.append(t)
        except uc.UcError:
            return None
        return sorted(set(targets))

    def _taken(self, br_pc, entry, end):
        s = br_pc
        while s - 4 >= entry and br_pc - (s - 4) <= 4 * 12 and self.kind(s - 4) in ('op', 'asm'):
            s -= 4
        for g in range(entry, end, 4):
            w = self.word(g)
            if (w & 0xFF000010) != 0x54000000 or bits(w, 0, 4) not in (9, 3):
                continue
            if g + sext(bits(w, 5, 19), 19) * 4 != s:
                continue
            cw = self.word(g - 4) if g - 4 >= entry else 0
            if (cw & 0x7F80001F) != 0x7100001F or bits(cw, 22, 1):
                continue
            limit = bits(cw, 10, 12)
            count = limit + 1 if bits(w, 0, 4) == 9 else limit
            if count <= 0 or count > 4096:
                continue
            self.set_at, self.regs, self.guard, self.cmp, self.count = s, [bits(cw, 5, 5)], g, g - 4, count
            b = g - 4
            while b - 4 >= entry and g - 4 - (b - 4) <= 4 * 12 and self.kind(b - 4) in ('op', 'asm'):
                b -= 4
            self.start = b
            return self._emulate(br_pc, entry, end, s, [bits(cw, 5, 5)], count, s)
        return None

    def _fallthrough(self, br_pc, entry, end):
        a = br_pc - 4
        guard = cmp = None
        while a >= entry and br_pc - a <= 4 * 16:
            w = self.word(a)
            if (w & 0xFF000010) == 0x54000000 and bits(w, 0, 4) in (8, 2):
                guard = a
                break
            if self.kind(a) in ('b', 'bl', 'blr', 'br', 'ret', 'trap'):
                return None
            a -= 4
        if guard is None:
            return None
        a = guard - 4
        while a >= entry and guard - a <= 4 * 8:
            if (self.word(a) & 0x7F80001F) == 0x7100001F:
                cmp = a
                break
            if self.kind(a) != 'op':
                return None
            a -= 4
        if cmp is None:
            return None
        w = self.word(cmp)
        if bits(w, 22, 1):
            return None
        limit = bits(w, 10, 12)
        count = limit + 1 if bits(self.word(guard), 0, 4) == 8 else limit
        if count <= 0 or count > 4096:
            return None
        start = cmp
        while start - 4 >= entry and cmp - (start - 4) <= 4 * 12 and self.kind(start - 4) in ('op', 'asm'):
            start -= 4
        regs = self._index_regs(start, cmp, bits(w, 5, 5))
        self.set_at, self.regs, self.start, self.cmp, self.count = cmp, regs, start, cmp, count
        rn = bits(self.word(br_pc), 5, 5)
        targets = []
        try:
            for i in range(count):
                if i in self.exclude:
                    continue
                self._reset(entry, start)
                if not self._step_to(start, cmp):
                    return None
                for r in regs:
                    self.mu.reg_write(A.UC_ARM64_REG_X0 + r if r < 29 else A.UC_ARM64_REG_X29, i)
                if not self._step_to(cmp, br_pc):
                    return None
                t = self._read(rn)
                if not (entry <= t < end) or t % 4:
                    return None
                targets.append(t)
        except uc.UcError:
            return None
        return sorted(set(targets))


_mdd = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
_mdd.detail = True
CALL_CLOBBERED = set(range(19)) | {30}


def _insn(mod, a):
    return next(_mdd.disasm(mod.image[a - mod.base:a - mod.base + 4], a), None)


def _regno(name):
    """X register number of a capstone register name (w and x views), 31 for sp/zr, None for others."""
    if name in ('fp', 'x29', 'w29'):
        return 29
    if name in ('lr', 'x30', 'w30'):
        return 30
    if name in ('sp', 'wsp', 'xzr', 'wzr'):
        return 31
    return int(name[1:]) if name[0] in 'xw' and name[1:].isdigit() else None


def _writes(ins):
    """X register numbers an instruction writes, by capstone (w and x views folded), plus the call clobbers."""
    if ins is None:
        return set(range(31))
    if ins.mnemonic in ('cmp', 'cmn', 'tst', 'ccmp', 'ccmn'):
        return set()  # capstone lists the first operand of these aliases as written
    try:
        _, w = ins.regs_access()
    except capstone.CsError:
        return set(range(31))
    out = set()
    for r in w:
        n = _regno(ins.reg_name(r))
        if n is not None and n < 31:
            out.add(n)
    if ins.mnemonic in ('bl', 'blr'):
        out |= CALL_CLOBBERED
    return out


class Reaching:
    """Reaching definitions of single registers over the resolver's blocks, with capstone's write sets."""

    def __init__(self, mod, cfg):
        self.mod, self.cfg, self.cache = mod, cfg, {}
        self.writes = {}
        for b, stop in zip(cfg.leaders, cfg.stops):
            for x in range(b, stop, 4):
                self.writes[x] = _writes(_insn(mod, x))

    def at_blocks(self, reg):
        """{block start: definitions of reg reaching it}; 'entry' for a value from outside the function."""
        if reg in self.cache:
            return self.cache[reg]
        cfg = self.cfg
        gen = {}
        for b, stop in zip(cfg.leaders, cfg.stops):
            gen[b] = next((x for x in range(stop - 4, b - 4, -4) if reg in self.writes[x]), None)
        ins = {b: {'entry'} if b in cfg.unknown else set() for b in cfg.leaders}
        changed = True
        while changed:
            changed = False
            for b in cfg.leaders:
                new = set(ins[b])
                for src, _ in cfg.preds[b]:
                    p = cfg.block(src)
                    new |= {gen[p]} if gen[p] is not None else ins[p]
                if new != ins[b]:
                    ins[b], changed = new, True
        self.cache[reg] = ins
        return ins

    def at(self, reg, a):
        """Definitions of reg reaching instruction a."""
        b = self.cfg.block(a)
        last = next((x for x in range(a - 4, b - 4, -4) if reg in self.writes[x]), None)
        return {last} if last is not None else self.at_blocks(reg)[b]

    def values(self, reg, a, active=None):
        """Constants reg holds at a, following adrp/add/adr and register copies; None if one is not constant.

        A definition already being evaluated (a cycle of copies) adds no value of its own.
        """
        active = set() if active is None else active
        out = set()
        for d in self.at(reg, a):
            if d == 'entry':
                return None
            if (reg, d) in active:
                continue
            active.add((reg, d))
            ins = _insn(self.mod, d)
            ops = ins.operands
            if ins.mnemonic in ('adrp', 'adr'):
                out.add(ops[1].imm)
            elif ins.mnemonic == 'add' and len(ops) == 3 and ops[2].type == capstone.arm64.ARM64_OP_IMM \
                    and _regno(ins.reg_name(ops[1].reg)) == reg:
                prev = self.values(reg, d, active)
                if prev is None:
                    return None
                imm = ops[2].imm << (ops[2].shift.value if ops[2].shift.type else 0)
                out |= {v + imm for v in prev}
            elif ins.mnemonic == 'mov' and len(ops) == 2 and ops[1].type == capstone.arm64.ARM64_OP_REG \
                    and ins.reg_name(ops[1].reg)[0] in 'xlf' and (_regno(ins.reg_name(ops[1].reg)) or 0) < 31:
                prev = self.values(_regno(ins.reg_name(ops[1].reg)), d, active)
                if prev is None:
                    return None
                out |= prev
            else:
                return None
        return out


class FunctionCheck:
    """The resolver's final tables and blocks for one function, with reaching definitions on demand."""

    def __init__(self, mod, entry, end):
        self.mod, self.entry, self.end = mod, entry, end
        self.tables = {a: t for a, t in jumptables._function(mod, entry, end).items() if t is not None}
        self.cfg = jumptables._Cfg(mod, entry, end, self.tables, {})
        self._reaching = self._facts = None

    @property
    def reaching(self):
        if self._reaching is None:
            self._reaching = Reaching(self.mod, self.cfg)
        return self._reaching

    @property
    def facts(self):
        if self._facts is None:
            self._facts = jumptables._facts(self.cfg) or {}
        return self._facts


def table_load(mod, s, br):
    """Last register-offset load in [s, br): the table read of a dispatch block."""
    a = br
    while a > s:
        a -= 4
        ins = _insn(mod, a)
        if ins and ins.mnemonic.startswith('ldr') and len(ins.operands) == 2 \
                and ins.operands[1].type == capstone.arm64.ARM64_OP_MEM and ins.operands[1].mem.index:
            return ins
    return None


def hoisted_base(fc, br):
    """(base register, its single constant at the dispatch block or None, verdict) of the table read before br.

    The verdict is 'local' when the dispatch block writes the base, 'verified' when every definition reaching the
    block (over the resolver's blocks and table edges, with capstone's write sets, through register copies) is an
    adrp/add or adr of one and the same constant and the resolver's fact does not name another, else a reason.
    """
    mod, s = fc.mod, fc.cfg.block(br)
    load = table_load(mod, s, br)
    if load is None:
        return None, None, 'no table load'
    reg = _regno(load.reg_name(load.operands[1].mem.base))
    if reg is None or reg == 31 or any(reg in _writes(_insn(mod, x)) for x in range(s, load.address, 4)):
        return reg, None, 'local'
    values = fc.reaching.values(reg, s)
    if values is None or len(values) != 1:
        return reg, None, 'not one constant' if values else 'not constant'
    value = values.pop()
    fact = fc.facts.get(s, {}).get(reg)
    if fact is not None and fact != 'z' and fact[0] == 'c' and fact[1] != value:
        return reg, value, 'facts disagree'
    return reg, value, 'verified'


def _reads(ins):
    try:
        r, _ = ins.regs_access()
    except capstone.CsError:
        return set(range(32))
    return {n for n in (_regno(ins.reg_name(x)) for x in r) if n is not None}


def _branch_target(ins):
    """Target of a direct b, b.cond, cbz/cbnz or tbz/tbnz, else None."""
    if ins is None or not ins.operands or not (ins.mnemonic in ('b', 'cbz', 'cbnz', 'tbz', 'tbnz')
                                               or ins.mnemonic.startswith('b.')):
        return None
    op = ins.operands[-1]
    return op.imm if op.type == capstone.arm64.ARM64_OP_IMM else None


def _ways_in(fc, lo, hi, tables):
    """Direct branches of the function into [lo, hi], or None when the range is entered otherwise too: by falling
    through into lo, at the function entry, or as a target of another table."""
    mod = fc.mod
    if lo <= fc.entry or any(lo <= t <= hi for ts in tables.values() if ts != 'all' for t in ts):
        return None
    prev = _insn(mod, lo - 4)
    if prev is None or prev.mnemonic not in ('b', 'br', 'ret'):
        return None
    out = []
    for a in range(fc.entry, fc.end, 4):
        ins = _insn(mod, a)
        t = _branch_target(ins)
        if t is not None and lo <= t <= hi:
            out.append((a, ins, t))
    return out


# Instructions an index slice may hold: integer arithmetic that reads only general registers and writes the flags at
# most (the flags-reading forms adc/sbc/csel/ccmp/... stay out).
_FLAG_SETTERS = {'cmp', 'cmn', 'tst', 'ands', 'bics', 'adds', 'subs', 'negs'}
_PURE_ALU = {'mov', 'movz', 'movn', 'movk', 'add', 'sub', 'and', 'orr', 'eor', 'bic', 'orn', 'eon', 'lsl', 'lsr',
             'asr', 'ror', 'ubfx', 'sbfx', 'ubfiz', 'sbfiz', 'bfi', 'bfxil', 'uxtb', 'uxth', 'uxtw', 'sxtb', 'sxth',
             'sxtw', 'neg', 'mvn', 'mul', 'madd', 'msub', 'udiv', 'sdiv', 'ubfm', 'sbfm', 'bfm'} | _FLAG_SETTERS
_DEPTH = 4   # predecessor blocks above the dispatch region


def _ends_block(ins):
    return ins is None or _branch_target(ins) is not None or ins.mnemonic in ('br', 'ret', 'brk', 'udf')


class _Ways:
    """Direct branches and fall-throughs of one function, with the addresses control may enter from elsewhere."""

    def __init__(self, fc, tables):
        mod = self.mod = fc.mod
        self.entry, self.into = fc.entry, {}
        for a in range(fc.entry, fc.end, 4):
            ins = _insn(mod, a)
            t = _branch_target(ins)
            if t is not None and fc.entry <= t < fc.end:
                self.into.setdefault(t, []).append((a, ins))
        self.roots = {fc.entry} | set(fc.cfg.unknown)
        self.roots |= {t for ts in tables.values() if ts != 'all' for t in ts}
        self.leaders = self.roots | set(self.into)

    def block_start(self, a):
        s = a
        while s > self.entry and s not in self.leaders and not _ends_block(_insn(self.mod, s - 4)):
            s -= 4
        return s

    def preds(self, s):
        """(branch or last instruction, insn, taken) for every way into s."""
        out = [(a, ins, True) for a, ins in self.into.get(s, ())]
        if s > self.entry:
            p = _insn(self.mod, s - 4)
            if p is not None and p.mnemonic not in ('b', 'br', 'ret', 'brk', 'udf'):
                out.append((s - 4, p, False))
        return out


def _edge_holds(oracle, ways, start, src, ins, taken, reg, k):
    """Whether the way out of the block [start, src] over the branch at src can be taken with x<reg> = k at src:
    False only when the branch condition is a function of x<reg> alone (its slice, back to the last write of x<reg>,
    is plain arithmetic and flag setting on x<reg>), evaluated by Unicorn; True otherwise."""
    if _branch_target(ins) is None or ins.mnemonic == 'b':
        return True
    mod = ways.mod

    def gp_reads(i):   # general registers read, zr dropped; None when sp or a non-general register is read
        try:
            r, _ = i.regs_access()
        except capstone.CsError:
            return None
        if i.mnemonic in ('cmp', 'cmn', 'tst'):   # capstone lists their first operand as written, not read
            r = list(r) + [op.reg for op in i.operands if op.type == capstone.arm64.ARM64_OP_REG]
        out = set()
        for x in r:
            name = i.reg_name(x)
            if name in ('xzr', 'wzr', 'nzcv'):
                continue
            n = _regno(name)
            if n is None or n == 31:
                return None
            out.add(n)
        return out

    def sets_flags(i):
        try:
            return i.mnemonic in _FLAG_SETTERS or 'nzcv' in (i.reg_name(r) for r in i.regs_access()[1])
        except capstone.CsError:
            return True

    if ins.mnemonic.startswith('b.'):
        setter = next((x for x in range(src - 4, start - 4, -4) if sets_flags(_insn(mod, x))), None)
        if setter is None or _insn(mod, setter).mnemonic not in _FLAG_SETTERS                 or any(reg in _writes(_insn(mod, x)) for x in range(setter, src, 4)):
            return True
        need, top, sl = gp_reads(_insn(mod, setter)), setter, [setter]
    else:
        need, top, sl = gp_reads(ins), src, []
    if need is None:
        return True
    for x in range(top - 4, start - 4, -4):
        i = _insn(mod, x)
        w = _writes(i)
        if reg in w:
            break   # x<reg> is k from here on
        if not w & need:
            continue
        r = gp_reads(i)
        if i.mnemonic not in _PURE_ALU or r is None:
            return True
        sl.append(x)
        need = (need - w) | r
    if need - {reg}:
        return True
    mu = oracle.mu
    try:
        for r in range(29):
            mu.reg_write(A.UC_ARM64_REG_X0 + r, SCRATCH_MEM + 0x80000)
        mu.reg_write(A.UC_ARM64_REG_X0 + reg if reg < 29 else A.UC_ARM64_REG_X29, k)
        for x in reversed(sl):
            mu.emu_start(x, 0, count=1)
        mu.emu_start(src, 0, count=1)
        went = mu.reg_read(A.UC_ARM64_REG_PC) != src + 4
    except uc.UcError:
        return True
    return went == taken


def _way_live(oracle, ways, src, ins, taken, reg, k, depth, seen):
    """Whether the way over src can carry x<reg> = k: its edge holds (_edge_holds) and its block either writes x<reg>
    or is entered with k (_reaches)."""
    start = ways.block_start(src)
    if not _edge_holds(oracle, ways, start, src, ins, taken, reg, k):
        return False
    if any(reg in _writes(_insn(ways.mod, x)) for x in range(start, src + 4, 4)):
        return True
    return _reaches(oracle, ways, start, reg, k, depth, seen)


def _reaches(oracle, ways, s, reg, k, depth, seen):
    """Whether control can enter s with x<reg> = k, over at most `depth` predecessor blocks."""
    if s in ways.roots or depth == 0 or s in seen:
        return True
    seen = seen | {s}
    return any(_way_live(oracle, ways, src, ins, taken, reg, k, depth - 1, seen)
               for src, ins, taken in ways.preds(s))


def excluded_indices(fc, br, oracle, tables, count):
    """Indices < count that cannot reach br: no way into the dispatch region [oracle.start, br] (its first instruction
    and every branch target inside it) can carry them. The index is the register of the oracle's bounds cmp, which no
    instruction from such an entry to br may write. Ways in are the module's direct branches and fall-throughs by
    capstone; the function entry, exception landing pads and every resolver table target count as entries from
    anywhere. Branch conditions are evaluated only where they depend on the index alone (_edge_holds). When the
    oracle enters the dispatch block by the taken bounds branch (_taken), that branch must be its only way in."""
    mod, lo = fc.mod, oracle.start
    if lo is None:
        return frozenset()
    if oracle.guard is not None:
        ways = _ways_in(fc, oracle.set_at, br, tables)
        if ways is None or [a for a, _, _ in ways] != [oracle.guard]:
            return frozenset()
    reg = bits(oracle.word(oracle.cmp), 5, 5)
    ways = _Ways(fc, tables)
    # The region falls through from lo to br (_fallthrough), or is the bounds-check block whose b.ls/b.lo is the only
    # way into the dispatch block (_taken).
    hi = br if oracle.guard is None else oracle.cmp
    if oracle.guard is not None and any(reg in _writes(_insn(mod, x)) for x in range(oracle.set_at, br, 4)):
        return frozenset()
    entries = [lo] + [t for t in range(lo + 4, hi + 4, 4) if t in ways.leaders]
    for e in entries:
        if e in ways.roots or any(reg in _writes(_insn(mod, x)) for x in range(e, hi, 4)):
            return frozenset()
    out = set()
    for k in range(count):
        live = False
        for e in entries:
            for src, ins, taken in ways.preds(e):
                if lo <= src < hi:
                    continue   # inside the region: its own conditions are ignored
                if _way_live(oracle, ways, src, ins, taken, reg, k, _DEPTH, frozenset()):
                    live = True
                    break
            if live:
                break
        if not live:
            out.add(k)
    return frozenset(out)


def oracle_wrong(fc, br, oracle, ours=None):
    """Why the oracle's targets for br are not the table's, from evidence independent of the resolver; or None.

    - the index the table is read with does not depend on the registers the oracle sets (it comes from another
      register or is reloaded from memory, which the oracle leaves at its scratch value);
    - the table base the oracle uses (its replay of ADRP/ADR/ADD-imm from the entry) is not the one constant that
      every reaching definition gives;
    - the oracle enumerates an index that no way into the dispatch region lets through (excluded_indices: a case
      peeled off before the table by `cmp wI, #k; b.ne`, `cbnz wI` or a range check on wI, possibly some blocks up),
      and without it the oracle gives the resolver's targets.
    """
    mod = fc.mod
    s = fc.cfg.block(br)
    load = table_load(mod, s, br)
    if load is None or oracle.set_at is None:
        return None
    live = {_regno(load.reg_name(load.operands[1].mem.index))}
    a = load.address
    while a > oracle.set_at:
        a -= 4
        ins = _insn(mod, a)
        w = _writes(ins)
        if live & w:
            live = (live - w) | _reads(ins)
    if not live & set(oracle.regs):
        return 'index not set by the oracle (live-in ' + ', '.join(f'x{r}' for r in sorted(live)) + ')'
    reg, value, verdict = hoisted_base(fc, br)
    if verdict == 'verified':
        oracle.watch = (load.address, reg)
        oracle.jump_table(br, fc.entry, fc.end)
        seen, oracle.watch, oracle.seen = oracle.seen, None, None
        if seen is not None and seen != value:
            return f'oracle base x{reg} = {seen:#x}, every reaching definition gives {value:#x}'
    if ours is not None:
        oracle.jump_table(br, fc.entry, fc.end)
        ex = excluded_indices(fc, br, oracle, fc.tables, oracle.count) if oracle.count else frozenset()
        if ex:
            oracle.exclude = ex
            try:
                again = oracle.jump_table(br, fc.entry, fc.end)
            finally:
                oracle.exclude = frozenset()
            if again == ours:
                return 'index ' + ', '.join(str(k) for k in sorted(ex)) + ' unreachable on every way in'
    return None


def plt_stubs(mod):
    """Addresses of `br x17` in `adrp x16; ldr x17, [x16, #o]; add x16, x16, #o; br x17` stubs."""
    w = mod.words
    m = ((w[3:] == 0xD61F0220) & ((w[:-3] & 0x9F00001F) == 0x90000010) & ((w[1:-2] & 0xFFC003FF) == 0xF9400211)
         & ((w[2:-1] & 0xFFC003FF) == 0x91000210))
    return (mod.lo + 4 * (np.nonzero(m)[0] + 3)).tolist()


def function_starts(mod):
    """Seeds plus BL targets in .text."""
    w = mod.words.astype(np.int64)
    pc = mod.lo + 4 * np.arange(len(w), dtype=np.int64)
    bl = (w & 0xFC000000) == 0x94000000
    t = pc[bl] + 4 * (((w[bl] & 0x3FFFFFF) ^ 0x2000000) - 0x2000000)
    starts = set(mod.seeds()) | {a for a in t.tolist() if mod.in_text(a)}
    return sorted(starts)


def disasm(mod, lo, hi, marks=()):
    out = []
    for ins in _md.disasm(mod.image[lo - mod.base:hi - mod.base], lo):
        tag = '>' if ins.address in marks else ' '
        out.append(f'    {tag}{ins.address - mod.base:8x}: {ins.mnemonic} {ins.op_str}')
    return out


def window(mod, br, entry, n=24):
    """Code before the br, and the bounds check that branches into its block if it sits elsewhere."""
    lo = max(entry, br - 4 * n)
    lines = disasm(mod, lo, br + 4, {br})
    labels = jumptables.branch_sources(mod, lo, br + 4)
    far = sorted({s for d, ss in labels.items() for s in ss if not lo <= s <= br})
    for s in far[:3]:
        lines.append(f'    ... reached from {s - mod.base:x}:')
        lines += disasm(mod, max(entry, s - 16), s + 4, {s})
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exefs', default=str(ROOT / 'build' / 'extracted' / 'oot111' / 'exefs'))
    ap.add_argument('--modules', default='main,sdk,subsdk0')
    ap.add_argument('--samples', type=int, default=8)
    ap.add_argument('--no-oracle', action='store_true')
    args = ap.parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG, 'w', encoding='utf-8')

    def out(s=''):
        print(s)
        log.write(s + '\n')

    bases = {n: b for n, b, _ in layout.layout(args.exefs)}
    failed = 0
    for name in args.modules.split(','):
        mod = Module(name, Path(args.exefs) / name, bases[name])
        starts = function_starts(mod)
        brs = (mod.lo + 4 * np.nonzero((mod.words & 0xFFFFFC1F) == 0xD61F0000)[0]).tolist()
        oracle = None if args.no_oracle else Oracle(mod)
        stats, t_res, t_orc = Counter(), 0.0, 0.0
        samples, fc = {}, None
        rows = open(LOG.with_name(f'jumptables-{name}.tsv'), 'w', encoding='utf-8')
        rows.write('br\tentry\tend\tcomputed\tresolver\toracle\tcategory\n')
        for br in brs:
            i = bisect.bisect_right(starts, br) - 1
            entry, end = starts[i], starts[i + 1] if i + 1 < len(starts) else mod.hi
            t0 = time.perf_counter()
            raw = jumptables.table_targets(mod, br, entry, end)
            ours = jumptables.resolve(mod, br, entry, end)
            comp = jumptables.is_computed(mod, br, entry)
            t1 = time.perf_counter()
            orc = oracle.jump_table(br, entry, end) if oracle else None
            t_res += t1 - t0
            t_orc += time.perf_counter() - t1
            stats['br'] += 1
            stats['computed'] += comp
            if raw is not None and ours is None:
                stats['targets outside function'] += 1
                samples.setdefault('targets outside function', []).append((br, entry, end, raw, orc))
            if (ours is not None or orc is not None) and (fc is None or (fc.entry, fc.end) != (entry, end)):
                fc = FunctionCheck(mod, entry, end)
            why = None
            if ours is not None:
                stats['resolved'] += 1
                stats['entries'] += len(ours)
                reg, _, verdict = hoisted_base(fc, br)
                stats[f'base {verdict}' if verdict in ('local', 'verified') else 'base not verified'] += 1
                if verdict not in ('local', 'verified'):
                    samples.setdefault('base not verified', []).append((br, entry, end, ours, orc))
                    rows.write(f'{br - mod.base:x}\t{entry - mod.base:x}\t{end - mod.base:x}\t-\t-\t-\t'
                               f'base x{reg}: {verdict}\n')
                if not comp:
                    stats['resolved, not computed'] += 1
                    samples.setdefault('resolved, not computed', []).append((br, entry, end, ours, orc))
            if orc is not None:
                stats['oracle resolved'] += 1
            if ours is not None and orc is not None:
                key = 'agree' if ours == orc else 'disagree'
            elif ours is not None:
                key = 'only resolver'
            elif orc is not None:
                key = 'only oracle'
            else:
                key = 'computed, unresolved' if comp else None
            if key in ('disagree', 'only oracle'):
                why = oracle_wrong(fc, br, oracle, ours)
                if why:
                    key = 'oracle wrong' if key == 'disagree' else 'only oracle, oracle wrong'
            if key or raw is not None:
                rows.write(f'{br - mod.base:x}\t{entry - mod.base:x}\t{end - mod.base:x}\t{int(comp)}\t'
                           f'{"-" if raw is None else len(raw)}\t{"-" if orc is None else len(orc)}\t{key}'
                           f'{": " + why if why else ""}\n')
            if key:
                stats[key] += 1
                if key != 'agree':
                    samples.setdefault(key, []).append((br, entry, end, ours if ours is not None else raw, orc, why))
        rows.close()
        plt = plt_stubs(mod)
        stats['PLT stubs'] = len(plt)
        stats['PLT stubs computed or resolved'] = sum(
            jumptables.is_computed(mod, a, a - 12) or jumptables.table_targets(mod, a, a - 12, a + 4) is not None
            for a in plt)
        out(f'== {name}: {len(starts)} functions')
        for k in ('br', 'computed', 'resolved', 'entries', 'oracle resolved', 'agree', 'disagree', 'oracle wrong',
                  'only resolver', 'only oracle', 'only oracle, oracle wrong', 'computed, unresolved',
                  'resolved, not computed', 'targets outside function', 'base local', 'base verified',
                  'base not verified', 'PLT stubs', 'PLT stubs computed or resolved'):
            out(f'  {k:32s} {stats[k]}')
        failed += stats['disagree'] + stats['base not verified'] + stats['PLT stubs computed or resolved']
        out(f'  {"resolver time":32s} {t_res:.2f} s')
        if oracle:
            out(f'  {"oracle time":32s} {t_orc:.2f} s')
        for key in ('disagree', 'only oracle', 'base not verified', 'oracle wrong', 'only oracle, oracle wrong',
                    'computed, unresolved', 'resolved, not computed', 'targets outside function', 'only resolver'):
            for br, entry, end, ours, orc, *why in samples.get(key, [])[:args.samples]:
                out(f'-- {key}: br {br - mod.base:x} in [{entry - mod.base:x}, {end - mod.base:x})'
                    + (f': {why[0]}' if why and why[0] else ''))
                fmt = lambda t: 'None' if t is None else f'{len(t)} [' + ' '.join(
                    f'{a - mod.base:x}' for a in t[:12]) + (' ...' if len(t) > 12 else '') + ']'
                out(f'   resolver {fmt(ours)}')
                out(f'   oracle   {fmt(orc)}')
                for line in window(mod, br, entry):
                    out(line)
    log.close()
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
