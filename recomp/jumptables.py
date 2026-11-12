"""Static jump-table resolution for computed `br`.

Clang lowers a dense switch to a bounds check (`cmp wI, #N; b.hi default`, or `b.ls` to a dispatch block placed
elsewhere), a table load (`ldrb/ldrh/ldrsw wO, [xT, xI, lsl #s]`) and an add to an anchor (`adr xB, L;
add xB, xB, wO, sxtb #2`, or the table address itself). The resolver walks back from the `br` over its block and the
predecessor blocks (_explore), evaluates each path forward symbolically (_run), and enumerates the values the path
conditions (bounds checks, ccmp chains, cbz/tbz) or the construction (masks, bit inserts, selects of constants)
allow for the unknowns the target depends on (_solve). Table entries come from .text/.rodata of the unrelocated
image. Anything it cannot prove is rejected.

Values are expression tuples over the registers at the top of the path. 32/64-bit views are normalised
(`lo(zx(e)) = e`, `uxtw`, `and #0xffffffff`, ...) so that the copies and extensions clang inserts compare equal.
Registers at a path top come from a function-wide must-analysis (_facts): constants, small constant sets, terms
over named values, zero-extension. That carries table bases clang hoists out of loops and index copies made
before them.

A function is resolved as a whole (_function): table targets are blocks and edges for the facts and the paths, and
the tables and the facts are iterated to a common fixpoint. Assumptions: control enters [entry, end) only at the
entry, at targets of direct branches (seen module-wide), at exception landing pads (.eh_frame LSDAs) and through
its own table jumps; an unresolved table never enters another switch between its bounds check and its `br`.
Work per function is capped (MAX_STEPS); past the cap nothing in the function is resolved.
"""
import bisect
import itertools
import weakref
from operator import is_not

import numpy as np

from lifter import bits, decode_bit_masks, sext

MAX_BLOCK = 64       # instructions in one block
MAX_DEPTH = 4        # predecessor blocks above the dispatch block
MAX_PATHS = 32       # paths explored per br
MAX_COUNT = 4096     # table entries
MAX_CANDS = 8        # candidates combined in pairs and triples
MAX_SET = 8          # constants a register fact may hold
MAX_STEPS = 1 << 19  # work units per function (instructions evaluated, blocks met, assignments tried)
MAX_ROUNDS = 8       # rounds of the per-function table/fact iteration

M32, M64 = 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF
HS, LO, HI, LS = 2, 3, 8, 9

_REGS = range(31)
_INDEX = weakref.WeakKeyDictionary()
_FUNCS = weakref.WeakKeyDictionary()
_PADS = weakref.WeakKeyDictionary()


def _mask(sz):
    return M32 if sz == 32 else M64


# ---------------------------------------------------------------- expressions

def C(v, sz=64):
    return ('c', v & _mask(sz))


def lo(e):
    """Low 32 bits of a 64-bit value."""
    k = e[0]
    if k == 'c':
        return ('c', e[1] & M32)
    if k in ('zx', 'sx'):
        return e[1]
    if k == 'oneof' and max(e[2]) <= M32:
        return e
    return ('lo', e)


def zx(e):
    return e if e[0] in ('c', 'oneof') else ('zx', e)


def sx(e):
    return C(sext(e[1], 32)) if e[0] == 'c' else ('sx', e)


def binop(op, sz, a, b):
    if a[0] == 'c' and b[0] == 'c':
        return C(_BIN[op](a[1], b[1], sz), sz)
    if op in ('add', 'orr', 'eor') and a[0] == 'c' and a[1] == 0:
        return b
    if op in ('add', 'sub', 'orr', 'eor') and b[0] == 'c' and b[1] == 0:
        return a
    if op == 'and' and b[0] == 'c' and sz == 64 and b[1] <= M32:
        return zx(binop('and', 32, lo(a), b)) if b[1] != M32 else zx(lo(a))
    if op == 'and' and b[0] == 'c' and b[1] == _mask(sz):
        return a
    return (op, sz, a, b)


def shift(op, sz, a, n):
    n %= sz
    if n == 0:
        return a
    if a[0] == 'c':
        return C(_SH[op](a[1], n, sz), sz)
    return (op, sz, a, n)


def bfm(sz, signed, immr, imms, x):
    """UBFM/SBFM of the sz-bit value x."""
    if not signed and immr == 0 and imms == 31 and sz == 64:
        return zx(lo(x))
    if signed and immr == 0 and imms == 31 and sz == 64:
        return sx(lo(x))
    if imms == sz - 1:
        return shift('asr' if signed else 'lsr', sz, x, immr)
    if not signed and imms + 1 == immr:
        return shift('lsl', sz, x, sz - immr)
    if not signed and immr == 0:
        return binop('and', sz, x, C((1 << (imms + 1)) - 1))
    if not signed and sz == 64 and immr <= imms < 32:
        return zx(bfm(32, False, immr, imms, lo(x)))
    e = ('bfm', sz, signed, immr, imms, x)
    return C(_bfm(sz, signed, immr, imms, x[1]), sz) if x[0] == 'c' else e


def bfi(sz, immr, imms, dst, src):
    """BFM (bfi, bfxil): bits of src inserted into dst."""
    if dst[0] == 'c' and src[0] == 'c':
        return C(_bfi(sz, immr, imms, dst[1], src[1]), sz)
    return ('bfi', sz, immr, imms, dst, src)


def _bfi(sz, immr, imms, d, v):
    wmask, tmask = decode_bit_masks(1 if sz == 64 else 0, imms, immr, sz, False)
    m = _mask(sz)
    bot = (d & ~wmask & m) | (_SH['ror'](v, immr, sz) if immr else v) & wmask
    return (d & ~tmask & m) | (bot & tmask)


def extend(e, option, sz):
    """Extended-register operand: option applied to the 64-bit register value e, result sz bits."""
    if option in (3, 7):
        return e if sz == 64 else lo(e)
    if option == 2:
        return zx(lo(e)) if sz == 64 else lo(e)
    if option == 6:
        return sx(lo(e)) if sz == 64 else lo(e)
    b, signed = 8 if option & 3 == 0 else 16, option >= 4
    v = lo(e)
    if v[0] == 'c':
        x = v[1] & ((1 << b) - 1)
        return C(sext(x, b) if signed else x, sz)
    return ('xt', b, signed, sz, v)


def _bfm(sz, signed, immr, imms, v):
    m = _mask(sz)
    if imms >= immr:
        w = imms - immr + 1
        f = (v >> immr) & ((1 << w) - 1)
        return (sext(f, w) if signed else f) & m
    f = v & ((1 << (imms + 1)) - 1)
    if signed:
        f = sext(f, imms + 1)
    return (f << (sz - immr)) & m


_BIN = {
    'add': lambda a, b, sz: (a + b) & _mask(sz),
    'sub': lambda a, b, sz: (a - b) & _mask(sz),
    'and': lambda a, b, sz: a & b,
    'orr': lambda a, b, sz: a | b,
    'eor': lambda a, b, sz: a ^ b,
    'bic': lambda a, b, sz: a & ~b & _mask(sz),
    'orn': lambda a, b, sz: (a | ~b) & _mask(sz),
    'eon': lambda a, b, sz: (a ^ ~b) & _mask(sz),
}
_SH = {
    'lsl': lambda a, n, sz: (a << n) & _mask(sz),
    'lsr': lambda a, n, sz: a >> n,
    'asr': lambda a, n, sz: (sext(a, sz) >> n) & _mask(sz),
    'ror': lambda a, n, sz: ((a >> n) | (a << (sz - n))) & _mask(sz),
}
_SHIFT_NAMES = ('lsl', 'lsr', 'asr', 'ror')


# ---------------------------------------------------------------- symbolic execution of one path

class _State:
    __slots__ = ('regs', 'flags', 'epoch', 'mem', 'conds', 'fresh')

    def __init__(self):
        self.regs = [('r', r) for r in range(32)]  # 31 is SP
        self.flags = None
        self.epoch = 0   # memory version: loads of different versions are different values
        self.mem = []    # [(base, offset, size, value)] stores of this version, disjoint
        self.conds = []   # conditions the branches on the path prove (see _edge)
        self.fresh = 0

    def unknown(self):
        self.fresh += 1
        return ('u', self.fresh)

    def get(self, r, sz, sp=False):
        if r == 31 and not sp:
            return C(0)
        e = self.regs[r]
        return e if sz == 64 else lo(e)

    def set(self, r, sz, e, sp=False):
        if r == 31 and not sp:
            return
        self.regs[r] = e if sz == 64 else zx(e)

    def kill(self, r, sz=64, sp=False):
        """r gets an unknown value; a 32-bit write leaves the upper half zero."""
        if r != 31 or sp:
            self.regs[r] = self.unknown() if sz == 64 else ('zx', self.unknown())

    def clobber(self):
        """Memory changed in an unknown way."""
        self.epoch += 1
        self.mem = []

    def kill_all(self):
        for r in range(32):
            self.regs[r] = self.unknown()
        self.flags = None
        self.clobber()

    def call(self):
        for r in list(range(19)) + [30]:
            self.regs[r] = self.unknown()
        self.flags = None
        self.clobber()

    def store(self, addr, size, value):
        """value (64-bit for 8 bytes, else 32-bit) stored at addr. Stores that may overlap are forgotten: same base
        and overlapping offsets, or a different base."""
        base, off = _split(addr)
        self.epoch += 1
        self.mem = [m for m in self.mem if m[0] == base and (m[1] + m[2] <= off or off + size <= m[1])]
        self.mem.append((base, off, size, value))

    def load(self, size, signed, sz, addr):
        """sz-bit value of the size-byte load at addr, forwarded from a store when the same bytes were stored."""
        base, off = _split(addr)
        for b, o, n, v in self.mem:
            if b == base and o == off and n == size:
                if size == 8:
                    return v
                if size == 4:
                    return sx(v) if signed and sz == 64 else v
                if signed and v[0] == 'c':
                    return C(sext(v[1] & ((1 << (8 * size)) - 1), 8 * size), sz)
                if signed:
                    return ('xt', 8 * size, True, sz, v)
                return binop('and', 32, v, C((1 << (8 * size)) - 1))
        return ('ld', size, signed, sz, addr, self.epoch)


def _split(addr):
    if addr[0] == 'add' and addr[3][0] == 'c':
        return addr[2], addr[3][1]
    if addr[0] == 'c':
        return ('c', 0), addr[1]
    return addr, 0


def _addr_imm(st, w, scale):
    """Base and offset of the load/store register (immediate and register offset) forms."""
    rn = bits(w, 5, 5)
    base = st.get(rn, 64, sp=True)
    if bits(w, 24, 1):  # unsigned offset
        return base, binop('add', 64, base, C(bits(w, 10, 12) << scale)), None
    if bits(w, 21, 1):  # register offset
        if bits(w, 10, 2) != 2:
            return None
        off = extend(st.get(bits(w, 16, 5), 64), bits(w, 13, 3), 64)
        if bits(w, 12, 1):
            off = shift('lsl', 64, off, scale)
        return base, binop('add', 64, base, off), None
    imm = sext(bits(w, 12, 9), 9)
    mode = bits(w, 10, 2)
    new = binop('add', 64, base, C(imm))
    if mode == 1:  # post-index
        return base, base, new
    if mode == 3:  # pre-index
        return base, new, new
    return base, new, None


def _ldst(st, w, pc):
    rt, rn = bits(w, 0, 5), bits(w, 5, 5)
    if (w & 0x3B000000) == 0x18000000:  # load literal
        opc, v = bits(w, 30, 2), bits(w, 26, 1)
        if not v and opc != 3:
            a = C(pc + sext(bits(w, 5, 19), 19) * 4)
            st.set(rt, 64, st.load(8 if opc == 1 else 4, opc == 2, 64, a) if opc else zx(st.load(4, False, 32, a)))
        return
    if (w & 0x3A000000) == 0x28000000:  # pair
        mode, load, v, opc = bits(w, 23, 2), bits(w, 22, 1), bits(w, 26, 1), bits(w, 30, 2)
        scale = 2 + (bits(w, 31, 1) if not v else opc)
        base = st.get(rn, 64, sp=True)
        new = binop('add', 64, base, C(sext(bits(w, 15, 7), 7) << scale))
        addr = base if mode == 1 else new
        rt2 = bits(w, 10, 5)
        if load and not v:
            sz = 64 if opc else 32
            st.kill(rt, sz)
            st.kill(rt2, sz)
        elif not load and not v and opc != 1:
            n = 1 << scale
            a, b = st.get(rt, 8 * n), st.get(rt2, 8 * n)
            st.store(addr, n, a)
            st.store(binop('add', 64, addr, C(n)), n, b)
        elif not load:
            st.clobber()
        if mode in (1, 3):
            st.set(rn, 64, new, sp=True)
        return
    if (w & 0x3B200C00) in (0x38000000, 0x38000400, 0x38000C00, 0x38200800) or (w & 0x3B000000) == 0x39000000:
        size, v, opc = bits(w, 30, 2), bits(w, 26, 1), bits(w, 22, 2)
        if v:
            scale = size | (bits(w, 23, 1) << 2)
            r = _addr_imm(st, w, scale)
            if r is None:
                st.kill_all()
                return
            if not (opc & 1):
                st.clobber()
            if r[2] is not None:
                st.set(rn, 64, r[2], sp=True)
            return
        r = _addr_imm(st, w, size)
        if r is None:
            st.kill_all()
            return
        _, addr, new = r
        if opc == 0:
            st.store(addr, 1 << size, st.get(rt, 64 if size == 3 else 32))
        elif opc == 1:
            st.set(rt, 64, zx(st.load(1 << size, False, 32, addr)) if size < 3 else st.load(8, False, 64, addr))
        elif size == 3:  # prfm
            pass
        elif opc == 2:
            st.set(rt, 64, st.load(1 << size, True, 64, addr))
        elif size < 2:
            st.set(rt, 32, st.load(1 << size, True, 32, addr))
        else:
            st.kill_all()
            return
        if new is not None:
            st.set(rn, 64, new, sp=True)
        return
    if (w & 0xBE000000) == 0x0C000000:  # SIMD structures (multiple and single)
        if not bits(w, 22, 1):
            st.clobber()
        if bits(w, 23, 1):
            st.kill(rn, sp=True)
        return
    sz = 64 if bits(w, 30, 2) == 3 else 32
    if (w & 0x3F000000) == 0x08000000:  # exclusives, acquire/release
        st.kill(rt, sz)
        st.kill(bits(w, 16, 5), 32)
        st.kill(bits(w, 10, 5), sz)
        st.clobber()
        return
    if (w & 0x3B200C00) == 0x38200000:  # LSE atomics
        st.kill(rt, sz)
        st.clobber()
        return
    st.kill_all()


def _dp_imm(st, w, pc):
    sf, rd, rn = bits(w, 31, 1), bits(w, 0, 5), bits(w, 5, 5)
    sz = 64 if sf else 32
    op = bits(w, 23, 3)
    if op in (0, 1):  # ADR, ADRP
        imm = sext((bits(w, 5, 19) << 2) | bits(w, 29, 2), 21)
        st.set(rd, 64, C((pc & ~0xFFF) + (imm << 12) if sf else pc + imm))
    elif op == 2:  # add/sub imm
        opc, s = bits(w, 30, 1), bits(w, 29, 1)
        imm = C(bits(w, 10, 12) << (12 if bits(w, 22, 1) else 0), sz)
        a = st.get(rn, sz, sp=True)
        r = binop('sub' if opc else 'add', sz, a, imm)
        if s:
            st.flags = ('sub' if opc else 'add', sz, a, imm)
        st.set(rd, sz, r, sp=not s)
    elif op == 4:  # logical imm
        opc, n = bits(w, 29, 2), bits(w, 22, 1)
        if not sf and n:
            st.kill_all()
            return
        imm, _ = decode_bit_masks(n, bits(w, 10, 6), bits(w, 16, 6), sz, True)
        r = binop(('and', 'orr', 'eor', 'and')[opc], sz, st.get(rn, sz), C(imm, sz))
        if opc == 3:
            st.flags = ('logic', sz, r)
        st.set(rd, sz, r, sp=opc != 3)
    elif op == 5:  # move wide
        opc, hw = bits(w, 29, 2), bits(w, 21, 2)
        imm = bits(w, 5, 16) << (hw * 16)
        if opc == 0:
            st.set(rd, sz, C(~imm, sz))
        elif opc == 2:
            st.set(rd, sz, C(imm, sz))
        elif opc == 3:
            old = st.get(rd, sz)
            st.set(rd, sz, C((old[1] & ~(0xFFFF << (hw * 16))) | imm, sz) if old[0] == 'c' else st.unknown())
        else:
            st.kill_all()
    elif op == 6:  # bitfield
        opc = bits(w, 29, 2)
        if opc == 1:  # BFM: bfi, bfxil
            st.set(rd, sz, bfi(sz, bits(w, 16, 6), bits(w, 10, 6), st.get(rd, sz), st.get(rn, sz)))
        elif opc == 3:
            st.kill(rd, sz)
        else:
            st.set(rd, sz, bfm(sz, opc == 0, bits(w, 16, 6), bits(w, 10, 6), st.get(rn, sz)))
    elif bits(w, 16, 5) == rn:  # ROR (EXTR with Rn = Rm)
        st.set(rd, sz, shift('ror', sz, st.get(rn, sz), bits(w, 10, 6)))
    else:
        st.kill(rd, sz)


def _dp_reg(st, w):
    sf, rd, rn, rm = bits(w, 31, 1), bits(w, 0, 5), bits(w, 5, 5), bits(w, 16, 5)
    sz = 64 if sf else 32
    if (w & 0x1F000000) == 0x0A000000:  # logical shifted register
        opc, n = bits(w, 29, 2), bits(w, 21, 1)
        b = shift(_SHIFT_NAMES[bits(w, 22, 2)], sz, st.get(rm, sz), bits(w, 10, 6))
        name = (('and', 'orr', 'eor', 'and'), ('bic', 'orn', 'eon', 'bic'))[n][opc]
        r = binop(name, sz, st.get(rn, sz), b)
        if opc == 3:
            st.flags = ('logic', sz, r)
        st.set(rd, sz, r)
    elif (w & 0x1F200000) == 0x0B000000:  # add/sub shifted register
        opc, s, typ = bits(w, 30, 1), bits(w, 29, 1), bits(w, 22, 2)
        if typ == 3:
            st.kill_all()
            return
        a = st.get(rn, sz)
        b = shift(_SHIFT_NAMES[typ], sz, st.get(rm, sz), bits(w, 10, 6))
        if s:
            st.flags = ('sub' if opc else 'add', sz, a, b)
        st.set(rd, sz, binop('sub' if opc else 'add', sz, a, b))
    elif (w & 0x1F200000) == 0x0B200000:  # add/sub extended register
        opc, s = bits(w, 30, 1), bits(w, 29, 1)
        a = st.get(rn, sz, sp=True)
        b = shift('lsl', sz, extend(st.get(rm, 64), bits(w, 13, 3), sz), bits(w, 10, 3))
        if s:
            st.flags = ('sub' if opc else 'add', sz, a, b)
        st.set(rd, sz, binop('sub' if opc else 'add', sz, a, b), sp=not s)
    elif (w & 0x1FE00000) == 0x1A400000 and bits(w, 29, 1):  # ccmp, ccmn
        b = C(rm, sz) if bits(w, 11, 1) else st.get(rm, sz)
        st.flags = ('ccmp', sz, st.get(rn, sz), b, not bits(w, 30, 1), bits(w, 12, 4), bits(w, 0, 4), st.flags)
    elif (w & 0x1FE00000) == 0x1A000000 and (w & 0xFC00):
        st.flags = None  # rmif, setf
    elif (w & 0x1FE00800) == 0x1A800000 and not bits(w, 29, 1):  # csel, csinc, csinv, csneg
        x, y = st.get(rn, sz), st.get(rm, sz)
        if bits(w, 10, 1):
            y = binop('add', sz, y, C(1)) if not bits(w, 30, 1) else binop('sub', sz, C(0), y)
        elif bits(w, 30, 1):
            y = binop('eor', sz, y, C(_mask(sz)))
        # Equal expressions are equal values, so a select carries what picks the arm: the condition and the flags
        # record, or a fresh unknown when the flags are not known (two `cset wN, ne` on different compares differ).
        sel = st.flags if st.flags is not None else st.unknown()
        st.set(rd, sz, x if x == y else ('csel', sz, x, y, bits(w, 12, 4), sel))
    else:  # adc, sbc, csel, 1/2/3-source
        st.kill(rd, sz)
        if bits(w, 29, 1):
            st.flags = None


def _fp_simd(st, w):
    if (w & 0x5F203C00) == 0x1E202000 or (w & 0x5F200C00) == 0x1E200400:  # fcmp, fccmp
        st.flags = None
    elif (w & 0x5F200000) == 0x1E000000 or (w & 0x5F20FC00) == 0x1E200000:  # conversions with a general register
        if bits(w, 16, 3) in (0, 1, 4, 5, 6):  # to a general register (fcvt*, fmov Rd, Vn)
            st.kill(bits(w, 0, 5), 64 if bits(w, 31, 1) else 32)
    elif (w & 0x9FE08400) == 0x0E000400 and bits(w, 11, 4) in (5, 7):  # smov, umov
        st.kill(bits(w, 0, 5), 64 if bits(w, 30, 1) else 32)


def _step(st, w, pc):
    """Effect of one non-branch instruction."""
    op0 = bits(w, 25, 4)
    if (op0 & 0b1110) == 0b1000:
        _dp_imm(st, w, pc)
    elif (op0 & 0b0101) == 0b0100:
        _ldst(st, w, pc)
    elif (op0 & 0b0111) == 0b0101:
        _dp_reg(st, w)
    elif (op0 & 0b0111) == 0b0111:
        _fp_simd(st, w)
    elif (w & 0xFC000000) == 0x94000000 or (w & 0xFFFFFC1F) == 0xD63F0000:  # bl, blr
        st.call()
    elif (w & 0xFFFFF01F) in (0xD503201F, 0xD503301F):  # hints, barriers
        pass
    elif (w & 0xFFF00000) == 0xD5300000:  # mrs
        st.kill(bits(w, 0, 5))
    elif (w & 0xFFF00000) == 0xD5100000:  # msr (system register): may write NZCV
        st.flags = None
    elif (w & 0xFFF80000) == 0xD5080000:  # sys: dc, ic, tlbi
        st.clobber()
    else:
        st.kill_all()


def _edge(st, w, taken):
    """Record the condition a conditional branch edge proves."""
    if (w & 0xFF000010) == 0x54000000:
        cond = bits(w, 0, 4)
        if cond < 14 and st.flags is not None:
            st.conds.append(('flags', st.flags, cond, taken))
    elif (w & 0x7E000000) == 0x34000000:  # cbz, cbnz
        sz = 64 if bits(w, 31, 1) else 32
        st.conds.append(('zero', sz, st.get(bits(w, 0, 5), sz), taken != bool(bits(w, 24, 1))))
    elif (w & 0x7E000000) == 0x36000000:  # tbz, tbnz
        bit = (bits(w, 31, 1) << 5) | bits(w, 19, 5)
        st.conds.append(('bit', st.get(bits(w, 0, 5), 64), bit, taken == bool(bits(w, 24, 1))))


def _init(st, facts):
    """Registers at the top of a path from the function-wide facts (see _facts)."""
    for r, v in facts.items():
        st.regs[r] = ('zx', ('lo', ('r', r))) if v == 'z' else ('oneof', r, v[1]) if v[0] == 'set' else v


class _Exhausted(Exception):
    """The per-function step budget ran out."""


_budget = [0]


def _spend(n):
    _budget[0] -= n
    if _budget[0] < 0:
        raise _Exhausted


def _run(mod, path, rn, facts, spend=True):
    """Symbolic state at the br that ends path [(start, last, taken)] (top block first) and the br register value.

    taken says which edge of a conditional branch at `last` the path follows; None for any other block end
    (fall-through, unconditional branch, table jump).
    """
    if spend:
        _spend(sum(last - start >> 2 for start, last, _ in path) + 1)
    st = _State()
    if facts:
        _init(st, facts)
    for start, last, taken in path[:-1]:
        for a in range(start, last, 4):
            _step(st, mod.word(a), a)
        w = mod.word(last)
        if taken is not None:
            _edge(st, w, taken)
        elif _branch_target(w, last) is None and not _is_br(w):
            _step(st, w, last)
    start, last, _ = path[-1]
    for a in range(start, last, 4):
        _step(st, mod.word(a), a)
    return st, st.get(rn, 64)


# ---------------------------------------------------------------- control flow

def _branch_target(w, pc):
    """Target of a direct branch (b, b.cond, cbz/cbnz, tbz/tbnz), else None."""
    if (w & 0xFC000000) == 0x14000000:
        return pc + sext(w & 0x3FFFFFF, 26) * 4
    if (w & 0xFF000010) == 0x54000000 or (w & 0x7E000000) == 0x34000000:
        return pc + sext(bits(w, 5, 19), 19) * 4
    if (w & 0x7E000000) == 0x36000000:
        return pc + sext(bits(w, 5, 14), 14) * 4
    return None


def _no_fallthrough(w):
    """b, b.al, br/ret/eret (branch-register forms other than blr), udf, brk, hlt."""
    return ((w & 0xFC000000) == 0x14000000 or (w & 0xFF00001E) == 0x5400000E
            or (w & 0xFE000000) == 0xD6000000 and (w & 0x01E00000) != 0x00200000
            or (w >> 16) == 0 or (w & 0xFFE0001F) in (0xD4200000, 0xD4400000))


def _ends_block(w):
    return _no_fallthrough(w) or (w & 0xFF000010) == 0x54000000 or (w & 0x7C000000) == 0x34000000


def _ends_block_np(w):
    return (((w & 0xFC000000) == 0x14000000) | ((w & 0xFF000010) == 0x54000000) | ((w & 0x7C000000) == 0x34000000)
            | (((w & 0xFE000000) == 0xD6000000) & ((w & 0x01E00000) != 0x00200000)) | ((w >> 16) == 0)
            | ((w & 0xFFE0001F) == 0xD4200000) | ((w & 0xFFE0001F) == 0xD4400000))


def _is_br(w):
    return (w & 0xFFFFFC1F) == 0xD61F0000


def _is_call(w):
    return (w & 0xFC000000) == 0x94000000 or (w & 0xFFFFFC1F) == 0xD63F0000


def _branch_index(mod):
    """(targets, sources) of every direct branch in .text, sorted by target; cached per module."""
    r = _INDEX.get(mod)
    if r is None:
        w = mod.words.astype(np.int64)
        pc = mod.lo + 4 * np.arange(len(w), dtype=np.int64)
        b = (w & 0xFC000000) == 0x14000000
        c = ((w & 0xFF000010) == 0x54000000) | ((w & 0x7E000000) == 0x34000000)
        t = (w & 0x7E000000) == 0x36000000
        off = np.zeros(len(w), dtype=np.int64)
        off[b] = ((w[b] & 0x3FFFFFF) ^ 0x2000000) - 0x2000000
        off[c] = (((w[c] >> 5) & 0x7FFFF) ^ 0x40000) - 0x40000
        off[t] = (((w[t] >> 5) & 0x3FFF) ^ 0x2000) - 0x2000
        m = b | c | t
        src, dst = pc[m], pc[m] + 4 * off[m]
        order = np.argsort(dst, kind='stable')
        r = _INDEX[mod] = (dst[order], src[order])
    return r


def branch_sources(mod, entry, end):
    """{target: [sources]} of the direct branches anywhere in .text that land in [entry, end)."""
    dst, src = _branch_index(mod)
    i, j = np.searchsorted(dst, [entry, end])
    out = {}
    for d, s in zip(dst[i:j].tolist(), src[i:j].tolist()):
        out.setdefault(d, []).append(s)
    return out


class _Cfg:
    """Basic blocks of [entry, end) with direct, fall-through and table edges.

    tables maps each table-shaped br to its targets so far, or None when it is unresolved and may reach any block.
    Leaders are the entry, direct branch targets, landing pads, table targets and the instruction after each
    block-ending instruction. `unknown` holds the blocks entered in ways the function does not show: the entry and
    targets of branches from other functions.
    """

    def __init__(self, mod, entry, end, tables, pads):
        self.mod, self.entry, self.end, self.pads = mod, entry, end, pads
        labels = branch_sources(mod, entry, end)
        words = mod.words[(entry - mod.lo) >> 2:(end - mod.lo) >> 2].astype(np.int64)
        after = (entry + 4 * np.nonzero(_ends_block_np(words))[0] + 4).tolist()
        targets = {t for ts in tables.values() if ts for t in ts if entry <= t < end}
        self.leaders = sorted({entry} | set(labels) | set(pads) | targets | {a for a in after if a < end})
        self.stops = self.leaders[1:] + [end]
        self.unknown = {entry} | {t for t, srcs in labels.items() if any(not entry <= a < end for a in srcs)}
        self.everywhere = [br for br, ts in tables.items() if ts is None]
        self.preds = {b: [] for b in self.leaders}  # [(last instruction of the predecessor, taken)]
        for b, stop in zip(self.leaders, self.stops):
            last = stop - 4
            w = mod.word(last)
            t = _branch_target(w, last)
            cond = (w & 0xFF000010) == 0x54000000 and (w & 0xE) != 0xE or (w & 0x7C000000) == 0x34000000
            if t is not None and entry <= t < end:
                self.preds[t].append((last, True if cond and t != stop else None))
            if not _no_fallthrough(w) and stop < end and t != stop:
                self.preds[stop].append((last, False if cond else None))
        for br, ts in tables.items():
            for t in ts or ():
                if entry <= t < end:
                    self.preds[t].append((br, None))

    def block(self, a):
        """Start of the block holding a."""
        return self.leaders[bisect.bisect_right(self.leaders, a) - 1]

    def path_preds(self, b):
        """[(start, last, taken)] predecessor segments of block b, or None if some entry is not visible.

        Unresolved table jumps are not counted: a table is assumed never to enter another switch's dispatch path
        after its bounds check.
        """
        if b in self.unknown or b in self.pads or not self.preds[b]:
            return None
        return [(self.block(src), src, taken) for src, taken in self.preds[b]]


def _facts(cfg):
    """{block start: {reg: fact}} for the registers whose value is known the same way on every path into the block.

    A fact is a constant, a small set of constants ('set', values), a term, or 'z' (a zero-extended 32-bit value).
    Terms are expressions over names ('v', A, r) / ('vw', A, r): the 64-bit / 32-bit value the most recent execution
    of instruction A wrote to r.
    They keep copies and extensions of one value equal across blocks (`mov w12, w8` before a loop, the check on w8
    inside it). A name dies when its instruction executes again.

    A forward must-analysis over cfg, optimistic for blocks not yet reached. An exception landing pad is entered from
    the calls of its call-site ranges with the caller-saved registers clobbered; an unresolved table jump may enter
    any block. None if the analysis does not settle.
    """
    mod, out, ins, calls, names = cfg.mod, {}, {}, {}, set()
    for _ in range(64):
        _spend(len(cfg.leaders))
        changed = False
        anywhere = None
        for br in cfg.everywhere:
            anywhere = _meet(anywhere, out.get(br))
        for b, stop in zip(cfg.leaders, cfg.stops):
            if b in cfg.unknown:
                d = {}
            else:
                d = None
                for src, _ in cfg.preds[b]:
                    d = _meet(d, out.get(src))
                for lo_, hi_ in cfg.pads.get(b, ()):
                    for a, c in calls.items():
                        if lo_ <= a < hi_:
                            d = _meet(d, c)
                d = _meet(d, anywhere)
                if d is None:
                    continue
            if ins.get(b) == d and stop - 4 in out:
                continue  # same input as last round: the transfer gives the same output
            ins[b] = d
            _spend(stop - b >> 2)
            st = _State()
            _init(st, d)
            regs = st.regs
            for a in range(b, stop, 4):
                w = mod.word(a)
                if _is_call(w):
                    c = {r: v for r, v in _facts_of(st).items() if 19 <= r < 30}
                    if calls.get(a) != c:
                        calls[a] = c
                        changed = True
                if a == stop - 4 and (_is_br(w) or _branch_target(w, a) is not None):
                    break
                before = regs[:31]
                _step(st, w, a)
                for r in itertools.compress(_REGS, map(is_not, regs, before)):
                    e = regs[r]
                    if e[0] != 'oneof' and not _clean(e):
                        regs[r] = ('zx', ('vw', a, r)) if e[0] == 'zx' else ('v', a, r)
                        names.add(a)
                if a in names:
                    for r in range(31):
                        if regs[r] is before[r] and _names(regs[r], a):
                            regs[r] = st.unknown()
            d = _facts_of(st)
            if out.get(stop - 4) != d:
                out[stop - 4] = d
                changed = True
        if not changed:
            return ins
    return None


def _clean(e, depth=8):
    """e is built from constants and names only, and is small (larger values get a name)."""
    k = e[0]
    if k in ('c', 'v', 'vw'):
        return True
    if k in ('r', 'u', 'ld', 'oneof') or not depth:
        return False
    return all(_clean(x, depth - 1) for x in e[1:] if isinstance(x, tuple))


def _names(e, a):
    """e mentions a name written by instruction a."""
    k = e[0]
    if k in ('v', 'vw'):
        return e[1] == a
    if k in ('c', 'r', 'u'):
        return False
    return any(_names(x, a) for x in e[1:] if isinstance(x, tuple))


def _facts_of(st):
    out = {}
    for r, e in enumerate(st.regs[:31]):
        if _clean(e):
            out[r] = e
        elif e[0] == 'oneof':
            out[r] = ('set', e[2])
        elif e[0] == 'zx':
            out[r] = 'z'
    return out


def _values(v):
    return frozenset((v[1],)) if v[0] == 'c' else v[1] if v[0] == 'set' else None


def _zext(v):
    return v == 'z' or v[0] == 'zx' or _values(v) is not None and max(_values(v)) <= M32


def _meet(a, b):
    if a is None:
        return b
    if b is None:
        return a
    out = {}
    for r, v in a.items():
        u = b.get(r)
        if u is None:
            continue
        if u == v:
            out[r] = v
            continue
        if u != 'z' and v != 'z':
            a_, b_ = _values(u), _values(v)
            if a_ is not None and b_ is not None and len(a_ | b_) <= MAX_SET:
                out[r] = ('set', a_ | b_)
                continue
        if _zext(u) and _zext(v):
            out[r] = 'z'
    return out


# ---------------------------------------------------------------- enumeration

class _Unknown(Exception):
    """The value depends on something the assignment does not fix (a register, or memory other than .text and
    .rodata)."""


def _eval(mod, e, env):
    """Value of e with the subexpressions in env [(expr, value)] fixed."""
    for x, v in env:
        if e is x:
            return v
    k = e[0]
    if k == 'c':
        return e[1]
    for x, v in env:
        if e == x:
            return v
    if k == 'lo':
        return _eval(mod, e[1], env) & M32
    if k == 'zx':
        return _eval(mod, e[1], env)
    if k == 'sx':
        return sext(_eval(mod, e[1], env), 32) & M64
    if k in _BIN:
        return _BIN[k](_eval(mod, e[2], env), _eval(mod, e[3], env), e[1])
    if k in _SH:
        return _SH[k](_eval(mod, e[2], env), e[3], e[1])
    if k == 'xt':
        _, b, signed, sz, x = e
        v = _eval(mod, x, env) & ((1 << b) - 1)
        return (sext(v, b) if signed else v) & _mask(sz)
    if k == 'bfm':
        return _bfm(e[1], e[2], e[3], e[4], _eval(mod, e[5], env))
    if k == 'bfi':
        return _bfi(e[1], e[2], e[3], _eval(mod, e[4], env), _eval(mod, e[5], env))
    if k == 'ld':
        _, size, signed, sz, addr, _ = e
        a = _eval(mod, addr, env)
        if not _const_mem(mod, a, size):
            raise _Unknown
        return mod.read(a, size, signed=signed) & _mask(sz)
    raise _Unknown


def _const_mem(mod, a, n):
    """[a, a+n) lies in .text or .rodata (no relocations, never written)."""
    for off, size in (mod.nso.text, mod.nso.rodata):
        if mod.base + off <= a and a + n <= mod.base + off + size:
            return True
    return False


def _holds(cond, f):
    n, z, c, v = f
    r = (z, c, n, v, c and not z, n == v, n == v and not z, True)[cond >> 1]
    return r != bool(cond & 1 and cond != 15)


def _nzcv(kind, sz, x, y):
    m, top = _mask(sz), sz - 1
    if kind == 'logic':
        return x >> top & 1, x == 0, 0, 0
    if kind == 'add':
        r = (x + y) & m
        return r >> top & 1, r == 0, x + y > m, (~(x ^ y) & (x ^ r)) >> top & 1
    r = (x - y) & m
    return r >> top & 1, r == 0, x >= y, ((x ^ y) & (x ^ r)) >> top & 1


def _flags(mod, f, env):
    """Possible NZCV values of the flags record f under env; None for one that is not determined."""
    k = f[0]
    if k == 'ccmp':
        _, sz, a, b, add, cond, imm, prev = f
        compare = fixed = False
        for p in _flags(mod, prev, env) if prev is not None else (None,):
            if p is None or _holds(cond, p):
                compare = True
            if p is None or not _holds(cond, p):
                fixed = True
        out = [(imm >> 3 & 1, imm >> 2 & 1, imm >> 1 & 1, imm & 1)] if fixed else []
        if compare:
            out += _flags(mod, ('add' if add else 'sub', sz, a, b), env)
        return out
    try:
        if k == 'logic':
            return [_nzcv(k, f[1], _eval(mod, f[2], env), 0)]
        return [_nzcv(k, f[1], _eval(mod, f[2], env), _eval(mod, f[3], env))]
    except _Unknown:
        return [None]


def _violated(mod, rec, env):
    """The path condition rec is provably false under env."""
    try:
        if rec[0] == 'flags':
            _, f, cond, holds = rec
            opts = _flags(mod, f, env)
            return None not in opts and all(_holds(cond, o) != holds for o in opts)
        if rec[0] == 'zero':
            _, sz, x, holds = rec
            return ((_eval(mod, x, env) & _mask(sz)) == 0) != holds
        _, x, bit, holds = rec
        return bool(_eval(mod, x, env) >> bit & 1) != holds
    except _Unknown:
        return False


_SWAP = {HS: LS, LS: HS, HI: LO, LO: HI, 10: 13, 13: 10, 11: 12, 12: 11, 0: 0, 1: 1}


def _bounds(st):
    """[(expr, values)] that the path conditions bound: x <= N, x < N, x == N, x == 0."""
    out = []
    for rec in st.conds:
        if rec[0] == 'zero':
            if rec[3]:
                out.append((rec[2], (0,)))
            continue
        if rec[0] != 'flags':
            continue
        _, f, cond, holds = rec
        if f[0] == 'ccmp':
            _, sz, a, b, add, ccond, imm, _ = f
            if add or _holds(cond, (imm >> 3 & 1, imm >> 2 & 1, imm >> 1 & 1, imm & 1)) == holds:
                continue  # the fixed flags alone could take this edge
            f = ('sub', sz, a, b)
        if f[0] != 'sub':
            continue
        _, sz, a, b = f
        c = cond if holds else cond ^ 1
        if a[0] == 'c' and b[0] != 'c':
            a, b, c = b, a, _SWAP.get(c)
        if b[0] != 'c' or a[0] == 'c':
            continue
        n = b[1]
        vals = range(n + 1) if c == LS else range(n) if c == LO else (n,) if c == 0 else None
        if vals is not None and 0 < len(vals) <= MAX_COUNT:
            out.append((lo(a) if a[0] in ('zx', 'sx') else a, vals))
    return out


def _subexprs(e, out):
    if e[0] in ('c', 'r', 'u', 'v', 'vw'):
        return
    out.append(e)
    if e[0] == 'oneof':
        return
    for x in e[1:]:
        if isinstance(x, tuple):
            _subexprs(x, out)


def _indices(e, out):
    """Subexpressions of the addresses of the loads in e (candidate table indices)."""
    if e[0] == 'ld':
        _subexprs(e[4], out)
    if e[0] not in ('c', 'r', 'u', 'v', 'vw'):
        for x in e[1:]:
            if isinstance(x, tuple):
                _indices(x, out)


def _domain(e):
    """Values e can take by construction (masks, narrow extracts, selects of constants), or None."""
    k = e[0]
    if k == 'and' and e[3][0] == 'c' and bin(e[3][1]).count('1') <= 10:
        m, out, x = e[3][1], [], 0
        while True:  # every submask of m
            out.append(x)
            x = (x - m) & m
            if not x:
                return out
    if k == 'bfm' and not e[2] and e[3] <= e[4] < e[3] + 10:
        return range(1 << (e[4] - e[3] + 1))
    if k == 'bfm' and not e[2] and e[4] < e[3] and e[4] < 8:  # ubfiz
        return tuple((v << (e[1] - e[3])) & _mask(e[1]) for v in range(1 << (e[4] + 1)))
    if k in ('orr', 'add', 'eor'):
        a, b = (_domain(x) if x[0] != 'c' else (x[1],) for x in e[2:])
        if a is None or b is None or len(a) * len(b) > 256:
            return None
        return tuple(sorted({_BIN[k](x, y, e[1]) for x in a for y in b}))
    if k == 'lsr' and e[3] > e[1] - 10:
        return range(1 << (e[1] - e[3]))
    if (k == 'xt' and not e[2]) or (k == 'ld' and not e[2] and e[1] == 1):
        return range(256)
    if k == 'csel':
        a, b = (_domain(x) if x[0] != 'c' else (x[1],) for x in e[2:4])
        return tuple(sorted(set(a) | set(b))) if a is not None and b is not None and len(a) + len(b) <= 256 else None
    if k in ('zx', 'lo'):
        d = _domain(e[1])
        return d if d is None or k == 'zx' else tuple(sorted({v & M32 for v in d}))
    if k == 'lsl':
        d = _domain(e[2])
        return None if d is None else tuple(sorted({(v << e[3]) & _mask(e[1]) for v in d}))
    if k == 'oneof':
        return tuple(sorted(e[2]))
    if k == 'bfi':
        _, sz, immr, imms, dst, _ = e
        dom = (dst[1],) if dst[0] == 'c' else _domain(dst)
        width, at = (imms - immr + 1, immr) if imms >= immr else (imms + 1, 0)
        if dom is None or width > 4 or len(dom) << width > 256:
            return None
        return tuple(sorted({_bfi(sz, immr, imms, d, i << at) for d in dom for i in range(1 << width)}))
    return None


def _solve(mod, t, st, masks):
    """Targets of t over the values the path proves for the unknowns in it, or None.

    Candidates are values bounded by path conditions and, with masks, values bounded by construction. One, two
    or three candidates must together fix t; every assignment the path conditions rule out is dropped.
    """
    try:
        return {_eval(mod, t, ())}
    except _Unknown:
        pass
    cands = _bounds(st)
    if masks:
        # A mask is a sound bound but may far exceed the table; with a bounds check on the path (of some other
        # value) it is only taken if no larger than that check.
        limit = min((len(v) for _, v in cands if len(v) > 1), default=MAX_COUNT)
        subs = []
        _indices(t, subs)
        for e in subs:
            d = _domain(e)
            if d is not None and len(d) <= limit:
                cands.append((e, d))
    seen, uniq = set(), []
    for e, d in sorted(cands, key=lambda c: len(c[1])):
        if e not in seen:
            seen.add(e)
            uniq.append((e, d))
    for k in (1, 2, 3):
        for combo in itertools.combinations(uniq if k == 1 else uniq[:MAX_CANDS], k):
            size = 1
            for _, d in combo:
                size *= len(d)
            if size > MAX_COUNT:
                continue
            out = set()
            for values in itertools.product(*(d for _, d in combo)):
                _spend(1)
                env = list(zip((e for e, _ in combo), values))
                if any(_violated(mod, rec, env) for rec in st.conds):
                    continue
                try:
                    out.add(_eval(mod, t, env))
                except _Unknown:
                    if out:
                        return None  # fixes t for some values only
                    break  # the first feasible value leaves t unfixed: not this combination
            else:
                if out:
                    return out
    return None


# ---------------------------------------------------------------- exception landing pads

def _uleb(img, o):
    v = sh = 0
    while True:
        b = img[o]
        o += 1
        v |= (b & 0x7F) << sh
        sh += 7
        if not b & 0x80:
            return v, o


def _sleb(img, o):
    v, o2 = _uleb(img, o)
    n = 7 * (o2 - o)
    return (v - (1 << n) if v >> (n - 1) & 1 else v), o2


def _enc(img, o, enc):
    """DWARF pointer encoding at image offset o: (value as image offset or plain number, next offset)."""
    if enc == 0xFF:
        return 0, o
    fmt = enc & 0x0F
    if fmt == 0x01:
        v, o2 = _uleb(img, o)
    elif fmt == 0x09:
        v, o2 = _sleb(img, o)
    else:
        n, signed = {0x00: (8, False), 0x02: (2, False), 0x03: (4, False), 0x04: (8, False), 0x0A: (2, True),
                     0x0B: (4, True), 0x0C: (8, True)}[fmt]
        v, o2 = int.from_bytes(img[o:o + n], 'little', signed=signed), o + n
    if enc & 0x70 == 0x10:
        v += o
    return v, o2


def landing_pads(mod):
    """{landing pad: [(start, end)]}: the call-site ranges of each pad, from .eh_frame and the LSDAs."""
    hit = _PADS.get(mod)
    if hit is not None:
        return hit
    img, hdr, pads = mod.image, mod.nso.eh_start, {}
    if hdr and hdr < len(img):
        o, _ = _enc(img, hdr + 4, img[hdr + 1])
        cies = {}
        while o + 8 <= len(img):
            length = int.from_bytes(img[o:o + 4], 'little')
            if length in (0, 0xFFFFFFFF):
                break
            nxt, cid = o + 4 + length, int.from_bytes(img[o + 4:o + 8], 'little')
            if cid == 0:
                p = o + 9
                aug = img[p:img.index(0, p)].decode('latin-1')
                p += len(aug) + 1
                _, p = _uleb(img, p)
                _, p = _sleb(img, p)
                p = p + 1 if img[o + 8] == 1 else _uleb(img, p)[1]
                info = {'R': 0, 'L': None}
                if aug.startswith('z'):
                    _, p = _uleb(img, p)
                    for ch in aug[1:]:
                        if ch == 'P':
                            _, p = _enc(img, p + 1, img[p])
                        elif ch in 'LR':
                            info[ch] = img[p]
                            p += 1
                cies[o] = info
            else:
                cie = cies.get(o + 4 - cid)
                if cie is not None and cie['L'] is not None:
                    start, p = _enc(img, o + 8, cie['R'])
                    _, p = _enc(img, p, cie['R'] & 0x0F)
                    _, p = _uleb(img, p)
                    lsda, _ = _enc(img, p, cie['L'])
                    if lsda:
                        _lsda(img, lsda, start, mod.base, pads)
            o = nxt
    _PADS[mod] = pads
    return pads


def _lsda(img, o, func, base, pads):
    lp_enc = img[o]
    lpstart, o = _enc(img, o + 1, lp_enc)
    if lp_enc == 0xFF:
        lpstart = func
    tt_enc = img[o]
    o += 1
    if tt_enc != 0xFF:
        _, o = _uleb(img, o)
    cs_enc = img[o]
    n, o = _uleb(img, o + 1)
    stop = o + n
    while o < stop:
        s, o = _enc(img, o, cs_enc & 0x0F)
        ln, o = _enc(img, o, cs_enc & 0x0F)
        lp, o = _enc(img, o, cs_enc & 0x0F)
        _, o = _uleb(img, o)
        if lp:
            pads.setdefault(base + lpstart + lp, []).append((base + func + s, base + func + s + ln))


# ---------------------------------------------------------------- per-function resolution

def _explore(cfg, facts, br_pc):
    """Targets of the table jump at br_pc over every path into its dispatch block, or None."""
    mod = cfg.mod
    rn = bits(mod.word(br_pc), 5, 5)
    budget = [MAX_PATHS]

    def explore(path, depth):
        budget[0] -= 1
        if budget[0] < 0:
            return None
        st, t = _run(mod, path, rn, facts.get(path[0][0]))
        r = _solve(mod, t, st, False)
        if r is not None:
            return r
        preds = cfg.path_preds(path[0][0]) if depth < MAX_DEPTH else None
        if preds:
            out = set()
            for seg in preds:
                r = explore([seg] + path, depth + 1)
                if r is None:
                    break
                out |= r
            else:
                return out
        return _solve(mod, t, st, True)

    out = explore([(cfg.block(br_pc), br_pc, None)], 0)
    if not out or any(a % 4 or not mod.in_text(a) for a in out):
        return None
    return out


def _round(mod, entry, end, brs, tables, pads):
    """Resolution of every br in brs over the blocks, facts and predecessors that the table edges `tables` give."""
    cfg = _Cfg(mod, entry, end, tables, pads)
    facts, new = None, {}
    for a in brs:
        r = _explore(cfg, {}, a)
        if r is None:
            if facts is None:
                facts = _facts(cfg) or {}
            if facts:
                r = _explore(cfg, facts, a)
        new[a] = r
    return new


def _function(mod, entry, end):
    """{br: set of targets or None} for the table-shaped brs of [entry, end); cached for the last function.

    Table edges change the blocks, the facts and the predecessors the paths may take, so the function is resolved
    as a whole. Growth: start without table edges, resolve, add the targets found as edges, until no table grows
    (a table found unresolved here may still resolve once the blocks before it are reached). Check: give every
    unresolved table edges to every block, resolve again, drop any table that no longer resolves and grow any
    whose result its edges do not cover (an unresolved one included), until all are covered. By induction over an
    execution, the facts and every resolved table then hold together. The work is capped at MAX_STEPS evaluated
    instructions; a function that needs more gets no tables.
    """
    hit = _FUNCS.get(mod)
    if hit is not None and hit[0] == (entry, end):
        return hit[1]
    w = mod.words[(entry - mod.lo) >> 2:(end - mod.lo) >> 2]
    brs = [entry + 4 * int(i) for i in np.nonzero((w & 0xFFFFFC1F) == 0xD61F0000)[0]]
    brs = [a for a in brs if is_computed(mod, a, entry)]
    pads = {a: r for a, r in landing_pads(mod).items() if entry <= a < end}
    _budget[0] = MAX_STEPS
    try:
        result = _resolve_all(mod, entry, end, brs, pads)
    except _Exhausted:
        result = {a: None for a in brs}
    _FUNCS[mod] = ((entry, end), result)
    return result


def _resolve_all(mod, entry, end, brs, pads):
    tables, result = {}, {a: None for a in brs}
    for _ in range(MAX_ROUNDS if brs else 0):
        grown = False
        for a, r in _round(mod, entry, end, brs, tables, pads).items():
            if r is not None and not r <= tables.get(a, set()):
                tables[a] = tables.get(a, set()) | r
                grown = True
        if not grown:
            break
    dead = set()
    for _ in range(MAX_ROUNDS if brs else 0):
        full = {a: tables.get(a) for a in brs}
        new = _round(mod, entry, end, brs, full, pads)
        bad = [a for a in brs if new[a] is None and full[a] is not None or new[a] is not None and a not in dead
               and not new[a] <= (full[a] or set())]
        if not bad:
            result = {a: new[a] if full[a] is not None else None for a in brs}
            break
        for a in bad:
            if new[a] is None:
                del tables[a]
                dead.add(a)
            else:
                tables[a] = tables.get(a, set()) | new[a]
    return result


# ---------------------------------------------------------------- interface

def table_targets(mod, br_pc, entry, end):
    """Every target the `br` at br_pc can reach through a proved table index, anywhere in .text; None if unproved."""
    if not is_computed(mod, br_pc, entry):
        return None
    t = _function(mod, entry, end).get(br_pc)
    return sorted(t) if t is not None else None


def resolve(mod, br_pc, entry, end):
    """Targets of the `br` at br_pc when it dispatches through a bounds-checked jump table inside [entry, end).

    mod is a recomp.module.Module. Returns a sorted list of distinct target addresses, every one in [entry, end)
    and 4-byte aligned, or None when the br is not a recognised table jump.
    """
    t = table_targets(mod, br_pc, entry, end)
    if t is None or t[0] < entry or t[-1] >= end:
        return None
    return t


def _table_shape(mod, t):
    """The br value is a code address, or address arithmetic over a loaded entry or an image address.

    A value loaded as a whole is a pointer (vtables, function-pointer arrays in .data): a call, not a table.
    """
    k = t[0]
    if k == 'c':
        return mod.in_text(t[1])
    if k in ('add', 'sub'):
        return _refers(mod, t)
    return False


def _refers(mod, e):
    """e contains a load or an image address and something unknown or loaded."""
    has_addr = has_var = False
    stack = [e]
    while stack:
        x = stack.pop()
        k = x[0]
        if k == 'c':
            has_addr |= mod.in_image(x[1])
        elif k in ('r', 'u', 'v', 'vw', 'ld'):
            has_var = True
            has_addr |= k == 'ld'
            if k == 'ld':
                stack.append(x[4])
        else:
            stack.extend(y for y in x[1:] if isinstance(y, tuple))
    return has_addr and has_var


def is_computed(mod, br_pc, entry):
    """True if the br register comes from address arithmetic (a table) rather than a load (a pointer call).

    Decided on the straight-line code before the br alone, so it does not depend on the function bounds.
    """
    w = mod.word(br_pc)
    if not _is_br(w):
        return False
    s = br_pc
    while s > entry and br_pc - s < 4 * MAX_BLOCK and not _ends_block(mod.word(s - 4)):
        s -= 4
    return _table_shape(mod, _run(mod, [(s, br_pc, None)], bits(w, 5, 5), None, spend=False)[1])
