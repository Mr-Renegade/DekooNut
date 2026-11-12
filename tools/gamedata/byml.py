"""Binary YAML (BYML / .bgyml, versions 2-7) for tools/build_gamedata.py: a reader that keeps every value node's offset,
and in-place patches of inline values (bool, int, uint, float), so a mod changes exactly the bytes of the values it sets
and the rest of the file stays byte-identical. Read-only of the input.

For a key the file lacks (most area files inherit their values from a parent or from code defaults), set_value()
adds it and write() rebuilds the whole file from the tree: key and string tables sorted, 8-byte values first, then the
containers depth first, equal ones shared. That is the shipped layout: almost every BYML file of the game comes out byte-identical,
the rest as the same tree, a little smaller (Nintendo's writer kept duplicate 8-byte values there, or
pointed at another copy of an equal container); tools/gamedata/test_tools.py checks it. Hash-keyed dicts (0x20, 0x21)
read as HashDict (int hash -> node), e.g. ModelSceneExtensionParam's LevelPaletteDict.

Layout: header "BY" (big-endian) or "YB" (little-endian), u16 version, u32 key table, u32 string table, u32 root.
Nodes: 0xA0 string (index into the string table), 0xC0 array, 0xC1 dict (sorted by key), 0xC2 string table,
0xD0 bool, 0xD1 int, 0xD2 float, 0xD3 uint, 0xD4 int64, 0xD5 uint64, 0xD6 double (8-byte ones by offset), 0xFF null.
"""
import struct

INLINE = {0xD0: "?", 0xD1: "i", 0xD2: "f", 0xD3: "I"}
BIG = {0xD4: "q", 0xD5: "Q", 0xD6: "d"}


class Value:
    """A leaf: its type, decoded value and the offset of its 4-byte slot (inline) or of its 8 bytes (int64/double)."""
    __slots__ = ("type", "value", "offset")

    def __init__(self, t, v, off):
        self.type, self.value, self.offset = t, v, off

    def __repr__(self):
        return f"{self.value!r}"

    def __eq__(self, o):
        return isinstance(o, Value) and o.type == self.type and _same(o.value, self.value)

    def __hash__(self):
        return hash((self.type, self.value))


def _same(a, b):
    """Equal values, with floats compared by their bits (so NaN equals NaN and -0.0 differs from 0.0)."""
    if isinstance(a, float) and isinstance(b, float): return struct.pack("<d", a) == struct.pack("<d", b)
    return a == b


class HashDict(dict):
    """A hash-keyed dict node (0x20: u32 hashes, 0x21: u64 hashes): int hash -> node."""

    def __init__(self, t=0x20, *a):
        super().__init__(*a)
        self.type = t


class Byml:
    def __init__(self, data):
        self.data = bytes(data)
        m = self.data[:2]
        if m == b"YB": self.e = "<"
        elif m == b"BY": self.e = ">"
        else: raise ValueError("not a BYML file")
        self.version, kt, st, root = struct.unpack_from(self.e + "HIII", self.data, 2)
        self.keys = self._strings(kt) if kt else []
        self.strings = self._strings(st) if st else []
        self.root = self._node(self.data[root], root) if root else None

    def _u24(self, off):
        b = self.data[off:off + 3]
        return int.from_bytes(b, "little" if self.e == "<" else "big")

    def _strings(self, off):
        n = self._u24(off + 1)
        offs = struct.unpack_from(self.e + f"{n + 1}I", self.data, off + 4)
        return [self.data[off + offs[i]:self.data.index(b"\0", off + offs[i])].decode("utf-8") for i in range(n)]

    def _value(self, t, slot):
        raw = struct.unpack_from(self.e + "I", self.data, slot)[0]
        if t in INLINE: return Value(t, struct.unpack_from(self.e + INLINE[t], self.data, slot)[0], slot)
        if t in BIG: return Value(t, struct.unpack_from(self.e + BIG[t], self.data, raw)[0], raw)
        if t == 0xA0: return Value(t, self.strings[raw], slot)
        if t == 0xFF: return Value(t, None, slot)
        if t in (0xC0, 0xC1, 0x20, 0x21): return self._node(t, raw)
        return Value(t, raw, slot)   # unknown leaf types are kept as their raw word

    def _node(self, t, off):
        n = self._u24(off + 1)
        if t == 0xC1:
            d = {}
            for i in range(n):
                e = off + 4 + 8 * i
                k = self._u24(e); vt = self.data[e + 3]
                d[self.keys[k]] = self._value(vt, e + 4)
            return d
        if t == 0xC0:
            types = self.data[off + 4:off + 4 + n]
            base = off + 4 + ((n + 3) & ~3)
            return [self._value(types[i], base + 4 * i) for i in range(n)]
        if t in (0x20, 0x21):   # hash-keyed dict: {u32 or u64 hash, u32 value} entries, then one type byte each
            esz = 8 if t == 0x20 else 12
            types = self.data[off + 4 + esz * n:off + 4 + esz * n + n]
            d = HashDict(t)
            for i in range(n):
                e = off + 4 + esz * i
                h = struct.unpack_from(self.e + ("I" if t == 0x20 else "Q"), self.data, e)[0]
                d[h] = self._value(types[i], e + esz - 4)
            return d
        raise ValueError(f"node type {t:#x} at {off:#x}")

    def get(self, path):
        """path: keys and list indices, e.g. ["ShadowPrePass", "StaticSdwWidth"] -> Value."""
        n = self.root
        for p in path: n = n[p]
        return n


def patch(data, val, new):
    """Bytes with val (a Value of an inline bool/int/uint/float or an 8-byte int64/uint64/double) set to new."""
    b = Byml(data)
    out = bytearray(data)
    fmt = INLINE.get(val.type) or BIG.get(val.type)
    if not fmt: raise ValueError(f"type {val.type:#x} cannot be patched in place")
    if val.type == 0xD0: fmt, new = "I", 1 if new else 0   # the whole 4-byte word ("?" is one byte: wrong in "BY" files)
    struct.pack_into(b.e + fmt, out, val.offset, new)
    return bytes(out)


def dump(node, indent=0, out=None):
    """Text of a tree (for logs and for finding keys)."""
    out = [] if out is None else out
    pad = "  " * indent
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, (dict, list)): out.append(f"{pad}{k}:"); dump(v, indent + 1, out)
            else: out.append(f"{pad}{k}: {v!r}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            if isinstance(v, (dict, list)): out.append(f"{pad}- [{i}]"); dump(v, indent + 1, out)
            else: out.append(f"{pad}- {v!r}")
    return out


# ---- writing a whole file (for edits the in-place patch cannot do: a key the file does not have yet)

CONTAINERS = (0xC0, 0xC1, 0x20, 0x21)


def leaf(v, t=None):
    """A Value for a Python value: bool -> 0xD0, int -> 0xD1 (0xD3 if t says so), float -> 0xD2, str -> 0xA0,
    None -> 0xFF. A Value is returned as it is."""
    if isinstance(v, Value): return v
    if t is None:
        if isinstance(v, bool): t = 0xD0
        elif isinstance(v, int): t = 0xD1
        elif isinstance(v, float): t = 0xD2
        elif isinstance(v, str): t = 0xA0
        elif v is None: t = 0xFF
        else: raise TypeError(f"no BYML type for {v!r}")
    if t == 0xD2: v = struct.unpack("<f", struct.pack("<f", v))[0]   # what the file can hold
    return Value(t, v, None)


def _kind(node):
    if isinstance(node, HashDict): return node.type
    if isinstance(node, dict): return 0xC1
    if isinstance(node, list): return 0xC0
    return node.type


def _collect(node, keys, strings):
    if isinstance(node, dict):
        for k, v in node.items():
            if not isinstance(node, HashDict): keys.add(k)
            _collect(v, keys, strings)
    elif isinstance(node, list):
        for v in node: _collect(v, keys, strings)
    elif node.type == 0xA0: strings.add(node.value)


def _string_table(strs, e):
    body, offs = bytearray(), []
    base = 4 + 4 * (len(strs) + 1)
    for s in strs:
        offs.append(base + len(body)); body += s.encode("utf-8") + b"\0"
    offs.append(base + len(body))
    out = bytearray([0xC2]) + len(strs).to_bytes(3, "little" if e == "<" else "big")
    out += struct.pack(e + f"{len(offs)}I", *offs) + body
    return out + b"\0" * (-len(out) % 4)


def _freeze(node):
    """A hashable copy of a subtree, for sharing equal containers and 8-byte values."""
    if isinstance(node, HashDict): return ("h", node.type, tuple((k, _freeze(v)) for k, v in sorted(node.items())))
    if isinstance(node, dict): return ("d", tuple((k, _freeze(v)) for k, v in sorted(node.items())))
    if isinstance(node, list): return ("a", tuple(_freeze(v) for v in node))
    v = node.value
    return ("v", node.type, struct.pack("<d", v) if isinstance(v, float) else v)


def write(root, version=7, big_endian=False):
    """A BYML file for a tree as Byml(...).root returns it (dicts, HashDicts, lists, Values or Python leaves). Keys and
    strings are sorted (byte order, as the game's binary search expects), dict entries by key, hash dicts by hash.
    Containers are written depth first after their parent, and equal containers and equal 8-byte values are written
    once and shared, which is how the shipped files are laid out (the round trip of the shipped files is checked by
    tools/gamedata/test_tools.py)."""
    e = ">" if big_endian else "<"
    u24 = lambda n: n.to_bytes(3, "big" if big_endian else "little")
    keys, strings = set(), set()
    if root is not None: _collect(root, keys, strings)
    keys, strings = sorted(keys), sorted(strings)
    kidx = {k: i for i, k in enumerate(keys)}
    sidx = {s: i for i, s in enumerate(strings)}
    buf = bytearray((b"BY" if big_endian else b"YB") + struct.pack(e + "HIII", version, 0, 0, 0))
    if keys:
        struct.pack_into(e + "I", buf, 4, len(buf)); buf += _string_table(keys, e)
    if strings:
        struct.pack_into(e + "I", buf, 8, len(buf)); buf += _string_table(strings, e)
    done = {}   # frozen subtree -> offset

    def word(v):
        t = v.type
        if t == 0xD0: return struct.pack(e + "I", 1 if v.value else 0)   # a 4-byte word, not struct's 1-byte "?"
        if t in INLINE: return struct.pack(e + INLINE[t], v.value)
        if t == 0xA0: return struct.pack(e + "I", sidx[v.value])
        if t == 0xFF: return b"\0\0\0\0"
        if t in BIG: return None
        return struct.pack(e + "I", v.value)   # an unknown leaf type kept as its raw word

    def node(n):
        f = _freeze(n)
        if f in done: return done[f]
        off = len(buf)
        done[f] = off
        t = _kind(n)
        later = []   # (slot offset, child) for containers and 8-byte values
        if t == 0xC1:
            items = sorted(n.items())
            buf.extend(bytes([0xC1]) + u24(len(items)))
            for k, v in items:
                v = leaf(v) if not isinstance(v, (dict, list)) else v
                buf.extend(u24(kidx[k]) + bytes([_kind(v)]))
                w = None if _kind(v) in CONTAINERS else word(v)
                if w is None: later.append((len(buf), v)); w = b"\0\0\0\0"
                buf.extend(w)
        elif t == 0xC0:
            vals = [leaf(v) if not isinstance(v, (dict, list)) else v for v in n]
            buf.extend(bytes([0xC0]) + u24(len(vals)) + bytes(_kind(v) for v in vals))
            buf.extend(b"\0" * (-len(buf) % 4))
            for v in vals:
                w = None if _kind(v) in CONTAINERS else word(v)
                if w is None: later.append((len(buf), v)); w = b"\0\0\0\0"
                buf.extend(w)
        else:   # hash dict
            items = sorted(n.items())
            vals = [leaf(v) if not isinstance(v, (dict, list)) else v for _, v in items]
            buf.extend(bytes([t]) + u24(len(items)))
            for (h, _), v in zip(items, vals):
                buf.extend(struct.pack(e + ("I" if t == 0x20 else "Q"), h))
                w = None if _kind(v) in CONTAINERS else word(v)
                if w is None: later.append((len(buf), v)); w = b"\0\0\0\0"
                buf.extend(w)
            buf.extend(bytes(_kind(v) for v in vals))
            buf.extend(b"\0" * (-len(buf) % 4))
        for slot, v in later:
            if _kind(v) in CONTAINERS: o = node(v)
            else:
                fv = _freeze(v)
                if fv in done: o = done[fv]
                else:
                    o = len(buf); done[fv] = o; buf.extend(struct.pack(e + BIG[v.type], v.value))
            struct.pack_into(e + "I", buf, slot, o)
        return off

    def big_values(n):   # the 8-byte values come first, after the string tables, in the order the nodes hold them
        if isinstance(n, dict): kids = [v for _, v in sorted(n.items())]
        elif isinstance(n, list): kids = n
        else: kids = []
        for v in kids:
            if isinstance(v, (dict, list)): continue
            v = leaf(v)
            if v.type in BIG:
                fv = _freeze(v)
                if fv not in done: done[fv] = len(buf); buf.extend(struct.pack(e + BIG[v.type], v.value))
        for v in kids:
            if isinstance(v, (dict, list)): big_values(v)

    if root is not None:
        big_values(root)
        struct.pack_into(e + "I", buf, 12, node(root))
    return bytes(buf)


def set_value(data, path, value, t=None):
    """data with the value at path (keys and list indices) set to value, adding the last key if its dict lacks it.
    An existing inline value of the same type is patched in place (the rest of the file stays byte-identical);
    anything else rebuilds the file with write(). The type is the old value's when there is one, else from t or the
    Python type (leaf)."""
    b = Byml(data)
    parent = b.root
    for p in path[:-1]: parent = parent[p]
    last = path[-1]
    old = parent[last] if (last in parent if isinstance(parent, dict) else last < len(parent)) else None
    if isinstance(old, Value):
        new = leaf(value, t or old.type)
        if new.type == old.type and (old.type in INLINE or old.type in BIG): return patch(data, old, new.value)
    else:
        new = value if isinstance(value, (dict, list)) else leaf(value, t)
    if isinstance(parent, list) and last == len(parent): parent.append(new)
    else: parent[last] = new
    return write(b.root, b.version, b.e == ">")
