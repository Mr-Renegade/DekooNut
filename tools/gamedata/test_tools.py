#!/usr/bin/env python3
"""Offline checks of tools/gamedata/sarc.py and byml.py against the player's own RomFS (read-only; writes nothing).

    python3 tools/gamedata/test_tools.py <romfs> [--all]

<romfs>: a folder with romfs.bin + index.json (tools/pack_romfs.py), or an extracted RomFS folder. Python 3.14.
1. Packs (Bootup.Nin_Ounce_NVN2, Foliage, ResidentCommon): the SARC rebuilt unchanged is byte-identical; the
   recompressed frame has the original's header fields (descriptor, dictionary ID, content size) and decompresses to
   the same bytes.
2. BYML: every .bgyml / .byml in those packs and in the loose gsys, agl, WorldMgr and System/GameGraphics folders is
   rewritten from its parsed tree, byte-identical. --all also does every pack under Pack/ (about 144,000 files).
3. Edits: a key patched in place, a missing key added (a scalar and a whole dict), a hash-dict value, and a member
   edited inside a pack, each read back, with every other value unchanged and a second rewrite stable.
4. RESTBL: the three packs have entries; restbl_grow adds the growth rounded up to 32.
Exit status 0 when every check passes.
"""
import json, struct, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byml, sarc   # noqa: E402

PACKS = ["Pack/Bootup.Nin_Ounce_NVN2.pack.zs", "Pack/Foliage.pack.zs", "Pack/ResidentCommon.pack.zs"]
LOOSE = ["gsys/", "agl/", "WorldMgr/", "System/GameGraphics/"]
RESTBL = "System/Resource/ResourceSizeTable.Product.111.Nin_Ounce_NVN2.rsizetable.zs"
FAILS = []


class Reader:
    def __init__(self, root):
        self.root = Path(root)
        if (self.root / "index.json").exists():
            j = json.loads((self.root / "index.json").read_text())
            self.files = {f["path"]: (f["offset"], f["size"]) for f in j["files"]}
            self.f = open(self.root / "romfs.bin", "rb")
        else:
            self.files = {str(p.relative_to(self.root)): None for p in self.root.rglob("*") if p.is_file()}
            self.f = None

    def read(self, path):
        if self.f is None: return (self.root / path).read_bytes()
        off, size = self.files[path]
        self.f.seek(off)
        return self.f.read(size)


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what)
    if not ok: FAILS.append(what)


def flat(node, p=""):
    """path -> (type, value) for every leaf, to compare two trees."""
    out = {}
    if isinstance(node, dict):
        for k, v in node.items(): out.update(flat(v, f"{p}/{k}"))
    elif isinstance(node, list):
        for i, v in enumerate(node): out.update(flat(v, f"{p}[{i}]"))
    else:
        v = node.value
        out[p] = (node.type, struct.pack("<d", v) if isinstance(v, float) else v)
    return out


def bymls_of(name, data):
    if name.endswith(".zs"):
        data, _ = sarc.decompress(data)
        name = name[:-3]
    if data[:4] == b"SARC":
        for n, b in sarc.Sarc(data).items(): yield from bymls_of(n, b)
    elif data[:2] in (b"YB", b"BY"):
        yield name, data


def roundtrip(label, items):
    """Each file rewritten from its tree: byte-identical, or else the same tree (Nintendo's writer sometimes keeps two
    copies of an equal 8-byte value or points at another copy of an equal container; write() shares them)."""
    n = same = equal = 0
    first, bad = [], []
    for name, data in items:
        n += 1
        b = byml.Byml(data)
        w = byml.write(b.root, b.version, b.e == ">")
        if w == data: same += 1; continue
        if flat(byml.Byml(w).root) == flat(b.root) and byml.write(byml.Byml(w).root, b.version, b.e == ">") == w:
            equal += 1
            if len(first) < 3: first.append(f"{name} {len(data)} -> {len(w)}")
        else: bad.append(name)
    check(not bad, f"BYML rewrite {label}: {n} files, {same} byte-identical, {equal} the same tree and stable"
          + (f" (e.g. {first})" if first else "") + (f"; DIFFERENT TREE: {bad[:5]}" if bad else ""))


def edit_checks(r):
    # in place: an existing bool
    path = "gsys/ModelSceneConfigParam/Main.gsys__ModelSceneConfigParam.bgyml"
    d = r.read(path)
    n = byml.set_value(d, ["CreateFilterAa"], False)
    a, b = flat(byml.Byml(d).root), flat(byml.Byml(n).root)
    check(len(n) == len(d) and b.pop("/CreateFilterAa") == (0xD0, False) and a.pop("/CreateFilterAa") == (0xD0, True)
          and a == b and sum(x != y for x, y in zip(d, n)) == 1, "in place: Main config CreateFilterAa True -> False, one byte")

    # missing scalar and missing dict in a loose area file
    path = "WorldMgr/WorldParam/HyruleField_Child.game__wm__WorldParam.bgyml"
    d = r.read(path)
    root0 = byml.Byml(d).root
    check("VolumeFogResScaleConsole" not in root0, "HyruleField_Child WorldParam has no VolumeFogResScaleConsole (inherited)")
    n = byml.set_value(d, ["VolumeFogResScaleConsole"], {"X": byml.leaf(1.0), "Y": byml.leaf(1.0), "Z": byml.leaf(1.0)})
    n = byml.set_value(n, ["SunriseHoure"], 5)             # existing int, in place
    n = byml.set_value(n, ["AddedTestFloat"], 0.25)        # missing scalar
    t = byml.Byml(n)
    a, b = flat(root0), flat(t.root)
    added = {k: b.pop(k) for k in list(b) if k.startswith(("/VolumeFogResScaleConsole", "/AddedTestFloat"))}
    a.pop("/SunriseHoure"); s = b.pop("/SunriseHoure")
    check(added == {"/VolumeFogResScaleConsole/X": (0xD2, struct.pack("<d", 1.0)), "/VolumeFogResScaleConsole/Y":
                    (0xD2, struct.pack("<d", 1.0)), "/VolumeFogResScaleConsole/Z": (0xD2, struct.pack("<d", 1.0)),
                    "/AddedTestFloat": (0xD2, struct.pack("<d", 0.25))} and s == (0xD1, 5) and a == b,
          "added: VolumeFogResScaleConsole {X,Y,Z} and a float key read back; every other value unchanged")
    check(t.keys == sorted(t.keys) and t.strings == sorted(t.strings), "added: key and string tables sorted")
    check(byml.write(t.root) == n, "added: a second rewrite is byte-identical (stable)")

    # hash dict, inside the Bootup pack, through edit_pack and the RESTBL
    pack = "Pack/Bootup.Nin_Ounce_NVN2.pack.zs"
    member = "System/GameGraphics/ModelSceneExtensionParam/Default.game__gfx__ModelSceneExtensionParam.bgyml"
    world = "WorldMgr/WorldParam/MainField.game__wm__WorldParam.bgyml"
    frame = r.read(pack)
    lt = ["DynamicResolutionParam", "LevelPaletteDict", 0x2ebeca77, "LoadThreshold"]
    f2, g1 = sarc.edit_pack(frame, member, lambda b: byml.set_value(b, lt, 100000.0))
    f3, g2 = sarc.edit_pack(f2, world, lambda b: byml.set_value(b, ["VolumeFogResScaleConsole", "X"], 1.0))
    old = sarc.Sarc(sarc.decompress(frame)[0]); new = sarc.Sarc(sarc.decompress(f3)[0])
    others = all(new[k] == v for k, v in old.items() if k not in (member, world)) and len(old.members) == len(new.members)
    v1 = byml.Byml(new[member]).get(lt).value
    v2 = byml.Byml(new[world]).get(["VolumeFogResScaleConsole", "X"]).value
    check(v1 == 100000.0 and v2 == 1.0 and others and g1 == 0 and g2 == 0,
          f"in a pack: LoadThreshold 95 -> {v1} (hash dict), MainField fog X 0.7 -> {v2}; {len(old.members) - 2} other members byte-identical; growth {g1 + g2} bytes")

    # a member that grows: a missing dict in the Foliage pack, with its RESTBL entry
    pack = "Pack/Foliage.pack.zs"
    member = "Foliage/ResFoliageScene/HyruleField_Child.game__foliage__ResFoliageScene.bgyml"
    frame = r.read(pack)
    lod = {"Dist_LowTree": byml.leaf(150.0), "Dist_Ornament": byml.leaf(90.0)}
    f2, grow = sarc.edit_pack(frame, member, lambda b: byml.set_value(b, ["LodSetting"], lod))
    old = sarc.Sarc(sarc.decompress(frame)[0]); new = sarc.Sarc(sarc.decompress(f2)[0])
    check(all(new[k] == v for k, v in old.items() if k != member) and len(old.members) == len(new.members)
          and all(new.members[k][1] >= min(8, old.members[k][1]) for k in old.members),
          f"grown member: the {len(old.members) - 1} other Foliage members byte-identical, still aligned")
    got = byml.Byml(new[member]).get(["LodSetting", "Dist_LowTree"]).value
    rt = sarc.decompress(r.read(RESTBL))[0]
    e0 = sarc.restbl_lookup(rt, pack[:-3])
    rt2 = sarc.restbl_grow(rt, pack[:-3], grow)
    e1 = sarc.restbl_lookup(rt2, pack[:-3])
    check(got == 150.0 and grow > 0 and e1 - e0 == (grow + 31) & ~31 and len(rt2) == len(rt)
          and sum(x != y for x, y in zip(rt, rt2)) <= 4,
          f"grown member: Foliage HyruleField_Child LodSetting added, Dist_LowTree {got}; pack +{grow} bytes; RESTBL {e0} -> {e1}")


def main():
    if len(sys.argv) < 2: sys.exit(__doc__)
    r = Reader(sys.argv[1])
    sarc.use_romfs(r)

    for p in PACKS:
        frame = r.read(p)
        raw, dname = sarc.decompress(frame)
        s = sarc.Sarc(raw)
        check(s.build() == raw, f"{p}: {len(s.members)} members, {dname}; SARC rebuilt unchanged is byte-identical ({len(raw)} bytes)")
        c = sarc.compress(raw, dname)
        hdr = lambda f: (f[4], sarc.frame_dict_id(f), int.from_bytes(f[6:10], "little"))
        check(hdr(c) == hdr(frame) and sarc.decompress(c)[0] == raw,
              f"{p}: recompressed {len(frame)} -> {len(c)} bytes; same frame header {hdr(frame)}; decompresses identical; "
              f"byte-identical frame: {c == frame}")

    roundtrip("in the 3 packs", (x for p in PACKS for x in bymls_of(p, r.read(p))))
    roundtrip("loose " + " ".join(LOOSE), ((p, r.read(p)) for p in sorted(r.files)
                                         if p.startswith(tuple(LOOSE)) and p.endswith((".bgyml", ".byml"))))
    if "--all" in sys.argv:
        n = same = 0
        diff = []
        for p in sorted(r.files):
            if not (p.startswith("Pack/") and p.endswith(".pack.zs")): continue
            raw, _ = sarc.decompress(r.read(p))
            n += 1
            if sarc.Sarc(raw).build() == raw: same += 1
            elif len(diff) < 5: diff.append(p)
        check(same == n, f"SARC rebuild of every pack under Pack/: {same} of {n} byte-identical" + (f" ({diff})" if diff else ""))
        roundtrip("every pack under Pack/", (x for p in sorted(r.files) if p.startswith("Pack/") and p.endswith(".pack.zs")
                                             for x in bymls_of(p, r.read(p))))
    edit_checks(r)
    rt = sarc.decompress(r.read(RESTBL))[0]
    check(all(sarc.restbl_lookup(rt, p[:-3]) for p in PACKS),
          "RESTBL entries: " + ", ".join(f"{p[5:-8]} {sarc.restbl_lookup(rt, p[:-3])}" for p in PACKS))
    print(f"{len(FAILS)} failed" if FAILS else "all passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
