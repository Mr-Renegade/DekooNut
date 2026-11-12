"""lod:<coef>: model LOD switches farther out on console. The game's ModelLodParam
(System/GameGraphics/ModelLodParam/Default, inside Pack/Bootup) sets only FovyScaleForNoraml and FovyScaleForTimeline;
LodStartDistCoefForConsole comes from the constructor (main 0x7000f7e7d0, object +0x48 = 1.0; handheld +0x4c = 0.8,
MaxLevelDistScale +0x50 = 1.0; field order from the reader 0x70031a2560). This mod adds LodStartDistCoefForConsole =
<coef> (default 1.5): models keep their detailed level that much farther. The key insert grows the pack, so the
resource size table's entry for the pack grows by the same amount (sarc.restbl_grow). Run after dynres when both are
in one pack (they edit the same pack; this one tracks only its own growth)."""
import byml, sarc

PACK = "Pack/Bootup.Nin_Ounce_NVN2.pack.zs"
RESTBL = "System/Resource/ResourceSizeTable.Product.111.Nin_Ounce_NVN2.rsizetable.zs"
MEMBER = "System/GameGraphics/ModelLodParam/Default.game__gfx__ModelLodParam.bgyml"
TARGETS = [PACK, RESTBL]   # in this order: the pack's growth is known before the table is edited
LOG, grown = [], {}


def apply(path, data, arg):
    coef = float(arg or "1.5")
    if not 0.5 <= coef <= 4.0: raise SystemExit("lod: a coefficient from 0.5 to 4")
    if path == PACK:
        new, grown[PACK] = sarc.edit_pack(data, MEMBER, lambda b: byml.set_value(b, ["LodStartDistCoefForConsole"], coef))
        LOG.append(f"{MEMBER}: LodStartDistCoefForConsole (code default 1.0) -> {coef}; pack +{grown[PACK]} bytes")
        return new
    if grown.get(PACK, 0) <= 0: return data
    raw, _ = sarc.decompress(data)
    LOG.append(f"RESTBL {PACK[:-3]}: +{(grown[PACK] + 31) & ~31}")
    return sarc.compress(sarc.restbl_grow(raw, PACK[:-3], grown[PACK]))
