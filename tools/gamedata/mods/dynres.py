"""dynres:<scale>: the game's internal 3D resolution, as a share of its 1706x960 game image, held fixed. The
scene size comes from dynamic resolution: System/GameGraphics/ModelSceneExtensionParam
(inside Pack/Bootup) has a palette of levels (docked 0x2ebeca77: 0.75, 0.711 ... 0.333; handheld 0x90e0a495) and drops a
level when the GPU load it measures passes LoadThreshold (95 / 108). As shipped the game stays at level 0, 0.75 of
1706x960 = the 1280x720 scene (BaseResoWidth/Height in ModelSceneConfigParam does not change it). This mod sets
level 0 to <scale> and both palettes' LoadThreshold to 1e6, so the level never drops: 0.9375 = 1600x900, 1.0 = 1706x960.
An in-place edit: the pack does not grow, the size table stays."""
import byml, sarc

PACK = "Pack/Bootup.Nin_Ounce_NVN2.pack.zs"
MEMBER = "System/GameGraphics/ModelSceneExtensionParam/Default.game__gfx__ModelSceneExtensionParam.bgyml"
TARGETS = [PACK]
LOG = []


def apply(path, data, arg):
    scale = float(arg or "0.9375")
    if not 0.25 <= scale <= 1.0: raise SystemExit("dynres: a scale from 0.25 to 1.0")

    def edit(b):
        pal = byml.Byml(b).root["DynamicResolutionParam"]["LevelPaletteDict"]
        for key in list(pal.keys()):
            b = byml.set_value(b, ["DynamicResolutionParam", "LevelPaletteDict", key, "LoadThreshold"], 1.0e6)
            b = byml.set_value(b, ["DynamicResolutionParam", "LevelPaletteDict", key, "DynamicResolutionLevelParamList", 0,
                                   "ResolutionReduceScale"], scale)
            LOG.append(f"palette {key}: LoadThreshold -> 1e6, level 0 scale -> {scale} (about {round(1706 * scale)}x{round(960 * scale)})")
        return b

    new, grown = sarc.edit_pack(data, MEMBER, edit)
    if grown: raise SystemExit(f"dynres: the pack grew by {grown} bytes (expected an in-place edit)")
    return new
