"""shadowfade:<start>/<end>: where the sun's dynamic shadows fade out (agl ShadowPrePassParam MainField
DynamicShadowFarFadeStart / End, shipped 80 / 200 m). Full-strength shadows reach farther;
no new pass, the receivers already sample the cascades. The last cascade ends near DepthShadowNear03 = 200 m, so an
end much above that may show the map's edge."""
from byml import Byml, patch

TARGETS = ["agl/ShadowPrePassParam/MainField.agl__ShadowPrePassParam.bgyml"]
LOG = []


def apply(path, data, arg):
    start, _, end = (arg or "150/250").partition("/")
    start, end = float(start), float(end)
    if not 0 < start < end <= 1000: raise SystemExit("shadowfade: <start>/<end> in metres, start < end <= 1000")
    for key, v in (("DynamicShadowFarFadeStart", start), ("DynamicShadowFarFadeEnd", end)):
        old = Byml(data).get([key]); LOG.append(f"{key} {old.value} -> {v}"); data = patch(data, old, v)
    return data
