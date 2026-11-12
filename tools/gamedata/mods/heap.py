"""heap:<MiB>: the GPU texture heap grows by that much (System/HeapSize HeapDef: DynamicTextureAllocatorMem and its
parent Graphics, shipped 312320 and 861184 KiB), for a bigger internal resolution. Recompressed with zstd level 19."""
import compression.zstd as zstd
from byml import Byml, patch

TARGETS = ["System/HeapSize/Product.Nin_Ounce_NVN2.HeapDef.byml.zs"]
LOG = []


def apply(path, data, arg):
    add = int(arg) * 1024
    d = zstd.decompress(data)
    for heap in ("DynamicTextureAllocatorMem", "Graphics"):
        v = Byml(d).get(["Heaps", heap, "Size"])   # the heaps sit under the root key "Heaps"
        LOG.append(f"{heap} Size {v.value} -> {v.value + add} KiB")
        d = patch(d, v, v.value + add)
    return zstd.compress(d, level=19)
