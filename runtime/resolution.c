// Game resolution: the size the game composes its frame in (canvas, UI, swapchain) and the size of its 3D scene, above the
// shipped 1920x1080 and 1600x900. Settings OOT_OUTPUT_RES and OOT_RENDER_SCALE; see docs/how-it-works.md, "Render resolution".
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

#include "cfg.h"
#include "host.h"

#define BASE_W 1920
#define BASE_H 1080
#define SCENE_W 1600
#define SCENE_H 900
#define DTA_KIB 113664u      // DynamicTextureAllocatorMem, shipped
#define SCENE_MAX 4096  // per side

static struct {
    int decided;
    unsigned ow, oh, sw, sh;   // output and scene
    int scene_set;             // the scene differs from the game's 5/6 of the output
} R;

static void decide(void) {
    if (R.decided) return;
    R.decided = 1;
    R.ow = BASE_W;
    R.oh = BASE_H;
    const char *o = cfg_value("OOT_OUTPUT_RES");
    unsigned w = 0, h = 0;
    char tail;
    if (sscanf(o, "%ux%u%c", &w, &h, &tail) == 2 && w >= 1280 && h >= 720 && w <= 7680 && h <= 4320) {
        R.ow = w & ~1u;
        R.oh = h & ~1u;
    } else {
        rt_log("resolution: OOT_OUTPUT_RES=%s is not <width>x<height> in 1280x720..7680x4320; 1920x1080", o);
    }
    const char *s = cfg_value("OOT_RENDER_SCALE");
    double k = 5.0 / 6.0;
    if (strcmp(s, "game")) {
        char *end;
        double pct = strtod(s, &end);
        if (end != s && !*end && pct >= 25 && pct <= 400) {
            k = pct / 100.0;
            R.scene_set = 1;
        } else {
            rt_log("resolution: OOT_RENDER_SCALE=%s is not game or a percentage 25..400; game", s);
        }
    }
    double sw = R.ow * k, sh = R.oh * k;
    double fit = fmin(1.0, fmin(SCENE_MAX / sw, SCENE_MAX / sh));
    if (fit < 1.0) R.scene_set = 1;
    R.sw = (unsigned)(sw * fit + 0.5);
    R.sh = (unsigned)(sh * fit + 0.5);
}

static int bigger(void) {
    decide();
    return R.ow != BASE_W || R.oh != BASE_H || R.scene_set;
}

uint64_t res_heap_extra_kib(void) {
    if (!bigger()) return 0;
    double grow = fmax(0.0, (double)R.ow * R.oh / (BASE_W * BASE_H) - 1.0) +
                  fmax(0.0, (double)R.sw * R.sh / (SCENE_W * SCENE_H) - 1.0);
    if (grow <= 0) return 0;
    return (uint64_t)ceil(DTA_KIB * (1.0 + grow) / 1024.0) * 1024 - DTA_KIB;
}

uint64_t res_heap_extra_bytes(void) { return ((res_heap_extra_kib() << 10) + 0x1fffff) & ~0x1fffffull; }

// ---- the HeapDef table. BYML version 2..7, little endian ("YB"): a hash node is {u8 0xC1, u24 count} and count
// entries {u24 key index, u8 type, u32 value}; an int (type 0xD1) is stored in the entry, so a size changes in place.
typedef struct { uint8_t *p; size_t n; uint32_t keys; } Byml;

static int u32_at(const Byml *b, size_t off, uint32_t *v) {
    if (off + 4 > b->n) return 0;
    memcpy(v, b->p + off, 4);
    return 1;
}

static const char *key_name(const Byml *b, uint32_t k) {
    uint32_t head, at;
    if (!u32_at(b, b->keys, &head) || (head & 0xff) != 0xc2 || k >= head >> 8 || !u32_at(b, b->keys + 4 + 4 * k, &at) ||
        b->keys + at >= b->n || !memchr(b->p + b->keys + at, 0, b->n - b->keys - at))
        return NULL;
    return (const char *)b->p + b->keys + at;
}

// The offset of the value of hash node `node`'s entry `name`, which must have type `type`; 0 when there is none.
static size_t field(const Byml *b, uint32_t node, const char *name, uint8_t type) {
    uint32_t head;
    if (!u32_at(b, node, &head) || (head & 0xff) != 0xc1) return 0;
    for (uint32_t i = 0; i < head >> 8; i++) {
        uint32_t e;
        if (!u32_at(b, node + 4 + 8 * (size_t)i, &e)) return 0;
        const char *k = key_name(b, e & 0xffffff);
        if (k && !strcmp(k, name)) return e >> 24 == type ? node + 8 + 8 * (size_t)i : 0;
    }
    return 0;
}

static size_t heap_size_field(const Byml *b, uint32_t heaps, const char *heap) {
    size_t at = field(b, heaps, heap, 0xc1);
    uint32_t node;
    return at && u32_at(b, at, &node) ? field(b, node, "Size", 0xd1) : 0;
}

int res_heapdef_build(const void *in, size_t n, void **out, size_t *out_n) {
    uint64_t extra = res_heap_extra_kib();
    *out = NULL;
    if (!extra) return 0;
    if (ZSTD_getDictID_fromFrame(in, n)) {
        rt_log("resolution: the HeapDef table needs a zstd dictionary; the game keeps its own resolution");
        return 0;
    }
    unsigned long long raw = ZSTD_getFrameContentSize(in, n);
    if (raw == ZSTD_CONTENTSIZE_UNKNOWN || raw == ZSTD_CONTENTSIZE_ERROR || raw > (64u << 20)) {
        rt_log("resolution: the HeapDef table is not a zstd frame with a size; the game keeps its own resolution");
        return 0;
    }
    Byml b = {malloc((size_t)raw), (size_t)raw, 0};
    if (!b.p) return 0;
    size_t got = ZSTD_decompress(b.p, b.n, in, n);
    uint32_t root, heaps = 0;
    uint16_t ver = 0;
    if (!ZSTD_isError(got) && got == b.n && b.n >= 16) memcpy(&ver, b.p + 2, 2);
    size_t heaps_at = 0, dta_at = 0, gfx_at = 0;
    if (ver >= 2 && ver <= 7 && !memcmp(b.p, "YB", 2) && u32_at(&b, 4, &b.keys) && u32_at(&b, 12, &root) &&
        (heaps_at = field(&b, root, "Heaps", 0xc1)) && u32_at(&b, heaps_at, &heaps)) {
        dta_at = heap_size_field(&b, heaps, "DynamicTextureAllocatorMem");
        gfx_at = heap_size_field(&b, heaps, "Graphics");
    }
    if (!dta_at || !gfx_at) {
        rt_log("resolution: the HeapDef table has no DynamicTextureAllocatorMem / Graphics sizes; the game keeps its own "
               "resolution");
        free(b.p);
        return 0;
    }
    int32_t dta, gfx;
    memcpy(&dta, b.p + dta_at, 4);
    memcpy(&gfx, b.p + gfx_at, 4);
    int32_t dta2 = dta + (int32_t)extra, gfx2 = gfx + (int32_t)extra;
    memcpy(b.p + dta_at, &dta2, 4);
    memcpy(b.p + gfx_at, &gfx2, 4);
    size_t cap = ZSTD_compressBound(b.n);
    void *z = malloc(cap);
    size_t zn = z ? ZSTD_compress(z, cap, b.p, b.n, 9) : 0;
    free(b.p);
    if (!z || ZSTD_isError(zn)) {
        free(z);
        return 0;
    }
    *out = z;
    *out_n = zn;
    rt_log("resolution: HeapDef DynamicTextureAllocatorMem %d -> %d KiB, Graphics %d -> %d KiB", dta, dta2, gfx, gfx2);
    return 1;
}

// ---- start: this game has no output and scene patch sets
void res_init(void) {
    if (!bigger()) {
        rt_log("resolution: the game's own (output 1920x1080, scene 1600x900)");
        return;
    }
    rt_log("resolution: the generated C has no output and scene patch sets; the game keeps its own resolution");
}


void res_sizes(unsigned out[2], unsigned scene[2]) {
    decide();
    out[0] = BASE_W, out[1] = BASE_H;
    scene[0] = SCENE_W, scene[1] = SCENE_H;
}
