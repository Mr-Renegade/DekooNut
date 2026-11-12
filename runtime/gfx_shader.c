// Shader service client: turns an NVN program into SPIR-V plus reflection through `oot_xlate.exe --serve` children,
// with a memory and disk cache keyed per program, answers and translator. See README, "Shader cache".
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "nvn_geometry_control.h"
#include "platform/platform.h"
#include "spv_fma_zero.h"
#include "spv_gmem_stub.h"

int vk_mem_flat_mode(void);   // vk.h pulls in Vulkan, which this file does without

#define CONTROL_SIZE 0x880
// NVN 2 compute c0: a 16-byte driver head, eight root tables (word counts +0xa9..0xb0), uniform/storage tables
// (+0xb1/+0xb2), then texture/image handles (+0xb3/+0xb6). nvn2_lower.cpp writes DKC1 and the counts.
#define NVN2_C0_LAYOUT 0x860

// Shared memory bytes of a lowered NVN 2 compute stage (nvn2_lower.cpp, after the bank counts), 0 otherwise.
static uint32_t nvn2_shared(const uint8_t *control) {
    uint32_t n = 0;
    if (!memcmp(control + NVN2_C0_LAYOUT, "DKC1", 4)) memcpy(&n, control + NVN2_C0_LAYOUT + 8, 4);
    return n;
}
#define CODE_SKIP 0x30                                     // graphics code starts after the first 0x30 bytes of the program
#define BUCKETS 4096
#define LINE_MAX_LEN 4096
#define KEY_VERSION 5  // bump when the key layout or the use of the output changes here
#define IN_VERSION 1                                       // legacy input-ID schema and non-GS .in storage version
#define USE_MAGIC 0x4d554b54u                              // "TKUM", .use files
#define USE_VERSION 3  // 3: per-table masks; version 2 files stay valid (use_parse)
#define IN_MAGIC 0x49534b54u                               // "TKSI", .in files
#define GS_IN_VERSION 2                                   // GS only: explicit packed answers plus input topology
#define ANSWERS_PACKED 147                                // legacy hash/record codec, independent of C struct padding
#define GS_ANS_PACKED (ANSWERS_PACKED + 1)
#define LEGACY_ANS_SIZE 152                               // version-1 .in ABI; do not replace with a new sizeof
#define MAX_SVC 32                                         // translator processes

typedef struct {                                           // which answers a shader's translation depends on
    uint32_t slots;                                        // c2 slots whose bindings the translation looked at
    uint32_t attrs;                                        // attr_type (VS input Locations)
    uint32_t fouts;                                        // frag_out_type (FS output Locations)
// Which part of those bindings it read: a format query on a sampled texture reads only its integer flags.
    uint32_t tex_type;                                     // tex_type, whole
    uint32_t tex_class;                                    // tex_type, its integer flags only
    uint32_t img_type, img_format;
} UseMask;

typedef struct UseEnt {                                    // code hash -> use mask; known 0 caches a disk miss
    struct UseEnt *next;
    uint8_t code_hash[32];
    int known;
    UseMask m;
} UseEnt;

struct GfxShader {
    struct GfxShader *next;
    uint8_t key[32];
    uint8_t iid[32];                                       // input id (see the keys at the top)
    uint32_t *spirv;                                       // NULL once compressed into spvz
    uint8_t *spvz;                                         // the checked and repaired module as one zstd frame
    uint64_t spvz_size;
    int spv_checked;                                       // the module is final: repaired and compressed (spv_check)
    uint64_t spv_original;                                 // the hash of the SPIR-V a repair replaced, or 0
    uint64_t words;
    char *reflection;                                      // the service's .res.json text, parsed into the fields below;
    uint64_t reflection_size;                              // NULL / 0 once parsed (disk_load_from)
    uint32_t tex_queried, img_queried;                     // its "tex_queried" / "img_queried" slot masks (use_of)
    // Reflection (reflect(), once per shader before it enters the cache)
    int stage;                                             // GFX_STAGE_*
    GfxShaderRes *res[GFX_RES_KINDS];
    int nres[GFX_RES_KINDS];
    uint32_t in_mask, out_mask;                            // user Locations of Input / Output variables
    GfxShaderIo in[32], out[32];
    uint32_t local[3];
    GfxShaderGeometry geometry;
    uint32_t geometry_skip[8];                           // source mask, independent of queried draw answers
    int geometry_source_passthrough;
    unsigned geometry_position;                         // v2: Position input/output, malformed interface bit 4
    Nvn2C0Layout c0;                                       // NVN 2 compute parameter-block layout, when has_c0
    int has_c0;
};

static PlatMutex cache_lock = PLAT_MUTEX_INIT;                  // memory cache
static GfxShader *buckets[BUCKETS];
static UseEnt *use_buckets[BUCKETS];                       // under cache_lock
static volatile int64_t n_requests, n_hits, n_failures, n_keys, n_xlated;   // n_keys: memory cache entries

typedef struct {                                           // one translator child
    PlatChild *child;                                      // NULL: not started
    int busy;                                              // a request owns it (under S.lock)
    size_t in_len;
    char inbuf[LINE_MAX_LEN * 2];
} Svc;

static struct {
    PlatMutex lock;
    PlatCond freed;
    int limit;                                             // children in use at most (0: svc_runtime_limit)
    Svc s[MAX_SVC];
} S = {PLAT_MUTEX_INIT, PLAT_COND_INIT};

static struct {                                            // translations in flight: cache key, or code hash while the
    PlatMutex lock;                                          // use mask is unknown; one translation per id at a time
    PlatCond done;
    uint8_t id[2 * MAX_SVC][32];
    int n;
} F = {PLAT_MUTEX_INIT, PLAT_COND_INIT};

#define PATH_CAP 1024                                  // bytes of a UTF-8 path
static char cache_dir[PATH_CAP];
static char xlate_exe[PATH_CAP];
static uint8_t xlate_id[32];                               // translator identity (see the keys at the top)
static int host_flat;                                      // vk_mem_flat_mode
static int host_f64 = 1, host_set;                         // gfx_shader_set_host
static uint32_t host_sbuf, host_sg[3] = {32, 32, 32}, host_gather;
static uint32_t host_gs_sg;                               // must be supplied explicitly by the host
// Translator fixes, one bit each, all on. Device-dependent bits are cleared when the device lacks them
// (gfx_shader_xfix_support); a mask other than all gets its own cache keys.
#define XFIX_ALL 0x7ffu
#define XFIX_FLOAT_CONTROLS (1u << 3)
#define XFIX_DEMOTE (1u << 4)
static uint32_t xfix_eff = XFIX_ALL, xfix_unsupported;
static PlatOnce paths_once = PLAT_ONCE_INIT;

// ---- small helpers

#define file_exists plat_file_exists
#define mkdir_p plat_file_mkdir_p
#define read_file plat_file_read
#define write_file plat_file_write

// ---- cache pack (<cache>/shader-cache.pack, written by tools/pack_shader_cache.py)
// "TKSPACK1", u32 count, u32 0, entries sorted by (kind, name), then data; name is the 32 bytes of the 64 hex digits.
// A pack that fails a check is ignored as a whole.
enum { PACK_SPV = 1, PACK_RES = 2, PACK_USE = 3, PACK_IN = 4 };
typedef struct { uint8_t name[32]; uint32_t kind, size; uint64_t offset; } PackEnt;
_Static_assert(sizeof(PackEnt) == 48, "shader-cache.pack entry layout");
static PlatFile *pack_file;
static PackEnt *pack_ix;
static uint32_t pack_n;

static int pack_cmp(const PackEnt *a, uint32_t kind, const uint8_t name[32]) {
    if (a->kind != kind) return a->kind < kind ? -1 : 1;
    return memcmp(a->name, name, 32);
}

static int pack_read_at(uint64_t offset, void *out, uint32_t n) {
    return plat_file_pread(pack_file, out, n, offset) == (int64_t)n;
}

// Opens <cache>/shader-cache.pack and reads its index (init_paths). Nothing happens when there is none.
static void pack_open(void) {
    char p[PATH_CAP + 32];
    snprintf(p, sizeof p, "%s/shader-cache.pack", cache_dir);
    PlatFile *h = plat_file_open(p);
    if (!h) return;
    PlatFileId id;
    uint8_t head[16];
    uint32_t n = 0;
    const char *why = NULL;
    PackEnt *ix = NULL;
    pack_file = h;
    if (!plat_file_id(h, &id) || !pack_read_at(0, head, sizeof head) || memcmp(head, "TKSPACK1", 8)) why = "header";
    if (!why) {
        memcpy(&n, head + 8, 4);
        if (n > (1u << 24) || (uint64_t)n * sizeof *ix + sizeof head > id.size ||
            !(ix = malloc((size_t)n * sizeof *ix + 1)))
            why = "index size";
        else if (!pack_read_at(sizeof head, ix, n * (uint32_t)sizeof *ix)) why = "index read";
    }
    for (uint32_t i = 0; !why && i < n; i++) {
        if (ix[i].offset + ix[i].size > id.size) why = "entry out of bounds";
        else if (i && pack_cmp(&ix[i - 1], ix[i].kind, ix[i].name) >= 0) why = "index order";
    }
    if (why) {
        rt_log("gfx_shader: %s is not a usable cache pack (%s); reading the loose files", p, why);
        free(ix);
        plat_file_close(h);
        pack_file = NULL;
        return;
    }
    pack_ix = ix, pack_n = n;
    rt_log("gfx_shader: cache pack %s: %u files", p, n);
}

static const PackEnt *pack_find(uint32_t kind, const uint8_t name[32]) {
    size_t lo = 0, hi = pack_n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        int c = pack_cmp(&pack_ix[mid], kind, name);
        if (!c) return &pack_ix[mid];
        if (c < 0) lo = mid + 1;
        else hi = mid;
    }
    return NULL;
}

// A packed file as plat_file_read returns a loose one: malloc'd, NUL after the last byte; NULL when absent or empty.
static uint8_t *pack_read(uint32_t kind, const uint8_t name[32], uint64_t *size) {
    const PackEnt *e = pack_find(kind, name);
    uint8_t *buf = e && e->size ? malloc((size_t)e->size + 1) : NULL;
    if (!buf) return NULL;
    if (!pack_read_at(e->offset, buf, e->size)) {
        free(buf);
        return NULL;
    }
    buf[e->size] = 0;
    *size = e->size;
    return buf;
}

// Appends a UTF-8 string as a JSON string body: ASCII as is, quotes and backslashes escaped, the rest as \uXXXX
// (UTF-16 units), so the service's stdin encoding does not matter.
static void json_esc(char **o, const char *s) {
    const unsigned char *p = (const unsigned char *)s;
    while (*p) {
        unsigned c = *p++;
        if (c >= 0x80) {
            int more = c >= 0xf0 ? 3 : c >= 0xe0 ? 2 : 1;
            c &= 0x3f >> more;
            while (more-- && (*p & 0xc0) == 0x80) c = c << 6 | (*p++ & 0x3f);
        }
        if (c == '"' || c == '\\') *(*o)++ = '\\', *(*o)++ = (char)c;
        else if (c >= 0x20 && c < 0x7f) *(*o)++ = (char)c;
        else if (c >= 0x10000) {
            c -= 0x10000;
            *o += sprintf(*o, "\\u%04x\\u%04x", 0xd800 + (c >> 10), 0xdc00 + (c & 0x3ff));
        } else *o += sprintf(*o, "\\u%04x", c);
    }
}

// ---- paths and the translator identity

static int hex32(const char *h, uint8_t out[32]) {        // 64 hex digits, then nothing but white space
    for (int i = 0; i < 64; i++) {
        char c = h[i] >= 'A' && h[i] <= 'F' ? (char)(h[i] | 32) : h[i];
        int v = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
        if (v < 0) return 0;
        out[i / 2] = (uint8_t)(i & 1 ? out[i / 2] | v : v << 4);
    }
    for (const char *p = h + 64; *p; p++)
        if (*p != '\n' && *p != '\r' && *p != ' ' && *p != '\t') return 0;
    return 1;
}

// <translator>.id next to the exe (tools/build_xlate.py), else the sha256 of the exe.
static void xlate_identity(void) {
    const char *how = "oot_xlate.id";
    char p[PATH_CAP + 8];
    strcpy(p, xlate_exe);
    char *dot = strrchr(p, '.'), *sep = plat_last_sep(p);
    if (dot && dot > sep) *dot = 0;
    strcat(p, ".id");
    uint64_t size = 0;
    char *f = (char *)read_file(p, &size);
    if (!f || !hex32(f, xlate_id)) {
        uint8_t *exe = read_file(xlate_exe, &size);
        how = exe ? "sha256 of the exe; tools/build_xlate.py writes oot_xlate.id" : "none: no translator";
        if (exe) rt_sha256(exe, (size_t)size, xlate_id);
        free(exe);
    }
    free(f);
    xfix_eff = XFIX_ALL;
    rt_log("gfx_shader: translator identity %02x%02x%02x%02x%02x%02x%02x%02x (%s)", xlate_id[0], xlate_id[1],
           xlate_id[2], xlate_id[3], xlate_id[4], xlate_id[5], xlate_id[6], xlate_id[7], how);
}

// Device support for the device-dependent translator fix bits (vk_caps.c, before gfx_shader_set_host).
void gfx_shader_xfix_support(int float_controls, int demote) {
    xfix_unsupported = (float_controls ? 0 : XFIX_FLOAT_CONTROLS) | (demote ? 0 : XFIX_DEMOTE);
}

// The repository root is the nearest parent of oot.exe holding build/xlate/oot_xlate.exe; the cache lives next to oot.exe.
static void use_preload_start(void);

static void init_paths(void *param) {
    (void)param;
    char exe[PATH_CAP], root[PATH_CAP] = "", exe_dir[PATH_CAP] = "";
    if (plat_exe_path(exe, sizeof exe)) {
        char *last = plat_last_sep(exe);
        if (last) snprintf(exe_dir, sizeof exe_dir, "%.*s", (int)(last - exe), exe);
        for (char *cut = plat_last_sep(exe); cut; cut = plat_last_sep(exe)) {
            *cut = 0;
            char probe[PATH_CAP + 32];
            snprintf(probe, sizeof probe, "%s/build/xlate/oot_xlate" PLAT_EXE_SUFFIX, exe);
            if (file_exists(probe)) {
                strcpy(root, exe);
                break;
            }
        }
    }
    const char *xlate_cfg = cfg_get("OOT_XLATE");
    char beside[PATH_CAP + 32];   // a packaged folder (tools/package_build.py) keeps the translator next to oot.exe
    if (exe_dir[0]) snprintf(beside, sizeof beside, "%s/oot_xlate" PLAT_EXE_SUFFIX, exe_dir);
    if (xlate_cfg && *xlate_cfg && strlen(xlate_cfg) < sizeof xlate_exe) strcpy(xlate_exe, xlate_cfg);
    else if (exe_dir[0] && file_exists(beside)) strcpy(xlate_exe, beside);
    else if (root[0]) snprintf(xlate_exe, sizeof xlate_exe, "%s/build/xlate/oot_xlate" PLAT_EXE_SUFFIX, root);
    if (exe_dir[0]) snprintf(cache_dir, sizeof cache_dir, "%s/shader-cache", exe_dir);
    xlate_identity();
    if (cache_dir[0]) pack_open();
    use_preload_start();
    return;
}

static void paths(void) { plat_once(&paths_once, init_paths, NULL); }

// Host answers (vk_caps.c, at device creation). Answers other than the defaults are mixed into the translator
// identity, so they get their own cache keys.
void gfx_shader_set_host(int float64, uint32_t sbuf_align, const uint32_t subgroup[3], uint32_t gather_bits) {
    paths();
    if (host_set++) return;
    if (xfix_eff & xfix_unsupported) {
        uint8_t xb[32 + 32];
        xfix_eff &= ~xfix_unsupported;
        int xn = snprintf((char *)xb + 32, 32, "TKXD %#x", xfix_eff);
        memcpy(xb, xlate_id, 32);
        rt_sha256(xb, 32 + (size_t)xn, xlate_id);
        rt_log("gfx_shader: translator fixes %#x on this device (%#x unsupported): own shader cache keys", xfix_eff,
               xfix_unsupported);
    }
    host_f64 = float64, host_sbuf = sbuf_align, host_gather = gather_bits;
    memcpy(host_sg, subgroup, sizeof host_sg);
    host_flat = vk_mem_flat_mode();
    if (float64 && !sbuf_align && subgroup[0] == 32 && subgroup[1] == 32 && subgroup[2] == 32 && !gather_bits && !host_flat) return;
    uint8_t buf[32 + 96];
    int n = snprintf((char *)buf + 32, 96, "TKHO f64=%d sbuf=%u sg=%u,%u,%u", float64, sbuf_align, subgroup[0], subgroup[1],
                     subgroup[2]);
    if (gather_bits) n += snprintf((char *)buf + 32 + n, (size_t)(96 - n), " gather=%u", gather_bits);
    if (host_flat) n += snprintf((char *)buf + 32 + n, (size_t)(96 - n), " flat=%d", host_flat);
    memcpy(buf, xlate_id, 32);
    rt_sha256(buf, 32 + (size_t)n, xlate_id);
    rt_log("gfx_shader: host options %s: own shader cache keys", (char *)buf + 37);
}

void gfx_shader_set_geometry_host(uint32_t subgroup) {
    if (!subgroup || (subgroup & (subgroup - 1))) rt_fatal("gfx_shader: invalid geometry subgroup size %u", subgroup);
    if (host_gs_sg == subgroup) return;
    if (host_gs_sg || n_requests || n_keys) rt_fatal("gfx_shader: geometry host answers changed after initialization");
    paths();
    host_gs_sg = subgroup;
    if (subgroup == 32) return;                           // the managed service's default; preserve legacy output keys
// A separate capability tag keeps the three-stage host API; it changes output keys and use-mask namespaces only.
    uint8_t buf[40];
    memcpy(buf, xlate_id, 32), memcpy(buf + 32, "TKHG", 4), memcpy(buf + 36, &subgroup, 4);
    rt_sha256(buf, sizeof buf, xlate_id);
}

// ---- the translator children

static void child_stop(Svc *v) {
    plat_child_stop(v->child);
    v->child = NULL;
    v->in_len = 0;
}

static int child_start(Svc *v) {
    paths();
    if (!xlate_exe[0] || !file_exists(xlate_exe)) {
        rt_log("gfx_shader: translator not found (%s); set OOT_XLATE", xlate_exe[0] ? xlate_exe : "<none>");
        return 0;
    }
    uint32_t err = 0;
    v->child = plat_child_start(xlate_exe, "--serve", &err);
    if (!v->child) {
        rt_log("gfx_shader: cannot start %s (error %lu)", xlate_exe, (unsigned long)err);
        return 0;
    }
    v->in_len = 0;
    return 1;
}

// One line out, one line in (no newline in the reply). 0 when the pipe broke.
static int child_exchange_raw(Svc *v, const char *line, char *reply, size_t reply_size) {
    if (!v->child && !child_start(v)) return 0;
    plat_inc64(&n_requests);
    size_t n = strlen(line);
    if (!plat_child_write(v->child, line, n)) return 0;
    if (!plat_child_write(v->child, "\n", 1)) return 0;
    for (;;) {
        char *nl = memchr(v->inbuf, '\n', v->in_len);
        if (nl) {
            size_t len = (size_t)(nl - v->inbuf), keep = len < reply_size - 1 ? len : reply_size - 1;
            memcpy(reply, v->inbuf, keep);
            while (keep && reply[keep - 1] == '\r') keep--;
            reply[keep] = 0;
            memmove(v->inbuf, nl + 1, v->in_len - len - 1);
            v->in_len -= len + 1;
            return 1;
        }
        if (v->in_len == sizeof v->inbuf) v->in_len = 0;  // an overlong line: drop it, wait for its end
        long got = plat_child_read(v->child, v->inbuf + v->in_len, sizeof v->inbuf - v->in_len);
        if (got <= 0) return 0;
        v->in_len += (size_t)got;
    }
}

static int child_exchange(Svc *v, const char *line, char *reply, size_t reply_size) {
    uint64_t p0 = perf_begin();   // render-thread time spent translating
    int ok = child_exchange_raw(v, line, reply, reply_size);
    perf_end(PERF_XLATE, p0);
    return ok;
}

static int svc_runtime_limit(void) {
    return 2;
}

static int svc_used(void) {                                // caller holds S.lock
    int used = 0;
    for (int i = 0; i < MAX_SVC; i++) used += S.s[i].busy || S.s[i].child;
    return used;
}

// A child for one request: an idle running one, else a new slot while fewer than S.limit are in use, else waits.
static Svc *svc_get(void) {
    plat_mutex_lock(&S.lock);
    if (!S.limit) S.limit = svc_runtime_limit();
    Svc *v;
    for (;;) {
        Svc *empty = NULL;
        v = NULL;
        for (int i = 0; i < MAX_SVC && !v; i++) {
            Svc *c = &S.s[i];
            if (!c->busy && c->child) v = c;
            else if (!c->busy && !empty) empty = c;
        }
        if (!v && empty && svc_used() < S.limit) v = empty;
        if (v) break;
        plat_cond_wait(&S.freed, &S.lock);
    }
    v->busy = 1;
    plat_mutex_unlock(&S.lock);
    return v;
}

// Returns a child; one over the limit (svc_set_limit lowered it) is stopped.
static void svc_put(Svc *v) {
    plat_mutex_lock(&S.lock);
    int stop = v->child && svc_used() > S.limit;
    if (!stop) v->busy = 0;
    plat_mutex_unlock(&S.lock);
    if (stop) {
        child_stop(v);
        plat_mutex_lock(&S.lock);
        v->busy = 0;
        plat_mutex_unlock(&S.lock);
    }
    plat_cond_broadcast(&S.freed);
}

// Sets the number of children; idle ones over it are stopped now, busy ones when they are returned.
static void svc_set_limit(int n) {
    Svc *stop[MAX_SVC];
    int k = 0;
    plat_mutex_lock(&S.lock);
    S.limit = n < 1 ? 1 : n > MAX_SVC ? MAX_SVC : n;
    for (int i = 0, used = svc_used(); i < MAX_SVC && used > S.limit; i++)
        if (!S.s[i].busy && S.s[i].child) S.s[i].busy = 1, stop[k++] = &S.s[i], used--;
    plat_mutex_unlock(&S.lock);
    for (int i = 0; i < k; i++) child_stop(stop[i]);
    plat_mutex_lock(&S.lock);
    for (int i = 0; i < k; i++) stop[i]->busy = 0;
    plat_mutex_unlock(&S.lock);
    plat_cond_broadcast(&S.freed);
}

// One request on a child; a dead child is restarted once. 0 when the translator is unavailable. The caller has given
// up its virtual core (kern_core_away): the wait for a child and the translation are host time.
static int svc_request(const char *line, char *reply, size_t reply_size) {
    Svc *v = svc_get();
    int ok = 0;
    for (int attempt = 0; attempt < 2 && !ok; attempt++)
        if (!(ok = child_exchange(v, line, reply, reply_size))) child_stop(v);
    svc_put(v);
    return ok;
}

// One translation per id at a time (F); F.id full also waits.
static void flight_enter(const uint8_t id[32]) {
    plat_mutex_lock(&F.lock);
    for (;;) {
        int busy = F.n == 2 * MAX_SVC;
        for (int i = 0; i < F.n && !busy; i++) busy = !memcmp(F.id[i], id, 32);
        if (!busy) break;
        plat_cond_wait(&F.done, &F.lock);
    }
    memcpy(F.id[F.n++], id, 32);
    plat_mutex_unlock(&F.lock);
}

static void flight_leave(const uint8_t id[32]) {
    plat_mutex_lock(&F.lock);
    for (int i = 0; i < F.n; i++)
        if (!memcmp(F.id[i], id, 32)) {
            memcpy(F.id[i], F.id[--F.n], 32);
            break;
        }
    plat_mutex_unlock(&F.lock);
    plat_cond_broadcast(&F.done);
}

// ---- cache

static GfxShader *mem_find(const uint8_t key[32]) {
    plat_mutex_lock_shared(&cache_lock);
    GfxShader *s = buckets[(key[0] | key[1] << 8) % BUCKETS];
    while (s && memcmp(s->key, key, 32)) s = s->next;
    plat_mutex_unlock_shared(&cache_lock);
    return s;
}

static volatile int64_t spvz_free_gen;   // counts freed shaders that had a compressed module (gfx_shader_spirv's rings)
static void shader_free(GfxShader *s) {
    if (s->spvz) plat_inc64(&spvz_free_gen);
    for (int k = 0; k < GFX_RES_KINDS; k++) free(s->res[k]);
    free(s->spirv), free(s->spvz), free(s->reflection), free(s);
}

enum { REFLECT_REQUIRE_C0 = 1, REFLECT_LOWERED_BODY = 2 };   // flags of reflect, disk_load_from and disk_load
static int reflect(GfxShader *s, int flags);

// Inserts s, or frees it and returns the entry another thread inserted first.
static GfxShader *mem_insert(GfxShader *s) {
    plat_mutex_lock(&cache_lock);
    GfxShader **head = &buckets[(s->key[0] | s->key[1] << 8) % BUCKETS], *r = *head;
    while (r && memcmp(r->key, s->key, 32)) r = r->next;
    if (r) {
        shader_free(s);
        s = r;
    } else {
        s->next = *head;
        *head = s;
        n_keys++;
    }
    plat_mutex_unlock(&cache_lock);
    return s;
}

static void key_hex(const uint8_t key[32], char out[65]) {
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) out[i * 2] = hex[key[i] >> 4], out[i * 2 + 1] = hex[key[i] & 15];
    out[64] = 0;
}

// A .res.json the translator wrote: one JSON object that names its stage, no NUL inside; an interrupted write leaves
// zeros, which reflect() would take for a vertex shader with no resources.
static int reflection_valid(const char *res, uint64_t size) {
    if (strlen(res) != size) return 0;
    const char *b = res, *e = res + size;
    while (b < e && isspace((unsigned char)*b)) b++;
    while (e > b && isspace((unsigned char)e[-1])) e--;
    if (e - b < 2 || *b != '{' || e[-1] != '}') return 0;
    const char *st = strstr(b, "\"stage\"");
    if (!st) return 0;
    st = strchr(st + 7, '"');
    return st && (!strncmp(st, "\"vs\"", 4) || !strncmp(st, "\"ps\"", 4) || !strncmp(st, "\"cs\"", 4) ||
                  !strncmp(st, "\"gs\"", 4));
}

// Every instruction after the header has a word count and the last one ends at the module's end: a module with
// zero-filled clusters fails.
static int spirv_complete(const uint32_t *w, uint64_t n) {
    uint64_t i = 5;
    while (i < n) {
        uint32_t len = w[i] >> 16;
        if (!len) return 0;
        i += len;
    }
    return i == n;
}

// Reads <key>.spv and <key>.res.json from the pack (packed 1) or the loose files; NULL if either is missing or
// malformed. iid: the input id, NULL when the caller sets it.
static GfxShader *disk_load_from(const uint8_t key[32], const uint8_t *iid, int packed, int flags) {
    static volatile int32_t n_bad;
    char hex[65], p[PATH_CAP + 80];
    key_hex(key, hex);
    uint64_t spv_size = 0, res_size = 0;
    snprintf(p, PATH_CAP + 80, "%s/%s.res.json", cache_dir, hex);
    uint8_t *res = packed ? pack_read(PACK_RES, key, &res_size) : read_file(p, &res_size);
    if (!res) return NULL;
    if (!reflection_valid((const char *)res, res_size)) {
        int32_t n = plat_inc32(&n_bad);
        if (n <= 3 || !(n & (n - 1))) rt_log("gfx_shader: %s.res.json is malformed, translating again (%d so far)", hex, n);
        free(res);
        return NULL;
    }
    snprintf(p, PATH_CAP + 80, "%s/%s.spv", cache_dir, hex);
    uint8_t *spv = packed ? pack_read(PACK_SPV, key, &spv_size) : read_file(p, &spv_size);
    uint32_t magic = 0;
    if (spv && spv_size >= 20) memcpy(&magic, spv, 4);
    if (!spv || spv_size % 4 || magic != 0x07230203 || !spirv_complete((const uint32_t *)spv, spv_size / 4)) {
        free(res), free(spv);
        return NULL;
    }
    GfxShader *s = calloc(1, sizeof *s);
    memcpy(s->key, key, 32);
    if (iid) memcpy(s->iid, iid, 32);
    s->spirv = (uint32_t *)spv;
    s->words = spv_size / 4;
    s->reflection = (char *)res;
    s->reflection_size = res_size;
    if (!reflect(s, flags)) { shader_free(s); return NULL; }
// The .res.json text is freed once parsed; the use masks stay in tex_queried / img_queried.
    free(s->reflection), s->reflection = NULL, s->reflection_size = 0;
    return s;
}

// The pack, then the loose files (a newer translation of a key the pack holds a bad copy of); NULL when neither has a
// valid pair (the caller translates again and writes loose files).
static GfxShader *disk_load(const uint8_t key[32], const uint8_t *iid, int flags) {
    GfxShader *s = pack_n ? disk_load_from(key, iid, 1, flags) : NULL;
    return s ? s : disk_load_from(key, iid, 0, flags);
}

// ---- request building

static const char *const tex_names[] = {NULL, "1d", "1darray", "buffer", "2d", "2darray", "2dms", "2dmsarray", "3d",
                                        "cube", "cubearray"};   // GfxShaderAnswers tex_type / img_type codes
static const char *const fmt_names[] = {   // img_format codes: the translator's TextureFormat enum order
    NULL, "R8Unorm", "R8Snorm", "R8Uint", "R8Sint", "R16Float", "R16Unorm", "R16Snorm", "R16Uint", "R16Sint", "R32Float",
    "R32Uint", "R32Sint", "R8G8Unorm", "R8G8Snorm", "R8G8Uint", "R8G8Sint", "R16G16Float", "R16G16Unorm", "R16G16Snorm",
    "R16G16Uint", "R16G16Sint", "R32G32Float", "R32G32Uint", "R32G32Sint", "R8G8B8A8Unorm", "R8G8B8A8Snorm",
    "R8G8B8A8Uint", "R8G8B8A8Sint", "R16G16B16A16Float", "R16G16B16A16Unorm", "R16G16B16A16Snorm", "R16G16B16A16Uint",
    "R16G16B16A16Sint", "R32G32B32A32Float", "R32G32B32A32Uint", "R32G32B32A32Sint", "R10G10B10A2Unorm",
    "R10G10B10A2Uint", "R11G11B10Float"};
#define COUNT(a) ((int)(sizeof a / sizeof *a))

static const char *attr_name(uint8_t t) { return t == 1 ? "sint" : t == 2 ? "uint" : "float"; }

// ---- reflection: the .res.json the translator writes (tools/xlate/Program.cs Translate) and the SPIR-V itself

// `"name":` inside [o, e): the first byte after the colon and its spaces, NULL when absent.
static const char *json_key(const char *o, const char *e, const char *name) {
    char pat[40];
    int n = snprintf(pat, sizeof pat, "\"%s\":", name);
    for (const char *p = o; p + n <= e; p++)
        if (*p == '"' && !memcmp(p, pat, (size_t)n)) {
            p += n;
            while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
            return p;
        }
    return NULL;
}

static long json_int(const char *o, const char *e, const char *name, long def) {
    const char *p = json_key(o, e, name);
    return p ? strtol(p, NULL, 10) : def;
}

static void json_str(const char *o, const char *e, const char *name, char *buf, size_t n) {
    const char *p = json_key(o, e, name);
    size_t k = 0;
    if (p && *p == '"')
        for (p++; p < e && *p != '"' && k + 1 < n; p++) buf[k++] = *p;
    buf[k] = 0;
}

static const char *json_space(const char *p, const char *end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) p++;
    return p;
}

// Bounded JSON token traversal for the versioned geometry contract; strtol cannot tell integers from floats.
static const char *json_token_end(const char *p, const char *end, unsigned depth) {
    p = json_space(p, end);
    if (p == end || depth > 64) return NULL;
    if (*p == '"') {
        for (p++; p < end; p++) {
            if (*p == '"') return p + 1;
            if ((unsigned char)*p < 32) return NULL;
            if (*p != '\\') continue;
            if (++p == end) return NULL;
            if (*p == 'u') {
                for (int i = 0; i < 4; i++) if (++p == end || !isxdigit((unsigned char)*p)) return NULL;
            } else if (!strchr("\"\\/bfnrt", *p)) return NULL;
        }
        return NULL;
    }
    if (*p == '{' || *p == '[') {
        int object = *p == '{';
        char close = object ? '}' : ']';
        p = json_space(p + 1, end);
        if (p < end && *p == close) return p + 1;
        for (;;) {
            if (object) {
                if (p == end || *p != '"' || !(p = json_token_end(p, end, depth + 1))) return NULL;
                p = json_space(p, end);
                if (p == end || *p != ':') return NULL;
                p++;
            }
            if (!(p = json_token_end(p, end, depth + 1))) return NULL;
            p = json_space(p, end);
            if (p == end) return NULL;
            if (*p == close) return p + 1;
            if (*p != ',') return NULL;
            p = json_space(p + 1, end);
        }
    }
    for (int i = 0; i < 3; i++) {
        const char *literal = i == 0 ? "true" : i == 1 ? "false" : "null";
        size_t n = strlen(literal);
        if ((size_t)(end - p) >= n && !memcmp(p, literal, n)) return p + n;
    }
    if (*p == '-') p++;
    if (p == end || !isdigit((unsigned char)*p)) return NULL;
    if (*p == '0') p++;
    else while (p < end && isdigit((unsigned char)*p)) p++;
    if (p < end && *p == '.') {
        if (++p == end || !isdigit((unsigned char)*p)) return NULL;
        while (p < end && isdigit((unsigned char)*p)) p++;
    }
    if (p < end && (*p == 'e' || *p == 'E')) {
        p++;
        if (p < end && (*p == '+' || *p == '-')) p++;
        if (p == end || !isdigit((unsigned char)*p)) return NULL;
        while (p < end && isdigit((unsigned char)*p)) p++;
    }
    return p;
}

// NULL: absent; end: malformed object or duplicate/escaped keys. Object keys emitted by our service are plain ASCII.
static const char *json_member(const char *o, const char *end, const char *name) {
    const char *p = json_space(o, end), *found = NULL;
    size_t size = strlen(name);
    if (p == end || *p++ != '{') return end;
    for (;;) {
        p = json_space(p, end);
        if (p < end && *p == '}') return found;
        const char *key = p, *ke = json_token_end(p, end, 0);
        if (!ke || *p != '"' || memchr(p, '\\', (size_t)(ke - p))) return end;
        p = json_space(ke, end);
        if (p == end || *p++ != ':') return end;
        p = json_space(p, end);
        if ((size_t)(ke - key) == size + 2 && !memcmp(key + 1, name, size)) {
            if (found) return end;
            found = p;
        }
        if (!(p = json_token_end(p, end, 0))) return end;
        p = json_space(p, end);
        if (p < end && *p == '}') return found;
        if (p == end || *p++ != ',') return end;
    }
}

static int json_string_is(const char *p, const char *end, const char *value) {
    size_t n = strlen(value);
    return p && p < end && (size_t)(end - p) >= n + 2 && *p == '"' && !memcmp(p + 1, value, n) && p[n + 1] == '"';
}

static int json_uint(const char *o, const char *end, const char *name, uint32_t *out) {
    const char *p = json_member(o, end, name), *q;
    if (!p || p == end || !isdigit((unsigned char)*p) || !(q = json_token_end(p, end, 0))) return 0;
    uint32_t v = 0;
    for (; p < q; p++) {
        if (!isdigit((unsigned char)*p) || v > (UINT32_MAX - (uint32_t)(*p - '0')) / 10) return 0;
        v = v * 10 + (uint32_t)(*p - '0');
    }
    *out = v;
    return 1;
}

static const char *const geometry_names[] = {NULL, "points", "lines", "lines_adjacency", "triangles", "triangles_adjacency"};
static const uint32_t geometry_vertices[] = {0, 1, 2, 4, 3, 6};
static const char *const geometry_outputs[] = {NULL, "points", "line_strip", "triangle_strip"};

static int geometry_skip_json(const char *js, const char *end, uint32_t skip[8]) {
    const char *p = json_member(js, end, "gs_passthrough_skip");
    if (!p || p == end || *p++ != '[') return 0;
    for (unsigned i = 0; i < 8; i++) {
        p = json_space(p, end);
        const char *q = json_token_end(p, end, 0);
        if (!q || p == q || !isdigit((unsigned char)*p)) return 0;
        uint32_t v = 0;
        for (; p < q; p++) {
            if (!isdigit((unsigned char)*p) || v > (UINT32_MAX - (uint32_t)(*p - '0')) / 10) return 0;
            v = v * 10 + (uint32_t)(*p - '0');
        }
        skip[i] = v;
        p = json_space(p, end);
        if (p == end || *p++ != (i == 7 ? ']' : ',')) return 0;
    }
    return nvn_geometry_control_mask_valid(skip);
}

static int geometry_json(const char *js, const char *end, GfxShaderGeometry *g, uint32_t skip[8], int *source) {
    memset(g, 0, sizeof *g);
    memset(skip, 0, 32);
    *source = 0;
    if (!json_uint(js, end, "geometry_version", &g->version) ||
        (g->version != GFX_GEOMETRY_VERSION && g->version != GFX_GEOMETRY_PASSTHROUGH_VERSION)) return 0;
    const char *input = json_member(js, end, "gs_input"), *o = json_member(js, end, "geometry");
    for (unsigned i = 1; i < COUNT(geometry_names); i++)
        if (json_string_is(input, end, geometry_names[i])) g->input = i;
    if (!g->input || !o || o == end || *o != '{') return 0;
    const char *oe = json_token_end(o, end, 0);
    if (!oe) return 0;
    const char *pass = json_member(o, oe, "passthrough");
    if (!pass || pass == oe || oe - pass < 5 || memcmp(pass, "false", 5)) return 0;
    if (!json_uint(o, oe, "input_vertices", &g->input_vertices) || g->input_vertices != geometry_vertices[g->input] ||
        !json_uint(o, oe, "invocations", &g->invocations) || !g->invocations || g->invocations > 255 ||
        !json_uint(o, oe, "max_output_vertices", &g->max_output_vertices) || !g->max_output_vertices || g->max_output_vertices > 4095)
        return 0;
    const char *output = json_member(o, oe, "output_topology");
    for (unsigned i = 1; i < COUNT(geometry_outputs); i++)
        if (json_string_is(output, oe, geometry_outputs[i])) g->output = i;
    const char *src = json_member(o, oe, "source_passthrough");
    if (g->version == GFX_GEOMETRY_PASSTHROUGH_VERSION) {
        if (!src || src == oe || oe - src < 4 || memcmp(src, "true", 4) || !geometry_skip_json(js, end, skip) ||
            g->input != GFX_GEOMETRY_TRIANGLES || g->input_vertices != 3 || g->invocations != 1 ||
            g->max_output_vertices != 3 || g->output != GFX_GEOMETRY_OUTPUT_TRIANGLE_STRIP) return 0;
        *source = 1;
    } else if (src || json_member(js, end, "gs_passthrough_skip")) return 0;
    return g->output != 0;
}

// The objects of the flat array `"name": [ {...}, ... ]`: calls fn for each object's [start, end). Returns the count.
static int json_objects(const char *js, const char *name, void (*fn)(GfxShader *, int, const char *, const char *),
                        GfxShader *s, int kind) {
    const char *end = js + strlen(js), *p = json_key(js, end, name);
    if (!p || *p != '[') return 0;
    int n = 0;
    for (p++; p < end && *p != ']'; p++) {
        if (*p != '{') continue;
        const char *q = memchr(p, '}', (size_t)(end - p));
        if (!q) break;
        if (fn) fn(s, kind, p, q);
        n++;
        p = q;
    }
    return n;
}

static int fmt_code(const char *name) {
    for (int i = 1; i < COUNT(fmt_names); i++)
        if (!strcmp(fmt_names[i], name)) return i;
    return 0;
}

// c2 slot of a texture or image handle (the translator's layout: combined handles at 0x20, 32 slots; images at
// 0x120; separate textures at 0x168), -1 outside those ranges or for a handle in another constant buffer.
static int handle_slot(int kind, int cbuf_slot, int handle, int separate) {
    if (cbuf_slot >= 0 && cbuf_slot != 2) return -1;
    int bytes = (handle & 0x3fff) * 4;
    if (kind == GFX_RES_IMG) return bytes >= 0x120 && bytes < 0x168 ? (bytes - 0x120) / 8 : -1;
    if (separate) return bytes >= 0x168 && bytes < 0x568 ? (bytes - 0x168) / 8 : -1;
    return bytes >= 0x20 && bytes < 0x120 ? (bytes - 0x20) / 8 : -1;
}

static void add_res(GfxShader *s, int kind, const char *o, const char *e) {
    GfxShaderRes *r = &s->res[kind][s->nres[kind]++];
    r->set = 0;
    r->binding = (uint16_t)json_int(o, e, "binding", 0);
    if (kind == GFX_RES_UBO || kind == GFX_RES_SSBO) {
        r->slot = (int16_t)json_int(o, e, "slot", -1);
        r->count = 1;
        if (kind == GFX_RES_SSBO) {
            r->cbuf_slot = (int8_t)json_int(o, e, "cb_slot", -1);
            r->handle = (uint16_t)json_int(o, e, "cb_offset", 0);
            r->flags = (uint8_t)json_int(o, e, "flags", 0);
// NVN 2 compute: the address's place in the packed bank, as the NVN 1 offset of the same buffer
// (uniform buffer u at 0x210 + 16u, storage buffer k at 0x310 + 16k), which is what ssbo_slot reads.
            int off = r->handle * 4, ub = (int)nvn2_c0_part(&s->c0, 1), sb = (int)nvn2_c0_part(&s->c0, 2),
                end = (int)nvn2_c0_part(&s->c0, 3);
            if (!(r->flags & GFX_RES_SSBO_GUARDED) && s->has_c0 && r->cbuf_slot == 0 &&
                off >= ub && off < end && (off - ub) % 16 == 0)
                r->handle = (uint16_t)((off < sb ? 0x210 + (off - ub) : 0x310 + (off - sb)) / 4);
        }
        return;
    }
    char buf[32];
    json_str(o, e, "format", buf, sizeof buf);
    const char *sep = json_key(o, e, "separate");
    r->type = (uint16_t)json_int(o, e, "type", 0);
    r->format = (uint8_t)fmt_code(buf);
    r->flags = (uint8_t)json_int(o, e, "flags", 0);
    r->cbuf_slot = (int8_t)json_int(o, e, "cbuf_slot", -1);
    int handle = (int)json_int(o, e, "handle", 0);
    r->handle = (uint16_t)handle;
    r->count = (uint16_t)json_int(o, e, "array", 1);
    r->separate = sep && *sep == 't';
    r->slot = (int16_t)handle_slot(kind, r->cbuf_slot, handle, r->separate);
    if (s->has_c0 && (r->cbuf_slot <= 0 || r->cbuf_slot == 2)) {   // NVN 2 compute: packed-bank handles, 8 bytes each
        int bytes = (handle & 0x3fff) * 4, at = (int)nvn2_c0_part(&s->c0, kind == GFX_RES_IMG ? 4 : 3);
        int n = kind == GFX_RES_IMG ? s->c0.images : s->c0.textures;
        r->slot = (int16_t)(bytes >= at && bytes < at + 8 * n && (bytes - at) % 8 == 0 ? (bytes - at) / 8 : -1);
    }
}

enum { OP_ENTRY_POINT = 15, OP_EXECUTION_MODE = 16, OP_TYPE_INT = 21, OP_TYPE_FLOAT = 22, OP_TYPE_VECTOR = 23,
       OP_TYPE_ARRAY = 28, OP_TYPE_STRUCT = 30, OP_TYPE_POINTER = 32, OP_CONSTANT = 43, OP_VARIABLE = 59, OP_DECORATE = 71,
       OP_MEMBER_DECORATE = 72,
       DEC_BUILTIN = 11, DEC_LOCATION = 30, DEC_COMPONENT = 31, SC_INPUT = 1, SC_OUTPUT = 3, EM_LOCAL_SIZE = 17 };

// The service emits one GS main with four execution modes; validate them against the JSON on every disk load.
static int geometry_spirv(const GfxShader *s) {
    const uint32_t *w = s->spirv;
    uint32_t entry = 0, modes = 0, capability = 0;
    int gs = s->stage == GFX_STAGE_GS;
    if (s->words < 5) return !gs;
    for (uint64_t i = 5; i < s->words;) {
        uint32_t len = w[i] >> 16, op = w[i] & 0xffff;
        if (!len || len > s->words - i) return 0;
        const uint32_t *a = w + i + 1;
        if (op == 17 && len >= 2) {                      // OpCapability
            if (a[0] == 5251) return 0;                  // GeometryShaderPassthroughNV
            if (a[0] == 2) capability = 1;               // Geometry
        } else if (op == 10 && len >= 2) {               // OpExtension
            static const char pass[] = "SPV_NV_geometry_shader_passthrough";
            if ((len - 1) * 4 >= sizeof pass && !memcmp(a, pass, sizeof pass)) return 0;
        } else if ((op == 71 && len >= 3 && a[1] == 5250) ||
                   (op == 72 && len >= 4 && a[2] == 5250)) return 0;   // PassthroughNV decoration/member decoration
        else if (op == OP_ENTRY_POINT) {
            if (len < 4 || (!gs && a[0] == 3)) return 0;
            if (gs) {
                const char *name = (const char *)(a + 2);
                if (entry || a[0] != 3 || !a[1] || a[1] >= w[3] ||
                    (len - 3) * 4 < 5 || memcmp(name, "main", 5)) return 0;
                entry = a[1];
            }
        } else if (gs && op == OP_EXECUTION_MODE) {
            if (len < 3 || !entry || a[0] != entry) return 0;
            uint32_t bit = 0;
            if (a[1] >= 19 && a[1] <= 23) {
                bit = 1;
                if (len != 3 || a[1] != 18 + s->geometry.input) return 0;
            } else if (a[1] == 0) {                       // Invocations
                bit = 2;
                if (len != 4 || a[2] != s->geometry.invocations) return 0;
            } else if (a[1] == 26) {                      // OutputVertices
                bit = 4;
                if (len != 4 || a[2] != s->geometry.max_output_vertices) return 0;
            } else if (a[1] >= 27 && a[1] <= 29) {
                bit = 8;
                if (len != 3 || a[1] != 26 + s->geometry.output) return 0;
            } else return 0;
            if (modes & bit) return 0;
            modes |= bit;
        }
        i += len;
    }
    return !gs || (entry && capability && modes == 15);
}

// Interface Locations and the compute workgroup size from the module; ids beyond 2^20 are ignored.
static void reflect_spirv(GfxShader *s) {
    const uint32_t *w = s->spirv;
    uint64_t n = s->words;
    s->local[0] = s->local[1] = s->local[2] = 1;
    if (n < 5) return;
    uint32_t bound = w[3] < (1u << 20) ? w[3] : (1u << 20);
    // per id: location + 1 (0 = none), component flag, scalar/vector description of types, pointee of pointers
    int32_t *loc = calloc(bound, sizeof *loc);
    uint8_t *packed = calloc(bound, 1), *ttype = calloc(bound, 1), *tcomps = calloc(bound, 1);
    uint8_t *width = calloc(bound, 1), *span = calloc(bound, 1), *length = calloc(bound, 1);
    uint8_t *array_element_span = calloc(bound, 1);
    uint32_t *pointee = calloc(bound, sizeof *pointee);
    int portable = s->geometry_source_passthrough;
    uint8_t *posdesc = portable ? calloc(bound, 1) : NULL, *poshas = portable ? calloc(bound, 1) : NULL;
    uint8_t *posvar = portable ? calloc(bound, 1) : NULL;
    uint32_t *posmember = portable ? calloc(bound, sizeof *posmember) : NULL;
    s->geometry_position = 0;
    if (!loc || !packed || !ttype || !tcomps || !width || !span || !length || !array_element_span || !pointee) goto out;
    if (portable && (!posdesc || !poshas || !posvar || !posmember)) goto out;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) break;
        const uint32_t *a = w + i + 1;
        uint32_t id = len > 1 ? a[0] : 0;
        if (op == OP_EXECUTION_MODE && len >= 6 && a[1] == EM_LOCAL_SIZE) {
            s->local[0] = a[2], s->local[1] = a[3], s->local[2] = a[4];
        } else if (op == OP_DECORATE && len >= 4 && id < bound) {
            if (a[1] == DEC_LOCATION) loc[id] = a[2] < 32 ? (int32_t)a[2] + 1 : 0;
            else if (a[1] == DEC_COMPONENT) packed[id] = 1;
            else if (portable && a[1] == DEC_BUILTIN && a[2] == 0) posvar[id] = 1;
            if (portable && a[1] == DEC_LOCATION && a[2] >= 32) s->geometry_position |= 4;
        } else if (portable && op == OP_MEMBER_DECORATE && len >= 5 && id < bound && a[2] == DEC_BUILTIN && a[3] == 0) {
            posmember[id] = !posmember[id] && a[1] < bound - 1 ? a[1] + 1 : UINT32_MAX;
        } else if (op == OP_TYPE_INT && len >= 4 && id < bound) {
            ttype[id] = a[2] ? GFX_TYPE_SINT : GFX_TYPE_UINT, tcomps[id] = 1;
            width[id] = a[1] <= UINT8_MAX ? (uint8_t)a[1] : 0, span[id] = 1;
        } else if (op == OP_TYPE_FLOAT && len >= 3 && id < bound) {
            ttype[id] = GFX_TYPE_FLOAT, tcomps[id] = 1;
            width[id] = a[1] <= UINT8_MAX ? (uint8_t)a[1] : 0, span[id] = 1;
        } else if (op == OP_TYPE_VECTOR && len >= 4 && id < bound && a[1] < bound) {
            ttype[id] = ttype[a[1]], tcomps[id] = (uint8_t)a[2];
            width[id] = width[a[1]], span[id] = span[a[1]] && a[2] >= 2 && a[2] <= 4;
            if (portable && ttype[id] == GFX_TYPE_FLOAT && width[id] == 32 && a[2] == 4) posdesc[id] = 1;
        } else if (portable && op == OP_TYPE_STRUCT && id < bound && posmember[id]) {
            poshas[id] = 1;
            uint32_t member = posmember[id];
            if (member < len - 1 && a[member] < bound && posdesc[a[member]] == 1) posdesc[id] = 2;
        } else if (op == OP_CONSTANT && len >= 4 && a[0] < bound && a[1] < bound) {
            uint32_t type = a[0], value = a[2];
            if (ttype[type] != GFX_TYPE_FLOAT && tcomps[type] == 1 &&
                ((width[type] == 32 && len == 4 && value &&
                  (ttype[type] == GFX_TYPE_UINT || !(value >> 31))) ||
                 (width[type] == 64 && len == 5 && (value || a[3]) &&
                  (ttype[type] == GFX_TYPE_UINT || !(a[3] >> 31)))))
                length[a[1]] = (width[type] == 64 && a[3]) || value > 32 ? 32 : (uint8_t)value;
        } else if (op == OP_TYPE_ARRAY && len >= 4 && id < bound && a[1] < bound) {
            ttype[id] = ttype[a[1]], tcomps[id] = tcomps[a[1]], width[id] = width[a[1]];
// A scalar/vector array takes one location per element, nested arrays included. Cap each factor before multiplying;
// unknown lengths and element widths stay unsupported.
            if (width[id] == 32 && a[2] < bound) {
                array_element_span[id] = span[a[1]];
                uint32_t slots = (uint32_t)span[a[1]] * length[a[2]];
                span[id] = slots > 32 ? 32 : (uint8_t)slots;
            }
            if (portable) {
                poshas[id] = poshas[a[1]];
                if (a[2] < bound && length[a[2]] == 3 && (posdesc[a[1]] == 1 || posdesc[a[1]] == 2))
                    posdesc[id] = (uint8_t)(posdesc[a[1]] * 4);
            }
        } else if (op == OP_TYPE_POINTER && len >= 4 && id < bound) {
            pointee[id] = a[2];
        } else if (op == OP_VARIABLE && len >= 4) {
            uint32_t type = a[0], var = a[1], sc = a[2];
            if (portable && (sc == SC_INPUT || sc == SC_OUTPUT) && var < bound && type < bound) {
                uint32_t t = pointee[type] < bound ? pointee[type] : 0;
                if (posvar[var] || poshas[t]) {
                    unsigned bit = sc == SC_INPUT ? 1 : 2;
                    unsigned expected = posvar[var] ? (sc == SC_INPUT ? 4 : 1) : (sc == SC_INPUT ? 8 : 2);
                    if (posdesc[t] != expected || (s->geometry_position & bit)) s->geometry_position |= 4;
                    else s->geometry_position |= bit;
                }
            }
            if ((sc == SC_INPUT || sc == SC_OUTPUT) && var < bound && type < bound && loc[var] > 0 && loc[var] <= 32) {
                int l = loc[var] - 1;
                uint32_t t = pointee[type] < bound ? pointee[type] : 0;
                GfxShaderIo io = {ttype[t], tcomps[t] ? tcomps[t] : 4, packed[var], width[t]};
                uint32_t slots = span[t];
                if (portable && ((sc == SC_INPUT ? (slots != 3 || array_element_span[t] != 1) :
                                                  (slots != 1 || array_element_span[t] != 0)) ||
                    ((sc == SC_INPUT ? s->in_mask : s->out_mask) >> l & 1))) s->geometry_position |= 4;
// A GS input's outer fixed array indexes vertices, not Locations; keep its element span.
                if (sc == SC_INPUT && s->stage == GFX_STAGE_GS && slots && array_element_span[t])
                    slots = array_element_span[t];
                for (uint32_t j = 0; j < slots && l + j < 32; j++) {
                    if (sc == SC_INPUT) s->in_mask |= 1u << (l + j), s->in[l + j] = io;
                    else s->out_mask |= 1u << (l + j), s->out[l + j] = io;
                }
            }
        }
        i += len;
    }
out:
    free(loc), free(packed), free(ttype), free(tcomps), free(width), free(span), free(length), free(array_element_span), free(pointee);
    free(posdesc), free(poshas), free(posvar), free(posmember);
}

// This ABI field is an exact JSON integer, not a numeric prefix such as 1.5 or 1e1. Missing means legacy version 0.
static int storage_contract_version(const char *js, const char *end) {
    const char *p = json_member(js, end, "storage_candidates_version");
    if (!p) return 0;
    uint32_t version;
    return json_uint(js, end, "storage_candidates_version", &version) && version <= 1 ? (int)version : -1;
}

static uint32_t json_slot_mask(const char *js, const char *name);

static int reflect(GfxShader *s, int flags) {
    const int require_c0 = flags & REFLECT_REQUIRE_C0;
    static const char *const kinds[GFX_RES_KINDS] = {"cbufs", "sbufs", "textures", "images"};
    const char *js = s->reflection;
    const char *end = js + strlen(js);
    s->tex_queried = json_slot_mask(js, "tex_queried"), s->img_queried = json_slot_mask(js, "img_queried");
    const char *whole = json_token_end(js, end, 0), *stage = json_member(js, end, "stage");
    if (!whole || json_space(whole, end) != end) return 0;
    s->stage = json_string_is(stage, end, "vs") ? GFX_STAGE_VS : json_string_is(stage, end, "ps") ? GFX_STAGE_FS :
               json_string_is(stage, end, "cs") ? GFX_STAGE_CS : json_string_is(stage, end, "gs") ? GFX_STAGE_GS : -1;
    if (s->stage < 0) return 0;
    if (s->stage == GFX_STAGE_GS) {
        if (!host_gs_sg || !geometry_json(js, end, &s->geometry, s->geometry_skip, &s->geometry_source_passthrough)) return 0;
    } else if (json_member(js, end, "geometry_version") || json_member(js, end, "gs_input") ||
               json_member(js, end, "geometry") || json_member(js, end, "gs_passthrough_skip")) return 0;
    if (!geometry_spirv(s)) return 0;
    // Driver contracts are unique root fields. Extra nested properties are metadata, not alternate layouts.
    const char *c0 = json_member(js, end, "c0"), *roots = json_member(js, end, "c0_roots"),
               *c0_version = json_member(js, end, "c0_version");
    if (c0 == end || roots == end || c0_version == end) return 0;
    if (s->stage == GFX_STAGE_GS && (require_c0 || c0 || roots || c0_version)) return 0;
    if (!c0 && (roots || c0_version)) return 0;
    if (c0) {
        uint32_t version;
        if (*c0 != '[' || !json_uint(js, end, "c0_version", &version) || version != 2 || !roots || *roots != '[') {
            rt_log("gfx_shader: NVN 2 reflection lacks root layout version 2; rebuild the shader translator");
            return 0;   // an old translator must not silently use image counts as root-table sizes
        }
        for (int a = 0; a < 2; a++) {
            char *q = (char *)(a ? roots : c0) + 1;
            uint8_t *dst = a ? s->c0.roots : &s->c0.images;
            int n = a ? NVN2_ROOT_TABLES : 4;
            for (int i = 0; i < n; i++) {
                while (isspace((unsigned char)*q)) q++;
                if (!isdigit((unsigned char)*q)) return 0;
                unsigned long v = strtoul(q, &q, 10);
                if (v > 255) return 0;
                dst[i] = (uint8_t)v;
                while (isspace((unsigned char)*q)) q++;
                if (*q++ != (i + 1 == n ? ']' : ',')) return 0;
            }
        }
        if (!nvn2_c0_valid(&s->c0, NVN2_C0_BYTES)) return 0;
        s->has_c0 = 1;
    }
    if (require_c0 && !s->has_c0) {
        rt_log("gfx_shader: translator omitted the NVN 2 root layout; rebuild the shader translator");
        return 0;
    }
    int storage_version = storage_contract_version(js, end);
    if ((require_c0 && storage_version != GFX_STORAGE_CANDIDATES_VERSION) ||
        (storage_version && (storage_version != GFX_STORAGE_CANDIDATES_VERSION ||
                             !s->has_c0 || s->stage != GFX_STAGE_CS))) {
        rt_log("gfx_shader: incompatible guarded storage contract; rebuild the shader translator");
        return 0;
    }
    for (int k = 0; k < GFX_RES_KINDS; k++) {
        int count = json_objects(js, kinds[k], NULL, s, k);
        s->res[k] = calloc(count ? count : 1, sizeof *s->res[k]);
        json_objects(js, kinds[k], add_res, s, k);
    }
    for (int i = 0; i < s->nres[GFX_RES_SSBO]; i++)
        if ((s->res[GFX_RES_SSBO][i].flags & GFX_RES_SSBO_GUARDED) &&
            storage_version != GFX_STORAGE_CANDIDATES_VERSION) {
            rt_log("gfx_shader: guarded storage resource without its versioned binding contract");
            return 0;
        }
    reflect_spirv(s);
    if (s->geometry_source_passthrough) {
        if (s->geometry_position != 3) return 0;
        uint32_t locations = 0;
        for (unsigned i = 0; i < 32; i++)
            if (!(s->geometry_skip[1 + i / 8] >> ((i % 8) * 4) & 15)) locations |= 1u << i;
        // Portable-v2 metadata describes this exact interface. Only a stage whose control says it was lowered from a
        // native one (nvn_geometry_control_pack_lowered) may read or declare more than it forwards.
        if (flags & REFLECT_LOWERED_BODY) {
            if ((s->in_mask & locations) != locations || (s->out_mask & locations) != locations) return 0;
        } else if (s->in_mask != locations || s->out_mask != locations) return 0;
        for (unsigned i = 0; i < 32; i++) if (locations >> i & 1) {
            const GfxShaderIo *a = &s->in[i], *b = &s->out[i];
            if (a->scalar_bits != 32 || b->scalar_bits != 32 || a->comps != 4 || b->comps != 4 ||
                a->packed || b->packed || a->type > GFX_TYPE_UINT || a->type != b->type) return 0;
        }
    }
    return 1;
}

// ---- use masks (see the cache key at the top)

// The slots of the int array `"name": [...]` below 32 as a mask.
static uint32_t json_slot_mask(const char *js, const char *name) {
    const char *end = js + strlen(js), *p = json_key(js, end, name);
    uint32_t m = 0;
    if (!p || *p != '[') return 0;
    for (p++; p < end && *p != ']';) {
        char *q;
        long v = strtol(p, &q, 10);
        if (q == p) {
            p++;
            continue;
        }
        if (v >= 0 && v < 32) m |= 1u << v;
        p = q;
    }
    return m;
}

static UseMask use_of(const GfxShader *s) {
    uint32_t type_q = s->tex_queried, format_q = s->img_queried;   // from the .res.json (reflect)
    UseMask m = {.slots = type_q | format_q, .tex_type = type_q | format_q, .img_type = type_q | format_q,
                 .img_format = type_q | format_q};
    if (s->stage == GFX_STAGE_VS) m.attrs = s->in_mask;
    else if (s->stage == GFX_STAGE_FS) m.fouts = s->out_mask & 0xff;
    if (s->stage == GFX_STAGE_CS && s->has_c0) {          // the translator's table of a queried slot is the resource's table
        uint32_t tex = 0, img = 0;
        for (uint32_t i = 0; i < s->nres[GFX_RES_TEX]; i++)
            if (s->res[GFX_RES_TEX][i].slot >= 0 && s->res[GFX_RES_TEX][i].slot < 32) tex |= 1u << s->res[GFX_RES_TEX][i].slot;
        for (uint32_t i = 0; i < s->nres[GFX_RES_IMG]; i++)
            if (s->res[GFX_RES_IMG][i].slot >= 0 && s->res[GFX_RES_IMG][i].slot < 32) img |= 1u << s->res[GFX_RES_IMG][i].slot;
        uint32_t neither = m.slots & ~(tex | img);        // a query with no resource: either table
        tex |= neither, img |= neither;
        m.tex_type = type_q & tex, m.tex_class = format_q & tex & ~type_q;
        m.img_type = type_q & img, m.img_format = format_q & img;
    }
    return m;
}

// The words of a .use file; 0 when they are not one. Version 2 files read every table of a slot, as they always did.
static int use_parse(const uint32_t *u, uint64_t bytes, UseMask *m) {
    if (!u || bytes < 20 || u[0] != USE_MAGIC) return 0;
    memset(m, 0, sizeof *m);
    m->slots = u[2], m->attrs = u[3], m->fouts = u[4];
    if (u[1] == 2 && bytes == 20) m->tex_type = m->img_type = m->img_format = m->slots;
    else if (u[1] == USE_VERSION && bytes == 36) m->tex_type = u[5], m->tex_class = u[6], m->img_type = u[7], m->img_format = u[8];
    else return 0;
    return 1;
}

// <sha256("TKSU", translator identity, code hash)>.use: masks are per translator.
static void use_name(const uint8_t h[32], uint8_t name[32]) {
    uint8_t buf[4 + 32 + 32];
    memcpy(buf, "TKSU", 4), memcpy(buf + 4, xlate_id, 32), memcpy(buf + 36, h, 32);
    rt_sha256(buf, sizeof buf, name);
}

static void use_path_of(const uint8_t name[32], char *p) {
    char hex[65];
    key_hex(name, hex);
    snprintf(p, PATH_CAP + 80, "%s/%s.use", cache_dir, hex);
}

static void use_path(const uint8_t h[32], char *p) {
    uint8_t name[32];
    use_name(h, name);
    use_path_of(name, p);
}

// Every .use file of the cache, read by a background thread once the paths are set, so the render thread opens no
// file. The table is read-only once use_files_ready is 1; later masks are in use_buckets.
#define USE_FILE_BUCKETS 65536
typedef struct UseFile { struct UseFile *next; uint8_t name[32]; UseMask m; } UseFile;
static UseFile *use_files[USE_FILE_BUCKETS];
static volatile int32_t use_files_ready;

static UseFile **use_file_slot(const uint8_t name[32]) {
    UseFile **e = &use_files[(name[0] | name[1] << 8) % USE_FILE_BUCKETS];
    while (*e && memcmp((*e)->name, name, 32)) e = &(*e)->next;
    return e;
}

static int hex_byte(const char *c) {
    int v = 0;
    for (int i = 0; i < 2; i++) {
        int d = isdigit((unsigned char)c[i]) ? c[i] - '0' : c[i] >= 'a' && c[i] <= 'f' ? c[i] - 'a' + 10 : -1;
        if (d < 0) return -1;
        v = v << 4 | d;
    }
    return v;
}

typedef struct { int n, bad; } UsePreload;

static int use_preload_file(const char *name, void *ctx) {
    UsePreload *st = ctx;
    char p[PATH_CAP + 80];
    UseFile e = {0};
    int ok = strlen(name) == 68;
    for (int i = 0; ok && i < 32; i++) {
        int v = hex_byte(name + 2 * i);
        ok = v >= 0, e.name[i] = (uint8_t)v;
    }
    uint64_t len = 0;
    uint32_t *u = NULL;
    if (ok) {
        snprintf(p, sizeof p, "%s/%s", cache_dir, name);
        u = (uint32_t *)read_file(p, &len);
        ok = use_parse(u, len, &e.m);
    }
    UseFile **slot = ok ? use_file_slot(e.name) : NULL, *x;
    if (slot && !*slot && (x = malloc(sizeof *x))) {
        *x = e, *slot = x, st->n++;
    } else if (!ok) {
        st->bad++;
    }
    free(u);
    return 1;
}

static void use_preload(void *arg) {
    (void)arg;
    plat_thread_priority(PLAT_PRIO_BELOW_NORMAL);
    uint64_t t0 = plat_now_ns();
    int n = 0, bad = 0;
    for (uint32_t i = 0; i < pack_n; i++) {   // the packed masks; the loose files below add the masks found since
        uint32_t u[9];
        UseFile **slot, *x;
        UseMask m;
        if (pack_ix[i].kind != PACK_USE) continue;
        if (pack_ix[i].size > sizeof u || !pack_read_at(pack_ix[i].offset, u, pack_ix[i].size) ||
            !use_parse(u, pack_ix[i].size, &m)) {
            bad++;
            continue;
        }
        if (!*(slot = use_file_slot(pack_ix[i].name)) && (x = malloc(sizeof *x))) {
            memcpy(x->name, pack_ix[i].name, 32);
            x->next = NULL, x->m = m;
            *slot = x, n++;
        }
    }
    UsePreload st = {0};
    plat_file_list(cache_dir, ".use", use_preload_file, &st);
    n += st.n, bad += st.bad;
    plat_xchg32(&use_files_ready, 1);
    rt_log("gfx_shader: %d use masks read in %.0f ms on a background thread (%d unreadable)", n,
           (plat_now_ns() - t0) / 1e6, bad);
    return;
}

static void use_preload_start(void) {
    if (!cache_dir[0]) return;
    plat_thread_spawn(use_preload, NULL, "use preload", 0);
}

static UseEnt **use_slot(const uint8_t h[32]) {         // caller holds cache_lock
    UseEnt **e = &use_buckets[(h[0] | h[1] << 8) % BUCKETS];
    while (*e && memcmp((*e)->code_hash, h, 32)) e = &(*e)->next;
    return e;
}

static void use_put(const uint8_t h[32], int known, const UseMask *m) {
    UseEnt **e = use_slot(h);
    if (!*e && !(*e = calloc(1, sizeof **e))) return;
    memcpy((*e)->code_hash, h, 32);
    if (known) (*e)->known = 1, (*e)->m = *m;
}

// 1 with *m when the use mask of h is known (memory, then <hash>.use; a disk miss is remembered).
static int use_find(const uint8_t h[32], UseMask *m) {
    plat_mutex_lock_shared(&cache_lock);
    UseEnt *e = *use_slot(h);
    int seen = e != NULL, known = e && e->known;
    if (known) *m = e->m;
    plat_mutex_unlock_shared(&cache_lock);
    if (seen) return known;
    uint8_t name[32];
    use_name(h, name);
    if (plat_cas32(&use_files_ready, 0, 0)) {   // complete: a name not in it has no file
        UseFile *u = *use_file_slot(name);
        if ((known = u != NULL)) *m = u->m;
    } else {
        char p[PATH_CAP + 80];
        use_path_of(name, p);
        uint64_t n = 0;
        uint32_t *f = (uint32_t *)pack_read(PACK_USE, name, &n);
        if (!f) f = (uint32_t *)read_file(p, &n);
        known = use_parse(f, n, m);
        free(f);
    }
    plat_mutex_lock(&cache_lock);
    use_put(h, known, m);
    plat_mutex_unlock(&cache_lock);
    return known;
}

static void use_store(const uint8_t h[32], const UseMask *m) {
    char p[PATH_CAP + 80];
    use_path(h, p);
    uint32_t f[9] = {USE_MAGIC, USE_VERSION, m->slots, m->attrs, m->fouts, m->tex_type, m->tex_class, m->img_type, m->img_format};
    plat_file_write_atomic(p, f, sizeof f);
    plat_mutex_lock(&cache_lock);
    use_put(h, 1, m);
    plat_mutex_unlock(&cache_lock);
}

static void put_u32(uint8_t **p, uint32_t v) { memcpy(*p, &v, 4), *p += 4; }

// Fixed-layout answers for hashing; returns the byte count (at most 256).
static size_t pack_answers(uint8_t *o, const GfxShaderAnswers *a) {
    uint8_t *p = o;
    GfxShaderAnswers zero = {0};
    if (!a) a = &zero;
    memcpy(p, a->tex_type, 32), p += 32;
    memcpy(p, a->img_type, 32), p += 32;
    memcpy(p, a->img_format, 32), p += 32;
    memcpy(p, a->attr_type, 32), p += 32;
    memcpy(p, a->frag_out_type, 8), p += 8;
    for (int i = 0; i < 3; i++) *p++ = (uint8_t)a->local[i], *p++ = (uint8_t)(a->local[i] >> 8);
    put_u32(&p, a->shared_bytes);
    *p++ = a->depth_minus_w;
    return (size_t)(p - o);
}

static void unpack_answers(const uint8_t *p, GfxShaderAnswers *a);

// Versioned GS storage/record codec. Hashing deliberately retains the exact legacy 147-byte sequence.
static size_t pack_geometry_answers(uint8_t *o, const GfxShaderAnswers *a) {
    size_t n = pack_answers(o, a);
    o[n++] = a ? a->geometry_input : 0;
    return n;
}

static void unpack_geometry_answers(const uint8_t *p, GfxShaderAnswers *a) {
    unpack_answers(p, a);
    a->geometry_input = p[ANSWERS_PACKED];
}

// The original .in format stores a 152-byte little-endian C ABI: two padding bytes before shared_bytes and three
// after depth_minus_w. Name neither padding as a new answer nor depend on the current struct's sizeof/offsets.
static void pack_legacy_input_answers(uint8_t out[LEGACY_ANS_SIZE], const GfxShaderAnswers *a) {
    uint8_t packed[ANSWERS_PACKED];
    pack_answers(packed, a);
    memset(out, 0, LEGACY_ANS_SIZE);
    memcpy(out, packed, 142), memcpy(out + 144, packed + 142, 4), out[148] = packed[146];
}

static void unpack_legacy_input_answers(const uint8_t in[LEGACY_ANS_SIZE], GfxShaderAnswers *a) {
    uint8_t packed[ANSWERS_PACKED];
    memcpy(packed, in, 142), memcpy(packed + 142, in + 144, 4), packed[146] = in[148];
    unpack_answers(packed, a);
}

static int build_code_hash(int stage, const uint8_t *control, const uint8_t *code, uint64_t code_size,
                           const uint8_t *c1, uint32_t c1_len, uint8_t geometry_input, uint8_t hash[32]) {
    if ((stage == GFX_STAGE_GS && (!geometry_input || geometry_input >= COUNT(geometry_names))) ||
        (stage != GFX_STAGE_GS && geometry_input)) return 0;
    size_t total = 4 + CONTROL_SIZE + 8 + (size_t)code_size + 4 + c1_len + (stage == GFX_STAGE_GS ? 12 : 0);
    uint8_t *buf = malloc(total), *p = buf;
    if (!buf) return 0;
    put_u32(&p, (uint32_t)stage);
    memcpy(p, control, CONTROL_SIZE), p += CONTROL_SIZE;
    memcpy(p, &code_size, 8), p += 8;
    memcpy(p, code, (size_t)code_size), p += code_size;
    put_u32(&p, c1_len);
    if (c1_len) memcpy(p, c1, c1_len);
    p += c1_len;
    if (stage == GFX_STAGE_GS) {
        memcpy(p, "TKGS", 4), p += 4;
        int metadata = nvn_geometry_control_get(control, CONTROL_SIZE, code, code_size, NULL);
        if (metadata < 0) return free(buf), 0;
        put_u32(&p, metadata ? GFX_GEOMETRY_PASSTHROUGH_VERSION : GFX_GEOMETRY_VERSION), put_u32(&p, geometry_input);
    }
    rt_sha256(buf, total, hash);
    free(buf);
    return 1;
}

// The answers with everything outside the mask zeroed; the translator ignores depth_minus_w (Program.cs).
static void mask_answers(const GfxShaderAnswers *a, const UseMask *m, GfxShaderAnswers *o) {
    memset(o, 0, sizeof *o);
    if (!a) return;
    for (int i = 0; i < 32; i++) {
        if (m->tex_type >> i & 1) o->tex_type[i] = a->tex_type[i];
        else if (m->tex_class >> i & 1) o->tex_type[i] = a->tex_type[i] & 0xc0;
        if (m->img_type >> i & 1) o->img_type[i] = a->img_type[i];
        if (m->img_format >> i & 1) o->img_format[i] = a->img_format[i];
        if (m->attrs >> i & 1) o->attr_type[i] = a->attr_type[i];
    }
    for (int i = 0; i < 8; i++)
        if (m->fouts >> i & 1) o->frag_out_type[i] = a->frag_out_type[i];
    memcpy(o->local, a->local, sizeof o->local);
    o->shared_bytes = a->shared_bytes;
}

static void key_of_input(const uint8_t iid[32], uint8_t key[32]) {
    uint8_t buf[8 + 32 + 32], *p = buf;
    memcpy(p, "TKSK", 4), p += 4;
    put_u32(&p, KEY_VERSION);
    memcpy(p, xlate_id, 32), p += 32;
    memcpy(p, iid, 32), p += 32;
    rt_sha256(buf, (size_t)(p - buf), key);
}

// The input id and cache key (m given), or the provisional key (m NULL: full answers; iid untouched).
static void build_key(const uint8_t code_hash[32], const GfxShaderAnswers *a, const UseMask *m, uint8_t key[32],
                      uint8_t iid[32]) {
    GfxShaderAnswers masked;
    uint8_t buf[8 + 32 + 32 + 256], *p = buf;
    if (m) {
        memcpy(p, "TKSI", 4), p += 4;
        put_u32(&p, IN_VERSION);
        memcpy(p, code_hash, 32), p += 32;
        mask_answers(a, m, &masked);
        p += pack_answers(p, &masked);
        rt_sha256(buf, (size_t)(p - buf), iid);
        key_of_input(iid, key);
        return;
    }
    memcpy(p, "TKSP", 4), p += 4;
    put_u32(&p, KEY_VERSION);
    memcpy(p, xlate_id, 32), p += 32;
    memcpy(p, code_hash, 32), p += 32;
    p += pack_answers(p, a);
    rt_sha256(buf, (size_t)(p - buf), key);
}

static size_t build_request(char *o0, const char *hex, const char *stage, const char *code_path, const uint8_t *c1,
                            uint32_t c1_len, const GfxShaderAnswers *a, const uint8_t *control) {
    char *o = o0;
    o += sprintf(o, "{\"job\":{\"key\":\"");
    json_esc(&o, hex);
    o += sprintf(o, "\",\"stage\":\"%s\",\"debug\":false,\"code\":\"", stage);
    json_esc(&o, code_path);
    o += sprintf(o, "\",\"c1\":\"");
    for (uint32_t i = 0; i < c1_len; i++) o += sprintf(o, "%02x", c1[i]);
    o += sprintf(o, "\"");
    if (!strcmp(stage, "gs")) {
        uint32_t skip[8];
        int pass = !memcmp(control + NVN_GEOMETRY_CONTROL_OFFSET, "TKGSMETA", 8);
        o += sprintf(o, ",\"geometry_version\":%u,\"gs_input\":\"%s\"",
                     pass ? GFX_GEOMETRY_PASSTHROUGH_VERSION : GFX_GEOMETRY_VERSION, geometry_names[a->geometry_input]);
        if (pass) {
            o += sprintf(o, ",\"gs_passthrough_skip\":[");
            for (unsigned i = 0; i < 8; i++) {
                skip[i] = nvn_geometry_control_u32(control + NVN_GEOMETRY_CONTROL_OFFSET + 24 + 4 * i);
                o += sprintf(o, "%s%u", i ? "," : "", skip[i]);
            }
            o += sprintf(o, "]");
        }
    }
    if (a) {
        const char *sep = "";
        o += sprintf(o, ",\"tex\":{");
        for (int i = 0; i < 32; i++) {
            unsigned type = a->tex_type[i] & 0x3f;
            if (type && type < COUNT(tex_names)) {
                const char *format = a->tex_type[i] & 0x40 ? "R32Uint" : a->tex_type[i] & 0x80 ? "R32Sint" : "R8G8B8A8Unorm";
                o += sprintf(o, "%s\"%d\":{\"type\":\"%s\",\"fmt\":\"%s\"}", sep, i, tex_names[type], format), sep = ",";
            }
        }
        sep = "";
        o += sprintf(o, "},\"img\":{");
        for (int i = 0; i < 32; i++)
            if (a->img_type[i] && a->img_type[i] < COUNT(tex_names)) {
                const char *f = a->img_format[i] && a->img_format[i] < COUNT(fmt_names) ? fmt_names[a->img_format[i]] : "R8G8B8A8Unorm";
                o += sprintf(o, "%s\"%d\":{\"type\":\"%s\",\"fmt\":\"%s\"}", sep, i, tex_names[a->img_type[i]], f), sep = ",";
            }
        sep = "";
        o += sprintf(o, "},\"attr\":{");
        for (int i = 0; i < 32; i++)
            if (a->attr_type[i]) o += sprintf(o, "%s\"%d\":\"%s\"", sep, i, attr_name(a->attr_type[i])), sep = ",";
        sep = "";
        o += sprintf(o, "},\"fout\":{");
        for (int i = 0; i < 8; i++)
            if (a->frag_out_type[i]) o += sprintf(o, "%s\"%d\":\"%s\"", sep, i, attr_name(a->frag_out_type[i])), sep = ",";
        o += sprintf(o, "},\"shared\":%u", a->shared_bytes > nvn2_shared(control) ? a->shared_bytes : nvn2_shared(control));
    }
    if (!a && nvn2_shared(control)) o += sprintf(o, ",\"shared\":%u", nvn2_shared(control));
    if (!strcmp(stage, "cs")) {
        uint32_t l[3] = {1, 1, 1};
        if (a && (a->local[0] | a->local[1] | a->local[2])) l[0] = a->local[0], l[1] = a->local[1], l[2] = a->local[2];
        else memcpy(l, control + 0x748, 12);
        o += sprintf(o, ",\"local\":[%u,%u,%u]", l[0], l[1], l[2]);
        const uint8_t *lay = control + NVN2_C0_LAYOUT;
        if (!memcmp(lay, "DKC1", 4)) {
            o += sprintf(o, ",\"storage_candidates_version\":%u,\"c0\":[%u,%u,%u,%u],\"c0_version\":2,\"c0_roots\":[",
                         GFX_STORAGE_CANDIDATES_VERSION, lay[4], lay[5], lay[6], lay[7]);
            for (int i = 0; i < NVN2_ROOT_TABLES; i++) o += sprintf(o, "%s%u", i ? "," : "", lay[12 + i]);
            o += sprintf(o, "]");
        }
    }
    /* Control byte +0x718 of a fragment shader: early fragment tests (early-Z force); part of the code hash already. */
    if (!strcmp(stage, "ps") && (control[0x718] & 1)) o += sprintf(o, ",\"early_fragment_tests\":true");
    int si = !strcmp(stage, "vs") ? 0 : !strcmp(stage, "cs") ? 2 : 1;   // legacy gfx_shader_set_host answers
    uint32_t subgroup = !strcmp(stage, "gs") ? host_gs_sg : host_sg[si];
    if (!host_f64) o += sprintf(o, ",\"softfloat64\":true");
    if (host_sbuf) o += sprintf(o, ",\"sbuf_align\":%u", host_sbuf);
    if (subgroup != 32) o += sprintf(o, ",\"subgroup\":%u", subgroup);
    if (host_gather) o += sprintf(o, ",\"gather_bias\":%u", host_gather);
    if (host_flat) o += sprintf(o, ",\"flat_gmem\":%d", host_flat);
    o += sprintf(o, ",\"xlate_fix\":%u", xfix_eff);
    o += sprintf(o, "},\"out\":\"");
    json_esc(&o, cache_dir);
    o += sprintf(o, "\"}");
    return (size_t)(o - o0);
}

// ---- public

static const GfxShader *fail(const char *why, const char *hex) {
    plat_inc64(&n_failures);
    rt_log("gfx_shader: translation failed (%s): %s", hex ? hex : "-", why);
    return NULL;
}

// A validated request: the program's code length, its c1 constants and the cache key. 0 (failure counted) when invalid.
typedef struct {
    const char *stage_name;
    int stage;
    uint8_t geometry_input;
    GfxShaderGeometry geometry;
    uint32_t geometry_skip[8];
    int geometry_source_passthrough;
    int lowered_body;                                      // nvn_geometry_control_is_lowered
    uint32_t length, c1_len;
    const uint8_t *c1;
    uint8_t code_hash[32];
    uint8_t key[32];                                       // cache key, or the provisional one with final 0
    uint8_t iid[32];                                       // input id with final 1
    int final, require_c0;                                 // DKC1 inputs must have versioned NVN 2 reflection
} Req;

// NVN 2 programs: control magic 0x98761234, version 0xC9 at +4, control size at +0x30, code size at +0x38.
int gfx_shader_is_nvn2(const uint8_t *control) {
    uint32_t magic, version;
    memcpy(&magic, control, 4), memcpy(&version, control + 4, 4);
    return magic == 0x98761234u && version == 0xC9u;
}

// gfx_shader_nvn2_dump_to writes an NVN 2 stage into a folder as <fnv>.ctl (control) and <fnv>.bin (code).
// An NVN 2 stage's control size and code size (the code with its inline constants, c[0x1]); 0 when implausible.
static int nvn2_sizes(const uint8_t *control, uint32_t *ctl_size, uint32_t *len) {
    uint32_t c1_end;
    memcpy(ctl_size, control + 0x30, 4), memcpy(len, control + 0x38, 4), memcpy(&c1_end, control + 0x44, 4);
    // Some stages leave the control size at 0; 0xd8 is the size of nearly all the others.
    if (!*ctl_size) *ctl_size = 0xd8;
    if (c1_end > *len && c1_end - *len <= 0x10000) *len = c1_end;
    return *ctl_size >= 0x40 && *ctl_size <= 0x10000 && *len && *len <= 0x400000;
}

// FNV-1a over an NVN 2 stage's control and code: the dump file name. 0 when it is not one.
uint64_t gfx_shader_nvn2_hash(const uint8_t *control, const uint8_t *code) {
    uint32_t ctl_size, len;
    if (!control || !code || !gfx_shader_is_nvn2(control) || !nvn2_sizes(control, &ctl_size, &len)) return 0;
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint32_t i = 0; i < ctl_size; i++) h = (h ^ control[i]) * 0x100000001b3ull;
    for (uint32_t i = 0; i < len; i++) h = (h ^ code[i]) * 0x100000001b3ull;
    return h;
}

void gfx_shader_nvn2_dump_to(const char *dir, const uint8_t *control, const uint8_t *code) {
    uint32_t ctl_size, len;
    uint64_t h = dir && *dir ? gfx_shader_nvn2_hash(control, code) : 0;
    if (!h || !nvn2_sizes(control, &ctl_size, &len)) return;
    static PlatMutex dump_lock = PLAT_MUTEX_INIT;
    char path[1024];
    plat_mutex_lock(&dump_lock);
    mkdir_p(dir);   // a missing folder would otherwise make every fopen below fail without a word
    for (int part = 0; part < 2; part++) {
        snprintf(path, sizeof path, "%s/%016llx.%s", dir, (unsigned long long)h, part ? "bin" : "ctl");
        FILE *f = fopen(path, "rb");
        if (f) { fclose(f); continue; }
        if ((f = fopen(path, "wb"))) {
            fwrite(part ? code : control, 1, part ? len : ctl_size, f);
            fclose(f);
        }
    }
    plat_mutex_unlock(&dump_lock);
}

static int req_init(Req *r, int stage, const uint8_t *control, const uint8_t *code, uint64_t code_size, const uint8_t *c1,
                    const GfxShaderAnswers *a) {
    memset(r, 0, sizeof *r);
    if (!control) return fail("missing control section", NULL), 0;
    // An NVN 2 control block read at the NVN 1 offsets (+0x700, +0x704) yields a bogus code length.
    if (gfx_shader_is_nvn2(control)) {
        char native[17];
        snprintf(native, sizeof native, "%016llx", (unsigned long long)gfx_shader_nvn2_hash(control, code));
        return fail("native Ampere stage was not lowered; see the nvn2 rejection for this hash", native), 0;
    }
    if (control && stage == GFX_STAGE_CS && !memcmp(control + NVN2_C0_LAYOUT, "DKC0", 4))
        return fail("old NVN 2 compute control: regenerate shader inputs with root layout version 2", NULL), 0;
    r->require_c0 = stage == GFX_STAGE_CS && !memcmp(control + NVN2_C0_LAYOUT, "DKC1", 4);
    r->stage = stage;
    r->geometry_input = a ? a->geometry_input : 0;
    r->stage_name = stage == GFX_STAGE_VS ? "vs" : stage == GFX_STAGE_FS ? "ps" : stage == GFX_STAGE_CS ? "cs" :
                    stage == GFX_STAGE_GS ? "gs" : NULL;
    if (!r->stage_name) return fail("stage is not translated", NULL), 0;
    uint32_t c1_off;
    memcpy(&c1_off, control + 0x700, 4), memcpy(&r->length, control + 0x704, 4);
    uint64_t skip = stage == GFX_STAGE_CS ? 0 : CODE_SKIP;
    if (!code || r->length > code_size || c1_off > r->length || skip >= r->length) return fail("bad control section", NULL), 0;
    if (stage == GFX_STAGE_GS) {
        uint32_t sph[5];
        if (!host_gs_sg) return fail("geometry host answers are not initialized", NULL), 0;
        if (!r->geometry_input || r->geometry_input >= COUNT(geometry_names) ||
            c1_off < CODE_SKIP + 0x60 || (c1_off - CODE_SKIP) % 8 || (r->length - CODE_SKIP) % 8)
            return fail("incomplete geometry request", NULL), 0;
        memcpy(sph, code + CODE_SKIP, sizeof sph);
        uint32_t output = sph[3] >> 24 & 15;
        if ((sph[0] >> 10 & 15) != 4 || !(sph[2] >> 24) || !(sph[4] & 0xfff) ||
            (output != 1 && output != 6 && output != 7)) return fail("unsupported geometry SPH", NULL), 0;
        int metadata = nvn_geometry_control_get(control, CONTROL_SIZE, code, r->length, r->geometry_skip);
        int pass = (sph[0] & (1u << 24)) != 0;
        if (metadata < 0 || pass != metadata || (pass && r->geometry_input != GFX_GEOMETRY_TRIANGLES))
            return fail("unsupported geometry source metadata", NULL), 0;
        if (!memcmp(control + NVN2_C0_LAYOUT, "DKC1", 4) || !memcmp(control + NVN2_C0_LAYOUT, "DKC0", 4))
            return fail("compute root layout supplied for geometry", NULL), 0;
        r->geometry = (GfxShaderGeometry){.version = GFX_GEOMETRY_VERSION, .input = r->geometry_input,
            .input_vertices = geometry_vertices[r->geometry_input], .output = output == 1 ? 1 : output == 6 ? 2 : 3,
            .invocations = sph[2] >> 24, .max_output_vertices = sph[4] & 0xfff};
        if (pass) {
            r->geometry.version = GFX_GEOMETRY_PASSTHROUGH_VERSION;
            r->geometry.max_output_vertices = 3;
            r->geometry_source_passthrough = 1;
            r->lowered_body = nvn_geometry_control_is_lowered(control);
        }
    } else if (r->geometry_input || !memcmp(control + NVN_GEOMETRY_CONTROL_OFFSET, "TKGSMETA", 8))
        return fail("geometry input supplied for another stage", NULL), 0;
    r->c1_len = r->length - c1_off;
    r->c1 = c1 ? c1 : code + c1_off;
    if (!build_code_hash(stage, control, code, r->length, r->c1, r->c1_len, r->geometry_input, r->code_hash))
        return fail("out of memory", NULL), 0;
    paths();
    UseMask m;
    r->final = use_find(r->code_hash, &m);
    build_key(r->code_hash, a, r->final ? &m : NULL, r->key, r->iid);
    return 1;
}

static int shader_matches_request(const GfxShader *s, const Req *r) {
    return s->stage == r->stage && (r->stage != GFX_STAGE_GS ||
        (!memcmp(&s->geometry, &r->geometry, sizeof r->geometry) &&
         s->geometry_source_passthrough == r->geometry_source_passthrough &&
         !memcmp(s->geometry_skip, r->geometry_skip, sizeof r->geometry_skip)));
}

static GfxShader *request_disk_load(const Req *r, const uint8_t key[32], const uint8_t *iid) {
    for (int packed = pack_n ? 1 : 0; packed >= 0; packed--) {
        GfxShader *s = disk_load_from(key, iid, packed, (r->require_c0 ? REFLECT_REQUIRE_C0 : 0) |
                                                        (r->lowered_body ? REFLECT_LOWERED_BODY : 0));
        if (!s) continue;
        if (shader_matches_request(s, r)) return s;
        shader_free(s);
    }
    return NULL;
}

static const GfxShader *translate(const Req *r, int stage, const uint8_t *control, const uint8_t *code,
                                  const GfxShaderAnswers *a);

static void ans_note(const Req *r, int stage, const uint8_t *control, const uint8_t *code, const GfxShaderAnswers *a,
                     const GfxShader *s);
static const GfxShader *get_async_one(Req r, int stage, const uint8_t *control, const uint8_t *code,
                                      const GfxShaderAnswers *a, int *pending);

static int known_failed(const uint8_t key[32]);
static void note_failed(const uint8_t key[32]);

// The synchronous path remembers failures as the background one does (X.waits).
static const GfxShader *get_one(const Req *r, int stage, const uint8_t *control, const uint8_t *code,
                                const GfxShaderAnswers *a) {
    GfxShader *s = mem_find(r->key);
    if (s) return shader_matches_request(s, r) ? (plat_inc64(&n_hits), s) : fail("cached stage contract differs from request", NULL);
    if (known_failed(r->key)) return NULL;
    paths();
    if (!cache_dir[0]) return fail("no cache directory", NULL);
    if (r->final && (s = request_disk_load(r, r->key, r->iid))) return plat_inc64(&n_hits), mem_insert(s);
    const GfxShader *t = translate(r, stage, control, code, a);
    if (!t) note_failed(r->key);
    return t;
}

static const GfxShader *get_noted(int stage, const uint8_t *control, const uint8_t *code, uint64_t code_size,
                                  const uint8_t *c1, const GfxShaderAnswers *a, int note) {
    Req r;
    if (!req_init(&r, stage, control, code, code_size, c1, a)) return NULL;
    const GfxShader *s = get_one(&r, stage, control, code, a);
    if (s && note) ans_note(&r, stage, control, code, a, s);
    return s;
}
const GfxShader *gfx_shader_get(int stage, const uint8_t *control, const uint8_t *code, uint64_t code_size,
                                const uint8_t *c1, const GfxShaderAnswers *a) {
    return get_noted(stage, control, code, code_size, c1, a, 1);
}
// The prewarm's and the preparation's gets pass note 0: they stay out of shader-answers.bin.

// ---- stored inputs (<cache>/in/<input id>.in, see the top)

typedef struct {
    uint32_t magic, version, stage, length, c1_len, answers_size;
} InHeader;                                                // then GfxShaderAnswers, control, code[length], c1[c1_len]

typedef struct {
    uint8_t *buf;                                          // the file; the pointers below point into it
    int stage;
    uint32_t length, c1_len;
    const uint8_t *control, *code, *c1;
    GfxShaderAnswers a;
} Inputs;

static void in_path(const uint8_t iid[32], char *p) {
    char hex[65];
    key_hex(iid, hex);
    snprintf(p, PATH_CAP + 80, "%s/in/%s.in", cache_dir, hex);
}

// Stores the inputs of a translation unless they are stored already; the file appears whole (written, then renamed).
static void in_store(const uint8_t iid[32], int stage, const uint8_t *control, const uint8_t *code, const Req *r,
                     const GfxShaderAnswers *a) {
    if (nvn_geometry_control_get(control, CONTROL_SIZE, code, r->length, NULL) < 0) return;
    char p[PATH_CAP + 80], tmp[PATH_CAP + 120];
    in_path(iid, p);
    if (pack_find(PACK_IN, iid) || file_exists(p)) return;
    int gs = stage == GFX_STAGE_GS;
    InHeader h = {IN_MAGIC, gs ? GS_IN_VERSION : IN_VERSION, (uint32_t)stage, r->length, r->c1_len,
                  gs ? GS_ANS_PACKED : LEGACY_ANS_SIZE};
    size_t n = sizeof h + h.answers_size + CONTROL_SIZE + r->length + r->c1_len;
    uint8_t *buf = malloc(n), *o = buf;
    if (!buf) return;
    memcpy(o, &h, sizeof h), o += sizeof h;
    if (gs) pack_geometry_answers(o, a);
    else pack_legacy_input_answers(o, a);
    o += h.answers_size;
    memcpy(o, control, CONTROL_SIZE), o += CONTROL_SIZE;
    memcpy(o, code, r->length), o += r->length;
    if (r->c1_len) memcpy(o, r->c1, r->c1_len);
    snprintf(tmp, PATH_CAP + 120, "%s/in", cache_dir);
    mkdir_p(tmp);
    plat_file_write_atomic(p, buf, n);
    free(buf);
}

// Reads stored inputs into *in (free in->buf); 0 when absent or malformed.
static int in_load_from(const uint8_t iid[32], Inputs *in, int packed) {
    char p[PATH_CAP + 80];
    in_path(iid, p);
    uint64_t n = 0;
    uint8_t *f = packed ? pack_read(PACK_IN, iid, &n) : read_file(p, &n);
    InHeader h;
    if (!f || n < sizeof h) return free(f), 0;
    memcpy(&h, f, sizeof h);
    int gs = h.stage == GFX_STAGE_GS;
    if (h.magic != IN_MAGIC || (gs ? h.version != GS_IN_VERSION || h.answers_size != GS_ANS_PACKED
                                 : h.version != IN_VERSION || h.answers_size != LEGACY_ANS_SIZE) ||
        (h.stage != GFX_STAGE_VS && h.stage != GFX_STAGE_FS && h.stage != GFX_STAGE_CS && !gs) ||
        h.c1_len > h.length || n != sizeof h + h.answers_size + CONTROL_SIZE + (uint64_t)h.length + h.c1_len)
        return free(f), 0;
    const uint8_t *o = f + sizeof h;
    if (gs) unpack_geometry_answers(o, &in->a);
    else unpack_legacy_input_answers(o, &in->a);
    if (gs && (!in->a.geometry_input || in->a.geometry_input >= COUNT(geometry_names))) return free(f), 0;
    o += h.answers_size;
    in->control = o, o += CONTROL_SIZE;
    uint32_t length, c1_at;
    memcpy(&length, in->control + 0x704, 4), memcpy(&c1_at, in->control + 0x700, 4);
    if (length != h.length || c1_at > length || length - c1_at != h.c1_len) return free(f), 0;
    in->code = o, o += h.length;
    in->c1 = o;
    in->buf = f, in->stage = (int)h.stage, in->length = h.length, in->c1_len = h.c1_len;
    return 1;
}

static int in_load(const uint8_t iid[32], Inputs *in) {
    return (pack_n && in_load_from(iid, in, 1)) || in_load_from(iid, in, 0);
}

// Portable v2 caches must retain their original inputs, even on the memory fast path.
static int input_matches_cached(const GfxShader *s, const uint8_t iid[32]) {
    int portable = s->stage == GFX_STAGE_GS && s->geometry.version == GFX_GEOMETRY_PASSTHROUGH_VERSION;
    int requires_provenance = portable;
    for (int packed = pack_n ? 1 : 0; packed >= 0; packed--) {
        Inputs in;
        if (!in_load_from(iid, &in, packed)) continue;
        // Classify the source as well as the output: a wrong-stage or v1 cached result must not hide a passthrough input.
        int tagged = !memcmp(in.control + NVN_GEOMETRY_CONTROL_OFFSET, "TKGSMETA", 8) ||
                     (in.stage == GFX_STAGE_GS && in.length >= CODE_SKIP + 4 &&
                      (nvn_geometry_control_u32(in.code + CODE_SKIP) & (1u << 24)));
        requires_provenance |= tagged;
        if (!portable || !tagged) { free(in.buf); continue; }
        Req r;
        int ok = req_init(&r, in.stage, in.control, in.code, in.length, in.c1, &in.a) &&
                 shader_matches_request(s, &r);
        if (ok && !r.lowered_body) {
            // Another cache layer may have allowed relaxed reflection. The input which actually matches this
            // result must grant that policy itself; an ordinary portable source retains its exact interface.
            uint32_t locations = 0;
            for (unsigned i = 0; i < 32; i++)
                if (!(r.geometry_skip[1 + i / 8] >> ((i % 8) * 4) & 15)) locations |= 1u << i;
            ok = s->in_mask == locations && s->out_mask == locations;
        }
        if (ok) {
            UseMask m = use_of(s);
            uint8_t key[32], rebuilt[32];
            build_key(r.code_hash, &in.a, &m, key, rebuilt);
            ok = !memcmp(iid, rebuilt, 32) && !memcmp(s->key, key, 32);
        }
        free(in.buf);
        if (ok) return 1;
    }
    return !requires_provenance;
}

static GfxShader *input_disk_load(const uint8_t key[32], const uint8_t iid[32]) {
    // Parse either interface policy; input_matches_cached must establish the exact source/input/cache identities
    // and require the strict interface unless that same matching input grants the lowered-body policy.
    GfxShader *s = disk_load(key, iid, REFLECT_LOWERED_BODY);
    if (!s || input_matches_cached(s, iid)) return s;
    shader_free(s);
    // A valid JSON/SPIR-V pair in the pack may still describe different inputs.
    s = pack_n ? disk_load_from(key, iid, 0, REFLECT_LOWERED_BODY) : NULL;
    if (s && !input_matches_cached(s, iid)) { shader_free(s); s = NULL; }
    return s;
}

// ---- translation

static const GfxShader *translate_one(const Req *r, int stage, const uint8_t *control, const uint8_t *code,
                                      const GfxShaderAnswers *a) {
    const uint8_t *key = r->key;
    GfxShader *s;
    UseMask m;
    uint8_t mk[32], miid[32];
    // Another thread may have translated it while this one waited in flight_enter.
    const uint8_t *look = r->final ? key : NULL, *look_iid = r->iid;
    if (!r->final && use_find(r->code_hash, &m)) build_key(r->code_hash, a, &m, mk, miid), look = mk, look_iid = miid;
    if (look && (s = mem_find(look)))
        return shader_matches_request(s, r) ? (plat_inc64(&n_hits), s) : fail("cached stage contract differs from request", NULL);
    if (look && (s = request_disk_load(r, look, look_iid))) return plat_inc64(&n_hits), mem_insert(s);

    uint64_t skip = stage == GFX_STAGE_CS ? 0 : CODE_SKIP;
    char hex[65], code_path[PATH_CAP + 80], spv_path[PATH_CAP + 80], res_path[PATH_CAP + 80];
    key_hex(key, hex);
    mkdir_p(cache_dir);
    snprintf(code_path, PATH_CAP + 80, "%s/%s.code", cache_dir, hex);
    snprintf(spv_path, PATH_CAP + 80, "%s/%s.spv", cache_dir, hex);
    snprintf(res_path, PATH_CAP + 80, "%s/%s.res.json", cache_dir, hex);
    if (!write_file(code_path, code + skip, (size_t)(r->length - skip))) return fail("cannot write the code file", hex);
    size_t cap = 2 * PATH_CAP * 6 + 2 * r->c1_len + 8192;
    char *req = malloc(cap), reply[LINE_MAX_LEN];
    if (!req) return plat_file_delete(code_path), fail("out of memory", hex);
    build_request(req, hex, r->stage_name, code_path, r->c1, r->c1_len, a, control);
    int ok = svc_request(req, reply, sizeof reply);
    free(req);
    plat_file_delete(code_path);
    if (!ok) return fail("translator unavailable", hex);
    if (strncmp(reply, "ok ", 3)) {
        plat_file_delete(spv_path), plat_file_delete(res_path);
        return fail(reply, hex);
    }
    plat_file_flush(spv_path), plat_file_flush(res_path);   // the translator wrote them; on disk before use or rename
    if (!(s = request_disk_load(r, key, r->iid))) return fail("translator output missing or malformed", hex);
    if (!r->final) {                                       // name the files after the cache key, remember the mask
        m = use_of(s);
        build_key(r->code_hash, a, &m, mk, miid);
        char mhex[65], to[PATH_CAP + 80];
        key_hex(mk, mhex);
        snprintf(to, PATH_CAP + 80, "%s/%s.spv", cache_dir, mhex);
        int moved = plat_file_replace(spv_path, to);
        snprintf(to, PATH_CAP + 80, "%s/%s.res.json", cache_dir, mhex);
        moved = moved && plat_file_replace(res_path, to);
        if (moved) {
            use_store(r->code_hash, &m);
        } else {                                           // keep the mask for this run only
            plat_file_delete(spv_path), plat_file_delete(res_path), rt_log("gfx_shader: cannot rename %s", hex);
            plat_mutex_lock(&cache_lock);
            use_put(r->code_hash, 1, &m);
            plat_mutex_unlock(&cache_lock);
        }
        memcpy(s->key, mk, 32), memcpy(s->iid, miid, 32);
    }
    in_store(s->iid, stage, control, code, r, a);
    plat_inc64(&n_xlated);
    return mem_insert(s);
}

// The service round trip for a request in neither cache; one translation per cache key at a time.
static const GfxShader *translate(const Req *r, int stage, const uint8_t *control, const uint8_t *code,
                                  const GfxShaderAnswers *a) {
    int away = kern_core_away();
    const uint8_t *id = r->final ? r->key : r->code_hash;
    flight_enter(id);
    const GfxShader *s = translate_one(r, stage, control, code, a);
    flight_leave(id);
    kern_core_back(&away);
    return s;
}

// ---- background translation (gfx_shader_get_async)
// One worker per translator child, each job owning copies of the control section and code. A key stays in X.waits
// while queued or running, and for good when it failed.
typedef struct XJob {
    struct XJob *next;
    int stage;
    Req r;                                                 // r.c1 points into code
    uint8_t *control, *code;
    GfxShaderAnswers a;
} XJob;

typedef struct XWait {
    struct XWait *next;
    uint8_t key[32];
    int failed;
} XWait;

static struct {
    PlatMutex lock;
    PlatCond work, done;                         // done: a job finished (gfx_shader_await)
    XJob *head, *tail;
    XWait *waits[256];
    int started;                                           // 1 running, -1 no thread could be created
} X = {PLAT_MUTEX_INIT, PLAT_COND_INIT, PLAT_COND_INIT};

static volatile int64_t n_async;

// Caller holds X.lock.
static XWait **wait_slot(const uint8_t key[32]) {
    XWait **w = &X.waits[key[0]];
    while (*w && memcmp((*w)->key, key, 32)) w = &(*w)->next;
    return w;
}

static int known_failed(const uint8_t key[32]) {
    plat_mutex_lock(&X.lock);
    XWait *w = *wait_slot(key);
    int f = w && w->failed;
    plat_mutex_unlock(&X.lock);
    return f;
}

static void note_failed(const uint8_t key[32]) {
    plat_mutex_lock(&X.lock);
    XWait **w = wait_slot(key);
    if (!*w && (*w = calloc(1, sizeof **w))) memcpy((*w)->key, key, 32);
    if (*w) (*w)->failed = 1;
    plat_mutex_unlock(&X.lock);
}

static void xlate_worker(void *arg) {
    (void)arg;
    for (;;) {
        plat_mutex_lock(&X.lock);
        while (!X.head) plat_cond_wait(&X.work, &X.lock);
        XJob *j = X.head;
        if (!(X.head = j->next)) X.tail = NULL;
        plat_mutex_unlock(&X.lock);
        const GfxShader *s = translate(&j->r, j->stage, j->control, j->code, &j->a);
        plat_mutex_lock(&X.lock);
        XWait **w = wait_slot(j->r.key), *e = *w;
        if (e && s) *w = e->next, free(e);
        else if (e) e->failed = 1;
        plat_mutex_unlock(&X.lock);
        plat_cond_broadcast(&X.done);
        if (s) plat_inc64(&n_async);
        free(j->control), free(j->code), free(j);
    }
    return;
}
const GfxShader *gfx_shader_get_async(int stage, const uint8_t *control, const uint8_t *code, uint64_t code_size,
                                      const GfxShaderAnswers *a, int *pending) {
    *pending = 0;
    Req r;
    if (!req_init(&r, stage, control, code, code_size, NULL, a)) return NULL;
    const GfxShader *s = get_async_one(r, stage, control, code, a, pending);
    if (s) ans_note(&r, stage, control, code, a, s);
    return s;
}

static const GfxShader *get_async_one(Req r, int stage, const uint8_t *control, const uint8_t *code,
                                      const GfxShaderAnswers *a, int *pending) {
    GfxShader *s = mem_find(r.key);
    if (s) return shader_matches_request(s, &r) ? (plat_inc64(&n_hits), s) : fail("cached stage contract differs from request", NULL);
    plat_mutex_lock_shared(&X.lock);
    XWait *w = *wait_slot(r.key);
    int failed = w && w->failed;
    plat_mutex_unlock_shared(&X.lock);
    if (w) return *pending = !failed, NULL;
    paths();
    if (!cache_dir[0]) return fail("no cache directory", NULL);
    if (r.final && (s = request_disk_load(&r, r.key, r.iid))) return plat_inc64(&n_hits), mem_insert(s);

    XJob *j = calloc(1, sizeof *j);
    XWait *e = calloc(1, sizeof *e);
    uint8_t *ctl = malloc(CONTROL_SIZE), *cd = malloc(r.length);
    plat_mutex_lock(&X.lock);
    if (!X.started) {
        int n = 0;
        for (int i = 0; i < svc_runtime_limit(); i++) {
            if (plat_thread_spawn(xlate_worker, NULL, "xlate", 0)) n++;
        }
        X.started = n ? 1 : -1;
        if (n) rt_log("gfx_shader: graphics shaders are translated by %d background workers", n);
        else rt_log("gfx_shader: cannot start the translation workers; translating synchronously");
    }
    XWait **slot = wait_slot(r.key);
    int raced = *slot != NULL;                             // another thread queued it between the checks
    int queued = !raced && X.started == 1 && j && e && ctl && cd;
    if (queued) {
        memcpy(ctl, control, CONTROL_SIZE), memcpy(cd, code, r.length);
        *j = (XJob){.stage = stage, .r = r, .control = ctl, .code = cd, .a = *a};
        j->r.c1 = cd + (r.length - r.c1_len);
        memcpy(e->key, r.key, 32);
        *slot = e;
        if (X.tail) X.tail->next = j;
        else X.head = j;
        X.tail = j;
        plat_cond_signal(&X.work);
    }
    plat_mutex_unlock(&X.lock);
    if (queued) return *pending = 1, NULL;
    free(j), free(e), free(ctl), free(cd);
    if (raced) return *pending = 1, NULL;
    return translate(&r, stage, control, code, a);         // no worker or out of memory
}

const GfxShader *gfx_shader_await(int stage, const uint8_t *control, const uint8_t *code, uint64_t code_size,
                                  const GfxShaderAnswers *a) {
    int pending;
    const GfxShader *s = gfx_shader_get_async(stage, control, code, code_size, a, &pending);
    Req r;
    if (!pending || !req_init(&r, stage, control, code, code_size, NULL, a)) return s;
    uint64_t p0 = perf_begin();   // the wait counts as the render thread's translation time
    plat_mutex_lock(&X.lock);
    for (XJob *j = X.head, *prev = NULL; j; prev = j, j = j->next)
        if (!memcmp(j->r.key, r.key, 32)) {                // queued: to the front, the worker takes it next
            if (prev) {
                prev->next = j->next;
                if (X.tail == j) X.tail = prev;
                j->next = X.head, X.head = j;
            }
            break;
        }
    XWait *w;
    while ((w = *wait_slot(r.key)) && !w->failed) plat_cond_wait(&X.done, &X.lock);
    plat_mutex_unlock(&X.lock);
    perf_end(PERF_XLATE, p0);
    if (w) return NULL;                                    // failed: logged by the worker
    s = gfx_shader_get_async(stage, control, code, code_size, a, &pending);
    return pending ? NULL : s;
}

uint64_t gfx_shader_async_done(void) { return (uint64_t)n_async; }

void gfx_shader_counts(uint64_t *keys, uint64_t *translated) { *keys = (uint64_t)n_keys, *translated = (uint64_t)n_xlated; }

const uint8_t *gfx_shader_key(const GfxShader *s) { return s->key; }

const uint8_t *gfx_shader_input_id(const GfxShader *s) { return s->iid; }

// By input id: memory, then the disk cache under the current translator, then (xlate 1) a translation from the stored
// inputs. NULL when none has it; a failed translation is counted as a failure.
const GfxShader *gfx_shader_load_input(const uint8_t iid[32], int xlate) {
    paths();
    if (!cache_dir[0]) return NULL;
    uint8_t key[32];
    key_of_input(iid, key);
    GfxShader *s = mem_find(key);
    if (s) return input_matches_cached(s, iid) ? s : NULL;
    if ((s = input_disk_load(key, iid))) return mem_insert(s);
    Inputs in;
    if (!xlate || !in_load(iid, &in)) return NULL;
    const GfxShader *t = get_noted(in.stage, in.control, in.code, in.length, in.c1, &in.a, 0);
    free(in.buf);
    // This lookup promises the requested input, not another cached translation.
    return t && !memcmp(t->iid, iid, 32) ? t : NULL;
}

// From the disk cache only, as a shader the caller owns and releases with gfx_shader_release_private. When the memory
// cache has it, that entry is returned with *owned 0 and must not be released. NULL when not cached.
const GfxShader *gfx_shader_load_private(const uint8_t iid[32], int *owned) {
    *owned = 0;
    paths();
    if (!cache_dir[0]) return NULL;
    uint8_t key[32];
    key_of_input(iid, key);
    GfxShader *s = mem_find(key);
    if (s) return input_matches_cached(s, iid) ? s : NULL;
    if ((s = input_disk_load(key, iid))) *owned = 1;
    return s;
}

void gfx_shader_release_private(const GfxShader *s) {
    if (s) shader_free((GfxShader *)s);
}

// ---- boot prewarm (gfx_shader_prewarm)

typedef struct {
    const uint8_t (*named)[32];                            // loaded into memory, results in out
    size_t n_named;
    const GfxShader **out;
    uint8_t (*other)[32];                                  // other stored inputs: translated when not cached, not loaded
    size_t n_other;
    volatile int64_t next, done;
    volatile int32_t missing, other_cached;
} Warm;

static void warm_worker(void *arg) {
    Warm *w = arg;
    for (;;) {
        size_t i = (size_t)plat_inc64(&w->next) - 1;
        if (i >= w->n_named + w->n_other) return;
        if (i < w->n_named) {
            if (!(w->out[i] = gfx_shader_load_input(w->named[i], 1))) plat_inc32(&w->missing);
        } else {
            const uint8_t *iid = w->other[i - w->n_named];
            uint8_t key[32];
            char hex[65], p[PATH_CAP + 80];
            key_of_input(iid, key);
            key_hex(key, hex);
            snprintf(p, PATH_CAP + 80, "%s/%s.spv", cache_dir, hex);
            if (pack_find(PACK_SPV, key) || file_exists(p)) plat_inc32(&w->other_cached);
            else gfx_shader_load_input(iid, 1);
        }
        plat_inc64(&w->done);
    }
}

static int iid_cmp(const void *x, const void *y) { return memcmp(x, y, 32); }

typedef struct {
    const uint8_t (*named)[32];
    size_t n_named, n, cap;
    uint8_t (*v)[32];
} InList;

static int in_collect(const char *file, void *ctx) {
    InList *l = ctx;
    char name[80];
    uint8_t iid[32];
    if (strlen(file) != 67) return 1;                      // 64 hex digits + ".in"
    memcpy(name, file, 64);
    name[64] = 0;
    if (!hex32(name, iid) || (l->n_named && bsearch(iid, l->named, l->n_named, 32, iid_cmp))) return 1;
    if (l->n == l->cap) {
        size_t c = l->cap ? l->cap * 2 : 1024;
        uint8_t (*t)[32] = realloc(l->v, c * 32);
        if (!t) return 0;
        l->v = t, l->cap = c;
    }
    memcpy(l->v[l->n++], iid, 32);
    return 1;
}

// The stored inputs not in named (sorted): ids parsed from the in/ file names.
static size_t other_inputs(const uint8_t (*named)[32], size_t n_named, uint8_t (**out)[32]) {
    char dir[PATH_CAP + 80];
    snprintf(dir, sizeof dir, "%s/in", cache_dir);
    InList l = {.named = named, .n_named = n_named};
    char name[68];
    for (uint32_t i = 0; i < pack_n; i++) {   // the packed inputs, under the name their loose file would have
        if (pack_ix[i].kind != PACK_IN) continue;
        key_hex(pack_ix[i].name, name);
        memcpy(name + 64, ".in", 4);
        if (!in_collect(name, &l)) break;
    }
    plat_file_list(dir, ".in", in_collect, &l);   // in_store writes no loose file for a packed input: no duplicates
    return *out = l.v, l.n;
}

void gfx_shader_prewarm(const uint8_t (*iids)[32], size_t n, const GfxShader **out, int threads, int procs) {
    paths();
    double t0 = (double)plat_now_ms();
    memset(out, 0, n * sizeof *out);
    if (!cache_dir[0]) return;
    uint64_t xl0 = (uint64_t)n_xlated, fail0 = (uint64_t)n_failures;
    uint8_t (*sorted)[32] = n ? malloc(n * 32) : NULL;
    if (sorted) memcpy(sorted, iids, n * 32), qsort(sorted, n, 32, iid_cmp);
    Warm w = {.named = iids, .n_named = n, .out = out};
    w.n_other = other_inputs(sorted, sorted ? n : 0, &w.other);
    free(sorted);
    size_t total = w.n_named + w.n_other;
    threads = threads < 1 ? 1 : threads > 64 ? 64 : threads;
    if ((size_t)threads > total) threads = (int)(total ? total : 1);
    svc_set_limit(procs);
    PlatThread *t[64];
    int k = 0;
    for (int i = 0; i < threads; i++)
        if ((t[k] = plat_thread_new(warm_worker, &w, "shader prewarm"))) k++;
    if (!k) warm_worker(&w);
    for (int m = 0; m < k; m++)
        while (!plat_thread_join_ns(t[m], 2000000000u))
            rt_log("gfx_shader: prewarm: %lld of %zu shaders, %llu translated, %.0f ms", (long long)w.done, total,
                   (unsigned long long)((uint64_t)n_xlated - xl0), (double)plat_now_ms() - t0);
    svc_set_limit(svc_runtime_limit());
    rt_log("gfx_shader: prewarm: %zu shaders named by pipeline keys (%ld not available), %zu other stored inputs (%ld "
           "cached); %llu translated, %llu failed, with up to %d translators and %d threads, in %.0f ms", w.n_named,
           (long)w.missing, w.n_other, (long)w.other_cached, (unsigned long long)((uint64_t)n_xlated - xl0),
           (unsigned long long)((uint64_t)n_failures - fail0), procs, k, (double)plat_now_ms() - t0);
    free(w.other);
}
// Repairs the module in place (spv_fma_zero.h, spv_gmem_stub.h); spv_original keeps the hash of the replaced words.
static void spv_repair(GfxShader *s) {
    if (!s->spirv || s->spv_original) return;
    uint64_t h = 0xcbf29ce484222325ull;
    const uint8_t *b = (const uint8_t *)s->spirv;
    for (uint64_t i = 0; i < s->words * 4; i++) h = (h ^ b[i]) * 0x100000001b3ull;
    uint64_t words = 0;
    unsigned fixed = 0;
    uint32_t *o = spv_fma_zero_fix(s->spirv, s->words, &words, &fixed);
    if (o) {
        free(s->spirv);
        s->spirv = o, s->words = words, s->spv_original = h;
        rt_log("gfx_shader: fma zero sign repaired in %016llx: %u selects", (unsigned long long)h, fixed);
    }
    o = host_flat != 2 ? NULL : spv_gmem_stub_fix(s->spirv, s->words, &words, &fixed);
    if (o) {
        free(s->spirv);
        s->spirv = o, s->words = words, s->spv_original = h;
        rt_log("gfx_shader: global load helper filled in %016llx: %u", (unsigned long long)h, fixed);
    }
}


// Checked modules are kept as zstd frames and decompressed into a per-thread ring: see README, "Shader cache".
// A returned pointer stays valid on that thread until SPVZ_RING - 1 other modules were asked for there.
#define SPVZ_RING 16
#define SPVZ_KEEP (256u << 10)   // a ring buffer larger than this is given back when its slot is reused
static void spvz_note(void) {
    static int said;
    if (!said) {
        said = 1;
        rt_log("gfx_shader: SPIR-V kept compressed in the shader cache, decompressed per use");
    }
}
static volatile int64_t spvz_decomp, spvz_hits;

// One of 16 sleeping locks by the shader's address: a module is checked once, two in parallel.
static PlatMutex spv_locks[16] = {PLAT_MUTEX_INIT, PLAT_MUTEX_INIT, PLAT_MUTEX_INIT, PLAT_MUTEX_INIT, PLAT_MUTEX_INIT,
                                  PLAT_MUTEX_INIT, PLAT_MUTEX_INIT, PLAT_MUTEX_INIT, PLAT_MUTEX_INIT, PLAT_MUTEX_INIT,
                                  PLAT_MUTEX_INIT, PLAT_MUTEX_INIT, PLAT_MUTEX_INIT, PLAT_MUTEX_INIT, PLAT_MUTEX_INIT,
                                  PLAT_MUTEX_INIT};
static void spv_check(const GfxShader *cs) {
    if (__atomic_load_n(&cs->spv_checked, __ATOMIC_ACQUIRE)) return;
    GfxShader *s = (GfxShader *)cs;
    PlatMutex *lock = &spv_locks[((uintptr_t)cs >> 6) * 0x9E3779B1u >> 28 & 15];
    plat_mutex_lock(lock);
    // spv_checked is set only after the repairs and compression; a thread that sees it set takes the module as final.
    if (!s->spv_checked) {
        spv_repair(s);
        spvz_note();
        if (s->spirv && s->words) {
            size_t raw = (size_t)s->words * 4, cap = ZSTD_compressBound(raw);
            uint8_t *z = malloc(cap);
            size_t n = z ? ZSTD_compress(z, cap, s->spirv, raw, 1) : 0;
            if (z && !ZSTD_isError(n) && n < raw) {
                uint8_t *t = realloc(z, n);
                s->spvz = t ? t : z, s->spvz_size = n;
                free(s->spirv);   // the final module (a repaired copy too): its bytes are in spvz now
                s->spirv = NULL;
            } else free(z);
        }
        __atomic_store_n(&s->spv_checked, 1, __ATOMIC_RELEASE);
    }
    plat_mutex_unlock(lock);
}

uint64_t gfx_shader_spirv_words(const GfxShader *s) {
    spv_check(s);
    return s->words;
}

const uint32_t *gfx_shader_spirv(const GfxShader *s, uint64_t *words) {
    spv_check(s);
    if (words) *words = s->words;
    if (!s->spvz) return s->spirv;
    static __thread struct { const GfxShader *s; uint32_t *buf; uint64_t cap, used; int64_t gen; } ring[SPVZ_RING];
    static __thread uint64_t uses;
    uint64_t now = ++uses;
    int64_t gen = __atomic_load_n(&spvz_free_gen, __ATOMIC_RELAXED);
    unsigned k = 0;
    for (int i = 0; i < SPVZ_RING; i++) {
        if (ring[i].s == s && ring[i].gen == gen) { ring[i].used = now; plat_inc64(&spvz_hits); return ring[i].buf; }
        if (ring[i].used < ring[k].used) k = (unsigned)i;
    }
    uint64_t raw = s->words * 4;
    if (ring[k].cap > SPVZ_KEEP && ring[k].cap > raw) {   // a big module's buffer does not stay with the thread
        free(ring[k].buf);
        ring[k].buf = NULL, ring[k].cap = 0;
    }
    if (ring[k].cap < raw) {
        uint32_t *b = realloc(ring[k].buf, raw);
        if (!b) rt_fatal("gfx_shader: out of memory decompressing a module (%llu bytes)", (unsigned long long)raw);
        ring[k].buf = b, ring[k].cap = raw;
    }
    ring[k].s = NULL;
    size_t n = ZSTD_decompress(ring[k].buf, raw, s->spvz, s->spvz_size);
    if (ZSTD_isError(n) || n != raw) rt_fatal("gfx_shader: a compressed module does not decompress (%s)", ZSTD_isError(n) ? ZSTD_getErrorName(n) : "short");
    ring[k].s = s, ring[k].used = now, ring[k].gen = gen;
    int64_t c = plat_inc64(&spvz_decomp);
    if (!(c & (c - 1)) && c >= 1024)
        rt_log("gfx_shader: SPIR-V: %lld modules decompressed, %lld ring hits", (long long)c, (long long)spvz_hits);
    return ring[k].buf;
}

int gfx_shader_stage(const GfxShader *s) { return s->stage; }
const GfxShaderGeometry *gfx_shader_geometry(const GfxShader *s) { return s->stage == GFX_STAGE_GS ? &s->geometry : NULL; }

int gfx_shader_resources(const GfxShader *s, int kind, const GfxShaderRes **res) {
    if (kind < 0 || kind >= GFX_RES_KINDS) return 0;
    if (res) *res = s->res[kind];
    return s->nres[kind];
}

const Nvn2C0Layout *gfx_shader_c0_layout(const GfxShader *s) { return s && s->has_c0 ? &s->c0 : NULL; }

// The slots of the use mask: answers outside it select the same translation (mask_answers), so a caller may leave
// their texture and image types zero. Parses the reflection: callers keep the result.
uint32_t gfx_shader_slot_mask(const GfxShader *s) { return use_of(s).slots; }

uint32_t gfx_shader_inputs(const GfxShader *s, GfxShaderIo io[32]) {
    if (io) memcpy(io, s->in, sizeof s->in);
    return s->in_mask;
}

uint32_t gfx_shader_outputs(const GfxShader *s, GfxShaderIo io[32]) {
    if (io) memcpy(io, s->out, sizeof s->out);
    return s->out_mask;
}

void gfx_shader_local_size(const GfxShader *s, uint32_t local[3]) { memcpy(local, s->local, sizeof s->local); }

// ---- answers met in play (shader-answers.bin, shader-programs.bin)
// Every (program, answers) a gfx_shader_get* resolves is recorded next to pipeline-keys.bin, so first-start
// preparation translates the variants play needs. Formats: see README, "Shader cache".
// A file with another header is rewritten; a torn tail is cut off at open.
#define ANS_VERSION 2                                     // independent GS input selector; readers retain version 1
#define ANS_PACKED GS_ANS_PACKED
#define ANS_REC (4 + 32 + 32 + ANS_PACKED + 1)
#define ANS_REC_V1 (4 + 32 + 32 + ANSWERS_PACKED + 1)
#define PRG_VERSION 1
#define PRG_HEAD (12 + 32 + CONTROL_SIZE)
typedef struct { char magic[8]; uint32_t version, rec_size; } RecHeader;
typedef struct { uint64_t *v; uint32_t cap, n; } Set64;    // open addressing, 0 empty
typedef struct { uint8_t *p; size_t n, cap; } Bytes;

static struct {
    PlatMutex lock;
    volatile int32_t on;
    char ans_path[PATH_CAP], prg_path[PATH_CAP];
    Set64 pairs, progs;                                    // input ids, code hashes seen (in the files or this run)
    Bytes ans, prg;                                        // pending
    uint64_t pairs_new;                                    // added this run
} RC = {PLAT_MUTEX_INIT};

static int set_add(Set64 *s, uint64_t h) {                 // 1 if new
    h = h ? h : 1;
    if ((s->n + 1) * 2 > s->cap) {
        uint32_t cap = s->cap ? s->cap * 2 : 4096;
        uint64_t *t = calloc(cap, sizeof *t);
        if (!t) return 0;
        for (uint32_t i = 0; i < s->cap; i++)
            if (s->v[i]) {
                uint32_t j = (uint32_t)s->v[i] & (cap - 1);
                while (t[j]) j = (j + 1) & (cap - 1);
                t[j] = s->v[i];
            }
        free(s->v), s->v = t, s->cap = cap;
    }
    uint32_t j = (uint32_t)h & (s->cap - 1);
    for (; s->v[j]; j = (j + 1) & (s->cap - 1))
        if (s->v[j] == h) return 0;
    s->v[j] = h, s->n++;
    return 1;
}

static int bytes_put(Bytes *b, const void *p, size_t n) {
    if (!n) return 1;
    if (b->n + n > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 65536;
        while (cap < b->n + n) cap *= 2;
        uint8_t *t = realloc(b->p, cap);
        if (!t) return 0;
        b->p = t, b->cap = cap;
    }
    memcpy(b->p + b->n, p, n), b->n += n;
    return 1;
}

static uint64_t hash8(const uint8_t *sha) {
    uint64_t v;
    memcpy(&v, sha, 8);
    return v;
}

static void unpack_answers(const uint8_t *p, GfxShaderAnswers *a) {
    memset(a, 0, sizeof *a);
    memcpy(a->tex_type, p, 32), p += 32;
    memcpy(a->img_type, p, 32), p += 32;
    memcpy(a->img_format, p, 32), p += 32;
    memcpy(a->attr_type, p, 32), p += 32;
    memcpy(a->frag_out_type, p, 8), p += 8;
    for (int i = 0; i < 3; i++) a->local[i] = (uint16_t)(p[0] | p[1] << 8), p += 2;
    memcpy(&a->shared_bytes, p, 4), p += 4;
    a->depth_minus_w = *p;
}

// Normalize both answer record versions to v2 in memory. The final byte is seed priority, never spare padding.
// Whole legacy records retain every byte and gain a zero selector. A torn final record is the only discarded data.
static int answers_decode(const uint8_t *file, size_t size, uint8_t **body, size_t *n, int *rewrite) {
    *body = NULL, *n = 0, *rewrite = 0;
    if (!size) { *rewrite = 1; return 1; }
    RecHeader h;
    if (size < sizeof h) return 0;
    memcpy(&h, file, sizeof h);
    int legacy = h.version == 1 && h.rec_size == ANS_REC_V1;
    if (memcmp(h.magic, "OOT_SANS", 8) || (!legacy && (h.version != ANS_VERSION || h.rec_size != ANS_REC))) return 0;
    size_t bytes = size - sizeof h, count = bytes / h.rec_size;
    if (count > SIZE_MAX / ANS_REC) return 0;
    uint8_t *out = calloc(count ? count : 1, ANS_REC);
    if (!out) return 0;
    for (size_t i = 0; i < count; i++) {
        const uint8_t *from = file + sizeof h + i * h.rec_size;
        uint8_t *to = out + i * ANS_REC;
        if (legacy) {
            memcpy(to, from, ANS_REC_V1 - 1);
            to[ANS_REC - 1] = from[ANS_REC_V1 - 1];
        } else memcpy(to, from, ANS_REC);
        uint32_t stage;
        memcpy(&stage, to, 4);
        uint8_t input = to[68 + ANSWERS_PACKED];
        if (stage == GFX_STAGE_GS ? (!input || input >= COUNT(geometry_names)) :
            (input || (stage != GFX_STAGE_VS && stage != GFX_STAGE_FS && stage != GFX_STAGE_CS))) {
            free(out);
            return 0;
        }
    }
    *body = out, *n = count * ANS_REC;
    *rewrite = legacy || bytes % h.rec_size;
    return 1;
}

// Read-only users normalize without modifying disk. Recording upgrades atomically before it can append a v2 record;
// an allocation/write/format failure leaves the old file intact and prevents recording into a mixed-version file.
static int answers_read(const char *path, uint8_t **body, size_t *n, int migrate) {
    *body = NULL, *n = 0;
    uint64_t size = 0;
    uint8_t *file = read_file(path, &size);
    if (!file) return 0;
    int rewrite = 0, ok = size <= SIZE_MAX && answers_decode(file, (size_t)size, body, n, &rewrite);
    free(file);
    if (ok && migrate && rewrite) {
        RecHeader h = {{'O','O','T','_','S','A','N','S'}, ANS_VERSION, ANS_REC};
        if (*n > SIZE_MAX - sizeof h) ok = 0;
        uint8_t *out = ok ? malloc(sizeof h + *n) : NULL;
        if (out) {
            memcpy(out, &h, sizeof h);
            if (*n) memcpy(out + sizeof h, *body, *n);
            ok = plat_file_write_atomic(path, out, sizeof h + *n);
            free(out);
        } else ok = 0;
    }
    if (!ok) {
        free(*body), *body = NULL, *n = 0;
        rt_log("gfx_shader: cannot read or upgrade %s; existing answers were retained", path);
    }
    return ok;
}

// Reads a record file's body (after its header) into *body; 0 when absent, and a file with another header is removed.
static int rec_read(const char *path, const char *magic, uint32_t version, uint32_t rec_size, uint8_t **body, size_t *n) {
    *body = NULL, *n = 0;
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    RecHeader h;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    int ok = len >= (long)sizeof h && fread(&h, sizeof h, 1, f) == 1 && !memcmp(h.magic, magic, 8) &&
             h.version == version && h.rec_size == rec_size;
    if (ok && len > (long)sizeof h) {
        *n = (size_t)len - sizeof h;
        ok = (*body = malloc(*n)) && fread(*body, 1, *n, f) == *n;
    }
    fclose(f);
    if (!ok) {
        free(*body), *body = NULL, *n = 0;
        if (len > 0) rt_log("gfx_shader: %s has another format or version; it is rewritten", path), remove(path);
    }
    return ok;
}

// Walks shader-programs.bin records; returns the byte length of the whole records. fn may be NULL.
static size_t prg_walk(const uint8_t *b, size_t n, void (*fn)(const uint8_t *rec, void *ctx), void *ctx) {
    size_t o = 0;
    while (n - o >= PRG_HEAD) {
        uint32_t len, c1;
        memcpy(&len, b + o + 4, 4), memcpy(&c1, b + o + 8, 4);
        size_t size = PRG_HEAD + (size_t)len + c1;
        if (len > (64u << 20) || c1 > len || size > n - o) break;
        if (fn) fn(b + o, ctx);
        o += size;
    }
    return o;
}

static void prg_seen(const uint8_t *rec, void *ctx) { set_add(ctx, hash8(rec + 12)); }

// The program hashes of shader-programs.bin from the record heads alone (44 bytes each). 1 with *whole = the byte
// length of the whole records; 0 when the file is missing or of another format.
static int prg_scan(const char *path, Set64 *set, size_t *whole, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    RecHeader h;
    fseeko(f, 0, SEEK_END);
    off_t len = ftello(f);
    fseeko(f, 0, SEEK_SET);
    if (len < (off_t)sizeof h || fread(&h, sizeof h, 1, f) != 1 || memcmp(h.magic, "OOT_SPRG", 8) || h.version != PRG_VERSION ||
        h.rec_size != 0) {
        fclose(f);
        return 0;
    }
    *n = (size_t)len - sizeof h;
    size_t o = 0;
    uint8_t head[44];
    while (*n - o >= PRG_HEAD) {
        if (fseeko(f, (off_t)(sizeof h + o), SEEK_SET) || fread(head, sizeof head, 1, f) != 1) break;
        uint32_t l, c1;
        memcpy(&l, head + 4, 4), memcpy(&c1, head + 8, 4);
        size_t size = PRG_HEAD + (size_t)l + c1;
        if (l > (64u << 20) || c1 > l || size > *n - o) break;
        set_add(set, hash8(head + 12));
        o += size;
    }
    fclose(f);
    *whole = o;
    return 1;
}

static void truncate_to(const char *path, size_t size) {
    if (!plat_file_truncate(path, size)) return;
    rt_log("gfx_shader: %s had a torn tail; cut to %zu bytes", path, size);
}

void gfx_shader_record_at(const char *keys_path) {
    if (!keys_path || !keys_path[0] || RC.on) return;
    plat_mutex_lock(&RC.lock);
    const char *slash = strrchr(keys_path, '\\'), *fs = strrchr(keys_path, '/');
    if (fs > slash) slash = fs;
    int dir = slash ? (int)(slash + 1 - keys_path) : 0;
    snprintf(RC.ans_path, sizeof RC.ans_path, "%.*sshader-answers.bin", dir, keys_path);
    snprintf(RC.prg_path, sizeof RC.prg_path, "%.*sshader-programs.bin", dir, keys_path);
    uint8_t *b;
    size_t n;
    if (answers_read(RC.ans_path, &b, &n, 1)) {
        for (size_t o = 0; o + ANS_REC <= n; o += ANS_REC) set_add(&RC.pairs, hash8(b + o + 36));
    } else if (file_exists(RC.ans_path)) {
        plat_mutex_unlock(&RC.lock);
        return;                                          // do not append a different format after a failed migration
    }
    free(b);
    size_t whole;
    if (prg_scan(RC.prg_path, &RC.progs, &whole, &n)) {
        if (whole != n) truncate_to(RC.prg_path, sizeof(RecHeader) + whole);
    } else if (rec_read(RC.prg_path, "OOT_SPRG", PRG_VERSION, 0, &b, &n)) {
        whole = prg_walk(b, n, prg_seen, &RC.progs);
        if (whole != n) truncate_to(RC.prg_path, sizeof(RecHeader) + whole);
        free(b);
    }
    rt_log("gfx_shader: recording answers met in play to %s (%u records, %u programs recorded before)", RC.ans_path,
           RC.pairs.n, RC.progs.n);
    plat_xchg32(&RC.on, 1);
    plat_mutex_unlock(&RC.lock);
}

// A get resolved (r, a) to s: queue the record when its input id is new, and its program when new.
static void ans_note(const Req *r, int stage, const uint8_t *control, const uint8_t *code, const GfxShaderAnswers *a,
                     const GfxShader *s) {
    if (!RC.on) return;
    if (nvn_geometry_control_get(control, CONTROL_SIZE, code, r->length, NULL) < 0) return;
    uint32_t st = (uint32_t)stage;
    plat_mutex_lock(&RC.lock);
    if (set_add(&RC.pairs, hash8(s->iid))) {
        uint8_t rec[ANS_REC] = {0};
        memcpy(rec, &st, 4), memcpy(rec + 4, r->code_hash, 32), memcpy(rec + 36, s->iid, 32);
        pack_geometry_answers(rec + 68, a);
        if (set_add(&RC.progs, hash8(r->code_hash))) {
            uint32_t head[3] = {st, r->length, r->c1_len};
            bytes_put(&RC.prg, head, 12), bytes_put(&RC.prg, r->code_hash, 32);
            bytes_put(&RC.prg, control, CONTROL_SIZE), bytes_put(&RC.prg, code, r->length);
            bytes_put(&RC.prg, r->c1, r->c1_len);
        }
        bytes_put(&RC.ans, rec, sizeof rec);
        RC.pairs_new++;
    }
    plat_mutex_unlock(&RC.lock);
}

static int rec_append(const char *path, const char *magic, uint32_t version, uint32_t rec_size, const Bytes *b) {
    FILE *f = fopen(path, "ab");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    int ok = 1;
    if (ftell(f) == 0) {
        RecHeader h = {{0}, version, rec_size};
        memcpy(h.magic, magic, 8);
        ok = fwrite(&h, sizeof h, 1, f) == 1;
    }
    ok = ok && fwrite(b->p, 1, b->n, f) == b->n;
    return !fclose(f) && ok;
}

static PlatMutex rec_io = PLAT_MUTEX_INIT;   // the record files: appends in order, no read during an append

void gfx_shader_record_flush(int force) {
    static volatile int64_t last;
    int64_t now = (int64_t)plat_now_ms();
    if (!RC.on || (!force && now - last < 2000)) return;
    last = now;
    // The files are written outside RC.lock, which the render thread takes to record a new answer.
    plat_mutex_lock(&rec_io);
    plat_mutex_lock(&RC.lock);
    Bytes ans = RC.ans, prg = RC.prg;
    RC.ans = (Bytes){0}, RC.prg = (Bytes){0};
    uint64_t total = RC.pairs.n, added = RC.pairs_new;
    plat_mutex_unlock(&RC.lock);
    if (prg.n && !rec_append(RC.prg_path, "OOT_SPRG", PRG_VERSION, 0, &prg)) rt_log("gfx_shader: cannot write %s", RC.prg_path);
    if (ans.n && !rec_append(RC.ans_path, "OOT_SANS", ANS_VERSION, ANS_REC, &ans)) rt_log("gfx_shader: cannot write %s", RC.ans_path);
    plat_mutex_unlock(&rec_io);
    if (ans.n) rt_log("gfx_shader: answers: %zu new records appended (%llu new this run, %llu recorded)", ans.n / ANS_REC,
                      (unsigned long long)added, (unsigned long long)total);
    free(ans.p), free(prg.p);
}

// ---- preparation from record files: the answers met in play or the seed from tools/prepare_shaders.py
// A seed record has a zero input id and its pad byte holds the list priority (0 title, 1 warm-up, 2 rest).
typedef struct { const uint8_t *rec; } PrgIdx;
typedef struct { PrgIdx *idx; size_t n; } PrgList;
typedef struct {
    PrgList progs;
    const uint8_t *ans;
    size_t n_ans;
    uint8_t (*iid)[32];                                    // per answers record: the resolved input id, zero on failure
    volatile int64_t next, done, missing, failed;
} Prep;

static int prg_cmp(const void *x, const void *y) { return memcmp(((const PrgIdx *)x)->rec + 12, ((const PrgIdx *)y)->rec + 12, 32); }

static void prg_collect(const uint8_t *rec, void *ctx) {
    PrgList *l = ctx;
    l->idx[l->n++].rec = rec;
}

static void prep_worker(void *arg) {
    Prep *p = arg;
    for (;;) {
        size_t i = (size_t)plat_inc64(&p->next) - 1;
        if (i >= p->n_ans) return;
        const uint8_t *r = p->ans + i * ANS_REC;
        uint8_t key_rec[12 + 32];
        memcpy(key_rec + 12, r + 4, 32);
        PrgIdx k = {key_rec}, *e = bsearch(&k, p->progs.idx, p->progs.n, sizeof k, prg_cmp);
        if (!e) {
            plat_inc64(&p->missing), plat_inc64(&p->done);
            continue;
        }
        uint32_t st, len;
        memcpy(&st, e->rec, 4), memcpy(&len, e->rec + 4, 4);
        const uint8_t *control = e->rec + 44, *code = control + CONTROL_SIZE, *c1 = code + len;
        GfxShaderAnswers a;
        unpack_geometry_answers(r + 68, &a);
        uint32_t answer_stage;
        memcpy(&answer_stage, r, 4);
        if (answer_stage != st) {
            plat_inc64(&p->failed), plat_inc64(&p->done);
            continue;
        }
        if (!memcmp(control + NVN_GEOMETRY_CONTROL_OFFSET, "TKGSMETA", 8)) {
            Req rebuilt;
            if (!req_init(&rebuilt, (int)st, control, code, len, c1, &a) ||
                memcmp(rebuilt.code_hash, e->rec + 12, 32)) {
                plat_inc64(&p->failed), plat_inc64(&p->done);
                continue;
            }
        }
        const GfxShader *s = get_noted((int)st, control, code, len, c1, &a, 0);
        if (!s) plat_inc64(&p->failed);
        else if (p->iid) memcpy(p->iid[i], s->iid, 32);
        plat_inc64(&p->done);
    }
}

// Translates every record of ans_path whose program is in prg_path on `threads` threads with up to `procs` translators.
// *ans_out, *iid_out (optional; free them): the answers records and the input id each resolved to (zero: failed).
// Returns the record count, -1 without the files.
static long long prepare_files(const char *ans_path, const char *prg_path, int threads, int procs, const char *what,
                               uint8_t **ans_out, uint8_t (**iid_out)[32]) {
    paths();
    double t0 = (double)plat_now_ms();
    uint8_t *ab = NULL, *pb = NULL;
    size_t an = 0, pn = 0;
    if (!answers_read(ans_path, &ab, &an, 0) ||
        !rec_read(prg_path, "OOT_SPRG", PRG_VERSION, 0, &pb, &pn)) {
        rt_log("gfx_shader: prepare: no %s (%s)", what, ans_path);
        free(ab), free(pb);
        return -1;
    }
    Prep p = {.progs = {calloc(pn / PRG_HEAD + 1, sizeof(PrgIdx)), 0}, .ans = ab, .n_ans = an / ANS_REC};
    if (iid_out) p.iid = calloc(p.n_ans + 1, 32);
    if (!p.progs.idx || (iid_out && !p.iid)) return free(p.progs.idx), free(p.iid), free(ab), free(pb), -1;
    prg_walk(pb, pn, prg_collect, &p.progs);
    qsort(p.progs.idx, p.progs.n, sizeof *p.progs.idx, prg_cmp);
    uint64_t xl0 = (uint64_t)n_xlated;
    threads = threads < 1 ? 1 : threads > 64 ? 64 : threads;
    svc_set_limit(procs);
    PlatThread *t[64];
    int k = 0;
    for (int i = 0; i < threads; i++)
        if ((t[k] = plat_thread_new(prep_worker, &p, "shader prepare"))) k++;
    if (!k) prep_worker(&p);
    for (int m = 0; m < k; m++)
        while (!plat_thread_join_ns(t[m], 5000000000u))
            rt_log("gfx_shader: prepare: %lld of %zu pairs, %llu translated, %.0f ms", (long long)p.done, p.n_ans,
                   (unsigned long long)((uint64_t)n_xlated - xl0), (double)plat_now_ms() - t0);
    svc_set_limit(svc_runtime_limit());
    rt_log("gfx_shader: prepare: %zu %s of %zu programs: %llu translated, %lld failed, %lld without a program, the rest "
           "cached; %.0f ms", p.n_ans, what, p.progs.n, (unsigned long long)((uint64_t)n_xlated - xl0), (long long)p.failed,
           (long long)p.missing, (double)plat_now_ms() - t0);
    free(p.progs.idx), free(pb);
    if (ans_out) *ans_out = ab;
    else free(ab);
    if (iid_out) *iid_out = p.iid;
    return (long long)p.n_ans;
}

// ---- prepared variants (shader-prepared.bin next to pipeline-keys.bin)
// "OOT_SPRP", version, record size; records: stage, priority, 2 zero bytes, input id.
#define PRP_VERSION 1
#define PRP_REC 36

int gfx_shader_prepare_seed(const char *dir, int threads, int procs) {
    char ans[PATH_CAP + 32], prg[PATH_CAP + 32], out[PATH_CAP + 32];
    snprintf(ans, sizeof ans, "%s/seed-answers.bin", dir);
    snprintf(prg, sizeof prg, "%s/seed-programs.bin", dir);
    snprintf(out, sizeof out, "%s/shader-prepared.bin", dir);
    double t0 = (double)plat_now_ms();
    uint64_t fail0 = (uint64_t)n_failures;
    uint8_t *ab = NULL, (*iid)[32] = NULL;
    long long n = prepare_files(ans, prg, threads, procs, "seed variants", &ab, &iid);
    if (n < 0) return 1;
    uint8_t *recs = calloc((size_t)n + 1, PRP_REC);
    static const uint8_t zero[32];
    size_t u = 0, by[3] = {0}, stages[6] = {0};
    Set64 seen = {0};
    for (long long i = 0; recs && i < n; i++) {           // seed records come in priority order: the first id wins
        if (!memcmp(iid[i], zero, 32) || !set_add(&seen, hash8(iid[i]))) continue;
        uint8_t *r = recs + u * PRP_REC;
        const uint8_t *a = ab + (size_t)i * ANS_REC;
        r[0] = a[0] < 6 ? a[0] : 0, r[1] = a[ANS_REC - 1] > 2 ? 2 : a[ANS_REC - 1];
        memcpy(r + 4, iid[i], 32);
        by[r[1]]++, stages[r[0]]++, u++;
    }
    free(seen.v);
    // The cache key of each seed record, in order (zero: not translated): the planner reads their reflection.
    char keys[PATH_CAP + 32];
    snprintf(keys, sizeof keys, "%s/seed-keys.bin", dir);
    uint8_t *kb = malloc((size_t)n * 32 + 1);
    for (long long i = 0; kb && i < n; i++) {
        if (memcmp(iid[i], zero, 32)) key_of_input(iid[i], kb + (size_t)i * 32);
        else memset(kb + (size_t)i * 32, 0, 32);
    }
    if (kb) plat_file_write_atomic(keys, kb, (size_t)n * 32);
    free(kb);
    RecHeader h = {{'O', 'O', 'T', '_', 'S', 'P', 'R', 'P'}, PRP_VERSION, PRP_REC};
    uint8_t *file = recs ? malloc(sizeof h + u * PRP_REC + 1) : NULL;
    int ok = file != NULL;
    if (ok) {
        memcpy(file, &h, sizeof h), memcpy(file + sizeof h, recs, u * PRP_REC);
        ok = plat_file_write_atomic(out, file, sizeof h + u * PRP_REC);
    }
    char msg[640];
    snprintf(msg, sizeof msg, "gfx_shader: prepared %zu shader variants (vertex %zu, fragment %zu, compute %zu; title %zu, "
             "warm-up %zu, other %zu) from %lld seed records, %llu translations failed, in %.0f ms; %s %s", u,
             stages[GFX_STAGE_VS], stages[GFX_STAGE_FS], stages[GFX_STAGE_CS], by[0], by[1], by[2], n,
             (unsigned long long)((uint64_t)n_failures - fail0), (double)plat_now_ms() - t0,
             ok ? "wrote" : "cannot write", out);
    rt_log("%s", msg);
    printf("%s\n", msg);
    free(file), free(recs), free(iid), free(ab);
    return ok ? 0 : 1;
}

// ---- lowering a whole game's NVN 2 stages ahead of play (oot --lower-shaders)
// Reads seed-nvn2.bin ("OOTNVN2S", version, count; records: stage, control length, code length, 4 zero bytes,
// census key, control, code) and runs each stage through nvn2_lower as nvnProgramSetShaders would, so the code hash
// matches play. Writes seed-programs<sfx>.bin and seed-lowered<sfx>.bin; `shard` of `shards` takes every shards-th stage.
int nvn2_lower(const uint8_t *control, const uint8_t *code, const uint8_t **out_control, const uint8_t **out_code);

int gfx_shader_lower_seed(const char *dir, unsigned shard, unsigned shards) {
    char in[PATH_CAP + 32], prg[PATH_CAP + 48], idx[PATH_CAP + 48], sfx[24] = "";
    if (shards > 1) snprintf(sfx, sizeof sfx, ".%u", shard);
    snprintf(in, sizeof in, "%s/seed-nvn2.bin", dir);
    snprintf(prg, sizeof prg, "%s/seed-programs%s.bin", dir, sfx);
    snprintf(idx, sizeof idx, "%s/seed-lowered%s.bin", dir, sfx);
    uint64_t size = 0;
    uint8_t *b = read_file(in, &size);
    if (!b || size < 16 || memcmp(b, "OOTNVN2S", 8)) {
        fprintf(stderr, "lower-shaders: no %s\n", in);
        free(b);
        return 1;
    }
    FILE *fp = fopen(prg, "wb"), *fi = fopen(idx, "wb");
    if (!fp || !fi) return fprintf(stderr, "lower-shaders: cannot write in %s\n", dir), 1;
    if (shards <= 1) {
        RecHeader h = {{'O', 'O', 'T', '_', 'S', 'P', 'R', 'G'}, PRG_VERSION, 0};
        fwrite(&h, sizeof h, 1, fp);
    }
    uint32_t count, n = 0, lowered = 0, failed = 0, tensor = 0;
    memcpy(&count, b + 12, 4);
    size_t o = 16;
    double t0 = (double)plat_now_ms();
    for (uint32_t i = 0; i < count && size - o >= 24; i++) {
        uint32_t stage, ctl_len, code_len;
        memcpy(&stage, b + o, 4), memcpy(&ctl_len, b + o + 4, 4), memcpy(&code_len, b + o + 8, 4);
        const uint8_t *key = b + o + 16, *ctl = b + o + 24, *code = ctl + ctl_len;
        o += 24 + (size_t)ctl_len + code_len;
        if (o > size) break;
        if (i % (shards ? shards : 1) != shard) continue;
        n++;
        uint8_t rec[8 + 16 + 32] = {0};
        uint32_t flags = 0, length = 0;
        const uint8_t *nctl, *ncode;
        if ((stage == GFX_STAGE_VS || stage == GFX_STAGE_FS || stage == GFX_STAGE_CS) &&
            nvn2_lower(ctl, code, &nctl, &ncode)) {
            uint32_t c1_off;
            memcpy(&c1_off, nctl + 0x700, 4), memcpy(&length, nctl + 0x704, 4);
            uint32_t c1_len = length - c1_off;
            flags = 1 | (nvn2_is_tensor(ncode) ? 2 : 0);
            if (build_code_hash((int)stage, nctl, ncode, length, ncode + c1_off, c1_len, 0, rec + 24)) {
                uint32_t head[3] = {stage, length, c1_len};
                fwrite(head, 12, 1, fp), fwrite(rec + 24, 32, 1, fp), fwrite(nctl, CONTROL_SIZE, 1, fp);
                fwrite(ncode, length, 1, fp), fwrite(ncode + c1_off, c1_len, 1, fp);
                lowered++;
                tensor += flags >> 1 & 1;
            } else flags = 0;
        } else if (stage == GFX_STAGE_VS || stage == GFX_STAGE_FS || stage == GFX_STAGE_CS) failed++;
        memcpy(rec, key, 8), memcpy(rec + 8, &stage, 4), memcpy(rec + 12, &flags, 4), memcpy(rec + 16, &length, 4);
        fwrite(rec, sizeof rec, 1, fi);
    }
    int ok = !ferror(fp) && !ferror(fi);
    ok = !fclose(fp) && ok;
    ok = !fclose(fi) && ok;
    free(b);
    printf("lower-shaders: shard %u of %u: %u stages, %u lowered (%u tensor), %u failed, %.0f ms\n", shard, shards, n,
           lowered, tensor, failed, (double)plat_now_ms() - t0);
    return ok ? 0 : 1;
}

// The variants the boot prebuild warms, most likely first: every input id play recorded (shader-answers.bin, priority
// 0), then the prepared ones (shader-prepared.bin, priority 1 + their list priority). One entry per input id.
size_t gfx_shader_warm_list(GfxWarmEnt **out) {
    *out = NULL;
    if (!RC.on) return 0;
    char prp[PATH_CAP + 32];
    snprintf(prp, sizeof prp, "%.*sshader-prepared.bin", (int)(strlen(RC.ans_path) - strlen("shader-answers.bin")),
             RC.ans_path);
    uint8_t *ab = NULL, *pb = NULL;
    size_t an = 0, pn = 0;
    plat_mutex_lock(&rec_io);                      // no append meanwhile
    answers_read(RC.ans_path, &ab, &an, 0);
    plat_mutex_unlock(&rec_io);
    rec_read(prp, "OOT_SPRP", PRP_VERSION, PRP_REC, &pb, &pn);
    size_t na = an / ANS_REC, np = pn / PRP_REC, n = 0;
    GfxWarmEnt *e = calloc(na + np + 1, sizeof *e);
    Set64 seen = {0};
    for (size_t i = 0; e && i < na; i++) {
        const uint8_t *r = ab + i * ANS_REC;
        if (!set_add(&seen, hash8(r + 36))) continue;
        memcpy(e[n].iid, r + 36, 32), e[n].stage = r[0], e[n].prio = 0, n++;
    }
    for (size_t i = 0; e && i < np; i++) {
        const uint8_t *r = pb + i * PRP_REC;
        if (!set_add(&seen, hash8(r + 4))) continue;
        memcpy(e[n].iid, r + 4, 32), e[n].stage = r[0], e[n].prio = (uint8_t)(1 + r[1]), n++;
    }
    free(seen.v), free(ab), free(pb);
    rt_log("gfx_shader: warm list: %zu variants (%zu met in play, %zu prepared)", n, na, np);
    *out = e;
    return n;
}
