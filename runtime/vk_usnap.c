// Uniform blocks as of the submit call: every uniform range a submission binds is copied into an arena at the start of
// its recording, so late draws do not read another object's matrix. See README, "Uniform snapshot".
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "host.h"
#include "nvn_api.h"
#include "vk.h"
#include <math.h>
#include "vk_exec.h"
#include "platform/platform.h"

#define ALIGN 256u
#define MAX_RANGE (64u << 10)
#define ARENA_MIN (4u << 20)
#define CALL_DEPTH 16

typedef struct Arena {
    struct Arena *next;
    VkBuffer buf;
    VmaAllocation alloc;
    uint8_t *map;
    uint64_t size;
} Arena;

typedef struct { uint64_t lo, hi; } Rng;
typedef struct { uint64_t lo, hi, off; } Reg;
typedef struct { uint64_t lo, hi, fence; } Wrote;

enum { usnap_mode = 1 };
static struct {
    int mode;                   // usnap_mode + 1 once the first submission began; 0 before
    int on;                     // this submission has a snapshot
    uint64_t fence, gen;
    Arena *arena;
    uint8_t *shadow;            // replay_fast bit 256: the snapshot's bytes in ordinary memory, for CPU readers
    uint64_t shadow_cap;
    int shadowed;               // this submission's snapshot is in shadow too
    Rng *col;                   // collected ranges
    int ncol, ccol;
    Reg *reg;                   // merged, sorted by lo, disjoint
    int nreg, creg;
    Rng *drop;                  // byte ranges written during this submission, sorted, disjoint
    int ndrop, cdrop;
    Wrote *wr;                  // GPU writes of submissions not known to be complete
    int nwr, cwr;
    // since the last summary
    uint64_t subs, regions, bytes, copy_ns, hits, drops, inflight, refused, off_subs;
} U;

static Arena *free_arenas;
static PlatMutex free_lock = PLAT_MUTEX_INIT;

static void *grow(void *p, int *cap, size_t elem) {
    int n = *cap ? *cap * 2 : 1024;
    void *q = realloc(p, (size_t)n * elem);
    if (!q) rt_fatal("vk_usnap: out of memory");
    *cap = n;
    return q;
}

static Arena *arena_get(uint64_t need) {
    plat_mutex_lock(&free_lock);
    Arena **pp = &free_arenas, *a = NULL;
    for (; *pp; pp = &(*pp)->next)
        if ((*pp)->size >= need) {
            a = *pp, *pp = a->next;
            break;
        }
    plat_mutex_unlock(&free_lock);
    if (a) return a;
    a = calloc(1, sizeof *a);
    if (!a) rt_fatal("vk_usnap: out of memory");
    a->size = need < ARENA_MIN ? ARENA_MIN : (need + (1u << 20) - 1) & ~(uint64_t)((1u << 20) - 1);
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = a->size,
                             .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT};
    // Device-local through the BAR when the device has it (the GPU reads constants on every draw), else host memory.
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                           VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    VkResult res = vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &a->buf, &a->alloc, &info);
    if (res != VK_SUCCESS) {
        rt_log("vk_usnap: arena of %llu bytes failed (%d): this submission binds live memory",
               (unsigned long long)a->size, (int)res);
        free(a);
        return NULL;
    }
    a->map = info.pMappedData;
    rt_log("vk_usnap: new arena of %.1f MiB", a->size / 1048576.0);
    return a;
}

static void arena_done(void *arg) {   // completion thread: the GPU has read it
    Arena *a = arg;
    plat_mutex_lock(&free_lock);
    a->next = free_arenas, free_arenas = a;
    plat_mutex_unlock(&free_lock);
}

static void collect_binding(const p_bind_buf *b) {
    if (!b->addr || !b->size) return;
    if (U.ncol == U.ccol) U.col = grow(U.col, &U.ccol, sizeof *U.col);
    uint64_t n = b->size < MAX_RANGE ? b->size : MAX_RANGE;
    U.col[U.ncol++] = (Rng){b->addr & ~(uint64_t)(ALIGN - 1), b->addr + n};
}

static void collect(const op_hdr *h, int depth) {
    for (;;) {
        if (h->size < sizeof *h || (h->size & 7) || h->op >= OP_COUNT_)
            rt_fatal("vk_usnap: corrupt command record at %p: op %u size %u", (const void *)h, h->op, h->size);
        const void *a = nvn_op_payload(h);
        if (h->op == OP_END) return;
        if (h->op == OP_JUMP) {
            h = (const op_hdr *)(uintptr_t)((const p_jump *)a)->target;
            continue;
        }
        if (h->op == OP_CALL && depth < CALL_DEPTH) collect(nvn_op_first(((const p_jump *)a)->target), depth + 1);
        if (h->op == OP_BIND_UBO) collect_binding(a);
        h = nvn_op_next(h);
    }
}

static int by_lo(const void *a, const void *b) {
    uint64_t x = ((const Rng *)a)->lo, y = ((const Rng *)b)->lo;
    return x < y ? -1 : x > y;
}
static int wr_by_lo(const void *a, const void *b) {
    uint64_t x = ((const Wrote *)a)->lo, y = ((const Wrote *)b)->lo;
    return x < y ? -1 : x > y;
}

// Sorts the collected ranges by lo. A frame can bind about 10,000 ranges and qsort is slow on that many, so they go
// through an LSD radix sort on (lo - lowest) / ALIGN (three passes of 11 bits); qsort is left for a span over 2 TiB.
#define RADIX_BITS 11
static void sort_col(void) {
    static Rng *tmp;
    static int ctmp;
    int n = U.ncol;
    if (n < 2) return;
    uint64_t lo = UINT64_MAX, hi = 0;
    for (int i = 0; i < n; i++) {
        if (U.col[i].lo < lo) lo = U.col[i].lo;
        if (U.col[i].lo > hi) hi = U.col[i].lo;
    }
    if ((hi - lo) / ALIGN >> (3 * RADIX_BITS)) {
        qsort(U.col, (size_t)n, sizeof *U.col, by_lo);
        return;
    }
    if (ctmp < n) {
        free(tmp);
        ctmp = U.ccol;
        if (!(tmp = malloc((size_t)ctmp * sizeof *tmp))) rt_fatal("vk_usnap: out of memory");
    }
    Rng *src = U.col, *dst = tmp;
    for (int pass = 0; pass < 3; pass++) {
        int shift = pass * RADIX_BITS, count[1 << RADIX_BITS] = {0};
        for (int i = 0; i < n; i++) count[(src[i].lo - lo) / ALIGN >> shift & ((1 << RADIX_BITS) - 1)]++;
        for (int k = 0, sum = 0; k < 1 << RADIX_BITS; k++) {
            int c = count[k];
            count[k] = sum, sum += c;
        }
        for (int i = 0; i < n; i++) dst[count[(src[i].lo - lo) / ALIGN >> shift & ((1 << RADIX_BITS) - 1)]++] = src[i];
        Rng *t = src;
        src = dst, dst = t;
    }
    if (src != U.col) memcpy(U.col, src, (size_t)n * sizeof *src);   // three passes end in tmp
}

// Keeps the GPU writes of submissions that are not done, merged into disjoint ranges sorted by lo (the terrain rewrites
// one block before every draw: hundreds of equal entries), each with the newest fence among its parts.
static void wr_prune(void) {
    uint64_t done = vk_sync_fence_done();
    int n = 0;
    for (int i = 0; i < U.nwr; i++)
        if (U.wr[i].fence > done) U.wr[n++] = U.wr[i];
    qsort(U.wr, (size_t)n, sizeof *U.wr, wr_by_lo);
    int m = 0;
    for (int i = 0; i < n; i++) {
        if (m && U.wr[i].lo <= U.wr[m - 1].hi) {
            if (U.wr[i].hi > U.wr[m - 1].hi) U.wr[m - 1].hi = U.wr[i].hi;
            if (U.wr[i].fence > U.wr[m - 1].fence) U.wr[m - 1].fence = U.wr[i].fence;
        } else {
            U.wr[m++] = U.wr[i];
        }
    }
    U.nwr = m;
}
static int wr_hits(uint64_t lo, uint64_t hi) {   // wr_prune ran: disjoint, sorted
    int a = 0, b = U.nwr;
    while (a < b) {
        int k = (a + b) / 2;
        if (U.wr[k].hi <= lo) a = k + 1;
        else b = k;
    }
    return a < U.nwr && U.wr[a].lo < hi;
}

static void summary(void) {
    if (U.subs + U.off_subs < 3600) return;
    double s = U.subs ? (double)U.subs : 1.0;
    rt_log("vk_usnap: %llu submissions with a snapshot, %llu without; per submission %.0f ranges, %.1f KiB copied in "
           "%.3f ms, %.0f uniform bindings from the copy, %.1f dropped by GPU writes, %.1f left out (GPU writes in "
           "flight), %.1f refused (not an imported pool)",
           (unsigned long long)U.subs, (unsigned long long)U.off_subs, U.regions / s, U.bytes / s / 1024.0,
           U.copy_ns / s / 1e6, U.hits / s, U.drops / s, U.inflight / s, U.refused / s);
    U.subs = U.off_subs = U.regions = U.bytes = U.copy_ns = U.hits = U.drops = U.inflight = U.refused = 0;
}

// Queue coroutines (vk_exec.c): submissions and bindings of queues that wait, added to the snapshot of the
// submission they will run in. Called from the hook only (inside vk_usnap_begin).
void (*vk_usnap_extra)(void);
void vk_usnap_add_state(const VkExec *ex) {
    for (int stage = 0; stage < 8; stage++)
        for (int slot = 0; slot < 18; slot++) collect_binding(&ex->ubo[stage][slot]);
}
void vk_usnap_add_handles(const uint64_t *handles, int n) {
    for (int i = 0; i < n; i++)
        if (handles[i]) collect(nvn_op_first(handles[i]), 0);
}
void vk_usnap_add_stream(const void *op) { if (op) collect(op, 0); }
void vk_usnap_add_returns(const void *const *ret, int depth) {
    for (int i = 0; i < depth; i++) if (ret[i]) collect(ret[i], 0);
}

void vk_usnap_begin(const VkExec *ex, const uint64_t *handles, int n, uint64_t fence) {
    if (!U.mode) U.mode = usnap_mode + 1;   // stored +1: 0 is "not read yet"
    U.on = 0, U.fence = fence, U.gen++, U.ndrop = 0, U.shadowed = 0;
    summary();
    uint64_t t0 = kern_now_ns();
    U.ncol = 0;
    // Replay bindings persist across submissions. A stream can draw with an earlier uniform binding without
    // another OP_BIND_UBO; that range needs this submission's snapshot just as a newly bound range does.
    for (int stage = 0; stage < 8; stage++)
        for (int slot = 0; slot < 18; slot++) collect_binding(&ex->ubo[stage][slot]);
    for (int i = 0; i < n; i++)
        if (handles[i]) collect(nvn_op_first(handles[i]), 0);
    if (vk_usnap_extra) vk_usnap_extra();
    sort_col();
    wr_prune();
    // Reject pending GPU writes per binding before merging allocations. Otherwise a write to one uniform would
    // also exclude every untouched neighbor that happens to be contiguous with it.
    int kept = 0;
    for (int i = 0; i < U.ncol; i++) {
        if (wr_hits(U.col[i].lo, U.col[i].hi)) U.inflight++;
        else U.col[kept++] = U.col[i];
    }
    U.ncol = kept;
    // Merge the safe ranges, keeping only what lies wholly in one imported pool.
    U.nreg = 0;
    uint64_t total = 0;
    for (int i = 0; i < U.ncol;) {
        uint64_t lo = U.col[i].lo, hi = U.col[i].hi;
        for (i++; i < U.ncol && U.col[i].lo <= hi; i++)
            if (U.col[i].hi > hi) hi = U.col[i].hi;
        VkMemSpan sp;
        if (!vk_mem_resolve(lo, hi - lo, &sp) || sp.alloc || !sp.pool || sp.avail < hi - lo ||
            (sp.cls != VK_MCLS_HOST_CACHED && sp.cls != VK_MCLS_HOST_UNC)) {
            U.refused++;
            continue;
        }
        if (U.nreg == U.creg) U.reg = grow(U.reg, &U.creg, sizeof *U.reg);
        U.reg[U.nreg++] = (Reg){lo, hi, total};
        total += (hi - lo + ALIGN - 1) & ~(uint64_t)(ALIGN - 1);
    }
    if (!U.nreg) return;
    if (!(U.arena = arena_get(total))) return;
    // The arena is device-local memory mapped through the BAR where the device has it (write-combined, not cached), and
    // reading it back is slow. So the ranges are copied into ordinary memory first and go to the arena from there in one
    // streaming copy; CPU readers take the ordinary copy.
    U.shadowed = 0;
    if (U.shadow_cap < total) {
        free(U.shadow);
        U.shadow_cap = (total + (1u << 20) - 1) & ~(uint64_t)((1u << 20) - 1);
        if (!(U.shadow = malloc(U.shadow_cap))) rt_fatal("vk_usnap: out of memory");
    }
    for (int i = 0; i < U.nreg; i++)
        memcpy(U.shadow + U.reg[i].off, (const void *)(uintptr_t)U.reg[i].lo, U.reg[i].hi - U.reg[i].lo);
    // the gaps between regions (each rounded up to ALIGN) are never bound: only the regions go up
    for (int i = 0; i < U.nreg; i++)
        memcpy(U.arena->map + U.reg[i].off, U.shadow + U.reg[i].off, U.reg[i].hi - U.reg[i].lo);
    U.shadowed = 1;
    vmaFlushAllocation(vk_ctx()->vma, U.arena->alloc, 0, total);   // a no-op on coherent memory
    U.on = 1;
    U.subs++, U.regions += (uint64_t)U.nreg, U.bytes += total, U.copy_ns += kern_now_ns() - t0;
}

void vk_usnap_end(uint64_t fence) {
    if (U.arena) vk_completion_add(fence, arena_done, U.arena), U.arena = NULL;
    U.on = 0, U.gen++;
}

// First region whose end is above a.
static int first_above(uint64_t a) {
    int lo = 0, hi = U.nreg;
    while (lo < hi) {
        int m = (lo + hi) / 2;
        if (U.reg[m].hi <= a) lo = m + 1;
        else hi = m;
    }
    return lo;
}

int vk_usnap_find(uint64_t addr, uint64_t n, uint64_t align, VkDescriptorBufferInfo *out) {
    if (!U.on || !n) return 0;
    int i = first_above(addr);
    if (i >= U.nreg) return 0;
    const Reg *r = &U.reg[i];
    if (addr < r->lo || addr + n > r->hi) return 0;
    int lo = 0, hi = U.ndrop;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (U.drop[mid].hi <= addr) lo = mid + 1;
        else hi = mid;
    }
    if (lo < U.ndrop && U.drop[lo].lo < addr + n) return 0;
    uint64_t off = r->off + (addr - r->lo);
    if (off % align) return 0;
    *out = (VkDescriptorBufferInfo){U.arena->buf, off, n};
    U.hits++;
    return 1;
}

// The snapshot's bytes of [addr, addr + n) for a CPU reader (mesh handle programs, vk_exec_mesh.h): 1 when this
// submission's snapshot holds the whole range, else 0 (the caller reads live memory).
int vk_usnap_read(uint64_t addr, uint64_t n, void *out) {
    VkDescriptorBufferInfo bi;
    if (!vk_usnap_find(addr, n, 1, &bi) || !U.arena || !U.arena->map) return 0;
    memcpy(out, (U.shadowed ? U.shadow : U.arena->map) + bi.offset, (size_t)n);
    return 1;
}

void vk_usnap_drop(uint64_t addr, uint64_t n) {
    if (!U.mode || !n) return;
    if (U.nwr == U.cwr) U.wr = grow(U.wr, &U.cwr, sizeof *U.wr);
    U.wr[U.nwr++] = (Wrote){addr, addr + n, U.fence};
    if (!U.on) return;
    int region = first_above(addr);
    if (region >= U.nreg || U.reg[region].lo >= addr + n) return;
    // Allocation ranges merge adjacent uniforms. Invalidating that entire allocation would also make an untouched
    // neighbor read live CPU memory, so retain the exact write intervals independently of the allocation layout.
    uint64_t lo = addr, hi = addr + n;
    int first = 0;
    while (first < U.ndrop && U.drop[first].hi < lo) first++;
    if (first < U.ndrop && U.drop[first].lo <= lo && U.drop[first].hi >= hi) return;
    int last = first;
    while (last < U.ndrop && U.drop[last].lo <= hi) {
        if (U.drop[last].lo < lo) lo = U.drop[last].lo;
        if (U.drop[last].hi > hi) hi = U.drop[last].hi;
        last++;
    }
    if (first == last && U.ndrop == U.cdrop) U.drop = grow(U.drop, &U.cdrop, sizeof *U.drop);
    memmove(U.drop + first + 1, U.drop + last, (size_t)(U.ndrop - last) * sizeof *U.drop);
    U.ndrop += 1 - (last - first);
    U.drop[first] = (Rng){lo, hi};
    U.drops++, U.gen++;
}

uint64_t vk_usnap_gen(void) { return U.gen; }

// The arenas back on the free list (vk_exec_draw_shutdown: the device is idle, every completion ran).
void vk_usnap_shutdown(void) {
    plat_mutex_lock(&free_lock);
    for (Arena *a = free_arenas, *next; a; a = next) next = a->next, vmaDestroyBuffer(vk_ctx()->vma, a->buf, a->alloc), free(a);
    free_arenas = NULL;
    plat_mutex_unlock(&free_lock);
}
