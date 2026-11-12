// The write-back machinery of the VRAM mirror: the write-back rule (wb_*), wb_dirty (wbd_*),
// pend_wait (pw_*), vk_mem_gpu_wrote and wb_record.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_pool_index.h"
#include "vk.h"
#include "platform/platform.h"
#include "vk_memory_int.h"
#include "wb_dirty_spv.h"

// ---- write-back must not land on CPU data
// A small range the CPU also writes stays out of write-back: a stale mirror copy would roll back newer CPU data.
// VRAM mirror and write-back: see README, "VRAM mirror and write-back".
enum { WB_ENTS = 1 << 15, WB_PER_SUB = 1024 };
#define WB_RING (512ull << 20)   // the largest ring; ranges up to WB_RING / 16 are watched
// The host ring the rule's copies go through (pinned host memory, allocated whole at first use, read on the completion thread).
// wb_done moves its tail as submissions complete; an item that would overwrite bytes in flight is left out (counted).
static const uint64_t wb_ring_size = 32ull << 20;
static uint64_t wb_pos, wb_hw, wb_dropped;   // replay thread; wb_pos counts bytes ever placed
static volatile uint64_t wb_done_pos;                      // completion thread: wb_pos at the end of the last done job
typedef struct { uint64_t addr; uint32_t size, sub, seen; volatile uint8_t skip; uint8_t have; uint8_t *last; } WbEnt;   // seen: msub of the last wb_find
typedef struct { PoolRec *r; uint64_t off; WbEnt *e; uint64_t ring; } WbItem;
typedef struct { int n; uint64_t end; WbItem it[]; } WbJob;
static WbEnt wb_ent[WB_ENTS];
static int wb_used;
static WbItem wb_sub[WB_PER_SUB];      // replay: small ranges of this submission
static int wb_n;
static VkBuffer wb_buf;
static VmaAllocation wb_alloc;
static uint8_t *wb_map;
static volatile int64_t wb_flagged, wb_checked;

// Every 6000 submissions, entries not looked up for 60000 submissions are freed (the slot becomes a tombstone), so the copies
// stay under the 1 GiB cap while the game moves through the world.
#define WB_TOMB 1ull
static uint64_t wb_bytes;   // the copies of all entries together stay under 1 GiB
static uint64_t wb_evicted;
static WbEnt *wb_new(WbEnt *e, uint64_t addr, uint32_t size) {   // e: an empty slot or a tombstone
    if (wb_used >= WB_ENTS / 2 || wb_bytes + size > (1024ull << 20)) return NULL;
    uint8_t *last = malloc(size);
    if (!last) return NULL;
    memset(e, 0, sizeof *e);
    e->last = last, e->size = size, e->addr = addr, e->seen = msub, wb_used++, wb_bytes += size;
    return e;
}
static WbEnt *wb_find(uint64_t addr, uint32_t size) {
    unsigned at = (unsigned)(((addr >> 4) ^ size) * 0x9E3779B1u) & (WB_ENTS - 1);
    WbEnt *tomb = NULL;
    for (int k = 0; k < 64; k++, at = (at + 1) & (WB_ENTS - 1)) {
        WbEnt *e = &wb_ent[at];
        if (e->addr == addr && e->size == size) { e->seen = msub; return e; }
        if (e->addr == WB_TOMB) { if (!tomb) tomb = e; continue; }
        if (e->addr) continue;
        return wb_new(tomb ? tomb : e, addr, size);
    }
    // 64 probes without an empty slot: the range is not in the table, so a tombstone passed on the way takes it
    // (tombstones never become empty again).
    return tomb ? wb_new(tomb, addr, size) : NULL;
}
static void wb_sweep(void) {   // replay thread, like wb_find
    // evict_free: the window for entries the rule never stopped (they only remember the last write-back's bytes); stopped
    // entries (the CPU writes them) keep evict's window.
    enum { evict = 60000, evict_free = 6000 };
    static int evict_said;
    if (!evict_said) {
        evict_said = 1;
        rt_log("mem: wb rule: entries unseen for %lld submissions are freed (%lld for entries never stopped)",
               (long long)evict, (long long)evict_free);
    }
    // Every 6000 submissions since the last sweep, not msub % 6000 == 0: the watched ranges come in the same submission each frame.
    static uint32_t swept;
    if ((uint32_t)(msub - swept) < 6000) return;
    swept = msub;
    uint64_t freed = 0, bytes = 0;
    for (int i = 0; i < WB_ENTS; i++) {
        WbEnt *e = &wb_ent[i];
        uint32_t win = (uint32_t)(e->skip ? evict : evict_free);
        if (e->addr <= WB_TOMB || (uint32_t)(msub - e->seen) < win) continue;
        if (e->sub && (uint32_t)(msub - e->sub) < win) continue;   // a job may still read it (win >= 6000 submissions)
        free(e->last);
        bytes += e->size, wb_bytes -= e->size, wb_used--, freed++;
        memset(e, 0, sizeof *e);
        e->addr = WB_TOMB;
    }
    wb_evicted += freed;
    if (freed && (wb_evicted < 2000 || !(msub % 120000)))
        rt_log("mem: wb rule: %llu entries (%.1f MiB) unseen for %lld submissions freed at submission %u; %d live, %.1f MiB",
               (unsigned long long)freed, bytes / 1048576.0, (long long)evict, msub, wb_used, wb_bytes / 1048576.0);
}

// The rule's table for the census (vk_mem_census; counts only, read unordered): entries, copies, stopped ones, by size and age.
void wb_census(void) {
    uint64_t n = 0, bytes = 0, skip = 0, small = 0, big = 0, s6k = 0, s60k = 0, tomb = 0;
    uint32_t now = msub;
    for (int i = 0; i < WB_ENTS; i++) {
        const WbEnt *e = &wb_ent[i];
        if (e->addr == WB_TOMB) { tomb++; continue; }
        if (!e->addr) continue;
        n++, bytes += e->size, skip += e->skip != 0;
        if (e->size <= 4096) small++; else big++;
        if (now - e->seen < 6000) s6k++;
        if (now - e->seen < 60000) s60k++;
    }
    rt_log("mem: wb rule census at submission %u: %llu entries, %.1f MiB of copies (%llu stopped; %llu up to 4 KiB, %llu "
           "larger); looked up in the last 6000 submissions %llu, 60000 %llu; %llu tombstones, %llu evicted so far",
           now, (unsigned long long)n, bytes / 1048576.0, (unsigned long long)skip, (unsigned long long)small,
           (unsigned long long)big, (unsigned long long)s6k, (unsigned long long)s60k, (unsigned long long)tomb,
           (unsigned long long)wb_evicted);
}

static void wb_done(void *arg) {   // completion thread
    WbJob *j = arg;
    for (int i = 0; i < j->n; i++) {
        WbItem *w = &j->it[i];
        WbEnt *e = w->e;
        const uint8_t *guest = wb_map + w->ring, *mirror = guest + e->size;
        plat_inc64(&wb_checked);
        if (e->have && memcmp(guest, e->last, e->size)) {
            e->skip = 1;
            for (uint64_t a = e->addr & ~0xfffull; a < e->addr + e->size; a += 0x1000) wbsp_add(a);
            if (plat_inc64(&wb_flagged) <= 400) rt_log("mem: write-back of %#llx+%#x stops: the CPU wrote it", (unsigned long long)e->addr, e->size);
        }
        memcpy(e->last, mirror, e->size);
        e->have = 1;
    }
    __atomic_store_n(&wb_done_pos, j->end, __ATOMIC_RELEASE);   // after the last read of the job's ring bytes
    free(j);
}

static uint64_t tr_prog;                  // replay thread: the producer of the next vk_mem_gpu_wrote
static uint32_t tr_bind;

void vk_mem_wb_tag(uint64_t prog, uint32_t bind) { tr_prog = prog, tr_bind = bind; }

// ---- wb_dirty: a compute pass (shaders/wb_dirty.comp, one workgroup per chunk) compares the mirror with a VRAM shadow of guest
// memory and writes only the changed 4 KiB chunks; ranges with unaligned ends and pools without a shadow take the plain copy.
#define WBD_RING (1u << 18)   // chunk records (32 bytes each): 8 MiB
typedef struct { uint64_t m, s, g; uint32_t words, pad; } WbdChunk;
static int wbd_state;          // 0 not set up, 1 on, -1 off
static VkPipelineLayout wbd_layout;
static VkPipeline wbd_pipe;
static VkBuffer wbd_ring, wbd_counts;
static VmaAllocation wbd_ring_a, wbd_counts_a;
static WbdChunk *wbd_map;
static volatile uint32_t *wbd_cnt;
static uint64_t wbd_ring_addr, wbd_counts_addr;
static uint32_t wbd_head;
static uint64_t wbd_issued, wbd_issued_words, wbd_last_chunks, wbd_last_words, wbd_shadows, wbd_shadow_bytes;

int wbd_on(void) {
    if (wbd_state) return wbd_state > 0;
    wbd_state = -1;
    VkCtx *c = vk_ctx();
    if (!c->has_buffer_device_address) { rt_log("mem: wb_dirty needs buffer device addresses: off"); return 0; }
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 24};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof wb_dirty_spv, .pCode = wb_dirty_spv};
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(c->device, &li, NULL, &wbd_layout) != VK_SUCCESS || vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS) {
        rt_log("mem: wb_dirty: no pipeline: off");
        return 0;
    }
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                                .module = m, .pName = "main"}, .layout = wbd_layout};
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &wbd_pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)WBD_RING * sizeof(WbdChunk),
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT};
    VkBufferCreateInfo bc = bi;
    bc.size = 64;
    VmaAllocationInfo ia, ic;
    if (r != VK_SUCCESS || vmaCreateBuffer(c->vma, &bi, &ai, &wbd_ring, &wbd_ring_a, &ia) != VK_SUCCESS ||
        vmaCreateBuffer(c->vma, &bc, &ai, &wbd_counts, &wbd_counts_a, &ic) != VK_SUCCESS) {
        rt_log("mem: wb_dirty: no pipeline or ring: off");
        return 0;
    }
    wbd_map = ia.pMappedData, wbd_cnt = ic.pMappedData;
    memset((void *)wbd_cnt, 0, 64);
    wbd_ring_addr = gmem_device(wbd_ring), wbd_counts_addr = gmem_device(wbd_counts);
    wbd_state = 1;
    rt_log("mem: wb_dirty on: write-back copies only the 4 KiB chunks whose bytes changed (VRAM shadows, compare on the GPU)");
    return 1;
}

// The pool's shadow, made with its mirror before any write-back; its content comes from the mirror's first full upload
// (copied mirror -> shadow by wbd_after_upload). A pool without a usable shadow takes the plain copy.
void wbd_mirror_made(PoolRec *r) {
    if (!wbd_on() || r->wbs_state) return;
    r->wbs_state = -1;
    VkCtx *c = vk_ctx();
    uint64_t n = r->import_hi - r->import_lo;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = n,
                             .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateBuffer(c->vma, &bi, &ai, &r->wbs, &r->wbs_a, NULL) != VK_SUCCESS) { r->wbs = VK_NULL_HANDLE; return; }
    r->wbs_addr = gmem_device(r->wbs), r->m_addr = gmem_device(r->mbuf), r->g_addr = gmem_device(r->buf);
    if (!r->wbs_addr || !r->m_addr || !r->g_addr) return;
    wbd_shadows++, wbd_shadow_bytes += n;
    r->wbs_state = 1;
}

// vk_mem_mirror_prologue, after the uploads into the mirrors and their barrier: the same bytes into the shadows.
void wbd_after_upload(VkCommandBuffer cb, const MRange *a, int n) {
    if (wbd_state != 1) return;
    int rec = 0;
    for (int i = 0; i < n; i++) {
        PoolRec *r = a[i].r;
        if (r->wbs_state != 1) continue;
        uint64_t lim = r->import_hi - r->import_lo, lo = a[i].off, hi = a[i].off + a[i].size;
        if (hi > lim) hi = lim;
        if (lo >= hi) continue;
        VkBufferCopy c = {lo, lo, hi - lo};
        vkCmdCopyBuffer(cb, r->mbuf, r->wbs, 1, &c);
        rec = 1;
    }
    if (rec) mbarrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                      VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
}

// Bytes that took the plain copy, by reason (the census in wbd_report): no usable shadow, an end not 4-byte aligned, a
// part over half the ring, the rest of a part after a ring wrap; and with wbd_split the unaligned ends alone.
enum { WBP_NOSHADOW, WBP_UNALIGNED, WBP_HALF, WBP_WRAP, WBP_ENDS, WBP_N };
static uint64_t wbd_plain[WBP_N];
// wbd_split: the 1-3 byte ends of unaligned ranges whose middle went to the compare pass; vk_mem_mirror_end copies them.
MRange *wbd_ends;
int wbd_nends, wbd_cends;
static void wbd_end_add(PoolRec *r, uint64_t off, uint64_t size) {
    if (!size) return;
    if (wbd_nends == wbd_cends) {
        int c = wbd_cends ? 2 * wbd_cends : 256;
        MRange *m = realloc(wbd_ends, sizeof *m * (size_t)c);
        if (!m) return;   // the end stays uncopied only if this fails: the caller then keeps the whole range plain
        wbd_ends = m, wbd_cends = c;
    }
    wbd_ends[wbd_nends++] = (MRange){r, off, size};
    wbd_plain[WBP_ENDS] += size;
}

static void wbd_dispatch(VkCommandBuffer cb, uint32_t first, uint32_t count) {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, wbd_pipe);
    for (uint32_t done = 0; done < count;) {
        uint32_t k = count - done > 65535 ? 65535 : count - done;
        struct { uint64_t list, counts; uint32_t first, n; } pc = {wbd_ring_addr, wbd_counts_addr, first + done, k};
        vkCmdPushConstants(cb, wbd_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
        vkCmdDispatch(cb, k, 1, 1);
        done += k;
    }
}

// vk_mem_mirror_end, before copy_ranges: the ranges it takes become size 0 (copy_ranges skips them). Returns 1 when it
// recorded anything.
int wbd_record(VkCommandBuffer cb, MRange *a, int n) {
    wbd_nends = 0;
    if (!wbd_on() || !n) return 0;
    qsort(a, (size_t)n, sizeof *a, range_cmp);
    uint32_t first = wbd_head, count = 0;
    int any = 0;
    for (int i = 0; i < n;) {   // merged per pool as copy_ranges merges
        PoolRec *r = a[i].r;
        uint64_t lim = r->import_hi - r->import_lo;
        int i0 = i;
        uint64_t lo = a[i].off, hi = a[i].off + a[i].size;
        for (i++; i < n && a[i].r == r && a[i].off <= hi; i++) if (a[i].off + a[i].size > hi) hi = a[i].off + a[i].size;
        if (hi > lim) hi = lim;
        if (lo >= hi) continue;
        if (r->wbs_state != 1) { wbd_plain[WBP_NOSHADOW] += hi - lo; continue; }   // the plain copy
        uint64_t clo = lo, chi = hi;   // what the compare pass takes
        if ((lo & 3) || (hi & 3)) {
            if (((lo + 3) & ~3ull) >= (hi & ~3ull)) { wbd_plain[WBP_UNALIGNED] += hi - lo; continue; }
            clo = (lo + 3) & ~3ull, chi = hi & ~3ull;
        }
        uint64_t nch = 0;
        for (uint64_t at = clo; at < chi; at = (at & ~0xfffull) + 0x1000) nch++;
        if (count + nch > WBD_RING / 2) { wbd_plain[WBP_HALF] += hi - lo; continue; }   // more than half the ring in one part
        // Wrap: what is queued so far goes first.
        if (wbd_head + count + nch > WBD_RING) {
            if (count) wbd_dispatch(cb, first, count), any = 1, wbd_issued += count;
            first = wbd_head = 0, count = 0;
        }
        int nends0 = wbd_nends;
        wbd_end_add(r, lo, clo - lo);
        wbd_end_add(r, chi, hi - chi);
        if (wbd_nends - nends0 != (clo > lo) + (hi > chi)) {   // no room for an end: the whole range stays plain
            wbd_nends = nends0;
            wbd_plain[WBP_UNALIGNED] += hi - lo;
            continue;
        }
        for (uint64_t at = clo; at < chi;) {
            uint64_t end = (at & ~0xfffull) + 0x1000;
            if (end > chi) end = chi;
            wbd_map[first + count++] = (WbdChunk){r->m_addr + at, r->wbs_addr + at, r->g_addr + at, (uint32_t)((end - at) / 4), 0};
            wbd_issued_words += (end - at) / 4;
            at = end;
        }
        for (int j = nends0; j < wbd_nends; j++) {   // the ends' bytes into the shadow too, so it keeps matching guest memory
            VkBufferCopy c = {wbd_ends[j].off, wbd_ends[j].off, wbd_ends[j].size};
            vkCmdCopyBuffer(cb, r->mbuf, r->wbs, 1, &c);
        }
        for (int k = i0; k < i; k++) a[k].size = 0;   // taken (its ends, if any, are in wbd_ends)
    }
    if (count) {
        wbd_head = first + count;
        wbd_issued += count;
        wbd_dispatch(cb, first, count);
        any = 1;
    }
    if (any) vk_exec_draw_new_cb();   // not the replay's pipeline any more: its next dispatch binds its own again
    return any;
}

void wbd_report(double frames) {
    if (wbd_state != 1) return;
    uint64_t ch = wbd_cnt[0], wd = wbd_cnt[1];
    double dch = (double)(ch - wbd_last_chunks), dwd = (double)(wd - wbd_last_words);
    wbd_last_chunks = ch, wbd_last_words = wd;
    rt_log("mem: wb_dirty: per frame %.0f chunks (%.2f MiB) compared, %.0f written (%.2f MiB, %.1f%%); shadows %llu (%.1f MiB)",
           wbd_issued / frames, wbd_issued_words * 4.0 / frames / 1048576.0, dch / frames, dwd * 4.0 / frames / 1048576.0,
           wbd_issued_words ? 100.0 * dwd / (double)wbd_issued_words : 0.0, (unsigned long long)wbd_shadows, wbd_shadow_bytes / 1048576.0);
    rt_log("mem: wb_dirty: plain copy per frame (MiB): no shadow %.2f, unaligned ends %.2f, over half the ring %.2f, after a "
           "ring wrap %.2f; wbd_split ends %.3f", wbd_plain[WBP_NOSHADOW] / frames / 1048576.0, wbd_plain[WBP_UNALIGNED] / frames / 1048576.0,
           wbd_plain[WBP_HALF] / frames / 1048576.0, wbd_plain[WBP_WRAP] / frames / 1048576.0, wbd_plain[WBP_ENDS] / frames / 1048576.0);
    memset(wbd_plain, 0, sizeof wbd_plain);
    wbd_issued = wbd_issued_words = 0;
}

// ---- pend_wait (pw_*): the slots and the counts its report reads.
#define PW_SLOTS 4096
extern volatile int64_t nvn_wait_event_calls;   // nvn_api_cmd.c: nvnCommandBufferWaitEvent calls (a no-op here)
typedef struct { uint64_t plo, phi; int pending; volatile int armed; volatile uint64_t fence; volatile int64_t gf, rf; } PwSlot;   // fence: newest arming
static PwSlot pw[PW_SLOTS];
static volatile int pw_n;
static PlatMutex pw_lock = PLAT_MUTEX_INIT;   // replay and completion threads; never the fault handler
static int *pw_sub, pw_nsub;                  // replay: the slots of the submission being recorded
static int pw_nsizes = -1;
static volatile int64_t pw_faults, pw_wait_ns, pw_timeouts, pw_arms, pw_unarmable, pw_unsubmitted;

static void pw_done(void *arg) {   // completion thread: the submission's write-back copies are in guest memory
    int *j = arg;
    plat_mutex_lock(&pw_lock);
    for (int k = 1; k <= j[0]; k++) {
        PwSlot *t = &pw[j[k]];
        if (t->pending > 0 && --t->pending == 0 && __atomic_load_n(&t->armed, __ATOMIC_ACQUIRE)) {
            plat_mem_trap(t->plo, t->phi - t->plo, 0);
            __atomic_store_n(&t->armed, 0, __ATOMIC_RELEASE);
        }
    }
    plat_mutex_unlock(&pw_lock);
    free(j);
}

void pw_submit(uint64_t fence) {   // vk_mem_mirror_end, after the write-back copies
    if (!pw_nsub) return;
    int *j = malloc(sizeof *j * (size_t)(pw_nsub + 1));
    if (!j) { pw_nsub = 0; return; }
    j[0] = pw_nsub;
    plat_mutex_lock(&pw_lock);
    for (int k = 0; k < pw_nsub; k++) {
        PwSlot *t = &pw[pw_sub[k]];
        j[k + 1] = pw_sub[k];
        t->pending++;
        __atomic_store_n(&t->fence, fence, __ATOMIC_RELEASE);
        if (__atomic_load_n(&t->armed, __ATOMIC_ACQUIRE)) continue;
        uint64_t end = 0;   // only pages committed read-write (a trap over a hole fails part way)
        if (plat_mem_state(t->plo, t->phi, &end) != PLAT_PROT_RW || end < t->phi) { pw_unarmable++; continue; }
        __atomic_store_n(&t->armed, 1, __ATOMIC_RELEASE);   // before the trap: a fault right after it finds it armed
        if (!plat_mem_trap(t->plo, t->phi - t->plo, 1)) { __atomic_store_n(&t->armed, 0, __ATOMIC_RELEASE); pw_unarmable++; continue; }
        pw_arms++;
    }
    plat_mutex_unlock(&pw_lock);
    pw_nsub = 0;
    vk_completion_add(fence, pw_done, j);
}

void pw_report(double frames) {
    if (frames > 0 && pw_nsizes < 0) {   // pend_wait never on: the WaitEvent count alone
        rt_log("mem: pend_wait: off; nvnCommandBufferWaitEvent %.2f per frame", plat_xchg64(&nvn_wait_event_calls, 0) / frames);
        return;
    }
    if (pw_nsizes < 0 || frames <= 0) return;
    int64_t f = plat_xchg64(&pw_faults, 0), w = plat_xchg64(&pw_wait_ns, 0), to = plat_xchg64(&pw_timeouts, 0);
    int64_t a = plat_xchg64(&pw_arms, 0), u = plat_xchg64(&pw_unarmable, 0);
    rt_log("mem: pend_wait: per frame %.2f arms, %.3f CPU accesses that waited (%.3f ms waited), %lld timeouts, %lld "
           "unarmable, %d ranges known, %.2f accesses before their submission went out (given back); nvnCommandBufferWaitEvent "
           "%.2f per frame", a / frames, f / frames, w / 1e6 / frames, (long long)to, (long long)u, pw_n,
           plat_xchg64(&pw_unsubmitted, 0) / frames, plat_xchg64(&nvn_wait_event_calls, 0) / frames);
}

void vk_mem_gpu_wrote(const VkMemSpan *sp, uint64_t size) {
    PoolRec *r = sp->pool;
    if (!r || r->mstate != MS_ON || sp->buffer != r->mbuf || !size) return;
    if ((int64_t)size <= 0) return;
    // The rule also watches ranges up to 64 KiB whose size is a whole number of 256-byte vfx particle records (0x4000, 0x4400, ...):
    // the CPU emits particles into them between the emitter dispatches.
    int watch = (int64_t)size <= 4096 || ((int64_t)size <= 65536 && size % 256 == 0);
    if (watch && size < (WB_RING >> 4)) {
        wb_sweep();
        WbEnt *e = wb_find(r->import_lo + sp->offset, (uint32_t)size);
        if (e && e->skip) return;
        if (e && e->sub != msub && wb_n < WB_PER_SUB) e->sub = msub, wb_sub[wb_n++] = (WbItem){r, sp->offset, e, 0};
    }
    range_add(&wbl, &nwbl, &cwbl, r, sp->offset, size);
}

// Records the ring copies; the caller's barrier has made the mirror's writes readable. 1: copies recorded.
int wb_record(VkCommandBuffer cb, uint64_t fence) {
    if (!wb_n) return 0;
    if (!wb_buf) {
        rt_log("mem: wb ring: %llu MiB of host memory for the write-back rule's copies", (unsigned long long)(wb_ring_size >> 20));
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = wb_ring_size, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &wb_buf, &wb_alloc, &info) != VK_SUCCESS) { wb_n = 0; return 0; }
        wb_map = info.pMappedData;
    }
    WbJob *j = malloc(sizeof *j + sizeof(WbItem) * (size_t)wb_n);
    if (!j) { wb_n = 0; return 0; }
    j->n = 0;
    uint64_t done = __atomic_load_n(&wb_done_pos, __ATOMIC_ACQUIRE);
    for (int i = 0; i < wb_n; i++) {
        WbItem w = wb_sub[i];
        uint64_t need = ((uint64_t)w.e->size * 2 + 15) & ~15ull, at = wb_pos, off = at % wb_ring_size;
        if (off + need > wb_ring_size) at += wb_ring_size - off, off = 0;   // never split: start again at the ring's start
        if (at + need - done > wb_ring_size) {   // would overwrite bytes a pending job still reads
            uint64_t nd = ++wb_dropped;
            if (!(nd & (nd - 1)))
                rt_log("mem: wb ring full (%llu MiB in flight): a watched range left out of this submission (%llu so far)",
                       (unsigned long long)((wb_pos - done) >> 20), (unsigned long long)wb_dropped);
            continue;
        }
        wb_pos = at + need;
        w.ring = off;
        if (wb_pos - done > wb_hw * 2 + (1 << 20)) {
            wb_hw = wb_pos - done;
            rt_log("mem: wb ring: %.1f MiB in flight at most so far (ring %llu MiB)", wb_hw / 1048576.0, (unsigned long long)(wb_ring_size >> 20));
        } else if (wb_pos - done > wb_hw) wb_hw = wb_pos - done;
        VkBufferCopy g = {w.off, w.ring, w.e->size}, m = {w.off, w.ring + w.e->size, w.e->size};
        vkCmdCopyBuffer(cb, w.r->buf, wb_buf, 1, &g);
        vkCmdCopyBuffer(cb, w.r->mbuf, wb_buf, 1, &m);
        j->it[j->n++] = w;
    }
    wb_n = 0;
    j->end = wb_pos;
    vk_completion_add(fence, wb_done, j);
    return 1;
}
