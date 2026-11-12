// Upload batches (command buffer, staging buffers, upload timings) and deferred destruction of the graphics layer.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "vk.h"
#include "platform/platform.h"
#include "vk_memory_int.h"

// ---------------------------------------------------------------- upload batches and deferred destruction

// The perf line's upload figures (OOT_TRACE=perf, the full overlay): GPU time, batch count, bytes staged and wait time of upload
// batches, which go to the queue outside vk_sync.c's timed submissions (perf_gpu_line). Each batch owns one timestamp pair;
// batches past UP_PAIRS go untimed.
#define UP_PAIRS 256
static VkQueryPool up_pool;
static int up_state, up_pairs;   // up_state: 0 not set up, 1 ready, -1 no timestamps
static float up_period;
static volatile int64_t up_gpu_ns, up_timed, up_batches, up_wait_ns, up_bytes;

// A timestamp pool of n pairs for the perf line's side timings (upload batches, gfx_present.c's blit), or NULL when
// the graphics queue has no 64-bit timestamps. *period: ns per tick.
VkQueryPool vk_ts_pool(uint32_t pairs, float *period) {
    VkCtx *c = vk_ctx();
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c->phys, &nq, NULL);
    VkQueueFamilyProperties *qp = calloc(nq ? nq : 1, sizeof *qp);
    if (!qp) return VK_NULL_HANDLE;
    vkGetPhysicalDeviceQueueFamilyProperties(c->phys, &nq, qp);
    uint32_t bits = c->gfx_family < nq ? qp[c->gfx_family].timestampValidBits : 0;
    free(qp);
    VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                .queryCount = 2 * pairs};
    VkQueryPool p = VK_NULL_HANDLE;
    // A counter narrower than 64 bits goes untimed.
    if (bits < 64 || c->props.limits.timestampPeriod <= 0 || vkCreateQueryPool(c->device, &qi, NULL, &p) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    *period = c->props.limits.timestampPeriod;
    return p;
}

static int up_ts_ready(void) {
    if (!up_state) up_state = (up_pool = vk_ts_pool(UP_PAIRS, &up_period)) ? 1 : -1;
    return up_state > 0;
}

void vk_upload_perf(uint64_t *gpu_ns, uint64_t *timed, uint64_t *batches, uint64_t *wait_ns, uint64_t *bytes) {
    *gpu_ns = (uint64_t)plat_xchg64(&up_gpu_ns, 0), *timed = (uint64_t)plat_xchg64(&up_timed, 0);
    *batches = (uint64_t)plat_xchg64(&up_batches, 0), *wait_ns = (uint64_t)plat_xchg64(&up_wait_ns, 0);
    *bytes = (uint64_t)plat_xchg64(&up_bytes, 0);
}


PlatRMutex cmd_lock;
static PlatOnce cmd_once = PLAT_ONCE_INIT;
VkCommandPool cmd_pool;
static Batch open_b;
static int open_active;
static Batch *inflight;
static int ninflight, cap_inflight;
Batch *spare;                  // completed batches whose command buffer and fence are reused
int nspare, cap_spare;
uint64_t next_seq, done_seq;   // batches handed out / completed (all batches up to done_seq)
Dead *dead;
int ndead, cap_dead;
uint64_t tl_protected;         // newest submission being recorded or submitted; guards deferred destruction

static void cmd_init(void *p) {
    (void)p;
    return;
}

void grow(void **arr, int *cap, size_t elem) {
    *cap = *cap ? *cap * 2 : 16;
    void *n = realloc(*arr, (size_t)*cap * elem);
    if (!n) rt_fatal("vk: out of memory (upload bookkeeping)");
    *arr = n;
}

static void free_staging(VkCtx *c, Batch *b) {
    for (int i = 0; i < b->nstg; i++) vmaDestroyBuffer(c->vma, b->stg[i].buf, b->stg[i].alloc);
    b->nstg = 0;
}

// Batches complete in submission order (one queue): retire from the front while their fences are signalled.
void retire(VkCtx *c, int wait_all) {
    int k = 0;
    for (; k < ninflight; k++) {
        Batch *b = &inflight[k];
        if (wait_all) vkWaitForFences(c->device, 1, &b->fence, VK_TRUE, UINT64_MAX);
        else if (vkGetFenceStatus(c->device, b->fence) != VK_SUCCESS) break;
        uint64_t v[2];
        if (b->timed && vkGetQueryPoolResults(c->device, up_pool, 2 * (uint32_t)b->ts, 2, sizeof v, v, sizeof v[0],
                                              VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && v[1] >= v[0]) {
            plat_add64(&up_gpu_ns, (int64_t)((double)(v[1] - v[0]) * up_period));
            plat_inc64(&up_timed);
        }
        free_staging(c, b);
        done_seq = b->seq;
        if (nspare == cap_spare) grow((void **)&spare, &cap_spare, sizeof *spare);
        spare[nspare++] = *b;
    }
    memmove(inflight, inflight + k, (size_t)(ninflight - k) * sizeof *inflight);
    ninflight -= k;
}

void collect_locked(VkCtx *c) {
    retire(c, 0);
    uint64_t tl = 0;
    if (c->timeline) vkGetSemaphoreCounterValue(c->device, c->timeline, &tl);
    int k = 0;
    for (int i = 0; i < ndead; i++) {
        Dead *d = &dead[i];
        if (d->seq > done_seq || d->tl > tl) { dead[k++] = *d; continue; }
        switch (d->kind) {
        case DEAD_IMAGE: vkDestroyImage(c->device, (VkImage)(uintptr_t)d->handle, NULL); break;
        case DEAD_IMAGE_VMA: vmaDestroyImage(c->vma, (VkImage)(uintptr_t)d->handle, d->alloc); break;
        case DEAD_VIEW: vkDestroyImageView(c->device, (VkImageView)(uintptr_t)d->handle, NULL); break;
        case DEAD_BUFFER_VMA: vmaDestroyBuffer(c->vma, (VkBuffer)(uintptr_t)d->handle, d->alloc); break;
        }
    }
    ndead = k;
}

VkCommandBuffer vk_upload_begin(void) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    VkCtx *c = vk_ctx();
    if (!c) rt_fatal("vk: upload before vk_init");
    if (open_active) return open_b.cb;
    if (!cmd_pool) {
        VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                      .queueFamilyIndex = c->gfx_family};
        if (vkCreateCommandPool(c->device, &pi, NULL, &cmd_pool) != VK_SUCCESS) rt_fatal("vk: upload command pool");
    }
    retire(c, 0);
    memset(&open_b, 0, sizeof open_b);
    if (nspare) {
        Batch s = spare[--nspare];
        open_b.cb = s.cb, open_b.fence = s.fence, open_b.stg = s.stg, open_b.cap = s.cap, open_b.ts = s.ts;
        vkResetCommandBuffer(open_b.cb, 0);
        vkResetFences(c->device, 1, &open_b.fence);
    } else {
        open_b.ts = up_pairs < UP_PAIRS ? up_pairs++ : -1;
        VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cmd_pool,
                                          .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
        VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkAllocateCommandBuffers(c->device, &ai, &open_b.cb) != VK_SUCCESS ||
            vkCreateFence(c->device, &fi, NULL, &open_b.fence) != VK_SUCCESS)
            rt_fatal("vk: upload command buffer");
    }
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(open_b.cb, &bi);
    if (open_b.ts >= 0 && perf_gpu_timing() && up_ts_ready()) {
        vkCmdResetQueryPool(open_b.cb, up_pool, 2 * (uint32_t)open_b.ts, 2);
        vkCmdWriteTimestamp(open_b.cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, up_pool, 2 * (uint32_t)open_b.ts);
        open_b.timed = 1;
    }
    open_b.seq = ++next_seq;
    vk_crumb(open_b.cb, VK_CRUMB_UPLOAD, open_b.seq);
    open_active = 1;
    return open_b.cb;
}

void vk_upload_end(void) { plat_rmutex_unlock(&cmd_lock); }

// Staging memory comes from a pool of its own that keeps STG_MIN_BLOCKS blocks from first use: in the default pools VMA creates a
// block under its memory type's lock, so a block made by the headroom thread or a burst of uploads would hold every staging
// allocation. Buffers over half a block go to the default pool.
#define STG_MIN_BLOCKS 2
static const unsigned long long stg_block = 128ull << 20;
VmaPool stg_pool;
int stg_pool_tried;

void *vk_staging(VkDeviceSize size, VkBuffer *buf, VmaAllocation *alloc) {
    VkCtx *c = vk_ctx();
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size ? size : 16,
                             .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo aci = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                   .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    if (!__atomic_load_n(&stg_pool_tried, __ATOMIC_ACQUIRE)) {
        plat_once(&cmd_once, cmd_init, NULL);
        plat_rmutex_lock(&cmd_lock);
        if (!stg_pool_tried) {
            uint32_t type;
            VmaPoolCreateInfo pi = {.blockSize = stg_block, .minBlockCount = STG_MIN_BLOCKS};
            if (vmaFindMemoryTypeIndexForBufferInfo(c->vma, &bi, &aci, &type) == VK_SUCCESS) {
                pi.memoryTypeIndex = type;
                if (vmaCreatePool(c->vma, &pi, &stg_pool) != VK_SUCCESS) stg_pool = NULL;
            }
            rt_log("vk: staging pool %s (%d blocks of %llu MiB kept)", stg_pool ? "created" : "unavailable: default pool",
                   STG_MIN_BLOCKS, (unsigned long long)(stg_block >> 20));
            __atomic_store_n(&stg_pool_tried, 1, __ATOMIC_RELEASE);
        }
        plat_rmutex_unlock(&cmd_lock);
    }
    if (stg_pool && bi.size <= stg_block / 2) aci.pool = stg_pool;
    VmaAllocationInfo info;
    VkBuffer b;
    VmaAllocation a;
    VkResult r = vmaCreateBuffer(c->vma, &bi, &aci, &b, &a, &info);
    if (r != VK_SUCCESS) rt_fatal("vk: staging buffer of %llu bytes failed (%d)", (unsigned long long)size, (int)r);
    plat_add64(&up_bytes, (int64_t)bi.size);
    *buf = b;
    if (alloc) *alloc = a;   // the caller destroys it (vmaDestroyBuffer) after vk_upload_flush(1)
    else {
        if (open_b.nstg == open_b.cap) grow((void **)&open_b.stg, &open_b.cap, sizeof *open_b.stg);
        open_b.stg[open_b.nstg++] = (Staging){b, a};
    }
    return info.pMappedData;
}

uint64_t vk_upload_flush(int wait) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    VkCtx *c = vk_ctx();
    uint64_t seq = next_seq;
    if (c && open_active) {
        for (int i = 0; i < open_b.nstg; i++) vmaFlushAllocation(c->vma, open_b.stg[i].alloc, 0, VK_WHOLE_SIZE);
        vk_crumb(open_b.cb, VK_CRUMB_UPLOAD_END, open_b.seq);
        if (open_b.timed)
            vkCmdWriteTimestamp(open_b.cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, up_pool, 2 * (uint32_t)open_b.ts + 1);
        vkEndCommandBuffer(open_b.cb);
        plat_inc64(&up_batches);
        VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &open_b.cb};
        vk_queue_lock();
        VkResult r = vkQueueSubmit(c->gfx_queue, 1, &si, open_b.fence);
        vk_queue_unlock();
        if (r != VK_SUCCESS) {
            vk_device_fault_report(r);
            rt_fatal("vk: upload submit failed (%d)", (int)r);
        }
        if (ninflight == cap_inflight) grow((void **)&inflight, &cap_inflight, sizeof *inflight);
        inflight[ninflight++] = open_b;
        open_active = 0;
    }
    if (c && wait) {
        uint64_t w0 = ninflight && done_seq < seq ? kern_now_ns() : 0;
        while (ninflight && done_seq < seq) {
            // A lost device returns at once and its fence never signals: without this check the loop spins forever.
            VkResult w = vkWaitForFences(c->device, 1, &inflight[0].fence, VK_TRUE, UINT64_MAX);
            if (w != VK_SUCCESS && w != VK_TIMEOUT) {
                vk_device_fault_report(w);
                rt_fatal("vk: upload wait failed (%d)", (int)w);
            }
            retire(c, 0);
        }
        if (w0) plat_add64(&up_wait_ns, (int64_t)(kern_now_ns() - w0));
    }
    if (c) collect_locked(c);
    plat_rmutex_unlock(&cmd_lock);
    return seq;
}

static void defer(int kind, uint64_t handle, VmaAllocation alloc) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    if (ndead == cap_dead) grow((void **)&dead, &cap_dead, sizeof *dead);
    // The open batch may record into the object: it counts as the newest user.
    dead[ndead++] = (Dead){kind, handle, alloc, open_active ? open_b.seq : next_seq, tl_protected};
    plat_rmutex_unlock(&cmd_lock);
}

int dead_pending(void) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    int n = ndead;
    plat_rmutex_unlock(&cmd_lock);
    return n;
}

void vk_defer_image(VkImage img, VmaAllocation alloc) {
    if (img) defer(alloc ? DEAD_IMAGE_VMA : DEAD_IMAGE, (uint64_t)(uintptr_t)img, alloc);
}
void vk_defer_view(VkImageView view) { if (view) defer(DEAD_VIEW, (uint64_t)(uintptr_t)view, NULL); }
void vk_defer_buffer(VkBuffer buf, VmaAllocation alloc) {
    if (buf) defer(DEAD_BUFFER_VMA, (uint64_t)(uintptr_t)buf, alloc);
}

void vk_mem_protect_until(uint64_t timeline_value) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    // New retirements conservatively share the frontier; already queued objects keep their earlier protection value.
    if (timeline_value > tl_protected) tl_protected = timeline_value;
    plat_rmutex_unlock(&cmd_lock);
}

void vk_mem_collect(void) {
    VkCtx *c = vk_ctx();
    if (!c) return;
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    collect_locked(c);
    plat_rmutex_unlock(&cmd_lock);
}
