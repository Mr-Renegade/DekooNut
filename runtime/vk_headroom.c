// VRAM headroom: the next VMA block of a memory type is allocated on a background thread, because a vkAllocateMemory on the
// render thread stalls it for milliseconds. VRAM headroom: see README, "VRAM headroom".
// Under VRAM pressure (use plus two blocks past 85% of the budget) no headroom is kept.

#include "host.h"
#include "vk.h"
#include "platform/platform.h"

#include <stdio.h>
#include <stdlib.h>

#define HR_PERIOD_MS 500
#define HR_SMALL_HEAP (1ull << 30)   // VMA_SMALL_HEAP_MAX_SIZE: smaller heaps use blocks of a heap eighth

static struct {
    int on, stop;
    PlatMutex lock;             // held around each top-up; vk_headroom_stop takes it before the allocator goes
    int started;
    PlatEvent ev;               // auto-reset: another thread allocated, or stop
    volatile uint32_t tid;      // the headroom thread's id
    volatile int64_t types;    // memory types a thread other than this one allocated device memory of
    volatile int64_t made;     // blocks this thread brought in
    volatile int64_t other;    // device memory of 32 MiB or more other threads allocated (VMA blocks, large dedicated)
    VmaAllocation anchor[64]; // per memory type: the half block that keeps the newest spare block from being deleted
} H;

static void VKAPI_PTR on_allocate(VmaAllocator a, uint32_t type, VkDeviceMemory m, VkDeviceSize size, void *user) {
    (void)a, (void)m, (void)size, (void)user;
    if (!H.on || type >= 64) return;
    if (plat_thread_id() == H.tid) {
        plat_inc64(&H.made);
        return;
    }
    if (size >= 32ull << 20) plat_inc64(&H.other);
    plat_or64(&H.types, (int64_t)(1ull << type));
    plat_event_set(&H.ev);
}

const VmaDeviceMemoryCallbacks *vk_headroom_callbacks(void) {
    static const VmaDeviceMemoryCallbacks cb = {.pfnAllocate = on_allocate};
    H.on = 1;
    return &cb;
}

static VkDeviceSize block_size(const VkCtx *c, uint32_t type) {
    VkDeviceSize heap = c->mem_props.memoryHeaps[c->mem_props.memoryTypes[type].heapIndex].size;
    return heap <= HR_SMALL_HEAP ? heap / 8 : 256ull << 20;   // VMA's default preferredLargeHeapBlockSize
}

static int room(const VkCtx *c, uint32_t type, VkDeviceSize block) {
    VmaBudget b[VK_MAX_MEMORY_HEAPS];
    vmaGetHeapBudgets(c->vma, b);
    const VmaBudget *h = &b[c->mem_props.memoryTypes[type].heapIndex];
    return h->budget && h->usage + 2 * block <= h->budget / 100 * 85;
}

static void drop_anchor(const VkCtx *c, uint32_t type) {
    if (H.anchor[type]) vmaFreeMemory(c->vma, H.anchor[type]);
    H.anchor[type] = NULL;
}

static void top_up(const VkCtx *c, uint32_t type) {
    VkDeviceSize block = block_size(c, type);
    if (!room(c, type, block)) {
        drop_anchor(c, type);
        return;
    }
    VkMemoryRequirements req = {.size = block / 2, .alignment = 256, .memoryTypeBits = 1u << type};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_UNKNOWN};
    VmaAllocation a;
    int64_t before = H.made;
    if (vmaAllocateMemory(c->vma, &req, &ai, &a, NULL) != VK_SUCCESS) return;
    if (H.made == before) {   // there was room
        vmaFreeMemory(c->vma, a);
        return;
    }
    drop_anchor(c, type);
    H.anchor[type] = a;
    static volatile int32_t logged;
    if (plat_inc32(&logged) <= 32)
        rt_log("vk: headroom: a %.0f MiB block of memory type %u allocated ahead (%lld so far; %lld allocations of 32 MiB "
               "or more on other threads)", (double)block / 1048576.0, type, (long long)H.made, (long long)H.other);
}

static void headroom_main(void *arg) {
    (void)arg;
    __atomic_store_n(&H.tid, plat_thread_id(), __ATOMIC_RELEASE);
    plat_thread_priority(PLAT_PRIO_BELOW_NORMAL);
    uint64_t watch = 0;   // memory types seen so far
    for (;;) {
        if (watch) plat_event_wait_ns(&H.ev, (uint64_t)HR_PERIOD_MS * 1000000u);
        else plat_event_wait(&H.ev);
        watch |= (uint64_t)plat_xchg64(&H.types, 0);
        plat_mutex_lock(&H.lock);
        const VkCtx *c = H.stop ? NULL : vk_ctx();
        for (uint32_t t = 0; c && t < c->mem_props.memoryTypeCount && t < 64; t++)
            if (watch >> t & 1) top_up(c, t);
        int stop = H.stop;
        if (stop && (c = vk_ctx()))
            for (uint32_t t = 0; t < 64; t++) drop_anchor(c, t);
        plat_mutex_unlock(&H.lock);
        if (stop) return;
    }
}

// Before the allocator is destroyed: waits for a running top-up, frees the anchors, and the thread allocates no more.
void vk_headroom_stop(void) {
    if (!H.started) return;
    plat_mutex_lock(&H.lock);
    H.stop = 1;
    const VkCtx *c = vk_ctx();
    for (uint32_t t = 0; c && t < 64; t++) drop_anchor(c, t);
    plat_mutex_unlock(&H.lock);
    plat_event_set(&H.ev);
}

void vk_headroom_start(void) {
    if (!H.on || H.started) return;
    plat_mutex_init(&H.lock);
    plat_event_init(&H.ev, 0, 0);
    if (!plat_thread_spawn(headroom_main, NULL, "vk headroom", 0)) {
        H.on = 0;
        rt_log("vk: headroom: no thread, new VMA blocks stay on the allocating thread");
        return;
    }
    H.started = 1;
    rt_log("vk: headroom: the next VMA block of each memory type in use is allocated on a background thread");
}
