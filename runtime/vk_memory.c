// Vulkan memory of the graphics layer: NVN memory pools as Vulkan buffers, the virtual-pool page table, GPU address
// resolution, page write generations for re-uploads, and the upload batch that vk_texture.c records into.
//
// Behaviour ported from MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/totk_vpool.c (request checks, unmap
// semantics, 64 KiB pages) and nvn_dirty.c (generation stamps: a texture is stale when a covered page carries a
// generation newer than its upload), @ b75be8cf (MIT, Copyright (c) 2026 jhaizhou; origin d2r-mobile); and from LAS
// Recomp runtime/nvn/vulkan/exec.c:2094-2131 (vk_pages_changed: every write-watch query resets the OS bits, so each
// query stamps the pages it saw with a new generation shared by all users) and textures.c:240-270 (deferred
// destruction) @ da35b83a (MIT, Copyright (c) 2026 Peiikai). MNG aliases host VM (mach_vm_remap) and protects pages
// (fault handler); here virtual pools are a GPU-side page table over device-local memory and dirty pages come from
// GetWriteWatch (mem_written_pages), so neither is needed.
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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

// Nanosecond timestamps; qpc_ms converts to ms.
static uint64_t qpc(void) { return plat_now_ns(); }

static double qpc_ms(uint64_t ticks) { return (double)ticks / 1e6; }

// ---------------------------------------------------------------- pools
#define WIN_SHIFT 25                    // windowed shadows: a window starts every 32 MiB and spans two steps
#define WIN_STEP (1ull << WIN_SHIFT)
static PoolRec *window_get(PoolRec *r, uint64_t addr, int create);

static void mirror_forget(PoolRec *r);   // pool_lock exclusive: off the mirrored list
static int shadow_on;                    // shadow mode: imported pools get shadow storage instead (vk_mem_class_log)
static int shadow_pool(VkCtx *c, PoolRec *r, uint64_t addr, uint64_t size);
static void shadow_late(PoolRec *r);
static void shadow_touch(PoolRec *r);
static void shadow_forget(PoolRec *r);
static void mirror_retiring(PoolRec *r);  // vk_pool_fini: kept until mirror_retire (or vk_mem_shutdown)
static void mirror_retire(void *arg);    // completion thread: destroys a finalized mirrored pool's record

static PlatMutex pool_lock = PLAT_MUTEX_INIT;
// Bumped under pool_lock (exclusive) by every change to the index or a page table: vk_mem_resolve's per-thread hits
// stay valid while it is unchanged.
static uint64_t pool_gen = 1;
static uint64_t storage_serial;
// Bumped when a buffer use may get another buffer for the same span (a mirror created, refused or forgotten) and at
// every submission's start (mirror syncs are per submission). vk_mem_gen: both together.
static uint64_t mirror_gen;
static pidx pool_index;                // POOL_IMPORT, POOL_PHYS and POOL_VIRT records by GPU address
static PoolRec **phys_list;            // physical pools by key, for MapVirtual requests
static int nphys, cap_phys;
static int32_t n_imports, n_import_failures;
static PFN_vkGetMemoryHostPointerPropertiesEXT p_host_props;

static uint32_t gmem_changes;   // raw pool table: mirrors made since the start
static int ft_on;                                         // the guest page table is in use
static void ft_pool_put(PoolRec *r);                      // the pool's pages now lead to its current memory
static void ft_pool_gone(const PoolRec *r);               // pool_lock exclusive: to the sink, what overlaps it stays
static VkBufferUsageFlags pool_usage(const VkCtx *c) {
    VkBufferUsageFlags u = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                           VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT |
                           VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                           VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    if (c->has_transform_feedback)
        u |= VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT | VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT;
    if (c->has_conditional_rendering) u |= VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT;
    if (c->has_buffer_device_address) u |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;   // the raw pool table, vk_mem_gmem_info
    return u;
}

static int memory_type(const VkCtx *c, uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < c->mem_props.memoryTypeCount; i++)
        if ((bits >> i & 1) && (c->mem_props.memoryTypes[i].propertyFlags & want) == want) return (int)i;
    for (uint32_t i = 0; i < c->mem_props.memoryTypeCount; i++)
        if (bits >> i & 1) return (int)i;
    return -1;
}

// CPU-visible storage: the guest pages themselves become the buffer's memory (VK_EXT_external_memory_host).
static int import_pool(VkCtx *c, PoolRec *r, uint64_t addr, uint64_t size) {
    if (shadow_on) return shadow_pool(c, r, addr, size);
    if (!p_host_props)
        p_host_props = (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(c->device, "vkGetMemoryHostPointerPropertiesEXT");
    uint64_t al = c->min_imported_host_alignment ? c->min_imported_host_alignment : 0x1000;
    r->import_lo = addr & ~(al - 1);
    r->import_hi = (addr + size + al - 1) & ~(al - 1);
    uint64_t n = r->import_hi - r->import_lo;
    VkMemoryHostPointerPropertiesEXT hp = {.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    VkResult res = p_host_props ? p_host_props(c->device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                               (void *)(uintptr_t)r->import_lo, &hp)
                                : VK_ERROR_EXTENSION_NOT_PRESENT;
    if (res != VK_SUCCESS || !hp.memoryTypeBits) {
        rt_log("vk: pool %#llx+%#llx: host pointer not importable (%d)", (unsigned long long)addr, (unsigned long long)size, (int)res);
        return 0;
    }
    VkExternalMemoryBufferCreateInfo eb = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                                           .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &eb, .size = n,
                             .usage = pool_usage(c), .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    if ((res = vkCreateBuffer(c->device, &bi, NULL, &r->buf)) != VK_SUCCESS) {
        rt_log("vk: pool %#llx: vkCreateBuffer for the import failed (%d)", (unsigned long long)addr, (int)res);
        return 0;
    }
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(c->device, r->buf, &req);
    int type = memory_type(c, req.memoryTypeBits & hp.memoryTypeBits,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type < 0 || req.size > n) {
        rt_log("vk: pool %#llx: no memory type for the import (buffer %#x, host %#x, size %#llx of %#llx)",
               (unsigned long long)addr, req.memoryTypeBits, hp.memoryTypeBits, (unsigned long long)req.size,
               (unsigned long long)n);
        vkDestroyBuffer(c->device, r->buf, NULL);
        r->buf = VK_NULL_HANDLE;
        return 0;
    }
    VkMemoryAllocateFlagsInfo device_address = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
                                                .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkImportMemoryHostPointerInfoEXT im = {.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
                                           .pNext = c->has_buffer_device_address ? &device_address : NULL,
                                           .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                           .pHostPointer = (void *)(uintptr_t)r->import_lo};
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &im, .allocationSize = n,
                               .memoryTypeIndex = (uint32_t)type};
    if ((res = vkAllocateMemory(c->device, &ai, NULL, &r->mem)) != VK_SUCCESS ||
        (res = vkBindBufferMemory(c->device, r->buf, r->mem, 0)) != VK_SUCCESS) {
        rt_log("vk: pool %#llx+%#llx: host memory import failed (%d)", (unsigned long long)addr, (unsigned long long)size, (int)res);
        if (r->mem) vkFreeMemory(c->device, r->mem, NULL);
        vkDestroyBuffer(c->device, r->buf, NULL);
        r->buf = VK_NULL_HANDLE;
        r->mem = VK_NULL_HANDLE;
        return 0;
    }
    plat_inc32(&n_imports);
    return 1;
}

// Physical pool: device-local memory of the pool size that both buffers and optimal-tiling images can be bound to.
static int alloc_physical(VkCtx *c, PoolRec *r) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = r->size, .usage = pool_usage(c),
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VkResult res = vkCreateBuffer(c->device, &bi, NULL, &r->buf);
    if (res != VK_SUCCESS) { rt_log("vk: physical pool: vkCreateBuffer failed (%d)", (int)res); return 0; }
    VkMemoryRequirements rb;
    vkGetBufferMemoryRequirements(c->device, r->buf, &rb);
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                            .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {256, 256, 1}, .mipLevels = 1, .arrayLayers = 1,
                            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
    VkDeviceImageMemoryRequirements di = {.sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS, .pCreateInfo = &ii};
    VkMemoryRequirements2 ri = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    vkGetDeviceImageMemoryRequirements(c->device, &di, &ri);
    uint32_t bits = rb.memoryTypeBits & ri.memoryRequirements.memoryTypeBits;
    if (!bits) bits = rb.memoryTypeBits;
    VkDeviceSize align = NVN_VPOOL_PAGE;
    if (rb.alignment > align) align = rb.alignment;
    if (ri.memoryRequirements.alignment > align) align = ri.memoryRequirements.alignment;
    VkMemoryRequirements req = {.size = (rb.size + align - 1) & ~(align - 1), .alignment = align, .memoryTypeBits = bits};
    VmaAllocationCreateInfo aci = {.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
                                   .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
    if ((res = vmaAllocateMemory(c->vma, &req, &aci, &r->alloc, NULL)) != VK_SUCCESS ||
        (res = vmaBindBufferMemory(c->vma, r->alloc, r->buf)) != VK_SUCCESS) {
        rt_log("vk: physical pool of %#llx bytes: device-local allocation failed (%d)", (unsigned long long)r->size, (int)res);
        if (r->alloc) vmaFreeMemory(c->vma, r->alloc);
        vkDestroyBuffer(c->device, r->buf, NULL);
        r->buf = VK_NULL_HANDLE;
        r->alloc = NULL;
        return 0;
    }
    {   // Switch pool memory starts zeroed; once per pool, not per image placed in it
        VkCommandBuffer cb = vk_upload_begin();
        vkCmdFillBuffer(cb, r->buf, 0, VK_WHOLE_SIZE, 0);
        VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                               .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
        VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
        vkCmdPipelineBarrier2(cb, &di);
        vk_upload_end();
    }
    return 1;
}

static void index_put(PoolRec *r) {
    __atomic_add_fetch(&pool_gen, 1, __ATOMIC_RELEASE);
    pidx_ent e = {.lo = r->lo, .hi = r->lo + r->size, .base = r->lo, .buf = r, .gpu = r->lo, .key = r->key};
    pidx_remove(&pool_index, r->key, NULL);   // an object initialized again without Finalize
    if (!pidx_insert(&pool_index, &e)) rt_fatal("vk: out of memory for the pool index");
}

static PoolRec *phys_find(const pool_t *key) {
    for (int i = 0; i < nphys; i++) if (phys_list[i]->key == key) return phys_list[i];
    return NULL;
}

static void destroy_rec(VkCtx *c, PoolRec *r) {
    if (r->wbs) vmaDestroyBuffer(c->vma, r->wbs, r->wbs_a), r->wbs = VK_NULL_HANDLE;
    if (r->mbuf) vmaDestroyBuffer(c->vma, r->mbuf, r->malloc_);
    if (r->salloc) vmaDestroyBuffer(c->vma, r->buf, r->salloc), r->buf = VK_NULL_HANDLE;
    if (r->buf) vkDestroyBuffer(c->device, r->buf, NULL);
    if (r->mem) vkFreeMemory(c->device, r->mem, NULL);
    if (r->alloc) vmaFreeMemory(c->vma, r->alloc);
    for (uint32_t k = 0; k < r->nwin; k++)
        if (r->win[k]) destroy_rec(c, r->win[k]);
    free(r->win);
    free(r->pt);
    free(r);
}

int vk_pool_init(pool_t *p) {
    VkCtx *c = vk_ctx();
    if (!c) { rt_log("vk: MemoryPoolInitialize before vk_init"); return 0; }
    // Reinitializing the same object retires its old backing before installing the new mapping. In particular,
    // descriptors at a reused address must not inherit registrations from a previous storage lifetime.
    int reinit = 0;
    plat_mutex_lock_shared(&pool_lock);
    for (int i = 0; i < pool_index.n; i++) if (pool_index.e[i].key == p) { reinit = 1; break; }
    plat_mutex_unlock_shared(&pool_lock);
    if (reinit) vk_pool_fini(p);
    // The GPU never sees a GPU_NO_ACCESS pool, except a physical one (0x189, or 0x18c CPU-visible) through the virtual
    // pools that map it; those get device-local memory.
    if ((p->flags & NVN_MEMORY_POOL_FLAGS_GPU_NO_ACCESS) && !(p->flags & NVN_MEMORY_POOL_FLAGS_PHYSICAL)) return 1;
    PoolRec *r = calloc(1, sizeof *r);
    if (!r) rt_fatal("vk: out of memory for a pool record");
    r->key = p;
    r->flags = p->flags;
    r->lo = (uintptr_t)p->mem;
    r->size = p->size;
    int ok;
    if (p->flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL) {
        r->kind = POOL_VIRT;
        r->npages = p->size / NVN_VPOOL_PAGE;
        r->pt = calloc(r->npages ? r->npages : 1, sizeof *r->pt);
        ok = r->pt != NULL;
    } else if (p->flags & NVN_MEMORY_POOL_FLAGS_PHYSICAL) {
        r->kind = POOL_PHYS;
        ok = alloc_physical(c, r);
    } else {
        r->kind = POOL_IMPORT;
        mem_thaw((uintptr_t)p->mem, p->size);   // the import may need write access to the pages
        ok = import_pool(c, r, (uintptr_t)p->mem, p->size);
        if (!ok) {
            // Not fatal for the game: the pool works for the CPU, and vk_mem_resolve reports its addresses as unbacked.
            long n = plat_inc32(&n_import_failures);
            if (n <= 8) rt_log("vk: pool %p (flags %#x, %#llx bytes) has no GPU buffer (%ld so far)", (void *)p->mem,
                               p->flags, (unsigned long long)p->size, n);
            r->buf = VK_NULL_HANDLE;
            ok = 1;
        }
    }
    if (!ok) { free(r->pt); free(r); return 0; }
    plat_mutex_lock(&pool_lock);
    if (!++storage_serial) ++storage_serial;
    r->storage_serial = storage_serial;
    index_put(r);
    if (ft_on) ft_pool_put(r);
    if (r->kind == POOL_PHYS) {
        if (nphys == cap_phys) {
            cap_phys = cap_phys ? cap_phys * 2 : 16;
            phys_list = realloc(phys_list, (size_t)cap_phys * sizeof *phys_list);
            if (!phys_list) rt_fatal("vk: out of memory for the physical pool list");
        }
        phys_list[nphys++] = r;
    }
    plat_mutex_unlock(&pool_lock);
    return 1;
}

void vk_pool_fini(pool_t *p) {
    VkCtx *c = vk_ctx();
    if (!c) return;
    pidx_ent e;
    plat_mutex_lock(&pool_lock);
    int had = pidx_remove(&pool_index, p, &e);
    __atomic_add_fetch(&pool_gen, 1, __ATOMIC_RELEASE);
    PoolRec *r = had ? e.buf : NULL;
    if (r && r->mstate == MS_ON) mirror_forget(r);
    if (r && r->kind == POOL_PHYS) {
        for (int i = 0; i < nphys; i++)
            if (phys_list[i] == r) { phys_list[i] = phys_list[--nphys]; break; }
        for (int i = 0; i < pool_index.n; i++) {   // virtual pages that still map it become unmapped
            PoolRec *v = pool_index.e[i].buf;
            if (v->kind != POOL_VIRT) continue;
            for (uint64_t k = 0; k < v->npages; k++)
                if (v->pt[k].phys == r) v->pt[k].phys = NULL;
        }
    }
    if (r && ft_on) ft_pool_gone(r);
    plat_mutex_unlock(&pool_lock);
    if (!r) return;
    // Take the texture lock only after dropping pool_lock: descriptor lookup locks texture, then pool.
    if (r->kind == POOL_IMPORT) vk_texture_descriptors_retire(r->lo, r->size, 0);
    else if (r->kind == POOL_PHYS) vk_texture_descriptors_retire(0, 0, r->storage_serial);
    if (r->smap) shadow_forget(r);
    for (uint32_t k = 0; k < r->nwin; k++)
        if (r->win[k] && r->win[k]->smap) shadow_forget(r->win[k]);
    if (r->kind == POOL_IMPORT) mem_thaw(r->lo, r->size);   // the storage goes back to the guest writable
    // Images cut from this range go first; a used buffer or memory waits until the GPU is done with it, since the
    // guest gets the storage back when this returns.
    vk_texture_range_gone(r->lo, r->size, r->alloc);
    if (r->used && r->kind != POOL_VIRT) {
        vk_upload_flush(1);
        vk_queue_lock();
        vkQueueWaitIdle(c->gfx_queue);
        vk_queue_unlock();
    }
    vk_mem_collect();
    // A mirrored pool's record and buffers outlive the submission being recorded, whose sync copies and write-backs may
    // name them (the game frees a pool only once the GPU is done with it, but its CPU writes may still be queued).
    if (r->mbuf) mirror_retiring(r), vk_completion_add(vk_sync_issued() + 1, mirror_retire, r);
    else destroy_rec(c, r);
}

// Requests were checked by the front end (nvn_api_core.c); apply them in order.
int vk_pool_map_virtual(pool_t *v, int n, const void *requests) {
    const nvn_mapping_t *rq = requests;
    plat_mutex_lock(&pool_lock);
    const pidx_ent *e = pidx_find(&pool_index, (uintptr_t)v->mem, 0);
    PoolRec *r = e && e->key == v ? e->buf : NULL;
    int ok = r && r->kind == POOL_VIRT;
    __atomic_add_fetch(&pool_gen, 1, __ATOMIC_RELEASE);
    for (int i = 0; ok && i < n; i++) {
        PoolRec *ph = rq[i].physical_pool ? phys_find((const pool_t *)(const void *)rq[i].physical_pool) : NULL;
        if (rq[i].physical_pool && !ph) {
            rt_log("vk: MapVirtual request %d names physical pool %p that has no Vulkan memory", i,
                   (const void *)rq[i].physical_pool);
            ok = 0;
            break;
        }
        uint64_t first = (uint64_t)rq[i].virtual_offset / NVN_VPOOL_PAGE, count = rq[i].size / NVN_VPOOL_PAGE;
        if (first + count > r->npages) { ok = 0; break; }
        for (uint64_t k = 0; k < count; k++)
            r->pt[first + k] = (VPage){ph, ph ? (uint64_t)rq[i].physical_offset + k * NVN_VPOOL_PAGE : 0};
    }
    if (r && ft_on) ft_pool_put(r);
    plat_mutex_unlock(&pool_lock);
    if (!ok) rt_log("vk: MapVirtual on pool %p failed", (void *)v->mem);
    return ok;
}

// Per-thread cache of resolved import and physical pools that overlap no other index entry, valid while pool_gen is
// unchanged. Direct-mapped by the 4 KiB page of the address, so a hit is one compare.
static int rec_class(const PoolRec *r) {
    if (r->kind != POOL_IMPORT) return VK_MCLS_DEVICE;
    return r->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS ? VK_MCLS_HOST_NOACC
         : r->flags & NVN_MEMORY_POOL_FLAGS_CPU_CACHED ? VK_MCLS_HOST_CACHED : VK_MCLS_HOST_UNC;
}

typedef struct { uint64_t gen, lo, hi, base, end; VkBuffer buf; VmaAllocation alloc; int cls; void *pool; } ResolveHit;
#define RHIT_SLOTS 512
static _Thread_local ResolveHit rhit[RHIT_SLOTS];

static ResolveHit *rhit_slot(uint64_t addr) { return &rhit[(addr >> 12 ^ addr >> 21) & (RHIT_SLOTS - 1)]; }

static int resolve_locked(uint64_t addr, uint64_t size, VkMemSpan *out, int create);

int vk_mem_inspect(uint64_t addr, uint64_t size, VkMemInspect *out) {
    if (!out) return 0;
    memset(out, 0, sizeof *out);
    if (!addr || !size || addr > UINT64_MAX - size) return 0;
    int ok = 0;
    VkMemInspect result = {0};
    plat_mutex_lock_shared(&pool_lock);
    // Match bind_range's one-byte resolver candidate, then check the whole request against that same backing.
    const pidx_ent *e = pidx_find(&pool_index, addr, 1);
    const PoolRec *r = e ? e->buf : NULL;
    if (!r || !r->size || r->lo > UINT64_MAX - r->size || addr < r->lo) goto done;
    uint64_t rel = addr - r->lo;
    if (rel >= r->size || size > r->size - rel) goto done;
    uint64_t remaining = r->size - rel;
    result.pool_flags = r->flags;
    if (r->kind == POOL_IMPORT) {
        if (!r->buf || addr < r->import_lo || addr + size > r->import_hi) goto done;
        result.pool_kind = VK_MEM_INSPECT_POOL_IMPORT;
        result.offset = addr - r->import_lo;
        result.avail = r->import_hi - addr;
        if (result.avail > remaining) result.avail = remaining;
        result.cls = rec_class(r);
        _Static_assert(MS_NONE == VK_MEM_INSPECT_MIRROR_NONE && MS_ON == VK_MEM_INSPECT_MIRROR_ON &&
                       MS_REFUSED == VK_MEM_INSPECT_MIRROR_REFUSED, "inspection mirror-state schema");
        result.mirror_state = r->mstate;
        result.mirror_present = r->mbuf != VK_NULL_HANDLE;
        result.shadow_storage = r->smap != NULL;
    } else if (r->kind == POOL_PHYS) {
        if (!r->buf) goto done;
        result.pool_kind = VK_MEM_INSPECT_POOL_PHYSICAL;
        result.offset = rel;
        result.avail = remaining;
        result.cls = VK_MCLS_DEVICE;
    } else if (r->kind == POOL_VIRT) {
        uint64_t k = rel / NVN_VPOOL_PAGE, in_page = rel % NVN_VPOOL_PAGE;
        uint64_t pages = r->size / NVN_VPOOL_PAGE;
        if (pages > r->npages) pages = r->npages;
        if (!r->pt || k >= pages) goto done;
        const PoolRec *ph = r->pt[k].phys;
        uint64_t start = r->pt[k].off;
        if (!ph || ph->kind != POOL_PHYS || !ph->buf || start % NVN_VPOOL_PAGE ||
            start > ph->size || NVN_VPOOL_PAGE > ph->size - start) goto done;
        uint64_t end = start + NVN_VPOOL_PAGE;
        for (uint64_t j = k + 1; j < pages && r->pt[j].phys == ph && r->pt[j].off == end; j++) {
            if (NVN_VPOOL_PAGE > ph->size - end) break;
            end += NVN_VPOOL_PAGE;
        }
        result.pool_kind = VK_MEM_INSPECT_POOL_VIRTUAL;
        result.offset = start + in_page;
        result.avail = end - result.offset;
        if (result.avail > remaining) result.avail = remaining;
        result.cls = VK_MCLS_DEVICE;
    } else {
        goto done;
    }
    if (result.avail < size) goto done;
    result.buffer_present = 1;
    *out = result;
    ok = 1;
done:
    plat_mutex_unlock_shared(&pool_lock);
    return ok;
}

int vk_mem_descriptor_key(uint64_t addr, uint64_t size, VkDescKey *out) {
    memset(out, 0, sizeof *out);
    if (!addr || !size || addr > UINT64_MAX - size) return 0;
    int ok = 0;
    plat_mutex_lock_shared(&pool_lock);
    const pidx_ent *e = pidx_find(&pool_index, addr, size);
    const PoolRec *r = e ? e->buf : NULL;
    if (r && r->kind == POOL_IMPORT) {
        *out = (VkDescKey){0, addr};
        ok = 1;
    } else if (r && r->kind == POOL_PHYS) {
        *out = (VkDescKey){r->storage_serial, addr - r->lo};
        ok = 1;
    } else if (r && r->kind == POOL_VIRT) {
        uint64_t rel = addr - r->lo, k = rel / NVN_VPOOL_PAGE;
        const PoolRec *ph = k < r->npages ? r->pt[k].phys : NULL;
        if (ph) {
            uint64_t off = r->pt[k].off + rel % NVN_VPOOL_PAGE;
            uint64_t end = r->pt[k].off + NVN_VPOOL_PAGE;
            for (uint64_t j = k + 1; end - off < size && j < r->npages && r->pt[j].phys == ph &&
                 r->pt[j].off == end; j++) end += NVN_VPOOL_PAGE;
            if (end - off >= size && off <= ph->size && size <= ph->size - off) {
                *out = (VkDescKey){ph->storage_serial, off};
                ok = 1;
            }
        }
    }
    plat_mutex_unlock_shared(&pool_lock);
    return ok;
}

int vk_mem_resolve(uint64_t addr, uint64_t size, VkMemSpan *out) {
    memset(out, 0, sizeof *out);
    uint64_t g = __atomic_load_n(&pool_gen, __ATOMIC_ACQUIRE), end = addr + size;
    const ResolveHit *h = rhit_slot(addr);
    if (end >= addr && h->gen == g && addr >= h->lo && end <= h->hi) {
        out->buffer = h->buf;
        out->offset = addr - h->base;
        out->avail = h->end - addr;
        out->alloc = h->alloc;
        out->cls = h->cls;
        out->pool = h->pool;
        if (h->alloc) out->alloc_offset = out->offset;
        if (shadow_on && h->pool && !((PoolRec *)h->pool)->stouched) shadow_touch(h->pool);
        return 1;
    }
    return resolve_locked(addr, size, out, 1);
}

// Like vk_mem_resolve but never makes a deferred shadow (callers only ask what backs an address).
int vk_mem_resolve_existing(uint64_t addr, uint64_t size, VkMemSpan *out) {
    return resolve_locked(addr, size, out, 0);
}

// The pool index lookup under the lock; a non-overlapped import or physical pool goes into the address's cache slot.
static int resolve_locked(uint64_t addr, uint64_t size, VkMemSpan *out, int create) {
    memset(out, 0, sizeof *out);
    plat_mutex_lock_shared(&pool_lock);
    const pidx_ent *e = pidx_find(&pool_index, (uintptr_t)addr, size);
    PoolRec *r = e ? e->buf : NULL;
    if (r && r->nwin) {   // windowed shadows: the window that holds addr and at least a step after it
        PoolRec *w = window_get(r, addr, create);
        int ok = w && addr + size <= w->import_hi;
        if (w && !ok) {
            static volatile int32_t lines;
            if (plat_inc32(&lines) <= 4)
                rt_log("mem: %#llx+%#llx runs past its shadow window (%#llx+%#llx): left unbacked",
                       (unsigned long long)addr, (unsigned long long)size, (unsigned long long)w->import_lo,
                       (unsigned long long)w->size);
        }
        if (ok) {
            int i = (int)(e - pool_index.e);
            if ((i == 0 || pool_index.maxhi[i - 1] <= e->lo) && (i + 1 == pool_index.n || pool_index.e[i + 1].lo >= e->hi)) {
                // The cache slot covers only the step whose lookups pick this window, so a hit and the locked lookup
                // agree.
                uint64_t s_lo = r->import_lo + ((addr - r->import_lo) & ~(WIN_STEP - 1)), s_hi = s_lo + WIN_STEP;
                ResolveHit *h = rhit_slot(addr);
                h->gen = __atomic_load_n(&pool_gen, __ATOMIC_RELAXED);
                h->lo = s_lo > e->lo ? s_lo : e->lo, h->hi = s_hi < e->hi ? s_hi : e->hi;
                h->buf = w->buf, h->cls = rec_class(w), h->base = w->import_lo, h->end = w->import_hi;
                h->alloc = NULL, h->pool = w;
            }
            out->cls = rec_class(w);
            out->pool = w;
            out->buffer = w->buf;
            out->offset = addr - w->import_lo;
            out->avail = w->import_hi - addr;
            if (w->smap && !w->stouched) shadow_touch(w);
            if (!r->used) plat_xchg32(&r->used, 1);
        }
        plat_mutex_unlock_shared(&pool_lock);
        if (!ok) memset(out, 0, sizeof *out);
        return ok;
    }
    if (create && r && r->kind == POOL_IMPORT && r->slazy && !__atomic_load_n(&r->buf, __ATOMIC_ACQUIRE)) shadow_late(r);
    int ok = 0;
    if (r && (r->kind == POOL_IMPORT ? r->buf != VK_NULL_HANDLE : r->kind == POOL_PHYS)) {
        int i = (int)(e - pool_index.e);
        if ((i == 0 || pool_index.maxhi[i - 1] <= e->lo) && (i + 1 == pool_index.n || pool_index.e[i + 1].lo >= e->hi)) {
            ResolveHit *h = rhit_slot(addr);
            h->gen = __atomic_load_n(&pool_gen, __ATOMIC_RELAXED);   // stable: changed only under the exclusive lock
            h->lo = e->lo, h->hi = e->hi, h->buf = r->buf, h->cls = rec_class(r);
            if (r->kind == POOL_IMPORT) h->base = r->import_lo, h->end = r->import_hi, h->alloc = NULL, h->pool = r;
            else h->base = r->lo, h->end = r->lo + r->size, h->alloc = r->alloc, h->pool = NULL;
        }
    }
    if (r && r->kind == POOL_IMPORT && r->buf) {
        out->cls = rec_class(r);
        out->pool = r;
        out->buffer = r->buf;
        out->offset = addr - r->import_lo;
        out->avail = r->import_hi - addr;
        if (r->smap && !r->stouched) shadow_touch(r);
        ok = 1;
    } else if (r && r->kind == POOL_PHYS) {
        out->buffer = r->buf;
        out->offset = out->alloc_offset = addr - r->lo;
        out->avail = r->size - out->offset;
        out->alloc = r->alloc;
        out->cls = VK_MCLS_DEVICE;
        ok = 1;
    } else if (r && r->kind == POOL_VIRT) {
        uint64_t rel = addr - r->lo, k = rel / NVN_VPOOL_PAGE;
        if (k < r->npages && r->pt[k].phys) {
            PoolRec *ph = r->pt[k].phys;
            uint64_t off = r->pt[k].off + rel % NVN_VPOOL_PAGE, end = r->pt[k].off + NVN_VPOOL_PAGE;
            for (uint64_t j = k + 1; j < r->npages && r->pt[j].phys == ph && r->pt[j].off == end; j++) end += NVN_VPOOL_PAGE;
            out->buffer = ph->buf;
            out->offset = out->alloc_offset = off;
            out->avail = end - off;
            out->alloc = ph->alloc;
            out->cls = VK_MCLS_DEVICE;
            ok = out->avail >= size;
            if (ok && !ph->used) plat_xchg32(&ph->used, 1);   // read first: the flag is only ever set
        }
    }
    if (ok && !r->used) plat_xchg32(&r->used, 1);
    plat_mutex_unlock_shared(&pool_lock);
    if (!ok) memset(out, 0, sizeof *out);
    return ok;
}

// Raw pool table (translator patch 0027), for shaders that follow a pointer read from memory: sorted 8-word entries
// {start, end, delta = device minus guest address}; entry 0 is empty. It is a ring, since a table changed in place
// would break shaders still reading it. Writes through it are not noted as GPU writes, so a pool reached only this
// way gets no mirror sync. See README, "Shader access to guest memory".
#define GMEM_ENTRIES 8192u
#define GMEM_BYTES (GMEM_ENTRIES * 32u)
#define GMEM_RING 32u
static VkBuffer gmem_buf;
static VmaAllocation gmem_alloc;
static uint8_t *gmem_ring;
static uint32_t *gmem_map;           // the current slot
static uint32_t gmem_slot;
static uint64_t gmem_used[GMEM_RING];  // the submission that last read each slot
static uint64_t gmem_seen = ~0ull;

VkDeviceAddress gmem_device(VkBuffer b) {
    VkBufferDeviceAddressInfo i = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = b};
    return vkGetBufferDeviceAddress(vk_ctx()->device, &i);
}

static void gmem_put(uint32_t *n, uint64_t *last, uint64_t lo, uint64_t hi, uint64_t delta) {
    if (lo < *last) lo = *last;       // pools can nest; the earlier one keeps the overlap
    if (lo >= hi) return;
    if (*n >= GMEM_ENTRIES) { static int once; if (!once++) rt_log("mem: raw pool table is full: later pools drop accesses"); return; }
    uint64_t *e = (uint64_t *)(gmem_map + (size_t)*n * 8);
    e[0] = lo, e[1] = hi, e[2] = delta;
    ++*n, *last = hi;
}

static void gmem_build(void) {
    uint32_t n = 1;
    uint64_t last = 0;
    uint32_t slot = (gmem_slot + 1) % GMEM_RING;
    for (uint32_t i = 0; i < GMEM_RING && gmem_used[slot] > vk_sync_fence_done(); i++) slot = (slot + 1) % GMEM_RING;
    if (gmem_used[slot] > vk_sync_fence_done()) { static int once; if (!once++) rt_log("mem: raw pool table: every slot is in flight"); }
    gmem_slot = slot, gmem_map = (uint32_t *)(gmem_ring + (size_t)slot * GMEM_BYTES);
    memset(gmem_map, 0, 32);
    plat_mutex_lock_shared(&pool_lock);
    for (int i = 0; i < pool_index.n; i++) {
        const pidx_ent *e = &pool_index.e[i];
        PoolRec *r = e->buf;
        if (!r) continue;
        if (r->kind == POOL_IMPORT) {
            VkBuffer b = r->mstate == MS_ON ? r->mbuf : r->buf;
            if (b) gmem_put(&n, &last, e->lo, e->hi, gmem_device(b) - r->import_lo);
        } else if (r->kind == POOL_PHYS) {
            if (r->buf) gmem_put(&n, &last, e->lo, e->hi, gmem_device(r->buf) - r->lo);
        } else if (r->kind == POOL_VIRT) {
            for (uint64_t k = 0; k < r->npages;) {
                PoolRec *ph = r->pt[k].phys;
                if (!ph || !ph->buf) { k++; continue; }
                uint64_t j = k + 1, off = r->pt[k].off;
                while (j < r->npages && r->pt[j].phys == ph && r->pt[j].off == off + (j - k) * NVN_VPOOL_PAGE) j++;
                uint64_t lo = r->lo + k * NVN_VPOOL_PAGE;
                gmem_put(&n, &last, lo, r->lo + j * NVN_VPOOL_PAGE, gmem_device(ph->buf) + off - lo);
                k = j;
            }
        }
    }
    plat_mutex_unlock_shared(&pool_lock);
    for (uint32_t i = n; i < GMEM_ENTRIES; i++) ((uint64_t *)(gmem_map + (size_t)i * 8))[0] = ~0ull;
    static unsigned builds;
    if (builds++ < 6 || builds % 256 == 0) rt_log("mem: raw pool table: %u ranges (build %u)", n - 1, builds);
}

static void gmem_sync_mirrors(void);

int vk_mem_gmem_info(VkBuffer *buffer, VkDeviceSize *offset, VkDeviceSize *size) {
    VkCtx *c = vk_ctx();
    if (!c->has_buffer_device_address) return 0;
    gmem_sync_mirrors();
    if (!gmem_buf) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)GMEM_BYTES * GMEM_RING,
                                 .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO, .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                                      .requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
        VmaAllocationInfo info;
        if (vmaCreateBuffer(c->vma, &bi, &ai, &gmem_buf, &gmem_alloc, &info) != VK_SUCCESS) return 0;
        gmem_ring = info.pMappedData;
    }
    uint64_t now = __atomic_load_n(&pool_gen, __ATOMIC_ACQUIRE) + __atomic_load_n(&gmem_changes, __ATOMIC_ACQUIRE);
    if (now != gmem_seen) gmem_seen = now, gmem_build();
    gmem_used[gmem_slot] = vk_sync_issued() + 1;
    *buffer = gmem_buf, *offset = (VkDeviceSize)gmem_slot * GMEM_BYTES, *size = GMEM_BYTES;
    return 1;
}

// ---- guest page table (translator patch 0032): see README, "Shader access to guest memory".
// A shader loads table[guest address >> 12] (a device page number), then accesses entry << 12 plus the page offset.
// One 32-bit entry per 4 KiB page of 36 bits of guest address; bit 39 (virtual pools) folds to the table's unused
// bottom, as the translator masks it away. The table is aligned to its size: the shader ORs, it does not add.
// Unmapped pages lead to the sink page. Changes go in queue order through the prologue of the part that needs them.
#define FT_ENTRIES (1u << 24)
#define FT_BYTES ((uint64_t)FT_ENTRIES * 4)
#define FT_SINK (FT_ENTRIES - 1)
#define FT_MASK ((1ull << 39) - 1)
typedef struct { uint32_t index, n, page, step; } FtRun;   // step 1: consecutive device pages, step 0: all one page
static VkBuffer ft_buf, ft_sink_buf;
static VmaAllocation ft_alloc, ft_sink_alloc;
static uint64_t ft_addr, ft_off;   // the table's device address and its offset in ft_buf
static uint32_t ft_sink_page;
static int ft_filled;
static FtRun *ft_runs;
static int ft_n, ft_cap;
static PlatMutex ft_lock;
static uint64_t ft_entries_sent;

// On where the device is capable: NVIDIA, on Linux, with device addresses. It must be decided before the first
// translation, as it is part of the shader cache keys.
static int ft_mode = -1;
void vk_mem_flat_choose(int capable) {
    ft_mode = capable ? 2 : 0;
}

int vk_mem_flat_mode(void) { return ft_mode > 0 ? ft_mode : 0; }

uint64_t vk_mem_flat_table(void) { return ft_on ? ft_addr : 0; }

static void ft_run(uint64_t lo, uint64_t hi, uint64_t device, int sink) {
    uint64_t span = hi - lo;
    lo &= FT_MASK, hi = lo + span;
    uint64_t first = lo >> 12, last = (hi + 0xfff) >> 12;
    if (sink ? 0 : (device & 0xfff) || (device >> 44)) {
        static int once;
        if (!once++) rt_log("mem: guest page table: device address %#llx does not fit an entry", (unsigned long long)device);
        return;
    }
    if (last > FT_SINK) {
        static int once;
        if (!once++) rt_log("mem: guest page table: guest address %#llx is past its range", (unsigned long long)lo);
        return;
    }
    if (first >= last) return;
    plat_mutex_lock(&ft_lock);
    if (ft_n == ft_cap) grow((void **)&ft_runs, &ft_cap, sizeof *ft_runs);
    ft_runs[ft_n++] = (FtRun){(uint32_t)first, (uint32_t)(last - first), sink ? ft_sink_page : (uint32_t)(device >> 12), !sink};
    plat_mutex_unlock(&ft_lock);
}

static void ft_pool_put(PoolRec *r) {
    if (r->kind == POOL_IMPORT) {
        VkBuffer b = r->mstate == MS_ON ? r->mbuf : r->buf;
        if (b) ft_run(r->import_lo, r->import_hi, gmem_device(b), 0);
    } else if (r->kind == POOL_PHYS) {
        if (r->buf) ft_run(r->lo, r->lo + r->size, gmem_device(r->buf), 0);
    } else if (r->kind == POOL_VIRT) {
        for (uint64_t k = 0; k < r->npages;) {
            PoolRec *ph = r->pt[k].phys;
            uint64_t j = k + 1, off = r->pt[k].off, lo = r->lo + k * NVN_VPOOL_PAGE;
            while (j < r->npages && r->pt[j].phys == ph && (!ph || r->pt[j].off == off + (j - k) * NVN_VPOOL_PAGE)) j++;
            if (ph && ph->buf) ft_run(lo, r->lo + j * NVN_VPOOL_PAGE, gmem_device(ph->buf) + off, 0);
            else ft_run(lo, r->lo + j * NVN_VPOOL_PAGE, 0, 1);
            k = j;
        }
    }
}

static void ft_pool_gone(const PoolRec *r) {
    uint64_t lo = r->kind == POOL_IMPORT ? r->import_lo : r->lo, hi = r->kind == POOL_IMPORT ? r->import_hi : r->lo + r->size;
    ft_run(lo, hi, 0, 1);
    for (int i = 0; i < pool_index.n; i++) {
        const pidx_ent *e = &pool_index.e[i];
        PoolRec *o = e->buf;
        if (o && o != r && ((e->lo < hi && e->hi > lo) || (r->kind == POOL_PHYS && o->kind == POOL_VIRT))) ft_pool_put(o);
    }
}

void vk_mem_flat_setup(void) {
    VkCtx *c = vk_ctx();
    if (vk_mem_flat_mode() != 2) return;
    if (!c->has_buffer_device_address || shadow_on) rt_fatal("mem: the guest page table needs device addresses and no shadow storage");
    plat_mutex_init(&ft_lock);
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = FT_BYTES * 2, .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT};
    VmaAllocationCreateInfo ai = {.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
    // Twice the size: a dedicated allocation ignores the alignment asked for, and the table starts at the aligned point inside.
    if (vmaCreateBuffer(c->vma, &bi, &ai, &ft_buf, &ft_alloc, NULL) != VK_SUCCESS) rt_fatal("mem: no memory for the guest page table");
    bi.size = 4096;
    if (vmaCreateBufferWithAlignment(c->vma, &bi, &ai, 4096, &ft_sink_buf, &ft_sink_alloc, NULL) != VK_SUCCESS) rt_fatal("mem: no memory for the sink page");
    uint64_t start = gmem_device(ft_buf);
    ft_addr = (start + FT_BYTES - 1) & ~(FT_BYTES - 1);
    ft_off = ft_addr - start;
    uint64_t sink = gmem_device(ft_sink_buf);
    if ((sink & 0xfff) || (sink >> 44)) rt_fatal("mem: sink page at %#llx", (unsigned long long)sink);
    ft_sink_page = (uint32_t)(sink >> 12);
    ft_on = 1;
    rt_log("mem: guest page table: %u entries at %#llx, sink page %#x", FT_ENTRIES, (unsigned long long)ft_addr, ft_sink_page);
}

// Before anything the part recorded, in the prologue's command buffer. Entries go inline: no staging buffer to keep alive.
static void ft_flush(VkCommandBuffer cb) {
    static uint32_t chunk[16384];
    plat_mutex_lock(&ft_lock);
    if (!ft_filled) vkCmdFillBuffer(cb, ft_sink_buf, 0, VK_WHOLE_SIZE, 0), vkCmdFillBuffer(cb, ft_buf, ft_off, FT_BYTES, ft_sink_page), ft_filled = 1;
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
    for (int i = 0; i < ft_n; i++) {
        const FtRun *f = &ft_runs[i];
        ft_entries_sent += f->n;
        if (!f->step) { vkCmdFillBuffer(cb, ft_buf, ft_off + (uint64_t)f->index * 4, (uint64_t)f->n * 4, f->page); continue; }
        for (uint32_t k = 0; k < f->n;) {
            uint32_t m = f->n - k < 16384 ? f->n - k : 16384;
            for (uint32_t x = 0; x < m; x++) chunk[x] = f->page + k + x;
            vkCmdUpdateBuffer(cb, ft_buf, ft_off + (uint64_t)(f->index + k) * 4, (uint64_t)m * 4, chunk);
            k += m;
        }
    }
    ft_n = 0;
    plat_mutex_unlock(&ft_lock);
    vkCmdPipelineBarrier2(cb, &di);
}

uint64_t vk_mem_gen(void) {
    return __atomic_load_n(&pool_gen, __ATOMIC_ACQUIRE) + __atomic_load_n(&mirror_gen, __ATOMIC_ACQUIRE);
}

int vk_mem_class_of(uint64_t addr) {
    VkMemSpan m;
    return vk_mem_resolve_existing(addr, 1, &m) ? m.cls : VK_MCLS_NONE;
}

// ---------------------------------------------------------------- census
static volatile int64_t traffic[VK_MCLS_N][VK_MUSE_N];

// Every caller is the replay (vk_exec*, under vk_sync_submit's lock), about ten adds per draw: one writer at a time, so
// a plain add, not a locked one. An add racing the census's reset may be lost.
void vk_mem_count(int cls, int use, uint64_t bytes) {
    if ((unsigned)cls < VK_MCLS_N && (unsigned)use < VK_MUSE_N)
        __atomic_store_n(&traffic[cls][use], __atomic_load_n(&traffic[cls][use], __ATOMIC_RELAXED) + (int64_t)bytes,
                         __ATOMIC_RELAXED);
}

static void flag_names(uint32_t f, char *buf, size_t n) {
    static const char *const names[] = {"CPU_NO_ACCESS", "CPU_UNCACHED", "CPU_CACHED", "GPU_NO_ACCESS", "GPU_UNCACHED",
                                        "GPU_CACHED", "SHADER_CODE", "COMPRESSIBLE", "PHYSICAL", "VIRTUAL"};
    size_t k = 0;
    buf[0] = 0;
    for (int i = 0; i < 10; i++)
        if (f >> i & 1) k += (size_t)snprintf(buf + k, k < n ? n - k : 0, "%s%s", k ? "|" : "", names[i]);
    if (!k) snprintf(buf, n, "none");
}

// One line per (kind, flags) group of live pools, then the bytes the replay bound or copied per pool class since the
// previous census.

static void mirror_log(uint64_t frame);
static void shadow_log(uint64_t frame);

static void vram_log(uint64_t frame);

void vk_mem_census(uint64_t frame) {
    static uint64_t window_start;
    if ((rt_trace & RT_TRACE_PERF) && frame % 600 == 0) vram_log(frame);
    if ((rt_trace & RT_TRACE_PERF) ? frame % 60 == 0 : frame == 600 || frame == 1800)
        mirror_log(frame), shadow_log(frame);   // mirror_log also logs the write-watch line
    if (frame == 1200) {   // loading is over: the first window starts here
        for (int c = 0; c < VK_MCLS_N; c++)
            for (int u = 0; u < VK_MUSE_N; u++) plat_xchg64(&traffic[c][u], 0);
        window_start = frame;
        return;
    }
    // Frame 600 as well: runs on slow machines end before 1800.
    if (frame != 600 && (frame % 1800 || !(frame == 1800 || (rt_trace & RT_TRACE_PERF)))) return;
    typedef struct { int kind; uint32_t flags; uint64_t n, bytes; } Group;
    Group g[64];
    int ng = 0;
    uint64_t total[4] = {0}, count[4] = {0};
    plat_mutex_lock_shared(&pool_lock);
    for (int i = 0; i < pool_index.n; i++) {
        const PoolRec *r = pool_index.e[i].buf;
        int j = 0;
        while (j < ng && (g[j].kind != r->kind || g[j].flags != r->flags)) j++;
        if (j == ng && ng < 64) g[ng++] = (Group){r->kind, r->flags, 0, 0};
        if (j < ng) g[j].n++, g[j].bytes += r->size;
        total[r->kind] += r->size, count[r->kind]++;
    }
    plat_mutex_unlock_shared(&pool_lock);
    rt_log("mem: census at frame %llu: import %llu pools %.1f MiB (%d without a GPU buffer), physical %llu pools %.1f MiB, "
           "virtual %llu pools %.1f MiB of address space",
           (unsigned long long)frame, (unsigned long long)count[POOL_IMPORT], total[POOL_IMPORT] / 1048576.0,
           n_import_failures, (unsigned long long)count[POOL_PHYS], total[POOL_PHYS] / 1048576.0,
           (unsigned long long)count[POOL_VIRT], total[POOL_VIRT] / 1048576.0);
    wb_census();
    static const char *const kinds[] = {"", "import", "physical", "virtual"};
    for (int j = 0; j < ng; j++) {
        char f[160];
        flag_names(g[j].flags, f, sizeof f);
        rt_log("mem:   %s %#x %s: %llu pools, %.1f MiB", kinds[g[j].kind], g[j].flags, f, (unsigned long long)g[j].n,
               g[j].bytes / 1048576.0);
    }
    uint64_t frames = window_start && frame > window_start ? frame - window_start : frame;
    static const char *const cls[] = {"", "host CPU_NO_ACCESS", "host CPU_UNCACHED", "host CPU_CACHED", "device", "scratch"};
    rt_log("mem: traffic per frame over %llu frames, MiB (bound ranges, not bytes read): class: vertex index uniform "
           "storage indirect copy-src gpu-write cpu-read", (unsigned long long)frames);
    for (int c = 1; c < VK_MCLS_N; c++) {
        double v[VK_MUSE_N];
        for (int u = 0; u < VK_MUSE_N; u++) v[u] = plat_xchg64(&traffic[c][u], 0) / 1048576.0 / (double)(frames ? frames : 1);
        rt_log("mem:   %s: %.2f %.2f %.2f %.2f %.3f %.2f %.2f %.3f", cls[c], v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
    }
    window_start = frame;
}

int vk_mem_vram(uint64_t *used, uint64_t *budget, int *exact) {
    VkCtx *c = vk_ctx();
    *used = *budget = 0, *exact = 0;
    if (!c) return 0;
    const VkPhysicalDeviceMemoryProperties *mp = &c->mem_props;
    if (c->has_memory_budget) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT b = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &b};
        vkGetPhysicalDeviceMemoryProperties2(c->phys, &p2);
        for (uint32_t h = 0; h < p2.memoryProperties.memoryHeapCount && h < VK_MAX_MEMORY_HEAPS; h++)
            if (p2.memoryProperties.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                *used += b.heapUsage[h], *budget += b.heapBudget[h];
        *exact = 1;
        return *budget != 0;
    }
    VmaBudget vb[VK_MAX_MEMORY_HEAPS];
    vmaGetHeapBudgets(c->vma, vb);
    for (uint32_t h = 0; h < mp->memoryHeapCount && h < VK_MAX_MEMORY_HEAPS; h++)
        if (mp->memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            *used += vb[h].statistics.blockBytes, *budget += mp->memoryHeaps[h].size;
    return *budget != 0;
}

// ---------------------------------------------------------------- page write generations
// Two-level table over the guest address space (below 2^39): 256 MiB chunks of 65,536 page generations, allocated
// when a page of the chunk is first stamped. 0 = never stamped.
#define GEN_CHUNK_SHIFT 28
#define GEN_PAGE_SHIFT 12
#define GEN_CHUNKS (RT_ADDRESS_LIMIT >> GEN_CHUNK_SHIFT)
#define GEN_PER_CHUNK (1u << (GEN_CHUNK_SHIFT - GEN_PAGE_SHIFT))
static uint32_t *gen_chunk[GEN_CHUNKS];
static uint32_t gen_now;                 // last generation handed out
static PlatMutex gen_lock = PLAT_MUTEX_INIT;
static uint64_t gen_pages[1024];
// The newest generation stamped on any page of each 2 MiB region (the write-watch snapshot regions below): the mirror
// sync skips regions with nothing newer than its pool's mirror.
#define REG_SHIFT 21
static uint32_t reg_max[RT_ADDRESS_LIMIT >> REG_SHIFT];

// Write-watch query statistics (gen_lock held exclusively): calls, bytes and time since the last mirror summary, and
// the queries that found written pages (they also flush TLBs) apart.
static uint64_t ww_calls, ww_bytes, ww_ticks, ww_dirty_calls, ww_dirty_ticks, ww_pages;

static uint64_t ww_query(uint64_t addr, uint64_t size) {
    uint64_t t0 = qpc();
    uint64_t n = mem_written_pages(addr, size, gen_pages, sizeof gen_pages / sizeof *gen_pages);
    uint64_t dt = qpc() - t0;
    ww_ticks += dt, ww_calls++, ww_bytes += size;
    if (n && n != UINT64_MAX) ww_dirty_calls++, ww_dirty_ticks += dt, ww_pages += n;
    return n;
}

// Frozen regions: a 2 MiB region that nothing wrote for freeze_idle submissions is made read-only (mem_freeze), and
// write-watch queries skip it until a write or a file read thaws it. A region may hold non-pool memory too: that is
// safe, since the only kernel writes into guest memory are file reads, which thaw their buffer first.
// See README, "Write watching and page generations".
#define FREEZE_PER_SUB 32
static const int freeze_idle = 60;
static uint32_t ww_ep = 1;                    // gen_lock: the submission being recorded (vk_mem_mirror_begin)
static uint32_t reg_last[RT_ADDRESS_LIMIT >> REG_SHIFT];    // gen_lock: ww_ep of the region's last stamp
static uint32_t reg_tried[RT_ADDRESS_LIMIT >> REG_SHIFT];   // gen_lock: ww_ep of its last freeze candidacy
static uint32_t *fcand;                       // gen_lock: regions to try at the next freeze pass
static int nfcand, cfcand;
static uint64_t n_frz_refused;                // gen_lock: candidates mem_freeze refused

uint32_t *gen_slot(uint64_t page_addr, int create) {
    uint64_t ci = page_addr >> GEN_CHUNK_SHIFT;
    if (ci >= GEN_CHUNKS) return NULL;
    if (!gen_chunk[ci]) {
        if (!create) return NULL;
        gen_chunk[ci] = calloc(GEN_PER_CHUNK, sizeof(uint32_t));
        if (!gen_chunk[ci]) rt_fatal("vk: out of memory for page generations");
    }
    return &gen_chunk[ci][(page_addr >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)];
}

// gen_lock held exclusively; g is the newest generation handed out.
static void stamp_page(uint64_t a, uint32_t g) {
    uint32_t *s = gen_slot(a, 1);
    if (!s) return;
    *s = g;
    reg_max[a >> REG_SHIFT] = g;
    reg_last[a >> REG_SHIFT] = ww_ep;
}

static void stamp_range(uint64_t lo, uint64_t hi, uint32_t g) {
    for (uint64_t a = lo; a < hi; a += 1ull << GEN_PAGE_SHIFT) stamp_page(a, g);
}

void vk_guest_written(uint64_t addr, uint64_t size) {
    if (!size || nvn_gpu_only(addr)) return;
    uint64_t lo = addr & ~0xfffull, hi = (addr + size + 0xfff) & ~0xfffull;
    plat_mutex_lock(&gen_lock);
    stamp_range(lo, hi, ++gen_now);
    plat_mutex_unlock(&gen_lock);
}

// ---- write-back echo
// A page that the write-back copied whole is marked as an echo, and mirror_sync skips a page whose newest stamp is its
// echo, so the write-back is not uploaded again. Only when nothing newer than the submission's recording was stamped
// there; a later CPU write keeps the plain stamp and is uploaded. See README, "Write watching and page generations".
#define ECHO_JOBS 64
#define ECHO_SPANS 4096
typedef struct { uint64_t lo, hi; } EchoSpan;           // page-aligned guest range the write-back copied whole
typedef struct { uint64_t fence; uint32_t g; int n, cap; EchoSpan *s; } EchoJob;
static EchoJob echo_jobs[ECHO_JOBS];                    // echo_lock: by fence % ECHO_JOBS
static PlatMutex echo_lock = PLAT_MUTEX_INIT;           // taken before gen_lock
static uint32_t *echo_chunk[GEN_CHUNKS];                // gen_lock: the stamp a page got as an echo, 0 none
static volatile int64_t echo_pages, echo_skipped;       // pages marked, page uploads skipped since the last mirror line
static volatile int64_t echo_rule_marks, echo_rule_skips;   // the same on pages that hold a range kept out by the rule

// Pages that hold a range the write-back rule keeps out of write-back (wb_done sets them; completion thread writes,
// any thread reads). Open addressing on the page number; full: the page is not added (diagnostic counts miss it).
#define WBSP_N 8192
static volatile uint64_t wbsp[WBSP_N];
void wbsp_add(uint64_t page) {
    unsigned at = (unsigned)((page >> 12) * 0x9E3779B1u) & (WBSP_N - 1);
    for (int k = 0; k < 64; k++, at = (at + 1) & (WBSP_N - 1)) {
        uint64_t v = __atomic_load_n(&wbsp[at], __ATOMIC_ACQUIRE);
        if (v == page) return;
        if (!v) { __atomic_store_n(&wbsp[at], page, __ATOMIC_RELEASE); return; }
    }
}
// The page whose echo marks, echo skips and uploads are logged.
static uint64_t echo_watch(void) {
    return 0;
}
static uint32_t echo_frame(void);

static uint32_t *echo_slot(uint64_t page_addr, int create) {
    uint64_t ci = page_addr >> GEN_CHUNK_SHIFT;
    if (ci >= GEN_CHUNKS) return NULL;
    if (!echo_chunk[ci]) {
        if (!create) return NULL;
        echo_chunk[ci] = calloc(GEN_PER_CHUNK, sizeof(uint32_t));
        if (!echo_chunk[ci]) return NULL;
    }
    return &echo_chunk[ci][(page_addr >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)];
}

// Replay thread, vk_mem_mirror_end: the spans the write-back of `fence` copies, before it is submitted.
static void echo_job(uint64_t fence, const EchoSpan *s, int n) {
    plat_mutex_lock(&gen_lock);
    uint32_t g = gen_now;   // every stamp up to here belongs to a write before the copy
    plat_mutex_unlock(&gen_lock);
    plat_mutex_lock(&echo_lock);
    EchoJob *j = &echo_jobs[fence % ECHO_JOBS];
    j->fence = 0, j->n = 0;   // a job that cannot be kept marks nothing: plain stamps
    if (n > j->cap && n <= ECHO_SPANS) {
        EchoSpan *ns = realloc(j->s, sizeof *ns * (size_t)n);
        if (ns) j->s = ns, j->cap = n;
    }
    if (n <= j->cap) {
        memcpy(j->s, s, sizeof *s * (size_t)n);
        j->n = n, j->g = g, j->fence = fence;
    }
    plat_mutex_unlock(&echo_lock);
}

void vk_gpu_written_at(uint64_t addr, uint64_t size, uint64_t fence) {
    if (shadow_on || !size || nvn_gpu_only(addr)) {
        vk_gpu_written(addr, size);
        return;
    }
    uint64_t lo = addr & ~0xfffull, hi = (addr + size + 0xfff) & ~0xfffull;
    EchoSpan hit[8];
    int nh = 0;
    uint32_t g0 = 0;
    plat_mutex_lock(&echo_lock);
    const EchoJob *j = &echo_jobs[fence % ECHO_JOBS];
    if (j->fence == fence) {
        g0 = j->g;
        for (int i = 0; i < j->n && nh < 8; i++)
            if (j->s[i].lo < hi && j->s[i].hi > lo) hit[nh++] = j->s[i];
    }
    plat_mutex_lock(&gen_lock);
    uint32_t g = ++gen_now;
    int64_t marked = 0;
    for (uint64_t a = lo; a < hi; a += 1ull << GEN_PAGE_SHIFT) {
        uint32_t *s = gen_slot(a, 1);
        if (!s) continue;
        uint32_t prior = *s;
        int in = 0;
        for (int k = 0; k < nh && !in; k++) in = a >= hit[k].lo && a < hit[k].hi;
        uint32_t *e = in ? echo_slot(a, 1) : NULL;
        stamp_page(a, g);
        // prior <= g0: no write seen since the recording. prior == *e: the page was an echo already (an overlapping
        // range of this fence, or an earlier write-back), so nothing wrote it since either.
        int rule = 0;
        if (e && (prior <= g0 || prior == *e)) *e = g, marked++;
        if (a == echo_watch())
            rt_log("echo watch %#llx: fence %llu frame %u: stamp %u -> %u, recorded at %u, echo %s, rule page %d",
                   (unsigned long long)a, (unsigned long long)fence, echo_frame(), prior, g, g0,
                   !in ? "no (not covered whole)" : e && *e == g ? "marked" : "not marked", rule);
    }
    plat_mutex_unlock(&gen_lock);
    plat_mutex_unlock(&echo_lock);
    if (marked) plat_add64(&echo_pages, marked);
}

int vk_mem_changed(uint64_t addr, uint64_t size, uint32_t since, uint32_t *now) {
    if (!size) { *now = since; return 0; }
    uint64_t lo = addr & ~0xfffull, hi = (addr + size + 0xfff) & ~0xfffull;
    plat_mutex_lock(&gen_lock);
    uint32_t g = ++gen_now;
    int tracked = 1, frozen = 1;
    for (uint64_t a = lo & ~((1ull << REG_SHIFT) - 1); a < hi && frozen; a += 1ull << REG_SHIFT) frozen = mem_frozen(a);
    while (!frozen) {
        uint64_t n = ww_query(lo, hi - lo);
        if (n == UINT64_MAX) { tracked = 0; break; }
        for (uint64_t k = 0; k < n; k++) stamp_page(gen_pages[k], g);
        if (n < sizeof gen_pages / sizeof *gen_pages) break;   // a full answer may have left pages for the next call
    }
    int changed = 0;
    for (uint64_t a = lo; tracked && a < hi && !changed; a += 1ull << GEN_PAGE_SHIFT) {
        uint32_t *s = gen_slot(a, 0);
        if (s && *s > since) changed = 1;
    }
    plat_mutex_unlock(&gen_lock);
    *now = g;
    return tracked ? changed : -1;
}

// Write-watch snapshots per submission: the guest address space in 2 MiB regions, each queried at most once per epoch
// (a guest submission), at the first check that covers it. A snapshot only adds observations to the generation table.
#define SNAP_SHIFT 21
#define SNAP_REGIONS (RT_ADDRESS_LIMIT >> SNAP_SHIFT)
static uint32_t snap_at[SNAP_REGIONS];      // epoch of the region's last snapshot; 0 never
static uint8_t snap_untracked[SNAP_REGIONS];
static uint64_t n_snap_checks, n_snap_calls, n_snap_untracked;

// ---- parallel pre-snap (3 workers): in a batch, snap_regions only collects its runs (pieces of at most WWB_PIECE
// regions, marked seen at once); ww_batch_run queries the pieces on the workers and this thread together (the kernel
// walk takes the mmap lock shared) and applies them in order with gen_lock still held.
#define WWB_PIECE 4   // regions (8 MiB) per piece
typedef struct { uint64_t r, k; uint32_t g; uint64_t at, n; int tracked; uint64_t calls, bytes, ticks, dcalls, dticks; } WwPiece;
static struct {
    int on, workers;
    WwPiece *p; int np, cp;
    uint64_t *pages; uint64_t npages, cpages;   // each piece's slot: at most its size in pages
    uint64_t batch;                             // high 32 bits of next and total
    uint64_t next, total, done;                 // atomic
    pthread_mutex_t m; pthread_cond_t cv; uint64_t posted;
    uint64_t batches, pieces, mine;             // stats: batches, pieces, pieces this thread ran
} wwb = {.m = PTHREAD_MUTEX_INITIALIZER, .cv = PTHREAD_COND_INITIALIZER};

static void ww_batch_add(uint64_t r, uint64_t k, uint32_t g, uint32_t epoch) {
    for (uint64_t s = r; s <= k; s += WWB_PIECE) {
        uint64_t e = s + WWB_PIECE - 1 <= k ? s + WWB_PIECE - 1 : k;
        if (wwb.np == wwb.cp) grow((void **)&wwb.p, &wwb.cp, sizeof *wwb.p);
        uint64_t slot = (e - s + 1) << (SNAP_SHIFT - 12);
        wwb.p[wwb.np++] = (WwPiece){.r = s, .k = e, .g = g, .at = wwb.npages, .tracked = 1};
        wwb.npages += slot;
        for (uint64_t j = s; j <= e; j++) snap_at[j] = epoch;   // seen: later runs of this batch skip it
    }
}

static void ww_piece_run(WwPiece *q) {
    uint64_t a = q->r << SNAP_SHIFT, b = (q->k + 1) << SNAP_SHIFT, cap = (q->k - q->r + 1) << (SNAP_SHIFT - 12);
    uint64_t t0 = qpc();
    uint64_t n = mem_written_pages(a, b - a, wwb.pages + q->at, cap);
    uint64_t dt = qpc() - t0;
    q->calls = 1, q->bytes = b - a, q->ticks = dt;
    if (n == UINT64_MAX) q->tracked = 0, q->n = 0;
    else {
        q->n = n;
        if (n) q->dcalls = 1, q->dticks = dt;
    }
}

static int ww_grab(void) {   // the next piece of the current batch, or -1
    uint64_t v = __atomic_fetch_add(&wwb.next, 1, __ATOMIC_ACQ_REL);
    uint64_t t = __atomic_load_n(&wwb.total, __ATOMIC_ACQUIRE);
    if ((v >> 32) != (t >> 32) || (uint32_t)v >= (uint32_t)t) return -1;
    return (int)(uint32_t)v;
}

static void *ww_worker(void *arg) {
    (void)arg;
    uint64_t seen = 0;
    for (;;) {
        pthread_mutex_lock(&wwb.m);
        while (wwb.posted == seen) pthread_cond_wait(&wwb.cv, &wwb.m);
        seen = wwb.posted;
        pthread_mutex_unlock(&wwb.m);
        for (int i; (i = ww_grab()) >= 0;) ww_piece_run(&wwb.p[i]), __atomic_add_fetch(&wwb.done, 1, __ATOMIC_RELEASE);
    }
    return NULL;
}

static int ww_batch_workers(void) {
    static int n = -1;
    if (n < 0) {
        n = 3;
        for (int i = 0; i < n; i++) {
            pthread_t t;
            if (pthread_create(&t, NULL, ww_worker, NULL)) { n = i; break; }
            pthread_detach(t);
        }
        if (n) rt_log("mem: the pre-snap write-watch queries run on %d workers and the replay thread", n);
    }
    return n;
}

static void ww_batch_begin(void) {
    wwb.on = ww_batch_workers() > 0;
    wwb.np = 0, wwb.npages = 0;
}

// Runs the batch's pieces and applies them (gen_lock held exclusively, as for snap_regions).
static void ww_batch_run(void) {
    if (!wwb.on) return;
    wwb.on = 0;
    if (!wwb.np) return;
    if (wwb.npages > wwb.cpages) {
        free(wwb.pages);
        wwb.cpages = wwb.npages + (wwb.npages >> 1);
        wwb.pages = malloc(wwb.cpages * sizeof *wwb.pages);
        if (!wwb.pages) rt_fatal("mem: out of memory (write-watch batch)");
    }
    uint64_t batch = ++wwb.batch << 32;
    __atomic_store_n(&wwb.done, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&wwb.total, batch | (uint32_t)wwb.np, __ATOMIC_RELEASE);
    __atomic_store_n(&wwb.next, batch, __ATOMIC_RELEASE);
    if (wwb.np > 1) {
        pthread_mutex_lock(&wwb.m);
        wwb.posted++;
        pthread_cond_broadcast(&wwb.cv);
        pthread_mutex_unlock(&wwb.m);
    }
    uint64_t mine = 0;
    for (int i; (i = ww_grab()) >= 0;) ww_piece_run(&wwb.p[i]), __atomic_add_fetch(&wwb.done, 1, __ATOMIC_RELEASE), mine++;
    while (__atomic_load_n(&wwb.done, __ATOMIC_ACQUIRE) < (uint64_t)wwb.np) __builtin_ia32_pause();
    wwb.batches++, wwb.pieces += (uint64_t)wwb.np, wwb.mine += mine;
    for (int i = 0; i < wwb.np; i++) {
        WwPiece *q = &wwb.p[i];
        ww_calls += q->calls, ww_bytes += q->bytes, ww_ticks += q->ticks, ww_dirty_calls += q->dcalls, ww_dirty_ticks += q->dticks;
        ww_pages += q->n;
        for (uint64_t j = 0; j < q->n; j++) stamp_page(wwb.pages[q->at + j], q->g);
        for (uint64_t j = q->r; j <= q->k; j++) {
            snap_untracked[j] = (uint8_t)!q->tracked;
            if (q->tracked && ww_ep - reg_last[j] >= (uint32_t)freeze_idle &&
                ww_ep - reg_tried[j] >= (uint32_t)freeze_idle) {
                reg_tried[j] = ww_ep;
                if (nfcand == cfcand) grow((void **)&fcand, &cfcand, sizeof *fcand);
                fcand[nfcand++] = (uint32_t)j;
            }
        }
    }
    if (wwb.batches % 3600 == 0)
        rt_log("mem: pre-snap workers: %llu batches, %.1f pieces each, %.0f%% of them on the replay thread",
               (unsigned long long)wwb.batches, (double)wwb.pieces / (double)wwb.batches, 100.0 * (double)wwb.mine / (double)wwb.pieces);
}

// gen_lock held exclusively. Queries the regions r0..r1 not snapped in epoch, one query per run; returns the queries.
static uint64_t snap_regions(uint64_t r0, uint64_t r1, uint32_t epoch) {
    uint32_t g = 0;
    uint64_t calls = 0;
    for (uint64_t r = r0; r <= r1;) {
        if (snap_at[r] == epoch) { r++; continue; }
        if (mem_frozen(r << SNAP_SHIFT)) {   // nothing in it changed since its freeze
            snap_at[r] = epoch, snap_untracked[r] = 0;
            r++;
            continue;
        }
        uint64_t k = r;   // one query over a run of regions not seen in this epoch
        while (k + 1 <= r1 && snap_at[k + 1] != epoch && !mem_frozen((k + 1) << SNAP_SHIFT)) k++;
        uint64_t a = r << SNAP_SHIFT, b = (k + 1) << SNAP_SHIFT;
        int tracked = 1;
        if (!g) g = ++gen_now;
        if (wwb.on) {   // the pre-snap batch (ww_batch_run): queried later on several threads
            ww_batch_add(r, k, g, epoch);
            calls++;
            r = k + 1;
            continue;
        }
        for (;;) {
            uint64_t n = ww_query(a, b - a);
            if (n == UINT64_MAX) { tracked = 0; break; }
            for (uint64_t j = 0; j < n; j++) stamp_page(gen_pages[j], g);
            if (n < sizeof gen_pages / sizeof *gen_pages) break;
        }
        calls++;
        for (uint64_t j = r; j <= k; j++) {
            snap_at[j] = epoch, snap_untracked[j] = (uint8_t)!tracked;
            if (tracked && ww_ep - reg_last[j] >= (uint32_t)freeze_idle &&
                ww_ep - reg_tried[j] >= (uint32_t)freeze_idle) {
                reg_tried[j] = ww_ep;
                if (nfcand == cfcand) grow((void **)&fcand, &cfcand, sizeof *fcand);
                fcand[nfcand++] = (uint32_t)j;
            }
        }
        r = k + 1;
    }
    return calls;
}

int vk_mem_changed_snap(uint64_t addr, uint64_t size, uint32_t since, uint32_t epoch, uint32_t *now) {
    if (!size) { *now = since; return 0; }
    uint64_t lo = addr & ~0xfffull, hi = (addr + size + 0xfff) & ~0xfffull;
    uint64_t r0 = lo >> SNAP_SHIFT, r1 = (hi - 1) >> SNAP_SHIFT;
    if (!epoch || r1 >= SNAP_REGIONS) return vk_mem_changed(addr, size, since, now);
    int excl = 0;
    plat_mutex_lock_shared(&gen_lock);
    for (uint64_t r = r0; r <= r1; r++)
        if (snap_at[r] != epoch) { excl = 1; break; }
    if (excl) {
        plat_mutex_unlock_shared(&gen_lock);
        plat_mutex_lock(&gen_lock);
        n_snap_calls += snap_regions(r0, r1, epoch);
    }
    int untracked = 0, changed = 0;
    for (uint64_t r = r0; r <= r1; r++) untracked |= snap_untracked[r];
    // Pages only in regions stamped after since (reg_max; REG_SHIFT is SNAP_SHIFT): an unchanged texture then costs no
    // page reads.
    for (uint64_t r = r0; !untracked && r <= r1 && !changed; r++) {
        if (reg_max[r] <= since) continue;
        uint64_t a = r << REG_SHIFT > lo ? r << REG_SHIFT : lo, b = (r + 1) << REG_SHIFT < hi ? (r + 1) << REG_SHIFT : hi;
        for (; a < b && !changed; a += 1ull << GEN_PAGE_SHIFT) {
            uint32_t *s = gen_slot(a, 0);
            if (s && *s > since) changed = 1;
        }
    }
    *now = gen_now;   // every observation so far is in the table
    __atomic_add_fetch(&n_snap_checks, 1, __ATOMIC_RELAXED);
    if (untracked) __atomic_add_fetch(&n_snap_untracked, 1, __ATOMIC_RELAXED);
    if (excl) plat_mutex_unlock(&gen_lock);
    else plat_mutex_unlock_shared(&gen_lock);
    return untracked ? -1 : changed;
}

void vk_mem_snap_stats(uint64_t *checks, uint64_t *calls, uint64_t *untracked) {
    *checks = __atomic_exchange_n(&n_snap_checks, 0, __ATOMIC_RELAXED);
    *calls = __atomic_exchange_n(&n_snap_calls, 0, __ATOMIC_RELAXED);
    *untracked = __atomic_exchange_n(&n_snap_untracked, 0, __ATOMIC_RELAXED);
}

// OOT_TRACE=perf: the replay notes each imported pool it binds as a buffer (VK_MEM_BOUND), for the frame trace.
void vk_mem_bound_note(void *pool) { ((PoolRec *)pool)->bufbound = 1; }

// gen_lock held exclusively. 0: the range is not write-watched.
static int query_stamp(uint64_t lo, uint64_t hi, uint32_t g) {
    for (;;) {
        uint64_t n = ww_query(lo, hi - lo);
        if (n == UINT64_MAX) return 0;
        for (uint64_t k = 0; k < n; k++) stamp_page(gen_pages[k], g);
        if (n < sizeof gen_pages / sizeof *gen_pages) return 1;
    }
}

// ---------------------------------------------------------------- shadow storage
// Where importing guest memory costs too much, a CPU-visible pool gets a host-visible buffer of its own at the offsets
// the import would have. On for Linux except NVIDIA (mirror_mode says why). See README, "Shadow storage".
// CPU -> shadow: vk_mem_shadow_sync, right before each submit, copies the pages stamped after a pool's sgen.
// GPU -> guest: GPU writes are noted at record time (vk_mem_gpu_pending) and copied back at the fence
// (vk_gpu_written); until then the CPU copies skip them.
typedef struct { uint64_t lo, hi; } SRange;
static PlatMutex shadow_lock = PLAT_MUTEX_INIT;   // after pool_lock; before gen_lock
static PoolRec **stouch;                          // shadow_lock: pools resolved since the last sync
static int nstouch, cstouch;
static SRange *spend;                             // shadow_lock: GPU writes recorded and not copied back yet
static int nspend, cspend;
static int64_t s_live_n, s_live_bytes;            // shadow_lock
static uint64_t s_syncs, s_copied, s_queries, s_untracked, s_wb_bytes, s_ticks;   // shadow_lock: since the last log

// Deferred: a pool gets its shadow at its first vk_mem_resolve (most CPU-visible pools only hold textures), under
// pool_lock shared. r->buf is stored last, so a reader that sees it sees the rest.
static PlatMutex lazy_lock = PLAT_MUTEX_INIT;   // after pool_lock; before shadow_lock and gen_lock
static int shadow_make(VkCtx *c, PoolRec *r);

// ---------------------------------------------------------------- windowed shadows
// A pool of 256 MiB or more gets shadows of windows instead, made on demand: window k covers
// [import_lo + k*WIN_STEP, + 2*WIN_STEP), so every lookup of an address in step k picks window k. A window has no
// index entry, so it gets no VRAM mirror: the GPU reads its host-visible shadow. See README, "Shadow storage".
static uint64_t window_min(void) {
    return 256ull << 20;
}

static int shadow_pool(VkCtx *c, PoolRec *r, uint64_t addr, uint64_t size) {
    r->import_lo = addr & ~0xfffull;
    r->import_hi = (addr + size + 0xfff) & ~0xfffull;
    (void)c;
    uint64_t n = r->import_hi - r->import_lo, min = window_min();
    if (n >= min) {
        r->nwin = (uint32_t)((n + WIN_STEP - 1) >> WIN_SHIFT);
        r->win = calloc(r->nwin, sizeof *r->win);
        if (!r->win) r->nwin = 0;
    }
    r->slazy = 1;
    return 1;
}

static void shadow_late(PoolRec *r) {
    plat_mutex_lock(&lazy_lock);
    if (!r->buf && r->slazy && !shadow_make(vk_ctx(), r)) r->slazy = 0;   // failed: stays unbacked
    plat_mutex_unlock(&lazy_lock);
}

// pool_lock held (shared at least): the window of a windowed pool that addr's lookups use; made when create is set.
static PoolRec *window_get(PoolRec *r, uint64_t addr, int create) {
    if (addr < r->import_lo) return NULL;
    uint64_t k = (addr - r->import_lo) >> WIN_SHIFT;
    if (k >= r->nwin) return NULL;
    PoolRec *w = __atomic_load_n(&r->win[k], __ATOMIC_ACQUIRE);
    if (w || !create) return w;
    plat_mutex_lock(&lazy_lock);
    w = r->win[k];
    if (!w && (w = calloc(1, sizeof *w))) {
        w->key = r->key;
        w->storage_serial = r->storage_serial;
        w->kind = POOL_IMPORT;
        w->flags = r->flags;
        w->import_lo = r->import_lo + (k << WIN_SHIFT);
        w->import_hi = w->import_lo + 2 * WIN_STEP < r->import_hi ? w->import_lo + 2 * WIN_STEP : r->import_hi;
        w->lo = w->import_lo, w->size = w->import_hi - w->import_lo;
        if (shadow_make(vk_ctx(), w)) {
            static volatile int32_t lines;
            if (plat_inc32(&lines) <= 8)
                rt_log("mem: pool %#llx+%#llx: window %llu shadowed, %#llx+%#llx (the pool is %u windows)", (unsigned long long)r->lo, (unsigned long long)r->size,
                       (unsigned long long)k, (unsigned long long)w->import_lo, (unsigned long long)w->size, r->nwin);
            __atomic_store_n(&r->win[k], w, __ATOMIC_RELEASE);
        } else {
            free(w);
            w = NULL;
        }
    }
    plat_mutex_unlock(&lazy_lock);
    return w;
}

static int shadow_make(VkCtx *c, PoolRec *r) {
    uint64_t n = r->import_hi - r->import_lo;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = n, .usage = pool_usage(c),
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    // Cached where the device has it: the fence copies read it back on the CPU.
    VmaAllocationCreateInfo ai = {.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                  .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                  .preferredFlags = VK_MEMORY_PROPERTY_HOST_CACHED_BIT};
    VmaAllocationInfo info;
    VkBuffer buf = VK_NULL_HANDLE;
    VkResult res = vmaCreateBuffer(c->vma, &bi, &ai, &buf, &r->salloc, &info);
    if (res != VK_SUCCESS) {
        rt_log("vk: pool %#llx+%#llx: shadow allocation failed (%d)", (unsigned long long)r->lo,
               (unsigned long long)r->size, (int)res);
        r->salloc = NULL;
        return 0;
    }
    r->smap = info.pMappedData;
    plat_mutex_lock(&gen_lock);
    r->sgen = gen_now;   // the copy below holds every write stamped so far; later stamps are copied again
    plat_mutex_unlock(&gen_lock);
    memcpy(r->smap, (const void *)(uintptr_t)r->import_lo, n);
    __atomic_store_n(&r->buf, buf, __ATOMIC_RELEASE);
    plat_mutex_lock(&shadow_lock);
    s_live_n++, s_live_bytes += (int64_t)n;
    plat_mutex_unlock(&shadow_lock);
    plat_inc32(&n_imports);
    return 1;
}

static void shadow_touch(PoolRec *r) {
    plat_mutex_lock(&shadow_lock);
    if (!r->stouched) {
        if (nstouch == cstouch) grow((void **)&stouch, &cstouch, sizeof *stouch);
        stouch[nstouch++] = r;
        r->stouched = 1;
    }
    plat_mutex_unlock(&shadow_lock);
}

// vk_pool_fini, after the pool left the index: off the touched list before its record goes.
static void shadow_forget(PoolRec *r) {
    plat_mutex_lock(&shadow_lock);
    for (int i = 0; i < nstouch; i++)
        if (stouch[i] == r) { stouch[i] = stouch[--nstouch]; break; }
    s_live_n--, s_live_bytes -= (int64_t)(r->import_hi - r->import_lo);
    plat_mutex_unlock(&shadow_lock);
}

// shadow_lock held: copies guest [a, b) into the shadow, leaving out the bytes of GPU writes not copied back yet
// (spend from index i on).
static void shadow_copy(PoolRec *r, uint64_t a, uint64_t b, int i) {
    for (; i < nspend; i++)
        if (spend[i].lo < b && spend[i].hi > a) {
            if (spend[i].lo > a) shadow_copy(r, a, spend[i].lo, i + 1);
            if (spend[i].hi < b) shadow_copy(r, spend[i].hi, b, i + 1);
            return;
        }
    memcpy(r->smap + (a - r->import_lo), (const void *)(uintptr_t)a, b - a);
    s_copied += b - a;
}

// shadow_lock and gen_lock held: stamps the pool's written pages and copies those newer than its shadow.
static void shadow_sync_pool(PoolRec *r) {
    uint64_t lo = r->import_lo, hi = r->import_hi;
    uint32_t g = ++gen_now;
    int tracked = 1;
    for (uint64_t a = lo; a < hi;) {   // one query per run of regions that are not frozen
        uint64_t b = (a | ((1ull << REG_SHIFT) - 1)) + 1;
        if (b > hi) b = hi;
        if (mem_frozen(a)) { a = b; continue; }
        while (b < hi && !mem_frozen(b)) b = b + (1ull << REG_SHIFT) < hi ? b + (1ull << REG_SHIFT) : hi;
        s_queries++;
        if (!query_stamp(a, b, g)) tracked = 0;
        a = b;
    }
    uint32_t since = r->sgen;
    r->sgen = gen_now;
    if (!tracked) {   // nothing tells which pages changed: all of them
        s_untracked++;
        shadow_copy(r, lo, hi, 0);
        return;
    }
    uint64_t run = 0;
    for (uint64_t k = lo >> REG_SHIFT; k <= (hi - 1) >> REG_SHIFT; k++) {
        uint64_t a = k << REG_SHIFT, b = a + (1ull << REG_SHIFT);
        if (a < lo) a = lo;
        if (b > hi) b = hi;
        const uint32_t *ch = gen_chunk[a >> GEN_CHUNK_SHIFT];   // a 2 MiB region lies in one chunk
        if (reg_max[k] <= since || !ch) {
            if (run) shadow_copy(r, run, a, 0), run = 0;
            continue;
        }
        for (uint64_t x = a; x < b; x += 1ull << GEN_PAGE_SHIFT) {
            int dirty = ch[(x >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)] > since;
            if (dirty && !run) run = x;
            else if (!dirty && run) shadow_copy(r, run, x, 0), run = 0;
        }
    }
    if (run) shadow_copy(r, run, hi, 0);
}

void vk_mem_shadow_sync(void) {
    if (!shadow_on) return;
    uint64_t t0 = qpc();
    plat_mutex_lock(&shadow_lock);
    if (nstouch) {
        plat_mutex_lock(&gen_lock);
        for (int i = 0; i < nstouch; i++) shadow_sync_pool(stouch[i]), stouch[i]->stouched = 0;
        plat_mutex_unlock(&gen_lock);
        s_syncs += (uint64_t)nstouch;
        nstouch = 0;
    }
    s_ticks += qpc() - t0;
    plat_mutex_unlock(&shadow_lock);
}

void vk_mem_gpu_pending(uint64_t addr, uint64_t size) {
    if (!shadow_on || !size) return;
    plat_mutex_lock(&shadow_lock);
    if (nspend == cspend) grow((void **)&spend, &cspend, sizeof *spend);
    spend[nspend++] = (SRange){addr, addr + size};
    plat_mutex_unlock(&shadow_lock);
}

void vk_gpu_written(uint64_t addr, uint64_t size) {
    if (shadow_on && size) {
        plat_mutex_lock_shared(&pool_lock);
        const pidx_ent *e = pidx_find(&pool_index, addr, size);
        PoolRec *r = e ? e->buf : NULL;
        if (r && r->nwin) r = window_get(r, addr, 0);   // the window the binding at addr used
        plat_mutex_lock(&shadow_lock);
        if (r && r->smap && addr >= r->import_lo && addr + size <= r->import_hi) {
            mem_thaw(addr, size);
            memcpy((void *)(uintptr_t)addr, r->smap + (addr - r->import_lo), size);
            s_wb_bytes += size;
        }
        for (int i = 0; i < nspend; i++)
            if (spend[i].lo == addr && spend[i].hi == addr + size) { spend[i] = spend[--nspend]; break; }
        plat_mutex_unlock(&shadow_lock);
        plat_mutex_unlock_shared(&pool_lock);
    }
    vk_guest_written(addr, size);
}

static void shadow_log(uint64_t frame) {
    if (!shadow_on) return;
    plat_mutex_lock(&shadow_lock);
    rt_log("mem: shadow at frame %llu: %lld pools %.1f MiB; since the last: %llu pool syncs, %llu queries, %.1f MiB "
           "copied in (%llu untracked), %.1f MiB written back, %.2f ms, %d GPU writes pending",
           (unsigned long long)frame, (long long)s_live_n, s_live_bytes / 1048576.0, (unsigned long long)s_syncs,
           (unsigned long long)s_queries, s_copied / 1048576.0, (unsigned long long)s_untracked, s_wb_bytes / 1048576.0,
           qpc_ms(s_ticks), nspend);
    s_syncs = s_queries = s_copied = s_untracked = s_wb_bytes = s_ticks = 0;
    plat_mutex_unlock(&shadow_lock);
}

// ---------------------------------------------------------------- VRAM mirror of buffer-bound pools
// In mirror mode (discrete GPU) an imported pool gets a device-local buffer over its imported range at its first
// buffer use (vk_mem_use); bindings and copies that touch the pool (vk_mem_pick) then use it at the same offsets.
// CPU -> mirror: mirror_sync, the first access of a submission, copies the pages stamped after the mirror's mgen.
// GPU -> guest: vk_mem_gpu_wrote ranges go back into the imported buffer before the fence (vk_mem_mirror_end).
// A pool stays imported when its mirror would not fit the VRAM budget, its range is not write-watched, it overlaps
// another pool or the allocation fails. See README, "VRAM mirror".
enum { MR_BUDGET, MR_UNTRACKED, MR_OVERLAP, MR_ALLOC, MR_N };

int vk_mirror_on;
enum { mirror_gap = 2 };               // pre-snap: hot pools at most this many 2 MiB regions apart share one query
static PoolRec **mlist;                // pool_lock: mirrored pools by import_lo
static int nm, mcap;
uint32_t msub;                  // replay: number of the submission being recorded
static uint32_t mepoch;                // replay: its write-watch snapshot epoch (vk_texture_epoch)
MRange *pend, *wbl;             // replay: copies for the next prologue; GPU writes into mirrors this submission
int npend, cpend, nwbl, cwbl;
static VkBufferCopy *mregions;         // replay: vkCmdCopyBuffer regions
static int cregions;
static PlatMutex flush_lock = PLAT_MUTEX_INIT;

typedef struct {
    uint64_t subs, syncs, upload_bytes, upload_copies, first_n, first_bytes, wb_bytes, wb_copies, wb_ranges;
    uint64_t presnap_queries, sync_queries, presnap_t, sync_t, prologues, presnap_bytes;
    uint64_t flush_ww_bytes;           // flush-box bytes left to the write watch
} MirrorAcc;
static MirrorAcc mrep;                 // replay: this submission
static MirrorAcc macc;                 // mirror_stat_lock: since the last summary
static PlatMutex mirror_stat_lock = PLAT_MUTEX_INIT;
static volatile int64_t m_live_n, m_live_bytes, m_refused_n[MR_N], m_refused_bytes[MR_N];
static volatile int64_t m_flush_calls, m_flush_bytes;

static void mirror_mode(int discrete, uint32_t vendor) {
    const int mode = discrete ? 1 : 0;   // 0 import, 1 mirror
    const char *why = discrete ? "auto, discrete gpu" : "auto, uma";
    vk_mirror_on = mode > 0;
    // Off on NVIDIA: the camera blocks (c3) differ from their shadow while noted as GPU writes, so shadow_copy leaves
    // its CPU bytes out and the fence copies the old shadow bytes over guest memory: draws read a stale view.
#ifdef _WIN32
    shadow_on = 0;
#else
    shadow_on = vendor != 0x10de;
#endif
    if (shadow_on) rt_log("mem: shadow storage: CPU-visible pools are host-visible copies of guest memory, not imports");
    rt_log("mem: mode %s (%s): %s", mode ? "mirror" : "import", why,
           mode ? "pools used as buffers get a device-local mirror" : "every CPU-visible pool is imported host memory");
}

void range_add(MRange **a, int *n, int *cap, PoolRec *r, uint64_t off, uint64_t size) {
    if (*n && (*a)[*n - 1].r == r && (*a)[*n - 1].off + (*a)[*n - 1].size == off) {
        (*a)[*n - 1].size += size;
        return;
    }
    if (*n == *cap) grow((void **)a, cap, sizeof **a);
    (*a)[(*n)++] = (MRange){r, off, size};
}

int range_cmp(const void *x, const void *y) {
    const MRange *a = x, *b = y;
    if (a->r != b->r) return (uintptr_t)a->r < (uintptr_t)b->r ? -1 : 1;
    return a->off < b->off ? -1 : a->off > b->off;
}

// Records the ranges as one vkCmdCopyBuffer per pool, overlapping and adjacent ranges merged; to_mirror: buf -> mbuf,
// else mbuf -> buf. Returns the commands recorded; adds what they move to *bytes and *copies.
static int copy_ranges(VkCommandBuffer cb, MRange *a, int n, int to_mirror, uint64_t *bytes, uint64_t *copies) {
    qsort(a, (size_t)n, sizeof *a, range_cmp);
    int rec = 0;
    for (int i = 0; i < n;) {
        PoolRec *r = a[i].r;
        uint64_t lim = r->import_hi - r->import_lo;
        int k = 0;
        for (; i < n && a[i].r == r; i++) {
            uint64_t lo = a[i].off, hi = a[i].off + a[i].size;
            if (hi > lim) hi = lim;
            if (lo >= hi) continue;
            if (k && lo <= mregions[k - 1].srcOffset + mregions[k - 1].size) {
                uint64_t end = mregions[k - 1].srcOffset + mregions[k - 1].size;
                if (hi > end) mregions[k - 1].size += hi - end;
                continue;
            }
            if (k == cregions) grow((void **)&mregions, &cregions, sizeof *mregions);
            mregions[k++] = (VkBufferCopy){lo, lo, hi - lo};
        }
        if (!k) continue;
        if (to_mirror) vkCmdCopyBuffer(cb, r->buf, r->mbuf, (uint32_t)k, mregions);
        else vkCmdCopyBuffer(cb, r->mbuf, r->buf, (uint32_t)k, mregions);
        rec++;
        for (int j = 0; j < k; j++) *bytes += mregions[j].size;
        *copies += (uint64_t)k;
    }
    return rec;
}

void mbarrier(VkCommandBuffer cb, VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds,
                     VkAccessFlags2 da) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = ss, .srcAccessMask = sa,
                           .dstStageMask = ds, .dstAccessMask = da};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

static void refuse(PoolRec *r, int why, const char *what) {
    r->mstate = MS_REFUSED;
    uint64_t n = r->import_hi - r->import_lo;
    if (plat_inc64(&m_refused_n[why]) <= 4) {
        char f[160];
        flag_names(r->flags, f, sizeof f);
        rt_log("mem: pool %#llx+%#llx (%s) stays imported: %s", (unsigned long long)r->lo, (unsigned long long)r->size,
               f, what);
    }
    plat_add64(&m_refused_bytes[why], (int64_t)n);
}

// Replay thread: the pool's mirror on its first buffer use.
static void mirror_create(PoolRec *r) {
    VkCtx *c = vk_ctx();
    uint64_t n = r->import_hi - r->import_lo;
    r->mstate = MS_REFUSED;
    __atomic_add_fetch(&mirror_gen, 1, __ATOMIC_RELEASE);
    if (r->kind != POOL_IMPORT || !r->buf || !n) return;
    plat_mutex_lock_shared(&pool_lock);
    int i = pidx_upper(&pool_index, r->lo) - 1;
    while (i >= 0 && pool_index.e[i].buf != r) i--;
    int alone = i >= 0 && (i == 0 || pool_index.maxhi[i - 1] <= pool_index.e[i].lo) &&
                (i + 1 == pool_index.n || pool_index.e[i + 1].lo >= pool_index.e[i].hi);
    plat_mutex_unlock_shared(&pool_lock);
    if (i < 0) return;   // finalized meanwhile
    if (!alone) return refuse(r, MR_OVERLAP, "it overlaps another pool");
    uint64_t r0 = r->import_lo >> SNAP_SHIFT, r1 = (r->import_hi - 1) >> SNAP_SHIFT;
    if (r1 >= SNAP_REGIONS) return refuse(r, MR_UNTRACKED, "outside the write-watched address space");
    uint64_t used, budget;
    int exact;
    if (vk_mem_vram(&used, &budget, &exact) && used + n + budget / 8 > budget) {
        char why[128];
        snprintf(why, sizeof why, "its %.1f MiB mirror does not fit the VRAM budget (%.0f of %.0f MiB in use)",
                 n / 1048576.0, used / 1048576.0, budget / 1048576.0);
        return refuse(r, MR_BUDGET, why);
    }
    plat_mutex_lock(&gen_lock);
    mrep.sync_queries += snap_regions(r0, r1, mepoch);
    int untracked = 0;
    for (uint64_t k = r0; k <= r1; k++) untracked |= snap_untracked[k];
    r->mgen = gen_now;   // the whole-range copy below holds every write observed so far
    plat_mutex_unlock(&gen_lock);
    if (untracked) return refuse(r, MR_UNTRACKED, "its range is not write-watched");
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = n, .usage = pool_usage(c),
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                  .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
    VkResult res = vmaCreateBufferWithAlignment(c->vma, &bi, &ai, ft_on ? 4096 : 1, &r->mbuf, &r->malloc_, NULL);   // a table entry names whole pages
    if (res != VK_SUCCESS) {
        r->mbuf = VK_NULL_HANDLE, r->malloc_ = NULL;
        char why[96];
        snprintf(why, sizeof why, "device-local allocation of %.1f MiB failed (%d)", n / 1048576.0, (int)res);
        return refuse(r, MR_ALLOC, why);
    }
    plat_mutex_lock(&pool_lock);
    if (nm == mcap) grow((void **)&mlist, &mcap, sizeof *mlist);
    int at = nm;
    while (at > 0 && mlist[at - 1]->import_lo > r->import_lo) at--;
    memmove(mlist + at + 1, mlist + at, (size_t)(nm - at) * sizeof *mlist);
    mlist[at] = r;
    nm++;
    plat_mutex_unlock(&pool_lock);
    r->acc_sub = msub;
    r->mstate = MS_ON;
    __atomic_add_fetch(&gmem_changes, 1, __ATOMIC_RELEASE);   // the raw pool table must now reach the mirror
    if (ft_on) ft_pool_put(r);
    range_add(&pend, &npend, &cpend, r, 0, n);
    wbd_mirror_made(r);   // its first content comes from this same full upload (wbd_after_upload)
    mrep.first_n++, mrep.first_bytes += n;
    plat_inc64(&m_live_n);
    plat_add64(&m_live_bytes, (int64_t)n);
}

static void mirror_forget(PoolRec *r) {
    __atomic_add_fetch(&mirror_gen, 1, __ATOMIC_RELEASE);
    for (int i = 0; i < nm; i++)
        if (mlist[i] == r) {
            memmove(mlist + i, mlist + i + 1, (size_t)(nm - i - 1) * sizeof *mlist);
            nm--;
            plat_dec64(&m_live_n);
            plat_add64(&m_live_bytes, -(int64_t)(r->import_hi - r->import_lo));
            break;
        }
}

// Finalized mirrored pools waiting for mirror_retire; vk_mem_shutdown destroys those whose callback never ran.
static PoolRec **retiring;
static int nretiring, cretiring;
static PlatMutex retire_lock = PLAT_MUTEX_INIT;

static void mirror_retiring(PoolRec *r) {
    plat_mutex_lock(&retire_lock);
    if (nretiring == cretiring) grow((void **)&retiring, &cretiring, sizeof *retiring);
    retiring[nretiring++] = r;
    plat_mutex_unlock(&retire_lock);
}

static void mirror_retire(void *arg) {
    plat_mutex_lock(&retire_lock);
    for (int i = 0; i < nretiring; i++)
        if (retiring[i] == arg) { retiring[i] = retiring[--nretiring]; break; }
    plat_mutex_unlock(&retire_lock);
    destroy_rec(vk_ctx(), arg);
}

// A pool's nvnMemoryPoolFlushMappedRange bounding box adds no copies when the write watch tracks every region of it;
// the pages the CPU did write are queued from their stamps. A box over a region the last query could not watch
// (snap_untracked) still goes up whole.
static int flush_ww_covers(uint64_t a, uint64_t b) {
    static int said;
    if (!said) {
        said = 1;
        rt_log("mem: flush boxes over write-watched regions add no copies (written pages go up from their stamps)");
    }
    for (uint64_t k = a >> SNAP_SHIFT; k <= (b - 1) >> SNAP_SHIFT; k++)
        if (snap_untracked[k] || snap_at[k] != mepoch) return 0;   // not watched, or not queried this submission
    return 1;
}

// ms_walk: set by gmem_sync_mirrors, which times its whole walk once instead of reading the clock twice per mirror.
static int ms_walk;
// Replay thread, the first access of this submission: queues the copies of the pages written since the last sync.
static void mirror_sync(PoolRec *r) {
    uint64_t t0 = ms_walk ? 0 : qpc();
    r->acc_sub = msub;
    uint64_t lo = r->import_lo, hi = r->import_hi;
    uint64_t r0 = lo >> SNAP_SHIFT, r1 = (hi - 1) >> SNAP_SHIFT;
    int excl = 0;
    plat_mutex_lock_shared(&gen_lock);
    for (uint64_t k = r0; k <= r1; k++)
        if (snap_at[k] != mepoch) { excl = 1; break; }
    if (excl) {
        plat_mutex_unlock_shared(&gen_lock);
        plat_mutex_lock(&gen_lock);
        mrep.sync_queries += snap_regions(r0, r1, mepoch);
    }
    uint32_t since = r->mgen;
    int64_t skipped = 0;
    for (uint64_t k = r0; k <= r1; k++) {
        if (reg_max[k] <= since) continue;
        uint64_t a = k << SNAP_SHIFT, b = a + (1ull << SNAP_SHIFT);
        if (a < lo) a = lo;
        if (b > hi) b = hi;
        const uint32_t *ch = gen_chunk[a >> GEN_CHUNK_SHIFT];   // a 2 MiB region lies in one chunk
        if (!ch) continue;
        const uint32_t *ec = echo_chunk[a >> GEN_CHUNK_SHIFT];
        for (uint64_t x = a; x < b; x += 1ull << GEN_PAGE_SHIFT) {
            uint32_t i = (uint32_t)((x >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)), v = ch[i];
            if (v <= since) continue;
            if (ec && ec[i] == v) {   // wb_echo: our own write-back, the mirror holds it
                skipped++;
                continue;
            }
            range_add(&pend, &npend, &cpend, r, x - lo, 1ull << GEN_PAGE_SHIFT);
        }
    }
    if (skipped) plat_add64(&echo_skipped, skipped);
    r->mgen = gen_now;
    if (excl) plat_mutex_unlock(&gen_lock);
    else plat_mutex_unlock_shared(&gen_lock);
    if (__atomic_load_n(&r->fl_hi, __ATOMIC_ACQUIRE)) {
        plat_mutex_lock(&flush_lock);
        uint64_t a = r->fl_lo, b = r->fl_hi;
        r->fl_hi = 0;
        plat_mutex_unlock(&flush_lock);
        a = a > lo ? a & ~0xfffull : lo;
        b = b < hi ? (b + 0xfff) & ~0xfffull : hi;
        if (a < b && flush_ww_covers(a, b)) mrep.flush_ww_bytes += b - a;   // the write watch has it
        else if (a < b) range_add(&pend, &npend, &cpend, r, a - lo, b - a);
    }
    mrep.syncs++;
    if (!ms_walk) mrep.sync_t += qpc() - t0;
}

// A shader on the raw table reads any pool without the replay seeing it, so the CPU's writes to every mirrored pool must
// reach its mirror this submission, not only those of pools a binding used.
static void gmem_sync_mirrors(void) {
    static uint32_t done;
    if (!vk_mirror_on || done == msub) return;
    done = msub;
    plat_mutex_lock_shared(&pool_lock);
    uint64_t t0 = qpc();
    ms_walk = 1;
    for (int i = 0; i < nm; i++) if (mlist[i]->acc_sub != msub) mirror_sync(mlist[i]);
    ms_walk = 0;
    mrep.sync_t += qpc() - t0;
    plat_mutex_unlock_shared(&pool_lock);
}

void vk_mem_use(VkMemSpan *sp) {
    PoolRec *r = sp->pool;
    if (r->mstate == MS_NONE) mirror_create(r);
    if (r->mstate != MS_ON) return;
    if (r->acc_sub != msub) mirror_sync(r);
    sp->buffer = r->mbuf;
}

void vk_mem_pick(VkMemSpan *sp) {
    PoolRec *r = sp->pool;
    if (!r || r->mstate != MS_ON) return;
    if (r->acc_sub != msub) mirror_sync(r);
    sp->buffer = r->mbuf;
}

int vk_mem_mirror_probe(uint64_t addr) {
    VkMemSpan sp;
    if (!vk_mirror_on || !vk_mem_resolve(addr, 1, &sp) || !sp.pool) return 0;
    mepoch = vk_texture_epoch();
    vk_mem_use(&sp);
    return ((PoolRec *)sp.pool)->mstate == MS_ON;
}

void nvn_pool_flushed(const pool_t *p, uint64_t off, uint64_t size) {
    if (!vk_mirror_on || !size || !p->mem) return;
    plat_inc64(&m_flush_calls);
    plat_add64(&m_flush_bytes, (int64_t)size);
    plat_mutex_lock_shared(&pool_lock);
    const pidx_ent *e = pidx_find(&pool_index, (uintptr_t)p->mem, 0);
    PoolRec *r = e && e->key == p ? e->buf : NULL;
    if (r && r->mstate == MS_ON) {
        uint64_t a = (uintptr_t)p->mem + off, b = a + size;
        plat_mutex_lock(&flush_lock);
        if (!r->fl_hi) r->fl_lo = a, r->fl_hi = b;
        else {
            if (a < r->fl_lo) r->fl_lo = a;
            if (b > r->fl_hi) r->fl_hi = b;
        }
        plat_mutex_unlock(&flush_lock);
    }
    plat_mutex_unlock_shared(&pool_lock);
}

// gen_lock exclusive: freezes up to FREEZE_PER_SUB candidate regions still idle. The query after the
// protection collects the writes made before it took effect.
static void freeze_pass(void) {
    for (int done = 0; nfcand && done < FREEZE_PER_SUB;) {
        uint64_t j = fcand[--nfcand], a = j << REG_SHIFT;
        if (ww_ep - reg_last[j] < (uint32_t)freeze_idle || mem_frozen(a)) continue;
        done++;
        if (!mem_freeze(a)) { n_frz_refused++; continue; }
        query_stamp(a, a + (1ull << REG_SHIFT), ++gen_now);
    }
}

void vk_mem_mirror_begin(void) {
    if (!vk_mirror_on) {   // import mode: the texture checks' snapshots still skip frozen regions
        plat_mutex_lock(&gen_lock);
        ww_ep++;
        freeze_pass();
        plat_mutex_unlock(&gen_lock);
        return;
    }
    uint64_t t0 = qpc();
    uint32_t prev = msub;
    msub = msub + 1 ? msub + 1 : 1;
    __atomic_add_fetch(&mirror_gen, 1, __ATOMIC_RELEASE);
    mepoch = vk_texture_epoch();
    uint64_t q = 0, run0 = 0, run1 = 0;
    int have = 0;
    plat_mutex_lock_shared(&pool_lock);
    plat_mutex_lock(&gen_lock);
    ww_ep++;
    uint64_t b0 = ww_bytes;
    ww_batch_begin();   // the runs below are queried together, in parallel
    for (int i = 0; i < nm; i++) {
        const PoolRec *r = mlist[i];
        if (r->acc_sub != prev) continue;
        uint64_t a = r->import_lo >> SNAP_SHIFT, b = (r->import_hi - 1) >> SNAP_SHIFT;
        if (have && a <= run1 + 1 + (uint64_t)mirror_gap) {
            if (b > run1) run1 = b;
            continue;
        }
        if (have) q += snap_regions(run0, run1, mepoch);
        run0 = a, run1 = b, have = 1;
    }
    if (have) q += snap_regions(run0, run1, mepoch);
    ww_batch_run();
    mrep.presnap_bytes += ww_bytes - b0;
    freeze_pass();
    plat_mutex_unlock(&gen_lock);
    plat_mutex_unlock_shared(&pool_lock);
    mrep.subs++;
    mrep.presnap_queries += q;
    mrep.presnap_t += qpc() - t0;
    if (ft_on) gmem_sync_mirrors();   // a shader on the page table can read any pool, and no binding says which
}

int vk_mem_mirror_pending(void) { return npend || (ft_on && (__atomic_load_n(&ft_n, __ATOMIC_ACQUIRE) || !ft_filled)); }

// ---- sky SH keep: the sky SH block's two accumulated windows keep their GPU bytes across uploads (saved before, put
// back after) and each is zeroed in GPU order just before the dispatch that sums into it. A whole-page upload would
// otherwise zero the sky SH and the exposure measurement would darken the scene. See README, "Sky SH keep".
static const struct { uint32_t off, size; uint64_t native; } skk_win[2] = {{0x30, 0x40, 0x499dbfd0721d2858ull},
                                                                          {0x80, 0x10, 0xab7699ee7f536755ull}};
static struct { int on, fixed; uint64_t addr; PoolRec *r; uint64_t off; int seen; } skk = {.addr = 0x25a4c7f00ull};
static VkBuffer skk_buf;
static VmaAllocation skk_alloc;
int vk_sky_sh_keep(void) {
    static int said;
    if (!said) {
        said = 1;
        rt_log("mem: sky SH keep: block %#llx: +0x30..+0x70 and +0x80..+0x90 keep their GPU bytes, each cleared "
               "before its sum dispatch", (unsigned long long)skk.addr);
    }
    return 1;
}
// The block the sum dispatches bind (vk_exec_draw.c sky_sh_before_dispatch: storage slot 0, 0xc0 bytes), taken from
// where the game binds it: a fixed address moves with the heap size.
void vk_sky_sh_block(uint64_t addr) {
    if (!vk_sky_sh_keep() || addr == skk.addr) return;
    rt_log("mem: sky SH keep: the sky SH block is at %#llx (was taken at %#llx): followed (frame %llu)", (unsigned long long)addr,
           (unsigned long long)skk.addr, (unsigned long long)vk_frame_no);
    skk.addr = addr, skk.r = NULL, skk.off = 0, skk.seen = 0;
}
// The window summed by the dispatch whose program has this native key, -1 if none.
int vk_sky_sh_window(uint64_t native) {
    for (int w = 0; w < 2; w++) if (skk_win[w].native == native) return w;
    return -1;
}

void vk_sky_sh_clear(VkCommandBuffer cb, int w) {   // replay, outside a render pass, before the sum dispatch
    if (!vk_sky_sh_keep() || !skk.r || !skk.r->mbuf || w < 0 || w > 1) return;
    if (w == 0) {
        mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    }
    mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
             VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    vkCmdFillBuffer(cb, skk.r->mbuf, skk.off + skk_win[w].off, skk_win[w].size, 0);
    mbarrier(cb, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
             VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
}

void vk_mem_mirror_prologue(VkCommandBuffer cb) {
    if (ft_on) ft_flush(cb);
    if (!npend) return;
    mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
             VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
    int keep = 0;   // the sky SH block is inside this upload and has GPU bytes worth keeping
    if (vk_sky_sh_keep()) {
        for (int i = 0; i < npend && !keep; i++) {
            uint64_t lo = pend[i].r->import_lo + pend[i].off;
            if (skk.addr < lo || skk.addr + 0xc0 > lo + pend[i].size || !pend[i].r->mbuf) continue;
            skk.r = pend[i].r, skk.off = skk.addr - pend[i].r->import_lo;
            if (!skk.seen) rt_log("mem: sky SH keep: the block's page goes up for the first time (frame %llu)", (unsigned long long)vk_frame_no);
            keep = skk.seen++ > 0;   // the first upload goes through
        }
        if (keep && !skk_buf) {
            VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4096,
                                     .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
            VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
            if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &skk_buf, &skk_alloc, NULL) != VK_SUCCESS) skk_buf = VK_NULL_HANDLE, keep = 0;
        }
        if (keep) {
            for (int w = 0; w < 2; w++)
                vkCmdCopyBuffer(cb, skk.r->mbuf, skk_buf, 1, &(VkBufferCopy){skk.off + skk_win[w].off, skk_win[w].off, skk_win[w].size});
            mbarrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT);
        }
    }
    copy_ranges(cb, pend, npend, 1, &mrep.upload_bytes, &mrep.upload_copies);
    if (keep) {
        mbarrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                 VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
        for (int w = 0; w < 2; w++)
            vkCmdCopyBuffer(cb, skk_buf, skk.r->mbuf, 1, &(VkBufferCopy){skk_win[w].off, skk.off + skk_win[w].off, skk_win[w].size});
    }
    mbarrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
             VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    wbd_after_upload(cb, pend, npend);   // wb_dirty: the shadows follow the uploads
    npend = 0;
    mrep.prologues++;
}

// wb_echo: the whole pages of the merged write-back ranges (copy_ranges merges them the same way), as guest addresses.
static void echo_spans(uint64_t fence) {
    static EchoSpan *sp;
    static int csp;
    int n = 0;
    for (int i = 0; i < nwbl;) {
        PoolRec *r = wbl[i].r;
        uint64_t lim = r->import_hi - r->import_lo, lo = wbl[i].off, hi = wbl[i].off + wbl[i].size;
        for (i++; i < nwbl && wbl[i].r == r && wbl[i].off <= hi; i++)
            if (wbl[i].off + wbl[i].size > hi) hi = wbl[i].off + wbl[i].size;
        if (hi > lim) hi = lim;
        uint64_t a = (r->import_lo + lo + 0xfff) & ~0xfffull, b = (r->import_lo + hi) & ~0xfffull;
        if (a >= b || hi - lo < 1) continue;
        if (n == csp) grow((void **)&sp, &csp, sizeof *sp);
        sp[n++] = (EchoSpan){a, b};
    }
    echo_job(fence, sp, n);
}

int vk_mem_mirror_end(VkCommandBuffer cb, uint64_t fence) {
    if (!vk_mirror_on) return 0;
    int rec = 0;
    if (nwbl) {
        mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                 VK_ACCESS_2_TRANSFER_READ_BIT);
        if (wb_record(cb, fence)) {   // the guest bytes are read before the write-back below changes them
            mbarrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            rec += 2;
        }
        for (int i = 0; i < nwbl; i++) wbl[i].r->wb_sub = msub;
        mrep.wb_ranges += (uint64_t)nwbl;
        // wb_dirty: the compare pass takes what it can (sizes 0), copy_ranges the rest; the full list comes back after
        // for the echo marks and the trap/census hooks
        static MRange *keep;
        static int ckeep;
        int dirty = 0;
        if (wbd_on()) {
            if (nwbl > ckeep) { MRange *k = realloc(keep, sizeof *k * (size_t)nwbl); if (k) keep = k, ckeep = nwbl; }
            if (nwbl <= ckeep) {
                mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
                         VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
                qsort(wbl, (size_t)nwbl, sizeof *wbl, range_cmp);
                memcpy(keep, wbl, sizeof *wbl * (size_t)nwbl);
                dirty = wbd_record(cb, wbl, nwbl);
                rec += dirty;
            }
        }
        rec += copy_ranges(cb, wbl, nwbl, 0, &mrep.wb_bytes, &mrep.wb_copies) + 2;
        if (dirty && wbd_nends) rec += copy_ranges(cb, wbd_ends, wbd_nends, 0, &mrep.wb_bytes, &mrep.wb_copies);   // wbd_split ends
        if (dirty) memcpy(wbl, keep, sizeof *wbl * (size_t)nwbl);
        pw_submit(fence);
        echo_spans(fence);   // wbl is sorted by copy_ranges
        mbarrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
        nwbl = 0;
    }
    plat_mutex_lock(&mirror_stat_lock);
    uint64_t *d = (uint64_t *)&macc, *s = (uint64_t *)&mrep;
    for (size_t i = 0; i < sizeof macc / sizeof(uint64_t); i++) d[i] += s[i];
    plat_mutex_unlock(&mirror_stat_lock);
    memset(&mrep, 0, sizeof mrep);
    return rec;
}

// One summary line: with OOT_TRACE=perf every 60 frames, else at frames 600 and 1800.
static void mirror_log(uint64_t frame) {
    static uint64_t last, ww_last;
    plat_mutex_lock(&gen_lock);
    uint64_t wc = ww_calls, wb = ww_bytes, wt = ww_ticks, dc = ww_dirty_calls, dt = ww_dirty_ticks, wp = ww_pages;
    ww_pages = 0;
    uint64_t nr = n_frz_refused;
    ww_calls = ww_bytes = ww_ticks = ww_dirty_calls = ww_dirty_ticks = n_frz_refused = 0;
    plat_mutex_unlock(&gen_lock);
    double wf = frame > ww_last ? (double)(frame - ww_last) : 1.0;
    ww_last = frame;
    uint64_t fz, ff, ft, fo;
    mem_freeze_stats(&fz, &ff, &ft, &fo);
    if (rt_trace & RT_TRACE_PERF)
        rt_log("mem: write-watch frame %llu: per frame, every consumer: %.1f queries over %.1f MiB, %.3f ms (%.1f found "
               "%.0f written pages, %.3f ms); frozen %.0f MiB; since the last line: freezes %llu, thaws by a write %llu, by a file read "
               "or mapping change %llu, refused %llu",
               (unsigned long long)frame, wc / wf, wb / wf / 1048576.0, qpc_ms(wt) / wf, dc / wf, wp / wf, qpc_ms(dt) / wf,
               fz * 2.0, (unsigned long long)ff, (unsigned long long)ft, (unsigned long long)fo, (unsigned long long)nr);
    if (!vk_mirror_on) return;
    plat_mutex_lock(&mirror_stat_lock);
    MirrorAcc a = macc;
    memset(&macc, 0, sizeof macc);
    plat_mutex_unlock(&mirror_stat_lock);
    double f = frame > last ? (double)(frame - last) : 1.0, M = 1048576.0;
    last = frame;
    uint64_t fc = (uint64_t)plat_xchg64(&m_flush_calls, 0), fb = (uint64_t)plat_xchg64(&m_flush_bytes, 0);
    char ver[160] = "";
    rt_log("mem: mirror frame %llu: pools %lld, %.1f MiB device-local; stay imported: budget %lld (%.1f MiB), untracked "
           "%lld, overlapping %lld, failed %lld; per frame: submits %.2f, syncs %.1f, upload %.3f MiB in %.1f ranges, "
           "first use %.2f pools %.3f MiB, write-back %.3f MiB in %.1f ranges, flush calls %.1f (%.3f MiB), queries "
           "%.1f pre-snap (%.1f MiB) + %.1f sync, prologues %.2f, tracking cpu %.3f ms (pre-snap %.3f, syncs %.3f); flush boxes left "
           "to the write watch %.3f MiB%s",
           (unsigned long long)frame, (long long)m_live_n, m_live_bytes / M, (long long)m_refused_n[MR_BUDGET],
           m_refused_bytes[MR_BUDGET] / M, (long long)m_refused_n[MR_UNTRACKED], (long long)m_refused_n[MR_OVERLAP],
           (long long)m_refused_n[MR_ALLOC], a.subs / f, a.syncs / f, a.upload_bytes / f / M, a.upload_copies / f,
           a.first_n / f, a.first_bytes / f / M, a.wb_bytes / f / M, a.wb_copies / f, fc / f, fb / f / M,
           a.presnap_queries / f, a.presnap_bytes / f / M, a.sync_queries / f, a.prologues / f, qpc_ms(a.presnap_t + a.sync_t) / f,
           qpc_ms(a.presnap_t) / f, qpc_ms(a.sync_t) / f, a.flush_ww_bytes / f / M, ver);
    int64_t ep = plat_xchg64(&echo_pages, 0), es = plat_xchg64(&echo_skipped, 0);
    int64_t erm = plat_xchg64(&echo_rule_marks, 0), ers = plat_xchg64(&echo_rule_skips, 0);
    rt_log("mem: mirror frame %llu: wb_echo %lld: per frame %.0f pages marked as write-back echoes, %.0f page uploads "
           "skipped (%.3f MiB); on pages with a range the rule keeps out: %lld marked, %lld skipped",
           (unsigned long long)frame, 1ll, ep / f, es / f, es * 4096.0 / f / M, (long long)erm, (long long)ers);
    pw_report(f);
    wbd_report(f);
}

// With OOT_TRACE=perf every 600 frames: what holds device-local memory (VMA blocks, texture entries, mirrors, dead objects).
static void vram_log(uint64_t frame) {
    VkCtx *c = vk_ctx();
    if (!c) return;
    VmaBudget vb[VK_MAX_MEMORY_HEAPS];
    vmaGetHeapBudgets(c->vma, vb);
    uint64_t blocks = 0, block_bytes = 0, allocs = 0, alloc_bytes = 0;
    for (uint32_t h = 0; h < c->mem_props.memoryHeapCount && h < VK_MAX_MEMORY_HEAPS; h++)
        if (c->mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            blocks += vb[h].statistics.blockCount, block_bytes += vb[h].statistics.blockBytes,
                allocs += vb[h].statistics.allocationCount, alloc_bytes += vb[h].statistics.allocationBytes;
    VkTexCensus t;
    vk_texture_census(&t);
    const double M = 1048576.0;
    rt_log("mem: vram frame %llu: vma %llu blocks %.0f MiB, %llu allocations %.0f MiB; textures live %llu (own memory %llu, "
           "%.0f MiB), released %llu (own memory %llu, %.0f MiB); mirrors %lld %.0f MiB; releases pending %d",
           (unsigned long long)frame, (unsigned long long)blocks, block_bytes / M, (unsigned long long)allocs,
           alloc_bytes / M, (unsigned long long)t.live, (unsigned long long)t.own_live, t.bytes_live / M,
           (unsigned long long)t.released, (unsigned long long)t.own_released, t.bytes_released / M, (long long)m_live_n,
           m_live_bytes / M, dead_pending());
}

// Device class: INTEGRATED -> uma; a discrete GPU with a DEVICE_LOCAL heap of 1 GiB or more beside a
// heap that is not device-local (host memory) -> discrete; otherwise uma (import everywhere). Sets the mirror mode.
void vk_mem_class_log(const VkPhysicalDeviceProperties *p, const VkPhysicalDeviceMemoryProperties *m) {
    int big_local = -1, host_heap = -1;
    for (uint32_t h = 0; h < m->memoryHeapCount; h++) {
        if (m->memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            if (m->memoryHeaps[h].size >= (1ull << 30) && big_local < 0) big_local = (int)h;
        } else if (host_heap < 0) host_heap = (int)h;
    }
    static const char *const types[] = {"other", "integrated", "discrete", "virtual", "cpu"};
    int t = (int)p->deviceType;
    const char *tn = t >= 0 && t <= 4 ? types[t] : "unknown";
    const char *cls, *why;
    if (p->deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) cls = "uma", why = "integrated gpu";
    else if (p->deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) cls = "uma", why = "not a discrete gpu";
    else if (big_local < 0) cls = "uma", why = "no device-local heap of 1 GiB or more";
    else if (host_heap < 0) cls = "uma", why = "every heap is device-local";
    else cls = "discrete", why = "device-local heap separate from host memory";
    int discrete = cls[0] == 'd';
    char line[2048];
    size_t k = (size_t)snprintf(line, sizeof line, "mem: class %s (%s; device type %d %s, %s)", cls, why, t, tn, p->deviceName);
    for (uint32_t h = 0; h < m->memoryHeapCount; h++)
        k += (size_t)snprintf(line + k, k < sizeof line ? sizeof line - k : 0, "; heap %u %.0f MiB flags %#x%s%s", h,
                              m->memoryHeaps[h].size / 1048576.0, m->memoryHeaps[h].flags,
                              m->memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT ? " DEVICE_LOCAL" : "",
                              m->memoryHeaps[h].flags & VK_MEMORY_HEAP_MULTI_INSTANCE_BIT ? " MULTI_INSTANCE" : "");
    rt_log("%s", line);
    k = (size_t)snprintf(line, sizeof line, "mem: types");
    for (uint32_t i = 0; i < m->memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = m->memoryTypes[i].propertyFlags;
        k += (size_t)snprintf(line + k, k < sizeof line ? sizeof line - k : 0, "; %u heap %u flags %#x%s%s%s%s%s", i,
                              m->memoryTypes[i].heapIndex, f, f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT ? " DEVICE_LOCAL" : "",
                              f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ? " HOST_VISIBLE" : "",
                              f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT ? " HOST_COHERENT" : "",
                              f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT ? " HOST_CACHED" : "",
                              f & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT ? " LAZY" : "");
    }
    rt_log("%s", line);
    mirror_mode(discrete, p->vendorID);
}

void vk_mem_stats(VkMemStats *s) {
    memset(s, 0, sizeof *s);
    plat_mutex_lock_shared(&pool_lock);
    for (int i = 0; i < pool_index.n; i++) {
        const PoolRec *r = pool_index.e[i].buf;
        if (r->kind == POOL_IMPORT) s->imported_pools++;
        else if (r->kind == POOL_PHYS) s->physical_pools++;
        else s->virtual_pools++;
    }
    plat_mutex_unlock_shared(&pool_lock);
    s->import_failures = (uint64_t)n_import_failures;
    s->upload_batches = next_seq;
}

void vk_mem_shutdown(void) {
    VkCtx *c = vk_ctx();
    if (!c) return;
    vk_upload_flush(1);
    vk_queue_lock();
    vkDeviceWaitIdle(c->device);
    vk_queue_unlock();
    vk_texture_shutdown();
    plat_rmutex_lock(&cmd_lock);
    retire(c, 1);
    uint64_t keep = tl_protected;
    tl_protected = 0;
    for (int i = 0; i < ndead; i++) dead[i].seq = 0, dead[i].tl = 0;   // the device is idle
    collect_locked(c);
    tl_protected = keep;
    for (int i = 0; i < nspare; i++) {
        vkDestroyFence(c->device, spare[i].fence, NULL);
        free(spare[i].stg);
    }
    nspare = 0;
    if (cmd_pool) vkDestroyCommandPool(c->device, cmd_pool, NULL);
    cmd_pool = VK_NULL_HANDLE;
    if (stg_pool) vmaDestroyPool(c->vma, stg_pool);   // after retire: no staging buffer is left in it
    stg_pool = NULL;
    stg_pool_tried = 0;
    plat_rmutex_unlock(&cmd_lock);
    plat_mutex_lock(&pool_lock);
    for (int i = 0; i < pool_index.n; i++) destroy_rec(c, pool_index.e[i].buf);
    pool_index.n = 0;
    nm = npend = nwbl = 0;   // mirrors went with their pools
    plat_mutex_lock(&shadow_lock);
    nstouch = nspend = 0;   // shadows too
    s_live_n = s_live_bytes = 0;
    plat_mutex_unlock(&shadow_lock);
    for (int i = 0; i < nretiring; i++) destroy_rec(c, retiring[i]);   // finalized ones whose callback never ran
    nretiring = 0;
    __atomic_add_fetch(&pool_gen, 1, __ATOMIC_RELEASE);
    nphys = 0;
    plat_mutex_unlock(&pool_lock);
    plat_mutex_lock(&gen_lock);
    for (uint64_t i = 0; i < GEN_CHUNKS; i++) free(gen_chunk[i]), gen_chunk[i] = NULL;
    plat_mutex_unlock(&gen_lock);
}

static uint32_t echo_frame(void) { return (uint32_t)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED); }
