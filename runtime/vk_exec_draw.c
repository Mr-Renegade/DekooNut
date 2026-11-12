// Draws, binds and dispatches of the Vulkan op-stream replay: the strong vk_exec_op_draw behind vk_exec.c's weak hook.
// Runs on the submitting thread under vk_sync_submit's lock; nothing here locks. Draw replay: see docs/how-it-works.md, "Draw replay".
// Provenance: the op semantics (bind ops, program and slot conventions, draw and dispatch flow, vertex streams with a
// zeroed buffer for unbound ones) follow MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/nvn_mtl.m:7409-7460,
// 9365-9488, 9489-9921, 10013-10079 @ b75be8cf (MIT, Copyright (c) 2026 jhaizhou); the uniform-buffer spans, texture and
// sampler descriptor lookups and the dispatch flow follow LAS Recomp runtime/nvn/vulkan/exec.c:1851-1908, 2300-2516
// @ da35b83a (MIT, Copyright (c) 2026 Peiikai). The support buffer layout is Ryujinx's SupportBuffer.cs (MIT).
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_hmap.h"
#include "nvn_geometry_control.h"
#include "nvn_indirect_reset.h"
#include "vk.h"
#include "vk_exec.h"
#include "vk_exec_draw_int.h"
#include "vk_view_format.h"
#include "tex.h"
#include "platform/platform.h"
#include "upscale_decode_spv.h"
#include "dlss_color_spv.h"
#include "dlss_motion_spv.h"
#include "dlss_output_spv.h"
#include "dlss_mask_spv.h"
#include "dlss_4k_composite_spv.h"

// Compute parameter layout of the dispatched program (NULL for NVN 1); read by vk_c0_fill.
static const Nvn2C0Layout *c0_lay;
static const uint32_t (*c0_push)[NVN2_ROOT_WORDS];
#define CHUNK_BYTES (4u << 20)
#define ZERO_BYTES 65536u
static uint8_t geometry_input;                              // normalized actual draw primitive; zero outside opt-in GS
#define ONESHOT_WAIT_MS 30000u   // prepare_draw: how long a one-shot draw waits for its pipeline before it is skipped

static void c0_lay_set(VkExec *ex, const GfxShader *cs) {
    c0_lay = gfx_shader_c0_layout(cs);
    c0_push = ex->push[1];
}

enum { W_PROGRAM, W_SHADER, W_PIPELINE, W_NO_VS, W_INDEX, W_UB, W_BIND, W_SSBO_ALIGN, W_SSBO_SLOT, W_RESTART, W_INDIRECT, W_GEOMETRY, W_COUNT };

static void warn_once(int key, const char *fmt, ...) {
    static volatile int32_t seen[W_COUNT];
    if (__atomic_exchange_n(&seen[key], 1, __ATOMIC_RELAXED)) return;
    char msg[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    rt_log("vk: %s (logged once)", msg);
}

static void *grow(void *p, int *cap, size_t elem) {
    *cap = *cap ? *cap * 2 : 8;
    p = realloc(p, elem * (size_t)*cap);
    if (!p) rt_fatal("vk: out of memory");
    return p;
}

// ---- resources of one command buffer, freed by its completion callback

typedef struct {
    VkImage image;
    VkFormat format;
    VkImageViewType type;
    uint32_t aspect, level0, levels, layer0, layers, swz;
} ViewKey;
_Static_assert(sizeof(ViewKey) == 40, "ViewKey is hashed as bytes: no padding");

static CbRes *cur;
static uint64_t cur_fence;

// Freed descriptor pools and full-size transient chunks are recycled: creating one is a driver memory allocation.
#define SPARE_POOLS 64
#define SPARE_CHUNKS 16
static PlatMutex spare_lock = PLAT_MUTEX_INIT;
static VkDescriptorPool spare_pool[SPARE_POOLS];
static Chunk spare_chunk[SPARE_CHUNKS];
static int nspare_pool, nspare_chunk;

typeof(B) B;   // vk_exec_draw_int.h

// Contents of B.ds_last (packed VkDescSlot descriptors): a draw whose set would hold the same reuses it. psig: the same
// for the last push per set number.
#define DS_SIG_BYTES (MAX_DESC * sizeof(VkDescSlot))
static struct { uint32_t n[2], pn[2]; uint8_t sig[2][DS_SIG_BYTES], psig[2][DS_SIG_BYTES]; } DS;

typeof(RS) RS = {-1};   // vk_exec_draw_int.h

// Bound-slot byte ranges of VkExec that a pipeline's descriptors read, merged where they touch, built once per pipeline
// info. ok 0: never skipped (too many descriptors or buffers).
#define RS_PTRS 4
typedef struct { uint32_t off, len; } RsRange;
typedef struct { int st; const GfxShaderRes *res; } RsPtr;
typedef struct { uint32_t n, bytes; uint8_t sup, c0, ok, np; RsPtr p[RS_PTRS]; RsRange r[]; } RsList;
static nvn_hmap rs_lists;                // VkPipeInfo pointer -> RsList, until shutdown
static const VkPipeInfo *rs_last;        // the last lookup
static const RsList *rs_last_l;

static uint32_t c0_dim[3], c0_binst;     // the dispatch size and base instance of the draw or dispatch being bound
static uint64_t bind_gen = 1;            // bumped by every bind op that writes ex->prog, ubo, ssbo, tex or img

static void cb_done(void *arg) {   // completion thread
    CbRes *r = arg;
    VkCtx *c = vk_ctx();
    for (int i = 0; i < r->nwritten; i++) vk_gpu_written_at(r->written[i].addr, r->written[i].size, r->fence);
    for (int i = 0; i < r->npools; i++) {
        vkResetDescriptorPool(c->device, r->pools[i], 0);
        plat_mutex_lock(&spare_lock);
        int kept = nspare_pool < SPARE_POOLS;
        if (kept) spare_pool[nspare_pool++] = r->pools[i];
        plat_mutex_unlock(&spare_lock);
        if (!kept) vkDestroyDescriptorPool(c->device, r->pools[i], NULL);
    }
    for (int i = 0; i < r->nchunks; i++) {
        plat_mutex_lock(&spare_lock);
        int kept = r->chunks[i].size == CHUNK_BYTES && nspare_chunk < SPARE_CHUNKS;
        if (kept) spare_chunk[nspare_chunk++] = r->chunks[i];
        plat_mutex_unlock(&spare_lock);
        if (!kept) vmaDestroyBuffer(c->vma, r->chunks[i].buf, r->chunks[i].alloc);
    }
    free(r->written), free(r->written_slot), free(r->pools), free(r->chunks), free(r);
}

void vk_exec_draw_new_cb(void) { memset(&B, 0, sizeof B); }

CbRes *cb_res(VkExec *ex) {
    if (cur && cur_fence == ex->fence) return cur;
    cur = calloc(1, sizeof *cur);
    if (!cur) rt_fatal("vk: out of memory");
    cur_fence = ex->fence;
    cur->fence = ex->fence;
    memset(&B, 0, sizeof B);
    vk_completion_add(ex->fence, cb_done, cur);
    return cur;
}

uint8_t *scratch(VkExec *ex, uint64_t size, VkBuffer *buf, VkDeviceSize *off) {
    CbRes *r = cb_res(ex);
    size = (size + 255) & ~255ull;
    Chunk *c = r->nchunks ? &r->chunks[r->nchunks - 1] : NULL;
    if (!c || c->used + size > c->size) {
        if (r->nchunks == r->cchunks) r->chunks = grow(r->chunks, &r->cchunks, sizeof *r->chunks);
        c = &r->chunks[r->nchunks];
        uint64_t n = size > CHUNK_BYTES ? size : CHUNK_BYTES;
        int reused = 0;
        if (n == CHUNK_BYTES) {
            plat_mutex_lock(&spare_lock);
            if (nspare_chunk) *c = spare_chunk[--nspare_chunk], reused = 1;
            plat_mutex_unlock(&spare_lock);
        }
        if (reused) {
            c->used = 0;
            r->nchunks++;
            *buf = c->buf;
            *off = 0;
            c->used = size;
            return c->map;
        }
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = n,
                                 .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                          VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        VkResult res = vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &c->buf, &c->alloc, &info);
        if (res != VK_SUCCESS) rt_fatal("vk: transient buffer of %llu bytes failed (%d)", (unsigned long long)n, (int)res);
        c->map = info.pMappedData, c->size = n, c->used = 0;
        r->nchunks++;
    }
    *buf = c->buf;
    *off = c->used;
    uint8_t *p = c->map + c->used;
    c->used += size;
    return p;
}

static uint32_t written_hash(uint64_t addr, uint64_t size) {
    uint64_t h = (addr ^ size * 0xc2b2ae3d27d4eb4full) * 0x9e3779b97f4a7c15ull;
    return (uint32_t)(h >> 32);
}

// Each distinct range once per command buffer, in first-write order (looked up through a table).
static void note_written(VkExec *ex, uint64_t addr, uint64_t size) {
    CbRes *r = cb_res(ex);
    if ((uint32_t)r->nwritten * 2 >= r->written_slots) {
        uint32_t slots = r->written_slots ? r->written_slots * 2 : 256;
        uint32_t *table = calloc(slots, sizeof *table);
        if (!table) rt_fatal("vk: out of memory");
        for (int i = 0; i < r->nwritten; i++) {
            uint32_t at = written_hash(r->written[i].addr, r->written[i].size) & (slots - 1);
            while (table[at]) at = (at + 1) & (slots - 1);
            table[at] = (uint32_t)i + 1;
        }
        free(r->written_slot);
        r->written_slot = table, r->written_slots = slots;
    }
    uint32_t mask = r->written_slots - 1, at = written_hash(addr, size) & mask;
    for (; r->written_slot[at]; at = (at + 1) & mask) {
        const Range *w = &r->written[r->written_slot[at] - 1];
        if (w->addr == addr && w->size == size) return;
    }
    if (r->nwritten == r->cwritten) r->written = grow(r->written, &r->cwritten, sizeof *r->written);
    r->written[r->nwritten++] = (Range){addr, size};
    r->written_slot[at] = (uint32_t)r->nwritten;
    vk_mem_gpu_pending(addr, size);
}

static int reset_mapped(void *context, uint64_t addr, uint32_t size) {
    (void)context;
    VkMemSpan span;
    return vk_mem_resolve(addr, size, &span);
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

// Whether two nonnull targets share a byte; their copies then need a barrier between them to keep the input order.
static int reset_targets_overlap(const uint64_t *pointers, uint32_t count) {
    uint64_t *sorted = malloc((size_t)count * sizeof *sorted);
    if (!sorted) rt_fatal("vk: out of memory checking indirect-reset targets");
    uint32_t n = 0;
    for (uint32_t i = 0; i < count; i++)
        if (pointers[i]) sorted[n++] = pointers[i];
    qsort(sorted, n, sizeof *sorted, cmp_u64);
    int overlap = 0;
    for (uint32_t i = 1; i < n && !overlap; i++) overlap = sorted[i] - sorted[i - 1] < NVN_INDIRECT_RESET_RECORD_BYTES;
    free(sorted);
    return overlap;
}

// Read the pointer list after submitting the prefix that may have written it, then validate every target before recording a write.
int vk_exec_reset_indirect(VkExec *ex, uint64_t list_addr, uint32_t descriptor_bytes, int64_t first, int64_t count) {
    NvnIndirectResetList list;
    if (!nvn_indirect_reset_list(list_addr, descriptor_bytes, first, count, &list)) return 0;
    uint64_t *pointers = malloc(list.bytes);
    if (!pointers) rt_fatal("vk: out of memory reading an indirect-reset pointer list");
    if (!vk_exec_read_buffer(ex, list.begin, list.bytes, pointers) ||
        !nvn_indirect_reset_targets(&list, pointers, reset_mapped, NULL)) {
        free(pointers);
        return 0;
    }
    VkBuffer source;
    VkDeviceSize source_offset;
    static const uint32_t record[5] = {0, 1, 0, 0, 0};
    memcpy(scratch(ex, sizeof record, &source, &source_offset), record, sizeof record);
    CbRes *resources = cb_res(ex);
    Chunk *chunk = &resources->chunks[resources->nchunks - 1];
    if (vmaFlushAllocation(vk_ctx()->vma, chunk->alloc, source_offset, sizeof record) != VK_SUCCESS)
        rt_fatal("vk: flushing indirect-reset values failed");
    int ordered = reset_targets_overlap(pointers, list.count);
    vk_exec_barrier(ex);   // the copies follow every earlier access to the records
    for (uint32_t i = 0; i < list.count; i++) {
        uint64_t addr = pointers[i];
        if (!addr) continue;
        VkMemSpan span;
        if (!vk_mem_resolve(addr, sizeof record, &span))
            rt_fatal("vk: an indirect-reset mapping changed during replay");
// Commit first-use imports to their mirror first so a later indirect bind reads the same backing.
        VK_MEM_USE(span);
        if (ordered) vk_exec_barrier(ex);   // preserves the input order of duplicate or overlapping target ranges
        VkBufferCopy copy = {source_offset, span.offset, sizeof record};
        vkCmdCopyBuffer(ex->cb, source, span.buffer, 1, &copy);
        ex->recorded++;
        vk_mem_gpu_wrote(&span, sizeof record);
        vk_usnap_drop(addr, sizeof record);
        vk_mem_count(span.cls, VK_MUSE_GPU_WRITE, sizeof record);
        note_written(ex, addr, sizeof record);
    }
    free(pointers);
    vk_exec_barrier(ex);   // subsequent shader accesses and indirect reads see every completed record
    return 1;
}

// ---- stand-ins for what a binding names but the game did not bind (zeros, a 1x1 image, the default sampler)

typeof(Z) Z;   // vk_exec_draw_int.h

static const struct { VkImageType t; VkImageViewType vt; uint32_t layers; VkSampleCountFlagBits s; VkImageCreateFlags f; } dinfo[DV_COUNT] = {
    {VK_IMAGE_TYPE_1D, VK_IMAGE_VIEW_TYPE_1D, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_1D, VK_IMAGE_VIEW_TYPE_1D_ARRAY, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_CUBE, 6, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY, 6, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, 1, VK_SAMPLE_COUNT_4_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 1, VK_SAMPLE_COUNT_4_BIT, 0}};

// A 1x1 stand-in image of view type dinfo[i] in `fmt`, moved to GENERAL by the upload batch. *sview: the storage view (a cube becomes a 2D array).
static void make_dummy(int i, VkFormat fmt, VkImage *image, VmaAllocation *alloc, VkImageView *view, VkImageView *sview) {
    VkCtx *c = vk_ctx();
    int ms = dinfo[i].s != VK_SAMPLE_COUNT_1_BIT;
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = dinfo[i].f, .imageType = dinfo[i].t,
                            .format = fmt, .extent = {1, 1, 1}, .mipLevels = 1, .arrayLayers = dinfo[i].layers,
                            .samples = dinfo[i].s, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | (ms ? 0 : VK_IMAGE_USAGE_STORAGE_BIT),
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ia = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ii, &ia, image, alloc, NULL) != VK_SUCCESS) rt_fatal("vk: stand-in image %d failed", i);
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = *image,
                                .viewType = dinfo[i].vt, .format = fmt,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, dinfo[i].layers}};
    if (vkCreateImageView(c->device, &vi, NULL, view) != VK_SUCCESS) rt_fatal("vk: stand-in view failed");
    *sview = *view;
    if (!ms && (dinfo[i].vt == VK_IMAGE_VIEW_TYPE_CUBE || dinfo[i].vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY)) {
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        if (vkCreateImageView(c->device, &vi, NULL, sview) != VK_SUCCESS) rt_fatal("vk: stand-in view failed");
    }
    VkImageMemoryBarrier2 bar = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                 .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                 .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                 .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                 .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                 .image = *image, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, dinfo[i].layers}};
    VkCommandBuffer cb = vk_upload_begin();
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &bar};
    vkCmdPipelineBarrier2(cb, &di);
    vk_upload_end();
}

// Storage image stand-ins in the format a shader declares (translator TextureFormat order); the view format must match.
static const VkFormat tex_format[40] = {
    VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SNORM, VK_FORMAT_R8_UINT, VK_FORMAT_R8_SINT,
    VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16_UNORM, VK_FORMAT_R16_SNORM, VK_FORMAT_R16_UINT, VK_FORMAT_R16_SINT,
    VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SNORM,
    VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_SINT, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_SNORM,
    VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16_SINT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SINT,
    VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_SINT,
    VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16B16A16_UINT,
    VK_FORMAT_R16G16B16A16_SINT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT,
    VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2B10G10R10_UINT_PACK32, VK_FORMAT_B10G11R11_UFLOAT_PACK32};
static struct { int used, di; VkFormat fmt; VkImage image; VmaAllocation alloc; VkImageView view, sview; } SD[48];

static VkImageView storage_dummy(int di, int code) {
    VkFormat fmt = code > 0 && code < 40 ? tex_format[code] : VK_FORMAT_R8G8B8A8_UNORM;
    if (fmt == VK_FORMAT_R8G8B8A8_UNORM || dinfo[di].s != VK_SAMPLE_COUNT_1_BIT) return Z.img[di].sview;
    for (int i = 0; i < 48; i++) {
        if (!SD[i].used) {
            SD[i].used = 1, SD[i].di = di, SD[i].fmt = fmt;
            make_dummy(di, fmt, &SD[i].image, &SD[i].alloc, &SD[i].view, &SD[i].sview);
            return SD[i].sview;
        }
        if (SD[i].di == di && SD[i].fmt == fmt) return SD[i].sview;
    }
    return Z.img[di].sview;
}

static void dummies_init(void) {
    if (Z.ready) return;
    VkCtx *c = vk_ctx();
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = ZERO_BYTES,
                             .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                      VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(c->vma, &bi, &ai, &Z.zbuf, &Z.zalloc, &info) != VK_SUCCESS) rt_fatal("vk: zero buffer failed");
    memset(info.pMappedData, 0, ZERO_BYTES);
    VkBufferViewCreateInfo bv = {.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, .buffer = Z.zbuf,
                                 .format = VK_FORMAT_R32G32B32A32_SFLOAT, .range = ZERO_BYTES};
    if (vkCreateBufferView(c->device, &bv, NULL, &Z.zview) != VK_SUCCESS) rt_fatal("vk: zero buffer view failed");
    samp_par sp;
    memset(&sp, 0, sizeof sp);
    sp.lod_max = 1000.0f;
    Z.samp = vk_sampler_for(&sp);
    sp.cmp_mode = 1, sp.cmp_func = NVN_DEPTH_FUNC_ALWAYS;
    Z.samp_shadow = vk_sampler_for(&sp);

    for (int i = 0; i < DV_COUNT; i++)
        make_dummy(i, VK_FORMAT_R8G8B8A8_UNORM, &Z.img[i].image, &Z.img[i].alloc, &Z.img[i].view, &Z.img[i].sview);
    Z.ready = 1;
}

static void twins_release(VkCtx *c);
static void mesh_release(VkCtx *c);

// Releases the stand-ins and the depth twins. The device must be idle.
static void c1_release(VkCtx *c);   // below: the compiler-constant buffers
void vk_exec_draw_shutdown(void) {
    VkCtx *c = vk_ctx();
    if (!c) return;
    twins_release(c);
    upscale_release();
    mesh_release(c);
    c1_release(c);
    for (int i = 0; Z.ready && i < DV_COUNT; i++) {
        if (Z.img[i].sview != Z.img[i].view) vkDestroyImageView(c->device, Z.img[i].sview, NULL);
        vkDestroyImageView(c->device, Z.img[i].view, NULL);
        vmaDestroyImage(c->vma, Z.img[i].image, Z.img[i].alloc);
    }
    for (int i = 0; i < 48 && SD[i].used; i++) {
        if (SD[i].sview != SD[i].view) vkDestroyImageView(c->device, SD[i].sview, NULL);
        vkDestroyImageView(c->device, SD[i].view, NULL);
        vmaDestroyImage(c->vma, SD[i].image, SD[i].alloc);
    }
    memset(SD, 0, sizeof SD);
    nvn_hm_clear(&rs_lists, free);
    rs_last = NULL, rs_last_l = NULL;
    for (int i = 0; i < nspare_pool; i++) vkDestroyDescriptorPool(c->device, spare_pool[i], NULL);
    for (int i = 0; i < nspare_chunk; i++) vmaDestroyBuffer(c->vma, spare_chunk[i].buf, spare_chunk[i].alloc);
    nspare_pool = nspare_chunk = 0;
    cur = NULL, cur_fence = 0;
    vk_usnap_shutdown();
    if (Z.ready) {
        vkDestroyBufferView(c->device, Z.zview, NULL);
        vmaDestroyBuffer(c->vma, Z.zbuf, Z.zalloc);
    }
    memset(&Z, 0, sizeof Z);
}

// ---- resource types

static VkImageViewType res_view_type(uint16_t t) {
    int arr = t & 0x100;
    switch (t & 0xff) {
    case 1: return arr ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
    case 4: return VK_IMAGE_VIEW_TYPE_3D;
    case 5: return arr ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE;
    default: return arr ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    }
}

static int dummy_index(VkImageViewType vt, int ms) {
    switch (vt) {
    case VK_IMAGE_VIEW_TYPE_1D: return DV_1D;
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY: return DV_1D_ARRAY;
    case VK_IMAGE_VIEW_TYPE_3D: return DV_3D;
    case VK_IMAGE_VIEW_TYPE_CUBE: return DV_CUBE;
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: return DV_CUBE_ARRAY;
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY: return ms ? DV_2D_MS_ARRAY : DV_2D_ARRAY;
    default: return ms ? DV_2D_MS : DV_2D;
    }
}

// GFX_TEX_* of a bound texture slot (the translator's tex_type / img_type answers).
static int gfx_type_of(const VkTexImage *img, const view_t *v, int hv) {
    if (hv && v->set_target) return vk_pipeline_tex_type(v->target);
    int ms = img->samples > 1;
    switch (img->view_type) {
    case VK_IMAGE_VIEW_TYPE_1D: return GFX_TEX_1D;
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY: return GFX_TEX_1D_ARRAY;
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY: return ms ? GFX_TEX_2D_MS_ARRAY : GFX_TEX_2D_ARRAY;
    case VK_IMAGE_VIEW_TYPE_3D: return GFX_TEX_3D;
    case VK_IMAGE_VIEW_TYPE_CUBE: return GFX_TEX_CUBE;
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: return GFX_TEX_CUBE_ARRAY;
    default: return ms ? GFX_TEX_2D_MS : GFX_TEX_2D;
    }
}

static VkComponentSwizzle swz_of(uint8_t s) {
    switch (s) {
    case NVN_TEXTURE_SWIZZLE_ZERO: return VK_COMPONENT_SWIZZLE_ZERO;
    case NVN_TEXTURE_SWIZZLE_ONE: return VK_COMPONENT_SWIZZLE_ONE;
    case NVN_TEXTURE_SWIZZLE_R: return VK_COMPONENT_SWIZZLE_R;
    case NVN_TEXTURE_SWIZZLE_G: return VK_COMPONENT_SWIZZLE_G;
    case NVN_TEXTURE_SWIZZLE_B: return VK_COMPONENT_SWIZZLE_B;
    default: return VK_COMPONENT_SWIZZLE_A;
    }
}

// A STENCIL-mode view samples the stencil plane, otherwise depth; the translator declares every sampled image float.
VkImageAspectFlags vk_exec_view_aspect(const VkTexImage *img, const view_t *v, int hv) {
    if (hv && v->set_dsmode && v->dsmode == NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL && (img->aspect & VK_IMAGE_ASPECT_STENCIL_BIT))
        return VK_IMAGE_ASPECT_STENCIL_BIT;
    return img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT     ? VK_IMAGE_ASPECT_DEPTH_BIT
           : img->aspect & VK_IMAGE_ASPECT_STENCIL_BIT ? VK_IMAGE_ASPECT_STENCIL_BIT
                                                       : VK_IMAGE_ASPECT_COLOR_BIT;
}

// Image views of registered textures, kept for their image's lifetime. Texture lock held; vk_exec_image_gone hands a
// destroyed image's views to the deferred release.
typedef struct ViewNode { struct ViewNode *next; ViewKey k; VkImageView view; } ViewNode;
static nvn_hmap view_of;                 // ViewKey -> VkImageView
static nvn_hmap views_by_image;          // VkImage -> its first ViewNode

// An image view of a registered texture for a shader resource of type `rtype` (GfxShaderRes.type); VK_NULL_HANDLE when
// the image cannot serve it (the caller binds a stand-in). Texture lock held.
VkImageView get_view(VkExec *ex, const VkTexImage *img, const view_t *v, int hv, uint16_t rtype, int storage) {
    (void)ex;
    VkFormat format = img->format;
    const uint8_t *channels = img->chan;
    int alpha_one = img->alpha_one;
    TexVkFormat mapped;
    // NVNtextureView's format is a bit reinterpretation: packed RGB data goes through R32_UINT views.
    if (hv && v->set_format && img->aspect == VK_IMAGE_ASPECT_COLOR_BIT) {
        if (!tex_vk_format(v->format, &mapped) || !vk_color_view_compatible(img->format, mapped.vk)) return VK_NULL_HANDLE;
        format = mapped.vk, channels = mapped.chan, alpha_one = mapped.alpha_one;
    }
    if (storage && vk_unorm_of_srgb(format)) format = vk_unorm_of_srgb(format);   // sRGB stored through UNORM (vk_texture.c)
    VkImageViewType vt = res_view_type(rtype);
    int ms = (rtype & 0x200) != 0;
    if ((img->samples > 1) != ms || (storage && ms)) return VK_NULL_HANDLE;
    if (storage && !(img->usage & VK_IMAGE_USAGE_STORAGE_BIT)) return VK_NULL_HANDLE;
    int cube = vt == VK_IMAGE_VIEW_TYPE_CUBE || vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    int ok = 0;
    switch (vt) {
    case VK_IMAGE_VIEW_TYPE_1D:
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY: ok = img->type == VK_IMAGE_TYPE_1D; break;
    case VK_IMAGE_VIEW_TYPE_3D: ok = img->type == VK_IMAGE_TYPE_3D; break;
    case VK_IMAGE_VIEW_TYPE_CUBE:
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
        ok = img->type == VK_IMAGE_TYPE_2D && (img->view_type == VK_IMAGE_VIEW_TYPE_CUBE || img->view_type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY);
        break;
    default: ok = img->type == VK_IMAGE_TYPE_2D; break;
    }
    if (!ok) return VK_NULL_HANDLE;
    uint32_t l0 = hv && v->set_levels ? v->base_level : 0;
    if (l0 >= img->levels) return VK_NULL_HANDLE;
    uint32_t nl = hv && v->set_levels && v->nlevels ? v->nlevels : img->levels - l0;
    if (l0 + nl > img->levels) nl = img->levels - l0;
    if (storage) nl = 1;
    uint32_t y0 = 0, ny = 1;
    if (vt != VK_IMAGE_VIEW_TYPE_3D) {
        y0 = hv && v->set_layers ? v->min_layer : 0;
        if (y0 >= img->layers) return VK_NULL_HANDLE;
        uint32_t rest = hv && v->set_layers && v->nlayers ? v->nlayers : img->layers - y0;
        if (y0 + rest > img->layers) rest = img->layers - y0;
        if (vt == VK_IMAGE_VIEW_TYPE_1D || vt == VK_IMAGE_VIEW_TYPE_2D) ny = 1;
        else if (vt == VK_IMAGE_VIEW_TYPE_CUBE) ny = 6;
        else if (vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) ny = rest / 6 * 6;
        else ny = rest;
        if (!ny || y0 + ny > img->layers) return VK_NULL_HANDLE;
    }
    if (storage && cube) vt = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    uint32_t aspect = vk_exec_view_aspect(img, v, hv);
    uint8_t sw[4] = {NVN_TEXTURE_SWIZZLE_R, NVN_TEXTURE_SWIZZLE_G, NVN_TEXTURE_SWIZZLE_B, NVN_TEXTURE_SWIZZLE_A};
    if (!storage) {   // view selectors replace the texture's, then the format remap
        memcpy(sw, hv && v->set_swizzle ? v->swz : img->swz, 4);
        if (alpha_one || channels[0])
            for (int i = 0; i < 4; i++) {
                if (alpha_one && sw[i] == NVN_TEXTURE_SWIZZLE_A) sw[i] = NVN_TEXTURE_SWIZZLE_ONE;
                else if (channels[0] && sw[i] >= NVN_TEXTURE_SWIZZLE_R && sw[i] <= NVN_TEXTURE_SWIZZLE_A)
                    sw[i] = channels[sw[i] - NVN_TEXTURE_SWIZZLE_R];
            }
    }
    uint32_t swz = (uint32_t)sw[0] | (uint32_t)sw[1] << 8 | (uint32_t)sw[2] << 16 | (uint32_t)sw[3] << 24;
    // NVN selectors use three bits per byte; bit 31 separates storage-only views.
    ViewKey k = {img->image, format, vt, aspect, l0, nl, y0, ny, swz | (storage ? 0x80000000u : 0)};
    if (!view_of.klen) nvn_hm_init(&view_of, sizeof(ViewKey)), nvn_hm_init(&views_by_image, sizeof(VkImage));
    VkImageView found = (VkImageView)nvn_hm_get(&view_of, &k);
    if (found) return found;
    VkFormatProperties properties;
    vkGetPhysicalDeviceFormatProperties(vk_ctx()->phys, format, &properties);
    VkFormatFeatureFlags required = storage ? VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT : VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    if (!(properties.optimalTilingFeatures & required)) return VK_NULL_HANDLE;
    VkImageViewCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img->image, .viewType = vt,
                                .format = format,
                                .components = {swz_of(sw[0]), swz_of(sw[1]), swz_of(sw[2]), swz_of(sw[3])},
                                .subresourceRange = {aspect, l0, nl, y0, ny}};
    VkImageViewUsageCreateInfo use = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
                                      .usage = storage ? VK_IMAGE_USAGE_STORAGE_BIT : VK_IMAGE_USAGE_SAMPLED_BIT};
    ci.pNext = &use;   // extended image usages need not be supported by every view format
    VkImageView view;
    if (vkCreateImageView(vk_ctx()->device, &ci, NULL, &view) != VK_SUCCESS) return VK_NULL_HANDLE;
    ViewNode *n = malloc(sizeof *n);
    if (!n) rt_fatal("vk: out of memory");
    *n = (ViewNode){nvn_hm_get(&views_by_image, &img->image), k, view};
    nvn_hm_put(&views_by_image, &img->image, n);
    nvn_hm_put(&view_of, &k, (void *)view);
    return view;
}

// An attachment view of a registered texture, kept for the image's lifetime. Attachment keys carry a swizzle word no
// sampled key has (those hold four selectors, each below 8). Texture lock guards the tables; vk_exec_pass_begin may hold it.
VkImageView vk_exec_target_view(const VkTexImage *img, VkFormat format, VkImageViewType type, uint32_t aspect,
                                uint32_t level, uint32_t base, uint32_t layers, int blocks) {
    ViewKey k = {img->image, format, type, aspect, level, 1, base, layers, blocks ? 0xfffffffeu : 0xffffffffu};
    vk_texture_lock();
    if (!view_of.klen) nvn_hm_init(&view_of, sizeof(ViewKey)), nvn_hm_init(&views_by_image, sizeof(VkImage));
    VkImageView view = (VkImageView)nvn_hm_get(&view_of, &k);
    if (!view) {
        VkImageViewCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img->image,
                                    .viewType = type, .format = format, .subresourceRange = {aspect, level, 1, base, layers}};
        VkImageViewUsageCreateInfo use = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
                                          .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
        if (blocks) ci.pNext = &use;   // not the BC image's sampling and transfer usages
        // A view in another format drops the image's storage usage, which that format may lack (create_tex gives
        // every colour texture that allows it storage usage).
        else if (format != img->format && (img->usage & VK_IMAGE_USAGE_STORAGE_BIT))
            use.usage = img->usage & ~VK_IMAGE_USAGE_STORAGE_BIT, ci.pNext = &use;
        if (vkCreateImageView(vk_ctx()->device, &ci, NULL, &view) != VK_SUCCESS)
            rt_fatal("vk: render target view (format %d) could not be created", (int)format);
        ViewNode *n = malloc(sizeof *n);
        if (!n) rt_fatal("vk: out of memory");
        *n = (ViewNode){nvn_hm_get(&views_by_image, &img->image), k, view};
        nvn_hm_put(&views_by_image, &img->image, n);
        nvn_hm_put(&view_of, &k, (void *)view);
    }
    vk_texture_unlock();
    return view;
}

int rf_mesh;   // record_fixed is recording a mesh-shader draw (vk_exec_mesh.h)
// DLSS projection jitter of the draw being recorded: ubo_info adds (rf_pj[0], rf_pj[1]) x the w row to the x and y rows
// of its camera block.
int rf_proj;
float rf_pj[2];
uint32_t ub_gen;   // bumped by every inline uniform update (op_update_ub): a patched camera block is stale then

void vk_exec_image_gone(VkImage image) {
    vk_exec_bc_gone(image);
    bl_image_gone(image);
    dlss_image_gone(image);
    if (!views_by_image.klen) return;
    for (ViewNode *n = nvn_hm_del(&views_by_image, &image), *next; n; n = next) {
        next = n->next;
        nvn_hm_del(&view_of, &n->k);
        vk_defer_view(n->view);
        free(n);
    }
}

// ---- program to shaders

// (program, stage, answers) -> shader, 4-way set-associative, most recently used first. A miss costs a SHA-256 of the shader code.
#define SHADER_SETS 2048
#define SHADER_WAYS 4
typedef struct { uint64_t serial; int stage; int used; GfxShaderAnswers a; const GfxShader *s; } ShaderEnt;
static ShaderEnt shader_tab[SHADER_SETS][SHADER_WAYS];

// A failed lowering leaves the compact native control, whose stage and length are not at the legacy offsets.
static int program_stage_info(const NVNshaderData *data, uint32_t *stage, uint32_t *length) {
    const uint8_t *ctl = data->control, *code = (const uint8_t *)(uintptr_t)data->data;
    if (!ctl || !code) return 0;
    if (gfx_shader_is_nvn2(ctl)) {
        uint32_t w0;
        memcpy(length, ctl + 0x38, 4);
        if (*length < 4) return 0;
        memcpy(&w0, code, 4);
        int graphics = ((w0 & 31) == 1 || (w0 & 31) == 2) && ((w0 >> 5) & 31) == 4;
        if (!graphics) { *stage = NVN_STAGE_CS; return 1; }
        switch ((w0 >> 10) & 15) {
        case 1: *stage = NVN_STAGE_VS; break;
        case 2: *stage = 3; break;   // tessellation control, as nvn2_lower
        case 3: *stage = 4; break;   // tessellation evaluation; mesh mode still needs explicit context
        case 4: *stage = NVN_STAGE_GS; break;
        case 5: *stage = NVN_STAGE_FS; break;
        default: return 0;
        }
    } else {
        memcpy(stage, ctl + 0x714, 4);
        memcpy(length, ctl + 0x704, 4);
    }
    return *stage < 6;
}

static int find_stage_scan(const prog_t *p, int nvn_stage, const uint8_t **ctl, const uint8_t **code, uint32_t *len) {
    for (uint32_t i = 0; i < p->nstage && i < 6; i++) {
        const uint8_t *c = p->sd[i].control;
        if (!c || !p->sd[i].data) continue;
        uint32_t st, length;
        if (!program_stage_info(&p->sd[i], &st, &length)) continue;
        if ((int)st != nvn_stage) continue;
        *len = length;
        *ctl = c, *code = (const uint8_t *)(uintptr_t)p->sd[i].data;
        return 1;
    }
    return 0;
}

// Shader and code length serving each NVN stage, per program serial. Direct-mapped; a collision only scans again.
#define STAGEM_N 512
static struct { uint64_t serial; int8_t idx[6]; uint32_t len[6]; } stagem[STAGEM_N];

static int find_stage(const prog_t *p, int nvn_stage, const uint8_t **ctl, const uint8_t **code, uint32_t *len) {
    if (!p->serial || (unsigned)nvn_stage >= 6) return find_stage_scan(p, nvn_stage, ctl, code, len);
    typeof(stagem[0]) *m = &stagem[p->serial * 0x9E3779B97F4A7C15ull >> 55];
    if (m->serial != p->serial) {
        m->serial = p->serial;
        memset(m->idx, -1, sizeof m->idx);
        for (uint32_t i = 0; i < p->nstage && i < 6; i++) {
            const uint8_t *c = p->sd[i].control;
            if (!c || !p->sd[i].data) continue;
            uint32_t st, length;
            if (!program_stage_info(&p->sd[i], &st, &length)) continue;
            if (m->idx[st] < 0) m->idx[st] = (int8_t)i, m->len[st] = length;
        }
    }
    int i = m->idx[nvn_stage];
    if (i < 0) return 0;
    *ctl = p->sd[i].control, *code = (const uint8_t *)(uintptr_t)p->sd[i].data, *len = m->len[nvn_stage];
    return 1;
}

// Texture and image types of a stage's first 32 slots, each kept with the handle and the generation and slot token it
// was looked up under (see slot_still).
static struct { uint64_t context, tex[32], img[32], ttok[32], itok[32]; uint32_t tgen[32], igen[32]; uint8_t tt[32], it[32], ifmt[32]; } AC[6];
static uint64_t descriptor_serial = 1;

// vk_texture_slot_token of the slot a texture or image handle names; an empty handle always resolves to nothing.
static uint64_t token_of(const VkExec *ex, uint64_t h) {
    return h ? vk_texture_slot_token(&ex->tex_pool, (int)(h & 0xfffff)) : 1;
}

static void descriptor_state(VkExec *ex);

// Whether what handle h resolved to under generation *egen and slot token etok still holds at generation gen: nothing
// changed, or not its slot. A token hit moves *egen to gen.
static int slot_still(const VkExec *ex, uint32_t *egen, uint64_t etok, uint64_t h, uint32_t gen) {
    if (*egen == gen) return 1;
    uint64_t t = token_of(ex, h);
    if (!t || t != etok) return 0;
    *egen = gen;
    return 1;
}

// The slots of (program, stage, GS input topology) whose types the translation reads, learned from its first shader;
// ~0u while unknown. answers_for leaves the other slots zero. Direct-mapped by serial and stage.
#define SLOTM_N 4096
static struct { uint64_t serial; int st; uint32_t m; } slotm[SLOTM_N];

static uint32_t slotm_idx(uint64_t serial, int st) { return (uint32_t)((serial * 8 + (uint64_t)st) * 0x9E3779B97F4A7C15ull >> 52); }

static int slot_mask_stage(int st) { return st == NVN_STAGE_GS ? st + 8 * geometry_input : st; }

static uint32_t slot_mask_of(uint64_t serial, int st) {
    st = slot_mask_stage(st);                           // GS topology can change which texture queries survive translation
    const typeof(slotm[0]) *e = &slotm[slotm_idx(serial, st)];
    return serial && e->serial == serial && e->st == st ? e->m : ~0u;
}

static void slot_mask_learn(uint64_t serial, int st, const GfxShader *s) {
    if (!serial || slot_mask_of(serial, st) != ~0u) return;
    st = slot_mask_stage(st);
    slotm[slotm_idx(serial, st)] = (typeof(slotm[0])){serial, st, gfx_shader_slot_mask(s)};
}

static uint8_t type_of_handle(const VkExec *ex, uint64_t h) {   // texture lock held
    view_t v;
    int hv;
    const VkTexImage *img = h ? vk_texture_by_id_locked(&ex->tex_pool, (int)(h & 0xfffff), &v, &hv) : NULL;
    return img ? (uint8_t)gfx_type_of(img, &v, hv) : 0;
}

// Storage instructions need the effective view format; unknown, the translator declares integer images as float.
static uint8_t format_of_handle(const VkExec *ex, uint64_t h) {   // texture lock held
    view_t v;
    int hv;
    const VkTexImage *img = h ? vk_texture_by_id_locked(&ex->tex_pool, (int)(h & 0xfffff), &v, &hv) : NULL;
    if (!img) return 0;
    VkFormat format = img->format;
    TexVkFormat mapped;
    if (hv && v.set_format && img->aspect == VK_IMAGE_ASPECT_COLOR_BIT &&
        tex_vk_format(v.format, &mapped) && vk_color_view_compatible(format, mapped.vk)) format = mapped.vk;
    for (int i = 1; i < 40; i++) if (tex_format[i] == format) return (uint8_t)i;
    return 0;
}

static uint8_t sampled_type_of_handle(const VkExec *ex, uint64_t h) {
    uint8_t type = type_of_handle(ex, h);
    switch (format_of_handle(ex, h)) {
    case 3: case 8: case 11: case 15: case 20: case 23: case 27: case 32: case 35: case 38:
        return type ? type | 0x40 : 0;
    case 4: case 9: case 12: case 16: case 21: case 24: case 28: case 33: case 36:
        return type ? type | 0x80 : 0;
    default: return type;
    }
}

// The answers of stage st for the slots in mask; the types of other slots stay zero.
static void answers_for(VkExec *ex, int st, const uint8_t *ctl, uint32_t mask, GfxShaderAnswers *a) {
    descriptor_state(ex);
    memset(a, 0, sizeof *a);
    if (st == NVN_STAGE_VS) vk_pipeline_answers(&ex->pipe, a, NULL);
    else if (st == NVN_STAGE_FS) vk_pipeline_answers(&ex->pipe, NULL, a);
    else if (st == NVN_STAGE_GS) a->geometry_input = geometry_input;
    else if (st == NVN_STAGE_CS)
        for (int k = 0; k < 3; k++) {
            uint32_t l;
            memcpy(&l, ctl + 0x748 + 4 * k, 4);
            a->local[k] = (uint16_t)(l ? l : 1);
        }
    typeof(AC[0]) *c = &AC[st];
    if (c->context != descriptor_serial) {
        memset(c, 0, sizeof *c);
        c->context = descriptor_serial;
    }
    // Generation and tokens are read before the lookups, unlocked: a racing registration is ordered after this draw.
    uint32_t gen = vk_texture_slots_gen();
    int locked = 0;
    for (uint32_t m = mask; m; m &= m - 1) {
        int slot = __builtin_ctz(m);
        uint64_t h = ex->tex[st][slot], hi = ex->img[st][slot];
        int tmiss = c->tex[slot] != h || !c->ttok[slot] || !slot_still(ex, &c->tgen[slot], c->ttok[slot], h, gen);
        int imiss = c->img[slot] != hi || !c->itok[slot] || !slot_still(ex, &c->igen[slot], c->itok[slot], hi, gen);
        uint64_t tt = tmiss ? token_of(ex, h) : 0, it = imiss ? token_of(ex, hi) : 0;
        if ((tmiss || imiss) && !locked) vk_texture_lock(), locked = 1;
        if (tmiss) c->tt[slot] = sampled_type_of_handle(ex, h), c->tex[slot] = h, c->ttok[slot] = tt, c->tgen[slot] = gen;
        if (imiss) c->it[slot] = type_of_handle(ex, hi), c->ifmt[slot] = format_of_handle(ex, hi),
                   c->img[slot] = hi, c->itok[slot] = it, c->igen[slot] = gen;
        a->tex_type[slot] = c->tt[slot], a->img_type[slot] = c->it[slot], a->img_format[slot] = c->ifmt[slot];
    }
    if (locked) vk_texture_unlock();
}

// Lowered native passthrough geometry stages run through the portable path. The varying skip mask (Maxwell register
// 0x1240) is inferred from what the vertex stage writes and the fragment stage reads.
static int native_gs_on(void) {
    static int logged;
    if (!logged) {
        logged = 1;
        rt_log("native GS: Bananza passthrough geometry enabled");
    }
    return 1;
}

static const GfxShader *native_gs_vs, *native_gs_fs;   // the draw's stages, set before its geometry lookup

static const uint8_t *native_gs_control(const uint8_t *ctl, const uint8_t *code, uint32_t len) {
    // nvn2_lower.cpp keeps these source controls for the process lifetime; keep the packed copies as long (shader_for borrows them).
    static struct { const uint8_t *ctl; uint32_t mask; uint8_t *copy; } *made;
    static size_t count, capacity;
    GfxShaderIo io[32];
    const uint32_t fs_in = gfx_shader_inputs(native_gs_fs, io), vs_out = gfx_shader_outputs(native_gs_vs, io);
    const uint32_t mask = fs_in & vs_out;
    for (size_t i = 0; i < count; i++)
        if (made[i].ctl == ctl && made[i].mask == mask) return made[i].copy;
    uint32_t skip[8];
    skip[0] = 0x0fffffffu;
    for (unsigned i = 1; i < 8; i++) skip[i] = UINT32_MAX;
    for (unsigned l = 0; l < 32; l++)
        if (mask >> l & 1) skip[1 + l / 8] &= ~(15u << (l % 8 * 4));
    uint8_t *copy = malloc(0x880);
    if (!copy) return NULL;
    memcpy(copy, ctl, 0x880);
    if (!nvn_geometry_control_pack_lowered(copy, 0x880, code, len, skip)) {
        rt_log("native GS: stage %016llx does not fit the portable control", (unsigned long long)nvn2_origin(code));
        free(copy);
        return NULL;
    }
    if (count == capacity) {
        if (capacity > SIZE_MAX / 2) { free(copy); return NULL; }
        size_t next = capacity ? capacity * 2 : 256;
        if (next > SIZE_MAX / sizeof *made) { free(copy); return NULL; }
        void *grown = realloc(made, next * sizeof *made);
        if (!grown) { free(copy); return NULL; }
        made = grown, capacity = next;
    }
    made[count].ctl = ctl, made[count].mask = mask, made[count].copy = copy;
    count++;
    rt_log("native GS: stage %016llx forwards locations %#x (vertex writes %#x, fragment reads %#x)",
           (unsigned long long)nvn2_origin(code), mask, vs_out, fs_in);
    return copy;
}

// *present: the program has this stage. NULL when it has not or the translation failed. pending (graphics stages,
// vk_pipeline_async_on): a shader not translated yet is queued, NULL returned with *pending = 1. await: wait for a queued one.
static const GfxShader *shader_for(VkExec *ex, int st, int *present, int *pending, int await) {
    const uint8_t *ctl, *code;
    uint32_t len;
    *present = 0;
    if (pending) *pending = 0;
    if (!ex->prog || !find_stage(ex->prog, st, &ctl, &code, &len)) return NULL;
    *present = 1;
    if (gfx_shader_is_nvn2(ctl)) {
        // Program creation rejected this native stage; answers_for needs a full legacy-shaped control block.
        warn_once(W_SHADER, "unlowered native stage %d of program %llu (%016llx): draws with it are skipped", st,
                  (unsigned long long)ex->prog->serial, (unsigned long long)gfx_shader_nvn2_hash(ctl, code));
        return NULL;
    }
    if (st == NVN_STAGE_GS && native_gs_on() && nvn2_origin(code)) {
        if (!native_gs_vs || !native_gs_fs) {   // its varyings follow from the other stages, which are still pending
            if (pending) *pending = 1;
            return NULL;
        }
        if (!(ctl = native_gs_control(ctl, code, len))) return NULL;
    }
    GfxShaderAnswers a;
    uint64_t serial = ex->prog->serial;
    answers_for(ex, st, ctl, slot_mask_of(serial, st), &a);
    ShaderEnt *set = NULL;
    if (serial) {
        uint64_t h = (serial * 0x9E3779B97F4A7C15ull ^ (uint64_t)st) ^ nvn_hm_hash(&a, sizeof a);   // a is zero-padded
        set = shader_tab[(h ^ h >> 29) & (SHADER_SETS - 1)];
        for (int w = 0; w < SHADER_WAYS && set[w].used; w++) {
            ShaderEnt *e = &set[w];
            if (e->serial != serial || e->stage != st || memcmp(&e->a, &a, sizeof a)) continue;
            if (w) {   // to the front
                ShaderEnt hit = *e;
                memmove(&set[1], &set[0], w * sizeof *set);
                set[0] = hit;
            }
            return set[0].s;
        }
    }
    int gs = st == NVN_STAGE_VS ? GFX_STAGE_VS : st == NVN_STAGE_FS ? GFX_STAGE_FS :
             st == NVN_STAGE_GS ? GFX_STAGE_GS : st == NVN_STAGE_CS ? GFX_STAGE_CS : -1;
    if (gs < 0) return NULL;
    const GfxShader *s = await ? gfx_shader_await(gs, ctl, code, len, &a)
                       : pending && vk_pipeline_async_on() ? gfx_shader_get_async(gs, ctl, code, len, &a, pending)
                                                           : gfx_shader_get(gs, ctl, code, len, NULL, &a);
    if (pending && *pending) return NULL;
    if (s) slot_mask_learn(serial, st, s);
    if (!s) warn_once(W_SHADER, "a stage %d shader of program %llu failed to translate: draws with it are skipped", st,
                      (unsigned long long)serial);
    if (!s) {   // log each untranslated stage once
        static uint64_t told[256];
        static int ntold;
        uint64_t id = gfx_shader_nvn2_hash(ctl, code), key = (id ? id : serial) * 31 + (uint64_t)st;
        int seen = 0;
        for (int i = 0; i < ntold && !seen; i++) seen = told[i] == key;
        if (!seen && ntold < 256) {
            told[ntold++] = key;
            rt_log("vk_exec: no translation for stage %d, nvn2 %016llx, program %llu: draws skipped, first in pass %ux%u",
                   st, (unsigned long long)id, (unsigned long long)serial, ex->pass_w, ex->pass_h);
        }
    }
    else if (set) {   // in front, the least recently used way drops out
        memmove(&set[1], &set[0], (SHADER_WAYS - 1) * sizeof *set);
        set[0] = (ShaderEnt){.serial = serial, .stage = st, .used = 1, .a = a, .s = s};
    }
    return s;
}

// ---- descriptors

static int nvn_stage_of(int gfx_stage) {
    return gfx_stage == GFX_STAGE_VS ? NVN_STAGE_VS : gfx_stage == GFX_STAGE_FS ? NVN_STAGE_FS :
           gfx_stage == GFX_STAGE_GS ? NVN_STAGE_GS : gfx_stage == GFX_STAGE_CS ? NVN_STAGE_CS :
           gfx_stage == GFX_STAGE_TASK ? 6 : gfx_stage == GFX_STAGE_MESH ? 7 : -1;
}

static VkDescriptorBufferInfo zero_info(uint64_t max) {
    return (VkDescriptorBufferInfo){Z.zbuf, 0, max < ZERO_BYTES ? max : ZERO_BYTES};
}

// vk_mem_count, also kept per (class, use) while a graphics build runs (RS).
static void count_use(int cls, int use, uint64_t n) {
    vk_mem_count(cls, use, n);
    if (!RS.rec) return;
    for (int i = 0; i < RS.ncnt; i++)
        if (RS.cnt[i].cls == cls && RS.cnt[i].use == use) {
            RS.cnt[i].bytes += n;
            return;
        }
    if (RS.ncnt < (int)(sizeof RS.cnt / sizeof RS.cnt[0])) RS.cnt[RS.ncnt++] = (RsCount){cls, use, n};
    else RS.cnt_over = 1;
}

// A buffer descriptor for [addr, addr+size) of guest memory: its backing when aligned, else a transient copy. 0 when
// neither works. written: the shader stores into it. direct_only: guarded aliases must share the real backing.
static int bind_range(VkExec *ex, uint64_t addr, uint64_t size, uint64_t align, uint64_t max, int use, int written,
                      int direct_only, VkDescriptorBufferInfo *out) {
    VkMemSpan sp;
    if (!addr || !vk_mem_resolve(addr, 1, &sp)) return 0;
    uint64_t n = size && size < sp.avail ? size : sp.avail;
    if (n > max) n = max;
    if (!n) return 0;
    if (sp.offset % align == 0) {
        VK_MEM_USE(sp);
        *out = (VkDescriptorBufferInfo){sp.buffer, sp.offset, n};
        if (written) {
            vk_mem_gpu_wrote(&sp, n);
            vk_usnap_drop(addr, n);
        }
        count_use(sp.cls, use, n);
        VK_MEM_BOUND(sp);
        return 1;
    }
    if (sp.alloc || direct_only) return 0;
    VkBuffer b;
    VkDeviceSize o;
    RS.volatile_ = 1;
    memcpy(scratch(ex, n, &b, &o), (const void *)(uintptr_t)addr, n);
    vk_mem_count(sp.cls, VK_MUSE_CPU_READ, n);
    vk_mem_count(VK_MCLS_SCRATCH, use, n);
    *out = (VkDescriptorBufferInfo){b, o, n};
    return 1;
}

static void c0_table(uint8_t *at, const p_bind_buf b[16]) {
    for (int r = 0; r < 16; r++) {
        uint64_t a = b[r].addr;
        uint32_t n = (uint32_t)b[r].size;
        memcpy(at + 16 * r, &a, 8);
        memcpy(at + 16 * r + 8, &n, 4);
    }
}

// Driver constant bank c0 as the compiled shaders read it: storage tables {u64 address, u32 size} at 0x110 (vertex),
// 0x510 (fragment), 0x310 (compute); compute dispatch size 0x200, uniform table 0x210; graphics max point size 0x610, base instance 0x644.
void vk_c0_fill(uint8_t bank[VK_C0_BYTES], int stage, const p_bind_buf ssbo[16], const p_bind_buf ubo[16],
                const uint32_t dispatch[3], uint32_t base_instance) {
    if (stage == NVN_STAGE_CS && c0_lay) {
        if (!nvn2_c0_pack(bank, VK_C0_BYTES, c0_lay, c0_push, ssbo, ubo))
            rt_fatal("vk: invalid NVN 2 compute parameter-block layout");
        memcpy(bank + 4, dispatch, 12);   // driver head: the group counts, read at c0+4 and c0+8 for the grid size
        return;
    }
    memset(bank, 0, VK_C0_BYTES);
    int base =stage == NVN_STAGE_VS ? 0x110 : stage == NVN_STAGE_FS ? 0x510 : stage == NVN_STAGE_CS ? 0x310 :
              stage == NVN_STAGE_GS ? 0x410 : -1;   // geometry: the table ssbo_slot expects
    if (base < 0) return;
    c0_table(bank + base, ssbo);
    if (stage == NVN_STAGE_CS) {
        memcpy(bank + 0x200, dispatch, 12);
        c0_table(bank + 0x210, ubo);
        return;
    }
    const float max_point = 2047.0f;
    memcpy(bank + 0x610, &max_point, 4);
    memcpy(bank + 0x644, &base_instance, 4);
}

// c0 from replay scratch; a bank equal to the stage's previous one reuses its range, so it costs no copy.
static VkDescriptorBufferInfo c0_info(VkExec *ex, int st) {
    if (st != NVN_STAGE_VS && st != NVN_STAGE_FS && st != NVN_STAGE_CS && st != NVN_STAGE_GS)
        rt_fatal("vk_exec: synthetic c0 is unavailable for stage %d", st);
    int k = st == NVN_STAGE_VS ? 0 : st == NVN_STAGE_FS ? 1 : st == NVN_STAGE_GS ? 3 : 2;
    // Equal inputs give the same bank, so neither the fill nor the 2 KiB compare is needed.
    typeof(B.c0_in[0]) in;
    memset(&in, 0, sizeof in);
    memcpy(in.ssbo, ex->ssbo[st], sizeof in.ssbo);
    if (st == NVN_STAGE_CS) memcpy(in.ubo, ex->ubo[st], sizeof in.ubo), memcpy(in.dim, c0_dim, sizeof in.dim);
    else in.binst = c0_binst;
    if (st == NVN_STAGE_CS && c0_lay) {
        in.lay = *c0_lay, in.has_lay = 1;
        for (unsigned i = 0; i < NVN2_ROOT_TABLES; i++)
            memcpy(in.roots[i], ex->push[1][i], 4u * c0_lay->roots[i]);
    }
    if (B.c0_valid[k] && !memcmp(&in, &B.c0_in[k], sizeof in))
        return (VkDescriptorBufferInfo){B.c0_buf[k], B.c0_off[k], VK_C0_BYTES};
    B.c0_in[k] = in;
    uint8_t tmp[VK_C0_BYTES];
    vk_c0_fill(tmp, st, ex->ssbo[st], ex->ubo[st], c0_dim, c0_binst);
    if (!B.c0_valid[k] || memcmp(tmp, B.c0[k], VK_C0_BYTES)) {
        memcpy(scratch(ex, VK_C0_BYTES, &B.c0_buf[k], &B.c0_off[k]), tmp, VK_C0_BYTES);
        memcpy(B.c0[k], tmp, VK_C0_BYTES);
        B.c0_valid[k] = 1;
    }
    return (VkDescriptorBufferInfo){B.c0_buf[k], B.c0_off[k], VK_C0_BYTES};
}

// Buffer range limits, read once (the device properties are a cold cache line).
static struct { uint64_t ubo_max, ubo_align, sbo_max, sbo_align; } LIM;

static void limits_init(void) {
    const VkPhysicalDeviceLimits *l = &vk_ctx()->props.limits;
    LIM.ubo_align = l->minUniformBufferOffsetAlignment, LIM.sbo_max = l->maxStorageBufferRange;
    LIM.sbo_align = l->minStorageBufferOffsetAlignment;
    __atomic_store_n(&LIM.ubo_max, l->maxUniformBufferRange, __ATOMIC_RELEASE);
}

// robustBufferAccess2 rounds a uniform read past a descriptor's range up to 16 bytes, as Maxwell does. Size 0 and sizes
// that are not a multiple of 16 escape that; counted here, newest at powers of two.
static void ubo_odd_size(int st, int slot, const p_bind_buf *b) {
    static uint64_t n0, n16;
    static int lines;
    uint64_t c = b->size ? ++n16 : ++n0;
    if (lines < 8 || !(c & (c - 1))) {
        lines++;
        rt_log("vk: uniform slot %d of stage %d bound at %#llx with size %#llx (descriptor fills so far: size 0 %llu, "
               "not a multiple of 16 %llu)", slot - 3, st, (unsigned long long)b->addr, (unsigned long long)b->size,
               (unsigned long long)n0, (unsigned long long)n16);
    }
}

// Compiler constants of a lowered NVN 2 stage (uniform slot 1, after the code in host memory), uploaded once per stage and
// kept until shutdown: a per-draw copy would make the descriptor build volatile (RS.volatile_). Keyed by the code pointer of nvn2_origin buffers; replay thread only.
#define C1_BLOCK (1u << 20)
#define C1_N 32768
static struct { const uint8_t *code; uint32_t c1, size; VkBuffer buf; VkDeviceSize off; } c1_tab[C1_N];
static struct { VkBuffer buf; VmaAllocation alloc; uint8_t *map; } c1_blk[256];
static uint32_t c1_nblk, c1_used = C1_BLOCK, c1_entries;
static uint64_t c1_bytes;

static int c1_const(const uint8_t *code, uint32_t c1, uint32_t size, uint64_t align, VkDescriptorBufferInfo *out) {
    if (size > C1_BLOCK / 4) return 0;
    uint32_t i = (uint32_t)(((uintptr_t)code * 0x9E3779B97F4A7C15ull) >> 49);
    for (;; i = (i + 1) & (C1_N - 1)) {
        if (c1_tab[i].code == code && c1_tab[i].c1 == c1 && c1_tab[i].size == size) {
            *out = (VkDescriptorBufferInfo){c1_tab[i].buf, c1_tab[i].off, size};
            return 1;
        }
        if (!c1_tab[i].code) break;
    }
    if (c1_entries > C1_N * 3 / 4 || !nvn2_origin(code)) return 0;   // full: the scratch copy
    if (align < 256) align = 256;
    uint32_t at = (uint32_t)((c1_used + align - 1) & ~(align - 1));
    if (at + size > C1_BLOCK) {
        if (c1_nblk == 256) return 0;
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = C1_BLOCK,
                                 .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                               VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &c1_blk[c1_nblk].buf, &c1_blk[c1_nblk].alloc, &info) != VK_SUCCESS) return 0;
        c1_blk[c1_nblk++].map = info.pMappedData;
        at = 0;
    }
    memcpy(c1_blk[c1_nblk - 1].map + at, code + c1, size);
    vmaFlushAllocation(vk_ctx()->vma, c1_blk[c1_nblk - 1].alloc, at, size);
    c1_used = at + size;
    c1_tab[i].code = code, c1_tab[i].c1 = c1, c1_tab[i].size = size, c1_tab[i].buf = c1_blk[c1_nblk - 1].buf, c1_tab[i].off = at;
    c1_entries++, c1_bytes += size;
    if (!(c1_entries & (c1_entries - 1)) && c1_entries >= 256)
        rt_log("vk_exec: %u compiler-constant blocks uploaded once (%.1f MiB in %u buffers)", c1_entries, c1_bytes / 1048576.0, c1_nblk);
    *out = (VkDescriptorBufferInfo){c1_tab[i].buf, c1_tab[i].off, size};
    return 1;
}

static void c1_release(VkCtx *c) {
    for (uint32_t i = 0; i < c1_nblk; i++) vmaDestroyBuffer(c->vma, c1_blk[i].buf, c1_blk[i].alloc);
    c1_nblk = 0, c1_used = C1_BLOCK, c1_entries = 0, memset(c1_tab, 0, sizeof c1_tab);
}

static VkDescriptorBufferInfo ubo_info(VkExec *ex, int st, int slot) {
    if (!__atomic_load_n(&LIM.ubo_max, __ATOMIC_ACQUIRE)) limits_init();
    uint64_t max = LIM.ubo_max, align = LIM.ubo_align;
    VkDescriptorBufferInfo bi;
    if (slot == 0) return c0_info(ex, st);
    if (slot == 1 && ex->prog) {   // compiler constants stored after the shader code
        const uint8_t *ctl, *code;
        uint32_t len, c1;
        if (find_stage(ex->prog, st, &ctl, &code, &len)) {
            memcpy(&c1, ctl + 0x700, 4);
            if (c1 < len && bind_range(ex, (uint64_t)(uintptr_t)code + c1, len - c1, align, max, VK_MUSE_UBO, 0, 0, &bi))
                return bi;
            if (c1 < len && len - c1 <= max) {   // NVN 2 stages: the lowered code (nvn2_lower.cpp) is host memory
                if (c1_const(code, c1, len - c1, align, &bi)) return bi;
                VkBuffer b;
                VkDeviceSize o;
                RS.volatile_ = 1;
                memcpy(scratch(ex, len - c1, &b, &o), code + c1, len - c1);
                return (VkDescriptorBufferInfo){b, o, len - c1};
            }
        }
    } else if (slot == 17 && st != NVN_STAGE_GS) {
        // Full-screen blits read a parameter block from bank 24 (moved to bank 17 by the lowering); a guess for a blit over the whole
        // render area: texel size 0x0/0x4 and 0x10/0x14, size 0x8/0xc, UV scale 1 at 0x18/0x1c, UV clamp 0,0..1,1 at 0x30-0x3c,
        // source rectangle 0,0..W-1,H-1 at 0x50-0x5c. OP_PUSH_CONST words replace it. No compute stage reads bank 24.
        float w = ex->pass_w ? (float)ex->pass_w : 1.0f, h = ex->pass_h ? (float)ex->pass_h : 1.0f;
        union { float f[64]; uint32_t u[64]; } blk = {{1 / w, 1 / h, w, h, 1 / w, 1 / h, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1}};
        blk.u[22] = (uint32_t)(w - 1), blk.u[23] = (uint32_t)(h - 1);
        int k = st == NVN_STAGE_CS;
        if (k) memset(&blk, 0, sizeof blk);
        for (int i = 0; i < 64; i++)
            if (ex->push_set[k][0] >> i & 1) blk.u[i] = ex->push[k][0][i];
        VkBuffer b;
        VkDeviceSize o;
        RS.volatile_ = 1;
        memcpy(scratch(ex, sizeof blk, &b, &o), &blk, sizeof blk);
        return (VkDescriptorBufferInfo){b, o, sizeof blk};
    } else if (slot >= 3 && slot - 3 < 18) {
        const p_bind_buf *b = &ex->ubo[st][slot - 3];
        if (b->addr && (!b->size || (b->size & 15))) ubo_odd_size(st, slot, b);
        if (rf_proj && slot == 3 && st != NVN_STAGE_FS && st != NVN_STAGE_CS && dlss_proj_block(ex, b, &bi)) return bi;
        if (b->size && vk_usnap_find(b->addr, b->size < max ? b->size : max, align, &bi)) {   // as of the submit call
            count_use(VK_MCLS_SCRATCH, VK_MUSE_UBO, bi.range);
            return bi;
        }
        if (bind_range(ex, b->addr, b->size, align, max, VK_MUSE_UBO, 0, 0, &bi)) return bi;
    }
    return zero_info(max);
}

// The NVN slot of a storage binding: the c0 entry holding its address (tables at 0x110 vertex, 0x510 fragment, 0x310 compute).
// A compute address at c0 0x210..0x30f is uniform block (offset - 0x210) / 16 read as storage: *ubo is set. -1: not in c0 (see ubo_ptr_res) or no known table.
static int ssbo_slot(const GfxShaderRes *r, int st, int *ubo) {
    *ubo = 0;
    if (r->cbuf_slot != 0) return -1;
    // Geometry stages use an ordinary table at c0+0x410; this only maps declared bindings.
    int off = r->handle * 4, base = st == NVN_STAGE_VS ? 0x110 : st == NVN_STAGE_FS ? 0x510 :
                                  st == NVN_STAGE_GS ? 0x410 : st == NVN_STAGE_CS ? 0x310 : -1;
    if (st == NVN_STAGE_CS && off >= 0x210 && off < 0x310 && (off - 0x210) % 16 == 0) {
        *ubo = 1;
        return (off - 0x210) / 16;
    }
    return base >= 0 && off >= base && off < base + 0x100 && (off - base) % 16 == 0 ? (off - base) / 16 : -1;
}

// 1 when the shader loads the storage buffer's address from a user uniform block (cbuf_slot 3.., uniform slot
// cbuf_slot - 3, byte offset handle * 4); otherwise it would get a zero buffer.
static int ubo_ptr_res(const GfxShaderRes *r) {
    return r->cbuf_slot >= 3 && r->cbuf_slot - 3 < 18;
}

// Inline uniform updates of the submission being replayed, newest last. op_update_ub writes them into the buffer's GPU copy; guest
// memory gets them from the write-back at the end of the submission. ubo_ptr reads the latest update covering the pointer, else memory.
#define UPD_N 4096
static struct {
    uint64_t fence;                      // the submission the entries belong to
    uint32_t n;                          // updates noted in it
    struct { uint64_t lo, hi; const uint8_t *data; } u[UPD_N];
} UPD;

static void upd_note(const VkExec *ex, const p_update_ub *u) {
    if (UPD.fence != ex->fence) UPD.fence = ex->fence, UPD.n = 0;
    UPD.u[UPD.n++ % UPD_N] = (typeof(UPD.u[0])){u->dst, u->dst + u->size, u->data};
}

// 1 when an update of this submission supplied [a, a+n).
int upd_read(const VkExec *ex, uint64_t a, void *out, size_t n) {
    if (UPD.fence != ex->fence) return 0;
    for (uint32_t i = UPD.n, end = UPD.n > UPD_N ? UPD.n - UPD_N : 0; i-- > end;) {
        const typeof(UPD.u[0]) *e = &UPD.u[i % UPD_N];
        if (a >= e->hi || a + n <= e->lo) continue;
        if (a < e->lo || a + n > e->hi) return 0;   // partly covered: memory
        memcpy(out, e->data + (a - e->lo), n);
        return 1;
    }
    return 0;
}

// That address as the bound block holds it at this point of the replay; 0 when the block is unbound or too short.
static uint64_t ubo_ptr(const VkExec *ex, int st, const GfxShaderRes *r) {
    const p_bind_buf *u = &ex->ubo[st][r->cbuf_slot - 3];
    uint64_t off = (uint64_t)r->handle * 4, a = 0;
    VkMemSpan sp;
    if (!u->addr || u->size < off + 8 || upd_read(ex, u->addr + off, &a, 8)) return a;
    if (vk_mem_resolve(u->addr + off, 8, &sp)) memcpy(&a, (const void *)(uintptr_t)(u->addr + off), 8);
    return a;
}

// Guarded NVN 2 candidates use the bytes the dispatch binds as uniform banks (bank 0: pushed roots and descriptor tables;
// bank 1: bounded compiler constants). bind_range decides whether a candidate has backing.
static p_bind_buf guarded_storage_binding(const VkExec *ex, int st, const GfxShaderRes *r) {
    uint64_t off = (uint64_t)r->handle * 4;
    p_bind_buf b = {0};
    if (st != NVN_STAGE_CS || !c0_lay) return b;
    if (r->cbuf_slot == 0) {
        if (off > VK_C0_BYTES - 8) return b;
        uint8_t bank[VK_C0_BYTES];
        vk_c0_fill(bank, st, ex->ssbo[st], ex->ubo[st], c0_dim, c0_binst);
        memcpy(&b.addr, bank + off, 8);
        size_t ub = nvn2_c0_part(c0_lay, 1), end = nvn2_c0_part(c0_lay, 3);
        if (off >= ub && off < end && (off - ub) % 16 == 0) {
            uint32_t bytes;
            memcpy(&bytes, bank + off + 8, 4);
            b.size = bytes;   // preserve a typed table's declared extent, not the remainder of its pool
        }
    } else if (r->cbuf_slot == 1 && ex->prog) {
        const uint8_t *ctl, *code;
        uint32_t len, c1;
        if (!find_stage(ex->prog, st, &ctl, &code, &len)) return b;
        memcpy(&c1, ctl + 0x700, 4);
        if (c1 > len || off > len - c1 || len - c1 - off < 8) return b;
        memcpy(&b.addr, code + c1 + off, 8);
    } else if (r->cbuf_slot == 17) {
        // ubo_info binds the synthetic compute push bank here, before its user-UBO mapping. Unwritten words are zero.
        if (off > sizeof ex->push[1][0] - 8) return b;
        uint32_t words[2] = {0};
        for (unsigned i = 0; i < 2; i++) {
            unsigned word = r->handle + i;
            if (ex->push_set[1][0] >> word & 1) words[i] = ex->push[1][0][word];
        }
        memcpy(&b.addr, words, 8);
    } else if (ubo_ptr_res(r)) {
        b.addr = ubo_ptr(ex, st, r);
    }
    return b;
}

static VkDescriptorBufferInfo ssbo_info(VkExec *ex, int st, const GfxShaderRes *r, int written) {
    if (r->cbuf_slot == GFX_GMEM_MAP_SLOT) {   // the raw pool table
        VkDescriptorBufferInfo table = {0};
        if (vk_mem_gmem_info(&table.buffer, &table.offset, &table.range)) return table;
        return zero_info(1);
    }
    if (!__atomic_load_n(&LIM.ubo_max, __ATOMIC_ACQUIRE)) limits_init();
    uint64_t max = LIM.sbo_max, align = LIM.sbo_align;
    VkDescriptorBufferInfo bi;
    int ubo = 0, slot = -1;
    p_bind_buf up = {0};
    const p_bind_buf *b = NULL;
    int guarded = !!(r->flags & GFX_RES_SSBO_GUARDED);
    if (!guarded) slot = ssbo_slot(r, st, &ubo);
    if (guarded) up = guarded_storage_binding(ex, st, r), b = &up;
    else if (slot >= 0 && slot < 16) b = ubo ? &ex->ubo[st][slot] : &ex->ssbo[st][slot];
    else if (ubo_ptr_res(r)) up.addr = ubo_ptr(ex, st, r), b = &up;   // size 0: up to the end of the backing
    if (b) {
        // A storage buffer with no declared size would be bound to the end of its pool (most of a multi-GB pool written back
        // every submission): cap it at the mapped size, at most 8 MiB; a guarded access past the cap branches away.
        static const uint64_t unsized_max = 8ull << 20;
        if (!ubo && !b->size && max > unsized_max) max = unsized_max;
        // A guarded candidate is flagged written if the shader stores anywhere, except inside a uniform buffer of this stage
        // (the game's own constant block, which the CPU rewrites; write-back would overwrite it).
        if (guarded && written) {
            for (int i = 0; written && i < 18; i++) {
                const p_bind_buf *u = &ex->ubo[st][i];
                if (u->addr && u->size && b->addr >= u->addr && b->addr + (b->size ? b->size : 1) <= u->addr + u->size) written = 0;
            }
        }
        // The producer tag of this binding's writes (NVN 2 stage name and SPIR-V binding), used by the write-back read trap.
        uint64_t wprog = 0;
        if (written) {
            const uint8_t *ctl, *code;
            uint32_t len;
            if (ex->prog && find_stage(ex->prog, st, &ctl, &code, &len)) wprog = nvn2_origin(code);
        }
        vk_mem_wb_tag(wprog, r->binding | (b->size ? 0u : 0x80000000u));   // bit 31: no declared size (bound up to 8 MiB)
        // vk_caps.c: shaders translated for sbuf_align address the buffer from its address rounded down to it.
        uint64_t lead = vk_caps.sbuf_align ? b->addr & (vk_caps.sbuf_align - 1) : 0;
        int bound = bind_range(ex, b->addr - lead, b->size ? b->size + lead : 0, align, max, ubo ? VK_MUSE_UBO : VK_MUSE_SSBO,
                               written, guarded, &bi);
        vk_mem_wb_tag(0, 0);
        if (bound) {
            if (written && bi.buffer != VK_NULL_HANDLE) {
                note_written(ex, b->addr - lead, bi.range);
                const uint8_t *ctl, *code;
                uint32_t len;
                if (vk_sync_noting() && ex->prog && find_stage(ex->prog, st, &ctl, &code, &len))
                    vk_sync_note_write(b->addr - lead, bi.range, nvn2_origin(code));
            }
            return bi;
        }
        if (b->addr && !guarded) warn_once(W_SSBO_ALIGN, "storage buffer %#llx is not backed or not aligned: a zero buffer is bound",
                               (unsigned long long)b->addr);
    } else {
        warn_once(W_SSBO_SLOT, "a storage buffer whose address is at c%d+%#x (stage %d) is not replayed: a zero buffer is bound",
                  r->cbuf_slot, r->handle * 4, st);
    }
    bi = zero_info(max);
    // Vulkan requires a nonzero range; one byte is zero elements of the runtime array and the guarded operation branches away.
    if (guarded) bi.range = 1;
    return bi;
}

// ---- feedback loops (vk.h VkCtx.feedback)
// A draw whose fragment shader samples an attachment the open pass wrote needs a barrier: VK_FEEDBACK_LAYOUT learns the
// image and uses the feedback-loop layout from then on; VK_FEEDBACK_RESTART restarts the pass behind a full barrier.
// Feedback loops: see docs/how-it-works.md, "Draw replay".
static int in_fbl(const VkExec *ex, VkImage img) {
    for (int i = 0; i < ex->nfbl; i++)
        if (ex->fbl_img[i] == img) return 1;
    return 0;
}

// The image and the levels [l0, l1) and layers [a0, a1) a texture's view covers.
typedef struct { VkImage image; uint32_t l0, l1, a0, a1; } SubRange;

static SubRange sub_of(const VkTexImage *img, const view_t *v, int hv) {
    SubRange r = {img->image, 0, img->levels, 0, img->layers};
    if (hv && v->set_levels) r.l0 = v->base_level, r.l1 = v->base_level + (v->nlevels ? v->nlevels : img->levels);
    if (hv && v->set_layers) r.a0 = v->min_layer, r.a1 = v->min_layer + (v->nlayers ? v->nlayers : img->layers);
    return r;
}

// Attachment slots of the open pass (bit 8 depth) whose subresource r covers.
static uint32_t attached_as(const VkExec *ex, const SubRange *r) {
    uint32_t m = 0;
    if (!r->image) return 0;
    for (uint32_t i = 0; i < 9; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        if ((i < 8 && i >= ex->pass_ncolor) || t->image != r->image) continue;
        if (t->level < r->l0 || t->level >= r->l1) continue;
        if (!t->is3d && (t->base_layer + t->layers <= r->a0 || t->base_layer >= r->a1)) continue;
        m |= 1u << i;
    }
    return m;
}

// The subresource a texture handle samples, kept while its slot resolves the same (slot_still). Direct-mapped.
#define FM_N 8192
static struct { uint64_t context, h, tok; uint32_t gen; SubRange r; } FM[FM_N];

static SubRange sampled_sub(const VkExec *ex, uint64_t h) {   // texture lock held
    view_t v;
    int hv;
    const VkTexImage *img = vk_texture_by_id_locked(&ex->tex_pool, (int)(h & 0xfffff), &v, &hv);
    return img ? sub_of(img, &v, hv) : (SubRange){0};
}

// Attachments of the open pass the fragment shader fs samples; kept while the shader, the texture bindings and the
// pass stay the same.
static uint32_t feedback_scan(VkExec *ex, const GfxShader *fs) {
    descriptor_state(ex);
    if (ex->fb_fs == fs && ex->fb_serial == ex->tex_serial && ex->fb_pass == ex->pass_serial) return ex->fb_sampled;
    const GfxShaderRes *res;
    int n = gfx_shader_resources(fs, GFX_RES_TEX, &res);
    int st = nvn_stage_of(gfx_shader_stage(fs));
    if (st < 0) return 0;
    uint32_t m = 0, gen = vk_texture_slots_gen();   // before any lookup
    int locked = 0;
    for (int r = 0; r < n; r++)
        for (int k = 0; k < (res[r].count ? res[r].count : 1); k++) {
            int slot = res[r].slot + k;
            uint64_t h = slot >= 0 && slot < 128 ? ex->tex[st][slot] : 0;
            if (!h) continue;
            typeof(FM[0]) *f = &FM[(h * 0x9E3779B97F4A7C15ull) >> 51];
            if (f->context == descriptor_serial && f->h == h && slot_still(ex, &f->gen, f->tok, h, gen)) {
                m |= attached_as(ex, &f->r);
                continue;
            }
            uint64_t tok = token_of(ex, h);   // before the lookup
            if (!locked) vk_texture_lock(), locked = 1;
            SubRange sr = sampled_sub(ex, h);
            m |= attached_as(ex, &sr);
            if (tok) *f = (typeof(*f)){.context = descriptor_serial, .h = h, .tok = tok, .gen = gen, .r = sr};
            else f->h = 0;
        }
    if (locked) vk_texture_unlock();
    ex->fb_fs = fs, ex->fb_serial = ex->tex_serial, ex->fb_pass = ex->pass_serial, ex->fb_sampled = m;
    return m;
}

static int geometry_feedback_safe(VkExec *ex, const GfxShader *vs, const GfxShader *gs) {
    // The in-pass feedback dependency is fragment-only; reject pre-raster reads of an attachment.
    if (gs && (feedback_scan(ex, vs) || feedback_scan(ex, gs))) {
        warn_once(W_GEOMETRY, "experimental geometry draw samples an attachment before rasterization: skipped");
        return 0;
    }
    return 1;
}

// Before the pipeline lookup of a draw with fragment shader fs: learns and restarts as the mode says. Returns the
// sampled attachments in the feedback layout (the pipeline's feedback flag and feedback_barrier).
static uint32_t feedback_prepare(VkExec *ex, const GfxShader *fs) {
    int mode = vk_ctx()->feedback;
    if (mode == VK_FEEDBACK_OFF || !fs || !ex->in_pass) return 0;
    uint32_t m = feedback_scan(ex, fs);
    uint32_t d = m & ex->fb_dirty;
    if (d) vk_exec_frame.fb_draws++;
    if (d & ~(mode == VK_FEEDBACK_LAYOUT ? ex->fbl_mask : 0)) {
        if (mode == VK_FEEDBACK_LAYOUT)
            for (uint32_t i = 0; i < 9; i++) {
                const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
                if ((d >> i & 1) && t->img && t->img->fb_usage) ((VkTexImage *)t->img)->fb_learned = 1;
            }
        vk_exec_pass_end(ex);
        vk_exec_pass_begin(ex);   // learned targets now in the feedback layout; the full barrier cleared fb_dirty
        vk_exec_frame.fb_restarts++;
    }
    return mode == VK_FEEDBACK_LAYOUT ? m & ex->fbl_mask : 0;
}

// By-region barrier inside the pass: earlier attachment writes become visible to this draw's fragment shader reads.
static void feedback_barrier(VkExec *ex) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                                           VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                           .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                           .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
                           .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(ex->cb, &di);
    ex->recorded++;
    ex->fb_dirty = 0;
    vk_exec_frame.fb_barriers++;
}

// Marks the attachments the draw with state s writes: colour targets with a nonzero effective write mask, depth with
// depth writes or the stencil test on.
static void feedback_written(VkExec *ex, const VkPipeState *s) {
    if (vk_ctx()->feedback == VK_FEEDBACK_OFF) return;
    if (ex->depth.image && !(ex->fb_dirty & 0x100) && ((s->ds.dt && s->ds.dw) || s->ds.st)) ex->fb_dirty |= 0x100;
    for (uint32_t i = 0; i < ex->pass_ncolor; i++)
        if (ex->color[i].image && !(ex->fb_dirty >> i & 1) && vk_pipeline_target_wmask(s, (int)i)) ex->fb_dirty |= 1u << i;
}

static void tex_info_full(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate,
                          VkDescriptorImageInfo *out, VkImage *image);


// Sampled-texture descriptors per (handle, resource type, separate), kept while the slot resolution and texture entry (slot_still)
// and the samplers (vk_sampler_gen) are unchanged; the layout is set again on every hit (in_fbl). A slot's first binding in
// a submission (vk_texture_epoch) still does its storage check. Storage images are not kept. Direct-mapped; texture lock held.
#define TD_N 8192
static struct {
    uint64_t context, h, tok, stok;   // stok: always 0
    uint32_t gen, epoch, sgen;
    uint16_t type;
    uint8_t separate, valid;
    VkImage image;
    VkDescriptorImageInfo out;
} TD[TD_N];

static void tex_info_lookup(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate,
                            VkDescriptorImageInfo *out) {
    descriptor_state(ex);
    VkImage image;
    if (!h || storage) {
        return tex_info_full(ex, r, h, storage, separate, out, &image);
    }
    uint32_t gen = vk_texture_slots_gen(), ep = vk_texture_epoch(), sg = vk_sampler_gen();   // before any lookup
    typeof(TD[0]) *m = &TD[(h * 0x9E3779B97F4A7C15ull ^ (uint64_t)r->type << 7 ^ (uint64_t)separate) >> 51];
    if (m->valid && m->context == descriptor_serial && m->h == h && m->type == r->type && m->separate == separate && m->sgen == sg &&
        slot_still(ex, &m->gen, m->tok, h, gen)) {
        if (m->epoch != ep) {   // the slot's first binding in this submission: only its storage check is due
            vk_texture_by_id_locked(&ex->tex_pool, (int)(h & 0xfffff), NULL, NULL);
            m->epoch = ep;
        }
        *out = m->out;
        out->imageLayout = m->image && ex->nfbl && in_fbl(ex, m->image) ? VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
                                                                          : VK_IMAGE_LAYOUT_GENERAL;
        return;
    }
    uint64_t tok = token_of(ex, h);   // before the lookup
    tex_info_full(ex, r, h, storage, separate, out, &image);
    *m = (typeof(*m)){.context = descriptor_serial, .h = h, .tok = tok, .gen = gen, .epoch = ep, .sgen = sg, .type = r->type,
                      .separate = (uint8_t)separate, .valid = 1, .image = image, .out = *out};
}

static void tex_info_full(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate,
                          VkDescriptorImageInfo *out, VkImage *image) {
    int ms = (r->type & 0x200) != 0;
    *image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler samp = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
    if (h) {
        view_t v;
        int hv;
        const VkTexImage *img = vk_texture_by_id_locked(&ex->tex_pool, (int)(h & 0xfffff), &v, &hv);
        if (img) view = get_view(ex, img, &v, hv, r->type, storage), *image = img->image;
        if (img && storage) vk_texture_storage_bound_locked(img);
        if (img && !storage && ex->nfbl && in_fbl(ex, img->image)) layout = VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT;
        if (!storage && !separate) {
            // A depth texture sampled without a compare reads one texel on the Switch whatever the filter (blending depths breaks shadow masks).
            int sid = (int)(h >> 20 & 0xfff);
            int point = img && (img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && !(r->type & 0x400) &&
                        vk_exec_view_aspect(img, &v, hv) != VK_IMAGE_ASPECT_STENCIL_BIT;
            samp = point ? vk_sampler_point_by_id_locked(&ex->samp_pool, sid) : vk_sampler_by_id_locked(&ex->samp_pool, sid);
        }
    }
    if (!view) {
        int di = dummy_index(res_view_type(r->type), ms);
        view = storage ? storage_dummy(di, r->format) : Z.img[di].view;
    }
    if (!samp) samp = r->type & 0x400 ? Z.samp_shadow : Z.samp;
    *out = (VkDescriptorImageInfo){separate || storage ? VK_NULL_HANDLE : samp, view, layout};
}

// ---- depth twins: colour images under a compare sampler in a compute shader
// Compute shaders sample an R32_SFLOAT colour texture through a shadow sampler, which AMD drivers cannot compare
// (VUID-vkCmdDispatch-None-06479). Such an image gets a depth twin, copied through a buffer before each dispatch that samples it; draws keep the colour image.
#define TWINS 8
typedef struct { VkImage src, img; VmaAllocation mem; VkImageView view[2]; uint32_t w, h, layers; VkFormat fmt; } Twin;
static Twin twins[TWINS];
static unsigned twin_next;
static int twin_dispatch;                // op_dispatch*: the resources being bound are a dispatch's
static VkBuffer twin_buf;
static VmaAllocation twin_buf_mem;
static VkDeviceSize twin_buf_cap;

static void twin_free(void *arg) {   // completion thread, after the last submission that may sample it
    Twin *t = arg;
    VkCtx *c = vk_ctx();
    for (int k = 0; k < 2; k++)
        if (t->view[k]) vkDestroyImageView(c->device, t->view[k], NULL);
    vmaDestroyImage(c->vma, t->img, t->mem);
    free(t);
}

typedef struct { VkBuffer buf; VmaAllocation mem; } TwinBuf;
static void twin_buf_free(void *arg) {   // completion thread, after the last submission that may copy through it
    TwinBuf *b = arg;
    vmaDestroyBuffer(vk_ctx()->vma, b->buf, b->mem);
    free(b);
}

static VkFormat twin_format(const VkTexImage *img) {
    if ((img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) || img->samples > 1 || img->depth > 1) return VK_FORMAT_UNDEFINED;
    VkFormat df = img->format == VK_FORMAT_R32_SFLOAT ? VK_FORMAT_D32_SFLOAT
                : img->format == VK_FORMAT_R16_UNORM  ? VK_FORMAT_D16_UNORM : VK_FORMAT_UNDEFINED;
    if (!df) return df;
    static int compares[2] = {-1, -1};   // per source format: the driver compares it as it is
    int i = df == VK_FORMAT_D16_UNORM;
    if (compares[i] < 0) {
        VkFormatProperties3 p3 = {.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};
        VkFormatProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, .pNext = &p3};
        vkGetPhysicalDeviceFormatProperties2(vk_ctx()->phys, img->format, &p2);
        compares[i] = (p3.optimalTilingFeatures & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT) != 0;
        rt_log("vk_exec: %s under a compare sampler: %s", i ? "R16_UNORM" : "R32_SFLOAT",
               compares[i] ? "compared by the driver" : "sampled through a depth twin in compute passes");
    }
    return compares[i] ? VK_FORMAT_UNDEFINED : df;
}

static void twins_release(VkCtx *c) {
    for (int i = 0; i < TWINS; i++) {
        for (int k = 0; k < 2; k++)
            if (twins[i].view[k]) vkDestroyImageView(c->device, twins[i].view[k], NULL);
        if (twins[i].img) vmaDestroyImage(c->vma, twins[i].img, twins[i].mem);
    }
    memset(twins, 0, sizeof twins);
    if (twin_buf) vmaDestroyBuffer(c->vma, twin_buf, twin_buf_mem);
    twin_buf = VK_NULL_HANDLE, twin_buf_mem = NULL, twin_buf_cap = 0;
}

// The twin of img holding its current texels, copies recorded into ex->cb; NULL when none is needed or it cannot be made.
static Twin *twin_update(VkExec *ex, const VkTexImage *img) {
    VkFormat df = twin_format(img);
    if (!df) return NULL;
    VkCtx *c = vk_ctx();
    uint32_t bpp = df == VK_FORMAT_D32_SFLOAT ? 4 : 2, layers = img->layers ? img->layers : 1;
    VkDeviceSize bytes = (VkDeviceSize)img->width * img->height * layers * bpp;
    if (bytes > (64u << 20)) return NULL;
    if (bytes > twin_buf_cap) {   // one copy buffer, grown; the old one goes after the submission being recorded
        VkDeviceSize want = twin_buf_cap ? twin_buf_cap * 2 : 1u << 16;
        while (want < bytes) want *= 2;
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = want,
                                 .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VkBuffer nb;
        VmaAllocation nm;
        if (vmaCreateBuffer(c->vma, &bi, &ai, &nb, &nm, NULL) != VK_SUCCESS) return NULL;
        if (twin_buf) {
            TwinBuf *old = malloc(sizeof *old);
            if (!old) rt_fatal("vk: out of memory (depth twin)");
            *old = (TwinBuf){twin_buf, twin_buf_mem};
            vk_completion_add(vk_sync_issued() + 1, twin_buf_free, old);
        }
        twin_buf = nb, twin_buf_mem = nm, twin_buf_cap = want;
    }
    Twin *t = NULL;
    for (int i = 0; i < TWINS && !t; i++)
        if (twins[i].img && twins[i].src == img->image && twins[i].w == img->width && twins[i].h == img->height &&
            twins[i].layers == layers && twins[i].fmt == df)
            t = &twins[i];
    if (!t) {
        t = &twins[twin_next++ % TWINS];
        if (t->img) {   // the submission being recorded may still sample the replaced twin
            Twin *old = malloc(sizeof *old);
            if (!old) rt_fatal("vk: out of memory (depth twin)");
            *old = *t;
            vk_completion_add(vk_sync_issued() + 1, twin_free, old);
        }
        *t = (Twin){.src = img->image, .w = img->width, .h = img->height, .layers = layers, .fmt = df};
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = df,
                                .extent = {img->width, img->height, 1}, .mipLevels = 1, .arrayLayers = layers,
                                .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(c->vma, &ci, &ai, &t->img, &t->mem, NULL) != VK_SUCCESS) {
            *t = (Twin){0};
            return NULL;
        }
        for (int k = 0; k < 2; k++) {
            VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = t->img,
                                        .viewType = k ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, .format = df,
                                        .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, k ? layers : 1}};
            if (vkCreateImageView(c->device, &vi, NULL, &t->view[k]) != VK_SUCCESS) t->view[k] = VK_NULL_HANDLE;
        }
        static int nlog;
        if (nlog++ < 8)
            rt_log("vk_exec: depth twin %ux%ux%u %s for a compute pass's compare sampler", img->width, img->height,
                   layers, df == VK_FORMAT_D32_SFLOAT ? "D32_SFLOAT" : "D16_UNORM");
    }
    // The source's earlier writes and the twin's earlier readers before the copies; the twin's old texels are dropped.
    VkMemoryBarrier2 m0 = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT};
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = t->img, .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, layers}};
    VkDependencyInfo d0 = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &m0,
                           .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(ex->cb, &d0);
    VkBufferImageCopy src = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layers},
                             .imageExtent = {img->width, img->height, 1}};
    vkCmdCopyImageToBuffer(ex->cb, img->image, VK_IMAGE_LAYOUT_GENERAL, twin_buf, 1, &src);
    VkMemoryBarrier2 m1 = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT};
    VkDependencyInfo d1 = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &m1};
    vkCmdPipelineBarrier2(ex->cb, &d1);
    VkBufferImageCopy dst = {.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, layers},
                             .imageExtent = {img->width, img->height, 1}};
    vkCmdCopyBufferToImage(ex->cb, twin_buf, t->img, VK_IMAGE_LAYOUT_GENERAL, 1, &dst);
    // The twin's texels before the dispatch; the source's next writers and the buffer's next copy after these copies.
    VkMemoryBarrier2 m2 = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo d2 = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &m2};
    vkCmdPipelineBarrier2(ex->cb, &d2);
    ex->recorded += 5;
    return t;
}

// The image descriptor of slot h; in a dispatch, a compare-sampled colour image the driver cannot compare is swapped
// for its depth twin.
static void tex_info(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate, VkDescriptorImageInfo *out) {
    tex_info_lookup(ex, r, h, storage, separate, out);
    if (!twin_dispatch || storage || separate || !h || !(r->type & 0x400)) return;
    VkImageViewType vt = res_view_type(r->type);
    if (vt != VK_IMAGE_VIEW_TYPE_2D && vt != VK_IMAGE_VIEW_TYPE_2D_ARRAY) return;
    const VkTexImage *img = vk_texture_by_id_locked(&ex->tex_pool, (int)(h & 0xfffff), NULL, NULL);
    Twin *t = img ? twin_update(ex, img) : NULL;
    VkImageView v = t ? t->view[vt == VK_IMAGE_VIEW_TYPE_2D_ARRAY] : VK_NULL_HANDLE;
    if (v) out->imageView = v, out->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
}

static VkDescriptorSet alloc_set(VkExec *ex, VkDescriptorSetLayout layout) {
    CbRes *r = cb_res(ex);
    VkCtx *c = vk_ctx();
    for (int attempt = 0; attempt < 2; attempt++) {
        if (r->npools) {
            VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                              .descriptorPool = r->pools[r->npools - 1], .descriptorSetCount = 1,
                                              .pSetLayouts = &layout};
            VkDescriptorSet set;
            VkResult res = vkAllocateDescriptorSets(c->device, &ai, &set);
            if (res == VK_SUCCESS) return set;
            if (res != VK_ERROR_OUT_OF_POOL_MEMORY && res != VK_ERROR_FRAGMENTED_POOL) rt_fatal("vk: descriptor set allocation failed (%d)", (int)res);
        }
        static const VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4096}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2048},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4096}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024},
            {VK_DESCRIPTOR_TYPE_SAMPLER, 1024},   // the fragment sampler arrays of mesh pipelines
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 256},
            {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 256}};
        VkDescriptorPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 256,
                                         .poolSizeCount = sizeof sizes / sizeof sizes[0], .pPoolSizes = sizes};
        if (r->npools == r->cpools) r->pools = grow(r->pools, &r->cpools, sizeof *r->pools);
        VkDescriptorPool pool = VK_NULL_HANDLE;
        plat_mutex_lock(&spare_lock);
        if (nspare_pool) pool = spare_pool[--nspare_pool];
        plat_mutex_unlock(&spare_lock);
        if (!pool && vkCreateDescriptorPool(c->device, &pi, NULL, &pool) != VK_SUCCESS) rt_fatal("vk: descriptor pool failed");
        r->pools[r->npools++] = pool;
    }
    rt_fatal("vk: descriptor set allocation failed twice");
    return VK_NULL_HANDLE;
}

static void support_in(const VkExec *ex, float in[3]) {   // what support_fill reads: viewport size, point size
    in[0] = ex->in_pass && ex->has_vp ? ex->vp[0][2] : (float)ex->pass_w;
    in[1] = ex->in_pass && ex->has_vp ? ex->vp[0][3] : (float)ex->pass_h;
    in[2] = ex->point_size > 0 ? ex->point_size : 1.0f;
}

static void support_fill(uint8_t *b, float w, float h, float point_size) {
    memset(b, 0, SUPPORT_BYTES);
    float inv[4] = {w > 0 ? 2.0f / w : 0, h > 0 ? 2.0f / h : 0, 0, 0}, size[4] = {w, h, 0, 0};
    memcpy(b + 144, inv, 16);
    memcpy(b + 160, size, 16);
    const float one = 1.0f;
    memcpy(b + 1392, &point_size, 4);   // point_size: vertex shaders that write none output it
    uint64_t table = vk_mem_flat_table();   // gmem_low, gmem_high
    memcpy(b + 1408, &table, 4), memcpy(b + 1424, (uint8_t *)&table + 4, 4);
    for (int i = 0; i < 73; i++) memcpy(b + 192 + 16 * i, &one, 4);   // render scale 1: the host renders at guest size
}

// A graphics draw whose descriptor inputs equal the previous graphics draw's in this command buffer skips building its sets
// (bind_sig compares the inputs; the side effects of a build repeat as no-ops). B.res_valid says the bound sets hold the
// last build; vk_exec_draw_new_cb clears it and every build clears it until it ends. See README, "Draw replay".

static void rs_add(RsRange *r, uint32_t *n, const VkExec *ex, const void *at, size_t len) {
    uint32_t off = (uint32_t)((const uint8_t *)at - (const uint8_t *)ex);
    if (*n && r[*n - 1].off + r[*n - 1].len == off) r[*n - 1].len += (uint32_t)len;
    else r[(*n)++] = (RsRange){off, (uint32_t)len};
}

static const RsList *rs_list(const VkExec *ex, const VkPipeInfo *info) {
    if (info == rs_last) return rs_last_l;
    if (!rs_lists.klen) nvn_hm_init(&rs_lists, sizeof info);
    RsList *l = nvn_hm_get(&rs_lists, &info);
    if (!l) {
        static RsRange tmp[MAX_DESC];
        RsPtr ptr[RS_PTRS];
        uint32_t n = 0, nd = 0, np = 0;
        int sup = 0, c0 = 0, ok = 1;
        for (uint32_t i = 0; i < info->n_bindings && ok; i++) {
            const VkPipeBinding *b = &info->bindings[i];
            const GfxShaderRes *r = b->res;
            uint32_t cnt = b->count ? b->count : 1;
            if ((nd += cnt) > MAX_DESC) { ok = 0; break; }   // the build stops on it
            int st = nvn_stage_of(b->stage);
            if (st < 0) rt_fatal("vk_exec: unsupported descriptor stage %u", b->stage);
            if (b->kind == VK_PIPE_SUPPORT) {
                sup = 1;
            } else if (b->kind == GFX_RES_UBO) {
                for (uint32_t k = 0; k < cnt; k++) {
                    int slot = r->slot + (int)k;
                    if (slot == 0) c0 = 1, rs_add(tmp, &n, ex, ex->ssbo[st], sizeof ex->ssbo[st]);
                    else if (slot >= 3 && slot - 3 < 18) rs_add(tmp, &n, ex, &ex->ubo[st][slot - 3], sizeof(p_bind_buf));
                }   // slot 1: the program (bind_sig); others: the zero buffer
            } else if (b->kind == GFX_RES_SSBO) {   // every descriptor of the binding is the same range
                int ubo, slot = ssbo_slot(r, st, &ubo);
                if (slot >= 0 && slot < 16) {
                    rs_add(tmp, &n, ex, ubo ? &ex->ubo[st][slot] : &ex->ssbo[st][slot], sizeof(p_bind_buf));
                } else if (ubo_ptr_res(r)) {
                    if (np == RS_PTRS) { ok = 0; break; }
                    rs_add(tmp, &n, ex, &ex->ubo[st][r->cbuf_slot - 3], sizeof(p_bind_buf));
                    ptr[np++] = (RsPtr){st, r};
                }
            } else if ((b->kind == GFX_RES_TEX || b->kind == GFX_RES_IMG) && b->type != VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER &&
                       b->type != VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) {
                int img = b->kind == GFX_RES_IMG, cap = img ? 32 : 128;
                for (uint32_t k = 0; k < cnt; k++) {
                    int slot = r->slot + (int)k;
                    if (slot >= 0 && slot < cap) rs_add(tmp, &n, ex, img ? &ex->img[st][slot] : &ex->tex[st][slot], 8);
                }
            }
        }
        uint64_t bytes = 0;
        for (uint32_t i = 0; i < n; i++) bytes += tmp[i].len;
        if (bytes > RS_BYTES) ok = 0;
        l = malloc(sizeof *l + sizeof *tmp * n);
        if (!l) rt_fatal("vk: out of memory");
        l->n = n, l->bytes = (uint32_t)bytes, l->sup = (uint8_t)sup, l->c0 = (uint8_t)c0, l->ok = (uint8_t)ok;
        l->np = (uint8_t)np;
        memcpy(l->p, ptr, sizeof *ptr * np);
        memcpy(l->r, tmp, sizeof *tmp * n);
        nvn_hm_put(&rs_lists, &info, l);
    }
    rs_last = info, rs_last_l = l;
    return l;
}

// The signature of a graphics draw's descriptor inputs into RS; 1 when it equals the stored one. Bound-slot bytes are
// compared only after a bind op or a pipeline change; check mode compares them always.
static int bind_sig(const VkExec *ex, const VkPipeInfo *info) {
    const RsList *l = rs_list(ex, info);
    if (!l->ok) return RS.hn = 0, RS.info = NULL, 0;
    uint64_t *h = RS.h, diff = 0;
    uint32_t n = 0;
#define PUT(v) do { uint64_t v_ = (uint64_t)(v); diff |= h[n] ^ v_; h[n++] = v_; } while (0)
    PUT((uintptr_t)info);
    PUT((uintptr_t)ex->prog);
    PUT(ex->prog->serial);
    PUT(ex->tex_pool.addr);
    PUT(ex->tex_pool.count);
    PUT(ex->samp_pool.addr);
    PUT(ex->samp_pool.count);
    PUT((uint64_t)vk_texture_slots_gen() << 32 | vk_texture_epoch());
    PUT((uint64_t)vk_sampler_gen() << 32 | (uint32_t)ex->nfbl);
    PUT(0);
    PUT(vk_mem_gen() + vk_usnap_gen());   // both only grow
    for (int i = 0; i < ex->nfbl && i < 9; i++) PUT((uintptr_t)ex->fbl_img[i]);
    if (l->sup) {
        float in[3];
        support_in(ex, in);
        uint64_t a;
        uint32_t ps;
        memcpy(&a, in, 8), memcpy(&ps, &in[2], 4);
        PUT(a);
        PUT(ps);
    }
    if (l->c0) PUT(c0_binst);
    for (int i = 0; i < l->np; i++) PUT(ubo_ptr(ex, l->p[i].st, l->p[i].res));
#undef PUT
    diff |= n != RS.hn;
    RS.hn = n;
    int fast = RS.info == info && RS.gen == bind_gen;
    if (fast) return !diff;
    const uint8_t *e = (const uint8_t *)ex;
    uint8_t *d = RS.body;
    int bd = 0;
    for (uint32_t i = 0; i < l->n; i++) {
        const RsRange *r = &l->r[i];
        if (!bd && memcmp(d, e + r->off, r->len)) bd = 1;
        if (bd) memcpy(d, e + r->off, r->len);
        d += r->len;
    }
    RS.info = info, RS.gen = bind_gen;
    return !diff && !bd;
}

// Writes allocated set s from its packed descriptors d: through the layout's update template, or with the write
// structs w[i0..i1) when it has none.
static void write_set(const VkPipeInfo *info, uint32_t s, VkDescriptorSet set, const VkDescSlot *d, VkWriteDescriptorSet *w,
                      uint32_t i0, uint32_t i1) {
    if (i1 == i0) return;
    if (info->tpl[s]) {
        vk_pipeline_update_tpl_fn()(vk_ctx()->device, set, info->tpl[s], d);
        return;
    }
    for (uint32_t k = i0; k < i1; k++) w[k].dstSet = set;
    vkUpdateDescriptorSets(vk_ctx()->device, i1 - i0, w + i0, 0, NULL);
}

// Bytes readable by the CPU at addr: pools made by the runtime (GPU-only addresses) have no guest memory behind them.
uint64_t dyn_readable(uint64_t addr, uint64_t want) {
    VkMemInspect mi;
    if (!vk_mem_inspect(addr, 8, &mi) || mi.pool_kind != VK_MEM_INSPECT_POOL_IMPORT) return 0;
    return mi.avail < want ? mi.avail : want;
}

// Writes the pipeline's sets (set 0, and set 1 with independent layouts): each is pushed into the command buffer, or
// allocated, written and bound; a graphics set equal to the last one allocated for that number is reused, an equal push is not recorded again.
// The descriptors go into one packed array d (a VkDescSlot each, binding order) that a set's update template reads; write structs only without a template.
static VkDescriptorBufferInfo mesh_resource_info(VkExec *ex, const VkPipeBinding *binding);

static void bind_resources(VkExec *ex, const VkPipeInfo *info) {
    static VkWriteDescriptorSet w[MAX_BINDINGS];
    static VkDescSlot d[MAX_DESC];
    static VkBufferView bv[MAX_DESC];
    if (info->n_bindings > MAX_BINDINGS) rt_fatal("vk: %u descriptor bindings exceed the replay's %d", info->n_bindings, MAX_BINDINGS);
    int gfx = info->bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS;
    int mesh = info->n_bindings && info->bindings[0].stage == GFX_STAGE_MESH;
    if (rf_proj) B.res_valid = 0;   // a projection-jittered draw binds a patched camera block: never skip into or out of it
    int skip = gfx && !mesh && !rf_proj, same = 0, was_valid = B.res_valid;
    // OOT_TRACE=perf: which binds could skip their build and did, every 600 frames
    static uint64_t nb[5], nb_frame;   // compute, mesh, jittered graphics, other graphics, skipped
    nb[!gfx ? 0 : mesh ? 1 : rf_proj ? 2 : 3]++;
    if ((rt_trace & RT_TRACE_PERF) && vk_frame_no >= nb_frame + 600) {
        if (nb_frame) {
            uint64_t all = nb[0] + nb[1] + nb[2] + nb[3];
            rt_log("perf: bind_resources per frame (600 frames): %.0f calls: compute %.0f, mesh %.0f, jittered %.0f, other graphics "
                   "%.0f of which %.0f skipped the build", all / 600.0, nb[0] / 600.0, nb[1] / 600.0, nb[2] / 600.0,
                   nb[3] / 600.0, nb[4] / 600.0);
        }
        uint64_t blb, bls;
        bindless_bind_stats(&blb, &bls);
        uint64_t blr[4];
        bindless_ring_stats(blr);
        if (nb_frame) rt_log("perf: bindless set 2 per frame (600 frames): %.0f binds, %.0f skipped; ids re-pointed "
                             "%.2f, commits into a new copy %.2f, binds with no free copy %.2f (ring of %llu)", blb / 600.0,
                             bls / 600.0, blr[2] / 600.0, blr[0] / 600.0, blr[1] / 600.0, (unsigned long long)blr[3]);
        memset(nb, 0, sizeof nb), nb_frame = vk_frame_no;
    }
    if (mesh) B.res_valid = 0;   // native pointer-table reads are not part of the ordinary bind signature
    if (skip) {
        same = bind_sig(ex, info);
        if (same && was_valid) {
            for (int i = 0; i < RS.ncnt; i++) vk_mem_count(RS.cnt[i].cls, RS.cnt[i].use, RS.cnt[i].bytes);
            nb[4]++;
            ex->recorded++;
            return;
        }
        B.res_valid = 0;   // until the build below ends
        RS.rec = 1, RS.ncnt = 0, RS.cnt_over = 0, RS.volatile_ = 0;
    }
    uint32_t n_sets = info->n_sets ? info->n_sets : 1;
    uint32_t lo[2] = {0, 0}, hi[2] = {0, 0}, d0[2] = {0, 0};   // bindings [lo, hi) and first descriptor per set
    int need_w = 0;                                             // a set without a template: fill the write structs
    for (uint32_t s = 0, i0 = 0; s < n_sets; s++) {
        uint32_t i1 = i0;
        while (i1 < info->n_bindings && (n_sets == 1 || info->bindings[i1].set == s)) i1++;
        lo[s] = i0, hi[s] = i1, i0 = i1;
        if (i1 > lo[s] && !info->tpl[s]) need_w = 1;
    }
    uint32_t nd = 0;
    for (uint32_t i = 0; i < info->n_bindings; i++) {
        const VkPipeBinding *b = &info->bindings[i];
        const GfxShaderRes *r = b->res;
        uint32_t cnt = b->count ? b->count : 1;
        if (nd + cnt > MAX_DESC) rt_fatal("vk: %u descriptors exceed the replay's %d", nd + cnt, MAX_DESC);
        if (n_sets == 2 && i == lo[1]) d0[1] = nd;
        VkDescSlot *o = &d[nd];
        int st = nvn_stage_of(b->stage), kind = 0;   // kind: 0 buffer, 1 image, 2 texel buffer view
        if (st < 0) rt_fatal("vk_exec: unsupported descriptor stage %u", b->stage);
        if (b->kind == VK_PIPE_SUPPORT) {
            float in[3];
            support_in(ex, in);
            if (!B.sup_valid || memcmp(in, B.sup_in, sizeof in)) {   // support_fill reads only these
                uint8_t tmp[SUPPORT_BYTES];
                support_fill(tmp, in[0], in[1], in[2]);
                memcpy(B.sup_in, in, sizeof in);
                if (!B.sup_valid || memcmp(tmp, B.support, SUPPORT_BYTES)) {   // a new buffer only when the contents changed
                    VkBuffer sb;
                    VkDeviceSize so;
                    memcpy(scratch(ex, SUPPORT_BYTES, &sb, &so), tmp, SUPPORT_BYTES);
                    memcpy(B.support, tmp, SUPPORT_BYTES);
                    B.sup_buf = sb, B.sup_off = so, B.sup_valid = 1;
                }
            }
            o->buffer = (VkDescriptorBufferInfo){B.sup_buf, B.sup_off, SUPPORT_BYTES};
        } else if ((b->stage == GFX_STAGE_MESH || b->stage == GFX_STAGE_TASK) &&
                   (b->kind == GFX_RES_UBO || b->kind == GFX_RES_SSBO)) {
            if (cnt != 1) rt_fatal("vk: native mesh descriptor arrays are unsupported");
            o->buffer = mesh_resource_info(ex, b);
        } else if (b->kind == GFX_RES_UBO) {
            for (uint32_t k = 0; k < cnt; k++) o[k].buffer = ubo_info(ex, st, r->slot + (int)k);
        } else if (b->kind == GFX_RES_SSBO) {
            for (uint32_t k = 0; k < cnt; k++) o[k].buffer = ssbo_info(ex, st, r, r->flags & 1);
        } else if (b->kind == GFX_RES_TEX || b->kind == GFX_RES_IMG) {
            int img = b->kind == GFX_RES_IMG;
            if (b->type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER || b->type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) {
                kind = 2;
                for (uint32_t k = 0; k < cnt; k++) {   // buffer textures are not replayed yet
                    o[k].w[0] = 0, o[k].w[1] = 0, o[k].w[2] = 0;
                    o[k].view = Z.zview;
                    if (need_w) bv[nd + k] = Z.zview;
                }
            } else {
                kind = 1;
                vk_texture_lock();   // tex_info looks up with the texture lock held
                for (uint32_t k = 0; k < cnt; k++) {
                    int slot = r->slot + (int)k;
                    uint64_t h = 0;
                    if (img) h = slot >= 0 && slot < 32 ? ex->img[st][slot] : 0;
                    else h = slot >= 0 && slot < 128 ? ex->tex[st][slot] : 0;
                    tex_info(ex, r, h, img, r->separate, &o[k].image);
                    o[k].w[2] = (uint32_t)o[k].image.imageLayout;   // zero the tail padding
                }
                vk_texture_unlock();
            }
        } else {
            rt_fatal("vk: binding %u has an unknown kind %d", (unsigned)b->binding, (int)b->kind);
        }
        if (need_w) {
            w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                          .dstBinding = b->binding, .descriptorCount = cnt, .descriptorType = b->type};
            if (kind == 0) w[i].pBufferInfo = &o->buffer;   // a VkDescSlot is as large as either info struct
            else if (kind == 1) w[i].pImageInfo = &o->image;
            else w[i].pTexelBufferView = &bv[nd];
        }
        nd += cnt;
    }
    // Set 1's first descriptor is noted at its first binding; with no bindings it starts at the end so set 0's byte count
    // covers only its descriptors (else its template would push 0 bytes).
    if (n_sets == 2 && lo[1] == info->n_bindings) d0[1] = nd;
    if (gfx && info->layout != B.ds_layout) {   // another layout: sets bound before may be disturbed
        B.ds_layout = info->layout;
        B.ds_bound[0] = B.ds_bound[1] = VK_NULL_HANDLE;
        B.ds_pushed[0] = B.ds_pushed[1] = 0;
    }
    for (uint32_t k = 0; k < n_sets; k++) {
        uint32_t s = k, i0 = lo[s], i1 = hi[s];
        const VkDescSlot *ds = &d[d0[s]];
        uint32_t dbytes = (uint32_t)(((s + 1 < n_sets ? d0[s + 1] : nd) - d0[s]) * sizeof *d);
        if (s && i1 == i0) continue;       // no fragment stage: set 1 is empty and unused
        if (s ? info->push1 : info->push) {
            uint32_t n = gfx ? dbytes : 0;
            if (n && B.ds_pushed[s] && DS.pn[s] == n && !memcmp(DS.psig[s], ds, n)) {
                continue;
            }
            if (info->tpl[s]) rec_push_tpl(vk_pipeline_push_tpl_fn(), ex->cb, info->tpl[s], info->layout, s, ds, dbytes);
            else rec_push(vk_pipeline_push_fn(), ex->cb, info->bind_point, info->layout, s, i1 - i0, w + i0);
            if (gfx) {
                B.ds_bound[s] = VK_NULL_HANDLE, B.ds_pushed[s] = n != 0;
                if (n) DS.pn[s] = n, memcpy(DS.psig[s], ds, n);
            }
        } else if (!gfx || s > 1) {
            VkDescriptorSet set = alloc_set(ex, s ? info->set_layout1 : info->set_layout);
            write_set(info, s, set, ds, w, i0, i1);
            vkCmdBindDescriptorSets(ex->cb, info->bind_point, info->layout, s, 1, &set, 0, NULL);
        } else {
            VkDescriptorSetLayout sl = s ? info->set_layout1 : info->set_layout;
            VkDescriptorSet set = B.ds_last[s];
            if (!set || B.ds_last_layout[s] != sl || DS.n[s] != dbytes || memcmp(DS.sig[s], ds, dbytes)) {
                set = alloc_set(ex, sl);
                write_set(info, s, set, ds, w, i0, i1);
                B.ds_last[s] = set, B.ds_last_layout[s] = sl, DS.n[s] = dbytes;
                memcpy(DS.sig[s], ds, dbytes);
            }
            if (B.ds_bound[s] != set) {
                vkCmdBindDescriptorSets(ex->cb, info->bind_point, info->layout, s, 1, &set, 0, NULL);
                B.ds_bound[s] = set, B.ds_pushed[s] = 0;
            }
        }
    }
    if (skip) {
        RS.rec = 0;
        // An equal next draw records nothing when every push set holds its push and the build is repeatable.
        int valid = !RS.volatile_ && !RS.cnt_over && RS.hn;
        for (uint32_t s = 0; s < n_sets; s++)
            if (hi[s] > lo[s] && (s ? info->push1 : info->push) && !B.ds_pushed[s]) valid = 0;
        B.res_valid = valid;
    }
    ex->recorded++;
}

// ---- draws

static void bind_pipeline(VkExec *ex, VkPipelineBindPoint bp, VkPipeline p) {
    int k = bp == VK_PIPELINE_BIND_POINT_COMPUTE;
    if (B.pipe[k] == p) return;
    vkCmdBindPipeline(ex->cb, bp, p);
    ex->recorded++;
    B.pipe[k] = p;
    if (!k) B.st_valid = 0;
}

static void record_fixed(VkExec *ex) {
    VkCommandBuffer cb = ex->cb;
    float jx = 0, jy = 0;   // the scene draws' sub-pixel shift (0 for every other draw)
    dlss_jitter_for(ex, &jx, &jy);
    if ((ex->dirty & VK_EXEC_DIRTY_VIEWPORT) || jx != B.vjit[0] || jy != B.vjit[1]) {
        float x = 0, y = 0, w = (float)ex->pass_w, h = (float)ex->pass_h;
        if (ex->has_vp) x = ex->vp[0][0], y = ex->vp[0][1], w = ex->vp[0][2], h = ex->vp[0][3];
        if (w < 1) w = 1;
        if (h < 1) h = 1;
        VkViewport v = {x + jx, y + h + jy, w, -h, ex->depth_range[0], ex->depth_range[1]};   // the y flip: negative height
        B.vjit[0] = jx, B.vjit[1] = jy;
        vkCmdSetViewportWithCount(cb, 1, &v);
        ex->recorded++;
    }
    if (ex->dirty & VK_EXEC_DIRTY_SCISSOR) {
        VkRect2D r = vk_exec_scissor(ex);
        vkCmdSetScissorWithCount(cb, 1, &r);
        ex->recorded++;
    }
    if (ex->dirty & VK_EXEC_DIRTY_DEPTH_BIAS) {
        vkCmdSetDepthBias(cb, ex->poly_offset.units, ex->poly_offset.clamp, ex->poly_offset.factor);
        ex->recorded++;
    }
    if (ex->dirty & VK_EXEC_DIRTY_BLEND_CONST) {
        vkCmdSetBlendConstants(cb, ex->blend_color);
        ex->recorded++;
    }
    if (ex->dirty & VK_EXEC_DIRTY_STENCIL) {
        for (int f = 0; f < 2; f++) {
            VkStencilFaceFlags face = f ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT;
            vkCmdSetStencilCompareMask(cb, face, ex->st_vmask[f]);
            vkCmdSetStencilWriteMask(cb, face, ex->st_wmask[f]);
            vkCmdSetStencilReference(cb, face, ex->st_ref[f]);
            ex->recorded += 3;
        }
    }
    ex->dirty = 0;
}

static void bind_vertex_buffers(VkExec *ex) {
    VkBuffer vb[17];
    VkDeviceSize vo[17], vsz[17], vstr[17];
    int vcls[17] = {0};
    void *vpool[17] = {0};
    int dyn = vk_pipeline_vertex_input_dynamic();
    // A stream whose range matches the one its bound buffer was resolved from, with no pool or mirror change since (vk_mem_gen), resolves to that buffer again.
    uint64_t gen = vk_mem_gen();
    int same_gen = B.vb_valid && B.vgen == gen;
    for (int b = 0; b < 17; b++) {
        vb[b] = Z.zbuf, vo[b] = 0, vsz[b] = ZERO_BYTES, vstr[b] = b < 16 ? ex->pipe.vstream[b].stride : 0;
        if (b == VK_PIPE_ZERO_BINDING) continue;
        const NVNbufferRange *r = &ex->vbuf[b];
        if (same_gen && B.vin[b].address == r->address && B.vin[b].size == r->size) {
            vb[b] = B.vb[b], vo[b] = B.vo[b], vsz[b] = B.vsz[b], vcls[b] = B.vcls[b], vpool[b] = B.vpool[b];
            continue;
        }
        VkMemSpan sp;
        if (!r->address || !vk_mem_resolve(r->address, 1, &sp)) continue;
        VK_MEM_USE(sp);
        uint64_t n = r->size && r->size < sp.avail ? r->size : sp.avail;
        vb[b] = sp.buffer, vo[b] = sp.offset, vsz[b] = n, vcls[b] = sp.cls, vpool[b] = sp.pool;
    }
    memcpy(B.vcls, vcls, sizeof vcls), memcpy(B.vpool, vpool, sizeof vpool);
    memcpy(B.vin, ex->vbuf, sizeof B.vin);
    B.vgen = gen;
    if (B.vb_valid && !memcmp(B.vb, vb, sizeof vb) && !memcmp(B.vo, vo, sizeof vo) && !memcmp(B.vsz, vsz, sizeof vsz) &&
        (dyn || !memcmp(B.vstr, vstr, sizeof vstr)))
        return;
    vkCmdBindVertexBuffers2(ex->cb, 0, 17, vb, vo, vsz, dyn ? NULL : vstr);
    ex->recorded++;
    for (int b = 0; b < 17; b++)
        if (vcls[b]) vk_mem_count(vcls[b], VK_MUSE_VB, vsz[b]);
    if (rt_trace & RT_TRACE_PERF)
        for (int b = 0; b < 17; b++)
            if (vpool[b]) vk_mem_bound_note(vpool[b]);
    memcpy(B.vb, vb, sizeof vb), memcpy(B.vo, vo, sizeof vo), memcpy(B.vsz, vsz, sizeof vsz), memcpy(B.vstr, vstr, sizeof vstr);
    B.vb_valid = 1;
}

// Which draws are skipped, rather than waited for, while their shaders or pipeline build: draws into per-frame targets,
// full-screen draws included (one-shot draws wait, see prepare_draw).
static int pipe_skip_mode(void) {
    enum { mode = 2 };
    static int logged;
    if (!logged) {
        logged = 1;
        rt_log("vk_exec: draws skipped while their pipeline builds: those and full-screen draws");
    }
    return mode;
}

// The previous draw's shaders, kept while its program, pipeline state, the first 32 texture and image handles of its vertex and
// fragment stages and every slot's resolution (vk_texture_slots_gen, read unlocked) stay the same. tex_serial (bumped by every
// texture or image bind) equal to the key's skips the 1 KiB handle compare; a rebind to the same handles still matches by it.
typedef struct {
    const prog_t *prog;
    uint64_t serial;
    uint32_t gen, tex_serial;
    uint64_t tex[3][32], img[3][32];
    uint8_t geometry_input;
    VkPipeState pipe;
} MemoKey;

static struct { int on, valid, has_fs, has_gs; MemoKey k; const GfxShader *vs, *fs, *gs; } SM = {-1};

// Numeric handles are relative to the active descriptor tables: per-handle memos hold only while the tables and their
// address mappings stay unchanged (a sampler table switch also changes combined image descriptors).
static void descriptor_state(VkExec *ex) {
    static struct { const VkExec *ex; dpool_ref tex, samp; uint64_t mem_gen; uint32_t slots_gen; } last;
    uint64_t mg = vk_mem_gen();
    uint32_t sg = vk_texture_slots_gen();
    if (last.ex != ex || last.tex.addr != ex->tex_pool.addr || last.tex.count != ex->tex_pool.count ||
        last.samp.addr != ex->samp_pool.addr || last.samp.count != ex->samp_pool.count || last.mem_gen != mg) {
        if (!++descriptor_serial) rt_fatal("vk_exec: descriptor context serial exhausted");
        SM.valid = 0;
        ex->fb_fs = NULL;
        last.ex = ex, last.tex = ex->tex_pool, last.samp = ex->samp_pool, last.mem_gen = mg;
    }
    if (last.slots_gen != sg) ex->fb_fs = NULL;
    last.slots_gen = sg;
}

static void memo_key(const VkExec *ex, uint32_t gen, MemoKey *k) {
    k->prog = ex->prog, k->serial = ex->prog->serial, k->gen = gen, k->tex_serial = ex->tex_serial;
    memcpy(k->tex[0], ex->tex[NVN_STAGE_VS], sizeof k->tex[0]), memcpy(k->tex[1], ex->tex[NVN_STAGE_FS], sizeof k->tex[1]);
    memcpy(k->img[0], ex->img[NVN_STAGE_VS], sizeof k->img[0]), memcpy(k->img[1], ex->img[NVN_STAGE_FS], sizeof k->img[1]);
    k->geometry_input = geometry_input;
    if (geometry_input) {
        memcpy(k->tex[2], ex->tex[NVN_STAGE_GS], sizeof k->tex[2]);
        memcpy(k->img[2], ex->img[NVN_STAGE_GS], sizeof k->img[2]);
    }
    memcpy(&k->pipe, &ex->pipe, sizeof k->pipe);
}

static int memo_match(VkExec *ex, uint32_t gen, MemoKey *k) {
    descriptor_state(ex);
    if (!SM.valid) return 0;
    if (k->prog != ex->prog || k->serial != ex->prog->serial || k->gen != gen || k->geometry_input != geometry_input ||
        memcmp(&k->pipe, &ex->pipe, sizeof k->pipe))
        return 0;
    if (geometry_input && (memcmp(k->tex[2], ex->tex[NVN_STAGE_GS], sizeof k->tex[2]) ||
                           memcmp(k->img[2], ex->img[NVN_STAGE_GS], sizeof k->img[2]))) return 0;
    if (k->tex_serial == ex->tex_serial) return 1;
    if (memcmp(k->tex[0], ex->tex[NVN_STAGE_VS], sizeof k->tex[0]) || memcmp(k->tex[1], ex->tex[NVN_STAGE_FS], sizeof k->tex[1]) ||
        memcmp(k->img[0], ex->img[NVN_STAGE_VS], sizeof k->img[0]) || memcmp(k->img[1], ex->img[NVN_STAGE_FS], sizeof k->img[1]))
        return 0;
    k->tex_serial = ex->tex_serial;
    return 1;
}

static struct { int on, valid; VkPipeState s; const VkPipeInfo *info; } PM = {-1};

// 1: an opted-in ordinary GS is required; 0: keep the VS/FS path; -1: this opted-in draw is unsupported. Raw NVN2 markers are read before any NVN1 control offsets.
static int geometry_draw(VkExec *ex, uint32_t prim) {
    geometry_input = 0;
    if (native_gs_on() && ex->prog) {   // native failures keep VS/FS; ordinary stages retain their separate opt-in
        const uint8_t *ctl, *code, *fctl, *fcode;
        uint32_t len, flen, sph[5];
        if (!find_stage(ex->prog, NVN_STAGE_GS, &ctl, &code, &len)) return 0;
        if (nvn2_origin(code)) {
            if (len < 0x30 + 0x60 || !find_stage(ex->prog, NVN_STAGE_FS, &fctl, &fcode, &flen) ||
                vk_geometry_input(prim) != GFX_GEOMETRY_TRIANGLES) return 0;
            memcpy(sph, code + 0x30, sizeof sph);
            if (!(sph[0] & (1u << 24)) || sph[2] >> 24 != 1 || (sph[3] >> 24 & 15) != 7 || (sph[4] & 0xfff) != 1) {
                warn_once(W_GEOMETRY, "native GS: a stage outside the passthrough subset keeps the vertex and fragment path");
                return 0;
            }
            geometry_input = GFX_GEOMETRY_TRIANGLES;
            return 1;
        }
    }
    return 0;
}


// Pass, shaders, pipeline, fixed state and resources for a draw. 0: the draw is skipped.

static uint64_t stage_origin(const VkExec *ex, int st);   // below: nvn2_origin of a stage, per program serial
static int prepare_draw(VkExec *ex, int prim, int32_t count, int restart) {
    if (ex->render_disabled) {
        return 0;
    }
    int want_gs = geometry_draw(ex, (uint32_t)prim);
    cb_res(ex);
    dummies_init();
    if (ex->pass_noatt) {   // no attachment: the pass covers this draw's viewport or scissor, reopened when it needs more
        uint32_t w, h;
        vk_exec_noatt_dims(ex, &w, &h);
        if (ex->in_pass && (w > ex->pass_w || h > ex->pass_h)) vk_exec_pass_end(ex);
        if (!ex->in_pass) ex->rt_resolved = 0;
    }
    if (!ex->in_pass) ex->dirty |= VK_EXEC_DIRTY_VIEWPORT | VK_EXEC_DIRTY_SCISSOR;
    vk_exec_pass_begin(ex);
    VkRect2D sc = vk_exec_scissor(ex);
    if (!sc.extent.width || !sc.extent.height) return 0;
    if (!ex->prog) return warn_once(W_PROGRAM, "a draw without a bound program is skipped"), 0;
    int has_vs, has_fs, has_gs = 0, vs_wait = 0, fs_wait = 0, gs_wait = 0;
    const GfxShader *vs, *fs, *gs = NULL;
    uint32_t gen = vk_texture_slots_gen();   // before shader_for, whose lookups may bump it
    if (SM.valid && memo_match(ex, gen, &SM.k)) {
        vs = SM.vs, fs = SM.fs, gs = SM.gs, has_vs = 1, has_fs = SM.has_fs, has_gs = SM.has_gs;
    } else {
        SM.valid = 0;
        vs = shader_for(ex, NVN_STAGE_VS, &has_vs, &vs_wait, 0);
        fs = shader_for(ex, NVN_STAGE_FS, &has_fs, &fs_wait, 0);
        native_gs_vs = vs, native_gs_fs = fs;
        if (want_gs) gs = shader_for(ex, NVN_STAGE_GS, &has_gs, &gs_wait, 0);
        if (vs && !vs_wait && !fs_wait && !gs_wait && (!has_fs || fs) && (!want_gs || (has_gs && gs)))
            memo_key(ex, gen, &SM.k), SM.vs = vs, SM.fs = fs, SM.gs = gs, SM.has_fs = has_fs, SM.has_gs = has_gs, SM.valid = 1;
    }
    if (!has_vs) {
        return warn_once(W_NO_VS, "a draw with a program that has no vertex stage is skipped"), 0;
    }
    // A draw into a target the previous frame did not render (or into none) waits for a pending shader or pipeline (a
    // one-shot draw would leave its target empty); a draw into per-frame targets is skipped until ready.
    int oneshot = -1;   // -1: not looked up
    pipe_skip_mode();
    if (vs_wait || fs_wait || gs_wait) {
        if (!(oneshot = !vk_exec_targets_recent(ex, 1))) {
            return perf_count(PERF_CNT_SHADER_SKIP), 0;
        }
        perf_count(PERF_CNT_ONESHOT_WAIT);
        if (vs_wait) vs = shader_for(ex, NVN_STAGE_VS, &has_vs, NULL, 1);
        if (fs_wait) fs = shader_for(ex, NVN_STAGE_FS, &has_fs, NULL, 1);
        native_gs_vs = vs, native_gs_fs = fs;
        if (gs_wait) gs = shader_for(ex, NVN_STAGE_GS, &has_gs, NULL, 1);
    }
    if (!vs || (has_fs && !fs) || (want_gs && (!has_gs || !gs))) {
        return 0;
    }
    if (!geometry_feedback_safe(ex, vs, gs)) return 0;
    uint32_t fbl = feedback_prepare(ex, fs);
    VkPipeState s = ex->pipe;
    s.vs = vs, s.fs = fs, s.gs = gs;
    s.feedback = (uint8_t)((fbl & 0xff ? 1 : 0) | (fbl & 0x100 ? 2 : 0));
    s.prim = (uint8_t)prim;
    s.depth_minus_w = (uint8_t)(nvn_depth_minus_w() != 0);
    s.y_flip = 1;
    s.prim_restart = (uint8_t)(restart != 0);
    const VkPipeInfo *info = NULL;
    int pending;
    // A one-shot draw waits for its pipeline; everything else is skipped while it builds.
    uint32_t wait_ms = 0;
    if (oneshot > 0) {
        wait_ms = ONESHOT_WAIT_MS;
    }
    // The previous draw's pipeline while its state repeats (PM): the cache key is made from the state alone, and a found
    // pipeline only changes from its fast link to its optimised build.
    VkPipeline p;
    const char *why = NULL;
    uint64_t w0 = 0;
    if (PM.valid && !memcmp(&PM.s, &s, sizeof s)) {
        info = PM.info, pending = 0;
        p = __atomic_load_n(&info->pipeline, __ATOMIC_ACQUIRE);
    } else {
        w0 = wait_ms ? kern_now_ns() : 0;   // the clock is read only for draws that may wait
        p = vk_pipeline_graphics_async(&s, &info, &pending, wait_ms);
        why = oneshot > 0 ? "one-shot, shader" : NULL;
        if (pending && oneshot < 0 && !vk_exec_targets_recent(ex, 1)) {
            perf_count(PERF_CNT_ONESHOT_WAIT);
            why = "one-shot";
            if (!w0) w0 = kern_now_ns();
            p = vk_pipeline_graphics_async(&s, &info, &pending, ONESHOT_WAIT_MS);
        }
        PM.valid = !pending && p && info;
        if (PM.valid) PM.s = s, PM.info = info;
    }
    uint64_t waited = w0 ? kern_now_ns() - w0 : 0;
    static int wait_logs;
    if (why && waited >= 2000000 && wait_logs < 200) {
        char tg[512];
        wait_logs++;
        vk_exec_targets_describe(ex, tg, sizeof tg);
        rt_log("vk_exec: frame %llu: a draw waited %.1f ms for its pipeline (%s%s): %u vertices, pass %ux%u, %s",
               (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1, waited / 1e6, why,
               pending ? ", skipped after all" : "", count, ex->pass_w, ex->pass_h, tg);
    }
    if (pending) {
        return 0;
    }
    if (!p || !info) {
        return warn_once(W_PIPELINE, "a graphics pipeline could not be created: draws with it are skipped"), 0;
    }
    bind_pipeline(ex, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    if (!B.st_valid || memcmp(&B.st, &s, sizeof s)) {
        vk_pipeline_set_dynamic(ex->cb, &s);
        ex->recorded++;
        B.st = s, B.st_valid = 1;
    }
    record_fixed(ex);
    bindless_bind(ex, info);
    bind_resources(ex, info);
    if (fbl & ex->fb_dirty) feedback_barrier(ex);
    feedback_written(ex, &s);
    bind_vertex_buffers(ex);
    dlss_track_camera(ex, NVN_STAGE_VS);
    vk_exec_oq_draw(ex);
    return 1;
}

// ---- ind_clamp: an indirect dispatch, mesh launch or indexed indirect draw reads its counts from a copy that a one-thread
// compute shader (shaders/ind_clamp.comp) checked: an axis over IC_AXIS or a total over IC_TOTAL is zeroed and counted.
// Records the GPU did not write this frame hold stale bytes. The copies go to a device-local ring of IC_SLOTS 32-byte slots.
#include "ind_clamp_spv.h"
#define IC_SLOTS 65536u
#define IC_AXIS 65535u
#define IC_TOTAL (1u << 20)
typedef struct { uint64_t prog, frame; uint32_t kind, raw; } IcInfo;
static struct {
    int state;   // 0 not tried, 1 ready, -1 unavailable
    VkPipeline pipe;
    VkPipelineLayout layout;
    VkBuffer ring, cnt;
    VmaAllocation ring_a, cnt_a;
    VkDeviceAddress ring_addr, cnt_addr;
    volatile uint32_t *cmap;
    uint32_t next, seen;
    IcInfo info[IC_SLOTS];
} IC;

static int ic_init(void) {
    if (IC.state) return IC.state > 0;
    IC.state = -1;
    VkCtx *c = vk_ctx();
    if (!c->has_buffer_device_address) { rt_log("ind_clamp: no buffer device address: off"); return 0; }
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)IC_SLOTS * 32,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateBuffer(c->vma, &bi, &ai, &IC.ring, &IC.ring_a, NULL) != VK_SUCCESS) { rt_log("ind_clamp: no ring: off"); return 0; }
    bi.size = 64, bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VmaAllocationCreateInfo hi = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                  .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(c->vma, &bi, &hi, &IC.cnt, &IC.cnt_a, &info) != VK_SUCCESS || !info.pMappedData) {
        rt_log("ind_clamp: no counter: off"); return 0;
    }
    IC.cmap = info.pMappedData;
    memset((void *)IC.cmap, 0, 64);
    VkBufferDeviceAddressInfo da = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = IC.ring};
    IC.ring_addr = vkGetBufferDeviceAddress(c->device, &da);
    da.buffer = IC.cnt;
    IC.cnt_addr = vkGetBufferDeviceAddress(c->device, &da);
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof ind_clamp_spv,
                                   .pCode = ind_clamp_spv};
    VkShaderModule m;
    if (vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS) { rt_log("ind_clamp: shader module failed: off"); return 0; }
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1,
                                     .pPushConstantRanges = &pr};
    VkComputePipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"}};
    int ok = vkCreatePipelineLayout(c->device, &li, NULL, &IC.layout) == VK_SUCCESS;
    pi.layout = IC.layout;
    ok = ok && vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &pi, NULL, &IC.pipe) == VK_SUCCESS;
    vkDestroyShaderModule(c->device, m, NULL);
    if (!ok) { rt_log("ind_clamp: pipeline failed: off"); return 0; }
    IC.state = 1;
    rt_log("ind_clamp: on: indirect records over %u groups per axis or %u in all are zeroed", IC_AXIS, IC_TOTAL);
    return 1;
}

// The record at (src, off): mode 0 a VkDispatchIndirectCommand, 1 a mesh launch count (one word; y = z = 1), 2 a
// VkDrawIndexedIndirectCommand. Returns 1 with *buf, *boff the checked copy to read instead; 0 (unavailable): read the original.
static int ind_clamp(VkExec *ex, VkBuffer src, VkDeviceSize off, int mode, VkBuffer *buf, VkDeviceSize *boff) {
    if (!ic_init()) return 0;
    uint32_t slot = IC.next++ % IC_SLOTS;
    IC.info[slot] = (IcInfo){ex->prog ? ex->prog->serial : 0, vk_frame_no + 1, (uint32_t)mode, 0};
    VkDeviceSize o = (VkDeviceSize)slot * 32;
    vk_exec_barrier(ex);   // the producer's writes before the copy (also ends a pass)
    VkBufferCopy cp = {off, o, mode == 1 ? 4 : mode == 2 ? 20 : 12};
    vkCmdCopyBuffer(ex->cb, src, IC.ring, 1, &cp);
    ex->recorded++;
    vk_exec_barrier(ex);
    vkCmdBindPipeline(ex->cb, VK_PIPELINE_BIND_POINT_COMPUTE, IC.pipe);
    struct { VkDeviceAddress rec, cnt; uint32_t mode, slot, max_axis, max_total; } pc = {IC.ring_addr + o, IC.cnt_addr,
        (uint32_t)mode, slot, IC_AXIS, IC_TOTAL};
    vkCmdPushConstants(ex->cb, IC.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(ex->cb, 1, 1, 1);
    ex->recorded += 3;
    vk_exec_barrier(ex);   // the checked record before the indirect read
    vk_exec_draw_new_cb();   // our pipeline and push constants replaced the replay's: bind everything again
    ex->dirty = VK_EXEC_DIRTY_ALL;
    *buf = IC.ring, *boff = o;
    return 1;
}

void vk_ind_clamp_frame(uint64_t frame) {
    if (IC.state <= 0 || !IC.cmap) return;
    uint32_t n = IC.cmap[0];
    if (n == IC.seen) return;
    uint32_t slot = IC.cmap[1] % IC_SLOTS;
    const IcInfo *i = &IC.info[slot];
    if (IC.seen < 64 || n / 100 != IC.seen / 100)
        rt_log("ind_clamp: frame %llu: %u records zeroed so far; newest: %s of program %llu (frame %llu), raw %u x %u x %u",
               (unsigned long long)frame, n, i->kind == 1 ? "mesh launch" : i->kind == 2 ? "indexed indirect draw" : "indirect dispatch", (unsigned long long)i->prog,
               (unsigned long long)i->frame, IC.cmap[2], IC.cmap[3], IC.cmap[4]);
    IC.seen = n;
}

#include "vk_exec_mesh.h"

// nvn2_origin of the program's stage that find_stage gives, per program serial: prepare_draw asks at every draw and each
// ask takes a lock and a map lookup. A zero (not lowered) is kept until nvn2_origin_gen moves. Replay thread only.
static uint64_t stage_origin(const VkExec *ex, int st) {
    static struct { uint64_t serial; const prog_t *prog; uint64_t o[3], zgen[3]; } m[256];
    const uint8_t *ctl, *code;
    uint32_t len;
    if (!ex->prog) return 0;
    int k = st == NVN_STAGE_VS ? 0 : st == NVN_STAGE_FS ? 1 : st == NVN_STAGE_CS ? 2 : -1;
    if (k < 0 || !ex->prog->serial) return find_stage(ex->prog, st, &ctl, &code, &len) ? nvn2_origin(code) : 0;
    typeof(m[0]) *e = &m[(ex->prog->serial * 0x9E3779B97F4A7C15ull) >> 56];
    if (e->serial != ex->prog->serial || e->prog != ex->prog) memset(e, 0, sizeof *e), e->serial = ex->prog->serial, e->prog = ex->prog;
    if (!e->o[k]) {
        uint64_t gen = nvn2_origin_gen();
        if (e->zgen[k] == gen) return 0;   // no origin, and none added since
        e->o[k] = find_stage(ex->prog, st, &ctl, &code, &len) ? nvn2_origin(code) : 0;
        e->zgen[k] = gen;
    }
    return e->o[k];
}

static void note_work(const VkExec *ex, int kind, uint64_t work) {
    if (!vk_sync_noting() || !ex->prog) return;
    int cs = kind >= 2;
    uint64_t a = stage_origin(ex, cs ? NVN_STAGE_CS : NVN_STAGE_VS);
    uint64_t b = !cs ? stage_origin(ex, NVN_STAGE_FS) : 0;
    vk_sync_note(kind, a, b, work);
}

static void bind_index(VkExec *ex, const void *data, uint64_t bytes, VkIndexType type) {
    VkBuffer b;
    VkDeviceSize o;
    memcpy(scratch(ex, bytes, &b, &o), data, bytes);
    vkCmdBindIndexBuffer(ex->cb, b, o, type);
    ex->recorded++;
}

static void op_draw(VkExec *ex, const p_draw *d) {
    c0_binst = (uint32_t)d->base_instance;
    if (d->count <= 0 || !prepare_draw(ex, d->prim, d->count, 0)) return;
    vk_exec_frame.draws++;
    uint32_t inst = d->instances > 0 ? (uint32_t)d->instances : 1, first = (uint32_t)d->first, n = (uint32_t)d->count;
    if (d->prim == NVN_DRAW_PRIMITIVE_QUADS && n >= 4) {   // quads become two triangles each
        uint32_t q = n / 4;
        uint32_t *ix = malloc(sizeof *ix * 6 * q);
        if (!ix) rt_fatal("vk: out of memory");
        for (uint32_t i = 0; i < q; i++) {
            uint32_t v = first + 4 * i, k[6] = {0, 1, 2, 0, 2, 3};
            for (int j = 0; j < 6; j++) ix[6 * i + j] = v + k[j];
        }
        bind_index(ex, ix, sizeof *ix * 6 * q, VK_INDEX_TYPE_UINT32);
        free(ix);
        vkCmdDrawIndexed(ex->cb, 6 * q, inst, 0, 0, (uint32_t)d->base_instance);
    } else if (d->prim == NVN_DRAW_PRIMITIVE_LINE_LOOP && n >= 2) {   // a strip that returns to its first vertex
        uint32_t *ix = malloc(sizeof *ix * (n + 1));
        if (!ix) rt_fatal("vk: out of memory");
        for (uint32_t i = 0; i < n; i++) ix[i] = first + i;
        ix[n] = first;
        bind_index(ex, ix, sizeof *ix * (n + 1), VK_INDEX_TYPE_UINT32);
        free(ix);
        vkCmdDrawIndexed(ex->cb, n + 1, inst, 0, 0, (uint32_t)d->base_instance);
    } else {
        vkCmdDraw(ex->cb, n, inst, first, (uint32_t)d->base_instance);
    }
    ex->recorded++;
    note_work(ex, 0, (uint64_t)n * inst);
}

// Primitive restart of an indexed draw: NVN restarts at the SetPrimitiveRestart value, Vulkan only at all-ones of the
// bound index type (byte indices are widened to 16 bits, op_draw_idx) and only for strips and fans.
static int restart_for(const VkExec *ex, int prim, int itype) {
    if (!ex->prim_restart.enable) return 0;
    switch (vk_pipeline_topology(prim)) {
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY: case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY: break;
    default: return 0;   // lists: a restart index only drops the unfinished primitive
    }
    static const uint32_t all_ones[3] = {0xffu, 0xffffu, 0xffffffffu};
    if (ex->prim_restart.index == all_ones[itype]) return 1;
    warn_once(W_RESTART, "primitive restart index %#x with %d-byte indices is not replayed: strips drawn without restart",
              ex->prim_restart.index, 1 << itype);
    return 0;
}

static void op_draw_idx(VkExec *ex, const p_draw_idx *d) {
    c0_binst = (uint32_t)d->base_instance;
    static const uint32_t isize[3] = {1, 2, 4};
    if (d->itype < 0 || d->itype > 2) return warn_once(W_INDEX, "index type %d is unknown: draw skipped", d->itype);
    int restart = restart_for(ex, d->prim, d->itype);
    if (d->count <= 0 || !prepare_draw(ex, d->prim, d->count, restart)) return;
    vk_exec_frame.draws++;
    uint64_t bytes = (uint64_t)d->count * isize[d->itype];
    VkMemSpan sp;
    if (!vk_mem_resolve(d->ibuf, bytes, &sp)) return warn_once(W_INDEX, "index buffer %#llx is not backed: draws with it are skipped", (unsigned long long)d->ibuf);
    if (d->itype == NVN_INDEX_TYPE_UNSIGNED_BYTE) {   // no uint8 index type here: widen (host-readable backing only)
        if (sp.alloc) return warn_once(W_INDEX, "byte indices in device-local memory: draw skipped");
        vk_mem_count(sp.cls, VK_MUSE_CPU_READ, bytes);
        vk_mem_count(VK_MCLS_SCRATCH, VK_MUSE_IB, 2 * bytes);
        uint16_t *ix = malloc(sizeof *ix * (size_t)d->count);
        if (!ix) rt_fatal("vk: out of memory");
        for (int i = 0; i < d->count; i++) {
            uint8_t v = ((const uint8_t *)(uintptr_t)d->ibuf)[i];
            ix[i] = restart && v == 0xff ? 0xffff : v;
        }
        bind_index(ex, ix, sizeof *ix * (size_t)d->count, VK_INDEX_TYPE_UINT16);
        free(ix);
    } else if (sp.offset % isize[d->itype]) {
        return warn_once(W_INDEX, "index buffer %#llx is not aligned to its index size: draw skipped", (unsigned long long)d->ibuf);
    } else {
        VK_MEM_USE(sp);
        vkCmdBindIndexBuffer(ex->cb, sp.buffer, sp.offset, d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
        ex->recorded++;
        vk_mem_count(sp.cls, VK_MUSE_IB, bytes);
        VK_MEM_BOUND(sp);
    }
    vkCmdDrawIndexed(ex->cb, (uint32_t)d->count, d->instances > 0 ? (uint32_t)d->instances : 1, 0, d->base_vertex,
                     (uint32_t)d->base_instance);
    ex->recorded++;
    note_work(ex, 0, (uint64_t)d->count * (uint64_t)(d->instances > 0 ? d->instances : 1));
}

// DrawElementsIndirect: the GPU reads the arguments from the pool buffer. The base instance stays out of c0 (0x644),
// which a transfer cannot patch inside a pass; gl_BaseInstance is the argument's. Byte indices are skipped.
static void op_draw_idx_ind(VkExec *ex, const p_draw_idx_ind *d) {
    c0_binst = 0;
    if (d->itype != NVN_INDEX_TYPE_UNSIGNED_SHORT && d->itype != NVN_INDEX_TYPE_UNSIGNED_INT)
        return warn_once(W_INDEX, "indirect draw with index type %d: skipped", d->itype);
    VkMemSpan ib, ip;
    if (!vk_mem_resolve(d->ind, sizeof(VkDrawIndexedIndirectCommand), &ip) || ip.offset % 4)
        return warn_once(W_INDIRECT, "indirect draw arguments at %#llx are not backed or not aligned: draws skipped", (unsigned long long)d->ind);
    if (!vk_mem_resolve(d->ibuf, 1, &ib) || ib.offset % (d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? 2 : 4))
        return warn_once(W_INDEX, "index buffer %#llx is not backed or not aligned: indirect draws skipped", (unsigned long long)d->ibuf);
    VkBuffer cb_buf = VK_NULL_HANDLE;   // ind_clamp's checked copy of the record (bit 2), else the game's (ip, below)
    VkDeviceSize cb_off = 0;
    if (!ex->render_disabled) {   // before prepare_draw: the check ends any pass
        VK_MEM_USE(ip);   // may move it to the mirror
        if (!ind_clamp(ex, ip.buffer, ip.offset, 2, &cb_buf, &cb_off)) cb_buf = VK_NULL_HANDLE;
    }
    if (!prepare_draw(ex, d->prim, 0x7fffffff, restart_for(ex, d->prim, d->itype))) return;
    vk_exec_frame.draws++;
    VK_MEM_USE(ip);
    VK_MEM_USE(ib);
    vkCmdBindIndexBuffer(ex->cb, ib.buffer, ib.offset, d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
    if (!cb_buf) cb_buf = ip.buffer, cb_off = ip.offset;
    vkCmdDrawIndexedIndirect(ex->cb, cb_buf, cb_off, 1, sizeof(VkDrawIndexedIndirectCommand));
    ex->recorded += 2;
    note_work(ex, 1, 0);
    // An imported pool's GPU address is its host address (bind_range reads it the same way); device memory is not.
    if (!ip.alloc && vk_sync_noting()) vk_sync_note_args(d->ind);
    vk_mem_count(ip.cls, VK_MUSE_IND, sizeof(VkDrawIndexedIndirectCommand));
    VK_MEM_BOUND(ip);
    VK_MEM_BOUND(ib);
}

// ---- dispatch and uniform buffer updates

// A compute dispatch skipped because its shader has no translation: the first one of each of 64 programs logs the
// storage images and buffers it would have written.
static void skipped_dispatch_log(VkExec *ex, const char *kind, int x, int y, int z) {
    static uint64_t seen[64];
    static int nseen;
    uint64_t serial = ex->prog ? ex->prog->serial : 0;
    for (int i = 0; i < nseen; i++) if (seen[i] == serial) return;
    if (nseen == 64) return;
    seen[nseen++] = serial;
    char s[768];
    int o = snprintf(s, sizeof s, "vk: skipped %s dispatch %dx%dx%d of program %llu:", kind, x, y, z, (unsigned long long)serial);
    vk_texture_lock();
    for (int slot = 0; slot < 32 && o < (int)sizeof s - 64; slot++) {
        uint64_t h = ex->img[NVN_STAGE_CS][slot];
        const VkTexImage *t = h ? vk_texture_by_id_locked(&ex->tex_pool, (int)(h & 0xfffff), NULL, NULL) : NULL;
        if (t) o += snprintf(s + o, sizeof s - o, " img%d=%ux%u/f%d@%#llx", slot, t->width, t->height, (int)t->format,
                             (unsigned long long)t->addr);
    }
    vk_texture_unlock();
    for (int slot = 0; slot < 16 && o < (int)sizeof s - 48; slot++)
        if (ex->ssbo[NVN_STAGE_CS][slot].addr)
            o += snprintf(s + o, sizeof s - o, " ssbo%d=%#llx+%#llx", slot, (unsigned long long)ex->ssbo[NVN_STAGE_CS][slot].addr,
                          (unsigned long long)ex->ssbo[NVN_STAGE_CS][slot].size);
    rt_log("%s", s);
}


// Before a dispatch that sums into the sky SH block (vk_memory.c skk_win) its window is zeroed in GPU order.
static void sky_sh_before_dispatch(VkExec *ex) {
    if (!vk_sky_sh_keep() || !ex->prog) return;
    static const void *last_prog;
    static int last_win = -1;
    if (ex->prog != last_prog) {
        uint64_t native = 0;
        for (uint32_t k = 0; k < ex->prog->nstage && k < 6 && !native; k++)
            native = nvn2_origin((const uint8_t *)(uintptr_t)ex->prog->sd[k].data);
        last_prog = ex->prog, last_win = vk_sky_sh_window(native);
    }
    if (last_win < 0) return;
    {   // the block is what the dispatch binds as storage slot 0 (0xc0 bytes): its address moves with the heap layout
        const p_bind_buf *b = &ex->ssbo[NVN_STAGE_CS][0];
        if (b->addr && b->size == 0xc0) vk_sky_sh_block(b->addr);
    }
    cb_res(ex);
    vk_exec_pass_end(ex);   // a fill is not allowed inside a render pass
    vk_sky_sh_clear(ex->cb, last_win);
}
static void op_dispatch(VkExec *ex, const p_dispatch *d) {
    sky_sh_before_dispatch(ex);
    if (d->x <= 0 || d->y <= 0 || d->z <= 0) return;
    if (!ex->prog) return warn_once(W_PROGRAM, "a dispatch without a bound program is skipped");
    // The game's neural upscaler is a chain of tensor-core (HMMA) compute stages, far too slow as software shuffles on a
    // GPU without tensor cores: they are dropped and the stage that would have written the larger picture gets a linear blit (upscale_standin).
    {
        for (uint32_t k = 0; k < ex->prog->nstage; k++)
            if (nvn2_is_tensor((const uint8_t *)(uintptr_t)ex->prog->sd[k].data)) {
                static int told;
                if (!told++) rt_log("tensor: the neural upscaler is replaced by a linear blit");
                if (!dlss_upscale(ex)) upscale_standin(ex);   // DLSS in place of the stand-in when its inputs fit
                return;
            }
    }
    cb_res(ex);
    dummies_init();
    int present;
    const GfxShader *cs = shader_for(ex, NVN_STAGE_CS, &present, NULL, 0);   // synchronous: a skipped dispatch changes results
    if (!cs) return skipped_dispatch_log(ex, "direct", d->x, d->y, d->z);
    const VkPipeInfo *info = NULL;
    VkPipeline p = vk_pipeline_compute(cs, &info);
    if (!p || !info) return warn_once(W_PIPELINE, "a compute pipeline is not ready or failed: its dispatches are skipped");
    vk_exec_barrier(ex);   // ends the pass; earlier writes are visible to the shader
    bind_pipeline(ex, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    c0_dim[0] = (uint32_t)d->x, c0_dim[1] = (uint32_t)d->y, c0_dim[2] = (uint32_t)d->z;
    c0_lay_set(ex, cs);
    twin_dispatch = 1;
    bindless_bind(ex, info);
    bind_resources(ex, info);
    twin_dispatch = 0;
    vkCmdDispatch(ex->cb, (uint32_t)d->x, (uint32_t)d->y, (uint32_t)d->z);
    ex->recorded++;
    note_work(ex, 2, (uint64_t)d->x * (uint64_t)d->y * (uint64_t)d->z);
    upscale_source_note(ex, info);
}

// DispatchComputeIndirect: NVN 1 shaders read the group counts at c0+0x200, so they are copied there on the GPU before
// dispatch; an NVN 2 bank at that offset holds roots or tables and stays unchanged.
static void op_dispatch_ind(VkExec *ex, const p_dispatch_ind *d) {
    sky_sh_before_dispatch(ex);
    if (!ex->prog) return warn_once(W_PROGRAM, "a dispatch without a bound program is skipped");
    // Tensor stages reached indirectly: same rule as op_dispatch, or the driver spends minutes on their SPIR-V.
    for (uint32_t k = 0; k < ex->prog->nstage; k++)
        if (nvn2_is_tensor((const uint8_t *)(uintptr_t)ex->prog->sd[k].data)) {
            static uint64_t count;
            uint64_t seen = count++;   // the first 8, then every 1024th
            if (!(seen & (seen < 8 ? 0 : 1023)))
                rt_log("tensor: indirect dispatch of tensor program %llu skipped (%llu so far)",
                       (unsigned long long)ex->prog->serial, (unsigned long long)count);
            return;
        }
    VkMemSpan ip;
    if (!vk_mem_resolve(d->ind, sizeof(VkDispatchIndirectCommand), &ip) || ip.offset % 4)
        return warn_once(W_INDIRECT, "indirect dispatch arguments at %#llx are not backed or not aligned: skipped", (unsigned long long)d->ind);
    cb_res(ex);
    dummies_init();
    int present;
    const GfxShader *cs = shader_for(ex, NVN_STAGE_CS, &present, NULL, 0);
    if (!cs) return skipped_dispatch_log(ex, "indirect", 0, 0, 0);
    const VkPipeInfo *info = NULL;
    VkPipeline p = vk_pipeline_compute(cs, &info);
    if (!p || !info) return warn_once(W_PIPELINE, "a compute pipeline is not ready or failed: its dispatches are skipped");
    vk_exec_barrier(ex);
    VK_MEM_USE(ip);
    VkBuffer ib = ip.buffer;   // the record the dispatch reads: the game's, or ind_clamp's checked copy
    VkDeviceSize io = ip.offset;
    ind_clamp(ex, ip.buffer, ip.offset, 0, &ib, &io);
    bind_pipeline(ex, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    c0_dim[0] = c0_dim[1] = c0_dim[2] = 0;
    c0_lay_set(ex, cs);
    B.c0_valid[2] = 0;   // a fresh c0 range: the copy below must not change one an earlier dispatch reads
    twin_dispatch = 1;
    bindless_bind(ex, info);
    bind_resources(ex, info);
    twin_dispatch = 0;
    if (B.c0_valid[2]) {   // the shader binds c0: the GPU's own group counts, at +0x200 (NVN 1) or +4 in the NVN 2 header
        VkBufferCopy cp = {io, B.c0_off[2] + (c0_lay ? 4 : 0x200), sizeof(VkDispatchIndirectCommand)};
        vkCmdCopyBuffer(ex->cb, ib, B.c0_buf[2], 1, &cp);
        ex->recorded++;
        vk_exec_barrier(ex);
        B.c0_valid[2] = 0;   // the range no longer holds B.c0[2]
    }
    vkCmdDispatchIndirect(ex->cb, ib, io);
    ex->recorded++;
    note_work(ex, 3, 0);
    vk_mem_count(ip.cls, VK_MUSE_IND, sizeof(VkDispatchIndirectCommand));
    VK_MEM_BOUND(ip);
}

static void op_update_ub(VkExec *ex, const p_update_ub *u) {
    if (!u->size) return;
    VkMemSpan sp;
    if (!vk_mem_resolve(u->dst, u->size, &sp)) return warn_once(W_UB, "UpdateUniformBuffer to %#llx is not backed: dropped", (unsigned long long)u->dst);
    cb_res(ex);
    vk_exec_barrier(ex);   // earlier draws finished with the old contents
    VkBuffer b;
    VkDeviceSize o;
    memcpy(scratch(ex, u->size, &b, &o), u->data, u->size);
    VK_MEM_PICK(sp);
    VkBufferCopy cp = {o, sp.offset, u->size};
    vkCmdCopyBuffer(ex->cb, b, sp.buffer, 1, &cp);
    ub_gen++;
    vk_mem_gpu_wrote(&sp, u->size);
    vk_usnap_drop(u->dst, u->size);
    ex->recorded++;
    vk_mem_count(sp.cls, VK_MUSE_GPU_WRITE, u->size);
    note_written(ex, u->dst, u->size);
    upd_note(ex, u);
}

// ---- bind ops

static void bind_handles(uint64_t *dst, int cap, const p_bind_handles *b) {
    if (b->stage < 0 || b->stage >= 8 || b->first < 0 || b->n < 0 || b->n > 32 || b->first + b->n > cap) {
        warn_once(W_BIND, "a bind of %d handles at %d of stage %d is out of range: ignored", b->n, b->first, b->stage);
        return;
    }
    memcpy(&dst[b->first], b->h, sizeof(uint64_t) * (size_t)b->n);
}

void vk_exec_op_draw(VkExec *ex, uint32_t op, const void *payload) {
    switch (op) {
    case OP_BIND_PROGRAM: {
        const p_bind_program *b = payload;
        ex->prog = b->prog, ex->prog_stages = b->stages;
        bind_gen++;
        break;
    }
    case OP_BIND_STATE: {
        const p_bind_state *b = payload;
        blend_st bs;
        switch (b->kind) {
        case 0:
            memcpy(&bs, b->bytes, sizeof bs);
            if (bs.target < 8) ex->pipe.blend[bs.target] = bs;
            break;
        case 1: memcpy(&ex->pipe.color, b->bytes, sizeof ex->pipe.color); break;
        case 2: memcpy(&ex->pipe.chmask, b->bytes, sizeof ex->pipe.chmask); break;
        case 3: memcpy(&ex->pipe.ms, b->bytes, sizeof ex->pipe.ms); break;
        case 4: memcpy(&ex->pipe.poly, b->bytes, sizeof ex->pipe.poly); break;
        case 5: memcpy(&ex->pipe.ds, b->bytes, sizeof ex->pipe.ds); break;
        case 6: memcpy(&ex->point_size, b->bytes, sizeof ex->point_size); break;
        case 7: memcpy(&ex->pipe.alpha_ref, b->bytes, sizeof ex->pipe.alpha_ref); break;
        }
        break;
    }
    case OP_BIND_VATTR: {
        const p_bind_vattr *b = payload;
        if (b->first < 0 || b->n < 0 || b->first + b->n > 16) break;
        memset(ex->pipe.vattr, 0, sizeof ex->pipe.vattr);   // the new list replaces the old one
        memcpy(&ex->pipe.vattr[b->first], b->st, sizeof(uint32_t) * (size_t)b->n);
        break;
    }
    case OP_BIND_VSTREAM: {
        const p_bind_vstream *b = payload;
        if (b->first < 0 || b->n < 0 || b->first + b->n > 16) break;
        memset(ex->pipe.vstream, 0, sizeof ex->pipe.vstream);
        memcpy(&ex->pipe.vstream[b->first], b->st, sizeof(vs_st) * (size_t)b->n);
        break;
    }
    case OP_BIND_VBUF: {
        const p_bind_vbuf *b = payload;
        if (b->first < 0 || b->n < 0 || b->first + b->n > 16) break;
        memcpy(&ex->vbuf[b->first], b->r, sizeof(NVNbufferRange) * (size_t)b->n);
        break;
    }
    case OP_BIND_UBO: {
        const p_bind_buf *b = payload;
        if (b->stage >= 0 && b->stage < 8 && b->index >= 0 && b->index < 18) ex->ubo[b->stage][b->index] = *b;
        bind_gen++;
        break;
    }
    case OP_PUSH_CONST: {
        const p_push_const *p = payload;
        if (p->table < 0 || p->table >= NVN2_ROOT_TABLES || (!p->compute && p->table != 0)) {
            static int warned;
            if (!warned++) rt_log("vk: push constants into root table %d dropped", p->table);
            break;
        }
        int k = p->compute != 0, w0 = p->offset / 4, n = p->bytes / 4;
        memcpy(&ex->push[k][p->table][w0], p->w, 4 * (size_t)n);
        ex->push_set[k][p->table] |= (n >= 64 ? ~0ull : ((1ull << n) - 1)) << w0;
        bind_gen++;
        break;
    }
    case OP_BIND_SSBO: {
        const p_bind_buf *b = payload;
        if (b->stage >= 0 && b->stage < 8 && b->index >= 0 && b->index < 16) ex->ssbo[b->stage][b->index] = *b;
        else {   // past the table: dropped (NVN 2 binds more storage buffers per stage than NVN 1's 16)
            static uint64_t seen[8];
            if (b->stage >= 0 && b->stage < 8 && b->index >= 0 && b->index < 64 && !(seen[b->stage] >> b->index & 1))
                seen[b->stage] |= 1ull << b->index, rt_log("vk: storage buffer %d of stage %d dropped", b->index, b->stage);
        }
        bind_gen++;
        break;
    }
    case OP_BIND_TEX: {
        const p_bind_handles *b = payload;
        bind_handles(b->stage >= 0 && b->stage < 8 ? ex->tex[b->stage] : NULL, 128, b);
        ex->tex_serial++;
        bind_gen++;
        break;
    }
    case OP_BIND_SAMP: {
        const p_bind_handles *b = payload;
        bind_handles(b->stage >= 0 && b->stage < 8 ? ex->samp[b->stage] : NULL, 128, b);
        break;
    }
    case OP_BIND_IMG: {
        const p_bind_handles *b = payload;
        bind_handles(b->stage >= 0 && b->stage < 8 ? ex->img[b->stage] : NULL, 32, b);
        ex->tex_serial++;   // the shader memo's handle check (feedback_scan reads only textures: a rescan at most)
        bind_gen++;
        break;
    }
    case OP_UPDATE_UB: op_update_ub(ex, payload); break;
    case OP_GEOMETRY_MODE: ex->geometry_mode = (uint32_t)((const p_flag *)payload)->value; break;
    case OP_DRAW_MESH: case OP_DRAW_MESH_IND: {
        mesh_stat[MS_CALL]++;
        if (op == OP_DRAW_MESH) mesh_draw(ex, payload);
        else mesh_draw_indirect(ex, payload);
        break;
    }
    case OP_DRAW: op_draw(ex, payload); break;
    case OP_DRAW_IDX: op_draw_idx(ex, payload); break;
    case OP_DISPATCH: op_dispatch(ex, payload); break;
    case OP_DRAW_IDX_IND: op_draw_idx_ind(ex, payload); break;
    case OP_DISPATCH_IND: op_dispatch_ind(ex, payload); break;
    default: rt_fatal("vk: op %s is not a draw op", nvn_op_name(op));
    }
}
