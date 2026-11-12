// Behaviour ported from MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn @ b75be8cf (MIT, Copyright (c) 2026
// jhaizhou; origin d2r-mobile): which textures mirror their pool memory (nvn_mtl.m tex_mirrored, 3347-3362), the
// "GPU-owned" rule that stops re-uploads once the GPU rendered into a mirrored texture (nvn_mtl.m tex_mtl, 3366-3372),
// the live texture at a storage address that a slot of a finalized texture follows (hle_nvn.c:373-393), and
// ReadTexels / WriteTexels (nvn_mtl.m:10105-10195). Image creation, upload regions and samplers are ported from LAS
// Recomp runtime/nvn/vulkan/textures.c:235-330, 778-844 @ da35b83a (MIT, Copyright (c) 2026 Peiikai).
//
// Vulkan textures of the graphics layer: a VkImage cache keyed by storage GPU address plus descriptor, uploads (deswizzle, ASTC
// decode in vk_astc.c), re-uploads on page write generations (vk_memory.c), ReadTexels / WriteTexels, descriptor slots and
// samplers. Every image lives in GENERAL layout; uploads go into the upload batch the replay submits (vk_upload_flush).
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_hmap.h"
#include "nvn_tex.h"
#include "tex.h"
#include "vk.h"
#include "vk_view_format.h"
#include "vk_exec.h"
#include "platform/platform.h"

typedef struct {
    uint64_t addr;
    nvn_texdesc d;
    uint8_t has_layout, layout[8], pad[7];
} TexKey;

typedef struct Tex {
    VkTexImage pub;               // first: vk_texture_get hands out &pub
    TexKey key;
    tex_t t;                      // the newest texture object seen with this key
    TexVkFormat f;
    VmaAllocation alloc;          // own memory; NULL when bound into pool memory
    VmaAllocation bound_to;       // the pool memory it is bound into
    uint32_t wgen;                // page generation of the last upload check
    uint32_t checked;             // submit_epoch of the last check from a draw binding
    int uploaded, released;
    uint32_t n_up;                // uploads of this entry (vk_texture_debug)
    int last_ch;                  // last vk_mem_changed result of make_fresh: 1 written, 0 clean, -1 not tracked
    struct Tex *next_at;          // other entries at the same storage address
    uint8_t gpu_dirty;            // mirrored, and the GPU wrote the image after the last write-back to the storage
    uint32_t gpu_gen;             // page generation when the GPU write that set gpu_dirty was recorded
    uint64_t gpu_seq;             // order of that GPU write among all of them (gpu_writes)
    uint64_t gpu_fence;           // fence of the submission that made that write (0: not recorded in one)
    uint64_t wr_fence;            // newest submission recorded to write the image (ReadTexels waits for it, not the replay queue)
    uint8_t listed;               // in owned[]
    uint8_t *snap;                // a texture that fills no page: its storage bytes when the GPU write was recorded
    uint32_t wt_n;                // WriteTexels into the storage since the last upload
    // tex_prestage: the copy at the end of submission pre_fence holds the image as of GPU write pre_seq, at pre_off of
    // the ring in lap pre_lap (pre_size bytes); pre_mark: the fence it is listed for.
    uint64_t pre_seq, pre_fence, pre_off, pre_lap, pre_size, pre_mark;
    struct Tex *rel_prev, *rel_next;   // while released: the list of released entries, oldest first
    uint64_t rel_bytes;           // while released: its own memory (0 when bound into pool memory), counted in rel_sum
    uint64_t rel_frame;           // gfx frame of the release (the age of a released entry taken back)
    uint8_t rel_intact;           // at the release: would come back intact (mirrored, never GPU-written), rel_sum[1]
    uint64_t slot_frame;          // frame (vk_frame_no) a texture slot last resolved to it, live or released (0: never)
    uint64_t wr_frame;   // frame of the last recorded GPU write (vk_texture_gpu_written)
    uint64_t born_frame;          // frame the entry was made in (diagnostics)
} Tex;

// A slot keeps what vk_texture_by_id resolved: the entry (ce, valid while its address generation agen is still cgen), the
// view and has_view, and cepoch (submit_epoch of the last freshness check), so a draw binding skips the key lookup.
typedef struct {
    struct Tex *ce;               // NULL: nothing cached (registered since, or never resolved)
    uint32_t cgen;                // agen of the slot's address when ce was resolved
    uint32_t cepoch;
    view_t cview;
    uint8_t chas_view, used, has_view;
    view_t v;
    tex_t t;
} TexSlot;
typedef struct { samp_par p; VkSampler s, sp; uint8_t used; } SampSlot;   // s: vk_sampler_for(&p) once looked up; sp:
                                                                           // its point-sampled form (vk_sampler_point_by_id_locked)
enum { DESC_TEXTURE = 1, DESC_SAMPLER };
typedef struct {
    uint32_t kind, serial;
    union { TexSlot tex; SampSlot sampler; } u;
} DescSlot;

static PlatRMutex tex_lock;
static PlatOnce tex_once = PLAT_ONCE_INIT;
static nvn_hmap by_key = {.klen = sizeof(TexKey)};
static nvn_hmap by_addr = {.klen = 8};
static nvn_hmap descriptors = {.klen = sizeof(VkDescKey)};
static nvn_hmap samplers = {.klen = sizeof(samp_par)};
static int nreleased;
static Tex *rel_head, *rel_tail;   // released entries in release order (sweep_released drops from the head)
// Own memory of the released entries (tex_lock): [0] all, [1] those that come back intact (rel_intact).
// Taken back since the last perf line, by age at the take-back: under 1 s, 1-10 s, 10-60 s, older; count, bytes.
static uint64_t rel_sum[2];
static uint64_t rt_back_n[4], rt_back_bytes[4];
static uint64_t n_uploads, n_reuploads;
// Re-uploads since the last perf line (tex_lock): all, and those of textures a WriteTexels wrote since their last upload.
static uint64_t ru_n, ru_bytes, ru_ns, ru_wt_n, ru_wt_bytes, ru_wt_ns;
// A new descriptor over the storage of a live image the GPU wrote gets that image's content (alias_copy); count per log line.
static uint64_t n_alias_copied, n_alias_upload, n_alias_uninit;
static uint32_t submit_epoch = 1;   // bumped per queue submit: draw bindings check a texture's storage once per submit
// Mirrored textures the GPU wrote into, sorted by address; changed under tex_lock and owned_lock, read under either.
static PlatMutex owned_lock = PLAT_MUTEX_INIT;
static Tex **owned;
static int nowned, cowned;
static uint64_t max_owned;   // largest storage size in owned[]
static uint64_t gpu_writes;  // GPU writes into mirrored textures so far: Tex.gpu_seq
// Bumped whenever the entry a slot resolves to can change: an entry created or destroyed, released or taken back into
// use (the slot lookup and the follow of a finalized slot read only by_key, by_addr and the released flags).
static uint32_t tex_gen = 1;
// Bumped with tex_gen and whenever a slot is registered: what vk_texture_by_id returns for any slot may have changed.
static uint32_t slots_gen = 1;
static uint32_t samp_gen = 1;   // bumped whenever a sampler slot is registered: vk_sampler_by_id may answer otherwise
static uint64_t n_checks;    // draw-binding storage checks (make_fresh_snap) since the last perf line
// Write-backs that went through image_texels (a GPU wait), by cause, and tex_prestage's counts (tex_lock; perf line).
static uint64_t wbk_n[2], wbk_ns[2], pre_staged, pre_used, pre_missed;   // tex_lock: since the last perf line
static int wbk_src;                                                     // tex_lock: 0 fence write-back, 1 CPU read
// tex_gen per storage address (hashed): a slot resolves through the entries at its own address only, because the game
// finalizes and initializes transient textures every frame and a global generation would empty every cache keyed on it.
#define AGEN_N 4096
static uint32_t agen[AGEN_N];
static uint32_t reg_serial;

static uint32_t agen_idx(uint64_t addr) { return (uint32_t)(addr * 0x9E3779B97F4A7C15ull >> 52); }

static void tex_gen_bump(uint64_t addr) {
    if (!++tex_gen) tex_gen = 1;
    slots_gen++;
    uint32_t *g = &agen[agen_idx(addr)];
    __atomic_store_n(g, *g + 1, __ATOMIC_RELEASE);
}

static volatile int32_t tex_ready;
static void tex_init(void *p) {
    (void)p;
    __atomic_store_n(&tex_ready, 1, __ATOMIC_RELEASE);
    return;
}
static void lock(void) {
    if (!__atomic_load_n(&tex_ready, __ATOMIC_ACQUIRE)) plat_once(&tex_once, tex_init, NULL);
    plat_rmutex_lock(&tex_lock);
}
static void unlock(void) { plat_rmutex_unlock(&tex_lock); }

static uint32_t umax(uint32_t a, uint32_t b) { return a > b ? a : b; }
static uint32_t mip(uint32_t v, unsigned lv) { return umax(v >> lv, 1); }

static void make_key(const tex_t *t, TexKey *k) {
    memset(k, 0, sizeof *k);
    k->addr = (uintptr_t)t->mem;
    k->d = t->d;
    k->has_layout = t->has_layout;
    memcpy(k->layout, t->layout, 8);
}

// The UNORM sibling of an sRGB colour format (a storage view of it), 0 for the others.
VkFormat vk_unorm_of_srgb(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_B8G8R8A8_SRGB: return VK_FORMAT_B8G8R8A8_UNORM;
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32: return VK_FORMAT_A8B8G8R8_UNORM_PACK32;
    case VK_FORMAT_R8G8_SRGB: return VK_FORMAT_R8G8_UNORM;
    case VK_FORMAT_R8_SRGB: return VK_FORMAT_R8_UNORM;
    default: return 0;
    }
}

// Keys equal apart from the descriptor's swizzle selectors: the same storage, format and geometry.
static int same_but_swizzle(const TexKey *a, const TexKey *b) {
    TexKey x = *a, y = *b;
    x.d.swizzle_r = x.d.swizzle_g = x.d.swizzle_b = x.d.swizzle_a = 0;
    y.d.swizzle_r = y.d.swizzle_g = y.d.swizzle_b = y.d.swizzle_a = 0;
    return !memcmp(&x, &y, sizeof x);
}

// Textures whose content the CPU provides in pool memory. Display, compressible, zcull,
// depth, multisample and GPU-only storage stays GPU-owned.
static int tex_mirrored(const tex_t *t) {
    if (!t->mem || nvn_gpu_only((uintptr_t)t->mem)) return 0;
    if (t->has_layout) return 1;
    const pool_t *p = t->pool;
    if (!p || !(p->flags & (NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_CPU_CACHED))) return 0;
    if (t->d.flags & (NVN_TEXTURE_FLAGS_DISPLAY | NVN_TEXTURE_FLAGS_COMPRESSIBLE | NVN_TEXTURE_FLAGS_ADAPTIVE_ZCULL |
                      NVN_TEXTURE_FLAGS_DEPTH16_PREFER_FAST_CLEAR | NVN_TEXTURE_FLAGS_ZCULL_SUPPORT_STENCIL))
        return 0;
    if (t->d.samples > 1 || t->d.target == NVN_TEXTURE_TARGET_BUFFER) return 0;
    if (t->d.format >= 0x31 && t->d.format <= 0x36) return 0;   // stencil / depth formats
    return 1;
}

static void log_limited(volatile int32_t *n, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void log_limited(volatile int32_t *n, const char *fmt, ...) {
    if (plat_inc32(n) > 16) return;
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    rt_log("vk: %s", buf);
}

static void barrier(VkCommandBuffer cb, VkPipelineStageFlags2 src, VkAccessFlags2 sa, VkPipelineStageFlags2 dst,
                    VkAccessFlags2 da) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = src, .srcAccessMask = sa,
                           .dstStageMask = dst, .dstAccessMask = da};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}
#define ALL_RW VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT

// ---------------------------------------------------------------- images
static VkFormatFeatureFlags features_of(VkFormat f) {
    VkFormatProperties p;
    vkGetPhysicalDeviceFormatProperties(vk_ctx()->phys, f, &p);
    return p.optimalTilingFeatures;
}

static VkImageAspectFlags aspect_of(const TexVkFormat *f) {
    VkImageAspectFlags a = 0;
    if (f->depth) a |= VK_IMAGE_ASPECT_DEPTH_BIT;
    if (f->stencil) a |= VK_IMAGE_ASPECT_STENCIL_BIT;
    return a ? a : VK_IMAGE_ASPECT_COLOR_BIT;
}

// Binds img into the pool memory behind its GPU address (physical pools, virtual pools through the page table) when the
// memory type, alignment and contiguous size allow; 0 otherwise. A depth / stencil image that needs more bytes than its NVN
// storage (guest_size) is not bound: RDNA2 drivers put depth compression metadata after the depth data, which would overwrite
// the next texture of the heap above 1080p. Colour images keep their place.
static int format_compressed(VkFormat f) {   // vkCmdClearColorImage takes no block-compressed format
    return (f >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK) ||
           (f >= VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK && f <= VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK);
}

static int bind_pool_memory(VkCtx *c, Tex *e, VkImage img, uint64_t guest_size) {
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(c->device, img, &req);
    if ((e->f.depth || e->f.stencil) && req.size > guest_size) return 0;
    VkMemSpan sp;
    if (!req.size) return 0;   // an image the device could not lay out (image_size)
    if (!vk_mem_resolve_existing(e->pub.addr, req.size, &sp) || !sp.alloc || sp.alloc_offset % req.alignment) return 0;
    VmaAllocationInfo info;
    vmaGetAllocationInfo(c->vma, sp.alloc, &info);
    if (!(req.memoryTypeBits >> info.memoryType & 1)) return 0;
    if (vmaBindImageMemory2(c->vma, sp.alloc, sp.alloc_offset, img, NULL) != VK_SUCCESS) return 0;
    e->bound_to = sp.alloc;
    return 1;
}

// A BC texture's blocks are drawn through an RGBA16UI view, one texel per 8-byte block: an uncompressed view of a
// BLOCK_TEXEL_VIEW_COMPATIBLE image, with EXTENDED_USAGE so the BC image can carry COLOR_ATTACHMENT. Whether the device takes
// that for this format and shape:
static int block_views_ok(const VkImageCreateInfo *ci) {
    VkPhysicalDeviceImageFormatInfo2 fi = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
                                           .format = ci->format, .type = ci->imageType, .tiling = ci->tiling,
                                           .usage = ci->usage, .flags = ci->flags};
    VkImageFormatProperties2 p = {.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
    if (vkGetPhysicalDeviceImageFormatProperties2(vk_ctx()->phys, &fi, &p) != VK_SUCCESS) return 0;
    const VkImageFormatProperties *q = &p.imageFormatProperties;
    return ci->extent.width <= q->maxExtent.width && ci->extent.height <= q->maxExtent.height &&
           ci->extent.depth <= q->maxExtent.depth && ci->mipLevels <= q->maxMipLevels &&
           ci->arrayLayers <= q->maxArrayLayers;
}

// UNDEFINED -> GENERAL; new depth / stencil images start cleared (the game copies depth without stencil).
static void init_image(Tex *e, VkCommandBuffer cb) {
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
                                .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = e->pub.image, .subresourceRange = {e->pub.aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(cb, &di);
    if (e->f.depth || e->f.stencil) {
        // A new depth image starts at the value the game clears depth to (1.0): a shadow map sampled before its first draw
        // must not read 0.
        static const float init_depth = 1.0f;
        VkClearDepthStencilValue zero = {init_depth, 0};
        VkImageSubresourceRange rg = {e->pub.aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        vkCmdClearDepthStencilImage(cb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &rg);
        barrier(cb, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
    } else if (!e->bound_to && !e->pub.mirrored && !format_compressed(e->f.vk)) {
        // GPU-only storage that got memory of its own starts at zero, as the pool it stands for. Images aliasing pool memory
        // are not cleared (that would erase other textures' renders); mirrored images are uploaded on first use.
        VkClearColorValue zero = {0};
        VkImageSubresourceRange rg = {e->pub.aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        vkCmdClearColorImage(cb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &rg);
        barrier(cb, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
    }
}

// defer_init: the caller records the image's first transition (init_image or alias_copy).
static Tex *create_tex(const tex_t *t, const TexKey *k, int defer_init) {
    static volatile int32_t n_unmapped, n_unsupported, n_failed, n_unbound;
    VkCtx *c = vk_ctx();
    const nvn_texdesc *d = &t->d;
    if (!c || d->target == NVN_TEXTURE_TARGET_BUFFER) return NULL;
    TexVkFormat f;
    if (!tex_vk_format(d->format, &f)) {
        log_limited(&n_unmapped, "texture format %#x has no Vulkan format: %s", d->format, f.reason);
        return NULL;
    }
    if (!tex_vk_format_supported(&f)) {
        log_limited(&n_unsupported, "texture format %#x: VkFormat %d lacks features %#x", d->format, (int)f.vk, (unsigned)f.need);
        return NULL;
    }
    nvn_texlayout L;
    nvn_tex_layout(d, &L);
    uint32_t w = umax(d->width, 1), h = umax(d->height, 1), depth = 1, layers = 1;
    VkImageType type = VK_IMAGE_TYPE_2D;
    switch (d->target) {
    case NVN_TEXTURE_TARGET_1D: type = VK_IMAGE_TYPE_1D; h = 1; break;
    case NVN_TEXTURE_TARGET_1D_ARRAY: type = VK_IMAGE_TYPE_1D; h = 1; layers = L.layers; break;
    case NVN_TEXTURE_TARGET_3D: type = VK_IMAGE_TYPE_3D; depth = umax(d->depth, 1); break;
    case NVN_TEXTURE_TARGET_2D_ARRAY: case NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY:
    case NVN_TEXTURE_TARGET_CUBEMAP: case NVN_TEXTURE_TARGET_CUBEMAP_ARRAY: layers = L.layers; break;
    default: break;
    }
    uint32_t levels = umax(d->levels, 1), most = 1;
    for (uint32_t s = umax(umax(w, h), depth); s > 1; s >>= 1) most++;
    if (levels > most) levels = most;
    uint32_t samples = d->samples > 1 ? (uint32_t)d->samples : 1;
    if (samples > 1) levels = 1;
    VkFormatFeatureFlags ff = features_of(f.vk);
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    int ds = f.depth || f.stencil;
    if (ds && (ff & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (!ds && (ff & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    // Storage usage for every single-sample colour texture whose format allows it: NVN 2 binds textures made without
    // NVN_TEXTURE_FLAGS_IMAGE as images, and the usage cannot be added later.
    if (!ds && (ff & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) && samples == 1) usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    VkImageCreateFlags flags = ds ? 0 : VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;   // format-reinterpreting views
    // RGB9E5 has no storage-image format. The game binds an R32_UINT view to write its packed bits; a compatible view
    // may supply a usage that the base sampling format cannot support.
    if (!ds && samples == 1 && f.vk == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 &&
        (features_of(VK_FORMAT_R32_UINT) & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) {
        usage |= VK_IMAGE_USAGE_STORAGE_BIT;
        flags |= VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
    }
    // Cube compatible only for cube targets (the replay makes cube views of nothing else). The flag on every square array of
    // 6+ layers changed how AMD's Windows driver lays the image out and broke terrain material arrays.
    if (type == VK_IMAGE_TYPE_2D && layers >= 6 && w == h && samples == 1 &&
        (d->target == NVN_TEXTURE_TARGET_CUBEMAP || d->target == NVN_TEXTURE_TARGET_CUBEMAP_ARRAY))
        flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    // NVN renders into a 3D texture slice by slice (a view's min layer is the depth plane), e.g. a grading LUT.
    // Vulkan needs a 2D view of the slice for that.
    if (type == VK_IMAGE_TYPE_3D && (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = flags, .imageType = type, .format = f.vk,
                            .extent = {w, h, depth}, .mipLevels = levels, .arrayLayers = layers,
                            .samples = (VkSampleCountFlagBits)samples, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    int bw, bh, block_views = 0;
    // A 3D BC image gets no block views: on RDNA2 such an image took a GPU write past the end of its memory (device lost).
    // Draws into it go through a stand-in (vk_exec.c bc_target).
    static int bv3d_said;
    if (!bv3d_said) {
        bv3d_said = 1;
        rt_log("vk: 3D BC textures have no block views");
    }
    if (!ds && samples == 1 && f.vk == f.ideal && nvn_fmt_bpb(d->format, &bw, &bh) && bw == 4 && bh == 4 &&
        type != VK_IMAGE_TYPE_3D) {
        VkImageCreateInfo bc = ci;   // BC formats: renderable through views of their blocks (block_views_ok)
        bc.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        bc.flags |= VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
        int ok = block_views_ok(&bc);
        if (ok) ci = bc, usage = bc.usage, flags = bc.flags, block_views = 1;
    }
    // Feedback loops (vk.h VkCtx.feedback): every attachable image may be sampled by a draw of a pass that renders it,
    // so it gets the usage now; adding it when a loop shows up would mean recreating the image.
    int fb_usage = c->feedback == VK_FEEDBACK_LAYOUT && !block_views &&
                   (usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT));
    if (fb_usage) ci.usage = usage |= VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
    Tex *e = calloc(1, sizeof *e);
    if (e) e->born_frame = vk_frame_no;
    if (!e) rt_fatal("vk: out of memory for a texture");
    e->key = *k;
    e->t = *t;
    e->f = f;
    e->pub.addr = k->addr;
    VkImage img = VK_NULL_HANDLE;
    // An RGBA32F render target gets memory of its own: the 320x180 fog volume map is read by the next frame's fog compute,
    // but a later target of the same frame reuses its pool range.
    int own_rt = f.vk == VK_FORMAT_R32G32B32A32_SFLOAT && type == VK_IMAGE_TYPE_2D && samples == 1 &&
                 (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) && nvn_gpu_only(k->addr);
    if (own_rt)   // a second descriptor here would not share bytes with this image after alias_copy
        rt_log("vk: texture %#llx (%ux%u) is a private render target%s", (unsigned long long)k->addr, w, h,
               nvn_hm_get(&by_addr, &k->addr) ? ", beside another descriptor" : "");
    if (nvn_gpu_only(k->addr) && !own_rt) {   // storage in a physical or virtual pool: alias the pool memory
        if (vkCreateImage(c->device, &ci, NULL, &img) == VK_SUCCESS && !bind_pool_memory(c, e, img, L.size)) {
            vkDestroyImage(c->device, img, NULL);
            img = VK_NULL_HANDLE;
            log_limited(&n_unbound, "texture %#llx (%ux%u fmt %#x) cannot alias its pool memory; own memory instead",
                        (unsigned long long)k->addr, w, h, d->format);
        }
    }
    if (!img) {
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VkResult r = vmaCreateImage(c->vma, &ci, &ai, &img, &e->alloc, NULL);
        if (r != VK_SUCCESS) {
            log_limited(&n_failed, "image creation failed (%d): nvn fmt %#x %ux%ux%u target %d levels %u layers %u", (int)r,
                        d->format, w, h, depth, d->target, levels, layers);
            free(e);
            return NULL;
        }
    }
    VkTexImage *p = &e->pub;
    p->image = img;
    p->format = f.vk;
    p->type = type;
    p->usage = usage;
    p->aspect = aspect_of(&f);
    p->width = w, p->height = h, p->depth = depth, p->levels = levels, p->layers = layers, p->samples = samples;
    p->view_type = type == VK_IMAGE_TYPE_1D ? (d->target == NVN_TEXTURE_TARGET_1D_ARRAY ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D)
                 : type == VK_IMAGE_TYPE_3D ? VK_IMAGE_VIEW_TYPE_3D
                 : d->target == NVN_TEXTURE_TARGET_CUBEMAP && (flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) ? VK_IMAGE_VIEW_TYPE_CUBE
                 : d->target == NVN_TEXTURE_TARGET_CUBEMAP_ARRAY && (flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY
                 : layers > 1 || d->target == NVN_TEXTURE_TARGET_2D_ARRAY || d->target == NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY
                     ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    p->alpha_one = f.alpha_one;
    memcpy(p->chan, f.chan, sizeof p->chan);
    p->swz[0] = (uint8_t)d->swizzle_r, p->swz[1] = (uint8_t)d->swizzle_g;
    p->swz[2] = (uint8_t)d->swizzle_b, p->swz[3] = (uint8_t)d->swizzle_a;
    if (!(p->swz[0] | p->swz[1] | p->swz[2] | p->swz[3]))   // descriptors built without nvnTextureBuilderSetDefaults
        p->swz[0] = NVN_TEXTURE_SWIZZLE_R, p->swz[1] = NVN_TEXTURE_SWIZZLE_G, p->swz[2] = NVN_TEXTURE_SWIZZLE_B,
        p->swz[3] = NVN_TEXTURE_SWIZZLE_A;
    p->mirrored = (uint8_t)tex_mirrored(t);
    p->bound = e->bound_to != NULL;
    p->block_views = (uint8_t)block_views;
    p->fb_usage = (uint8_t)fb_usage;

    if (!defer_init) {
        init_image(e, vk_upload_begin());
        vk_upload_end();
    }
    return e;
}

static int owned_lower(uint64_t a) {
    int lo = 0, hi = nowned;
    while (lo < hi) {
        int m = (lo + hi) / 2;
        if (owned[m]->pub.addr < a) lo = m + 1;
        else hi = m;
    }
    return lo;
}

static void own_add(Tex *e) {
    if (e->listed) return;
    plat_mutex_lock(&owned_lock);
    if (nowned == cowned) {
        cowned = cowned ? 2 * cowned : 256;
        if (!(owned = realloc(owned, sizeof *owned * (size_t)cowned))) rt_fatal("vk: out of memory for the owned-texture list");
    }
    int i = owned_lower(e->pub.addr);
    memmove(owned + i + 1, owned + i, sizeof *owned * (size_t)(nowned - i));
    owned[i] = e;
    __atomic_store_n(&nowned, nowned + 1, __ATOMIC_RELEASE);
    if (e->t.L.size > max_owned) max_owned = e->t.L.size;
    e->listed = 1;
    plat_mutex_unlock(&owned_lock);
}

static void own_remove(Tex *e) {
    if (!e->listed) return;
    plat_mutex_lock(&owned_lock);
    for (int i = owned_lower(e->pub.addr); i < nowned && owned[i]->pub.addr == e->pub.addr; i++)
        if (owned[i] == e) {
            memmove(owned + i, owned + i + 1, sizeof *owned * (size_t)(nowned - i - 1));
            __atomic_store_n(&nowned, nowned - 1, __ATOMIC_RELEASE);
            break;
        }
    e->listed = 0;
    plat_mutex_unlock(&owned_lock);
}

static int tex_intact(const Tex *e) { return e->pub.mirrored && !e->pub.gpu_owned && !e->gpu_dirty; }

static uint64_t tex_own_bytes(const Tex *e) {
    VkCtx *c = vk_ctx();
    if (!e->alloc || !c) return 0;
    VmaAllocationInfo ai;
    vmaGetAllocationInfo(c->vma, e->alloc, &ai);
    return ai.size;
}

static void rel_append(Tex *e) {
    e->rel_bytes = tex_own_bytes(e);
    e->rel_intact = (uint8_t)tex_intact(e);
    e->rel_frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    rel_sum[0] += e->rel_bytes;
    if (e->rel_intact) rel_sum[1] += e->rel_bytes;
    e->rel_next = NULL;
    e->rel_prev = rel_tail;
    if (rel_tail) rel_tail->rel_next = e;
    else rel_head = e;
    rel_tail = e;
}

// A released entry that a texture slot resolved to lately is in use (slot_frame says when). The sweeps below and entry_of's
// replacement of released entries spare an entry used within REL_USED_FRAMES; evicting it would leave its slot following
// whatever image lay at its address (wrong lighting in part of the screen) or a new zeroed image.
#define REL_USED_FRAMES 600   // 10 s
// Since the last perf line (tex_lock): in-use released entries a new texture at their address would have replaced (kept),
// those RELEASED_MAX evicted anyway; slots whose own entry was gone.
static uint64_t n_inuse_kept, n_inuse_capped, n_orphan_shaped, n_orphan_made;
static uint64_t n_slot_empty, n_slot_uploaded;   // slots whose own image was made at a lookup: GPU-only (empty), storage-backed
static int rel_in_use(const Tex *e, uint64_t frame) {
    return e->slot_frame && frame - e->slot_frame < REL_USED_FRAMES + 2;   // slot_frame is vk_frame_no: a frame behind
}
static void rel_unlink(Tex *e) {
    rel_sum[0] -= e->rel_bytes;
    if (e->rel_intact) rel_sum[1] -= e->rel_bytes;
    e->rel_bytes = 0, e->rel_intact = 0;
    if (e->rel_prev) e->rel_prev->rel_next = e->rel_next;
    else rel_head = e->rel_next;
    if (e->rel_next) e->rel_next->rel_prev = e->rel_prev;
    else rel_tail = e->rel_prev;
    e->rel_prev = e->rel_next = NULL;
}

static void pre_forget(Tex *e);

// Why the entry being destroyed goes (set by the caller), and a count and log of GPU-only entries destroyed although a
// slot sampled them within the last 10 s: the slot's next lookup then makes a new empty image (slot-empty).
static const char *destroy_why = "?";
static uint64_t n_sampled_destroyed;
// The GPU-only images destroyed last (a ring): a slot-empty line says what became of the image the slot had.
typedef struct { uint64_t addr, frame, slot_frame, wr_frame; uint32_t w, h, format; const char *why; } GoneRec;
static GoneRec gone_ring[512];
static uint32_t gone_n;
static void destroy_tex(Tex *e) {
    if (!e->pub.mirrored)
        gone_ring[gone_n++ & 511] = (GoneRec){e->key.addr, vk_frame_no, e->slot_frame, e->wr_frame, e->key.d.width, e->key.d.height,
                                              e->key.d.format, destroy_why};
    if (!e->pub.mirrored && e->slot_frame && vk_frame_no - e->slot_frame < 600) {
        n_sampled_destroyed++;
        static volatile int32_t told;
        if (plat_add32(&told, 1) <= 400)
            rt_log("vk: sampled-destroyed f%llu: %#llx %ux%u fmt %#x (%s, last sampled %llu frames ago): %s",
                   (unsigned long long)vk_frame_no, (unsigned long long)e->key.addr, e->key.d.width, e->key.d.height,
                   e->key.d.format, e->released ? "released" : "live", (unsigned long long)(vk_frame_no - e->slot_frame), destroy_why);
    }
    destroy_why = "?";
    own_remove(e);
    pre_forget(e);
    tex_gen_bump(e->key.addr);
    nvn_hm_del(&by_key, &e->key);
    Tex *head = nvn_hm_get(&by_addr, &e->key.addr);
    if (head == e) {
        if (e->next_at) nvn_hm_put(&by_addr, &e->key.addr, e->next_at);
        else nvn_hm_del(&by_addr, &e->key.addr);
    } else {
        for (Tex *p = head; p; p = p->next_at)
            if (p->next_at == e) { p->next_at = e->next_at; break; }
    }
    if (e->released) nreleased--, rel_unlink(e);
    vk_exec_image_gone(e->pub.image);
    vk_defer_image(e->pub.image, e->alloc);
    free(e->snap);
    free(e);
}

// Released entries with another descriptor at the same address are replaced; past RELEASED_MAX the oldest goes at each
// release (it comes back on demand), so thousands of images are not destroyed in one frame. Terrain chunk maps the GPU
// writes at stream-in must not be dropped by that rule: the next draw would upload stale storage bytes.
#define RELEASED_MAX 2048
static void sweep_released(void) {
    static int keep_said;
    if (!keep_said) {
        keep_said = 1;
        rt_log("vk: released textures at the cap: storage-backed entries first, then GPU-only images, then the oldest");
    }
    // Dropped first: entries whose storage holds their content (mirrored, not GPU-dirty); then GPU-only images (render
    // targets the game draws again); then the oldest of the rest. Entries a slot still resolves to (rel_in_use) go only in a
    // fourth pass, when nothing else is left.
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    for (int pass = 0; pass < 4 && nreleased > RELEASED_MAX; pass++)
        for (Tex *e = rel_head, *next; e && nreleased > RELEASED_MAX; e = next) {
            next = e->rel_next;
            if (pass < 3 && rel_in_use(e, frame)) continue;
            if (pass == 3) n_inuse_capped++;
            if (pass >= 2 || (pass == 0 ? e->pub.mirrored && !e->gpu_dirty : !e->pub.mirrored)) destroy_why = "entry limit", destroy_tex(e);
        }
}

// Above 90% of the device-local budget each submit drops up to PRESSURE_STEP of the oldest released entries that come back
// intact (mirrored, never GPU-written), so a 4 GiB card does not spill into system memory. A render target lives only in
// its image and is kept.
#define PRESSURE_STEP 32
enum { vram_pressure = 90 };
static uint64_t n_pressure_drops;

static void sweep_pressure(void) {   // tex_lock held
    static int pressure_said;
    if (!pressure_said) {
        pressure_said = 1;
        rt_log("vk: released textures under VRAM pressure: %s%d%%", "dropped above ", vram_pressure);
    }
    if (!rel_head || submit_epoch & 7) return;   // the budget query every 8 submits
    uint64_t used, budget;
    int exact;
    if (!vk_mem_vram(&used, &budget, &exact) || used <= budget / 100 * (uint64_t)vram_pressure) return;
    int dropped = 0;
    for (Tex *e = rel_head, *next; e && dropped < PRESSURE_STEP; e = next) {
        next = e->rel_next;
        if (!e->pub.mirrored || e->pub.gpu_owned || e->gpu_dirty) continue;
        destroy_why = "vram pressure", destroy_tex(e);
        dropped++, n_pressure_drops++;
    }
}

static uint64_t n_cap_drops, cap_drop_bytes, n_cap_stuck;

// Released texture cache: see docs/how-it-works.md, "Released texture cache". Released GPU-only render targets are freed by evidence: a
// released image is dead when another GPU-written image was written over its storage after its last use (same address, or
// starting elsewhere and covering an eighth of it or more); a dead one goes after REL_DEAD_UNUSED (3) unused frames, any
// other GPU-only one after REL_KEEP_FRAMES (3600; 600 above 90% of the budget). Mirrored entries are never touched here.
// "Used" is the latest of its release, a slot resolving to it and a GPU write. The pass runs every 15 frames, at most 256 images.
static const uint64_t REL_DEAD_UNUSED = 3;
#define REL_KEEP_FRAMES 3600
static uint64_t rel_trim_frame;
static uint64_t n_dead_drops, dead_drop_bytes, n_old_drops, old_drop_bytes;

static uint64_t tex_last_used(const Tex *e) {
    uint64_t f = e->rel_frame;
    if (e->slot_frame > f) f = e->slot_frame;
    if (e->wr_frame > f) f = e->wr_frame;
    return f;
}
static int dead_by_addr(const void *a, const void *b) {
    uint64_t x = (*(Tex *const *)a)->key.addr, y = (*(Tex *const *)b)->key.addr;
    return x < y ? -1 : x > y;
}
// [olo, ohi) written at o_written lies over [lo, hi) last used at last_used.
static int shifted_over(uint64_t lo, uint64_t hi, uint64_t last_used, uint64_t olo, uint64_t ohi, uint64_t o_written) {
    if (hi <= lo || ohi <= olo || olo == lo || o_written <= last_used) return 0;
    if (olo >= lo && ohi <= hi) return 0;
    uint64_t a = lo > olo ? lo : olo, b = hi < ohi ? hi : ohi;
    return b > a && (b - a) * 8 >= hi - lo;
}

static void sweep_trim(void) {   // tex_lock held
    static int dead_said;
    if (!dead_said) {
        dead_said = 1;
        rt_log("vk: released render targets: dead ones (drawn over since their last use) go after 3 unused frames, the "
               "others after a minute");
    }
    uint64_t frame = vk_frame_no;
    if (frame < rel_trim_frame + 15 || !rel_head) return;
    rel_trim_frame = frame;
    uint64_t keep = REL_KEEP_FRAMES, used, budget;
    int exact;
    if (vk_mem_vram(&used, &budget, &exact) && used > budget / 100 * (uint64_t)vram_pressure) keep = 600;
    // every GPU-written GPU-only image by where its storage starts
    static Tex **by;
    static int cby;
    int n = 0;
    uint64_t longest = 0;
    if (cby < (int)by_key.n + 1) {
        free(by);
        cby = (int)by_key.n + 256;
        if (!(by = malloc(sizeof *by * (size_t)cby))) { cby = 0; return; }
    }
    for (uint32_t i = 0; i < by_key.cap; i++) {
        Tex *t = by_key.vals[i];
        if (!t || t->pub.mirrored || !t->wr_frame || !t->t.L.size) continue;
        by[n++] = t;
        if (t->t.L.size > longest) longest = t->t.L.size;
    }
    qsort(by, (size_t)n, sizeof *by, dead_by_addr);
    Tex *gone[256];
    int ngone = 0;
    for (Tex *e = rel_head; e && ngone < 256; e = e->rel_next) {
        if (e->pub.mirrored) continue;
        uint64_t last = tex_last_used(e), unused = frame > last ? frame - last : 0;
        if (unused < REL_DEAD_UNUSED) continue;
        int dead = 0;
        if (unused >= keep) dead = 2;
        else if (e->t.L.size) {
            uint64_t lo = e->key.addr, hi = lo + e->t.L.size, from = lo > longest ? lo - longest : 0;
            int a = 0, b = n;
            while (a < b) { int m = (a + b) / 2; if (by[m]->key.addr < from) a = m + 1; else b = m; }
            int one = e->pub.levels == 1 && e->pub.layers == 1 && e->pub.depth <= 1;
            for (; a < n && by[a]->key.addr < hi && !dead; a++) {
                const Tex *o = by[a];
                if (o == e || o->wr_frame <= last) continue;
                if (o->key.addr == lo) dead = 1;   // drawn over from the same start: another size or format there since
                else if (one && shifted_over(lo, hi, last, o->key.addr, o->key.addr + o->t.L.size, o->wr_frame)) dead = 1;
            }
        }
        if (dead) {
            if (dead == 1) n_dead_drops++, dead_drop_bytes += e->rel_bytes;
            else n_old_drops++, old_drop_bytes += e->rel_bytes;
            gone[ngone++] = e;
        }
    }
    for (int i = 0; i < ngone; i++) destroy_why = "drawn over or a minute unused", destroy_tex(gone[i]);
}

// OOT_TRACE=perf, every 600 frames (tex_lock held): what the released entries are. Classes: gpu-only (not mirrored), mirrored
// and GPU-written (dirty: not written back; clean: written back), intact (mirrored, never GPU-written); "virtual": the address
// has no CPU view (>= 2^39); "under a live entry": another texture was put over that memory since.
static int census_by_addr(const void *a, const void *b) {
    uint64_t x = (*(Tex *const *)a)->key.addr, y = (*(Tex *const *)b)->key.addr;
    return x < y ? -1 : x > y;
}
static void rel_census(uint64_t frame) {
    const double M = 1048576.0;
    uint64_t n[4] = {0}, b[4] = {0}, an[4] = {0}, ab[4] = {0}, vn = 0, vb = 0, on = 0, ob = 0, live = 0, live_b = 0;
    struct { uint32_t w, h; int fmt; uint64_t n, bytes; } top[48];
    int ntop = 0;
    Tex **all = malloc(sizeof *all * (size_t)(by_key.n + 1));
    if (!all) return;
    int na = 0;
    for (uint32_t i = 0; i < by_key.cap; i++)
        if (by_key.vals[i]) all[na++] = by_key.vals[i];
    qsort(all, (size_t)na, sizeof *all, census_by_addr);
    uint8_t *under = calloc((size_t)na + 1, 1);
    for (int i = 0; under && i < na; i++) {
        uint64_t end = all[i]->key.addr + (all[i]->t.L.size ? all[i]->t.L.size : 1);
        for (int j = i + 1; j < na && all[j]->key.addr < end; j++) {
            if (all[i]->released && !all[j]->released) under[i] = 1;
            if (all[j]->released && !all[i]->released) under[j] = 1;
        }
    }
    for (int i = 0; i < na; i++) {
        const Tex *e = all[i];
        if (!e->released) { live++, live_b += tex_own_bytes(e); continue; }
        int c = !e->pub.mirrored ? 0 : e->gpu_dirty ? 1 : e->pub.gpu_owned ? 2 : 3;
        n[c]++, b[c] += e->rel_bytes;
        uint64_t age = frame - e->rel_frame;
        int a = age < 60 ? 0 : age < 600 ? 1 : age < 3600 ? 2 : 3;
        an[a]++, ab[a] += e->rel_bytes;
        if (e->key.addr >= 0x8000000000ull) vn++, vb += e->rel_bytes;
        if (under && under[i]) on++, ob += e->rel_bytes;
        int k = 0;
        while (k < ntop && (top[k].w != e->pub.width || top[k].h != e->pub.height || top[k].fmt != (int)e->pub.format)) k++;
        if (k == ntop && ntop < 48) top[ntop++] = (typeof(top[0])){e->pub.width, e->pub.height, (int)e->pub.format, 0, 0};
        if (k < ntop) top[k].n++, top[k].bytes += e->rel_bytes;
    }
    free(all), free(under);
    rt_log("perf: frame %llu: released census: gpu-only %llu (%.0f MiB), mirrored GPU-written dirty %llu (%.0f), written back "
           "%llu (%.0f), intact %llu (%.0f); virtual addresses %llu (%.0f MiB); under a live entry %llu (%.0f MiB); age under "
           "1 s %llu (%.0f), 1-10 s %llu (%.0f), 10-60 s %llu (%.0f), older %llu (%.0f); in use %llu entries (%.0f MiB own)",
           (unsigned long long)frame, (unsigned long long)n[0], b[0] / M, (unsigned long long)n[1], b[1] / M,
           (unsigned long long)n[2], b[2] / M, (unsigned long long)n[3], b[3] / M, (unsigned long long)vn, vb / M,
           (unsigned long long)on, ob / M, (unsigned long long)an[0], ab[0] / M, (unsigned long long)an[1], ab[1] / M,
           (unsigned long long)an[2], ab[2] / M, (unsigned long long)an[3], ab[3] / M, (unsigned long long)live, live_b / M);
    char line[700];
    int len = 0;
    for (int r = 0; r < 8; r++) {   // the eight sizes that hold most
        int best = -1;
        for (int k = 0; k < ntop; k++)
            if (top[k].n && (best < 0 || top[k].bytes > top[best].bytes)) best = k;
        if (best < 0) break;
        len += snprintf(line + len, sizeof line - (size_t)len, " %ux%u fmt %d: %llu (%.0f MiB);", top[best].w, top[best].h,
                        top[best].fmt, (unsigned long long)top[best].n, top[best].bytes / M);
        top[best].n = 0;
    }
    if (len) rt_log("perf: frame %llu: released census, largest sizes:%s", (unsigned long long)frame, line);
}

// ---------------------------------------------------------------- uploads
static uint32_t texel_alpha_one(int format, uint32_t *elem) {
    switch (format) {
    case 0x19: *elem = 1; return 0xFF;
    case 0x1A: *elem = 1; return 0x7F;
    case 0x1B: case 0x1C: *elem = 1; return 1;
    case 0x1D: *elem = 2; return 0x3C00;
    case 0x1E: *elem = 2; return 0xFFFF;
    case 0x1F: *elem = 2; return 0x7FFF;
    case 0x20: case 0x21: *elem = 2; return 1;
    case 0x22: *elem = 4; return 0x3F800000;
    default: *elem = 4; return 1;   // 0x23, 0x24
    }
}

// 3-component texels to 4 (n texels).
static void expand_rgb(int format, const uint8_t *src, uint8_t *dst, size_t n) {
    uint32_t e, one = texel_alpha_one(format, &e);
    for (size_t i = 0; i < n; i++, src += 3 * e, dst += 4 * e) {
        memcpy(dst, src, 3 * e);
        memcpy(dst + 3 * e, &one, e);
    }
}

static void shrink_rgb(int format, const uint8_t *src, uint8_t *dst, size_t n) {
    uint32_t e;
    texel_alpha_one(format, &e);
    for (size_t i = 0; i < n; i++, src += 4 * e, dst += 3 * e) memcpy(dst, src, 3 * e);
}

// Copies the whole storage into the image (every level and layer the image has).
static void upload(Tex *e) {
    static volatile int32_t n_bad, n_undecoded;
    const tex_t *t = &e->t;
    const nvn_texdesc *d = &t->d;
    int bw, bh, bpb = nvn_fmt_bpb(d->format, &bw, &bh);
    if (!bpb || !t->mem) return;
    // The game released the storage but still holds an id of the texture: the entry stays not uploaded.
    static volatile int32_t n_unmapped;
    if (!mem_guest_mapped((uint64_t)(uintptr_t)t->mem, t->L.size)) {
        log_limited(&n_unmapped, "texture %#llx fmt %#x %ux%u: its storage %#llx+%#llx is not mapped (released by the game): not uploaded",
                    (unsigned long long)e->pub.addr, d->format, d->width, d->height,
                    (unsigned long long)(uintptr_t)t->mem, (unsigned long long)t->L.size);
        return;
    }
    uint64_t p0 = perf_begin(), u0 = kern_now_ns();
    int linear = nvn_tex_is_linear(d), is3d = d->target == NVN_TEXTURE_TARGET_3D, one_d = e->pub.type == VK_IMAGE_TYPE_1D;
    uint32_t levels = linear ? 1 : e->pub.levels, layers = linear ? 1 : e->pub.layers;
    int astc = e->f.astc_bw != 0, rgb = e->f.expand_rgb;
    int agpu = astc && vk_astc_gpu();   // raw blocks to the GPU
    uint32_t n = levels * layers, k = 0;
    VkBufferImageCopy *rg = calloc(n, sizeof *rg);
    VkAstcRegion *ar = agpu ? calloc(n, sizeof *ar) : NULL;
    VkDeviceSize total = 0;
    for (uint32_t lv = 0; lv < levels; lv++)
        for (uint32_t ly = 0; ly < layers; ly++) {
            uint32_t w = mip(d->width, lv), h = one_d ? 1 : mip(d->height, lv), sl = is3d ? mip(d->depth, lv) : 1;
            uint64_t texels = (uint64_t)w * h * sl, raw = (uint64_t)(w + bw - 1) / bw * bpb * ((h + bh - 1) / bh) * sl;
            uint64_t out = agpu ? raw : astc ? texels * 4 : rgb ? texels * (uint64_t)bpb / 3 * 4 : raw;
            rg[k++] = (VkBufferImageCopy){.bufferOffset = total, .imageSubresource = {e->pub.aspect, lv, ly, 1},
                                          .imageExtent = {w, h, sl}};
            total += (out + 15) & ~15ull;
        }
    VkCommandBuffer cb = vk_upload_begin();
    VkBuffer sb, rbuf = VK_NULL_HANDLE;
    uint8_t *stg = vk_staging(total, &sb, NULL);
    k = 0;
    for (uint32_t lv = 0; lv < levels; lv++)
        for (uint32_t ly = 0; ly < layers; ly++, k++) {
            uint32_t w = rg[k].imageExtent.width, h = rg[k].imageExtent.height, sl = rg[k].imageExtent.depth;
            uint32_t rb = (w + bw - 1) / bw * (uint32_t)bpb, rows = (h + bh - 1) / bh;
            size_t raw = (size_t)rb * rows * sl;
            uint8_t *dst = stg + rg[k].bufferOffset, *tmp = (astc && !agpu) || rgb ? malloc(raw) : NULL;
            if (nvn_deswizzle(d, lv, ly, t->mem, tmp ? tmp : dst, rb)) {
                log_limited(&n_bad, "texture %#llx fmt %#x target %d: level %u layer %u has no CPU layout; uploaded as zeros",
                            (unsigned long long)e->pub.addr, d->format, d->target, lv, ly);
                memset(tmp ? tmp : dst, 0, raw);
            }
            if (agpu) {
                ar[k] = (VkAstcRegion){.src = rg[k].bufferOffset, .w = w, .h = h, .slices = sl, .level = lv, .layer = ly};
            } else if (astc) {
                uint64_t a0 = perf_begin();
                for (uint32_t z = 0; z < sl; z++)
                    astc_decode_ldr_mt(tmp + (size_t)z * rb * rows, w, h, e->f.astc_bw, e->f.astc_bh, e->f.astc_srgb,
                                       dst + (size_t)z * w * h * 4);
                perf_end(PERF_ASTC, a0);
                rg[k].bufferRowLength = w, rg[k].bufferImageHeight = h;
            } else if (rgb) {
                expand_rgb(d->format, tmp, dst, (size_t)w * h * sl);
                rg[k].bufferRowLength = w, rg[k].bufferImageHeight = h;
            } else {
                rg[k].bufferRowLength = rb / (uint32_t)bpb * (uint32_t)bw, rg[k].bufferImageHeight = rows * (uint32_t)bh;
            }
            free(tmp);
        }
    barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    int checked = 0;
    if (agpu) {
        uint64_t a0 = perf_begin();
        checked = vk_astc_record(cb, sb, total, ar, (int)n, e->f.astc_bw, e->f.astc_bh, e->f.astc_srgb, e->f.astc_bc, e->pub.image,
                                 e->pub.aspect, rbuf);
        perf_end(PERF_ASTC, a0);
        if (!checked)
            log_limited(&n_undecoded, "ASTC texture %#llx (fmt %#x %ux%u) not decoded on the GPU: left as it was",
                        (unsigned long long)e->pub.addr, d->format, d->width, d->height);
    } else {
        vkCmdCopyBufferToImage(cb, sb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, n, rg);
    }
    barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
    vk_upload_end();
    free(ar);
    free(rg);
    uint64_t dt = kern_now_ns() - u0;
    if (e->uploaded) {
        n_reuploads++, perf_count(PERF_CNT_TEX_REUP);
        ru_n++, ru_bytes += t->L.size, ru_ns += dt;
        if (e->wt_n) {
            ru_wt_n++, ru_wt_bytes += t->L.size, ru_wt_ns += dt;
            if (rt_trace & RT_TRACE_PERF)
                rt_log("perf: frame %llu: re-upload after %u WriteTexels: texture %#llx fmt %#x %ux%u, %.1f KiB, %.3f ms",
                       (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), e->wt_n,
                       (unsigned long long)e->pub.addr, d->format, d->width, d->height, t->L.size / 1024.0, dt / 1e6);
        }
    } else {
        n_uploads++, perf_count(PERF_CNT_TEX_UP);
    }
    e->wt_n = 0;
    e->uploaded = 1;
    e->n_up++;
    perf_end(PERF_TEX, p0);
}

// Whether e's storage fills no 4 KiB page of its own, so the write tracking can only see writes to pages it shares.
static int fills_no_page(const Tex *e) {
    return ((e->pub.addr + e->t.L.size) & ~0xfffull) <= ((e->pub.addr + 0xfff) & ~0xfffull);
}

// A texture that fills no page keeps a copy of its storage bytes from when a GPU write to it was recorded
// (vk_texture_gpu_written), up to SNAP_MAX bytes.
enum { SNAP_MAX = 4096 };
static volatile int32_t n_kept;

// vk_mem_changed over the pages wholly inside e's storage: a neighbour sharing the first or last page is not a write to e.
// A texture that fills no page shares all of its pages, so a page write counts there only when e's own bytes differ from the
// copy taken at the GPU write.
static int storage_rewritten(const Tex *e, uint32_t since, uint32_t *now) {
    uint64_t lo = (e->pub.addr + 0xfff) & ~0xfffull, hi = (e->pub.addr + e->t.L.size) & ~0xfffull;
    if (hi <= lo) lo = e->pub.addr, hi = e->pub.addr + e->t.L.size;
    int ch = vk_mem_changed(lo, hi - lo, since, now);
    if (ch != 1 || !since || !e->snap || memcmp(e->snap, (const void *)(uintptr_t)e->pub.addr, e->t.L.size)) return ch;
    log_limited(&n_kept, "write-back of %#llx kept: fmt %#x %ux%u, %llu bytes; a neighbour wrote its page",
                (unsigned long long)e->pub.addr, e->t.d.format, e->t.d.width, e->t.d.height,
                (unsigned long long)e->t.L.size);
    return 0;
}

// A mirrored texture is uploaded on first use and again when its storage pages carry a newer write generation.
static void make_fresh(Tex *e) {
    if (!e->pub.mirrored || e->pub.gpu_owned) return;
    uint64_t size = e->t.L.size;
    if (!e->uploaded) {
        vk_mem_changed(e->pub.addr, size, 0, &e->wgen);   // resets the write bits before the storage is read
        upload(e);
        return;
    }
    uint32_t now;
    int ch = vk_mem_changed(e->pub.addr, size, e->wgen, &now);
    e->wgen = now;
    e->last_ch = ch;
    if (ch) upload(e);   // 1 written, -1 not tracked: both upload again
}

// make_fresh for draw bindings: the page check answers from this submission's write-watch snapshot (vk_mem_changed_snap)
// instead of a system call per texture. A write the snapshot missed keeps its watch bit and is caught by the next one.
static void make_fresh_snap(Tex *e) {
    if (!e->pub.mirrored || e->pub.gpu_owned) return;
    uint32_t now;
    n_checks++;
    int ch = vk_mem_changed_snap(e->pub.addr, e->t.L.size, e->uploaded ? e->wgen : 0, submit_epoch, &now);
    e->wgen = now;
    if (!e->uploaded) { upload(e); return; }
    e->last_ch = ch;
    if (ch) upload(e);
}

// ---------------------------------------------------------------- aliases
// Bytes per block and block size of the image's VkFormat (ASTC decodes to RGBA8, RGB expands to RGBA); 0 when the
// VkFormat substitutes another one of unknown size.
static int img_block(const Tex *e, int *bw, int *bh) {
    if (e->f.astc_bc) { *bw = *bh = 4; return 16; }
    if (e->f.astc_bw) { *bw = *bh = 1; return 4; }
    int b = nvn_fmt_bpb(e->t.d.format, bw, bh);
    if (e->f.expand_rgb) { *bw = *bh = 1; return b / 3 * 4; }
    return e->f.vk == e->f.ideal ? b : 0;
}

// The new entry e can take src's content texel for texel: colour, one sample, the same shape and storage layout, and
// image formats of one block size.
static int alias_compatible(const Tex *e, const Tex *src) {
    const VkTexImage *p = &e->pub, *q = &src->pub;
    if (e->f.depth || e->f.stencil || src->f.depth || src->f.stencil || p->samples > 1 || q->samples > 1) return 0;
    if (p->type != q->type || p->width != q->width || p->height != q->height || p->depth != q->depth) return 0;
    if (e->t.L.size != src->t.L.size || nvn_tex_is_linear(&e->t.d) != nvn_tex_is_linear(&src->t.d)) return 0;
    int ew, eh, sw, sh, eb = img_block(e, &ew, &eh), sb = img_block(src, &sw, &sh);
    return eb && eb == sb && ew == sw && eh == sh;
}

// src holds content the storage bytes do not: the GPU wrote it and, for a mirrored texture, it was not written back
// and the CPU has not rewritten the storage since (the new entry's upload then reads the newer bytes).
static int alias_has_gpu_content(const Tex *src) {
    if (!src->pub.gpu_owned) return 0;
    if (!src->pub.mirrored) return 1;
    uint32_t now;
    return src->gpu_dirty && !storage_rewritten(src, src->gpu_gen, &now);
}

// Maxwell reads the same bytes through both descriptors: src's content goes into the new image e at the replay's current
// position, before any later command uses e. Through a buffer, since bound images share memory and image copies may not
// overlap; e leaves UNDEFINED between the halves. 0 when no replay is recording (e is then not initialised yet).
static int alias_copy(Tex *e, const Tex *src) {
    int reopen;
    VkCommandBuffer cb = vk_exec_inline_begin(&reopen);
    if (!cb) return 0;
    int bw, bh, bpb = img_block(e, &bw, &bh);
    uint32_t levels = e->pub.levels < src->pub.levels ? e->pub.levels : src->pub.levels;
    uint32_t layers = e->pub.layers < src->pub.layers ? e->pub.layers : src->pub.layers, n = levels * layers, k = 0;
    VkBufferImageCopy *rg = calloc(n, sizeof *rg);
    if (!rg) rt_fatal("vk: out of memory for an alias copy");
    VkDeviceSize total = 0;
    for (uint32_t lv = 0; lv < levels; lv++)
        for (uint32_t ly = 0; ly < layers; ly++, k++) {
            uint32_t w = mip(e->pub.width, lv), h = mip(e->pub.height, lv), z = mip(e->pub.depth, lv);
            rg[k] = (VkBufferImageCopy){.bufferOffset = total, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, lv, ly, 1},
                                        .imageExtent = {w, h, z}};
            total += ((VkDeviceSize)(w + bw - 1) / bw * ((h + bh - 1) / bh) * z * bpb + 15) & ~15ull;
        }
    VkBuffer buf = vk_exec_inline_buffer(total);
    if (!buf) {
        free(rg);
        vk_exec_inline_end(reopen);
        return 0;
    }
    barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    vkCmdCopyImageToBuffer(cb, src->pub.image, VK_IMAGE_LAYOUT_GENERAL, buf, n, rg);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT};
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                                .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = e->pub.image,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb,
                           .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(cb, &di);
    vkCmdCopyBufferToImage(cb, buf, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, n, rg);
    barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
    vk_exec_inline_end(reopen);
    free(rg);
    return 1;
}

static uint64_t n_overlap_seed, n_overlap_none, n_init_inline;

static Tex *entry_of(const tex_t *t) {
    TexKey k;
    make_key(t, &k);
    Tex *e = nvn_hm_get(&by_key, &k);
    if (!e) {
        static int alias_said;
        if (!alias_said) {
            alias_said = 1;
            rt_log("vk: texture aliases: %s", "a new image takes the live image's content");
        }
        if (!(e = create_tex(t, &k, 1))) return NULL;
        perf_count(PERF_CNT_TEX_NEW);
        // released entries of another descriptor at this address are replaced by the new one; a live one holding
        // GPU-written content is the alias source, preferring the one that differs only in swizzle (same_but_swizzle)
        int alias = 0;
        Tex *src = NULL;
        for (Tex *p = nvn_hm_get(&by_addr, &k.addr), *nx; p; p = nx) {
            nx = p->next_at;
            if (p->released) {
                // A released render target stays when another descriptor is made at its address: it is dead only once that
                // one has been drawn (sweep_trim). Replacing it here would leave a slot that sampled it without its image. A
                // storage-backed one is replaced.
                if (!p->pub.mirrored) { n_inuse_kept++; continue; }
                destroy_why = "new texture at its address", destroy_tex(p);
                continue;
            }
            alias = 1;
            if ((!src || (same_but_swizzle(&p->key, &k) && !same_but_swizzle(&src->key, &k))) &&
                alias_compatible(e, p) && alias_has_gpu_content(p))
                src = p;
        }
        if (alias) perf_count(PERF_CNT_TEX_ALIAS);
        {
            if (src && alias_copy(e, src)) {
                n_alias_copied++;
                if (e->pub.mirrored) {   // the content is src's GPU render, newer than the storage bytes
                    vk_mem_changed(e->pub.addr, e->t.L.size, 0, &e->wgen);
                    e->uploaded = 1;
                    e->pub.gpu_owned = 1;
                }
            } else {
                {
                    if (!e->pub.mirrored && !e->bound_to) n_overlap_none++;
                    static volatile int32_t told;
                    if (!e->pub.mirrored && plat_add32(&told, 1) <= 3000) {   // diagnostic: why a new GPU-only image starts empty
                        const Tex *near = NULL;
                        uint64_t nd = ~0ull;
                        for (uint32_t i = 0; i < by_key.cap; i++) {
                            const Tex *q = by_key.vals[i];
                            if (!q || q == e || q->pub.mirrored || !q->wr_frame) continue;
                            uint64_t d = q->key.addr > e->key.addr ? q->key.addr - e->key.addr : e->key.addr - q->key.addr;
                            if (d < nd) nd = d, near = q;
                        }
                        rt_log("vk: new-empty f%llu: %#llx %ux%u nvn%#x size %#llx %s, alias source %s, overlap source no; nearest drawn image: "
                               "%#llx %ux%u nvn%#x size %#llx %s, written %lld frames ago",
                               (unsigned long long)vk_frame_no, (unsigned long long)e->key.addr, e->pub.width, e->pub.height, e->key.d.format,
                               (unsigned long long)e->t.L.size, e->bound_to ? "bound into its pool" : "own memory", src ? "yes" : "no",
                               (unsigned long long)(near ? near->key.addr : 0), near ? near->pub.width : 0, near ? near->pub.height : 0,
                               near ? near->key.d.format : 0, (unsigned long long)(near ? near->t.L.size : 0),
                               near ? (near->released ? "released" : "live") : "-", near ? (long long)(vk_frame_no - near->wr_frame) : -1ll);
                    }
                    // An image bound into pool memory gets its first transition (UNDEFINED -> GENERAL) at the replay's position,
                    // not in the upload command buffer: that runs before the whole part, and earlier commands of the part may
                    // still read the pool memory the new image aliases.
                    int reopen;
                    VkCommandBuffer icb = e->bound_to ? vk_exec_inline_begin(&reopen) : VK_NULL_HANDLE;
                    if (icb) {
                        init_image(e, icb);
                        vk_exec_inline_end(reopen);
                        n_init_inline++;
                    } else {
                        init_image(e, vk_upload_begin());
                        vk_upload_end();
                    }
                }
                if (alias) {
                    if (e->pub.mirrored) n_alias_upload++;
                    else n_alias_uninit++;
                }
            }
        }
        e->next_at = nvn_hm_get(&by_addr, &k.addr);
        nvn_hm_put(&by_addr, &k.addr, e);
        nvn_hm_put(&by_key, &k, e);
        tex_gen_bump(k.addr);
    }
    if (e->released) {
        uint64_t age = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) - e->rel_frame;
        int b = age < 60 ? 0 : age < 600 ? 1 : age < 3600 ? 2 : 3;
        rt_back_n[b]++, rt_back_bytes[b] += e->rel_bytes;
        e->released = 0, nreleased--, rel_unlink(e), tex_gen_bump(k.addr);
    }
    e->t = *t;
    return e;
}

// vk_dump: upload bookkeeping of an image vk_texture_get / vk_texture_by_id returned. pages_written: whether its
// storage pages were written since the last check (1, 0, -1 not tracked); the query leaves that check's state alone.
void vk_texture_debug(const VkTexImage *img, uint32_t *uploads, int *last_change, int *pages_written) {
    lock();
    const Tex *e = (const Tex *)img;   // pub is the first member
    uint32_t now;
    *uploads = e->n_up;
    *last_change = e->last_ch;
    *pages_written = e->pub.mirrored && e->uploaded ? vk_mem_changed(e->pub.addr, e->t.L.size, e->wgen, &now) : 0;
    unlock();
}

const VkTexImage *vk_texture_get(const tex_t *t) {
    if (!t || !t->mem) return NULL;
    lock();
    Tex *e = entry_of(t);
    if (e) make_fresh(e);
    unlock();
    return e ? &e->pub : NULL;
}

// Small render targets the CPU reads without NVN's invalidate (vk_texture_write_back_small); small_pending: one may be
// dirty. wb_limit is the largest target (bytes) that goes back when the CPU waits for a fence. wb_trace: OOT_TRACE=gfx, read together with it.
enum { WB_SMALL_DEFAULT = 64 << 10 };
static int small_pending;
static const uint64_t wb_limit = 262144;
static int wb_trace;
static int wb_cfg_read;

static void wb_cfg_init(void) {
    if (wb_cfg_read) return;
    wb_trace = (rt_trace & RT_TRACE_GFX) != 0;
    wb_cfg_read = 1;
    rt_log("vk: fence write-back limit %llu bytes", (unsigned long long)wb_limit);
}

static void pre_note(Tex *e);

// A GPU write of e in the submission being recorded. A finalized slot resolves to the same-shape image at its address the
// GPU wrote last, and slots keep their resolution while their address generation stands: when e takes "written last" over
// from another image at the address, the generation moves.
static uint64_t n_wr_takeover;
static void wr_stamp(Tex *e) {
    uint64_t f = vk_exec_recording_fence();
    if (f > e->wr_fence) {
        for (Tex *p = nvn_hm_get(&by_addr, &e->key.addr); p; p = p->next_at)
            if (p != e && p->wr_fence && p->wr_fence >= e->wr_fence) {
                tex_gen_bump(e->key.addr), n_wr_takeover++;
                break;
            }
        e->wr_fence = f;
    }
    e->wr_frame = vk_frame_no ? vk_frame_no : 1;
}

void vk_texture_gpu_written(const tex_t *t) {
    if (!t || !t->mem) return;
    vk_exec_frame_written(t);
    lock();
    Tex *e = entry_of(t);
    if (e) {
        make_fresh(e);   // a partial render keeps what the storage held before
        e->pub.gpu_owned = 1;
        wr_stamp(e);
        if (e->pub.mirrored) {
            e->gpu_dirty = 1;
            storage_rewritten(e, 0, &e->gpu_gen);   // CPU writes up to here are older than this render
            if (!e->snap && e->t.L.size <= SNAP_MAX && fills_no_page(e) && !(e->snap = malloc(SNAP_MAX)))
                rt_fatal("vk: out of memory for a texture's storage copy");
            if (e->snap) memcpy(e->snap, (const void *)(uintptr_t)e->pub.addr, e->t.L.size);
            e->gpu_seq = ++gpu_writes;
            e->gpu_fence = vk_exec_recording_fence();
            wb_cfg_init();
            if (e->t.L.size <= wb_limit) __atomic_store_n(&small_pending, 1, __ATOMIC_RELEASE);
            own_add(e);
            pre_note(e);
        }
    }
    unlock();
}

void vk_texture_release(const tex_t *t) {
    if (!t || !t->mem) return;
    TexKey k;
    make_key(t, &k);
    lock();
    Tex *e = nvn_hm_get(&by_key, &k);
    if (e && !e->released) {
        e->released = 1;
        nreleased++;
        rel_append(e);
        tex_gen_bump(k.addr);
        sweep_released();
    }
    unlock();
}

void vk_texture_range_gone(uint64_t lo, uint64_t size, VmaAllocation alloc) {
    lock();
    Tex **list = malloc(sizeof *list * (size_t)(by_key.n + 1));
    int n = 0;
    for (uint32_t i = 0; i < by_key.cap; i++) {
        Tex *e = by_key.vals[i];
        if (e && ((e->key.addr >= lo && e->key.addr - lo < size) || (alloc && e->bound_to == alloc))) list[n++] = e;
    }
    for (int i = 0; i < n; i++) destroy_why = "its memory range went", destroy_tex(list[i]);
    free(list);
    unlock();
}

// ---------------------------------------------------------------- descriptor slots
static int descriptor_key(const dpool_ref *pool, int id, uint32_t limit, VkDescKey *key) {
    if (!pool || !pool->addr || !pool->count || pool->count > limit || id < 0 || (uint32_t)id >= pool->count ||
        pool->addr % NVN_DESCRIPTOR_BYTES || pool->addr > UINT64_MAX - (uint64_t)pool->count * NVN_DESCRIPTOR_BYTES) return 0;
    // A descriptor table that lies whole in one imported pool keys each entry by its address, so the pool lookup per texture
    // handle per draw (vk_mem_descriptor_key) is asked once per table and memory generation. Texture lock held.
    static struct { uint64_t addr, gen; uint32_t count; uint8_t import; } memo[2];
    uint64_t at = pool->addr + (uint64_t)(uint32_t)id * NVN_DESCRIPTOR_BYTES;
    {
        uint64_t g = vk_mem_gen();
        int m = memo[0].addr == pool->addr ? 0 : memo[1].addr == pool->addr ? 1 : -1;
        if (m < 0 || memo[m].gen != g || memo[m].count != pool->count) {
            static int turn;
            if (m < 0) m = turn++ & 1;
            VkDescKey whole;
            int ok = vk_mem_descriptor_key(pool->addr, (uint64_t)pool->count * NVN_DESCRIPTOR_BYTES, &whole);
            VkDescKey want = {0, pool->addr};
            memo[m].addr = pool->addr, memo[m].gen = g, memo[m].count = pool->count;
            memo[m].import = (uint8_t)(ok && !memcmp(&whole, &want, sizeof want));
        }
        if (memo[m].import) {
            memset(key, 0, sizeof *key);
            *key = (VkDescKey){0, at};
            return 1;
        }
    }
    return vk_mem_descriptor_key(at, NVN_DESCRIPTOR_BYTES, key);
}

// Texture lock held. Entries are addressed by their bytes, so wrappers and overlapping table views share updates.
static DescSlot *descriptor_slot(const dpool_ref *pool, int id, uint32_t limit, int create) {
    VkDescKey key;
    if (!descriptor_key(pool, id, limit, &key)) return NULL;
    DescSlot *d = nvn_hm_get(&descriptors, &key);
    if (!d && create) {
        d = calloc(1, sizeof *d);
        if (!d) rt_fatal("vk: out of memory for a descriptor");
        nvn_hm_put(&descriptors, &key, d);
    }
    return d;
}

// ---- texture ids for the bindless set: the list the bindless code takes the due ids from (bl_dirty).
enum { BLD_CAP = 8192 };
static uint32_t bl_dirty[BLD_CAP];
volatile int vk_texture_bl_ndirty;   // read without the lock (bindless_bind)
static uint64_t n_bl_reg, n_bl_waited, n_bl_lost;

// The ids due (texture lock held): up to cap of them, removed from the list.
int vk_texture_bl_dirty_locked(uint32_t *ids, int cap) {
    int n = vk_texture_bl_ndirty < cap ? vk_texture_bl_ndirty : cap;
    memcpy(ids, bl_dirty + vk_texture_bl_ndirty - n, sizeof *ids * (size_t)n);
    __atomic_store_n(&vk_texture_bl_ndirty, vk_texture_bl_ndirty - n, __ATOMIC_RELEASE);
    return n;
}

static Tex *slot_resolve(TexSlot *s, view_t *view, int *has_view);
// The image of texture id, only if its texture has one already (texture lock held); otherwise NULL and the id waits for
// the image (wait != 0).
const VkTexImage *vk_texture_by_id_have_locked(const dpool_ref *pool, int id, view_t *view, int *has_view, int wait) {
    DescSlot *d = descriptor_slot(pool, id, 1u << 20, 0);
    if (!d || d->kind != DESC_TEXTURE || !d->u.tex.t.mem) return NULL;
    TexKey k;
    make_key(&d->u.tex.t, &k);
    if (nvn_hm_get(&by_key, &k)) return vk_texture_by_id_locked(pool, id, view, has_view);
    if (wait) n_bl_waited++;   // due again when entry_of makes the image (bl_wait_made)
    return NULL;
}

static void descriptor_registered(DescSlot *d, uint32_t kind) {
    memset(d, 0, sizeof *d);
    if (!++reg_serial) ++reg_serial;
    d->serial = reg_serial, d->kind = kind;
    slots_gen++, samp_gen++;   // a registration can replace bytes previously used by either kind
}

void vk_texture_register(const dpool_ref *pool, int id, const tex_t *t, const view_t *v) {
    static volatile int32_t n_unbacked;
    if (!pool || id < 0 || id >= (1 << 20) || (uint32_t)id >= pool->count) rt_fatal("vk: texture slot %d out of pool range", id);
    lock();
    DescSlot *d = descriptor_slot(pool, id, 1u << 20, 1);
    if (!d) {
        log_limited(&n_unbacked, "texture descriptor slot %d has no mapped backing: not registered", id);
        unlock();
        return;
    }
    descriptor_registered(d, DESC_TEXTURE);
    TexSlot *s = &d->u.tex;
    s->t = *t;
    s->used = 1;
    s->ce = NULL;
    s->has_view = v != NULL;
    if (v) s->v = *v;
    else memset(&s->v, 0, sizeof s->v);
    unlock();
}

// The entry slot s resolves to, and the view and has_view a caller gets with it (the slot's, with the descriptor's
// swizzle when the entry is another descriptor at the same storage). NULL when the slot names no texture.
static Tex *slot_resolve(TexSlot *s, view_t *view, int *has_view) {
    *view = s->v;
    *has_view = s->has_view;
    TexKey k;
    make_key(&s->t, &k);
    Tex *e = nvn_hm_get(&by_key, &k);
    if (!e || e->released) {   // a finalized slot follows the live texture at its storage
        // A finalized slot follows a live entry at its storage: preferably one that differs from the slot descriptor only in
        // swizzle (transient render targets share storage, and the newest one at the address is usually another size); else,
        // whether or not the slot's own entry still exists, only a live entry of the descriptor's shape (size, depth, layers,
        // samples, storage size) whose format a view of the slot's format can read (never depth <-> colour); otherwise the slot
        // gets its own image (entry_of: a mirrored one from its storage, a GPU-only one zeroed). Following a transient of
        // another size would flip the slot between images from draw to draw. Slot resolution: see docs/how-it-works.md, "Slot resolution".
        static int keep_own_said;
        if (!keep_own_said) {
            keep_own_said = 1;
            rt_log("vk: finalized texture slots follow a live entry of their own shape, else their own image");
        }
        // Among the entries of the slot's shape at the address (released too), the one the GPU wrote last holds what the
        // shared memory holds (wr_fence; a live one wins a tie, then the slot's own).
        Tex *own = e, *any = NULL, *shaped = NULL, *newest = NULL;
        e = NULL;
        TexVkFormat kf;
        int kf_ok = !own && tex_vk_format(k.d.format, &kf);
        for (Tex *p = nvn_hm_get(&by_addr, &k.addr); p; p = p->next_at) {
            if (p == own) continue;
            int fits = own ? p->pub.width == own->pub.width && p->pub.height == own->pub.height &&
                             p->pub.depth == own->pub.depth && p->pub.layers == own->pub.layers &&
                             p->pub.samples == own->pub.samples && p->t.L.size == own->t.L.size &&
                             (p->pub.format == own->pub.format || vk_color_view_compatible(p->pub.format, own->pub.format))
                           : kf_ok && p->key.d.width == k.d.width && p->key.d.height == k.d.height &&
                             p->key.d.depth == k.d.depth && p->key.d.samples == k.d.samples &&
                             p->key.d.target == k.d.target && p->key.d.levels == k.d.levels &&
                             (p->pub.format == kf.vk || vk_color_view_compatible(p->pub.format, kf.vk));
            // Only an image whose content the GPU made: a mirrored texture's content is its storage, and its slot gets its
            // own entry, uploaded from there.
            if (!fits || !p->wr_fence || (p->pub.mirrored && !p->pub.gpu_owned)) continue;
            if (!newest || p->wr_fence > newest->wr_fence || (p->wr_fence == newest->wr_fence && newest->released && !p->released))
                newest = p;
        }
        if (newest && own && own->wr_fence >= newest->wr_fence) newest = NULL;   // the slot's own image is as new
        for (Tex *p = nvn_hm_get(&by_addr, &k.addr); p; p = p->next_at) {
            if (p->released) continue;
            if (!any) any = p;
            if (same_but_swizzle(&p->key, &k)) { e = p; break; }
            if (!shaped && own && p->pub.width == own->pub.width && p->pub.height == own->pub.height &&
                p->pub.depth == own->pub.depth && p->pub.layers == own->pub.layers &&
                p->pub.samples == own->pub.samples && p->t.L.size == own->t.L.size &&
                (p->pub.format == own->pub.format || vk_color_view_compatible(p->pub.format, own->pub.format)))
                shaped = p;
            if (!shaped && kf_ok && (!p->pub.mirrored || p->pub.gpu_owned) && p->key.d.width == k.d.width &&
                p->key.d.height == k.d.height && p->key.d.depth == k.d.depth && p->key.d.samples == k.d.samples &&
                p->key.d.target == k.d.target && p->key.d.levels == k.d.levels &&
                (p->pub.format == kf.vk || vk_color_view_compatible(p->pub.format, kf.vk)))
                shaped = p;
        }
        if (!e && newest) e = newest;
        if (!e) e = own ? (shaped ? shaped : own) : shaped;
        if (!own && any) { if (e) n_orphan_shaped++; else n_orphan_made++; }
    }
    if (!e && s->t.mem) {   // the slot's own image is made here: a GPU-only one starts empty (zeroed), i.e. the draw samples black
        e = entry_of(&s->t);
        if (e && !e->pub.mirrored) {
            n_slot_empty++;
            static volatile int32_t told;
            if (plat_add32(&told, 1) <= 2000) {
                char note[512];
                int o = 0;
                note[0] = 0;
                for (uint32_t i = 0; i < 512 && i < gone_n; i++) {   // newest first: the image of this shape that was here
                    const GoneRec *g = &gone_ring[(gone_n - 1 - i) & 511];
                    if (g->addr != k.addr || g->w != k.d.width || g->h != k.d.height || g->format != k.d.format) continue;
                    o = snprintf(note, sizeof note, "; its image went %llu frames ago (%s; then last sampled %lld, last written %lld frames before)",
                                 (unsigned long long)(vk_frame_no - g->frame), g->why, g->slot_frame ? (long long)(g->frame - g->slot_frame) : -1ll,
                                 g->wr_frame ? (long long)(g->frame - g->wr_frame) : -1ll);
                    break;
                }
                for (Tex *p = nvn_hm_get(&by_addr, &k.addr); p && o < (int)sizeof note - 96; p = p->next_at)
                    if (p != e)
                        o += snprintf(note + o, sizeof note - (size_t)o, "; there: %ux%u fmt %#x %s%s written %lld ago", p->key.d.width, p->key.d.height,
                                      p->key.d.format, p->released ? "released" : "live", p->pub.mirrored ? " mirrored" : "",
                                      p->wr_frame ? (long long)(vk_frame_no - p->wr_frame) : -1ll);
                rt_log("vk: slot-empty f%llu: a slot at %#llx (%ux%u fmt %#x, levels %u) had no image to sample: a new empty one%s",
                       (unsigned long long)vk_frame_no, (unsigned long long)k.addr, k.d.width, k.d.height, k.d.format, k.d.levels, note);
            }
        } else if (e) n_slot_uploaded++;
    }
    // Following live storage must keep the descriptor's interpretation of its bits: an RGB9E5 sampler can follow R11G11B10F
    // storage written through a uint view at the same address.
    if (e && s->t.d.format != e->t.d.format && !(s->has_view && s->v.set_format)) {
        TexVkFormat mapped;
        if (tex_vk_format(s->t.d.format, &mapped) && vk_color_view_compatible(e->pub.format, mapped.vk)) {
            view->format = s->t.d.format;
            view->set_format = 1;
            *has_view = 1;
        }
    }
    // A followed texture holds the content, but sampling takes the slot descriptor's swizzle (UI text is RGBA8 sampled
    // through an (A,A,A,A) descriptor).
    if (e && !(s->has_view && s->v.set_swizzle)) {
        const nvn_texdesc *d = &s->t.d;
        uint8_t sw[4] = {(uint8_t)d->swizzle_r, (uint8_t)d->swizzle_g, (uint8_t)d->swizzle_b, (uint8_t)d->swizzle_a};
        if (!(sw[0] | sw[1] | sw[2] | sw[3]))   // descriptors built without nvnTextureBuilderSetDefaults
            sw[0] = NVN_TEXTURE_SWIZZLE_R, sw[1] = NVN_TEXTURE_SWIZZLE_G, sw[2] = NVN_TEXTURE_SWIZZLE_B,
            sw[3] = NVN_TEXTURE_SWIZZLE_A;
        if (memcmp(sw, e->pub.swz, 4)) {
            if (!*has_view) memset(view, 0, sizeof *view);
            memcpy(view->swz, sw, 4);
            view->set_swizzle = 1;
            *has_view = 1;
        }
    }
    return e;
}

void vk_texture_lock(void) { lock(); }
void vk_texture_unlock(void) { unlock(); }
uint32_t vk_texture_slots_gen(void) { return slots_gen; }

uint64_t vk_texture_slot_token(const dpool_ref *pool, int id) {
    // tex_lock is recursive: callers may already hold it, or may be checking a cache before acquiring it.
    lock();
    DescSlot *d = descriptor_slot(pool, id, 1u << 20, 0);
    uint64_t token = d && d->kind == DESC_TEXTURE ? (uint64_t)d->serial << 32 |
        __atomic_load_n(&agen[agen_idx((uintptr_t)d->u.tex.t.mem)], __ATOMIC_ACQUIRE) : 0;
    unlock();
    return token;
}
uint32_t vk_sampler_gen(void) { return __atomic_load_n(&samp_gen, __ATOMIC_RELAXED); }

// What sampler slot id resolves to, as a token that changes when the slot may answer otherwise: the serial of the
// descriptor entry its bytes key (a registration gives a new one, and other bytes key another entry); 0 when it is not
// a registered sampler. Lets a memo keep a sampler through registrations of other slots, which move samp_gen.
uint64_t vk_sampler_slot_token(const dpool_ref *pool, int id) {
    lock();
    DescSlot *d = descriptor_slot(pool, id, 1u << 12, 0);
    uint64_t token = d && d->kind == DESC_SAMPLER ? d->serial : 0;
    unlock();
    return token;
}

const VkTexImage *vk_texture_by_id(const dpool_ref *pool, int id, view_t *view, int *has_view) {
    lock();
    const VkTexImage *r = vk_texture_by_id_locked(pool, id, view, has_view);
    unlock();
    return r;
}

const VkTexImage *vk_texture_by_id_locked(const dpool_ref *pool, int id, view_t *view, int *has_view) {
    DescSlot *d = descriptor_slot(pool, id, 1u << 20, 0);
    if (!d || d->kind != DESC_TEXTURE) return NULL;
    TexSlot *s = &d->u.tex;
    uint32_t g = agen[agen_idx((uintptr_t)s->t.mem)];
    if (!s->ce || s->cgen != g) {   // resolved again: an entry at its address was created, dropped or released, or the
        view_t v;                   // slot registered (s->ce cleared); an empty result is not kept: image creation may
        int hv;                     // succeed later
        s->ce = slot_resolve(s, &v, &hv);   // entry_of may create an entry and bump g: g read before it
        s->cview = v, s->chas_view = (uint8_t)hv;
        s->cgen = g;
        s->cepoch = 0;
    }
    Tex *e = s->ce;
    // Diagnostic (OOT_SAMPLE_AGE_LOG=1): a GPU-only image of 256+ pixels a side sampled although the GPU last wrote it 2 or
    // more frames ago (or never); once per image and frame, 6000 lines.
    static int age_log = -1;
    if (age_log < 0) age_log = getenv("OOT_SAMPLE_AGE_LOG") && atoi(getenv("OOT_SAMPLE_AGE_LOG"));
    if (age_log && e && !e->pub.mirrored && e->pub.width >= 256 && e->pub.height >= 256 && e->slot_frame != vk_frame_no &&
        (!e->wr_frame || vk_frame_no - e->wr_frame >= 2)) {
        static volatile int32_t told;
        if (plat_add32(&told, 1) <= 6000)
            rt_log("vk: sample-age f%llu slot %d: %#llx %ux%u fmt %#x %s, written %lld frames ago", (unsigned long long)vk_frame_no, id,
                   (unsigned long long)e->key.addr, e->pub.width, e->pub.height, e->key.d.format, e->released ? "released" : "live",
                   e->wr_frame ? (long long)(vk_frame_no - e->wr_frame) : -1ll);
    }
    if (e) {   // in use (rel_in_use). Marked while the entry is live too: a target sampled while live, then finalized and
               // at once overlaid by a new texture of another shape was replaced before any slot met it released, and
               // its slot got a new, zeroed image.
        uint64_t fr = vk_frame_no;
        if (e->slot_frame != fr) e->slot_frame = fr ? fr : 1;
    }
    // Bindings of one submit share one check, answered from the submit's write-watch snapshot. After one check an entry is
    // either uploaded or never uploads, so a slot that checked its entry in this submit has nothing left to do.
    if (e && s->cepoch != submit_epoch) {
        if (!e->uploaded || e->checked != submit_epoch) {
            make_fresh_snap(e);
            e->checked = submit_epoch;
        }
        s->cepoch = submit_epoch;
    }
    if (view) *view = has_view ? s->cview : s->v;
    if (has_view) *has_view = view ? s->chas_view : s->has_view;
    return e ? &e->pub : NULL;
}

// vk_dump: the swizzle selectors of the descriptor registered in slot id; 0 when the slot is empty.
int vk_texture_slot_swz(const dpool_ref *pool, int id, uint8_t swz[4]) {
    lock();
    DescSlot *s = descriptor_slot(pool, id, 1u << 20, 0);
    int ok = s && s->kind == DESC_TEXTURE;
    if (ok) {
        const nvn_texdesc *d = &s->u.tex.t.d;
        swz[0] = (uint8_t)d->swizzle_r, swz[1] = (uint8_t)d->swizzle_g, swz[2] = (uint8_t)d->swizzle_b;
        swz[3] = (uint8_t)d->swizzle_a;
    }
    unlock();
    return ok;
}

uint32_t vk_texture_epoch(void) { return __atomic_load_n(&submit_epoch, __ATOMIC_RELAXED); }

void vk_texture_new_submit(void) {
    lock();
    if (!++submit_epoch) submit_epoch = 1;
    sweep_pressure();
    sweep_trim();
    static uint64_t log_at;
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) / 60;
    if ((rt_trace & RT_TRACE_PERF) && f != log_at) {
        log_at = f;
        rt_log("perf: frame %llu: texture aliases: content copied %llu, filled from storage %llu, left uninitialised %llu; new GPU-only "
               "images with memory of their own: %llu took the content of the image their storage overlaps, %llu started empty",
               (unsigned long long)f * 60, (unsigned long long)n_alias_copied, (unsigned long long)n_alias_upload,
               (unsigned long long)n_alias_uninit, (unsigned long long)n_overlap_seed, (unsigned long long)n_overlap_none);
        if (n_init_inline) rt_log("perf: frame %llu: %llu pool-bound images got their first transition in replay order",
                                  (unsigned long long)f * 60, (unsigned long long)n_init_inline);
        n_alias_copied = n_alias_upload = n_alias_uninit = n_overlap_seed = n_overlap_none = n_init_inline = 0;
        rt_log("perf: frame %llu: texture re-uploads %llu (%.1f MiB, %.2f ms), after WriteTexels %llu (%.1f MiB, %.2f ms)",
               (unsigned long long)f * 60, (unsigned long long)ru_n, ru_bytes / 1048576.0, ru_ns / 1e6,
               (unsigned long long)ru_wt_n, ru_wt_bytes / 1048576.0, ru_wt_ns / 1e6);
        ru_n = ru_bytes = ru_ns = ru_wt_n = ru_wt_bytes = ru_wt_ns = 0;
        uint64_t ck, calls, untr;
        vk_mem_snap_stats(&ck, &calls, &untr);
        rt_log("perf: frame %llu: binding storage checks %llu (texture %llu), write-watch snapshot queries %llu, untracked %llu",
               (unsigned long long)f * 60, (unsigned long long)ck, (unsigned long long)n_checks, (unsigned long long)calls,
               (unsigned long long)untr);
        n_checks = 0;
        const double M = 1048576.0;
        uint64_t bn = rt_back_n[0] + rt_back_n[1] + rt_back_n[2] + rt_back_n[3];
        uint64_t bb = rt_back_bytes[0] + rt_back_bytes[1] + rt_back_bytes[2] + rt_back_bytes[3];
        rt_log("perf: frame %llu: released textures kept %d, dropped under VRAM pressure %llu; own memory %.0f MiB (intact "
               "%.0f MiB); by the cache cap %llu (%.0f MiB), %llu submits over it with nothing to drop; taken back %llu "
               "(%.0f MiB): under 1 s %llu, 1-10 s %llu (%.0f MiB), 10-60 s %llu (%.0f MiB), older %llu (%.0f MiB)",
               (unsigned long long)f * 60, nreleased, (unsigned long long)n_pressure_drops, rel_sum[0] / M, rel_sum[1] / M,
               (unsigned long long)n_cap_drops, cap_drop_bytes / M, (unsigned long long)n_cap_stuck,
               (unsigned long long)bn, bb / M, (unsigned long long)rt_back_n[0], (unsigned long long)rt_back_n[1],
               rt_back_bytes[1] / M, (unsigned long long)rt_back_n[2], rt_back_bytes[2] / M,
               (unsigned long long)rt_back_n[3], rt_back_bytes[3] / M);
        n_pressure_drops = n_cap_drops = cap_drop_bytes = n_cap_stuck = 0;
        memset(rt_back_n, 0, sizeof rt_back_n), memset(rt_back_bytes, 0, sizeof rt_back_bytes);
        rt_log("perf: frame %llu: released render targets dropped: drawn over %llu (%.0f MiB), a minute unused %llu (%.0f MiB); "
               "kept at a new texture's address %llu, in use but evicted at the entry limit %llu; slots without "
               "their own entry while another shape was live: followed a same-shape image %llu, own image made again %llu",
               (unsigned long long)f * 60, (unsigned long long)n_dead_drops, dead_drop_bytes / M, (unsigned long long)n_old_drops,
               old_drop_bytes / M, (unsigned long long)n_inuse_kept, (unsigned long long)n_inuse_capped,
               (unsigned long long)n_orphan_shaped, (unsigned long long)n_orphan_made);
        n_dead_drops = dead_drop_bytes = n_old_drops = old_drop_bytes = 0;
        rt_log("perf: frame %llu: slots given a new image at a lookup: empty (GPU-only) %llu, from storage %llu; \"written last\" at an "
               "address went to another image %llu times",
               (unsigned long long)f * 60, (unsigned long long)n_slot_empty, (unsigned long long)n_slot_uploaded,
               (unsigned long long)n_wr_takeover);
        n_wr_takeover = 0;
        rt_log("perf: frame %llu: GPU-only texture ids registered for the bindless set %llu, of them waited for their image %llu, "
               "dropped (list full) %llu", (unsigned long long)f * 60, (unsigned long long)n_bl_reg, (unsigned long long)n_bl_waited,
               (unsigned long long)n_bl_lost);
        n_bl_reg = n_bl_waited = n_bl_lost = 0;
        if (n_sampled_destroyed) rt_log("perf: frame %llu: GPU-only entries destroyed within 10 s of a slot sampling them: %llu",
                                        (unsigned long long)f * 60, (unsigned long long)n_sampled_destroyed);
        n_sampled_destroyed = 0;
        n_slot_empty = n_slot_uploaded = 0;
        n_inuse_kept = n_inuse_capped = n_orphan_shaped = n_orphan_made = 0;
        if (f % 10 == 0) rel_census(f * 60);
        rt_log("perf: frame %llu: texture write-backs through a GPU wait: at a fence %llu (%.2f ms), for a CPU read %llu "
               "(%.2f ms); tex_prestage %lld: staged %llu, used %llu, not usable %llu", (unsigned long long)f * 60,
               (unsigned long long)wbk_n[0], wbk_ns[0] / 1e6, (unsigned long long)wbk_n[1], wbk_ns[1] / 1e6,
               262144ll, (unsigned long long)pre_staged, (unsigned long long)pre_used,
               (unsigned long long)pre_missed);
        wbk_n[0] = wbk_n[1] = wbk_ns[0] = wbk_ns[1] = pre_staged = pre_used = pre_missed = 0;
    }
    unlock();
}

// ---------------------------------------------------------------- texel reads and writes
typedef struct { unsigned level, layer0; int nz, is3d, bw, bh, bpb; size_t wb, hb, row, img, x0, y0; } Region;

static int region_of(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint64_t row, uint64_t img, Region *g) {
    g->bpb = nvn_fmt_bpb(t->d.format, &g->bw, &g->bh);
    if (!g->bpb || r->width <= 0 || r->height <= 0 || r->depth < 0 || r->xoffset < 0 || r->yoffset < 0 || r->zoffset < 0 ||
        r->xoffset % g->bw || r->yoffset % g->bh)
        return 0;
    g->level = v && v->set_levels ? v->base_level : 0;
    g->layer0 = v && v->set_layers ? v->min_layer : 0;
    g->nz = r->depth ? r->depth : 1;
    g->is3d = t->d.target == NVN_TEXTURE_TARGET_3D;
    g->wb = (size_t)((r->width + g->bw - 1) / g->bw) * (size_t)g->bpb;
    g->hb = (size_t)(r->height + g->bh - 1) / (size_t)g->bh;
    g->row = row ? row : g->wb;
    g->img = img ? img : g->row * g->hb;
    g->x0 = (size_t)(r->xoffset / g->bw) * (size_t)g->bpb;
    g->y0 = (size_t)(r->yoffset / g->bh);
    return 1;
}

// Pool bytes <-> caller rows of the region alone (nvn_texels_box); a linear copy of the whole level is too slow for a small
// region of a large texture while the guest thread holds its core and the texture lock.
static int storage_texels(const tex_t *t, const Region *g, const NVNcopyRegion *r, uint8_t *mem, int write) {
    for (int z = 0; z < g->nz; z++) {
        unsigned layer = g->is3d ? 0 : g->layer0 + (unsigned)r->zoffset + (unsigned)z;
        uint32_t slice = g->is3d ? (uint32_t)(r->zoffset + z) : 0;
        if (nvn_texels_box(&t->d, g->level, layer, (uint32_t)g->x0, (uint32_t)g->y0, slice, (uint32_t)g->wb,
                           (uint32_t)g->hb, t->mem, mem + (size_t)z * g->img, g->row, !write))
            return 0;
    }
    return 1;
}

static int read_aspect(const Tex *e, VkImageAspectFlags *a) {
    if (!e->f.depth && !e->f.stencil) { *a = VK_IMAGE_ASPECT_COLOR_BIT; return 1; }
    switch (e->t.d.format) {   // packed depth-stencil storage has no single-aspect copy
    case 0x31: *a = VK_IMAGE_ASPECT_STENCIL_BIT; return 1;
    case 0x32: case 0x34: *a = VK_IMAGE_ASPECT_DEPTH_BIT; return 1;
    default: return 0;
    }
}

// Image <-> caller rows through a staging buffer; the GPU copy is waited for.
static int image_texels(Tex *e, const Region *g, const NVNcopyRegion *r, uint8_t *mem, int write) {
    static volatile int32_t n_astc, n_ds;
    VkCtx *c = vk_ctx();
    VkImageAspectFlags aspect;
    if (e->f.astc_bw && !write) {
        log_limited(&n_astc, "ReadTexels of ASTC texture %#llx: the image holds decoded RGBA8; zeros returned",
                    (unsigned long long)e->pub.addr);
        return 0;
    }
    if (!read_aspect(e, &aspect)) {
        log_limited(&n_ds, "texels of packed depth-stencil format %#x are not copied", e->t.d.format);
        return 0;
    }
    if (g->level >= e->pub.levels) return 0;
    uint32_t w = (uint32_t)r->width, h = (uint32_t)r->height;
    int rgb = e->f.expand_rgb, astc = e->f.astc_bw != 0, agpu = astc && write && vk_astc_gpu();
    size_t texels = (size_t)w * h, ib = agpu ? 0 : astc ? 4 : rgb ? (size_t)g->bpb / 3 * 4 : 0;
    size_t slice = ib ? texels * ib : g->wb * g->hb;   // agpu: the raw blocks, decoded by vk_astc_record
    VkBuffer sb;
    VmaAllocation sa;
    uint8_t *stg = vk_staging(slice * (size_t)g->nz, &sb, &sa);
    VkBufferImageCopy *rg = calloc((size_t)g->nz, sizeof *rg);
    for (int z = 0; z < g->nz; z++) {
        uint8_t *s = stg + slice * (size_t)z;
        const uint8_t *m = mem + (size_t)z * g->img;
        if (write) {
            if (agpu) {
                for (size_t y = 0; y < g->hb; y++) memcpy(s + y * g->wb, m + y * g->row, g->wb);
            } else if (astc) {
                uint8_t *tight = malloc(g->wb * g->hb);
                for (size_t y = 0; y < g->hb; y++) memcpy(tight + y * g->wb, m + y * g->row, g->wb);
                uint64_t a0 = perf_begin();
                astc_decode_ldr_mt(tight, w, h, e->f.astc_bw, e->f.astc_bh, e->f.astc_srgb, s);
                perf_end(PERF_ASTC, a0);
                free(tight);
            } else if (rgb) {
                for (size_t y = 0; y < g->hb; y++) expand_rgb(e->t.d.format, m + y * g->row, s + y * w * ib, w);
            } else {
                for (size_t y = 0; y < g->hb; y++) memcpy(s + y * g->wb, m + y * g->row, g->wb);
            }
        }
        rg[z] = (VkBufferImageCopy){.bufferOffset = slice * (size_t)z,
                                    .imageSubresource = {aspect, g->level, g->is3d ? 0 : g->layer0 + (uint32_t)r->zoffset + (uint32_t)z, 1},
                                    .imageOffset = {r->xoffset, r->yoffset, g->is3d ? r->zoffset + z : 0},
                                    .imageExtent = {w, h, 1}};
    }
    VkCommandBuffer cb = vk_upload_begin();
    if (write) {
        vmaFlushAllocation(c->vma, sa, 0, VK_WHOLE_SIZE);
        barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        int done = 0;
        // A BC3 image takes whole 4x4 blocks, or a part that ends on the mip's edge; a region of a 5x5 or 6x6 footprint
        // may end inside a block (vk_astc_record checks the start corner).
        uint32_t mw = e->t.d.width >> g->level, mh = e->t.d.height >> g->level;
        int bc_cut = e->f.astc_bc && (((w & 3) && (uint32_t)r->xoffset + w != (mw ? mw : 1)) ||
                                      ((h & 3) && (uint32_t)r->yoffset + h != (mh ? mh : 1)));
        if (agpu && !bc_cut) {   // bc_cut: nothing recorded, logged below as not decoded
            VkAstcRegion *ar = calloc((size_t)g->nz, sizeof *ar);
            for (int z = 0; z < g->nz; z++)
                ar[z] = (VkAstcRegion){.src = slice * (size_t)z, .w = w, .h = h, .slices = 1, .level = rg[z].imageSubresource.mipLevel,
                                       .layer = rg[z].imageSubresource.baseArrayLayer, .offset = rg[z].imageOffset};
            uint64_t a0 = perf_begin();
            done = vk_astc_record(cb, sb, slice * (size_t)g->nz, ar, g->nz, e->f.astc_bw, e->f.astc_bh, e->f.astc_srgb, e->f.astc_bc,
                                  e->pub.image, aspect, VK_NULL_HANDLE);
            perf_end(PERF_ASTC, a0);
            free(ar);
        }
        if (!agpu) vkCmdCopyBufferToImage(cb, sb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, (uint32_t)g->nz, rg);
        else if (!done) log_limited(&n_astc, "WriteTexels of ASTC texture %#llx not decoded on the GPU: left as it was",
                                    (unsigned long long)e->pub.addr);
        barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
    } else {
        barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        vkCmdCopyImageToBuffer(cb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, sb, (uint32_t)g->nz, rg);
        barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                VK_ACCESS_2_HOST_READ_BIT);
    }
    vk_upload_end();
    vk_upload_flush(1);
    if (!write) {
        vmaInvalidateAllocation(c->vma, sa, 0, VK_WHOLE_SIZE);
        for (int z = 0; z < g->nz; z++) {
            const uint8_t *s = stg + slice * (size_t)z;
            uint8_t *m = mem + (size_t)z * g->img;
            for (size_t y = 0; y < g->hb; y++) {
                if (rgb) shrink_rgb(e->t.d.format, s + y * w * ib, m + y * g->row, w);
                else memcpy(m + y * g->row, s + y * g->wb, g->wb);
            }
        }
    }
    free(rg);
    vmaDestroyBuffer(c->vma, sb, sa);
    return 1;
}

static void zero_rows(const Region *g, uint8_t *dst) {
    for (int z = 0; z < g->nz; z++)
        for (size_t y = 0; y < g->hb; y++) memset(dst + (size_t)z * g->img + y * g->row, 0, g->wb);
}

int vk_texture_read_image(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row, uint64_t img) {
    Region g;
    if (!t || !region_of(t, v, r, row, img, &g)) return 0;
    lock();
    Tex *e = entry_of(t);
    int ok = 0;
    if (e) {
        make_fresh(e);
        ok = image_texels(e, &g, r, dst, 0);
    }
    unlock();
    if (!ok) zero_rows(&g, dst);
    return ok;
}

// ReadTexels: the pool bytes are the content of a mirrored texture the GPU has not rendered into; every other texture
// is read back from its image.
void vk_read_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row, uint64_t img) {
    static volatile int32_t n_bad;
    Region g;
    if (!region_of(t, v, r, row, img, &g)) {
        log_limited(&n_bad, "ReadTexels: invalid region %d,%d,%d %dx%dx%d (format %#x)", r->xoffset, r->yoffset,
                    r->zoffset, r->width, r->height, r->depth, t->d.format);
        return;
    }
    lock();
    Tex *e = entry_of(t);
    int ok = 0;
    if (e && e->pub.mirrored && !e->pub.gpu_owned) ok = storage_texels(t, &g, r, dst, 0);
    else if (e) {
        make_fresh(e);
        ok = image_texels(e, &g, r, dst, 0);
    }
    unlock();
    if (!ok) zero_rows(&g, dst);
}

// WriteTexels: a mirrored texture gets the texels in its pool bytes (and in its image once the GPU owns it); every
// other texture in its image.
void vk_write_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, const uint8_t *src, uint64_t row, uint64_t img) {
    static volatile int32_t n_bad;
    Region g;
    if (!region_of(t, v, r, row, img, &g) || (v && v->set_format && v->format != t->d.format)) {
        log_limited(&n_bad, "WriteTexels: invalid region %d,%d,%d %dx%dx%d or view format (format %#x)", r->xoffset,
                    r->yoffset, r->zoffset, r->width, r->height, r->depth, t->d.format);
        return;
    }
    lock();
    Tex *e = entry_of(t);
    if (e && e->pub.mirrored) {
        if (!storage_texels(t, &g, r, (uint8_t *)src, 1))
            log_limited(&n_bad, "WriteTexels: region outside level %u of texture %#llx", g.level, (unsigned long long)e->pub.addr);
        vk_guest_written((uintptr_t)t->mem, t->L.size);
        if (e->pub.gpu_owned) image_texels(e, &g, r, (uint8_t *)src, 1);
        else if (e->uploaded) e->wt_n++;
    } else if (e) {
        image_texels(e, &g, r, (uint8_t *)src, 1);
    }
    unlock();
}

// ---------------------------------------------------------------- CPU reads of GPU-written storage
// A mirrored texture the GPU rendered into holds its texels in its image only; its storage keeps what the CPU last put
// there. The game still memcpys GPU-generated maps between pools on the CPU, so such a read writes the image back first.
static int pre_take(Tex *e);

static void write_back(Tex *e) {
    if (pre_take(e)) return;
    uint64_t t0 = kern_now_ns();
    const nvn_texdesc *d = &e->t.d;
    int linear = nvn_tex_is_linear(d), is3d = d->target == NVN_TEXTURE_TARGET_3D;
    uint32_t levels = linear ? 1 : e->pub.levels, layers = linear || is3d ? 1 : e->pub.layers;
    for (uint32_t lv = 0; lv < levels; lv++)
        for (uint32_t ly = 0; ly < layers; ly++) {
            view_t v = {.set_levels = 1, .set_layers = 1, .base_level = (uint8_t)lv, .min_layer = (uint16_t)ly};
            NVNcopyRegion r = {.width = (int)mip(d->width, lv),
                               .height = d->target == 0 || d->target == 3 ? 1 : (int)mip(d->height, lv),
                               .depth = is3d ? (int)mip(d->depth, lv) : 1};
            Region g;
            if (!region_of(&e->t, &v, &r, 0, 0, &g)) continue;
            uint8_t *buf = malloc(g.img * (size_t)g.nz);
            if (!buf) rt_fatal("vk: out of memory for a texture write-back");
            if (image_texels(e, &g, &r, buf, 0)) storage_texels(&e->t, &g, &r, buf, 1);
            free(buf);
        }
    e->gpu_dirty = 0;
    wbk_n[wbk_src]++, wbk_ns[wbk_src] += kern_now_ns() - t0;
}

// Mirrored textures overlapping [addr, addr+size) that the GPU wrote since their last write-back.
static int owned_dirty(uint64_t addr, uint64_t size, Tex **hit) {
    int n = 0;
    for (int i = owned_lower(addr > max_owned ? addr - max_owned : 0); i < nowned && owned[i]->pub.addr < addr + size; i++) {
        Tex *e = owned[i];
        if (__atomic_load_n(&e->gpu_dirty, __ATOMIC_RELAXED) && e->pub.addr + e->t.L.size > addr) {
            if (!hit) return 1;
            hit[n++] = e;
        }
    }
    return n;
}

// A storage image may be written by the shaders of the submission being recorded (texture lock held, vk_exec_draw.c
// tex_info). Its storage is not marked GPU-owned: only the write fence for ReadTexels.
void vk_texture_storage_bound_locked(const VkTexImage *img) {
    Tex *e = (Tex *)img;   // pub is the first member
    wr_stamp(e);   // also the frame (wr_frame): a compute shader's output counted as never written before
}

// What an image is, for a log line (texture lock held).
void vk_texture_describe_locked(const VkTexImage *img, char *buf, size_t cap) {
    const Tex *e = (const Tex *)img;   // pub is the first member
    uint64_t f = vk_frame_no;
    snprintf(buf, cap, "%#llx %ux%u vk%d nvn%#x L%u %s%s%s%s image %p, made %lld, written %lld, sampled %lld frames ago",
             (unsigned long long)e->key.addr, e->pub.width, e->pub.height, (int)e->pub.format, e->key.d.format, e->key.d.levels,
             e->released ? "released" : "live", e->pub.mirrored ? " mirrored" : "", e->pub.gpu_owned ? " gpu-written" : "",
             e->bound_to ? " pool-bound" : " own-memory",
             (void *)e->pub.image, (long long)(f - e->born_frame), e->wr_frame ? (long long)(f - e->wr_frame) : -1ll,
             e->slot_frame ? (long long)(f - e->slot_frame) : -1ll);
}

uint64_t vk_texture_last_write(const tex_t *t) {
    lock();
    Tex *e = entry_of(t);
    uint64_t f = e ? e->wr_fence : 0;
    unlock();
    return f;
}

size_t vk_texture_texel_span(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint64_t row, uint64_t img) {
    Region g;
    if (!region_of(t, v, r, row, img, &g)) return 0;
    return (size_t)(g.nz - 1) * g.img + (g.hb - 1) * g.row + g.wb;
}

int vk_texture_cpu_stale(uint64_t addr, uint64_t size) {
    if (!size || !__atomic_load_n(&nowned, __ATOMIC_ACQUIRE)) return 0;
    plat_mutex_lock_shared(&owned_lock);
    int dirty = owned_dirty(addr, size, NULL);
    plat_mutex_unlock_shared(&owned_lock);
    return dirty;
}

// The newest submission recorded to write a GPU-written mirrored texture overlapping [addr, addr+size); 0: none.
uint64_t vk_texture_dirty_fence(uint64_t addr, uint64_t size) {
    if (!size || !__atomic_load_n(&nowned, __ATOMIC_ACQUIRE)) return 0;
    lock();
    uint64_t f = 0;
    for (int i = owned_lower(addr > max_owned ? addr - max_owned : 0); i < nowned && owned[i]->pub.addr < addr + size; i++) {
        const Tex *e = owned[i];
        if (!e->gpu_dirty || e->pub.addr + e->t.L.size <= addr) continue;
        if (e->gpu_fence > f) f = e->gpu_fence;
        if (e->wr_fence > f) f = e->wr_fence;
    }
    unlock();
    return f;
}

// The image is the newest content only while nothing wrote the storage since the GPU render (the game may reuse the storage
// for new data and memcpy that on). Every hit is checked before any is written back, since a write-back is itself a CPU
// write; hits then go back oldest render first, so the newer render wins where descriptors share storage.
void vk_texture_write_back(uint64_t addr, uint64_t size) {
    lock();
    Tex **hit = malloc(sizeof *hit * (size_t)(nowned + 1));
    if (!hit) rt_fatal("vk: out of memory for a texture write-back");
    int n = owned_dirty(addr, size, hit), k = 0;   // owned[] cannot change while tex_lock is held
    for (int i = 0; i < n; i++) {
        uint32_t now;
        if (storage_rewritten(hit[i], hit[i]->gpu_gen, &now) == 1) hit[i]->gpu_dirty = 0;
        else hit[k++] = hit[i];
    }
    for (int i = 1; i < k; i++)
        for (int j = i; j > 0 && hit[j - 1]->gpu_seq > hit[j]->gpu_seq; j--) {
            Tex *x = hit[j];
            hit[j] = hit[j - 1], hit[j - 1] = x;
        }
    wbk_src = 1;
    for (int i = 0; i < k; i++) write_back(hit[i]);
    wbk_src = 0;
    for (int i = 0; i < k; i++) perf_count(PERF_CNT_WB);
    for (int i = k; i < n; i++) perf_count(PERF_CNT_WB_SKIP);
    free(hit);
    unlock();
}

// A game may render a small map into a CPU-cached pool every frame and read the bytes without nvnMemoryPoolInvalidateMappedRange;
// on the Switch they are there once the frame's fence has passed. So when the CPU waits for a fence, mirrored targets up to
// wb_limit bytes rendered by work up to that fence go back to their storage unless the CPU rewrote it since.
int vk_texture_small_dirty(void) { return __atomic_load_n(&small_pending, __ATOMIC_ACQUIRE); }

// OOT_TRACE=gfx: a dirty target the fence write-back leaves alone because it is over the limit, or sends back although it is
// over the default limit. One line per target address and outcome; a full table stops logging.
static void wb_note(const Tex *e, int skipped) {
    static uint64_t seen[2][64];
    static int n_seen[2];
    for (int i = 0; i < n_seen[skipped]; i++)
        if (seen[skipped][i] == e->pub.addr) return;
    if (n_seen[skipped] == 64) return;
    seen[skipped][n_seen[skipped]++] = e->pub.addr;
    const pool_t *p = e->t.pool;
    unsigned fl = p ? p->flags : 0;
    rt_log("vk: fence write-back %s %#llx %ux%u fmt %#x size %llu pool %s (limit %llu)",
           skipped ? "skipped-for-size" : "over-default", (unsigned long long)e->pub.addr, e->t.d.width, e->t.d.height,
           e->t.d.format, (unsigned long long)e->t.L.size,
           !p ? "none" : (fl & NVN_MEMORY_POOL_FLAGS_CPU_CACHED) ? "cpu-cached" : (fl & NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED) ? "cpu-uncached" : "other",
           (unsigned long long)wb_limit);
}

void vk_texture_write_back_small(uint64_t fence) {
    lock();
    wb_cfg_init();
    __atomic_store_n(&small_pending, 0, __ATOMIC_RELAXED);
    plat_mutex_lock_shared(&owned_lock);
    int n = 0, later = 0;
    for (int i = 0; i < nowned; i++) n += owned[i]->gpu_dirty && owned[i]->t.L.size <= wb_limit;
    Tex **hit = n ? malloc(sizeof *hit * (size_t)n) : NULL;
    if (n && !hit) rt_fatal("vk: out of memory for a texture write-back");
    n = 0;
    for (int i = 0; i < nowned; i++) {
        Tex *e = owned[i];
        if (!e->gpu_dirty) continue;
        if (e->t.L.size > wb_limit) {
            if (wb_trace && e->gpu_fence <= fence) {
                wb_note(e, 1);
                perf_count(PERF_CNT_WB_SKIP);
            }
            continue;
        }
        if (e->gpu_fence > fence) later = 1;
        else {
            if (wb_trace && e->t.L.size > WB_SMALL_DEFAULT) wb_note(e, 0);
            hit[n++] = e;
        }
    }
    plat_mutex_unlock_shared(&owned_lock);
    if (later) __atomic_store_n(&small_pending, 1, __ATOMIC_RELEASE);
    int k = 0;
    for (int i = 0; i < n; i++) {
        uint32_t now;
        if (storage_rewritten(hit[i], hit[i]->gpu_gen, &now) == 1) hit[i]->gpu_dirty = 0;
        else hit[k++] = hit[i];
    }
    for (int i = 1; i < k; i++)
        for (int j = i; j > 0 && hit[j - 1]->gpu_seq > hit[j]->gpu_seq; j--) {
            Tex *x = hit[j];
            hit[j] = hit[j - 1], hit[j - 1] = x;
        }
    for (int i = 0; i < k; i++) write_back(hit[i]);
    for (int i = 0; i < k; i++) perf_count(PERF_CNT_WB);
    for (int i = k; i < n; i++) perf_count(PERF_CNT_WB_SKIP);
    free(hit);
    unlock();
}

// ---------------------------------------------------------------- tex_prestage
// tex_prestage: see docs/how-it-works.md, "Texture write-back". A small GPU target is written back to its storage when the CPU waits for
// its fence or reads it; image_texels does that behind the upload batch with a GPU wait under the texture and upload locks.
// With tex_prestage (262144 bytes) the end of each submission copies such targets into a host-visible ring; a write-back of
// the same render whose fence is done reads the ring instead: same bytes, no GPU wait.
#define PRE_RING (16ull << 20)
#define PRE_REGS 64
typedef struct { Region g; NVNcopyRegion r; size_t slice, ib; uint64_t off; } PreReg;
static VkBuffer pre_buf;
static VmaAllocation pre_alloc;
static uint8_t *pre_map;
static int pre_failed;
static uint64_t pre_head, pre_lapn = 1;   // tex_lock: next free byte of the ring and the lap it is in
static Tex **pre_list;                    // tex_lock: targets rendered by the submission being recorded
static int npre, cpre;

// The tex_prestage size as it applies now: 0 while the world/calm gate of acq_sync is closed (nvn_world_calm); the prestage
// must not run during world entry.
static int64_t pre_lim(void) {
    extern int nvn_world_calm(void);
    int64_t v = 262144;
    return !nvn_world_calm() ? 0 : v;
}

static void pre_note(Tex *e) {   // tex_lock
    int64_t lim = pre_lim();
    if (lim < 0) lim = -lim;
    if (lim <= 0 || e->t.L.size > (uint64_t)lim || e->f.astc_bw || e->pre_mark == e->gpu_fence) return;
    if (npre == cpre) {
        int cap = cpre ? cpre * 2 : 64;
        Tex **n = realloc(pre_list, sizeof *n * (size_t)cap);
        if (!n) return;
        pre_list = n, cpre = cap;
    }
    e->pre_mark = e->gpu_fence;
    pre_list[npre++] = e;
}

static void pre_forget(Tex *e) {   // tex_lock
    for (int i = 0; i < npre; i++)
        if (pre_list[i] == e) pre_list[i] = NULL;
    e->pre_fence = 0;
}

// The regions write_back reads, in its order, with their offsets in the staged copy; -1: too many.
static int pre_regions(const Tex *e, PreReg *out, uint64_t *total) {
    const nvn_texdesc *d = &e->t.d;
    int linear = nvn_tex_is_linear(d), is3d = d->target == NVN_TEXTURE_TARGET_3D, n = 0;
    uint32_t levels = linear ? 1 : e->pub.levels, layers = linear || is3d ? 1 : e->pub.layers;
    uint64_t off = 0;
    for (uint32_t lv = 0; lv < levels; lv++)
        for (uint32_t ly = 0; ly < layers; ly++) {
            view_t v = {.set_levels = 1, .set_layers = 1, .base_level = (uint8_t)lv, .min_layer = (uint16_t)ly};
            NVNcopyRegion r = {.width = (int)mip(d->width, lv),
                               .height = d->target == 0 || d->target == 3 ? 1 : (int)mip(d->height, lv),
                               .depth = is3d ? (int)mip(d->depth, lv) : 1};
            Region g;
            if (!region_of(&e->t, &v, &r, 0, 0, &g) || g.level >= e->pub.levels) continue;   // image_texels: nothing
            if (n == PRE_REGS) return -1;
            size_t ib = e->f.expand_rgb ? (size_t)g.bpb / 3 * 4 : 0;
            size_t slice = ib ? (size_t)r.width * (size_t)r.height * ib : g.wb * g.hb;
            out[n++] = (PreReg){g, r, slice, ib, off};
            off += ((uint64_t)slice * (uint64_t)g.nz + 15) & ~15ull;
        }
    *total = off;
    return n;
}

// A plain image only, every region inside it: colour, the exact format the NVN format maps to (no stand-in, no RGB
// expansion, no block views), one sample, 2D, and each copy within the image's own levels, layers and extent, with the
// texel size the staging layout assumes. 0: leave it to the old path. why: the first reason it is left.
static int pre_plain(const Tex *e, const PreReg *pr, int n, const char **why) {
    const VkTexImage *p = &e->pub;
    int bw, bh, bpb = nvn_fmt_bpb(e->t.d.format, &bw, &bh);
    *why = !p->image ? "no image" : e->f.depth || e->f.stencil ? "depth/stencil" : e->f.vk != e->f.ideal ? "fallback format"
         : e->f.chan[0] ? "stand-in channels" : e->f.expand_rgb ? "rgb expansion" : e->f.astc_bw ? "astc"
         : p->block_views ? "block views" : p->samples > 1 ? "multisampled" : p->type != VK_IMAGE_TYPE_2D ? "not 2D"
         : p->format != e->f.vk ? "image format differs" : !bpb ? "no block size" : NULL;
    for (int k = 0; !*why && k < n; k++) {
        const PreReg *r = &pr[k];
        if (r->g.level >= p->levels || r->g.layer0 + (uint32_t)r->r.zoffset + (uint32_t)r->g.nz > p->layers) *why = "level or layer out of the image";
        else if ((uint32_t)r->r.xoffset + (uint32_t)r->r.width > mip(p->width, r->g.level) ||
                 (uint32_t)r->r.yoffset + (uint32_t)r->r.height > mip(p->height, r->g.level)) *why = "extent out of the image";
        else if (r->slice != (size_t)((r->r.width + bw - 1) / bw) * (size_t)((r->r.height + bh - 1) / bh) * (size_t)bpb)
            *why = "staging size differs from the copy";
    }
    return !*why;
}

int vk_texture_prestage(VkCommandBuffer cb, uint64_t fence) {
    if (!__atomic_load_n(&npre, __ATOMIC_RELAXED)) return 0;
    lock();
    int64_t lim = pre_lim(), dry = lim < 0;
    if (dry) lim = -lim;   // tex_prestage < 0: dry run, the candidates are checked and logged, nothing is recorded
    if (!dry && lim > 0 && !pre_buf && !pre_failed) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = PRE_RING,
                                 .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &pre_buf, &pre_alloc, &info) == VK_SUCCESS) pre_map = info.pMappedData;
        else pre_buf = VK_NULL_HANDLE, pre_failed = 1, rt_log("vk: tex_prestage: no host-visible ring; off");
    }
    int rec = 0;
    for (int i = 0; lim > 0 && (pre_buf || dry) && i < npre; i++) {
        Tex *e = pre_list[i];
        if (!e) continue;
        e->pre_mark = 0;
        VkImageAspectFlags aspect;
        PreReg pr[PRE_REGS];
        uint64_t total;
        if (!e->gpu_dirty || e->gpu_fence != fence || e->t.L.size > (uint64_t)lim || e->f.astc_bw ||
            e->pub.samples > 1 || !e->pub.image || !read_aspect(e, &aspect))
            continue;
        int n = pre_regions(e, pr, &total);
        if (n <= 0 || total > PRE_RING / 4) continue;
        const char *why;
        int plain = pre_plain(e, pr, n, &why);
        if (dry || !plain) {
            static int logged;
            if (logged < 300) {
                logged++;
                rt_log("vk: tex_prestage%s %#llx fmt %#x vk %d %ux%u levels %u layers %u, %d regions %llu bytes: %s",
                       dry ? " dry" : "", (unsigned long long)e->pub.addr, e->t.d.format, (int)e->pub.format, e->pub.width,
                       e->pub.height, e->pub.levels, e->pub.layers, n, (unsigned long long)total, plain ? "would stage" : why);
            }
            continue;
        }
        if (pre_head + total > PRE_RING) pre_head = 0, pre_lapn++;
        if (!rec) barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        for (int k = 0; k < n; k++) {
            const PreReg *p = &pr[k];
            VkBufferImageCopy rg[16];
            for (int z0 = 0; z0 < p->g.nz; z0 += 16) {
                int m = p->g.nz - z0 < 16 ? p->g.nz - z0 : 16;
                for (int j = 0; j < m; j++) {
                    int z = z0 + j;
                    rg[j] = (VkBufferImageCopy){
                        .bufferOffset = pre_head + p->off + p->slice * (size_t)z,
                        .imageSubresource = {aspect, p->g.level, p->g.is3d ? 0 : p->g.layer0 + (uint32_t)p->r.zoffset + (uint32_t)z, 1},
                        .imageOffset = {p->r.xoffset, p->r.yoffset, p->g.is3d ? p->r.zoffset + z : 0},
                        .imageExtent = {(uint32_t)p->r.width, (uint32_t)p->r.height, 1}};
                }
                vkCmdCopyImageToBuffer(cb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, pre_buf, (uint32_t)m, rg);
                rec++;
            }
        }
        e->pre_seq = e->gpu_seq, e->pre_fence = fence, e->pre_off = pre_head, e->pre_lap = pre_lapn, e->pre_size = total;
        pre_head += total;
        pre_staged++;
    }
    if (rec) barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                     VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    npre = 0;
    unlock();
    return rec;
}

// write_back, tex_lock: the staged copy when it is this render's, nothing recorded a later write of the image (wr_fence
// also counts storage-image writes, which leave gpu_seq alone) and its fence is done (then the GPU wrote the ring and no
// later copy was recorded into those bytes). 1: written back from it.
static int pre_take(Tex *e) {
    if (!e->pre_fence || !pre_map) return 0;
    int ok = e->pre_seq == e->gpu_seq && e->wr_fence <= e->pre_fence && e->pre_fence <= vk_sync_fence_done() &&
             (e->pre_lap == pre_lapn || (e->pre_lap + 1 == pre_lapn && e->pre_off >= pre_head));
    PreReg pr[PRE_REGS];
    uint64_t total;
    int n = ok ? pre_regions(e, pr, &total) : -1;
    if (n < 0 || total != e->pre_size) {
        pre_missed++;
        return 0;
    }
    vmaInvalidateAllocation(vk_ctx()->vma, pre_alloc, e->pre_off, e->pre_size);
    for (int k = 0; k < n; k++) {
        const PreReg *p = &pr[k];
        const Region *g = &p->g;
        uint32_t w = (uint32_t)p->r.width;
        uint8_t *buf = malloc(g->img * (size_t)g->nz);
        if (!buf) rt_fatal("vk: out of memory for a texture write-back");
        for (int z = 0; z < g->nz; z++) {
            const uint8_t *s = pre_map + e->pre_off + p->off + p->slice * (size_t)z;
            uint8_t *m = buf + (size_t)z * g->img;
            for (size_t y = 0; y < g->hb; y++) {
                if (p->ib) shrink_rgb(e->t.d.format, s + y * w * p->ib, m + y * g->row, w);
                else memcpy(m + y * g->row, s + y * g->wb, g->wb);
            }
        }
        storage_texels(&e->t, g, &p->r, buf, 1);
        free(buf);
    }
    e->gpu_dirty = 0;
    pre_used++;
    return 1;
}

// ---------------------------------------------------------------- samplers
static VkCompareOp compare_op(uint32_t f) {
    switch (f) {
    case NVN_COMPARE_FUNC_NEVER: return VK_COMPARE_OP_NEVER;
    case NVN_COMPARE_FUNC_LESS: return VK_COMPARE_OP_LESS;
    case NVN_COMPARE_FUNC_EQUAL: return VK_COMPARE_OP_EQUAL;
    case NVN_COMPARE_FUNC_LEQUAL: return VK_COMPARE_OP_LESS_OR_EQUAL;
    case NVN_COMPARE_FUNC_GREATER: return VK_COMPARE_OP_GREATER;
    case NVN_COMPARE_FUNC_NOTEQUAL: return VK_COMPARE_OP_NOT_EQUAL;
    case NVN_COMPARE_FUNC_GEQUAL: return VK_COMPARE_OP_GREATER_OR_EQUAL;
    default: return VK_COMPARE_OP_ALWAYS;
    }
}

static struct { int init, mirror_clamp_edge, minmax, border_no_format; } sf;

static void sampler_features(VkCtx *c) {
    if (sf.init) return;
    VkPhysicalDeviceCustomBorderColorFeaturesEXT cbf = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT};
    VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                            .pNext = c->has_custom_border_color ? &cbf : NULL};
    VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
    vkGetPhysicalDeviceFeatures2(c->phys, &f2);
    sf.mirror_clamp_edge = f12.samplerMirrorClampToEdge;
    sf.minmax = f12.samplerFilterMinmax;
    sf.border_no_format = c->has_custom_border_color && cbf.customBorderColors && cbf.customBorderColorWithoutFormat;
    sf.init = 1;
}

static VkSampler make_sampler(VkCtx *c, const samp_par *d) {
    sampler_features(c);
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = d->magf == NVN_MAG_FILTER_LINEAR ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    uint32_t mf = d->minf;
    si.minFilter = mf == 1 || mf == 3 || mf == 5 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    si.mipmapMode = mf == 4 || mf == 5 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    VkSamplerAddressMode modes[3];
    const uint8_t wrap[3] = {d->ws, d->wt, d->wr};
    int border = 0;
    for (int i = 0; i < 3; i++) {
        switch (wrap[i]) {
        case NVN_WRAP_MODE_CLAMP: case NVN_WRAP_MODE_CLAMP_TO_EDGE: modes[i] = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; break;
        case NVN_WRAP_MODE_MIRROR_CLAMP: case NVN_WRAP_MODE_MIRROR_CLAMP_TO_EDGE:
            modes[i] = sf.mirror_clamp_edge ? VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
            break;
        case NVN_WRAP_MODE_MIRROR_CLAMP_TO_BORDER: case NVN_WRAP_MODE_CLAMP_TO_BORDER:
            modes[i] = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            border = 1;
            break;
        case NVN_WRAP_MODE_MIRRORED_REPEAT: modes[i] = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT; break;
        default: modes[i] = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        }
    }
    si.addressModeU = modes[0], si.addressModeV = modes[1], si.addressModeW = modes[2];
    si.minLod = d->lod_min;
    si.maxLod = d->lod_max > 1000 ? VK_LOD_CLAMP_NONE : d->lod_max;
    if (mf <= NVN_MIN_FILTER_LINEAR) si.maxLod = si.minLod + 0.25f;   // not mipmapped: stay on the base level
    if (si.maxLod < si.minLod) si.maxLod = si.minLod;
    float lim = c->props.limits.maxSamplerLodBias;
    float bias = d->lod_bias;
    si.mipLodBias = fmaxf(-lim, fminf(lim, bias));
    if (d->aniso > 1) {
        si.anisotropyEnable = VK_TRUE;
        si.maxAnisotropy = fminf(d->aniso, c->props.limits.maxSamplerAnisotropy);
    }
    if (d->cmp_mode) {
        si.compareEnable = VK_TRUE;
        si.compareOp = compare_op(d->cmp_func);
    }
    VkSamplerCustomBorderColorCreateInfoEXT cb = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT};
    VkSamplerReductionModeCreateInfo rm = {.sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO};
    const void **tail = &si.pNext;
    if (border) {
        const float *b = d->border;
        if (b[0] == 0 && b[1] == 0 && b[2] == 0 && b[3] == 0) si.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        else if (b[0] == 0 && b[1] == 0 && b[2] == 0 && b[3] == 1) si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        else if (b[0] == 1 && b[1] == 1 && b[2] == 1 && b[3] == 1) si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        else if (sf.border_no_format) {
            memcpy(cb.customBorderColor.float32, b, 16);
            si.borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT;
            *tail = &cb;
            tail = &cb.pNext;
        } else si.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    }
    if (d->reduction && sf.minmax && !si.compareEnable) {
        rm.reductionMode = d->reduction == NVN_SAMPLER_REDUCTION_MIN ? VK_SAMPLER_REDUCTION_MODE_MIN : VK_SAMPLER_REDUCTION_MODE_MAX;
        *tail = &rm;
    }
    VkSampler s;
    VkResult r = vkCreateSampler(c->device, &si, NULL, &s);
    if (r != VK_SUCCESS) rt_fatal("vk: vkCreateSampler failed (%d)", (int)r);
    return s;
}

void vk_sampler_register(const dpool_ref *pool, int id, const samp_par *s) {
    static volatile int32_t n_unbacked;
    if (!pool || id < 0 || id >= (1 << 12) || (uint32_t)id >= pool->count) rt_fatal("vk: sampler slot %d out of pool range", id);
    lock();
    DescSlot *d = descriptor_slot(pool, id, 1u << 12, 1);
    if (!d) {
        log_limited(&n_unbacked, "sampler descriptor slot %d has no mapped backing: not registered", id);
        unlock();
        return;
    }
    descriptor_registered(d, DESC_SAMPLER);
    d->u.sampler.p = *s;
    d->u.sampler.used = 1;
    unlock();
}

// Caller holds the lock.
static VkSampler sampler_for(VkCtx *c, const samp_par *s) {
    samp_par k;
    memset(&k, 0, sizeof k);
    k = *s;
    VkSampler v = (VkSampler)nvn_hm_get(&samplers, &k);
    if (!v) {
        v = make_sampler(c, &k);
        nvn_hm_put(&samplers, &k, (void *)v);
    }
    return v;
}

VkSampler vk_sampler_for(const samp_par *s) {
    VkCtx *c = vk_ctx();
    if (!c || !s) return VK_NULL_HANDLE;
    lock();
    VkSampler v = sampler_for(c, s);
    unlock();
    return v;
}

// Samplers live until shutdown, so a slot keeps the one its parameters map to until it is registered again.
VkSampler vk_sampler_by_id_locked(const dpool_ref *pool, int id) {
    VkCtx *c = vk_ctx();
    if (!c) return VK_NULL_HANDLE;
    VkSampler v = VK_NULL_HANDLE;
    DescSlot *d = descriptor_slot(pool, id, 1u << 12, 0);
    if (d && d->kind == DESC_SAMPLER) {
        SampSlot *s = &d->u.sampler;
        if (!s->s) s->s = sampler_for(c, &s->p);
        v = s->s;
    }
    return v;
}

VkSampler vk_sampler_point_by_id_locked(const dpool_ref *pool, int id) {
    VkCtx *c = vk_ctx();
    DescSlot *d = descriptor_slot(pool, id, 1u << 12, 0);
    if (!c || !d || d->kind != DESC_SAMPLER) return VK_NULL_HANDLE;
    SampSlot *s = &d->u.sampler;
    if (!s->sp) {
        samp_par k = s->p;
        k.magf = NVN_MAG_FILTER_NEAREST;
        k.minf = k.minf == NVN_MIN_FILTER_LINEAR ? NVN_MIN_FILTER_NEAREST :
                 k.minf > NVN_MIN_FILTER_LINEAR ? NVN_MIN_FILTER_NEAREST_MIPMAP_NEAREST : k.minf;
        k.aniso = 1;
        s->sp = sampler_for(c, &k);
    }
    return s->sp;
}

VkSampler vk_sampler_by_id(const dpool_ref *pool, int id) {
    lock();
    VkSampler v = vk_sampler_by_id_locked(pool, id);
    unlock();
    return v;
}

int vk_sampler_params(const dpool_ref *pool, int id, samp_par *out) {
    lock();
    DescSlot *d = descriptor_slot(pool, id, 1u << 12, 0);
    int ok = d && d->kind == DESC_SAMPLER;
    if (ok) *out = d->u.sampler.p;
    unlock();
    return ok;
}

// A backing lifetime ended. Keep imported entries while another live imported alias still maps the same bytes.
// pool_lock is not held by the caller; taking it through descriptor_key while holding tex_lock preserves lock order.
void vk_texture_descriptors_retire(uint64_t addr, uint64_t size, uint64_t storage) {
    lock();
    VkDescKey *gone = malloc(sizeof *gone * (size_t)(descriptors.n + 1));
    if (!gone) rt_fatal("vk: out of memory retiring descriptors");
    uint32_t n = 0;
    for (uint32_t i = 0; i < descriptors.cap; i++) {
        if (!descriptors.vals[i]) continue;
        VkDescKey key;
        memcpy(&key, descriptors.keys + (size_t)i * descriptors.klen, sizeof key);
        if (key.storage != storage) continue;
        if (!storage) {
            if (key.offset >= addr ? key.offset - addr >= size : addr - key.offset >= NVN_DESCRIPTOR_BYTES) continue;
            VkDescKey live;
            if (vk_mem_descriptor_key(key.offset, NVN_DESCRIPTOR_BYTES, &live) && !memcmp(&live, &key, sizeof key)) continue;
        }
        gone[n++] = key;
    }
    for (uint32_t i = 0; i < n; i++) free(nvn_hm_del(&descriptors, &gone[i]));
    if (n) slots_gen++, samp_gen++;
    free(gone);
    unlock();
}

// ---------------------------------------------------------------- statistics and shutdown
void vk_texture_stats(uint64_t *images, uint64_t *uploads, uint64_t *reuploads) {
    lock();
    if (images) *images = by_key.n;
    if (uploads) *uploads = n_uploads;
    if (reuploads) *reuploads = n_reuploads;
    unlock();
}

void vk_texture_census(VkTexCensus *s) {
    VkCtx *c = vk_ctx();
    memset(s, 0, sizeof *s);
    lock();
    for (uint32_t i = 0; i < by_key.cap; i++) {
        const Tex *e = by_key.vals[i];
        if (!e) continue;
        uint64_t bytes = 0;
        if (e->alloc && c) {
            VmaAllocationInfo ai;
            vmaGetAllocationInfo(c->vma, e->alloc, &ai);
            bytes = ai.size;
        }
        if (e->released) s->released++, s->own_released += e->alloc != NULL, s->bytes_released += bytes;
        else s->live++, s->own_live += e->alloc != NULL, s->bytes_live += bytes;
    }
    unlock();
}

void vk_texture_shutdown(void) {
    VkCtx *c = vk_ctx();
    lock();
    Tex **list = malloc(sizeof *list * (size_t)(by_key.n + 1));
    int n = 0;
    for (uint32_t i = 0; i < by_key.cap; i++)
        if (by_key.vals[i]) list[n++] = by_key.vals[i];
    for (int i = 0; i < n; i++) destroy_tex(list[i]);
    free(list);
    for (uint32_t i = 0; i < samplers.cap; i++)
        if (samplers.vals[i] && c) vkDestroySampler(c->device, (VkSampler)samplers.vals[i], NULL);
    free(by_key.keys), free(by_key.vals), free(by_addr.keys), free(by_addr.vals), free(samplers.keys), free(samplers.vals);
    by_key = (nvn_hmap){.klen = sizeof(TexKey)};
    by_addr = (nvn_hmap){.klen = 8};
    samplers = (nvn_hmap){.klen = sizeof(samp_par)};
    for (uint32_t i = 0; i < descriptors.cap; i++) free(descriptors.vals[i]);
    free(descriptors.keys), free(descriptors.vals);
    descriptors = (nvn_hmap){.klen = sizeof(VkDescKey)};
    nreleased = 0;
    rel_head = rel_tail = NULL;
    rel_sum[0] = rel_sum[1] = 0;
    unlock();
}
