// The op-stream replay core of the Vulkan back end: walks a submission's command handles into its command buffer, keeps
// render targets and fixed-function state, opens and closes dynamic rendering passes, records clears and barriers. Draws and
// binds are in vk_exec_draw.c, copies, counters and events in vk_exec_xfer.c. Images stay in GENERAL layout (vk_texture.c);
// full barriers order passes, clears and what those record between them.
//
// The op loop (END / BEGIN / JUMP / CALL, which ops are pure state, the state defaults, render targets kept until they
// change, clears skipped while rendering is disabled) follows MacNativeGames
// games/totk/third_party/d2r_gfx/gfx/nvn/nvn_mtl.m:10588-10700, 7409-7460, 5449-5457
// (https://github.com/ly05010419/MacNativeGames @ b75be8cfa704df3701e86130c8ced268e79d6b11, MIT, Copyright (c) 2026
// jhaizhou). Pass begin and end (GENERAL-layout attachments, LOAD/STORE, render area from the smallest attachment, a
// full barrier before each pass), the scissor clip and attachment clears follow LAS Recomp
// runtime/nvn/vulkan/exec.c:855-1149, 3553-3778 (https://github.com/peiikai/LASRecomp @
// da35b83a9611d046a9a681aae668d7085954d37f, MIT, Copyright (c) 2026 Peiikai).
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_tex.h"
#include "platform/platform.h"
#include "tex.h"
#include "vk.h"
#include "vk_exec.h"

#define WEAK __attribute__((weak))
#define CALL_DEPTH 8

typedef struct {
    VkImage image;
    VkFormat format;
    VkImageViewType type;
    VkImageAspectFlags aspect;
    uint32_t level, base, layers;
    VkImageView view;
} ViewEnt;

static VkExec X;
static int x_ready;
static _Thread_local int x_here;   // this thread is inside vk_exec_record (vk_exec_inline_begin)

// ---- OOT_TRACE=gfx frame counters

VkExecFrame vk_exec_frame;
static struct { const uint8_t *mem; uint32_t w, h, fmt, n; } fw[64];   // textures written this frame
static int nfw;

uint64_t vk_exec_recording_fence(void) { return x_here ? X.fence : 0; }

void vk_exec_frame_written(const tex_t *t) {
    if (!(rt_trace & RT_TRACE_GFX) || !t) return;
    for (int i = 0; i < nfw; i++)
        if (fw[i].mem == t->mem) { fw[i].n++; return; }
    if (nfw == 64) return;
    fw[nfw].mem = t->mem, fw[nfw].w = t->d.width, fw[nfw].h = t->d.height, fw[nfw].fmt = (uint32_t)t->d.format;
    fw[nfw++].n = 1;
}

void vk_exec_frame_log(const tex_t *p, uint64_t frame) {
    if (!(rt_trace & RT_TRACE_GFX)) return;
    VkExecFrame *f = &vk_exec_frame;
    int hit = 0;
    char list[512] = "";
    size_t o = 0;
    for (int i = 0; i < nfw; i++) {
        if (p && fw[i].mem == p->mem) hit = 1;
        if (o < sizeof list - 48)
            o += (size_t)snprintf(list + o, sizeof list - o, " %p:%ux%u/%#x*%u", (const void *)fw[i].mem, fw[i].w,
                                  fw[i].h, fw[i].fmt, fw[i].n);
    }
    rt_log("frame %llu: submits %llu passes %llu draws %llu/%llu clears %llu copies %llu dispatches %llu; present %p "
           "%ux%u/%#x written %d; feedback draws %llu barriers %llu restarts %llu layouts %llu; targets%s",
           (unsigned long long)frame, (unsigned long long)f->submits,
           (unsigned long long)f->passes, (unsigned long long)f->draws, (unsigned long long)f->draws_seen,
           (unsigned long long)f->clears, (unsigned long long)f->copies, (unsigned long long)f->dispatches,
           p ? (const void *)p->mem : NULL, p ? p->d.width : 0, p ? p->d.height : 0, p ? (uint32_t)p->d.format : 0,
           hit, (unsigned long long)f->fb_draws, (unsigned long long)f->fb_barriers, (unsigned long long)f->fb_restarts,
           (unsigned long long)f->fb_layouts, list);
    memset(f, 0, sizeof *f);
    nfw = 0;
}

WEAK void vk_exec_op_draw(VkExec *ex, uint32_t op, const void *payload) {
    (void)ex, (void)payload;
    rt_fatal("vk: op %s is not replayed yet", nvn_op_name(op));
}

WEAK void vk_exec_op_xfer(VkExec *ex, uint32_t op, const void *payload) {
    (void)ex, (void)payload;
    rt_fatal("vk: op %s is not replayed yet", nvn_op_name(op));
}

enum { W_UNRENDERABLE, W_LEVEL, W_LAYER, W_VIEW_FORMAT, W_BLOCK_VIEW, W_BLOCK_LAYERS, W_CLEAR_TARGET, W_CLEAR_MASK, W_COUNT };

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

// NVN defaults of the replay state.
static void state_reset(VkExec *ex) {
    memset(ex, 0, sizeof *ex);
    ex->st_vmask[0] = ex->st_vmask[1] = ex->st_wmask[0] = ex->st_wmask[1] = 0xff;
    ex->depth_range[1] = 1.0f;
    vk_pipe_state_defaults(&ex->pipe);
    ex->barrier_at = -1;
}

// ---- render targets

static uint32_t level_extent(uint32_t v, uint32_t level) {
    v >>= level;
    return v ? v : 1;
}

// A view format other than the texture's is used when both map directly and have the same texel size (colour images are
// created MUTABLE_FORMAT); for a BC texture with block_views (blocks), an uncompressed format whose texel is one block.
static int view_format(const tex_t *t, const view_t *v, int blocks, VkFormat *out) {
    TexVkFormat a, b;
    int bw, bh, vbw, vbh;
    if (!tex_vk_format(t->d.format, &a) || a.vk != a.ideal || !tex_vk_format(v->format, &b) || b.vk != b.ideal ||
        b.depth || b.stencil)
        return 0;
    int bpb = nvn_fmt_bpb(t->d.format, &bw, &bh), vbpb = nvn_fmt_bpb(v->format, &vbw, &vbh);
    if (!bpb || bpb != vbpb || (!blocks && (bw != 1 || bh != 1)) || vbw != 1 || vbh != 1) return 0;
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(vk_ctx()->phys, b.vk, &fp);
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) return 0;
    *out = b.vk;
    return 1;
}

// ---- 3D BC render targets: see docs/how-it-works.md, "3D BC render targets".
// Draws through an uncompressed block view of a 3D BC image go to a per-level stand-in, copied back into the level at pass end.
typedef struct { VkImage bc, img; VmaAllocation alloc; VkFormat fmt; uint32_t level, w, h, d; } BcStandIn;
#define BC_STANDINS 64   // one per level drawn: 6 volumes of up to 7 levels
static BcStandIn bcs[BC_STANDINS];
static int nbcs;
static PlatMutex bcs_lock = PLAT_MUTEX_INIT;   // the replay adds and reads; destroy_tex (any thread) removes

static void image_barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, uint32_t layers) {
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
                                .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                .oldLayout = from, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(cb, &di);
}

static void memory_barrier(VkCommandBuffer cb, VkPipelineStageFlags2 src, VkAccessFlags2 sa, VkPipelineStageFlags2 dst,
                           VkAccessFlags2 da) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = src, .srcAccessMask = sa,
                           .dstStageMask = dst, .dstAccessMask = da};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

// The stand-in of level level of a 3D BC image for draws through view format fmt, made on first use; 0 when none can
// be made (logged). Replay thread.
static int bc_standin(const VkTexImage *img, uint32_t level, VkFormat fmt, BcStandIn *out) {
    static int warned;
    plat_mutex_lock(&bcs_lock);
    int full = nbcs == BC_STANDINS;
    for (int i = 0; i < nbcs; i++)
        if (bcs[i].bc == img->image && bcs[i].level == level && bcs[i].fmt == fmt) {
            *out = bcs[i];
            plat_mutex_unlock(&bcs_lock);
            return 1;
        }
    plat_mutex_unlock(&bcs_lock);
    if (full) {
        if (!warned++) rt_log("vk: more than %d 3D BC render targets: the others are left out", BC_STANDINS);
        return 0;
    }
    uint32_t lw = level_extent(img->width, level), lh = level_extent(img->height, level);
    BcStandIn s = {img->image, VK_NULL_HANDLE, NULL, fmt, level, (lw + 3) / 4, (lh + 3) / 4,
                   level_extent(img->depth, level)};
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
                            .extent = {s.w, s.h, 1}, .mipLevels = 1, .arrayLayers = s.d, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                     VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(vk_ctx()->vma, &ci, &ai, &s.img, &s.alloc, NULL) != VK_SUCCESS) {
        if (!warned++) rt_log("vk: stand-in for a 3D BC render target could not be created: left out");
        return 0;
    }
    // In the upload batch, which goes to the queue before this part: GENERAL, then the texture's blocks.
    VkCommandBuffer cb = vk_upload_begin();
    image_barrier(cb, s.img, VK_IMAGE_LAYOUT_UNDEFINED, s.d);
    memory_barrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                   VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkImageCopy rg = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1},
                      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, s.d},
                      .extent = {lw, lh, s.d}};   // texels of the compressed source
    vkCmdCopyImage(cb, img->image, VK_IMAGE_LAYOUT_GENERAL, s.img, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
    memory_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                   VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    vk_upload_end();
    plat_mutex_lock(&bcs_lock);
    bcs[nbcs++] = s;
    plat_mutex_unlock(&bcs_lock);
    rt_log("vk: 3D BC render target %ux%ux%u level %u drawn through a %ux%u x %u stand-in (format %d)", img->width,
           img->height, img->depth, level, s.w, s.h, s.d, (int)fmt);
    *out = s;
    return 1;
}

// vk_exec_draw.c vk_exec_image_gone: the stand-in of a BC image that goes, freed once the GPU is done with it.
void vk_exec_bc_gone(VkImage image) {
    plat_mutex_lock(&bcs_lock);
    for (int i = 0; i < nbcs; i++)
        if (bcs[i].bc == image) {
            vk_defer_image(bcs[i].img, bcs[i].alloc);
            bcs[i--] = bcs[--nbcs];
        }
    plat_mutex_unlock(&bcs_lock);
}

// At a pass end: the planes the pass drew into a stand-in go to its texture.
static void bc_copy_back(VkExec *ex, const VkExecTarget *t) {
    plat_mutex_lock(&bcs_lock);
    for (int i = 0; i < nbcs; i++) {
        if (bcs[i].img != t->image) continue;
        memory_barrier(ex->cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                       VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkImageCopy rg = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, t->base_layer, t->layers},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, bcs[i].level, 0, 1},
                          .dstOffset = {0, 0, (int32_t)t->base_layer},
                          .extent = {bcs[i].w, bcs[i].h, t->layers}};   // texels of the uncompressed source
        vkCmdCopyImage(ex->cb, t->image, VK_IMAGE_LAYOUT_GENERAL, bcs[i].bc, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
        memory_barrier(ex->cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
        ex->recorded++;
        break;
    }
    plat_mutex_unlock(&bcs_lock);
}

// A draw into a 3D BC image without block views through an uncompressed view of its blocks: the target is the
// image's stand-in (bc_standin). 1 when o is filled (target or left out), 0 for any other target.
static int bc_target(const tex_t *t, const view_t *v, int has_v, const VkTexImage *img, VkExecTarget *o) {
    int bw, bh;
    if (img->type != VK_IMAGE_TYPE_3D || img->block_views || !nvn_fmt_bpb(t->d.format, &bw, &bh) || bw != 4 || bh != 4)
        return 0;
    VkFormat fmt;
    BcStandIn s;
    uint32_t level = has_v && v->set_levels ? v->base_level : 0;
    if (level >= img->levels) {
        warn_once(W_LEVEL, "render target level %u of a %u-level texture: left out", level, img->levels);
        return 1;
    }
    if (!has_v || !v->set_format || !view_format(t, v, 1, &fmt) || !bc_standin(img, level, fmt, &s)) {
        warn_once(W_BLOCK_VIEW, "render target format %#x (%ux%u) through view format %#x: no block view, left out",
                  t->d.format, t->d.width, t->d.height, has_v && v->set_format ? v->format : 0);
        return 1;
    }
    uint32_t base = 0, layers = s.d;
    if (v->set_layers) {
        base = v->min_layer;
        layers = v->nlayers ? v->nlayers : (base < s.d ? s.d - base : 0);
    }
    if (base >= s.d || !layers) return 1;
    if (base + layers > s.d) layers = s.d - base;
    o->image = s.img;   // not img->image: view_of makes this command buffer's view of the stand-in
    o->img = img;
    o->format = fmt;
    o->view_type = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    o->aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    o->level = 0, o->base_layer = base, o->layers = layers;
    o->width = s.w, o->height = s.h, o->samples = 1;
    o->nvn_format = v->format;
    return 1;
}

static void resolve_target(const tex_t *t, const view_t *v, int has_v, int depth_slot, VkExecTarget *o) {
    memset(o, 0, sizeof *o);
    o->tex = t;
    if (!t) return;
    const VkTexImage *img = vk_texture_get(t);
    if (img && !depth_slot && bc_target(t, v, has_v, img, o)) return;   // a 3D BC image: its stand-in
    VkImageUsageFlags need = depth_slot ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    int is3d = img && img->type == VK_IMAGE_TYPE_3D && !depth_slot;   // slices drawn through 2D views (2D_ARRAY_COMPATIBLE)
    if (!img || !(img->usage & need) || (img->type != VK_IMAGE_TYPE_2D && !is3d)) {
        warn_once(W_UNRENDERABLE, "render target format %#x target %d (%ux%u) is not renderable here: left out",
                  t->d.format, t->d.target, t->d.width, t->d.height);
        return;
    }
    uint32_t level = has_v && v->set_levels ? v->base_level : 0;
    if (level >= img->levels) {
        warn_once(W_LEVEL, "render target level %u of a %u-level texture: left out", level, img->levels);
        return;
    }
    // A 3D level's depth planes take the place of layers (Vulkan views them as the array layers of a 2D array view).
    uint32_t nlay = is3d ? level_extent(img->depth, level) : img->layers;
    uint32_t base = 0, layers = nlay;
    if (has_v && v->set_layers) {
        base = v->min_layer;
        layers = v->nlayers ? v->nlayers : (base < nlay ? nlay - base : 0);
    }
    if (base >= nlay || !layers) {
        warn_once(W_LAYER, "render target layer %u of a %u-layer texture: left out", base, nlay);
        return;
    }
    if (base + layers > nlay) layers = nlay - base;
    VkFormat fmt = img->format;
    int32_t nvn_fmt = t->d.format;
    int blocks = !depth_slot && img->block_views;
    if (blocks) {   // a BC texture takes draws only through an uncompressed view of its blocks
        if (!has_v || !v->set_format || !view_format(t, v, 1, &fmt)) {
            warn_once(W_BLOCK_VIEW, "render target format %#x (%ux%u) through view format %#x: no block view, left out",
                      t->d.format, t->d.width, t->d.height, has_v && v->set_format ? v->format : 0);
            return;
        }
        nvn_fmt = v->format;
        if (layers > 1) {   // an uncompressed view of a BC image has one layer (no maintenance6 here)
            warn_once(W_BLOCK_LAYERS, "render target format %#x: block view of %u layers, first layer only", t->d.format,
                      layers);
            layers = 1;
        }
    } else if (!depth_slot && has_v && v->set_format && v->format != t->d.format) {
        if (view_format(t, v, 0, &fmt)) nvn_fmt = v->format;
        else warn_once(W_VIEW_FORMAT, "render target view format %#x of a format %#x texture: texture format used",
                       v->format, t->d.format);
    }
    o->image = img->image;
    o->img = img;
    o->format = fmt;
    o->view_type = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    o->aspect = depth_slot ? img->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)
                           : VK_IMAGE_ASPECT_COLOR_BIT;
    o->level = level, o->base_layer = base, o->layers = layers;
    o->is3d = (uint8_t)is3d;
    o->blocks = (uint8_t)blocks;
    o->width = level_extent(img->width, level), o->height = level_extent(img->height, level);
    if (blocks) o->width = (o->width + 3) / 4, o->height = (o->height + 3) / 4;   // the view's extent: 4x4 blocks
    o->samples = img->samples ? img->samples : 1;
    o->nvn_format = nvn_fmt;
    if (!o->aspect) o->image = VK_NULL_HANDLE;
}

// The size of a pass without attachments: the far corner of viewport 0, else of the scissor, 1..16384. Sizing by the
// viewport's width alone puts an offset viewport outside the scissor.
void vk_exec_noatt_dims(const VkExec *ex, uint32_t *w, uint32_t *h) {
    double x1 = 0, y1 = 0;
    if (ex->has_vp) {
        x1 = ceil(fmax(ex->vp[0][0], ex->vp[0][0] + ex->vp[0][2]));
        y1 = ceil(fmax(ex->vp[0][1], ex->vp[0][1] + ex->vp[0][3]));
    }
    if ((x1 < 1 || y1 < 1) && ex->has_scissor) x1 = (double)ex->scissor.x + ex->scissor.w, y1 = (double)ex->scissor.y + ex->scissor.h;
    *w = x1 < 1 ? 1 : x1 > 16384 ? 16384 : (uint32_t)x1;
    *h = y1 < 1 ? 1 : y1 > 16384 ? 16384 : (uint32_t)y1;
}

// Resolves ex->rt to this command buffer's images and derives the pass parameters and the pipeline's target formats.
static void resolve_rt(VkExec *ex) {
    if (ex->rt_resolved) return;
    ex->rt_resolved = 1;
    const p_set_rt *r = &ex->rt;
    int n = r->n < 0 ? 0 : r->n > 8 ? 8 : r->n;
    uint32_t w = UINT32_MAX, h = UINT32_MAX, layers = UINT32_MAX, samples = 0;
    for (int i = 0; i < 8; i++) {
        VkExecTarget *t = &ex->color[i];
        resolve_target(i < n ? r->color[i] : NULL, &r->cview[i], i < n && r->has_cview[i], 0, t);
        ex->pass_color[i] = t->image ? t->format : VK_FORMAT_UNDEFINED;
        ex->pipe.color_fmt[i] = t->image ? t->nvn_format : 0;
        if (!t->image) continue;
        if (t->width < w) w = t->width;
        if (t->height < h) h = t->height;
        if (t->layers < layers) layers = t->layers;
        if (!samples) samples = t->samples;
    }
    VkExecTarget *d = &ex->depth;
    resolve_target(r->depth, &r->dview, r->has_dview, 1, d);
    ex->pass_depth = d->image && (d->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) ? d->format : VK_FORMAT_UNDEFINED;
    ex->pass_stencil = d->image && (d->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) ? d->format : VK_FORMAT_UNDEFINED;
    ex->pipe.depth_fmt = d->image ? d->nvn_format : 0;
    if (d->image) {
        if (d->width < w) w = d->width;
        if (d->height < h) h = d->height;
        if (d->layers < layers) layers = d->layers;
        if (!samples) samples = d->samples;
    }
    ex->pass_noatt = w == UINT32_MAX;
    if (ex->pass_noatt) {
        vk_exec_noatt_dims(ex, &w, &h);
        layers = 1;
        static int logged;
        if (logged < 8 && ex->has_vp && (ex->vp[0][0] != 0 || ex->vp[0][1] != 0 || ex->vp[0][2] != w || ex->vp[0][3] != h))
            logged++, rt_log("vk_exec: pass without attachments %ux%u for viewport %g,%g %gx%g", w, h, ex->vp[0][0],
                             ex->vp[0][1], ex->vp[0][2], ex->vp[0][3]);
    }
    ex->pass_w = w, ex->pass_h = h, ex->pass_layers = layers;
    ex->pass_ncolor = (uint32_t)n;
    ex->pass_samples = samples ? samples : 1;
    ex->pipe.samples = (uint8_t)ex->pass_samples;
}

// The attachment view of a target. A registered texture's view lives as long as its image (vk_exec_target_view); any
// other image's view is made for this command buffer and freed at its fence.
static VkImageView view_of(VkExec *ex, const VkExecTarget *t) {
    if (t->img && t->img->image == t->image)
        return vk_exec_target_view(t->img, t->format, t->view_type, t->aspect, t->level, t->base_layer, t->layers, t->blocks);
    ViewEnt *v = ex->views;
    for (int i = 0; i < ex->nviews; i++)
        if (v[i].image == t->image && v[i].format == t->format && v[i].type == t->view_type && v[i].aspect == t->aspect &&
            v[i].level == t->level && v[i].base == t->base_layer && v[i].layers == t->layers)
            return v[i].view;
    VkImageViewCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = t->image,
                                .viewType = t->view_type, .format = t->format,
                                .subresourceRange = {t->aspect, t->level, 1, t->base_layer, t->layers}};
    VkImageViewUsageCreateInfo use = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
                                      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
    if (t->blocks) ci.pNext = &use;   // not the BC image's sampling and transfer usages
    VkImageView view;
    if (vkCreateImageView(vk_ctx()->device, &ci, NULL, &view) != VK_SUCCESS)
        rt_fatal("vk: render target view (format %d, %ux%u) could not be created", (int)t->format, t->width, t->height);
    if (ex->nviews == ex->cap_views) {
        ex->cap_views = ex->cap_views ? ex->cap_views * 2 : 16;
        ex->views = realloc(ex->views, sizeof(ViewEnt) * (size_t)ex->cap_views);
        if (!ex->views) rt_fatal("vk: out of memory");
        v = ex->views;
    }
    v[ex->nviews++] = (ViewEnt){t->image, t->format, t->view_type, t->aspect, t->level, t->base_layer, t->layers, view};
    return view;
}

typedef struct { int n; VkImageView v[]; } ViewList;

static void free_views(void *arg) {   // completion thread, once the submission that used them is done
    ViewList *l = arg;
    VkDevice dev = vk_ctx()->device;
    for (int i = 0; i < l->n; i++) vkDestroyImageView(dev, l->v[i], NULL);
    free(l);
}

// ---- target history: the frame each target subresource was last rendered in, and the one before. prepare_draw makes a
// one-shot draw (a volume or LUT generated once) wait for its pipeline instead of skipping it. Direct-mapped: a collision
// makes a draw wait, never skip.
typedef struct { VkImage image; uint32_t level, layer; uint64_t last, prev; } TargetFrames;
static TargetFrames tfr[1024];

static TargetFrames *tfr_slot(const VkExecTarget *t) {
    uint64_t h = ((uint64_t)(uintptr_t)t->image ^ (uint64_t)t->level << 56 ^ (uint64_t)t->base_layer << 40) *
                 0x9E3779B97F4A7C15ull;
    return &tfr[h >> 54];
}

static int tfr_match(const TargetFrames *e, const VkExecTarget *t) {
    return e->image == t->image && e->level == t->level && e->layer == t->base_layer;
}

static void tfr_note(const VkExecTarget *t, uint64_t stamp) {
    TargetFrames *e = tfr_slot(t);
    if (!tfr_match(e, t)) *e = (TargetFrames){t->image, t->level, t->base_layer, 0, 0};
    if (e->last != stamp) e->prev = e->last, e->last = stamp;
}

static int tfr_recent(const VkExecTarget *t, uint64_t stamp) {
    const TargetFrames *e = tfr_slot(t);
    return tfr_match(e, t) && (e->last == stamp ? e->prev : e->last) == stamp - 1;
}

int vk_exec_targets_recent(VkExec *ex, int any) {
    resolve_rt(ex);
    uint64_t stamp = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    int n = 0, recent = 0;
    for (int i = 0; i < 9; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        if (!t->image) continue;
        n++;
        if (tfr_recent(t, stamp)) recent++;
        else if (!any) return 0;
    }
    return any ? recent > 0 : n > 0;
}

// One line naming each target of the current pass: format, extent, layer and the two frames it was last rendered in
// (prepare_draw's pipeline wait log).
void vk_exec_targets_describe(VkExec *ex, char *buf, size_t cap) {
    resolve_rt(ex);
    size_t n = 0;
    buf[0] = 0;
    for (int i = 0; i < 9 && n < cap; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        if (!t->image) continue;
        const TargetFrames *e = tfr_slot(t);
        int m = tfr_match(e, t);
        int w = snprintf(buf + n, cap - n, "%s%c%d fmt %d %ux%u L%u seen %llu/%llu", n ? "; " : "", i < 8 ? 'c' : 'd',
                         i < 8 ? i : 0, (int)t->format, t->width, t->height, t->base_layer,
                         (unsigned long long)(m ? e->last : 0), (unsigned long long)(m ? e->prev : 0));
        if (w < 0) break;
        n += (size_t)w;
    }
    if (!n) snprintf(buf, cap, "no targets");
}

// ---- passes and barriers

static int has_image(const VkImage *list, int n, VkImage img) {
    for (int i = 0; i < n; i++)
        if (list[i] == img) return 1;
    return 0;
}

static VkImageMemoryBarrier2 layout_barrier(VkImage img, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to) {
    return (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                   .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                                   .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                   .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                   .oldLayout = from, .newLayout = to, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img,
                                   .subresourceRange = {aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
}

// Full barrier. The images in the feedback layout become want[] (pass begin) or none (every other barrier): the others
// go back to GENERAL and the new ones leave it, in the same command.
static void barrier_to(VkExec *ex, const VkImage *want, const VkImageAspectFlags *wasp, int nwant) {
    vk_exec_pass_end(ex);
    VkImageMemoryBarrier2 ib[18];
    uint32_t n = 0;
    for (int i = 0; i < ex->nfbl; i++)
        if (!has_image(want, nwant, ex->fbl_img[i]))
            ib[n++] = layout_barrier(ex->fbl_img[i], ex->fbl_aspect[i], VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT,
                                     VK_IMAGE_LAYOUT_GENERAL);
    for (int i = 0; i < nwant; i++)
        if (!has_image(ex->fbl_img, ex->nfbl, want[i]))
            ib[n++] = layout_barrier(want[i], wasp[i], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT);
    for (int i = 0; i < nwant; i++) ex->fbl_img[i] = want[i], ex->fbl_aspect[i] = wasp[i];
    ex->nfbl = nwant;
    if (ex->barrier_at == ex->recorded && !n) return;
    vk_exec_frame.fb_layouts += n;
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb,
                           .imageMemoryBarrierCount = n, .pImageMemoryBarriers = ib};
    vkCmdPipelineBarrier2(ex->cb, &di);
    ex->barrier_at = ++ex->recorded;
}

void vk_exec_barrier(VkExec *ex) { barrier_to(ex, NULL, NULL, 0); }

void vk_exec_pass_begin(VkExec *ex) {
    if (ex->in_pass) return;
    resolve_rt(ex);
    // Learned targets (VkTexImage.fb_learned) are attached in the feedback layout
    VkImage want[9];
    VkImageAspectFlags wasp[9];
    int nwant = 0;
    uint32_t fbl = 0;
    for (uint32_t i = 0; i < 9; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        if ((i < 8 && i >= ex->pass_ncolor) || !t->image || !t->img || !t->img->fb_learned) continue;
        fbl |= 1u << i;
        if (has_image(want, nwant, t->image)) continue;
        want[nwant] = t->image, wasp[nwant++] = i < 8 ? VK_IMAGE_ASPECT_COLOR_BIT : t->img->aspect;
    }
    barrier_to(ex, want, wasp, nwant);   // passes, clears, copies and dispatches see each other's results
    ex->fbl_mask = fbl;
    ex->fb_dirty = 0;
    ex->pass_serial++;
    VkRenderingAttachmentInfo ca[8], da = {0}, sa = {0};
    for (uint32_t i = 0; i < ex->pass_ncolor; i++) {
        const VkExecTarget *t = &ex->color[i];
        ca[i] = (VkRenderingAttachmentInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        if (!t->image) continue;
        ca[i].imageView = view_of(ex, t);
        ca[i].imageLayout = fbl >> i & 1 ? VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT : VK_IMAGE_LAYOUT_GENERAL;
        ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        ca[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;    }
    const VkExecTarget *d = &ex->depth;
    if (d->image) {
        VkRenderingAttachmentInfo a = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = view_of(ex, d),
                                       .imageLayout = fbl >> 8 & 1 ? VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
                                                                   : VK_IMAGE_LAYOUT_GENERAL,
                                       .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
        if (d->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) da = a;
        if (d->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) sa = a;
    }
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {ex->pass_w, ex->pass_h}},
                          .layerCount = ex->pass_layers, .colorAttachmentCount = ex->pass_ncolor,
                          .pColorAttachments = ca,
                          .pDepthAttachment = ex->pass_depth ? &da : NULL,
                          .pStencilAttachment = ex->pass_stencil ? &sa : NULL};
    vkCmdBeginRendering(ex->cb, &ri);
    vk_exec_frame.passes++;
    ex->recorded++;
    ex->in_pass = 1;
}

void vk_exec_pass_end(VkExec *ex) {
    if (!ex->in_pass) return;
    vk_exec_oq_pass_end(ex);
    vkCmdEndRendering(ex->cb);
    ex->recorded++;
    ex->in_pass = 0;
    for (int i = 0; i < ex->pass_ncolor && i < 8; i++)   // stand-ins of 3D BC targets go to their textures
        if (ex->color[i].image && ex->color[i].img && ex->color[i].img->image != ex->color[i].image)
            bc_copy_back(ex, &ex->color[i]);
    uint64_t stamp = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    for (int i = 0; i < 8; i++)
        if (ex->color[i].image) vk_texture_gpu_written(ex->color[i].tex), tfr_note(&ex->color[i], stamp);
    if (ex->depth.image) vk_texture_gpu_written(ex->depth.tex), tfr_note(&ex->depth, stamp);
}

VkCommandBuffer vk_exec_inline_begin(int *reopen) {
    VkExec *ex = &X;
    *reopen = 0;
    if (!x_here || !ex->cb) return VK_NULL_HANDLE;
    *reopen = ex->in_pass;
    vk_exec_barrier(ex);   // ends the pass; feedback-layout targets go back to GENERAL
    return ex->cb;
}

void vk_exec_inline_end(int reopen) {
    X.recorded++;
    if (reopen) vk_exec_pass_begin(&X);
}

typedef struct { VkBuffer b; VmaAllocation a; } InlineBuf;
static void free_inline_buf(void *arg) {   // completion thread
    InlineBuf *t = arg;
    vmaDestroyBuffer(vk_ctx()->vma, t->b, t->a);
    free(t);
}

VkBuffer vk_exec_inline_buffer(VkDeviceSize size) {
    InlineBuf *t = calloc(1, sizeof *t);
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                             .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (!t || vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &t->b, &t->a, NULL) != VK_SUCCESS) { free(t); return VK_NULL_HANDLE; }
    vk_completion_add(X.fence, free_inline_buf, t);
    return t->b;
}

VkRect2D vk_exec_scissor(const VkExec *ex) {
    int64_t w = ex->pass_w, h = ex->pass_h;
    if (!ex->has_scissor) return (VkRect2D){{0, 0}, {(uint32_t)w, (uint32_t)h}};
    int64_t x0 = ex->scissor.x, y0 = ex->scissor.y;
    int64_t x1 = x0 + ex->scissor.w, y1 = y0 + ex->scissor.h;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > w) x1 = w;
    if (y1 > h) y1 = h;
    if (x1 <= x0 || y1 <= y0) return (VkRect2D){{0, 0}, {0, 0}};
    return (VkRect2D){{(int32_t)x0, (int32_t)y0}, {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}};
}

// ---- clears

// Outside a pass, a clear of a whole target (no narrower scissor, the target as large as the render area) clears the
// image directly instead of opening a pass for it.
static int whole_target(const VkExec *ex, const VkExecTarget *t) {
    // vkCmdClear*Image addresses a 3D level whole, not one depth plane: slices of a 3D target clear in a pass; it
    // cannot clear a compressed image: views of BC blocks clear in a pass too
    if (ex->in_pass || t->is3d || t->blocks || t->width != ex->pass_w || t->height != ex->pass_h) return 0;
    if (!ex->has_scissor) return 1;
    const p_scissor *s = &ex->scissor;
    return s->x <= 0 && s->y <= 0 && (int64_t)s->x + s->w >= t->width && (int64_t)s->y + s->h >= t->height;
}

static void clear_attachment(VkExec *ex, const VkClearAttachment *a) {
    vk_exec_pass_begin(ex);
    VkClearRect cr = {vk_exec_scissor(ex), 0, ex->pass_layers};
    if (!cr.rect.extent.width || !cr.rect.extent.height) return;
    vkCmdClearAttachments(ex->cb, 1, a, 1, &cr);
    ex->recorded++;
    ex->fb_dirty |= a->aspectMask & VK_IMAGE_ASPECT_COLOR_BIT ? 1u << a->colorAttachment : 0x100u;
}

static void clear_color(VkExec *ex, const p_clear_color *c) {
    vk_exec_frame.clears++;
    if (ex->render_disabled) return;
    resolve_rt(ex);
    int i = c->index;
    if (i < 0 || i >= (int)ex->pass_ncolor || !ex->color[i].image) {
        warn_once(W_CLEAR_TARGET, "colour clear of target %d, which is not bound or not renderable: skipped", i);
        return;
    }
    uint32_t mask = (uint32_t)c->mask & 0xf;
    if (!mask) return;
    if (mask != 0xf) {   // vkCmdClearAttachments ignores write masks; a masked clear needs a draw (vk_exec_draw.c)
        warn_once(W_CLEAR_MASK, "colour clear with channel mask %#x is not replayed: skipped", mask);
        return;
    }
    const VkExecTarget *t = &ex->color[i];
    VkClearColorValue cv;
    memcpy(cv.float32, c->rgba, sizeof cv.float32);   // integer targets: the payload's bits as the front end stored them
    if (whole_target(ex, t)) {
        vk_exec_barrier(ex);
        VkImageSubresourceRange r = {VK_IMAGE_ASPECT_COLOR_BIT, t->level, 1, t->base_layer, t->layers};
        vkCmdClearColorImage(ex->cb, t->image, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &r);
        ex->recorded++;
        vk_texture_gpu_written(t->tex);
        return;
    }
    VkClearAttachment a = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .colorAttachment = (uint32_t)i};
    a.clearValue.color = cv;
    clear_attachment(ex, &a);
}

static void clear_ds(VkExec *ex, const p_clear_ds *c) {
    if (ex->render_disabled) return;
    resolve_rt(ex);
    const VkExecTarget *t = &ex->depth;
    if (!t->image) return;
    VkImageAspectFlags asp = 0;
    if (c->depth_mask && (t->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) asp |= VK_IMAGE_ASPECT_DEPTH_BIT;
    uint32_t sm = (uint32_t)c->stencil_mask & 0xff;
    if (sm && (t->aspect & VK_IMAGE_ASPECT_STENCIL_BIT)) {
        if (sm == 0xff) asp |= VK_IMAGE_ASPECT_STENCIL_BIT;
        else {
            VkRect2D scissor = vk_exec_scissor(ex);
            if (scissor.extent.width && scissor.extent.height) {
                // ClearAttachments ignores the write mask: draw into each layer with ALWAYS/REPLACE in a separate pass, keeping
                // the depth plane and stencil bits outside sm, and the clear out of the game's occlusion queries.
                vk_exec_barrier(ex);
                VkExecTarget layer = *t;
                layer.layers = 1;
                layer.view_type = VK_IMAGE_VIEW_TYPE_2D;
                for (uint32_t i = 0; i < t->layers; i++) {
                    layer.base_layer = t->base_layer + i;
                    vk_clear_stencil_masked(ex->cb, t->format, t->samples, t->aspect, view_of(ex, &layer),
                                             ex->pass_w, ex->pass_h, scissor, sm, (uint32_t)c->stencil);
                    ex->recorded += 9;
                    vk_exec_frame.passes++;
                }
                vk_texture_gpu_written(t->tex);
                vk_exec_draw_new_cb();
                ex->dirty = VK_EXEC_DIRTY_ALL;
                vk_exec_barrier(ex);
            }
        }
    }
    if (!asp) return;
    float d = c->depth < 0.0f ? 0.0f : c->depth > 1.0f ? 1.0f : c->depth;
    VkClearDepthStencilValue v = {d, (uint32_t)c->stencil & 0xff};
    if (whole_target(ex, t)) {
        vk_exec_barrier(ex);
        VkImageSubresourceRange r = {asp, t->level, 1, t->base_layer, t->layers};
        vkCmdClearDepthStencilImage(ex->cb, t->image, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &r);
        ex->recorded++;
        vk_texture_gpu_written(t->tex);
        return;
    }
    VkClearAttachment a = {.aspectMask = asp};
    a.clearValue.depthStencil = v;
    clear_attachment(ex, &a);
}

// ---- the op loop

static void set_rt(VkExec *ex, const p_set_rt *r) {
    if (!memcmp(&ex->rt, r, sizeof *r)) return;   // the same targets again: the pass goes on
    vk_exec_pass_end(ex);
    ex->rt = *r;
    ex->rt_resolved = 0;
}

// ---- command buffer parts: see docs/how-it-works.md, "Command buffer parts".
// A submission is split into parts after 64 draws and dispatches and at least 1000 us of replay; a split may end a render pass
// (reopened at the next draw). Only the last part signals the fence. Bindings and dynamic state are lost at a split.
enum { split_n = 64 };
static int split_draws, part_start;
static const uint64_t split_ns = 1000 * 1000;
static uint64_t part_t0;
static uint64_t n_parts, n_subs, n_inpass;

static void maybe_split(VkExec *ex) {
    static int logged;
    if (!logged) {
        logged = 1;
        rt_log("vk: replay command buffer split every %d draws (inside passes too), parts of at least %d us", split_n, 1000);
    }
    if (split_draws < split_n) return;
    // Past the draw count every op asks; inside a pass the clock is read at every 16th of them only: the split comes at
    // most 15 ops late.
    static unsigned asks;
    if (ex->in_pass && ++asks % 16) return;
    if (kern_now_ns() - part_t0 < split_ns) return;   // split_draws stays: the next draw asks again
    split_draws = 0;
    if (ex->recorded == part_start) return;
    if (ex->in_pass) vk_exec_pass_end(ex), n_inpass++;
    ex->cb = vk_sync_split();
    part_start = ex->recorded;
    part_t0 = kern_now_ns();
    ex->dirty = VK_EXEC_DIRTY_ALL;
    vk_exec_draw_new_cb();
    n_parts++;
}

// An ordered read of the GPU-visible backing at this replay position (imported guest bytes can be older than their mirror).
int vk_exec_read_buffer(VkExec *ex, uint64_t addr, uint64_t size, void *out) {
    if (!x_here || ex != &X || !ex->cb || !out || !addr || !size || size > (8ull << 20) ||
        addr > UINT64_MAX - size || (addr & 3) || (size & 3)) return 0;
    VkMemSpan source;
    if (!vk_mem_resolve(addr, size, &source) || !source.buffer || source.avail < size || (source.offset & 3)) return 0;
    VK_MEM_PICK(source);   // choose an existing mirror; a read does not create a new one
    if (!source.buffer) return 0;
    VkCtx *c = vk_ctx();
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = NULL;
    VmaAllocationInfo mapped = {0};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    if (vmaCreateBuffer(c->vma, &bi, &ai, &buffer, &allocation, &mapped) != VK_SUCCESS) return 0;
    if (!mapped.pMappedData) { vmaDestroyBuffer(c->vma, buffer, allocation); return 0; }
    vk_exec_barrier(ex);   // end rendering and order all earlier writes before the transfer
    VkBufferCopy copy = {.srcOffset = source.offset, .dstOffset = 0, .size = size};
    vkCmdCopyBuffer(ex->cb, source.buffer, buffer, 1, &copy);
    VkBufferMemoryBarrier2 host = {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = buffer, .offset = 0, .size = size};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                           .bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &host};
    vkCmdPipelineBarrier2(ex->cb, &di);
    ex->recorded += 2;
    vk_mem_count(source.cls, VK_MUSE_COPY_SRC, size);
    VK_MEM_BOUND(source);
    ex->cb = vk_sync_split_wait();   // this prefix, not ex->fence (only the final part signals that value)
    split_draws = 0;
    part_start = ex->recorded;
    part_t0 = kern_now_ns();
    ex->barrier_at = -1;
    ex->dirty = VK_EXEC_DIRTY_ALL;
    vk_exec_draw_new_cb();
    n_parts++;
    VkResult result = vmaInvalidateAllocation(c->vma, allocation, 0, size);
    if (result == VK_SUCCESS) memcpy(out, mapped.pMappedData, (size_t)size);
    // split_wait is fatal on submission/wait failure, so this allocation is never freed while the copy is in flight.
    vmaDestroyBuffer(c->vma, buffer, allocation);
    return result == VK_SUCCESS;
}

// Event waits in replay order (waitev_*, a diagnostic): a wait whose event has not reached its value means the game queued
// this work before the work it waits for. One line "waitev-early f<frame>: ..." per such wait.
void *vk_exec_cur_queue;   // nvn_api_core.c nvnQueueSubmitCommands: the queue of the submission being replayed
static struct { volatile uint32_t *val; uint32_t v, known; uint64_t fence, frame; void *queue; } wev[64];
static uint64_t wev_waits, wev_early;
static uint64_t wev_cpu_waits, wev_cpu_unmet, wev_cpu_pending_t, wev_cpu_pending_frame;
static volatile uint32_t *volatile wev_cpu_pending;   // the CPU event the last replay ran past unsignalled
// nvn_api_core.c nvnEventSignal: the CPU signals an event; reports how late if the replay ran past a wait for it.
static volatile uint32_t *volatile co_cpu_wait;   // the CPU event a queue stands at (co_wait_blocks)
static uint64_t co_kicks;
int vk_exec_cpu_signal_wakes(volatile uint32_t *val) {   // 1: a queue stands at a wait for this CPU event
    if (__atomic_load_n(&co_cpu_wait, __ATOMIC_SEQ_CST) != val) return 0;
    __atomic_store_n(&co_cpu_wait, NULL, __ATOMIC_SEQ_CST);
    co_kicks++;
    return 1;
}
void vk_exec_cpu_signal(volatile uint32_t *val) {
    if (val != wev_cpu_pending) return;
    wev_cpu_pending = NULL;
    static volatile int32_t told;
    if (plat_add32(&told, 1) <= 2000)
        rt_log("vk_exec: waitev-cpu-late f%llu: CPU event %p signalled %.2f ms after the replay ran past its wait (frame %llu)",
               (unsigned long long)vk_frame_no, (void *)val, (double)(kern_now_ns() - wev_cpu_pending_t) / 1e6,
               (unsigned long long)wev_cpu_pending_frame);
}
static int wev_slot(volatile uint32_t *val) {
    int i = 0;
    while (i < 63 && wev[i].val && wev[i].val != val) i++;
    wev[i].val = val;
    return i;
}
static void waitev_signal(VkExec *ex, const void *a) {
    const p_signal_event *p = a;
    int i = wev_slot(p->val);
    if (p->mode == NVN_EVENT_SIGNAL_MODE_ADD) wev[i].v = (wev[i].known ? wev[i].v : *p->val) + p->value;
    else wev[i].v = p->value;
    wev[i].known = 1, wev[i].fence = ex->fence, wev[i].frame = vk_frame_no, wev[i].queue = vk_exec_cur_queue;
}
static void waitev_wait(VkExec *ex, const void *a) {
    const p_wait_event *p = a;
    int i = wev_slot(p->val);
    wev_waits++;
    if (!wev[i].known) {   // never signalled by a command buffer: a CPU event (nvnEventSignal). Is it signalled yet?
        uint32_t live = *p->val;
        int ok = p->mode == NVN_EVENT_WAIT_MODE_EQUAL ? live == p->value : (int32_t)(live - p->value) >= 0;
        wev_cpu_waits++;
        if (!ok) {
            wev_cpu_unmet++;
            wev_cpu_pending = p->val, wev_cpu_pending_t = kern_now_ns(), wev_cpu_pending_frame = vk_frame_no;
            static volatile int32_t told;
            if (plat_add32(&told, 1) <= 6000)
                rt_log("vk_exec: waitev-cpu f%llu: queue %p, fence %llu replays past a wait for CPU event %p %s %u; it is at %u (after %d "
                       "commands; %llu of %llu CPU waits so far)", (unsigned long long)vk_frame_no, vk_exec_cur_queue, (unsigned long long)ex->fence,
                       (void *)p->val, p->mode == NVN_EVENT_WAIT_MODE_EQUAL ? "==" : ">=", p->value, live, ex->recorded,
                       (unsigned long long)wev_cpu_unmet, (unsigned long long)wev_cpu_waits);
        }
        return;
    }
    int met = p->mode == NVN_EVENT_WAIT_MODE_EQUAL ? wev[i].v == p->value : (int32_t)(wev[i].v - p->value) >= 0;
    if (met) return;
    static volatile int32_t told;
    wev_early++;
    if (plat_add32(&told, 1) <= 64)
        rt_log("vk_exec: waitev-early f%llu: queue %p, fence %llu waits for event %p %s %u; in replay order it is at %u (signalled last by "
               "queue %p, fence %llu, frame %llu); %llu of %llu waits so far",
               (unsigned long long)vk_frame_no, vk_exec_cur_queue, (unsigned long long)ex->fence, (void *)p->val,
               p->mode == NVN_EVENT_WAIT_MODE_EQUAL ? "==" : ">=", p->value, wev[i].v, wev[i].queue, (unsigned long long)wev[i].fence,
               (unsigned long long)wev[i].frame, (unsigned long long)wev_early, (unsigned long long)wev_waits);
    // the frame's pattern: which (queue, event) pairs waited early; a line when it differs from the frame before's
    static uint64_t pat_frame, pat[24], last[24];
    static int npat, nlast;
    static volatile int32_t pat_told;
    if (vk_frame_no != pat_frame) {
        int same = npat == nlast && !memcmp(pat, last, sizeof pat[0] * (size_t)npat);
        if (!same && pat_frame && plat_add32(&pat_told, 1) <= 20000) {
            char line[512];
            int o = 0;
            for (int k = 0; k < npat && o < (int)sizeof line - 24; k++)
                o += snprintf(line + o, sizeof line - (size_t)o, " q%03llx/e%02llx", (unsigned long long)(pat[k] >> 32 & 0xfff), (unsigned long long)(pat[k] & 0xff));
            rt_log("vk_exec: waitev-pattern f%llu: %d early waits:%s (the frame before: %d)", (unsigned long long)pat_frame, npat, line, nlast);
        }
        memcpy(last, pat, sizeof pat), nlast = npat, npat = 0, pat_frame = vk_frame_no;
    }
    uint64_t key = ((uint64_t)(uintptr_t)vk_exec_cur_queue >> 12 & 0xfff) << 32 | ((uint64_t)(uintptr_t)p->val & 0xff);
    int k = 0;
    while (k < npat && pat[k] != key) k++;
    if (k == npat && npat < 24) {   // kept sorted, so the order of the waits does not matter
        while (k > 0 && pat[k - 1] > key) pat[k] = pat[k - 1], k--;
        pat[k] = key, npat++;
    }
}

// Where a command stream stands: the next record and the records to go back to after each OP_CALL. A stream stopped at
// an event wait (the queue coroutines below) is taken up again from its cursor.
typedef struct { const op_hdr *pc, *ret[CALL_DEPTH + 1]; int depth; } WalkCur;
static int co_wait_blocks(VkExec *ex, const p_wait_event *p);

// Replays the stream from the cursor. 0: its end was reached; 1: it stands at an event wait that is not met (the cursor
// names the wait).
static int walk(VkExec *ex, WalkCur *wc) {
    const op_hdr *h = wc->pc;
    for (;;) {
        if (h->size < sizeof *h || (h->size & 7) || h->op >= OP_COUNT_)
            rt_fatal("vk: corrupt command record at %p: op %u size %u", (const void *)h, h->op, h->size);
        const void *a = nvn_op_payload(h);
        switch (h->op) {
        case OP_END:
            if (!wc->depth) return 0;
            h = wc->ret[--wc->depth];   // back to the record after the call
            continue;
        case OP_BEGIN: break;
        case OP_JUMP: h = (const op_hdr *)(uintptr_t)((const p_jump *)a)->target; continue;
        case OP_CALL:
            if (wc->depth >= CALL_DEPTH) rt_fatal("vk: command buffer calls nested deeper than %d", CALL_DEPTH);
            wc->ret[wc->depth++] = nvn_op_next(h);
            h = nvn_op_first(((const p_jump *)a)->target);
            continue;
        case OP_SET_RT: set_rt(ex, a); break;
        case OP_CLEAR_COLOR: clear_color(ex, a); break;
        case OP_CLEAR_DS: clear_ds(ex, a); break;
        case OP_VIEWPORTS: {
            const p_viewports *v = a;
            for (int i = 0; i < v->n && v->first + i < 16; i++)
                if (v->first + i >= 0) memcpy(ex->vp[v->first + i], v->v[i], sizeof ex->vp[0]);
            ex->has_vp = 1;
            ex->dirty |= VK_EXEC_DIRTY_VIEWPORT;
            break;
        }
        case OP_SCISSOR:
            ex->scissor = *(const p_scissor *)a;
            ex->has_scissor = 1;
            ex->dirty |= VK_EXEC_DIRTY_SCISSOR;
            break;
        case OP_DEPTH_RANGE: {
            const p_depth_range *r = a;
            ex->depth_range[0] = r->n, ex->depth_range[1] = r->f;
            ex->dirty |= VK_EXEC_DIRTY_VIEWPORT;
            break;
        }
        case OP_STENCIL: {
            const p_stencil *s = a;
            for (int f = 0; f < 2; f++) {
                if (!(s->face & (1 << f))) continue;
                if (s->which == 0) ex->st_vmask[f] = (uint32_t)s->value_mask;
                else if (s->which == 1) ex->st_wmask[f] = (uint32_t)s->mask;
                else ex->st_ref[f] = (uint32_t)s->ref;
            }
            ex->dirty |= VK_EXEC_DIRTY_STENCIL;
            break;
        }
        case OP_POLY_OFFSET:
            ex->poly_offset = *(const p_poly_offset *)a;
            ex->dirty |= VK_EXEC_DIRTY_DEPTH_BIAS;
            break;
        case OP_BLEND_COLOR:
            memcpy(ex->blend_color, ((const p_blend_color *)a)->rgba, sizeof ex->blend_color);
            ex->dirty |= VK_EXEC_DIRTY_BLEND_CONST;
            break;
        case OP_DEPTH_CLAMP: ex->pipe.depth_clamp = ((const p_flag *)a)->value != 0; break;
        case OP_RASTER_DISCARD: ex->pipe.raster_discard = ((const p_flag *)a)->value != 0; break;
        case OP_PRIM_RESTART: ex->prim_restart = *(const p_prim_restart *)a; break;
        case OP_COPY_STRIDE: ex->copy_stride = ((const p_copy_stride *)a)->stride; break;
        case OP_COPY_IMAGE_STRIDE: ex->copy_image_stride = ((const p_copy_stride *)a)->stride; break;
        case OP_SET_TEX_POOL: ex->tex_pool = ((const p_set_pool *)a)->pool, ex->tex_serial++; break;
        case OP_SET_SAMP_POOL: ex->samp_pool = ((const p_set_pool *)a)->pool, ex->tex_serial++; break;
        case OP_BARRIER: vk_exec_barrier(ex); break;
        case OP_WAIT_SYNC: break;   // submit preparation waited before any replay/snapshot; source may be gone
        case OP_WAIT_EVENT:   // the stream stops here until the event is met
            if (co_wait_blocks(ex, a)) { wc->pc = h; return 1; }
            waitev_wait(ex, a);
            break;
        case OP_DRAW_MESH: case OP_DRAW_MESH_IND:
        case OP_DRAW: case OP_DRAW_IDX: case OP_DISPATCH: case OP_UPDATE_UB: case OP_DRAW_IDX_IND: case OP_DISPATCH_IND:
            if (h->op == OP_DRAW || h->op == OP_DRAW_IDX || h->op == OP_DRAW_IDX_IND) vk_exec_frame.draws_seen++, split_draws++;
            if (h->op == OP_DISPATCH || h->op == OP_DISPATCH_IND) vk_exec_frame.dispatches++, split_draws++;
            __attribute__((fallthrough));   // counted above; replayed with the binds below
        case OP_BIND_PROGRAM: case OP_BIND_STATE: case OP_BIND_VATTR: case OP_BIND_VSTREAM: case OP_BIND_VBUF:
        case OP_BIND_UBO: case OP_BIND_SSBO: case OP_BIND_TEX: case OP_BIND_SAMP: case OP_BIND_IMG: case OP_PUSH_CONST:
        case OP_GEOMETRY_MODE:
            vk_exec_op_draw(ex, h->op, a);
            break;
        case OP_COPY_B2B: case OP_COPY_B2T: case OP_COPY_T2B: case OP_COPY_T2T: case OP_CLEAR_TEX:
            if (h->op != OP_CLEAR_TEX) vk_exec_frame.copies++;
            __attribute__((fallthrough));
        case OP_SIGNAL_EVENT: case OP_REPORT_COUNTER: case OP_RESET_COUNTER: case OP_RENDER_ENABLE: case OP_RENDER_COND:
        case OP_CLEAR_BUF:
            if (h->op == OP_SIGNAL_EVENT) waitev_signal(ex, a);
            vk_exec_op_xfer(ex, h->op, a);
            break;
        default: rt_fatal("vk: op %s has no replay handler", nvn_op_name(h->op));
        }
        if (split_draws >= split_n) maybe_split(ex);
        h = nvn_op_next(h);
    }
}

// ---- The queues as coroutines: see docs/how-it-works.md, "Queue coroutines".
// Each queue keeps its pending submissions, its place (WalkCur) and its own NVN state; a stream runs until a wait is not met.
// What cannot run stays queued until the next submission, the CPU's signal (vk_exec_co_kick) or the present (waits ignored).
enum { CO_MAX = 8, CO_SUBS = 32 };
typedef struct { uint64_t *h; int n, i; uint64_t fence; } CoSub;
typedef struct Co {
    void *queue;
    CoSub sub[CO_SUBS];
    int head, count;
    WalkCur cur;
    int in_handle;                 // cur is inside sub[head].h[sub[head].i]
    VkExec st;                     // its NVN state while another queue runs
    int has_st;
    const p_wait_event *blocked;   // the wait it stands at
} Co;
static Co *co_tab[CO_MAX];
static int co_n, co_force;
static Co *co_active;              // the queue whose NVN state X holds
static uint64_t co_switches, co_blocks, co_forced, co_forced_subs, co_ran_subs;

static int co_on(void) {
    static int on = -1;
    if (on < 0) {
        on = 1;
        if (on) rt_log("vk_exec: the NVN queues run as coroutines: a stream stops at an event wait that is not met");
    }
    return on;
}

static int co_event_met(const p_wait_event *p) {
    int i = wev_slot(p->val);
    uint32_t v = wev[i].known ? wev[i].v : *p->val;   // a command buffer's event in replay order, else the CPU's, live
    return p->mode == NVN_EVENT_WAIT_MODE_EQUAL ? v == p->value : (int32_t)(v - p->value) >= 0;
}

static int co_wait_blocks(VkExec *ex, const p_wait_event *p) {
    (void)ex;
    if (!co_on() || co_force || co_event_met(p)) return 0;
    if (!wev[wev_slot(p->val)].known) {   // the CPU's: named first, then read again, so its signal cannot fall in between
        __atomic_store_n(&co_cpu_wait, p->val, __ATOMIC_SEQ_CST);
        if (co_event_met(p)) return 0;
    }
    co_blocks++;
    return 1;
}

// The NVN state of a queue into X; what belongs to the command buffer being recorded stays.
static void co_load(VkExec *ex, const VkExec *st) {
    static VkExec live;   // large: not on the stack
    live = *ex;
    *ex = *st;
    ex->cb = live.cb, ex->fence = live.fence, ex->submitted_ticks = live.submitted_ticks, ex->recorded = live.recorded;
    ex->in_pass = 0, ex->dirty = VK_EXEC_DIRTY_ALL, ex->rt_resolved = 0;
    ex->views = live.views, ex->nviews = live.nviews, ex->cap_views = live.cap_views, ex->barrier_at = live.barrier_at;
    memcpy(ex->fbl_img, live.fbl_img, sizeof ex->fbl_img), memcpy(ex->fbl_aspect, live.fbl_aspect, sizeof ex->fbl_aspect);
    ex->nfbl = live.nfbl, ex->fbl_mask = live.fbl_mask, ex->fb_dirty = live.fb_dirty;
    ex->fb_fs = NULL, ex->fb_serial = live.fb_serial, ex->fb_pass = live.fb_pass, ex->fb_sampled = live.fb_sampled;
    ex->tex_serial = live.tex_serial + 1, ex->pass_serial = live.pass_serial + 1;
}

static void co_switch(VkExec *ex, Co *q) {
    if (co_active == q) return;
    vk_exec_pass_end(ex);
    if (ex->recorded) vk_exec_barrier(ex);   // the next stream sees this one's writes
    if (co_active) co_active->st = *ex, co_active->has_st = 1;
    if (q->has_st) co_load(ex, &q->st);
    else ex->dirty = VK_EXEC_DIRTY_ALL, ex->rt_resolved = 0, ex->tex_serial++;   // its first run: the state as it is
    co_active = q;
    co_switches++;
}

static int co_runnable(const Co *q) { return q->count && (!q->blocked || co_force || co_event_met(q->blocked)); }

// Runs the queue's stream until its submissions are done or it stands at a wait.
static void co_exec(VkExec *ex, Co *q) {
    q->blocked = NULL;
    while (q->count) {
        CoSub *sb = &q->sub[q->head];
        if (!q->in_handle) {
            if (sb->i == sb->n) {
                free(sb->h);
                q->head = (q->head + 1) % CO_SUBS, q->count--;
                co_ran_subs++;
                if (co_force) co_forced_subs++;
                continue;
            }
            if (!sb->h[sb->i]) rt_fatal("vk: command handle %d of a submission is null", sb->i);
            q->cur = (WalkCur){.pc = nvn_op_first(sb->h[sb->i])};
            q->in_handle = 1;
        }
        if (walk(ex, &q->cur)) {
            q->blocked = nvn_op_payload(q->cur.pc);
            return;
        }
        q->in_handle = 0, sb->i++;
    }
}

// Every queue that can run, the given one first, until none can.
static void co_run(VkExec *ex, Co *first) {
    for (;;) {
        Co *q = first && co_runnable(first) ? first : NULL;
        for (int i = 0; i < co_n && !q; i++)
            if (co_runnable(co_tab[i])) q = co_tab[i];
        if (!q) break;
        co_switch(ex, q);
        co_exec(ex, q);
    }
}

static Co *co_of(void *queue) {
    for (int i = 0; i < co_n; i++)
        if (co_tab[i]->queue == queue) return co_tab[i];
    if (co_n == CO_MAX) rt_fatal("vk: more than %d NVN queues submit commands", CO_MAX);
    Co *q = calloc(1, sizeof *q);
    if (!q) rt_fatal("vk: out of memory");
    q->queue = queue;
    rt_log("vk_exec: queue %p is coroutine %d", queue, co_n);
    return co_tab[co_n++] = q;
}

static void co_report(void) {
    static uint64_t last_frame;
    if (vk_frame_no < last_frame + 600) return;
    last_frame = vk_frame_no;
    int pending = 0;
    for (int i = 0; i < co_n; i++) pending += co_tab[i]->count;
    rt_log("vk_exec: queue coroutines, frame %llu: since the last line %llu submissions run, %llu switches, %llu stops at a wait, %llu runs "
           "started by a CPU signal, %llu flushes with the waits ignored (%llu submissions); %d submissions pending now",
           (unsigned long long)vk_frame_no, (unsigned long long)co_ran_subs, (unsigned long long)co_switches, (unsigned long long)co_blocks,
           (unsigned long long)co_kicks, (unsigned long long)co_forced, (unsigned long long)co_forced_subs, pending);
    co_ran_subs = co_switches = co_blocks = co_kicks = co_forced = co_forced_subs = 0;
}

// A submission of the queue being replayed (vk_exec_cur_queue) joins its queue; then whatever can run, runs.
static void co_submit(VkExec *ex, const uint64_t *handles, int n, uint64_t fence) {
    Co *q = co_of(vk_exec_cur_queue);
    if (q->count == CO_SUBS) {   // far behind (its event never comes): everything pending runs in the old order
        co_force = 1, co_forced++;
        co_run(ex, NULL);
        co_force = 0;
    }
    CoSub *sb = &q->sub[(q->head + q->count) % CO_SUBS];
    *sb = (CoSub){.h = malloc(sizeof *sb->h * (size_t)n), .n = n, .fence = fence};
    if (!sb->h) rt_fatal("vk: out of memory");
    memcpy(sb->h, handles, sizeof *sb->h * (size_t)n);
    q->count++;
    co_run(ex, q);
    co_report();
}

// vk_usnap_begin's hook: the uniform ranges of every pending submission and of every waiting queue's bindings.
static void co_usnap_extra(void) {
    for (int i = 0; i < co_n; i++) {
        Co *q = co_tab[i];
        if (!q->count) continue;
        if (q != co_active && q->has_st) vk_usnap_add_state(&q->st);
        for (int k = 0; k < q->count; k++) {
            const CoSub *sb = &q->sub[(q->head + k) % CO_SUBS];
            vk_usnap_add_handles(sb->h + sb->i, sb->n - sb->i);
        }
        if (q->in_handle) vk_usnap_add_stream(q->cur.pc), vk_usnap_add_returns((const void *const *)q->cur.ret, q->cur.depth);
    }
}

// Pending submissions? (nvn_api: whether a kick or a flush has anything to do.)
int vk_exec_co_pending(void) {
    int n = 0;
    for (int i = 0; i < co_n; i++) n += co_tab[i]->count;
    return n;
}

// The work of a submission the back end makes for a kick (the CPU signalled an event a queue stands at) or a flush
// (force: a present or a finish: everything left runs with the waits ignored).
static void co_work(VkExec *ex, void *context) {
    int force = context != NULL;
    if (force && vk_exec_co_pending()) co_force = 1, co_forced++;
    co_run(ex, NULL);
    co_force = 0;
}
int vk_exec_record_co(VkCommandBuffer cb, uint64_t fence, uint64_t submitted_ticks, int force) {
    static int marker;
    return vk_exec_record_work(cb, fence, submitted_ticks, co_work, force ? &marker : NULL);
}

static int record_impl(VkCommandBuffer cb, const uint64_t *handles, int n, uint64_t fence, uint64_t submitted_ticks,
                       VkExecWorkFn work, void *context) {
    VkExec *ex = &X;
    if (!x_ready) state_reset(ex), x_ready = 1;
    ex->cb = cb;
    ex->fence = fence;
    ex->submitted_ticks = submitted_ticks;
    ex->recorded = 0;
    ex->in_pass = 0;
    ex->dirty = VK_EXEC_DIRTY_ALL;
    ex->rt_resolved = 0;   // images may have changed since the last command buffer
    ex->nviews = 0;
    ex->barrier_at = -1;
    ex->nfbl = 0;   // the previous command buffer ended with a full barrier: every image is in GENERAL
    vk_exec_frame.submits++;
    split_draws = part_start = 0;
    part_t0 = kern_now_ns();
    x_here = 1;
    int memory = n || work;
    if (memory) {
        vk_mem_mirror_begin();
        if (co_n) vk_usnap_extra = co_usnap_extra;   // the waiting queues' uniform ranges too
        vk_usnap_begin(ex, handles, n, fence);
    }
    if (work) work(ex, context);
    else {
        if (co_on()) co_submit(ex, handles, n, fence);
        else
        for (int i = 0; i < n; i++) {
            if (!handles[i]) rt_fatal("vk: command handle %d of a submission is null", i);
            WalkCur wc = {.pc = nvn_op_first(handles[i])};
            walk(ex, &wc);
        }
    }
    vk_exec_pass_end(ex);
    x_here = 0;
    if (ex->recorded) vk_exec_barrier(ex);   // later submissions see this one's writes
    if (memory) ex->recorded += vk_texture_prestage(ex->cb, fence);   // tex_prestage: small targets to the host ring
    if (memory) ex->recorded += vk_mem_mirror_end(ex->cb, fence), vk_usnap_end(fence);   // VRAM mirror bytes the GPU wrote go back to guest memory
    if (ex->nviews) {
        ViewList *l = malloc(sizeof *l + sizeof(VkImageView) * (size_t)ex->nviews);
        if (!l) rt_fatal("vk: out of memory");
        l->n = ex->nviews;
        for (int i = 0; i < ex->nviews; i++) l->v[i] = ((ViewEnt *)ex->views)[i].view;
        vk_completion_add(fence, free_views, l);
        ex->nviews = 0;
    }
    ex->cb = VK_NULL_HANDLE;
    if (memory) n_subs++;
    if ((rt_trace & RT_TRACE_PERF) && n_subs >= 240) {
        rt_log("perf: replay command buffer parts per submission %.2f (%llu splits inside a pass)", (double)(n_subs + n_parts) / (double)n_subs,
               (unsigned long long)n_inpass);
        n_subs = n_parts = n_inpass = 0;
    }
    return ex->recorded - part_start;   // what the last part holds
}

int vk_exec_record(VkCommandBuffer cb, const uint64_t *handles, int n, uint64_t fence, uint64_t submitted_ticks) {
    return record_impl(cb, handles, n, fence, submitted_ticks, NULL, NULL);
}

int vk_exec_record_work(VkCommandBuffer cb, uint64_t fence, uint64_t submitted_ticks, VkExecWorkFn work, void *context) {
    if (!work) rt_fatal("vk: work recording without a callback");
    return record_impl(cb, NULL, 0, fence, submitted_ticks, work, context);
}
