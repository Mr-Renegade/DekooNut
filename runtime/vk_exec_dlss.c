// The upscaler's stand-in and DLSS (DLSS Super Resolution, its jitter and the 4K composite).
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "vk.h"
#include "vk_exec.h"
#include "vk_exec_draw_int.h"
#include "tex.h"
#include "platform/platform.h"
#include "upscale_decode_spv.h"
#include "dlss_color_spv.h"
#include "dlss_motion_spv.h"
#include "dlss_output_spv.h"
#include "dlss_mask_spv.h"
#include "dlss_4k_composite_spv.h"

// ---- the upscaler's stand-in
// The picture a compute shader wrote last (the video's NV12 -> RGBA conversion is one), kept as the source of the blit.
static struct { VkImage image; uint64_t addr; uint32_t w, h; VkFormat format; VkImageUsageFlags usage; int fresh, packed; } up_src;   // fresh: not blitted since written
// up_packed.image: the packed bits as RGB9E5; .linear: the colour after the decode (shaders/upscale_decode.comp).
static struct {
    VkImage image, linear;
    VmaAllocation alloc, linear_alloc;
    VkImageView view, linear_view;
    uint32_t w, h;
} up_packed;
// up_bits: an 8-bit UNORM image of the destination's size. The stand-in scales into it and copies the bits on into an
// sRGB destination (upscale_standin).
static struct { VkImage image; VmaAllocation alloc; uint32_t w, h; VkFormat format; } up_bits;
static struct {
    int tried;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipe;
    VkSampler sampler;
} up_decode;

void upscale_release(void) {
    VkCtx *c = vk_ctx();
    if (up_packed.view) vkDestroyImageView(c->device, up_packed.view, NULL);
    if (up_packed.linear_view) vkDestroyImageView(c->device, up_packed.linear_view, NULL);
    if (up_packed.image) vmaDestroyImage(c->vma, up_packed.image, up_packed.alloc);
    if (up_packed.linear) vmaDestroyImage(c->vma, up_packed.linear, up_packed.linear_alloc);
    if (up_bits.image) vmaDestroyImage(c->vma, up_bits.image, up_bits.alloc);
    memset(&up_bits, 0, sizeof up_bits);
    if (up_decode.pipe) vkDestroyPipeline(c->device, up_decode.pipe, NULL);
    if (up_decode.sampler) vkDestroySampler(c->device, up_decode.sampler, NULL);
    if (up_decode.layout) vkDestroyPipelineLayout(c->device, up_decode.layout, NULL);
    if (up_decode.dsl) vkDestroyDescriptorSetLayout(c->device, up_decode.dsl, NULL);
    memset(&up_decode, 0, sizeof up_decode);
    memset(&up_packed, 0, sizeof up_packed);
    memset(&up_src, 0, sizeof up_src);
}

static void upscale_decode_init(VkCtx *c) {
    up_decode.tried = 1;
    VkDescriptorSetLayoutBinding b[2] = {
        {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT}};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 2, .pBindings = b};
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST,
                              .minFilter = VK_FILTER_NEAREST, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                              .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0};
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                   .codeSize = sizeof upscale_decode_spv, .pCode = upscale_decode_spv};
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(c->device, &dci, NULL, &up_decode.dsl) != VK_SUCCESS ||
        vkCreateSampler(c->device, &si, NULL, &up_decode.sampler) != VK_SUCCESS)
        rt_fatal("tensor: the upscale decode's descriptors could not be created");
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &up_decode.dsl};
    if (vkCreatePipelineLayout(c->device, &li, NULL, &up_decode.layout) != VK_SUCCESS ||
        vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS)
        rt_fatal("tensor: the upscale decode's shader could not be created");
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = up_decode.layout};
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &up_decode.pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    if (r != VK_SUCCESS) rt_fatal("tensor: the upscale decode's pipeline could not be created (%d)", (int)r);
}

static VkImage upscale_rgb9e5(VkExec *ex) {
    VkCtx *c = vk_ctx();
    if (!up_decode.tried) upscale_decode_init(c);
    if (!up_packed.image || up_packed.w != up_src.w || up_packed.h != up_src.h) {
        if (up_packed.image) {
            vk_defer_view(up_packed.view), vk_defer_view(up_packed.linear_view);
            vk_defer_image(up_packed.image, up_packed.alloc), vk_defer_image(up_packed.linear, up_packed.linear_alloc);
        }
        memset(&up_packed, 0, sizeof up_packed);
        VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, .extent = {up_src.w, up_src.h, 1}, .mipLevels = 1,
            .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(c->vma, &ii, &ai, &up_packed.image, &up_packed.alloc, NULL) != VK_SUCCESS)
            rt_fatal("tensor: RGB9E5 conversion image allocation failed");
        ii.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        ii.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if (vmaCreateImage(c->vma, &ii, &ai, &up_packed.linear, &up_packed.linear_alloc, NULL) != VK_SUCCESS)
            rt_fatal("tensor: decoded colour image allocation failed");
        VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = up_packed.image,
                                    .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_E5B9G9R9_UFLOAT_PACK32,
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        if (vkCreateImageView(c->device, &vi, NULL, &up_packed.view) != VK_SUCCESS)
            rt_fatal("tensor: RGB9E5 view creation failed");
        vi.image = up_packed.linear, vi.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        if (vkCreateImageView(c->device, &vi, NULL, &up_packed.linear_view) != VK_SUCCESS)
            rt_fatal("tensor: decoded colour view creation failed");
        up_packed.w = up_src.w, up_packed.h = up_src.h;
    }
    // Independent memory: the copy keeps the packed bits, touches no aliased pool storage and does no float conversion.
    VkImageMemoryBarrier2 image = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = up_packed.image, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &image};
    vkCmdPipelineBarrier2(ex->cb, &di);
    VkImageCopy copy = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .extent = {up_src.w, up_src.h, 1}};
    vkCmdCopyImage(ex->cb, up_src.image, VK_IMAGE_LAYOUT_GENERAL, up_packed.image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
    image.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, image.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    image.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, image.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    image.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    vkCmdPipelineBarrier2(ex->cb, &di);
    // The packing pass stored sqrt(colour), and the network's output is the colour itself: square it here
    // (shaders/upscale_decode.comp). The last frame's blit has read .linear; its old content goes.
    image.image = up_packed.linear, image.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image.srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT, image.srcAccessMask = 0;
    image.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    vkCmdPipelineBarrier2(ex->cb, &di);
    VkDescriptorImageInfo in = {up_decode.sampler, up_packed.view, VK_IMAGE_LAYOUT_GENERAL},
                          out = {VK_NULL_HANDLE, up_packed.linear_view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &in},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &out}};
    vkCmdBindPipeline(ex->cb, VK_PIPELINE_BIND_POINT_COMPUTE, up_decode.pipe);
    B.pipe[1] = VK_NULL_HANDLE;   // not the game's: the next dispatch binds its own again (its sets are pushed every time)
    rec_push(vk_pipeline_push_fn(), ex->cb, VK_PIPELINE_BIND_POINT_COMPUTE, up_decode.layout, 0, 2, w);
    vkCmdDispatch(ex->cb, (up_src.w + 7) / 8, (up_src.h + 7) / 8, 1);
    image.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    image.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, image.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    image.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT, image.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier2(ex->cb, &di);
    ex->recorded++;
    return up_packed.linear;
}

// The bound storage image of the compute stage with the largest area (and its size), or NULL.
static const VkTexImage *largest_storage_image(VkExec *ex) {
    const VkTexImage *best = NULL;
    for (int slot = 0; slot < 8; slot++) {
        uint64_t h = ex->img[NVN_STAGE_CS][slot];
        if (!h) continue;
        const VkTexImage *t = vk_texture_by_id_locked(&ex->tex_pool, (int)(h & 0xfffff), NULL, NULL);
        if (t && t->type == VK_IMAGE_TYPE_2D && t->layers == 1 && (!best || (uint64_t)t->width * t->height > (uint64_t)best->width * best->height)) best = t;
    }
    return best;
}

// After a compute dispatch: the largest storage image its shader declares (not every slot: the slots keep the images of
// earlier dispatches, the stand-in's own destination among them).
void upscale_source_note(VkExec *ex, const VkPipeInfo *info) {
    const VkTexImage *best = NULL;
    int packed = 0;
    vk_texture_lock();
    for (uint32_t i = 0; i < info->n_bindings; i++) {
        const VkPipeBinding *b = &info->bindings[i];
        if (b->kind != GFX_RES_IMG || !b->res || b->res->slot < 0 || b->res->slot >= 32) continue;
        uint64_t h = ex->img[NVN_STAGE_CS][b->res->slot];
        view_t view;
        int has_view = 0;
        const VkTexImage *t = h ? vk_texture_by_id_locked(&ex->tex_pool, (int)(h & 0xfffff), &view, &has_view) : NULL;
        if (t && t->type == VK_IMAGE_TYPE_2D && (!best || (uint64_t)t->width * t->height > (uint64_t)best->width * best->height)) {
            best = t;
            packed = has_view && view.set_format && view.format == NVN_FORMAT_R32UI &&
                     (t->format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || t->format == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32);
        }
    }
    if (best && ((best->width >= 1280 && best->height >= 720) || packed))
        up_src.image = best->image, up_src.addr = best->addr, up_src.w = best->width, up_src.h = best->height,
        up_src.format = best->format, up_src.usage = best->usage, up_src.fresh = 1, up_src.packed = packed;
    vk_texture_unlock();
}

// The 8-bit UNORM format whose bits an sRGB destination shares, or VK_FORMAT_UNDEFINED.
static VkFormat srgb_bits_format(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_B8G8R8A8_SRGB: return VK_FORMAT_B8G8R8A8_UNORM;
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32: return VK_FORMAT_A8B8G8R8_UNORM_PACK32;
    default: return VK_FORMAT_UNDEFINED;
    }
}

// The UNORM image the stand-in scales into before the bits go to an sRGB destination; NULL when it cannot be made.
static VkImage upscale_bits_image(uint32_t w, uint32_t h, VkFormat format) {
    if (up_bits.image && up_bits.w == w && up_bits.h == h && up_bits.format == format) return up_bits.image;
    VkCtx *c = vk_ctx();
    if (up_bits.image) vk_defer_image(up_bits.image, up_bits.alloc);
    memset(&up_bits, 0, sizeof up_bits);
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = format,
        .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ii, &ai, &up_bits.image, &up_bits.alloc, NULL) != VK_SUCCESS) {
        rt_log("tensor: %ux%u image for the stand-in's sRGB destination could not be made: blitting directly", w, h);
        memset(&up_bits, 0, sizeof up_bits);
        return VK_NULL_HANDLE;
    }
    up_bits.w = w, up_bits.h = h, up_bits.format = format;
    return up_bits.image;
}

// A tensor stage was dropped: if it binds a storage image larger than the last compute picture, scale that picture into it.
// An 8-bit UNORM picture into an sRGB destination keeps its bits: a blit would code an already sRGB-coded picture twice.
void upscale_standin(VkExec *ex) {
    if (!up_src.image || !up_src.fresh) return;   // once per new picture: every stage of the chain would do it again
    vk_texture_lock();
    // A packed source whose own image is RGB9E5 holds the packing pass's bits as they are: decode it too, as an
    // R11G11B10 image with an RGB9E5 view bound here is.
    int rgb9e5 = up_src.packed && up_src.format == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
    if (!rgb9e5 && up_src.packed && up_src.format == VK_FORMAT_B10G11R11_UFLOAT_PACK32)
        for (int slot = 0; slot < 32; slot++) {
            uint64_t handle = ex->tex[NVN_STAGE_CS][slot];
            view_t view;
            int has_view = 0;
            const VkTexImage *input = handle ? vk_texture_by_id_locked(&ex->tex_pool, (int)(handle & 0xfffff), &view, &has_view) : NULL;
            if (input && input->addr == up_src.addr && input->width == up_src.w && input->height == up_src.h &&
                ((has_view && view.set_format && view.format == NVN_FORMAT_RGB9E5F) ||
                 input->format == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32)) { rgb9e5 = 1; break; }
        }
    const VkTexImage *t = largest_storage_image(ex);
    VkImage dst = VK_NULL_HANDLE;
    uint32_t w = 0, h = 0;
    VkFormat dst_format = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags dst_usage = 0;
    if (t && t->width > up_src.w && t->height > up_src.h && t->image != up_src.image) {
        dst = t->image, w = t->width, h = t->height, dst_format = t->format, dst_usage = t->usage;
        vk_texture_storage_bound_locked(t);   // as a storage binding does: the GPU writes this image (what later samples must see)
    }
    vk_texture_unlock();
    if (!dst) return;
    cb_res(ex);
    vk_exec_barrier(ex);   // ends the pass; the source is written
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                           .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(ex->cb, &di);
    VkImage source = rgb9e5 ? upscale_rgb9e5(ex) : up_src.image;
    VkImageBlit rg = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .srcOffsets = {{0, 0, 0}, {(int32_t)up_src.w, (int32_t)up_src.h, 1}},
                      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}}};
    up_src.fresh = 0;
    VkFormat bits = srgb_bits_format(dst_format);
    VkImage mid = !rgb9e5 && bits != VK_FORMAT_UNDEFINED && bits == up_src.format
                      ? upscale_bits_image(w, h, bits) : VK_NULL_HANDLE;
    if (mid) {
        VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, .srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT, .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = mid, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VkDependencyInfo mdi = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
        vkCmdPipelineBarrier2(ex->cb, &mdi);   // the last copy out of it has read it; its old content goes
        vkCmdBlitImage(ex->cb, source, VK_IMAGE_LAYOUT_GENERAL, mid, VK_IMAGE_LAYOUT_GENERAL, 1, &rg, VK_FILTER_LINEAR);
        ib.srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT, ib.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        ib.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, ib.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        ib.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        vkCmdPipelineBarrier2(ex->cb, &mdi);
        VkImageCopy cp = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                          .extent = {w, h, 1}};
        vkCmdCopyImage(ex->cb, mid, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_GENERAL, 1, &cp);   // same bits, no coding
        mb.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        static uint64_t n_bits;
        ++n_bits;
        if (!(n_bits & (n_bits - 1)))   // 1, 2, 4, 8, ...: how long the path stays in use
            rt_log("tensor: stand-in keeps the bits of a %ux%u UNORM picture in its sRGB destination (%llu times, frame %llu)",
                   w, h, (unsigned long long)n_bits, (unsigned long long)vk_frame_no);
    } else {
        vkCmdBlitImage(ex->cb, source, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_GENERAL, 1, &rg, VK_FILTER_LINEAR);
        mb.srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
    }
    static int decoded;
    if (rgb9e5 && !decoded++) rt_log("tensor: packed source decoded through an RGB9E5 image and squared");

    mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, mb.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier2(ex->cb, &di);
    ex->recorded++;
    static int told;
    if (!told++) rt_log("tensor: upscale stand-in blits %ux%u to %ux%u (formats %d -> %d, usage %#x -> %#x)", up_src.w, up_src.h, w, h, (int)up_src.format, (int)dst_format, (unsigned)up_src.usage, (unsigned)dst_usage);
}

// ---- DLSS Super Resolution in place of the stand-in (vk_dlss.c); see README, "DLSS".
// The game has no motion vectors or jitter: motion comes from the scene depth reprojected through this and the previous
// camera (vertex cbuf 3); the colour is the game's sqrt-coded LDR input. Anything unexpected leaves the stand-in, with the
// reason logged when it changes and every 120th repeat.
static struct {
    uint64_t epoch;                      // upscales tried so far + 1: what the camera below belongs to
    uint64_t cam_epoch;                  // 0: no camera since the last upscale
    uint32_t cw, ch;                     // its colour target's size
    float cur[16], prev[16];             // view-projection of this frame and of the one before (cbuf 3 +0x30, +0x180)
    VkImage depth;                       // the depth target of that draw (D32)
    int stage;
    uint64_t last_ok;                    // epoch of the last frame DLSS drew
    float last_cur[16];
    const char *reason;
    uint64_t repeats, drawn;
    struct { VkImage img; VmaAllocation a; VkImageView v; } packed, color, motion, out, mask, pdepth, pcolor, out4k[2], ref[2];
    uint32_t rw, rh, ow, oh;
    // 4K path: DLSS's window-size image and the game target's copy right after it (the reference), two of each by
    // upscale parity: the present of frame N reads N's while the replay already writes N+1's
    uint32_t k4w, k4h, k4ow, k4oh;       // their sizes (0: not made)
    uint32_t k4sw, k4sh;                 // the window's size
    uint64_t k4_ep[2];                   // the upscale each pair holds
    uint64_t k4_token, k4_taken;         // the newest pair (atomic), the one the last present took (vk_dlss4k_take)
    int k4_broken;                       // the 4K feature or its images failed: the 960p path for the rest of the run
    const char *k4_reason;
    // Jitter: Halton(2,3) viewport shift of the draws into the scene depth, only while DLSS drew the last upscale
    VkImage jdepth;                      // the scene depth of the last upscale DLSS drew
    uint64_t jit_epoch, jit_hold;        // the upscale the shift below is for; no shift before this upscale (after a miss)
    float jx, jy;
    uint64_t pdepth_epoch;   // the upscale whose depth and colour pdepth, pcolor hold (0: none)
    int init;                            // 1 pipelines made, -1 failed
    VkDescriptorSetLayout dsl, dsl3;     // 2 bindings (in, out); 5 for the mask pass (depth, mask, previous depth, colour, previous colour)
    VkSampler sampler;
    struct { VkPipelineLayout layout; VkPipeline pipe; } pc, pm, po, pk;   // colour, motion, output, mask passes
} DL = {.epoch = 1};

void dlss_image_gone(VkImage image) {
    if (image && image == DL.depth) DL.depth = VK_NULL_HANDLE, DL.cam_epoch = 0;
    if (image && image == DL.jdepth) DL.jdepth = VK_NULL_HANDLE;
}

static float dlss_halton(uint32_t i, uint32_t b) {
    float f = 1, r = 0;
    for (; i; i /= b) f /= (float)b, r += f * (float)(i % b);
    return r;
}
// This frame's sub-pixel shift for a draw with a 16:9 depth target from 128 pixels wide up to the scene's size, as a
// projection jitter (dlss_proj_block). Only right after an upscale DLSS drew, and not for 60 upscales after one it did not.
// Halton(2,3) over 72 phases.
void dlss_jitter_for(const VkExec *ex, float *jx, float *jy) {
    rf_proj = 0;
    if (!DL.jdepth || !vk_dlss_on()) return;
    const VkExecTarget *d = &ex->depth;
    uint32_t w = d->image ? d->width : 0, h = d->image ? d->height : 0;
    if (!(w >= 128 && w <= DL.cw && (uint64_t)w * DL.ch == (uint64_t)h * DL.cw)) return;
    uint64_t e = DL.epoch;
    if (!DL.last_ok || DL.last_ok + 1 != e || e < DL.jit_hold) return;
    if (DL.jit_epoch != e) {
        uint32_t i = (uint32_t)(e % 72) + 1;
        DL.jit_epoch = e, DL.jx = dlss_halton(i, 2) - 0.5f, DL.jy = dlss_halton(i, 3) - 0.5f;
    }
    // projection jitter: the camera block moves the draw (ubo_info), the viewport stays
    rf_proj = 1;
    rf_pj[0] = 2.0f * DL.jx / (float)DL.cw;
    rf_pj[1] = -2.0f * DL.jy / (float)DL.ch;   // NDC y is up
}
// Projection jitter: a draw's c3 block with the scene camera layout (0x230 bytes or more; w row (0, 0, -1, 0) at +0xa0) is
// bound as a copy whose x and y matrix rows have a x the w row added. Every pass size then moves by the jitter, and screen
// positions derived from the clip position move with it (a viewport shift would double the shift). Copies are cached per
// block address, submission, upscale and inline-update generation.
int dlss_proj_block(VkExec *ex, const p_bind_buf *b, VkDescriptorBufferInfo *out) {
    if (!b->addr || b->size < 0x230 || b->size > 0x4000) return 0;
    // 256 entries by address; the block is read straight into its scratch copy instead of through blk.
    static struct { uint64_t addr, size, fence, epoch; uint32_t gen; float a[2]; VkBuffer buf; VkDeviceSize off; int ok; } C[256];
    int k = (int)((b->addr * 0x9E3779B97F4A7C15ull) >> 56);
    if (C[k].addr == b->addr && C[k].size == b->size && C[k].fence == ex->fence && C[k].epoch == DL.epoch && C[k].gen == ub_gen &&
        C[k].a[0] == rf_pj[0] && C[k].a[1] == rf_pj[1]) {
        if (!C[k].ok) return 0;
        RS.volatile_ = 1;
        *out = (VkDescriptorBufferInfo){C[k].buf, C[k].off, b->size};
        return 1;
    }
    C[k].addr = b->addr, C[k].size = b->size, C[k].fence = ex->fence, C[k].epoch = DL.epoch, C[k].gen = ub_gen, C[k].ok = 0;
    C[k].a[0] = rf_pj[0], C[k].a[1] = rf_pj[1];
    static float blk[0x4000 / 4];
    // The w rows first (16 bytes at +0xa0): most blocks of this size are per-object blocks, not the camera.
    float w[4];
    if (!dlss_read(ex, b->addr + 0xa0, w, sizeof w) || w[2] != -1.0f || w[3] != 0.0f || w[0] != 0.0f || w[1] != 0.0f)
        return 0;
    // Matrices are patched in a local copy of the first 0xb0 bytes, then written over the scratch copy (write-combined: never read back).
    float *src = blk;
    if (!dlss_read(ex, b->addr, blk, 0xb0)) return 0;
    if (src[0xa8 / 4] != -1.0f || src[0xac / 4] != 0.0f || src[0xa0 / 4] != 0.0f || src[0xa4 / 4] != 0.0f) return 0;
    for (int r = 0; r < 2; r++) {   // view-projection at +0x30 (w row +0x60), projection at +0x70 (w row +0xa0)
        float *m = src + (r ? 0x70 : 0x30) / 4;
        for (int i = 0; i < 4; i++) m[i] += rf_pj[0] * m[12 + i], m[4 + i] += rf_pj[1] * m[12 + i];
    }
    uint8_t *p = scratch(ex, b->size, &C[k].buf, &C[k].off);
    if (!dlss_read(ex, b->addr, p, b->size)) return 0;
    memcpy(p + 0x30, (const uint8_t *)blk + 0x30, 0x80);
    C[k].ok = 1;
    RS.volatile_ = 1;
    *out = (VkDescriptorBufferInfo){C[k].buf, C[k].off, b->size};
    return 1;
}

static void dlss_jitter_frame(void) {
    static int told;
    if (!told) {
        told = 1;
        rt_log("dlss: jitter on (Halton(2,3), 72 phases, viewport shift of every draw with a depth "
               "target of the scene's size, DLSS's offset sign %+d)", 1);
        rt_log("dlss: jitter mode %d amplitude %.3f", 13, 1.0f);
    }
}

// The bytes as the bound block holds them here: an inline update of this submission, the uniform snapshot, guest memory.
int dlss_read(const VkExec *ex, uint64_t a, void *out, size_t n) {
    if (upd_read(ex, a, out, n) || vk_usnap_read(a, n, out)) return 1;
    if (dyn_readable(a, n) < n) return 0;
    memcpy(out, (const void *)(uintptr_t)a, n);
    return 1;
}

// Any draw into a scene-sized B10G11R11 target qualifies; the camera block is recognised by its contents. Reflections and
// half-size passes share its layout: the largest target since the last upscale wins.
void dlss_track_camera(VkExec *ex, int stage) {
    if (!vk_dlss_on()) return;
    const VkExecTarget *c = &ex->color[0], *d = &ex->depth;
    if (!c->image || c->format != VK_FORMAT_B10G11R11_UFLOAT_PACK32 || c->width < 640 || c->height < 360 || c->is3d) return;
    if (DL.cam_epoch == DL.epoch && (uint64_t)c->width * c->height <= (uint64_t)DL.cw * DL.ch) return;
    if (!d->image || !d->img || d->width != c->width || d->height != c->height || d->samples > 1) return;
    if (d->img->format != VK_FORMAT_D32_SFLOAT || !(d->img->usage & VK_IMAGE_USAGE_SAMPLED_BIT)) {
        static int told;
        if (!told++) rt_log("dlss: a %ux%u scene draw has depth format %d (view %d), not a sampled D32: not used",
                            c->width, c->height, (int)d->img->format, (int)d->format);
        return;
    }
    const p_bind_buf *u = &ex->ubo[stage][0];   // bank 3
    float cam[0x230 / 4];
    if (!u->addr || u->size < sizeof cam || !dlss_read(ex, u->addr, cam, sizeof cam)) return;
    if (cam[0x120 / 4] != (float)c->width || cam[0x124 / 4] != (float)c->height || cam[0xa8 / 4] != -1.0f || cam[0xac / 4] != 0.0f)
        return;
    DL.cam_epoch = DL.epoch, DL.cw = c->width, DL.ch = c->height, DL.depth = d->image, DL.stage = stage;
    extern void nav_state_camera(uint64_t addr);
    nav_state_camera(u->addr);   // OOT_STATE_FILE's eye and fwd (nav_state.c)
    memcpy(DL.cur, cam + 0x30 / 4, sizeof DL.cur);
    memcpy(DL.prev, cam + 0x180 / 4, sizeof DL.prev);
    static int told;
    if (!told++) rt_log("dlss: scene camera found (stage %d bank 3, target %ux%u, depth %ux%u D32)", stage, c->width, c->height,
                        d->width, d->height);
}

// previous * inverse(current), both row-major. 0 when current is singular.
static int dlss_clip_to_previous(const float *cur, const float *prev, float *out) {
    double rows[4][8];
    for (int i = 0; i < 4; i++)
        for (int k = 0; k < 8; k++) rows[i][k] = k < 4 ? cur[i * 4 + k] : (k - 4 == i);
    for (int col = 0; col < 4; col++) {
        int piv = col;
        for (int r = col + 1; r < 4; r++) if (fabs(rows[r][col]) > fabs(rows[piv][col])) piv = r;
        if (piv != col) for (int k = 0; k < 8; k++) { double t = rows[col][k]; rows[col][k] = rows[piv][k], rows[piv][k] = t; }
        double div = rows[col][col];
        if (!isfinite(div) || fabs(div) < 1e-10) return 0;
        for (int k = 0; k < 8; k++) rows[col][k] /= div;
        for (int r = 0; r < 4; r++) {
            if (r == col) continue;
            double sc = rows[r][col];
            for (int k = 0; k < 8; k++) rows[r][k] -= sc * rows[col][k];
        }
    }
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            double v = 0;
            for (int i = 0; i < 4; i++) v += prev[r * 4 + i] * rows[i][c + 4];
            out[r * 4 + c] = (float)v;
            if (!isfinite(out[r * 4 + c])) return 0;
        }
    return 1;
}

static int dlss_fail(const char *reason) {
    DL.jit_hold = DL.epoch + 60;   // no shift for a second
    if (reason != DL.reason || ++DL.repeats % 120 == 0)
        rt_log("dlss: the stand-in draws upscale %llu (%s)", (unsigned long long)(DL.epoch - 1), reason);
    if (reason != DL.reason) DL.reason = reason, DL.repeats = 0;
    return 0;
}

static int dlss_pipeline(VkCtx *c, const uint32_t *code, size_t bytes, uint32_t push, VkPipelineLayout *layout, VkPipeline *pipe,
                         const VkDescriptorSetLayout *dsl) {
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, push};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = dsl, .pushConstantRangeCount = push ? 1 : 0, .pPushConstantRanges = &pr};
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = bytes, .pCode = code};
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(c->device, &li, NULL, layout) != VK_SUCCESS || vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS)
        return 0;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = *layout};
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    return r == VK_SUCCESS;
}

static int dlss_pipelines(void) {
    if (DL.init) return DL.init > 0;
    VkCtx *c = vk_ctx();
    DL.init = -1;
    VkDescriptorSetLayoutBinding b[5] = {
        {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding = 4, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT}};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, .bindingCount = 2, .pBindings = b};
    VkDescriptorSetLayoutCreateInfo dci3 = dci;
    dci3.bindingCount = 5;
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
                              .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST, .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
    if (vkCreateDescriptorSetLayout(c->device, &dci, NULL, &DL.dsl) != VK_SUCCESS || vkCreateDescriptorSetLayout(c->device, &dci3, NULL, &DL.dsl3) != VK_SUCCESS ||
        vkCreateSampler(c->device, &si, NULL, &DL.sampler) != VK_SUCCESS ||
        !dlss_pipeline(c, dlss_color_spv, sizeof dlss_color_spv, 0, &DL.pc.layout, &DL.pc.pipe, &DL.dsl) ||
        !dlss_pipeline(c, dlss_motion_spv, sizeof dlss_motion_spv, 72, &DL.pm.layout, &DL.pm.pipe, &DL.dsl3) ||
        !dlss_pipeline(c, dlss_output_spv, sizeof dlss_output_spv, 0, &DL.po.layout, &DL.po.pipe, &DL.dsl) ||
        !dlss_pipeline(c, dlss_mask_spv, sizeof dlss_mask_spv, 76, &DL.pk.layout, &DL.pk.pipe, &DL.dsl3)) {
        rt_log("dlss: the motion, colour or output pass could not be created: the stand-in stays");
        return 0;
    }
    DL.init = 1;
    return 1;
}

static int dlss_image(typeof(DL.packed) *o, uint32_t w, uint32_t h, VkFormat f, VkImageUsageFlags usage) {
    VkCtx *c = vk_ctx();
    if (o->img) vk_defer_view(o->v), vk_defer_image(o->img, o->a);
    memset(o, 0, sizeof *o);
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = f, .extent = {w, h, 1},
                            .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ii, &ai, &o->img, &o->a, NULL) != VK_SUCCESS) { memset(o, 0, sizeof *o); return 0; }
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = o->img, .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                .format = f, .subresourceRange = {f == VK_FORMAT_D32_SFLOAT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkCreateImageView(c->device, &vi, NULL, &o->v) != VK_SUCCESS) { vmaDestroyImage(c->vma, o->img, o->a); memset(o, 0, sizeof *o); return 0; }
    return 1;
}

static int dlss_images(uint32_t rw, uint32_t rh, uint32_t ow, uint32_t oh) {
    if (DL.packed.img && DL.rw == rw && DL.rh == rh && DL.ow == ow && DL.oh == oh) return 1;
    const VkImageUsageFlags st = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    DL.rw = DL.rh = DL.ow = DL.oh = 0;
    if (!dlss_image(&DL.packed, rw, rh, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT) ||
        !dlss_image(&DL.color, rw, rh, VK_FORMAT_R16G16B16A16_SFLOAT, st) || !dlss_image(&DL.motion, rw, rh, VK_FORMAT_R16G16_SFLOAT, st) ||
        !dlss_image(&DL.out, ow, oh, VK_FORMAT_R16G16B16A16_SFLOAT, st | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT) ||
        !dlss_image(&DL.mask, rw, rh, VK_FORMAT_R8_UNORM, st) ||
        !dlss_image(&DL.pdepth, rw, rh, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT) ||
        !dlss_image(&DL.pcolor, rw, rh, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT))
        return 0;
    DL.pdepth_epoch = 0;
    DL.rw = rw, DL.rh = rh, DL.ow = ow, DL.oh = oh;
    return 1;
}

static VkImageMemoryBarrier2 dlss_ib(VkImage img, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to,
                                     VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds, VkAccessFlags2 da);

// DLSS to the window's size when it is 3840x2160. 0 (the 960p feature and FSR) with
// the reason logged when it changes: another window size, the feature or the images failed.
static int dlss_4k_for(uint32_t *w, uint32_t *h) {
    static const float scale = 1.0f;   // DLSS's output as a share of the window, the composite scales the rest
    static int told;
    if (!told) {
        told = 1;
        rt_log("dlss: DLSS to the window's size (3840x2160 only), a composite with the game frame's "
               "HUD in place of FSR");
    }
    if (DL.k4_broken) return 0;
    uint32_t sw = 0, sh = 0;
    vk_present_extent(&sw, &sh);
    const char *why = sw != 3840 || sh != 2160 ? "the window is not 3840x2160" : NULL;
    if (why != DL.k4_reason) {
        if (why) rt_log("dlss: 4K off for now (%s, %ux%u): the 960p feature and FSR", why, sw, sh);
        else rt_log("dlss: 4K on: the game's scene (1280x720 as shipped) -> %ux%u", sw, sh);
        DL.k4_reason = why;
    }
    if (why) return 0;
    DL.k4sw = sw, DL.k4sh = sh;
    float sc = scale;
    *w = ((uint32_t)(sw * sc) + 1) & ~1u, *h = ((uint32_t)(sh * sc) + 1) & ~1u;
    return 1;
}
static int dlss_images4k(uint32_t w, uint32_t h, uint32_t ow, uint32_t oh) {
    if (DL.k4w == w && DL.k4h == h && DL.k4ow == ow && DL.k4oh == oh) return 1;
    const VkImageUsageFlags st = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    DL.k4w = DL.k4h = DL.k4ow = DL.k4oh = 0;
    DL.k4_ep[0] = DL.k4_ep[1] = 0;
    for (int i = 0; i < 2; i++)
        if (!dlss_image(&DL.out4k[i], w, h, VK_FORMAT_R16G16B16A16_SFLOAT, st) ||
            !dlss_image(&DL.ref[i], ow, oh, VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT))
            return 0;
    DL.k4w = w, DL.k4h = h, DL.k4ow = ow, DL.k4oh = oh;
    return 1;
}
static void dlss_4k_off(const char *why) {
    DL.k4_broken = 1;
    rt_log("dlss: 4K off for this run (%s): the 960p feature and FSR", why);
}

static uint32_t dl_calls_fresh, dl_calls_stale, dl_calls_nosrc;   // dlss_upscale calls since the last present was queued
// The pair of the newest 4K upscale, once: the present queued next goes with it (vk_backend.c present_image).
uint64_t vk_dlss4k_take(void) {
    uint64_t t = __atomic_load_n(&DL.k4_token, __ATOMIC_ACQUIRE);
    if (!t || t == DL.k4_taken) {
        // a present with no upscale of its own since the last one: FSR draws the game frame (diagnostic line, 400 at most)
        static uint32_t n_none;
        uint32_t cf = __atomic_exchange_n(&dl_calls_fresh, 0, __ATOMIC_RELAXED), cs = __atomic_exchange_n(&dl_calls_stale, 0, __ATOMIC_RELAXED),
                 cn = __atomic_exchange_n(&dl_calls_nosrc, 0, __ATOMIC_RELAXED);
        if (t && n_none++ < 400)
            rt_log("dlss: 4K: no-upscale f%llu: the present has no new 4K upscale (the last was %llu; %u so far; upscale stages since the "
                   "last present: %u with a new source, %u with the same, %u without; last miss: %s): FSR draws the game frame",
                   (unsigned long long)vk_frame_no, (unsigned long long)t, n_none, cf, cs, cn, DL.reason ? DL.reason : "none");
        return 0;
    }
    __atomic_store_n(&dl_calls_fresh, 0, __ATOMIC_RELAXED), __atomic_store_n(&dl_calls_stale, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&dl_calls_nosrc, 0, __ATOMIC_RELAXED);
    DL.k4_taken = t;
    return t;
}

// The composite (present thread, vk_present's command buffer): pipeline, its output image and a view per present slot.
static struct {
    int tried, ok, logged;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipe;
    VkSampler lin;
    struct { VkImage img; VmaAllocation a; VkImageView v; } outimg;
    uint32_t w, h;
    VkImageView src_view[8];
    const char *why;
} K4;
static const float k4_lo = 0.025f, k4_hi = 0.12f;   // coverage ramp of the difference

static int k4_init(VkCtx *c) {
    K4.tried = 1;
    VkDescriptorSetLayoutBinding b[4];
    for (uint32_t i = 0; i < 4; i++)
        b[i] = (VkDescriptorSetLayoutBinding){.binding = i, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .descriptorType = i == 1 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, .bindingCount = 4, .pBindings = b};
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR,
                              .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST, .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
    if (vkCreateDescriptorSetLayout(c->device, &dci, NULL, &K4.dsl) != VK_SUCCESS || vkCreateSampler(c->device, &si, NULL, &K4.lin) != VK_SUCCESS ||
        !dlss_pipeline(c, dlss_4k_composite_spv, sizeof dlss_4k_composite_spv, 28, &K4.layout, &K4.pipe, &K4.dsl)) {
        rt_log("dlss: the 4K composite could not be created: FSR stays");
        return 0;
    }
    return K4.ok = 1;
}

static int k4_skip(const char *why) {
    if (why != K4.why) rt_log("dlss: the 4K composite skips a present (%s): FSR draws it", why), K4.why = why;
    return 0;
}

// 1: the window image of this present is DLSS's 4K scene plus what the game drew over its upscale (HUD, post effects),
// recorded into cb and blitted into dst's out rectangle (TRANSFER_DST); 0: FSR or the blit draws it as before.
int vk_dlss4k_composite(VkCommandBuffer cb, unsigned slot, uint64_t token, VkImage src, VkFormat fmt, int sw, int sh,
                        const int crop[4], VkImage dst, const int out[4]) {
    if (!token) return 0;
    int i = (int)(token & 1);
    if (DL.k4_ep[i] != token) return k4_skip("its 4K image was overwritten");
    if ((uint32_t)crop[2] != DL.k4ow || (uint32_t)crop[3] != DL.k4oh) return k4_skip("the game frame is not the upscale's size");
    if (out[0] || out[1] || (uint32_t)out[2] != DL.k4sw || (uint32_t)out[3] != DL.k4sh) return k4_skip("the picture does not fill the window");
    VkCtx *c = vk_ctx();
    if (!K4.tried) k4_init(c);
    if (!K4.ok) return 0;
    if (K4.w != DL.k4sw || K4.h != DL.k4sh) {
        if (!dlss_image((typeof(DL.packed) *)&K4.outimg, DL.k4sw, DL.k4sh, VK_FORMAT_R16G16B16A16_SFLOAT,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
            K4.w = K4.h = 0;
            return k4_skip("no composite image");
        }
        K4.w = DL.k4sw, K4.h = DL.k4sh;
    }
    // the game frame through a view of its own format; an sRGB one reads decoded (as the blit), a UNORM one holds coded values
    int srgb = fmt == VK_FORMAT_R8G8B8A8_SRGB || fmt == VK_FORMAT_B8G8R8A8_SRGB || fmt == VK_FORMAT_A8B8G8R8_SRGB_PACK32;
    int coded = !srgb && (fmt == VK_FORMAT_R8G8B8A8_UNORM || fmt == VK_FORMAT_B8G8R8A8_UNORM || fmt == VK_FORMAT_A8B8G8R8_UNORM_PACK32 ||
                          fmt == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || fmt == VK_FORMAT_A2R10G10B10_UNORM_PACK32);
    if (!srgb && !coded) return k4_skip("the game frame's format is not 8-bit or 10-bit RGBA");
    slot %= 8;
    if (K4.src_view[slot]) vkDestroyImageView(c->device, K4.src_view[slot], NULL), K4.src_view[slot] = VK_NULL_HANDLE;
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = src, .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                .format = fmt, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkCreateImageView(c->device, &vi, NULL, &K4.src_view[slot]) != VK_SUCCESS) {
        K4.src_view[slot] = VK_NULL_HANDLE;
        return k4_skip("no view of the game frame");
    }
    if (!K4.logged++)
        rt_log("dlss: 4K composite: game frame %dx%d (format %d, %s), crop %dx%d at %d,%d, DLSS %ux%u, window %ux%u; ramp %.3f..%.3f",
               sw, sh, (int)fmt, coded ? "sRGB-coded values" : "decoded sRGB", crop[2], crop[3], crop[0], crop[1], DL.k4w, DL.k4h,
               DL.k4sw, DL.k4sh, k4_lo, k4_hi);
    const VkPipelineStageFlags2 ALL = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, CS = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                XF = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    const VkImageLayout G = VK_IMAGE_LAYOUT_GENERAL;
    const VkImageAspectFlags C = VK_IMAGE_ASPECT_COLOR_BIT;
    const VkAccessFlags2 SR = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, MW = VK_ACCESS_2_MEMORY_WRITE_BIT;
    // the replay's writes (an earlier submission on this queue) become visible; last present's blit read the output first
    VkImageMemoryBarrier2 ib[4] = {
        dlss_ib(src, C, G, G, ALL, MW, CS, SR), dlss_ib(DL.out4k[i].img, C, G, G, ALL, MW, CS, SR),
        dlss_ib(DL.ref[i].img, C, G, G, ALL, MW, CS, SR),
        dlss_ib(K4.outimg.img, C, VK_IMAGE_LAYOUT_UNDEFINED, G, XF | CS, 0, CS, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT)};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 4, .pImageMemoryBarriers = ib};
    vkCmdPipelineBarrier2(cb, &di);
    VkDescriptorImageInfo ii[4] = {{K4.lin, DL.out4k[i].v, G}, {VK_NULL_HANDLE, K4.outimg.v, G}, {K4.lin, DL.ref[i].v, G},
                                   {K4.lin, K4.src_view[slot], G}};
    VkWriteDescriptorSet wr[4];
    for (uint32_t k = 0; k < 4; k++)
        wr[k] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = k, .descriptorCount = 1,
                                       .descriptorType = k == 1 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                       .pImageInfo = &ii[k]};
    struct { float rect[4], lo, hi; uint32_t flags; } pc = {
        {(float)crop[0] / (float)sw, (float)crop[1] / (float)sh, (float)crop[2] / (float)sw, (float)crop[3] / (float)sh},
        k4_lo, k4_hi, (uint32_t)coded};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, K4.pipe);
    rec_push(vk_pipeline_push_fn(), cb, VK_PIPELINE_BIND_POINT_COMPUTE, K4.layout, 0, 4, wr);
    vkCmdPushConstants(cb, K4.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(cb, (DL.k4sw + 7) / 8, (DL.k4sh + 7) / 8, 1);
    ib[0] = dlss_ib(K4.outimg.img, C, G, G, CS, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, XF, VK_ACCESS_2_TRANSFER_READ_BIT);
    ib[1] = dlss_ib(src, C, G, G, CS, 0, ALL, 0);   // later writes of the game frame wait for this read
    di.imageMemoryBarrierCount = 2;
    vkCmdPipelineBarrier2(cb, &di);
    VkImageBlit region = {.srcSubresource = {C, 0, 0, 1}, .srcOffsets = {{0, 0, 0}, {(int)DL.k4sw, (int)DL.k4sh, 1}},
                          .dstSubresource = {C, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {(int)DL.k4sw, (int)DL.k4sh, 1}}};
    vkCmdBlitImage(cb, K4.outimg.img, G, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_NEAREST);
    K4.why = NULL;
    return 1;
}

static VkImageMemoryBarrier2 dlss_ib(VkImage img, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to, VkPipelineStageFlags2 ss,
                                     VkAccessFlags2 sa, VkPipelineStageFlags2 ds, VkAccessFlags2 da) {
    return (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = ss, .srcAccessMask = sa,
        .dstStageMask = ds, .dstAccessMask = da, .oldLayout = from, .newLayout = to, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img, .subresourceRange = {aspect, 0, 1, 0, 1}};
}
static void dlss_barrier(VkExec *ex, const VkImageMemoryBarrier2 *ib, uint32_t n, int memory) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = memory ? 1 : 0, .pMemoryBarriers = &mb,
                           .imageMemoryBarrierCount = n, .pImageMemoryBarriers = ib};
    vkCmdPipelineBarrier2(ex->cb, &di);
    ex->recorded++;
}
// in -> binding 0, out (storage) -> 1, extra[i] (sampled, GENERAL) -> 2 + i.
static void dlss_pass3(VkExec *ex, VkPipelineLayout layout, VkPipeline pipe, VkImageView in, VkImageLayout in_layout, VkImageView out,
                       const VkImageView *extra, uint32_t nextra, const void *push, uint32_t push_bytes, uint32_t w, uint32_t h) {
    VkDescriptorImageInfo ii[2 + 4] = {{DL.sampler, in, in_layout}, {VK_NULL_HANDLE, out, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet wr[2 + 4];
    if (nextra > 4) nextra = 4;
    for (uint32_t i = 0; i < 2 + nextra; i++) {
        if (i >= 2) ii[i] = (VkDescriptorImageInfo){DL.sampler, extra[i - 2], VK_IMAGE_LAYOUT_GENERAL};
        wr[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = i, .descriptorCount = 1,
                                       .descriptorType = i == 1 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                       .pImageInfo = &ii[i]};
    }
    vkCmdBindPipeline(ex->cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    B.pipe[1] = VK_NULL_HANDLE;   // not the game's: the next dispatch binds its own again (its sets are pushed every time)
    rec_push(vk_pipeline_push_fn(), ex->cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 2 + nextra, wr);
    if (push_bytes) vkCmdPushConstants(ex->cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
    vkCmdDispatch(ex->cb, (w + 7) / 8, (h + 7) / 8, 1);
    ex->recorded += 3;
}
static void dlss_pass(VkExec *ex, VkPipelineLayout layout, VkPipeline pipe, VkImageView in, VkImageLayout in_layout, VkImageView out,
                      const void *push, uint32_t push_bytes, uint32_t w, uint32_t h) {
    dlss_pass3(ex, layout, pipe, in, in_layout, out, NULL, 0, push, push_bytes, w, h);
}
static VkImageView dlss_view(VkImage img, VkFormat f, VkImageAspectFlags aspect) {
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                .format = f, .subresourceRange = {aspect, 0, 1, 0, 1}};
    VkImageView v = VK_NULL_HANDLE;
    if (vkCreateImageView(vk_ctx()->device, &vi, NULL, &v) != VK_SUCCESS) return VK_NULL_HANDLE;
    vk_defer_view(v);   // freed once the GPU is past this submission
    return v;
}

// 1: DLSS drew this upscale; 0: the stand-in draws it.
int dlss_upscale(VkExec *ex) {
    if (!vk_dlss_on()) return 0;
    if (!up_src.image || !up_src.fresh) {   // counted for vk_dlss4k_take's diagnostic line
        __atomic_fetch_add(up_src.image ? &dl_calls_stale : &dl_calls_nosrc, 1, __ATOMIC_RELAXED);
        return 0;
    }
    __atomic_fetch_add(&dl_calls_fresh, 1, __ATOMIC_RELAXED);
    uint64_t e = DL.epoch++;
    if (DL.cam_epoch != e) return dlss_fail("no scene camera this frame");
    if (!DL.depth) return dlss_fail("no scene depth this frame");
    if (up_src.w != DL.cw || up_src.h != DL.ch) return dlss_fail("colour and camera of different sizes");
    // The colour: the packing pass's sqrt-coded RGB9E5 (an RGB9E5 image, or an R11G11B10 one it wrote through an RGB9E5 view).
    int rgb9e5 = up_src.packed && up_src.format == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
    vk_texture_lock();
    if (!rgb9e5 && up_src.packed && up_src.format == VK_FORMAT_B10G11R11_UFLOAT_PACK32)
        for (int slot = 0; slot < 32 && !rgb9e5; slot++) {
            uint64_t handle = ex->tex[NVN_STAGE_CS][slot];
            view_t view;
            int has_view = 0;
            const VkTexImage *in = handle ? vk_texture_by_id_locked(&ex->tex_pool, (int)(handle & 0xfffff), &view, &has_view) : NULL;
            rgb9e5 = in && in->addr == up_src.addr && in->width == up_src.w && in->height == up_src.h &&
                     ((has_view && view.set_format && view.format == NVN_FORMAT_RGB9E5F) || in->format == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32);
        }
    const VkTexImage *t = largest_storage_image(ex);
    VkImage dst = t ? t->image : VK_NULL_HANDLE;
    uint32_t ow = t ? t->width : 0, oh = t ? t->height : 0;
    int target_ok = t && t->format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 && t->image != up_src.image && t->samples <= 1 &&
                    llabs((long long)ow * up_src.h - (long long)oh * up_src.w) <= (long long)up_src.w;
    // The game's upscale target has its input's shape (within a pixel) and is at least as large; the 4K path draws the
    // window's 3840x2160 either way.
    int storage = t && (t->usage & VK_IMAGE_USAGE_STORAGE_BIT);
    if (rgb9e5 && target_ok && storage) vk_texture_storage_bound_locked(t);   // the GPU writes it (what later samples must see)
    vk_texture_unlock();
    {   // the game's scene and upscale target sizes, once (a mod pack's internal resolution changes the first)
        static uint32_t told_w, told_h;
        if (told_w != up_src.w || told_h != up_src.h) {
            told_w = up_src.w, told_h = up_src.h;
            rt_log("dlss: the game's scene %ux%u, its upscale target %ux%u%s", up_src.w, up_src.h, ow, oh, target_ok ? "" : " (not used: not its shape)");
        }
    }
    if (!rgb9e5) return dlss_fail("the upscale's colour is not the packed RGB9E5");
    if (!target_ok) return dlss_fail("unexpected upscale images");
    if (!storage) return dlss_fail("unsupported storage format");
    float m[16];
    if (!dlss_clip_to_previous(DL.cur, DL.prev, m)) return dlss_fail("camera matrix not invertible");
    if (!dlss_pipelines()) return dlss_fail("no motion pass");
    if (!dlss_images(up_src.w, up_src.h, ow, oh)) return dlss_fail("DLSS images could not be made");
    uint32_t k4w = 0, k4h = 0;
    int k4 = dlss_4k_for(&k4w, &k4h);
    if (k4 && !dlss_images4k(k4w, k4h, ow, oh)) k4 = 0, dlss_4k_off("its images could not be made");
    const int bi = (int)(e & 1);   // this upscale's pair
    VkImageView depth_view = dlss_view(DL.depth, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT);
    VkImageView dst_view = dlss_view(dst, VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_IMAGE_ASPECT_COLOR_BIT);
    if (!depth_view || !dst_view) return dlss_fail("no upscale target view");
    dlss_jitter_frame();
    int reset = !DL.last_ok || DL.last_ok + 1 != e;
    for (int i = 0; i < 16 && !reset; i++) reset = fabsf(DL.prev[i] - DL.last_cur[i]) >= 0.001f;
    // DLSS's BiasCurrentColor mask (dlss_mask.comp) from last frame's depth and colour, mode 3: depth test (disocclusion,
    // 0.1 relative error) and colour test (last frame's colour reprojected outside this frame's 3x3 box, 0.05 sqrt-coded).
    // Only when last frame's copies are the previous upscale's.
    static const int mask_on = 3;
    static const float mask_t = 0.1f, mask_c = 0.05f;
    static int mask_told;
    if (!mask_told) {
        mask_told = 1;
        rt_log("dlss: bias mask on (depth + colour; depth %.3f of the distance, colour %.3f)", mask_t, mask_c);
    }
    int use_mask = mask_on && !reset && DL.pdepth_epoch && DL.pdepth_epoch + 1 == e;
    static const float mv0_t = 0.05f;   // the motion pass's threshold (its mode 1, zero motion, is not used)
    int use_prev = mask_on && !reset && DL.pdepth_epoch && DL.pdepth_epoch + 1 == e;

    const VkPipelineStageFlags2 ALL = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, CS = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                CP = VK_PIPELINE_STAGE_2_COPY_BIT;
    const VkAccessFlags2 RW = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, SR = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                         SW = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    const VkImageLayout G = VK_IMAGE_LAYOUT_GENERAL, U = VK_IMAGE_LAYOUT_UNDEFINED, RO = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    const VkImageAspectFlags C = VK_IMAGE_ASPECT_COLOR_BIT, D = VK_IMAGE_ASPECT_DEPTH_BIT;
    cb_res(ex);
    vk_exec_barrier(ex);   // ends the pass; the colour, the depth and the target are written
    VkImageMemoryBarrier2 ib[9] = {
        dlss_ib(DL.packed.img, C, U, G, ALL, 0, CP, VK_ACCESS_2_TRANSFER_WRITE_BIT),
        dlss_ib(DL.color.img, C, U, G, ALL, 0, CS, SW), dlss_ib(DL.motion.img, C, U, G, ALL, 0, CS, SW),
        dlss_ib(DL.out.img, C, U, G, ALL, 0, ALL, RW), dlss_ib(DL.mask.img, C, U, G, ALL, 0, CS, SW),
        // last frame's depth and colour: kept when the mask reads them, otherwise discarded (also their first use)
        dlss_ib(DL.pdepth.img, D, use_prev ? G : U, G, ALL, VK_ACCESS_2_MEMORY_WRITE_BIT, ALL, RW),
        dlss_ib(DL.pcolor.img, C, use_mask ? G : U, G, ALL, VK_ACCESS_2_MEMORY_WRITE_BIT, ALL, RW),
        dlss_ib(DL.out4k[bi].img, C, U, G, ALL, 0, ALL, RW), dlss_ib(DL.ref[bi].img, C, U, G, ALL, 0, ALL, RW)};
    dlss_barrier(ex, ib, k4 ? 9 : 7, 1);
    VkImageCopy copy = {.srcSubresource = {C, 0, 0, 1}, .dstSubresource = {C, 0, 0, 1}, .extent = {up_src.w, up_src.h, 1}};
    vkCmdCopyImage(ex->cb, up_src.image, G, DL.packed.img, G, 1, &copy);   // the bits as they are: no float conversion
    ex->recorded++;
    ib[0] = dlss_ib(DL.packed.img, C, G, G, CP, VK_ACCESS_2_TRANSFER_WRITE_BIT, CS, SR);
    dlss_barrier(ex, ib, 1, 0);
    dlss_pass(ex, DL.pc.layout, DL.pc.pipe, DL.packed.v, G, DL.color.v, NULL, 0, up_src.w, up_src.h);
    {   // the motion pass: binding 2 = last frame's depth (read in mode 1 only)
        float push[18];
        memcpy(push, m, sizeof m), push[16] = mv0_t;
        uint32_t mode = 0u;
        memcpy(&push[17], &mode, 4);
        VkImageView extra[1] = {DL.pdepth.v};
        dlss_pass3(ex, DL.pm.layout, DL.pm.pipe, depth_view, G, DL.motion.v, extra, 1, push, sizeof push, up_src.w, up_src.h);
    }
    if (use_mask) {   // after the colour pass, whose output the mask reads
        float push[19];
        memcpy(push, m, sizeof m), push[16] = mask_t, push[17] = mask_c;
        uint32_t mode = (uint32_t)mask_on;
        memcpy(&push[18], &mode, 4);
        VkImageMemoryBarrier2 cb = dlss_ib(DL.color.img, C, G, G, CS, SW, CS, SR);
        dlss_barrier(ex, &cb, 1, 0);
        VkImageView extra[3] = {DL.pdepth.v, DL.color.v, DL.pcolor.v};
        dlss_pass3(ex, DL.pk.layout, DL.pk.pipe, depth_view, G, DL.mask.v, extra, 3, push, sizeof push, up_src.w, up_src.h);
    }
    // DLSS reads its inputs as sampled images in SHADER_READ_ONLY_OPTIMAL and writes the output in GENERAL.
    ib[0] = dlss_ib(DL.color.img, C, G, RO, CS, SW, CS, SR);
    ib[1] = dlss_ib(DL.motion.img, C, G, RO, CS, SW, CS, SR);
    ib[2] = dlss_ib(DL.depth, D, G, RO, CS, SR, CS, SR);
    ib[3] = dlss_ib(DL.mask.img, C, G, RO, CS, SW, CS, SR);
    dlss_barrier(ex, ib, 4, 0);
    VkDlssImage in_c = {DL.color.img, DL.color.v, VK_FORMAT_R16G16B16A16_SFLOAT, up_src.w, up_src.h, C},
                in_d = {DL.depth, depth_view, VK_FORMAT_D32_SFLOAT, up_src.w, up_src.h, D},
                in_m = {DL.motion.img, DL.motion.v, VK_FORMAT_R16G16_SFLOAT, up_src.w, up_src.h, C},
                out = {DL.out.img, DL.out.v, VK_FORMAT_R16G16B16A16_SFLOAT, ow, oh, C},
                out4k = {DL.out4k[bi].img, DL.out4k[bi].v, VK_FORMAT_R16G16B16A16_SFLOAT, k4w, k4h, C},
                in_k = {DL.mask.img, DL.mask.v, VK_FORMAT_R8_UNORM, up_src.w, up_src.h, C};
    const char *why = NULL;
    rec_drain();   // NGX records into this command buffer itself: every call queued for the recorder goes first
    // the shift this upscale's scene draws got (dlss_jitter_for), as DLSS's offset
    float jit[2] = {DL.jx, DL.jy};
    const float *jp = DL.jit_epoch == e ? jit : NULL;
    int ok = vk_dlss_evaluate(ex->cb, &in_c, &in_d, &in_m, use_mask ? &in_k : NULL, k4 ? &out4k : &out, jp, reset, &why);
    if (!ok && k4 && why && !strncmp(why, "DLSS feature", 12)) {   // no 4K feature: the 960p one, for the rest of the run
        dlss_4k_off(why);
        k4 = 0, why = NULL;
        ok = vk_dlss_evaluate(ex->cb, &in_c, &in_d, &in_m, use_mask ? &in_k : NULL, &out, jp, reset, &why);
    }
    memset(&B, 0, sizeof B);   // NGX bound its own pipelines and descriptors: forget what was bound
    ib[0] = dlss_ib(DL.color.img, C, RO, G, ALL, SR, ALL, RW);
    ib[1] = dlss_ib(DL.motion.img, C, RO, G, ALL, SR, ALL, RW);
    ib[2] = dlss_ib(DL.depth, D, RO, G, ALL, SR, ALL, RW);   // the game's depth goes back to GENERAL either way
    ib[3] = dlss_ib(DL.mask.img, C, RO, G, ALL, SR, ALL, RW);
    dlss_barrier(ex, ib, 4, 1);
    DL.pdepth_epoch = 0;
    if (mask_on) {   // this frame's depth for next frame's mask (the barrier above made it readable)
        VkImageCopy dc = {.srcSubresource = {D, 0, 0, 1}, .dstSubresource = {D, 0, 0, 1}, .extent = {up_src.w, up_src.h, 1}};
        vkCmdCopyImage(ex->cb, DL.depth, G, DL.pdepth.img, G, 1, &dc);
        VkImageCopy cc = {.srcSubresource = {C, 0, 0, 1}, .dstSubresource = {C, 0, 0, 1}, .extent = {up_src.w, up_src.h, 1}};
        vkCmdCopyImage(ex->cb, DL.color.img, G, DL.pcolor.img, G, 1, &cc);
        ex->recorded += 2;
        DL.pdepth_epoch = e;
    }
    if (!ok) {
        return dlss_fail(why ? why : "DLSS declined the frame");
    }
    dlss_pass(ex, DL.po.layout, DL.po.pipe, k4 ? DL.out4k[bi].v : DL.out.v, G, dst_view, NULL, 0, ow, oh);
    dlss_barrier(ex, NULL, 0, 1);
    if (k4) {   // the reference: the game's target as DLSS left it, before the game draws over it
        VkImageCopy rc = {.srcSubresource = {C, 0, 0, 1}, .dstSubresource = {C, 0, 0, 1}, .extent = {ow, oh, 1}};
        vkCmdCopyImage(ex->cb, dst, G, DL.ref[bi].img, G, 1, &rc);
        ex->recorded++;
        dlss_barrier(ex, NULL, 0, 1);
        DL.k4_ep[bi] = e;
        __atomic_store_n(&DL.k4_token, e, __ATOMIC_RELEASE);
    }
    DL.jdepth = DL.depth;   // the next frame's scene draws into this depth may be shifted
    up_src.fresh = 0;
    DL.last_ok = e;
    memcpy(DL.last_cur, DL.cur, sizeof DL.cur);
    DL.reason = NULL;
    if (!(DL.drawn++ % 600))
        rt_log("dlss: DLSS drew upscale %llu, %ux%u -> %ux%u (%llu drawn so far%s%s)", (unsigned long long)e, up_src.w, up_src.h,
               k4 ? k4w : ow, k4 ? k4h : oh, (unsigned long long)DL.drawn, reset ? ", history reset" : "", jp ? ", jittered" : "");
    return 1;
}
