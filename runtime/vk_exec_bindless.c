// The bindless set 2: texture and sampler pools indexed through handles read from a storage table.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "host.h"
#include "nvn_api.h"
#include "nvn_hmap.h"
#include "vk.h"
#include "vk_exec.h"
#include "vk_exec_draw_int.h"
#include "tex.h"

// Samples through a handle read from a storage table (translator patch tools/xlate/patches/0035): see README,
// "Bindless texture set".
enum { BL_TEX = 32768, BL_SAMP = 4096, BL_KINDS = 8, BL_SAMP_BINDING = 8 };
static VkDescriptorSetLayout bl_layout, bl_empty;
static VkDescriptorSet bl_set;
static int bl_ready;                          // bl_init: 1 ready, -1 failed
static VkImage bl_done_tex[BL_TEX];           // the image each texture id's descriptors were written for
// An image destroyed while ids of set 2 still named it (vk_exec_image_gone): those ids get the stand-ins at once and
// every table is rescanned. bl_by_image: VkImage -> id + 1, or BL_MULTI when written for more than one id.
static nvn_hmap bl_by_image;
static uint32_t bl_gen;                       // bumped by bl_image_gone: tables scanned at another generation scan again
static uint64_t bl_gone_ids, bl_gone_images;
#define BL_MULTI ((void *)(uintptr_t)UINT32_MAX)
// A ring of BL_RING copies of set 2: changes are queued (bl_put) and committed at the next bind into a copy no pending
// submission bound, so no set is written while the GPU may read it. See README, "Bindless texture set".
enum { BL_RING_MAX = 16 };
enum { BL_RING = 16 };
static int bl_sync = 1;                      // 1: the ring; 0: no pool for the copies (bl_init), one set
static VkDescriptorSet bl_sets[BL_RING_MAX];
static uint64_t bl_used[BL_RING_MAX];             // timeline value of the latest submission that bound the copy
static int bl_cur;
typedef struct { uint32_t binding, elem; VkImageView view; VkSampler samp; } BlPut;
static BlPut *bl_q;
static int bl_nq, bl_capq, bl_qreset;         // queued changes; bl_qreset: one of them is a destroyed image's stand-in
static uint64_t bl_commits, bl_waits, bl_direct;
static const int bl_dummy[BL_KINDS] = {DV_2D, DV_2D_ARRAY, DV_CUBE, DV_3D, DV_2D, DV_2D_ARRAY, DV_CUBE, DV_CUBE_ARRAY};

static VkDescriptorSetLayout bl_once(VkDescriptorSetLayout *slot, const VkDescriptorSetLayoutCreateInfo *ci) {
    VkDescriptorSetLayout have = __atomic_load_n(slot, __ATOMIC_ACQUIRE), made = VK_NULL_HANDLE;
    if (have) return have;
    if (vkCreateDescriptorSetLayout(vk_ctx()->device, ci, NULL, &made) != VK_SUCCESS) return VK_NULL_HANDLE;
    if (__atomic_compare_exchange_n(slot, &have, made, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) return made;
    vkDestroyDescriptorSetLayout(vk_ctx()->device, made, NULL);
    return have;
}

VkDescriptorSetLayout vk_bindless_empty(void) {
    VkDescriptorSetLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    return bl_once(&bl_empty, &ci);
}

VkDescriptorSetLayout vk_bindless_layout(void) {
    if (__atomic_load_n(&bl_layout, __ATOMIC_ACQUIRE)) return bl_layout;
    VkDescriptorSetLayoutBinding b[BL_KINDS + 1];
    VkDescriptorBindingFlags f[BL_KINDS + 1];
    for (uint32_t i = 0; i <= BL_KINDS; i++) {
        int samp = i == BL_SAMP_BINDING;
        b[i] = (VkDescriptorSetLayoutBinding){i, samp ? VK_DESCRIPTOR_TYPE_SAMPLER : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                              samp ? BL_SAMP : BL_TEX, VK_SHADER_STAGE_ALL, NULL};
        f[i] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
               VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT;
    }
    VkDescriptorSetLayoutBindingFlagsCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
                                                      .bindingCount = BL_KINDS + 1, .pBindingFlags = f};
    VkDescriptorSetLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .pNext = &fi,
                                          .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
                                          .bindingCount = BL_KINDS + 1, .pBindings = b};
    return bl_once(&bl_layout, &ci);
}

static int bl_init(void) {
    if (bl_ready) return bl_ready > 0;
    bl_ready = -1;
    VkDevice dev = vk_ctx()->device;
    VkDescriptorSetLayout l = vk_bindless_layout();
    if (!l) return rt_log("vk: bindless: no set layout"), 0;
    uint32_t nsets = bl_sync == 1 ? BL_RING : 1;
    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, BL_TEX * BL_KINDS * nsets}, {VK_DESCRIPTOR_TYPE_SAMPLER, BL_SAMP * nsets}};
    VkDescriptorPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                     .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT, .maxSets = nsets, .poolSizeCount = 2, .pPoolSizes = ps};
    VkDescriptorPool pool;
    if (vkCreateDescriptorPool(dev, &pi, NULL, &pool) != VK_SUCCESS) {
        if (nsets == 1) return rt_log("vk: bindless: no pool"), 0;
        rt_log("vk: bindless: no pool for %u copies: one set, the old way", nsets);
        bl_sync = 0, nsets = 1, ps[0].descriptorCount = BL_TEX * BL_KINDS, ps[1].descriptorCount = BL_SAMP, pi.maxSets = 1;
        if (vkCreateDescriptorPool(dev, &pi, NULL, &pool) != VK_SUCCESS) return rt_log("vk: bindless: no pool"), 0;
    }
    VkDescriptorSetLayout ls[BL_RING_MAX];
    for (int i = 0; i < BL_RING_MAX; i++) ls[i] = l;
    VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = pool,
                                      .descriptorSetCount = nsets, .pSetLayouts = ls};
    if (vkAllocateDescriptorSets(dev, &ai, bl_sets) != VK_SUCCESS) return rt_log("vk: bindless: no set"), 0;
    bl_set = bl_sets[0], bl_cur = 0;
    // Stand-ins everywhere: a handle whose texture was not written yet reads zero instead of an unwritten descriptor.
    const int *dummy = bl_dummy;
    VkDescriptorImageInfo *ii = malloc(sizeof *ii * BL_TEX);
    if (!ii) return 0;
    for (uint32_t k = 0; k <= BL_KINDS; k++) {
        int samp = k == BL_SAMP_BINDING;
        uint32_t n = samp ? BL_SAMP : BL_TEX;
        for (uint32_t i = 0; i < n; i++)
            ii[i] = samp ? (VkDescriptorImageInfo){.sampler = Z.samp_shadow}
                         : (VkDescriptorImageInfo){.imageView = Z.img[dummy[k]].view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = bl_set, .dstBinding = k, .descriptorCount = n,
                                  .descriptorType = samp ? VK_DESCRIPTOR_TYPE_SAMPLER : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = ii};
        for (uint32_t c = 0; c < nsets; c++) w.dstSet = bl_sets[c], vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);
    }
    free(ii);
    bl_ready = 1;
    rt_log("vk: bindless: mode %d: %s", bl_sync, bl_sync == 1 ? "changes go into a copy no pending submission bound (ring of 4)"
                                                                     : "changes written into the bound set at once");
    nvn_hm_init(&bl_by_image, sizeof(VkImage));
    rt_log("vk: bindless: set 2 ready (%d textures, %d samplers)", BL_TEX, BL_SAMP);
    return 1;
}

// One descriptor of set 2 (texture lock held): written into the bound set at once, or queued for bl_commit.
static void bl_put(uint32_t binding, uint32_t elem, VkImageView view, VkSampler samp, int reset) {
    if (bl_sync != 1) {
        VkDescriptorImageInfo ii = {.sampler = samp, .imageView = view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = bl_set, .dstBinding = binding,
                                  .dstArrayElement = elem, .descriptorCount = 1,
                                  .descriptorType = samp ? VK_DESCRIPTOR_TYPE_SAMPLER : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                  .pImageInfo = &ii};
        vkUpdateDescriptorSets(vk_ctx()->device, 1, &w, 0, NULL);
        return;
    }
    if (reset) {   // a destroyed image's views may go before the commit: its queued writes are dropped (order kept)
        int k = 0;
        for (int i = 0; i < bl_nq; i++)
            if (bl_q[i].binding != binding || bl_q[i].elem != elem) bl_q[k++] = bl_q[i];
        bl_nq = k;
    }
    if (bl_nq == bl_capq) {
        bl_capq = bl_capq ? 2 * bl_capq : 256;
        if (!(bl_q = realloc(bl_q, sizeof *bl_q * (size_t)bl_capq))) rt_fatal("vk: out of memory for bindless changes");
    }
    bl_q[bl_nq++] = (BlPut){binding, elem, view, samp};
    bl_qreset |= reset;
}

// Writes the queued changes into set (texture lock held).
static void bl_write_queued(VkDescriptorSet set) {
    enum { CHUNK = 256 };
    VkWriteDescriptorSet w[CHUNK];
    VkDescriptorImageInfo ii[CHUNK];
    for (int i = 0; i < bl_nq; i += CHUNK) {
        int n = bl_nq - i < CHUNK ? bl_nq - i : CHUNK;
        for (int j = 0; j < n; j++) {
            const BlPut *q = &bl_q[i + j];
            ii[j] = (VkDescriptorImageInfo){.sampler = q->samp, .imageView = q->view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
            w[j] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = q->binding,
                                          .dstArrayElement = q->elem, .descriptorCount = 1,
                                          .descriptorType = q->samp ? VK_DESCRIPTOR_TYPE_SAMPLER : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                          .pImageInfo = &ii[j]};
        }
        vkUpdateDescriptorSets(vk_ctx()->device, (uint32_t)n, w, 0, NULL);
    }
}

static uint32_t bl_ids[BL_TEX], bl_nids, bl_ck_gen;
static uint8_t bl_known[BL_TEX];
static uint64_t bl_ck_runs, bl_ck_fixed;
static uint32_t bl_sids[BL_SAMP], bl_nsids;
static uint8_t bl_known_s[BL_SAMP];

// The recheck commits often, so a commit copies only the descriptors that can differ between ring copies: all start with
// the same stand-ins (bl_init) and only bl_put changes one, for an id the scan saw (bl_ids, bl_sids).
static void bl_copy_known(VkDescriptorSet src, VkDescriptorSet dst) {
    enum { CHUNK = 512 };
    VkCopyDescriptorSet cp[CHUNK];
    int n = 0;
    for (uint32_t k = 0; k <= BL_KINDS; k++) {
        int samp = k == BL_SAMP_BINDING;
        uint32_t cnt = samp ? bl_nsids : bl_nids;
        for (uint32_t i = 0; i < cnt; i++) {
            cp[n++] = (VkCopyDescriptorSet){.sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET, .srcSet = src, .srcBinding = k,
                                            .srcArrayElement = samp ? bl_sids[i] : bl_ids[i], .dstSet = dst, .dstBinding = k,
                                            .dstArrayElement = samp ? bl_sids[i] : bl_ids[i], .descriptorCount = 1};
            if (n == CHUNK) vkUpdateDescriptorSets(vk_ctx()->device, 0, NULL, (uint32_t)n, cp), n = 0;
        }
    }
    if (n) vkUpdateDescriptorSets(vk_ctx()->device, 0, NULL, (uint32_t)n, cp);
}

// Before a bind (texture lock held): the queued changes go into a copy no pending submission bound.
static void bl_commit(void) {
    if (bl_sync != 1 || !bl_nq) return;
    VkCtx *c = vk_ctx();
    uint64_t tl = 0;
    vkGetSemaphoreCounterValue(c->device, c->timeline, &tl);
    int j = -1;
    for (int i = 1; i < BL_RING && j < 0; i++)
        if (bl_used[(bl_cur + i) % BL_RING] <= tl) j = (bl_cur + i) % BL_RING;
    if (j < 0) {   // every other copy is still in use: wait for the next bind, unless a destroyed image is named
        bl_waits++;
        if (!bl_qreset) return;
        bl_direct++;
        bl_write_queued(bl_sets[bl_cur]);
        bl_nq = 0, bl_qreset = 0;
        return;
    }
    bl_copy_known(bl_sets[bl_cur], bl_sets[j]);
    bl_write_queued(bl_sets[j]);
    if (++bl_commits <= 16 || !(bl_commits & (bl_commits - 1)))
        rt_log("vk: bindless: %d changes into copy %d (copy %d last bound by %llu, timeline at %llu); %llu commits, "
               "%llu waited for a free copy, %llu written in place", bl_nq, j, bl_cur, (unsigned long long)bl_used[bl_cur],
               (unsigned long long)tl, (unsigned long long)bl_commits, (unsigned long long)bl_waits,
               (unsigned long long)bl_direct);
    bl_cur = j, bl_set = bl_sets[j];
    bl_nq = 0, bl_qreset = 0;
}

// bl_by_image upkeep when id tid changes from image old to image img (texture lock held, as bl_image_gone).
static void bl_map_set(uint32_t tid, VkImage old, VkImage img) {
    if (old && nvn_hm_get(&bl_by_image, &old) == (void *)(uintptr_t)(tid + 1)) nvn_hm_del(&bl_by_image, &old);
    void *have = nvn_hm_get(&bl_by_image, &img);
    if (!have) nvn_hm_put(&bl_by_image, &img, (void *)(uintptr_t)(tid + 1));
    else if (have != (void *)(uintptr_t)(tid + 1)) nvn_hm_put(&bl_by_image, &img, BL_MULTI);
}

static uint8_t bl_all[BL_TEX];
static void bl_put_all(uint32_t binding, uint32_t elem, VkImageView view);
static void bl_reset_id(uint32_t tid) {
    if (bl_all[tid]) {   // Written into every copy, so the stand-ins go into every copy, at once
        for (uint32_t k = 0; k < BL_KINDS; k++)
            if (bl_all[tid] >> k & 1) bl_put_all(k, tid, Z.img[bl_dummy[k]].view);
        bl_all[tid] = 0;
    }
    for (uint32_t k = 0; k < BL_KINDS; k++) bl_put(k, tid, Z.img[bl_dummy[k]].view, VK_NULL_HANDLE, 1);
    bl_done_tex[tid] = VK_NULL_HANDLE;
    bl_gone_ids++;
}

// vk_exec_image_gone (destroy_tex, texture lock held): the ids of set 2 still written for image get the stand-ins.
void bl_image_gone(VkImage image) {
    if (bl_ready <= 0 || !image || !bl_by_image.klen) return;
    void *v = nvn_hm_del(&bl_by_image, &image);
    if (!v) return;
    uint64_t before = bl_gone_ids;
    if (v == BL_MULTI) {
        for (uint32_t t = 1; t < BL_TEX; t++)
            if (bl_done_tex[t] == image) bl_reset_id(t);
    } else {
        uint32_t t = (uint32_t)(uintptr_t)v - 1;
        if (t < BL_TEX && bl_done_tex[t] == image) bl_reset_id(t);
    }
    if (bl_gone_ids == before) return;
    bl_gen++;
    if (++bl_gone_images <= 16 || !(bl_gone_images & (bl_gone_images - 1)))
        rt_log("vk: bindless: a destroyed image was still written for %llu id(s): stand-ins until the rescan (%llu images, %llu ids so far, frame %llu)",
               (unsigned long long)(bl_gone_ids - before), (unsigned long long)bl_gone_images, (unsigned long long)bl_gone_ids,
               (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED));
}

// Texture id tid of set 2 now names img (texture lock held): its descriptors for every image kind.
static void bl_write_id(VkExec *ex, uint32_t tid, const VkTexImage *img, const view_t *v, int hv) {
    static const uint16_t rtype[BL_KINDS] = {2, 0x102, 5, 4, 2 | 0x400, 0x102 | 0x400, 5 | 0x400, 0x105};
    bl_map_set(tid, bl_done_tex[tid], img->image);
    bl_done_tex[tid] = img->image;
    for (uint32_t k = 0; k < BL_KINDS; k++) {
        // a kind this image cannot serve gets the stand-in, not the previous image's view
        VkImageView view = k >= 4 && k < 7 && !(img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) ? VK_NULL_HANDLE
                         : get_view(ex, img, v, hv, rtype[k], 0);
        if (!view) view = Z.img[bl_dummy[k]].view;
        if (!view) continue;
        bl_put(k, tid, view, VK_NULL_HANDLE, 0);
    }
}

// vk_texture.c bl_dirty: ids of GPU-only textures the game registered are written as soon as the texture has an image.
// They are not "known" ids (a new one comes with every render target), so each goes into every copy of the ring at once.
static uint64_t bl_reg_written, bl_reg_writes;
static void bl_put_all(uint32_t binding, uint32_t elem, VkImageView view) {
    VkDescriptorImageInfo ii = {.imageView = view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[BL_RING_MAX];
    uint32_t n = bl_sync == 1 ? (uint32_t)BL_RING : 1;
    for (uint32_t c = 0; c < n; c++)
        w[c] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = bl_sync == 1 ? bl_sets[c] : bl_set,
                                      .dstBinding = binding, .dstArrayElement = elem, .descriptorCount = 1,
                                      .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &ii};
    vkUpdateDescriptorSets(vk_ctx()->device, n, w, 0, NULL);
    bl_reg_writes += n;
    int k = 0;   // a change queued for this descriptor before is older
    for (int i = 0; i < bl_nq; i++)
        if (bl_q[i].binding != binding || bl_q[i].elem != elem) bl_q[k++] = bl_q[i];
    bl_nq = k;
}
const VkTexImage *vk_texture_by_id_have_locked(const dpool_ref *pool, int id, view_t *view, int *has_view, int wait);
static void bl_registered(VkExec *ex) {   // texture lock held
    static const uint16_t rtype[BL_KINDS] = {2, 0x102, 5, 4, 2 | 0x400, 0x102 | 0x400, 5 | 0x400, 0x105};
    uint32_t ids[256];
    int n;
    while ((n = vk_texture_bl_dirty_locked(ids, 256)) > 0)
        for (int i = 0; i < n; i++) {
            uint32_t tid = ids[i];
            if (!tid || tid >= BL_TEX) continue;
            view_t v;
            int hv;
            const VkTexImage *img = vk_texture_by_id_have_locked(&ex->tex_pool, (int)tid, &v, &hv, 1);
            if (!img || bl_done_tex[tid] == img->image) continue;
            bl_map_set(tid, bl_done_tex[tid], img->image);
            bl_done_tex[tid] = img->image;
            uint8_t kinds = 0;
            for (uint32_t k = 0; k < BL_KINDS; k++) {
                VkImageView view = k >= 4 && k < 7 && !(img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) ? VK_NULL_HANDLE
                                 : get_view(ex, img, &v, hv, rtype[k], 0);
                if (view) kinds |= (uint8_t)(1u << k), bl_put_all(k, tid, view);
                else if (bl_all[tid] >> k & 1) bl_put_all(k, tid, Z.img[bl_dummy[k]].view);   // the image before served it
            }
            bl_all[tid] = kinds;
            bl_reg_written++;
        }
}

// The ids the material tables named are resolved again whenever a texture slot may resolve to another image
// (vk_texture_slots_gen), not only at the table scan: the game points the same ids at other transient targets per frame.
static void bl_know(uint32_t tid) {
    if (!bl_known[tid]) bl_known[tid] = 1, bl_ids[bl_nids++] = tid;
}
static void bl_recheck_ids(VkExec *ex) {
    static int logged;   // the line goes out once
    if (!logged) {
        logged = 1;
        rt_log("vk: bindless: known ids follow slot changes at once");
    }
    if (!bl_nids) return;
    uint32_t gen = vk_texture_slots_gen();   // before any lookup
    if (gen == bl_ck_gen) return;
    bl_ck_gen = gen, bl_ck_runs++;
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    vk_texture_lock();
    for (uint32_t i = 0; i < bl_nids; i++) {
        uint32_t tid = bl_ids[i];
        view_t v;
        int hv;
        const VkTexImage *img = vk_texture_by_id_locked(&ex->tex_pool, (int)tid, &v, &hv);
        if (!img || bl_done_tex[tid] == img->image) continue;
        if (++bl_ck_fixed <= 4000 || !(bl_ck_fixed & (bl_ck_fixed - 1)))
            rt_log("vk: bindless: recheck: id %u now %ux%u format %d (addr %#llx), frame %llu; %llu stale so far in %llu rechecks",
                   tid, img->width, img->height, (int)img->format, (unsigned long long)img->addr, (unsigned long long)frame,
                   (unsigned long long)bl_ck_fixed, (unsigned long long)bl_ck_runs);
        bl_write_id(ex, tid, img, &v, hv);
    }
    vk_texture_unlock();
}

static void bl_scan(VkExec *ex) {
    static struct { uint64_t addr, next; uint32_t gen; } tables[16];
    VkImage *done_tex = bl_done_tex;
    static VkSampler done_samp[BL_SAMP];
    static uint8_t seen_tex[BL_TEX], seen_samp[BL_SAMP];
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    const p_bind_buf *u = &ex->ssbo[NVN_STAGE_FS][3];
    if (!u->addr || u->size < 0x100000) return;   // the material table; smaller buffers in this slot hold other data
    int slot = 0;   // each table on its own clock: this slot holds other buffers in other draws
    while (slot < 15 && tables[slot].addr && tables[slot].addr != u->addr) slot++;
    if (tables[slot].addr == u->addr && frame < tables[slot].next && tables[slot].gen == bl_gen) return;
    uint64_t len = dyn_readable(u->addr, u->size);
    if (len < 0x100000) return;
    tables[slot].addr = u->addr, tables[slot].next = frame + 16, tables[slot].gen = bl_gen;
    memset(seen_tex, 0, sizeof seen_tex);
    memset(seen_samp, 0, sizeof seen_samp);
    const uint8_t *p = (const uint8_t *)(uintptr_t)u->addr;
    int new_tex = 0, new_samp = 0, handles = 0, uniq = 0, misses = 0;
    vk_texture_lock();
    for (uint64_t o = 16; o + 8 <= len; o += 8) {
        if ((o & 31) != 16 && (o & 31) != 24) continue;
        uint64_t h = *(const uint64_t *)(p + o);
        uint32_t tid = (uint32_t)(h & 0xfffff), sid = (uint32_t)(h >> 20 & 0xfff);
        if ((h >> 32) || !tid || tid >= BL_TEX) continue;
        handles++;
        if (!seen_tex[tid]) {
            seen_tex[tid] = 1;
            view_t v;
            int hv;
            const VkTexImage *img = vk_texture_by_id_locked(&ex->tex_pool, (int)tid, &v, &hv);
            uniq++;
            if (!img) misses++;
            if (img && done_tex[tid] != img->image && new_tex < 6)
                rt_log("vk: bindless: id %u %ux%u layers %u format %d view type %d (%s%s%s, addr %#llx)", tid, img->width, img->height,
                       img->layers, (int)img->format, (int)img->view_type, img->mirrored ? "mirrored" : "gpu-only",
                       img->gpu_owned ? ", gpu-written" : "", img->bound ? ", pool-bound" : "", (unsigned long long)img->addr);
            if (img && done_tex[tid] != img->image) {
                bl_write_id(ex, tid, img, &v, hv);
                new_tex++;
            }
            bl_know(tid);
        }
        if (!seen_samp[sid]) {
            seen_samp[sid] = 1;
            VkSampler s = vk_sampler_by_id_locked(&ex->samp_pool, (int)sid);
            if (!bl_known_s[sid]) bl_known_s[sid] = 1, bl_sids[bl_nsids++] = sid;
            if (s && done_samp[sid] != s) {
                done_samp[sid] = s;
                new_samp++;
                bl_put(BL_SAMP_BINDING, sid, VK_NULL_HANDLE, s, 0);
            }
        }
    }
    vk_texture_unlock();
    if (new_tex || new_samp)
        rt_log("vk: bindless: table %#llx+%#llx: %d handles, %d distinct ids (%d not in the pool, pool %u), %d textures and %d samplers written, frame %llu", (unsigned long long)u->addr,
               (unsigned long long)len, handles, uniq, misses, ex->tex_pool.count, new_tex, new_samp, (unsigned long long)frame);
}

// A graphics draw binds set 2 only when it could have changed or been disturbed: another pipeline layout than the last
// bind's, another copy of the set (a commit) or a new command buffer. bl_nq is peeked without the lock: a change queued
// right after the peek goes in at the next draw. Compute binds per dispatch: the runtime's own compute passes push other
// layouts in between.
static uint64_t bl_binds, bl_bind_skips;

void bindless_bind(VkExec *ex, const VkPipeInfo *info) {
    if (!Z.ready || !bl_init()) return;
    bl_scan(ex);
    if (__atomic_load_n(&vk_texture_bl_ndirty, __ATOMIC_ACQUIRE)) {   // registered GPU-only textures
        vk_texture_lock();
        bl_registered(ex);
        vk_texture_unlock();
    }
    bl_recheck_ids(ex);
    int gfx = info->bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS;
    VkDescriptorSet set = bl_set;
    if (bl_sync == 1 && (!gfx || __atomic_load_n(&bl_nq, __ATOMIC_RELAXED) || B.bl_set != set || bl_used[bl_cur] < ex->fence)) {
        vk_texture_lock();
        bl_commit();
        set = bl_set;
        if (bl_used[bl_cur] < ex->fence) bl_used[bl_cur] = ex->fence;
        vk_texture_unlock();
    }
    if (gfx) {
        if (B.bl_layout == info->layout && B.bl_set == set) { bl_bind_skips++; return; }
        B.bl_layout = info->layout, B.bl_set = set;
    }
    bl_binds++;
    vkCmdBindDescriptorSets(ex->cb, info->bind_point, info->layout, 2, 1, &set, 0, NULL);
}

// What set 2 holds for a texture id (diagnostic).
void bindless_debug_id(uint32_t tid, char *buf, size_t cap) {
    if (tid >= BL_TEX) { snprintf(buf, cap, "id out of range"); return; }
    snprintf(buf, cap, "set 2: image %p, %s, kinds written by registration %#x", (void *)bl_done_tex[tid], bl_known[tid] ? "known" : "not known",
             bl_all[tid]);
}

void bindless_bind_stats(uint64_t *binds, uint64_t *skips) { *binds = bl_binds, *skips = bl_bind_skips, bl_binds = bl_bind_skips = 0; }

// Since the last call: commits into a new copy, binds that found no free copy, ids the recheck re-pointed; the ring size.
void bindless_ring_stats(uint64_t out[4]) {
    static uint64_t c0, w0, f0;
    out[0] = bl_commits - c0, out[1] = bl_waits - w0, out[2] = bl_ck_fixed - f0, out[3] = (uint64_t)BL_RING;
    if (bl_reg_written)
        rt_log("perf: bindless: %llu registered GPU-only texture ids written since the last line (%llu descriptor writes)",
               (unsigned long long)bl_reg_written, (unsigned long long)bl_reg_writes);
    bl_reg_written = bl_reg_writes = 0;
    c0 = bl_commits, w0 = bl_waits, f0 = bl_ck_fixed;
}
