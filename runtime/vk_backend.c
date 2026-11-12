// The Vulkan graphics back end: gfx_vulkan_backend wires the memory and texture layer (vk_memory.c, vk_texture.c),
// submission and fences (vk_sync.c) and the present. Every entry starts the device on first use.
// Present: see docs/how-it-works.md, "Present thread".
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "flight.h"
#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "vk.h"
#include "prof.h"
#include "vk_exec.h"
#include "platform/platform.h"

uint64_t vk_frame_no;   // the newest present, for diagnostics that start at a frame

#define WEAK __attribute__((weak))

typedef struct { VkImage image; VkFormat format; int w, h, has_crop, crop[4]; uint64_t frame, at, queued, k4; } PresentJob;
// k4: the 4K composite's token for this frame (vk_dlss4k_take), taken where the present is queued.

static PlatMutex pl = PLAT_MUTEX_INIT;
static PlatCond pcv = PLAT_COND_INIT;   // job queued, job taken, job submitted
static PresentJob job;
static int pending, busy, present_thread;

// The host present goes out once the blit is submitted; pacing is left to nvnWindowAcquireTexture alone, since holding it
// to the display time would keep every vk_present_drain waiting and skew the game's frame-rate control.
static void present_hold(void) {
    static int logged;
    if (!logged) {
        logged = 1;
        rt_log("vk: the host present goes out once its blit is submitted");
    }
}

static void present_main(void *arg) {
    (void)arg;
    uint64_t last = 0, last_queued = 0;
    plat_mutex_lock(&pl);
    for (;;) {
        while (!pending) plat_cond_wait(&pcv, &pl);
        PresentJob j = job;
        pending = 0;
        busy = 1;
        plat_cond_broadcast(&pcv);
        plat_mutex_unlock(&pl);
        if (flight_on) vk_thumbs_frame(j.frame);
        vk_present_dlss4k(j.k4);
        vk_present_blit(j.image, VK_IMAGE_LAYOUT_GENERAL, j.format, j.w, j.h, j.has_crop ? j.crop : NULL);
        vk_sync_submit(NULL, 0, 0); // no guest reports; this value only covers the blit
        plat_mutex_lock(&pl);
        busy = 0;
        plat_cond_broadcast(&pcv);
        plat_mutex_unlock(&pl);
        uint64_t t0 = kern_now_ns();
        present_hold();
        uint64_t t1 = kern_now_ns();
        vk_present_flip();
        // OOT_TRACE=present: one line per present (intervals, wait and overshoot past the target).
        if (rt_trace & RT_TRACE_PRESENT)
            rt_log("present: frame %llu interval %.3f ms, game %.3f ms, waited %.3f ms, %.3f ms past target",
                   (unsigned long long)j.frame, last ? (t1 - last) / 1e6 : 0.0,
                   last_queued ? (j.queued - last_queued) / 1e6 : 0.0, (t1 - t0) / 1e6,
                   j.at && t1 > j.at ? (t1 - j.at) / 1e6 : 0.0);
        last = t1;
        last_queued = j.queued;
        vk_pipeline_autosave();   // starts the autosave thread once
        plat_mutex_lock(&pl);
    }
    return;
}

void vk_present_drain(void) {
    uint64_t p0 = perf_begin();
    plat_mutex_lock(&pl);
    while (pending || busy) plat_cond_wait(&pcv, &pl);
    plat_mutex_unlock(&pl);
    perf_end(PERF_DRAIN, p0);
}

// The present after the frame count: resolves the image and hands the blit to the present thread. queued: when the game
// presented.
static void present_image(const tex_t *t, const int crop[4], uint64_t at_ns, uint64_t queued, uint64_t frame) {
    static volatile int32_t n_skipped;
    vk_exec_frame_log(t, frame);
    vk_mem_census(frame);
    vk_frame_no = frame;
    vk_ind_clamp_frame(frame);
    { extern void nav_state_present(uint64_t); nav_state_present(frame); }   // OOT_STATE_FILE (nav_state.c)
    const VkTexImage *img = vk_texture_get(t);
    if (!img || !(img->aspect & VK_IMAGE_ASPECT_COLOR_BIT)) {
        if (plat_inc32(&n_skipped) <= 8)
            rt_log("vk: present of texture %p (format %#x) skipped: no colour image", (const void *)t, t ? t->d.format : 0);
        return;
    }
    PresentJob j = {.image = img->image, .format = img->format, .w = (int)img->width, .h = (int)img->height, .frame = frame,
                    .at = at_ns, .queued = queued, .k4 = vk_dlss4k_take()};
    if (crop) {
        j.has_crop = 1;
        for (int i = 0; i < 4; i++) j.crop[i] = crop[i];
    }
    vk_upload_flush(0);   // an upload vk_texture_get just recorded goes before the blit
    plat_mutex_lock(&pl);
    if (!present_thread) {
        if (!plat_thread_spawn(present_main, NULL, "vk present", 0)) rt_fatal("vk: cannot start the present thread");
        present_thread = 1;
    }
    while (pending) plat_cond_wait(&pcv, &pl);   // one present in waiting at most
    job = j;
    pending = 1;
    plat_cond_broadcast(&pcv);
    plat_mutex_unlock(&pl);
}

// A submission that runs the queues that can run now (vk_exec.c queue coroutines): after the CPU signalled
// an event a queue stood at (nvn_api_core.c nvnEventSignal), or with force before a present or a finish. 0: nothing
// pending. The value returned is a fence as vk_be_submit's.
static int co_record(VkCommandBuffer cb, uint64_t fence, uint64_t submitted_ticks, void *context) {
    return vk_exec_record_co(cb, fence, submitted_ticks, context != NULL);
}
uint64_t vk_co_run(int force) {
    extern uint64_t rt_ticks(void);   // cpu.h
    static int marker;
    if (!vk_exec_co_pending()) return 0;
    vk_sync_start();
    vk_present_drain();
    vk_texture_new_submit();
    return vk_sync_submit_record(co_record, force ? &marker : NULL, rt_ticks());
}

static uint64_t submit_sync(const uint64_t *handles, int n, uint64_t submitted_ticks) {
    vk_sync_start();
    vk_present_drain();
    vk_texture_new_submit();
    return vk_sync_submit(handles, n, submitted_ticks);
}

static void vk_be_present(const tex_t *t, const int crop[4], uint64_t at_ns) {
    vk_sync_start();
    uint64_t frame = __atomic_add_fetch(&gfx_counters.frames, 1, __ATOMIC_RELAXED);
    perf_present(frame);   // before vk_exec_frame_log, which resets the frame's exec counts the flight recorder reads
    present_image(t, crop, at_ns, kern_now_ns(), frame);
}

// vk_pipeline.c drops a released program's pipelines here.
WEAK void vk_program_release(const prog_t *p) { (void)p; }

static int vk_be_pool_init(pool_t *p) { vk_sync_start(); return vk_pool_init(p); }
static void vk_be_pool_fini(pool_t *p) { vk_sync_start(); vk_present_drain(); vk_pool_fini(p); }
static int vk_be_pool_map_virtual(pool_t *v, int n, const void *requests) {
    vk_sync_start();
    vk_present_drain();
    return vk_pool_map_virtual(v, n, requests);
}
static void vk_be_texture_register(const dpool_ref *pool, int id, const tex_t *t, const view_t *v) { vk_sync_start(); vk_texture_register(pool, id, t, v); }
static void vk_be_sampler_register(const dpool_ref *pool, int id, const samp_par *s) { vk_sync_start(); vk_sampler_register(pool, id, s); }
static void vk_be_texture_release(const tex_t *t) {
    vk_sync_start();
    vk_present_drain();
    vk_texture_release(t);
}
static void vk_be_program_release(const prog_t *p) { vk_program_release(p); }
static uint64_t vk_be_submit(const uint64_t *handles, int n, uint64_t submitted_ticks) {
    return submit_sync(handles, n, submitted_ticks);
}
static void vk_be_finish(void) { vk_sync_start(); vk_present_drain(); vk_sync_finish(); }
static void vk_be_read_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row,
                              uint64_t img) {
    vk_sync_start();
    vk_read_texels(t, v, r, dst, row, img);
}
static void vk_be_write_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, const uint8_t *src, uint64_t row,
                               uint64_t img) {
    vk_sync_start();
    vk_present_drain();
    vk_write_texels(t, v, r, src, row, img);
}
static void vk_be_guest_written(uint64_t addr, uint64_t size) { vk_sync_start(); vk_guest_written(addr, size); }

static void vk_be_guest_read(uint64_t addr, uint64_t size) {
    if (!vk_texture_cpu_stale(addr, size)) return;
    vk_sync_start();
    vk_present_drain();
    vk_texture_write_back(addr, size);
}
// After a fence wait: the value is done, so its replay is too; a later submission the replay records meanwhile marks
// its targets with a later fence, which write_back_small leaves alone. No drain.
static void vk_be_cpu_synced(uint64_t fence) {
    if (!vk_texture_small_dirty()) return;
    vk_sync_start();
    vk_present_drain();
    vk_texture_write_back_small(fence);
}

const GfxBackend gfx_vulkan_backend = {
    .name = "vulkan",
    .pool_init = vk_be_pool_init,
    .pool_fini = vk_be_pool_fini,
    .pool_map_virtual = vk_be_pool_map_virtual,
    .texture_register = vk_be_texture_register,
    .sampler_register = vk_be_sampler_register,
    .texture_release = vk_be_texture_release,
    .program_release = vk_be_program_release,
    .submit = vk_be_submit,
    .fence_wait = vk_sync_fence_wait,
    .fence_done = vk_sync_fence_done,
    .present = vk_be_present,
    .finish = vk_be_finish,
    .read_texels = vk_be_read_texels,
    .write_texels = vk_be_write_texels,
    .guest_written = vk_be_guest_written,
    .guest_read = vk_be_guest_read,
    .cpu_synced = vk_be_cpu_synced,
};

GfxStats gfx_counters;

GfxStats gfx_stats(void) {
    GfxStats s;
    s.frames = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    s.nvn_calls = __atomic_load_n(&gfx_counters.nvn_calls, __ATOMIC_RELAXED);
    s.nvn_distinct = __atomic_load_n(&gfx_counters.nvn_distinct, __ATOMIC_RELAXED);
    s.unimplemented = __atomic_load_n(&gfx_counters.unimplemented, __ATOMIC_RELAXED);
    s.validation_errors = __atomic_load_n(&gfx_counters.validation_errors, __ATOMIC_RELAXED);
    s.shader_failures = __atomic_load_n(&gfx_counters.shader_failures, __ATOMIC_RELAXED);
    s.draws_skipped = __atomic_load_n(&gfx_counters.draws_skipped, __ATOMIC_RELAXED);
    return s;
}

const GfxBackend *gfx_backend(void) { return &gfx_vulkan_backend; }

void vk_backend_shutdown(void) {
    vk_present_drain();
    vk_sync_shutdown();
    vk_mem_shutdown();
    vk_shutdown();
}
