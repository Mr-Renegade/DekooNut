// Adapted from MacNativeGames games/totk/runtime/loader.c (lines 456-477) and services.c (lines 219-235) @ b75be8cf
// (MIT, Copyright (c) 2026 MacNativeGames contributors) and LAS Recomp runtime/main.c (lines 16-35, 141-159) @
// da35b83 (MIT, Copyright (c) 2026 Peiikai).
// The boot driver: the guest main thread runs the game's own nn::init::Start
// with three host callbacks, the second of which runs DT_INIT of main, subsdk0 and sdk; Start then calls nnMain.
// With --seconds S rt_frame_presented ends the run once the world clock passed S.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "flight.h"
#include "cfg.h"
#include "fs.h"
#include "gfx.h"
#include "nvn_api.h"
#include "platform/platform.h"
#include "vk.h"
#include "platform/platform.h"

#define SYM_INIT_START "_ZN2nn4init5StartEmmPFvvES2_S2_"
#define CTOR_DONE 2                      // a module's constructor flag after its DT_INIT ran the init_array

static volatile int32_t inits_done;
uint64_t rt_guest_start_ns;

static uint32_t ctor_state(const RtLoaded *m) { return LD32(m->ctor_flag); }

static const char *caller(const Cpu *c, char *buf, int n) { return ldr_describe(c->x[30] - 4, buf, n); }

// main.npdm META: the main thread's priority (+0xE), core (+0xF) and stack size (+0x1C). kernel.c builds the main
// thread with the 1.2.1 values; a different npdm means a different game build.
static void check_npdm(void) {
    size_t n;
    uint8_t *npdm = ldr_read_verified(rt_config()->exefs, "main.npdm", &n);
    if (n < 0x80 || memcmp(npdm, "META", 4)) rt_fatal("boot: main.npdm has no META header");
    uint32_t stack;
    memcpy(&stack, npdm + 0x1C, 4);
    int prio = npdm[0xE], core = npdm[0xF];
    free(npdm);
    if (prio != 44 || core != 0 || stack != 0x100000)
        rt_fatal("boot: main.npdm asks for priority %d, core %d, stack %#x; kernel.c builds 44, 0, 0x100000", prio,
                 core, stack);
}

// ---- the three callbacks of nn::init::Start (rtld's job on hardware)
static void cb_loader_ready(Cpu *c) {
    (void)c;
    rt_log("boot: rtld callback: modules linked");
}

// DT_INIT registers the module with musl (__nnmusl_init_dso) and runs its init_array; the constructor flag reads 2
// afterwards.
static void cb_module_inits(Cpu *c) {
    if (plat_cas32(&inits_done, 0, 0)) rt_fatal("boot: module constructors requested twice");
    // Load order (main, subsdk*, sdk), as rtld runs them.
    for (int k = 0; k < ldr_count(); k++) {
        const RtLoaded *m = ldr_module(k);
        uint64_t entries = m->init_array_size / 8;
        if (!m->init) rt_fatal("boot: %s has no DT_INIT", m->name);
        rt_log("boot: ENTER %s DT_INIT (%llu init_array entries)", m->name, (unsigned long long)entries);
        rt_call(c, m->init, 0);
        uint32_t state = ctor_state(m);
        rt_logf("boot: DT_INIT %s state=%u (%llu)\n", m->name, state, (unsigned long long)entries);
        if (state != CTOR_DONE) rt_fatal("boot: %s DT_INIT left its constructor flag at %u, not 2", m->name, state);
    }
    plat_xchg32(&inits_done, 1);
}

static void cb_module_finis(Cpu *c) {
    char lr[128];
    rt_fatal("boot: the game asked for module finalizers (from %s): the guest exited", caller(c, lr, sizeof lr));
}

// ---- checkpoints (host.h)
static void stats_line(void) {
    RtStats s = rt_stats();
    rt_logf("boot: passed (%llu SVCs, %llu threads, %llu host calls)\n", (unsigned long long)s.svcs,
           (unsigned long long)s.threads, (unsigned long long)s.host_calls);
}

// ---- --seconds S
// A PNG with stored (uncompressed) deflate blocks: no compression library needed (rt_write_png is declared in vk.h).
static uint32_t crc32_update(uint32_t crc, const uint8_t *p, size_t n) {
    static uint32_t table[256];
    if (!table[1])
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
    for (size_t i = 0; i < n; i++) crc = table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return crc;
}

static void png_chunk(FILE *f, const char *type, const uint8_t *data, uint32_t n) {
    uint8_t hdr[8] = {(uint8_t)(n >> 24), (uint8_t)(n >> 16), (uint8_t)(n >> 8), (uint8_t)n};
    memcpy(hdr + 4, type, 4);
    uint32_t crc = crc32_update(crc32_update(~0u, hdr + 4, 4), data, n) ^ ~0u;
    uint8_t tail[4] = {(uint8_t)(crc >> 24), (uint8_t)(crc >> 16), (uint8_t)(crc >> 8), (uint8_t)crc};
    fwrite(hdr, 1, 8, f);
    if (n) fwrite(data, 1, n, f);
    fwrite(tail, 1, 4, f);
}

void rt_write_png(const char *path, const uint8_t *rgba, uint32_t w, uint32_t h) {
    uint64_t row = 1 + 4ull * w, raw_n = row * h;
    uint8_t *raw = malloc(raw_n);
    if (!raw) rt_fatal("boot: out of memory for the shot");
    for (uint32_t y = 0; y < h; y++) {
        raw[y * row] = 0;   // filter: none
        memcpy(raw + y * row + 1, rgba + 4ull * w * y, 4ull * w);
    }
    uint64_t blocks = (raw_n + 65534) / 65535;
    uint8_t *z = malloc(2 + raw_n + 5 * blocks + 4);
    if (!z) rt_fatal("boot: out of memory for the shot");
    uint64_t zn = 0;
    z[zn++] = 0x78, z[zn++] = 0x01;
    uint32_t a = 1, b = 0;
    for (uint64_t off = 0; off < raw_n; off += 65535) {
        uint32_t k = raw_n - off > 65535 ? 65535 : (uint32_t)(raw_n - off);
        z[zn++] = off + k >= raw_n;
        z[zn++] = (uint8_t)k, z[zn++] = (uint8_t)(k >> 8), z[zn++] = (uint8_t)~k, z[zn++] = (uint8_t)(~k >> 8);
        memcpy(z + zn, raw + off, k);
        zn += k;
        for (uint32_t i = 0; i < k; i++) a = (a + raw[off + i]) % 65521, b = (b + a) % 65521;
    }
    uint32_t ad = b << 16 | a;
    z[zn++] = (uint8_t)(ad >> 24), z[zn++] = (uint8_t)(ad >> 16), z[zn++] = (uint8_t)(ad >> 8), z[zn++] = (uint8_t)ad;
    FILE *f = fopen(path, "wb");
    if (!f) rt_fatal("boot: cannot write %s", path);
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 13, 10, 26, 10};
    uint8_t ihdr[13] = {(uint8_t)(w >> 24), (uint8_t)(w >> 16), (uint8_t)(w >> 8), (uint8_t)w,
                        (uint8_t)(h >> 24), (uint8_t)(h >> 16), (uint8_t)(h >> 8), (uint8_t)h, 8, 6, 0, 0, 0};
    fwrite(sig, 1, 8, f);
    png_chunk(f, "IHDR", ihdr, 13);
    png_chunk(f, "IDAT", z, (uint32_t)zn);
    png_chunk(f, "IEND", NULL, 0);
    if (fclose(f)) rt_fatal("boot: cannot write %s", path);
    free(z);
    free(raw);
}

// The presentation allocation may be larger than the active image (OoT uses 1706x960 of 2560x1440). Match
// gfx_present.c's crop clipping so diagnostic screenshots do not include the unused right and bottom margins.
static NVNcopyRegion shot_region(const tex_t *t, const int crop[4]) {
    int x = 0, y = 0, w = (int)t->d.width, h = (int)t->d.height;
    if (crop && crop[2] > 0 && crop[3] > 0) x = crop[0], y = crop[1], w = crop[2], h = crop[3];
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if ((int64_t)x + w > t->d.width) w = (int)t->d.width - x;
    if ((int64_t)y + h > t->d.height) h = (int)t->d.height - y;
    if (w <= 0 || h <= 0) rt_fatal("boot: screenshot crop is outside the presented texture");
    return (NVNcopyRegion){.xoffset = x, .yoffset = y, .width = w, .height = h, .depth = 1};
}

static void save_shot(const tex_t *t, const int crop[4], const char *path) {
    int fmt = t->d.format;
    int bgra = fmt == NVN_FORMAT_BGRA8 || fmt == NVN_FORMAT_BGRA8_SRGB;
    if (!bgra && fmt != NVN_FORMAT_RGBA8 && fmt != NVN_FORMAT_RGBA8_SRGB)
        rt_fatal("boot: --shot supports 8-bit RGBA and BGRA targets, the presented texture has NVN format %#x", fmt);
    NVNcopyRegion r = shot_region(t, crop);
    uint32_t w = (uint32_t)r.width, h = (uint32_t)r.height;
    uint8_t *px = malloc(4ull * w * h);
    if (!px) rt_fatal("boot: out of memory for the shot");
    view_t view;
    memset(&view, 0, sizeof view);
    gfx_backend()->read_texels(t, &view, &r, px, 4ull * w, 4ull * w * h);
    if (bgra)
        for (uint64_t i = 0; i < 4ull * w * h; i += 4) {
            uint8_t x = px[i];
            px[i] = px[i + 2];
            px[i + 2] = x;
        }
    rt_write_png(path, px, w, h);
    free(px);
    rt_logf("boot: shot %s (%ux%u)\n", path, w, h);
}

// OOT_SHOTS=lo-hi:step: every step-th present in [lo, hi] saved as ./f<n>.png (n the present), box-filtered to
// 480x270; OOT_SHOTS=w<lo>-<hi>:<step> steps by world-clock seconds instead (the first present at or after each step).
static void series_shot(const tex_t *t, const int crop[4], uint64_t n) {
    static int parsed;
    static unsigned long long lo, hi, step, next;
    static const char *const dir = ".";
    static int world;
    if (!parsed) {
        const char *e = cfg_get("OOT_SHOTS");
        double wl, wh, ws;
        if (e && e[0] == 'w' && sscanf(e + 1, "%lf-%lf:%lf", &wl, &wh, &ws) == 3 && ws > 0)   // world-clock seconds
            world = 1, lo = (unsigned long long)(wl * 1e6), hi = (unsigned long long)(wh * 1e6),
            step = (unsigned long long)(ws * 1e6), next = lo;
        else if (!e || sscanf(e, "%llu-%llu:%llu", &lo, &hi, &step) != 3 || !step) hi = 0;
        if (hi) plat_file_mkdir_p(dir);
        parsed = 1;
    }
    if (world) {   // the first present at or after each step of the world clock
        int64_t w = frame_step_world_us();
        if (w < 0 || (unsigned long long)w < next || next > hi) return;
        while (next <= (unsigned long long)w) next += step;
    } else if (n < lo || n > hi || (n - lo) % step) {
        return;
    }
    int fmt = t->d.format;
    int bgra = fmt == NVN_FORMAT_BGRA8 || fmt == NVN_FORMAT_BGRA8_SRGB;
    if (!bgra && fmt != NVN_FORMAT_RGBA8 && fmt != NVN_FORMAT_RGBA8_SRGB) return;
    NVNcopyRegion r = shot_region(t, crop);
    uint32_t w = (uint32_t)r.width, h = (uint32_t)r.height, ow = 480, oh = 270;
    uint8_t *px = malloc(4ull * w * h), *out = malloc(4ull * ow * oh);
    if (!px || !out) rt_fatal("boot: out of memory for a OOT_SHOTS frame");
    view_t view;
    memset(&view, 0, sizeof view);
    gfx_backend()->read_texels(t, &view, &r, px, 4ull * w, 4ull * w * h);
    for (uint32_t y = 0; y < oh; y++)
        for (uint32_t x = 0; x < ow; x++) {
            uint32_t x0 = x * w / ow, x1 = (x + 1) * w / ow, y0 = y * h / oh, y1 = (y + 1) * h / oh;
            if (x1 <= x0) x1 = x0 + 1;
            if (y1 <= y0) y1 = y0 + 1;
            uint32_t sum[4] = {0}, k = 0;
            for (uint32_t sy = y0; sy < y1; sy++)
                for (uint32_t sx = x0; sx < x1; sx++, k++)
                    for (int c = 0; c < 4; c++) sum[c] += px[4ull * (sy * (uint64_t)w + sx) + (uint32_t)c];
            uint8_t *o = out + 4ull * (y * ow + x);
            for (int c = 0; c < 4; c++) o[c] = (uint8_t)(sum[bgra && c != 3 ? 2 - c : c] / k);
            o[3] = 255;
        }
    char path[PLAT_PATH_MAX];
    snprintf(path, sizeof path, "%s/f%05llu.png", dir, (unsigned long long)n);
    rt_write_png(path, out, ow, oh);
    free(out);
    free(px);
}

// OOT_BURST_FILE=<path> OOT_BURST_DIR=<dir>: when that file appears, the next N presents (N: the number in the file,
// default 240) go to <dir>/b<n>.raw as 480x270 RGBA, then the file is removed. The read waits for the GPU.
static void burst_shot(const tex_t *t, const int crop[4], uint64_t n) {
    static const char *file, *dir; static int ready, left, full; static uint8_t *px, *out;
    if (!ready) { ready = 1, file = getenv("OOT_BURST_FILE"), dir = getenv("OOT_BURST_DIR"); if (dir) plat_file_mkdir_p(dir); }
    if (!file || !dir) return;
    if (!left) {
        FILE *f = fopen(file, "r");
        if (!f) return;
        if (fscanf(f, "%d", &left) != 1 || left <= 0) left = 240;
        // "<N> full": the frames at the game image's own size as <dir>/b<n>.png
        char word[8] = "";
        full = fscanf(f, "%7s", word) == 1 && !strcmp(word, "full");
        fclose(f);
    }
    int fmt = t->d.format;
    int bgra = fmt == NVN_FORMAT_BGRA8 || fmt == NVN_FORMAT_BGRA8_SRGB;
    if (!bgra && fmt != NVN_FORMAT_RGBA8 && fmt != NVN_FORMAT_RGBA8_SRGB) return;
    NVNcopyRegion r = shot_region(t, crop);
    uint32_t w = (uint32_t)r.width, h = (uint32_t)r.height, ow = 480, oh = 270;
    if (!px) px = malloc(4ull * 4096 * 2304), out = malloc(4ull * ow * oh);
    if ((uint64_t)w * h > 4096ull * 2304) return;
    view_t view;
    memset(&view, 0, sizeof view);
    gfx_backend()->read_texels(t, &view, &r, px, 4ull * w, 4ull * w * h);
    if (full) {   // comparison shots: the game image's own size, PNG
        uint8_t *rgba = malloc(4ull * w * h);
        if (rgba) {
            for (uint64_t i = 0; i < (uint64_t)w * h; i++)
                for (int c = 0; c < 4; c++) rgba[4 * i + c] = c == 3 ? 255 : px[4 * i + (bgra ? 2 - c : c)];
            char fp[PLAT_PATH_MAX];
            snprintf(fp, sizeof fp, "%s/b%07llu.png", dir, (unsigned long long)n);
            rt_write_png(fp, rgba, w, h);
            free(rgba);
        }
        if (!--left) remove(file);
        return;
    }
    for (uint32_t y = 0; y < oh; y++)   // every other source texel of a box: fast, and a flicker covers more than one
        for (uint32_t x = 0; x < ow; x++) {
            uint32_t x0 = x * w / ow, x1 = (x + 1) * w / ow, y0 = y * h / oh, y1 = (y + 1) * h / oh, sum[3] = {0}, k = 0;
            for (uint32_t sy = y0; sy < y1; sy += 2)
                for (uint32_t sx = x0; sx < x1; sx += 2, k++)
                    for (int c = 0; c < 3; c++) sum[c] += px[4ull * (sy * (uint64_t)w + sx) + (uint32_t)c];
            uint8_t *o = out + 4ull * (y * ow + x);
            for (int c = 0; c < 3; c++) o[c] = (uint8_t)(sum[bgra ? 2 - c : c] / (k ? k : 1));
            o[3] = 255;
        }
    char path[PLAT_PATH_MAX];
    snprintf(path, sizeof path, "%s/b%07llu.raw", dir, (unsigned long long)n);
    FILE *f = fopen(path, "wb");
    if (f) fwrite(out, 4, (size_t)ow * oh, f), fclose(f);
    if (!--left) remove(file);
}

// OOT_THUMB_FILE=<path>: every flight thumbnail (OOT_FLIGHT=1) appended as u32 frame + 320x180 RGB. Unlike a burst it
// waits for nothing: the GPU copies each present into a ring and the frame is taken once its fence has signalled.
static void thumb_sink(uint64_t n) {
    static int state; static FILE *f; static uint64_t next = 1; static uint8_t px[FLIGHT_THUMB_W * FLIGHT_THUMB_H * 4], rgb[FLIGHT_THUMB_W * FLIGHT_THUMB_H * 3];
    if (!state) { const char *e = getenv("OOT_THUMB_FILE"); f = e && *e ? fopen(e, "wb") : NULL; state = f ? 1 : -1; }
    if (state < 0) return;
    if (n > next + 300) next = n - 300;   // the ring holds 600; older ones are gone
    for (; next < n; next++) {
        if (!vk_thumbs_read(next, px)) { if (n - next < 8) break; continue; }   // not done yet: next present
        for (int i = 0; i < FLIGHT_THUMB_W * FLIGHT_THUMB_H; i++) rgb[3 * i] = px[4 * i], rgb[3 * i + 1] = px[4 * i + 1], rgb[3 * i + 2] = px[4 * i + 2];
        uint32_t fr = (uint32_t)next;
        fwrite(&fr, 4, 1, f), fwrite(rgb, sizeof rgb, 1, f);
    }
}

// A --seconds run ends at the first present after the world clock (game time, frame_step_world_us) passed S seconds,
// so a run covers the same game time at any frame rate.
void rt_frame_presented(const tex_t *t, const int crop[4], uint64_t n) {
    const RtConfig *cfg = rt_config();
    series_shot(t, crop, n);
    burst_shot(t, crop, n);
    thumb_sink(n);
    int64_t w = cfg->seconds ? frame_step_world_us() : -1;
    if (w < 0 || (double)w < cfg->seconds * 1e6) return;
    if (cfg->shot) save_shot(t, crop, cfg->shot);
    GfxStats g = gfx_stats();
    rt_logf("boot: frames %llu presented (nvn calls %llu, distinct %llu, unimplemented %llu, validation errors %llu, "
           "shader failures %llu, draws skipped %llu)\n", (unsigned long long)g.frames,
           (unsigned long long)g.nvn_calls, (unsigned long long)g.nvn_distinct, (unsigned long long)g.unimplemented,
           (unsigned long long)g.validation_errors, (unsigned long long)g.shader_failures,
           (unsigned long long)g.draws_skipped);
    stats_line();
    if (rt_trace & RT_TRACE_HOSTCALL) hostfn_report();   // which host functions the run leaned on
    fflush(stdout);
    fflush(stderr);
    fs_save_flush();
    vk_pipeline_save();   // no-op without the Vulkan back end
    prof_write();
    fflush(stderr);
    plat_exit(0);
}

// ---- steps 9-14
static uint64_t cb_addr[3];

static void guest_main(Cpu *c) {
    uint32_t handle = kern_main_thread(&c);
    uint64_t start = ldr_symbol(SYM_INIT_START);
    if (!start) rt_fatal("boot: %s is not exported", SYM_INIT_START);
    char d[128];
    rt_log("boot: ENTER nn::init::Start at %s, main thread %#x", ldr_describe(start, d, sizeof d), handle);
    // nn::init::Start(main thread handle, arguments region, loader_ready, module_inits, module_finis)
    rt_call(c, start, 5, (uint64_t)handle, (uint64_t)0, cb_addr[0], cb_addr[1], cb_addr[2]);
    rt_fatal("boot: nn::init::Start returned");
}

// The guest starts once the recorded shaders are loaded and the pipeline and mesh jobs are queued; the builds go on
// while it initialises (a draw that meets a queued job promotes it).
static PlatMutex warm_lock = PLAT_MUTEX_INIT;
static PlatCond warm_cv = PLAT_COND_INIT;
static int warm_queued;
static void prewarm_release(void) {   // the guest may start
    plat_mutex_lock(&warm_lock);
    if (!warm_queued) {
        warm_queued = 1;
        plat_mutex_unlock(&warm_lock);
        vk_exec_mesh_prewarm();   // after the shader prewarm: the fragment shaders are loaded (mode 1 only queues)
        plat_mutex_lock(&warm_lock);
        warm_queued = 2;
        plat_cond_broadcast(&warm_cv);
    }
    plat_mutex_unlock(&warm_lock);
}

static void prewarm_main(void *arg) {
    (void)arg;
    vk_pipeline_prewarm_queued = prewarm_release;
    vk_pipeline_prewarm();
    prewarm_release();   // a prewarm that returned before queuing (no keys, prebuild off)
    rt_log("boot: prewarm builds done %llu ms after process start",
           (unsigned long long)(plat_process_uptime_ns() / 1000000));
    return;
}

void rt_boot(void) {
    check_npdm();
    if (!rt_config()->shot) vk_pipeline_async_enable();   // a --shot image must not depend on build timing
    // Shaders and pipelines recorded in earlier runs are made ready while the modules load; the guest waits for them.
    PlatThread *warm = plat_thread_new(prewarm_main, NULL, "prewarm");
    ldr_load(rt_config()->exefs);
    rt_patches_init();
    frame_step_init();
    res_init();
    cb_addr[0] = hostfn_register("<rtld loader_ready>", cb_loader_ready);
    cb_addr[1] = hostfn_register("<rtld module_inits>", cb_module_inits);
    cb_addr[2] = hostfn_register("<rtld module_finis>", cb_module_finis);
    hostfn_seal();
    uint64_t held = plat_now_ms();
    if (warm) {
        plat_mutex_lock(&warm_lock);
        while (warm_queued != 2) plat_cond_wait(&warm_cv, &warm_lock);
        plat_mutex_unlock(&warm_lock);
        plat_thread_detach(warm);
    } else {
        vk_pipeline_prewarm(), vk_exec_mesh_prewarm();
    }
    held = plat_now_ms() - held;
    rt_log("boot: guest starts %llu ms after process start (%llu ms of it waiting for the prewarm)",
           (unsigned long long)(plat_process_uptime_ns() / 1000000), (unsigned long long)held);
    rt_guest_start_ns = kern_now_ns();
    kern_run_on_guest_main(guest_main);
    rt_fatal("boot: the guest main thread ended");
}
