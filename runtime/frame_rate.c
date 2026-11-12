// Adapted from MacNativeGames games/totk/runtime/game_patches.c @ ebe621d (MIT, Copyright (c) 2026 MacNativeGames
// contributors): runtime_frame_step_update (2338bb0: no 1/120 s floor).
// Frame rate: the frame limit (OOT_FPS_LIMIT) and the world clock (frame_step_world_us sums the steps from the first present).
// The game steps whole 60 Hz vsyncs and the pacer follows its interval, so the limit stops at 60. OOT_TRACE=perf adds a line per 120 presents.
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "layout.h"
#include "platform/platform.h"

#define STEP_MAX_S (1.0 / 15.0)    // longest step; a longer frame slows the game down

static uint64_t last_ns;

void frame_step_init(void) {
    rt_log("frame rate: the game steps in whole display intervals");
}

// The frame limit (menu or OOT_FPS_LIMIT): any whole rate from 30 to 360, or display (the window's monitor rate).
#define LIMIT_MIN_HZ 30
#define LIMIT_MAX_HZ 360
static unsigned limit_seen;
static int limit_read;
static int limit_hz = -1;   // 0: display; -1 before the first read

static int limit_parse(void) {
    const char *e = cfg_get("OOT_FPS_LIMIT");
    if (!e || !*e) return 60;
    if (!strcmp(e, "display")) return 0;
    char *end;
    long n = strtol(e, &end, 10);
    if (!*end && n >= LIMIT_MIN_HZ && n <= LIMIT_MAX_HZ) return (int)n;
    rt_log("frame rate: OOT_FPS_LIMIT=%s is not 30..360 or display; 60", e);
    return 60;
}

static void frame_limit_live(void) {
    unsigned v = cfg_version();
    if (limit_read && v == limit_seen) return;
    limit_read = 1;
    limit_seen = v;
    int hz = limit_parse();
    if (hz == limit_hz) return;
    limit_hz = hz;
    if (hz > 60 || hz == 0)
        rt_log("frame rate: limits above 60 need the real-time step; 60 fps");
    else rt_log("frame rate: limit %d fps", hz);
}

// The frame rate the limit gives, for the overlay: the setting, the monitor's rate for "display", 60 at most; 0 before the first read.
double frame_step_limit_hz(void) {
    int hz = limit_hz;
    if (hz < 0) return 0;
    double r = hz ? hz : gfx_window_refresh_hz();
    if (r < LIMIT_MIN_HZ) r = 60;
    return r > 60 ? 60 : r;
}

// ---- OOT_TRACE=perf speed lines
static uint64_t win_t0, win_frames, win_hist[4];   // win_hist: frames by present interval 1, 2, 3+ ([0] unused)

static void speed_report(uint64_t now, int interval) {
    if (!win_t0) {
        win_t0 = now;
        return;
    }
    win_frames++;
    win_hist[interval < 1 ? 1 : interval > 3 ? 3 : interval]++;
    if (win_frames < 120) return;
    double wall = (double)(now - win_t0) * 1e-9;
    rt_log("fps: %llu frames in %.3f s (%.1f fps), whole-vsync steps; interval 1/2/3+ %llu/%llu/%llu",
           (unsigned long long)win_frames, wall, (double)win_frames / wall, (unsigned long long)win_hist[1],
           (unsigned long long)win_hist[2], (unsigned long long)win_hist[3]);
    win_t0 = now;
    win_frames = 0;
    memset(win_hist, 0, sizeof win_hist);
}

static _Atomic int64_t world_us = -1;
static _Atomic uint64_t game_us;   // game time stepped since the first present (frame_step_ticks)
static uint64_t presents;

int64_t frame_step_world_us(void) { return atomic_load_explicit(&world_us, memory_order_relaxed); }
uint64_t frame_step_ticks(void) { return atomic_load_explicit(&game_us, memory_order_relaxed) * 60 / 1000000; }

static void world_clock(double dt) {
    presents++;
    int64_t w = atomic_load_explicit(&world_us, memory_order_relaxed);
    if (w < 0) {
        rt_log("frame step: world clock starts at present %llu, %.2f s after the guest started",
               (unsigned long long)presents, rt_guest_start_ns ? (kern_now_ns() - rt_guest_start_ns) / 1e9 : 0.0);
        atomic_store_explicit(&world_us, 0, memory_order_relaxed);
        return;
    }
    double step = dt > STEP_MAX_S ? STEP_MAX_S : dt;   // whole-vsync steps: the display time stands in
    atomic_store_explicit(&world_us, w + (int64_t)(step * 1e6 + 0.5), memory_order_relaxed);
}

void frame_step_present(uint64_t display_ns, int interval) {
    frame_limit_live();
    double dt = last_ns ? (double)(display_ns - last_ns) * 1e-9 : 0;
    last_ns = display_ns;
    // The game clock: what the game stepped this frame, whole vsyncs.
    double game = (interval > 0 ? interval : 1) / 60.0;
    atomic_store_explicit(&game_us, atomic_load_explicit(&game_us, memory_order_relaxed) + (uint64_t)(game * 1e6 + 0.5),
                          memory_order_relaxed);
    world_clock(dt);
    if (rt_trace & RT_TRACE_PERF) speed_report(display_ns, interval);
}
