// The frames the display showed, for the performance overlay. On Wayland the compositor answers a feedback request with
// the time the window's next content update was shown, or that it was dropped (the presentation-time protocol). The
// driver's frame generation commits its frames to the same surface, so they are counted like the game's own.
// On Windows the graphics kernel counts the frames the process put on the window's display, generated ones included.
#include <stdint.h>
#include <string.h>

#include "gfx.h"
#include "host.h"
#include "platform/platform.h"

#ifdef __linux__
#include <dlfcn.h>
#include <poll.h>

// libwayland-client's types and calls, as far as they are used: the library is the one the window system loaded.
struct wl_interface;
struct wl_message { const char *name, *signature; const struct wl_interface **types; };
struct wl_interface { const char *name; int version, method_count; const struct wl_message *methods;
                      int event_count; const struct wl_message *events; };
static void *(*display_create_queue)(void *display);
static void *(*proxy_create_wrapper)(void *proxy);
static void (*proxy_set_queue)(void *proxy, void *queue);
static void *(*proxy_marshal_flags)(void *proxy, uint32_t opcode, const struct wl_interface *iface, uint32_t version,
                                    uint32_t flags, ...);
static int (*proxy_add_listener)(void *proxy, void (**impl)(void), void *data);
static void (*proxy_destroy)(void *proxy);
static uint32_t (*proxy_get_version)(void *proxy);
static int (*display_roundtrip_queue)(void *display, void *queue);
static int (*display_prepare_read_queue)(void *display, void *queue);
static int (*display_dispatch_queue_pending)(void *display, void *queue);
static int (*display_read_events)(void *display);
static void (*display_cancel_read)(void *display);
static int (*display_flush)(void *display);
static int (*display_get_fd)(void *display);
static const struct wl_interface *registry_if;

// wp_presentation version 1 and its feedback object.
static const struct wl_interface feedback_if;
static const struct wl_interface *no_types[7];
static const struct wl_interface *feedback_types[] = {NULL, &feedback_if};   // the surface, the new feedback
static const struct wl_message presentation_requests[] = {{"destroy", "", no_types}, {"feedback", "on", feedback_types}};
static const struct wl_message presentation_events[] = {{"clock_id", "u", no_types}};
static const struct wl_interface presentation_if = {"wp_presentation", 1, 2, presentation_requests, 1, presentation_events};
static const struct wl_message feedback_events[] = {{"sync_output", "o", no_types}, {"presented", "uuuuuuu", no_types},
                                                    {"discarded", "", no_types}};
static const struct wl_interface feedback_if = {"wp_presentation_feedback", 1, 0, NULL, 3, feedback_events};

#define ARM_NS 2000000ull          // a request every 2 ms: one is waiting at each of the surface's commits
#define WINDOW_NS 500000000ull     // the rate is taken over half a second of shown frames
#define WANTED_NS 2000000000ull    // requests stop this long after the overlay last asked

static void *g_display, *g_surface, *g_queue, *g_registry, *g_presentation;
static uint32_t g_presentation_name;
static int g_pending;
static uint64_t g_last_shown, g_first, g_frames;   // the last shown time; the window's first one and its frame count
static volatile int64_t g_wanted_ns, g_rate_ns;
static volatile int64_t g_rate_milli;              // frames per second x 1000

static void on_global(void *data, void *registry, uint32_t name, const char *iface, uint32_t version) {
    (void)data, (void)registry, (void)version;
    if (!strcmp(iface, presentation_if.name)) g_presentation_name = name;
}
static void on_global_remove(void *data, void *registry, uint32_t name) { (void)data, (void)registry, (void)name; }
static void (*registry_listener[])(void) = {(void (*)(void))on_global, (void (*)(void))on_global_remove};

static void on_sync_output(void *data, void *fb, void *output) { (void)data, (void)fb, (void)output; }

// Several requests share one commit and report the same time: a new time is a new frame on the display.
static void on_presented(void *data, void *fb, uint32_t sec_hi, uint32_t sec_lo, uint32_t nsec, uint32_t refresh,
                         uint32_t seq_hi, uint32_t seq_lo, uint32_t flags) {
    (void)data, (void)refresh, (void)seq_hi, (void)seq_lo, (void)flags;
    proxy_destroy(fb);
    g_pending--;
    uint64_t t = (((uint64_t)sec_hi << 32) | sec_lo) * 1000000000ull + nsec;
    if (t == g_last_shown) return;
    if (t < g_last_shown || t - g_last_shown > WINDOW_NS) g_first = t, g_frames = 0;   // a gap: a new window
    g_last_shown = t;
    g_frames++;
    if (t - g_first < WINDOW_NS) return;
    plat_xchg64(&g_rate_milli, (int64_t)((double)(g_frames - 1) * 1e12 / (double)(t - g_first)));
    plat_xchg64(&g_rate_ns, (int64_t)kern_now_ns());
    g_first = t, g_frames = 1;
}

static void on_discarded(void *data, void *fb) {
    (void)data;
    proxy_destroy(fb);
    g_pending--;
}
static void (*feedback_listener[])(void) = {(void (*)(void))on_sync_output, (void (*)(void))on_presented,
                                            (void (*)(void))on_discarded};

static int load_calls(void) {
    void *lib = dlopen("libwayland-client.so.0", RTLD_NOW | RTLD_GLOBAL);
    if (!lib) return 0;
#define CALL(var, name) if (!(var = dlsym(lib, name))) return 0
    CALL(display_create_queue, "wl_display_create_queue");
    CALL(proxy_create_wrapper, "wl_proxy_create_wrapper");
    CALL(proxy_set_queue, "wl_proxy_set_queue");
    CALL(proxy_marshal_flags, "wl_proxy_marshal_flags");
    CALL(proxy_add_listener, "wl_proxy_add_listener");
    CALL(proxy_destroy, "wl_proxy_destroy");
    CALL(proxy_get_version, "wl_proxy_get_version");
    CALL(display_roundtrip_queue, "wl_display_roundtrip_queue");
    CALL(display_prepare_read_queue, "wl_display_prepare_read_queue");
    CALL(display_dispatch_queue_pending, "wl_display_dispatch_queue_pending");
    CALL(display_read_events, "wl_display_read_events");
    CALL(display_cancel_read, "wl_display_cancel_read");
    CALL(display_flush, "wl_display_flush");
    CALL(display_get_fd, "wl_display_get_fd");
    CALL(registry_if, "wl_registry_interface");
#undef CALL
    return 1;
}

// A queue of its own, so the events are handled on this thread and the window system's are left alone.
static int bind_presentation(void) {
    g_queue = display_create_queue(g_display);
    void *wrapper = g_queue ? proxy_create_wrapper(g_display) : NULL;
    if (!wrapper) return 0;
    proxy_set_queue(wrapper, g_queue);
    g_registry = proxy_marshal_flags(wrapper, 1 /* get_registry */, registry_if, proxy_get_version(wrapper), 0, NULL);
    if (!g_registry) return 0;
    proxy_add_listener(g_registry, registry_listener, NULL);
    if (display_roundtrip_queue(g_display, g_queue) < 0 || !g_presentation_name) return 0;
    g_presentation = proxy_marshal_flags(g_registry, 0 /* bind */, &presentation_if, 1, 0, g_presentation_name,
                                         presentation_if.name, 1, NULL);
    return g_presentation != NULL;
}

static void shown_main(void *arg) {
    (void)arg;
    if (!load_calls() || !bind_presentation()) {
        rt_log("gfx: shown frames are not counted (no presentation feedback from the compositor)");
        return;
    }
    rt_log("gfx: shown frames are counted from the compositor's presentation feedback");
    struct pollfd pfd = {.fd = display_get_fd(g_display), .events = POLLIN};
    uint64_t armed = 0;
    for (;;) {
        uint64_t now = kern_now_ns();
        int wanted = now - (uint64_t)g_wanted_ns < WANTED_NS;
        if (wanted && now - armed >= ARM_NS && g_pending < 64) {
            void *fb = proxy_marshal_flags(g_presentation, 1 /* feedback */, &feedback_if,
                                           proxy_get_version(g_presentation), 0, g_surface, NULL);
            if (fb) proxy_add_listener(fb, feedback_listener, NULL), g_pending++;
            armed = now;
        }
        while (display_prepare_read_queue(g_display, g_queue) != 0) display_dispatch_queue_pending(g_display, g_queue);
        display_flush(g_display);
        if (poll(&pfd, 1, wanted ? 2 : 250) > 0) display_read_events(g_display);
        else display_cancel_read(g_display);
        if (display_dispatch_queue_pending(g_display, g_queue) < 0) return;   // the connection is gone
    }
}

void gfx_shown_start(void *wl_display, void *wl_surface) {
    static int started;
    if (started || !wl_display || !wl_surface) return;
    started = 1;
    g_display = wl_display, g_surface = wl_surface;
    plat_thread_spawn(shown_main, NULL, "shown frames", 0);
}

double gfx_shown_fps(void) {
    uint64_t now = kern_now_ns();
    plat_xchg64(&g_wanted_ns, (int64_t)now);
    if (now - (uint64_t)g_rate_ns > 1500000000ull) return 0;
    return (double)g_rate_milli / 1000.0;
}

#elif defined(_WIN32)
#include "input_win.h"

// The graphics kernel counts the frames each process put on a display (a VidPn source), the ones the driver generates
// included. The toolchain has no header for the call, so its structs are written out as far as they are used.
typedef struct { WCHAR DeviceName[32]; UINT hAdapter; LUID AdapterLuid; UINT VidPnSourceId; } KmtOpenAdapter;
typedef struct { UINT hAdapter; } KmtCloseAdapter;
typedef struct { ULONG Frame, CancelledFrame, QueuedPresent; ULONG64 Reserved[8]; } KmtSourceFrames;
typedef struct {
    UINT Type;
    LUID AdapterLuid;
    HANDLE hProcess;
    union { KmtSourceFrames frames; ULONG64 all[97]; } Result;   // as large as the largest answer of the call
    UINT VidPnSourceId;
} KmtQueryStatistics;
_Static_assert(sizeof(KmtQueryStatistics) == 808, "the layout of D3DKMT_QUERYSTATISTICS");
#define KMT_PROCESS_VIDPNSOURCE 8   // D3DKMT_QUERYSTATISTICS_PROCESS_VIDPNSOURCE
LONG WINAPI D3DKMTOpenAdapterFromGdiDisplayName(KmtOpenAdapter *open);
LONG WINAPI D3DKMTCloseAdapter(const KmtCloseAdapter *close);
LONG WINAPI D3DKMTQueryStatistics(KmtQueryStatistics *query);

#define WINDOW_NS 500000000ull   // the rate is taken over half a second of the counter

static HMONITOR g_mon;   // the monitor the adapter and source below are of
static LUID g_luid;
static UINT g_source;

static int source_find(HMONITOR mon) {
    MONITORINFOEXW mi = {.cbSize = sizeof mi};
    KmtOpenAdapter open = {0};
    if (!GetMonitorInfoW(mon, (MONITORINFO *)&mi)) return 0;
    memcpy(open.DeviceName, mi.szDevice, sizeof open.DeviceName);
    if (D3DKMTOpenAdapterFromGdiDisplayName(&open)) return 0;
    KmtCloseAdapter close = {open.hAdapter};
    D3DKMTCloseAdapter(&close);
    g_luid = open.AdapterLuid, g_source = open.VidPnSourceId, g_mon = mon;
    return 1;
}

// The process's frame count on the window's monitor. 0: none; 2: the monitor changed, so the count starts again.
static int frames_read(ULONG *frames) {
    HWND w = gfx_window_handle();
    HMONITOR mon = w ? MonitorFromWindow(w, MONITOR_DEFAULTTONEAREST) : NULL;
    int found = 1;
    if (!mon) return 0;
    if (mon != g_mon) {
        if (!source_find(mon)) return 0;
        found = 2;
    }
    KmtQueryStatistics q = {.Type = KMT_PROCESS_VIDPNSOURCE, .AdapterLuid = g_luid, .hProcess = GetCurrentProcess(),
                            .VidPnSourceId = g_source};
    if (D3DKMTQueryStatistics(&q)) return g_mon = NULL, 0;
    *frames = q.Result.frames.Frame;
    return found;
}

void gfx_shown_start(void *wl_display, void *wl_surface) { (void)wl_display, (void)wl_surface; }

// Called by the thread that draws the overlay. 0 while the counter stands still (nothing of the process is shown).
double gfx_shown_fps(void) {
    static uint64_t t0, rate_ns;
    static ULONG f0;
    static int have = -1;   // -1 before the first read
    static double rate;
    uint64_t now = kern_now_ns();
    if (!t0 || now - t0 >= WINDOW_NS) {
        ULONG f = 0;
        int r = frames_read(&f);
        if (r == 1 && have == 1 && now - t0 < 4 * WINDOW_NS)
            rate = (double)(ULONG)(f - f0) * 1e9 / (double)(now - t0), rate_ns = now;
        if ((r != 0) != have)
            rt_log(r ? "gfx: shown frames are counted by the graphics kernel (the process's frames on the display)"
                     : "gfx: shown frames are not counted (no frame count from the graphics kernel)");
        t0 = now, f0 = f, have = r != 0;
    }
    return rate_ns && now - rate_ns < 1500000000ull ? rate : 0;
}

#else

void gfx_shown_start(void *wl_display, void *wl_surface) { (void)wl_display, (void)wl_surface; }
double gfx_shown_fps(void) { return 0; }

#endif
