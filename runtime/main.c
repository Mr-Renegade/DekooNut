// oot.exe entry: reserves the guest address space, parses the command line and runs one mode.
//   oot.exe [--exefs DIR] [--romfs DIR] [--profile DIR] [--seconds S [--shot PNG]]
// --seconds exits 0 at the present after S seconds of world clock (game time since the save's first frame); --shot saves it.
// Also here: the runtime services the generated code calls (rt_brk, rt_udf, rt_ticks, profiler hook), logging, the crash reporter.
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "platform/platform.h"
#include "sha256.h"

#define MAX_MODULES 16

// ---- logging, tracing, hashing
unsigned rt_trace;

// Log file: a game run whose stderr is not a file also writes logs\oot.log next to oot.exe; each start rotates the earlier
// logs to oot.1.log .. oot.4.log; a file stops at 64 MiB. Lines from before log_open wait in a static buffer: mem_init runs
// first so nothing else lands in the guest address space, and an earlier open would put the CRT's heap there.
#define LOG_KEEP 5
#define LOG_CAP (64ull << 20)
enum { LOG_EARLY, LOG_FILE, LOG_OFF };
static int g_log_state;
static PlatAppend *g_log_file;
static char g_log_path[PLAT_PATH_MAX * 3];
static char g_log_early[32768];
static size_t g_log_early_n;
static uint64_t g_log_bytes;

void rt_log_raw(const char *p, size_t n) {
    fwrite(p, 1, n, stderr);
    int state = __atomic_load_n(&g_log_state, __ATOMIC_ACQUIRE);
    if (state == LOG_EARLY) {
        size_t at = __atomic_fetch_add(&g_log_early_n, n, __ATOMIC_RELAXED);
        if (at + n <= sizeof g_log_early) memcpy(g_log_early + at, p, n);
    } else if (state == LOG_FILE) {
        uint64_t at = __atomic_fetch_add(&g_log_bytes, n, __ATOMIC_RELAXED);
        if (at + n <= LOG_CAP) {
            plat_append_write(g_log_file, p, n);
        } else if (at <= LOG_CAP) {   // the first write past the cap says so, once
            static const char cut[] = "[host] log: the file reached 64 MiB, the rest of this run goes to stderr only\n";
            plat_append_write(g_log_file, cut, sizeof cut - 1);
        }
    }
}

void rt_vlogf(const char *fmt, va_list ap) {
    char buf[2048];
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    if (n > 0) rt_log_raw(buf, n < (int)sizeof buf ? (size_t)n : sizeof buf - 1);
}

void rt_logf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    rt_vlogf(fmt, ap);
    va_end(ap);
}

void rt_log(const char *fmt, ...) {
    char buf[2048];
    int n = snprintf(buf, sizeof buf, "[host] ");
    va_list ap;
    va_start(ap, fmt);
    int k = vsnprintf(buf + n, sizeof buf - n - 1, fmt, ap);
    va_end(ap);
    n = k < 0 ? n : n + k < (int)sizeof buf - 1 ? n + k : (int)sizeof buf - 2;
    buf[n++] = '\n';
    rt_log_raw(buf, n);
}

const char *rt_log_path(void) {
    return __atomic_load_n(&g_log_state, __ATOMIC_ACQUIRE) == LOG_FILE ? g_log_path : NULL;
}

// Opens the log file for a game run (game = 1) and writes the buffered lines to it; anything else drops them. Runs on
// the main thread before the boot starts other threads.
static void log_open(int game) {
    char dir[PLAT_PATH_MAX * 3], from[PLAT_PATH_MAX * 3 + 16], to[PLAT_PATH_MAX * 3 + 16];
    char *cut = NULL;
    if (game && !plat_stderr_path(dir, sizeof dir) && plat_exe_path(dir, sizeof dir)) {
        cut = plat_last_sep(dir);
        char *f = strrchr(dir, '/');
        if (f > cut) cut = f;
    }
    if (cut && (size_t)(cut - dir) + sizeof "/logs" <= sizeof dir) {
        strcpy(cut, "/logs");
        plat_file_mkdir_p(dir);
        for (int i = LOG_KEEP - 1; i > 0; i--) {   // oot.3.log -> oot.4.log, ..., oot.log -> oot.1.log
            if (i > 1) snprintf(from, sizeof from, "%s/oot.%d.log", dir, i - 1);
            else snprintf(from, sizeof from, "%s/oot.log", dir);
            snprintf(to, sizeof to, "%s/oot.%d.log", dir, i);
            if (plat_file_exists(from)) plat_file_replace(from, to);
        }
        snprintf(g_log_path, sizeof g_log_path, "%s/oot.log", dir);
        g_log_file = plat_append_open(g_log_path);
    }
    if (!g_log_file) {
        __atomic_store_n(&g_log_state, LOG_OFF, __ATOMIC_RELEASE);
        return;
    }
    size_t n = __atomic_load_n(&g_log_early_n, __ATOMIC_RELAXED);
    if (n > sizeof g_log_early) n = sizeof g_log_early;
    plat_append_write(g_log_file, g_log_early, n);
    g_log_bytes = n;
    __atomic_store_n(&g_log_state, LOG_FILE, __ATOMIC_RELEASE);
    rt_log("log: %s", g_log_path);
}

static void parse_trace(void) {
    const char *e = cfg_get("OOT_TRACE");   // no default: nothing is traced when unset
    if (!e) return;
    static const struct { const char *name; unsigned bit; } names[] = {
        {"svc", RT_TRACE_SVC}, {"host", RT_TRACE_HOSTCALL}, {"ipc", RT_TRACE_IPC}, {"fs", RT_TRACE_FS},
        {"gfx", RT_TRACE_GFX}, {"perf", RT_TRACE_PERF}, {"sched", RT_TRACE_SCHED}, {"audio", RT_TRACE_AUDIO},
        {"present", RT_TRACE_PRESENT}, {"codec", RT_TRACE_CODEC}, {"all", ~0u}};
    for (const char *p = e; *p;) {
        size_t n = strcspn(p, ",");
        int known = 0;
        for (unsigned i = 0; i < sizeof names / sizeof *names; i++)
            if (strlen(names[i].name) == n && !strncmp(p, names[i].name, n)) rt_trace |= names[i].bit, known = 1;
        if (!known && n) rt_log("OOT_TRACE: unknown item '%.*s' (svc, host, ipc, fs, gfx, perf, sched, audio, present, codec, all)", (int)n, p);
        p += n + (p[n] == ',');
    }
}

void rt_sha256(const void *p, size_t n, uint8_t out[32]) { sha256(p, n, out); }

// ---- configuration
static RtConfig cfg;
const RtConfig *rt_config(void) { return &cfg; }

// ---- runtime services the generated code calls (cpu.h)

void rt_brk(Cpu *c, uint32_t imm, uint64_t pc) {
    char at[128], lr[128];
    rt_fatal("brk #%#x at %s (lr %s)", imm, ldr_describe(pc, at, sizeof at), ldr_describe(c->x[30], lr, sizeof lr));
}

void rt_udf(Cpu *c, uint64_t pc) {
    char at[128], lr[128];
    rt_fatal("undefined instruction at %s (lr %s)", ldr_describe(pc, at, sizeof at),
             ldr_describe(c->x[30], lr, sizeof lr));
}

// CNTPCT_EL0: the Switch counter runs at 19.2 MHz, the Switch 2's at 31.25 MHz (the SDK converts with 31250000 / 10^9);
// at 19.2 MHz the game would pace itself at 20 FPS. Set before main, so no thread reads it half written.
static PlatScale tick_scale;

uint32_t rt_tick_hz(void) { return 31250000u; }

__attribute__((constructor)) static void tick_scale_init(void) { tick_scale = plat_scale(rt_tick_hz(), plat_counter_hz()); }

uint64_t rt_ticks(void) { return plat_scaled(tick_scale, plat_counter()); }
uint64_t rt_ticks_to_ns(uint64_t t) { return (uint64_t)((unsigned __int128)t * 1000000000u / rt_tick_hz()); }
uint64_t rt_ns_to_ticks(uint64_t ns) { return (uint64_t)((unsigned __int128)ns * rt_tick_hz() / 1000000000u); }

// ---- crash reporter
typedef struct { uintptr_t host; uint64_t guest; } FnRev;
static FnRev *fn_rev;
static uint64_t fn_rev_n;

static int by_host(const void *a, const void *b) {
    uintptr_t x = ((const FnRev *)a)->host, y = ((const FnRev *)b)->host;
    return x < y ? -1 : x > y;
}

// The lifted function that contains a host code address, by the nearest function start below it; 0 outside the lifted
// code.
static uint64_t lifted_function(uintptr_t rip) {
    if (!fn_rev) {
        static RtModule mods[MAX_MODULES];
        int n = rt_linked_modules(mods, MAX_MODULES);
        uint64_t total = 0;
        for (int i = 0; i < n; i++) total += mods[i].count;
        fn_rev = malloc((total ? total : 1) * sizeof *fn_rev);
        if (!fn_rev) return 0;
        for (int i = 0; i < n; i++)
            for (uint64_t k = 0; k < mods[i].count; k++)
                fn_rev[fn_rev_n++] = (FnRev){(uintptr_t)mods[i].fns[k], mods[i].addrs[k]};
        qsort(fn_rev, fn_rev_n, sizeof *fn_rev, by_host);
    }
    // The lifted code is one range with the host code outside it (kernel.c ap_init checks that).
    if (!fn_rev_n || rip > fn_rev[fn_rev_n - 1].host) return 0;
    uint64_t lo = 0, hi = fn_rev_n;
    while (lo < hi) {
        uint64_t mid = (lo + hi) / 2;
        if (fn_rev[mid].host <= rip) lo = mid + 1;
        else hi = mid;
    }
    return lo ? fn_rev[lo - 1].guest : 0;
}

static int readable(uint64_t a, uint64_t n) { return plat_mem_readable(a, n); }

// Itanium C++ RTTI of the object a guest register points to: [obj] is the vtable, [vtable - 8] the type_info, [type_info + 8]
// its mangled name. Returns 0 unless every step is readable and the name is printable.
static int rtti_name(uint64_t obj, char *out, int n) {
    if (!obj || (obj & 7) || !readable(obj, 8)) return 0;
    uint64_t vt = LD64(obj);
    if ((vt & 7) || vt < 0x7000000000ull || !readable(vt - 8, 8)) return 0;
    uint64_t ti = LD64(vt - 8);
    if ((ti & 7) || ti < 0x7000000000ull || !readable(ti, 16)) return 0;
    uint64_t nm = LD64(ti + 8);
    if (nm < 0x7000000000ull || !readable(nm, 4)) return 0;
    int i = 0;
    for (; i < n - 1; i++) {
        if (!readable(nm + i, 1)) return 0;
        unsigned char ch = *(const unsigned char *)(uintptr_t)(nm + i);
        if (!ch) break;
        if (ch < 0x20 || ch > 0x7e) return 0;
        out[i] = (char)ch;
    }
    out[i] = 0;
    return i >= 2;
}

static void crash_handler(const PlatCrash *x) {
    static volatile int32_t busy;
    if (plat_xchg32(&busy, 1)) plat_exit(71);
    char d[160];
    uintptr_t rip = x->rip;
    char module[128];
    uintptr_t module_base;
    if (plat_module_of(rip, module, sizeof module, &module_base))
        rt_logf("FATAL: %s (%#x) at host %p (%s+%#llx)", x->what, x->code, (void *)rip, module,
                (unsigned long long)(rip - module_base));
    else
        rt_logf("FATAL: %s (%#x) at host %p (private or unmapped memory)", x->what, x->code, (void *)rip);
    if (x->access >= 0)
        rt_logf(", %s address %#llx", x->access == 1 ? "writing" : x->access == 8 ? "executing" : "reading",
                (unsigned long long)x->addr);
    rt_log_raw("\n", 1);
    if (x->has_registers) {
        static const char *const names[16] = {"rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
                                              "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"};
        for (int i = 0; i < 16; i++)
            rt_log("host register: %s=%#llx", names[i], (unsigned long long)x->registers[i]);
        uint64_t stack[16];
        if (plat_mem_read(x->registers[4], stack, sizeof stack))
            for (int i = 0; i < 16; i++)
                rt_log("host stack: +%#x=%#llx", i * 8, (unsigned long long)stack[i]);
    }
    uint64_t f = lifted_function(rip);
    if (f) rt_logf("  in lifted function f_%llx = %s\n", (unsigned long long)f, ldr_describe(f, d, sizeof d));
    for (int i = 0; i < x->nframes; i++) {   // abort and invalid parameter: where the host code called it
        uint64_t g = lifted_function(x->frames[i]);
        rt_logf("  host frame %2d: oot+%#llx", i, (unsigned long long)(x->frames[i] - (rip - x->offset)));
        if (g) rt_logf(" in lifted function f_%llx", (unsigned long long)g);
        rt_log_raw("\n", 1);
    }
    Cpu *c = kern_cpu();
    if (c) {
        for (int i = 0; i < 31; i++)
            rt_logf("  x%-2d %016llx%s", i, (unsigned long long)c->x[i], i % 4 == 3 ? "\n" : "");
        rt_logf("  sp  %016llx\n  pc  %016llx  lr %s\n", (unsigned long long)c->sp, (unsigned long long)c->pc,
                ldr_describe(c->x[30], d, sizeof d));
        // The stack words above sp hold the callee-saved registers of the callers; the vtable word of each tells the call chain's objects apart.
        for (int i = 0; i < 24 && readable(c->sp + 8 * i, 8); i++) {
            uint64_t w = LD64(c->sp + 8 * i);
            if (w >= 0x100000000ull && !(w & 7) && readable(w, 8))
                rt_logf("  [sp+%#x] %#llx -> %#llx\n", 8 * i, (unsigned long long)w, (unsigned long long)LD64(w));
            else
                rt_logf("  [sp+%#x] %#llx\n", 8 * i, (unsigned long long)w);
        }
        for (int i = 0; i < 29; i++) {   // registers that hold a C++ object: name its class
            char nm[120];
            if (rtti_name(c->x[i], nm, sizeof nm)) rt_logf("  x%-2d is a %s\n", i, nm);
        }
        uint64_t fp = c->x[29];
        for (int n = 0; n < 32 && fp && !(fp & 7) && readable(fp, 16); n++) {
            uint64_t next = LD64(fp), lr = LD64(fp + 8);
            rt_logf("  guest frame %2d: %s\n", n, ldr_describe(lr, d, sizeof d));
            if (next <= fp || next - fp > (64u << 20)) break;
            fp = next;
        }
    } else {
        rt_logf("  not on a guest thread\n");
    }
    fflush(stderr);
    plat_exit(71);
}

static int usage(const char *argv0) {
    fprintf(stderr, "usage: %s [--exefs DIR] [--romfs DIR] [--profile DIR] [--prepare-shaders | --warm-shaders | --seconds S [--shot PNG]]\n", argv0);
    return 2;
}

// Paths for a copied folder. Per path: OOT_EXEFS, OOT_ROMFS, OOT_PROFILE_DIR, else the folder next to oot.exe if it is there,
// else the path baked in at build time. A relative value is relative to oot.exe's folder.
static char pp_exefs[PLAT_PATH_MAX * 2], pp_romfs[PLAT_PATH_MAX * 2], pp_profile[PLAT_PATH_MAX * 2];

static const char *portable_path(const char *dir, const char *var, const char *sub, const char *must, char *out, size_t cap,
                                 const char *fallback) {
    const char *e = cfg_get(var);
    if (e && *e) {
        int abs_path = e[0] == '\\' || e[0] == '/' || e[1] == ':';
        if (abs_path) snprintf(out, cap, "%s", e);
        else snprintf(out, cap, "%s/%s", dir, e);
        return out;
    }
    char probe[PLAT_PATH_MAX * 3];
    snprintf(out, cap, "%s/%s", dir, sub);
    if (!must) return plat_dir_exists(out) ? out : fallback;   // the folder itself is the sign
    snprintf(probe, sizeof probe, "%s/%s", out, must);
    return plat_file_exists(probe) ? out : fallback;
}

static void portable_paths(RtConfig *c) {
    char exe[PLAT_PATH_MAX * 2];
    int n = plat_exe_path(exe, sizeof exe);
    char *cut = n && n < sizeof exe ? plat_last_sep(exe) : NULL;
    if (!cut) return;
    *cut = 0;
    c->exefs = portable_path(exe, "OOT_EXEFS", "exefs", "main.npdm", pp_exefs, sizeof pp_exefs, c->exefs);
    c->romfs = portable_path(exe, "OOT_ROMFS", "romfs", "romfs.bin", pp_romfs, sizeof pp_romfs, c->romfs);
    c->profile = portable_path(exe, "OOT_PROFILE_DIR", "profile", NULL, pp_profile, sizeof pp_profile, c->profile);
}

int main(int argc, char **argv) {
    mem_init();   // first: claim the guest address space before anything else can land there; reads oot.ini
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    plat_crash_hook(crash_handler);
    parse_trace();
    prof_start();
    // Timed guest waits (kernel.c block_inner) sleep in whole milliseconds: a 1 ms wait costs a full tick at 15.6 ms.
    plat_timer_1ms();
    cfg = (RtConfig){.exefs = rt_default_exefs, .romfs = rt_default_romfs, .profile = rt_default_profile};
    portable_paths(&cfg);   // command-line paths below still win
    const char *mode = NULL;
    for (int i = 1; i < argc; i++) {
        const char **path = !strcmp(argv[i], "--exefs") ? &cfg.exefs : !strcmp(argv[i], "--romfs") ? &cfg.romfs
                          : !strcmp(argv[i], "--profile") ? &cfg.profile : NULL;
        if (path) {
            if (++i == argc) return usage(argv[0]);
            *path = argv[i];
        } else if (!strcmp(argv[i], "--seconds")) {
            char *end;
            if (++i == argc) return usage(argv[0]);
            double s = strtod(argv[i], &end);
            if (*end || !(s > 0) || s > 1e6) return usage(argv[0]);
            cfg.seconds = s;
        } else if (!strcmp(argv[i], "--shot")) {
            if (++i == argc) return usage(argv[0]);
            cfg.shot = argv[i];
        } else if (!mode && !strncmp(argv[i], "--", 2)) {
            mode = argv[i];
        } else {
            return usage(argv[0]);
        }
    }
    if (cfg.shot && !cfg.seconds) return usage(argv[0]);
    if (cfg.seconds && mode) return usage(argv[0]);
    log_open(!mode);
    rt_log("build: %s (commit of the source tree; .dirty = uncommitted changes)", rt_port_version);
    if (!mode) {
        rt_boot();
        return 0;
    }
    if (!strcmp(mode, "--lower-shaders")) return vk_pipeline_lower_shaders();
    if (!strcmp(mode, "--prepare-shaders")) return vk_pipeline_prepare_shaders();
    if (!strcmp(mode, "--warm-shaders")) return vk_pipeline_warm_shaders();
    return usage(argv[0]);
}
