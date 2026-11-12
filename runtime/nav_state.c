// OOT_STATE_FILE=<file>: one JSON line (frame, ms, camera eye and fwd, Link's position) replaced atomically about 10 times a
// second by a thread of its own; guest memory is read with process_vm_readv, so a stale pointer gives an error, not a crash.
#if defined(__linux__)
#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

// The main camera block: its first three vec4 are the view matrix's rows (vk_usnap.c). The fixed address holds only for the
// stock heap layout; nav_state_camera replaces it with the scene camera the DLSS tracker recognised (vk_exec_dlss.c).
#define CAMERA_BLOCK 0x25e93b000ull
static volatile uint64_t cam_addr = CAMERA_BLOCK;
void nav_state_camera(uint64_t addr) { if (addr) cam_addr = addr; }
// Found by behaviour: a static slot in main's data holds a 0x32068-byte object with Link's live position, a facing vector
// after it and a string object with the map's name.
#define PLAYER_SLOT 0x7004e44598ull
#define PLAYER_POS 0x3b8              // 3 floats, then 3 floats facing
#define PLAYER_MAP 0x430              // pointer to the name, inline at the end of a small string object

static const char *path;
static volatile uint64_t frame_now;

static int peek(uint64_t addr, void *out, size_t n) {
    struct iovec l = {out, n}, r = {(void *)(uintptr_t)addr, n};
    return process_vm_readv(getpid(), &l, 1, &r, 1, 0) == (ssize_t)n;
}

// OOT_INVINCIBLE=1 (tests only): Link's life is put back every 20 ms on this thread. The player actor has vtable main+0x4b4c460
// and a resource record at +16 named "Player" at +88; the life is the u32 at [actor + 7184] + 8. The actor is searched among the
// pointers of the PLAYER_SLOT object and of the objects those point to, again whenever it stops validating (area load).
// Only that one word is written; reads and the write go through process_vm_readv/writev.
#define PLAYER_VTABLE 0x7004b4c460ull
#define PLAYER_LIFE_PTR 7184
static int is_player(uint64_t a) {
    uint64_t vt = 0, res = 0;
    uint8_t name[8] = {0};
    return a >= 0x100000000ull && !(a & 7) && peek(a, &vt, 8) && vt == PLAYER_VTABLE && peek(a + 16, &res, 8) && res &&
           peek(res + 88, name, 7) && !memcmp(name, "Player", 7);
}
static uint64_t find_player(uint64_t obj) {
    enum { WORDS = 0x32068 / 8 };
    static uint64_t w[WORDS], w2[64];
    if (is_player(obj)) return obj;
    if (!peek(obj, w, sizeof w)) return 0;
    for (int i = 0; i < WORDS; i++)
        if (is_player(w[i])) return w[i];
    for (int i = 0; i < WORDS; i++) {   // one level down: the first 64 words of each object a word points to
        uint64_t p = w[i];
        if (p < 0x100000000ull || p >= 0x7000000000ull || (p & 7) || !peek(p, w2, sizeof w2)) continue;
        for (int k = 0; k < 64; k++)
            if (is_player(w2[k])) return w2[k];
    }
    return 0;
}
// The player actor is not always reachable from the player slot object: search every writable mapping of the guest's low
// address space for the vtable word (4 MiB reads, each hit checked with is_player). Run only when there is no actor, at most every 30 s.
static uint64_t heap_scan(void) {
    enum { CHUNK = 4 << 20 };
    static uint64_t buf[CHUNK / 8];
    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) return 0;
    char line[512];
    uint64_t found = 0;
    while (!found && fgets(line, sizeof line, f)) {
        unsigned long long lo, hi;
        char perm[8];
        if (sscanf(line, "%llx-%llx %7s", &lo, &hi, perm) != 3 || perm[0] != 'r' || perm[1] != 'w') continue;
        if (lo < 0x100000000ull || hi > 0x7000000000ull) continue;   // below the modules: the heap block and the alias region
        for (uint64_t at = lo; at < hi && !found; at += CHUNK) {
            uint64_t n = hi - at < CHUNK ? hi - at : CHUNK;
            if (!peek(at, buf, n)) continue;
            for (uint64_t i = 0; i < n / 8; i++)
                if (buf[i] == PLAYER_VTABLE && is_player(at + i * 8)) { found = at + i * 8; break; }
        }
    }
    fclose(f);
    return found;
}

static void invincible_tick(void) {
    static uint64_t actor, life_at, tries;
    static uint32_t full;
    static int told;
    uint64_t obj = 0, lp = 0;
    if (!peek(PLAYER_SLOT, &obj, 8) || !obj) { actor = 0; return; }
    if (!actor || !is_player(actor)) {
        actor = 0, full = 0;
        static uint64_t every = 50, next;   // a search a second while there is no actor, then rarer (up to 30 s)
        if (tries++ < next) return;
        next = tries + every;
        if (every < 1500) every *= 2;
        if (!(actor = find_player(obj)) && !(actor = heap_scan())) {
            if (told < 3 && tries > 500) told = 3, fprintf(stderr, "[host] nav: OOT_INVINCIBLE: no player actor found (player slot object, every writable guest mapping)\n");
            return;
        }
    }
    uint32_t life = 0;
    if (!peek(actor + PLAYER_LIFE_PTR, &lp, 8) || !lp || !peek(lp + 8, &life, 4)) return;
    if (lp + 8 != life_at) life_at = lp + 8, full = 0;
    if (life > full && life <= 400) {
        full = life;
        if (told < 2) told++, fprintf(stderr, "[host] nav: OOT_INVINCIBLE: player actor %#llx, life %u at %#llx: kept there\n",
                                      (unsigned long long)actor, life, (unsigned long long)life_at);
    } else if (life < full) {
        struct iovec l = {&full, 4}, r = {(void *)(uintptr_t)life_at, 4};
        process_vm_writev(getpid(), &l, 1, &r, 1, 0);
    }
}

static void *writer(void *arg) {
    (void)arg;
    char tmp[1024];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    const char *inv = getenv("OOT_INVINCIBLE");
    int invincible = inv && *inv && strcmp(inv, "0");
    for (unsigned tick = 0;; tick++) {
        struct timespec ts = {0, 20000000};
        nanosleep(&ts, NULL);
        if (invincible) invincible_tick();
        if (tick % 5) continue;   // the state file every 100 ms
        float v[12];
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        FILE *f = fopen(tmp, "w");
        if (!f) continue;
        fprintf(f, "{\"frame\":%llu,\"ms\":%lld", (unsigned long long)frame_now, (long long)now.tv_sec * 1000 + now.tv_nsec / 1000000);
        if (peek(cam_addr, v, sizeof v)) {
            float e[3], t[3] = {v[3], v[7], v[11]};
            for (int i = 0; i < 3; i++) e[i] = -(v[i] * t[0] + v[4 + i] * t[1] + v[8 + i] * t[2]);
            fprintf(f, ",\"eye\":[%.3f,%.3f,%.3f],\"fwd\":[%.4f,%.4f,%.4f]", e[0], e[1], e[2], -v[8], -v[9], -v[10]);
        }
        uint64_t obj = 0, name_at = 0;
        float p[6];
        char map[64] = "";
        int world = 0;
        if (peek(PLAYER_SLOT, &obj, 8) && obj && peek(obj + PLAYER_POS, p, sizeof p) && peek(obj + PLAYER_MAP, &name_at, 8) && name_at &&
            peek(name_at, map, sizeof map - 1)) {
            map[sizeof map - 1] = 0;
            for (char *c = map; *c; c++) if (*c < 32 || *c > 126 || *c == '"' || *c == '\\') *c = '?';
            world = map[0] != 0 && (p[0] != 0 || p[1] != 0 || p[2] != 0) && p[0] == p[0] && p[1] == p[1] && p[2] == p[2];
            fprintf(f, ",\"pos\":[%.3f,%.3f,%.3f],\"face\":[%.4f,%.4f,%.4f],\"map\":\"%s\"", p[0], p[1], p[2], p[3], p[4], p[5], map);
        }
        fprintf(f, ",\"world\":%d}\n", world);
        fclose(f);
        rename(tmp, path);
    }
    return NULL;
}

void nav_state_present(uint64_t frame) {
    static int state;   // 0 not read, 1 on, -1 off
    if (!state) {
        path = getenv("OOT_STATE_FILE");
        state = path && *path ? 1 : -1;
        if (state > 0) {
            pthread_t t;
            pthread_attr_t a;
            pthread_attr_init(&a);
            pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);
            if (pthread_create(&t, &a, writer, NULL)) state = -1;
        }
    }
    frame_now = frame;
}
#else
#include <stdint.h>
void nav_state_present(uint64_t frame) { (void)frame; }
void nav_state_camera(uint64_t addr) { (void)addr; }
#endif
