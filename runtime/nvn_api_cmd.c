// NVN front end, command-buffer slice: command buffers, their memory, recording
// as op records (nvn_obj.h), recording copies behind command handles, and the submit-time resolution of deferred binds
// (nvn_submit_prepare, called by nvnQueueSubmitCommands in nvn_api_core.c). Ported from MacNativeGames
// games/totk/third_party/d2r_gfx/gfx/nvn/hle_nvn.c @ b75be8cf (MIT, Copyright (c) 2026 jhaizhou; origin d2r-mobile),
// lines 647-1062, 1261-1286, 1486-1760 and 1975-1990, and the *Deferred and memory-callback entry points of
// totk_nvn_ext.c in the same directory. Metal, mach, the translation thread, Apple diagnostics, the per-frame
// statistics and the long-session sweeps (copy marks in control memory, texture pin sweep) dropped.
// Command recording and recording copies: see README, "NVN command recording".
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "gfx.h"
#include "host.h"
#include "vk.h"
#include "nvn_api.h"
#include "nvn_hmap.h"
#include "platform/platform.h"

#define CMD_MAGIC MAGIC('C', 'M', 'D', 'B')
#define COPY_MAGIC 0x59504f43444d43ull     // "CMDCOPY": prefix of a live recording copy
#define OP_RESERVE 24                       // room always kept at the end of the recording buffer
#define MIN_COMMAND_SIZE 512                // NVN_DEVICE_INFO_COMMAND_BUFFER_MIN_COMMAND_SIZE (nvn_api_core.c devinfo)
#define MIN_CONTROL_SIZE 256                // NVN_DEVICE_INFO_COMMAND_BUFFER_MIN_CONTROL_SIZE
#define CTL_PER_RECORDING 64                // control memory one recording is reported to use
#define COPY_PAGE_SHIFT 16                  // recording copy index granularity (release_copies)

static cmd_t *cmd_of(const Cpu *c) { return arg_p(c, 0); }

// ---------- memory callbacks
// NVN calls the game's memory callback when a recording needs command or control memory and none is left. Records live in
// host memory here, so it is called only where the hardware certainly would: a recording starts without command memory, or
// ends without control memory. It runs on the calling guest thread; Set{Command,Control}MemoryCallbackEnabled(false) suppress it.
enum { CB_NO_COMMAND = 1, CB_NO_CONTROL = 2 };
static PlatMutex cb_lock = PLAT_MUTEX_INIT;
static nvn_hmap cb_off = {.klen = 8};       // command buffer -> disabled callback kinds | 0x100

static unsigned cb_disabled(const cmd_t *o) {
    plat_mutex_lock_shared(&cb_lock);
    uintptr_t v = (uintptr_t)nvn_hm_get(&cb_off, &o);
    plat_mutex_unlock_shared(&cb_lock);
    return (unsigned)v & 0xff;
}

static void cb_set_disabled(const cmd_t *o, unsigned kind, int disabled) {
    plat_mutex_lock(&cb_lock);
    unsigned v = (unsigned)(uintptr_t)nvn_hm_get(&cb_off, &o) & 0xff;
    v = disabled ? v | kind : v & ~kind;
    if (v) nvn_hm_put(&cb_off, &o, (void *)(uintptr_t)(v | 0x100));
    else nvn_hm_del(&cb_off, &o);
    plat_mutex_unlock(&cb_lock);
}

static void memory_event(Cpu *c, cmd_t *o, int event, uint64_t min) {
    if (!o->cb || (cb_disabled(o) & (event == NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY ? CB_NO_COMMAND
                                                                                                     : CB_NO_CONTROL)))
        return;
    NVN_LOG("command buffer %p: memory callback %p (%s memory, %llu bytes)", (void *)o, (void *)o->cb,
            event == NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY ? "command" : "control",
            (unsigned long long)min);
    rt_call(c, (uint64_t)(uintptr_t)o->cb, 4, (uint64_t)(uintptr_t)o, (uint64_t)event, min,
            (uint64_t)(uintptr_t)o->cb_data);
}

// ---------- recording buffer
static void put_hdr(uint8_t *p, uint32_t op, uint32_t size) {
    op_hdr h = {op, size};
    memcpy(p, &h, sizeof h);
}

static void cmd_reserve(cmd_t *o, size_t need) {
    if (o->seg && o->cur + need + OP_RESERVE <= o->end) return;
    size_t used = o->seg ? (size_t)(o->cur - o->seg) : 0;
    size_t cap = o->seg ? (size_t)(o->end - o->seg) : 0;
    if (!cap) cap = 64 << 10;
    while (used + need + OP_RESERVE > cap) cap *= 2;
    uint8_t *b = realloc(o->seg, cap);
    // A block this size fails only without commit, or when seg is no block of ours (an overwritten or copied command buffer object).
    if (!b)
        rt_fatal("nvn: command recording buffer of %zu bytes (command buffer %p, magic %#x, buffer %p of %zu bytes, %zu "
                 "used, %zu asked)", cap, (void *)o, o->magic, (void *)o->seg, o->seg ? (size_t)(o->end - o->seg) : 0,
                 used, need);
    o->seg = b;
    o->cur = b + used;
    o->end = b + cap;
}

// Space for one record (header included); the header's op is filled by the caller.
static uint8_t *cmd_alloc(cmd_t *o, uint32_t size) {
    size = (size + 7) & ~7u;
    cmd_reserve(o, size);
    uint8_t *p = o->cur;
    o->cur += size;
    o->cmd_used += size;
    o->hw_used += size;
    o->last_rec = size;
    put_hdr(p, 0, size);
    return p;
}

// The game sums GetCommandMemoryUsed over recordings to size resource heaps, so usage is what the real NVN push-buffer
// encoding takes: DrawElements 12 words, DrawElementsBaseVertex 11, BindVertexAttribState 1+16, BindVertexStreamState 16+n,
// BindVertexBuffer 6 each.
static void hw_fix(cmd_t *o, size_t bytes) {
    o->hw_used += bytes;
    o->hw_used -= o->last_rec;
    o->last_rec = bytes;
}

// A zeroed payload of one record.
static void *rec(cmd_t *o, uint32_t op, size_t payload) {
    uint32_t sz = (uint32_t)((8 + payload + 7) & ~(size_t)7);
    uint8_t *p = cmd_alloc(o, sz);
    put_hdr(p, op, sz);
    memset(p + 8, 0, sz - 8);
    return p + 8;
}
#define REC(o, OP, T) ((T *)rec((o), (OP), sizeof(T)))
// Array-carrying binds take the prefix fields plus n elements; the replay reads only the first n.
#define RECN(o, OP, T, ARR, N) ((T *)rec((o), (OP), offsetof(T, ARR) + sizeof(((T *)0)->ARR[0]) * (size_t)(N)))

static void copy_view(view_t *dst, uint8_t *has, const NVNtextureView *v) {
    *has = v != NULL;
    if (v) *dst = *(const view_t *)v;
}

// ---------- texture pins
// Texture pins: records point at an immutable host copy of each texture initialization, since the game may finalize or reuse
// an NVNtexture between recording and submit. See README, "NVN command recording".
static PlatMutex pin_lock = PLAT_MUTEX_INIT;
static nvn_hmap pins = {.klen = 8};         // serial -> tex_t copy

// Pins are keyed by the texture's content, not its serial: the game re-initializes its transient textures every frame with a
// new serial, and nothing in the replay reads a pinned copy's serial. The copy keeps the serial of the first initialization.
static nvn_hmap pins_by_content = {.klen = 8};   // content hash -> tex_t copy
static uint64_t pin_content_hash(const tex_t *t, tex_t *tmp) {
    *tmp = *t;
    tmp->serial = 0;
    uint64_t h = 0xcbf29ce484222325ull;
    const uint8_t *b = (const uint8_t *)tmp;
    for (size_t i = 0; i < sizeof *tmp; i++) h = (h ^ b[i]) * 0x100000001b3ull;
    return h ? h : 1;
}
static int pin_dedup(void) {
    static int told;
    if (!told++) rt_log("nvn: texture pins shared by content");
    return 1;
}
static const tex_t *tex_pin_dedup(const tex_t *t) {
    tex_t tmp, other;
    uint64_t h = pin_content_hash(t, &tmp);
    for (int probe = 0; probe < 4; probe++, h = h * 0x9e3779b97f4a7c15ull + 1) {   // a collision with other content: next key
        plat_mutex_lock_shared(&pin_lock);
        const tex_t *r = nvn_hm_get(&pins_by_content, &h);
        plat_mutex_unlock_shared(&pin_lock);
        if (r) {
            other = *r;
            other.serial = 0;
            if (!memcmp(&other, &tmp, sizeof tmp)) return r;
            continue;
        }
        plat_mutex_lock(&pin_lock);
        r = nvn_hm_get(&pins_by_content, &h);
        if (!r) {
            tex_t *copy = malloc(sizeof *copy);
            if (!copy) rt_fatal("nvn: texture pin copy");
            *copy = *t;
            nvn_hm_put(&pins_by_content, &h, copy);
            r = copy;
        }
        plat_mutex_unlock(&pin_lock);
        other = *r;
        other.serial = 0;
        if (!memcmp(&other, &tmp, sizeof tmp)) return r;
    }
    rt_fatal("nvn: texture pin: four content hash collisions");
}

static const tex_t *tex_pin(const NVNtexture *nt) {
    const tex_t *t = (const tex_t *)nt;
    if (!t) return NULL;
    uint64_t serial = t->serial;
    if (!serial) rt_fatal("nvn: a recording references the uninitialized texture object %p", (const void *)t);
    if (pin_dedup()) return tex_pin_dedup(t);
    plat_mutex_lock_shared(&pin_lock);
    const tex_t *r = nvn_hm_get(&pins, &serial);
    plat_mutex_unlock_shared(&pin_lock);
    if (r) return r;
    plat_mutex_lock(&pin_lock);
    r = nvn_hm_get(&pins, &serial);
    if (!r) {
        tex_t *copy = malloc(sizeof *copy);
        if (!copy) rt_fatal("nvn: texture pin copy");
        *copy = *t;
        nvn_hm_put(&pins, &serial, copy);
        r = copy;
    }
    plat_mutex_unlock(&pin_lock);
    return r;
}

// ---------- recording copies
// A handle points just past a copy_prefix. Copies are indexed by the 64 KiB block of their control-memory position, so adding
// control memory releases the copies recorded there without walking every live copy.
typedef struct { uint64_t magic, bytes, made_at, used_at, deferred; } copy_prefix;   // deferred: 0 unknown, 1 none, 2 some
_Static_assert(sizeof(copy_prefix) % 8 == 0, "records stay 8-byte aligned");
typedef struct { uintptr_t lo, hi; uint8_t *copy; } cmd_copy;
typedef struct { cmd_copy *v; uint32_t n, cap; } copy_bucket;
static PlatMutex copy_lock = PLAT_MUTEX_INIT;    // guards the two maps, the counters and every prefix
static nvn_hmap copy_pages = {.klen = 8};   // page -> copy_bucket
static nvn_hmap copy_live = {.klen = 8};    // handle -> copy_prefix of a live copy
static uint64_t ncopies, copy_bytes, copies_made, nsubmits;
// Asynchronous replay: copies released by the game, and per-submission clones, that a queued submission still reads; freed
// once nvn_submit_count() reaches their used_at. Empty with synchronous replay.
static copy_prefix **zombies;
static uint32_t nzombies, cap_zombies;

static void zombie_add(copy_prefix *pf) {   // copy_lock held
    if (nzombies == cap_zombies) {
        cap_zombies = cap_zombies ? cap_zombies * 2 : 64;
        zombies = realloc(zombies, sizeof *zombies * cap_zombies);
        if (!zombies) rt_fatal("nvn: out of memory (released recording copies)");
    }
    zombies[nzombies++] = pf;
}

static void zombie_sweep(uint64_t replayed) {   // copy_lock held
    for (uint32_t i = 0; i < nzombies;) {
        if (zombies[i]->used_at <= replayed) {
            free(zombies[i]);
            zombies[i] = zombies[--nzombies];
        } else {
            i++;
        }
    }
}

static copy_prefix *copy_find(uint64_t handle) {   // copy_lock held
    return nvn_hm_get(&copy_live, &handle);
}

int nvn_handle_live(uint64_t handle) {
    plat_mutex_lock_shared(&copy_lock);
    int live = copy_find(handle) != NULL;
    plat_mutex_unlock_shared(&copy_lock);
    return live;
}

uint64_t nvn_handle_bytes(uint64_t handle) {
    plat_mutex_lock_shared(&copy_lock);
    const copy_prefix *pf = copy_find(handle);
    uint64_t n = pf ? pf->bytes : 0;
    plat_mutex_unlock_shared(&copy_lock);
    return n;
}

// Marks a live copy used by submission now; its size from OP_BEGIN to OP_END inclusive.
static uint64_t copy_touch(uint64_t handle, uint64_t now, const char *what) {
    plat_mutex_lock_shared(&copy_lock);
    copy_prefix *pf = copy_find(handle);
    uint64_t bytes = 0;
    if (pf) __atomic_store_n(&pf->used_at, now, __ATOMIC_RELAXED), bytes = pf->bytes;
    plat_mutex_unlock_shared(&copy_lock);
    if (!pf) rt_fatal("nvn: %s: command handle %#llx is not a live recording (released or never recorded)", what,
                      (unsigned long long)handle);
    return bytes;
}

// The recording [p, p + n), OP_BEGIN to OP_END, copied into one host block of its exact size (one allocation, one copy).
static uint8_t *flatten_recording(const uint8_t *p, size_t n) {
    uint8_t *out = malloc(sizeof(copy_prefix) + n);
    if (!out) rt_fatal("nvn: command copy allocation");
    memcpy(out + sizeof(copy_prefix), p, n);
    for (size_t at = 0;;) {   // the records stay checked: a corrupt one stops here, not in the replay
        op_hdr h;
        memcpy(&h, p + at, sizeof h);
        if (h.size < 8 || (h.size & 7) || h.op >= OP_COUNT_ || h.op == OP_JUMP || h.op == OP_CALL || at + h.size > n ||
            (!at && h.op != OP_BEGIN))
            rt_fatal("nvn: corrupt recording at %p (op %u size %u)", (const void *)(p + at), h.op, h.size);
        at += h.size;
        if (h.op == OP_END) {
            if (at != n) rt_fatal("nvn: recording at %p ends at %zu of %zu bytes", (const void *)p, at, n);
            break;
        }
    }
    *(copy_prefix *)out = (copy_prefix){COPY_MAGIC, n, 0, 0};
    return out + sizeof(copy_prefix);
}

static void copy_add(uintptr_t lo, uintptr_t hi, uint8_t *copy) {
    copy_prefix *pf = (copy_prefix *)copy - 1;
    uint64_t handle = (uintptr_t)copy;
    uintptr_t page = lo >> COPY_PAGE_SHIFT;
    plat_mutex_lock(&copy_lock);
    pf->made_at = nsubmits;
    copy_bucket *b = nvn_hm_get(&copy_pages, &page);
    if (!b) {
        b = calloc(1, sizeof *b);
        if (!b) rt_fatal("nvn: command copy index");
        nvn_hm_put(&copy_pages, &page, b);
    }
    if (b->n == b->cap) {
        b->cap = b->cap ? b->cap * 2 : 8;
        b->v = realloc(b->v, b->cap * sizeof *b->v);
        if (!b->v) rt_fatal("nvn: command copy index");
    }
    b->v[b->n++] = (cmd_copy){lo, hi, copy};
    nvn_hm_put(&copy_live, &handle, pf);
    ncopies++;
    copy_bytes += pf->bytes;
    if (++copies_made % 20000 == 0)
        NVN_LOG("recording copies: %llu made, %llu live, %.1f MiB, submit %llu", (unsigned long long)copies_made,
                (unsigned long long)ncopies, copy_bytes / 1048576.0, (unsigned long long)nsubmits);
    plat_mutex_unlock(&copy_lock);
}

// Adding control memory again retires every recording copy made in it (the driver's handle lived there).
static void release_copies(uintptr_t lo, uintptr_t hi) {
    if (hi <= lo) return;
    uint64_t replayed = nvn_submit_count();
    plat_mutex_lock(&copy_lock);
    if (nzombies) zombie_sweep(replayed);
    for (uintptr_t pg = lo >> COPY_PAGE_SHIFT; pg <= (hi - 1) >> COPY_PAGE_SHIFT; pg++) {
        copy_bucket *b = nvn_hm_get(&copy_pages, &pg);
        if (!b) continue;
        for (uint32_t i = 0; i < b->n;) {
            cmd_copy *k = &b->v[i];
            if (!(k->lo < hi && lo < k->hi)) { i++; continue; }
            uint64_t handle = (uintptr_t)k->copy;
            copy_prefix *pf = (copy_prefix *)k->copy - 1;
            nvn_hm_del(&copy_live, &handle);
            ncopies--;
            copy_bytes -= pf->bytes;
            pf->magic = 0;
            if (pf->used_at > replayed) zombie_add(pf);   // a queued submission still reads it
            else free(pf);
            b->v[i] = b->v[--b->n];
        }
        if (!b->n) {
            nvn_hm_del(&copy_pages, &pg);
            free(b->v);
            free(b);
        }
    }
    plat_mutex_unlock(&copy_lock);
}

// ---------- CommandBuffer: object and memory
// The inline fast-path fields are kept at ~0 / 0 so NVN 55 inline code never finds room and calls the API (cmd_t).
NVN_API(nvnCommandBufferInitialize) {
    cmd_t *o = cmd_of(c);
    // Initialize over an object that was never finalized drops its recording buffer (seg lives in host memory). Logged
    // only: the old bytes may be a copy whose buffer another object owns, so freeing it here is not safe.
    static uint32_t reinits;
    if (o->magic == CMD_MAGIC && o->seg && __atomic_fetch_add(&reinits, 1, __ATOMIC_RELAXED) < 8)
        rt_log("nvn: Initialize over the command buffer %p, which was not finalized; its %zu-byte recording buffer %p "
               "is dropped", (void *)o, (size_t)(o->end - o->seg), (void *)o->seg);
    memset(o, 0, sizeof *o);
    o->magic = CMD_MAGIC;
    o->dev = arg_p(c, 1);
    o->inline_cur = ~0ull;
    o->inline_end = 0;
    cb_set_disabled(o, CB_NO_COMMAND | CB_NO_CONTROL, 0);
    ret_x(c, 1);
}
// Handles outlive their recorder.
NVN_API(nvnCommandBufferFinalize) {
    cmd_t *o = cmd_of(c);
    free(o->seg);
    o->seg = o->cur = o->end = NULL;
    o->magic = 0;
    cb_set_disabled(o, CB_NO_COMMAND | CB_NO_CONTROL, 0);
}
NVN_API(nvnCommandBufferSetMemoryCallback) { cmd_of(c)->cb = arg_p(c, 1); }
NVN_API(nvnCommandBufferSetMemoryCallbackData) { cmd_of(c)->cb_data = arg_p(c, 1); }
NVN_API(nvnCommandBufferGetMemoryCallback) { ret_x(c, (uintptr_t)cmd_of(c)->cb); }
NVN_API(nvnCommandBufferGetMemoryCallbackData) { ret_x(c, (uintptr_t)cmd_of(c)->cb_data); }
NVN_API(nvnCommandBufferSetCommandMemoryCallbackEnabled) { cb_set_disabled(cmd_of(c), CB_NO_COMMAND, !nvn_arg_bool(c, 1)); }
NVN_API(nvnCommandBufferSetControlMemoryCallbackEnabled) { cb_set_disabled(cmd_of(c), CB_NO_CONTROL, !nvn_arg_bool(c, 1)); }
// Accounting only; command memory can be shared by independent recordings with distinct control memory
NVN_API(nvnCommandBufferAddCommandMemory) {
    cmd_t *o = cmd_of(c);
    const pool_t *p = arg_p(c, 1);
    o->cmd_base = (uintptr_t)(p ? p->mem : NULL) + (uintptr_t)arg_x(c, 2);
    o->game_size = arg_x(c, 3);
    o->cmd_used = 0;
    o->hw_used = 0;
}
// The SDK encodes handles in control memory, so reusing that storage retires the copies recorded there
NVN_API(nvnCommandBufferAddControlMemory) {
    cmd_t *o = cmd_of(c);
    void *m = arg_p(c, 1);
    size_t n = arg_x(c, 2);
    release_copies((uintptr_t)m, (uintptr_t)m + n);
    o->ctl = m;
    o->ctl_size = n;
    o->ctl_used = 0;
}
// Usage is bounded by the memory the game gave, so the game never sees more used than it provided
static size_t min_size(size_t a, size_t b) { return a < b ? a : b; }
NVN_API(nvnCommandBufferGetControlMemorySize) { ret_x(c, cmd_of(c)->ctl_size); }
NVN_API(nvnCommandBufferGetControlMemoryUsed) { const cmd_t *o = cmd_of(c); ret_x(c, min_size(o->ctl_used, o->ctl_size)); }
NVN_API(nvnCommandBufferGetControlMemoryFree) { const cmd_t *o = cmd_of(c); ret_x(c, o->ctl_size - min_size(o->ctl_used, o->ctl_size)); }
NVN_API(nvnCommandBufferGetCommandMemorySize) { ret_x(c, cmd_of(c)->game_size); }
NVN_API(nvnCommandBufferGetCommandMemoryUsed) { const cmd_t *o = cmd_of(c); ret_x(c, min_size(o->hw_used, o->game_size)); }
NVN_API(nvnCommandBufferGetCommandMemoryFree) { const cmd_t *o = cmd_of(c); ret_x(c, o->game_size - min_size(o->hw_used, o->game_size)); }

// ---------- CommandBuffer: recording
NVN_API(nvnCommandBufferBeginRecording) {
    cmd_t *o = cmd_of(c);
    if (o->recording) rt_log("nvn: BeginRecording on command buffer %p while it records", (void *)o);
    o->recording = 1;
    o->cur = o->seg;   // the previous recording was copied at EndRecording
    o->rec_start = o->cmd_used;
    put_hdr(cmd_alloc(o, 8), OP_BEGIN, 8);
    o->hw_used -= 8;   // BEGIN and END have no hardware equivalent
    o->hw_start = o->hw_used;
    if (!o->game_size) memory_event(c, o, NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY, MIN_COMMAND_SIZE);
}
NVN_API(nvnCommandBufferEndRecording) {
    cmd_t *o = cmd_of(c);
    if (!o->recording || !o->seg) rt_fatal("nvn: EndRecording on command buffer %p, which does not record", (void *)o);
    if (!o->ctl_size) memory_event(c, o, NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_CONTROL_MEMORY, MIN_CONTROL_SIZE);
    put_hdr(cmd_alloc(o, 8), OP_END, 8);
    o->hw_used -= 8;
    o->recording = 0;
    uint8_t *copy = flatten_recording(o->seg, (size_t)(o->cur - o->seg));
    // Finalizing the recorder does not retire its handle; only reusing the control memory it was recorded in does.
    uintptr_t lo = (uintptr_t)o->ctl + (o->ctl_used < o->ctl_size ? o->ctl_used : 0);
    uintptr_t hi = o->ctl ? lo + 1 : 0;
    o->ctl_used += CTL_PER_RECORDING;
    copy_add(lo, hi, copy);
    ret_x(c, (uintptr_t)copy);
}
NVN_API(nvnCommandBufferIsRecording) { ret_x(c, cmd_of(c)->recording != 0); }

// The called recording is copied into the caller at the call position, as a handle may be gone by the time the caller is
// submitted. Deferred binds stay deferred. The records go over as one block (flatten_recording checked them).
static void inline_recording(cmd_t *o, uint64_t handle, const char *what) {
    if (!handle) rt_fatal("nvn: %s with a null handle", what);
    uint64_t bytes = copy_touch(handle, __atomic_load_n(&nsubmits, __ATOMIC_RELAXED), what);
    if (bytes <= 16) return;
    size_t body = (size_t)bytes - 16;
    cmd_reserve(o, body);
    memcpy(o->cur, (const uint8_t *)(uintptr_t)handle + 8, body);
    o->cur += body;
    o->cmd_used += body;
    o->hw_used += body;
    o->last_rec = 0;
}
NVN_API(nvnCommandBufferCallCommands) {
    cmd_t *o = cmd_of(c);
    int n = (int)arg_x(c, 1);
    const uint64_t *h = arg_p(c, 2);
    for (int i = 0; i < n; i++) inline_recording(o, h[i], "CallCommands");
}
NVN_API(nvnCommandBufferCopyCommands) {
    cmd_t *o = cmd_of(c);
    int n = (int)arg_x(c, 1);
    const uint64_t *h = arg_p(c, 2);
    for (int i = 0; i < n; i++) inline_recording(o, h[i], "CopyCommands");
}

// ---------- CommandBuffer: commands with CPU-visible results (performed by the back end in GPU order)
NVN_API(nvnCommandBufferSignalEvent) {
    const ev_t *e = arg_p(c, 1);
    *REC(cmd_of(c), OP_SIGNAL_EVENT, p_signal_event) = (p_signal_event){e->val, (uint32_t)arg_x(c, 2), (uint32_t)arg_x(c, 5)};
}
NVN_API(nvnCommandBufferResetCounter) { REC(cmd_of(c), OP_RESET_COUNTER, p_reset_counter)->type = (uint32_t)arg_x(c, 1); }
NVN_API(nvnCommandBufferReportCounter) {
    *REC(cmd_of(c), OP_REPORT_COUNTER, p_report_counter) = (p_report_counter){(uint32_t)arg_x(c, 1), arg_x(c, 2)};
}
NVN_API(nvnCommandBufferCopyBufferToBuffer) {
    *REC(cmd_of(c), OP_COPY_B2B, p_copy_b2b) = (p_copy_b2b){arg_x(c, 1), arg_x(c, 2), arg_x(c, 3)};
}
// The data is copied at record time (the caller may change it afterwards) and written into the buffer at replay
NVN_API(nvnCommandBufferUpdateUniformBuffer) {
    uint64_t dst = arg_x(c, 1) + arg_x(c, 3), n = arg_x(c, 4);
    const void *data = arg_p(c, 5);
    if (n > (16u << 20)) rt_fatal("nvn: UpdateUniformBuffer of %#llx bytes", (unsigned long long)n);
    p_update_ub *r = rec(cmd_of(c), OP_UPDATE_UB, offsetof(p_update_ub, data) + n);
    r->dst = dst;
    r->size = n;
    memcpy(r->data, data, n);
}

// ---------- CommandBuffer: render state, payloads of nvn_obj.h
NVN_API(nvnCommandBufferSetRenderTargets) {
    int n = (int)arg_x(c, 1);
    const NVNtexture *const *ct = arg_p(c, 2);
    const NVNtextureView *const *cv = arg_p(c, 3);
    const NVNtexture *dt = arg_p(c, 4);
    const NVNtextureView *dv = arg_p(c, 5);
    if (n < 0 || n > 8) {
        rt_log("nvn: SetRenderTargets with %d colour targets", n);
        n = n < 0 ? 0 : 8;
    }
    p_set_rt *r = REC(cmd_of(c), OP_SET_RT, p_set_rt);
    r->n = n;
    for (int i = 0; i < n; i++) {
        r->color[i] = ct ? tex_pin(ct[i]) : NULL;
        copy_view(&r->cview[i], &r->has_cview[i], cv ? cv[i] : NULL);
    }
    r->depth = tex_pin(dt);
    copy_view(&r->dview, &r->has_dview, dv);
}
NVN_API(nvnCommandBufferClearColor) {
    p_clear_color *r = REC(cmd_of(c), OP_CLEAR_COLOR, p_clear_color);
    r->index = (int)arg_x(c, 1);
    r->mask = (int)arg_x(c, 3);
    memcpy(r->rgba, arg_p(c, 2), 16);
}
NVN_API(nvnCommandBufferClearDepthStencil) {
    *REC(cmd_of(c), OP_CLEAR_DS, p_clear_ds) = (p_clear_ds){arg_s(c, 0), nvn_arg_bool(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3)};
}
static void set_viewports(cmd_t *o, int first, int n, const float *v) {
    if (first < 0 || n < 0 || first + n > 16) {
        rt_log("nvn: SetViewports first %d n %d", first, n);
        return;
    }
    p_viewports *r = REC(o, OP_VIEWPORTS, p_viewports);
    r->first = first;
    r->n = n;
    for (int i = 0; i < n; i++) memcpy(r->v[i], v + 4 * i, 16);
}
NVN_API(nvnCommandBufferSetViewports) { set_viewports(cmd_of(c), (int)arg_x(c, 1), (int)arg_x(c, 2), arg_p(c, 3)); }
NVN_API(nvnCommandBufferSetViewport) {
    float v[4] = {(float)(int)arg_x(c, 1), (float)(int)arg_x(c, 2), (float)(int)arg_x(c, 3), (float)(int)arg_x(c, 4)};
    set_viewports(cmd_of(c), 0, 1, v);
}
static void set_scissor(cmd_t *o, int x, int y, int w, int h) { *REC(o, OP_SCISSOR, p_scissor) = (p_scissor){x, y, w, h}; }
NVN_API(nvnCommandBufferSetScissor) {
    set_scissor(cmd_of(c), (int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3), (int)arg_x(c, 4));
}
// The replay keeps one scissor (index 0); other indices matter only to shaders that select a viewport index, so
// differing ones are reported
NVN_API(nvnCommandBufferSetScissors) {
    int first = (int)arg_x(c, 1), n = (int)arg_x(c, 2);
    const int *r = arg_p(c, 3);
    if (first < 0 || n < 0 || first + n > 16) rt_fatal("nvn: SetScissors first %d n %d", first, n);
    for (int i = 1; i < n; i++)
        if (memcmp(r + 4 * i, r, 16)) {
            static int once;
            if (!once++) rt_log("nvn: SetScissors: differing scissor %d ignored (single-scissor replay)", first + i);
        }
    if (first == 0 && n > 0) set_scissor(cmd_of(c), r[0], r[1], r[2], r[3]);
}
static void set_depth_range(cmd_t *o, float n, float f) { *REC(o, OP_DEPTH_RANGE, p_depth_range) = (p_depth_range){n, f}; }
NVN_API(nvnCommandBufferSetDepthRange) { set_depth_range(cmd_of(c), arg_s(c, 0), arg_s(c, 1)); }
NVN_API(nvnCommandBufferSetDepthRanges) {
    int first = (int)arg_x(c, 1), n = (int)arg_x(c, 2);
    const float *r = arg_p(c, 3);
    if (first < 0 || n < 0 || first + n > 16) rt_fatal("nvn: SetDepthRanges first %d n %d", first, n);
    for (int i = 1; i < n; i++)
        if (memcmp(r + 2 * i, r, 8)) {
            static int once;
            if (!once++) rt_log("nvn: SetDepthRanges: differing range %d ignored (single-range replay)", first + i);
        }
    if (first == 0 && n > 0) set_depth_range(cmd_of(c), r[0], r[1]);
}
NVN_API(nvnCommandBufferSetDepthClamp) { REC(cmd_of(c), OP_DEPTH_CLAMP, p_flag)->value = nvn_arg_bool(c, 1) != 0; }
NVN_API(nvnCommandBufferSetRenderEnable) { REC(cmd_of(c), OP_RENDER_ENABLE, p_flag)->value = nvn_arg_bool(c, 1) != 0; }
// Rasterizer discard runs vertex work without rasterising; with no transform feedback that equals not drawing.
NVN_API(nvnCommandBufferSetRasterizerDiscard) { REC(cmd_of(c), OP_RASTER_DISCARD, p_flag)->value = nvn_arg_bool(c, 1) != 0; }
// Strips with 16-bit indices restart at 0xffff; the replay applies it per indexed draw (op_draw_idx)
NVN_API(nvnCommandBufferSetPrimitiveRestart) {
    *REC(cmd_of(c), OP_PRIM_RESTART, p_prim_restart) = (p_prim_restart){nvn_arg_bool(c, 1) != 0, (uint32_t)arg_x(c, 2)};
}
// Rendering is enabled when the two NVNcounterData.counter values at addr / addr+16 compare equal (mode 0) or not equal
// (mode 1). The game gates ClearColor commands with it; the replay evaluates it in GPU order.
NVN_API(nvnCommandBufferSetRenderEnableConditional) {
    int m = (int)arg_x(c, 1);
    if (m != 0 && m != 1) rt_fatal("nvn: SetRenderEnableConditional: unknown mode %d", m);
    p_render_cond *r = REC(cmd_of(c), OP_RENDER_COND, p_render_cond);
    r->mode = m;
    r->addr = arg_x(c, 2);
}
NVN_API(nvnCommandBufferBindProgram) {
    *REC(cmd_of(c), OP_BIND_PROGRAM, p_bind_program) = (p_bind_program){arg_p(c, 1), (int)arg_x(c, 2)};
}
// Mesh program identity belongs to replay state: a draw may inherit a binding from another command buffer or thread.
NVN_API(nvnCommandBufferDrawMeshTasks) {
    *REC(cmd_of(c), OP_DRAW_MESH, p_draw_mesh) = (p_draw_mesh){(uint32_t)arg_x(c, 1), (uint32_t)arg_x(c, 2), c->x[30]};
}
NVN_API(nvnCommandBufferDrawMeshTasksIndirect) {
    *REC(cmd_of(c), OP_DRAW_MESH_IND, p_draw_mesh_ind) = (p_draw_mesh_ind){arg_x(c, 1), c->x[30]};
}
NVN_API(nvnCommandBufferSetGeometryProcessingMode) {
    *REC(cmd_of(c), OP_GEOMETRY_MODE, p_flag) = (p_flag){(int)arg_x(c, 1)};
}
// The state object's bytes are copied (p_bind_state.kind order)
static void bind_state(const Cpu *c, int kind, size_t n) {
    p_bind_state *r = REC(cmd_of(c), OP_BIND_STATE, p_bind_state);
    r->kind = kind;
    memcpy(r->bytes, arg_p(c, 1), n);
}
NVN_API(nvnCommandBufferBindBlendState) { bind_state(c, 0, sizeof(blend_st)); }
NVN_API(nvnCommandBufferBindColorState) { bind_state(c, 1, sizeof(color_st)); }
NVN_API(nvnCommandBufferBindChannelMaskState) { bind_state(c, 2, sizeof(chmask_st)); }
NVN_API(nvnCommandBufferBindMultisampleState) { bind_state(c, 3, sizeof(ms_st)); }
NVN_API(nvnCommandBufferBindPolygonState) { bind_state(c, 4, sizeof(poly_st)); }
NVN_API(nvnCommandBufferBindDepthStencilState) { bind_state(c, 5, sizeof(ds_st)); }
// Recorded as bind-state kind 6 with the float as payload
NVN_API(nvnCommandBufferSetPointSize) {
    p_bind_state *r = REC(cmd_of(c), OP_BIND_STATE, p_bind_state);
    float s = arg_s(c, 0);
    r->kind = 6;
    memcpy(r->bytes, &s, sizeof s);
}
// Recorded as bind-state kind 7 (spirv_alpha_test)
NVN_API(nvnCommandBufferSetAlphaRef) {
    p_bind_state *r = REC(cmd_of(c), OP_BIND_STATE, p_bind_state);
    float s = arg_s(c, 0);
    r->kind = 7;
    memcpy(r->bytes, &s, sizeof s);
}
NVN_API(nvnCommandBufferBindVertexAttribState) {
    cmd_t *o = cmd_of(c);
    int n = (int)arg_x(c, 1);
    if (n < 0 || n > 16) { rt_log("nvn: BindVertexAttribState n %d", n); return; }
    p_bind_vattr *r = RECN(o, OP_BIND_VATTR, p_bind_vattr, st, n);
    r->first = 0;
    r->n = n;
    memcpy(r->st, arg_p(c, 2), 4 * (size_t)n);
    hw_fix(o, 68);   // the driver always writes all 16 attribute formats: 17 words
}
NVN_API(nvnCommandBufferBindVertexStreamState) {
    cmd_t *o = cmd_of(c);
    int n = (int)arg_x(c, 1);
    if (n < 0 || n > 16) { rt_log("nvn: BindVertexStreamState n %d", n); return; }
    p_bind_vstream *r = RECN(o, OP_BIND_VSTREAM, p_bind_vstream, st, n);
    r->first = 0;
    r->n = n;
    memcpy(r->st, arg_p(c, 2), 8 * (size_t)n);
    hw_fix(o, 4 * (16 + (size_t)n));
}
static void bind_vertex_buffers(cmd_t *o, int first, int n, const NVNbufferRange *rg) {
    if (first < 0 || n < 0 || first + n > 16) { rt_log("nvn: BindVertexBuffers first %d n %d", first, n); return; }
    p_bind_vbuf *r = RECN(o, OP_BIND_VBUF, p_bind_vbuf, r, n);
    r->first = first;
    r->n = n;
    memcpy(r->r, rg, sizeof *rg * (size_t)n);
    hw_fix(o, 24 * (size_t)n);
}
NVN_API(nvnCommandBufferBindVertexBuffers) { bind_vertex_buffers(cmd_of(c), (int)arg_x(c, 1), (int)arg_x(c, 2), arg_p(c, 3)); }
NVN_API(nvnCommandBufferBindVertexBuffer) {
    NVNbufferRange r = {arg_x(c, 2), arg_x(c, 3)};
    bind_vertex_buffers(cmd_of(c), (int)arg_x(c, 1), 1, &r);
}
// Uniform and storage buffers: one record per binding (CB_SIZE/ADDRESS 4 words + CB_BIND 1 on hardware).
static void bind_buffers(const Cpu *c, uint32_t op) {   // (cmd, stage, first, n, const NVNbufferRange *ranges)
    cmd_t *o = cmd_of(c);
    int st = (int)arg_x(c, 1), first = (int)arg_x(c, 2), n = (int)arg_x(c, 3);
    const NVNbufferRange *rg = arg_p(c, 4);
    for (int i = 0; i < n; i++) {
        *REC(o, op, p_bind_buf) = (p_bind_buf){st, first + i, rg[i].address, rg[i].size};
        hw_fix(o, 20);
    }
}
static void bind_buffer(const Cpu *c, uint32_t op) {    // (cmd, stage, index, NVNbufferAddress address, size_t size)
    cmd_t *o = cmd_of(c);
    *REC(o, op, p_bind_buf) = (p_bind_buf){(int)arg_x(c, 1), (int)arg_x(c, 2), arg_x(c, 3), arg_x(c, 4)};
    hw_fix(o, 20);
}
NVN_API(nvnCommandBufferBindUniformBuffer) { bind_buffer(c, OP_BIND_UBO); }
NVN_API(nvnCommandBufferBindUniformBuffers) { bind_buffers(c, OP_BIND_UBO); }
NVN_API(nvnCommandBufferBindStorageBuffer) { bind_buffer(c, OP_BIND_SSBO); }
NVN_API(nvnCommandBufferBindStorageBuffers) { bind_buffers(c, OP_BIND_SSBO); }
// NVN 2. The driver clamps bytes to 0x100, drops calls under 4 bytes and writes words: graphics through SET_ROOT_TABLE_SELECTOR
// (method 0x504, table & 7 | offset << 8) plus LOAD_ROOT_TABLE (0x508), compute through MME macro 0x53. Graphics shaders read
// table 0 as constant bank 24 (the full-screen blits get 1/w, 1/h, w, h this way). The Deferred entry points store the direct
// data pointer and the low byte of offset without reading data; the submit interpreter reloads that pointer.
static void push_constants(const Cpu *c, int compute, int deferred) {
    cmd_t *o = cmd_of(c);
    int table = (int)(arg_x(c, 1) & 7), offset = deferred ? (uint8_t)arg_x(c, 2) : (int)arg_x(c, 2);
    uint32_t bytes = (uint32_t)arg_x(c, 3);
    if (bytes > 0x100) bytes = 0x100;
    if (bytes < 4) return;
    if (offset < 0 || offset + bytes > 0x100 || (offset & 3)) {
        rt_log("nvn: Push%sConstants offset %#x bytes %#x past the 256-byte root table", compute ? "Compute" : "Graphics",
               offset, bytes);
        return;
    }
    uint32_t n = bytes / 4;
    p_push_const *r = RECN(o, OP_PUSH_CONST, p_push_const, w, n);   // holds only n words of w: set fields one by one
    r->compute = compute, r->table = table, r->offset = offset, r->bytes = (int32_t)(n * 4);
    r->src = deferred ? arg_x(c, 4) : 0;
    if (!deferred) memcpy(r->w, arg_p(c, 4), n * 4);
    hw_fix(o, 4 * ((size_t)n + 2));   // method header, selector or macro parameter, n words
}
NVN_API(nvnCommandBufferPushGraphicsConstants) { push_constants(c, 0, 0); }
NVN_API(nvnCommandBufferPushComputeConstants) { push_constants(c, 1, 0); }
NVN_API(nvnCommandBufferPushGraphicsConstantsDeferred) { push_constants(c, 0, 1); }
NVN_API(nvnCommandBufferPushComputeConstantsDeferred) { push_constants(c, 1, 1); }
// Texture, image and separate handle binds. A deferred bind (NVN 55 *Deferred entry points) keeps the pointer to the game's
// handle array: the game fills its material handle tables after recording and before the submit (nvn_submit_prepare).
// hw_bytes: the hardware size of the plain variant.
static void bind_handles(cmd_t *o, uint32_t op, int st, int first, int n, const uint64_t *h, int deferred, size_t hw_bytes) {
    if (n < 0 || n > 32) { rt_log("nvn: bind of %d handles", n); return; }
    p_bind_handles *r = RECN(o, op, p_bind_handles, h, n);
    r->stage = st;
    r->first = first;
    r->n = n;
    r->src = deferred ? (uint64_t)(uintptr_t)h : 0;
    memcpy(r->h, h, 8 * (size_t)n);
    hw_fix(o, hw_bytes ? hw_bytes : 16 + 4 * (size_t)n);   // macro 7 + method 0x1a2c + CB_POS + one word per binding
}
static void bind_handle_array(const Cpu *c, uint32_t op, int deferred, int wide) {   // (cmd, stage, first, n, handles)
    int n = (int)arg_x(c, 3);
    bind_handles(cmd_of(c), op, (int)arg_x(c, 1), (int)arg_x(c, 2), n, arg_p(c, 4), deferred,
                 wide && n >= 0 && n <= 32 ? 16 + 8 * (size_t)n : 0);
}
static void bind_one_handle(const Cpu *c, uint32_t op, size_t hw_bytes) {   // (cmd, stage, index, handle)
    uint64_t v = arg_x(c, 3);
    bind_handles(cmd_of(c), op, (int)arg_x(c, 1), (int)arg_x(c, 2), 1, &v, 0, hw_bytes);
}
// Combined texture handles share the texture slots with separate textures (sampler id in bits 20 and up). Image handle =
// texture pool index; images and textures share their registration.
NVN_API(nvnCommandBufferBindTexture) { bind_one_handle(c, OP_BIND_TEX, 0); }
NVN_API(nvnCommandBufferBindTextures) { bind_handle_array(c, OP_BIND_TEX, 0, 0); }
NVN_API(nvnCommandBufferBindImage) { bind_one_handle(c, OP_BIND_IMG, 24); }
NVN_API(nvnCommandBufferBindImages) { bind_handle_array(c, OP_BIND_IMG, 0, 1); }
NVN_API(nvnCommandBufferBindSeparateTexture) { bind_one_handle(c, OP_BIND_TEX, 24); }
NVN_API(nvnCommandBufferBindSeparateTextures) { bind_handle_array(c, OP_BIND_TEX, 0, 1); }
NVN_API(nvnCommandBufferBindSeparateSampler) { bind_one_handle(c, OP_BIND_SAMP, 24); }
NVN_API(nvnCommandBufferBindSeparateSamplers) { bind_handle_array(c, OP_BIND_SAMP, 0, 1); }
NVN_API(nvnCommandBufferBindTexturesDeferred) { bind_handle_array(c, OP_BIND_TEX, 1, 0); }
NVN_API(nvnCommandBufferBindImagesDeferred) { bind_handle_array(c, OP_BIND_IMG, 1, 1); }
NVN_API(nvnCommandBufferBindSeparateTexturesDeferred) { bind_handle_array(c, OP_BIND_TEX, 1, 1); }
NVN_API(nvnCommandBufferBindSeparateSamplersDeferred) { bind_handle_array(c, OP_BIND_SAMP, 1, 1); }
// Buffer ranges still bind at record time
NVN_API(nvnCommandBufferBindUniformBuffersDeferred) {
    static int once;
    if (!once++) NVN_LOG("BindUniformBuffersDeferred: buffer ranges bind at record time");
    bind_buffers(c, OP_BIND_UBO);
}
NVN_API(nvnCommandBufferBindStorageBuffersDeferred) {
    static int once;
    if (!once++) NVN_LOG("BindStorageBuffersDeferred: buffer ranges bind at record time");
    bind_buffers(c, OP_BIND_SSBO);
}
NVN_API(nvnCommandBufferSetStencilValueMask) {
    *REC(cmd_of(c), OP_STENCIL, p_stencil) = (p_stencil){(int)arg_x(c, 1), (int)arg_x(c, 2), 0, 0, 0};
}
NVN_API(nvnCommandBufferSetStencilMask) {
    *REC(cmd_of(c), OP_STENCIL, p_stencil) = (p_stencil){(int)arg_x(c, 1), 0, (int)arg_x(c, 2), 0, 1};
}
NVN_API(nvnCommandBufferSetStencilRef) {
    *REC(cmd_of(c), OP_STENCIL, p_stencil) = (p_stencil){(int)arg_x(c, 1), 0, 0, (int)arg_x(c, 2), 2};
}
// A hint for the hardware's stencil culling; the stencil test itself still decides, so the Vulkan replay needs nothing
NVN_API(nvnCommandBufferSetStencilCullCriteria) { (void)c; }
NVN_API(nvnCommandBufferSetPolygonOffsetClamp) {
    *REC(cmd_of(c), OP_POLY_OFFSET, p_poly_offset) = (p_poly_offset){arg_s(c, 0), arg_s(c, 1), arg_s(c, 2)};
}
NVN_API(nvnCommandBufferDrawArrays) {
    *REC(cmd_of(c), OP_DRAW, p_draw) = (p_draw){(int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3), 0, 1};
}
// Zero instances draw nothing
NVN_API(nvnCommandBufferDrawArraysInstanced) {
    int inst = (int)arg_x(c, 5);
    if (inst <= 0) return;
    *REC(cmd_of(c), OP_DRAW, p_draw) = (p_draw){(int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3), (int)arg_x(c, 4), inst};
}
NVN_API(nvnCommandBufferDrawElements) {
    cmd_t *o = cmd_of(c);
    *REC(o, OP_DRAW_IDX, p_draw_idx) = (p_draw_idx){(int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3), 0, 0, 1, arg_x(c, 4)};
    hw_fix(o, 48);
}
NVN_API(nvnCommandBufferDrawElementsBaseVertex) {
    cmd_t *o = cmd_of(c);
    *REC(o, OP_DRAW_IDX, p_draw_idx) =
        (p_draw_idx){(int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3), (int)arg_x(c, 5), 0, 1, arg_x(c, 4)};
    hw_fix(o, 44);
}
// Assumed to use the 6-parameter draw macro of the base-vertex form (unverified)
NVN_API(nvnCommandBufferDrawElementsInstanced) {
    cmd_t *o = cmd_of(c);
    *REC(o, OP_DRAW_IDX, p_draw_idx) = (p_draw_idx){(int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3),
                                                    (int)arg_x(c, 5), (int)arg_x(c, 6), (int)arg_x(c, 7), arg_x(c, 4)};
    hw_fix(o, 44);
}
// The arguments at indirect are often GPU-written, so the replay draws with vkCmdDrawIndexedIndirect from the pool buffer.
NVN_API(nvnCommandBufferDrawElementsIndirect) {
    *REC(cmd_of(c), OP_DRAW_IDX_IND, p_draw_idx_ind) = (p_draw_idx_ind){(int)arg_x(c, 1), (int)arg_x(c, 2), arg_x(c, 3), arg_x(c, 4)};
}
// caller = the game's return address
NVN_API(nvnCommandBufferDispatchCompute) {
    *REC(cmd_of(c), OP_DISPATCH, p_dispatch) = (p_dispatch){(int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3), c->x[30]};
}
// vkCmdDispatchIndirect
NVN_API(nvnCommandBufferDispatchComputeIndirect) {
    *REC(cmd_of(c), OP_DISPATCH_IND, p_dispatch_ind) = (p_dispatch_ind){arg_x(c, 1), c->x[30]};
}
NVN_API(nvnCommandBufferClearBuffer) {
    *REC(cmd_of(c), OP_CLEAR_BUF, p_clear_buf) = (p_clear_buf){arg_x(c, 1), arg_x(c, 2), (uint32_t)arg_x(c, 3), 0};
}
// The value lands when the GPU reaches this point: recorded as a 4-byte ClearBuffer so it stays in GPU order with the counter
// resets and works in CPU-invisible pools.
NVN_API(nvnCommandBufferReportValue) {
    *REC(cmd_of(c), OP_CLEAR_BUF, p_clear_buf) = (p_clear_buf){arg_x(c, 2), 4, (uint32_t)arg_x(c, 1), 0};
}
// Colour reduction is a Maxwell bandwidth hint for blending into compressed targets: no effect on the result.
NVN_API(nvnCommandBufferSetColorReductionEnable) { (void)c; }
NVN_API(nvnCommandBufferSetColorReductionThresholds) { (void)c; }
NVN_API(nvnCommandBufferBarrier) { REC(cmd_of(c), OP_BARRIER, p_barrier)->bits = (int)arg_x(c, 1); }
NVN_API(nvnCommandBufferSetTexturePool) {
    dpool_ref pool;
    if (!nvn_dpool_snapshot(arg_p(c, 1), 1u << 20, &pool)) rt_fatal("nvn: invalid texture descriptor pool");
    REC(cmd_of(c), OP_SET_TEX_POOL, p_set_pool)->pool = pool;
}
NVN_API(nvnCommandBufferSetSamplerPool) {
    dpool_ref pool;
    if (!nvn_dpool_snapshot(arg_p(c, 1), 1u << 12, &pool)) rt_fatal("nvn: invalid sampler descriptor pool");
    REC(cmd_of(c), OP_SET_SAMP_POOL, p_set_pool)->pool = pool;
}
NVN_API(nvnCommandBufferCopyBufferToTexture) {
    p_copy_b2t *o = REC(cmd_of(c), OP_COPY_B2T, p_copy_b2t);
    o->src = arg_x(c, 1);
    o->tex = tex_pin(arg_p(c, 2));
    copy_view(&o->view, &o->has_view, arg_p(c, 3));
    o->r = *(const NVNcopyRegion *)arg_p(c, 4);
    o->flags = (int)arg_x(c, 5);
}
// p_copy_b2t with src = the destination address
NVN_API(nvnCommandBufferCopyTextureToBuffer) {
    p_copy_b2t *o = REC(cmd_of(c), OP_COPY_T2B, p_copy_b2t);
    o->src = arg_x(c, 4);
    o->tex = tex_pin(arg_p(c, 1));
    copy_view(&o->view, &o->has_view, arg_p(c, 2));
    o->r = *(const NVNcopyRegion *)arg_p(c, 3);
    o->flags = (int)arg_x(c, 5);
}
static void clear_texture(const Cpu *c, int is_uint) {
    p_clear_tex *o = REC(cmd_of(c), OP_CLEAR_TEX, p_clear_tex);
    o->tex = tex_pin(arg_p(c, 1));
    copy_view(&o->view, &o->has_view, arg_p(c, 2));
    o->r = *(const NVNcopyRegion *)arg_p(c, 3);
    o->mask = (int)arg_x(c, 5);
    o->is_uint = (uint8_t)is_uint;
    memcpy(o->c.u, arg_p(c, 4), 16);
}
NVN_API(nvnCommandBufferClearTexture) { clear_texture(c, 0); }
NVN_API(nvnCommandBufferClearTextureui) { clear_texture(c, 1); }
NVN_API(nvnCommandBufferCopyTextureToTexture) {
    p_copy_t2t *o = REC(cmd_of(c), OP_COPY_T2T, p_copy_t2t);
    o->src = tex_pin(arg_p(c, 1));
    o->dst = tex_pin(arg_p(c, 4));
    copy_view(&o->sv, &o->has_sv, arg_p(c, 2));
    copy_view(&o->dv, &o->has_dv, arg_p(c, 5));
    o->sr = *(const NVNcopyRegion *)arg_p(c, 3);
    o->dr = *(const NVNcopyRegion *)arg_p(c, 6);
    o->flags = (int)arg_x(c, 7);
}
// Level 0 of src into level 0 of dst, whole extents, filtered. Recorded as a linear-filter texture copy: copy_t2t
// resolves a multisample source into an equal-size target and blits (2x2 average at half size) between sizes.
NVN_API(nvnCommandBufferDownsample) {
    p_copy_t2t *o = REC(cmd_of(c), OP_COPY_T2T, p_copy_t2t);
    o->src = tex_pin(arg_p(c, 1));
    o->dst = tex_pin(arg_p(c, 2));
    if (o->src) o->sr = (NVNcopyRegion){0, 0, 0, (int)o->src->d.width, (int)o->src->d.height, 1};
    if (o->dst) o->dr = (NVNcopyRegion){0, 0, 0, (int)o->dst->d.width, (int)o->dst->d.height, 1};
    o->flags = NVN_COPY_FLAGS_LINEAR_FILTER_BIT;
}
NVN_API(nvnCommandBufferSetCopyRowStride) { REC(cmd_of(c), OP_COPY_STRIDE, p_copy_stride)->stride = (int64_t)arg_x(c, 1); }
// Bytes between the images (depth slices or array layers) of a buffer <-> texture copy, like the row stride for rows
NVN_API(nvnCommandBufferSetCopyImageStride) { REC(cmd_of(c), OP_COPY_IMAGE_STRIDE, p_copy_stride)->stride = (int64_t)arg_x(c, 1); }
NVN_API(nvnCommandBufferSetBlendColor) { memcpy(REC(cmd_of(c), OP_BLEND_COLOR, p_blend_color)->rgba, arg_p(c, 1), 16); }

// The SDK retains the pointer at record time and reads the sync's current producer at submit; preparation fulfils the wait.
NVN_API(nvnCommandBufferWaitSync) {
    cmd_t *o = cmd_of(c);
    REC(o, OP_WAIT_SYNC, p_wait_sync)->sync = arg_p(c, 1);
    hw_fix(o, 0);   // the SDK records a CPU control entry here; GPU wait commands are emitted at submit
}
// Commands without an effect on this replay: shader scratch memory and the tessellation patch size have no host counterpart.
volatile int64_t nvn_wait_event_calls;  // counted for the pend_wait report (vk_writeback.c pw_report)
// WaitEvent is recorded (OP_WAIT_EVENT) and the replay honours it: each NVN queue is a coroutine that stops at a wait that is
// not met (vk_exec.c "queue coroutines").
NVN_API(nvnCommandBufferWaitEvent) {
    __atomic_add_fetch(&nvn_wait_event_calls, 1, __ATOMIC_RELAXED);
    const ev_t *e = arg_p(c, 1);
    if (!e || e->magic != MAGIC('E', 'V', 'N', 'T')) return;
    *REC(cmd_of(c), OP_WAIT_EVENT, p_wait_event) = (p_wait_event){e->val, (uint32_t)arg_x(c, 2), (uint32_t)arg_x(c, 3)};
}
NVN_API(nvnCommandBufferSetShaderScratchMemory) { (void)c; }
NVN_API(nvnCommandBufferSetPatchSize) { (void)c; }

// ---------- submit
// What the console reads at the submit call is read here, on the game thread: every handle must be a live recording, and
// deferred binds and pushes take the data the game's arrays hold now. The result is written into the recording copy.
static uint64_t def_reads, def_changed;

// Only QueueFenceSync and window acquire produce syncs: both name work outside this submission that is already queued.
// Recorded FenceSync and imported/external producers are unimplemented. Waiting for their maximum before the whole batch is
// conservative: it orders CPU completion callbacks and GPU work before any deferred read or backend snapshot.
static uint64_t wait_target(const uint8_t *p, int depth) {
    uint64_t target = 0;
    for (;;) {
        op_hdr h;
        memcpy(&h, p, sizeof h);
        if (h.size < 8 || (h.size & 7) || h.op >= OP_COUNT_)
            rt_fatal("nvn: corrupt recording at %p while preparing WaitSync (op %u size %u)", (const void *)p, h.op, h.size);
        if (h.op == OP_END) return target;
        if (h.op == OP_JUMP) {
            if (h.size < 16) rt_fatal("nvn: short WaitSync jump record at %p", (const void *)p);
            memcpy(&p, p + 8, 8);
            continue;
        }
        uint64_t value = 0;
        if (h.op == OP_CALL) {
            if (h.size < 16) rt_fatal("nvn: short WaitSync call record at %p", (const void *)p);
            uint64_t called;
            memcpy(&called, p + 8, 8);
            if (depth >= 16) rt_fatal("nvn: WaitSync preparation: call depth over 16");
            value = wait_target((const uint8_t *)(uintptr_t)called, depth + 1);
        } else if (h.op == OP_WAIT_SYNC) {
            if (h.size < 8 + sizeof(p_wait_sync)) rt_fatal("nvn: short WaitSync record at %p", (const void *)p);
            const sync_t *source = ((const p_wait_sync *)(const void *)(p + 8))->sync;
            if (!source) rt_fatal("nvn: CommandBufferWaitSync with a null sync");
            sync_t snapshot;
            memcpy(&snapshot, source, sizeof snapshot);   // one value-at-submit read; never reread while waiting
            if (snapshot.magic != MAGIC('S', 'Y', 'N', 'C') ||
                (snapshot.fenced != 0 && snapshot.fenced != 1 && snapshot.fenced != 3))
                rt_fatal("nvn: CommandBufferWaitSync with an invalid sync state");
            // an unfenced SDK sync emits no GPU wait; an acquire sync (3) holds nothing back (fence 0; see
            // nvn_api_core.c acq_wait)
            if (snapshot.fenced == 1) value = snapshot.fence;
            else if (snapshot.fenced == 3 && snapshot.fence) {
                extern void nvn_acq_note(int path, int pending);
                nvn_acq_note(2, gfx_backend()->fence_done() < snapshot.fence);
            }
        }
        if (value > target) target = value;
        p += h.size;
    }
}

static void wait_submission(const uint64_t *handles, int n) {
    uint64_t target = 0;
    for (int i = 0; i < n; i++) {
        uint64_t value = wait_target((const uint8_t *)(uintptr_t)handles[i], 0);
        if (value > target) target = value;
    }
    if (!target) return;
    const GfxBackend *b = gfx_backend();
    // No copy/backend lock is held here (QueueSubmitCommands gave its virtual core up). fence_done/fence_wait include
    // completion callbacks, unlike the raw Vulkan timeline.
    if (b->fence_done() < target && !b->fence_wait(target, UINT64_MAX))
        rt_fatal("nvn: CommandBufferWaitSync could not reach producer fence %llu", (unsigned long long)target);
}

static int resolve_deferred(const uint8_t *p, int depth) {   // returns the number of deferred binds and pushes
    int nd = 0;
    for (;;) {
        op_hdr h;
        memcpy(&h, p, sizeof h);
        if (h.size < 8 || (h.size & 7) || h.op >= OP_COUNT_)
            rt_fatal("nvn: corrupt recording at %p while resolving deferred binds (op %u size %u)", (const void *)p,
                     h.op, h.size);
        if (h.op == OP_END) return nd;
        if (h.op == OP_JUMP) {
            memcpy(&p, p + 8, 8);
            continue;
        }
        if (h.op == OP_CALL) {
            uint64_t t;
            memcpy(&t, p + 8, 8);
            if (depth >= 16) rt_fatal("nvn: deferred bind resolution: call depth over 16");
            nd += resolve_deferred((const uint8_t *)(uintptr_t)t, depth + 1);
        } else if (h.op == OP_PUSH_CONST) {
            if (h.size < 8 + offsetof(p_push_const, w)) rt_fatal("nvn: short push-constant record at %p", (const void *)p);
            p_push_const *b = (p_push_const *)(uintptr_t)(p + 8);
            if (b->src) {
                if (b->bytes <= 0 || b->bytes > (int)sizeof b->w || (b->bytes & 3) ||
                    h.size < 8 + offsetof(p_push_const, w) + (size_t)b->bytes)
                    rt_fatal("nvn: invalid deferred push-constant record at %p", (const void *)p);
                nd++;
                memcpy(b->w, (const void *)(uintptr_t)b->src, (size_t)b->bytes);
            }
        } else if (h.op == OP_BIND_TEX || h.op == OP_BIND_IMG || h.op == OP_BIND_SAMP) {
            p_bind_handles *b = (p_bind_handles *)(uintptr_t)(p + 8);
            if (b->src && b->n > 0 && b->n <= 32) {
                nd++;
                uint64_t cur[32];
                memcpy(cur, (const void *)(uintptr_t)b->src, 8 * (size_t)b->n);
                __atomic_fetch_add(&def_reads, 1, __ATOMIC_RELAXED);
                if (memcmp(cur, b->h, 8 * (size_t)b->n)) {
                    if (__atomic_fetch_add(&def_changed, 1, __ATOMIC_RELAXED) < 6)
                        NVN_LOG("deferred bind read new handles at submit: stage %d slot %d, %#llx at record -> %#llx",
                                b->stage, b->first, (unsigned long long)b->h[0], (unsigned long long)cur[0]);
                    memcpy(b->h, cur, 8 * (size_t)b->n);
                }
            }
        }
        p += h.size;
    }
}

void nvn_submit_prepare(const uint64_t *handles, int n) {
    uint64_t now = __atomic_add_fetch(&nsubmits, 1, __ATOMIC_RELAXED);
    for (int i = 0; i < n; i++) {
        uint64_t handle = handles[i];
        copy_touch(handle, now, "QueueSubmitCommands");
    }
    wait_submission(handles, n);
    for (int i = 0; i < n; i++) resolve_deferred((const uint8_t *)(uintptr_t)handles[i], 0);
}

uint64_t nvn_submit_count(void) { return __atomic_load_n(&nsubmits, __ATOMIC_RELAXED); }

const char *nvn_op_name(uint32_t op) {
    static const char *const names[] = {
        "END", "JUMP", "BEGIN", "CALL", "SIGNAL_EVENT", "REPORT_COUNTER", "COPY_B2B", "UPDATE_UB", "SET_RT",
        "CLEAR_COLOR", "CLEAR_DS", "VIEWPORTS", "SCISSOR", "DEPTH_RANGE", "BIND_PROGRAM", "BIND_STATE", "BIND_VATTR",
        "BIND_VSTREAM", "BIND_VBUF", "BIND_UBO", "BIND_SSBO", "BIND_TEX", "BIND_SAMP", "STENCIL", "POLY_OFFSET", "DRAW",
        "DRAW_IDX", "DISPATCH", "BARRIER", "SET_TEX_POOL", "SET_SAMP_POOL", "COPY_B2T", "COPY_STRIDE", "BLEND_COLOR",
        "COPY_T2B", "BIND_IMG", "COPY_T2T", "RESET_COUNTER", "CLEAR_TEX", "DEPTH_CLAMP", "RENDER_ENABLE",
        "RASTER_DISCARD", "RENDER_COND", "PRIM_RESTART", "DRAW_IDX_IND", "DISPATCH_IND", "CLEAR_BUF", "PUSH_CONST", "WAIT_SYNC", "COPY_IMAGE_STRIDE",
        "DRAW_MESH", "DRAW_MESH_IND", "GEOMETRY_MODE", "WAIT_EVENT",
    };
    _Static_assert(sizeof names / sizeof *names == OP_COUNT_, "one name per op");
    return op < OP_COUNT_ ? names[op] : "?";
}
