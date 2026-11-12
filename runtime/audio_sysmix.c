// Switch 2 sound mixer service "audsmx:u" (sysmix): the PCM output streams nn::audio::OutputStream uses. The wire
// format is the console's. Commands and the shared buffer layout: see docs/how-it-works.md, "Sysmix audio service".
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <SDL3/SDL.h>

#include "audren.h"
#include "cfg.h"
#include "host.h"
#include "ipc.h"

enum { MAX_STREAMS = 16, BLOCK_FRAMES = 240, TICK_MS = 5 };
enum { READ_OFF = 0x44, WRITE_OFF = 0x1004, PAYLOAD_OFF = 0x1040 };   // shared buffer: read index, write index, PCM ring
enum { FMT_S16 = 2, FMT_F32 = 5 };                        // sample types: PCM int16, PCM float
enum { R_SMX_NOT_SUPPORTED = 0x40299u, R_SMX_INVALID_ARG = 0x20099u, R_SMX_BAD_ADDRESS = 0x1a099u,
       R_SMX_FAILED = 0x00c99u, R_SMX_INSUFFICIENT = 0x04099u, R_SMX_NOT_FOUND = 0x14099u };

typedef struct {
    int used, open, engaged, faulted;
    uint32_t id, flags;
    uint8_t *buf;
    uint64_t size;
    uint32_t format, frame_bytes, slots, consumer, data_mode;
    int32_t threshold;
    float gain;
    uint32_t state_event, data_event;   // writable event handles
    SDL_AudioStream *out;
} Sysmix;

static Sysmix streams[MAX_STREAMS];
static PlatMutex lock = PLAT_MUTEX_INIT;
static uint32_t next_id = 1;
static volatile int32_t pump_started;

static uint32_t ld32(const Sysmix *s, uint64_t off) {
    return __atomic_load_n((const uint32_t *)(s->buf + off), __ATOMIC_ACQUIRE);
}

// Free slots the game may still write; signals the data event when there are enough (holding lock).
static void notify(Sysmix *s, uint32_t filled) {
    if (s->data_mode == 2 && s->slots - 1 - filled >= (uint32_t)s->threshold) kern_event_signal(s->data_event);
}

static int occupancy(const Sysmix *s, uint32_t *read, uint32_t *write, uint32_t *filled) {
    *read = ld32(s, READ_OFF);
    *write = ld32(s, WRITE_OFF);
    if (*read != s->consumer || *read >= s->slots || *write >= s->slots) return 0;
    *filled = (uint32_t)(((uint64_t)*write + s->slots - *read) % s->slots);
    return 1;
}

static void fault(Sysmix *s, const char *why) {
    if (!s->faulted) rt_log("sysmix: stream %u fault: %s", s->id, why);
    s->faulted = 1, s->engaged = 0;
    kern_event_signal(s->state_event);
}

// One tick of one stream (holding lock).
static void pump_one(Sysmix *s) {
    uint32_t read, write, filled;
    if (!occupancy(s, &read, &write, &filled)) return fault(s, "ring indexes out of range");
    if (!s->engaged) return notify(s, filled);
    uint32_t taken = filled < BLOCK_FRAMES ? filled : BLOCK_FRAMES;
    if (taken) {
        int16_t out[BLOCK_FRAMES * 2];
        for (uint32_t i = 0; i < taken; i++) {
            const uint8_t *p = s->buf + PAYLOAD_OFF + (uint64_t)((read + i) % s->slots) * s->frame_bytes;
            for (int ch = 0; ch < 2; ch++) {
                float v;
                if (s->format == FMT_S16) {
                    int16_t x;
                    memcpy(&x, p + ch * 2, 2);
                    v = (float)x / 32768.0f;
                } else {
                    memcpy(&v, p + ch * 4, 4);
                }
                v = isfinite(v) ? v * s->gain : 0.0f;
                v = v > 1.0f ? 1.0f : v < -1.0f ? -1.0f : v;
                int32_t q = (int32_t)(v * 32768.0f);
                out[i * 2 + ch] = (int16_t)(q > 32767 ? 32767 : q < -32768 ? -32768 : q);
            }
        }
        if (s->out) {
            // Backpressure: the device holds at most ~120 ms; a full queue leaves the frames in the game's ring.
            if (SDL_GetAudioStreamQueued(s->out) > 48000 / 8 * 4) return;
            if (!SDL_PutAudioStreamData(s->out, out, (int)(taken * 4))) return;
        }
        uint32_t next = (read + taken) % s->slots, expected = read;
        if (!__atomic_compare_exchange_n((uint32_t *)(s->buf + READ_OFF), &expected, next, 0, __ATOMIC_ACQ_REL,
                                         __ATOMIC_ACQUIRE))
            return fault(s, "the game moved the read index");
        s->consumer = next;
    }
    if (occupancy(s, &read, &write, &filled)) notify(s, filled);
}

static void pump_thread(void *arg) {
    (void)arg;
    for (;;) {
        plat_sleep_ms(TICK_MS);
        plat_mutex_lock(&lock);
        for (int i = 0; i < MAX_STREAMS; i++)
            if (streams[i].used && streams[i].open && !streams[i].faulted) pump_one(&streams[i]);
        plat_mutex_unlock(&lock);
    }
}

static void ensure_pump(void) {
    if (plat_cas32(&pump_started, 1, 0) == 0 && !plat_thread_spawn(pump_thread, NULL, "sysmix", 0))
        rt_fatal("sysmix: cannot start the mixing thread");
}

static void close_stream_locked(Sysmix *s) {
    s->open = s->engaged = 0;
    if (s->out) {
        SDL_DestroyAudioStream(s->out);
        s->out = NULL;
    }
    kern_event_signal(s->state_event);
}

// ---- stream object
static void stream_close_cb(void *state) {
    Sysmix *s = state;
    plat_mutex_lock(&lock);
    close_stream_locked(s);
    s->used = 0;
    plat_mutex_unlock(&lock);
}

static void stream_handler(IpcSession *sess, const IpcRequest *rq, IpcResponse *rs) {
    Sysmix *s = sess->state;
    switch (rq->cmd) {
    case 0: {   // OpenStream
        uint32_t type = IPC_ARG(rq, uint32_t, 0), ch = IPC_ARG(rq, uint32_t, 4), rate = IPC_ARG(rq, uint32_t, 8);
        uint32_t frames = IPC_ARG(rq, uint32_t, 12), prefill = IPC_ARG(rq, uint32_t, 16), flags = IPC_ARG(rq, uint32_t, 20);
        plat_mutex_lock(&lock);
        uint32_t r = 0;
        uint64_t slots = (uint64_t)frames + 1;
        uint32_t fb = ch * (type == FMT_S16 ? 2u : 4u);
        if (s->open) r = R_SMX_FAILED;
        else if ((type != FMT_S16 && type != FMT_F32) || ch != 2 || rate != 48000 || prefill != 0 || flags != 0)
            r = R_SMX_NOT_SUPPORTED;
        else if (frames == 0 || frames >= 0x7fffffffu) r = R_SMX_INVALID_ARG;
        else if (PAYLOAD_OFF + slots * fb > s->size) r = R_SMX_INSUFFICIENT;
        if (!r) {
            s->slots = (uint32_t)slots, s->format = type, s->frame_bytes = fb;
            uint32_t rd = ld32(s, READ_OFF), wr = ld32(s, WRITE_OFF);
            if (rd >= s->slots || wr >= s->slots) r = R_SMX_BAD_ADDRESS;
            else {
                s->consumer = rd, s->faulted = 0, s->open = 1, s->engaged = 0, s->gain = 1.0f;
                const char *mode = cfg_get("OOT_AUDIO");
                if (!(mode && !strcmp(mode, "none")) && SDL_InitSubSystem(SDL_INIT_AUDIO)) {
                    SDL_AudioSpec spec = {SDL_AUDIO_S16, 2, 48000};
                    s->out = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, NULL, NULL);
                    if (!s->out) rt_log("sysmix: stream %u has no output device (%s): its audio is dropped", s->id, SDL_GetError());
                }
                kern_event_signal(s->state_event);
                notify(s, (uint32_t)(((uint64_t)wr + s->slots - rd) % s->slots));
                rt_log("sysmix: stream %u open: %s stereo 48000, %u frames, device %s", s->id,
                       type == FMT_S16 ? "int16" : "float", frames, s->out ? "yes" : "none");
            }
        }
        plat_mutex_unlock(&lock);
        rs->result = r;
        return;
    }
    case 1: plat_mutex_lock(&lock); close_stream_locked(s); plat_mutex_unlock(&lock); return;
    case 2: {   // GetStreamState
        plat_mutex_lock(&lock);
        uint32_t v = s->open && !s->faulted ? (s->engaged ? 2u : 1u) : 0u;
        plat_mutex_unlock(&lock);
        IPC_OUT(rs, v);
        return;
    }
    case 3: { plat_mutex_lock(&lock); float g = s->gain; plat_mutex_unlock(&lock); IPC_OUT(rs, g); return; }
    case 4: {   // SetStreamGain(f32 gain, u32 pad, s64 ramp ns)
        float g = IPC_ARG(rq, float, 0);
        int64_t ramp = IPC_ARG(rq, int64_t, 8);
        if (ramp != 0) rs->result = R_SMX_NOT_SUPPORTED;
        else if (!isfinite(g) || g < 0.0f || g > 128.0f) rs->result = R_SMX_INVALID_ARG;
        else { plat_mutex_lock(&lock); s->gain = g; plat_mutex_unlock(&lock); }
        return;
    }
    case 5: rs->copy[rs->ncopy++] = kern_event_readable(s->state_event); return;      // AcquireStreamStateChangeEvent
    case 6: return;                                                                      // ReleaseStreamStateChangeEvent
    case 7: {   // AcquireStreamDataEvent(u32 mode, s32 threshold)
        uint32_t mode = IPC_ARG(rq, uint32_t, 0);
        int32_t th = IPC_ARG(rq, int32_t, 4);
        plat_mutex_lock(&lock);
        if (mode != 0 && mode != 2) rs->result = R_SMX_NOT_SUPPORTED;
        else if (!s->open || s->faulted || th < 0 || (uint32_t)th >= s->slots) rs->result = R_SMX_INVALID_ARG;
        else {
            uint32_t rd, wr, filled;
            if (!occupancy(s, &rd, &wr, &filled)) { fault(s, "ring indexes out of range"); rs->result = R_SMX_BAD_ADDRESS; }
            else {
                s->data_mode = mode, s->threshold = th;
                notify(s, filled);
                rs->copy[rs->ncopy++] = kern_event_readable(s->data_event);
            }
        }
        plat_mutex_unlock(&lock);
        return;
    }
    case 8: plat_mutex_lock(&lock); s->data_mode = 0, s->threshold = 0; plat_mutex_unlock(&lock); return;
    case 9: case 10: case 11: rs->result = R_SMX_NOT_SUPPORTED; return;   // ClockStream, PrimeStream, PurgeStream
    case 12: return;                                                       // RenameStream
    default: ipc_unimplemented(sess, rq);
    }
}

// ---- engaging (holding lock)
static Sysmix *find_stream(uint32_t id) {
    for (int i = 0; i < MAX_STREAMS; i++)
        if (streams[i].used && streams[i].id == id) return &streams[i];
    return NULL;
}

static uint32_t engage_locked(Sysmix *s, int engage) {
    if (!s) return R_SMX_NOT_FOUND;
    if (!s->open || s->faulted) return R_SMX_FAILED;
    if (s->engaged == engage) return 0;
    if (s->out) {
        if (engage) SDL_ResumeAudioStreamDevice(s->out);
        else SDL_PauseAudioStreamDevice(s->out);
    }
    s->engaged = engage;
    kern_event_signal(s->state_event);
    return 0;
}

// The device manager (audio_nx2.c) starts a stream by attaching it to a device and stops it by detaching it.
uint32_t sysmix_set_engaged(uint32_t id, int engage) {
    plat_mutex_lock(&lock);
    uint32_t r = engage_locked(find_stream(id), engage);
    plat_mutex_unlock(&lock);
    return r;
}

// ---- user session
static void user_session_handler(IpcSession *sess, const IpcRequest *rq, IpcResponse *rs) {
    switch (rq->cmd) {
    case 1: {   // CreateStream
        uint32_t device = IPC_ARG(rq, uint32_t, 0), flags = IPC_ARG(rq, uint32_t, 4);
        uint64_t buffer = IPC_ARG(rq, uint64_t, 16), size = IPC_ARG(rq, uint64_t, 24);
        if (device != 1) { rs->result = R_SMX_INVALID_ARG; return; }
        if (flags != 0x24) { rs->result = R_SMX_NOT_SUPPORTED; return; }
        if (!buffer || (buffer & 0xfff) || (size & 0xfff) || size < 0x2000 || size > UINT64_MAX - buffer) {
            rs->result = R_SMX_BAD_ADDRESS;
            return;
        }
        uint8_t *host = ipc_guest(buffer, size);
        plat_mutex_lock(&lock);
        Sysmix *s = NULL;
        for (int i = 0; i < MAX_STREAMS && !s; i++)
            if (!streams[i].used) s = &streams[i];
        if (!s) { plat_mutex_unlock(&lock); rs->result = 0x3c699u; return; }   // out of sessions
        *s = (Sysmix){.used = 1, .id = next_id++, .flags = flags, .buf = host, .size = size, .gain = 1.0f,
                      .state_event = kern_event_new(), .data_event = kern_event_new()};
        uint32_t id = s->id;
        plat_mutex_unlock(&lock);
        ensure_pump();
        rt_log("sysmix: stream %u created: buffer %#llx size %#llx", id, (unsigned long long)buffer,
               (unsigned long long)size);
        IPC_OUT(rs, id);
        rs->move[rs->nmove++] = ipc_session_new("ISysmixStream", stream_handler, s, stream_close_cb);
        return;
    }
    case 2: case 3: {   // EngageStreams, DisengageStreams: a buffer of u32 ids
        IpcBuffer b = rq->na && rq->a[0].size ? rq->a[0] : rq->nx ? rq->x[0] : (IpcBuffer){0};
        if (!b.size || b.size % 4) { rs->result = R_SMX_INVALID_ARG; return; }
        const uint32_t *ids = ipc_guest(b.addr, b.size);
        int engage = rq->cmd == 2;
        plat_mutex_lock(&lock);
        for (uint64_t i = 0; i < b.size / 4 && !rs->result; i++) rs->result = engage_locked(find_stream(ids[i]), engage);
        plat_mutex_unlock(&lock);
        return;
    }
    default: ipc_unimplemented(sess, rq);
    }
}

void audsmx_handler(IpcSession *s, const IpcRequest *rq, IpcResponse *rs) {
    if (rq->cmd == 0) {   // OpenUserSession(the process handle)
        rs->move[rs->nmove++] = ipc_session_new("ISysmixUserSession", user_session_handler, NULL, NULL);
        return;
    }
    ipc_unimplemented(s, rq);
}
