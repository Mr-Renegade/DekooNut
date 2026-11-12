// Switch 2 audio device manager, "auddmg:u": the nn::audio virtual output devices (nn::audio::ListVirtualOutputDevices and
// friends) the SDK reaches before it opens an output stream. Reference for the wire format: Lichii/Eden
// src/core/hle/service/audio/nx2_audio_device_manager.h (GPL-3.0-or-later, read for the command layout only).
// Audio device manager: see README, "Audio device manager".
#include <stdint.h>
#include <string.h>

#include "audren.h"
#include "host.h"
#include "ipc.h"

static const char virtual_device[] = "AudioVirtualOutput0";
static const char physical_device[] = "AudioTvOutput";
enum { NAME_SIZE = 0x100, PROPS_SIZE = 0x228 };

// The first output buffer the caller gave: a B/W buffer, or the C (pointer) buffer the service answers through.
static IpcBuffer out_buf(const IpcRequest *rq, int *pointer) {
    *pointer = !(rq->nb && rq->b[0].size) && !(rq->nw && rq->w[0].size) && rq->nc && rq->c[0].size;
    if (*pointer) return rq->c[0];
    if (rq->nb && rq->b[0].size) return rq->b[0];
    return rq->nw ? rq->w[0] : (IpcBuffer){0};
}

static void answer_out(IpcResponse *rs, const IpcBuffer *b, int pointer) {
    if (pointer) rs->xout[rs->nxout++] = (IpcBuffer){b->addr, b->size, b->index};
}

static void put_text(uint8_t *dst, const char *s) {
    size_t n = strlen(s);
    memcpy(dst, s, n < NAME_SIZE - 1 ? n : NAME_SIZE - 1);
}

static void vdm_handler(IpcSession *s, const IpcRequest *rq, IpcResponse *rs) {
    static uint32_t state_event;   // never signalled: the device does not change
    int pointer;
    switch (rq->cmd) {
    case 0: {   // ListVirtualOutputDevices
        IpcBuffer b = out_buf(rq, &pointer);
        uint32_t n = b.size >= NAME_SIZE ? 1 : 0;
        if (n) {
            uint8_t *dst = ipc_guest(b.addr, b.size);
            memset(dst, 0, NAME_SIZE);
            put_text(dst, virtual_device);
            answer_out(rs, &b, pointer);
        }
        IPC_OUT(rs, n);
        return;
    }
    case 100: {   // GetDefaultVirtualOutputDevice
        IpcBuffer b = out_buf(rq, &pointer);
        if (b.size >= NAME_SIZE) {
            uint8_t *dst = ipc_guest(b.addr, b.size);
            memset(dst, 0, NAME_SIZE);
            put_text(dst, virtual_device);
            answer_out(rs, &b, pointer);
        }
        return;
    }
    case 300: return;   // IsVirtualOutputDeviceStreamFormatSupported: the host mixer takes any PCM format, result 0
    case 401: {         // GetVirtualOutputDeviceProperties
        IpcBuffer b = out_buf(rq, &pointer);
        if (b.size < PROPS_SIZE) {
            rs->result = 0xCA01u;   // invalid size
            return;
        }
        uint8_t *dst = ipc_guest(b.addr, b.size);
        memset(dst, 0, PROPS_SIZE);
        // Name at 0, type 3 at 0x100, 2 at 0x104, stereo 0x108, 48 kHz 0x10c, physical name 0x110, latency ns 0x218,
        // device id 0x220 (nonzero: sysmix CreateStream refuses 0).
        put_text(dst, virtual_device);
        const uint32_t v100 = 3, v104 = 2, channels = 2, rate = 48000, dev_id = 1;
        const uint64_t latency_ns = 10000000;
        memcpy(dst + 0x100, &v100, 4), memcpy(dst + 0x104, &v104, 4), memcpy(dst + 0x108, &channels, 4);
        memcpy(dst + 0x10c, &rate, 4);
        put_text(dst + 0x110, physical_device);
        memcpy(dst + 0x218, &latency_ns, 8), memcpy(dst + 0x220, &dev_id, 4);
        answer_out(rs, &b, pointer);
        return;
    }
    case 500: {   // AcquireVirtualOutputDeviceStateEvent
        uint32_t ev = __atomic_load_n(&state_event, __ATOMIC_ACQUIRE);
        if (!ev) {
            uint32_t mine = kern_event_new(), expected = 0;
            ev = __atomic_compare_exchange_n(&state_event, &expected, mine, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)
                     ? mine : expected;
        }
        rs->copy[rs->ncopy++] = kern_event_readable(ev);
        return;
    }
    case 700: {   // AttachStream(u32 stream, u32 functionality, name buffer): starts the stream
        uint32_t id = IPC_ARG(rq, uint32_t, 0), functionality = IPC_ARG(rq, uint32_t, 4);
        rs->result = functionality == 1 ? sysmix_set_engaged(id, 1) : 0x40299u;   // 1 = output; else not supported
        return;
    }
    case 800: rs->result = sysmix_set_engaged(IPC_ARG(rq, uint32_t, 0), 0); return;   // DetachStream: stops it
    default: ipc_unimplemented(s, rq);
    }
}

void auddmg_handler(IpcSession *s, const IpcRequest *rq, IpcResponse *rs) {
    if (rq->cmd == 0) {   // OpenVirtualDeviceManager (16 bytes and the pid)
        rs->move[rs->nmove++] = ipc_session_new("IAudioVirtualDeviceManager", vdm_handler, NULL, NULL);
        return;
    }
    ipc_unimplemented(s, rq);
}
