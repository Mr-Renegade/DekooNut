// Adapted from MacNativeGames games/totk/runtime/ipc.c @ b75be8c (MIT, Copyright (c) 2026 MacNativeGames
// contributors). Message framing per libnx sf/hipc.h, cmif.h, tipc.h (ISC), read for the layout only.
// In-process Horizon IPC: SVCs 0x15, 0x1F, 0x21, 0x22, the sm: service manager and the service table. Requests are read
// from the calling thread's message buffer and answered in place; unknown ports, services and commands are named fatals.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audren.h"
#include "host.h"
#include "ipc.h"
#include "platform/platform.h"

#define CMIF_IN_MAGIC 0x49434653u    // "SFCI"
#define CMIF_OUT_MAGIC 0x4f434653u   // "SFCO"
#define DOMAIN_OBJECTS 64            // object ids of one domain; 0 is invalid and 1 is the session itself
enum { DOMAIN_SEND_MESSAGE = 1, DOMAIN_CLOSE = 2 };
static PlatMutex domain_lock = PLAT_MUTEX_INIT;
#define POINTER_BUFFER_SIZE 0x8000
#define R_INVALID_SIZE 0xCA01u
#define R_INVALID_ADDRESS 0xCC01u

enum { T_CLOSE = 2, T_REQUEST = 4, T_CONTROL = 5, T_REQUEST_CTX = 6, T_CONTROL_CTX = 7, T_TIPC_CLOSE = 15,
       T_TIPC_BASE = 16 };

// Services sm: hands out. Anything else is a named fatal.
static const struct { const char *name; IpcHandler handler; } services[] = {
    {"audren:u", audren_manager_handler},
    {"aud:u", aud_user_handler},   // Switch 2: the way the SDK reaches audren:u
    {"auddmg:u", auddmg_handler},   // Switch 2: virtual output devices (audio_nx2.c)
    {"audsmx:u", audsmx_handler},   // Switch 2: sysmix PCM output streams (audio_sysmix.c)
};

static volatile int32_t g_live;
static volatile int64_t g_released, g_requests;
static _Thread_local const Cpu *t_req;   // guest context of the request being served

// ---- sessions
uint32_t ipc_session_new(const char *name, IpcHandler handler, void *state, void (*close)(void *state)) {
    IpcSession *s = calloc(1, sizeof *s);
    if (!s) rt_fatal("ipc: session allocation");
    *s = (IpcSession){name, handler, state, close, 1};
    plat_inc32(&g_live);
    return kern_handle_new(KERN_SESSION, s);
}

static uint32_t session_clone(IpcSession *s) {
    ipc_session_retain(s);
    return kern_handle_new(KERN_SESSION, s);
}

void ipc_session_retain(IpcSession *s) {
    plat_inc32(&s->refs);
}

void ipc_session_release(IpcSession *s) {
    if (plat_dec32(&s->refs)) return;
    if (rt_trace & RT_TRACE_IPC) rt_log("ipc: %s closed", s->name);
    if (s->close) s->close(s->state);
    if (s->objs) {
        for (unsigned i = 2; i < DOMAIN_OBJECTS; i++)
            if (s->objs[i]) ipc_session_release(s->objs[i]);
        free(s->objs);
    }
    free(s);
    plat_dec32(&g_live);
    plat_inc64(&g_released);
}

void ipc_handle_closed(int kind, void *obj) {
    if (kind == KERN_TMEM) {
        free(obj);
        return;
    }
    if (kind != KERN_SESSION) rt_fatal("ipc_handle_closed: handle kind %d", kind);
    ipc_session_release(obj);
}

// ---- helpers for services
static int readable(uint64_t a, uint64_t n) { return plat_mem_readable(a, n); }

void ipc_backtrace(void) {
    const Cpu *c = t_req;
    if (!c) return;
    char d[160];
    rt_logf("  ipc request from %s\n", ldr_describe(c->x[30], d, sizeof d));
    uint64_t fp = c->x[29];
    for (int n = 0; n < 24 && fp && !(fp & 7) && readable(fp, 16); n++) {
        uint64_t next = LD64(fp), lr = LD64(fp + 8);
        rt_logf("  guest frame %2d: %s\n", n, ldr_describe(lr, d, sizeof d));
        if (next <= fp || next - fp > (64u << 20)) break;
        fp = next;
    }
}

void *ipc_guest(uint64_t addr, uint64_t size) {
    if ((!addr && size) || addr + size < addr || addr + size > RT_ADDRESS_LIMIT) {
        ipc_backtrace();
        rt_fatal("ipc: client buffer %#llx+%#llx", (unsigned long long)addr, (unsigned long long)size);
    }
    return (void *)(uintptr_t)addr;
}

const void *ipc_arg(const IpcRequest *rq, uint32_t off, uint32_t n) {
    if (off + n > rq->data_size) {
        ipc_backtrace();
        rt_fatal("ipc: command %u argument %u+%u beyond %u bytes", rq->cmd, off, n, rq->data_size);
    }
    return rq->data + off;
}

void ipc_out(IpcResponse *rs, const void *p, uint32_t n) {
    if (rs->data_size + n > sizeof rs->data) rt_fatal("ipc: response larger than %u bytes", (unsigned)sizeof rs->data);
    memcpy(rs->data + rs->data_size, p, n);
    rs->data_size += n;
}

void ipc_unimplemented(const IpcSession *s, const IpcRequest *rq) {
    fflush(stdout);
    rt_logf("ipc: %s command %u (type %u, data %u bytes, x%u a%u b%u w%u c%u, copy %u move %u)\n", s->name,
            rq->cmd, rq->type, rq->data_size, rq->nx, rq->na, rq->nb, rq->nw, rq->nc, rq->ncopy, rq->nmove);
    ipc_backtrace();
    rt_fatal("ipc: %s command %u is not implemented", s->name, rq->cmd);
}

// ---- HIPC framing
static uint64_t static_addr(const uint32_t *d) {
    return d[1] | ((uint64_t)((d[0] >> 12) & 0xf) << 32) | ((uint64_t)((d[0] >> 6) & 0x3f) << 36);
}
static uint64_t buffer_addr(const uint32_t *d) {
    return d[1] | ((uint64_t)((d[2] >> 28) & 0xf) << 32) | ((uint64_t)((d[2] >> 2) & 0x3fffff) << 36);
}
static uint64_t buffer_size(const uint32_t *d) { return d[0] | ((uint64_t)((d[2] >> 24) & 0xf) << 32); }

static void parse(const uint32_t *m, IpcRequest *rq, const uint32_t **data_words, unsigned *ndata) {
    memset(rq, 0, sizeof *rq);
    rq->type = m[0] & 0xffff;
    rq->nx = (m[0] >> 16) & 0xf;
    rq->na = (m[0] >> 20) & 0xf;
    rq->nb = (m[0] >> 24) & 0xf;
    rq->nw = (m[0] >> 28) & 0xf;
    *ndata = m[1] & 0x3ff;
    unsigned recv_mode = (m[1] >> 10) & 0xf, special = m[1] >> 31;
    if (rq->nx > IPC_MAX_BUF || rq->na > IPC_MAX_BUF || rq->nb > IPC_MAX_BUF || rq->nw > IPC_MAX_BUF)
        rt_fatal("ipc: too many buffers (x%u a%u b%u w%u)", rq->nx, rq->na, rq->nb, rq->nw);
    const uint32_t *p = m + 2;
    if (special) {
        uint32_t h = *p++;
        rq->has_pid = h & 1;
        rq->ncopy = (h >> 1) & 0xf;
        rq->nmove = (h >> 5) & 0xf;
        if (rq->ncopy > 8 || rq->nmove > 8) rt_fatal("ipc: too many handles (copy %u move %u)", rq->ncopy, rq->nmove);
        if (rq->has_pid) {
            memcpy(&rq->pid, p, 8);
            p += 2;
        }
        for (unsigned i = 0; i < rq->ncopy; i++) rq->copy[i] = *p++;
        for (unsigned i = 0; i < rq->nmove; i++) rq->move[i] = *p++;
    }
    for (unsigned i = 0; i < rq->nx; i++, p += 2) rq->x[i] = (IpcBuffer){static_addr(p), p[0] >> 16, p[0] & 0x3f};
    IpcBuffer *lists[3] = {rq->a, rq->b, rq->w};
    unsigned counts[3] = {rq->na, rq->nb, rq->nw};
    for (unsigned k = 0; k < 3; k++)
        for (unsigned i = 0; i < counts[k]; i++, p += 3) lists[k][i] = (IpcBuffer){buffer_addr(p), buffer_size(p), i};
    *data_words = p;
    const uint32_t *recv = p + *ndata;
    rq->nc = recv_mode == 2 ? 1 : recv_mode > 2 ? recv_mode - 2 : 0;
    if (rq->nc > IPC_MAX_BUF) rt_fatal("ipc: %u receive buffers", rq->nc);
    for (unsigned i = 0; i < rq->nc; i++, recv += 2)
        rq->c[i] = (IpcBuffer){recv[0] | ((uint64_t)(recv[1] & 0xffff) << 32), recv[1] >> 16, i};
    if ((const uint8_t *)recv - (const uint8_t *)m > IPC_MSG_BYTES) rt_fatal("ipc: request exceeds the message buffer");
}

static void respond(uint32_t *m, int tipc, const IpcResponse *rs, int domain, const uint32_t *obj_ids, unsigned nobj) {
    uint32_t out[IPC_MSG_BYTES / 4];
    memset(out, 0, sizeof out);
    unsigned special = rs->ncopy || rs->nmove;
    uint32_t *p = out + 2;
    if (special) {
        *p++ = (rs->ncopy << 1) | (rs->nmove << 5);
        for (unsigned i = 0; i < rs->ncopy; i++) *p++ = rs->copy[i];
        for (unsigned i = 0; i < rs->nmove; i++) *p++ = rs->move[i];
    }
    for (unsigned i = 0; i < rs->nxout; i++, p += 2) {
        uint64_t a = rs->xout[i].addr;
        p[0] = (rs->xout[i].index & 0x3f) | (uint32_t)(((a >> 36) & 0x3f) << 6) | (uint32_t)(((a >> 32) & 0xf) << 12) |
               (uint32_t)(rs->xout[i].size << 16);
        p[1] = (uint32_t)a;
    }
    uint8_t *words = (uint8_t *)p, *payload;
    unsigned header = tipc ? 4 : domain ? 32 : 16;   // domain: DomainOutHeader (16) then the CMIF header (16)
    payload = tipc ? words : (uint8_t *)out + (((words - (uint8_t *)out) + 15) & ~15);
    unsigned used = (unsigned)(payload - words) + header + rs->data_size + (domain ? 4 * nobj : 0);
    unsigned nwords = tipc ? (used + 3) / 4 : (used + 16 + 3) / 4;   // CMIF keeps alignment slack like libnx
    if ((size_t)(words - (uint8_t *)out) + nwords * 4 > IPC_MSG_BYTES) rt_fatal("ipc: response exceeds the message buffer");
    if (tipc) memcpy(payload, &rs->result, 4);
    else {
        uint32_t h[4] = {CMIF_OUT_MAGIC, 0, rs->result, 0};
        if (domain) {
            uint32_t d[4] = {nobj, 0, 0, 0};
            memcpy(payload, d, 16);
        }
        memcpy(payload + (domain ? 16 : 0), h, 16);
    }
    memcpy(payload + header, rs->data, rs->data_size);
    if (domain && nobj) memcpy(payload + header + rs->data_size, obj_ids, 4 * nobj);
    out[0] = (rs->nxout & 0xf) << 16;
    out[1] = (nwords & 0x3ff) | (special ? 1u << 31 : 0);
    memcpy(m, out, IPC_MSG_BYTES);
}

// CMIF control requests.
static void control(IpcSession *s, const IpcRequest *rq, IpcResponse *rs) {
    switch (rq->cmd) {
    case 0: {   // ConvertCurrentObjectToDomain: the session becomes object 1 of a domain; its requests now carry a domain header
        plat_mutex_lock(&domain_lock);
        if (!s->objs) {
            s->objs = calloc(DOMAIN_OBJECTS, sizeof *s->objs);
            if (!s->objs) rt_fatal("ipc: domain allocation");
        }
        plat_mutex_unlock(&domain_lock);
        uint32_t id = 1;
        IPC_OUT(rs, id);
        return;
    }
    case 3: { uint16_t n = POINTER_BUFFER_SIZE; IPC_OUT(rs, n); return; }   // QueryPointerBufferSize
    case 2: case 4: rs->move[rs->nmove++] = session_clone(s); return;          // CloneCurrentObject(Ex)
    default:
        ipc_backtrace();
        rt_fatal("ipc: %s control command %u is not implemented", s->name, rq->cmd);
    }
}

// ---- domains: a session converted by control command 0 holds up to DOMAIN_OBJECTS objects by id
static uint32_t domain_add(IpcSession *d, IpcSession *child) {   // takes the caller's reference on child
    plat_mutex_lock(&domain_lock);
    for (uint32_t id = 2; id < DOMAIN_OBJECTS; id++)
        if (!d->objs[id]) {
            d->objs[id] = child;
            plat_mutex_unlock(&domain_lock);
            return id;
        }
    plat_mutex_unlock(&domain_lock);
    ipc_backtrace();
    rt_fatal("ipc: %s: the domain holds %d objects already", d->name, DOMAIN_OBJECTS);
}

static void domain_close(IpcSession *d, uint32_t id) {
    IpcSession *child = NULL;
    plat_mutex_lock(&domain_lock);
    if (id >= 2 && id < DOMAIN_OBJECTS) child = d->objs[id], d->objs[id] = NULL;
    plat_mutex_unlock(&domain_lock);
    if (child) ipc_session_release(child);
}

// The object a domain request is for, retained (id 1 is the domain's own session); NULL for an unknown id.
static IpcSession *domain_target(IpcSession *d, uint32_t id) {
    IpcSession *t = NULL;
    plat_mutex_lock(&domain_lock);
    if (id == 1) t = d;
    else if (id > 1 && id < DOMAIN_OBJECTS) t = d->objs[id];
    if (t) ipc_session_retain(t);
    plat_mutex_unlock(&domain_lock);
    return t;
}

static void dispatch(IpcSession *s, uint32_t *m) {
    IpcRequest rq;
    const uint32_t *words;
    unsigned ndata;
    parse(m, &rq, &words, &ndata);
    if (rq.type == T_CLOSE || rq.type == T_TIPC_CLOSE) return;
    plat_inc64(&g_requests);
    IpcResponse rs;
    memset(&rs, 0, sizeof rs);
    int tipc = rq.type >= T_TIPC_BASE;
    IpcSession *target = s;   // the object the request is for: s, or one of its domain's objects
    int domain_msg = 0;
    const uint8_t *bytes = (const uint8_t *)words, *end = bytes + ndata * 4;
    if (tipc) {
        rq.cmd = rq.type - T_TIPC_BASE;
        rq.data = bytes;
        rq.data_size = (uint32_t)(end - bytes);
    } else if (rq.type == T_REQUEST || rq.type == T_REQUEST_CTX || rq.type == T_CONTROL || rq.type == T_CONTROL_CTX) {
        const uint8_t *payload = (const uint8_t *)m + (((bytes - (const uint8_t *)m) + 15) & ~15);
        const uint8_t *cmif = payload;
        uint32_t data_len = payload + 16 <= end ? (uint32_t)(end - payload - 16) : 0;
        if (s->objs && (rq.type == T_REQUEST || rq.type == T_REQUEST_CTX)) {
            // A domain request: DomainInHeader {u8 type, u8 objects in, u16 data size, u32 object id, u32 pad, u32 token},
            // then the CMIF header and arguments (the data size counts both), then the in object ids.
            uint8_t dtype = payload[0], nin = payload[1];
            uint16_t dsize;
            uint32_t oid;
            memcpy(&dsize, payload + 2, 2);
            memcpy(&oid, payload + 4, 4);
            if (dtype == DOMAIN_CLOSE) {
                domain_close(s, oid);
                IpcResponse ok;
                memset(&ok, 0, sizeof ok);
                respond(m, 0, &ok, 0, NULL, 0);
                return;
            }
            if (dtype != DOMAIN_SEND_MESSAGE || nin || dsize < 16) {
                ipc_backtrace();
                rt_fatal("ipc: %s: domain request type %u with %u in objects and %u bytes", s->name, dtype, nin, dsize);
            }
            target = domain_target(s, oid);
            if (!target) {
                ipc_backtrace();
                rt_fatal("ipc: %s: domain object %u does not exist", s->name, oid);
            }
            domain_msg = 1;
            cmif = payload + 16;
            data_len = dsize - 16;
        }
        uint32_t h[4];
        memcpy(h, cmif, 16);
        if (h[0] != CMIF_IN_MAGIC) {
            ipc_backtrace();
            rt_fatal("ipc: %s: CMIF magic %#x", s->name, h[0]);
        }
        rq.cmd = h[2];
        rq.data = cmif + 16;
        rq.data_size = data_len;
    } else {
        fflush(stdout);
        rt_logf("ipc: %s message type %u:", s->name, rq.type);
        for (unsigned i = 0; i < 16; i++) rt_logf(" %08x", m[i]);
        rt_log_raw("\n", 1);
        ipc_backtrace();
        rt_fatal("ipc: %s message type %u is not implemented", s->name, rq.type);
    }
    if (rq.type == T_CONTROL || rq.type == T_CONTROL_CTX) control(s, &rq, &rs);
    else target->handler(target, &rq, &rs);
    uint32_t obj_ids[8];
    unsigned nobj = 0;
    if (domain_msg) {   // objects the service returns are ids in the domain, not moved handles
        for (unsigned i = 0; i < rs.nmove; i++) {
            IpcSession *child = kern_session_acquire(rs.move[i]);
            if (!child) rt_fatal("ipc: %s: a returned object is not a session", target->name);
            kern_handle_close(rs.move[i]);
            obj_ids[nobj++] = domain_add(s, child);
        }
        rs.nmove = 0;
    }
    if (rt_trace & RT_TRACE_IPC) {
        char d[160];
        rt_log("ipc: %s %s %u -> %#x (copy %u move %u) from %s", target->name,
               tipc ? "tipc" : rq.type == T_CONTROL || rq.type == T_CONTROL_CTX ? "control" : "cmd", rq.cmd,
               rs.result, rs.ncopy, rs.nmove, t_req ? ldr_describe(t_req->x[30], d, sizeof d) : "host");
    }
    respond(m, tipc, &rs, domain_msg, obj_ids, nobj);
    if (target != s) ipc_session_release(target);
}

// ---- sm: service manager (CMIF, and TIPC as the SDK's sm client speaks it since 12.0.0)
static void sm_handler(IpcSession *s, const IpcRequest *rq, IpcResponse *rs) {
    switch (rq->cmd) {
    case 0: return;   // RegisterClient / Initialize (pid)
    case 1: {         // GetServiceHandle(name)
        char name[9] = {0};
        memcpy(name, ipc_arg(rq, 0, 8), 8);
        for (unsigned i = 0; i < sizeof services / sizeof *services; i++)
            if (!strcmp(services[i].name, name)) {
                rs->move[rs->nmove++] = ipc_session_new(services[i].name, services[i].handler, NULL, NULL);
                rt_log("ipc: opened service %s", services[i].name);
                return;
            }
        fflush(stdout);
        ipc_backtrace();
        rt_fatal("sm: service \"%s\" is not implemented", name);
    }
    default: ipc_unimplemented(s, rq);
    }
}

// ---- SVCs
int ipc_svc(Cpu *c, uint32_t imm) {
    uint64_t *x = c->x;
    switch (imm) {
    case 0x1F: {   // ConnectToNamedPort(x1 name -> w1 handle)
        char name[13] = {0};
        const char *p = (const char *)(uintptr_t)x[1];
        if (!p || !readable(x[1], 1)) rt_fatal("ConnectToNamedPort: name at %#llx", (unsigned long long)x[1]);
        for (int i = 0; i < 12 && p[i]; i++) name[i] = p[i];
        if (strcmp(name, "sm:")) {
            t_req = c;
            ipc_backtrace();
            rt_fatal("ConnectToNamedPort(\"%s\"): only sm: is served", name);
        }
        x[0] = 0;
        x[1] = ipc_session_new("sm:", sm_handler, NULL, NULL);
        return 1;
    }
    case 0x21: case 0x22: {   // SendSyncRequest(w0 handle) / WithUserBuffer(x0 buffer, x1 size, w2 handle)
        uint32_t h = (uint32_t)(imm == 0x21 ? x[0] : x[2]);
        IpcSession *s = kern_session_acquire(h);
        if (!s) {
            rt_log("ipc: svc %#x on %#x, not a session handle", imm, h);
            x[0] = IPC_R_BAD_HANDLE;
            return 1;
        }
        uint32_t *m = (uint32_t *)(uintptr_t)(imm == 0x21 ? c->tpidrro : x[0]);
        if (!m || (imm == 0x22 && x[1] < IPC_MSG_BYTES))
            rt_fatal("ipc: svc %#x message buffer %p size %#llx", imm, (void *)m, (unsigned long long)x[1]);
        const Cpu *outer = t_req;
        t_req = c;
        dispatch(s, m);
        t_req = outer;
        ipc_session_release(s);
        x[0] = 0;
        return 1;
    }
    case 0x15: {   // CreateTransferMemory(x1 address, x2 size, w3 permission -> w1 handle)
        if ((x[1] & 0xfff) || x[1] + x[2] < x[1] || x[1] + x[2] > RT_ADDRESS_LIMIT) {
            x[0] = R_INVALID_ADDRESS;
            return 1;
        }
        if (!x[2] || (x[2] & 0xfff)) {
            x[0] = R_INVALID_SIZE;
            return 1;
        }
        IpcTransferMemory *t = malloc(sizeof *t);
        if (!t) rt_fatal("ipc: transfer memory allocation");
        *t = (IpcTransferMemory){x[1], x[2], (uint32_t)x[3]};
        x[0] = 0;
        x[1] = kern_handle_new(KERN_TMEM, t);
        return 1;
    }
    }
    return 0;
}
