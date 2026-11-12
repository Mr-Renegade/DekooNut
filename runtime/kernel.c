// Adapted from MacNativeGames games/totk/runtime/threads.c @ b75be8c (MIT, Copyright (c) 2026 MacNativeGames
// contributors; itself from d2r-mobile dm/host/hle_thread.c), d2r-mobile dm/host/thread.c @ 53e7c5b (MIT, Copyright
// (c) 2026 jhaizhou) and LAS Recomp runtime/kernel.c @ da35b83 (MIT, Copyright (c) 2026 Peiikai); the preemption of
// virtual cores from MacNativeGames threads.c @ 33e8fa8 (MIT, as above).
// Threads, synchronization, virtual cores and fibers of the Horizon kernel. One guest thread is one host thread with a 256 MiB
// reserved stack and its own Cpu; every wait hangs on one global lock G with one condition variable per thread.
// Guest threads and virtual cores: see docs/how-it-works.md, "Guest threads and virtual cores".
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "kernel.h"
#include "platform/platform.h"

#define R_OK 0u
#define R_TIMEOUT KERN_R_TIMEOUT
#define R_CANCELLED KERN_R_CANCELLED
#define R_BAD_HANDLE KERN_R_BAD_HANDLE
#define R_BAD_STATE KERN_R_BAD_STATE
#define R_INVALID_ENUM KERN_R_INVALID_ENUM
#define WAITERS_BIT KERN_WAITERS_BIT
#define H_CUR_THREAD KERN_H_CURRENT_THREAD

#define MAX_H 4096
#define NCORE 6  // Switch 2 applications run on cores 0-5 (main.npdm ACI0 ThreadInfo)
#define HOST_STACK (256ull << 20)        // reserved, committed on demand
#define TLR_SIZE 0x200
#define THREAD_OBJ_SIZE 0x400  // "current thread object" at TLR+0x1F8
#define TLS_SIZE 0x2000
#define MAIN_STACK (1ull << 20)          // main.npdm META+0x1C
#define MAIN_PRIO 44                     // META+0xE
#define MAIN_CORE 0                      // META+0xF
#define FIBER_MAX 1024

enum { O_FREE, O_THREAD, O_EVENT_R, O_EVENT_W, O_SESSION = KERN_SESSION, O_TMEM = KERN_TMEM };
_Static_assert(O_EVENT_W < O_SESSION, "handle kinds");
enum { W_NONE = KERN_WAIT_NONE, W_LOCK = KERN_WAIT_LOCK, W_CONDVAR = KERN_WAIT_CONDVAR, W_ADDRESS = KERN_WAIT_ADDRESS,
       W_SYNC = KERN_WAIT_SYNC };

// Scheduler accounting states (OOT_TRACE=sched).
enum { S_OFF, S_RUN, S_CORE, S_HOST, S_SLEEP, S_SYNC, S_CV, S_LOCK, S_ADDR, S_N };
static const char *const s_name[S_N] = {"off", "run", "core", "host", "sleep", "sync", "cv", "lock", "addr"};

typedef struct Event { int signaled, refs; } Event;
typedef struct Fiber Fiber;

struct RtThread {
    int preempt;               // first member: rt_dispatch reads it (host.h kern_preempt_pending)
    int host_depth;            // second member: host calls and SVCs in progress, while kern_host_track (host.h)
    uint32_t handle;
    uint64_t id;
    Cpu *cpu;                  // the thread's own context
    Cpu *cur;                  // the context running now: cpu, or the current fiber's
    uint64_t entry, arg, stack_top;
    int prio, core;
    uint64_t mask;
    int started, exited, closed, cancel, is_main;
    PlatCond cv;
    // current wait
    int wkind, woke;
    uint64_t waddr, wkey;
    uint32_t wtag, wres;
    uint64_t wseq;             // registration order: equal priority wakes first come first
    RtThread *next;            // wait list
    RtThread *all;             // every thread
    // virtual core
    int cur_core;              // core held while running guest code, -1 while blocked
    uint64_t cseq;
    RtThread *cnext;
    uint8_t *tlr, *tls;
    jmp_buf exit_jmp;          // the thread start; ExitThread longjmps to it
    // fibers
    PlatFiber *base_fiber;     // the host thread as a host fiber, after the first switch
    Fiber *fiber;              // current guest fiber, NULL on the thread's own context
    Fiber *leaving;            // fiber just switched away from, released by whoever resumes next
    int exit_pending;          // ExitThread on a fiber: unwind once back on the thread's own context
    // OOT_TRACE=sched accounting (holding G)
    uint32_t tid;              // host thread id
    int st;                    // S_* state since st_since
    uint64_t st_since, st_ns[S_N], nblk, npre, pre_since, cpu_last;
    uint64_t wobj;             // what the current wait is on: owner handle, key, address or first handle
    uint32_t waker;            // the thread whose signal ended the last wait (0: a host thread or a timeout)
    uint64_t wake_at, grant_at;   // when a wait was signalled and a core granted, until the thread runs again
    uint64_t lw_ns, lw_n, lw_max, lg_ns, lg_n, lg_max, sl_n, sl_req, sl_act;   // those latencies; SleepThread asked/took
    uint64_t sp_n, sp_ho, sp_site;   // spin wait passes and core handoffs, the last loop head (the thread writes them)
    RtThread *target;          // as a core waiter: the owner core_kick asked to give its core up for it (holding G)
    // asynchronous preemption (holding G)
    PlatIntr *hthr;            // the host thread, for plat_intr_stop / resume; open while it may own a core
    uint64_t ap_rip;           // the interrupted host rip kern_ap_stub returns to
    uint64_t ap_last;          // the last redirect attempt on this owner
    int ap_pending;            // redirected to kern_ap_stub, which has not reached kern_ap_yield yet
};

static PlatMutex G = PLAT_MUTEX_INIT;
static struct { int type; void *obj; } g_h[MAX_H];
static _Thread_local RtThread *tcur;

// A native fiber can resume on another OS thread: keep the TLS lookup out of the suspended function (noinline, barrier).
static __attribute__((noinline)) RtThread *thread_after_switch(void) {
    __asm__ volatile("" ::: "memory");
    return tcur;
}
_Static_assert(offsetof(RtThread, preempt) == 0, "host.h kern_preempt_pending reads the first int");
_Static_assert(offsetof(RtThread, host_depth) == sizeof(int), "host.h kern_host_depth reads the second int");
static RtThread *g_all, *g_wait, *g_main;
static uint64_t g_next_id = 0x51, g_wseq, g_created;
static uint64_t g_main_stack;

static void lock(void) { plat_mutex_lock(&G); }
static void unlock(void) { plat_mutex_unlock(&G); }

uint64_t kern_now_ns(void) { return plat_now_ns(); }

// ---- OOT_TRACE=sched accounting (holding G)
#define RW_N 32
static RtThread *g_render;  // the thread that presents (sched trace)
typedef struct { int st; uint32_t waker; uint64_t obj, ns; } WaitRec;
static WaitRec g_rw[RW_N], g_mw[RW_N];                   // the render and main thread's waits by object and waker
static uint64_t g_pre_req, g_pre_n, g_pre_ns, g_pre_max, g_pre_stale;  // preempt requests; releases after one and their latency
static uint64_t g_sync_calls, g_sync_blocks;              // WaitSynchronization calls and the blocks inside them

static void wait_add(WaitRec *w, int st, uint64_t obj, uint32_t waker, uint64_t ns) {
    for (int i = 0; i < RW_N; i++)
        if (!w[i].ns || (w[i].st == st && w[i].obj == obj && w[i].waker == waker)) {
            w[i].st = st;
            w[i].obj = obj;
            w[i].waker = waker;
            w[i].ns += ns;
            return;
        }
}

// Close the current state's interval and enter st.
static void acct(RtThread *t, int st) {
    if (!(rt_trace & RT_TRACE_SCHED)) return;
    uint64_t now = kern_now_ns();
    if (t->st_since) {
        t->st_ns[t->st] += now - t->st_since;
        if (t->st >= S_HOST && (t == g_render || t->is_main)) {
            uint32_t waker = t->st == S_HOST || t->st == S_SLEEP ? 0 : t->waker;
            wait_add(t == g_render ? g_rw : g_mw, t->st, t->wobj, waker, now - t->st_since);
        }
    }
    t->st = st;
    t->st_since = now;
}

static int st_of(int kind) {
    return kind == W_LOCK ? S_LOCK : kind == W_CONDVAR ? S_CV : kind == W_ADDRESS ? S_ADDR : S_SYNC;
}

// ---- handles (caller holds G)
static uint32_t h_new(int type, void *obj) {
    for (int i = 1; i < MAX_H; i++)
        if (!g_h[i].type) {
            g_h[i].type = type;
            g_h[i].obj = obj;
            return 0x1000 + i;
        }
    rt_fatal("kernel: handle table full (%d handles)", MAX_H);
}

static void *h_peek(uint32_t h, int type) {
    if (h == H_CUR_THREAD && type == O_THREAD) return tcur;
    unsigned i = h - 0x1000;
    if (i == 0 || i >= MAX_H || g_h[i].type != type) return NULL;
    return g_h[i].obj;
}

static void *h_get(uint32_t h, int type) {
    void *o = h_peek(h, type);
    if (!o) rt_log("kernel: invalid handle %#x (want kind %d)", h, type);
    return o;
}

static RtThread *cur(uint32_t svc) {
    if (!tcur) rt_fatal("kernel: svc %#x on a host thread (no guest thread)", svc);
    return tcur;
}

// ---- wait primitives (holding G)
static void wait_unlink(RtThread *t) {
    for (RtThread **p = &g_wait; *p; p = &(*p)->next)
        if (*p == t) {
            *p = t->next;
            break;
        }
    t->next = NULL;
}

// The waiter to wake in kernel order. Lock waits match by lock address, condvar waits by key, arbiter waits by address.
static RtThread *pick(int kind, uint64_t a) {
    RtThread *b = NULL;
    for (RtThread *t = g_wait; t; t = t->next) {
        if (t->wkind != kind || (kind == W_CONDVAR ? t->wkey : t->waddr) != a) continue;
        if (!b || t->prio < b->prio || (t->prio == b->prio && t->wseq < b->wseq)) b = t;
    }
    return b;
}

static void wake(RtThread *t, uint32_t res) {
    wait_unlink(t);
    t->wkind = W_NONE;
    t->woke = 1;
    t->wres = res;
    t->waker = tcur ? tcur->handle : 0;
    if (rt_trace & RT_TRACE_SCHED) t->wake_at = kern_now_ns();
    plat_cond_signal(&t->cv);
}

static void wake_sync_waiters(void) {
    for (RtThread *w = g_wait; w;) {
        RtThread *n = w->next;
        if (w->wkind == W_SYNC) wake(w, (uint32_t)-1);
        w = n;
    }
}

// SleepConditionVariableSRW counts milliseconds: round up (early is not legal) and re-check the deadline against the counter.
static uint32_t block_inner(RtThread *t, int kind, int64_t ns) {
    t->wkind = kind;
    t->woke = 0;
    t->wseq = ++g_wseq;
    t->next = g_wait;
    g_wait = t;
    if (ns == 0) {
        wait_unlink(t);
        t->wkind = W_NONE;
        return R_TIMEOUT;
    }
    uint64_t now = kern_now_ns();
    int forever = ns < 0 || (uint64_t)ns > UINT64_MAX - now;
    uint64_t deadline = forever ? 0 : now + (uint64_t)ns;
    while (!t->woke) {
        if (forever) {
            plat_cond_wait(&t->cv, &G);
            continue;
        }
        now = kern_now_ns();
        if (now >= deadline) {
            wait_unlink(t);
            t->wkind = W_NONE;
            return R_TIMEOUT;
        }
        plat_cond_wait_ns(&t->cv, &G, deadline - now);
    }
    if (t->wake_at) {
        uint64_t d = kern_now_ns() - t->wake_at;
        t->lw_ns += d;
        t->lw_n++;
        if (d > t->lw_max) t->lw_max = d;
        t->wake_at = 0;
    }
    return t->wres;
}

// ---- virtual cores
// A guest thread holds one of the application cores while it runs guest code: the game keys per-core data (allocators, job
// queues) by the current core number without locks. See README, "Guest threads and virtual cores".
static RtThread *g_core_owner[NCORE], *g_core_wait;

static void ap_init(void);

static int cores_enabled(void) {
    static int ready;
    if (!ready) {
        ready = 1;
        ap_init();
    }
    return 1;
}

static int core_allowed(const RtThread *t, int c) { return (int)((t->mask >> c) & 1); }

// ---- asynchronous preemption: see docs/how-it-works.md, "Asynchronous preemption".
// Preemption disable: the SDK's spin locks count TLR+0x100 (disable_count) and the switch is deferred to SVC 0x36 when
// TLR+0x102 (interrupt_flag) is set; such an owner is never preempted.
#define AP_FIRST_NS 250000ull   // the waiter spins this long before the first interruption (most waits end sooner)
#define AP_GAP_NS 1000000ull    // and interrupts the same owner at most this often
static int g_ap = -1;
static int g_ap_log;              // redirects logged so far (AP_LOG_N)
#define AP_LOG_N 40
static uintptr_t g_gen_lo, g_gen_hi;   // host code of the lifted modules: first and last function start
static uint64_t g_ap_sent, g_ap_taken, g_ap_host, g_ap_pc, g_ap_deferred, g_ap_fail, g_ap_spin_ok;
void kern_ap_stub(void);

static void ap_init(void) {
    if (g_ap >= 0) return;
    g_ap = 1;
    RtModule m[16];
    int n = rt_linked_modules(m, 16);
    uintptr_t lo = UINTPTR_MAX, hi = 0;
    for (int i = 0; i < n; i++)
        for (uint64_t k = 0; k < m[i].count; k++) {
            uintptr_t f = (uintptr_t)m[i].fns[k];
            if (f < lo) lo = f;
            if (f > hi) hi = f;
        }
    const uintptr_t host[] = {(uintptr_t)kern_ap_stub, (uintptr_t)kern_checkpoint, (uintptr_t)rt_svc,
                              (uintptr_t)rt_dispatch, (uintptr_t)rt_call};
    int bad = !n || lo >= hi;
    for (unsigned i = 0; i < sizeof host / sizeof *host; i++) bad |= host[i] >= lo && host[i] <= hi;
    if (bad) {
        rt_log("kernel: asynchronous preemption off: the lifted code range %#llx-%#llx holds host code",
               (unsigned long long)lo, (unsigned long long)hi);
        g_ap = 0;
    } else {
        g_gen_lo = lo;
        g_gen_hi = hi;
        kern_host_track = 1;
    }
    rt_log("kernel: asynchronous preemption of virtual cores %s, priority inheritance and preemption disable %s; lifted "
           "code exe+%#llx-exe+%#llx", "on", g_ap ? "on" : "off",
           (unsigned long long)(g_gen_lo ? g_gen_lo - plat_exe_base() : 0),
           (unsigned long long)(g_gen_hi ? g_gen_hi - plat_exe_base() : 0));
}

// Horizon's priority inheritance: o counts at the priority of the best thread blocked on a guest mutex it owns. Holding G.
static int eff_prio(const RtThread *o) {
    int p = o->prio;
    for (const RtThread *w = g_wait; w; w = w->next)
        if (w->wkind == W_LOCK && w->prio < p && w->waddr &&
            (__atomic_load_n((const uint32_t *)(uintptr_t)w->waddr, __ATOMIC_RELAXED) & ~WAITERS_BIT) == o->handle)
            p = w->prio;
    return p;
}

// The priority the virtual cores go by.
static int prio_of(const RtThread *t) { return g_ap > 0 ? eff_prio(t) : t->prio; }

// TLR+0x100 disable_count, +0x102 interrupt_flag (the flag is set from outside, as the kernel does).
static volatile uint16_t *ap_tlr(const RtThread *t) { return (volatile uint16_t *)(t->tlr + 0x100); }
static int ap_disabled(const RtThread *t) { return g_ap > 0 && ap_tlr(t)[0]; }

// kern_ap_stub: entered in place of the interrupted instruction of top-level game code. It keeps every register the game code
// may have live (volatile GPRs, flags, YMM0-15, MXCSR), calls kern_ap_yield and jumps back to kern_ap_rip(), stored first in
// the return slot so a stack walk sees the interrupted function as its caller. No ret: call/return pairs stay balanced.
#ifdef _WIN32
__asm__(".text\n"
        ".globl kern_ap_stub\n"
        ".def kern_ap_stub; .scl 2; .type 32; .endef\n"
        ".p2align 4\n"
        "kern_ap_stub:\n"
        ".seh_proc kern_ap_stub\n"
        "    leaq -8(%rsp), %rsp\n"   // not sub: the flags are still the interrupted code's until pushfq
        ".seh_stackalloc 8\n"
        "    pushfq\n"
        ".seh_stackalloc 8\n"
        "    pushq %rax\n"
        ".seh_pushreg %rax\n"
        "    pushq %rcx\n"
        ".seh_pushreg %rcx\n"
        "    pushq %rdx\n"
        ".seh_pushreg %rdx\n"
        "    pushq %r8\n"
        ".seh_pushreg %r8\n"
        "    pushq %r9\n"
        ".seh_pushreg %r9\n"
        "    pushq %r10\n"
        ".seh_pushreg %r10\n"
        "    pushq %r11\n"
        ".seh_pushreg %r11\n"
        "    pushq %rbp\n"
        ".seh_pushreg %rbp\n"
        "    movq %rsp, %rbp\n"
        ".seh_setframe %rbp, 0\n"
        ".seh_endprologue\n"
        "    andq $-32, %rsp\n"
        "    subq $576, %rsp\n"   // 0: shadow space, 32: MXCSR saved, 36: MXCSR default, 64: YMM0-15
        "    vmovdqa %ymm0, 64(%rsp)\n"
        "    vmovdqa %ymm1, 96(%rsp)\n"
        "    vmovdqa %ymm2, 128(%rsp)\n"
        "    vmovdqa %ymm3, 160(%rsp)\n"
        "    vmovdqa %ymm4, 192(%rsp)\n"
        "    vmovdqa %ymm5, 224(%rsp)\n"
        "    vmovdqa %ymm6, 256(%rsp)\n"
        "    vmovdqa %ymm7, 288(%rsp)\n"
        "    vmovdqa %ymm8, 320(%rsp)\n"
        "    vmovdqa %ymm9, 352(%rsp)\n"
        "    vmovdqa %ymm10, 384(%rsp)\n"
        "    vmovdqa %ymm11, 416(%rsp)\n"
        "    vmovdqa %ymm12, 448(%rsp)\n"
        "    vmovdqa %ymm13, 480(%rsp)\n"
        "    vmovdqa %ymm14, 512(%rsp)\n"
        "    vmovdqa %ymm15, 544(%rsp)\n"
        "    stmxcsr 32(%rsp)\n"
        "    movl $0x1f80, 36(%rsp)\n"
        "    ldmxcsr 36(%rsp)\n"
        "    cld\n"
        "    vzeroupper\n"
        "    call kern_ap_rip\n"
        "    movq %rax, 72(%rbp)\n"   // the return slot: rbp + 8 pushes + pushfq
        "    call kern_ap_yield\n"
        "    ldmxcsr 32(%rsp)\n"
        "    vmovdqa 64(%rsp), %ymm0\n"
        "    vmovdqa 96(%rsp), %ymm1\n"
        "    vmovdqa 128(%rsp), %ymm2\n"
        "    vmovdqa 160(%rsp), %ymm3\n"
        "    vmovdqa 192(%rsp), %ymm4\n"
        "    vmovdqa 224(%rsp), %ymm5\n"
        "    vmovdqa 256(%rsp), %ymm6\n"
        "    vmovdqa 288(%rsp), %ymm7\n"
        "    vmovdqa 320(%rsp), %ymm8\n"
        "    vmovdqa 352(%rsp), %ymm9\n"
        "    vmovdqa 384(%rsp), %ymm10\n"
        "    vmovdqa 416(%rsp), %ymm11\n"
        "    vmovdqa 448(%rsp), %ymm12\n"
        "    vmovdqa 480(%rsp), %ymm13\n"
        "    vmovdqa 512(%rsp), %ymm14\n"
        "    vmovdqa 544(%rsp), %ymm15\n"
        "    movq %rbp, %rsp\n"
        "    popq %rbp\n"
        "    popq %r11\n"
        "    popq %r10\n"
        "    popq %r9\n"
        "    popq %r8\n"
        "    popq %rdx\n"
        "    popq %rcx\n"
        "    popq %rax\n"
        "    popfq\n"
        "    leaq 8(%rsp), %rsp\n"
        "    jmpq *-8(%rsp)\n"
        ".seh_endproc\n");
#else
// The System V version: the interrupted code may keep data in the 128 bytes below its stack pointer (the red zone), so the
// return slot and the saves go below that (slot at entry - 136); the exit restores rsp and jumps through the slot.
__asm__(".text\n"
        ".globl kern_ap_stub\n"
        ".type kern_ap_stub, @function\n"
        ".p2align 4\n"
        "kern_ap_stub:\n"
        "    .cfi_startproc\n"
        "    leaq -136(%rsp), %rsp\n"   // not sub: the flags are still the interrupted code's until pushfq
        "    pushfq\n"
        "    pushq %rax\n"
        "    pushq %rcx\n"
        "    pushq %rdx\n"
        "    pushq %rsi\n"
        "    pushq %rdi\n"
        "    pushq %r8\n"
        "    pushq %r9\n"
        "    pushq %r10\n"
        "    pushq %r11\n"
        "    pushq %rbp\n"
        "    movq %rsp, %rbp\n"
        "    andq $-32, %rsp\n"
        "    subq $576, %rsp\n"   // 32: MXCSR saved, 36: MXCSR default, 64: YMM0-15
        "    vmovdqa %ymm0, 64(%rsp)\n"
        "    vmovdqa %ymm1, 96(%rsp)\n"
        "    vmovdqa %ymm2, 128(%rsp)\n"
        "    vmovdqa %ymm3, 160(%rsp)\n"
        "    vmovdqa %ymm4, 192(%rsp)\n"
        "    vmovdqa %ymm5, 224(%rsp)\n"
        "    vmovdqa %ymm6, 256(%rsp)\n"
        "    vmovdqa %ymm7, 288(%rsp)\n"
        "    vmovdqa %ymm8, 320(%rsp)\n"
        "    vmovdqa %ymm9, 352(%rsp)\n"
        "    vmovdqa %ymm10, 384(%rsp)\n"
        "    vmovdqa %ymm11, 416(%rsp)\n"
        "    vmovdqa %ymm12, 448(%rsp)\n"
        "    vmovdqa %ymm13, 480(%rsp)\n"
        "    vmovdqa %ymm14, 512(%rsp)\n"
        "    vmovdqa %ymm15, 544(%rsp)\n"
        "    stmxcsr 32(%rsp)\n"
        "    movl $0x1f80, 36(%rsp)\n"
        "    ldmxcsr 36(%rsp)\n"
        "    cld\n"
        "    vzeroupper\n"
        "    call kern_ap_rip\n"
        "    movq %rax, 88(%rbp)\n"   // the return slot: rbp + 10 pushes + pushfq
        "    call kern_ap_yield\n"
        "    ldmxcsr 32(%rsp)\n"
        "    vmovdqa 64(%rsp), %ymm0\n"
        "    vmovdqa 96(%rsp), %ymm1\n"
        "    vmovdqa 128(%rsp), %ymm2\n"
        "    vmovdqa 160(%rsp), %ymm3\n"
        "    vmovdqa 192(%rsp), %ymm4\n"
        "    vmovdqa 224(%rsp), %ymm5\n"
        "    vmovdqa 256(%rsp), %ymm6\n"
        "    vmovdqa 288(%rsp), %ymm7\n"
        "    vmovdqa 320(%rsp), %ymm8\n"
        "    vmovdqa 352(%rsp), %ymm9\n"
        "    vmovdqa 384(%rsp), %ymm10\n"
        "    vmovdqa 416(%rsp), %ymm11\n"
        "    vmovdqa 448(%rsp), %ymm12\n"
        "    vmovdqa 480(%rsp), %ymm13\n"
        "    vmovdqa 512(%rsp), %ymm14\n"
        "    vmovdqa 544(%rsp), %ymm15\n"
        "    movq %rbp, %rsp\n"
        "    popq %rbp\n"
        "    popq %r11\n"
        "    popq %r10\n"
        "    popq %r9\n"
        "    popq %r8\n"
        "    popq %rdi\n"
        "    popq %rsi\n"
        "    popq %rdx\n"
        "    popq %rcx\n"
        "    popq %rax\n"
        "    popfq\n"
        "    leaq 136(%rsp), %rsp\n"
        "    jmpq *-136(%rsp)\n"
        "    .cfi_endproc\n"
        ".size kern_ap_stub, .-kern_ap_stub\n");
#endif

enum { AP_NONE, AP_RETRY, AP_SENT };

// The waiter w interrupts its target owner if that runs top-level game code (holding G; nothing here takes a lock while it is stopped).
static int ap_try(RtThread *w) {
    RtThread *o = w->target;
    if (!o || o->cur_core < 0 || !o->preempt) return AP_NONE;
    if (!o->hthr || o->ap_pending) return AP_RETRY;
    volatile uint16_t *tl = ap_tlr(o);
    if (__atomic_load_n(&o->host_depth, __ATOMIC_RELAXED) > 0) {
        g_ap_host++;
        return AP_RETRY;
    }
    if (tl[0]) {   // preemption disabled: SVC 0x36 switches
        tl[1] = 1;
        g_ap_deferred++;
        return AP_RETRY;
    }
    uint64_t now = kern_now_ns();
    if (o->ap_last && now - o->ap_last < AP_GAP_NS) return AP_RETRY;
    o->ap_last = now;
    uint64_t rip, rsp;
    if (!plat_intr_stop(o->hthr, &rip, &rsp)) {
        g_ap_fail++;
        return AP_RETRY;
    }
    int r = AP_RETRY;
    uint64_t to = 0;
    if (rip < g_gen_lo || rip > g_gen_hi)
        g_ap_pc++;
    else if (__atomic_load_n(&o->host_depth, __ATOMIC_RELAXED) > 0)
        g_ap_host++;
    else if (tl[0]) {
        tl[1] = 1;
        g_ap_deferred++;
    } else {
        o->ap_rip = rip;
        o->ap_pending = 1;
        to = (uint64_t)(uintptr_t)kern_ap_stub;
    }
    if (!plat_intr_resume(o->hthr, to)) {
        o->ap_pending = 0;
        g_ap_fail++;
    } else if (to) {
        g_ap_sent++;
        r = AP_SENT;
    }
    if (r == AP_SENT && g_ap_log < AP_LOG_N) {
        g_ap_log++;
        rt_log("kernel: interrupt %#x p%d at exe+%#llx rsp %#llx for %#x p%d", o->handle, o->prio,
               (unsigned long long)(o->ap_rip - plat_exe_base()), (unsigned long long)rsp, w->handle, w->prio);
    }
    return r;
}

int kern_host_track;
static uint64_t g_sync36;

// perf_present, every present: the asynchronous preemption counters every 600 frames.
void kern_core_diag_frame(uint64_t frame, uint64_t interval_ns) {
    if (g_ap > 0 && frame && frame % 600 == 0) {  // asynchronous preemption counters since the start
        lock();
        uint64_t v[8] = {g_ap_sent, g_ap_taken, g_ap_host, g_ap_pc, g_ap_deferred, g_ap_fail, g_ap_spin_ok, g_sync36};
        unlock();
        rt_log("kernel: frame %llu: asynchronous preemption: %llu interrupted, %llu cores given up there; refused: "
               "%llu in a host call, %llu outside game code, %llu preemption disabled, %llu failed; %llu waits ended "
               "in the first spin; SVC 0x36 %llu",
               (unsigned long long)frame, (unsigned long long)v[0], (unsigned long long)v[1], (unsigned long long)v[2],
               (unsigned long long)v[3], (unsigned long long)v[4], (unsigned long long)v[5], (unsigned long long)v[6],
               (unsigned long long)v[7]);
    }
}

static void core_grant(int c) {
    RtThread *b = NULL, **bp = NULL;
    int bpri = 0;
    for (RtThread **p = &g_core_wait; *p; p = &(*p)->cnext) {
        if (!core_allowed(*p, c)) continue;
        int pr = prio_of(*p);
        if (!b || pr < bpri || (pr == bpri && (*p)->cseq < b->cseq)) {
            b = *p;
            bp = p;
            bpri = pr;
        }
    }
    if (!b) return;
    *bp = b->cnext;
    b->cnext = NULL;
    b->cur_core = c;
    g_core_owner[c] = b;
    if (rt_trace & RT_TRACE_SCHED) b->grant_at = kern_now_ns();
    plat_cond_signal(&b->cv);
}

static void core_put(RtThread *t) {
    if (t->cur_core < 0) return;
    int c = t->cur_core;
    if (t->preempt && t->pre_since) {   // sched trace: time from the preempt request to this release
        uint64_t d = kern_now_ns() - t->pre_since;
        g_pre_n++;
        g_pre_ns += d;
        if (d > g_pre_max) g_pre_max = d;
        t->npre++;
    }
    t->pre_since = 0;
    t->cur_core = -1;
    t->preempt = 0;
    g_core_owner[c] = NULL;
    core_grant(c);
}

// Whether a waiter better than t may run on core c.
static int core_better_waiter(const RtThread *t, int c) {
    int tp = prio_of(t);
    for (const RtThread *w = g_core_wait; w; w = w->cnext)
        if (core_allowed(w, c) && prio_of(w) < tp) return 1;
    return 0;
}

// Asks holders to give their cores up, one holder per waiter: waiters best first, each claims the worst holder worse than
// itself on a core it may run on that no better waiter claimed (a pending request is reused).
static void core_kick(void) {
    RtThread *w[64];
    int wp[64], op[NCORE], n = 0, claimed = 0;
    for (int i = 0; i < NCORE; i++) op[i] = g_core_owner[i] ? prio_of(g_core_owner[i]) : 0;
    for (RtThread *p = g_core_wait; p && n < 64; p = p->cnext) {
        int i = n++, pp = prio_of(p);
        for (; i > 0 && (wp[i - 1] > pp || (wp[i - 1] == pp && w[i - 1]->cseq > p->cseq)); i--) {
            w[i] = w[i - 1];
            wp[i] = wp[i - 1];
        }
        w[i] = p;
        wp[i] = pp;
    }
    for (int k = 0; k < n && claimed != (1 << NCORE) - 1; k++) {
        int v = -1;
        for (int i = 0; i < NCORE; i++) {
            RtThread *o = g_core_owner[i];
            if ((claimed >> i & 1) || !core_allowed(w[k], i) || !o || op[i] <= wp[k]) continue;
            RtThread *b = v < 0 ? NULL : g_core_owner[v];
            if (!b || op[i] > op[v] || (op[i] == op[v] && o->preempt && !b->preempt)) v = i;
        }
        if (v < 0) continue;
        claimed |= 1 << v;
        RtThread *o = g_core_owner[v];
        w[k]->target = o;
        if (o->preempt) continue;
        if (rt_trace & RT_TRACE_SCHED) {
            o->pre_since = kern_now_ns();
            g_pre_req++;
        }
        __atomic_store_n(&o->preempt, 1, __ATOMIC_RELEASE);
    }
}

static void core_take(RtThread *t) {
    if (t->cur_core >= 0) return;
    cores_enabled();
    if (!(t->mask & ((1u << NCORE) - 1)))
        rt_fatal("kernel: thread %#x core mask %#llx has no application core", t->handle, (unsigned long long)t->mask);
    int c = t->core >= 0 && t->core < NCORE && core_allowed(t, t->core) && !g_core_owner[t->core] ? t->core : -1;
    for (int i = 0; c < 0 && i < NCORE; i++)
        if (core_allowed(t, i) && !g_core_owner[i]) c = i;
    if (c >= 0) {
        t->cur_core = c;
        g_core_owner[c] = t;
        acct(t, S_RUN);
        return;
    }
    acct(t, S_CORE);
    t->cseq = ++g_wseq;
    t->cnext = g_core_wait;
    g_core_wait = t;
    t->target = NULL;
    core_kick();
    if (t->target && t->cur_core < 0) {   // most waits end within microseconds: spin before interrupting
        uint64_t end = kern_now_ns() + AP_FIRST_NS;
        unlock();
        while (__atomic_load_n(&t->cur_core, __ATOMIC_RELAXED) < 0 && kern_now_ns() < end) plat_pause();
        lock();
        if (t->cur_core >= 0) g_ap_spin_ok++;
    }
    while (t->cur_core < 0) {
        uint32_t ms = 0;   // 0: until signalled
        if (ap_try(t) != AP_NONE) ms = 1;   // the owner keeps its core: again in a millisecond
        if (ms) plat_cond_wait_ms(&t->cv, &G, ms);
        else plat_cond_wait(&t->cv, &G);
    }
    if (t->grant_at) {
        uint64_t d = kern_now_ns() - t->grant_at;
        t->lg_ns += d;
        t->lg_n++;
        if (d > t->lg_max) t->lg_max = d;
        t->grant_at = 0;
    }
    t->target = NULL;
    acct(t, S_RUN);
}

static uint32_t block(RtThread *t, int kind, int64_t ns) {
    t->waker = 0;
    if (ns == 0) return block_inner(t, kind, ns);
    core_put(t);
    acct(t, st_of(kind));
    t->nblk++;
    uint32_t r = block_inner(t, kind, ns);
    core_take(t);
    return r;
}

int kern_core_away(void) {
    RtThread *t = tcur;
    if (!t || t->cur_core < 0) return 0;
    lock();
    core_put(t);
    acct(t, S_HOST);
    if (rt_trace & RT_TRACE_SCHED)   // the caller as an exe RVA (oot.map)
        t->wobj = (uintptr_t)__builtin_return_address(0) - plat_exe_base();
    unlock();
    return 1;
}

void kern_core_back(int *away) {
    RtThread *t = tcur;
    if (!*away || !t) return;
    lock();
    core_take(t);
    unlock();
    *away = 0;
}

// ---- spin waits (recomp/patches.py `spin` lines): a pass of a guest loop that polls memory. See README, "Spin waits".
#define SPIN_GAP_NS 20000ull
static const uint64_t g_spin_handoff_ns = 50 * 1000;
static _Thread_local uint64_t sw_head, sw_t0, sw_last;

void rt_spin_gate(Cpu *c, uint64_t head) {
    (void)c;
    plat_pause();
    RtThread *t = tcur;
    if (!t) return;
    t->sp_n++;
    t->sp_site = head;
    uint64_t now = kern_now_ns();
    if (head != sw_head || now - sw_last > SPIN_GAP_NS) {   // a new wait
        sw_head = head;
        sw_t0 = now;
    }
    sw_last = now;
    if (t->cur_core < 0 || now - sw_t0 < g_spin_handoff_ns ||
        !__atomic_load_n(&g_core_wait, __ATOMIC_RELAXED))   // a racy look: a waiter missed now is seen at the next pass
        return;
    t->sp_ho++;
    int away = kern_core_away();
    plat_yield();
    kern_core_back(&away);
    sw_last = kern_now_ns();
}

// A pending preempt request, at an SVC or an indirect call: the core goes over only if a better thread still waits for it.
void kern_checkpoint(void) {
    RtThread *t = tcur;
    if (!t || !__atomic_load_n(&t->preempt, __ATOMIC_RELAXED)) return;
    lock();
    int better = t->cur_core >= 0 && core_better_waiter(t, t->cur_core);
    if (better && ap_disabled(t)) {   // preemption disabled: the switch comes at SVC 0x36
        ap_tlr(t)[1] = 1;
        t->preempt = 0;
        g_ap_deferred++;
    } else if (better) {
        core_put(t);   // clears preempt (after the sched trace reads it)
        core_take(t);
    } else {
        t->preempt = 0;
        t->pre_since = 0;
        g_pre_stale++;
    }
    unlock();
}

// kern_ap_stub, on the interrupted owner: where to jump back to, then the core given up as at a checkpoint.
uint64_t kern_ap_rip(void) { return tcur->ap_rip; }

void kern_ap_yield(void) {
    RtThread *t = tcur;
    lock();
    t->ap_pending = 0;
    if (t->preempt && t->cur_core >= 0 && !ap_disabled(t) && core_better_waiter(t, t->cur_core)) {
        g_ap_taken++;
        core_put(t);
        core_take(t);
    }
    unlock();
}

// SVC 0x36 SynchronizePreemptionState: a preemption deferred by disable_count happens here.
static void ap_sync(RtThread *t) {
    lock();
    if (g_ap > 0 && t->cur_core >= 0 && core_better_waiter(t, t->cur_core)) {
        core_put(t);
        core_take(t);
    }
    unlock();
}


// OOT_TRACE=sched: every 120 presents, the time split of the busiest threads and of the render thread, the longest waits of
// the render and main thread by object and waker, and the preemption latency.
#define REP_N 8
void kern_sched_report(uint64_t frame) {
    static uint64_t last;
    typedef struct {
        uint32_t h;
        int prio, render;
        uint64_t mask, entry, ns[S_N], nblk, npre, cpu, lw_ns, lw_n, lw_max, lg_ns, lg_n, lg_max, sl_n, sl_req, sl_act;
        uint64_t sp_n, sp_ho, sp_site;
    } Row;
    Row rows[REP_N + 1];
    WaitRec rw[2][4] = {0};   // the render and the main thread's longest waits
    int nr = 0;
    RtThread *self = tcur;
    lock();
    if (g_render != self) {
        g_render = self;
        memset(g_rw, 0, sizeof g_rw);
    }
    if (frame % 120) {
        unlock();
        return;
    }
    uint64_t now = kern_now_ns(), span = last ? now - last : 0;
    last = now;
    RtThread *top[REP_N + 1] = {0};
    uint64_t busy[REP_N + 1] = {0}, all_run = 0, all_core = 0;
    for (RtThread *t = g_all; t; t = t->all) {
        acct(t, t->st);   // close the open interval
        uint64_t b = t->st_ns[S_RUN] + t->st_ns[S_CORE] + t->st_ns[S_HOST];
        all_run += t->st_ns[S_RUN];
        all_core += t->st_ns[S_CORE];
        if (t == self) continue;
        for (int i = 0; i < REP_N; i++)
            if (!top[i] || b > busy[i]) {
                memmove(top + i + 1, top + i, (REP_N - 1 - i) * sizeof *top);
                memmove(busy + i + 1, busy + i, (REP_N - 1 - i) * sizeof *busy);
                top[i] = t;
                busy[i] = b;
                break;
            }
    }
    if (self) {   // the render thread first
        memmove(top + 1, top, REP_N * sizeof *top);
        top[0] = self;
    }
    for (int i = 0; i <= REP_N && top[i]; i++) {
        RtThread *t = top[i];
        Row *r = &rows[nr++];
        r->h = t->handle;
        r->prio = t->prio;
        r->render = t == self;
        r->mask = t->mask;
        r->entry = t->entry;
        memcpy(r->ns, t->st_ns, sizeof r->ns);
        r->nblk = t->nblk;
        r->npre = t->npre;
        r->lw_ns = t->lw_ns, r->lw_n = t->lw_n, r->lw_max = t->lw_max;
        r->lg_ns = t->lg_ns, r->lg_n = t->lg_n, r->lg_max = t->lg_max;
        r->sl_n = t->sl_n, r->sl_req = t->sl_req, r->sl_act = t->sl_act;
        r->sp_n = t->sp_n, r->sp_ho = t->sp_ho, r->sp_site = t->sp_site;
        r->cpu = 0;
        uint64_t c = t->tid ? plat_thread_cpu_ns(t->tid) : 0;
        if (c) {
            r->cpu = t->cpu_last ? c - t->cpu_last : 0;
            t->cpu_last = c;
        }
    }
    for (int k = 0; k < 2; k++) {
        const WaitRec *src = k ? g_mw : g_rw;
        for (int i = 0; i < RW_N && src[i].ns; i++)
            for (int j = 0; j < 4; j++)
                if (src[i].ns > rw[k][j].ns) {
                    memmove(rw[k] + j + 1, rw[k] + j, (3 - j) * sizeof **rw);
                    rw[k][j] = src[i];
                    break;
                }
    }
    uint64_t pre_stale = g_pre_stale, pre_req = g_pre_req, pre_n = g_pre_n, pre_ns = g_pre_ns, pre_max = g_pre_max;
    uint64_t sc = g_sync_calls, sb = g_sync_blocks;
    for (RtThread *t = g_all; t; t = t->all) {
        memset(t->st_ns, 0, sizeof t->st_ns);
        t->nblk = t->npre = 0;
        t->lw_ns = t->lw_n = t->lw_max = t->lg_ns = t->lg_n = t->lg_max = t->sl_n = t->sl_req = t->sl_act = 0;
        t->sp_n = t->sp_ho = 0;   // racy against the thread's own increments: a few passes may be lost
    }
    memset(g_rw, 0, sizeof g_rw);
    memset(g_mw, 0, sizeof g_mw);
    g_pre_req = g_pre_n = g_pre_ns = g_pre_max = g_pre_stale = 0;
    g_sync_calls = g_sync_blocks = 0;
    unlock();
    if (!span) return;
    rt_log("sched: frame %llu: %.2f s, cores busy %.2f of %d, core wait %.2f thread-s, preempt req %llu released %llu "
           "avg %.2f ms max %.2f ms, WaitSynchronization %llu calls %llu blocks, stale preempt %llu%s",
           (unsigned long long)frame, span / 1e9, (double)all_run / span, NCORE, all_core / 1e9,
           (unsigned long long)pre_req, (unsigned long long)pre_n, pre_n ? pre_ns / 1e6 / pre_n : 0.0, pre_max / 1e6,
           (unsigned long long)sc, (unsigned long long)sb, (unsigned long long)pre_stale, self ? "" : " (render thread is not a guest thread)");
    for (int i = 0; i < nr; i++) {
        Row *r = &rows[i];
        char d[128], s[256];
        int o = 0;
        for (int k = S_RUN; k < S_N; k++)
            o += snprintf(s + o, sizeof s - o, " %s %.0f%%", s_name[k], 100.0 * r->ns[k] / span);
        ldr_describe(r->entry, d, sizeof d);
        rt_log("sched: %s%#x p%d m%llx%s cpu %.0f%% blocks %llu preempted %llu (%s)", r->render ? "render " : "",
               r->h, r->prio, (unsigned long long)r->mask, s, 100.0 * r->cpu / span, (unsigned long long)r->nblk,
               (unsigned long long)r->npre, r->entry ? d : "main");
        rt_log("sched:   %#x wake to run avg %.1f us max %.0f us (%llu), core grant to run avg %.1f us max %.0f us "
               "(%llu), sleep %llu asked avg %.1f us took avg %.1f us", r->h, r->lw_n ? r->lw_ns / 1e3 / r->lw_n : 0.0,
               r->lw_max / 1e3, (unsigned long long)r->lw_n, r->lg_n ? r->lg_ns / 1e3 / r->lg_n : 0.0, r->lg_max / 1e3,
               (unsigned long long)r->lg_n, (unsigned long long)r->sl_n, r->sl_n ? r->sl_req / 1e3 / r->sl_n : 0.0,
               r->sl_n ? r->sl_act / 1e3 / r->sl_n : 0.0);
        if (r->sp_n) {
            ldr_describe(r->sp_site, d, sizeof d);
            rt_log("sched:   %#x spin wait passes %llu, core handoffs %llu, last at %s", r->h,
                   (unsigned long long)r->sp_n, (unsigned long long)r->sp_ho, d);
        }
    }
    for (int k = 0; k < 2; k++)
        for (int j = 0; j < 4 && rw[k][j].ns; j++) {
            char by[32] = "";
            if (rw[k][j].waker) snprintf(by, sizeof by, ", woken by %#x", rw[k][j].waker);
            rt_log("sched: %s wait %d: %s on %s%#llx%s: %.1f%%", k ? "main" : "render", j + 1, s_name[rw[k][j].st],
                   rw[k][j].st == S_HOST ? "exe+" : "", (unsigned long long)rw[k][j].obj, by, 100.0 * rw[k][j].ns / span);
        }
}

// ---- locks: tag word = owner handle | WAITERS_BIT
static inline uint32_t ld32(uint64_t a) { return __atomic_load_n((uint32_t *)(uintptr_t)a, __ATOMIC_SEQ_CST); }
static inline void st32(uint64_t a, uint32_t v) { __atomic_store_n((uint32_t *)(uintptr_t)a, v, __ATOMIC_SEQ_CST); }
static inline int cas32(uint64_t a, uint32_t *exp, uint32_t v) {
    return __atomic_compare_exchange_n((uint32_t *)(uintptr_t)a, exp, v, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}

// Release the lock at addr: hand it to the best waiter, or clear it.
static void lock_handoff(uint64_t addr) {
    RtThread *w = pick(W_LOCK, addr);
    int more = 0;
    if (!w) {
        st32(addr, 0);
        return;
    }
    for (RtThread *t = g_wait; t; t = t->next)
        if (t != w && t->wkind == W_LOCK && t->waddr == addr) {
            more = 1;
            break;
        }
    st32(addr, w->wtag | (more ? WAITERS_BIT : 0));
    wake(w, 0);
}

// A condvar waiter takes its lock: directly if free (and wakes), else it becomes a lock waiter behind the others.
static void lock_acquire_for(RtThread *t, uint64_t addr, uint32_t tag) {
    uint32_t v = ld32(addr);
    for (;;) {
        if (v == 0) {
            if (cas32(addr, &v, tag)) {
                wake(t, 0);
                return;
            }
            continue;
        }
        if (cas32(addr, &v, v | WAITERS_BIT)) break;
    }
    acct(t, S_LOCK);
    t->wobj = v & ~WAITERS_BIT;
    t->wkind = W_LOCK;
    t->waddr = addr;
    t->wtag = tag;
    t->wseq = ++g_wseq;
}

// ---- threads
static RtThread *thr_new(int prio, int core, uint64_t mask) {
    RtThread *t = calloc(1, sizeof *t);
    Cpu *c = plat_aligned_alloc(sizeof(Cpu), 64);
    if (!t || !c) rt_fatal("kernel: thread allocation");
    memset(c, 0, sizeof *c);
    plat_cond_init(&t->cv);
    t->prio = prio;
    t->core = core;
    t->mask = mask;
    t->cur_core = -1;
    // TLR (0x200) and the "current thread object" the SDK reads through TLR+0x1F8, in one guest-visible block.
    t->tlr = mem_kernel_alloc(TLR_SIZE + THREAD_OBJ_SIZE, TLR_SIZE);
    *(uint64_t *)(t->tlr + 0x1F8) = (uint64_t)(uintptr_t)(t->tlr + TLR_SIZE);
    t->tls = mem_kernel_alloc(TLS_SIZE, 16);
    c->tpidrro = (uint64_t)(uintptr_t)t->tlr;
    c->tpidr = (uint64_t)(uintptr_t)t->tls;
    c->excl_addr = ~0ull;
    c->thread = t;
    t->cpu = t->cur = c;
    return t;
}

static void thr_link(RtThread *t) {   // holding G
    t->handle = h_new(O_THREAD, t);
    // The SDK takes the caller's own thread handle from TLR+0x110 as the owner tag of its mutexes; a zero there equals the
    // word of every free mutex, so every mutex would look held.
    *(uint32_t *)(t->tlr + 0x110) = t->handle;
    t->id = g_next_id++;
    t->all = g_all;
    g_all = t;
    g_created++;
}

static void thr_free(RtThread *t) {   // holding G; t has exited and its handle is closed
    for (RtThread **p = &g_all; *p; p = &(*p)->all)
        if (*p == t) {
            *p = t->all;
            break;
        }
    mem_kernel_free(t->tlr, TLR_SIZE + THREAD_OBJ_SIZE);
    mem_kernel_free(t->tls, TLS_SIZE);
    plat_aligned_free(t->cpu);
    free(t);
}

// The end of every guest thread, on its own host context after the unwind: give the core up, signal the thread
// object, free it if its handle is already closed. t is not touched after the unlock.
static void thread_finish(RtThread *t) {
    if (t->base_fiber) plat_fiber_unself();
    t->base_fiber = NULL;
    tcur = NULL;
    lock();
    core_put(t);
    acct(t, S_OFF);
    // Core waiters that core_kick pointed at t: t may be freed once its handle closes, and ap_try would then read freed
    // memory (o->cur_core, o->hthr) and could reach pthread_kill on it. t owns no core now, so none can pick it again.
    for (RtThread *w = g_core_wait; w; w = w->cnext)
        if (w->target == t) w->target = NULL;
    if (t->hthr) plat_intr_free(t->hthr);   // no waiter interrupts a thread without a core
    t->hthr = NULL;
    t->exited = 1;
    wake_sync_waiters();
    if (t->closed) thr_free(t);
    unlock();
}

// ExitThread: back to the thread start with longjmp. On Windows the default MinGW setjmp records the stack frame the CRT
// validates (SetContextIpValidation). __builtin_setjmp is not usable: Win64 LLVM saves a frame address that is not rbp.
static __attribute__((noreturn)) void exit_now(RtThread *t) { longjmp(t->exit_jmp, 1); }

// The calling host thread, for ap_try.
static PlatIntr *ap_open_self(void) {
    cores_enabled();
    PlatIntr *h = plat_intr_self();
    if (!h) rt_log("kernel: no interruption handle for this thread: it is not preempted asynchronously");
    return h;
}

static void thread_main(void *p) {
    RtThread *t = p;
    tcur = t;
    t->tid = plat_thread_id();
    PlatIntr *self = ap_open_self();
    lock();
    t->hthr = self;
    while (!t->started) plat_cond_wait(&t->cv, &G);
    core_take(t);
    unlock();
    Cpu *c = t->cpu;
    c->sp = t->stack_top & ~15ull;
    if (!setjmp(t->exit_jmp)) {
#ifdef _WIN32   // the CRT needs this frame to validate longjmp when SetContextIpValidation is enabled
        if (!((_JUMP_BUFFER *)t->exit_jmp)->Frame)
            rt_fatal("kernel: setjmp recorded no frame; ExitThread would fail fast");
#endif
        rt_call(c, t->entry, 1, t->arg);
        char d[128];
        rt_fatal("guest thread %#x: entry %s returned (Horizon threads end with ExitThread)", t->handle,
                 ldr_describe(t->entry, d, sizeof d));
    }
    thread_finish(tcur);
    return;
}

uint64_t kern_threads_created(void) {
    lock();
    uint64_t n = g_created;
    unlock();
    return n;
}

Cpu *kern_cpu(void) { return tcur ? tcur->cur : NULL; }

uint32_t kern_main_thread(Cpu **cpu) {
    if (tcur) {
        if (!tcur->is_main) rt_fatal("kern_main_thread on guest thread %#x, not the main thread", tcur->handle);
        *cpu = tcur->cpu;
        return tcur->handle;
    }
    if (g_main) rt_fatal("kern_main_thread: the main thread already runs on another host thread");
    const RtGuestLayout *L = mem_layout();
    uint64_t stack = L->stack.base + L->stack.size - MAIN_STACK;
    if (!plat_mem_commit(stack, MAIN_STACK))
        rt_fatal("kernel: commit the main thread stack at %#llx failed", (unsigned long long)stack);
    RtThread *t = thr_new(MAIN_PRIO, MAIN_CORE, 7);
    t->started = 1;
    t->is_main = 1;
    t->stack_top = stack + MAIN_STACK;
    t->cpu->sp = t->stack_top;
    t->tid = plat_thread_id();
    perf_main_thread();
    PlatIntr *self = ap_open_self();
    lock();
    t->hthr = self;
    thr_link(t);
    g_main = t;
    g_main_stack = stack;
    core_take(t);
    unlock();
    tcur = t;
    *cpu = t->cpu;
    return t->handle;
}

void kern_main_stack(uint64_t *base, uint64_t *size) {
    uint64_t b = __atomic_load_n(&g_main_stack, __ATOMIC_ACQUIRE);
    *base = b;
    *size = b ? MAIN_STACK : 0;
}

static void guest_main(void *p) {
    void (*fn)(Cpu *) = (void (*)(Cpu *))p;
    Cpu *c;
    kern_main_thread(&c);
    if (!setjmp(tcur->exit_jmp)) {
#ifdef _WIN32   // the CRT needs this frame to validate longjmp when SetContextIpValidation is enabled
        if (!((_JUMP_BUFFER *)tcur->exit_jmp)->Frame)
            rt_fatal("kernel: setjmp recorded no frame; ExitThread would fail fast");
#endif
        fn(c);
    }
    thread_finish(tcur);
    return;
}

void kern_run_on_guest_main(void (*fn)(Cpu *c)) {
    PlatThread *h = plat_thread_new_ex(guest_main, (void *)fn, "guest main", HOST_STACK);
    if (!h) rt_fatal("kernel: cannot start the guest main thread");
    plat_thread_join(h);
}

// Sleeps go to the host timer, short ones too: job workers poll with SleepThread(20 us), and spun each sleep would cost its
// full length in CPU. On the host timer one wakes up to PLAT_WAIT_LATE_NS late.
static void sleep_ns(RtThread *t, int64_t ns) {
    (void)t;
    plat_wait_ns((uint64_t)ns);
}

// ---- synchronization objects
static int sig_obj(uint32_t h, int *bad) {   // 1 = signaled (holding G)
    unsigned i = h - 0x1000;
    if (h == H_CUR_THREAD) return 0;
    if (i == 0 || i >= MAX_H) {
        *bad = 1;
        return 0;
    }
    switch (g_h[i].type) {
    case O_THREAD: return ((RtThread *)g_h[i].obj)->exited;
    case O_EVENT_R: return ((Event *)g_h[i].obj)->signaled;
    default: *bad = 1; return 0;
    }
}

// CloseHandle: an event goes with its last end; a thread object once it has also exited.
static uint32_t handle_close(uint32_t h) {
    unsigned i = h - 0x1000;
    int ok = 0, kind = 0;
    void *ipc = NULL;
    lock();
    if (i > 0 && i < MAX_H) {
        int type = g_h[i].type;
        void *o = g_h[i].obj;
        if (type == O_EVENT_R || type == O_EVENT_W) {
            Event *e = o;
            if (--e->refs == 0) free(e);
            ok = 1;
        } else if (type == O_SESSION || type == O_TMEM) {
            kind = type;
            ipc = o;
            ok = 1;
        } else if (type == O_THREAD) {
            RtThread *t = o;
            t->closed = 1;
            if (t->exited) thr_free(t);
            ok = 1;
        }
        if (ok) {
            g_h[i].type = O_FREE;
            g_h[i].obj = NULL;
        }
    }
    unlock();
    if (ipc) ipc_handle_closed(kind, ipc);
    return ok ? R_OK : R_BAD_HANDLE;
}

void kern_handle_close(uint32_t h) {
    if (handle_close(h)) rt_fatal("kern_handle_close(%#x): bad handle", h);
}

int kern_svc(Cpu *c, uint32_t imm) {
    uint64_t *x = c->x;
    switch (imm) {
    case 0x08: {   // CreateThread(-> w1 handle; x1 entry, x2 arg, x3 stack top, w4 priority, w5 core)
        int prio = (int)x[4], core = (int)x[5];
        if (!x[1] || !x[3] || prio < 28 || prio > 59 || core < -3 || core > 2) {
            x[0] = R_BAD_STATE;
            return 1;
        }
        RtThread *t = thr_new(prio, core >= 0 ? core : MAIN_CORE, core >= 0 ? 1ull << core : 7);
        t->entry = x[1];
        t->arg = x[2];
        t->stack_top = x[3] & ~15ull;
        lock();
        thr_link(t);
        unlock();
        if (!plat_thread_spawn(thread_main, t, "guest", HOST_STACK)) rt_fatal("kernel: cannot start a guest thread");
        if (rt_trace & RT_TRACE_SVC) {
            char d[128];
            rt_log("CreateThread %#x entry %s stack %#llx prio %d core %d", t->handle,
                   ldr_describe(t->entry, d, sizeof d), (unsigned long long)t->stack_top, prio, core);
        }
        x[0] = R_OK;
        x[1] = t->handle;
        return 1;
    }
    case 0x09: {   // StartThread(w0)
        lock();
        RtThread *t = h_get((uint32_t)x[0], O_THREAD);
        if (t) {
            t->started = 1;
            plat_cond_signal(&t->cv);
        }
        unlock();
        x[0] = t ? R_OK : R_BAD_HANDLE;
        return 1;
    }
    case 0x0A: {   // ExitThread
        RtThread *t = cur(imm);
        if (t->fiber) {   // on a guest fiber: unwind from the thread's own context (kern_fiber_switch resumes it)
            t->exit_pending = 1;
            t->leaving = t->fiber;
            t->fiber = NULL;
            t->cur = t->cpu;
            plat_fiber_switch(t->base_fiber);
            rt_fatal("kernel: ExitThread on a fiber came back");
        }
        exit_now(t);
    }
    case 0x0B: {   // SleepThread(x0 ns); 0, -1, -2 yield
        int64_t ns = (int64_t)x[0];
        RtThread *t = cur(imm);
        if (ns > 0) {
            lock();
            core_put(t);
            acct(t, S_SLEEP);
            t->wobj = (uint64_t)ns;
            t->nblk++;
            unlock();
            uint64_t t0 = kern_now_ns();
            sleep_ns(t, ns);
            uint64_t took = kern_now_ns() - t0;
            lock();
            t->sl_n++;
            t->sl_req += (uint64_t)ns;
            t->sl_act += took;
            core_take(t);
            unlock();
            return 1;
        }
        // Yield: hand the core to a waiting thread of equal or better priority.
        lock();
        int give = 0;
        int tp = prio_of(t);
        for (RtThread *w = g_core_wait; w && !give; w = w->cnext)
            give = t->cur_core >= 0 && core_allowed(w, t->cur_core) && prio_of(w) <= tp;
        if (give) {
            core_put(t);
            core_take(t);
        }
        unlock();
        if (!give) plat_yield();
        return 1;
    }
    case 0x0C: {   // GetThreadPriority(w1 handle -> w1)
        lock();
        RtThread *t = h_get((uint32_t)x[1], O_THREAD);
        x[0] = t ? R_OK : R_BAD_HANDLE;
        if (t) x[1] = (uint64_t)t->prio;
        unlock();
        return 1;
    }
    case 0x0D: {   // SetThreadPriority(w0 handle, w1 priority)
        if ((uint32_t)x[1] < 28 || (uint32_t)x[1] > 59) {
            x[0] = R_BAD_STATE;
            return 1;
        }
        lock();
        RtThread *t = h_get((uint32_t)x[0], O_THREAD);
        if (t) t->prio = (int)x[1];
        x[0] = t ? R_OK : R_BAD_HANDLE;
        unlock();
        return 1;
    }
    case 0x0E: {   // GetThreadCoreMask(w2 handle -> w1 core, x2 mask)
        lock();
        RtThread *t = h_get((uint32_t)x[2], O_THREAD);
        x[0] = t ? R_OK : R_BAD_HANDLE;
        if (t) {
            x[1] = (uint64_t)(uint32_t)t->core;
            x[2] = t->mask;
        }
        unlock();
        return 1;
    }
    case 0x0F: {   // SetThreadCoreMask(w0 handle, w1 ideal core, x2 mask): -1 process default, -2/-3 keep the ideal core
        lock();
        RtThread *t = h_get((uint32_t)x[0], O_THREAD);
        if (t) {
            int core = (int)x[1];
            if (core >= 0) t->core = core;
            else if (core == -1) t->core = MAIN_CORE;
            if (x[2]) t->mask = x[2];
            if (t == tcur && t->cur_core >= 0 && !core_allowed(t, t->cur_core)) {   // move off a forbidden core
                core_put(t);
                core_take(t);
            }
        }
        x[0] = t ? R_OK : R_BAD_HANDLE;
        unlock();
        return 1;
    }
    case 0x10: {   // GetCurrentProcessorNumber: the virtual core held
        RtThread *t = cur(imm);
        x[0] = (uint64_t)(uint32_t)(t->cur_core >= 0 ? t->cur_core : t->core);
        return 1;
    }
    case 0x36: {   // SynchronizePreemptionState: the SDK's spin locks call it after clearing TLR+0x100 (disable_count)
                   // when TLR+0x102 (interrupt_flag) is set; rt_svc's checkpoint has already honoured a pending preempt
        RtThread *t = cur(imm);
        volatile uint16_t *flag = (volatile uint16_t *)(t->tlr + 0x102);
        __atomic_fetch_add(&g_sync36, 1, __ATOMIC_RELAXED);
        *flag = 0;
        ap_sync(t);
        x[0] = R_OK;
        return 1;
    }
    case 0x25: {   // GetThreadId(w1 handle -> x1)
        lock();
        RtThread *t = h_get((uint32_t)x[1], O_THREAD);
        x[0] = t ? R_OK : R_BAD_HANDLE;
        if (t) x[1] = t->id;
        unlock();
        return 1;
    }

    case 0x1A: {   // ArbitrateLock(w0 owner handle, x1 address, w2 tag)
        RtThread *t = cur(imm);
        uint64_t addr = x[1];
        uint32_t r = R_OK;
        lock();
        if (ld32(addr) == ((uint32_t)x[0] | WAITERS_BIT)) {
            t->waddr = addr;
            t->wtag = (uint32_t)x[2];
            t->wobj = (uint32_t)x[0];
            r = block(t, W_LOCK, -1);
        }
        unlock();
        x[0] = r;
        return 1;
    }
    case 0x1B:   // ArbitrateUnlock(x0 address)
        lock();
        lock_handoff(x[0]);
        unlock();
        x[0] = R_OK;
        return 1;
    case 0x1C: {   // WaitProcessWideKeyAtomic(x0 lock, x1 key, w2 tag, x3 timeout)
        // The key is set before the lock is released: the SDK's Signal reads the key in user mode and skips svc 0x1D when it is 0,
        // so releasing first lets another thread change the condition and wake nobody. Linking into the wait list happens in
        // block, still holding G.
        RtThread *t = cur(imm);
        uint64_t addr = x[0], key = x[1];
        lock();
        st32(key, 1);
        t->wkey = key;
        t->waddr = addr;
        t->wtag = (uint32_t)x[2];
        lock_handoff(addr);
        t->wobj = key;
        uint32_t r = block(t, W_CONDVAR, (int64_t)x[3]);   // on timeout user mode takes the lock itself
        unlock();
        x[0] = r;
        return 1;
    }
    case 0x1D: {   // SignalProcessWideKey(x0 key, w1 count; <= 0 all)
        uint64_t key = x[0];
        int cnt = (int)x[1], n = 0;
        lock();
        for (RtThread *w; (cnt <= 0 || n < cnt) && (w = pick(W_CONDVAR, key)); n++) lock_acquire_for(w, w->waddr, w->wtag);
        int left = 0;
        for (RtThread *w = g_wait; w && !left; w = w->next) left = w->wkind == W_CONDVAR && w->wkey == key;
        if (!left) st32(key, 0);
        unlock();
        x[0] = R_OK;
        return 1;
    }

    case 0x34: {   // WaitForAddress(x0 address, w1 type, x2 value, x3 timeout): 0 if less, 1 decrement if less, 2 if equal,
                   // 3 if equal on 64 bits (the Switch 2 SDK's atomic wait uses it)
        RtThread *t = cur(imm);
        uint64_t addr = x[0];
        int type = (int)x[1];
        int32_t val = (int32_t)x[2];
        if (type < 0 || type > 3) {
            x[0] = R_INVALID_ENUM;
            return 1;
        }
        lock();
        uint32_t r, v = ld32(addr);
        int ok;
        if (type == 3) ok = (int64_t)LD64(addr) == (int64_t)x[2];
        else ok = type == 2 ? (int32_t)v == val : (int32_t)v < val;
        if (ok && type == 1)
            while (!cas32(addr, &v, v - 1))
                if (!((int32_t)v < val)) {
                    ok = 0;
                    break;
                }
        if (!ok) r = R_BAD_STATE;
        else {
            t->waddr = addr;
            t->wobj = addr;
            r = block(t, W_ADDRESS, (int64_t)x[3]);
        }
        unlock();
        x[0] = r;
        return 1;
    }
    case 0x35: {   // SignalToAddress(x0 address, w1 type, w2 value, w3 count): 0 wake, 1 increment if equal,
                   // 2 if equal modify by the number of waiters
        uint64_t addr = x[0];
        int type = (int)x[1], cnt = (int)x[3];
        int32_t val = (int32_t)x[2];
        if (type < 0 || type > 2) {
            x[0] = R_INVALID_ENUM;
            return 1;
        }
        lock();
        int nw = 0;
        for (RtThread *w = g_wait; w; w = w->next) nw += w->wkind == W_ADDRESS && w->waddr == addr;
        uint32_t r = R_OK;
        if (type != 0) {
            int32_t nv = val + 1;
            if (type == 2) {
                if (cnt <= 0) nv = nw > 0 ? val - 1 : val + 1;
                else nv = nw == 0 ? val + 1 : nw <= cnt ? val - 1 : val;
            }
            uint32_t e = (uint32_t)val;
            if (!cas32(addr, &e, (uint32_t)nv)) r = R_BAD_STATE;
        }
        if (!r)
            for (int n = 0; cnt <= 0 || n < cnt; n++) {
                RtThread *w = pick(W_ADDRESS, addr);
                if (!w) break;
                wake(w, 0);
            }
        unlock();
        x[0] = r;
        return 1;
    }

    case 0x45: {   // CreateEvent(-> w1 writable, w2 readable)
        Event *e = calloc(1, sizeof *e);
        if (!e) rt_fatal("kernel: event allocation");
        e->refs = 2;
        lock();
        uint32_t hw = h_new(O_EVENT_W, e), hr = h_new(O_EVENT_R, e);
        unlock();
        x[0] = R_OK;
        x[1] = hw;
        x[2] = hr;
        return 1;
    }
    case 0x11: case 0x12: case 0x17: {   // SignalEvent(writable), ClearEvent(either end), ResetSignal(readable)
        uint32_t h = (uint32_t)x[0];
        uint32_t r = R_OK;
        lock();
        Event *e = imm == 0x11 ? h_peek(h, O_EVENT_W) : imm == 0x17 ? h_peek(h, O_EVENT_R)
                 : h_peek(h, O_EVENT_W) ? h_peek(h, O_EVENT_W) : h_peek(h, O_EVENT_R);
        if (!e) {
            r = imm == 0x17 && h_peek(h, O_THREAD) ? R_BAD_STATE : R_BAD_HANDLE;
            if (r == R_BAD_HANDLE) rt_log("kernel: event svc %#x: invalid handle %#x", imm, h);
        } else if (imm == 0x11) {
            e->signaled = 1;
            wake_sync_waiters();
        } else if (imm == 0x17 && !e->signaled) {
            r = R_BAD_STATE;
        } else {
            e->signaled = 0;
        }
        unlock();
        x[0] = r;
        return 1;
    }
    case 0x18: {   // WaitSynchronization(-> w1 index; x1 handles, w2 count, x3 timeout)
        RtThread *t = cur(imm);
        const uint32_t *hs = (const uint32_t *)(uintptr_t)x[1];
        int n = (int)x[2];
        int64_t ns = (int64_t)x[3];
        if (n < 0 || n > 64 || (!hs && n)) {
            x[0] = R_BAD_STATE;
            return 1;
        }
        uint64_t start = kern_now_ns();
        uint32_t r;
        int idx = -1;
        lock();
        g_sync_calls++;
        for (;;) {
            if (t->cancel) {
                t->cancel = 0;
                r = R_CANCELLED;
                break;
            }
            int bad = 0;
            for (int i = 0; i < n && idx < 0; i++)
                if (sig_obj(hs[i], &bad)) idx = i;
            if (bad) {
                r = R_BAD_HANDLE;
                break;
            }
            if (idx >= 0) {
                r = R_OK;
                break;
            }
            int64_t left = ns;
            if (ns > 0) {
                uint64_t spent = kern_now_ns() - start;
                if (spent >= (uint64_t)ns) {
                    r = R_TIMEOUT;
                    break;
                }
                left = ns - (int64_t)spent;
            }
            t->wobj = n ? hs[0] | (uint64_t)n << 32 : 0;
            g_sync_blocks++;
            if (block(t, W_SYNC, left) == R_TIMEOUT) {
                r = R_TIMEOUT;
                break;
            }
        }
        unlock();
        x[0] = r;
        x[1] = idx >= 0 ? (uint64_t)idx : 0;
        return 1;
    }
    case 0x19: {   // CancelSynchronization(w0 thread)
        lock();
        RtThread *t = h_get((uint32_t)x[0], O_THREAD);
        if (t) {
            t->cancel = 1;
            if (t->wkind == W_SYNC) wake(t, (uint32_t)-1);
        }
        unlock();
        x[0] = t ? R_OK : R_BAD_HANDLE;
        return 1;
    }
    case 0x16:   // CloseHandle(w0)
        x[0] = handle_close((uint32_t)x[0]);
        return 1;
    }
    return 0;
}

// ---- handles and events for host services (ipc.c and the service overrides)
uint32_t kern_handle_new(int kind, void *obj) {
    if (kind != KERN_SESSION && kind != KERN_TMEM) rt_fatal("kern_handle_new: kind %d", kind);
    lock();
    uint32_t h = h_new(kind, obj);
    unlock();
    return h;
}

void *kern_handle_get(uint32_t h, int kind) {
    lock();
    void *o = h_peek(h, kind);
    unlock();
    return o;
}

struct IpcSession *kern_session_acquire(uint32_t h) {
    lock();
    struct IpcSession *s = h_peek(h, O_SESSION);
    if (s) ipc_session_retain(s);
    unlock();
    return s;
}

uint32_t kern_event_new(void) {
    Event *e = calloc(1, sizeof *e);
    if (!e) rt_fatal("kernel: event allocation");
    e->refs = 1;
    lock();
    uint32_t hw = h_new(O_EVENT_W, e);
    unlock();
    return hw;
}

// Copy-handle transfer of the readable end: each call gives the client its own handle to the same event.
uint32_t kern_event_readable(uint32_t hw) {
    lock();
    Event *e = h_peek(hw, O_EVENT_W);
    if (!e) rt_fatal("kern_event_readable: %#x is not a writable event handle", hw);
    e->refs++;
    uint32_t hr = h_new(O_EVENT_R, e);
    unlock();
    return hr;
}

void kern_event_signal(uint32_t hw) {
    lock();
    Event *e = h_peek(hw, O_EVENT_W);
    if (!e) rt_fatal("kern_event_signal: %#x is not a writable event handle", hw);
    e->signaled = 1;
    wake_sync_waiters();
    unlock();
}

// System events of host services: the SDK's own nn::os::CreateSystemEvent creates the kernel event (SVC 0x45) on a
// guest thread; the writable handle is kept, so any thread (host audio tick included) can signal it later.
static struct { uint64_t ev; uint32_t wh; } g_se[512];
static int g_nse;
static PlatMutex g_se_lock = PLAT_MUTEX_INIT;

static uint32_t se_find(uint64_t ev) {
    plat_mutex_lock_shared(&g_se_lock);
    uint32_t wh = 0;
    for (int i = 0; i < g_nse && !wh; i++)
        if (g_se[i].ev == ev) wh = g_se[i].wh;
    plat_mutex_unlock_shared(&g_se_lock);
    return wh;
}

void kern_bind_system_event(Cpu *c, uint64_t ev, int autoclear) {
    if (se_find(ev)) return;
    if (!tcur || !c) rt_fatal("kern_bind_system_event(%#llx) on a host thread", (unsigned long long)ev);
    static uint64_t create, getw;
    if (!create) {
        create = ldr_symbol("_ZN2nn2os17CreateSystemEventEPNS0_15SystemEventTypeENS0_14EventClearModeEb");
        getw = ldr_symbol("_ZN2nn2os30GetWritableHandleOfSystemEventEPKNS0_15SystemEventTypeE");
        if (!create || !getw) rt_fatal("kern_bind_system_event: the SDK's system event functions are not loaded");
    }
    uint32_t r = (uint32_t)rt_call(c, create, 3, ev, (uint64_t)(autoclear ? 1 : 0), (uint64_t)1);
    if (r) rt_fatal("kern_bind_system_event: CreateSystemEvent(%#llx) failed %#x", (unsigned long long)ev, r);
    uint32_t wh = (uint32_t)rt_call(c, getw, 1, ev);
    plat_mutex_lock(&g_se_lock);
    if (g_nse == (int)(sizeof g_se / sizeof *g_se)) rt_fatal("kern_bind_system_event: table full");
    g_se[g_nse].ev = ev;
    g_se[g_nse++].wh = wh;
    plat_mutex_unlock(&g_se_lock);
}

// An event not bound yet is bound on the spot with manual clear, which needs a guest thread.
void kern_signal_system_event(uint64_t ev) {
    uint32_t wh = se_find(ev);
    if (!wh) {
        kern_bind_system_event(kern_cpu(), ev, 0);
        wh = se_find(ev);
    }
    kern_event_signal(wh);
}

// ---- fibers
// The SDK switches fibers by swapping register context. Each guest fiber gets a host fiber (own 256 MiB reserved stack) and its
// own Cpu; a switch carries the thread state (TLR, TLS pointer, FPCR/FPSR) to the target's Cpu and clears the exclusive monitor.
enum { F_NEW, F_STARTED, F_DONE };
struct Fiber {
    uint64_t addr, func, arg, stack, size;
    Cpu *cpu;
    PlatFiber *host;
    int state;
    volatile int running;
};
static Fiber *g_fibers[FIBER_MAX];
static PlatMutex g_fiber_lock = PLAT_MUTEX_INIT;

static Fiber *fiber_find(uint64_t addr) {
    Fiber *f = NULL;
    plat_mutex_lock_shared(&g_fiber_lock);
    for (int i = 0; i < FIBER_MAX && !f; i++)
        if (g_fibers[i] && g_fibers[i]->addr == addr) f = g_fibers[i];
    plat_mutex_unlock_shared(&g_fiber_lock);
    return f;
}

static void fiber_switch_to(RtThread *t, const Cpu *from, Fiber *to);

// After a host fiber switch: release the context left behind, and finish an ExitThread issued on a fiber.
static void fiber_resumed(void) {
    RtThread *t = thread_after_switch();
    rt_fpcr_sync(t->cur);   // the host fiber resumes with its own MXCSR (platform.h fibers switch it)
    if (t->leaving) {
        __atomic_store_n(&t->leaving->running, 0, __ATOMIC_RELEASE);
        t->leaving = NULL;
    }
    if (t->exit_pending && !t->fiber) exit_now(t);
}

static void fiber_main(void *p) {
    Fiber *f = p;
    fiber_resumed();
    Cpu *c = f->cpu;
    thread_after_switch()->host_depth = 0;   // each context restores its own depth on resume
    uint64_t next = rt_call(c, f->func, 1, f->arg);
    f->state = F_DONE;
    Fiber *n = NULL;
    if (next && !(n = fiber_find(next)))
        rt_fatal("fiber %#llx returned %#llx, which is not an initialized fiber", (unsigned long long)f->addr,
                 (unsigned long long)next);
    fiber_switch_to(thread_after_switch(), c, n);   // NULL: back to the current thread's own context
    rt_fatal("fiber %#llx resumed after it finished", (unsigned long long)f->addr);
}

static void fiber_switch_to(RtThread *t, const Cpu *from, Fiber *to) {
    if (!t->base_fiber) {
        t->base_fiber = plat_fiber_self();
        if (!t->base_fiber) rt_fatal("kernel: cannot make the thread a fiber");
    }
    Cpu *dst = to ? to->cpu : t->cpu;
    if (to) {
        if (to->state == F_DONE) rt_fatal("SwitchToFiber(%#llx): the fiber has finished", (unsigned long long)to->addr);
        int idle = 0;
        if (!__atomic_compare_exchange_n(&to->running, &idle, 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
            rt_fatal("SwitchToFiber(%#llx): the fiber is running", (unsigned long long)to->addr);
        if (to->state == F_NEW) {
            dst->sp = (to->stack + to->size) & ~15ull;
            to->state = F_STARTED;
        }
    }
    dst->tpidr = from->tpidr;
    dst->tpidrro = from->tpidrro;
    dst->fpcr = from->fpcr;
    dst->fpsr = from->fpsr;
    dst->excl_addr = ~0ull;
    dst->thread = t;
    t->leaving = t->fiber;
    t->fiber = to;
    t->cur = dst;
    int depth = t->host_depth;
    plat_fiber_switch(to ? to->host : t->base_fiber);
    thread_after_switch()->host_depth = depth;
    fiber_resumed();
}

// InitializeFiber(x0 FiberType*, x1 function, x2 argument, x3 stack, x4 stack size, w5 flags)
void kern_fiber_initialize(Cpu *c) {
    uint64_t addr = c->x[0];
    Fiber *f = fiber_find(addr);
    if (f && f->running) rt_fatal("InitializeFiber(%#llx): the fiber is running", (unsigned long long)addr);
    if (!f) {
        f = calloc(1, sizeof *f);
        Cpu *fc = plat_aligned_alloc(sizeof(Cpu), 64);
        if (!f || !fc) rt_fatal("kernel: fiber allocation");
        f->cpu = fc;
        plat_mutex_lock(&g_fiber_lock);
        int i = 0;
        while (i < FIBER_MAX && g_fibers[i]) i++;
        if (i == FIBER_MAX) rt_fatal("kernel: more than %d fibers", FIBER_MAX);
        g_fibers[i] = f;
        plat_mutex_unlock(&g_fiber_lock);
    } else if (f->host) {
        plat_fiber_free(f->host);
    }
    memset(f->cpu, 0, sizeof *f->cpu);
    f->cpu->excl_addr = ~0ull;
    f->addr = addr;
    f->func = c->x[1];
    f->arg = c->x[2];
    f->stack = c->x[3];
    f->size = c->x[4];
    f->state = F_NEW;
    f->running = 0;
    f->host = plat_fiber_new(HOST_STACK, fiber_main, f);
    if (!f->host) rt_fatal("kernel: cannot create a host fiber");
}

// FinalizeFiber(x0 FiberType*)
void kern_fiber_finalize(Cpu *c) {
    uint64_t addr = c->x[0];
    plat_mutex_lock(&g_fiber_lock);
    Fiber *f = NULL;
    for (int i = 0; i < FIBER_MAX && !f; i++)
        if (g_fibers[i] && g_fibers[i]->addr == addr) {
            f = g_fibers[i];
            g_fibers[i] = NULL;
        }
    plat_mutex_unlock(&g_fiber_lock);
    if (!f) rt_fatal("FinalizeFiber(%#llx): not an initialized fiber", (unsigned long long)addr);
    if (f->running) rt_fatal("FinalizeFiber(%#llx): the fiber is running", (unsigned long long)addr);
    plat_fiber_free(f->host);
    plat_aligned_free(f->cpu);
    free(f);
}

// SwitchToFiber(x0 FiberType*)
void kern_fiber_switch(Cpu *c) {
    RtThread *t = tcur;
    if (!t) rt_fatal("SwitchToFiber on a host thread");
    if (!c->x[0]) {   // SwitchToFiber(nullptr): back to the thread's own context (GetCurrentFiber reports NULL)
        if (t->fiber) fiber_switch_to(t, c, NULL);
        return;
    }
    Fiber *to = fiber_find(c->x[0]);
    if (!to) rt_fatal("SwitchToFiber(%#llx): not an initialized fiber", (unsigned long long)c->x[0]);
    if (to == t->fiber) return;
    fiber_switch_to(t, c, to);
}

// GetCurrentFiber() -> x0, NULL on the thread's own context
void kern_fiber_current(Cpu *c) {
    RtThread *t = tcur;
    c->x[0] = t && t->fiber ? t->fiber->addr : 0;
}

HOST_OVERRIDE(KERN_SYM_INITIALIZE_FIBER, kern_fiber_initialize);
HOST_OVERRIDE(KERN_SYM_FINALIZE_FIBER, kern_fiber_finalize);
HOST_OVERRIDE(KERN_SYM_SWITCH_TO_FIBER, kern_fiber_switch);
HOST_OVERRIDE(KERN_SYM_GET_CURRENT_FIBER, kern_fiber_current);
