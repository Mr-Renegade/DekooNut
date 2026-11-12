// Guest nonlocal returns must unwind the host call stack as well as ARM64 state.
#include <stdlib.h>
#include "guest_jump.h"
#include "host.h"

// Fibers share a host thread, so every lookup and removal also matches Cpu.
static _Thread_local RtGuestJump *jumps;

RtGuestJump *rt_guest_jump_prepare(RtGuestJumpScope *scope) {
    Cpu *c = scope->cpu;
    RtGuestJump **p = &jumps;
    while (*p && ((*p)->owner != scope || (*p)->buffer != c->x[0])) p = &(*p)->next;
    RtGuestJump *j = *p;
    if (j) *p = j->next;
    if (!j) {
        j = malloc(sizeof *j);
        if (!j) rt_fatal("guest setjmp: out of memory");
    }
    j->next = jumps;
    jumps = j;
    j->owner = scope;
    j->buffer = c->x[0];
    j->saved = *c;
    j->depth = c->thread ? *kern_host_depth(c->thread) : 0;
    return j;
}

void rt_guest_jump_cleanup(RtGuestJumpScope *scope) {
    for (RtGuestJump **p = &jumps; *p;) {
        RtGuestJump *j = *p;
        if (j->owner == scope) { *p = j->next; free(j); }
        else p = &j->next;
    }
}

void rt_guest_jump_landed(Cpu *c) {
    // longjmp leaves the destination first among this Cpu's active contexts.
    RtGuestJump *j = jumps;
    while (j && j->owner->cpu != c) j = j->next;
    if (!j) rt_fatal("guest longjmp: missing landing context");
    *c = j->saved;
    c->x[0] = (uint32_t)j->value;
    if (c->thread) *kern_host_depth(c->thread) = j->depth;
    rt_fpcr_sync(c);
}

void rt_guest_longjmp(Cpu *c) {
    RtGuestJump *target = jumps;
    while (target && (target->owner->cpu != c || target->buffer != c->x[0])) target = target->next;
    if (!target) rt_fatal("guest longjmp: no active setjmp for buffer %#llx", (unsigned long long)c->x[0]);
    target->value = (int)c->x[1];
    if (!target->value) target->value = 1;
    // Contexts belonging to unwound host frames cannot remain in the registry.
    for (RtGuestJump **p = &jumps; *p && *p != target;) {
        RtGuestJump *j = *p;
        if (j->owner->cpu == c) { *p = j->next; free(j); }
        else p = &j->next;
    }
    longjmp(target->host, 1);
}

HOST_OVERRIDE("longjmp", rt_guest_longjmp);
