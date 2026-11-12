// Host continuations for guest setjmp calls. Only generated callers include this.
#pragma once
#include <setjmp.h>
#include "cpu.h"

typedef struct RtGuestJumpScope { Cpu *cpu; } RtGuestJumpScope;
typedef struct RtGuestJump {
    jmp_buf host;
    Cpu saved;
    struct RtGuestJump *next;
    RtGuestJumpScope *owner;
    uint64_t buffer;
    int depth, value;
} RtGuestJump;

RtGuestJump *rt_guest_jump_prepare(RtGuestJumpScope *scope);
void rt_guest_jump_landed(Cpu *c);
void rt_guest_jump_cleanup(RtGuestJumpScope *scope);
void rt_guest_longjmp(Cpu *c) __attribute__((noreturn));
