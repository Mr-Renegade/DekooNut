// Guest address -> host function lookup for indirect branches (rt_dispatch, declared in cpu.h).
// One flat table over every module's code range, indexed by (pc - lo) >> 2; built once, then read without locks.
#pragma once
#include <stdbool.h>
#include <stdint.h>

#include "cpu.h"

// A lifted module: its place in the guest layout and the generator's sorted function table.
typedef struct RtModule {
    const char *name;
    uint64_t base, size;      // guest image (layout.h MOD_<NAME>_BASE / _SIZE)
    const uint64_t *addrs;    // <mod>_fn_addrs: guest function starts, strictly ascending
    const GuestFn *fns;       // <mod>_fn_ptrs: the lifted function for each start
    uint64_t count;           // <mod>_fn_count
} RtModule;

// Module glue written by tools/build_game.py (build/<version>/modules.c): the layout's modules and the linked ones.
typedef struct RtLayoutEntry { const char *name; uint64_t base, size; } RtLayoutEntry;
extern const RtLayoutEntry rt_layout[];
extern const int rt_layout_count;
int rt_linked_modules(RtModule *out, int max);

// Register a host function at a guest address outside every module (HLE entry points); before rt_dispatch_init only.
void rt_register_host(uint64_t addr, GuestFn fn);

// Build the table (modules kept by pointer). Call once, before any guest thread starts. Fatal on a malformed table.
void rt_dispatch_init(const RtModule *mods, int count);

// The host function for a guest address, or NULL when nothing is there. Does not run guest code.
GuestFn rt_dispatch_lookup(uint64_t pc);

// "0x7000001234 (main+0x1234)" for a guest address inside a module, else the bare address. Returns buf.
const char *rt_guest_name(uint64_t pc, char *buf, int size);

// Runtime fatal error: message to stderr, then exit. Used by the dispatcher and the stubs.
void rt_fatal(const char *fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));
