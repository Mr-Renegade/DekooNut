// SPDX-License-Identifier: GPL-3.0-or-later
// Internal native mesh replay, included by vk_exec_draw.c after its ordinary draw helpers.
// The cache contains locally compiled SPIR-V and MSH3 resource metadata, never repository assets.
// Indirect pointer resolution follows Lichii buffer_cache.h::BindGraphicsStorageBuffer; pointers are read from
// ordered GPU-visible memory before rendering begins, not from possibly stale guest bytes.
#pragma once
#ifndef _WIN32
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#define MESH_BINDINGS 64
#define MESH_C0_BYTES 0x900

typedef struct {
    GfxShaderRes reflection;   // first, so VkPipeBinding.res can refer to the common resource declaration
    uint32_t indirect;
    uint32_t texture_type;
    p_bind_buf bound;
    int kind;
    uint32_t *hp, hp_words;    // "H": the handle program of a texture whose handle the stage reads from a storage buffer
} MeshResource;

typedef struct MeshModule {
    struct MeshModule *next;
    uint64_t key, native;
    uint32_t stage, regs[3], count, words;
    uint32_t array_slots;   // texture slots compiled as 2D arrays: bit per slot, 0 for the plain module (see mesh_module_fit)
    uint32_t *code;
    int compile_pid;   // a background oot_mesh_xlate is at work on it (Linux)
    int gave_up;       // its compiler failed, or it has no compiler: its draws stay skipped
    MeshResource resources[MESH_BINDINGS];
} MeshModule;

typedef struct MeshPipeline {
    struct MeshPipeline *next;
    MeshModule *mesh, *task;
    uint8_t state_key[32];
    VkPipeInfo info;
    VkPipeBinding bindings[MAX_BINDINGS];
    GfxShaderRes samplers[MAX_BINDINGS];   // by binding index: a sampler array's resource, see mesh_pipeline_get
    int building;   // queued or building on a mesh builder thread; info.pipeline is valid once it is 0
    int waited_out;   // a wait for this build already ran out of budget
    uint64_t keys[3];   // mesh, task and fragment keys, for the builder's log line
} MeshPipeline;

static MeshModule *mesh_modules;
static MeshPipeline *mesh_pipelines;
static unsigned mesh_pipeline_count;

// Run-wide replay counters.
enum { MS_CALL, MS_ISSUED, MS_IND_CALL, MS_GROUPS,
       MS_NO_MESH, MS_NO_TASK, MS_INIT, MS_LIMIT, MS_RESOURCE, MS_SCISSOR, MS_FS, MS_PIPE, MS_OTHER, MS_BUILDING, MS_COUNT };
static uint64_t mesh_stat[MS_COUNT];
static struct { uint32_t width; uint64_t issued, groups; } mesh_stat_target[16];
static void mesh_stat_issued(uint32_t width, uint32_t groups) {
    int i = 0;
    while (i < 15 && mesh_stat_target[i].issued && mesh_stat_target[i].width != width) i++;
    mesh_stat_target[i].width = width, mesh_stat_target[i].issued++, mesh_stat_target[i].groups += groups;
    mesh_stat[MS_ISSUED]++, mesh_stat[MS_GROUPS] += groups;
}

static uint64_t mesh_source_key(const uint8_t *ctl, const uint8_t *code);
static int mesh_program_source(const prog_t *program, int kind, const uint8_t **control, const uint8_t **code);

static uint64_t mesh_source_key(const uint8_t *ctl, const uint8_t *code) {
    return gfx_shader_is_nvn2(ctl) ? gfx_shader_nvn2_hash(ctl, code) : nvn2_origin(code);
}

static int mesh_program_scan(const prog_t *program, int kind, const uint8_t **control, const uint8_t **code) {
    for (unsigned i = 0; i < program->nstage; i++) {
        const uint8_t *c = program->sd[i].control, *b = (const uint8_t *)(uintptr_t)program->sd[i].data;
        if (nvn2_native_kind(c, b) == kind) { *control = c; *code = b; return 1; }
    }
    return 0;
}

// Per-program memo (prog_t.serial changes at every ProgramSetShaders) of the task (6), mesh (7) and fragment stages and
// their keys, so a mesh draw does not rescan and rehash every stage. Replay thread only; other threads scan.
enum { MPM_N = 256 };
static struct MeshProgMemo {
    uint64_t serial; const prog_t *prog;
    const uint8_t *ctl[3], *code[3];
    uint64_t src[3], pk[3];
    uint64_t zgen[3];            // nvn2_origin_gen when a zero key was last computed
    uint8_t have[3], keyed[3], tried[3];
} mpm[MPM_N];
static int mpm_slot(int kind) { return kind == 6 ? 0 : kind == 7 ? 1 : kind == NVN_STAGE_FS ? 2 : -1; }
static struct MeshProgMemo *mpm_get(const prog_t *program) {
    if (!program->serial || !vk_exec_recording_fence()) return NULL;
    struct MeshProgMemo *m = &mpm[(program->serial * 0x9E3779B97F4A7C15ull) >> 56];
    if (m->serial != program->serial || m->prog != program) {
        memset(m, 0, sizeof *m);
        m->serial = program->serial, m->prog = program;
        static const int kinds[3] = {6, 7, NVN_STAGE_FS};
        for (int k = 0; k < 3; k++) m->have[k] = (uint8_t)mesh_program_scan(program, kinds[k], &m->ctl[k], &m->code[k]);
    }
    return m;
}

static int mesh_program_source(const prog_t *program, int kind, const uint8_t **control, const uint8_t **code) {
    int k = mpm_slot(kind);
    struct MeshProgMemo *m = k >= 0 ? mpm_get(program) : NULL;
    if (!m) return mesh_program_scan(program, kind, control, code);
    if (!m->have[k]) return 0;
    *control = m->ctl[k], *code = m->code[k];
    return 1;
}

// mesh_source_key and nvn2_program_key of the program's stage of that kind, memoized as above (0 when it has none).
static int mesh_stage_keys(const prog_t *program, int kind, const uint8_t **control, const uint8_t **code, uint64_t *src,
                           uint64_t *pk) {
    int k = mpm_slot(kind);
    struct MeshProgMemo *m = k >= 0 ? mpm_get(program) : NULL;
    if (!m) {
        if (!mesh_program_scan(program, kind, control, code)) return 0;
        *src = mesh_source_key(*control, *code);
        if (pk) *pk = nvn2_program_key(*control, *code);
        return 1;
    }
    if (!m->have[k]) return 0;
    *control = m->ctl[k], *code = m->code[k];
    if (!m->keyed[k]) {   // a zero key (not lowered or registered yet) is asked again, once a lowering was published
        // The keys depend on the stage bytes and the origin table only: else a stage that never completes is rehashed every draw.
        uint64_t gen = nvn2_origin_gen();
        if (!m->tried[k] || m->zgen[k] != gen) {
            m->src[k] = mesh_source_key(m->ctl[k], m->code[k]), m->pk[k] = nvn2_program_key(m->ctl[k], m->code[k]);
            m->keyed[k] = m->src[k] && m->pk[k];
            m->tried[k] = 1, m->zgen[k] = gen;
        }
    }
    *src = m->src[k];
    if (pk) *pk = m->pk[k];
    return 1;
}

// The mesh cache: mesh-cache in the executable's directory when a build has its own mesh compiler, else build/mesh-cache
// (shared). Its files are compiled locally from game data and stay on this PC.
static char mesh_exe_dir[PLAT_PATH_MAX];   // set by mesh_cache_dir when the cache is the executable's own
static const char *mesh_cache_dir(void) {
    static char directory[PLAT_PATH_MAX];
    static int ready;
    if (ready) return directory[0] ? directory : NULL;
    ready = 1;
#ifndef _WIN32
    {
        char self[PLAT_PATH_MAX];
        ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
        if (n > 0) {
            self[n] = 0;
            char *slash = strrchr(self, '/');
            if (slash) {
                struct stat st;
                *slash = 0;
                snprintf(directory, sizeof directory, "%s/mesh-cache", self);
                if (!stat(directory, &st) && S_ISDIR(st.st_mode)) snprintf(mesh_exe_dir, sizeof mesh_exe_dir, "%s", self);
                else snprintf(directory, sizeof directory, "%s/../mesh-cache", self);
            }
        }
    }
#endif
    if (directory[0]) rt_log("mesh: cache %s%s", directory, mesh_exe_dir[0] ? " (own cache, oot_mesh_xlate beside the executable)" : "");
    return directory[0] ? directory : NULL;
}

// <key>-mesh.spv, or <key>-a<slots>-mesh.spv for a module compiled with those texture slots as 2D arrays.
static void mesh_module_path(char *path, size_t size, const char *directory, const MeshModule *m, const char *tail) {
    const char *stage = m->stage == 7 ? "mesh" : "task";
    if (m->array_slots) snprintf(path, size, "%s/%016llx-a%08x-%s.spv%s", directory, (unsigned long long)m->key, m->array_slots, stage, tail);
    else snprintf(path, size, "%s/%016llx-%s.spv%s", directory, (unsigned long long)m->key, stage, tail);
}

// A program the cache has not seen compiles in the background (the compiler of tools/build_mesh.py, or oot_mesh_xlate
// beside the executable for an own cache; at most 4 at a time). Its draws are skipped until the module is there, and for
// good if the compiler refuses it.
static int mesh_compiles;
static void mesh_compile_start(MeshModule *m, const char *directory, const uint8_t *control, const uint8_t *code) {
#ifndef _WIN32
    if (mesh_compiles >= 4) return;
    char source[PLAT_PATH_MAX * 2], ctl[PLAT_PATH_MAX * 2], bin[PLAT_PATH_MAX * 2], out[PLAT_PATH_MAX * 2], tool[PLAT_PATH_MAX * 2];
    char context[PLAT_PATH_MAX * 2 + 32];
    snprintf(source, sizeof source, "%s/src", directory);
    gfx_shader_nvn2_dump_to(source, control, code);
    snprintf(ctl, sizeof ctl, "%s/%016llx.ctl", source, (unsigned long long)m->key);
    snprintf(bin, sizeof bin, "%s/%016llx.bin", source, (unsigned long long)m->key);
    mesh_module_path(out, sizeof out, directory, m, "");
    if (m->array_slots) {   // the compiler takes every other slot as 2D
        snprintf(context, sizeof context, "%s/%016llx-a%08x.context", source, (unsigned long long)m->key, m->array_slots);
        FILE *f = fopen(context, "w");
        if (!f) { m->gave_up = 1; return; }
        fputs("MTC1\n", f);
        for (int slot = 0; slot < 32; slot++)
            if (m->array_slots >> slot & 1) fprintf(f, "T %d %#x %d\n", slot, 0x1000 + slot, GFX_TEX_2D_ARRAY);
        fclose(f);
    }
    if (mesh_exe_dir[0]) snprintf(tool, sizeof tool, "%s/oot_mesh_xlate", mesh_exe_dir);
    else snprintf(tool, sizeof tool, "%s/../mesh/oot_mesh_xlate", directory);
    if (access(tool, X_OK)) { m->gave_up = 1; return; }
    char *const argv[] = {tool, "compile-ctl", bin, ctl, out, m->array_slots ? context : NULL, NULL};
    pid_t pid;
    extern char **environ;
    if (posix_spawn(&pid, tool, NULL, NULL, argv, environ)) { m->gave_up = 1; return; }
    m->compile_pid = (int)pid;
    mesh_compiles++;
    rt_log("mesh: compiling %016llx in the background (2D array slots %#x)", (unsigned long long)m->key, m->array_slots);
#else
    (void)directory, (void)control, (void)code;
    m->gave_up = 1;
#endif
}

// 1: no compile is running for m (any more), so the cache can be read again.
static int mesh_compile_poll(MeshModule *m) {
#ifndef _WIN32
    if (!m->compile_pid) return 1;
    int status = 0;
    pid_t done = waitpid(m->compile_pid, &status, WNOHANG);
    if (done == 0) return 0;
    m->compile_pid = 0;
    mesh_compiles--;
    if (done < 0 || !WIFEXITED(status) || WEXITSTATUS(status)) {
        m->gave_up = 1;
        rt_log("mesh: %016llx does not compile: its draws stay skipped", (unsigned long long)m->key);
    }
#else
    (void)m;
#endif
    return 1;
}

// Handle program: a mesh stage texture whose handle is read from a storage buffer (tools/mesh exports an "H" line) is
// bound per draw from the handle its program gives, evaluated over the stage's constant banks and storage buffers in
// guest memory: see README, "Mesh handle programs". It goes into texture slot MESH_HP_SLOT0 + n, past the 32 bank 2 names.
#define MESH_HP_SLOT0 96
#define MESH_HP_SLOTS 32
static int mesh_hp_on(void) {
    static int logged;   // the line goes out once
    if (!logged) {
        logged = 1;
        rt_log("mesh: textures read through a handle program: bound per draw from the evaluated handle");
    }
    return 1;
}

// A guest word, from this submission's uniform snapshot when it holds it, else live imported memory.
static int mesh_hp_word(uint64_t addr, int snap, uint32_t *out) {
    if (snap && vk_usnap_read(addr, 4, out)) return 1;
    if (dyn_readable(addr, 4) < 4) return 0;
    memcpy(out, (const void *)(uintptr_t)addr, 4);
    return 1;
}

enum { HP_IMM, HP_CBUF, HP_CBUF_DYN, HP_STORAGE, HP_ADD, HP_SUB, HP_MUL, HP_AND, HP_OR, HP_XOR, HP_SHL, HP_SHR, HP_SAR,
       HP_NEG, HP_NOT, HP_UMIN, HP_UMAX, HP_SMIN, HP_SMAX, HP_SLT, HP_SLE, HP_SGT, HP_SGE, HP_ULT, HP_ULE, HP_UGT, HP_UGE,
       HP_EQ, HP_NE, HP_LAND, HP_LOR, HP_LXOR, HP_LNOT, HP_SELECT, HP_TEE, HP_LOAD };   // handle program opcodes

typedef struct { VkExec *ex; MeshModule *m; int ok; } HpCtx;

static uint32_t hp_cbuf(HpCtx *c, uint32_t bank, uint32_t offset) {
    uint32_t v = 0, base = c->m->stage == 6 ? 0x670 : 0x770;
    int stage = c->m->stage;
    if (offset & 3) { c->ok = 0; return 0; }
    if (bank == 0) {   // the driver bank mesh_resource_info builds: storage table, largest point size
        if (offset >= base && offset < base + 0x100) {
            const p_bind_buf *b = &c->ex->ssbo[stage][(offset - base) / 16];
            unsigned w = (offset - base) % 16 / 4;
            return w == 0 ? (uint32_t)b->addr : w == 1 ? (uint32_t)(b->addr >> 32) : w == 2 ? (uint32_t)b->size : 0;
        }
        if (offset == 0x610) { float f = 2047.0f; memcpy(&v, &f, 4); return v; }
        return 0;
    }
    if (bank == 1) {   // the program's inline constants
        const uint8_t *control, *code;
        uint32_t begin, end;
        if (!mesh_program_source(c->ex->prog, stage, &control, &code) || !gfx_shader_is_nvn2(control)) { c->ok = 0; return 0; }
        memcpy(&begin, control + 0x40, 4), memcpy(&end, control + 0x44, 4);
        if (begin >= end || offset + 4 > end - begin) { c->ok = 0; return 0; }
        memcpy(&v, code + begin + offset, 4);
        return v;
    }
    if (bank < 3 || bank - 3 >= 18) { c->ok = 0; return 0; }
    const p_bind_buf *b = &c->ex->ubo[stage][bank - 3];
    // An inline update of this submission (UpdateUniformBuffer) first, as ubo_ptr reads its blocks.
    if (!b->addr || offset + 4 > b->size) c->ok = 0;
    else if (!upd_read(c->ex, b->addr + offset, &v, 4) && !mesh_hp_word(b->addr + offset, 1, &v)) c->ok = 0;
    return v;
}

static uint32_t hp_storage(HpCtx *c, uint32_t binding, uint32_t offset) {
    uint32_t v = 0, base = c->m->stage == 6 ? 0x670 : 0x770, n = 0;
    for (unsigned i = 0; i < c->m->count; i++) {
        const MeshResource *r = &c->m->resources[i];
        if (r->kind != GFX_RES_SSBO || n++ != binding) continue;
        // A storage buffer is bound at its address rounded down to 16 bytes; the program's offsets count from there.
        uint64_t addr = r->indirect == UINT32_MAX ? c->ex->ssbo[c->m->stage][(r->reflection.handle * 4u - base) / 16].addr
                                                  : r->bound.addr;
        if (!addr || !mesh_hp_word((addr & ~15ull) + offset, 0, &v)) c->ok = 0;
        return v;
    }
    c->ok = 0;
    return 0;
}

// Runs r's handle program and puts the handle in the stage's texture slot for r. 0: a word
// could not be read or the program is malformed (the draw is skipped).
static int mesh_hp_eval(VkExec *ex, MeshModule *m, const MeshResource *r) {
    HpCtx c = {ex, m, 1};
    uint32_t stack[64], slots[256];
    unsigned size = 0;
    memset(slots, 0, sizeof slots);
#define HP_POP() (size ? stack[--size] : 0u)
#define HP_PUSH(x) do { uint32_t hp_v_ = (x); if (size < 64) stack[size++] = hp_v_; } while (0)
    for (uint32_t pc = 0; pc < r->hp_words && c.ok;) {
        uint32_t op = r->hp[pc++];
#define HP_OPERAND() (pc < r->hp_words ? r->hp[pc++] : 0u)
        if (op == HP_IMM) { HP_PUSH(HP_OPERAND()); continue; }
        if (op == HP_CBUF) { uint32_t bank = HP_OPERAND(), off = HP_OPERAND(); HP_PUSH(hp_cbuf(&c, bank, off)); continue; }
        if (op == HP_CBUF_DYN) { uint32_t bank = HP_OPERAND(), off = HP_POP(); HP_PUSH(hp_cbuf(&c, bank, off)); continue; }
        if (op == HP_STORAGE) { uint32_t binding = HP_OPERAND(), off = HP_POP(); HP_PUSH(hp_storage(&c, binding, off)); continue; }
        if (op == HP_NEG || op == HP_NOT || op == HP_LNOT) {
            uint32_t a = HP_POP();
            HP_PUSH(op == HP_NEG ? 0u - a : op == HP_NOT ? ~a : (a == 0));
            continue;
        }
        if (op == HP_TEE) { uint32_t slot = HP_OPERAND(); if (slot < 256 && size) slots[slot] = stack[size - 1]; continue; }
        if (op == HP_LOAD) { uint32_t slot = HP_OPERAND(); HP_PUSH(slot < 256 ? slots[slot] : 0u); continue; }
        if (op == HP_SELECT) {
            uint32_t f = HP_POP(), t = HP_POP(), cond = HP_POP();
            HP_PUSH(cond ? t : f);
            continue;
        }
        uint32_t b = HP_POP(), a = HP_POP();
        int32_t sa = (int32_t)a, sb = (int32_t)b;
        switch (op) {
        case HP_ADD: HP_PUSH(a + b); break;
        case HP_SUB: HP_PUSH(a - b); break;
        case HP_MUL: HP_PUSH(a * b); break;
        case HP_AND: HP_PUSH(a & b); break;
        case HP_OR: HP_PUSH(a | b); break;
        case HP_XOR: HP_PUSH(a ^ b); break;
        case HP_SHL: HP_PUSH(b >= 32 ? 0u : a << b); break;
        case HP_SHR: HP_PUSH(b >= 32 ? 0u : a >> b); break;
        case HP_SAR: HP_PUSH((uint32_t)(sa >> (b >= 32 ? 31 : b))); break;
        case HP_UMIN: HP_PUSH(a < b ? a : b); break;
        case HP_UMAX: HP_PUSH(a > b ? a : b); break;
        case HP_SMIN: HP_PUSH((uint32_t)(sa < sb ? sa : sb)); break;
        case HP_SMAX: HP_PUSH((uint32_t)(sa > sb ? sa : sb)); break;
        case HP_SLT: HP_PUSH(sa < sb); break;
        case HP_SLE: HP_PUSH(sa <= sb); break;
        case HP_SGT: HP_PUSH(sa > sb); break;
        case HP_SGE: HP_PUSH(sa >= sb); break;
        case HP_ULT: HP_PUSH(a < b); break;
        case HP_ULE: HP_PUSH(a <= b); break;
        case HP_UGT: HP_PUSH(a > b); break;
        case HP_UGE: HP_PUSH(a >= b); break;
        case HP_EQ: HP_PUSH(a == b); break;
        case HP_NE: HP_PUSH(a != b); break;
        case HP_LAND: HP_PUSH(a && b); break;
        case HP_LOR: HP_PUSH(a || b); break;
        case HP_LXOR: HP_PUSH(!a != !b); break;
        default: c.ok = 0; break;
        }
#undef HP_OPERAND
    }
    uint32_t handle = size == 1 ? stack[0] : 0;
    if (size != 1) c.ok = 0;
#undef HP_POP
#undef HP_PUSH
    static unsigned logged;
    if (logged < 8 && (++logged, 1))
        rt_log("mesh: handle program of %016llx binding %u: %s, handle %#x", (unsigned long long)m->key,
               (unsigned)r->reflection.binding, c.ok ? "evaluated" : "unreadable", handle);
    if (!c.ok) return 0;
    ex->tex[m->stage][r->reflection.slot] = handle;
    return 1;
}

static MeshModule *mesh_module_get(uint64_t key, uint64_t native, unsigned stage, uint32_t array_slots, const uint8_t *control,
                                   const uint8_t *source) {
    const char *directory = mesh_cache_dir();
    if (!directory || !key || !native) return NULL;
    MeshModule *m = NULL;
    for (MeshModule *i = mesh_modules; i; i = i->next)
        if (i->key == key && i->stage == stage && i->array_slots == array_slots) { m = i; break; }
    if (m && m->code) return m;
    if (m && (m->gave_up || !mesh_compile_poll(m) || m->gave_up)) return NULL;
    if (!m) {
        m = calloc(1, sizeof *m);
        if (!m) rt_fatal("vk: out of memory loading a mesh module");
        m->key = key, m->stage = stage, m->array_slots = array_slots, m->next = mesh_modules, mesh_modules = m;
    }
    VkPhysicalDeviceSubgroupProperties subgroup = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 properties = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &subgroup};
    vkGetPhysicalDeviceProperties2(vk_ctx()->phys, &properties);
    VkSubgroupFeatureFlags required = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT |
        VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT;
    VkShaderStageFlags bit = stage == 7 ? VK_SHADER_STAGE_MESH_BIT_EXT : VK_SHADER_STAGE_TASK_BIT_EXT;
    if (subgroup.subgroupSize != 32 || !(subgroup.supportedStages & bit) ||
        (subgroup.supportedOperations & required) != required) {
        rt_log("mesh: cache profile requires 32-lane vote/ballot/shuffle support in stage %u", stage);
        return NULL;
    }
    char path[PLAT_PATH_MAX * 2];
    mesh_module_path(path, sizeof path, directory, m, ".bindings");
    FILE *f = fopen(path, "r");
    if (!f) {
        if (!m->compile_pid && !m->gave_up) mesh_compile_start(m, directory, control, source);
        return NULL;
    }
    char magic[8]; unsigned declared, count; unsigned long long identity;
    int ok = fscanf(f, "%7s %u %llx %u %u %u %u", magic, &declared, &identity,
                    &m->regs[0], &m->regs[1], &m->regs[2], &count) == 7 && !strcmp(magic, "MSH3") &&
             declared == stage && identity == native && count <= MESH_BINDINGS;
    for (unsigned i = 0; ok && i < count; i++) {
        MeshResource *r = &m->resources[i];
        char kind; unsigned binding, bank, elements, offset = 0, written = 0;
        ok = fscanf(f, " %c %u %u", &kind, &binding, &bank) == 3 && binding == i;
        if (!ok) break;
        if (kind == 'U') {
            ok = fscanf(f, "%u", &elements) == 1 && (bank <= 1 || (bank >= 3 && bank <= 16));
            r->kind = GFX_RES_UBO, r->reflection.slot = (int16_t)bank;
        } else if (kind == 'S') {
            ok = fscanf(f, "%u %u %u %u", &offset, &elements, &written, &r->indirect) == 4 && bank == 0;
            unsigned base = stage == 6 ? 0x670 : 0x770;
            ok = ok && offset >= base && offset < base + 0x100 && (offset - base) % 16 == 0 &&
                 (r->indirect == UINT32_MAX || (r->indirect < 512 && !(r->indirect & 3)));
            r->kind = GFX_RES_SSBO, r->reflection.cbuf_slot = (int8_t)bank;
            r->reflection.handle = (uint16_t)(offset / 4), r->reflection.flags = written ? 1 : 0;
        } else if (kind == 'H') {   // H binding type words w...: a texture handle computed per draw (mesh_hp_eval)
            unsigned hp_index = 0;
            for (unsigned j = 0; j < i; j++) hp_index += m->resources[j].hp != NULL;
            ok = mesh_hp_on() && fscanf(f, "%u", &r->hp_words) == 1 && r->hp_words && r->hp_words <= 4096 &&
                 (bank == GFX_TEX_2D || bank == GFX_TEX_2D_ARRAY) && hp_index < MESH_HP_SLOTS;
            if (ok && !(r->hp = malloc(sizeof *r->hp * r->hp_words))) rt_fatal("vk: out of memory reading a handle program");
            for (unsigned w = 0; ok && w < r->hp_words; w++) ok = fscanf(f, "%u", &r->hp[w]) == 1;
            elements = 1, r->texture_type = bank;
            r->kind = GFX_RES_TEX, r->reflection.cbuf_slot = 2;
            r->reflection.slot = (int16_t)(MESH_HP_SLOT0 + hp_index);
            r->reflection.type = (uint16_t)(3 | (r->texture_type == GFX_TEX_2D_ARRAY ? 0x100 : 0));
            r->reflection.handle = 0;
        } else if (kind == 'T') {
            ok = fscanf(f, "%u %u %u", &offset, &elements, &r->texture_type) == 3 && bank == 2 &&
                 offset >= 0x20 && offset < 0x120 && (offset - 0x20) % 8 == 0 &&
                 (r->texture_type == GFX_TEX_2D || r->texture_type == GFX_TEX_2D_ARRAY);
            r->kind = GFX_RES_TEX, r->reflection.cbuf_slot = 2;
            r->reflection.slot = (int16_t)((offset - 0x20) / 8);
            r->reflection.type = (uint16_t)(3 | (r->texture_type == GFX_TEX_2D_ARRAY ? 0x100 : 0));
            r->reflection.handle = (uint16_t)(offset / 4);
        } else ok = 0;
        ok = ok && elements == 1;
        r->reflection.binding = (uint16_t)binding, r->reflection.count = 1;
    }
    char extra;
    if (ok && fscanf(f, " %c", &extra) == 1) ok = 0;
    fclose(f);
    if (!ok || vk_ctx()->props.limits.minStorageBufferOffsetAlignment > 16) {
        rt_log("mesh: unsupported or invalid binding metadata for %016llx", (unsigned long long)key);
        m->gave_up = 1;
        return NULL;
    }
    mesh_module_path(path, sizeof path, directory, m, "");
    f = fopen(path, "rb");
    if (!f) { m->gave_up = 1; return NULL; }
    fseek(f, 0, SEEK_END); long size = ftell(f); rewind(f);
    if (size < 20 || size > 16 * 1024 * 1024 || size % 4) { fclose(f); m->gave_up = 1; return NULL; }
    uint32_t *code = malloc((size_t)size);
    if (!code) rt_fatal("vk: out of memory reading a mesh module");
    ok = fread(code, 1, (size_t)size, f) == (size_t)size && code[0] == 0x07230203;
    fclose(f);
    unsigned entries = 0;
    for (unsigned at = 5; ok && at < (unsigned)size / 4;) {
        unsigned len = code[at] >> 16, op = code[at] & 65535;
        if (!len || len > (unsigned)size / 4 - at) { ok = 0; break; }
        if (op == 15) {   // OpEntryPoint: TaskEXT 5364, MeshEXT 5365
            ok = len >= 4 && code[at + 1] == (stage == 6 ? 5364u : 5365u);
            entries++;
        }
        if (op == 71 && len >= 4 && code[at + 2] == 34 && code[at + 3]) ok = 0;   // DescriptorSet
        if (op == 71 && len >= 4 && code[at + 2] == 33 && code[at + 3] >= count) ok = 0;   // Binding
        at += len;
    }
    if (!ok || entries != 1) { free(code); m->gave_up = 1; return NULL; }
    m->native = native, m->count = count, m->words = (unsigned)size / 4, m->code = code;
    rt_log("mesh: loaded %s %016llx (%u descriptors, 2D array slots %#x)", stage == 7 ? "mesh" : "task", (unsigned long long)key, count, array_slots);
    return m;
}

// The compiler cannot see what a stage's texture slots hold, so the plain module takes them all as 2D. A slot bound to a
// 2D array needs a module compiled that way: this returns that variant (NULL while it compiles or if it will not).
static MeshModule *mesh_module_fit(VkExec *ex, MeshModule *m, const uint8_t *control, const uint8_t *source) {
    uint32_t slots = 0;
    for (unsigned i = 0; i < m->count; i++) {
        const MeshResource *r = &m->resources[i];
        if (r->kind != GFX_RES_TEX || r->hp) continue;   // a handle program's texture type is the compiler's own
        descriptor_state(ex);
        vk_texture_lock();
        uint8_t type = sampled_type_of_handle(ex, ex->tex[m->stage][r->reflection.slot]);
        vk_texture_unlock();
        if (type == GFX_TEX_2D_ARRAY) slots |= 1u << r->reflection.slot;
    }
    return slots ? mesh_module_get(m->key, m->native, m->stage, slots, control, source) : m;
}

static int mesh_resources_prepare(VkExec *ex, MeshModule *m) {
    if (!m) return 1;
    // Each holder is downloaded once even when several descriptors read different pointer entries from it.
    struct { unsigned offset, bytes; uint8_t data[520]; } tables[MESH_BINDINGS];
    unsigned ntables = 0, base = m->stage == 6 ? 0x670 : 0x770;
    for (unsigned i = 0; i < m->count; i++) {
        MeshResource *r = &m->resources[i];
        if (r->kind == GFX_RES_TEX && r->hp) continue;   // evaluated below, once the storage pointers are known
        if (r->kind == GFX_RES_TEX) {
            descriptor_state(ex);
            vk_texture_lock();
            uint8_t type = sampled_type_of_handle(ex, ex->tex[m->stage][r->reflection.slot]);
            vk_texture_unlock();
            if (type != r->texture_type) {
                return 0;
            }
        }
        if (r->kind != GFX_RES_SSBO) continue;
        unsigned offset = r->reflection.handle * 4;
        r->bound = ex->ssbo[m->stage][(offset - base) / 16];
        if (r->indirect == UINT32_MAX || !r->bound.addr) continue;
        unsigned k = 0;
        while (k < ntables && tables[k].offset != offset) k++;
        if (k == ntables) tables[ntables++] = (typeof(tables[0])){.offset = offset};
        if (tables[k].bytes < r->indirect + 8) tables[k].bytes = r->indirect + 8;
    }
    for (unsigned k = 0; k < ntables; k++) {
        p_bind_buf b = ex->ssbo[m->stage][(tables[k].offset - base) / 16];
        if (b.size && b.size < tables[k].bytes) return 0;
        // Holders are pointer tables the CPU writes, so guest memory has them without waiting for the GPU.
        if (!b.addr) return 0;
        memcpy(tables[k].data, (const void *)(uintptr_t)b.addr, tables[k].bytes);
    }
    for (unsigned i = 0; i < m->count; i++) {
        MeshResource *r = &m->resources[i];
        if (r->kind != GFX_RES_SSBO || r->indirect == UINT32_MAX) continue;
        unsigned k = 0;
        while (k < ntables && tables[k].offset != r->reflection.handle * 4u) k++;
        r->bound = (p_bind_buf){0};
        if (k < ntables) memcpy(&r->bound.addr, tables[k].data + r->indirect, 8);
    }
    for (unsigned i = 0; i < m->count; i++) {   // textures named by a handle program
        MeshResource *r = &m->resources[i];
        if (r->kind != GFX_RES_TEX || !r->hp) continue;
        if (!mesh_hp_eval(ex, m, r)) return 0;
        descriptor_state(ex);
        vk_texture_lock();
        uint8_t type = sampled_type_of_handle(ex, ex->tex[m->stage][r->reflection.slot]);
        vk_texture_unlock();
        if (type != r->texture_type) {
            static unsigned noted;
            if (noted++ < 8)
                rt_log("mesh: handle program texture of %016llx: handle %#llx is type %u, wanted %u: draw skipped",
                       (unsigned long long)m->key, (unsigned long long)ex->tex[m->stage][r->reflection.slot], type, r->texture_type);
            return 0;
        }
    }
    return 1;
}

static VkDescriptorBufferInfo mesh_resource_info(VkExec *ex, const VkPipeBinding *binding) {
    int stage = nvn_stage_of(binding->stage);
    const MeshResource *r = (const MeshResource *)binding->res;
    if (r->kind == GFX_RES_UBO) {
        if (r->reflection.slot == 1) {
            const uint8_t *control, *code;
            uint32_t begin, end, length;
            if (!mesh_program_source(ex->prog, stage, &control, &code)) return zero_info(65536);
            if (gfx_shader_is_nvn2(control)) {
                memcpy(&begin, control + 0x40, 4), memcpy(&end, control + 0x44, 4);
                memcpy(&length, control + 0x38, 4);
            } else {
                memcpy(&begin, control + 0x700, 4), memcpy(&end, control + 0x704, 4);
                length = end;
            }
            if (begin >= end || end > length || end - begin > 65536) return zero_info(65536);
            VkBuffer buffer; VkDeviceSize offset;
            memcpy(scratch(ex, end - begin, &buffer, &offset), code + begin, end - begin);
            return (VkDescriptorBufferInfo){buffer, offset, end - begin};
        }
        if (r->reflection.slot) return ubo_info(ex, stage, r->reflection.slot);
        uint8_t bank[MESH_C0_BYTES] = {0};
        // NVN 2 control stage tags are task 6 and mesh 7. Their observed driver storage tables start here.
        c0_table(bank + (stage == 6 ? 0x670 : 0x770), ex->ssbo[stage]);
        const float max_point = 2047.0f;
        memcpy(bank + 0x610, &max_point, 4);
        VkBuffer buffer; VkDeviceSize offset;
        memcpy(scratch(ex, sizeof bank, &buffer, &offset), bank, sizeof bank);
        return (VkDescriptorBufferInfo){buffer, offset, sizeof bank};
    }
    if (!__atomic_load_n(&LIM.ubo_max, __ATOMIC_ACQUIRE)) limits_init();
    uint64_t lead = r->bound.addr & 15u;
    uint64_t size = r->bound.size ? r->bound.size + lead : 0;
    // Read-only pointer-table pools may be up to 64 MiB (instance data can reach 12-18 MiB past the base); writable ones keep 8.
    uint64_t cap = r->indirect != UINT32_MAX && !(r->reflection.flags & 1) ? 64u << 20 : 8u << 20;
    uint64_t max = LIM.sbo_max < cap ? LIM.sbo_max : cap;
    VkDescriptorBufferInfo info;
    if (r->bound.addr && bind_range(ex, r->bound.addr - lead, size, LIM.sbo_align, max,
                                   VK_MUSE_SSBO, r->reflection.flags & 1, 1, &info)) {
        if (r->reflection.flags & 1) note_written(ex, r->bound.addr - lead, info.range);
        return info;
    }
    return zero_info(1);
}

static VkShaderModule mesh_shader_module(const MeshModule *m, unsigned offset) {
    uint32_t *code = malloc((size_t)m->words * 4);
    if (!code) rt_fatal("vk: out of memory rebinding a mesh module");
    memcpy(code, m->code, (size_t)m->words * 4);
    for (unsigned at = 5; at < m->words; at += code[at] >> 16)
        if ((code[at] & 65535) == 71 && (code[at] >> 16) >= 4 && code[at + 2] == 33) code[at + 3] += offset;
    VkShaderModuleCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                   .codeSize = (size_t)m->words * 4, .pCode = code};
    VkShaderModule module = VK_NULL_HANDLE;
    vkCreateShaderModule(vk_ctx()->device, &ci, NULL, &module);
    free(code);
    return module;
}

// One log line per pipeline: serial and mesh, task and fragment keys, the names the shader dumps use.
static void mesh_pipeline_note(const VkExec *ex, const char *what, const MeshModule *mesh, const MeshModule *task) {
    const uint8_t *control, *code;
    if (!ex || !ex->prog) {   // the boot prewarm (vk_exec_mesh_prewarm): no draw
        rt_log("mesh: %s (prewarm): mesh %016llx task %016llx", what, (unsigned long long)mesh->key,
               (unsigned long long)(task ? task->key : 0));
        return;
    }
    uint64_t fragment = mesh_program_source(ex->prog, NVN_STAGE_FS, &control, &code) ? mesh_source_key(control, code) : 0;
    rt_log("mesh: %s: program %llu mesh %016llx task %016llx fragment %016llx", what, (unsigned long long)ex->prog->serial,
           (unsigned long long)mesh->key, (unsigned long long)(task ? task->key : 0), (unsigned long long)fragment);
}

// Mesh pipelines are built on two builder threads, not in the replay: see README, "Mesh pipeline builds".
typedef struct MeshJob {
    struct MeshJob *next;
    MeshPipeline *p;
    VkPipeState state;
    VkShaderModule ms, ts;
    unsigned fragment_offset;
} MeshJob;
static struct {
    PlatMutex lock;
    PlatCond work, done;
    MeshJob *head, *tail;
    int started, queued_now;
    uint64_t queued, built, failed, slow, wait_frame, wait_ns;
    double max_ms;
} MA;

static int mesh_async_ms(void) {
    enum { ms = 20 };
    static int logged;   // the line goes out once
    if (!logged) {
        logged = 1;
        rt_log("mesh: pipelines built on mesh builder threads; a draw waits for its own at most %d ms", ms);
    }
    return ms;
}

static void mesh_build_main(void *arg) {
    (void)arg;
    plat_mutex_lock(&MA.lock);
    for (;;) {
        while (!MA.head) plat_cond_wait(&MA.work, &MA.lock);
        MeshJob *j = MA.head;
        if (!(MA.head = j->next)) MA.tail = NULL;
        plat_mutex_unlock(&MA.lock);
        uint64_t t0 = plat_now_ns();
        VkPipeline pipe = vk_pipeline_mesh_create(&j->state, j->p->info.layout, j->ms, j->ts, j->fragment_offset);
        double ms = (plat_now_ns() - t0) / 1e6;
        VkDevice device = vk_ctx()->device;
        if (j->ms) vkDestroyShaderModule(device, j->ms, NULL);
        if (j->ts) vkDestroyShaderModule(device, j->ts, NULL);
        plat_mutex_lock(&MA.lock);
        j->p->info.pipeline = pipe;
        __atomic_store_n(&j->p->building, 0, __ATOMIC_RELEASE);
        MA.queued_now--;
        if (pipe) MA.built++;
        else MA.failed++;
        if (ms > MA.max_ms) MA.max_ms = ms;
        if (ms >= 50) MA.slow++;
        plat_cond_broadcast(&MA.done);
        if (!pipe || ms >= 50)
            rt_log("mesh: builder %s mesh %016llx task %016llx fragment %016llx in %.1f ms; built %llu, failed %llu, over 50 ms "
                   "%llu, max %.1f ms, still queued %d, draws skipped while building %llu", pipe ? "built" : "FAILED",
                   (unsigned long long)j->p->keys[0], (unsigned long long)j->p->keys[1], (unsigned long long)j->p->keys[2], ms,
                   (unsigned long long)MA.built, (unsigned long long)MA.failed, (unsigned long long)MA.slow, MA.max_ms,
                   MA.queued_now, (unsigned long long)mesh_stat[MS_BUILDING]);
        free(j);
    }
}

// The replay's wait for a pipeline on a builder: until it is done, at most what is left of this frame's budget. 1 when
// done. A build whose wait ran out once is not waited for again, or every frame would spend its budget on the same build.
static int mesh_async_wait(MeshPipeline *p) {
    if (!__atomic_load_n(&p->building, __ATOMIC_ACQUIRE)) return 1;
    if (p->waited_out) return 0;
    uint64_t budget = (uint64_t)mesh_async_ms() * 1000000ull;
    if (MA.wait_frame != vk_frame_no) MA.wait_frame = vk_frame_no, MA.wait_ns = 0;
    if (MA.wait_ns >= budget) return 0;
    uint64_t t0 = plat_now_ns(), p0 = perf_begin();
    plat_mutex_lock(&MA.lock);
    while (__atomic_load_n(&p->building, __ATOMIC_ACQUIRE)) {
        uint64_t spent = plat_now_ns() - t0;
        if (MA.wait_ns + spent >= budget) break;
        plat_cond_wait_ns(&MA.done, &MA.lock, budget - MA.wait_ns - spent);
    }
    plat_mutex_unlock(&MA.lock);
    perf_end(PERF_PIPE, p0);
    MA.wait_ns += plat_now_ns() - t0;
    int done = !__atomic_load_n(&p->building, __ATOMIC_ACQUIRE);
    if (!done) p->waited_out = 1;
    return done;
}

// Queues the build; the job owns the modules. 0 when no builder thread could be started (the caller builds itself).
static int mesh_async_queue(MeshPipeline *p, const VkPipeState *state, VkShaderModule ms, VkShaderModule ts, unsigned fragment_offset) {
    if (!MA.started) {
        MA.started = 1;
        for (int i = 0; i < 2; i++)
            if (plat_thread_spawn(mesh_build_main, NULL, "mesh builder", 0)) MA.started = 2;
        if (MA.started != 2) rt_log("mesh: no mesh builder thread; pipelines are built in the replay");
    }
    if (MA.started != 2) return 0;
    MeshJob *j = calloc(1, sizeof *j);
    if (!j) rt_fatal("vk: out of memory queueing a mesh pipeline");
    j->p = p, j->state = *state, j->ms = ms, j->ts = ts, j->fragment_offset = fragment_offset;
    p->building = 1;
    plat_mutex_lock(&MA.lock);
    if (MA.tail) MA.tail->next = j;
    else MA.head = j;
    MA.tail = j;
    MA.queued++, MA.queued_now++;
    plat_cond_signal(&MA.work);
    plat_mutex_unlock(&MA.lock);
    return 1;
}

// ---- mesh pipeline notes: <mesh cache>/mesh-keys.bin, see README, "Mesh pipeline builds"
#define MESH_NOTES_VERSION 1
typedef struct {
    uint64_t mesh, mesh_native, task, task_native;
    uint32_t mesh_slots, task_slots, has_fs, pad;
    uint8_t fs[32];
    uint8_t state[sizeof(VkPipeState)];
} MeshNote;
typedef struct { char magic[8]; uint32_t version, rec_size; } MeshNotesHeader;
static struct { int ready, ok; char path[PLAT_PATH_MAX * 2]; uint64_t seen[4096]; uint32_t n; uint64_t written; } MN;
static uint64_t mesh_note_hash(const MeshNote *r) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < sizeof *r; i++) h = (h ^ ((const uint8_t *)r)[i]) * 0x100000001b3ull;
    return h ? h : 1;
}
static int mesh_note_seen(uint64_t h) {   // 1 when already there; adds it otherwise
    uint32_t j = (uint32_t)(h % 4096);
    for (uint32_t i = 0; i < 4096; i++, j = (j + 1) % 4096) {
        if (MN.seen[j] == h) return 1;
        if (!MN.seen[j]) { if (MN.n < 3600) MN.seen[j] = h, MN.n++; return 0; }
    }
    return 0;
}
// Reads the file (once): calls fn for each record when given; checks the header, starts the file again when it differs.
static void mesh_notes_open(void (*fn)(const MeshNote *)) {
    if (MN.ready) return;
    MN.ready = 1;
    const char *dir = mesh_cache_dir();
    if (!dir) return;
    snprintf(MN.path, sizeof MN.path, "%s/mesh-keys.bin", dir);
    MeshNotesHeader h, want = {"OOTMSHN", MESH_NOTES_VERSION, sizeof(MeshNote)};
    FILE *f = fopen(MN.path, "rb");
    if (f && fread(&h, sizeof h, 1, f) == 1 && !memcmp(&h, &want, sizeof h)) {
        MeshNote r;
        while (fread(&r, sizeof r, 1, f) == 1)
            if (!mesh_note_seen(mesh_note_hash(&r)) && fn) fn(&r);
        fclose(f);
        MN.ok = 1;
        return;
    }
    if (f) fclose(f);
    f = fopen(MN.path, "wb");
    if (f && fwrite(&want, sizeof want, 1, f) == 1) MN.ok = 1;
    if (f) fclose(f);
}
static void mesh_note_add(const VkPipeState *state, const MeshModule *mesh, const MeshModule *task) {
    mesh_notes_open(NULL);
    if (!MN.ok) return;
    MeshNote r;
    memset(&r, 0, sizeof r);
    r.mesh = mesh->key, r.mesh_native = mesh->native, r.mesh_slots = mesh->array_slots;
    if (task) r.task = task->key, r.task_native = task->native, r.task_slots = task->array_slots;
    if (state->fs) r.has_fs = 1, memcpy(r.fs, gfx_shader_input_id(state->fs), 32);
    VkPipeState s = *state;
    s.vs = s.gs = s.fs = NULL;
    memcpy(r.state, &s, sizeof s);
    if (mesh_note_seen(mesh_note_hash(&r))) return;
    FILE *f = fopen(MN.path, "ab");
    if (f) MN.written += fwrite(&r, sizeof r, 1, f), fclose(f);
}

// *building: set when the pipeline exists but its build is not done (the draw is skipped this time).
static MeshPipeline *mesh_pipeline_get(const VkExec *ex, const VkPipeState *state, MeshModule *mesh, MeshModule *task,
                                       int *building) {
    uint8_t key[32];
    *building = 0;
    if (!vk_pipeline_mesh_key(state, key)) return NULL;
    for (MeshPipeline *p = mesh_pipelines; p; p = p->next)
        if (p->mesh == mesh && p->task == task && !memcmp(p->state_key, key, sizeof key)) {
            if (__atomic_load_n(&p->building, __ATOMIC_ACQUIRE) && !mesh_async_wait(p)) return *building = 1, NULL;
            return p->info.pipeline ? p : NULL;
        }
    // Use the ordinary graphics builder's normalized state: it leaves out unused vertex inputs and dynamic depth/blend
    // values, which comparing all NVN state would also compare.
    if (1 + mesh->count + (task ? task->count : 0) > MAX_BINDINGS) return NULL;
    if (ex) mesh_note_add(state, mesh, task);   // for the next boot's prewarm
    MeshPipeline *p = calloc(1, sizeof *p);
    if (!p) rt_fatal("vk: out of memory creating a mesh pipeline");
    memcpy(p->state_key, key, sizeof key);
    p->mesh = mesh, p->task = task;
    p->next = mesh_pipelines, mesh_pipelines = p, mesh_pipeline_count++;
    p->info.bindings = p->bindings, p->info.bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS, p->info.n_sets = 1;
    p->bindings[p->info.n_bindings++] = (VkPipeBinding){.binding = 0, .stage = GFX_STAGE_MESH,
        .kind = VK_PIPE_SUPPORT, .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .count = 1, .stages = VK_SHADER_STAGE_FRAGMENT_BIT};
    unsigned offset = 1;
    MeshModule *modules[2] = {mesh, task};
    for (unsigned k = 0; k < 2; k++) if (modules[k]) {
        MeshModule *m = modules[k];
        for (unsigned i = 0; i < m->count; i++) {
            MeshResource *r = &m->resources[i];
            p->bindings[p->info.n_bindings++] = (VkPipeBinding){.binding = (uint16_t)(offset + i),
                .stage = m->stage == 7 ? GFX_STAGE_MESH : GFX_STAGE_TASK, .kind = (uint8_t)r->kind,
                .type = r->kind == GFX_RES_UBO ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : r->kind == GFX_RES_TEX ?
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .count = 1, .stages = m->stage == 7 ? VK_SHADER_STAGE_MESH_BIT_EXT : VK_SHADER_STAGE_TASK_BIT_EXT,
                .res = &r->reflection};
        }
        offset += m->count;
    }
    unsigned fragment_offset = offset - 1;
    int sampler_arrays = 0;
    for (int kind = 0; state->fs && kind < GFX_RES_KINDS; kind++) {
        const GfxShaderRes *resources;
        int count = gfx_shader_resources(state->fs, kind, &resources);
        for (int i = 0; i < count; i++) {
            const GfxShaderRes *r = &resources[i];
            if (p->info.n_bindings >= MAX_BINDINGS || !r->binding || r->set) {
                mesh_pipeline_note(ex, "too many or unplaced fragment resources; skipping pipeline", mesh, task);
                return NULL;
            }
            VkDescriptorType type = kind == GFX_RES_UBO ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER :
                kind == GFX_RES_SSBO ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : kind == GFX_RES_IMG ?
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : r->separate ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE :
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            if ((kind == GFX_RES_TEX || kind == GFX_RES_IMG) && (r->type & 0xff) == 2) {
                mesh_pipeline_note(ex, "fragment buffer textures are unsupported; skipping pipeline", mesh, task);
                return NULL;
            }
            const GfxShaderRes *bound = r;
            // Type 0 is the sampler half of the translator's bindless pair (an OpTypeSampler array beside a separate image
            // array, both indexed by a handle from game memory). It copies its resource with the image array's shadow bit.
            if (kind == GFX_RES_TEX && !(r->type & 0xff)) {
                GfxShaderRes *sampler = &p->samplers[p->info.n_bindings];
                *sampler = *r;
                for (int k = 0; k < count; k++)
                    if (resources[k].separate && resources[k].cbuf_slot == r->cbuf_slot && resources[k].handle == r->handle)
                        sampler->type |= resources[k].type & 0x400;
                type = VK_DESCRIPTOR_TYPE_SAMPLER, bound = sampler, sampler_arrays++;
            }
            p->bindings[p->info.n_bindings++] = (VkPipeBinding){.binding = (uint16_t)(fragment_offset + r->binding),
                .stage = GFX_STAGE_FS, .kind = (uint8_t)kind, .type = type, .count = r->count ? r->count : 1,
                .stages = VK_SHADER_STAGE_FRAGMENT_BIT, .res = bound};
        }
    }
    VkDescriptorSetLayoutBinding bindings[MAX_BINDINGS];
    for (unsigned i = 0; i < p->info.n_bindings; i++) {
        const VkPipeBinding *b = &p->bindings[i];
        bindings[i] = (VkDescriptorSetLayoutBinding){b->binding, b->type, b->count, b->stages, NULL};
    }
    VkCtx *c = vk_ctx();
    VkDescriptorSetLayoutCreateInfo si = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = p->info.n_bindings, .pBindings = bindings};
    if (vkCreateDescriptorSetLayout(c->device, &si, NULL, &p->info.set_layout) != VK_SUCCESS) return NULL;
    VkDescriptorSetLayout sets3[3] = {p->info.set_layout, vk_bindless_empty(), vk_bindless_layout()};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = sets3[2] ? 3 : 1, .pSetLayouts = sets3};
    if (vkCreatePipelineLayout(c->device, &li, NULL, &p->info.layout) != VK_SUCCESS) return NULL;
    VkShaderModule ms = mesh_shader_module(mesh, 1), ts = task ? mesh_shader_module(task, 1 + mesh->count) : VK_NULL_HANDLE;
    if (ms && (!task || ts)) {
        const uint8_t *control, *code;
        p->keys[0] = mesh->key, p->keys[1] = task ? task->key : 0;
        p->keys[2] = ex && mesh_program_source(ex->prog, NVN_STAGE_FS, &control, &code) ? mesh_source_key(control, code) : 0;
        if (mesh_async_queue(p, state, ms, ts, fragment_offset)) {
            if (sampler_arrays) mesh_pipeline_note(ex, "fragment sampler array pipeline queued", mesh, task);
            if (!ex || !mesh_async_wait(p)) return *building = 1, NULL;   // the boot prewarm does not wait
            if (!p->info.pipeline) mesh_pipeline_note(ex, "pipeline creation failed", mesh, task);
            return p->info.pipeline ? p : NULL;
        }
    }
    if (ms && (!task || ts)) p->info.pipeline = vk_pipeline_mesh_create(state, p->info.layout, ms, ts, fragment_offset);
    if (ms) vkDestroyShaderModule(c->device, ms, NULL);
    if (ts) vkDestroyShaderModule(c->device, ts, NULL);
    if (!p->info.pipeline) mesh_pipeline_note(ex, "pipeline creation failed", mesh, task);
    else if (sampler_arrays) mesh_pipeline_note(ex, "fragment sampler array pipeline built", mesh, task);
    return p->info.pipeline ? p : NULL;
}

// indirect: the guest record {count, first}; its count is copied on the GPU into a {count, 1, 1} record (a CPU read
// would wait for the GPU at every call). mesh_pair: see README, "Mesh depth and colour pass pairing".
static struct { uint64_t addr, frame; } MP[256];
static uint64_t mp_noted, mp_dropped;
// The per-object block: the mesh stage's c4 when it is small (0x100), else its c5 (c4 is the 0x4240-byte block every
// mesh draw binds).
static uint64_t mesh_pair_key(const VkExec *ex) {
    const p_bind_buf *c4 = &ex->ubo[7][1], *c5 = &ex->ubo[7][2];
    if (c4->addr && c4->size && c4->size <= 0x100) return c4->addr;
    return c5->addr;
}
static void mesh_pair_note(const VkExec *ex) {
    if (ex->pipe.ds.dw || ex->pipe.ds.df != 3) return;   // only colour passes: depth EQUAL, no depth write
    uint64_t a = mesh_pair_key(ex);
    if (!a) return;
    unsigned h = (unsigned)((a >> 8) * 0x9e3779b1u) & 255;
    for (unsigned i = 0; i < 8; i++) {
        unsigned k = (h + i) & 255;
        if (MP[k].addr == a || !MP[k].addr || MP[k].frame + 8 < vk_frame_no) { MP[k].addr = a, MP[k].frame = vk_frame_no; mp_noted++; return; }
    }
}
static int mesh_pair_drop(const VkExec *ex) {
    if (!ex->pipe.ds.dw) return 0;   // only pre-pass draws (depth write)
    uint64_t a = mesh_pair_key(ex);
    if (!a) return 0;
    unsigned h = (unsigned)((a >> 8) * 0x9e3779b1u) & 255;
    for (unsigned i = 0; i < 8; i++) {
        unsigned k = (h + i) & 255;
        if (MP[k].addr == a && MP[k].frame + 3 >= vk_frame_no) {   // vk_frame_no is the newest present: a few frames of slack
            if (!(mp_dropped++ & 4095)) rt_log("mesh: mesh_pair: %llu pre-pass draws skipped so far for objects whose colour draw was skipped (%llu notes)",
                                              (unsigned long long)mp_dropped, (unsigned long long)mp_noted);
            return 1;
        }
    }
    return 0;
}

static int mesh_draw_at(VkExec *ex, const p_draw_mesh *draw, uint64_t indirect) {
    if (!mesh_cache_dir() || !vk_ctx()->has_mesh_shader || ex->render_disabled || !ex->prog || draw->first || !draw->count)
        return 0;
    const uint8_t *ctl, *code;
    uint64_t src7, pk7;
    if (!mesh_stage_keys(ex->prog, 7, &ctl, &code, &src7, &pk7)) return mesh_stat[MS_OTHER]++, 0;
    // Native control byte 0x5f is the output primitive's vertex count: 3 in almost every mesh stage, 1 (points) in two.
    // tools/mesh always declares triangles, so such stages are drawn as triangles all the same (and show streaks).
    if (mesh_pair_drop(ex)) return 1;   // mesh_pair: this object's colour draw was skipped
    MeshModule *mesh = mesh_module_get(src7, pk7, 7, 0, ctl, code), *task = NULL;
    if (mesh) mesh = mesh_module_fit(ex, mesh, ctl, code);
    if (!mesh) {
        mesh_pair_note(ex);
        return mesh_stat[MS_NO_MESH]++, 0;
    }
    uint64_t src6, pk6;
    if (mesh_stage_keys(ex->prog, 6, &ctl, &code, &src6, &pk6)) {
        task = mesh_module_get(src6, pk6, 6, 0, ctl, code);
        if (task) task = mesh_module_fit(ex, task, ctl, code);
        // Task translation uses only MeshInitShader (thread count and payload size). MeshShaderA/B describe the
        // mesh stage, and a shared task program can legitimately feed mesh stages with different output limits.
        if (!task || !vk_ctx()->has_task_shader) return mesh_pair_note(ex), mesh_stat[MS_NO_TASK]++, 0;
        if (mesh->regs[2] != task->regs[2]) return mesh_stat[MS_INIT]++, 0;
    }
    static VkPhysicalDevice limits_device;
    static VkPhysicalDeviceMeshShaderPropertiesEXT limits;
    if (limits_device != vk_ctx()->phys) {
        limits = (VkPhysicalDeviceMeshShaderPropertiesEXT){.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 properties = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &limits};
        vkGetPhysicalDeviceProperties2(vk_ctx()->phys, &properties);
        limits_device = vk_ctx()->phys;
    }
    uint32_t max_x = task ? limits.maxTaskWorkGroupCount[0] : limits.maxMeshWorkGroupCount[0];
    uint32_t max_total = task ? limits.maxTaskWorkGroupTotalCount : limits.maxMeshWorkGroupTotalCount;
    if (draw->count > max_x || draw->count > max_total) return mesh_stat[MS_LIMIT]++, 0;
    if (!mesh_resources_prepare(ex, mesh) || !mesh_resources_prepare(ex, task)) return mesh_stat[MS_RESOURCE]++, 0;
    cb_res(ex); dummies_init();
    VkBuffer record = VK_NULL_HANDLE;
    VkDeviceSize record_offset = 0;
    if (indirect) {
        VkMemSpan source;
        if (!vk_mem_resolve(indirect, 8, &source) || (source.offset & 3)) return mesh_stat[MS_OTHER]++, 0;
        VK_MEM_USE(source);   // may move it to the mirror, the buffer the culling wrote
        if (!source.buffer) return mesh_stat[MS_OTHER]++, 0;
        uint32_t *arguments = (uint32_t *)scratch(ex, 16, &record, &record_offset);
        arguments[0] = 0, arguments[1] = arguments[2] = 1;
        vk_exec_barrier(ex);   // ends the pass; the culling's writes are visible to the copy
        if (!ind_clamp(ex, source.buffer, source.offset, 1, &record, &record_offset)) {   // a checked copy
            VkBufferCopy copy = {.srcOffset = source.offset, .dstOffset = record_offset, .size = 4};
            vkCmdCopyBuffer(ex->cb, source.buffer, record, 1, &copy);
            ex->recorded++;
            vk_exec_barrier(ex);
        }
        VK_MEM_BOUND(source);
    }
    if (!ex->in_pass) ex->dirty |= VK_EXEC_DIRTY_VIEWPORT | VK_EXEC_DIRTY_SCISSOR;
    vk_exec_pass_begin(ex);
    VkRect2D scissor = vk_exec_scissor(ex);
    if (!scissor.extent.width || !scissor.extent.height) return mesh_stat[MS_SCISSOR]++, 0;
    int present;
    const GfxShader *fragment = shader_for(ex, NVN_STAGE_FS, &present, NULL, 1);
    if (present && !fragment) return mesh_pair_note(ex), mesh_stat[MS_FS]++, 0;
    uint32_t feedback = feedback_prepare(ex, fragment);
    VkPipeState state = ex->pipe;
    state.vs = state.gs = NULL, state.fs = fragment, state.prim = NVN_DRAW_PRIMITIVE_TRIANGLES;
    state.depth_minus_w = nvn_depth_minus_w() != 0, state.y_flip = 1, state.prim_restart = 0;
    state.feedback = (uint8_t)((feedback & 0xff ? 1 : 0) | (feedback & 0x100 ? 2 : 0));
    static unsigned state_reports;
    if (state_reports++ < 4) rt_log("mesh: state depth test %u write %u function %u cull %u", state.ds.dt, state.ds.dw,
                                  state.ds.df, state.poly.cull);
    int building;
    MeshPipeline *pipeline = mesh_pipeline_get(ex, &state, mesh, task, &building);
    if (!pipeline) return mesh_pair_note(ex), mesh_stat[building ? MS_BUILDING : MS_PIPE]++, 0;
    bind_pipeline(ex, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->info.pipeline);
    vk_pipeline_set_dynamic(ex->cb, &state); B.st_valid = 0;
    rf_mesh = 1, record_fixed(ex), rf_mesh = 0;   // rf_mesh: record_fixed is recording a mesh draw
    bindless_bind(ex, &pipeline->info); bind_resources(ex, &pipeline->info);
    if (feedback & ex->fb_dirty) feedback_barrier(ex);
    feedback_written(ex, &state); vk_exec_oq_draw(ex);
    dlss_track_camera(ex, 7);   // the mesh stage's bank 3
    static PFN_vkCmdDrawMeshTasksEXT issue;
    if (!issue) issue = (PFN_vkCmdDrawMeshTasksEXT)vkGetDeviceProcAddr(vk_ctx()->device, "vkCmdDrawMeshTasksEXT");
    if (!issue) return 0;
    if (indirect) {
        static PFN_vkCmdDrawMeshTasksIndirectEXT issue_indirect;
        if (!issue_indirect) issue_indirect = (PFN_vkCmdDrawMeshTasksIndirectEXT)vkGetDeviceProcAddr(vk_ctx()->device, "vkCmdDrawMeshTasksIndirectEXT");
        if (!issue_indirect) return 0;
        rec_draw_mesh_indirect(issue_indirect, ex->cb, record, record_offset, 1, 12);
    } else rec_draw_mesh(issue, ex->cb, draw->count, 1, 1);
    ex->recorded++; vk_exec_frame.draws++;
    mesh_stat_issued(ex->depth.image ? ex->depth.width : 0, draw->count);
    static unsigned reports;
    if (reports++ < 8) rt_log("mesh: issued direct program %llu with %u task groups", (unsigned long long)ex->prog->serial, draw->count);
    return 1;
}

static int mesh_draw(VkExec *ex, const p_draw_mesh *draw) { return mesh_draw_at(ex, draw, 0); }

static int mesh_draw_indirect(VkExec *ex, const p_draw_mesh_ind *draw) {
    if (!mesh_cache_dir() || !vk_ctx()->has_mesh_shader || !ex->prog) return 0;
    const uint8_t *control, *code;
    uint64_t src7, pk7;
    if (!mesh_stage_keys(ex->prog, 7, &control, &code, &src7, &pk7) || !mesh_module_get(src7, pk7, 7, 0, control, code)) return 0;
    // SDK 22's direct draw writes the MME parameters as count, first: the GPU reads them (mesh_draw_at).
    mesh_stat[MS_IND_CALL]++;
    return mesh_draw_at(ex, &(p_draw_mesh){.first = 0, .count = 1, .caller = draw->caller}, draw->addr);
}

static void mesh_release(VkCtx *c) {
    while (mesh_pipelines) {
        MeshPipeline *p = mesh_pipelines; mesh_pipelines = p->next;
        if (p->info.pipeline) vkDestroyPipeline(c->device, p->info.pipeline, NULL);
        if (p->info.layout) vkDestroyPipelineLayout(c->device, p->info.layout, NULL);
        if (p->info.set_layout) vkDestroyDescriptorSetLayout(c->device, p->info.set_layout, NULL);
        free(p);
    }
    while (mesh_modules) { MeshModule *m = mesh_modules; mesh_modules = m->next; free(m->code); free(m); }
    mesh_pipeline_count = 0;
}

// At boot, every pipeline in mesh-keys.bin goes to the mesh builders. Modules come from the mesh cache only: a note
// whose module is not there is left out, as compiling it needs the guest's code.
static uint64_t mp_notes, mp_queued, mp_missing;
static void mesh_prewarm_one(const MeshNote *r) {
    mp_notes++;
    const char *dir = mesh_cache_dir();
    char path[PLAT_PATH_MAX * 2];
    MeshModule probe = {.key = r->mesh, .stage = 7, .array_slots = r->mesh_slots};
    mesh_module_path(path, sizeof path, dir, &probe, ".bindings");
    if (!plat_file_exists(path)) { mp_missing++; return; }
    if (r->task) {
        probe = (MeshModule){.key = r->task, .stage = 6, .array_slots = r->task_slots};
        mesh_module_path(path, sizeof path, dir, &probe, ".bindings");
        if (!plat_file_exists(path)) { mp_missing++; return; }
    }
    MeshModule *mesh = mesh_module_get(r->mesh, r->mesh_native, 7, r->mesh_slots, NULL, NULL);
    MeshModule *task = r->task ? mesh_module_get(r->task, r->task_native, 6, r->task_slots, NULL, NULL) : NULL;
    if (!mesh || (r->task && !task)) { mp_missing++; return; }
    VkPipeState s;
    memcpy(&s, r->state, sizeof s);
    s.vs = s.gs = s.fs = NULL;
    if (r->has_fs && !(s.fs = gfx_shader_load_input(r->fs, 0))) { mp_missing++; return; }
    int building = 0;
    mesh_pipeline_get(NULL, &s, mesh, task, &building);
    mp_queued += building;
}
void vk_exec_mesh_prewarm(void) {
    if (!mesh_cache_dir() || !vk_ctx()->has_mesh_shader) return;
    double t0 = (double)plat_now_ms();
    mesh_notes_open(mesh_prewarm_one);
    rt_log("mesh: prewarm: %llu noted pipelines in %s, %llu queued to the builders, %llu left out (module or fragment "
           "not cached) in %.0f ms", (unsigned long long)mp_notes, MN.path, (unsigned long long)mp_queued,
           (unsigned long long)mp_missing, (double)plat_now_ms() - t0);
}
