// NVN 2 program stage -> the NVN 1 shape the rest of the host takes (gfx_shader.c, vk_exec_draw.c): the bundled
// lowering library (runtime/third_party/eglantine) rewrites the Ampere code as Maxwell, laid out as the translator
// reads it, and a synthetic control block carries the fields the host reads.
#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "shader_recompiler/frontend/ampere/program_capture.h"

extern "C" void rt_log(const char *fmt, ...);
extern "C" uint64_t gfx_shader_nvn2_hash(const uint8_t *control, const uint8_t *code);

namespace {
constexpr size_t kControl = 0x880;   // gfx_shader.c CONTROL_SIZE
constexpr size_t kC0Layout = 0x860;  // "DKC1" + compute parameter layout (gfx_shader.c NVN2_C0_LAYOUT)
constexpr uint64_t kMaxwellNop = 0x50b0000000070f00ull;
constexpr uint64_t kMaxwellExit = 0xe30000000007000full;

struct Lowered { std::vector<uint8_t> control; std::vector<uint64_t> code; uint64_t origin, native_key; int kind, prim_vertices; bool tensor = false; };
std::mutex lock;
// Held across lower() alone (lowerings run one at a time); lock covers the tables only, so the replay's per-draw
// questions do not wait for a lowering.
std::mutex lower_lock;
// Keyed by content, not by address: a game frees program memory and loads other shaders at the same place. The key is
// gfx_shader_nvn2_hash (the dump name).
std::map<uint64_t, Lowered *> cache;
// Lowered code buffer -> its key, for nvn2_origin: the draw path asks it on every draw (vk_exec_draw.c
// light_quad_fault_skip), so it must not scan the cache.
std::unordered_map<const uint8_t *, uint64_t> origins;
std::atomic<uint64_t> origins_gen{1};   // bumped by every origin added: a cached "no origin" is stale once it moves
// Lowered code buffers of stages that use the tensor cores (HMMA, LDSM); without them each HMMA expands to dozens of
// shuffles.
std::unordered_set<const uint8_t *> tensor_stages;
bool uses_tensor(const std::vector<Shader::Ampere::Instruction> &native) {
    for (const auto &insn : native) {
        const uint32_t op = insn.Bits(0, 12);
        if (op == 0x23c || op == 0x83b) return true;   // HMMA.16816/.1688, LDSM
    }
    return false;
}

uint64_t native_key(const std::vector<Shader::Ampere::Instruction> &native) {
    uint64_t hash = 0xcbf29ce484222325ull;
    for (const auto &instruction : native)
        for (uint32_t word : instruction.words)
            for (unsigned byte = 0; byte < 4; byte++) hash = (hash ^ ((word >> (8 * byte)) & 255)) * 0x100000001b3ull;
    return hash;
}

void put32(std::vector<uint8_t> &c, size_t at, uint32_t v) { std::memcpy(c.data() + at, &v, 4); }

Lowered *lower(const uint8_t *control, const uint8_t *code) {
    uint32_t code_len, w0;
    std::memcpy(&code_len, control + 0x38, 4), std::memcpy(&w0, code, 4);
    const bool gfx = ((w0 & 31) == 1 || (w0 & 31) == 2) && ((w0 >> 5) & 31) == 4;   // SPH v4
    const uint32_t hdr = gfx ? 0x80 : 0;
    if (code_len <= hdr) {
        rt_log("nvn2: stage with %#x bytes of code (header %#x, word 0 %#x): nothing to lower", code_len, hdr, w0);
        return nullptr;
    }

    std::vector<Shader::Ampere::Instruction> native;
    Shader::Ampere::ProgramCaptureState capture;
    for (uint32_t at = hdr; at + 16 <= code_len; at += 16) {
        Shader::Ampere::Instruction insn;
        std::memcpy(insn.words.data(), code + at, 16);
        native.push_back(insn);
        const auto r = capture.Observe(insn, static_cast<uint32_t>(native.size() - 1));
        if (r != Shader::Ampere::ProgramCaptureState::Result::Continue) break;
    }
    // The SPH shader type picks the stage options: 4 geometry, 2 tessellation control, 3 tessellation evaluation. The
    // "full" matrix mode (tensor off) is what this host asks for.
    Shader::Ampere::LoweringOptions options;
    if (gfx) {
        const uint32_t sph_type = (w0 >> 10) & 15;
        options.geometry = sph_type == 4;
        if (sph_type == 2) options.tessellation_stage = Shader::Ampere::TessellationStage::Control;
        else if (sph_type == 3) options.tessellation_stage = Shader::Ampere::TessellationStage::Evaluation;
    }
    const auto lowered = Shader::Ampere::LowerToMaxwell(native, options);
    if (!lowered) {
        rt_log("nvn2: Ampere lowering failed at +%#x: %.*s (native %016llx, SPH type %u)",
               lowered.error_offset, (int)lowered.error.size(), lowered.error.data(),
               (unsigned long long)gfx_shader_nvn2_hash(control, code), gfx ? (w0 >> 10) & 15 : 0);
        return nullptr;
    }

    auto *out = new Lowered;
    out->kind = control[0x56];
    out->prim_vertices = control[0x5f];   // mesh stages: output primitive vertices (1 points, 3 triangles)
    out->native_key = native_key(native);
    out->tensor = uses_tensor(native);
    auto &m = out->code;
    // The lowering places instruction 0 at 0x10 modulo 32 with a scheduling word at every 32-byte boundary counted from
    // the decoder's start (graphics: SPH+0x50, compute: 0), so the code starts with a scheduling word and a NOP. For
    // graphics the 0x30 bytes the host skips and the first 0x50 bytes of the SPH come first.
    if (gfx) {
        m.assign(6, 0);
        m.resize(16);
        std::memcpy(m.data() + 6, code, 0x50);
        // Native geometry stages (vk_exec_draw.c, native_gs_control): the portable geometry contract reads a kind-1 version-3
        // header whose fields sit at the same offsets in the native one, so a lowered geometry stage is relabelled.
        if (((w0 >> 10) & 15) == 4) {
            const uint32_t relabelled = (w0 & ~uint32_t{0x3ff}) | 1u | 3u << 5;
            std::memcpy(m.data() + 6, &relabelled, 4);
        }
    }
    m.push_back(0), m.push_back(kMaxwellNop);
    size_t pos = 0x10;   // from the code start
    // A full scheduling group of EXITs closes the program: the decoder runs on until a block ends. The compiler's end
    // marker (an unconditional branch to itself after the last EXIT) becomes an EXIT too, or the endless loop hangs the GPU.
    for (size_t i = 0; i < lowered.instructions.size() + 4; i++) {
        if (pos % 32 == 0) m.push_back(0), pos += 8;
        uint64_t insn = i < lowered.instructions.size() ? lowered.instructions[i] : kMaxwellExit;
        if ((insn >> 52) == 0xe24 && ((insn >> 20) & 0xffffff) == 0xfffff8 && ((insn >> 16) & 15) == 7) insn = kMaxwellExit;
        m.push_back(insn), pos += 8;
    }

    // Inline constants (c[0x1]) follow the code in the NVN 2 buffer (control +0x40 to +0x44) and go after the Maxwell
    // code, where control +0x700 points. A BRX jump table in them holds native offsets (index * 16); the lowered BRX
    // jumps to Maxwell addresses, so those are rewritten.
    uint32_t c1_at, c1_end;
    std::memcpy(&c1_at, control + 0x40, 4), std::memcpy(&c1_end, control + 0x44, 4);
    const uint32_t c1_off = static_cast<uint32_t>(m.size() * 8);
    if (c1_end > c1_at && c1_end - c1_at <= 0x10000) {
        std::vector<uint32_t> c1((c1_end - c1_at) / 4);
        std::memcpy(c1.data(), code + c1_at, c1.size() * 4);
        bool brx = false;
        for (const auto &insn : native) brx |= insn.Bits(0, 12) == 0x949 || insn.Bits(0, 12) == 0x958;
        // Any 16-aligned word inside the program counts as a table entry, so an integer constant that looks like one is
        // rewritten too.
        if (brx)
            for (uint32_t &w : c1)
                if (w % 16 == 0 && w / 16 < lowered.source_instruction_offsets.size()) {
                    const uint32_t i = lowered.source_instruction_offsets[w / 16];
                    // The lowering's address_of (0x50-based). Instruction i sits at address_of(i) - 0x40 from the code
                    // start; the lowered BRX adds its base (-0x40) to the table value.
                    w = 0x50 + i * 8 + ((i + 1) / 3) * 8;
                }
        c1.resize((c1.size() + 1) & ~size_t{1});
        m.resize(m.size() + c1.size() / 2);
        std::memcpy(m.data() + c1_off / 8, c1.data(), c1.size() * 4);
    }

    out->control.assign(kControl, 0);
    const uint32_t len = static_cast<uint32_t>(m.size() * 8);
    const uint32_t type = (w0 >> 10) & 15;   // SPH shader type -> NVN stage
    const uint32_t stage = !gfx ? 5 : type == 5 ? 1 : type == 4 ? 2 : type == 2 ? 3 : type == 3 ? 4 : 0;
    put32(out->control, 0x700, c1_off);
    put32(out->control, 0x704, len);
    put32(out->control, 0x714, stage);
    if (!gfx) {
        std::memcpy(out->control.data() + 0x748, control + 0x58, 12);   // local size (NVN 2 control +0x58)
        // The compute driver bank is packed per program from these counts (gfx_shader.c, "NVN 2 compute c0"): images
        // +0xb6, uniform buffers +0xb1, storage buffers +0xb2, textures +0xb3. DKC1 adds the eight root-table word counts;
        // the marker and counts enter the code hash.
        const uint8_t lay[8] = {'D', 'K', 'C', '1', control[0xb6], control[0xb1], control[0xb2], control[0xb3]};
        std::memcpy(out->control.data() + kC0Layout, lay, sizeof lay);
        // Shared memory bytes (NVN 2 control +0x64): the translator sizes the workgroup's shared array from it.
        std::memcpy(out->control.data() + kC0Layout + 8, control + 0x64, 4);
        std::memcpy(out->control.data() + kC0Layout + 12, control + 0xa9, 8);
    }
    return out;
}
}  // namespace

// 1 and the replacement control and code for an NVN 2 stage, 0 when it does not lower.
extern "C" int nvn2_lower(const uint8_t *control, const uint8_t *code, const uint8_t **out_control,
                          const uint8_t **out_code) {
    const uint64_t key = gfx_shader_nvn2_hash(control, code);
    if (!key) {
        uint32_t w[6];
        std::memcpy(w, control + 0x30, sizeof w);
        rt_log("nvn2: stage with a control block of unexpected sizes (+0x30 %#x +0x38 %#x +0x44 %#x, code %p): not "
               "lowered", w[0], w[2], w[5], (const void *)code);
        return 0;
    }
    auto publish = [&](Lowered *l) {   // lock held: the result in the tables (nullptr: does not lower)
        cache.emplace(key, l);
        if (!l) return;
        l->origin = key;
        origins.emplace(reinterpret_cast<const uint8_t *>(l->code.data()), key);
        origins_gen.fetch_add(1, std::memory_order_release);
        if (l->tensor) tensor_stages.insert(reinterpret_cast<const uint8_t *>(l->code.data()));
    };
    auto answer = [&](const Lowered *l) {
        if (!l) return 0;
        *out_control = l->control.data();
        *out_code = reinterpret_cast<const uint8_t *>(l->code.data());
        return 1;
    };
    {
        std::lock_guard<std::mutex> g(lock);
        auto it = cache.find(key);
        if (it != cache.end()) return answer(it->second);
    }
    std::lock_guard<std::mutex> lg(lower_lock);
    {
        std::lock_guard<std::mutex> g(lock);   // another thread may have lowered it while this one waited
        auto it = cache.find(key);
        if (it != cache.end()) return answer(it->second);
    }
    rt_log("nvn2: lowering stage %016llx", (unsigned long long)key);
    Lowered *l = lower(control, code);
    std::lock_guard<std::mutex> g(lock);
    publish(l);
    return answer(l);
}

// The dump name (gfx_shader_nvn2_hash) of the NVN 2 stage a lowered code buffer came from, or 0. Lowered buffers live
// for the process, so their addresses never name another stage.
// Changes whenever nvn2_origin may give a non-zero answer where it gave 0 before (vk_exec_draw.c stage_origin).
extern "C" uint64_t nvn2_origin_gen(void) { return origins_gen.load(std::memory_order_acquire); }

// A per-thread direct-mapped memo in front of the table: the replay asks several times per draw. An origin never
// changes; "none" is asked again once a lowering has been published since (origins_gen).
extern "C" uint64_t nvn2_origin(const uint8_t *code) {
    struct Memo { const uint8_t *code; uint64_t origin, gen; };
    static thread_local Memo memo[1024];
    Memo &m = memo[(reinterpret_cast<uintptr_t>(code) * 0x9E3779B97F4A7C15ull) >> 54];
    const uint64_t gen = origins_gen.load(std::memory_order_acquire);
    if (code && m.code == code && (m.origin || m.gen == gen)) return m.origin;
    uint64_t o;
    {
        std::lock_guard<std::mutex> g(lock);
        auto it = origins.find(code);
        o = it == origins.end() ? 0 : it->second;
    }
    m.code = code, m.origin = o, m.gen = gen;
    return o;
}

extern "C" int nvn2_native_kind(const uint8_t *control, const uint8_t *code) {
    if (!control || !code) return -1;
    {
        std::lock_guard<std::mutex> g(lock);
        const auto it = origins.find(code);
        if (it != origins.end()) return cache.at(it->second)->kind;
    }
    uint32_t magic, version;
    std::memcpy(&magic, control, 4), std::memcpy(&version, control + 4, 4);
    return magic == 0x98761234 && version == 201 ? control[0x56] : -1;
}

extern "C" int nvn2_native_prim_vertices(const uint8_t *control, const uint8_t *code) {
    if (!control || !code) return -1;
    {
        std::lock_guard<std::mutex> g(lock);
        const auto it = origins.find(code);
        if (it != origins.end()) return cache.at(it->second)->prim_vertices;
    }
    uint32_t magic, version;
    std::memcpy(&magic, control, 4), std::memcpy(&version, control + 4, 4);
    return magic == 0x98761234 && version == 201 ? control[0x5f] : -1;
}

extern "C" uint64_t nvn2_program_key(const uint8_t *control, const uint8_t *code) {
    if (!control || !code) return 0;
    {
        std::lock_guard<std::mutex> g(lock);
        const auto it = origins.find(code);
        if (it != origins.end()) return cache.at(it->second)->native_key;
    }
    if (!gfx_shader_nvn2_hash(control, code)) return 0;
    uint32_t len, w0;
    std::memcpy(&len, control + 0x38, 4), std::memcpy(&w0, code, 4);
    uint32_t header = ((w0 & 31) == 1 || (w0 & 31) == 2) && ((w0 >> 5) & 31) == 4 ? 0x80 : 0;
    Shader::Ampere::ProgramCaptureState capture;
    std::vector<Shader::Ampere::Instruction> native;
    for (uint32_t off = header; off + 16 <= len; off += 16) {
        Shader::Ampere::Instruction instruction;
        std::memcpy(instruction.words.data(), code + off, 16);
        native.push_back(instruction);
        auto result = capture.Observe(instruction, static_cast<uint32_t>(native.size() - 1));
        if (result == Shader::Ampere::ProgramCaptureState::Result::InvalidTarget) return 0;
        if (result == Shader::Ampere::ProgramCaptureState::Result::Complete) return native_key(native);
    }
    return 0;
}

// 1 when a lowered code buffer is a stage that uses the tensor cores (HMMA/LDSM).
extern "C" int nvn2_is_tensor(const uint8_t *code) {
    std::lock_guard<std::mutex> g(lock);
    return tensor_stages.count(code) != 0;
}
