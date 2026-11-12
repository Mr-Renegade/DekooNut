// Runtime repair of a translator gap: a graphics stage's global load through a pointer read from memory matches no bound
// buffer, so its helper is `return 0`. The rewrite gives such a helper the body patch 0033 emits (see README, "Global
// loads in graphics stages"). Only modules with a `return 0` helper of two int pointers change, so other pipelines
// stay cached. Limit: the helper takes 32-bit reads (every module seen reads floats; not proven for all).
#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
    SGS_Name = 5, SGS_Extension = 10, SGS_ExtInstImport = 11, SGS_MemoryModel = 14, SGS_Capability = 17, SGS_TypeBool = 20,
    SGS_TypeInt = 21, SGS_TypeVector = 23, SGS_TypeStruct = 30, SGS_TypePointer = 32, SGS_Constant = 43, SGS_Function = 54,
    SGS_FunctionParameter = 55, SGS_FunctionEnd = 56, SGS_FunctionCall = 57, SGS_Variable = 59, SGS_Load = 61,
    SGS_AccessChain = 65, SGS_CompositeConstruct = 80, SGS_Bitcast = 124, SGS_ULessThan = 176, SGS_ShiftRightLogical = 194,
    SGS_ShiftLeftLogical = 196, SGS_BitwiseOr = 197, SGS_BitwiseAnd = 199, SGS_Select = 169, SGS_IEqual = 170, SGS_Label = 248,
    SGS_ReturnValue = 254,
    SGS_StorageUniform = 2, SGS_StoragePhysical = 5349, SGS_AddressingPhysical = 5348, SGS_CapPhysical = 5347,
    SGS_AccessAligned = 2,
    SGS_GmemLow = 9, SGS_GmemHigh = 10,                // members of the translator's SupportBuffer (patch 0033)
    SGS_MaxStubs = 8, SGS_BodyWords = 200,
};

typedef struct { uint64_t ret; uint32_t a, b, type_int; } SgsStub;

// Returns a new module and its size, or NULL to keep the module. `stubs` says how many helpers got a body.
static uint32_t *spv_gmem_stub_fix(const uint32_t *w, uint64_t words, uint64_t *out_words, unsigned *stubs) {
    if (words < 5 || w[0] != 0x07230203) return NULL;
    uint32_t bound = w[3];
    uint32_t tuint = 0, tbool = 0, tv2 = 0, tptr = 0, support = 0, support_type = 0, support_ptr = 0;
    uint64_t first_fn = 0, cap_at = 0, ext_at = 0, model_at = 0;
    int physical = 0;
    SgsStub st[SGS_MaxStubs];
    unsigned n_stub = 0;
    uint8_t *is_zero = calloc(bound, 1), *called = calloc(bound, 1);
    uint32_t *out = NULL;
    if (!is_zero || !called) goto done;

    for (uint64_t p = 5; p < words;) {
        uint32_t op = w[p] & 0xffff, n = w[p] >> 16;
        if (!n || p + n > words) goto done;
        if (op == SGS_Capability) {
            if (!cap_at) cap_at = p;
            if (w[p + 1] == SGS_CapPhysical) physical = 1;
        }
        if ((op == SGS_ExtInstImport || op == SGS_MemoryModel) && !ext_at) ext_at = p;
        if (op == SGS_MemoryModel) model_at = p;
        if (op == SGS_Name && !memcmp(&w[p + 2], "support_buffer", 15) && n >= 6) support = w[p + 1];
        if (op == SGS_TypeBool) tbool = w[p + 1];
        if (op == SGS_TypeInt && w[p + 2] == 32 && !w[p + 3]) tuint = w[p + 1];
        if (op == SGS_TypeVector && w[p + 3] == 2 && w[p + 2] == tuint && tuint) tv2 = w[p + 1];
        if (op == SGS_TypePointer && w[p + 2] == SGS_StorageUniform && w[p + 3] == tuint && tuint) tptr = w[p + 1];
        if (op == SGS_Constant && n == 4 && !w[p + 3] && w[p + 2] < bound) is_zero[w[p + 2]] = 1;
        if (op == SGS_Variable && w[p + 2] == support) support_ptr = w[p + 1];
        if (op == SGS_FunctionCall && w[p + 3] < bound) called[w[p + 3]] = 1;
        if (op == SGS_Function && !first_fn) first_fn = p;
        p += n;
    }
    if (physical || !support || !support_ptr || !tuint || !tbool || !first_fn || !cap_at || !ext_at || !model_at) goto done;
    // The support buffer's struct must have gmem_low and gmem_high, as uint.
    for (uint64_t p = 5; p < first_fn; p += w[p] >> 16)
        if ((w[p] & 0xffff) == SGS_TypePointer && w[p + 1] == support_ptr) support_type = w[p + 3];
    int fields = 0;
    for (uint64_t p = 5; p < first_fn; p += w[p] >> 16)
        if ((w[p] & 0xffff) == SGS_TypeStruct && w[p + 1] == support_type && (w[p] >> 16) >= 2 + SGS_GmemHigh + 1)
            fields = w[p + 2 + SGS_GmemLow] == tuint && w[p + 2 + SGS_GmemHigh] == tuint;
    if (!fields) goto done;

    // A helper: OpFunction, two parameters, OpLabel, OpReturnValue of an int constant 0, OpFunctionEnd.
    for (uint64_t p = first_fn; p < words; p += w[p] >> 16) {
        if ((w[p] & 0xffff) != SGS_Function) continue;
        uint64_t a = p + 5, b = a + 3, l = b + 3, r = l + 2, e = r + 2;
        if (e >= words || !called[w[p + 2]] || (w[a] & 0xffff) != SGS_FunctionParameter || (w[b] & 0xffff) != SGS_FunctionParameter ||
            (w[l] & 0xffff) != SGS_Label || (w[r] & 0xffff) != SGS_ReturnValue || (w[e] & 0xffff) != SGS_FunctionEnd ||
            w[r + 1] >= bound || !is_zero[w[r + 1]]) continue;
        if (n_stub == SGS_MaxStubs) goto done;
        st[n_stub++] = (SgsStub){r, w[a + 2], w[b + 2], w[p + 1]};
    }
    if (!n_stub) goto done;

    out = malloc((words + 96 + (uint64_t)n_stub * SGS_BodyWords) * 4);
    if (!out) goto done;
    uint32_t next = bound, ptr = next++, k[11];
    static const uint32_t kv[11] = {0x7f, 20, 12, 0x10, (1u << 24) - 1, 2, 0xffc, SGS_GmemLow, SGS_GmemHigh, ((1u << 24) - 1) << 2, 0};
    for (int i = 0; i < 11; i++) k[i] = next++;
    enum { K7f, K20, K12, K10, KMax, K2, KFfc, KLow, KHigh, KSinkOff, KZero };
    int new_v2 = !tv2, new_ptr = !tptr;
    if (new_v2) tv2 = next++;
    if (new_ptr) tptr = next++;
    uint64_t o = 5;
    memcpy(out, w, 20);
    for (uint64_t p = 5; p < words;) {
        uint32_t n = w[p] >> 16;
        if (p == cap_at) out[o++] = 2u << 16 | SGS_Capability, out[o++] = SGS_CapPhysical;
        if (p == ext_at) {
            static const char name[32] = "SPV_KHR_physical_storage_buffer";
            out[o++] = 9u << 16 | SGS_Extension;
            memcpy(&out[o], name, 32), o += 8;
        }
        if (p == first_fn) {
            if (new_v2) out[o++] = 4u << 16 | SGS_TypeVector, out[o++] = tv2, out[o++] = tuint, out[o++] = 2;
            if (new_ptr) out[o++] = 4u << 16 | SGS_TypePointer, out[o++] = tptr, out[o++] = SGS_StorageUniform, out[o++] = tuint;
            out[o++] = 4u << 16 | SGS_TypePointer, out[o++] = ptr, out[o++] = SGS_StoragePhysical, out[o++] = tuint;
            for (int i = 0; i < 11; i++) out[o++] = 4u << 16 | SGS_Constant, out[o++] = tuint, out[o++] = k[i], out[o++] = kv[i];
        }
        const SgsStub *s = NULL;
        for (unsigned i = 0; i < n_stub; i++) if (st[i].ret == p) s = &st[i];
        if (!s) {
            memcpy(&out[o], &w[p], (size_t)n * 4);
            if (p == model_at) out[o + 1] = SGS_AddressingPhysical;
            o += n, p += n;
            continue;
        }
#define SGS_OP2(code, type, a_, b_) (out[o++] = 5u << 16 | (code), out[o++] = (type), out[o++] = next, out[o++] = (a_), out[o++] = (b_), next++)
#define SGS_OP1(code, type, a_) (out[o++] = 4u << 16 | (code), out[o++] = (type), out[o++] = next, out[o++] = (a_), next++)
        uint32_t lo_i = SGS_OP1(SGS_Load, s->type_int, s->a), hi_i = SGS_OP1(SGS_Load, s->type_int, s->b);
        uint32_t lo = SGS_OP1(SGS_Bitcast, tuint, lo_i), hi = SGS_OP1(SGS_Bitcast, tuint, hi_i);
        uint32_t table_lo_p = (out[o++] = 5u << 16 | SGS_AccessChain, out[o++] = tptr, out[o++] = next, out[o++] = support,
                               out[o++] = k[KLow], next++);
        uint32_t table_hi_p = (out[o++] = 5u << 16 | SGS_AccessChain, out[o++] = tptr, out[o++] = next, out[o++] = support,
                               out[o++] = k[KHigh], next++);
        uint32_t table_lo = SGS_OP1(SGS_Load, tuint, table_lo_p), table_hi = SGS_OP1(SGS_Load, tuint, table_hi_p);
        uint32_t high = SGS_OP2(SGS_BitwiseAnd, tuint, hi, k[K7f]);
        uint32_t high_bits = SGS_OP2(SGS_ShiftLeftLogical, tuint, high, k[K20]);
        uint32_t low_bits = SGS_OP2(SGS_ShiftRightLogical, tuint, lo, k[K12]);
        uint32_t index = SGS_OP2(SGS_BitwiseOr, tuint, high_bits, low_bits);
        uint32_t in = SGS_OP2(SGS_ULessThan, tbool, high, k[K10]);
        uint32_t pick = (out[o++] = 6u << 16 | SGS_Select, out[o++] = tuint, out[o++] = next, out[o++] = in, out[o++] = index,
                         out[o++] = k[KMax], next++);
        uint32_t entry_off = SGS_OP2(SGS_ShiftLeftLogical, tuint, pick, k[K2]);
        uint32_t entry_lo = SGS_OP2(SGS_BitwiseOr, tuint, table_lo, entry_off);
        uint32_t entry_addr = SGS_OP2(SGS_CompositeConstruct, tv2, entry_lo, table_hi);
        uint32_t entry_p = SGS_OP1(SGS_Bitcast, ptr, entry_addr);
        uint32_t entry = (out[o++] = 6u << 16 | SGS_Load, out[o++] = tuint, out[o++] = next, out[o++] = entry_p,
                          out[o++] = SGS_AccessAligned, out[o++] = 4, next++);
        uint32_t page = SGS_OP2(SGS_ShiftLeftLogical, tuint, entry, k[K12]);
        uint32_t in_page = SGS_OP2(SGS_BitwiseAnd, tuint, lo, k[KFfc]);
        uint32_t host_lo = SGS_OP2(SGS_BitwiseOr, tuint, page, in_page);
        uint32_t host_hi = SGS_OP2(SGS_ShiftRightLogical, tuint, entry, k[K20]);
        uint32_t value_addr = SGS_OP2(SGS_CompositeConstruct, tv2, host_lo, host_hi);
        uint32_t value_p = SGS_OP1(SGS_Bitcast, ptr, value_addr);
        uint32_t value = (out[o++] = 6u << 16 | SGS_Load, out[o++] = tuint, out[o++] = next, out[o++] = value_p,
                          out[o++] = SGS_AccessAligned, out[o++] = 4, next++);
        // The last entry is the sink page, shared by everything unmapped: wild stores land there, so it reads as 0 here.
        uint32_t sink_lo = SGS_OP2(SGS_BitwiseOr, tuint, table_lo, k[KSinkOff]);
        uint32_t sink_addr = SGS_OP2(SGS_CompositeConstruct, tv2, sink_lo, table_hi);
        uint32_t sink_p = SGS_OP1(SGS_Bitcast, ptr, sink_addr);
        uint32_t sink = (out[o++] = 6u << 16 | SGS_Load, out[o++] = tuint, out[o++] = next, out[o++] = sink_p,
                         out[o++] = SGS_AccessAligned, out[o++] = 4, next++);
        uint32_t unmapped = SGS_OP2(SGS_IEqual, tbool, entry, sink);
        uint32_t safe = (out[o++] = 6u << 16 | SGS_Select, out[o++] = tuint, out[o++] = next, out[o++] = unmapped, out[o++] = k[KZero],
                         out[o++] = value, next++);
        uint32_t result = SGS_OP1(SGS_Bitcast, s->type_int, safe);
        out[o++] = 2u << 16 | SGS_ReturnValue, out[o++] = result;
#undef SGS_OP1
#undef SGS_OP2
        p += n;
    }
    out[3] = next;
    *out_words = o, *stubs = n_stub;
done:
    free(is_zero), free(called);
    return out;
}
