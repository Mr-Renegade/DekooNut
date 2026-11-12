// Runtime repair of one translator mistake, so that it needs no new shader preparation. The translator writes every 32-bit
// FFMA as select(a == 0 || b == 0, c, fma(a, b, c)); the zero branch returns c, but hardware returns +0 + c, which differs for
// c = -0 and shows in 1 / fma(0, x, -0): +inf on the console, -inf here (dissolve materials at rest lose every pixel).
// The rewrite touches only selects whose value reaches the denominator of 1 / x and whose c is a sign flip: c becomes
// (bits(c) == 0x80000000 ? +0 : c), tested on the bits because a compiler may drop x + 0 or a float compare. Other modules
// stay byte for byte the same, so their pipelines stay cached.
// Limit: an FFMA without .FMZ and with a negative product gives -0 on hardware, which this turns into +0.
#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
    SFZ_ExtInstImport = 11, SFZ_ExtInst = 12, SFZ_TypeBool = 20, SFZ_TypeInt = 21, SFZ_TypeFloat = 22,
    SFZ_Constant = 43, SFZ_Function = 54, SFZ_Load = 61, SFZ_Store = 62, SFZ_Bitcast = 124, SFZ_FNegate = 127,
    SFZ_FDiv = 136, SFZ_Select = 169, SFZ_IEqual = 170, SFZ_BitwiseXor = 198, SFZ_GlslFma = 50,
};

typedef struct {
    const uint32_t *w;
    const uint32_t *def;   // by id: word index of the instruction that defines it, for the opcodes read here
    const uint32_t *held;  // by variable id: the value last stored in it (the translator stores its locals once)
    uint8_t *select;       // by id: this select gets the repair
    uint32_t tfloat, glsl;
    unsigned n;
} SfzScan;

// Is this value -x, written as a float negate or as an xor of the sign bit, perhaps by way of variables?
static int sfz_is_flip(const SfzScan *z, uint32_t id) {
    const uint32_t *w = z->w;
    for (int hop = 0; hop < 8; hop++) {
        uint32_t p = z->def[id];
        if (!p) return 0;
        uint32_t op = w[p] & 0xffff;
        if (op == SFZ_FNegate) return 1;
        if (op == SFZ_BitwiseXor) {
            uint32_t a = z->def[w[p + 3]], b = z->def[w[p + 4]];
            return (a && (w[a] & 0xffff) == SFZ_Constant && w[a + 3] == 0x80000000u) ||
                   (b && (w[b] & 0xffff) == SFZ_Constant && w[b + 3] == 0x80000000u);
        }
        if (op == SFZ_Bitcast) id = w[p + 3];
        else if (op == SFZ_Load) id = z->held[w[p + 3]];
        else return 0;
    }
    return 0;
}

static void sfz_mark(SfzScan *z, uint32_t id) {
    const uint32_t *w = z->w;
    uint32_t p = z->def[id];
    if (!p || (w[p] & 0xffff) != SFZ_Select || w[p + 1] != z->tfloat || z->select[id]) return;
    uint32_t f = z->def[w[p + 5]];
    if (!f || (w[f] & 0xffff) != SFZ_ExtInst || w[f + 3] != z->glsl || w[f + 4] != SFZ_GlslFma) return;
    if (sfz_is_flip(z, w[p + 4])) z->select[id] = 1, z->n++;
}

// The repaired module (malloc) and its size in words; NULL when there is nothing to repair or the module is not
// understood. *fixed: selects repaired.
static uint32_t *spv_fma_zero_fix(const uint32_t *w, uint64_t words, uint64_t *out_words, unsigned *fixed) {
    *fixed = 0;
    if (words < 6 || w[0] != 0x07230203 || w[3] < 2 || w[3] > (1u << 26)) return NULL;
    uint32_t bound = w[3], tfloat = 0, tuint = 0, tbool = 0, glsl = 0, first_fn = 0;
    uint32_t *def = calloc(bound, 4), *held = calloc(bound, 4);
    uint8_t *var = calloc(bound, 1), *select = calloc(bound, 1);
    uint32_t *out = NULL;
    SfzScan z = {w, def, held, select, 0, 0, 0};
    if (!def || !held || !var || !select) goto done;

    for (uint64_t p = 5; p < words;) {
        uint32_t op = w[p] & 0xffff, n = w[p] >> 16;
        if (!n || p + n > words) goto done;
        uint32_t id = 0;
        switch (op) {
        case SFZ_ExtInstImport:
            if (n >= 5 && !memcmp(&w[p + 2], "GLSL.std.450", 12)) glsl = w[p + 1];
            break;
        case SFZ_TypeBool: tbool = w[p + 1]; break;
        case SFZ_TypeInt: if (w[p + 2] == 32 && !w[p + 3]) tuint = w[p + 1]; break;
        case SFZ_TypeFloat: if (w[p + 2] == 32) tfloat = w[p + 1]; break;
        case SFZ_Function: if (!first_fn) first_fn = (uint32_t)p; break;
        case SFZ_Store: if (w[p + 1] < bound) held[w[p + 1]] = w[p + 2]; break;
        case SFZ_Constant: case SFZ_ExtInst: case SFZ_Load: case SFZ_Select: case SFZ_Bitcast: case SFZ_FNegate:
        case SFZ_BitwiseXor:
            id = w[p + 2];
            break;
        default: break;
        }
        if (id) {
            if (id >= bound) goto done;
            def[id] = (uint32_t)p;
        }
        p += n;
    }
    if (!tfloat || !tuint || !tbool || !glsl || !first_fn) goto done;
    z.tfloat = tfloat, z.glsl = glsl;

    for (uint64_t p = 5; p < words; p += w[p] >> 16) {   // denominators of 1 / x
        if ((w[p] & 0xffff) != SFZ_FDiv || w[p + 1] != tfloat) continue;
        uint32_t num = def[w[p + 3]], den = w[p + 4], d = def[den];
        if (!num || (w[num] & 0xffff) != SFZ_Constant || w[num + 3] != 0x3f800000 || !d) continue;
        if ((w[d] & 0xffff) == SFZ_Load) var[w[d + 3]] = 1;
        else sfz_mark(&z, den);
    }
    for (int again = 1, pass = 0; again && pass < 8; pass++) {   // the values those variables hold
        again = 0;
        for (uint64_t p = 5; p < words; p += w[p] >> 16) {
            if ((w[p] & 0xffff) != SFZ_Store || !var[w[p + 1]]) continue;
            uint32_t v = w[p + 2], d = def[v];
            if (!d) continue;
            if ((w[d] & 0xffff) == SFZ_Load) {
                if (!var[w[d + 3]]) var[w[d + 3]] = 1, again = 1;
            } else sfz_mark(&z, v);
        }
    }
    if (!z.n) goto done;

    uint64_t size = words + 8 + 15ull * z.n;
    out = malloc(size * 4);
    if (!out) goto done;
    uint32_t next = bound, ksign = next++, kzero = next++;
    uint64_t o = 5;
    memcpy(out, w, 20);
    for (uint64_t p = 5; p < words;) {
        uint32_t op = w[p] & 0xffff, n = w[p] >> 16;
        if (p == first_fn) {
            out[o++] = 4u << 16 | SFZ_Constant, out[o++] = tuint, out[o++] = ksign, out[o++] = 0x80000000u;
            out[o++] = 4u << 16 | SFZ_Constant, out[o++] = tfloat, out[o++] = kzero, out[o++] = 0;
        }
        memcpy(&out[o], &w[p], (size_t)n * 4);
        if (op == SFZ_Select && select[w[p + 2]]) {
            uint32_t bits = next++, neg = next++, c = next++, t = w[p + 4];
            out[o++] = 4u << 16 | SFZ_Bitcast, out[o++] = tuint, out[o++] = bits, out[o++] = t;
            out[o++] = 5u << 16 | SFZ_IEqual, out[o++] = tbool, out[o++] = neg, out[o++] = bits, out[o++] = ksign;
            out[o++] = 6u << 16 | SFZ_Select, out[o++] = tfloat, out[o++] = c, out[o++] = neg, out[o++] = kzero,
                out[o++] = t;
            memcpy(&out[o], &w[p], (size_t)n * 4);
            out[o + 4] = c;
        }
        o += n, p += n;
    }
    out[3] = next;
    *out_words = o, *fixed = z.n;
done:
    free(def), free(held), free(var), free(select);
    return out;
}
