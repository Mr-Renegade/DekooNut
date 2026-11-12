// The NaN guard: a SPIR-V pass for fragment shaders, applied by vk_pipeline.c make_module_linked
// before vkCreateShaderModule. Free of Vulkan.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void rt_log(const char *fmt, ...);   // host.h; declared here so the file needs nothing else

// In fragment shaders every store to a float colour output (not a built-in) becomes c = FClamp(v, -1e30, 1e30),
// r = OpSelect(OpIsNan(v), 0, c), so one NaN or Inf pixel cannot reach bloom and exposure sums. Missing declarations are
// added. NULL when the shader has no such store or anything is not understood.
uint32_t *spirv_nan_guard(const uint32_t *w, uint64_t words, uint64_t *out_n) {
    static int said;
    static volatile int64_t patched, stores;
    if (!said) { said = 1; rt_log("pipeline: NaN guard on: float colour outputs of fragment shaders store NaN as 0 and clamp to +-1e30"); }
    if (words < 5 || words > (1u << 22)) return NULL;
    uint32_t bound = w[3];
    if (!bound || bound > (1u << 22)) return NULL;
    int frag = 0;
    uint32_t fty = 0, bty = 0, glsl = 0;
    uint64_t first_fn = 0, memmodel = 0;
    // per id: 1 float type, 2..4 float vector of that size, plus pointer info
    uint8_t *kind = calloc(bound, 1), *outp = calloc(bound, 1), *builtin = calloc(bound, 1);
    uint32_t *pointee = calloc(bound, 4), *bvec = calloc(5, 4);
    uint32_t vecty[5] = {0};
    if (!kind || !outp || !builtin || !pointee || !bvec) goto fail;
    for (uint64_t i = 5; i < words;) {
        uint32_t op = w[i] & 0xffff, n = w[i] >> 16;
        if (!n || i + n > words) goto fail;
        if (op == 15 && w[i + 1] == 4) frag = 1;                                         // OpEntryPoint Fragment
        if (op == 11 && n >= 3 && !strncmp((const char *)(w + i + 2), "GLSL.std.450", 12)) glsl = w[i + 1];   // OpExtInstImport
        if (op == 14 && !memmodel) memmodel = i;                                         // OpMemoryModel
        if (op == 54 && !first_fn) first_fn = i;                                         // OpFunction
        if (op == 71 && n >= 3 && w[i + 2] == 11 && w[i + 1] < bound) builtin[w[i + 1]] = 1;   // OpDecorate BuiltIn
        if (op == 20) bty = w[i + 1];                                                    // OpTypeBool
        if (op == 22 && n == 3 && w[i + 2] == 32 && w[i + 1] < bound) kind[w[i + 1]] = 1, fty = w[i + 1];   // OpTypeFloat 32
        if (op == 23 && w[i + 1] < bound && w[i + 2] < bound) {                          // OpTypeVector
            if (kind[w[i + 2]] == 1 && w[i + 3] >= 2 && w[i + 3] <= 4) kind[w[i + 1]] = (uint8_t)w[i + 3], vecty[w[i + 3]] = w[i + 1];
            if (bty && w[i + 2] == bty && w[i + 3] <= 4) bvec[w[i + 3]] = w[i + 1];
        }
        if (op == 32 && w[i + 1] < bound && w[i + 3] < bound && w[i + 2] == 3) pointee[w[i + 1]] = w[i + 3];   // OpTypePointer Output
        if (op == 59 && w[i + 2] < bound && w[i + 1] < bound && w[i + 3] == 3 && pointee[w[i + 1]] && kind[pointee[w[i + 1]]] && !builtin[w[i + 2]])
            outp[w[i + 2]] = 1;                                                          // OpVariable Output of float type
        if ((op == 65 || op == 66) && w[i + 2] < bound && w[i + 3] < bound && outp[w[i + 3]] && pointee[w[i + 1]] && kind[pointee[w[i + 1]]])
            outp[w[i + 2]] = 2;                                                          // OpAccessChain into one, to a float
        i += n;
    }
    if (!frag || !fty || !first_fn || !memmodel) goto fail;
    // the stores, and the vector sizes they need
    int need[5] = {0}, nst = 0;
    for (uint64_t i = 5; i < words; i += w[i] >> 16)
        if ((w[i] & 0xffff) == 62 && w[i + 1] < bound && outp[w[i + 1]]) {
            nst++;
        }
    if (!nst) goto fail;
    // pointer id -> value type: walk again (types are known now)
    uint32_t *valty = calloc(bound, 4);
    if (!valty) goto fail;
    for (uint64_t i = 5; i < words; i += w[i] >> 16) {
        uint32_t op = w[i] & 0xffff;
        if ((op == 59 || op == 65 || op == 66) && w[i + 2] < bound && outp[w[i + 2]]) valty[w[i + 2]] = pointee[w[i + 1]];
    }
    for (uint64_t i = 5; i < words; i += w[i] >> 16)
        if ((w[i] & 0xffff) == 62 && w[i + 1] < bound && outp[w[i + 1]]) need[kind[valty[w[i + 1]]]] = 1;
    uint32_t *o = malloc(((size_t)words + 256 + (size_t)nst * 20) * 4);
    if (!o) { free(valty); goto fail; }
    uint64_t k = 0;
    uint32_t c_pos = 0, c_neg = 0, c_zero = 0, cv_pos[5] = {0}, cv_neg[5] = {0}, cv_zero[5] = {0};
    union { float f; uint32_t u; } big = {1e30f}, nbig = {-1e30f};
    for (uint64_t i = 0; i < words;) {
        uint32_t op = i < 5 ? 0 : (w[i] & 0xffff), n = i < 5 ? 5 : (w[i] >> 16);
        if (i == memmodel && !glsl) {   // OpExtInstImport "GLSL.std.450"
            glsl = bound++;
            o[k++] = 6u << 16 | 11, o[k++] = glsl;
            memcpy(o + k, "GLSL.std.450\0\0\0", 16), k += 4;
        }
        if (i == first_fn) {   // declarations the guards use
            if (!bty) bty = bound++, o[k++] = 2u << 16 | 20, o[k++] = bty;
            c_pos = bound++, o[k++] = 4u << 16 | 43, o[k++] = fty, o[k++] = c_pos, o[k++] = big.u;
            c_neg = bound++, o[k++] = 4u << 16 | 43, o[k++] = fty, o[k++] = c_neg, o[k++] = nbig.u;
            c_zero = bound++, o[k++] = 4u << 16 | 43, o[k++] = fty, o[k++] = c_zero, o[k++] = 0;
            for (int m = 2; m <= 4; m++) {
                if (!need[m] || !vecty[m]) continue;
                if (!bvec[m]) bvec[m] = bound++, o[k++] = 4u << 16 | 23, o[k++] = bvec[m], o[k++] = bty, o[k++] = (uint32_t)m;
                uint32_t *cv[3] = {&cv_pos[m], &cv_neg[m], &cv_zero[m]}, cs[3] = {c_pos, c_neg, c_zero};
                for (int q = 0; q < 3; q++) {
                    *cv[q] = bound++;
                    o[k++] = (uint32_t)(3 + m) << 16 | 44, o[k++] = vecty[m], o[k++] = *cv[q];   // OpConstantComposite
                    for (int x = 0; x < m; x++) o[k++] = cs[q];
                }
            }
        }
        if (op == 62 && w[i + 1] < bound && outp[w[i + 1]]) {
            uint32_t vt = valty[w[i + 1]], m = kind[vt], v = w[i + 2];
            if (m == 1 || (m >= 2 && m <= 4 && vecty[m] && bvec[m])) {
                uint32_t bt = m == 1 ? bty : bvec[m], pos = m == 1 ? c_pos : cv_pos[m], neg = m == 1 ? c_neg : cv_neg[m],
                         zero = m == 1 ? c_zero : cv_zero[m];
                uint32_t isn = bound++, cl = bound++, r = bound++;
                o[k++] = 4u << 16 | 156, o[k++] = bt, o[k++] = isn, o[k++] = v;                                  // OpIsNan
                o[k++] = 8u << 16 | 12, o[k++] = vt, o[k++] = cl, o[k++] = glsl, o[k++] = 43, o[k++] = v, o[k++] = neg, o[k++] = pos;   // FClamp
                o[k++] = 6u << 16 | 169, o[k++] = vt, o[k++] = r, o[k++] = isn, o[k++] = zero, o[k++] = cl;     // OpSelect
                memcpy(o + k, w + i, (size_t)n * 4);
                o[k + 2] = r;
                k += n, i += n;
                __atomic_add_fetch(&stores, 1, __ATOMIC_RELAXED);
                continue;
            }
        }
        memcpy(o + k, w + i, (size_t)n * 4), k += n, i += n;
    }
    o[3] = bound;
    free(valty);
    free(kind), free(outp), free(builtin), free(pointee), free(bvec);
    if (__atomic_add_fetch(&patched, 1, __ATOMIC_RELAXED) <= 4 || !(patched & 1023))
        rt_log("pipeline: NaN guard: %lld fragment shaders guarded so far (%lld output stores)", (long long)patched, (long long)stores);
    *out_n = k;
    return o;
fail:
    free(kind), free(outp), free(builtin), free(pointee), free(bvec);
    return NULL;
}

