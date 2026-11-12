// Original host-only metadata for portable Maxwell passthrough GS. This is not
// an NVN1 SDK control layout. The complete control enters shader identity/storage.
#ifndef NVN_GEOMETRY_CONTROL_H
#define NVN_GEOMETRY_CONTROL_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define NVN_GEOMETRY_CONTROL_OFFSET 0x800u
#define NVN_GEOMETRY_CONTROL_SIZE 0x40u
#define NVN_GEOMETRY_CONTROL_PORTABLE 1u
#define NVN_GEOMETRY_CONTROL_NATIVE 2u
// Portable metadata made by the host for a lowered native stage (vk_exec_draw.c, native_gs_control). Its body may read
// or declare more locations than it forwards; nothing else may (gfx_shader.c, reflect).
#define NVN_GEOMETRY_CONTROL_LOWERED 3u

static inline uint32_t nvn_geometry_control_u32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static inline void nvn_geometry_control_put(uint8_t *p, uint32_t n) {
    for (unsigned i = 0; i < 4; i++) p[i] = (uint8_t)(n >> (8 * i));
}

// Position's four components and whole user Locations only; a set bit means skip.
static inline int nvn_geometry_control_mask_valid(const uint32_t skip[8]) {
    if (!skip || skip[0] != 0x0fffffffu) return 0;
    for (unsigned i = 1; i <= 4; i++) for (unsigned bit = 0; bit < 32; bit += 4) {
        unsigned nibble = skip[i] >> bit & 15;
        if (nibble && nibble != 15) return 0;
    }
    return skip[5] == UINT32_MAX && skip[6] == UINT32_MAX && skip[7] == UINT32_MAX;
}

// Full legacy-shaped code, including its 0x30-byte prefix, never an SPH-only slice.
static inline int nvn_geometry_control_context(const uint8_t *control, size_t control_size,
                                                const uint8_t *code, uint64_t code_size) {
    if (!control || control_size < 0x880 || !code) return 0;
    if (nvn_geometry_control_u32(control) == 0x98761234u &&
        nvn_geometry_control_u32(control + 4) == 0xc9u) return 0;
    uint32_t length = nvn_geometry_control_u32(control + 0x704);
    uint32_t c1 = nvn_geometry_control_u32(control + 0x700);
    if (nvn_geometry_control_u32(control + 0x714) != 2 || length > code_size ||
        c1 > length || c1 < 0x90 || (c1 - 0x30) % 8 || (length - 0x30) % 8 ||
        !memcmp(control + 0x860, "DKC0", 4) || !memcmp(control + 0x860, "DKC1", 4)) return 0;
    uint32_t w0 = nvn_geometry_control_u32(code + 0x30);
    return (w0 & 31) == 1 && (w0 >> 5 & 31) == 3 && (w0 >> 10 & 15) == 4 && (w0 & (1u << 24)) &&
           (nvn_geometry_control_u32(code + 0x38) >> 24) == 1 &&
           (nvn_geometry_control_u32(code + 0x3c) >> 24 & 15) == 7 &&
           (nvn_geometry_control_u32(code + 0x40) & 0xfff) == 1;
}

// 1: canonical portable metadata; 0: absent; -1: malformed or native metadata.
// No native address/context contract is admitted. Output is cleared on failure.
static inline int nvn_geometry_control_get(const uint8_t *control, size_t control_size,
                                            const uint8_t *code, uint64_t code_size, uint32_t skip[8]) {
    static const char magic[8] = {'T','K','G','S','M','E','T','A'};
    if (skip) memset(skip, 0, 32);
    if (!control || control_size < 0x880) return -1;
    if (nvn_geometry_control_u32(control) == 0x98761234u &&
        nvn_geometry_control_u32(control + 4) == 0xc9u) return -1;
    const uint8_t *p = control + NVN_GEOMETRY_CONTROL_OFFSET;
    if (memcmp(p, magic, sizeof magic)) return 0;
    if (nvn_geometry_control_u32(p + 8) != 1 || nvn_geometry_control_u32(p + 12) != NVN_GEOMETRY_CONTROL_SIZE ||
        (nvn_geometry_control_u32(p + 16) != NVN_GEOMETRY_CONTROL_PORTABLE &&
         nvn_geometry_control_u32(p + 16) != NVN_GEOMETRY_CONTROL_LOWERED) ||
        nvn_geometry_control_u32(p + 20) != 1 || !nvn_geometry_control_context(control, control_size, code, code_size)) return -1;
    for (unsigned i = 56; i < 64; i++) if (p[i]) return -1;
    uint32_t words[8];
    for (unsigned i = 0; i < 8; i++) words[i] = nvn_geometry_control_u32(p + 24 + i * 4);
    if (!nvn_geometry_control_mask_valid(words)) return -1;
    if (skip) memcpy(skip, words, sizeof words);
    return 1;
}

// Only an already-owned synthetic control may be changed. A nonempty unrecognized
// region is never overwritten. Failure leaves every destination byte unchanged.
static inline int nvn_geometry_control_pack_portable(uint8_t *control, size_t control_size,
                                                     const uint8_t *code, uint64_t code_size, const uint32_t skip[8]) {
    if (!nvn_geometry_control_context(control, control_size, code, code_size) ||
        !nvn_geometry_control_mask_valid(skip)) return 0;
    uint8_t *p = control + NVN_GEOMETRY_CONTROL_OFFSET;
    int nonzero = 0;
    for (unsigned i = 0; i < NVN_GEOMETRY_CONTROL_SIZE; i++) nonzero |= p[i];
    if (nonzero && nvn_geometry_control_get(control, control_size, code, code_size, NULL) != 1) return 0;
    uint8_t envelope[NVN_GEOMETRY_CONTROL_SIZE] = {0};
    memcpy(envelope, "TKGSMETA", 8);
    nvn_geometry_control_put(envelope + 8, 1);
    nvn_geometry_control_put(envelope + 12, NVN_GEOMETRY_CONTROL_SIZE);
    nvn_geometry_control_put(envelope + 16, NVN_GEOMETRY_CONTROL_PORTABLE);
    nvn_geometry_control_put(envelope + 20, 1);
    for (unsigned i = 0; i < 8; i++) nvn_geometry_control_put(envelope + 24 + i * 4, skip[i]);
    memcpy(p, envelope, sizeof envelope);
    return 1;
}

// The same metadata, marked as made for a lowered native stage. The mark is part of the control, so it enters the
// shader's identity. Failure leaves every destination byte unchanged.
static inline int nvn_geometry_control_pack_lowered(uint8_t *control, size_t control_size,
                                                    const uint8_t *code, uint64_t code_size, const uint32_t skip[8]) {
    if (!nvn_geometry_control_pack_portable(control, control_size, code, code_size, skip)) return 0;
    nvn_geometry_control_put(control + NVN_GEOMETRY_CONTROL_OFFSET + 16, NVN_GEOMETRY_CONTROL_LOWERED);
    return 1;
}

// For a control nvn_geometry_control_get accepted.
static inline int nvn_geometry_control_is_lowered(const uint8_t *control) {
    return nvn_geometry_control_u32(control + NVN_GEOMETRY_CONTROL_OFFSET + 16) == NVN_GEOMETRY_CONTROL_LOWERED;
}

#endif
