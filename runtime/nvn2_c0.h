// NVN 2 compute parameter block: driver head, padded root tables, buffer tables and handles.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "nvn_obj.h"

#define NVN2_C0_BYTES 0x800u
#define NVN2_ROOT_TABLES 8
#define NVN2_ROOT_WORDS 64

typedef struct {
    uint8_t images, ubos, ssbos, textures;
    uint8_t roots[NVN2_ROOT_TABLES];   // declared words, each table padded to four words in c0
} Nvn2C0Layout;

// Parts: 0 roots, 1 uniform buffers, 2 storage buffers, 3 texture handles, 4 image handles, 5 end.
static inline size_t nvn2_c0_part(const Nvn2C0Layout *l, unsigned part) {
    size_t at = 0x10;
    if (!part) return at;
    for (unsigned i = 0; i < NVN2_ROOT_TABLES; i++) at += ((l->roots[i] + 3u) & ~3u) * 4u;
    if (part == 1) return at;
    at += 16u * l->ubos;
    if (part == 2) return at;
    at += 16u * l->ssbos;
    if (part == 3) return at;
    at += 8u * l->textures;
    return part == 4 ? at : at + 8u * l->images;
}

static inline int nvn2_c0_valid(const Nvn2C0Layout *l, size_t capacity) {
    if (l->ubos > 16 || l->ssbos > 16 || l->images > 32 || l->textures > 128) return 0;
    for (unsigned i = 0; i < NVN2_ROOT_TABLES; i++) if (l->roots[i] > NVN2_ROOT_WORDS) return 0;
    return nvn2_c0_part(l, 5) <= capacity;
}

// Shader control +0xa9..+0xb0 supplies the root counts. The SDK writes the root words, padded to 16 bytes, before
// its uniform/storage tables. The roots are command-stream state, not dimensions inferred from the bound images.
// Unwritten root state starts at zero.
static inline int nvn2_c0_pack(uint8_t *bank, size_t capacity, const Nvn2C0Layout *l,
                               const uint32_t roots[NVN2_ROOT_TABLES][NVN2_ROOT_WORDS],
                               const p_bind_buf ssbo[16], const p_bind_buf ubo[16]) {
    if (!nvn2_c0_valid(l, capacity)) return 0;
    memset(bank, 0, capacity);
    size_t at = nvn2_c0_part(l, 0);
    for (unsigned i = 0; i < NVN2_ROOT_TABLES; i++) {
        memcpy(bank + at, roots[i], 4u * l->roots[i]);
        at += ((l->roots[i] + 3u) & ~3u) * 4u;
    }
    for (unsigned k = 0; k < 2; k++) {
        const p_bind_buf *b = k ? ssbo : ubo;
        unsigned n = k ? l->ssbos : l->ubos;
        for (unsigned i = 0; i < n; i++, at += 16) {
            uint32_t bytes = (uint32_t)b[i].size;
            memcpy(bank + at, &b[i].addr, 8);
            memcpy(bank + at + 8, &bytes, 4);
        }
    }
    return 1;   // handles stay zero: the translator resolves their declarations to host descriptor slots
}
