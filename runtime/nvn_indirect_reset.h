// CPU-only planning for the NVN 2 indirect-record reset: bounds and overlap of the selected pointer list are checked before
// anything is written; targets stay opaque GPU addresses.
#pragma once
#include <stdint.h>

#define NVN_INDIRECT_RESET_RECORD_BYTES 20u
#define NVN_INDIRECT_RESET_MAX_RECORDS (1u << 20)

typedef struct { uint64_t begin; uint32_t bytes, count; } NvnIndirectResetList;

// Returns 1 only when the complete target range is backed by one contiguous mapping; a query, it must not write or read GPU memory.
typedef int (*NvnIndirectResetMapped)(void *context, uint64_t address, uint32_t bytes);

static inline int nvn_indirect_reset_list(uint64_t address, uint32_t descriptor_bytes, int64_t first, int64_t count,
                                         NvnIndirectResetList *out) {
    if (!out) return 0;
    *out = (NvnIndirectResetList){0};
    if (!address || first < 0 || count <= 0 || count > NVN_INDIRECT_RESET_MAX_RECORDS) return 0;
    uint64_t entries = descriptor_bytes / sizeof(uint64_t), start = (uint64_t)first, n = (uint64_t)count;
    if (start > entries || n > entries - start) return 0;
    uint64_t offset = start * sizeof(uint64_t), bytes = n * sizeof(uint64_t);
    if (address > UINT64_MAX - offset) return 0;
    uint64_t begin = address + offset;
    if (begin > UINT64_MAX - bytes) return 0;
    *out = (NvnIndirectResetList){begin, (uint32_t)bytes, (uint32_t)n};
    return 1;
}

// Validate every nonnull record before any write. Overlap is checked against the selected first/count subrange only.
static inline int nvn_indirect_reset_targets(const NvnIndirectResetList *list, const uint64_t *pointers,
                                            NvnIndirectResetMapped mapped, void *context) {
    if (!list || !pointers || !mapped || !list->begin || !list->count ||
        list->count > NVN_INDIRECT_RESET_MAX_RECORDS ||
        list->bytes != (uint64_t)list->count * sizeof(uint64_t) || list->begin > UINT64_MAX - list->bytes) return 0;
    uint64_t end = list->begin + list->bytes;
    for (uint32_t i = 0; i < list->count; i++) {
        uint64_t address = pointers[i];
        if (!address) continue;
        if (address % sizeof(uint32_t) || address > UINT64_MAX - NVN_INDIRECT_RESET_RECORD_BYTES ||
            (address < end && list->begin < address + NVN_INDIRECT_RESET_RECORD_BYTES) ||
            !mapped(context, address, NVN_INDIRECT_RESET_RECORD_BYTES)) return 0;
    }
    return 1;
}
