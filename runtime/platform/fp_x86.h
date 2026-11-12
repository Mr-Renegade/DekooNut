// FP control on x86-64: MXCSR (rounding control bits 13-14, FTZ bit 15, DAZ bit 6, exception flags and masks).
// Inline on purpose: lifted FP code (simd.h) reads and writes it on every operation that depends on the guest FPCR.
#pragma once
#include <stdint.h>
#include <xmmintrin.h>

#define PLAT_FP_RC_MASK 0x6000u   // rounding control: 0 nearest, 0x2000 down, 0x4000 up, 0x6000 toward zero
#define PLAT_FP_FTZ 0x8000u
#define PLAT_FP_DAZ 0x0040u

static inline uint32_t plat_fp_get(void) { return _mm_getcsr(); }
static inline void plat_fp_set(uint32_t v) { _mm_setcsr(v); }
