// The function-local flag helpers of runtime/cpu.h (Flags, fc_*, fnz_*, nz_*, adc*f) against the packed helpers
// (subs64() and friends, COND_xx), which tests/diff/lift.py checks against Unicorn. Built and run by
// tests/flags_check.py; prints the first mismatches and exits 1 on any.
#include <stdio.h>

#include "cpu.h"

static const uint64_t EDGE[] = {0, 1, 2, 0x7f, 0x80, 0xff, 0x7fff, 0x8000, 0xffff, 0x7fffffff, 0x80000000,
                                0x80000001, 0xfffffffe, 0xffffffff, 0x100000000ull, 0x7fffffffffffffffull,
                                0x8000000000000000ull, 0x8000000000000001ull, 0xfffffffffffffffeull,
                                0xffffffffffffffffull, 0x123456789abcdef0ull, 0xfedcba9876543210ull};
#define NEDGE (sizeof EDGE / sizeof EDGE[0])

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint64_t next(void) {
    rng ^= rng << 13, rng ^= rng >> 7, rng ^= rng << 17;
    return rng;
}

static int fails;
static void bad(const char *what, int cc, uint64_t a, uint64_t b, uint64_t got, uint64_t want) {
    if (fails++ < 20)
        printf("%s cc %d a %llx b %llx: got %llx want %llx\n", what, cc, (unsigned long long)a, (unsigned long long)b,
               (unsigned long long)got, (unsigned long long)want);
}

static int cond(const Cpu *c, int cc) {
    switch (cc) {
    case CC_EQ: return COND_EQ(c);
    case CC_NE: return COND_NE(c);
    case CC_CS: return COND_CS(c);
    case CC_CC: return COND_CC(c);
    case CC_MI: return COND_MI(c);
    case CC_PL: return COND_PL(c);
    case CC_VS: return COND_VS(c);
    case CC_VC: return COND_VC(c);
    case CC_HI: return COND_HI(c);
    case CC_LS: return COND_LS(c);
    case CC_GE: return COND_GE(c);
    case CC_LT: return COND_LT(c);
    case CC_GT: return COND_GT(c);
    default: return COND_LE(c);
    }
}

// The condition argument is not constant here, so this also covers the switch default paths.
#define CHECK(NAME, OLD, NEW, FC, FNZ)                                                     \
    do {                                                                                   \
        Cpu c = {0};                                                                       \
        Flags f = {0};                                                                     \
        uint64_t ro = OLD, rn = NEW;                                                       \
        if (ro != rn) bad(NAME " result", -1, a, b, rn, ro);                               \
        for (int cc = 0; cc < 14; cc++)                                                    \
            if (!FC(&f, cc) != !cond(&c, cc)) bad(NAME, cc, a, b, FC(&f, cc), cond(&c, cc)); \
        f.nz = 0;                                                                          \
        if (FNZ(&f) != c.nzcv) bad(NAME " packed", -1, a, b, FNZ(&f), c.nzcv);             \
        Flags g = {.nz = c.nzcv};                                                          \
        for (int cc = 0; cc < 14; cc++)                                                    \
            if (!fc_nz(&g, cc) != !cond(&c, cc)) bad(NAME " nz", cc, a, b, fc_nz(&g, cc), cond(&c, cc)); \
    } while (0)

static void one(uint64_t a, uint64_t b) {
    uint32_t a32 = (uint32_t)a, b32 = (uint32_t)b;
    CHECK("sub64", subs64(&c, a, b), fsub64(&f, a, b), fc_sub64, fnz_sub64);
    CHECK("sub32", subs32(&c, a32, b32), fsub32(&f, a32, b32), fc_sub32, fnz_sub32);
    CHECK("add64", adds64(&c, a, b), fadd64(&f, a, b), fc_add64, fnz_add64);
    CHECK("add32", adds32(&c, a32, b32), fadd32(&f, a32, b32), fc_add32, fnz_add32);
    CHECK("logic64", logic64(&c, a & b), flogic64(&f, a & b), fc_logic64, fnz_logic64);
    CHECK("logic32", logic32(&c, a32 & b32), flogic32(&f, a32 & b32), fc_logic32, fnz_logic32);
    {
        Cpu c = {0};
        if (nz_sub64(a, b) != (subs64(&c, a, b), c.nzcv)) bad("nz_sub64", -1, a, b, nz_sub64(a, b), c.nzcv);
        if (nz_sub32(a32, b32) != (subs32(&c, a32, b32), c.nzcv)) bad("nz_sub32", -1, a, b, nz_sub32(a32, b32), c.nzcv);
        if (nz_add64(a, b) != (adds64(&c, a, b), c.nzcv)) bad("nz_add64", -1, a, b, nz_add64(a, b), c.nzcv);
        if (nz_add32(a32, b32) != (adds32(&c, a32, b32), c.nzcv)) bad("nz_add32", -1, a, b, nz_add32(a32, b32), c.nzcv);
    }
    for (int cin = 0; cin < 2; cin++) {
        Cpu c = {.nzcv = (uint64_t)cin << 29};
        Flags f = {0};
        if (adc64f(cin, a, b) != adc64(&c, a, b)) bad("adc64", cin, a, b, adc64f(cin, a, b), adc64(&c, a, b));
        if (sbc64f(cin, a, b) != sbc64(&c, a, b)) bad("sbc64", cin, a, b, sbc64f(cin, a, b), sbc64(&c, a, b));
        if (adc32f(cin, a32, b32) != adc32(&c, a32, b32)) bad("adc32", cin, a, b, adc32f(cin, a32, b32), adc32(&c, a32, b32));
        if (sbc32f(cin, a32, b32) != sbc32(&c, a32, b32)) bad("sbc32", cin, a, b, sbc32f(cin, a32, b32), sbc32(&c, a32, b32));
        uint64_t r;
#define CARRY(NAME, NEW, OLD)                                                              \
        c.nzcv = (uint64_t)cin << 29;                                                      \
        r = NEW;                                                                           \
        if (r != OLD || f.nz != c.nzcv) bad(NAME, cin, a, b, f.nz, c.nzcv);
        CARRY("adcs64", adcs64f(&f, cin, a, b), adcs64(&c, a, b))
        CARRY("sbcs64", sbcs64f(&f, cin, a, b), sbcs64(&c, a, b))
        CARRY("adcs32", adcs32f(&f, cin, a32, b32), adcs32(&c, a32, b32))
        CARRY("sbcs32", sbcs32f(&f, cin, a32, b32), sbcs32(&c, a32, b32))
    }
}

int main(void) {
    long n = 0;
    for (size_t i = 0; i < NEDGE; i++)
        for (size_t j = 0; j < NEDGE; j++, n++) one(EDGE[i], EDGE[j]);
    for (int k = 0; k < 2000000; k++, n++) {
        uint64_t a = next(), b = next();
        switch (k & 3) {
        case 1: b = a; break;                       // equal operands
        case 2: b = a + (next() & 3) - 1; break;    // neighbours
        case 3: a &= 0xff, b &= 0xff; break;        // small values
        }
        one(a, b);
    }
    printf("%ld operand pairs, %d mismatches\n", n, fails);
    return fails != 0;
}
