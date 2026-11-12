// SPDX-License-Identifier: GPL-3.0-or-later
// oot_icon: one ASTC texture level in block-linear storage to raw RGBA8, with the runtime's own block-linear copy
// (runtime/tex_deswizzle.c) and ASTC decoder (runtime/tex_astc.c). tools/game_icon.py builds and calls it.
//
//   oot_icon <in> <out> <width> <height> <block_w> <block_h> <block_height_log2> <srgb 0|1>
//
// <in> holds the level's storage (at least the bytes its GOB blocks cover); <out> gets width * height * 4 bytes.
// Exit 0 on success, 1 with one line on stderr otherwise.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tex.h"

// The runtime spreads large copies over its worker pool; here one thread does all rows.
void tex_mt_rows(uint32_t n, TexRowsFn fn, void *ctx) { fn(ctx, 0, n); }

static int fail(const char *msg) { fprintf(stderr, "oot_icon: %s\n", msg); return 1; }

int main(int argc, char **argv) {
    if (argc != 9) return fail("usage: oot_icon <in> <out> <width> <height> <block_w> <block_h> <block_height_log2> <srgb>");
    long w = atol(argv[3]), h = atol(argv[4]), bw = atol(argv[5]), bh = atol(argv[6]), th = atol(argv[7]);
    int srgb = atoi(argv[8]);
    if (w < 1 || h < 1 || w > 16384 || h > 16384 || bw < 4 || bw > 12 || bh < 4 || bh > 12 || th < 0 || th > 5)
        return fail("bad size arguments");
    uint32_t row_bytes = (uint32_t)((w + bw - 1) / bw) * 16, rows = (uint32_t)((h + bh - 1) / bh);
    uint32_t gob_rows = 8u << th;
    size_t need = (size_t)(row_bytes + 63) / 64 * 512 * ((size_t)1 << th) * ((rows + gob_rows - 1) / gob_rows);

    FILE *f = fopen(argv[1], "rb");
    if (!f) return fail("cannot open the input");
    uint8_t *bl = calloc(need, 1);
    size_t got = bl ? fread(bl, 1, need, f) : 0;
    fclose(f);
    if (got != need) return fail("the input is shorter than the texture level");

    uint8_t *blocks = malloc((size_t)row_bytes * rows), *rgba = malloc((size_t)w * h * 4);
    if (!blocks || !rgba) return fail("out of memory");
    tex_bl_copy(bl, blocks, row_bytes, row_bytes, rows, 1, (unsigned)th, 0, 1);
    astc_decode_ldr(blocks, (uint32_t)w, (uint32_t)h, (uint32_t)bw, (uint32_t)bh, srgb != 0, rgba);

    f = fopen(argv[2], "wb");
    if (!f) return fail("cannot create the output");
    size_t n = (size_t)w * h * 4;
    int bad = fwrite(rgba, 1, n, f) != n;
    if (fclose(f) || bad) return fail("cannot write the output");
    free(bl); free(blocks); free(rgba);
    return 0;
}
