/* zbic_dec: decode one ZBIC segment (zstd with the "ZBIC" frame magic and BIC-coded FSE tables, NSO flag bit 7,
 * firmware 22.0.0+) to a file. Built against hexkyz/zstd 02a8f10 (BSD-3-Clause, Meta Platforms) with
 * ZSTD_ZBIC_SUPPORT=1. The decoder itself is not reimplemented here.
 *
 *   zbic_dec <in> <expected_size> <out>
 *
 * Exit 0 only when the frame decodes to exactly expected_size bytes. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zstd.h"

int main(int argc, char **argv)
{
    if (argc != 4) { fprintf(stderr, "usage: zbic_dec <in> <expected_size> <out>\n"); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 2; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *in = malloc((size_t)n);
    if (!in || fread(in, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "read failed\n"); return 2; }
    fclose(f);
    size_t want = strtoull(argv[2], NULL, 0);
    unsigned char *out = malloc(want ? want : 1);
    if (!out) { fprintf(stderr, "out of memory\n"); return 2; }
    size_t got = ZSTD_decompress(out, want, in, (size_t)n);
    if (ZSTD_isError(got)) { fprintf(stderr, "zbic: %s\n", ZSTD_getErrorName(got)); return 1; }
    if (got != want) { fprintf(stderr, "zbic: size mismatch %zu != %zu\n", got, want); return 1; }
    FILE *o = fopen(argv[3], "wb");
    if (!o || fwrite(out, 1, got, o) != got) { fprintf(stderr, "write failed\n"); return 2; }
    fclose(o);
    return 0;
}
