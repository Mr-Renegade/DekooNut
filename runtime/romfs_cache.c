// A copy of the RomFS files the game has read, on a faster disk (romfs_cache.h): a file the game read is copied by a thread of
// its own, from the OS's file cache, to <cache>/<index>.bin, and later opens read that copy.
// The cache directory is $XDG_CACHE_HOME/oot-pc-recomp/romfs (%LOCALAPPDATA%\oot-pc-recomp\romfs on Windows), at most 8 GB;
// copying stops below 10 GB free. Off when it is on the image's own disk. A stamp file (image size, write time) empties it for another image.
// Only <index>.bin, <index>.tmp and the stamp are ever written or removed.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "host.h"
#include "platform/platform.h"
#include "romfs_cache.h"

enum { RING = 8192, COPY_CHUNK = 8u << 20 };

static char dir[1024];
static int on;
static PlatFile *src;
static uint64_t nfiles, budget, used, floor_free;
static uint8_t *known;   // per file: 1 queued or copied
static struct { uint64_t i, off, size; } ring[RING];
static unsigned head, tail;
static PlatMutex lock = PLAT_MUTEX_INIT;
static PlatCond wake;
static uint64_t copied_n, copied_bytes;

static void path_of(char *out, size_t cap, uint64_t i, const char *ext) {
    snprintf(out, cap, "%s/%llx%s", dir, (unsigned long long)i, ext);
}

static int mkdirs(const char *p) {
    char b[1024];
    snprintf(b, sizeof b, "%s", p);
    for (char *q = b + 1; *q; q++)
        if (*q == '/' || *q == '\\') {
            char c = *q;
            *q = 0;
            plat_fs_mkdir(b);
            *q = c;
        }
    int r = plat_fs_mkdir(b);
    return r == PLAT_FS_OK || r == PLAT_FS_EXISTS;
}

static int is_ours(const char *name) {   // <hex>.bin or <hex>.tmp
    size_t n = strlen(name);
    if (n < 5 || (strcmp(name + n - 4, ".bin") && strcmp(name + n - 4, ".tmp"))) return 0;
    for (size_t k = 0; k < n - 4; k++)
        if (!strchr("0123456789abcdef", name[k])) return 0;
    return 1;
}

static int clear_cb(const char *name, int kind, uint64_t size, void *ctx) {
    (void)size, (void)ctx;
    if (kind == PLAT_FS_FILE && is_ours(name)) {
        char p[1200];
        snprintf(p, sizeof p, "%s/%s", dir, name);
        plat_fs_unlink(p);
    }
    return 1;
}

static int scan_cb(const char *name, int kind, uint64_t size, void *ctx) {
    (void)ctx;
    if (kind != PLAT_FS_FILE || !is_ours(name)) return 1;
    size_t n = strlen(name);
    if (!strcmp(name + n - 4, ".tmp")) return clear_cb(name, kind, size, ctx);   // a copy cut short
    uint64_t i = strtoull(name, NULL, 16);
    if (i < nfiles) known[i] = 1;
    used += size;
    return 1;
}

static void copy_one(uint64_t i, uint64_t off, uint64_t size, uint8_t *buf) {
    uint64_t freeb = plat_fs_free_bytes(dir);
    if (used + size > budget || (freeb != UINT64_MAX && freeb < floor_free + size)) return;
    char tmp[1200], fin[1200];
    path_of(tmp, sizeof tmp, i, ".tmp");
    path_of(fin, sizeof fin, i, ".bin");
    plat_fs_unlink(tmp);
    PlatFile *o = plat_fs_create(tmp, NULL);
    if (!o) return;
    uint64_t done = 0;
    while (done < size) {
        uint32_t n = size - done > COPY_CHUNK ? COPY_CHUNK : (uint32_t)(size - done);
        int64_t r = plat_file_pread(src, buf, n, off + done);
        if (r <= 0 || plat_file_pwrite(o, buf, (uint32_t)r, done) != r) break;
        done += (uint64_t)r;
    }
    plat_file_close(o);
    if (done != size || plat_fs_rename(tmp, fin, 0) != PLAT_FS_OK) {
        plat_fs_unlink(tmp);
        return;
    }
    plat_mutex_lock(&lock);
    used += size;
    copied_n++;
    copied_bytes += size;
    uint64_t n = copied_n, b = copied_bytes;
    plat_mutex_unlock(&lock);
    if (n == 1 || n % 1000 == 0) rt_log("romfs cache: %llu files copied this run (%.0f MB)", (unsigned long long)n, b / 1e6);
}

static void writer(void *arg) {
    (void)arg;
    uint8_t *buf = malloc(COPY_CHUNK);
    if (!buf) return;
    for (;;) {
        plat_mutex_lock(&lock);
        while (head == tail) plat_cond_wait(&wake, &lock);
        uint64_t i = ring[tail % RING].i, off = ring[tail % RING].off, size = ring[tail % RING].size;
        tail++;
        plat_mutex_unlock(&lock);
        copy_one(i, off, size, buf);
    }
}

void romfs_cache_init(PlatFile *image, uint64_t file_count) {
    const char *x = getenv("XDG_CACHE_HOME"), *h = getenv("HOME"), *l = getenv("LOCALAPPDATA");
    if (x && *x) snprintf(dir, sizeof dir, "%s/oot-pc-recomp/romfs", x);
    else if (h && *h) snprintf(dir, sizeof dir, "%s/.cache/oot-pc-recomp/romfs", h);
    else if (l && *l) snprintf(dir, sizeof dir, "%s/oot-pc-recomp/romfs", l);
    else return;
    if (!mkdirs(dir)) {
        rt_log("romfs cache: cannot create %s; off", dir);
        return;
    }
    PlatFileId img;
    if (!plat_file_id(image, &img)) return;
    // The stamp: the image it caches. Its own volume tells whether the cache is on the image's disk.
    char sp[1200];
    snprintf(sp, sizeof sp, "%s/image.stamp", dir);
    char want[128];
    int wn = snprintf(want, sizeof want, "%llu %llu\n", (unsigned long long)img.size, (unsigned long long)img.write_time);
    char have[128] = {0};
    PlatFile *s = plat_file_open(sp);
    if (s) {
        plat_file_pread(s, have, sizeof have - 1, 0);
        PlatFileId sid;
        int same_disk = plat_file_id(s, &sid) && sid.volume == img.volume;
        plat_file_close(s);
        if (same_disk) {
            rt_log("romfs cache: %s is on the image's own disk; off", dir);
            return;
        }
    }
    nfiles = file_count;
    known = calloc(file_count ? file_count : 1, 1);
    if (!known) return;
    if (strcmp(have, want)) {   // another image (or none yet): start empty
        plat_fs_list(dir, clear_cb, NULL);
        plat_fs_unlink(sp);
        PlatFile *o = plat_fs_create(sp, NULL);
        if (!o) return;
        plat_file_pwrite(o, want, (uint32_t)wn, 0);
        PlatFileId sid;
        int same_disk = plat_file_id(o, &sid) && sid.volume == img.volume;
        plat_file_close(o);
        if (same_disk) {
            rt_log("romfs cache: %s is on the image's own disk; off", dir);
            return;
        }
    }
    plat_fs_list(dir, scan_cb, NULL);
    budget = (uint64_t)(8.0 * 1e9);
    floor_free = (uint64_t)(10.0 * 1e9);
    src = image;
    plat_cond_init(&wake);
    if (!plat_thread_spawn(writer, NULL, "romfs-cache", 0)) return;
    on = 1;
    rt_log("romfs cache: %s, %.2f GB held, limit %.1f GB, keeps %.1f GB of the disk free", dir, used / 1e9, budget / 1e9,
           floor_free / 1e9);
}

PlatFile *romfs_cache_open(uint64_t i, uint64_t size) {
    if (!on || i >= nfiles || !known[i]) return NULL;
    char p[1200];
    path_of(p, sizeof p, i, ".bin");
    PlatFile *f = plat_file_open(p);
    if (!f) return NULL;
    PlatFileId id;
    if (!plat_file_id(f, &id) || id.size != size) {   // queued, not copied yet, or damaged
        plat_file_close(f);
        return NULL;
    }
    return f;
}

void romfs_cache_note(uint64_t i, uint64_t off, uint64_t size) {
    if (!on || i >= nfiles || !size) return;
    plat_mutex_lock(&lock);
    if (!known[i] && head - tail < RING) {
        known[i] = 1;
        ring[head % RING].i = i, ring[head % RING].off = off, ring[head % RING].size = size;
        head++;
        plat_cond_signal(&wake);
    }
    plat_mutex_unlock(&lock);
}
