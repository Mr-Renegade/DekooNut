// Adapted from MacNativeGames games/totk/runtime/fs_host.c @ b75be8cf (MIT, Copyright (c) 2026 MacNativeGames
// contributors); the probe follows MacNativeGames' romfs.c runtime_fs_probe and fs_host.c runtime_save_probe.
// nn::fs host implementation of the functions main imports. RomFS goes to romfs.c; save, temporary and cache storage are
// host directories under the profile directory (rt_config()->profile). Paths are UTF-8. See README, "Save storage".
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fs.h"
#include "host.h"
#include "platform/platform.h"
#include "romfs_path.h"

#define OOT_APPLICATION_ID RT_APPLICATION_ID
#define FILE_BASE 0x200000000ull
#define DIR_BASE 0x300000000ull
#define NFILES 1024
#define NDIRS 256
#define NMOUNTS 16
#define PATH_CAP 1100

enum { M_FREE, M_SAVE, M_TEMP, M_CACHE, M_SAVE_RO };   // M_SAVE: the working copy; M_SAVE_RO: the committed save

typedef struct { int kind; char name[32]; char root[1024]; } Mount;
typedef struct { PlatFile *h; int mode, save; } File;   // h NULL = free; save: on a save-data mount
typedef struct { char *name; int kind; uint64_t size; } Listed;
typedef struct { Listed *e; int n, next, mode, used; } Dir;   // the directory's entries at open; used 0 = free

static Mount mounts[NMOUNTS];
static File files[NFILES];
static Dir dirs[NDIRS];
static PlatMutex lock = PLAT_MUTEX_INIT;
static char profile[1024], save_root[1100], work_root[1100];
static volatile int32_t storage_ready;

static void need(int c, const char *m) {
    if (!c) rt_fatal("fs: %s", m);
}

// ---- directory listings: every entry but "." and "..", taken whole before the caller changes the directory.
typedef struct { Listed *e; int n, cap; } Listing;

static int listing_add(const char *name, int kind, uint64_t size, void *ctx) {
    Listing *l = ctx;
    if (l->n == l->cap) {
        l->cap = l->cap ? l->cap * 2 : 16;
        l->e = realloc(l->e, (size_t)l->cap * sizeof *l->e);
        need(l->e != NULL, "directory listing");
    }
    l->e[l->n].name = strdup(name);
    need(l->e[l->n].name != NULL, "directory listing");
    l->e[l->n].kind = kind, l->e[l->n].size = size;
    l->n++;
    return 1;
}

static int list_dir(const char *dir, Listing *l) {
    *l = (Listing){0};
    return plat_fs_list(dir, listing_add, l);
}

static void listing_free(Listed *e, int n) {
    for (int i = 0; i < n; i++) free(e[i].name);
    free(e);
}

static char *join(const char *dir, const char *name) {
    size_t n = strlen(dir), k = strlen(name);
    char *p = malloc(n + k + 2);
    need(p != NULL, "path allocation");
    memcpy(p, dir, n);
    p[n] = '/';
    memcpy(p + n + 1, name, k + 1);
    return p;
}

// Save in progress: a save-data file open for writing, or save data changed in the last 3 s and not committed yet.
static volatile int32_t save_writers, save_dirty;
static volatile int64_t save_changed_ns;
static void save_wait_commit(void);
static void save_touched(void) {
    save_wait_commit();   // every change to save data comes here first: none lands in a commit's snapshot
    plat_xchg64(&save_changed_ns, (int64_t)plat_now_ns());
    plat_xchg32(&save_dirty, 1);
}
int fs_save_busy(void) {
    if (plat_cas32(&save_writers, 0, 0) > 0) return 1;
    return plat_cas32(&save_dirty, 0, 0) &&
           plat_now_ns() - (uint64_t)plat_cas64(&save_changed_ns, 0, 0) < 3000000000ull;
}

// ---- host directories
static int make_dir(const char *p) {
    int e = plat_fs_mkdir(p);
    return !e || (e == PLAT_FS_EXISTS && plat_fs_kind(p) == PLAT_FS_DIR);
}

// p and every missing parent.
static void make_dirs(const char *p) {
    char buf[1100];
    need(snprintf(buf, sizeof buf, "%s", p) < (int)sizeof buf, "profile path");
    for (char *q = buf + 1; *q; q++)
        if ((*q == '/' || *q == '\\') && q[-1] != ':') {
            char c = *q;
            *q = 0;
            make_dir(buf);
            *q = c;
        }
    if (!make_dir(buf)) rt_fatal("fs: cannot create directory %s", buf);
}

// Remove a directory tree; a link (reparse point, symbolic link) is removed as a link, never followed. 0 also when
// nothing is there.
static int remove_tree(const char *path) {
    int kind = plat_fs_kind(path);
    if (kind == PLAT_FS_MISSING) return 0;
    if (kind == PLAT_FS_FILE) return plat_fs_unlink(path) ? -1 : 0;
    int bad = 0;
    if (kind == PLAT_FS_DIR) {
        Listing l;
        list_dir(path, &l);
        for (int i = 0; i < l.n; i++) {
            char *child = join(path, l.e[i].name);
            bad |= remove_tree(child);
            free(child);
        }
        listing_free(l.e, l.n);
    }
    return bad | (plat_fs_rmdir(path) ? -1 : 0);
}

// ---- save transactions
// Commit: (1) copy work to new and flush; (2) rename new to ready-<n>, the commit point; (3) rename save/<uid> to old;
// (4) rename ready-<n> to save/<uid>; (5) move old to save-backup/<UTC time>, keep the newest three.
// Recovery at start (save_recover): an unfinished new is deleted, the highest ready-<n> is rolled forward, an absent
// save/<uid> gets old back. Every change to the working copy waits in save_touched while a commit runs.
#define SAVE_BACKUPS 3
#define COPY_CHUNK (1u << 20)

static char tx_root[1100], backup_root[1100], new_root[1100], old_root[1100];
static char ready_root[1100];   // the newest finished commit not swapped in, empty when none
static unsigned ready_seq;      // the highest ready-<n> this run made or found
static PlatMutex tx_lock = PLAT_MUTEX_INIT;   // taken before lock, never inside it
static int save_mounts;                  // M_SAVE mounts: the working copy exists while this is above 0

// Probe hook: Commit stops at its tx_crash_at-th step (from 1) and leaves the disk as a process killed there would.
static int tx_crash_at, tx_step, tx_dead;
static int crashed(void) {
    if (tx_crash_at && ++tx_step == tx_crash_at) tx_dead = 1;
    return tx_dead;
}

static int exists(const char *p) { return plat_fs_kind(p) != PLAT_FS_MISSING; }

// Rename a directory durably, retried every 50 ms up to `tries` times while another process holds a file in it. 0 on success.
#define MOVE_TRIES 40   // 2 s
static int move_dir(const char *from, const char *to, int tries) {
    for (int i = 0; i < tries; i++) {
        int e = plat_fs_rename(from, to, 1);
        if (!e) return 0;
        if (e != PLAT_FS_BUSY) return -1;
        plat_sleep_ms(50);
    }
    return -1;
}

// Copy one file; the source may be open for writing (a save file the game has not closed). 0 on success.
static int copy_file(const char *src, const char *dst, int durable, uint64_t *bytes) {
    PlatFile *in = plat_fs_open(src, 0, NULL);
    if (!in) return -1;
    PlatFile *out = plat_fs_create(dst, NULL);
    if (!out) {
        plat_file_close(in);
        return -1;
    }
    uint8_t *buf = malloc(COPY_CHUNK);
    need(buf != NULL, "copy buffer");
    int ok = 1;
    for (uint64_t at = 0;;) {
        int64_t got = plat_file_pread(in, buf, COPY_CHUNK, at);
        if (got < 0) ok = 0;
        if (!ok || !got) break;
        if (plat_file_pwrite(out, buf, (uint32_t)got, at) != got) ok = 0;
        if (!ok) break;
        at += (uint64_t)got;
        *bytes += (uint64_t)got;
    }
    if (ok && durable) ok = plat_file_fsync(out);
    free(buf);
    plat_file_close(in);
    plat_file_close(out);
    return ok ? 0 : -1;
}

// Whether files a and b hold the same bytes; 0 when either cannot be read.
static int same_file(const char *a, const char *b) {
    PlatFile *h[2] = {plat_fs_open(a, 0, NULL), plat_fs_open(b, 0, NULL)};
    PlatFileId ia, ib;
    int same = h[0] && h[1] && plat_file_id(h[0], &ia) && plat_file_id(h[1], &ib) && ia.size == ib.size;
    uint8_t *buf = same ? malloc(2 * (size_t)COPY_CHUNK) : NULL;
    if (same) need(buf != NULL, "compare buffer");
    for (uint64_t at = 0; same;) {
        int64_t ga = plat_file_pread(h[0], buf, COPY_CHUNK, at), gb = plat_file_pread(h[1], buf + COPY_CHUNK, COPY_CHUNK, at);
        if (ga < 0 || ga != gb || memcmp(buf, buf + COPY_CHUNK, (size_t)ga)) same = 0;
        else if (!ga) break;
        at += (uint64_t)ga;
    }
    free(buf);
    for (int i = 0; i < 2; i++)
        if (h[i]) plat_file_close(h[i]);
    return same;
}

// Copy the tree src to dst, which must not exist; links are skipped. durable: every file and directory is flushed.
// base (or NULL): the committed save; a file equal to base's file at the same path becomes a hard link to it
// (*nlinked). 0 on success.
static int copy_tree(const char *src, const char *dst, const char *base, int durable, int *nfiles, int *nlinked,
                     uint64_t *bytes) {
    if (plat_fs_mkdir(dst)) return -1;
    Listing l;
    if (list_dir(src, &l)) return -1;
    int bad = 0;
    for (int i = 0; i < l.n && !bad; i++) {
        if (l.e[i].kind == PLAT_FS_LINK) continue;
        char *s = join(src, l.e[i].name), *d = join(dst, l.e[i].name);
        char *b = base ? join(base, l.e[i].name) : NULL;
        if (b && !exists(b)) free(b), b = NULL;
        if (l.e[i].kind == PLAT_FS_DIR) bad = copy_tree(s, d, b, durable, nfiles, nlinked, bytes);
        else {
            if (b && same_file(s, b) && !plat_fs_link(b, d)) bad = 0, ++*nlinked;
            else bad = copy_file(s, d, durable, bytes);
            *nfiles += !bad;
            if (!bad && durable && crashed()) bad = -1;
        }
        free(s);
        free(d);
        free(b);
    }
    listing_free(l.e, l.n);
    if (!bad && durable) plat_fs_sync_dir(dst);
    return bad;
}

static int listed_cmp(const void *a, const void *b) {
    return strcmp(((const Listed *)a)->name, ((const Listed *)b)->name);
}

// Whether paths a and b are one file (hard links of it).
static int same_inode(const char *a, const char *b) {
    PlatFile *h[2] = {plat_fs_open(a, 0, NULL), plat_fs_open(b, 0, NULL)};
    PlatFileId ia, ib;
    int same = h[0] && h[1] && plat_file_id(h[0], &ia) && plat_file_id(h[1], &ib) && ia.volume == ib.volume &&
               ia.file_id == ib.file_id;
    for (int i = 0; i < 2; i++)
        if (h[i]) plat_file_close(h[i]);
    return same;
}

// Whether tree a holds the names and kinds of tree b with each file a hard link of it.
static int same_tree(const char *a, const char *b) {
    Listing la, lb;
    int ea = list_dir(a, &la), eb = list_dir(b, &lb);
    int same = !ea && !eb && la.n == lb.n;
    if (same) {
        qsort(la.e, (size_t)la.n, sizeof *la.e, listed_cmp);
        qsort(lb.e, (size_t)lb.n, sizeof *lb.e, listed_cmp);
    }
    for (int i = 0; same && i < la.n; i++) {
        same = !strcmp(la.e[i].name, lb.e[i].name) && la.e[i].kind == lb.e[i].kind && la.e[i].kind != PLAT_FS_LINK;
        if (!same) break;
        char *pa = join(a, la.e[i].name), *pb = join(b, lb.e[i].name);
        same = la.e[i].kind == PLAT_FS_DIR ? same_tree(pa, pb) : same_inode(pa, pb);
        free(pa);
        free(pb);
    }
    listing_free(la.e, la.n);
    listing_free(lb.e, lb.n);
    return same;
}

static int name_cmp(const void *a, const void *b) { return strcmp(a, b); }

// The backups' folder names (digits first), sorted: oldest first, as rotate_backup names them in creation order.
static int backup_names(char (*names)[48]) {
    int n = 0;
    Listing l;
    list_dir(backup_root, &l);
    for (int i = 0; i < l.n; i++)
        if (l.e[i].kind == PLAT_FS_DIR && l.e[i].name[0] >= '0' && l.e[i].name[0] <= '9' && strlen(l.e[i].name) < 48 &&
            n < 64)
            strcpy(names[n++], l.e[i].name);
    listing_free(l.e, l.n);
    qsort(names, (size_t)n, sizeof *names, name_cmp);
    return n;
}

// Move old to save-backup/<UTC time>, keeping the newest SAVE_BACKUPS. The name must sort after every existing one
// (same millisecond, clock set back): a lower name would be the first pruned.
static void rotate_backup(void) {
    if (!exists(old_root)) return;
    static char names[64][48];
    int n = backup_names(names);
    uint64_t ns = plat_unix_time_ns();
    struct tm t;
    plat_gmtime((int64_t)(ns / 1000000000u), &t);
    char name[48];
    int len = snprintf(name, sizeof name, "%04d%02d%02d-%02d%02d%02d-%03u", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday,
                       t.tm_hour, t.tm_min, t.tm_sec, (unsigned)(ns / 1000000u % 1000u));
    const char *newest = n ? names[n - 1] : "";
    if (strcmp(name, newest) <= 0 && strlen(newest) >= (size_t)len) {
        unsigned k = 0;
        sscanf(newest + len, "-%u", &k);   // no suffix: k stays 0
        do   // until it sorts after newest, even after an unpadded "-1"
            snprintf(name, sizeof name, "%.*s-%03u", len, newest, ++k);
        while (strcmp(name, newest) <= 0 && k < 100000);
    }
    char *dst = join(backup_root, name);
    if (move_dir(old_root, dst, MOVE_TRIES)) {
        rt_log("fs: cannot move the replaced save to save-backup; deleted");
        remove_tree(old_root);
    }
    free(dst);
    n = backup_names(names);
    for (int i = 0; i + SAVE_BACKUPS < n; i++) {
        char *p = join(backup_root, names[i]);
        remove_tree(p);
        free(p);
    }
}

// The newest save: the finished commit waiting to be swapped in, else the committed save.
static const char *newest_save(void) { return *ready_root ? ready_root : save_root; }

// Steps (3) to (5): swap ready_root in for the committed save. 0 on success; on failure ready_root stays the newest
// save. (3) is tried for 200 ms only.
#define SWAP_TRIES 4
static int swap_in(void) {
    if (!*ready_root) return 0;
    rotate_backup();   // an old left by a failed (5): (3) needs its name
    if (exists(save_root) && move_dir(save_root, old_root, SWAP_TRIES)) {
        char why[512];
        plat_fs_busy_reason(save_root, why, sizeof why);
        rt_log("fs: save commit: the committed save cannot be replaced (%s); the commit waits in %s and the next "
               "Commit or start retries", *why ? why : "a file in it is in use", ready_root);
        return -1;
    }
    if (crashed()) return -1;
    if (move_dir(ready_root, save_root, MOVE_TRIES)) {
        rt_log("fs: save commit: cannot move the finished commit to the committed save; it waits in %s", ready_root);
        move_dir(old_root, save_root, MOVE_TRIES);   // else save/<uid> stays absent and the next start rolls forward
        return -1;
    }
    *ready_root = 0;
    if (crashed()) return -1;
    rotate_backup();
    return 0;
}

// The number n of a "ready-<n>" name, 0 for any other name.
static unsigned ready_number(const char *name) {
    char *end;
    if (strncmp(name, "ready-", 6) || name[6] < '0' || name[6] > '9') return 0;
    unsigned long n = strtoul(name + 6, &end, 10);
    return *end || n > 0xffffffffu ? 0 : (unsigned)n;
}

// Startup: the newest finished commit into ready_root; the older ones, which it supersedes, are deleted.
static void ready_scan(void) {
    Listing l;
    list_dir(tx_root, &l);
    *ready_root = 0, ready_seq = 0;
    for (int i = 0; i < l.n; i++)
        if (l.e[i].kind == PLAT_FS_DIR && ready_number(l.e[i].name) > ready_seq) ready_seq = ready_number(l.e[i].name);
    for (int i = 0; i < l.n; i++) {
        unsigned n = l.e[i].kind == PLAT_FS_DIR ? ready_number(l.e[i].name) : 0;
        char *p = n ? join(tx_root, l.e[i].name) : NULL;
        if (n == ready_seq && p) snprintf(ready_root, sizeof ready_root, "%s", p);
        else if (p) remove_tree(p);
        free(p);
    }
    listing_free(l.e, l.n);
}

// Startup: finish or roll back an interrupted Commit and drop uncommitted changes (the rule above).
static void save_recover(void) {
    const char *did = "nothing to recover";
    ready_scan();
    int has_save = exists(save_root);
    if (exists(new_root) && !has_save && !*ready_root) {
        if (move_dir(new_root, save_root, MOVE_TRIES)) rt_fatal("fs: cannot complete an interrupted save commit");
        did = "completed an interrupted commit";
        has_save = 1;
    } else if (exists(new_root)) {
        if (remove_tree(new_root)) rt_fatal("fs: cannot delete an unfinished save commit");
        did = "rolled back an unfinished commit";
    }
    if (!has_save && !*ready_root && exists(old_root)) {
        if (move_dir(old_root, save_root, MOVE_TRIES)) rt_fatal("fs: cannot roll back an interrupted save commit");
        did = "rolled back an interrupted commit";
    } else if (exists(old_root))
        did = "moved the save an interrupted commit replaced to save-backup";
    rotate_backup();
    if (*ready_root)
        did = swap_in() ? "a finished commit still waits in save-tx; the game loads it from there"
                        : "completed a finished commit";
    if (exists(work_root)) {
        if (remove_tree(work_root)) rt_fatal("fs: cannot delete the old save working copy");
        if (!strcmp(did, "nothing to recover"))
            did = "deleted the last run's working copy (changes after its last Commit)";
    }
    rt_log("host save recovery: %s", did);
}

// The first save mount: a fresh working copy of the newest save.
static void work_create(void) {
    int n = 0, linked = 0;
    uint64_t bytes = 0;
    if (remove_tree(work_root) || copy_tree(newest_save(), work_root, NULL, 0, &n, &linked, &bytes))
        rt_fatal("fs: cannot make the save working copy");
    rt_log("host save working copy: %d files, %.1f MiB", n, bytes / 1048576.0);
}

// The last save unmount: changes no Commit took are dropped.
static void work_discard(void) {
    if (plat_xchg32(&save_dirty, 0)) rt_log("fs: save unmounted without Commit; changes discarded");
    if (remove_tree(work_root)) rt_log("fs: save working copy not fully deleted (a file still open?)");
}

// Commit of the save mount, tx_lock held. Steps (1) to (5) above; a failure before (2) leaves the newest save as it
// was, one after it leaves the commit waiting in ready_root.
static uint32_t save_commit_locked(void) {
    uint64_t t0 = plat_now_ns(), bytes = 0;
    int nfiles = 0, linked = 0;
    tx_step = tx_dead = 0;
    const char *base = exists(newest_save()) ? newest_save() : NULL;
    if (remove_tree(new_root) || copy_tree(work_root, new_root, base, 1, &nfiles, &linked, &bytes)) {
        if (tx_dead) return FS_R_INVALID_ARGUMENT;
        rt_log("fs: save commit: copying the working copy failed; the newest save is unchanged");
        remove_tree(new_root);
        return FS_R_INVALID_ARGUMENT;
    }
    if (crashed()) return FS_R_INVALID_ARGUMENT;
    if (base && linked == nfiles && same_tree(new_root, base)) {   // the game rewrote equal bytes: no save, no backup
        remove_tree(new_root);
        plat_xchg32(&save_dirty, 0);
        int waits = swap_in();
        if (tx_dead) return FS_R_INVALID_ARGUMENT;
        rt_log("fs commit save: %d files, none changed, in %.0f ms%s", nfiles, (plat_now_ns() - t0) / 1e6,
               waits ? "; the newest save waits in save-tx" : "");
        return 0;
    }
    char ready[1100];
    snprintf(ready, sizeof ready, "%s/ready-%010u", tx_root, ready_seq + 1);
    if (move_dir(new_root, ready, MOVE_TRIES)) {
        rt_log("fs: save commit: cannot finish the staging copy; the newest save is unchanged");
        remove_tree(new_root);
        return FS_R_INVALID_ARGUMENT;
    }
    ready_seq++;
    if (*ready_root) remove_tree(ready_root);   // superseded: the working copy it was made from held all of it
    snprintf(ready_root, sizeof ready_root, "%s", ready);
    plat_xchg32(&save_dirty, 0);
    if (crashed()) return FS_R_INVALID_ARGUMENT;
    int waits = swap_in();
    if (tx_dead) return FS_R_INVALID_ARGUMENT;
    rt_log("fs commit save: %d files, %d changed (%.1f MiB written), %d unchanged linked, in %.0f ms%s", nfiles,
           nfiles - linked, bytes / 1048576.0, linked, (plat_now_ns() - t0) / 1e6, waits ? "; waits in save-tx" : "");
    return 0;
}

// The background commit: commit_pending from the game's Commit until the thread has run save_commit_locked.
static PlatMutex pend_lock = PLAT_MUTEX_INIT;
static PlatCond pend_cv = PLAT_COND_INIT;
static volatile int32_t commit_pending;

static void save_wait_commit(void) {
    if (!plat_cas32(&commit_pending, 0, 0)) return;
    plat_mutex_lock(&pend_lock);
    while (commit_pending) plat_cond_wait(&pend_cv, &pend_lock);
    plat_mutex_unlock(&pend_lock);
}

void fs_save_flush(void) { save_wait_commit(); }

static uint32_t commit_run(void) {
    plat_mutex_lock(&tx_lock);
    uint32_t r = save_commit_locked();
    plat_dec32(&save_writers);
    plat_mutex_unlock(&tx_lock);
    plat_mutex_lock(&pend_lock);
    commit_pending = 0;
    plat_mutex_unlock(&pend_lock);
    plat_cond_broadcast(&pend_cv);
    return r;
}

static void commit_main(void *arg) {
    (void)arg;
    commit_run();
    return;
}

static uint32_t save_commit(void) {
    save_wait_commit();   // the previous Commit
    static int quiet;
    if (!plat_cas32(&save_dirty, 0, 0)) {
        if (!quiet++) rt_log("fs commit save: nothing changed (logged once)");
        return 0;
    }
    plat_inc32(&save_writers);   // fs_save_busy until the thread is done
    plat_xchg32(&commit_pending, 1);
    if (plat_thread_spawn(commit_main, NULL, "save commit", 0)) {
        return 0;
    }
    rt_log("fs: cannot start the save commit thread; committing on the game thread");
    return commit_run();
}

// ---- storage roots
static void storage_init(void *param) {
    (void)param;
    if (!*profile)
        need(snprintf(profile, sizeof profile, "%s", rt_config()->profile) < (int)sizeof profile, "profile path");
    uint64_t uid[2];
    char p[PATH_CAP];
    make_dirs(profile);
    account_uid(uid);
    snprintf(p, sizeof p, "%s/save", profile);
    make_dirs(p);
    need(snprintf(save_root, sizeof save_root, "%s/save/%016llx%016llx", profile, (unsigned long long)uid[1],
                  (unsigned long long)uid[0]) < (int)sizeof save_root, "save path");
    need(snprintf(tx_root, sizeof tx_root, "%s/save-tx", profile) < (int)sizeof tx_root, "save path");
    snprintf(work_root, sizeof work_root, "%s/work", tx_root);
    snprintf(backup_root, sizeof backup_root, "%s/save-backup", profile);
    make_dirs(tx_root);
    make_dirs(backup_root);
    snprintf(new_root, sizeof new_root, "%s/new", tx_root);
    snprintf(old_root, sizeof old_root, "%s/old", tx_root);
    save_recover();   // before save_root is created: an absent save_root means a commit to complete
    make_dirs(save_root);
    snprintf(p, sizeof p, "%s/temp", profile);
    remove_tree(p);
    make_dirs(p);
    snprintf(p, sizeof p, "%s/cache", profile);
    make_dirs(p);
    plat_xchg32(&storage_ready, 1);
    rt_log("host save storage: %s", save_root);
    return;
}

static void ready(void) {
    static PlatOnce once = PLAT_ONCE_INIT;
    plat_once(&once, storage_init, NULL);
}

void fs_host_set_profile(const char *dir) {
    if (plat_cas32(&storage_ready, 0, 0)) rt_fatal("fs_host_set_profile after the first storage use");
    need(dir && snprintf(profile, sizeof profile, "%s", dir) < (int)sizeof profile, "profile path");
}

// ---- mounts and paths
static uint32_t mount_add(const char *name, int kind, const char *root) {
    if (!name || !*name || strlen(name) >= 32 || strchr(name, ':') || strchr(name, '/')) return FS_R_INVALID_ARGUMENT;
    char as_path[40];
    snprintf(as_path, sizeof as_path, "%s:", name);
    if (romfs_owns(as_path)) return FS_R_ALREADY_EXISTS;   // the RomFS mount's name
    plat_mutex_lock(&lock);
    int slot = -1;
    for (int i = 0; i < NMOUNTS; i++) {
        if (mounts[i].kind && !strcmp(mounts[i].name, name)) {
            plat_mutex_unlock(&lock);
            return FS_R_ALREADY_EXISTS;
        }
        if (!mounts[i].kind && slot < 0) slot = i;
    }
    if (slot < 0) rt_fatal("fs mount table full");
    mounts[slot].kind = kind;
    strcpy(mounts[slot].name, name);
    snprintf(mounts[slot].root, sizeof mounts[slot].root, "%s", root);
    plat_mutex_unlock(&lock);
    rt_log("fs mount %s: -> %s", name, root);
    return 0;
}

// A path component every host stores under exactly this name: not empty, ".", "..", no reserved character, no trailing
// dot or space (Windows strips those, which would alias two names).
static int component_ok(const char *c, size_t n) {
    if (!n || (n == 1 && c[0] == '.') || (n == 2 && c[0] == '.' && c[1] == '.')) return 0;
    if (c[n - 1] == '.' || c[n - 1] == ' ') return 0;
    for (size_t i = 0; i < n; i++)
        if ((unsigned char)c[i] < 32 || strchr("\\:*?\"<>|", c[i])) return 0;
    return 1;
}

// "name:/a/b" -> host path; 0 on success; *kind = the mount's kind (M_SAVE ...) when kind is given.
static uint32_t host_path_kind(const char *path, char *out, size_t cap, int *kind) {
    if (!path) return FS_R_INVALID_PATH;
    const char *colon = strchr(path, ':');
    if (!colon || colon[1] != '/') return FS_R_INVALID_PATH;
    size_t n = (size_t)(colon - path);
    char root[1024];
    int k = M_FREE;
    plat_mutex_lock_shared(&lock);
    for (int i = 0; i < NMOUNTS; i++)
        if (mounts[i].kind && strlen(mounts[i].name) == n && !strncmp(mounts[i].name, path, n)) {
            strcpy(root, mounts[i].root);
            k = mounts[i].kind;
        }
    plat_mutex_unlock_shared(&lock);
    if (k == M_FREE) return FS_R_PATH_NOT_FOUND;
    if (kind) *kind = k;
    const char *rest = colon + 1;
    for (const char *q = rest + 1; *q;) {
        const char *e = strchr(q, '/');
        size_t len = e ? (size_t)(e - q) : strlen(q);
        if (!e && !len) break;
        if (!component_ok(q, len)) return FS_R_INVALID_PATH;
        q = e ? e + 1 : q + len;
    }
    if (snprintf(out, cap, "%s%s", root, rest) >= (int)cap) return FS_R_INVALID_PATH;
    size_t L = strlen(out);
    while (L > strlen(root) && out[L - 1] == '/') out[--L] = 0;
    return 0;
}

static uint32_t host_path(const char *path, char *out, size_t cap) { return host_path_kind(path, out, cap, NULL); }

// A change to save data through a path (create, delete, rename, directories) marks the save as in progress.
static uint32_t save_change_path(const char *path, char *out, size_t cap) {
    int kind = M_FREE;
    uint32_t r = host_path_kind(path, out, cap, &kind);
    if (!r && kind == M_SAVE_RO) return FS_R_UNSUPPORTED_OPEN;
    if (!r && kind == M_SAVE) save_touched();
    return r;
}

static uint32_t fs_result(int e) {
    switch (e) {
    case PLAT_FS_OK: return 0;
    case PLAT_FS_NOT_FOUND: return FS_R_PATH_NOT_FOUND;
    case PLAT_FS_EXISTS: return FS_R_ALREADY_EXISTS;
    case PLAT_FS_NOT_EMPTY: return FS_R_NOT_EMPTY;
    default: return FS_R_INVALID_ARGUMENT;
    }
}

// ---- storage mounts
static int my_uid(const uint64_t *uid) {
    uint64_t me[2];
    account_uid(me);
    return uid && uid[0] == me[0] && uid[1] == me[1];
}

static uint32_t mount_save(const char *name, const uint64_t *uid) {
    ready();
    if (!my_uid(uid)) return FS_R_TARGET_NOT_FOUND;
    plat_mutex_lock(&tx_lock);
    if (!save_mounts) work_create();
    uint32_t r = mount_add(name, M_SAVE, work_root);
    if (!r) save_mounts++;
    else if (!save_mounts) remove_tree(work_root);
    plat_mutex_unlock(&tx_lock);
    return r;
}

static uint8_t save_exists(uint64_t application, const uint64_t *uid) {
    ready();
    uint8_t r = application == OOT_APPLICATION_ID && my_uid(uid);
    static volatile int32_t logged;   // which application ids the game asks about (ours is RT_APPLICATION_ID)
    if (logged < 8 && plat_inc32(&logged) <= 8)
        rt_log("fs: save data of application %016llx asked for (ours %016llx, user %s): %s", (unsigned long long)application,
               (unsigned long long)OOT_APPLICATION_ID, my_uid(uid) ? "ours" : "other", r ? "present" : "not present");
    return r;
}

// Other titles' save data (data-link bonuses) is not present on this host.
static uint32_t mount_save_ro(const char *name, uint64_t application, const uint64_t *uid) {
    if (!save_exists(application, uid)) return FS_R_TARGET_NOT_FOUND;
    return mount_add(name, M_SAVE_RO, save_root);
}

static uint32_t mount_temp(const char *name) {
    ready();
    char p[PATH_CAP];
    snprintf(p, sizeof p, "%s/temp", profile);
    return mount_add(name, M_TEMP, p);
}

static void cache_dir(int index, char *out, size_t cap) { snprintf(out, cap, "%s/cache/%d", profile, index); }

static uint32_t create_cache(int index, int64_t size, int64_t journal) {
    ready();
    if (index < 0 || size < 0 || journal < 0) return FS_R_INVALID_ARGUMENT;
    char p[PATH_CAP];
    cache_dir(index, p, sizeof p);
    return fs_result(plat_fs_mkdir(p));
}

static uint32_t delete_cache(int index) {
    ready();
    char p[PATH_CAP];
    cache_dir(index, p, sizeof p);
    return remove_tree(p) ? FS_R_TARGET_NOT_FOUND : 0;
}

static uint32_t mount_cache(const char *name, int index) {
    ready();
    char p[PATH_CAP];
    cache_dir(index, p, sizeof p);
    if (plat_fs_kind(p) != PLAT_FS_DIR) return FS_R_TARGET_NOT_FOUND;
    return mount_add(name, M_CACHE, p);
}

typedef struct { int32_t index; uint8_t reserved[0x1c]; } CacheStorageInfo;
_Static_assert(sizeof(CacheStorageInfo) == 0x20, "nn::fs::CacheStorageInfo");
typedef struct { int list[64]; int count, next; } CacheList;

static uint32_t open_cache_list(uint64_t *out) {
    ready();
    if (!out) return FS_R_INVALID_ARGUMENT;
    CacheList *l = calloc(1, sizeof *l);
    need(l != NULL, "cache list allocation");
    char p[PATH_CAP];
    snprintf(p, sizeof p, "%s/cache", profile);
    Listing ls;
    list_dir(p, &ls);
    for (int i = 0; i < ls.n; i++) {
        char *end;
        long v = strtol(ls.e[i].name, &end, 10);
        if (ls.e[i].kind == PLAT_FS_DIR && *ls.e[i].name && !*end && v >= 0 && l->count < 64) l->list[l->count++] = (int)v;
    }
    listing_free(ls.e, ls.n);
    *out = (uint64_t)(uintptr_t)l;
    return 0;
}

static uint32_t read_cache_list(int *count, CacheStorageInfo *out, uint64_t h, int max) {
    CacheList *l = (CacheList *)(uintptr_t)h;
    if (!l || !count || max < 0 || (!out && max)) return FS_R_INVALID_ARGUMENT;
    int n = 0;
    while (n < max && l->next < l->count) {
        memset(&out[n], 0, sizeof *out);
        out[n++].index = l->list[l->next++];
    }
    *count = n;
    return 0;
}

static void close_cache_list(uint64_t h) { free((void *)(uintptr_t)h); }

static void unmount(const char *name) {
    save_wait_commit();   // else the working copy could go before a Commit queued just now has copied it
    plat_mutex_lock(&tx_lock);
    plat_mutex_lock(&lock);
    for (int i = 0; i < NMOUNTS; i++)
        if (name && mounts[i].kind && !strcmp(mounts[i].name, name)) {
            int save = mounts[i].kind == M_SAVE;
            mounts[i].kind = M_FREE;
            plat_mutex_unlock(&lock);
            if (save && !--save_mounts) work_discard();
            plat_mutex_unlock(&tx_lock);
            rt_log("fs unmount %s", name);
            return;
        }
    plat_mutex_unlock(&lock);
    plat_mutex_unlock(&tx_lock);
    size_t n = name ? strlen(name) : 0;
    char probe[40];
    if (!n || n >= 32 || (snprintf(probe, sizeof probe, "%s:", name), !romfs_owns(probe)))
        rt_fatal("fs: unmount of unknown mount %s", name ? name : "(null)");
}

static uint32_t commit(const char *name) {
    int found = 0, save = 0;
    plat_mutex_lock_shared(&lock);
    for (int i = 0; i < NMOUNTS; i++)
        if (name && mounts[i].kind && !strcmp(mounts[i].name, name)) found = 1, save = mounts[i].kind == M_SAVE;
    plat_mutex_unlock_shared(&lock);
    if (!found) return FS_R_PATH_NOT_FOUND;
    if (save) return save_commit();   // durability comes from the staging copy, not the working copy's handles
    plat_mutex_lock_shared(&lock);
    for (int i = 0; i < NFILES; i++)
        if (files[i].h && (files[i].mode & FS_OPEN_WRITE)) plat_file_fsync(files[i].h);
    plat_mutex_unlock_shared(&lock);
    return 0;
}

// ---- files
static File *file_of(uint64_t h, PlatFile **fh, int *mode) {
    if (h < FILE_BASE || h >= FILE_BASE + NFILES) return NULL;
    plat_mutex_lock_shared(&lock);
    File *f = &files[h - FILE_BASE];
    PlatFile *v = f->h;
    if (fh) *fh = f->h;
    if (mode) *mode = f->mode;
    plat_mutex_unlock_shared(&lock);
    return v ? f : NULL;
}

// Positional reads: bytes read at off, 0 at the end, -1 on an error.
static int64_t host_read_at(PlatFile *f, void *buf, uint32_t n, uint64_t off) {
    mem_thaw((uintptr_t)buf, n);   // the kernel's write would fail on a frozen page instead of faulting
    int64_t got = plat_file_pread(f, buf, n, off);
    if (got == -2)   // a read-only page in the buffer: a frozen region missed its thaw
        rt_log("fs: read of %u bytes into %p refused: the buffer is not writable", n, buf);
    return got < 0 ? -1 : got;
}

static uint32_t open_file(uint64_t *out, const char *path, int mode) {
    if (romfs_owns(path)) return romfs_open(out, path, mode);
    fs_trace(mode & FS_OPEN_WRITE ? "write" : "read", path);
    if (!out || !(mode & (FS_OPEN_READ | FS_OPEN_WRITE)) || (mode & ~7)) return FS_R_INVALID_ARGUMENT;
    char p[PATH_CAP];
    int kind = M_FREE;
    uint32_t r = host_path_kind(path, p, sizeof p, &kind);
    if (r) return r;
    if (plat_fs_kind(p) != PLAT_FS_FILE) return FS_R_PATH_NOT_FOUND;
    if (kind == M_SAVE_RO && (mode & FS_OPEN_WRITE)) return FS_R_UNSUPPORTED_OPEN;
    int save_writer = kind == M_SAVE && (mode & FS_OPEN_WRITE);
    int err = 0;
    PlatFile *f = plat_fs_open(p, (mode & FS_OPEN_WRITE) != 0, &err);
    if (!f) return fs_result(err);
    plat_mutex_lock(&lock);
    for (int i = 0; i < NFILES; i++)
        if (!files[i].h) {
            files[i] = (File){f, mode, save_writer};
            *out = FILE_BASE + (uint64_t)i;
            plat_mutex_unlock(&lock);
            if (save_writer) {
                plat_inc32(&save_writers);
                save_touched();
            }
            return 0;
        }
    plat_mutex_unlock(&lock);
    rt_fatal("fs: host file handle table full (%d)", NFILES);
}

static uint32_t read_file(uint64_t *out, uint64_t h, int64_t offset, void *buffer, uint64_t size) {
    if (romfs_handle(h)) return romfs_read(out, h, offset, buffer, size);
    PlatFile *f;
    int mode;
    if (!out || offset < 0 || (!buffer && size) || !file_of(h, &f, &mode)) return FS_R_INVALID_ARGUMENT;
    if (!(mode & FS_OPEN_READ)) return FS_R_OPEN_MODE;
    uint64_t done = 0;
    while (done < size) {
        uint32_t chunk = size - done > 0x40000000 ? 0x40000000 : (uint32_t)(size - done);
        int64_t n = host_read_at(f, (uint8_t *)buffer + done, chunk, (uint64_t)offset + done);
        if (n < 0) rt_fatal("fs: host save read failure");
        if (!n) break;
        done += (uint64_t)n;
    }
    *out = done;
    return 0;
}

static uint32_t read_exact(uint64_t h, int64_t off, void *buf, uint64_t size) {
    uint64_t n = 0;
    uint32_t r = read_file(&n, h, off, buf, size);
    return r ? r : n == size ? 0 : FS_R_INVALID_ARGUMENT;
}

static uint32_t write_file(uint64_t h, int64_t offset, const void *buffer, uint64_t size, const int32_t *option) {
    PlatFile *f;
    int mode;
    File *file = offset < 0 || (!buffer && size) ? NULL : file_of(h, &f, &mode);
    if (!file) return FS_R_INVALID_ARGUMENT;
    if (!(mode & FS_OPEN_WRITE)) return FS_R_OPEN_MODE;
    PlatFileId id;
    if (!plat_file_id(f, &id)) rt_fatal("fs: host save size");
    if ((uint64_t)offset + size > id.size && !(mode & FS_OPEN_APPEND)) return FS_R_APPEND_REQUIRED;
    if (file->save) save_touched();
    for (uint64_t done = 0; done < size;) {
        uint32_t chunk = size - done > 0x40000000 ? 0x40000000 : (uint32_t)(size - done);
        int64_t put = plat_file_pwrite(f, (const uint8_t *)buffer + done, chunk, (uint64_t)offset + done);
        if (put <= 0) rt_fatal("fs: host save write failure");
        done += (uint64_t)put;
    }
    if (option && (*option & 1)) plat_file_fsync(f);   // WriteOptionFlag_Flush
    return 0;
}

static uint32_t file_size(int64_t *out, uint64_t h) {
    if (romfs_handle(h)) return romfs_size(out, h);
    PlatFile *f;
    PlatFileId id;
    if (!out || !file_of(h, &f, NULL) || !plat_file_id(f, &id)) return FS_R_INVALID_ARGUMENT;
    *out = (int64_t)id.size;
    return 0;
}

static uint32_t set_file_size(uint64_t h, int64_t size) {
    PlatFile *f;
    int mode;
    File *file = size < 0 ? NULL : file_of(h, &f, &mode);
    if (!file) return FS_R_INVALID_ARGUMENT;
    if (!(mode & FS_OPEN_WRITE)) return FS_R_OPEN_MODE;
    if (file->save) save_touched();
    return fs_result(plat_file_set_size(f, (uint64_t)size));
}

static uint32_t flush_file(uint64_t h) {
    PlatFile *f;
    int mode;
    if (!file_of(h, &f, &mode)) return FS_R_INVALID_ARGUMENT;
    if (mode & FS_OPEN_WRITE) plat_file_fsync(f);
    return 0;
}

static void close_file(uint64_t h) {
    if (romfs_handle(h)) {
        romfs_close(h);
        return;
    }
    if (!file_of(h, NULL, NULL)) rt_fatal("fs: close of invalid file handle %#llx", (unsigned long long)h);
    plat_mutex_lock(&lock);
    PlatFile *f = files[h - FILE_BASE].h;
    int save = files[h - FILE_BASE].save;
    files[h - FILE_BASE] = (File){0};
    plat_mutex_unlock(&lock);
    plat_file_close(f);
    if (save) {
        save_touched();
        plat_dec32(&save_writers);
    }
}

static uint32_t create_file(const char *path, int64_t size) {
    char p[PATH_CAP];
    uint32_t r = save_change_path(path, p, sizeof p);
    if (r) return r;
    if (size < 0) return FS_R_INVALID_ARGUMENT;
    int err = 0;
    PlatFile *f = plat_fs_create(p, &err);
    if (!f) return fs_result(err);
    err = size ? plat_file_set_size(f, (uint64_t)size) : 0;
    plat_file_close(f);
    return fs_result(err);
}

static uint32_t delete_file(const char *path) {
    char p[PATH_CAP];
    uint32_t r = save_change_path(path, p, sizeof p);
    if (r) return r;
    if (plat_fs_kind(p) == PLAT_FS_DIR) return FS_R_INVALID_ARGUMENT;   // DeleteFile on a directory
    return fs_result(plat_fs_unlink(p));
}

static uint32_t rename_file(const char *from, const char *to) {
    char a[PATH_CAP], b[PATH_CAP];
    uint32_t r = save_change_path(from, a, sizeof a);
    if (!r) r = host_path(to, b, sizeof b);
    if (r) return r;
    size_t n = strcspn(from, ":");
    if (strcspn(to, ":") != n || strncmp(from, to, n)) return FS_R_INVALID_PATH;   // across mounts
    if (exists(b)) return FS_R_ALREADY_EXISTS;
    return fs_result(plat_fs_rename(a, b, 0));
}

static uint32_t entry_type(int *out, const char *path) {
    if (romfs_owns(path)) return romfs_entry_type(out, path);
    char p[PATH_CAP];
    uint32_t r = host_path(path, p, sizeof p);
    if (r) return r;
    if (!out) return FS_R_INVALID_ARGUMENT;
    int kind = plat_fs_kind(p);
    if (kind == PLAT_FS_MISSING) return FS_R_PATH_NOT_FOUND;
    *out = kind == PLAT_FS_DIR ? FS_ENTRY_DIRECTORY : FS_ENTRY_FILE;
    return 0;
}

// ---- directories
typedef struct { char name[769]; uint8_t pad0[3]; int8_t type; uint8_t pad1[3]; int64_t size; } DirectoryEntry;
_Static_assert(sizeof(DirectoryEntry) == 0x310, "nn::fs::DirectoryEntry");

static uint32_t create_directory(const char *path) {
    char p[PATH_CAP];
    uint32_t r = save_change_path(path, p, sizeof p);
    if (r) return r;
    return fs_result(plat_fs_mkdir(p));
}

static uint32_t delete_recursive(const char *path) {
    char p[PATH_CAP];
    uint32_t r = save_change_path(path, p, sizeof p);
    if (r) return r;
    if (plat_fs_kind(p) != PLAT_FS_DIR) return FS_R_PATH_NOT_FOUND;
    return remove_tree(p) ? FS_R_INVALID_ARGUMENT : 0;
}

static uint32_t open_directory(uint64_t *out, const char *path, int mode) {
    Listing l;
    if (romfs_owns(path)) {   // the RomFS: the children of the folder, taken from its sorted file index
        if (!out || mode < 1 || mode > 3) return FS_R_INVALID_ARGUMENT;
        l = (Listing){0};
        uint32_t rr = romfs_list(path, listing_add, &l);
        if (rr) {
            listing_free(l.e, l.n);
            return rr;
        }
    } else {
        char p[PATH_CAP];
        uint32_t r = host_path(path, p, sizeof p);
        if (r) return r;
        if (!out || mode < 1 || mode > 3) return FS_R_INVALID_ARGUMENT;
        if (plat_fs_kind(p) != PLAT_FS_DIR) return FS_R_PATH_NOT_FOUND;
        int e = list_dir(p, &l);
        if (e) return fs_result(e);
    }
    plat_mutex_lock(&lock);
    for (int i = 0; i < NDIRS; i++)
        if (!dirs[i].used) {
            dirs[i] = (Dir){l.e, l.n, 0, mode, 1};
            *out = DIR_BASE + (uint64_t)i;
            plat_mutex_unlock(&lock);
            return 0;
        }
    plat_mutex_unlock(&lock);
    rt_fatal("fs: host directory handle table full (%d)", NDIRS);
}

static uint32_t read_directory(int64_t *count, DirectoryEntry *out, uint64_t h, int64_t max) {
    if (h < DIR_BASE || h >= DIR_BASE + NDIRS || !count || max < 0 || (!out && max)) return FS_R_INVALID_ARGUMENT;
    Dir *d = &dirs[h - DIR_BASE];
    if (!d->used) return FS_R_INVALID_ARGUMENT;
    int64_t n = 0;
    while (n < max && d->next < d->n) {
        const Listed *e = &d->e[d->next++];
        size_t k = strlen(e->name) + 1;
        if (k > 769) continue;   // over 768 bytes: nn::fs cannot name it
        int is_dir = e->kind == PLAT_FS_DIR;
        if (!(d->mode & (is_dir ? 1 : 2))) continue;
        memset(&out[n], 0, sizeof *out);
        memcpy(out[n].name, e->name, k);
        out[n].type = is_dir ? FS_ENTRY_DIRECTORY : FS_ENTRY_FILE;
        out[n].size = is_dir ? 0 : (int64_t)e->size;
        n++;
    }
    *count = n;
    return 0;
}

static void close_directory(uint64_t h) {
    if (h < DIR_BASE || h >= DIR_BASE + NDIRS || !dirs[h - DIR_BASE].used)
        rt_fatal("fs: close of invalid directory handle %#llx", (unsigned long long)h);
    plat_mutex_lock(&lock);
    Dir d = dirs[h - DIR_BASE];
    dirs[h - DIR_BASE] = (Dir){0};
    plat_mutex_unlock(&lock);
    listing_free(d.e, d.n);
}

// ---- overrides: AAPCS64 thunks. FileHandle, DirectoryHandle, CacheStorageListHandle
// and ApplicationId are 8-byte structs passed in one register; Uid, ReadOption and WriteOption by reference.
#define I32(i) ((int32_t)arg_x(c, i))
#define I64(i) ((int64_t)arg_x(c, i))
// The first 120 opens of speech files are logged.
static void fs_repeat_note(const char *path) {
    if (path && (rt_trace & RT_TRACE_AUDIO) && (strstr(path, "Voice/") || strstr(path, "Sound/Resource/Stream/Dm_"))) {
        static int32_t voice_opens;
        int n = __atomic_add_fetch(&voice_opens, 1, __ATOMIC_RELAXED);
        if (n <= 120) rt_log("fs: voice open %d: %s", n, path);
    }
}
static void ov_open_file(Cpu *c) { fs_repeat_note(arg_p(c, 1)); ret_x(c, open_file(arg_p(c, 0), arg_p(c, 1), I32(2))); }
static void ov_file_size(Cpu *c) { ret_x(c, file_size(arg_p(c, 0), arg_x(c, 1))); }
static void ov_close_file(Cpu *c) { close_file(arg_x(c, 0)); }
static void ov_entry_type(Cpu *c) { ret_x(c, entry_type(arg_p(c, 0), arg_p(c, 1))); }
static void ov_read_file(Cpu *c) { ret_x(c, read_file(arg_p(c, 0), arg_x(c, 1), I64(2), arg_p(c, 3), arg_x(c, 4))); }
static void ov_read_options(Cpu *c) { ret_x(c, read_file(arg_p(c, 0), arg_x(c, 1), I64(2), arg_p(c, 3), arg_x(c, 4))); }
static void ov_read_exact(Cpu *c) { ret_x(c, read_exact(arg_x(c, 0), I64(1), arg_p(c, 2), arg_x(c, 3))); }
static void ov_write_file(Cpu *c) { ret_x(c, write_file(arg_x(c, 0), I64(1), arg_p(c, 2), arg_x(c, 3), arg_p(c, 4))); }
static void ov_set_file_size(Cpu *c) { ret_x(c, set_file_size(arg_x(c, 0), I64(1))); }
static void ov_flush_file(Cpu *c) { ret_x(c, flush_file(arg_x(c, 0))); }
static void ov_create_file(Cpu *c) { ret_x(c, create_file(arg_p(c, 0), I64(1))); }
static void ov_delete_file(Cpu *c) { ret_x(c, delete_file(arg_p(c, 0))); }
static void ov_rename_file(Cpu *c) { ret_x(c, rename_file(arg_p(c, 0), arg_p(c, 1))); }
static void ov_create_directory(Cpu *c) { ret_x(c, create_directory(arg_p(c, 0))); }
static void ov_delete_recursive(Cpu *c) { ret_x(c, delete_recursive(arg_p(c, 0))); }
static void ov_open_directory(Cpu *c) { ret_x(c, open_directory(arg_p(c, 0), arg_p(c, 1), I32(2))); }
static void ov_read_directory(Cpu *c) { ret_x(c, read_directory(arg_p(c, 0), arg_p(c, 1), arg_x(c, 2), I64(3))); }
static void ov_close_directory(Cpu *c) { close_directory(arg_x(c, 0)); }
static void ov_mount_save(Cpu *c) { ret_x(c, mount_save(arg_p(c, 0), arg_p(c, 1))); }
static void ov_mount_save_ro(Cpu *c) { ret_x(c, mount_save_ro(arg_p(c, 0), arg_x(c, 1), arg_p(c, 2))); }
static void ov_save_exists(Cpu *c) { ret_x(c, save_exists(arg_x(c, 0), arg_p(c, 1))); }
// nn::fs::EnsureSaveData(const account::Uid &): the save already "exists"; mount_save makes its working copy.
static void ov_ensure_save(Cpu *c) {
    ready();
    ret_x(c, my_uid(arg_p(c, 0)) ? 0 : FS_R_TARGET_NOT_FOUND);
}
static void ov_mount_temp(Cpu *c) { ret_x(c, mount_temp(arg_p(c, 0))); }
static void ov_unmount(Cpu *c) { unmount(arg_p(c, 0)); }
static void ov_commit(Cpu *c) { ret_x(c, commit(arg_p(c, 0))); }
static void ov_create_cache(Cpu *c) { ret_x(c, create_cache(I32(0), I64(1), I64(2))); }
static void ov_delete_cache(Cpu *c) { ret_x(c, delete_cache(I32(0))); }
static void ov_mount_cache(Cpu *c) { ret_x(c, mount_cache(arg_p(c, 0), I32(1))); }
static void ov_open_cache_list(Cpu *c) { ret_x(c, open_cache_list(arg_p(c, 0))); }
static void ov_read_cache_list(Cpu *c) { ret_x(c, read_cache_list(arg_p(c, 0), arg_p(c, 1), arg_x(c, 2), I32(3))); }
static void ov_close_cache_list(Cpu *c) { close_cache_list(arg_x(c, 0)); }
// The fs priority only orders I/O inside the system fs process; host I/O is unordered.
static void ov_set_priority(Cpu *c) { (void)c; }
#undef I32
#undef I64

#define S_OPEN_FILE "_ZN2nn2fs8OpenFileEPNS0_10FileHandleEPKci"
#define S_FILE_SIZE "_ZN2nn2fs11GetFileSizeEPlNS0_10FileHandleE"
#define S_CLOSE_FILE "_ZN2nn2fs9CloseFileENS0_10FileHandleE"
#define S_ENTRY_TYPE "_ZN2nn2fs12GetEntryTypeEPNS0_18DirectoryEntryTypeEPKc"
#define S_READ_FILE "_ZN2nn2fs8ReadFileEPmNS0_10FileHandleElPvm"
#define S_READ_OPTIONS "_ZN2nn2fs8ReadFileEPmNS0_10FileHandleElPvmRKNS0_10ReadOptionE"
#define S_READ_EXACT "_ZN2nn2fs8ReadFileENS0_10FileHandleElPvm"
#define S_WRITE_FILE "_ZN2nn2fs9WriteFileENS0_10FileHandleElPKvmRKNS0_11WriteOptionE"
#define S_SET_FILE_SIZE "_ZN2nn2fs11SetFileSizeENS0_10FileHandleEl"
#define S_FLUSH_FILE "_ZN2nn2fs9FlushFileENS0_10FileHandleE"
#define S_CREATE_FILE "_ZN2nn2fs10CreateFileEPKcl"
#define S_DELETE_FILE "_ZN2nn2fs10DeleteFileEPKc"
#define S_RENAME_FILE "_ZN2nn2fs10RenameFileEPKcS2_"
#define S_CREATE_DIRECTORY "_ZN2nn2fs15CreateDirectoryEPKc"
#define S_DELETE_RECURSIVE "_ZN2nn2fs26DeleteDirectoryRecursivelyEPKc"
#define S_OPEN_DIRECTORY "_ZN2nn2fs13OpenDirectoryEPNS0_15DirectoryHandleEPKci"
#define S_READ_DIRECTORY "_ZN2nn2fs13ReadDirectoryEPlPNS0_14DirectoryEntryENS0_15DirectoryHandleEl"
#define S_CLOSE_DIRECTORY "_ZN2nn2fs14CloseDirectoryENS0_15DirectoryHandleE"
#define S_MOUNT_SAVE "_ZN2nn2fs13MountSaveDataEPKcRKNS_7account3UidE"
#define S_MOUNT_SAVE_RO "_ZN2nn2fs21MountSaveDataReadOnlyEPKcNS_13ApplicationIdERKNS_7account3UidE"
#define S_SAVE_EXISTS "_ZN2nn2fs18IsSaveDataExistingENS_13ApplicationIdERKNS_7account3UidE"
#define S_MOUNT_TEMP "_ZN2nn2fs21MountTemporaryStorageEPKc"
#define S_UNMOUNT "_ZN2nn2fs7UnmountEPKc"
#define S_COMMIT "_ZN2nn2fs6CommitEPKc"
#define S_CREATE_CACHE "_ZN2nn2fs18CreateCacheStorageEill"
#define S_DELETE_CACHE "_ZN2nn2fs18DeleteCacheStorageEi"
#define S_MOUNT_CACHE "_ZN2nn2fs17MountCacheStorageEPKci"
#define S_OPEN_CACHE_LIST "_ZN2nn2fs20OpenCacheStorageListEPNS0_22CacheStorageListHandleE"
#define S_READ_CACHE_LIST "_ZN2nn2fs20ReadCacheStorageListEPiPNS0_16CacheStorageInfoENS0_22CacheStorageListHandleEi"
#define S_CLOSE_CACHE_LIST "_ZN2nn2fs21CloseCacheStorageListENS0_22CacheStorageListHandleE"
#define S_SET_PRIORITY "_ZN2nn2fs26SetPriorityOnCurrentThreadENS0_8PriorityE"
// romfs.c
#define S_MOUNT_ROM "_ZN2nn2fs8MountRomEPKcPvm"
#define S_QUERY_CACHE "_ZN2nn2fs22QueryMountRomCacheSizeEPm"
#define S_INITIALIZE "_ZN2nn2fs24InitializeForApplicationEv"

HOST_OVERRIDE(S_OPEN_FILE, ov_open_file);
HOST_OVERRIDE(S_FILE_SIZE, ov_file_size);
HOST_OVERRIDE(S_CLOSE_FILE, ov_close_file);
HOST_OVERRIDE(S_ENTRY_TYPE, ov_entry_type);
HOST_OVERRIDE(S_READ_FILE, ov_read_file);
HOST_OVERRIDE(S_READ_OPTIONS, ov_read_options);
HOST_OVERRIDE(S_READ_EXACT, ov_read_exact);
HOST_OVERRIDE(S_WRITE_FILE, ov_write_file);
HOST_OVERRIDE(S_SET_FILE_SIZE, ov_set_file_size);
HOST_OVERRIDE(S_FLUSH_FILE, ov_flush_file);
HOST_OVERRIDE(S_CREATE_FILE, ov_create_file);
HOST_OVERRIDE(S_DELETE_FILE, ov_delete_file);
HOST_OVERRIDE(S_RENAME_FILE, ov_rename_file);
HOST_OVERRIDE(S_CREATE_DIRECTORY, ov_create_directory);
HOST_OVERRIDE(S_DELETE_RECURSIVE, ov_delete_recursive);
HOST_OVERRIDE(S_OPEN_DIRECTORY, ov_open_directory);
HOST_OVERRIDE(S_READ_DIRECTORY, ov_read_directory);
HOST_OVERRIDE(S_CLOSE_DIRECTORY, ov_close_directory);
HOST_OVERRIDE(S_MOUNT_SAVE, ov_mount_save);
HOST_OVERRIDE(S_MOUNT_SAVE_RO, ov_mount_save_ro);
HOST_OVERRIDE(S_SAVE_EXISTS, ov_save_exists);
HOST_OVERRIDE("_ZN2nn2fs14EnsureSaveDataERKNS_7account3UidE", ov_ensure_save);
HOST_OVERRIDE(S_MOUNT_TEMP, ov_mount_temp);
HOST_OVERRIDE(S_UNMOUNT, ov_unmount);
HOST_OVERRIDE(S_COMMIT, ov_commit);
HOST_OVERRIDE(S_CREATE_CACHE, ov_create_cache);
HOST_OVERRIDE(S_DELETE_CACHE, ov_delete_cache);
HOST_OVERRIDE(S_MOUNT_CACHE, ov_mount_cache);
HOST_OVERRIDE(S_OPEN_CACHE_LIST, ov_open_cache_list);
HOST_OVERRIDE(S_READ_CACHE_LIST, ov_read_cache_list);
HOST_OVERRIDE(S_CLOSE_CACHE_LIST, ov_close_cache_list);
HOST_OVERRIDE(S_SET_PRIORITY, ov_set_priority);
