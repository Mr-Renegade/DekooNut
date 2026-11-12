// From MacNativeGames games/totk/runtime/romfs_path.h @ b75be8cf (MIT, Copyright (c) 2026 MacNativeGames
// contributors).
// RomFS path validation (romfs.c).
#pragma once
#include <stddef.h>
#include <string.h>

// A RomFS-relative path: non-empty, no leading '/', no '\', no empty, "." or ".." component.
static inline int romfs_path_valid_relative(const char *path) {
    if (!path || !*path || *path == '/' || strchr(path, '\\')) return 0;
    for (const char *part = path; *part;) {
        const char *end = strchr(part, '/');
        size_t size = end ? (size_t)(end - part) : strlen(part);
        if (!size || (size == 1 && part[0] == '.') || (size == 2 && part[0] == '.' && part[1] == '.')) return 0;
        if (!end) break;
        part = end + 1;
    }
    return 1;
}

// "<mount>:/<path>" -> "<path>", or NULL. Repeated separators collapse (audio stream requests join a mount root and a
// leading-slash resource); a collapsed result lives in a per-thread buffer until this thread's next call. Traversal and
// backslashes are rejected.
static inline const char *romfs_path_relative(const char *mount, const char *path) {
    if (!mount || !*mount || !path) return NULL;
    size_t size = strlen(mount);
    if (strncmp(path, mount, size) || path[size] != ':' || path[size + 1] != '/') return NULL;
    path += size + 2;
    while (*path == '/') ++path;
    if (strstr(path, "//")) {
        static _Thread_local char collapsed[4096];
        size_t n = 0;
        for (const char *q = path; *q; q++) {
            if (*q == '/' && (q[1] == '/' || !q[1])) continue;
            if (n + 1 >= sizeof collapsed) return NULL;
            collapsed[n++] = *q;
        }
        collapsed[n] = 0;
        path = collapsed;
    }
    return romfs_path_valid_relative(path) ? path : NULL;
}
