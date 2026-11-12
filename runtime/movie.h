// Adapted from MacNativeGames games/totk/runtime/movie_host.h @ b75be8cf (MIT, Copyright (c) 2026 MacNativeGames
// contributors).
// What ov_movie.c (the host movie::Decoder) uses. The game reaches the decoder through the
// HOST_OVERRIDE entries named in movie_symbols.
#pragma once
#include <stdint.h>

#include "host.h"

// Entries in the order of movie_symbols[]: the imported constructor, then the 19 vtable slots of movie::Decoder (each an ABS64 relocation by name).
enum { MOVIE_CTOR, MOVIE_D2, MOVIE_D0, MOVIE_CONFIGURE, MOVIE_REGISTER_EVENTS, MOVIE_GET_OUTPUT_FORMAT, MOVIE_START,
       MOVIE_STOP, MOVIE_RESET, MOVIE_FLUSH, MOVIE_ACQUIRE_INPUT, MOVIE_GET_INPUT, MOVIE_SEND_INPUT, MOVIE_ACQUIRE_OUTPUT,
       MOVIE_GET_OUTPUT, MOVIE_GET_OUTPUT_SIZE, MOVIE_RELEASE_OUTPUT, MOVIE_GET_LAST_ERROR, MOVIE_GET_OUTPUT_MD,
       MOVIE_GET_OUTPUT_BUFFER_FORMAT, MOVIE_ENTRIES };
extern const char *const movie_symbols[MOVIE_ENTRIES];

// Guest addresses of the functions the decoder calls with rt_call, the vtable address the constructor stores (vtable + 0x10)
// and the original code of every entry; filled from the loaded modules at the first entry call.
typedef struct {
    uint64_t signal_event, buf_set_range, buf_set_data, buf_base, buf_capacity;
    uint64_t md_find_int32, md_find_string, md_set_int32, md_set_rect, md_set_string;
    uint64_t vptr;
    uint64_t orig[MOVIE_ENTRIES];
} MovieGuest;

enum { MOVIE_EV_INPUT, MOVIE_EV_OUTPUT, MOVIE_EV_FORMAT, MOVIE_EV_ERROR };   // DecoderEvents: EventType * per event
enum { MOVIE_TYPE_VP9 = 3, MOVIE_MODE_BUFFER = 1, MOVIE_FORMAT_NV12 = 4, MOVIE_FLAG_EOS = 4 };

typedef struct { uint64_t frames, decode_ns, max_ns, errors, dropped, handed; } MovieStats;

