// A copy of the RomFS files the game has read, on a faster disk than the image's (romfs_cache.c).
#pragma once
#include <stdint.h>

#include "platform/platform.h"

void romfs_cache_init(PlatFile *image, uint64_t file_count);   // once, after the image opened
PlatFile *romfs_cache_open(uint64_t i, uint64_t size);         // file i's copy (size bytes), or NULL
void romfs_cache_note(uint64_t i, uint64_t off, uint64_t size);   // the game read file i (image range): copy it
