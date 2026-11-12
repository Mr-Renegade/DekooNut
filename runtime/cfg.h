// Settings: the OOT_* environment variables, with oot.ini next to oot.exe behind them.
#pragma once

// getenv for a setting: the environment variable when set, else the oot.ini key (the name without "OOT_", any case),
// else the device profile's default (cfg.c), else NULL. A name without the OOT_ prefix goes to getenv unchanged.
// The pointer stays valid for the process. Safe from any thread once cfg_init has returned.
const char *cfg_get(const char *name);

// Reads oot.ini once (from mem_init, before any setting is read); logs the file, unknown keys and environment overrides.
void cfg_init(void);

// Replaces the loaded keys with those of the file at path (tests). 1 when the file was read, 0 when missing.
// Not thread-safe: nothing else may call cfg_get meanwhile.
int cfg_load(const char *path);

// Live settings (the in-game menu). cfg_set stores value as the oot.ini value of name (OOT_ prefixed), marks it for
// cfg_save and bumps cfg_version. cfg_get returns it unless the environment sets name. Any thread.
void cfg_set(const char *name, const char *value);
// Bumped by every cfg_set: a consumer that keeps a parsed value compares it and parses again on change.
unsigned cfg_version(void);
// 1 when the environment sets name: cfg_set cannot change what cfg_get returns (the menu greys the setting out).
int cfg_from_env(const char *name);
// 1 when the player set name (the environment or oot.ini), 0 when cfg_get gives a device profile's default or NULL.
int cfg_is_set(const char *name);
// The device profile in use, "deck" or "desktop" (OOT_DEVICE_PROFILE, else what the machine is).
const char *cfg_device_profile(void);
// Writes the keys cfg_set changed since the last save into oot.ini (cfg_save_to: the file at path). Only those lines
// change; comments and every other line stay byte for byte. 1 on success or with nothing to write.
int cfg_save(void);
int cfg_save_to(const char *path);

// Reset (the menu's "Reset all settings to defaults"): removes the active line of each key (names without OOT_) at the
// next cfg_save and its value from memory; keys the environment sets are skipped. Bumps cfg_version; returns the count.
int cfg_reset(const char *const *keys, int n);
// 0 when the last cfg_save / cfg_save_to failed to write (the values stay set in memory, the next save tries again).
int cfg_last_save_ok(void);
// Every setting with a value, for a bug report ("Copy system info"): the oot.ini and cfg_set keys (name without
// OOT_, upper case; from_env 0), then the known OOT_ variables the environment sets (from_env 1).
void cfg_each_active(void (*cb)(const char *key, const char *value, int from_env, void *ctx), void *ctx);

// What the menu knows about a setting. key: without OOT_, upper case. def: what the consumer uses when the key is
// unset ("" = no fixed default, the consumer picks per machine). values: "0|1", "mailbox|fifo|immediate", a range
// "0..300", or NULL for free text. live: applies at once; the others are read at start.
typedef struct { const char *key, *def, *values; int live; } CfgMeta;
// The row of key (prefix and case ignored); NULL for a key the menu does not show.
const CfgMeta *cfg_meta(const char *key);
// The i-th row (0-based, table order); NULL past the last.
const CfgMeta *cfg_meta_at(int i);

// The setting's value for consumers: the environment or oot.ini value, else the CfgMeta default. name has the OOT_
// prefix; a name without a row is a bug (rt_fatal). A row with an empty default has no fixed value: its consumer reads
// cfg_get and keeps its own "unset" branch. An empty value counts as unset.
const char *cfg_value(const char *name);
// cfg_value parsed. A set value that does not parse (int: whole string; double: strtod) gives the default's number.
// cfg_bool: "0" is false, anything else true.
int cfg_int(const char *name);
double cfg_double(const char *name);
int cfg_bool(const char *name);
