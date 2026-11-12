// Adapted from MacNativeGames games/totk/runtime/audio_reverb.c @ b75be8c (MIT, Copyright (c) 2026 MacNativeGames
// contributors), based on Ryujinx.Audio (MIT, Copyright (c) Ryujinx Team and Contributors).
#ifndef OOT_AUDIO_REVERB_H
#define OOT_AUDIO_REVERB_H
#include <stdint.h>
typedef struct {
    uint8_t input[6], output[6]; uint16_t channel_max, channels;
    uint32_t reserved, sample_rate;
    float room_hf, hf_reference, decay_time, hf_decay_ratio, room_gain, reflections_gain,
          reverb_gain, diffusion, reflection_delay, reverb_delay, density, dry_gain;
    uint8_t status;
} Reverb3dParam;
_Static_assert(__builtin_offsetof(Reverb3dParam, status) == 72, "reverb3d wire layout");
typedef struct Reverb3d Reverb3d;
Reverb3d *audio_reverb_create(const Reverb3dParam *p);
void audio_reverb_free(Reverb3d *s);
void audio_reverb_update(Reverb3d *s, const Reverb3dParam *p);
void audio_reverb_process(Reverb3d *s, float *const input[6], float *const output[6], int frames);
#endif
