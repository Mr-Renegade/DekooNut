# Quality levels

A level is a set of `oot.ini` keys (and, for Default, a game-data pack). Setup writes them; `setup/quality.py` is the
table, heaviest first, and `suggest()` picks the heaviest level a PC's video memory and memory meet.

| level | scene | game frame | upscale | grass | `oot.ini` |
|---|---|---|---|---|---|
| Default | 1630x917 | 2560x1440 | DLSS, preset M | on | `romfs_mods = gamedata/hs625`, `gfx_scale = 0` |
| Balanced | 1280x720 | 1706x960 | DLSS, preset K, then FSR to the window | off | `gfx_scale = 2`, `dlss = fast`, `grass = off` |
| Light | 1280x720 | 1706x960 | a plain filter, then FSR to the window | off | `gfx_scale = 2`, `dlss = off`, `grass = off` |

## What each key costs

Measured on an RTX 3070 (8 GB, driver 580) with a Ryzen 5 9600X and 30 GB of memory, a 2560x1440 window, the Hyrule
Field test save, standing still; the frame is bound by the graphics card in every row. GPU time is per frame.

| change | GPU time | fps |
|---|---|---|
| Default | 22.0 ms | 41 |
| `dlss = fast` | 20.6 ms | 44 |
| `dlss = off` | 18.8 ms | 47 |
| `grass = off` | 19.3 ms | 46 |
| scene 1280x720 in a 1706x960 frame (`gfx_scale = 2`, no pack) | 20.5 ms | 44 |
| Balanced (the three together, DLSS fast) | 16.1 ms | 55 |
| Light (the three together, DLSS off) | 14.4 ms | 60 |

- DLSS at 2560x1440 costs 4.5 ms with preset M and 2.1 ms with preset K on this card (1.1 ms for a 1706x960 target).
- The scene size is a weak lever: 1086x611 18.9 ms, 1280x720 20.5, 1630x917 22.0, 1920x1080 23.8. The game's handheld
  mode (960x540) still needs 18.0 ms.
- Most of the frame is the foliage, drawn by mesh shaders whose cost follows the number of plants, not the pixels: with
  none of it drawn the same view takes 12.0 ms. The grass (eight mesh stages) is 2.7 ms of that and is what
  `grass = off` leaves out; trees, flowers and undergrowth stay. The largest single draw left is the distant forest,
  2.7 ms.
- Running through the field and in Kakariko the frame is lighter: Balanced holds 60 fps in both; Default 44 to 47.
- Skipping the distant forest's colour draw as well gives 13.5 ms and 60 fps with DLSS fast, but its depth pre-pass
  then leaves the trees' outlines in the sky colour: not offered until both passes can be left out together.
- Memory: the game holds 11.4 to 12.0 GiB resident at every level (7.3 GiB of it the game's own memory). Balanced
  under a 12 GB limit on the game's processes, about what a 16 GB PC leaves it, ran the same test at the same frame
  rates with 0.46 GiB swapped out and no frame over 50 ms after the load. That is a limit on a 30 GB machine, not a
  run on a 16 GB PC.
- Video memory: the game keeps to a budget of about 6 GB on an 8 GB card and reaches it at every level; it then drops
  released textures (1,610 in a minute of running at Default, 808 at Balanced) without a visible effect in these runs.
- The first start on a PC compiles every pipeline: about 2.5 minutes before the title on this machine, and two freezes
  of 12 and 9 seconds at the first world load. Later starts take 7 seconds.

On an RTX 4080 Super the Default level runs at 60 fps with 11.9 ms of GPU time at 3840x2160.
