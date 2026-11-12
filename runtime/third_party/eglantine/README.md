# Eglantine Ampere lowering (GPL-3.0-or-later)

`shader_recompiler/frontend/ampere/` (`lower_to_maxwell.h`, `program_capture.h`, `instruction.h`, `geometry_exit.h`,
`register_pressure.h`) is copied unchanged from [zortos293/Lichii](https://github.com/zortos293/Lichii) (branch `nx2`), a
descendant of Eglantine (an Eden-derived Switch 2 emulator) and of [Franseven/PokeDinner-a-Eglantine-Fork](https://git.eden-emu.dev/Franseven/PokeDinner-a-Eglantine-Fork),
at the commit in `UPSTREAM_COMMIT`. The six "DKB Recomp" fixes earlier copies carried (LDC small sizes, S2R THREAD_KILL,
F2F and I2F source forms, BRA.U) are not re-applied: Donkey Kong Bananza has not been rebuilt against this version. Copyright 2026 Eden Emulator
Project, GPL-3.0-or-later; their encoding reference is Mesa NAK's sm70_encode.rs (MIT, Copyright 2022 Collabora,
Ltd.).

They rewrite Switch 2 (Ampere, SM86) shader code as Maxwell instructions, which the shader translator this project
already uses can take. `common/common_types.h` is ours: the fixed-width type names the headers expect.

Because these files are GPL-3.0-or-later, DKB Recomp as a whole is distributed under GPL-3.0-or-later (see the
repository's LICENSE).
