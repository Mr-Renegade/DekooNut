# How DekooNut works

Notes on the runtime, by topic. Comments in the code point here by section name.

## Overview

This is a native PC port of the Switch 2 game The Legend of Zelda: Ocarina of Time. The game's ARM code is recompiled to C
ahead of time. The result is linked with a runtime (`runtime/`) that provides what the console provided.

The runtime reserves the guest address space and runs each guest thread as a host thread. Horizon's kernel (threads,
waits, fibers, core ownership) is implemented by the host. Indirect branches in the recompiled code go through one flat
dispatch table.

System services are host code. The runtime serves the RomFS from the player's extracted image with optional overlays. It
keeps saves in host directories with Horizon's journaled commit. It implements the audio services with its own mixer, the
input services (pads, motion, keyboard, mouse) and a movie decoder.

The graphics API (NVN) is recorded into host buffers by the runtime. At submit the recording is replayed on Vulkan. The
replay mirrors guest memory to the GPU, tracks which pages the CPU wrote, and writes GPU results back where the game reads
them. Game shaders are translated to SPIR-V by an external translator, cached on disk, and prepared ahead of play.
Pipelines are built in the background and prewarmed at boot.

The frame is presented through a present thread with fixed 60 Hz pacing. The game's own upscaler, FSR, or DLSS scales it to
the window. Settings come from environment variables, `oot.ini` and device profiles. A few test hooks (a state file, a
flight recorder, GPU breadcrumbs) help to find faults.

## Guest code, memory and threads

### Guest memory

`runtime/memory.c`. Every region of the guest address space is a plain reservation. Pages are committed and decommitted at
4 KiB granularity, and decommitted pages come back zeroed, so no stale-page bookkeeping is needed. The stack and heap
block and the alias region are reserved write-watched (`plat_mem_reserve`). `mem_written_pages` reports the pages the CPU
wrote, which the Vulkan back end uploads again. The process is re-run when a region cannot be taken, because Windows
places DLLs, heaps and stacks at random addresses.

Frozen regions. A 2 MiB region of the watched reservations that nothing wrote for a while is made read-only
(`mem_freeze`), so the back end's write-watch queries skip it: no page of a frozen region changes before a thaw. The first
write faults and `thaw_handler` makes the region writable again before the write retries. A fault in a region another
thread thawed meanwhile retries too. A page trapped no-access (`plat_mem_trap`, the write-back read trap) goes on to the
access hook after its region is thawed. A kernel write into guest memory (a file read) fails on a read-only page instead
of faulting, so file reads thaw their buffer first (`mem_thaw`), and so does every commit, decommit and protection
change. A region stays thawed for `FRZ_HOLD_MS` after a `mem_thaw`, because a file read still in flight writes its buffer
later. `frz_lock` is a leaf lock: nothing writes guest memory under it.

### Guest threads and virtual cores

`runtime/kernel.c` is the Horizon kernel's threads, synchronization and fibers. One guest thread is one host thread with
a 256 MiB reserved host stack (every guest BL is a host call) and its own `Cpu`. The SDK's nn::os runs as lifted code;
only the kernel under it is the host. Every wait hangs on one global lock `G` with one condition variable per thread. The
wake order is the kernel's: lower priority value first, then first come. Semantics follow the public Horizon kernel
(KConditionVariable, KAddressArbiter, KSynchronization).

Horizon runs at most one thread per core, and the game keys per-core data (allocators, job queues) by the current core
number without locks. A guest thread therefore holds one of the application cores (`NCORE`, 0-5) while it runs guest code.
It gives the core up only where Horizon would switch threads: waits, sleep, yield, exit, blocking host calls. A free core
goes to the best waiting thread. A waiter better than a holder asks one holder to give its core up (`core_kick`). The
holder does so at its next SVC or indirect call (`kern_checkpoint`, `rt_dispatch`), as Horizon runs the better thread at
once. Guest code with neither keeps the core until it reaches one. Priority inheritance (`eff_prio`) makes a thread that
owns a guest mutex count at the priority of the best thread blocked on it, for granting, kicking, preempting and yielding.

### Asynchronous preemption

Horizon gives a core to a strictly better thread the moment it is ready. Here a better waiter only marks the owner, which
yields at its next SVC or indirect call. An owner in a long stretch of game code with neither would keep the core for
milliseconds. So the waiter interrupts the owner (`plat_intr_stop`) once it has waited `AP_FIRST_NS`, then every
millisecond. If the owner's host rip is inside the lifted modules' code (`g_gen_lo..g_gen_hi`) and its host depth is 0
(not in a host call or SVC, nor in game code a host call made), it holds no host lock, so its rip is moved to
`kern_ap_stub`. The stub saves every volatile register, yields like a checkpoint (`kern_ap_yield`) and jumps back.
Otherwise the owner is only marked. The waiter never writes the owner's stack.

Horizon's preemption disable is honoured. The SDK's spin locks count `TLR+0x100` (disable_count) while they try or hold
the lock. They call SVC 0x36 after clearing it when `TLR+0x102` (interrupt_flag) is set. Such an owner is not preempted,
at a checkpoint either; the interrupt flag is set and the switch happens at that SVC. Preempting it anyway hangs the game
at "Continue" (the lock holder loses its core to a thread that then spins on that lock). Priority inheritance matters for
the same hang: without it a better thread takes the core of the low-priority owner of a mutex it then waits for.

### Spin waits

`runtime/kernel.c` (`rt_spin_gate`, the `spin` lines of `recomp/patches.py`). A pass of a guest loop that polls memory
until another thread changes it fills only its own core on the Switch, where a better thread pinned there preempts it.
Here it holds a virtual core and a host core. Pollers that take a lock on every pass hand its cache line between host
cores, so the thread whose item it is rarely gets it. Every pass pauses the host core once. A wait is a run of passes
through the same loop at most `SPIN_GAP_NS` apart. Once one has lasted 50 us and a thread waits for a virtual core, each
pass gives the core to the waiter whatever its priority, yields the host CPU and takes a core again. Shorter waits keep the
core, since a lock held for a moment would otherwise make the next pass wait for another thread's next SVC.

### Fibers

The SDK switches fibers by swapping register context, which breaks "RET = C return". Each guest fiber gets a host fiber
(own 256 MiB reserved host stack) and its own `Cpu`, so a suspended fiber's registers and host frames stay untouched while
others run, and a fiber may resume on another thread. A switch carries the thread state (TLR, TLS pointer, FPCR/FPSR) over
to the target's `Cpu` and clears the exclusive monitor. The fiber overrides (`InitializeFiber`, `SwitchToFiber`,
`GetCurrentFiber`) are registered as `HOST_OVERRIDE` entries in `kernel.c`.

### Indirect-branch dispatch

`runtime/dispatch.h`, `runtime/dispatch.c`, `runtime/hostfn.c`. Indirect branches in lifted code look the target up in one
flat table that covers every module's code range. The table has one `uint32_t` slot per 4-byte guest instruction over
`[lo, hi)`, where `lo` is the lowest function start of any module and `hi` is past the highest. A slot is 0 when no
function starts there. Otherwise it holds the host function's address minus `anchor` (an address just below the lowest
host function). Four bytes are enough because a single PE image keeps all lifted code within 4 GiB. A lookup is one range
check and one load (slot `(pc - lo) >> 2`).

Memory is 4 bytes per 4 guest bytes of range, but the gaps between code ranges (rodata, data, bss, alignment) are reserved
and never written, so the OS never backs them. Resident memory is about the total `.text` size. `rt_dispatch_init` fills
the table once and makes it read-only, so any number of threads read it without locks or per-thread caches.

Host functions (HOST_OVERRIDE / HOST_OBSERVER) live at synthetic guest addresses outside every module. `hostfn_seal`
registers them and builds the table, and guest threads start only after that. The host (HLE) functions at addresses outside
the modules live in a small hash table fixed at init and are looked up only after the flat table misses.

### Thread interruption and write watch

`runtime/platform/linux.c`. Thread interruption replaces Windows' SuspendThread and Get/SetThreadContext with a signal
handshake. The caller moves the handle from IDLE to REQ and signals the thread. The thread's handler stores where it was
and moves REQ to STOPPED, then spins until the caller moves it on to GO (with the redirect in `to`). The handler then
changes its own return context and goes back to IDLE. A caller that sees no STOPPED within `INTR_WAIT_NS` (the thread is
blocked in the kernel and runs no handler) takes REQ back to IDLE, and a handler that then finds IDLE returns at once.

Write watch uses a userfaultfd in asynchronous write-protect mode (Linux 6.7) over the watched reservations. The kernel
clears a page's protection on the first write without a round trip to the runtime, and `PAGEMAP_SCAN` lists the pages
whose protection was cleared and protects them again. Without it (an older kernel) `plat_mem_written` reports the ranges
as untracked and the back end uploads them in full. Reservations are `PROT_NONE MAP_NORESERVE` mappings. Commit is
mprotect read-write and decommit is `MADV_DONTNEED` plus `PROT_NONE`. Since Linux keeps no committed/reserved
distinction, a byte per guest page (`pstate`) records it.

The locks are a reader-writer lock in one 32-bit word (bit 31 writer, bit 30 sleepers, bits 0-29 readers) over a futex. A
release that finds the sleeper bit clears it and wakes all sleepers, and readers do not enter while it is set.

### Host event descriptors

`runtime/ov_eventfd.c`. The game embeds gRPC. Its event engines wake their poller through an eventfd
(`nn::socket::EventFd`) and wait with `nn::socket::Poll`. If every nn::socket call failed, no engine would initialize and
the game would stop at start. So these calls work on descriptors the host owns, with console semantics:
- `EventFd(initval, flags)`: a descriptor with a 64-bit counter; flag 0x1 = semaphore, 0x800 = non-blocking.
- `Read(fd, buf, n)`, n >= 8: returns the counter and resets it to 0 (semaphore: returns 1, counter - 1). It waits for a
  nonzero counter unless the descriptor is non-blocking.
- `Write(fd, buf, n)`, n >= 8: adds the value to the counter.
- `Poll(fds, n, timeout_ms)`: POLLIN while the counter is nonzero, POLLOUT always, POLLNVAL for any other descriptor.
- `Close`, `Fcntl`, `Ioctl` succeed on the host's descriptors.

Everything else in nn::socket fails, as there are no sockets.

## System services

### RomFS overlays

`runtime/romfs.c` serves a read-only game RomFS from the player's extracted image. The inputs are `romfs.bin`
(`rt_config()->romfs`, whose six IVFC levels and SHA-256 `tools/extract.py` verified) and `romfs-index.bin` next to
`oot.exe`, which `tools/romfs_index.py` wrote together with `romfs_config.h`. `fs_init` checks the index's SHA-256 and
layout and the image size. Reads are positional (`plat_file_pread`), so threads read the image in parallel without a
file-pointer lock.

Files listed in a directory's `manifest.tsv` are served in place of the originals. Each line names a RomFS path, the
overlay's size and SHA-256 and the original's SHA-256. A `-` adds a file the RomFS lacks; added files are numbered after
the RomFS's own (`file_count + k` in path order). The directories read, in this order, are `romfs-patches` next to the
index and the one named by `OOT_ROMFS_MODS` (unset or `0` means none). A missing path, a size or hash mismatch or a
changed original stops the start. An overlay file is opened when the game opens it. Its identity (volume, file id, size,
last write) is kept from the start check, and an open that finds another file there stops the game.

### Save storage

`runtime/fs_host.c` implements nn::fs. RomFS paths go to the read-only backend (`romfs.c`). Save, temporary and cache
storage are host directories under the profile directory (default `build/profile`):
- `save/<uid hi><uid lo>/` is the committed application save, created on first use.
- `save-tx/` holds the save mount's working copy and Commit's staging.
- `save-backup/<time>/` holds the three saves before the current one.
- `temp/` is emptied at start; `cache/<index>/` is cache storage.

The save mount is journaled as on Horizon. Changes go to a working copy (`save-tx/work`, made from the newest save when the
first save mount appears, deleted at the last Unmount and at start). They become the committed save atomically at Commit.
A crash or an Unmount before Commit leaves the previous save. Commit returns 0 once its steps are queued. A background
thread runs them while changes to the working copy wait, and Unmount, the next Commit and `fs_save_flush` (exit paths) wait
too. A failure is logged, not returned.

Commit steps:
1. Copy `work` to `new`, flushing every file and directory. A file equal to the newest save's becomes a hard link to it.
2. Rename `new` to `ready-<n>` (the commit point) and delete the `ready-<n>` it supersedes.
3. Rename `save/<uid>` to `old`.
4. Rename `ready-<n>` to `save/<uid>`.
5. Move `old` to `save-backup/<UTC time>` and keep the newest three.

Every rename is durable (`plat_fs_rename`). Windows refuses step 3 while any file under `save/<uid>` is open in any process
(a virus scanner, Explorer's preview, a sync tool), or after an elevated run made the folder. Then `ready-<n>` stays the
newest save, later working copies are made from it, and the next Commit and the next start try steps 3 to 5 again. If a
commit wrote equal bytes only, no save and no backup is made.

Recovery at start (`save_recover`): an unfinished `new` is deleted (rollback), unless `save/<uid>` and `ready-<n>` are both
absent, when `new` is a finished copy and becomes `save/<uid>`. The highest `ready-<n>` is the newest save and steps 3 to 5
run (roll forward); other `ready-<n>` are deleted. If `save/<uid>` is absent and no `ready-<n>` exists, `old` goes back.
Then `old` moves to `save-backup` and `work` (changes no Commit took) is deleted.

### Audio renderer service

`runtime/audren.c`, mixer in `audio_mixer.c`. The SDK's `nn::audio` client builds `RequestUpdate` input unchanged. The
service validates each section against its header and keeps a server-side state for every voice. It advances the voices one
render frame (`sample_count / sample_rate` seconds) per tick and signals the renderer's event. A tick runs each time the
output device takes the ring below its target (`audio_mixer_pace_ns`), or on the host clock when there is no device or the
device stalls. Voice lifetime follows the console: Stop turns a started voice into Stopping, and the next render frame
returns every queued wave buffer before it becomes Stopped. A buffer with no samples counts as played when the voice
reaches it, so it never holds up the buffers behind it. On Switch 2 the SDK reaches the managers through `aud:u`; only its
renderer manager is served.

### Audio mixer

`runtime/audio_mixer.c`. Per 5 ms frame the render thread (`audren.c`, renderer lock held) decodes every started voice
(PCM8/16/24/32, float, DSP-ADPCM) along the sample path the voice clock consumes and resamples it linearly. It runs the
voice biquads, scales the voice into its destination mix, folds the submixes, runs the mix effects and writes the device
sink's stereo mix into a single-producer single-consumer ring. The SDL audio thread drains the ring without a lock. An
empty ring plays silence and never blocks the render thread or the guest. With the `audio` setting `none` the mixer mixes
without a device. Output starts after 20 ms of queued audio and again after an underrun.

The device's clock paces the render thread (`audio_mixer_pace_ns`). A frame is rendered when the ring holds less than the
output target, `max(prefill, two device periods)`, so the queue and the delay to the speakers stay put and the host and
device clocks cannot drift apart. Without a device, while the renderer is stopped, and while the device takes nothing (a
stall), the render thread keeps real time on its own timer. What piled up meanwhile is skipped on the SDL thread once the
device resumes (more than `QUEUE_SLACK_MS` above the target plus the device's request).

### Audio device manager

`runtime/audio_nx2.c` implements the Switch 2 service "auddmg:u" (the nn::audio virtual output devices the SDK queries
before it opens an output stream). The host mixes everything into one stereo 48 kHz output, so there is a single virtual
device. Command 0 of the service returns the manager object. Its commands are: 0 (count, names in an out buffer of 0x100
bytes each), 100 (default device name), 300 (stream format check, result only), 401 (the 0x228-byte property block), 500
(copy handle of a state-change event, never signalled), 700 (attach a stream to the device) and 800 (detach). Attach starts
the sysmix stream (`audio_sysmix.c`) on the device; detach stops it.

### Sysmix audio service

`runtime/audio_sysmix.c`. The Switch 2 mixer service `audsmx:u` carries the PCM output streams of
`nn::audio::OutputStream`. `audsmx:u` command 0 (copy handle: the process) opens the user session. The session has command
1 (u32 device, u32 flags, u64 applet resource user id, u64 buffer, u64 size; name buffer) which returns a u32 id and a
stream, and commands 2 / 3 (buffer of u32 ids) which engage / disengage streams. A stream has these commands:
- 0 open (u32 sample type, channels, rate, frames, prefill, flags), 1 close.
- 2 state (0 closed, 1 open, 2 engaged).
- 3 gain (f32), 4 gain with a ramp (f32 gain, u32 pad, s64 ramp ns).
- 5 / 6 acquire / release the state event, 7 / 8 acquire / release the data event (u32 mode, s32 threshold).
- 9-11 clock, prime, purge (not supported), 12 rename.

The stream's buffer is shared with the game: a u32 read index at +0x44 (advanced by the service), a u32 write index at
+0x1004 (advanced by the game) and, from +0x1040, a ring of `frames + 1` stereo frames of 16-bit or float PCM at 48 kHz.
Every 5 ms the mixing thread takes up to 240 frames of each engaged stream, applies its gain, passes them to an SDL
playback stream and moves the read index. The data event is signalled (mode 2) while at least `threshold` ring slots are
free; that is how the game learns it may write more. With the `audio` setting `none` the frames are consumed without a
device, so the game never stalls.

### Input blocking

`runtime/input.c`. While text entry (`swkbd_*.c`) or the in-game menu is open the game reads no input, as behind the
keyboard applet. After either closes the game still reads none until every button and stick direction is released, so the
press that closed it stays there (`g_block` and `g_menu`: 0 off, 1 open, 2 closed and waiting for the release).

### Cursor capture

`runtime/input_sdl.c` (SDL) and `input_win.c` (Win32) follow the same rules. The cursor is captured (SDL relative mouse
mode) while the window has the focus, the mouse camera is on and no text entry or menu is open. Esc releases it until a
click in the window, and closing the menu that Esc opened gives the capture back. Losing the focus releases it until the
window has the focus again. `input.c` asks through `input_cursor_allowed`; the back end owns the capture itself.

### Six-axis fusion

`runtime/motion_fusion.c` turns gyro and accelerometer samples into `nn::hid::SixAxisSensorState`. Per sample the output is
acceleration in G, angular velocity in revolutions per second, angle (the integrated revolutions per axis) and direction
(the controller's three axes in a reference frame whose z is up). The orientation is integrated from the gyro. While the
controller is not being accelerated (|a| within 15 % of 1 G) it is pulled toward the measured gravity (Mahony filter, gain
0.5/s), as the Switch's fusion does. Yaw has no reference and drifts with the gyro's bias. The newest `MOTION_RING` samples
are kept and reads are newest first. A restart (StartSixAxisSensor) drops the samples before it: reads return nothing until
the next sample, which starts a fresh frame and angle, while sampling numbers go on.

### Movie decoder

The game's movies are VP9. The notes describe two replacement layers. They are given here as written.

`runtime/ov_codec.c` replaces the SDK's `nn::codec::video` VP9 path, which opens the console's video service over IPC and
so cannot run on the host. The decoder calls are host functions (libvpx for VP8 and VP9, dav1d for AV1, all handing over
8-bit NV12). The demuxer (`nn::movie::MediaExtractor`) stays guest code. What the game's movie player expects:
- Each output buffer i is an image pool of its own, tied to i by `RegisterVp9DecoderBuffer`. The player reads the pool's
  size at +0x20 / +0x22 (u16 width, height), which `InitializeDecoderImagePool` takes from the preference (+8 u16 width -
  1, +10 u16 height - 1).
- It acquires an output buffer before it submits input, and aborts on any failure there. Acquire reserves the buffer the
  next shown picture goes to and returns a fence (a `NativeSyncHandle` whose first word the host fills with a token:
  `FENCE_TAG | decoder << 8 | buffer`). The player waits on it through `RequestSyncPointEvent`, which is signalled when
  `SubmitVp9DecoderInputBuffer` has decoded a picture into that buffer (at once if it already has).
- A submission with no reserved buffer answers "not ready" (retry code 0x76 of the wrapper). A few packets (QUEUED) are
  accepted ahead of their buffers and decoded when a buffer is reserved. After a movie ends the player flushes and restarts
  with the buffers it still holds, submitting its third packet before releasing one.
- The picture is read with `ReadDecoderOutputImageNv12` (Y plane, then interleaved UV, each with the caller's pitch).
  `GetDecoderImagePoolNativeImageHandle` (the GPU path) returns 0.
- `movie::VideoDecoder::Stop` flushes, then acquires and releases buffers for as long as the oldest queued frame's fence is
  reached. Flush puts held pictures back to "waiting" so this ends; if every fence were reached it would not.

Sync point events are guest system events. They are signalled at once because the host has no sync point to wait for,
except for decoder fences, which fire when their buffer holds a picture.

`runtime/ov_movie.c`. The SDK's movie library decodes through NVDEC, which the host does not provide, so `movie::Decoder`
is replaced by libvpx while the game's demuxer (`movie::Extractor`) still runs as guest code. HOST_OVERRIDE entries replace
the decoder's vtable slots. A VP9 buffer-mode decoder (type 3, mode 1) is a host object (+0 the library's vtable, +8 the
host `Movie`). Any other decoder runs the original code. Behaviour follows the library's `DecoderImpl`:
- Output format 4 is packed NV12 copied into the client's buffer.
- Input and output buffers are indices handed out first-in first-out; flag 4 marks end of stream.
- The four client events (input available, output available, format changed, error) are signalled with
  `nn::os::SignalEvent` on a guest thread.

Decoding runs on one host worker thread per decoder, never on the game's threads. `SendInputBufferForDecode` queues and
returns, and the events the worker owes are signalled at the next entry call or present. Pacing: the game shows one decoded
frame per game frame, so frames are handed over when their presentation time is due (counted from the first frame, in
`movie_tick` at each present, at most one per present). If the game falls behind, a frame more than 1.5 periods late is
dropped when the next one is due too, so the picture keeps time with the sound.

Both replacements are in the runtime: `ov_codec.c` serves `nn::codec::video` (libvpx and dav1d, demuxer
`nn::movie::MediaExtractor`), `ov_movie.c` serves `movie::Decoder` (libvpx, demuxer `movie::Extractor`).

## Graphics front end (NVN recording)

### NVN command recording

`runtime/nvn_api_cmd.c`, `runtime/nvn_api.h`. Records are written into a growable host buffer per command buffer, never into
the game's command memory. Games hand the same command memory to several command buffers recording at once (budgeted for
real NVN encodings), and the runtime's records are larger. The game's command and control memory is only accounted.
`GetCommandMemoryUsed` reports what the real push-buffer encoding would take (`hw_fix`), because the game sums it to size
resource heaps.

`EndRecording` copies the finished recording (a flat `OP_BEGIN ... OP_END` sequence of 8-byte-aligned `{op_hdr, payload}`
records) into host memory. The handle is that copy, and it lives until the game adds the same control memory again (the
SDK encodes handles in control memory). Copies are indexed by the 64 KiB block of their control-memory position, so
releasing them does not walk every live copy (4 KiB pages would cost hundreds of lookups under the exclusive lock the
recording threads wait on). `CallCommands` and `CopyCommands` inline the called recording, so copies hold no `OP_CALL` or
`OP_JUMP`. Copies released while a queued submission still reads them go to a zombie list, freed once `nvn_submit_count()`
reaches their `used_at`. `nvn_api_core.c` counts calls into a shared counter 1024 at a time so recording threads do not
fight over its cache line.

Pools. A pool in the GPU window (CPU-invisible) keeps its GPU address in `mem` and the game's storage in `game_mem`. Every
other pool uses the game's storage and the GPU address equals the guest address.

Texture pins. Recordings refer to textures by pointer, but the game may finalize or reuse an NVNtexture between recording
and submit. Records therefore point at an immutable host copy per texture (one per `tex_t.serial`), kept for the whole run.
Pins are keyed by content (every field but the serial). The game re-initializes its transient textures every frame with the
same storage and a new serial, and the replay finds textures by address and description, so equal contents share one copy.

Memory callbacks. NVN calls the game's callback when a recording needs command or control memory and none is left. Records
live in host memory, so it is called only where the hardware certainly would (a recording starts without command memory or
ends without control memory), on the calling guest thread.

Submit. `nvn_submit_prepare` (called from `nvnQueueSubmitCommands`) reads on the game thread what the console reads at
submit time. Every handle must be a live recording. Deferred binds and pushes (payload `src != 0`) take the data the game's
arrays hold now, because the game fills its tables for the next frame as soon as the submit returns. Syncs come only from
`QueueFenceSync` and window acquire. Waiting for their maximum before the batch orders CPU completion callbacks and GPU
work before any deferred read.

### Texture-view and sampler state getters

`runtime/nvn_ext.c`. Entry points that do not draw: debug labels, markers, groups and checkpoints (ignored), timestamps
(counters are written in guest counter ticks, the base of GetSystemTick, converted with `rt_ticks_to_ns`), pool and buffer
invalidation (pool storage is written by the GPU directly), builder, view, sampler and fixed-function state queries, and
the Tegra tiled-cache and ZCull hints. Setters of state with no host equivalent store the value and the replay reports a
non-default use once. Lines are one pixel wide, point size comes from the vertex shader (the support buffer), and an enabled
depth-bounds range is drawn without the test and reported. Only the default provoking vertex (last vertex) is accepted.

## Replay on Vulkan

### Draw replay

`runtime/vk_exec_draw.c` is the strong `vk_exec_op_draw` behind `vk_exec.c`'s weak hook. The bind ops fill the `VkExec`
slots. A draw opens the render pass and resolves the bound program to translated shaders (the translator's answers come
from the bound state). It asks `vk_pipeline.c` for the pipeline, pushes or allocates set 0 as `VkPipeInfo.bindings` says,
and records the dynamic state, the vertex and index buffers and the draw. A dispatch ends the pass and does the same for the
compute shader. `UPDATE_UB` is a buffer copy from transient memory between full barriers. Nothing locks: it runs on the
submitting thread under `vk_sync_submit`'s lock.

Each command buffer keeps transient memory (support buffers, copied uniform ranges, generated indices), descriptor pools,
image views and the storage-buffer ranges the GPU may write. All of it is freed from one completion callback once the
submission's fence is done. The written ranges are reported with `vk_gpu_written` there, because the write watch does not
see GPU writes. Spare descriptor pools and full-size transient chunks go back to free lists instead of being destroyed,
since each creation is a driver allocation.

Per-draw lookups are cached because a scene issues thousands of draws:
- (program, stage, answers) -> shader is a 4-way set-associative table; a miss costs a SHA-256 of the shader code.
- The stage and code length of each program are direct-mapped by program serial. Texture and image types of a stage's first
  32 slots are kept with the handle and a (generation, slot token) pair. They are looked up again only when the handle
  changed or `slot_still` says the slot may resolve differently.
- The slots whose types a translation reads are learned from its first shader, so `answers_for` does not look up all 64
  handles.
- The previous draw's shaders are reused while program, pipeline state, texture handles and slot resolution are unchanged
  (`tex_serial` skips the 1 KiB handle compare).
- Image views and attachment views are kept for their image's lifetime. Sampled-texture descriptors are cached per (handle,
  resource type, separate) while the slot resolution and samplers are unchanged.

Descriptor build skip. A graphics draw whose descriptor inputs equal the previous graphics draw's in the same command
buffer skips building its sets. The signature (`bind_sig`) covers the pipeline info and program, the texture, epoch and
sampler generations, `vk_mem_gen`, the images in the feedback layout, and per binding the support inputs, c0's storage
table, the bound uniform and storage ranges and the texture and image handles. Storage buffer addresses read from user
uniform blocks are compared on every draw, since the block changes without a bind op. The side effects of a build repeat as
no-ops (c0 and support buffers kept in `B`, `note_written` ranges, write-backs, mirror sync, cached views). A build that
copied guest bytes into transient memory is never skipped afterwards. `B.res_valid` says the bound sets hold the last
build; `vk_exec_draw_new_cb` clears it and so does every build until it ends.

Driver constant bank c0. Shaders read the NVN driver constant bank c0 as a buffer:
- storage tables of `{u64 address, u32 size}` every 16 bytes at 0x110 (vertex), 0x510 (fragment), 0x310 (compute);
- the geometry table at 0x410, the compute dispatch size at 0x200 and its uniform table at 0x210;
- the largest point size at 0x610 and the base instance at 0x644.

`vk_c0_fill` builds it; an equal bank reuses the previous range. Storage buffers whose address a shader loads from a user
uniform block are found through `ubo_ptr`. It reads the latest inline uniform update covering the pointer (a ring of the
submission's updates), else memory. Compiler constants of a lowered NVN 2 stage (uniform slot 1) are uploaded once per stage
and kept to shutdown, because a per-draw copy would make the draw's descriptor build volatile. Storage bindings with no
declared size are capped at the mapped size, at most 8 MiB, so a guarded candidate does not bind most of a multi-GB pool and
write it all back.

Feedback loops. A fragment shader may sample a target of the pass that an earlier draw of the pass wrote. Vulkan leaves that
undefined without a barrier. Two settings choose the handling. In the first the image is learned (`VkTexImage.fb_learned`)
and the pass restarts once with it in ATTACHMENT_FEEDBACK_LOOP_OPTIMAL. Later passes attach it in that layout, its draws use
a pipeline with the feedback flag, and a by-region barrier precedes them. In the second the pass ends and restarts behind a
full barrier.

Depth twins. A compute shader that samples an R32_SFLOAT colour image through a shadow sampler needs a compare-capable
format, which some drivers do not give for that format. Such an image gets a depth twin of the same size (D32_SFLOAT, or
D16_UNORM for R16_UNORM). The twin is copied through a buffer right before each dispatch that samples it (a dispatch is
outside any render pass), and the dispatch samples the twin. Draws keep the colour image.

ind_clamp. Indirect dispatches, mesh launches and indexed indirect draws read their counts from a copy that a one-thread
compute shader (`shaders/ind_clamp.comp`) checked first. An axis over `IC_AXIS` or a total over `IC_TOTAL` is zeroed and
counted with its raw values. The game fills these records on the GPU, and a record its producer did not write this time
holds stale bytes (a float read as a count keeps the GPU busy for seconds). The copies go to a device-local ring of
`IC_SLOTS` 32-byte slots.

Draws that wait. A draw into a target the previous frame did not render waits for its pending shader or pipeline, because
one-shot draws (volumes, LUTs) would otherwise leave the target empty. Draws into per-frame targets are skipped until ready.

### Uniform snapshot

`runtime/vk_usnap.c` keeps uniform blocks as they were at the submit call. NVN starts the GPU when `nvnQueueSubmitCommands`
returns and the GPU reads the constants soon after. Here the replay records for several milliseconds, and the GPU reads a
pool's bytes only when it runs the part that uses them (the VRAM mirror's sync copies run then, `vk_memory.c`). Meanwhile
the game, having handed the frame over, rewrites per-object uniform blocks on other threads with no fence, so late draws
would read another object's matrix. At the start of a submission's recording (in `vk_exec_record`) every uniform range it
binds is copied into an arena of its own, and the replay's uniform bindings use the copy.

Only ranges of imported pools (memory the CPU writes) are copied, aligned out to 256 bytes (the largest
`minUniformBufferOffsetAlignment`) and at most 64 KiB from the bound address, with overlapping ones merged. A range the GPU
wrote in a submission that has not completed is left out: its bytes reach guest memory only with that submission's
write-back, and the mirror holds them until then. A GPU write within the submission (UpdateUniformBuffer, a storage buffer a
shader writes, a buffer copy or fill, an image-to-buffer copy: `vk_usnap_drop`) drops the copies it touches from then on, so
later draws read the result. The arena goes back to a free list once the submission's fence is done (completion thread). CPU
readers (mesh handle programs) take a shadow copy in ordinary memory, since the arena is device-local memory written through
the BAR and reading it back is slow.

### Command buffer parts

`runtime/vk_exec.c`. NVN starts the GPU as soon as commands are submitted, but one Vulkan command buffer per submission
would start it only after the whole replay. Once 64 draws and dispatches have been recorded, and the part spans at least
1000 us of replay, the part goes to the queue (`vk_sync_split`) and replay continues in a new command buffer. A split may
fall inside a render pass: the pass ends, the part is submitted, and the next draw begins the pass again with its
attachments loaded. A pass break is cheap, while one long pass keeps the GPU idle until it is all recorded. Only
command-buffer state is lost at a split (bindings and dynamic state, see `vk_exec_draw_new_cb` and `ex->dirty`). Barriers,
image layouts and queries survive. Fences, completion callbacks and reports stay per submission and only the last part
signals the value.

### Recorder thread

`runtime/vk_rec.c` makes the Vulkan calls of the replay on a thread of its own. `vk_sync.c` brackets each part of a
submission with `rec_begin` (after `vkBeginCommandBuffer`) and `rec_end` (before `vkEndCommandBuffer`). In between, every
`vkCmd*` on that command buffer copies its arguments into a ring packet, in call order, and the recorder makes the real
call. Calls into any other command buffer (uploads, the window, probes) are made at once. A call the ring has no packet for
(a structure with a pNext chain) waits for the ring to empty and runs on the calling thread. `rec_end` returns once every
call was made; `rec_end_cb` lets the recorder also make the last calls, `vkEndCommandBuffer` included. Submitting and the
command pool stay on the submitting thread.

### Queue coroutines

`runtime/vk_exec.c`. The game drives several NVN queues that pace each other with events. A worker queue's submission comes
in pieces, each behind a wait for an event the main queue signals at a set point of its frame. Then comes the main queue's
submission, which waits for an event the CPU signals once the frame's data is written, signals the worker's events and
waits for its answers. On the console each queue runs when its waits are met. Replaying in submit order with waits ignored
would run the worker's whole submission first on the previous frame's pictures.

So each queue keeps its pending submissions, its position (`WalkCur`) and its own NVN state. A stream runs until a wait that
is not met. Then another queue that can run takes over (state swapped in, pass ended, full barrier), into the same Vulkan
command buffer. Events signalled by command buffers count in replay order (`wev[]`); an event no command buffer ever
signalled belongs to the CPU and is read from its memory. What cannot run stays queued until the next submission, the CPU's
signal (`vk_exec_cpu_signal` then `vk_exec_co_kick`), or the present, which runs everything left with waits ignored.

### Replay target history

`runtime/vk_exec.c`. For each target subresource the replay remembers the frame it was last rendered in and the one before.
`prepare_draw` uses it to let a draw into a target the previous frame did not render (a volume or LUT generated once) wait
for its shaders and pipeline instead of skipping the draw. It is per subresource, not per image, because a volume may be
generated a few slices per frame. It is direct-mapped, so a collision makes a draw wait rather than skip.

### Transfers, counters and conditional rendering

`runtime/vk_exec_xfer.c`. Buffers are guest memory: `vk_mem_resolve` maps an address to the VkBuffer behind its pool.
Textures are the `vk_texture.c` images (linear texel layout, GENERAL layout; block-linear storage is only converted when a
mirrored texture is uploaded), so a copy is one Vulkan transfer command. The pitch of a buffer side comes from the payload
(`OP_COPY_STRIDE` or the tight row). Every copy ends the open pass and starts with a full barrier. An image destination is
marked GPU-written at record time. A CPU-visible buffer range is reported to `vk_gpu_written` once the GPU is done (the
write watch does not see GPU writes, so textures stored there re-upload).

Events and counters are CPU-visible side effects. Their values go into guest memory on the completion thread through
`vk_completion_add`, in submit order: event WRITE / ADD, counter `{0, timestamp}` for every type but samples passed. A
timestamp report writes 0 in the counter word, because the game gates its clears on two such words being unequal.

Samples passed is a running count on the queue. RESET_COUNTER zeroes it, each draw after it adds the samples that passed the
depth and stencil tests, and REPORT_COUNTER writes the running value. The game resets and reports it several times a frame
and reads the result on the CPU; a 0 culls what the query tested. The replay counts with occlusion queries:
- After a reset the first draw of a pass begins a query (precise when the device has `occlusionQueryPrecise`).
- It ends with the pass, or at the next report or reset, and the next draw begins another (a query begun inside a render
  pass must end in it).
- The game always resets, draws the tested proxies, reports, so counting stops at a report and restarts at the next reset.
  If the game reports twice without a reset in between, counting stays on for good (cumulative NVN semantics).
- The queries closed since the last report or reset form a segment, handed to the completion thread with that report or
  reset. It sums the segment's results into the running count (a reset drops the sum), and writes a report's count into
  guest memory.
- Slots come from one ring. The completion thread reads and host-resets them (`hostQueryReset`, Vulkan 1.2), so the replay
  records no reset commands. A full ring leaves the draw uncounted and the segment's count at least 1, as does a query
  still unavailable after its fence. Without `hostQueryReset` the counts stay 0.

Timestamp reports. On the console the GPU runs a submission as soon as the game submits it. Here the replay comes first and
the completion later still, so "now" at completion is mostly the replay's delay and every frame would measure longer than
1/60 s. The report therefore keeps an entry-tick approximation copied into its callback payload.

Pending reports are kept in a table by address (not a ring of the last few), since the game conditions colour draws on
reports of its depth pass. A small ring would forget running reports, and the condition would then read an older query's
count from memory, often 0, and drop a draw. A slot is free once its report is done.

Conditional rendering is a CPU-side skip, not `VK_EXT_conditional_rendering`: the extension compares one 32-bit word with
zero, while NVN compares the counter words at addr and addr + 16. The condition is evaluated when the op is recorded, from
guest memory. A counter reported by an earlier submission that is not done yet counts as 0 if it is a timestamp and as
unknown if it is samples passed. Unknown renders: the report usually sits in the command buffer being recorded, so it cannot
be waited for, and the word in memory still holds an older count. Rendering costs at most the work the condition would have
saved. The condition sets `VkExec.render_disabled`, which `vk_exec.c` honours for clears and `vk_exec_draw.c` for draws and
dispatches. A condition that cannot be read (address outside every pool) renders.

### 3D BC render targets

`runtime/vk_exec.c`, `runtime/vk_texture.c`. A 3D block-compressed image gets no block views, because on RDNA2 such an image
takes GPU writes past the end of its memory. The game's BC4 encoder draws into 32^3 and 64^3 volumes through an
uncompressed view of their blocks. Those draws go to a stand-in, a 2D array of one level's blocks in the view's format with
one layer per depth plane. It starts with the level's content so a partial pass keeps the rest. Each pass end copies the
planes it drew back into the level (a compressed to size-compatible uncompressed copy). Stand-ins are made on first use,
freed when their image goes (`vk_exec_image_gone`), and limited to `BC_STANDINS`.

### GPU capability fallbacks

`runtime/vk_caps.c` collects what the runtime branches on, logs it once at device creation and holds the fallbacks for GPUs
other than the development GPU.
- Without `shaderFloat64` the translator lowers FP64 to FP32 ("softfloat64"), so no shader declares Float64.
- With a `minStorageBufferOffsetAlignment` above 4 the translator computes storage offsets from the guest address rounded
  down to that alignment. `vk_exec_draw.c` `ssbo_info` binds the rounded-down address, so shader writes reach guest memory
  instead of a copy.
- Without a graphics pipeline library with fast linking (the default on AMD's RDNA 1/2 driver branch) `vk_caps.gpl` is 0,
  the extension stays disabled and pipelines are built whole.
- Without a 32-lane pin (no `subgroupSizeControl`, or 32 outside min..max) stages run at the device's `subgroupSize` (SPIR-V
  1.3 without a required size) and the translator is told that size.

The defaults (Float64, no storage rounding, 32 everywhere) leave the shader cache keys unchanged (`gfx_shader.c`).

## Textures

### Texture cache and re-upload

`runtime/vk_texture.c`. Images are cached by storage GPU address plus descriptor. A texture that mirrors CPU-provided pool
memory is uploaded on first use and again when its pages' write generation (`vk_memory.c`) shows a write. Draw bindings
check once per submission against that submission's write-watch snapshot. Once the GPU renders into a mirrored texture it
becomes GPU-owned and is no longer re-uploaded; the GPU-written image is the newest content. A new descriptor over the
storage of a live image the GPU wrote gets that image's content (`alias_copy`) through a buffer, because bound images share
memory.

### Slot resolution

`runtime/vk_texture.c`. A texture slot caches the entry it resolved to, valid while the generation of its storage address
stands. The generation is per hashed address, because the game finalizes and re-creates transient render targets every frame
and a global one would empty every cache. A finalized slot follows a live entry at its storage:
- first one that differs from the slot's descriptor only in swizzle;
- otherwise one of the same shape (size, depth, layers, samples, storage size) whose format a view of the slot's format can
  read (never depth to colour);
- otherwise it gets its own image again.

Among same-shape entries the one the GPU wrote last wins. Following a different-sized transient would flip the slot between
images from draw to draw.

### Released texture cache

`runtime/vk_texture.c`. A finalized texture is kept released, because the game re-initializes it a few frames later.
Released entries used by a slot within `REL_USED_FRAMES` are never evicted. Past `RELEASED_MAX` the oldest goes at each
release, with entries that hold their content in storage dropped first, then GPU-only images, then the rest. Above 90% of
the device-local budget a few of the oldest entries that come back intact are dropped per submit. Every 15 frames a sweep
frees released GPU-only images by evidence. An image is dead when another GPU-written image was written over its storage
after its last use, and goes after a few unused frames. Any other goes after `REL_KEEP_FRAMES`.

### Texture write-back

`runtime/vk_texture.c`. A mirrored texture the GPU rendered into holds its texels only in its image, but the game still
reads that storage with the CPU (it copies GPU-generated maps between pools). Such reads write the image back first, newest
content only (nothing wrote the storage since the render), oldest render first.

A game may also render a small map into a CPU-cached pool and read it without an invalidate; on the console the bytes are
there once the frame's fence has passed. So when the CPU waits for a fence, mirrored targets up to `wb_limit` bytes rendered
by work up to that fence go back. Doing that through the upload batch costs a GPU wait under the texture and upload locks.
So `tex_prestage` copies such targets (up to 262144 bytes) into a host-visible ring at the end of each submission, and a
write-back of that same render whose fence is done reads the ring instead.

### Block-linear textures

`runtime/tex_deswizzle.c` converts Tegra X1 block-linear layouts to and from linear (public descriptions of the layout
exist). A GOB is 64 bytes x 8 rows = 512 bytes. The offset of (x byte, y row) within a GOB is
`(x%64/32)*256 + (y%8/2)*64 + (x%32/16)*32 + (y%2)*16 + x%16`. A block is 1 GOB wide x 2^th GOBs high x 2^td GOBs deep,
ordered y first then z inside a block, and blocks fill the level in x -> y -> z row-major order. The Switch 2 GOB differs by
swapping bits 5 and 6 of that offset. Level and layer start offsets and the block height and depth of a level come from
`nvn_tex.c`; `tex_deswizzle.c` only does the addressing inside a level and the copy.

### ASTC textures

`runtime/vk_astc.c`, `runtime/tex_astc.c`. The mode is native when the device reports `textureCompressionASTC_LDR` and the
format's features. Otherwise it is gpu, a compute decode into RGBA8 (`shaders/astc_decode.comp`). Otherwise it is cpu
(`astc_decode_ldr_mt` on the uploading thread) when the decoder's pipeline cannot be made. In gpu mode a second pass
(`shaders/bc3_encode.comp`) encodes the decoded texels to BC3, so the image takes 8 bits per texel instead of 32, at the cost
of BC3 quality (slightly blockier textures, less precise data textures such as normal maps).

The decode is recorded into the upload batch (`vk_memory.c`) in chunks of at most 32 MiB of texels. A chunk's raw blocks go
from staging to a device-local scratch buffer, its dispatches write RGBA8 texels behind them, and a buffer-to-image copy puts
them into the image. The next chunk reuses the scratch after a barrier. The scratch buffer is freed once the batch completes
(`vk_defer_buffer`). The CPU decoder (`tex_astc.c`) matches the reference 8-bit ASTC decode bit for bit: endpoints expand to
16 bits, texel = `(e0 * (64 - w) + e1 * w + 32) >> 6`, output the top 8 bits; error blocks give magenta.

### ASTC to BC3

`runtime/shaders/bc3_encode.comp` is the second pass of the ASTC decode (`vk_astc.c`): RGBA8 texels become BC3 blocks, so a
decoded ASTC texture takes 8 bits per texel in video memory instead of 32.
- Colour: the BC1 four-colour block of the box around the block's colours, its diagonal turned to the sign of the colours'
  covariance and inset by 1/16 of the range (the bounding-box method). Each texel takes the nearest of the four palette
  entries along that axis.
- Alpha: the eight-value BC3 alpha block between the block's smallest and largest alpha.

The bytes are what the texels hold, so an sRGB image keeps sRGB-encoded values, as BC3_SRGB expects.

## Pipelines and descriptors

### Pipelines and descriptor sets

`runtime/vk_pipeline.c`, declared in `vk.h`. Each pipeline has one descriptor set 0 (or set 0 and set 1 when independent
pipeline layouts are available). Binding 0 is the support buffer, a uniform buffer every stage shares. Then come the vertex
stage's bindings, then the fragment stage's (its SPIR-V is renumbered). Set 0 is a push-descriptor set unless the pipeline
needs more than 32 descriptors; `VkPipeInfo.bindings` says what goes where. There are no push constants.

`VkPipeState` is everything a graphics pipeline depends on as the op stream carries it. Fields covered by dynamic state are
read only by `vk_pipeline_set_dynamic` and left out of the pipeline key. Descriptors go into one packed `VkDescSlot` array
(24 bytes each, unused bytes zero so equal descriptors compare equal) that a descriptor update template reads directly.
After `vk_pipeline_async_enable` a missing pipeline is built by background workers and the draw is skipped until it is
ready. Pipelines built in a run are recorded to `pipeline-keys.bin` and prebuilt at the next boot.

### Pipelines and descriptor bindings

`runtime/vk_pipeline.c` maps the NVN state collected by the op-stream replay (`vk_exec.c`, a `VkPipeState`) to Vulkan
graphics and compute pipelines, numbers descriptor bindings and keeps the caches.

Binding model. The translator (`tools/xlate/Program.cs`) puts every shader resource in DescriptorSet 0 and numbers the
bindings of each shader from 1 in the order it meets them. Binding 0 is the support buffer (a std140 uniform buffer every
shader declares). Numbering is per shader, so a vertex and a fragment stage of one program collide. The runtime therefore
uses one descriptor set per pipeline and no push constants. Binding 0 is the support buffer shared by all stages. The vertex
stage keeps its bindings 1..n; the fragment stage's bindings b >= 1 move to b + n (the SPIR-V Binding decorations are
rewritten when the module is created); compute keeps its own. Descriptor types come from the reflection:
- constant buffers are uniform buffers, storage buffers are storage buffers;
- textures are combined image samplers (uniform texel buffer for buffer textures, sampled image for separate ones);
- images are storage images (storage texel buffer for buffer images);
- array lengths are descriptor counts.

Set 0 is a push-descriptor set when its descriptors fit `maxPushDescriptors` (32 on the development GPU). A pipeline layout
may hold only one push set, so a set per stage is not possible. Larger layouts are ordinary sets the replay allocates
(`VkPipeInfo.push` is 0). A fixed layout per stage combination would hold about 150 descriptors and could never be pushed,
so layouts are made per distinct binding list and shared by all pipelines with that list. `VkPipeInfo.bindings` tells the
replay which NVN slot feeds each binding.

Fragment inputs the vertex stage does not write (stages are translated separately) get outputs added to the vertex module,
because Vulkan rejects such pairs. They start at (0, 0, 0, 1), which is what Maxwell reads from an unwritten output
(translator patch `tools/xlate/patches/0006`).

Pipeline key. Everything Vulkan 1.3 allows as dynamic state is dynamic: viewports and scissors with count, depth bias,
blend constants, stencil masks and references, cull mode, front face, topology (within its class), depth and stencil tests
and ops, depth bias enable, rasterizer discard, and vertex input (`VK_EXT_vertex_input_dynamic_state`, else binding strides
only). Where `VK_EXT_extended_dynamic_state3` offers them, depth clamp, polygon mode, blend enables, equations, write masks
and alpha to coverage are dynamic too. The key holds the rest: shaders, target formats, sample count, topology class, depth
clip mode, logic op and whatever the GPU cannot make dynamic. The `VkPipelineCache` is saved to `pipeline-cache.bin` next to
the executable.

Stage-independent layouts (`P.indep`, graphics pipeline libraries). The vertex bindings go in set 0 and the fragment
bindings in set 1, each with its own support buffer at binding 0. The pipeline layout is created with the independent-sets
flag so stage libraries can share it. Only set 0 can be a push set; set 1 is allocated per draw from the command buffer's
pools. Each vertex shader declares the union of the inputs of every fragment shader it has been paired with (from
`pipeline-keys.bin` at prewarm, then each pipeline built), so the vertex module and its pre-rasterization library do not
depend on one pipeline's fragment shader.

### Graphics pipeline libraries

With `VK_EXT_graphics_pipeline_library` (`vk_caps.gpl`, in `runtime/vk_pipeline.c`) a pipeline is linked from four
libraries, each cached by the part of the pipeline key it depends on:
- vertex input (topology class, vertex attributes when vertex input is not dynamic);
- pre-rasterization (vertex shader, the outputs its union adds, static rasterization state);
- fragment shader (fragment shader, dual-source rewrite, samples);
- fragment output (targets, blend, samples).

With independent layouts each stage's set depends on its shader alone, so the pre-rasterization library uses `{set 0, null}`
and the fragment shader library `{null, set 1}`. The fast link runs on the thread of the draw that needs it, without
link-time optimisation. The optimised pipeline is the job `pipe_miss` queued, a monolithic build with the same modules. When
it is done it replaces the fast one in the `Pipe`. Libraries and fast pipelines live until `vk_pipeline_shutdown`, because
recorded draws still use them.

### Background pipeline builds

`vk_pipeline_async_enable` starts a few worker threads (at most four). A cache miss in `vk_pipeline_graphics_async` enters
the `Pipe` as building and queues a job. The replay skips draws that use it until a worker publishes the handle. The job
holds copies of the state and the key; what they point to outlives it (shaders live for the process, layouts until
`vk_pipeline_shutdown`, which drains the queue first). Workers call `vkCreateGraphicsPipelines` without `P.lock`.

There are two queues: draws' misses and the boot prebuild. Workers take the prebuild queue only when the draw queue is empty
and run its jobs at below-normal priority so guest threads win. A draw that finds a prebuild job still queued moves it to the
draw queue; one that finds it running raises its worker's priority. Compute pipelines are built synchronously because a
skipped dispatch can change what the game computes. Shader libraries a waiting draw needs are queued ahead of every
pipeline job, so an idle worker builds each at once and the vertex and fragment libraries of one draw build in parallel. A
library job no worker took is built by the first thread that needs it.

### Pipeline key file and prewarm

Every graphics pipeline built in a game run appends a `KeyRec` to `pipeline-keys.bin` (next to the pipeline cache). It holds
the `VkPipeState` the pipeline was built from, with shader pointers zeroed and the shaders named by their `gfx_shader` input
ids (which do not depend on the translator), so the record rebuilds the pipeline without guest memory.

At the next boot `vk_pipeline_prewarm` (from `rt_boot`, before the guest runs) loads each record's shaders. It translates
the ones not cached under the current translator from their stored inputs (`gfx_shader_prewarm`; a record whose shader has
neither is dropped). It enters each `Pipe` as building under the key a draw would compute, queues it on the low queue and
builds the queue on every core.

Records are deduplicated by a hash of their bytes. They are collected under `P.lock` by the worker that built the pipeline
and appended by the present thread (`vk_pipeline_autosave`) or at exit (`vk_pipeline_save`). A file with another magic,
version or record size is ignored and rewritten. `KEYS_VERSION` must be bumped when `VkPipeState` or the record changes
meaning without changing size. The record keeps a frozen prefix of `VkPipeState` (up to `gs`) so adding fields does not
reinterpret old records.

### Stage library prebuild

Two places build what a draw would otherwise build while the game waits (`runtime/vk_pipeline.c`).

`oot.exe --prepare-shaders` (`lb_warm_prepared`) runs after the translation on every core. For each prepared variant
(`shader-prepared.bin`) it builds the pre-rasterization library of a vertex shader, the fragment-shader library of a
fragment shader and the pipeline of a compute shader, each from a private shader copy that is destroyed at once. What
remains is the driver's own disk cache; a library built again in a later process then takes far less time.
`pipeline-cache.bin` is not written there, because it would grow with every library and be read into memory at every boot.

At boot (`lb_worker`), after the pipeline-key prewarm, `cpus/4` threads at idle priority build the compute pipelines of
every recorded and prepared compute shader and keep them (`vk_pipeline_compute` builds compute synchronously on the replay
thread). The short `P.lock` sections and shader loads run at normal priority so a preempted thread never holds the lock the
replay needs. No library is kept: a kept library costs process memory to save a rebuild the driver cache makes nearly free.

The library state used is the vertex output union known now (`vs_union_peek`: the prewarm fills it from `pipeline-keys.bin`,
the preparation has none), the clamp, polygon mode and depth mode most recorded pipelines use (defaults without a key file:
clamp off, fill, -w..w), single-sampled, no dual source. State that is dynamic on this GPU is 0 in every key. A draw that
meets such a variant in a new pipeline then gets its libraries from `libs[]` or the driver cache and fast-links
(`gpl_link`).

The preparation keeps a journal, `shader-built.log` next to the executable, so a stopped preparation resumes where it
stopped and a module that kills the process (the driver's compiler can run out of memory) is not tried again. A line
`D <cache key>` means built into the driver cache (or nothing to do). `S <cache key>` means a big module's build began. Two
`S` and no `D` is a build that twice never came back (once may be the run being stopped); it is skipped and counted. The
cache key holds the translator's identity, so a new translator starts over by itself. `oot.exe --warm-shaders` repeats the
builds from the shader cache without translating. The AMD driver cache is keyed to the exe path and limited in size, so
preparing another build dir evicts the libraries, and a library a draw needs would then be slow to build in play.

## Memory mirror and write-back

### Write watching and page generations

`runtime/vk_memory.c`. A two-level table over the guest address space keeps a generation number per page (256 MiB chunks of
65,536 page generations, allocated on the first stamp; 0 = never stamped). A write-watch query (`GetWriteWatch` /
`PAGEMAP_SCAN`) resets the OS dirty bits, so each query stamps the pages it saw with a new generation shared by all
consumers (mirror sync, texture staleness checks). A texture is stale when a covered page carries a newer generation than
its upload. Queries run on 2 MiB regions, at most once per epoch (a guest submission); a snapshot only adds observations to
the table. The pre-snap of each submission asks about hundreds of runs, so in a batch the runs are split into pieces of at
most 4 regions, queried on 3 workers plus the replay thread (the kernel walk takes the mmap lock shared), and applied in
order with `gen_lock` held.

Frozen regions. A 2 MiB region that nothing wrote for `freeze_idle` submissions is made read-only (`mem_freeze`) and skipped
by queries until a write or a file read thaws it. Most pool memory is static and a query costs time per MiB. Regions may
also hold non-pool guest memory; that is safe because the only kernel writes into guest memory are file reads, which thaw
their buffer first, and every user-mode write faults and thaws.

Write-back echo. A written binding goes mirror -> guest at the end of its submission, and at its fence the pages are stamped
as written for the texture checks. A page the write-back covered whole is marked as an echo, and `mirror_sync` skips a page
whose newest stamp is its echo, so the same bytes are not uploaded back. This only applies when nothing newer than the
submission's recording was stamped on the page; a CPU write seen later keeps the plain stamp.

### VRAM mirror and write-back

`runtime/vk.h` declares the interface, `runtime/vk_memory.c` and `runtime/vk_writeback.c` implement it. On a discrete GPU an
imported pool would be read over PCIe on every draw. In mirror mode (`vk_mirror_on`) the pool gets a device-local buffer over
its imported range the first time it is used as a vertex, index, uniform, storage or indirect buffer. Those bindings, and
buffer copies, fills and uniform updates, then use the mirror at the same offsets; the guest memory stays what the CPU reads
and writes.
- `VK_MEM_USE` at a vertex, index, uniform, storage or indirect binding creates the pool's mirror on first use and makes
  `sp.buffer` the mirror. `VK_MEM_PICK` takes an existing mirror for copies and fills.
- `vk_mem_gpu_wrote` records a GPU write of a range so its bytes go back to guest memory at the end of the submission.
- CPU -> mirror: the first access of a submission syncs the mirror. Pages stamped after the mirror's `mgen`, plus
  `FlushMappedRange` ranges, go into copies recorded in a prologue command buffer that `vk_sync.c` submits before the part
  that touched the mirror first (`vk_mem_mirror_prologue` is the only path from guest memory into a mirror). A flush
  bounding box is not copied when the write watch tracks every region of it, since the CPU-written pages are queued from
  their stamps anyway.
- GPU -> guest: ranges the replay wrote into a mirror are copied back, exact bytes, at the end of the submission, before the
  fence signals.

A pool stays imported when the mirror would not fit the VRAM budget, its range is not write-watched, it overlaps another
pool, or the allocation fails. The mode is off on NVIDIA in the shadow case because of stale camera blocks (see
`mirror_mode`).

Write-back rule. A written binding goes back whole, and nothing tells which bytes a shader stored. Guarded candidates flag
buffers the shader never touches, and the game rewrites some of those from the CPU while the GPU runs, so a stale copy over
a newer CPU write would roll back the camera or the sky sum. The rule therefore watches small ranges (up to 4096 bytes, and
up to 64 KiB for sizes that are a whole number of 256-byte particle records).
- Just before each write-back the guest and mirror bytes of a watched range are copied into a host ring (pinned memory,
  allocated at first use). `wb_done` moves its tail as submissions complete, and an item that would overwrite bytes in
  flight is left out of that submission.
- At the fence, guest bytes that differ from what the last write-back left mean the CPU wrote the range. From then on its
  GPU data lives in the mirror only, and the CPU's next write still reaches the mirror through the page upload of
  `mirror_sync`. The first write-back after the CPU's first write still lands.
- The entry table is swept every 6000 submissions and entries unseen for 60000 submissions are freed, so the copies stay
  under 1 GiB while the game moves through the world.

wb_dirty. Each pool that writes back gets a VRAM shadow of what guest memory holds. It is filled from the import at its
first write-back, and refreshed from the mirror after every upload into the mirror and by the write-back itself.
`runtime/shaders/wb_dirty.comp` (recorded by `vk_writeback.c`) compares one chunk per workgroup (at most 4 KiB, never across
a page, one workgroup per 4 KiB chunk). Here m is the VRAM mirror (the GPU's bytes), s the VRAM shadow, g guest memory (the
pool's import). When m and s differ anywhere in the chunk the whole chunk is written to g and s. When equal, the chunk holds
no new GPU result and nothing crosses PCIe. Ranges whose ends are not 4-byte aligned (their 1 to 3 byte ends are copied
separately), parts over half the chunk ring and pools without a shadow take the plain copy.

pend_wait. A write-back range the CPU may read before the GPU finished is trapped (`plat_mem_trap`, Linux only); the access
hook waits for the submission's fence. `pw_submit` arms only pages committed read-write.

### Shadow storage

`runtime/vk_memory.c`. Importing guest memory costs too much on Linux (re-protecting pages that amdgpu imported as a userptr
invalidates the buffer at the next submit, and the import's write pinning clears the write watch). A CPU-visible pool then
gets a host-visible buffer of its own, at the offsets the import would have. The mode is on for Linux except NVIDIA.
- CPU -> shadow: before each submit, the pools the replay resolved are queried and pages stamped after the pool's `sgen` are
  copied.
- GPU -> guest: for CPU-visible pools the replay notes every GPU write at record time (`vk_mem_gpu_pending`) and reports it
  at the fence with `vk_gpu_written`. That call copies the bytes back into guest memory before telling the texture cache
  (`vk_guest_written`). Until then the CPU copies skip them, so a page both write keeps the GPU's bytes. `vk_gpu_written_at`
  is the same for a submission whose mirror write-back covered whole pages, which keep a generation the mirror sync knows as
  its own echo.

Shadows are deferred: most CPU-visible pools only hold textures, so the shadow is made at the first resolve. A pool of
256 MiB or more gets windows instead (one 2.4 GB pool is bound through one 8 MiB range). Window k covers
`[import_lo + k*32 MiB, + 64 MiB)`, so every lookup of an address in step k picks window k. A window has no index entry, so
it gets no VRAM mirror.

### Sky SH keep

`runtime/vk_memory.c`. The game's sky spherical-harmonics block has rows 1-2 stored by the CPU every frame, row 8 written by
an early compute, and rows 3-6 accumulated by a sum compute and cleared by the CPU every frame. Its write-back stops ("the
CPU wrote it"), so the guest copy of rows 3-6 stays 0. The CPU's store of rows 1-2 would then upload the whole page at the
start of the submission with the exposure measurement, which would read the sky SH as 0 and darken the scene. On the Switch a
CPU store touches only its own bytes. So the block's two accumulated windows (`+0x30..+0x70`, `+0x80..+0x90`) keep their GPU
bytes across uploads (saved before, put back after; the first upload goes through). Each window is zeroed in GPU order just
before the dispatch that sums into it.

### VRAM headroom

`runtime/vk_headroom.c`. VMA places an allocation in an existing block of its memory type and calls `vkAllocateMemory` (32 to
256 MiB) only when none has room, which stalls the render thread. A new `VkDeviceMemory` made by any other thread (VMA's
allocate callback) wakes the headroom thread for that memory type, and it checks every 500 ms. A check allocates half a
default block of that type. If a block has room the allocation lands there and is freed at once. Otherwise VMA allocates a
new block, and the allocation stays as that block's anchor, because VMA deletes an empty block at the next free of its type.
The previous anchor is freed when a new one is made, so at most one spare block per memory type is held back. Under VRAM
pressure (use plus two blocks past 85% of the budget, since the spare block counts against the budget that decides whether a
pool gets a mirror) the anchor goes and nothing new is allocated.

## Shaders

### Shader cache

`runtime/gfx_shader.c` turns an NVN program (control section, code, c1 constants and the translator's answers) into SPIR-V
plus reflection through `oot_xlate.exe --serve` children. They start lazily, talk over pipes and live for the process (2 at
run time, more for the boot prewarm). Results are cached in memory and on disk (`shader-cache/<key>.spv` and
`<key>.res.json` next to `oot.exe`), so a shader is translated once per machine and translator.

Keys. The code hash is sha256 over the stage (u32), the 0x880-byte control section, code_size (u64), the code, and the c1
length (u32) and bytes. A geometry shader also hashes its versioned input-topology suffix.
- For each code hash and translator the cache keeps a use mask (`<sha256("TKSU", translator id, code hash)>.use`): the
  texture and image slots the translator queried (`tex_queried`, `img_queried` in the .res.json), the vertex attributes a VS
  declares and the colour targets an FS declares.
- The input id is sha256 over "TKSI", IN_VERSION, the code hash and the answers with everything outside the mask zeroed, so
  stale bindings in slots a shader never reads do not create new ids. It does not depend on the translator.
- The cache key is sha256 over "TKSK", KEY_VERSION, the translator identity and the input id.
- A code hash with no known mask is translated under a provisional key ("TKSP", full answers). Its files are renamed to the
  cache key and the mask is stored afterwards.

Translator identity is `build/xlate/oot_xlate.id` (a sha256 that `tools/build_xlate.py` writes over the sources it builds
from), or the sha256 of the exe when that file is missing. A translator change therefore changes every cache key without a
manual bump. KEY_VERSION is for changes on the runtime side (key layout, use of the output).

Cache pack. `tools/pack_shader_cache.py` folds the .spv, .res.json, .use and `in/*.in` files into
`<cache>/shader-cache.pack` (one file instead of four per shader) and deletes them. The pack is read-only at run time. Every
read tries the pack first and then the loose file, and new translations are written as loose files until the tool runs
again. Layout: "TKSPACK1", u32 count, u32 0, entries sorted by (kind, name), then the data. A pack that fails a check is
ignored as a whole.

Stored inputs. Every translation also writes `<cache>/in/<input id>.in` (magic "TKSI"; stage, full answers, control section,
code, c1). A pipeline key record names its shaders by input id, so it can be translated again under a new translator without
guest memory (`gfx_shader_load_input`, `gfx_shader_prewarm`).

Reflection. A shader entering the cache has its .res.json parsed into `GfxShaderRes` arrays and its SPIR-V scanned for
interface Locations and the workgroup size.

SPIR-V storage. A module is repaired and checked once (`spv_checked` is set only afterwards), then kept as a zstd frame
(level 1) and the words are freed. `gfx_shader_spirv` decompresses into a per-thread ring of SPVZ_RING buffers. A buffer
above SPVZ_KEEP is freed when its slot is reused. Asking again for a shader still in the ring returns the same buffer, and
the slot reused is the one returned longest ago, counting ring hits as returns. So a pointer stays valid until
SPVZ_RING - 1 other modules were requested on that thread. This matters because `create_graphics_with_mesh` takes the vertex
module, then the fragment module, then builds the vertex module again. A ring entry also records the count of freed
compressed shaders, so a shader allocated at a freed one's address never gets the old words. Callers that only need the size
use `gfx_shader_spirv_words`, which never decompresses.

Answers met in play. Every (program, answers) that a `gfx_shader_get*` call resolves is recorded next to `pipeline-keys.bin`,
so first-start preparation translates exactly the variants play needs.
- `shader-answers.bin`: "OOT_SANS", version, record size; records of stage (u32), code hash, input id, packed full answers,
  GS input selector, priority. One record per input id; the full answers are kept so a translator with a wider mask still
  gets them. Under a new translator the ids are new and play appends records for them.
- `shader-programs.bin`: "OOT_SPRG", version, 0; records of stage, length, c1 length (u32 each), code hash, control section,
  code, c1. One per code hash, written before the answers that name it. It holds game content, so it stays in the ignored
  cache directory.

A file with another header is rewritten and a torn tail is cut off at open. A seed record (from `tools/prepare_shaders.py`,
`oot --prepare-shaders`) has a zero input id and its pad byte holds the list priority (0 title, 1 warm-up, 2 rest).
`shader-prepared.bin` ("OOT_SPRP") lists the prepared variants the boot warm-up reads.

Lowering NVN 2 stages ahead of play (`oot --lower-shaders`). Each vertex, fragment and compute stage of `seed-nvn2.bin` goes
through `nvn2_lower` as `nvnProgramSetShaders` would send it, so the stage gets the code hash it gets in play. Output is
`seed-programs<sfx>.bin` and `seed-lowered<sfx>.bin` (census key, stage, flags, length, code hash). Processes shard the work,
because the lowering holds one lock.

### Shader access to guest memory

`runtime/vk_memory.c`. Some translated shaders follow a pointer read from memory, so they need the guest address space
without a bound buffer. Two mechanisms exist.

The raw pool table (translator patch 0027) is a ring of tables of sorted 8-word entries `{start, end, delta}`, where delta
is the device address minus the guest address. Entry 0 is empty and unused entries start past every address. A ring is
needed because a table changed in place would break shaders still running on it. Writes through it are not noted as GPU
writes, so a pool reached only this way gets no mirror sync.

The guest page table (patch 0032) is the one actually used on capable devices (NVIDIA on Linux, with device addresses). It
is part of the shader cache keys, so it is decided before the first translation. A shader loads `table[guest address >> 12]`,
a device page number, and accesses `entry << 12` plus the page offset: no search, no bounds check, no branch. There is one
32-bit entry per 4 KiB page of 36 bits of guest address, and bit 39 (virtual pools) folds to the unused bottom of the table.
The table is aligned to its size because the shader ORs. A flat sparse buffer is not an option: this driver gives sparse
buffers only device-local memory, and the live pools do not fit beside the textures. A pool's pages can lead to its import,
its mirror, or a physical pool through a virtual one. Changes go in queue order through the prologue of the part that needs
them (`vk_sync.c`). Unmapped pages lead to a sink page, so a wild pointer reads junk and its writes go nowhere.

### Global loads in graphics stages

`runtime/spv_gmem_stub.h` repairs one translator gap at run time, so no new shader preparation is needed. A graphics stage's
global load whose address is a pointer read from memory matches no bound buffer. `GlobalToStorage` then builds a multi-target
helper with no targets, and that helper is `return 0`. Compute stages take the guarded path, which ends in the guest page
table (translator patch 0033). Graphics stages never reach it, so these loads read 0.

This breaks per-vertex probe lighting. The vertex shader mixes 8 probes, each read through a pointer (a valid flag and 27
values, 216 loads). All flags read 0, so each weight falls to 1e-5 and the normalising `1 / sum(weights)` is about 1e5. The
one term that does not come through a pointer is scaled by it, which shows as a strong green ambient light.

The rewrite gives such a helper the body patch 0033 emits: the table load, then the load at the device address of the guest
page. Only modules with a `return 0` helper of two int pointers change, so other pipelines stay cached. The helper takes
32-bit reads: a byte or short load would need its place in the word and the helper does not say its width. Every module seen
has one helper and those read floats, but that is not proven for all.

### Bindless texture set

`runtime/vk_exec_bindless.c` implements descriptor set 2 for shaders that sample through a handle read from a storage table
(world materials: baked light, shadow lookups). The translator turns such samples into indexed reads of set 2
(`tools/xlate/patches/0035`): one binding per image kind with the whole texture pool, and binding 8 the sampler pool. Every
pipeline layout holds the set, and it is bound after each `bind_resources`.

The pool can hold far more textures than the world uses and every one resolved here gets an image, so only the handles in
the table the materials read (fragment storage slot 3, 32-byte records, u64 handles at +16 and +24) are written. The table
is assumed to be in that slot and is rescanned every 16 frames, and also whenever a texture slot may resolve to another
image (`vk_texture_slots_gen`), because the game points the same ids at other transient targets from frame to frame. A
texture registered between scans reads the stand-in until the next one.

Set 2 is kept as a ring of `BL_RING` copies. A change is queued (`bl_put`). At the next bind (`bl_commit`) the current copy
is copied into one no pending submission bound (its last binding's timeline value is reached), the queued changes go into
that one and it becomes current. So no set is written while the GPU may read it: work submitted earlier reads the guest's
material table live (imported memory) and could otherwise take a just-changed id to a view created after its submission. If
no copy is free, the changes wait for the next bind (stand-ins for destroyed images are then written at once). More changes
per frame than free copies leave the later ones waiting (`bl_waits`). An image destroyed while ids of set 2 still named it
gets the stand-ins at once in every copy, and every table is scanned again at the next draw.

### Mesh handle programs

`runtime/vk_exec_mesh.h` replays the native mesh and task stages. A stage texture whose handle the stage reads from a storage
buffer is exported by `tools/mesh` as an `H` line in the module's `.bindings` file, followed by a small stack program
(immediates, constant-bank and storage reads, ALU, select and tee/load slots). `mesh_hp_eval` evaluates it for each draw over
the stage's constant banks and storage buffers in guest memory (the uniform snapshot first, then live memory) and puts the
handle into texture slot `MESH_HP_SLOT0 + n`, past the 32 slots bank 2 can name. Distant model meshlets sample a 2D array
through such a handle; without it their draws are skipped. A word that cannot be read or a malformed program skips the draw.

### Mesh pipeline builds

A mesh pipeline seen for the first time is built on one of two mesh builder threads. The draw that asked waits for it up to
20 ms, and at most 20 ms in all per frame. After that its draws are skipped (counted as "pipeline-building") until the build
is done, because the driver can take very long to compile a mesh pipeline and the render thread would freeze. A build whose
wait ran out once is not waited for again.

Every mesh pipeline a draw asked for is also noted in `<mesh cache>/mesh-keys.bin`: the mesh and task modules (cache key,
native key, array slots), the fragment shader's input id and the normalized state with shader pointers zeroed. The
pipeline-key prewarm of `vk_pipeline.c` covers vertex and fragment pipelines only. Without these notes a mesh pipeline would
be built in play at every first visit after a driver update or cache wipe. `vk_exec_mesh_prewarm` queues every noted
pipeline to the builders at boot (modules come from the mesh cache only; a note whose module is missing is left out). A draw
that meets one still building waits or skips as for its own build. A file with another magic, version or record size is
started again.

### Mesh depth and colour pass pairing

The game draws each mesh object twice: a depth pre-pass (depth write, no colour) and a colour pass with depth EQUAL. When
the colour pass skips a draw (its pipeline still builds, a stage has no module, a fragment did not translate), the pre-pass
has already claimed the pixels and whatever colour the target held would show there. Both passes bind the same per-object
block at mesh-stage c4 (or c5 when c4 is the large per-frame block). So a skipped colour draw notes that address
(`mesh_pair_note`), and pre-pass draws with a noted address from this or the last few frames are skipped too
(`mesh_pair_drop`). The object is absent and what lies behind it shows.

## Presentation

### Present thread

`runtime/vk_backend.c`. The game texture is resolved on the calling thread and its blit is handed to a present thread, so
`nvnQueuePresentTexture` returns at once. Queue order does the rest: the submissions that wrote the texture are already on
the queue. Every call that may write or free GPU state the blit reads (submit, finish, pool and texture teardown, texel
writes) first waits until the queued blit was submitted (`vk_present_drain`). The present thread sends the host present once
the blit is submitted (`present_hold`). Pacing is left to `nvnWindowAcquireTexture`, because holding the present to the
display time would keep every drain waiting and make the game's frame-rate control count the wait as work.

### Present pacing

`runtime/nvn_api_obj.c`. The vsync thread ticks at a fixed 60 Hz whatever the monitor does, so one game frame lasts
`interval` ticks. `pace` gives the present slot: the previous slot plus one period.
- Early: wait for the slot.
- Late by less than a period: present at once and keep the phase.
- Behind by a period or more: restart from now, with no catch-up burst.

`nvnWindowAcquireTexture` blocks (not the present) until the image's previous present plus one period has passed, since the
game often presents with a lock held; that is the time the next frame goes to the display. The acquire sync means "the
display no longer reads this image", which on the Switch is true once the image presented after it is on screen. So it
carries the fence of the work submitted before the latest present (`nvnSyncWait` and `nvnQueueWaitSync` wait for it,
`nvn_api_core.c` `acq_wait`).

`nvnQueuePresentTexture` queues the copy and returns at once. The display time comes from the pacer: the back end's present
thread holds the host present until then (back-edge pacing, as the Switch's display flips at vblank), so the present cadence
does not follow the replay time, and one period later the image is free for the next acquire. The added latency is the slack
of the frame (period minus its work), as on the Switch.

### DLSS

`runtime/vk_exec_dlss.c`, `runtime/vk_dlss.c`. The game has no motion vectors and no jitter.
- Camera motion is derived from the scene depth reprojected through this and the previous frame's camera (vertex cbuf 3).
- Colour is the game's tonemapped, sqrt-coded input to its own upscaler (LDR). DLSS's output is squared into the upscaler's
  target as the game's network does.
- Link's skinned pose is not tracked, so DLSS keeps the camera motion for it.
- Jitter is a Halton(2,3) shift over 72 phases applied as a projection jitter. The camera block of a draw (cbuf 3 with the
  scene camera layout) is bound as a patched copy, because a viewport shift would not move screen positions the shader
  derives from its own clip position.
- The 4K path runs DLSS at the window size and a composite replaces FSR. The game frame minus a reference (the game's upscale
  target right after DLSS's image was box-filtered into it) is what the game drew over the scene, and it is added to the 4K
  scene.
- Anything unexpected leaves the stand-in upscaler, with the reason logged when it changes.

### DLSS bias mask

`runtime/shaders/dlss_mask.comp` produces DLSS's BiasCurrentColor mask (1 = trust the current colour). The motion vectors are
the camera's only, so whatever moves on its own (actors, swaying trees) would leave history where it was. Two tests
(`params.mode` bit 0 and bit 1) give the mask; it is the larger of the two.
- Depth: each pixel's depth is reprojected into the previous frame as the motion pass does. Where last frame's depth there
  disagrees with the expected depth by more than `params.threshold` of the distance, the history belongs to something else.
  The relative error comes from NDC depth being linear in 1/w: B = dz/d(1/w) measured along the pixel's own ray, so no
  near/far plane is needed.
- Colour: last frame's DLSS input colour, fetched bilinearly at the reprojected place, is compared with the 3x3 box of this
  frame's colour around the pixel. Still detail stays inside the box and history of something moved falls outside it.
  Outside by `params.color_threshold` to 3x that ramps the mask from 0 to 1.

### Render resolution

`runtime/resolution.c` sets the size the game composes its frame in (its output: canvas, UI, swapchain) and the size of its
3D scene, above the shipped 1920x1080 and 1600x900. The game does the rendering itself; a patch makes it read both sizes
from the patch page. The `output_res` setting (`<w>x<h>`, default 1920x1080) sets the output and `render_scale` the scene
relative to it: "game" (default, 5/6, so 1600x900 at 1080p) or a percentage. Above 100 the game supersamples its scene. The
game's final pass (FSR 1 EASU) scales the scene to the output, and the present scales the output to the window
(`vk_fsr.c`). This game has no output, scene, aspect or fov patch sets: the game keeps its own sizes and only the heap growth
follows the settings.

Heap growth. agl's DynamicTextureAllocator is one fixed heap, "DynamicTextureAllocatorMem" in
`System/HeapSize/Product.Nin_NX_NVN.HeapDef.byml.zs` (113664 KiB, carved from "Graphics", 201728 KiB). The shipped size runs
out at the title screen at 2560x1440. Both heaps grow by the same amount: the shipped size times the output's pixel ratio
above 1920x1080 plus the scene's above 1600x900. `romfs.c` serves the rewritten table (`res_heapdef_build`) in place of the
original and `memory.c` adds the amount to the application pool, so no other heap shrinks. Without the rewritten table (a
RomFS that lacks it, an unknown format) the patch sets stay off.

### SDL window thread

`runtime/gfx_window_sdl.c`. SDL needs its video calls on the thread that initialised video, so one thread owns the window and
runs the event loop. Other threads hand it work with `run_on_window` (a user event and a wait). Video is initialised before
the window exists, because the Vulkan instance asks SDL for its surface extensions first. The `window_size` setting is in
pixels. On displays that scale windows (Wayland, HiDPI) the window and the mouse use the display's units, and `input_sdl.c`
and the menu scale back with `gfx_window_sdl_density`.

## Settings

### Settings file

`runtime/cfg.c` reads `oot.ini`, which gives defaults to the `OOT_*` environment variables. The environment wins over the
file. The format is `key = value` lines. A `#` or `;` at the start of a line begins a comment and `[sections]` only group. A
key is a variable name without `OOT_` (an `OOT_` prefix is accepted), case-insensitive. A value is the rest of the line,
trimmed, optionally in double quotes; an empty value counts as unset.

`cfg_set` changes a value in memory (readers see the new pointer; the old string is kept, so a pointer `cfg_get` handed out
stays valid). `cfg_save` writes only the changed keys back, keeping the rest of the file. Readers take no lock: an entry is
complete before `n_entries` counts it and value pointers are swapped atomically. A new variable is added to the `known` list
and, if the menu shows it, to the `metas` table, which holds the default and the accepted values.

Device profiles are defaults for a known machine, below the environment and `oot.ini` and above the table. Any key the
player sets wins, and resetting a key in the menu returns to the profile's value. The `device_profile` setting is `auto` (the Steam Deck when the firmware says so), `desktop` (none) or `deck`. The Deck
profile (LCD Jupiter, OLED Galileo; 1280x800, 16:10) runs the game at the screen's resolution with its own 5/6 scene. The
aspect patch gives the camera its 16:10 in `resolution.c`, and the `romfs-patches-1280x800` overlay the HUD's. It runs
fullscreen with vsync as gamescope composes Game Mode, at 40 fps, which divides 40 Hz and the OLED's 120 Hz.

## Test hooks

### Test state file

`runtime/nav_state.c`. With the `state_file` setting a thread writes one JSON line about ten times a second: frame, ms, the
main camera's eye and forward vector, and Link's position. The camera block is at a fixed address that holds only for the
stock heap layout (a changed heap layout, or `OOT_GFX_SCALE=0`, moves it). `nav_state_camera` replaces it with the scene
camera the DLSS tracker recognised by its contents. Link's position is found by behaviour: a static slot in main's data
holds a 0x32068-byte object with the position, a facing vector and a string object with the map name.

A test-only setting puts the player's life back every 20 ms. The player actor is found by its vtable (reachable from the slot
object, else by a scan of the guest's writable mappings every 30 s at most) and the one life word is written with
process_vm_writev.

### Flight recorder

`runtime/flight.c`, enabled by the `flight` setting. A ring keeps the last 600 presents: per present a record (interval,
render-thread blocked time per perf kind, event counts) and a 320x180 thumbnail blitted on the GPU by the present thread
(`gfx_present.c`). A dump goes to `<exe dir>/../flight/<YYYYMMDD-HHMMSS>-f<frame>/` as `frames.tsv` plus one `f<frame>.png`
per thumbnail, written by a background thread. Triggers are F8 while the game window has focus and a frame interval above
the `flight_spike_ms` setting (default 100, 0 is off, at most one dump per 10 s). A dump includes the frames up to
`FLIGHT_POST` presents after the trigger, so the aftermath is in it. Records come from `perf_present` (`perf.c`). The counts
are differences of the running totals of the `trace` setting's perf mode and of `vk_exec_frame`.

### GPU breadcrumbs and NaN guard

`runtime/vk_crumbs.c`: markers written with `VK_AMD_buffer_marker` at the top and bottom of the pipe into two words of a
host-coherent buffer. After a device loss the hang lies between the two values. A code holds the crumb kind in the top 4
bits and a number in the low 28.

`runtime/spirv_guard.c`: a SPIR-V pass for fragment shaders that makes each store to a float colour output
`clamp(v, -1e30, 1e30)` with NaN replaced by 0, so one NaN or Inf pixel cannot reach bloom and exposure sums.
