// Vulkan-internal header of the graphics layer. Host code only; the guest never sees it.
#pragma once
#include <vulkan/vulkan.h>
#include <stdint.h>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#include "vk_mem_alloc.h"
#pragma clang diagnostic pop
#include "nvn_obj.h"   // pool_t, tex_t, view_t, samp_par, NVNcopyRegion (memory and textures block)
#include "gfx.h"       // GfxShader, GfxShaderRes, GfxShaderAnswers (pipelines block)

// Required (device choice fails without them): Vulkan 1.3; VK_KHR_swapchain, VK_KHR_push_descriptor, VK_EXT_depth_clip_control,
// VK_EXT_external_memory_host and the core features vk_device.c checks. Optional ones set the has_* members.
typedef struct VkCtx {
    VkInstance instance;
    VkPhysicalDevice phys;
    VkDevice device;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties mem_props;
    uint32_t gfx_family;                    // graphics + compute + presentation on win32
    uint32_t xfer_family;                   // a transfer-only family when the GPU has one, else gfx_family
    VkQueue gfx_queue;                      // graphics + present
    VkQueue xfer_queue;                     // equals gfx_queue when xfer_shared
    int xfer_shared;                        // 1: no separate queue; serialise with gfx_queue
    VmaAllocator vma;
    VkSemaphore timeline;                   // one timeline semaphore; values are issued by vk_sync.c
    int has_custom_border_color, has_eds, has_eds2, has_eds3, has_vertex_input_dynamic_state,
        has_conditional_rendering, has_transform_feedback, has_provoking_vertex, has_device_fault, has_memory_budget,
        has_robustness2, has_buffer_device_address, has_fragment_shader_interlock, has_buffer_marker,
        has_mesh_shader, has_task_shader, has_mesh_queries,
        has_astc_ldr;                       // textureCompressionASTC_LDR (enabled when reported)
    // Feedback loops (vk_exec_draw.c feedback_prepare): VK_FEEDBACK_LAYOUT needs VK_EXT_attachment_feedback_loop_layout and
    // dynamicRenderingLocalRead; else VK_FEEDBACK_RESTART; VK_FEEDBACK_OFF samples in GENERAL without a barrier.
    int feedback;                           // VK_FEEDBACK_*
    uint32_t min_imported_host_alignment;   // VK_EXT_external_memory_host
    int device_index;                       // index in vkEnumeratePhysicalDevices
} VkCtx;
enum { VK_FEEDBACK_OFF, VK_FEEDBACK_RESTART, VK_FEEDBACK_LAYOUT };

// vk_caps.c: fallbacks for GPUs other than the RX 9070, decided at device creation.
typedef struct {
    int float64;                       // shaderFloat64 enabled; 0: shaders are translated with softfloat64
    uint32_t sbuf_align;               // nonzero: storage buffers bind at their address rounded down to this
    uint32_t sg_size;                  // requiredSubgroupSize on sg_pin_stages (32)
    VkShaderStageFlags sg_pin_stages;  // 0: no pin, stages run at the device's subgroupSize
    int gpl;                           // VK_EXT_graphics_pipeline_library enabled with fast linking
    int amd;                           // an AMD GPU (vendor 0x1002), any driver
    int amd_sc;                        // AMD's Windows driver branch for RDNA 1 and 2 ("AMD proprietary shader compiler")
} VkCaps;
extern VkCaps vk_caps;
void vk_caps_driver(VkPhysicalDevice p);             // first, before vkCreateDevice: vk_caps.amd, amd_sc
void vk_caps_features(VkPhysicalDeviceFeatures *f);   // before vkCreateDevice
// Before vkCreateDevice, with the driver's graphics pipeline library feature and properties: sets vk_caps.gpl; 1 = enable them.
int vk_caps_gpl(const VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT *f,
                const VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT *p);
void vk_caps_init(VkCtx *c);                          // after it: pretend limits, fallbacks, the boot report

// vk_dlss.c: NVIDIA DLSS Super Resolution in place of the game's upscale.
typedef struct { VkImage image; VkImageView view; VkFormat format; uint32_t w, h; VkImageAspectFlags aspect; } VkDlssImage;
int vk_dlss_on(void);                       // NGX is up and has SuperSampling
void vk_dlss_instance_ext(const char **ext, uint32_t *n, uint32_t cap);   // before vkCreateInstance
void vk_dlss_device_ext(const char **ext, uint32_t *n, uint32_t cap, const VkExtensionProperties *ep, uint32_t ne);
void vk_dlss_init(VkCtx *c);                // after the device exists
int vk_dlss_evaluate(VkCommandBuffer cb, const VkDlssImage *color, const VkDlssImage *depth, const VkDlssImage *motion,
                     const VkDlssImage *bias, const VkDlssImage *out, const float *jitter, int reset, const char **why);
// bias: NULL or R8 mask; jitter: NULL or {x, y} in render pixels
// The token of the upscale whose 4K image goes with the game frame presented next (0: none), taken in present order.
uint64_t vk_dlss4k_take(void);
int vk_dlss4k_composite(VkCommandBuffer cb, unsigned slot, uint64_t token, VkImage src, VkFormat fmt, int sw, int sh,
                        const int crop[4], VkImage dst, const int out[4]);
void vk_present_extent(uint32_t *w, uint32_t *h);   // gfx_present.c: the swapchain's size (0 before the first)
void vk_present_dlss4k(uint64_t token);              // gfx_present.c: the token of the next vk_present_blit

VkFormat vk_unorm_of_srgb(VkFormat f);      // vk_texture.c: the UNORM sibling of an sRGB colour format, 0 for the rest

VkCtx *vk_ctx(void);                        // NULL until vk_init() succeeded
int vk_init(void);                          // idempotent; 1 on success, 0 on failure (reason logged with rt_log)
void vk_shutdown(void);
// Call with the result of a failed queue submit, present or wait: VK_ERROR_DEVICE_LOST logs the VK_EXT_device_fault report
// once per process, after the breadcrumbs (vk_crumbs_report). Any other result returns at once.
void vk_device_fault_report(VkResult r);

// ---- GPU breadcrumbs (vk_crumbs.c): with VK_AMD_buffer_marker each crumb has the GPU write its code into host memory twice
// (command processor reached it, all work before it finished), which brackets the work that hung. Without it every call returns at once.
enum { VK_CRUMB_SUBMIT = 1, VK_CRUMB_SUBMIT_END, VK_CRUMB_UPLOAD, VK_CRUMB_UPLOAD_END, VK_CRUMB_ASTC, VK_CRUMB_PRESENT,
       VK_CRUMB_PRESENT_END, VK_CRUMB_WORK };
void vk_crumbs_init(VkCtx *c);       // after the allocator exists
void vk_crumbs_release(void);
// Records crumb kind with number n (low 28 bits kept) into cb. Only the thread that records cb may call it (vk_rec.h).
void vk_crumb(VkCommandBuffer cb, int kind, uint64_t n);
void vk_crumbs_report(void);         // logs the two codes and what they name

// ---- window and present (gfx_present.c). Host threads only.
// vk_present_image: blits the crop rectangle {x, y, w, h} of src (NULL, or w/h <= 0: the whole w x h image), scaled to the
// swapchain extent, and presents. src is in TRANSFER_SRC_OPTIMAL (the caller's barrier) or GENERAL (vk_texture.c images).
// Its writes must be submitted to gfx_queue first. Starts the window on first use; a minimised window or lost swapchain skips the frame.
void vk_present_image(VkImage src, VkImageLayout layout, VkFormat fmt, int w, int h, const int crop[4]);
// vk_present_image split in two: vk_present_blit submits the blit and leaves the present waiting, vk_present_flip
// queues it. The next blit or a swapchain rebuild queues a waiting present first.
void vk_present_blit(VkImage src, VkImageLayout layout, VkFormat fmt, int w, int h, const int crop[4]);
// FSR 1 on the present (vk_fsr.c): records EASU (+ RCAS) of src's crop rectangle into the rectangle out of dst, a swapchain
// image in TRANSFER_DST; src is in GENERAL. 0 = nothing recorded (bilinear, no upscale needed, or no FSR): the caller blits.
int vk_fsr_record(VkCommandBuffer cb, unsigned slot, VkImage src, VkFormat fmt, int sw, int sh, const int crop[4],
                  VkImage dst, const int out[4]);
void vk_fsr_release(void);   // before the device goes away
// Colour adjustment on the present (vk_color.c): with OOT_COLOR_* away from their defaults, records a pass writing src's
// adjusted crop rectangle into an RGBA16F image in GENERAL and returns it (crop then {0, 0, w, h}); VK_NULL_HANDLE = nothing recorded.
VkImage vk_color_record(VkCommandBuffer cb, unsigned slot, VkImage src, VkFormat fmt, const int crop[4]);
void vk_color_release(void);   // before the device goes away
void vk_present_flip(void);
// Every submission to ctx->gfx_queue (and xfer_queue when shared) takes this lock; vk_present_image and the swapchain
// rebuild do. Recursive.
void vk_queue_lock(void);
void vk_queue_unlock(void);
// The rectangle {x, y, w, h} of a dw x dh swapchain image that a cw x ch picture is presented in (OOT_ASPECT); the rest is black.
void gfx_fit_rect(int cw, int ch, int dw, int dh, int out[4]);
// Probe copy of the last presented swapchain image: with on, presents read it back into a host buffer and wait for the GPU.
// *px stays valid until the next present. Returns 0 with no frame.
void vk_present_capture(int on);
int vk_present_captured(const uint8_t **px, int *w, int *h, VkFormat *fmt);

// The window system's part of device creation (gfx_window_win.c with Win32, gfx_window_sdl.c with SDL): the instance
// extensions a surface needs (their count, at most max), and whether a queue family of pd can present to it.
int gfx_window_vk_extensions(const char **out, int max);
int gfx_window_vk_presentable(VkInstance inst, VkPhysicalDevice pd, uint32_t family);
void gfx_window_resize(int client_w, int client_h);   // programmatic resize; returns after the swapchain is rebuilt
void gfx_vsync_bind_counter(volatile int32_t *counter);  // host-side stand-in for a bound event: incremented per tick
unsigned gfx_window_focus_messages(void);          // OE_MSG_FOCUS_STATE_CHANGED pushes made by the window thread
void gfx_window_release_vk(void);                  // vk_shutdown calls it first: destroys swapchain, surface, present objects

// ---- memory and textures (vk_memory.c, vk_texture.c)
// Guest or host threads; every call needs vk_init(). The first nine have the GfxBackend signatures of gfx.h (vk_backend.c builds gfx_vulkan_backend).
int  vk_pool_init(pool_t *p);              // CPU-visible: guest storage imported (VK_EXT_external_memory_host) as one
                                           // VkBuffer; physical: device-local memory; virtual: a 64 KiB page table
void vk_pool_fini(pool_t *p);              // drops images cut from the pool; waits for the GPU if it used the pool
int  vk_pool_map_virtual(pool_t *v, int n, const void *requests);   // nvn_mapping_t[n], checked by the front end
void vk_texture_register(const dpool_ref *pool, int id, const tex_t *t, const view_t *v);
void vk_sampler_register(const dpool_ref *pool, int id, const samp_par *s);
void vk_texture_release(const tex_t *t);   // TextureFinalize: the image stays for slots that still name its storage
void vk_read_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row, uint64_t img);
void vk_write_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, const uint8_t *src, uint64_t row, uint64_t img);
void vk_guest_written(uint64_t addr, uint64_t size);   // a host write write-watch may miss: textures there re-upload

// GPU address -> backing. avail: bytes from offset that stay contiguous in buffer. alloc/alloc_offset: the device-local memory
// behind physical and (through the page table) virtual pool addresses, NULL for imported pools. cls: VK_MCLS_*.
typedef struct {
    VkBuffer buffer;
    VkDeviceSize offset, avail;
    VmaAllocation alloc;
    VkDeviceSize alloc_offset;
    int cls;
    void *pool;   // the imported pool record behind the span (NULL for device memory), for VK_MEM_BOUND
} VkMemSpan;

// Passive snapshot for the serialized replay thread: takes the shared pool lock but does not resolve, use or pick a span or
// touch mirrors, shadows or pool use. Success describes a present normal backing buffer; avail is capped by the registered pool.
enum { VK_MEM_INSPECT_POOL_IMPORT = 1, VK_MEM_INSPECT_POOL_PHYSICAL, VK_MEM_INSPECT_POOL_VIRTUAL };
enum { VK_MEM_INSPECT_MIRROR_NONE, VK_MEM_INSPECT_MIRROR_ON, VK_MEM_INSPECT_MIRROR_REFUSED };
typedef struct {
    uint64_t offset, avail;
    uint32_t pool_kind, pool_flags;
    int32_t cls;                        // VK_MCLS_* before optional mirror picking
    uint8_t mirror_state, mirror_present, shadow_storage, buffer_present;
} VkMemInspect;
int vk_mem_inspect(uint64_t addr, uint64_t size, VkMemInspect *out);   // nonnull out: zeroed on any failure
void vk_ind_clamp_frame(uint64_t frame);   // vk_exec_draw.c: logs records the indirect clamp zeroed
VkDescriptorSetLayout vk_bindless_layout(void);   // set 2: the texture and sampler pools
VkDescriptorSetLayout vk_bindless_empty(void);

// Canonical descriptor bytes: imported storage uses its host address; device storage a unique physical lifetime and byte offset.
typedef struct { uint64_t storage, offset; } VkDescKey;
int vk_mem_descriptor_key(uint64_t addr, uint64_t size, VkDescKey *out);
void vk_texture_descriptors_retire(uint64_t addr, uint64_t size, uint64_t storage);
// Traffic census (vk_mem_census): bytes the replay binds or copies, by pool class and use. DEVICE is physical memory; SCRATCH the
// replay's transient copies. CPU_READ: bytes host code reads in place of the GPU; GPU_WRITE: what copies, fills, counter reports and uniform updates write.
enum { VK_MCLS_NONE, VK_MCLS_HOST_NOACC, VK_MCLS_HOST_UNC, VK_MCLS_HOST_CACHED, VK_MCLS_DEVICE, VK_MCLS_SCRATCH, VK_MCLS_N };
enum { VK_MUSE_VB, VK_MUSE_IB, VK_MUSE_UBO, VK_MUSE_SSBO, VK_MUSE_IND, VK_MUSE_COPY_SRC, VK_MUSE_GPU_WRITE,
       VK_MUSE_CPU_READ, VK_MUSE_N };
void vk_mem_count(int cls, int use, uint64_t bytes);
// OOT_TRACE=perf (vk_memory.c): the replay notes each imported pool it binds as a vertex, index, uniform, storage or indirect buffer.
void vk_mem_bound_note(void *pool);
#define VK_MEM_BOUND(sp) do { if ((rt_trace & RT_TRACE_PERF) && (sp).pool) vk_mem_bound_note((sp).pool); } while (0)
// VRAM mirror of buffer-bound pools (vk_memory.c); vk_mirror_on is set at device selection for a discrete GPU. VK_MEM_USE creates a
// pool's mirror on first use, VK_MEM_PICK takes an existing one, vk_mem_gpu_wrote queues a mirror's GPU-written range for write-back
// at the end of the submission. VRAM mirror and write-back: see README, "VRAM mirror and write-back".
extern int vk_mirror_on;
void vk_mem_use(VkMemSpan *sp);
void vk_mem_pick(VkMemSpan *sp);
#define VK_MEM_USE(sp) do { if (vk_mirror_on && (sp).pool) vk_mem_use(&(sp)); } while (0)
#define VK_MEM_PICK(sp) do { if (vk_mirror_on && (sp).pool) vk_mem_pick(&(sp)); } while (0)
void vk_mem_gpu_wrote(const VkMemSpan *sp, uint64_t size);
// The sky SH block keeps its GPU bytes across page uploads; its sum rows are cleared in GPU order before the sky ambient sum dispatch.
int vk_sky_sh_keep(void);
int vk_sky_sh_window(uint64_t native);
void vk_sky_sh_block(uint64_t addr);   // the address the sum dispatch binds the block at
void vk_sky_sh_clear(VkCommandBuffer cb, int window);
// Uniform blocks as of the submit call (vk_usnap.c): vk_exec_record copies the ranges a submission binds before recording; ubo
// bindings take the copy (0: not copied), GPU writes of the replay drop the copies they touch.
struct VkExec;
void vk_usnap_begin(const struct VkExec *ex, const uint64_t *handles, int n, uint64_t fence);
void vk_usnap_end(uint64_t fence);
// Set by vk_exec.c's queue coroutines: the hook vk_usnap_begin calls after its own collection, and what it may add
extern void (*vk_usnap_extra)(void);
void vk_usnap_add_state(const struct VkExec *ex);
void vk_usnap_add_handles(const uint64_t *handles, int n);
void vk_usnap_add_stream(const void *op);
void vk_usnap_add_returns(const void *const *ret, int depth);
int vk_usnap_find(uint64_t addr, uint64_t n, uint64_t align, VkDescriptorBufferInfo *out);
int vk_usnap_read(uint64_t addr, uint64_t n, void *out);
void vk_usnap_drop(uint64_t addr, uint64_t n);   // the replay makes the GPU write [addr, addr+n) of guest memory
uint64_t vk_usnap_gen(void);                      // changes whenever a binding may resolve differently
extern uint64_t vk_frame_no;
void vk_usnap_shutdown(void);                     // vk_exec_draw_shutdown: frees the arenas
void vk_mem_mirror_begin(void);
int vk_mem_mirror_end(VkCommandBuffer cb, uint64_t fence);
int vk_mem_mirror_pending(void);
void vk_mem_mirror_prologue(VkCommandBuffer cb);
// Shadow storage (vk_memory.c): the replay notes every GPU write into CPU-visible memory at record time (vk_mem_gpu_pending);
// vk_gpu_written reports it at the fence and copies the bytes back into guest memory before telling the texture cache.
void vk_mem_shadow_sync(void);
void vk_mem_gpu_pending(uint64_t addr, uint64_t size);
void vk_gpu_written(uint64_t addr, uint64_t size);
// The same at the fence of submission `fence`: pages its mirror write-back covered whole keep a
// generation the mirror sync knows as its own echo (vk_memory.c).
void vk_gpu_written_at(uint64_t addr, uint64_t size, uint64_t fence);
int vk_mem_mirror_probe(uint64_t addr);   // the pool at addr gets its mirror as on a first buffer use; 1 when it has one
void vk_mem_class_log(const VkPhysicalDeviceProperties *p, const VkPhysicalDeviceMemoryProperties *m);   // once
int vk_mem_class_of(uint64_t addr);               // VK_MCLS_* of the pool at addr (NONE when unbacked)
void vk_mem_census(uint64_t frame);               // at each present: pool census and traffic, logged at frame 1800 and,
                                                  // with OOT_TRACE=perf, every 1800 frames
// VRAM in use and its budget, summed over the DEVICE_LOCAL heaps. exact 1: VK_EXT_memory_budget (the whole process),
// 0: this allocator's own allocations against the heap sizes. 0 when unknown.
int vk_mem_vram(uint64_t *used, uint64_t *budget, int *exact);
// VRAM headroom (vk_headroom.c): the VMA callbacks that see new device memory, and the thread that allocates the next block ahead of the render thread.
const VmaDeviceMemoryCallbacks *vk_headroom_callbacks(void);
void vk_headroom_start(void);
void vk_headroom_stop(void);   // before vmaDestroyAllocator
int vk_mem_resolve(uint64_t addr, uint64_t size, VkMemSpan *out);   // 1 when all of [addr, addr+size) is backed
// The same, but a CPU-visible pool whose shadow storage is not made yet stays unbacked instead of getting it.
int vk_mem_resolve_existing(uint64_t addr, uint64_t size, VkMemSpan *out);
// Changes whenever vk_mem_resolve plus VK_MEM_USE may give another span for an address: a pool added, finalized or
// mapped, a mirror created, refused or forgotten, a submission started.
uint64_t vk_mem_gen(void);
// Page write generations over guest memory (mem_written_pages plus vk_guest_written). Returns 1 when a page of the
// range was written after generation `since`, 0 when not, -1 when the range is not write-watched (treat as written);
// *now is the generation to pass next time.
int vk_mem_changed(uint64_t addr, uint64_t size, uint32_t since, uint32_t *now);
// The same from per-epoch snapshots of 2 MiB regions (no system call once queried in this epoch); writes made during the epoch are seen one epoch late.
int vk_mem_changed_snap(uint64_t addr, uint64_t size, uint32_t since, uint32_t epoch, uint32_t *now);
void vk_mem_snap_stats(uint64_t *checks, uint64_t *calls, uint64_t *untracked);   // since the last call

// Upload batch: one command buffer on gfx_queue that texture uploads and texel copies record into; begin/end bracket the
// recording (recursive lock). vk_staging returns a mapped host buffer that lives until the batch completes, or with alloc
// non-NULL until the caller's vmaDestroyBuffer after vk_upload_flush(1). The replay calls vk_upload_flush(0) before each
// submission that may read uploaded images. Returns the batch sequence number; wait=1 blocks until it completed.
VkCommandBuffer vk_upload_begin(void);
void vk_upload_end(void);
void *vk_staging(VkDeviceSize size, VkBuffer *buf, VmaAllocation *alloc);
uint64_t vk_upload_flush(int wait);
// Upload batch figures since the previous call (the perf line).
void vk_upload_perf(uint64_t *gpu_ns, uint64_t *timed, uint64_t *batches, uint64_t *wait_ns, uint64_t *bytes);
// A timestamp query pool of `pairs` pairs on the graphics queue, or VK_NULL_HANDLE without 64-bit timestamps.
VkQueryPool vk_ts_pool(uint32_t pairs, float *period);
void vk_exec_mesh_prewarm(void);   // noted mesh pipelines go to the mesh builders at boot (vk_exec_mesh.h)
void vk_mem_wb_tag(uint64_t prog, uint32_t bind);   // the producer of the next GPU write, for the write-back read trap (0: none)
// Deferred destruction: freed once the upload batches recorded so far and the timeline value last noted have completed.
void vk_defer_image(VkImage img, VmaAllocation alloc);   // alloc NULL: vkDestroyImage only
void vk_defer_view(VkImageView view);
void vk_exec_image_gone(VkImage image);   // destroy_tex, texture lock held: the replay's views of image go with it
void vk_defer_buffer(VkBuffer buf, VmaAllocation alloc);
// ---- ASTC (vk_astc.c)
enum { VK_ASTC_CPU, VK_ASTC_GPU, VK_ASTC_NATIVE, VK_ASTC_CHECK };
int vk_astc_mode(void);                 // decided once per device (logged); VK_ASTC_CPU before vk_init
int vk_astc_native(VkFormat astc_fmt);  // 1: images of this ASTC format are created as such and get the raw blocks
int vk_astc_gpu(void);                  // 1: ASTC images decoded to RGBA8 by the compute decoder (vk_astc_record)
int vk_astc_bc(void);                   // 1: those images are BC3 instead, encoded after the decode
// One image region decoded from raw blocks: tight rows of 16-byte blocks at byte src of the source buffer (slices one
// after the other), w x h texels and slices deep, written at offset of level / layer.
typedef struct { VkDeviceSize src; uint32_t w, h, slices, level, layer; VkOffset3D offset; } VkAstcRegion;
// Records into cb, the open upload batch: src (TRANSFER_SRC, src_size bytes of blocks) goes to a device-local scratch buffer,
// a dispatch per region decodes it and the texels are copied into img (GENERAL); the caller records its barriers around the call.
// readback (optional): also gets the decoded texels. bc: img is BC3 (TexVkFormat.astc_bc); the texels are encoded to BC3 blocks
// before the copy and readback is ignored. 0 when the decoder is unavailable or the scratch buffer cannot be allocated.
int vk_astc_record(VkCommandBuffer cb, VkBuffer src, VkDeviceSize src_size, const VkAstcRegion *rg, int n, uint32_t bw,
                   uint32_t bh, int srgb, int bc, VkImage img, VkImageAspectFlags aspect, VkBuffer readback);
void vk_astc_stats(uint64_t *dispatches, uint64_t *texels);   // since start
// GPU timestamps around each vk_astc_record's dispatches (logged per footprint and the largest one)
void vk_astc_time_enable(int on);
int vk_astc_time_get(uint32_t bw, uint32_t bh, uint64_t *texels, uint64_t *calls, double *us);   // totals so far
void vk_astc_time_summary(void);        // logs the totals
void vk_astc_release(void);             // vk_device.c, before the device goes: frees the decoder's objects
// Protect objects retired from here on until this timeline value completes; announce before recording their users (a
// destruction frontier, not a report that the value was submitted).
void vk_mem_protect_until(uint64_t timeline_value);
void vk_mem_collect(void);                         // frees what has completed
typedef struct { uint64_t imported_pools, physical_pools, virtual_pools, import_failures, upload_batches; } VkMemStats;
void vk_mem_stats(VkMemStats *s);
void vk_mem_shutdown(void);                        // before vk_shutdown: waits idle, frees every memory and texture object

// Textures: images are keyed by storage GPU address plus descriptor and stay in GENERAL. A mirrored texture (CPU-written pool
// storage) is uploaded on first use and again when its pages were written; once the GPU wrote it (vk_texture_gpu_written) the
// image is its content. A returned pointer stays valid until the next call that may drop textures.
typedef struct {
    VkImage image;
    VkFormat format;               // after the fallbacks of tex_vk_format (ASTC -> RGBA8, RGB -> RGBA)
    VkImageType type;
    VkImageViewType view_type;     // view of the whole image
    VkImageAspectFlags aspect;
    VkImageUsageFlags usage;
    uint32_t width, height, depth, levels, layers, samples;
    uint64_t addr;                 // storage GPU address
    uint8_t alpha_one;             // X format: sample alpha as 1 through the view swizzle
    uint8_t chan[4];               // TexVkFormat.chan: component order of a stand-in format, applied by the view swizzle
    uint8_t mirrored, gpu_owned;
    uint8_t bound;                 // aliases physical-pool memory (physical and virtual pools)
    uint8_t block_views;           // BC image rendered through uncompressed views of its blocks (only those may use COLOR_ATTACHMENT)
    uint8_t swz[4];               // NVN texture swizzle selectors (nvnTextureBuilderSetSwizzle); a view swizzle replaces them
    uint8_t fb_usage;              // created with ATTACHMENT_FEEDBACK_LOOP usage (VK_FEEDBACK_LAYOUT, attachable images)
    uint8_t fb_learned;            // a draw sampled it as a written attachment: its passes use the feedback-loop layout
} VkTexImage;
void vk_texture_describe_locked(const VkTexImage *img, char *buf, size_t cap);   // what an image is, for a log line
// vk_texture.c: ids of GPU-only textures due for the bindless set, and an id's image if it has one
extern volatile int vk_texture_bl_ndirty;
int vk_texture_bl_dirty_locked(uint32_t *ids, int cap);
struct dpool_ref;
const VkTexImage *vk_texture_get(const tex_t *t);   // created and made current; NULL for BUFFER or unmapped formats
// A texture slot; a slot whose texture was finalized follows the live texture at the same storage (hle_nvn.c).
const VkTexImage *vk_texture_by_id(const dpool_ref *pool, int id, view_t *view, int *has_view);
// The replay takes the texture lock once around its per-draw lookups (vk_texture_lock) and uses the _locked lookups;
// vk_texture_slots_gen changes whenever a slot may resolve to another image or view.
void vk_texture_lock(void);
void vk_texture_unlock(void);
const VkTexImage *vk_texture_by_id_locked(const dpool_ref *pool, int id, view_t *view, int *has_view);
VkSampler vk_sampler_by_id_locked(const dpool_ref *pool, int id);
// The slot's sampler with nearest filtering and no anisotropy (a depth texture read without a compare lands on one texel on the Switch). VK_NULL_HANDLE for an empty slot.
VkSampler vk_sampler_point_by_id_locked(const dpool_ref *pool, int id);
uint32_t vk_texture_slots_gen(void);
// What vk_texture_by_id(id) returns can change only when this does (slot registered again, or a texture entry at its storage
// address created, dropped, released or reused). Unlocked; read it before the lookup it guards. 0: never registered.
uint64_t vk_texture_slot_token(const dpool_ref *pool, int id);
uint32_t vk_sampler_gen(void);                      // changes whenever a sampler slot is registered
uint64_t vk_sampler_slot_token(const dpool_ref *pool, int id);   // what the slot resolves to; 0 not a registered sampler
void vk_texture_gpu_written(const tex_t *t);        // rendered or copied into by the GPU: no re-upload from storage
void vk_texture_storage_bound_locked(const VkTexImage *img);   // bound as a storage image in the recording submission
uint64_t vk_texture_last_write(const tex_t *t);    // timeline value of the newest submission recorded to write it; 0: none
size_t vk_texture_texel_span(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint64_t row, uint64_t img);   // bytes ReadTexels spans; 0 for a bad region
int vk_texture_small_dirty(void);                   // a small mirrored target holds GPU texels its storage lacks
void vk_texture_write_back_small(uint64_t fence);
// tex_prestage: at the end of a submission, small mirrored targets it rendered are copied into a host-visible ring, so a write-back after their fence need not wait for the GPU.
int vk_texture_prestage(VkCommandBuffer cb, uint64_t fence);   // writes those rendered by work up to fence back to storage
void vk_texture_debug(const VkTexImage *img, uint32_t *uploads, int *last_change, int *pages_written);   // vk_dump
int vk_texture_slot_swz(const dpool_ref *pool, int id, uint8_t swz[4]);
int vk_texture_read_image(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row,
                          uint64_t img);            // always from the image (GPU copy, waited for); 0 on failure
int vk_texture_cpu_stale(uint64_t addr, uint64_t size);    // a GPU-written mirrored texture there has stale storage
uint64_t vk_texture_dirty_fence(uint64_t addr, uint64_t size);   // newest recorded writer of such a texture; 0: none
void vk_texture_write_back(uint64_t addr, uint64_t size);   // writes those images back into their storage (waits)
void vk_texture_range_gone(uint64_t lo, uint64_t size, VmaAllocation alloc);   // vk_pool_fini: drop the images there
void vk_texture_stats(uint64_t *images, uint64_t *uploads, uint64_t *reuploads);
// Entries now, live and released (TextureFinalize, kept until sweep_released), with the bytes of their own memory
// (images not bound into pool memory).
typedef struct { uint64_t live, released, own_live, own_released, bytes_live, bytes_released; } VkTexCensus;
void vk_texture_census(VkTexCensus *s);
void vk_texture_shutdown(void);                     // vk_mem_shutdown calls it
void vk_texture_new_submit(void);                   // a queue submit starts: draw bindings check storage again
uint32_t vk_texture_epoch(void);                    // the write-watch snapshot epoch vk_texture_new_submit started
VkSampler vk_sampler_by_id(const dpool_ref *pool, int id);
VkSampler vk_sampler_for(const samp_par *s);        // cached by content
int vk_sampler_params(const dpool_ref *pool, int id, samp_par *out);

// 8-bit RGBA PNG with stored deflate blocks (boot.c); stops with rt_fatal when the file cannot be written.
void rt_write_png(const char *path, const uint8_t *rgba, uint32_t w, uint32_t h);

// ---- submission, synchronisation, completion (vk_sync.c, vk_backend.c)
// Fence values are ctx->timeline values: each submission signals the next one, in submit order. A value counts as done
// once the GPU reached it and the completion thread ran its side effects. Guest or host threads.
void vk_sync_start(void);                  // vk_init (fatal without a device) and the completion thread; idempotent
uint64_t vk_sync_submit(const uint64_t *handles, int n, uint64_t submitted_ticks); // immutable front-end entry tick
uint64_t vk_sync_reserve(void);            // asynchronous replay: the value the next queued submission signals
int vk_sync_fence_wait(uint64_t value, uint64_t timeout_ns);   // 1 when done; 0 on timeout or a value never submitted
uint64_t vk_sync_fence_done(void);         // highest done value
uint64_t vk_sync_issued(void);             // newest submitted value (the one being recorded is this + 1)
void vk_sync_finish(void);                 // every submitted value done
void vk_sync_shutdown(void);               // finish, stop the completion thread, free the command buffers
// The op-stream replay (vk_exec.c): records the commands of one submission into cb (begun and ended by the caller); fence is the
// value it will signal. Returns the commands recorded into the command buffer it holds at the end (0: stays empty, not submitted).
int vk_exec_record(VkCommandBuffer cb, const uint64_t *handles, int n, uint64_t fence, uint64_t submitted_ticks);
// Record internal GPU work through the submission and completion lifetime of NVN command handles. Synchronous; only the queue
// owner or an idle backend may call it, without simultaneous producers; never nest submission or replay inside the callback.
typedef int (*VkSyncRecordFn)(VkCommandBuffer cb, uint64_t fence, uint64_t submitted_ticks, void *context);
uint64_t vk_sync_submit_record(VkSyncRecordFn record, void *context, uint64_t submitted_ticks);
// Inside vk_exec_record, outside a render pass: ends and submits the command buffer recorded so far (without a signal) and
// returns a new begun one; the submission's timeline value is signalled by its last part.
VkCommandBuffer vk_sync_split(void);
// Submit the recorded prefix and wait for its dedicated fence, without advancing the full submission's timeline.
// Replay thread, outside a render pass. Returns a fresh recording command buffer; old resources keep their lifetime.
VkCommandBuffer vk_sync_split_wait(void);
// OOT_TRACE=perf: the replay notes each draw or dispatch of the part being timed (kind 0 draw, 1 indirect draw, 2 dispatch,
// 3 indirect dispatch; a, b: NVN 2 names of its vertex and fragment stages, or compute and 0; work: vertices x instances or groups,
// 0 when indirect). vk_sync_noting: 1 while the part is timed. vk_sync_note_args: the indirect draw just noted reads its arguments
// from this host address. vk_sync_note_write: a stage stores into [addr, addr+size) of guest memory.
int vk_sync_noting(void);
void vk_sync_note(int kind, uint64_t a, uint64_t b, uint64_t work);
void vk_sync_note_args(uint64_t addr);
void vk_sync_note_write(uint64_t addr, uint64_t size, uint64_t stage);
// Runs fn(arg) on the completion thread once fence is done on the GPU (event writes, counter reports, sync state).
// Callbacks run in the order they were added, after those of earlier values; a value already done runs at once there.
void vk_completion_add(uint64_t fence, void (*fn)(void *), void *arg);
// gfx_vulkan_backend (vk_backend.c): presents go to a present thread and run after the submissions made before them.
void vk_present_drain(void);               // returns once every queued present was submitted to gfx_queue
void vk_program_release(const prog_t *p);  // GfxBackend.program_release: vk_pipeline.c; weak no-op in vk_backend.c
void vk_backend_shutdown(void);            // presents, vk_sync_shutdown, vk_mem_shutdown, vk_shutdown in that order

// ---- pipelines (vk_pipeline.c)
// Binding model: one descriptor set, set 0, per pipeline: the support buffer at binding 0, then the vertex stage's bindings,
// then the fragment stage's. Pipelines and descriptor sets: see README, "Pipelines and descriptor sets".
// Everything a graphics pipeline depends on, as the op stream carries it; fields covered by dynamic state are left out of the pipeline key.
typedef struct VkPipeState {
    const GfxShader *vs, *fs;              // gfx_shader_get results; fs NULL: no fragment stage (depth only)
    int32_t color_fmt[8];                  // NVNformat of colour target i (OP_SET_RT), 0 none
    int32_t depth_fmt;                     // NVNformat of the depth/stencil target, 0 none
    uint8_t samples;                       // sample count of the targets; 0 or 1 single-sampled
    uint8_t prim;                          // NVN_DRAW_PRIMITIVE_* of the draw
    uint8_t depth_minus_w;                 // nvn_depth_minus_w(): clip-space depth is -w..w
    uint8_t depth_clamp, raster_discard;   // OP_DEPTH_CLAMP, OP_RASTER_DISCARD
    uint8_t y_flip;                        // viewports have negative heights (upper-left origin): the front face is not mirrored
    uint8_t prim_restart;                  // indices restart the strip or fan at the all-ones value (OP_PRIM_RESTART); dynamic state only
    uint8_t feedback;                      // the fragment shader samples an attachment in the feedback layout: 1 colour, 2 depth/stencil
    blend_st blend[8];                    // OP_BIND_STATE kind 0, indexed by blend_st.target
    color_st color;                        // kinds 1-5
    chmask_st chmask;
    ms_st ms;
    poly_st poly;
    ds_st ds;
    uint32_t vattr[16];                    // OP_BIND_VATTR: va_st words; attribute i feeds vertex shader Location i
    vs_st vstream[16];                     // OP_BIND_VSTREAM
    // Keep the legacy fields above byte-for-byte stable for pipeline-key records.
    const GfxShader *gs;                   // experimental ordinary GS; NULL keeps the existing VS/FS paths
    float alpha_ref;                       // OP_BIND_STATE kind 7: alpha test reference. After gs, so not in the pipeline-key records
} VkPipeState;
void vk_pipe_state_defaults(VkPipeState *s);   // NVN defaults of every state object (shaders and formats left 0)
enum { VK_PIPE_SUPPORT = GFX_RES_KINDS };      // VkPipeBinding.kind of the support buffer
#define VK_PIPE_ZERO_BINDING 16                // vertex binding of attribute-less shader inputs: a zeroed buffer of 16+ bytes, stride 0
typedef struct {
    uint16_t binding;                      // in its set
    uint8_t stage;                         // GFX_STAGE_* that declared it (support buffer: the first stage)
    uint8_t kind;                          // GFX_RES_* or VK_PIPE_SUPPORT
    VkDescriptorType type;
    uint16_t count;
    VkShaderStageFlags stages;
    const GfxShaderRes *res;               // the stage's reflection entry; NULL for the support buffer
    uint8_t set;                           // descriptor set: 0, or 1 for the fragment stage with independent layouts
} VkPipeBinding;
typedef struct {
    VkPipeline pipeline;                   // VK_NULL_HANDLE when creation failed
    VkPipelineLayout layout;
    VkDescriptorSetLayout set_layout;      // set 0
    int push;                              // 1: write set 0 with vkCmdPushDescriptorSetKHR; 0: allocate and bind a set
    VkPipelineBindPoint bind_point;
    uint32_t n_bindings;
    const VkPipeBinding *bindings;
    uint32_t vs_inputs;                    // Locations the vertex shader reads
    // Independent layouts (vk_caps.gpl): set 0 holds the vertex stage's bindings, set 1 the fragment stage's, ordered by set;
    // at most one is a push set. n_sets 1: everything in set 0.
    VkDescriptorSetLayout set_layout1;
    int push1;
    uint32_t n_sets;
    // Descriptor update template per set (vk_pipeline.c make_tpl): reads one VkDescSlot per descriptor, in binding order,
    // from the start of the set's descriptors. VK_NULL_HANDLE: the set is written with VkWriteDescriptorSet structs.
    VkDescriptorUpdateTemplate tpl[2];
} VkPipeInfo;
// One descriptor of the packed array a descriptor update template reads (VkPipeInfo.tpl), 24 bytes for every type; the
// bytes a type leaves unused are zero, so equal descriptors compare equal byte for byte.
typedef union { VkDescriptorImageInfo image; VkDescriptorBufferInfo buffer; VkBufferView view; uint64_t w[3]; } VkDescSlot;
// Return the cached pipeline (created on first use; failures are cached and logged once); valid until vk_pipeline_shutdown; thread-safe.
// Mesh pipeline raster and output state follows the ordinary graphics path; task/mesh modules and layout come from the caller.
VkPipeline vk_pipeline_mesh_create(const VkPipeState *s, VkPipelineLayout layout, VkShaderModule mesh,
                                   VkShaderModule task, uint32_t fragment_binding_offset);
// Process-local digest of the normalized mesh pipeline state; excludes dynamic state and unused vertex inputs.
int vk_pipeline_mesh_key(const VkPipeState *s, uint8_t key[32]);
void vk_mem_flat_choose(int capable);
int vk_mem_flat_mode(void);       // 2: the guest page table, 0: guarded candidates
void vk_mem_flat_setup(void);       // once the device is ready, before the first pool
uint64_t vk_mem_flat_table(void);  // its device address, 0 when not in use
int vk_mem_gmem_info(VkBuffer *buffer, VkDeviceSize *offset, VkDeviceSize *size);   // the raw pool table, up to date (vk_memory.c)
// The replay's variant: after vk_pipeline_async_enable a missing pipeline is built by background workers and this returns
// VK_NULL_HANDLE with *pending 1 until ready (the draw is skipped). wait_ms > 0 waits up to that long first. Pipelines built
// in a run are recorded to pipeline-keys.bin and prebuilt at the next boot.
VkPipeline vk_pipeline_graphics_async(const VkPipeState *s, const VkPipeInfo **info, int *pending, uint32_t wait_ms);
// rt_boot calls it unless --shot is given.
void vk_pipeline_async_enable(void);
int vk_pipeline_async_on(void);   // 1 after vk_pipeline_async_enable: graphics shaders are translated in the background too
// Boot prewarm (rt_boot): loads the shaders pipeline-keys.bin names, translating uncached ones from their stored inputs, and
// builds every recorded pipeline on all cores. No-op without vk_pipeline_async_enable.
void vk_pipeline_prewarm(void);
// Called once by vk_pipeline_prewarm when the recorded shaders are loaded and its pipeline jobs queued, before the builds
// (boot.c: the guest may start there). NULL: not called.
extern void (*vk_pipeline_prewarm_queued)(void);
VkPipeline vk_pipeline_compute(const GfxShader *cs, const VkPipeInfo **info);
// Records every dynamic state that VkPipeState determines, after binding a graphics pipeline (cull, front face, topology,
// depth and stencil, bias enable, discard, vertex input and, where dynamic, depth clamp, polygon mode, blend state, alpha
// to coverage). The replay sets viewports, scissors, bias, blend constants, stencil masks and references and the vertex
// buffers (strides NULL when vk_pipeline_vertex_input_dynamic()).
void vk_pipeline_set_dynamic(VkCommandBuffer cb, const VkPipeState *s);
int vk_pipeline_vertex_input_dynamic(void);    // 1: vertex input is VK_EXT_vertex_input_dynamic_state
// Translator answers from the bound state: attr_type from the vertex attribute formats (vs), frag_out_type from the
// colour target formats (fs). Other fields are left as they are; either pointer may be NULL.
void vk_pipeline_answers(const VkPipeState *s, GfxShaderAnswers *vs, GfxShaderAnswers *fs);
int vk_pipeline_tex_type(int nvn_target);      // NVN_TEXTURE_TARGET_* -> GFX_TEX_* (GfxShaderAnswers.tex_type)
uint32_t vk_pipeline_target_wmask(const VkPipeState *s, int i);   // colour write mask of target i the pipeline uses
VkPrimitiveTopology vk_pipeline_topology(int nvn_prim);   // quads, line loop: the replay rewrites those draws; polygon -> fan
uint8_t vk_geometry_input(uint32_t nvn_primitive);    // GFX_GEOMETRY_*; 0 for unsupported/emulated input primitives
VkFormat vk_vertex_format(int nvn_format, int *type);     // attribute format (supported fallback), GFX_TYPE_* in *type
PFN_vkCmdPushDescriptorSetKHR vk_pipeline_push_fn(void);  // after the first vk_pipeline_* call
// Descriptor update templates (after the first vk_pipeline_* call): NULL on a device below Vulkan 1.1 or, for the push variant,
// vkCmdPushDescriptorSetWithTemplateKHR is missing.
PFN_vkCmdPushDescriptorSetWithTemplateKHR vk_pipeline_push_tpl_fn(void);
PFN_vkUpdateDescriptorSetWithTemplate vk_pipeline_update_tpl_fn(void);
typedef struct { uint64_t graphics, compute, failures, hits, layouts, push_layouts, async_built, pending_skips; } VkPipeStats;
void vk_pipeline_stats(VkPipeStats *s);
void vk_pipeline_save(void);               // writes the VkPipelineCache to pipeline-cache.bin next to oot.exe
void vk_pipeline_autosave(void);           // vk_pipeline_save at most every 20 s, when pipelines were built since
void vk_pipeline_shutdown(void);           // waits for background builds, saves, destroys pipelines and cache; before vk_shutdown

// Fixed-function clear helper. Caller supplies a one-layer view and orders it with surrounding work.
void vk_clear_stencil_masked(VkCommandBuffer cb, VkFormat format, uint32_t samples, VkImageAspectFlags aspect,
                             VkImageView view, uint32_t width, uint32_t height, VkRect2D scissor,
                             uint32_t mask, uint32_t value);
void vk_clear_shutdown(VkDevice device);   // after device idle, before destroying the device

#include "vk_rec.h"   // last: its macros route the vkCmd* calls of every file that includes vk.h
