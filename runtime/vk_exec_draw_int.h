// Internal to the draw replay: what vk_exec_draw.c shares with the files split off from it:
// vk_exec_bindless.c (the bindless set 2) and vk_exec_dlss.c (the upscaler's stand-in).
#pragma once
#include <stdint.h>

#include "nvn_obj.h"
#include "vk.h"
#include "vk_exec.h"

#define NVN_STAGE_VS 0
#define NVN_STAGE_FS 1
#define NVN_STAGE_GS 2
#define NVN_STAGE_CS 5

#define MAX_BINDINGS 128
#define MAX_DESC 1024
#define SUPPORT_BYTES 1440u   // support buffer (std140): alpha_test, is_bgra[8], viewport_inverse, viewport_size, ...

// ---- vk_exec_draw.c: stand-ins for what a binding names but the game did not bind
enum { DV_1D, DV_1D_ARRAY, DV_2D, DV_2D_ARRAY, DV_3D, DV_CUBE, DV_CUBE_ARRAY, DV_2D_MS, DV_2D_MS_ARRAY, DV_COUNT };
extern struct {
    int ready;
    VkBuffer zbuf;
    VmaAllocation zalloc;
    VkBufferView zview;
    VkSampler samp, samp_shadow;
    struct { VkImage image; VmaAllocation alloc; VkImageView view, sview; } img[DV_COUNT];
} Z;

// ---- vk_exec_draw.c: resources of one command buffer, freed by its completion callback
typedef struct { VkBuffer buf; VmaAllocation alloc; uint8_t *map; uint64_t size, used; } Chunk;
typedef struct { uint64_t addr, size; } Range;
typedef struct {
    Chunk *chunks;
    int nchunks, cchunks;
    VkDescriptorPool *pools;
    int npools, cpools;
    Range *written;
    int nwritten, cwritten;
    uint64_t fence;                      // the submission these belong to (vk_gpu_written_at)
    uint32_t *written_slot;              // note_written: open-addressed index of written, entry = index + 1, 0 empty
    uint32_t written_slots;              // a power of two
} CbRes;
CbRes *cb_res(VkExec *ex);
uint8_t *scratch(VkExec *ex, uint64_t size, VkBuffer *buf, VkDeviceSize *off);

// State of the command buffer being recorded that the Vulkan command buffer itself holds.
extern struct {
    VkPipeline pipe[2];                  // bound per bind point: 0 graphics, 1 compute
    VkPipeState st;                      // the state vk_pipeline_set_dynamic last recorded
    int st_valid;
    VkBuffer vb[17];                     // vertex buffers last bound
    VkDeviceSize vo[17], vsz[17], vstr[17];
    int vcls[17];                        // their pool class and pool record, for the census
    void *vpool[17];
    NVNbufferRange vin[16];              // the ranges they were resolved from, under vk_mem_gen vgen
    uint64_t vgen;
    int vb_valid;
    uint8_t support[SUPPORT_BYTES];      // last support buffer and where it lives
    float sup_in[3];                     // the support_fill arguments it was filled from
    VkBuffer sup_buf;
    VkDeviceSize sup_off;
    int sup_valid;
    uint8_t c0[4][VK_C0_BYTES];          // last c0 bank per stage (vertex, fragment, compute, geometry) and where it lives
    VkBuffer c0_buf[4];
    VkDeviceSize c0_off[4];
    int c0_valid[4];
    struct { p_bind_buf ssbo[16], ubo[16]; uint32_t dim[3], binst; Nvn2C0Layout lay; uint8_t has_lay;
             uint32_t roots[NVN2_ROOT_TABLES][NVN2_ROOT_WORDS]; } c0_in[4];   // what vk_c0_fill read for c0[k]
    VkPipelineLayout ds_layout;          // graphics layout the sets below were last pushed or bound with
    VkPipelineLayout bl_layout;          // graphics layout the bindless set 2 was last bound with in this command buffer
    VkDescriptorSet bl_set;              // and the copy of set 2 bound then (bindless_bind)
    VkDescriptorSet ds_bound[2];         // allocated set bound per set number (0: none or pushed since)
    VkDescriptorSet ds_last[2];          // last allocated and written set per set number, contents in DS
    VkDescriptorSetLayout ds_last_layout[2];
    int ds_pushed[2];                    // set number holds the push of DS.psig under ds_layout
    int res_valid;                       // the graphics sets bound hold what the inputs in RS give (bind_resources)
    float vjit[2];                       // the shift in the viewport last recorded
} B;

// The inputs of the last graphics bind_resources (bind_sig) and what its build counted for the traffic census, which a
// skipped build counts again. RS_BYTES bounds the bound-slot bytes a pipeline reads (RsList): per binding at most one
// c0 storage table, otherwise one bound range or handle per descriptor.
#define RS_BYTES (MAX_BINDINGS * sizeof(((VkExec *)0)->ssbo[0]) + MAX_DESC * sizeof(p_bind_buf))
typedef struct { int cls, use; uint64_t bytes; } RsCount;
extern struct {
    int on;                              // unused
    uint32_t hn;                         // words of h the last signature filled; 0: none
    uint64_t h[32];                      // generations, descriptor tables, feedback images, support inputs, base instance
    const VkPipeInfo *info;              // the pipeline whose RsList ranges body holds, as of bind op generation gen
    uint64_t gen;
    uint8_t body[RS_BYTES];
    int rec;                             // a graphics build is running: count_use records, scratch copies mark volatile
    int volatile_;                       // the build copied guest bytes into transient memory: a rebuild differs
    RsCount cnt[2 * VK_MCLS_N];
    int ncnt, cnt_over;
} RS;

// ---- vk_exec_draw.c: draws, uniform reads
extern int rf_mesh;     // record_fixed is recording a mesh-shader draw
extern int rf_proj;     // the draw being recorded gets a projection jitter
extern float rf_pj[2];
extern uint32_t ub_gen; // bumped by every inline uniform update (op_update_ub)
int upd_read(const VkExec *ex, uint64_t a, void *out, size_t n);
VkImageView get_view(VkExec *ex, const VkTexImage *img, const view_t *v, int hv, uint16_t rtype, int storage);
// Bytes readable by the CPU at addr: pools made by the runtime (GPU-only addresses) have no guest memory behind them.
uint64_t dyn_readable(uint64_t addr, uint64_t want);

// ---- vk_exec_bindless.c: the bindless set 2
void bl_image_gone(VkImage image);
void bindless_bind(VkExec *ex, const VkPipeInfo *info);
void bindless_bind_stats(uint64_t *binds, uint64_t *skips);   // since the last call
void bindless_ring_stats(uint64_t out[4]);

// ---- vk_exec_dlss.c: the upscaler's stand-in
void upscale_release(void);
void upscale_source_note(VkExec *ex, const VkPipeInfo *info);
void upscale_standin(VkExec *ex);
void dlss_image_gone(VkImage image);
void dlss_jitter_for(const VkExec *ex, float *jx, float *jy);   // the viewport shift
int dlss_proj_block(VkExec *ex, const p_bind_buf *b, VkDescriptorBufferInfo *out);
int dlss_read(const VkExec *ex, uint64_t a, void *out, size_t n);
void dlss_track_camera(VkExec *ex, int stage);   // the scene camera of a draw
int dlss_upscale(VkExec *ex);
