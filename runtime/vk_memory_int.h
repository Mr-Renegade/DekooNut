// Internal interface of vk_memory.c and the files split from it: the types and state they share.
#pragma once
#include <stdint.h>

#include "vk.h"
#include "platform/platform.h"

void vk_exec_draw_new_cb(void);   // vk_exec_draw.c: forget what the replay bound in its command buffer


// ---- vk_memory.c: pools, page write generations, the mirror
enum { POOL_IMPORT = 1, POOL_PHYS, POOL_VIRT };
enum { MS_NONE, MS_ON, MS_REFUSED };

typedef struct { struct PoolRec *phys; uint64_t off; } VPage;   // virtual page -> physical pool and offset

typedef struct PoolRec {
    const pool_t *key;
    uint64_t storage_serial;          // unique backing lifetime; virtual aliases use their physical record's serial
    int kind;
    uint32_t flags;
    uint64_t lo, size;                 // GPU address range of the pool
    uint64_t import_lo, import_hi;     // POOL_IMPORT: the imported host range (aligned outwards)
    VkBuffer buf;
    VkDeviceMemory mem;                // POOL_IMPORT
    VmaAllocation alloc;               // POOL_PHYS
    VPage *pt;                         // POOL_VIRT: size / 64 KiB entries
    uint64_t npages;
    volatile int32_t used;                // resolved at least once: the GPU may still read it at finalize
    uint8_t bufbound;                  // ever bound as a vertex, index, uniform, storage or indirect buffer
    // VRAM mirror (POOL_IMPORT, "VRAM mirror" below). mstate and mbuf change on the replay thread only.
    uint8_t mstate;                    // MS_NONE until the first buffer use, then MS_ON or MS_REFUSED
    VkBuffer mbuf;                     // MS_ON: device-local copy of [import_lo, import_hi), same offsets as buf
    VmaAllocation malloc_;
    uint32_t acc_sub;                  // replay: the last submission that synced the mirror (mirror_sync)
    uint32_t wb_sub;                   // replay: the last submission that wrote mirror bytes back to guest memory
    uint32_t mgen;                     // gen_lock: every page write stamped up to this generation is in the mirror
                                       // (or in a copy queued before its next use)
    uint64_t fl_lo, fl_hi;             // flush_lock: nvnMemoryPoolFlushMappedRange bytes not copied yet (fl_hi 0: none)
    // Shadow storage (POOL_IMPORT in shadow mode, "shadow storage" below): buf is a host-visible copy of the guest pages.
    VmaAllocation salloc;
    uint8_t *smap;                     // its mapping; offsets as in buf (from import_lo)
    uint32_t sgen;                     // gen_lock: every page write stamped up to this generation is in the shadow
    volatile uint8_t stouched;         // on the touched list: the GPU may use it in the submission being recorded
    uint8_t slazy;                     // shadow mode: the shadow is made at the first vk_mem_resolve (shadow_late)
    // Windowed shadow (a deferred shadow of a pool of 256 MiB or more, "windowed shadows" below): the pool
    // itself never gets buf; win[k] is a record of its own (no index entry) shadowing [import_lo + k*WIN_STEP, +2 steps).
    struct PoolRec **win;
    uint32_t nwin;
    // wb_dirty: a VRAM copy of what guest memory holds (made at the pool's first write-back, refreshed by every upload
    // and every write-back), and the device addresses of it, the mirror and the import. wbs_state 0 none, 1 made, -1 failed.
    VkBuffer wbs;
    VmaAllocation wbs_a;
    uint64_t wbs_addr, m_addr, g_addr;
    int8_t wbs_state;
} PoolRec;
typedef struct { PoolRec *r; uint64_t off, size; } MRange;
extern uint32_t msub;                  // replay: number of the submission being recorded
extern MRange *pend, *wbl;             // replay: copies for the next prologue; GPU writes into mirrors this submission
extern int npend, cpend, nwbl, cwbl;
VkDeviceAddress gmem_device(VkBuffer b);
uint32_t *gen_slot(uint64_t page_addr, int create);
void wbsp_add(uint64_t page);
void range_add(MRange **a, int *n, int *cap, PoolRec *r, uint64_t off, uint64_t size);
int range_cmp(const void *x, const void *y);
void mbarrier(VkCommandBuffer cb, VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds,
              VkAccessFlags2 da);

// ---- vk_writeback.c: the write-back rule, the read trap, wb_dirty, pend_wait
extern MRange *wbd_ends;
extern int wbd_nends, wbd_cends;
void wb_census(void);
int wbd_on(void);
void wbd_mirror_made(PoolRec *r);   // wb_dirty: the shadow is made with the mirror
void wbd_after_upload(VkCommandBuffer cb, const MRange *a, int n);
int wbd_record(VkCommandBuffer cb, MRange *a, int n);
void wbd_report(double frames);
void pw_submit(uint64_t fence);   // vk_mem_mirror_end, after the write-back copies
void pw_report(double frames);
int wb_record(VkCommandBuffer cb, uint64_t fence);

// ---- vk_upload.c: upload batches and deferred destruction
typedef struct { VkBuffer buf; VmaAllocation alloc; } Staging;
typedef struct {
    VkCommandBuffer cb;
    VkFence fence;
    uint64_t seq;
    Staging *stg;
    int nstg, cap;
    int ts, timed;   // its timestamp pair in up_pool (-1: none), and whether this use of it writes the pair
} Batch;

enum { DEAD_IMAGE = 1, DEAD_IMAGE_VMA, DEAD_VIEW, DEAD_BUFFER_VMA };
typedef struct { int kind; uint64_t handle; VmaAllocation alloc; uint64_t seq, tl; } Dead;

extern PlatRMutex cmd_lock;
extern VkCommandPool cmd_pool;
extern Batch *spare;                  // completed batches whose command buffer and fence are reused
extern int nspare, cap_spare;
extern uint64_t next_seq, done_seq;   // batches handed out / completed (all batches up to done_seq)
extern Dead *dead;
extern int ndead, cap_dead;
extern uint64_t tl_protected;         // newest submission being recorded or submitted; guards deferred destruction
extern VmaPool stg_pool;
extern int stg_pool_tried;
void grow(void **arr, int *cap, size_t elem);
void retire(VkCtx *c, int wait_all);
void collect_locked(VkCtx *c);
int dead_pending(void);
