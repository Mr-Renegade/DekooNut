// Internal declarations shared by vk_pipeline.c and the files split from it: the SPIR-V opcodes and
// decorations the rewrites use, the device state P, and the functions one file calls in another.
#pragma once
#include <stdint.h>

#include "gfx.h"
#include "platform/platform.h"
#include "vk.h"

enum { SPV_OP_ENTRY_POINT = 15, SPV_OP_TYPE_INT = 21, SPV_OP_TYPE_FLOAT = 22, SPV_OP_TYPE_VECTOR = 23,
       SPV_OP_TYPE_POINTER = 32, SPV_OP_CONSTANT = 43, SPV_OP_CONSTANT_COMPOSITE = 44,
       SPV_OP_CONSTANT_NULL = 46, SPV_OP_FUNCTION = 54, SPV_OP_VARIABLE = 59,
       SPV_OP_DECORATE = 71, SPV_DEC_BUILTIN = 11, SPV_DEC_LOCATION = 30, SPV_DEC_INDEX = 32, SPV_DEC_BINDING = 33, SPV_DEC_DESCRIPTOR_SET = 34,
       SPV_SC_OUTPUT = 3 };

// vk_pipeline.c: device capabilities and the pipeline cache (set up on first use, under P.lock)
struct PipelineDev {
    PlatMutex lock;
    volatile int32_t ready;
    VkDevice device;
    VkPipelineCache cache;
    uint32_t max_push;
    int indep;                         // stage-independent layouts (vk_caps.gpl): vertex set 0, fragment set 1
    int gpl;                           // draws whose pipeline is building get a fast-linked one (vk_caps.gpl)
    struct { uint64_t links, wlinks, dlinks, requeued, link_fail, defer, lib_n[4], lib_fail, swaps, slow, pre_used[4]; double link_ms, link_max, lib_ms[4], lib_max[4], dlink_ms, dlink_max; } g;
    int dyn_vi, dyn_clamp, dyn_poly, dyn_blend_en, dyn_blend_eq, dyn_wmask, dyn_a2c;
    int f_logic_op, f_depth_clamp, f_fill_non_solid, f_dual_src;
    int f_geometry, f_vertex_stores, f_gs_point_size, f_clip, f_cull;
    int pv_last;                       // provokingVertexLast enabled: NVN's default is the last vertex
    VkShaderStageFlags sg32_stages;    // stages that accept requiredSubgroupSize = 32 (0: not available)
    VkShaderStageFlags subgroup_stages;
    VkSubgroupFeatureFlags subgroup_operations;
    VkDynamicState dyn[32];
    uint32_t n_dyn;
    PFN_vkCmdPushDescriptorSetKHR push;
    int tpl;                           // descriptor update templates on (make_tpl)
    PFN_vkCreateDescriptorUpdateTemplate tpl_create;
    PFN_vkDestroyDescriptorUpdateTemplate tpl_destroy;
    PFN_vkUpdateDescriptorSetWithTemplate tpl_update;
    PFN_vkCmdPushDescriptorSetWithTemplateKHR tpl_push;
    PFN_vkCmdSetVertexInputEXT set_vi;
    PFN_vkCmdSetDepthClampEnableEXT set_clamp;
    PFN_vkCmdSetPolygonModeEXT set_poly;
    PFN_vkCmdSetColorBlendEnableEXT set_blend_en;
    PFN_vkCmdSetColorBlendEquationEXT set_blend_eq;
    PFN_vkCmdSetColorWriteMaskEXT set_wmask;
    PFN_vkCmdSetAlphaToCoverageEnableEXT set_a2c;
    char cache_path[PLAT_PATH_MAX * 2];
    VkPipeStats stats;
};
extern struct PipelineDev P;
int ensure_init(void);   // 1 once the device capabilities and the cache are set up

// vk_pipeline_geometry.c: NULL when the pipeline's vertex, geometry and fragment interfaces link within the device
// limits, else why not
const char *geometry_validate(const VkPipeState *s, const VkPhysicalDeviceLimits *limits);

// vk_pipeline_spirv.c: the SPIR-V rewrites (each returns a new module the caller frees, or NULL)
uint32_t *spirv_rebind(const uint32_t *w, uint64_t n, uint32_t off);
uint32_t *spirv_set(const uint32_t *w, uint64_t n, uint32_t set);
uint32_t *spirv_dual_source(const uint32_t *w, uint64_t n, uint64_t *out_n);
uint32_t *spirv_alpha_test(const uint32_t *w, uint64_t n, unsigned func, float ref, uint64_t *out_n);
uint32_t *spirv_add_outputs(const uint32_t *w, uint64_t n, uint32_t missing, const GfxShaderIo io[32], uint64_t *out_n);
uint32_t *spirv_fixup_linked(const uint32_t *w, uint64_t n, uint64_t *out_n, int geometry_chain);
uint32_t *spirv_fixup(const uint32_t *w, uint64_t n, uint64_t *out_n);
