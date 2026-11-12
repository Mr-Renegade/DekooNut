// Original masked stencil clear. Vulkan attachment clears ignore stencil write masks, so a fullscreen triangle
// replaces only the selected bits. The caller supplies a one-layer view, orders surrounding accesses, and forgets
// graphics bindings afterwards. No guest shader, game state or descriptor resources are used.
#include <stdlib.h>

#include "host.h"
#include "vk.h"
#include "stencil_clear_vert_spv.h"
#include "stencil_clear_frag_spv.h"

typedef struct ClearPipe {
    VkFormat format;
    VkSampleCountFlagBits samples;
    VkImageAspectFlags aspect;
    VkPipeline pipeline;
    struct ClearPipe *next;
} ClearPipe;

static VkPipelineLayout clear_layout;
static ClearPipe *clear_pipes;

static VkShaderModule clear_module(const uint32_t *words, size_t size) {
    VkShaderModuleCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                  .codeSize = size, .pCode = words};
    VkShaderModule module;
    VkResult rc = vkCreateShaderModule(vk_ctx()->device, &ci, NULL, &module);
    if (rc != VK_SUCCESS) rt_fatal("vk: stencil clear shader failed (%d)", rc);
    return module;
}

static VkPipeline clear_pipeline(VkFormat format, VkSampleCountFlagBits samples, VkImageAspectFlags aspect) {
    VkDevice device = vk_ctx()->device;
    for (ClearPipe *p = clear_pipes; p; p = p->next)
        if (p->format == format && p->samples == samples && p->aspect == aspect) return p->pipeline;
    if (!clear_layout) {
        VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        VkResult rc = vkCreatePipelineLayout(device, &ci, NULL, &clear_layout);
        if (rc != VK_SUCCESS) rt_fatal("vk: stencil clear layout failed (%d)", rc);
    }
    VkShaderModule vs = clear_module(stencil_clear_vert_spv, sizeof stencil_clear_vert_spv);
    VkShaderModule fs = clear_module(stencil_clear_frag_spv, sizeof stencil_clear_frag_spv);
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vs, .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = fs, .pName = "main"},
    };
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                               .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
                                                 .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                              .rasterizationSamples = samples};
    VkStencilOpState stencil = {.failOp = VK_STENCIL_OP_REPLACE, .passOp = VK_STENCIL_OP_REPLACE,
                                .depthFailOp = VK_STENCIL_OP_REPLACE, .compareOp = VK_COMPARE_OP_ALWAYS,
                                .compareMask = 0xff};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                                                .stencilTestEnable = VK_TRUE, .front = stencil, .back = stencil};
    VkPipelineColorBlendStateCreateInfo blend = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    VkDynamicState dynamics[] = {VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT, VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT,
                                 VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE};
    VkPipelineDynamicStateCreateInfo dyn = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                            .dynamicStateCount = sizeof dynamics / sizeof dynamics[0],
                                            .pDynamicStates = dynamics};
    VkPipelineRenderingCreateInfo render = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                                             .depthAttachmentFormat = aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? format : 0,
                                             .stencilAttachmentFormat = format};
    VkGraphicsPipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &render,
                                       .stageCount = 2, .pStages = stages, .pVertexInputState = &vi,
                                       .pInputAssemblyState = &ia, .pViewportState = &vp, .pRasterizationState = &rs,
                                       .pMultisampleState = &ms, .pDepthStencilState = &ds, .pColorBlendState = &blend,
                                       .pDynamicState = &dyn, .layout = clear_layout};
    ClearPipe *p = calloc(1, sizeof *p);
    if (!p) rt_fatal("vk: stencil clear pipeline allocation failed");
    VkResult rc = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &ci, NULL, &p->pipeline);
    vkDestroyShaderModule(device, fs, NULL);
    vkDestroyShaderModule(device, vs, NULL);
    if (rc != VK_SUCCESS) rt_fatal("vk: stencil clear pipeline failed (%d)", rc);
    p->format = format, p->samples = samples, p->aspect = aspect, p->next = clear_pipes;
    clear_pipes = p;
    return p->pipeline;
}

// The replay serializes this helper and releases it after waiting for the device at shutdown. A one-layer view
// lets every selected layer run without requiring a vertex Layer output or geometry shader support.
void vk_clear_stencil_masked(VkCommandBuffer cb, VkFormat format, uint32_t samples, VkImageAspectFlags aspect,
                             VkImageView view, uint32_t width, uint32_t height, VkRect2D scissor,
                             uint32_t mask, uint32_t value) {
    VkPipeline pipe = clear_pipeline(format, (VkSampleCountFlagBits)samples, aspect);
    VkRenderingAttachmentInfo att = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = view,
                                      .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                                      .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {width, height}},
                           .layerCount = 1, .pDepthAttachment = aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? &att : NULL,
                           .pStencilAttachment = &att};
    VkViewport vp = {0, 0, (float)width, (float)height, 0, 1};
    vkCmdBeginRendering(cb, &ri);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdSetViewportWithCount(cb, 1, &vp);
    vkCmdSetScissorWithCount(cb, 1, &scissor);
    vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, mask & 0xff);
    vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, value & 0xff);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
}

void vk_clear_shutdown(VkDevice device) {
    while (clear_pipes) {
        ClearPipe *p = clear_pipes;
        clear_pipes = p->next;
        vkDestroyPipeline(device, p->pipeline, NULL);
        free(p);
    }
    if (clear_layout) vkDestroyPipelineLayout(device, clear_layout, NULL);
    clear_layout = VK_NULL_HANDLE;
}
