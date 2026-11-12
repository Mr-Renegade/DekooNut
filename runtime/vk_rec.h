// The recorder thread (vk_rec.c): the vkCmd* calls the replay makes into its command buffer are made by a thread of
// their own. vk_sync.c brackets each part of a submission with rec_begin and rec_end (rec_end_cb ends the part: the
// recorder makes the last calls, vkEndCommandBuffer included). In between, each vkCmd* copies its arguments into a
// ring packet, in call order, and the recorder makes the real call. Calls into other command buffers are made at once;
// a call with no packet (a pNext chain) waits for the ring to empty and is made on the calling thread. rec_end returns
// once the recorder made every call. Every vkCmd* goes through the rc_* wrappers below (the macros at the end);
// extension commands called through function pointers have rec_* wrappers of their own.
#pragma once

void rec_begin(VkCommandBuffer cb);   // the submitting thread, sub_lock held
void rec_end(void);                   // waits for the recorder; no-op when rec_begin did not start it
// The end of part cb: the breadcrumb crumb (0: none, vk_crumbs.c), then vkEndCommandBuffer, both made by the recorder
// when it records cb, and the wait for it (1); otherwise rec_end and the breadcrumb, and the caller ends cb (0).
int rec_end_cb(VkCommandBuffer cb, int crumb, uint64_t n);
void rec_drain(void);                 // waits until the recorder made every call queued so far
void rec_draw_mesh(PFN_vkCmdDrawMeshTasksEXT fn, VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z);
void rec_draw_mesh_indirect(PFN_vkCmdDrawMeshTasksIndirectEXT fn, VkCommandBuffer cb, VkBuffer buffer,
                            VkDeviceSize offset, uint32_t count, uint32_t stride);
void rec_stats_log(void);             // packets, bytes, drains and waits since the last call

// Extension commands called through device function pointers.
void rec_crumb(VkCommandBuffer cb, int kind, uint64_t n);   // vk_crumb in order with this command buffer's packets
void rec_push_tpl(PFN_vkCmdPushDescriptorSetWithTemplateKHR fn, VkCommandBuffer cb, VkDescriptorUpdateTemplate tpl,
                  VkPipelineLayout layout, uint32_t set, const void *data, uint32_t bytes);   // data: bytes read by tpl
void rec_push(PFN_vkCmdPushDescriptorSetKHR fn, VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout layout,
              uint32_t set, uint32_t n, const VkWriteDescriptorSet *w);
void rec_vertex_input(PFN_vkCmdSetVertexInputEXT fn, VkCommandBuffer cb, uint32_t nb,
                      const VkVertexInputBindingDescription2EXT *b, uint32_t na,
                      const VkVertexInputAttributeDescription2EXT *a);
void rec_bool(PFN_vkCmdSetDepthClampEnableEXT fn, VkCommandBuffer cb, VkBool32 v);   // also alpha to coverage
void rec_polygon_mode(PFN_vkCmdSetPolygonModeEXT fn, VkCommandBuffer cb, VkPolygonMode m);
void rec_blend_enable(PFN_vkCmdSetColorBlendEnableEXT fn, VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBool32 *v);
void rec_blend_equation(PFN_vkCmdSetColorBlendEquationEXT fn, VkCommandBuffer cb, uint32_t first, uint32_t n,
                        const VkColorBlendEquationEXT *v);
void rec_write_mask(PFN_vkCmdSetColorWriteMaskEXT fn, VkCommandBuffer cb, uint32_t first, uint32_t n,
                    const VkColorComponentFlags *v);

#define REC_CMDS(X)                                                                                                       \
    X(BeginQuery, (VkCommandBuffer cb, VkQueryPool p, uint32_t q, VkQueryControlFlags f), (cb, p, q, f))                  \
    X(BeginRendering, (VkCommandBuffer cb, const VkRenderingInfo *ri), (cb, ri))                                          \
    X(BindDescriptorSets, (VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l, uint32_t first, uint32_t n,   \
                           const VkDescriptorSet *s, uint32_t nd, const uint32_t *d), (cb, bp, l, first, n, s, nd, d))   \
    X(BindIndexBuffer, (VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkIndexType t), (cb, b, o, t))                   \
    X(BindPipeline, (VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline p), (cb, bp, p))                             \
    X(BindVertexBuffers2, (VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBuffer *b, const VkDeviceSize *o,     \
                           const VkDeviceSize *sz, const VkDeviceSize *st), (cb, first, n, b, o, sz, st))                \
    X(BlitImage, (VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n,              \
                  const VkImageBlit *r, VkFilter f), (cb, s, sl, d, dl, n, r, f))                                        \
    X(ClearAttachments, (VkCommandBuffer cb, uint32_t na, const VkClearAttachment *a, uint32_t nr, const VkClearRect *r), \
      (cb, na, a, nr, r))                                                                                                 \
    X(ClearColorImage, (VkCommandBuffer cb, VkImage i, VkImageLayout l, const VkClearColorValue *c, uint32_t n,          \
                        const VkImageSubresourceRange *r), (cb, i, l, c, n, r))                                          \
    X(ClearDepthStencilImage, (VkCommandBuffer cb, VkImage i, VkImageLayout l, const VkClearDepthStencilValue *c,        \
                               uint32_t n, const VkImageSubresourceRange *r), (cb, i, l, c, n, r))                       \
    X(CopyBuffer, (VkCommandBuffer cb, VkBuffer s, VkBuffer d, uint32_t n, const VkBufferCopy *r), (cb, s, d, n, r))     \
    X(CopyBufferToImage, (VkCommandBuffer cb, VkBuffer s, VkImage d, VkImageLayout dl, uint32_t n,                       \
                          const VkBufferImageCopy *r), (cb, s, d, dl, n, r))                                             \
    X(CopyImage, (VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n,              \
                  const VkImageCopy *r), (cb, s, sl, d, dl, n, r))                                                       \
    X(CopyImageToBuffer, (VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkBuffer d, uint32_t n,                       \
                          const VkBufferImageCopy *r), (cb, s, sl, d, n, r))                                             \
    X(Dispatch, (VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z), (cb, x, y, z))                                 \
    X(DispatchIndirect, (VkCommandBuffer cb, VkBuffer b, VkDeviceSize o), (cb, b, o))                                    \
    X(Draw, (VkCommandBuffer cb, uint32_t n, uint32_t i, uint32_t f, uint32_t fi), (cb, n, i, f, fi))                    \
    X(DrawIndexed, (VkCommandBuffer cb, uint32_t n, uint32_t i, uint32_t f, int32_t v, uint32_t fi), (cb, n, i, f, v, fi)) \
    X(DrawIndexedIndirect, (VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t s), (cb, b, o, n, s))   \
    X(EndQuery, (VkCommandBuffer cb, VkQueryPool p, uint32_t q), (cb, p, q))                                             \
    X(EndRendering, (VkCommandBuffer cb), (cb))                                                                          \
    X(FillBuffer, (VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkDeviceSize n, uint32_t v), (cb, b, o, n, v))        \
    X(PipelineBarrier2, (VkCommandBuffer cb, const VkDependencyInfo *di), (cb, di))                                      \
    X(PushConstants, (VkCommandBuffer cb, VkPipelineLayout l, VkShaderStageFlags st, uint32_t o, uint32_t n,             \
                      const void *v), (cb, l, st, o, n, v))                                                              \
    X(ResetQueryPool, (VkCommandBuffer cb, VkQueryPool p, uint32_t f, uint32_t n), (cb, p, f, n))                        \
    X(ResolveImage, (VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n,           \
                     const VkImageResolve *r), (cb, s, sl, d, dl, n, r))                                                 \
    X(SetBlendConstants, (VkCommandBuffer cb, const float c[4]), (cb, c))                                                \
    X(SetCullMode, (VkCommandBuffer cb, VkCullModeFlags m), (cb, m))                                                     \
    X(SetDepthBias, (VkCommandBuffer cb, float c, float cl, float s), (cb, c, cl, s))                                    \
    X(SetDepthBiasEnable, (VkCommandBuffer cb, VkBool32 v), (cb, v))                                                     \
    X(SetDepthCompareOp, (VkCommandBuffer cb, VkCompareOp v), (cb, v))                                                   \
    X(SetDepthTestEnable, (VkCommandBuffer cb, VkBool32 v), (cb, v))                                                     \
    X(SetDepthWriteEnable, (VkCommandBuffer cb, VkBool32 v), (cb, v))                                                    \
    X(SetFrontFace, (VkCommandBuffer cb, VkFrontFace v), (cb, v))                                                        \
    X(SetPrimitiveRestartEnable, (VkCommandBuffer cb, VkBool32 v), (cb, v))                                              \
    X(SetPrimitiveTopology, (VkCommandBuffer cb, VkPrimitiveTopology v), (cb, v))                                        \
    X(SetRasterizerDiscardEnable, (VkCommandBuffer cb, VkBool32 v), (cb, v))                                             \
    X(SetScissorWithCount, (VkCommandBuffer cb, uint32_t n, const VkRect2D *r), (cb, n, r))                              \
    X(SetStencilCompareMask, (VkCommandBuffer cb, VkStencilFaceFlags f, uint32_t v), (cb, f, v))                         \
    X(SetStencilOp, (VkCommandBuffer cb, VkStencilFaceFlags f, VkStencilOp a, VkStencilOp b, VkStencilOp c,              \
                     VkCompareOp d), (cb, f, a, b, c, d))                                                                \
    X(SetStencilReference, (VkCommandBuffer cb, VkStencilFaceFlags f, uint32_t v), (cb, f, v))                           \
    X(SetStencilTestEnable, (VkCommandBuffer cb, VkBool32 v), (cb, v))                                                   \
    X(SetStencilWriteMask, (VkCommandBuffer cb, VkStencilFaceFlags f, uint32_t v), (cb, f, v))                           \
    X(SetViewportWithCount, (VkCommandBuffer cb, uint32_t n, const VkViewport *v), (cb, n, v))                           \
    X(WriteTimestamp, (VkCommandBuffer cb, VkPipelineStageFlagBits s, VkQueryPool p, uint32_t q), (cb, s, p, q))         \
    X(WriteTimestamp2, (VkCommandBuffer cb, VkPipelineStageFlags2 s, VkQueryPool p, uint32_t q), (cb, s, p, q))

#define REC_DECL(name, params, args) void rc_vkCmd##name params;
REC_CMDS(REC_DECL)
#undef REC_DECL

#ifndef VK_REC_IMPL
#define vkCmdBeginQuery rc_vkCmdBeginQuery
#define vkCmdBeginRendering rc_vkCmdBeginRendering
#define vkCmdBindDescriptorSets rc_vkCmdBindDescriptorSets
#define vkCmdBindIndexBuffer rc_vkCmdBindIndexBuffer
#define vkCmdBindPipeline rc_vkCmdBindPipeline
#define vkCmdBindVertexBuffers2 rc_vkCmdBindVertexBuffers2
#define vkCmdBlitImage rc_vkCmdBlitImage
#define vkCmdClearAttachments rc_vkCmdClearAttachments
#define vkCmdClearColorImage rc_vkCmdClearColorImage
#define vkCmdClearDepthStencilImage rc_vkCmdClearDepthStencilImage
#define vkCmdCopyBuffer rc_vkCmdCopyBuffer
#define vkCmdCopyBufferToImage rc_vkCmdCopyBufferToImage
#define vkCmdCopyImage rc_vkCmdCopyImage
#define vkCmdCopyImageToBuffer rc_vkCmdCopyImageToBuffer
#define vkCmdDispatch rc_vkCmdDispatch
#define vkCmdDispatchIndirect rc_vkCmdDispatchIndirect
#define vkCmdDraw rc_vkCmdDraw
#define vkCmdDrawIndexed rc_vkCmdDrawIndexed
#define vkCmdDrawIndexedIndirect rc_vkCmdDrawIndexedIndirect
#define vkCmdEndQuery rc_vkCmdEndQuery
#define vkCmdEndRendering rc_vkCmdEndRendering
#define vkCmdFillBuffer rc_vkCmdFillBuffer
#define vkCmdPipelineBarrier2 rc_vkCmdPipelineBarrier2
#define vkCmdPushConstants rc_vkCmdPushConstants
#define vkCmdResetQueryPool rc_vkCmdResetQueryPool
#define vkCmdResolveImage rc_vkCmdResolveImage
#define vkCmdSetBlendConstants rc_vkCmdSetBlendConstants
#define vkCmdSetCullMode rc_vkCmdSetCullMode
#define vkCmdSetDepthBias rc_vkCmdSetDepthBias
#define vkCmdSetDepthBiasEnable rc_vkCmdSetDepthBiasEnable
#define vkCmdSetDepthCompareOp rc_vkCmdSetDepthCompareOp
#define vkCmdSetDepthTestEnable rc_vkCmdSetDepthTestEnable
#define vkCmdSetDepthWriteEnable rc_vkCmdSetDepthWriteEnable
#define vkCmdSetFrontFace rc_vkCmdSetFrontFace
#define vkCmdSetPrimitiveRestartEnable rc_vkCmdSetPrimitiveRestartEnable
#define vkCmdSetPrimitiveTopology rc_vkCmdSetPrimitiveTopology
#define vkCmdSetRasterizerDiscardEnable rc_vkCmdSetRasterizerDiscardEnable
#define vkCmdSetScissorWithCount rc_vkCmdSetScissorWithCount
#define vkCmdSetStencilCompareMask rc_vkCmdSetStencilCompareMask
#define vkCmdSetStencilOp rc_vkCmdSetStencilOp
#define vkCmdSetStencilReference rc_vkCmdSetStencilReference
#define vkCmdSetStencilTestEnable rc_vkCmdSetStencilTestEnable
#define vkCmdSetStencilWriteMask rc_vkCmdSetStencilWriteMask
#define vkCmdSetViewportWithCount rc_vkCmdSetViewportWithCount
#define vkCmdWriteTimestamp rc_vkCmdWriteTimestamp
#define vkCmdWriteTimestamp2 rc_vkCmdWriteTimestamp2
#endif
