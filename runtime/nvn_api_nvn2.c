// NVN 2 procs the game asks for beyond NVN 1 (runtime/nvn_procs_nvn2.inc). Only calls the host can honour without a new
// NVN 2 structure layout are here; the rest stop with their name (nvn_procs.c).
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_obj.h"
#include "platform/platform.h"

// ---- Queues: one queue model; a queue's primary queue and ray-tracing memory change nothing.
NVN_API(nvnQueueBuilderSetPrimaryQueue) { NVN_LOG("QueueBuilderSetPrimaryQueue %p", arg_p(c, 1)); }
NVN_API(nvnQueueBuilderSetRayTracingMemorySize) { (void)c; }
NVN_API(nvnQueueBuilderGetRayTracingMemorySize) { ret_x(c, 0); }
NVN_API(nvnQueueGetTotalRayTracingMemoryUsed) { ret_x(c, 0); }

// ---- Device
// int nvnDeviceGetFlags(const NVNdevice *d): the flags the device was initialized with
NVN_API(nvnDeviceGetFlags) { ret_x(c, ((const dev_t_ *)arg_p(c, 0))->flags); }
// NVNboolean nvnDeviceRegisterFastClear{Color,Colori,Colorui}Index(NVNdevice *, const T *color, NVNformat, int index),
// ...Stencil(NVNdevice *, int value): like the NVN 1 forms (nvn_api_core.c), "not registered" keeps the normal clear
NVN_API(nvnDeviceRegisterFastClearColorIndex) { ret_x(c, 0); }
NVN_API(nvnDeviceRegisterFastClearColoriIndex) { ret_x(c, 0); }
NVN_API(nvnDeviceRegisterFastClearColoruiIndex) { ret_x(c, 0); }
NVN_API(nvnDeviceRegisterFastClearStencil) { ret_x(c, 0); }

// ---- Device message buffer (driver debug messages): the host driver writes none
NVN_API(nvnDeviceMessageBufferSetDefaults) { (void)c; }
NVN_API(nvnDeviceMessageBufferSetDevice) { (void)c; }
NVN_API(nvnDeviceMessageBufferSetStorage) { (void)c; }
NVN_API(nvnDeviceMessageBufferBeginMessages) { (void)c; }
NVN_API(nvnDeviceMessageBufferEndMessages) { (void)c; }
NVN_API(nvnDeviceMessageBufferReset) { (void)c; }
NVN_API(nvnDeviceMessageBufferGetNextMessage) { ret_x(c, 0); }

// ---- Multisample state: coverage truncation is a hint the Vulkan back end has no switch for
NVN_API(nvnMultisampleStateSetCoverageTruncationEnable) { (void)c; }
NVN_API(nvnMultisampleStateGetCoverageTruncationEnable) { ret_x(c, 0); }

// ---- Window colour space and HDR metadata: the host presents SDR sRGB
// void nvnWindowBuilderSetColorSpace(NVNwindowBuilder *, NVNcolorSpace), NVNcolorSpace ...GetColorSpace(...): 0, sRGB
NVN_API(nvnWindowBuilderSetColorSpace) { NVN_LOG("WindowBuilderSetColorSpace %d", (int)arg_x(c, 1)); }
NVN_API(nvnWindowBuilderGetColorSpace) { ret_x(c, 0); }
// void ...Set{Hdr}Display/ContentMetadata(obj, const metadata *): kept by no one
NVN_API(nvnWindowBuilderSetHdrDisplayMetadata) { (void)c; }
NVN_API(nvnWindowBuilderSetHdrContentMetadata) { (void)c; }
NVN_API(nvnWindowSetHdrDisplayMetadata) { (void)c; }
NVN_API(nvnWindowSetHdrContentMetadata) { (void)c; }

// nvnCommandBufferSetComputeOverlapMode: the host already inserts full barriers between dispatches and passes on one queue,
// so every overlap mode keeps that ordering.
NVN_API(nvnCommandBufferSetComputeOverlapMode) { (void)c; }

// ---- Command buffer calls new in NVN 2 that the Vulkan back end does not implement: accepted, recorded nowhere, the first
// call of each logged.
#define NVN2_NOT_RECORDED(n)                                                                                         \
    NVN_API(n) {                                                                                                     \
        static volatile int32_t logged;                                                                              \
        (void)c;                                                                                                     \
        if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED)) rt_log("nvn2: " #n " accepted, not recorded yet");   \
    }
NVN2_NOT_RECORDED(nvnCommandBufferSetRecordingMode)
NVN2_NOT_RECORDED(nvnCommandBufferBeginDeviceCommands)
NVN2_NOT_RECORDED(nvnCommandBufferEndDeviceCommands)
NVN2_NOT_RECORDED(nvnCommandBufferSubmitDeviceCommands)
NVN2_NOT_RECORDED(nvnCommandBufferBindAccelerationStructure)
NVN2_NOT_RECORDED(nvnCommandBufferBindAccelerationStructures)
NVN2_NOT_RECORDED(nvnCommandBufferBindAccelerationStructuresDeferred)
NVN2_NOT_RECORDED(nvnCommandBufferBuildAccelerationStructures)
NVN2_NOT_RECORDED(nvnCommandBufferCompactAccelerationStructure)
NVN2_NOT_RECORDED(nvnCommandBufferCopyAccelerationStructure)
NVN2_NOT_RECORDED(nvnCommandBufferRelocateAccelerationStructure)
NVN2_NOT_RECORDED(nvnCommandBufferDispatchRays)
NVN2_NOT_RECORDED(nvnCommandBufferDispatchRaysIndirect)
NVN2_NOT_RECORDED(nvnCommandBufferBindDeviceMessageBuffer)
NVN2_NOT_RECORDED(nvnCommandBufferBindDeviceMessageBufferDeferred)
NVN2_NOT_RECORDED(nvnCommandBufferResetDeviceMessageBuffer)
NVN2_NOT_RECORDED(nvnCommandBufferResetDeviceMessageBufferDeferred)
NVN2_NOT_RECORDED(nvnCommandBufferBindShadingRateImage)
NVN2_NOT_RECORDED(nvnCommandBufferSetShadingRatePaletteMode)
NVN2_NOT_RECORDED(nvnCommandBufferShadingRateImagePalette)
NVN2_NOT_RECORDED(nvnCommandBufferComputeDirtyTileMap)
// Direct and single indirect mesh draws are recorded in nvn_api_cmd.c; the other variants are accepted and do nothing.
#define NVN2_MESH_NOT_RECORDED(n) NVN_API(nvnCommandBuffer##n) { (void)c; }
NVN2_MESH_NOT_RECORDED(DrawMeshTasks3D)
NVN2_MESH_NOT_RECORDED(DrawMeshTasks3DIndirect)
NVN2_MESH_NOT_RECORDED(MultiDrawMeshTasks3DIndirect)
NVN2_MESH_NOT_RECORDED(MultiDrawMeshTasks3DIndirectCount)
NVN2_MESH_NOT_RECORDED(MultiDrawMeshTasksIndirect)
NVN2_MESH_NOT_RECORDED(MultiDrawMeshTasksIndirectCount)
NVN2_NOT_RECORDED(nvnCommandBufferEnableExclusiveScissor)
NVN2_NOT_RECORDED(nvnCommandBufferSetExclusiveScissor)
NVN2_NOT_RECORDED(nvnCommandBufferSetExclusiveScissors)
NVN2_NOT_RECORDED(nvnCommandBufferEvaluateDlss)
NVN2_NOT_RECORDED(nvnCommandBufferInsertDebugMarkerCompute)
NVN2_NOT_RECORDED(nvnCommandBufferInsertDebugMarkerGraphics)
NVN2_NOT_RECORDED(nvnCommandBufferPushDebugGroupCompute)
NVN2_NOT_RECORDED(nvnCommandBufferPushDebugGroupGraphics)
NVN2_NOT_RECORDED(nvnCommandBufferPopDebugGroupCompute)
NVN2_NOT_RECORDED(nvnCommandBufferPopDebugGroupGraphics)
// nvnCommandBufferPush{Graphics,Compute}Constants and their Deferred forms are recorded in nvn_api_cmd.c (OP_PUSH_CONST).
NVN2_NOT_RECORDED(nvnCommandBufferSetComputeEnable)
NVN2_NOT_RECORDED(nvnCommandBufferSetComputeEnableConditional)
NVN2_NOT_RECORDED(nvnCommandBufferSetCopyEnable)
NVN2_NOT_RECORDED(nvnCommandBufferSetCopyEnableConditional)
NVN2_NOT_RECORDED(nvnCommandBufferSetConservativeRasterMode)
NVN2_NOT_RECORDED(nvnCommandBufferSetRepresentativeFragmentTest)

// ---- Procs OoT 1.1.1 asks for in addition (SDK 22): same policy. Answering 0 means no primary queue, ray-tracing modules,
// DLSS network textures, cooperative vectors or dirty tile maps; a fast clear "not registered" keeps the normal clear.
#define NVN2_RET0(n)                                                                                               \
    NVN_API(n) {                                                                                                   \
        static volatile int32_t logged;                                                                            \
        ret_x(c, 0);                                                                                               \
        if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED)) rt_log("nvn2: " #n " answers 0");                  \
    }
NVN2_RET0(nvnDeviceRegisterFastClearDepthIndex)
NVN2_RET0(nvnDeviceRegisterFastClearStencilIndex)
NVN2_RET0(nvnQueueBuilderGetPrimaryQueue)
NVN2_RET0(nvnProgramBuilderGetRayTracingModuleCount)
NVN2_RET0(nvnProgramBuilderGetRayTracingModuleData)
NVN2_RET0(nvnDlssNetworkBuilderGetTexturePool)
NVN2_RET0(nvnDlssNetworkBuilderGetTextureIDs)
NVN2_RET0(nvnDeviceGetCooperativeVectorMatrixSize)
NVN2_RET0(nvnDeviceConvertCooperativeVectorMatrix)
NVN2_RET0(nvnDirtyTileMapParametersGetTexture)
NVN2_RET0(nvnDirtyTileMapParametersGetInputs)
NVN2_RET0(nvnDirtyTileMapParametersGetOutputs)
// uint64_t acquire timeout: the host waits for the swapchain image as long as it takes
NVN_API(nvnWindowBuilderGetAcquireTimeout) { ret_x(c, ~0ull); }
NVN_API(nvnWindowGetAcquireTimeout) { ret_x(c, ~0ull); }
// calls with no host effect
NVN2_NOT_RECORDED(nvnCommandBufferComputeDirtyTileMapEx)
NVN2_NOT_RECORDED(nvnCommandBufferConvertCooperativeVectorMatrix)
NVN2_NOT_RECORDED(nvnCommandBufferSetComputeWorkloadPriority)
NVN2_NOT_RECORDED(nvnCommandBufferSetGraphicsWorkloadPriority)
NVN2_NOT_RECORDED(nvnCommandBufferSetResourceOccupancy)
NVN2_NOT_RECORDED(nvnDirtyTileMapParametersSetConstantColorTable)
NVN2_NOT_RECORDED(nvnDirtyTileMapParametersSetDefaults)
NVN2_NOT_RECORDED(nvnDirtyTileMapParametersSetDepthStencilMask)
NVN2_NOT_RECORDED(nvnDirtyTileMapParametersSetExpandFootprint)
NVN2_NOT_RECORDED(nvnDirtyTileMapParametersSetInputs)
NVN2_NOT_RECORDED(nvnDirtyTileMapParametersSetOutputs)
NVN2_NOT_RECORDED(nvnDirtyTileMapParametersSetTexture)
NVN2_NOT_RECORDED(nvnDlssNetworkResize)
NVN2_NOT_RECORDED(nvnMultisampleStateGetSampleLocations)
NVN2_NOT_RECORDED(nvnSamplerPoolWriteDescriptorTableData)
NVN2_NOT_RECORDED(nvnTexturePoolWriteDescriptorTableData)
NVN2_NOT_RECORDED(nvnWindowBuilderSetAcquireTimeout)
NVN2_NOT_RECORDED(nvnWindowSetAcquireTimeout)
NVN2_RET0(nvnDirtyTileMapParametersGetConstantColorTable)
NVN2_RET0(nvnDirtyTileMapParametersGetDepthStencilMask)
NVN2_RET0(nvnDirtyTileMapParametersGetExpandFootprint)
NVN2_NOT_RECORDED(nvnCommandBufferSetFloatBlendClamp)
NVN2_RET0(nvnCommandBufferGetAccelerationStructureVisualizationData)
