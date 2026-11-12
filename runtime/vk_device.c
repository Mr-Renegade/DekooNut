// Vulkan instance, physical device choice, logical device, VMA and the timeline semaphore.
// Ported from LAS Recomp, runtime/nvn/vulkan/device.c:33-330
// (https://github.com/peiikai/LASRecomp @ da35b83a9611d046a9a681aae668d7085954d37f, MIT, Copyright (c) 2026 Peiikai),
// without SDL. The VMA implementation is runtime/vk_vma.cpp (VMA is C++ only).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "vk.h"
#include "platform/platform.h"

static VkCtx g_vk;
static int g_ready;
static PlatMutex g_lock = PLAT_MUTEX_INIT;

static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++) if (!strcmp(e[i].extensionName, name)) return 1;
    return 0;
}

// Mesa's drivers (RADV on Linux and the Steam Deck) keep a shader cache of their own, by default in ~/.cache and up to
// 1 GB; unless MESA_SHADER_CACHE_DIR names a folder, it goes to mesa-cache beside the exe (the SD card is roomier).
static void mesa_cache_beside_exe(void) {
    char dir[PLAT_PATH_MAX * 2];
    if (plat_getenv("MESA_SHADER_CACHE_DIR", dir, sizeof dir)) return;
    int len = plat_exe_path(dir, sizeof dir - 16);
    char *cut = len ? plat_last_sep(dir) : NULL;
    if (!cut) return;
    strcpy(cut + 1, "mesa-cache");
    plat_setenv("MESA_SHADER_CACHE_DIR", dir);
}

// NVIDIA's driver cache (~/.cache/nvidia, shared, 1 GB by default) is too small for a game whose shaders were built
// ahead, so it gets a folder of its own beside the exe. __GL_SHADER_DISK_CACHE_PATH in the environment wins.
static void nvidia_cache_beside_exe(void) {
    char dir[PLAT_PATH_MAX * 2];
    if (plat_getenv("__GL_SHADER_DISK_CACHE_PATH", dir, sizeof dir)) return;
    int len = plat_exe_path(dir, sizeof dir - 16);
    char *cut = len ? plat_last_sep(dir) : NULL;
    if (!cut) return;
    strcpy(cut + 1, "nvidia-cache");
    plat_fs_mkdir(dir);
    plat_setenv("__GL_SHADER_DISK_CACHE_PATH", dir);
    plat_setenv("__GL_SHADER_DISK_CACHE", "1");
    plat_setenv("__GL_SHADER_DISK_CACHE_SIZE", "17179869184");
    plat_setenv("__GL_SHADER_DISK_CACHE_SKIP_CLEANUP", "1");
}

static int create_instance(void) {
    mesa_cache_beside_exe();
    nvidia_cache_beside_exe();
    const char *ext[8];
    uint32_t n_ext = (uint32_t)gfx_window_vk_extensions(ext, 7);
    vk_dlss_instance_ext(ext, &n_ext, 7);   // what NGX asks for (VK_KHR_get_physical_device_properties2)
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "TotK Recomp",
                             .pEngineName = "TotK Recomp", .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
                               .enabledExtensionCount = n_ext, .ppEnabledExtensionNames = ext,
                               .enabledLayerCount = 0};
    VkResult r = vkCreateInstance(&ci, NULL, &g_vk.instance);
    if (r != VK_SUCCESS) { rt_log("vk: vkCreateInstance failed (%d); is a Vulkan 1.3 driver installed?", (int)r); return 0; }
    return 1;
}

static const char *const k_required_ext[] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME,
    VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME};

// Name of the first missing required extension, or NULL.
static const char *missing_required(VkPhysicalDevice pd) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(pd, NULL, &n, NULL);
    VkExtensionProperties *e = calloc(n ? n : 1, sizeof *e);
    vkEnumerateDeviceExtensionProperties(pd, NULL, &n, e);
    const char *miss = NULL;
    for (size_t i = 0; i < sizeof k_required_ext / sizeof *k_required_ext && !miss; i++)
        if (!has_ext(e, n, k_required_ext[i])) miss = k_required_ext[i];
    free(e);
    return miss;
}

// Graphics+compute family that can present to the window.
static uint32_t find_gfx_family(VkPhysicalDevice pd, const VkQueueFamilyProperties *qp, uint32_t nq) {
    for (uint32_t i = 0; i < nq; i++)
        if ((qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            gfx_window_vk_presentable(g_vk.instance, pd, i))
            return i;
    return UINT32_MAX;
}

static int name_has(const char *s, const char *sub) {
    for (size_t n = strlen(sub); *s; s++)
        if (!strncasecmp(s, sub, n)) return 1;
    return 0;
}

// First discrete GPU with Vulkan 1.3, the required extensions and a graphics+present queue; OOT_GPU_INDEX (index) or
// OOT_GPU_NAME (case-insensitive part of the name) overrides.
static int pick_device(void) {
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(g_vk.instance, &n, NULL);
    if (!n) { rt_log("vk: no Vulkan device"); return 0; }
    VkPhysicalDevice *pd = calloc(n, sizeof *pd);
    vkEnumeratePhysicalDevices(g_vk.instance, &n, pd);
    const char *e = cfg_get("OOT_GPU_INDEX");   // no fixed default: the best device when unset
    int want = e && *e ? atoi(e) : -1;
    const char *name = cfg_get("OOT_GPU_NAME");
    int best = -1, best_score = -1;
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pd[i], &p);
        rt_log("vk: gpu %u: %s (type %d, api %u.%u.%u)", i, p.deviceName, (int)p.deviceType, VK_API_VERSION_MAJOR(p.apiVersion),
               VK_API_VERSION_MINOR(p.apiVersion), VK_API_VERSION_PATCH(p.apiVersion));
        if (want >= 0 && (int)i != want) continue;
        if (name && *name && !name_has(p.deviceName, name)) continue;
        if (p.apiVersion < VK_API_VERSION_1_3) { rt_log("vk: gpu %u skipped: Vulkan below 1.3", i); continue; }
        const char *miss = missing_required(pd[i]);
        if (miss) { rt_log("vk: gpu %u skipped: missing extension %s", i, miss); continue; }
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd[i], &nq, NULL);
        VkQueueFamilyProperties *qp = calloc(nq ? nq : 1, sizeof *qp);
        vkGetPhysicalDeviceQueueFamilyProperties(pd[i], &nq, qp);
        uint32_t fam = find_gfx_family(pd[i], qp, nq);
        free(qp);
        if (fam == UINT32_MAX) { rt_log("vk: gpu %u skipped: no graphics+compute+present queue family", i); continue; }
        int score = (int)i == want ? 100 : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 3
                  : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2 : 1;
        if (score > best_score) { best_score = score; best = (int)i; }
    }
    if (best < 0) {
        if (want >= 0) rt_log("vk: OOT_GPU_INDEX=%d is not usable (see above)", want);
        else if (name && *name) rt_log("vk: no usable gpu matches OOT_GPU_NAME=%s (see above)", name);
        else rt_log("vk: no device with Vulkan 1.3 and the required extensions (see above)");
        free(pd);
        return 0;
    }
    g_vk.phys = pd[best];
    g_vk.device_index = best;
    free(pd);
    vkGetPhysicalDeviceProperties(g_vk.phys, &g_vk.props);
    vkGetPhysicalDeviceMemoryProperties(g_vk.phys, &g_vk.mem_props);
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk.phys, &nq, NULL);
    VkQueueFamilyProperties *qp = calloc(nq ? nq : 1, sizeof *qp);
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk.phys, &nq, qp);
    g_vk.gfx_family = find_gfx_family(g_vk.phys, qp, nq);
    g_vk.xfer_family = g_vk.gfx_family;
    for (uint32_t i = 0; i < nq; i++)   // a family that does transfer only (async DMA engine)
        if ((qp[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && !(qp[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))) {
            g_vk.xfer_family = i;
            break;
        }
    free(qp);
    rt_log("vk: using gpu %d: %s", best, g_vk.props.deviceName);
    vk_mem_class_log(&g_vk.props, &g_vk.mem_props);
    return 1;
}

typedef struct {
    VkPhysicalDeviceFeatures2 f2;
    VkPhysicalDeviceVulkan11Features f11;
    VkPhysicalDeviceVulkan12Features f12;
    VkPhysicalDeviceVulkan13Features f13;
    VkPhysicalDeviceDepthClipControlFeaturesEXT clip;
    VkPhysicalDeviceCustomBorderColorFeaturesEXT cbc;
    VkPhysicalDeviceExtendedDynamicStateFeaturesEXT eds;
    VkPhysicalDeviceExtendedDynamicState2FeaturesEXT eds2;
    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT eds3;
    VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT vids;
    VkPhysicalDeviceConditionalRenderingFeaturesEXT cond;
    VkPhysicalDeviceTransformFeedbackFeaturesEXT xfb;
    VkPhysicalDeviceProvokingVertexFeaturesEXT pv;
    VkPhysicalDeviceRobustness2FeaturesEXT rob2;
    VkPhysicalDeviceFaultFeaturesEXT fault;
    VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT fsi;
    VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gpl;
    VkPhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT fbl;
    VkPhysicalDeviceDynamicRenderingLocalReadFeaturesKHR lr;
    VkPhysicalDeviceMeshShaderFeaturesEXT mesh;
} FeatureChain;

static int create_device(void) {
    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(g_vk.phys, NULL, &ne, NULL);
    VkExtensionProperties *ep = calloc(ne ? ne : 1, sizeof *ep);
    vkEnumerateDeviceExtensionProperties(g_vk.phys, NULL, &ne, ep);
    const char *ext[32];
    uint32_t n_ext = 0;
    for (size_t i = 0; i < sizeof k_required_ext / sizeof *k_required_ext; i++) ext[n_ext++] = k_required_ext[i];

    FeatureChain *c = calloc(1, sizeof *c);
    void **tail = &c->f2.pNext;
#define CHAIN(member, stype) do { c->member.sType = stype; *tail = &c->member; tail = &c->member.pNext; } while (0)
    c->f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    CHAIN(f11, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES);
    CHAIN(f12, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
    CHAIN(f13, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
    CHAIN(clip, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT);
#define OPTIONAL_EXT(flag, name, member, stype) do { \
        if (has_ext(ep, ne, name)) { g_vk.flag = 1; ext[n_ext++] = name; CHAIN(member, stype); } } while (0)
    OPTIONAL_EXT(has_custom_border_color, VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME, cbc, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT);
    OPTIONAL_EXT(has_eds, VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME, eds, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT);
    OPTIONAL_EXT(has_eds2, VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME, eds2, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT);
    OPTIONAL_EXT(has_eds3, VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME, eds3, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT);
    OPTIONAL_EXT(has_vertex_input_dynamic_state, VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME, vids, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT);
    OPTIONAL_EXT(has_conditional_rendering, VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME, cond, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT);
    OPTIONAL_EXT(has_transform_feedback, VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME, xfb, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT);
    OPTIONAL_EXT(has_provoking_vertex, VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME, pv, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT);
    OPTIONAL_EXT(has_robustness2, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME, rob2, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT);
    OPTIONAL_EXT(has_device_fault, VK_EXT_DEVICE_FAULT_EXTENSION_NAME, fault, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT);
    // The translator's SPIR-V declares fragment shader interlock whenever the host reports it (vk_pipeline.c spirv_fixup).
    OPTIONAL_EXT(has_fragment_shader_interlock, VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME, fsi,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT);
    // Feedback loops (vk.h VkCtx.feedback): the layout, and barriers inside dynamic rendering
    int has_fbl = 0, has_lr = 0;
    if (has_ext(ep, ne, VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME) &&
        has_ext(ep, ne, VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME)) {
        has_fbl = has_lr = 1;
        ext[n_ext++] = VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME;
        ext[n_ext++] = VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME;
        CHAIN(fbl, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_FEATURES_EXT);
        CHAIN(lr, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES_KHR);
    }
    if (has_ext(ep, ne, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) { g_vk.has_memory_budget = 1; ext[n_ext++] = VK_EXT_MEMORY_BUDGET_EXTENSION_NAME; }
    // GPU breadcrumbs for a lost device (vk_crumbs.c): AMD's drivers and RADV report it.
    if (has_ext(ep, ne, VK_AMD_BUFFER_MARKER_EXTENSION_NAME)) { g_vk.has_buffer_marker = 1; ext[n_ext++] = VK_AMD_BUFFER_MARKER_EXTENSION_NAME; }
    OPTIONAL_EXT(has_mesh_shader, VK_EXT_MESH_SHADER_EXTENSION_NAME, mesh, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT);
    // NGX's device extensions (VK_NVX_binary_import, VK_NVX_image_view_handle); before the graphics pipeline
    // library, whose two extensions must stay last (vk_caps_gpl may drop them again).
    vk_dlss_device_ext(ext, &n_ext, 30, ep, ne);
    // Graphics pipeline library: chained last so it can be dropped again when vk_caps_gpl declines it.
    void **gpl_link = NULL;
    if (has_ext(ep, ne, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME) && has_ext(ep, ne, VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME)) {
        ext[n_ext++] = VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME;
        ext[n_ext++] = VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME;
        gpl_link = tail;
        CHAIN(gpl, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT);
    }
    free(ep);

    // The driver fills the chain with what it supports; the same chain then enables all of it (robustBufferAccess
    // stays on: guest shaders index constant buffers freely and out-of-range reads must stay harmless).
    vkGetPhysicalDeviceFeatures2(g_vk.phys, &c->f2);
    c->mesh.multiviewMeshShader = VK_FALSE;
    c->mesh.primitiveFragmentShadingRateMeshShader = VK_FALSE;
    g_vk.has_mesh_shader = c->mesh.meshShader != 0;
    g_vk.has_task_shader = c->mesh.taskShader != 0;
    g_vk.has_mesh_queries = c->mesh.meshShaderQueries != 0;
    rt_log("vk: mesh shaders %d, task shaders %d, mesh queries %d",
           g_vk.has_mesh_shader, g_vk.has_task_shader, g_vk.has_mesh_queries);
    vk_caps_driver(g_vk.phys);
    vk_caps_features(&c->f2.features);
    if (gpl_link) {
        VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT gp = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &gp};
        vkGetPhysicalDeviceProperties2(g_vk.phys, &p2);
        if (!vk_caps_gpl(&c->gpl, &gp)) { *gpl_link = NULL; n_ext -= 2; }
    }
    const VkPhysicalDeviceFeatures *h = &c->f2.features;
    const char *miss = NULL;
    if (!c->f12.timelineSemaphore) miss = "timelineSemaphore";
    else if (!c->f13.dynamicRendering) miss = "dynamicRendering";
    else if (!c->f13.synchronization2) miss = "synchronization2";
    else if (!h->shaderStorageImageWriteWithoutFormat) miss = "shaderStorageImageWriteWithoutFormat";
    else if (!c->clip.depthClipControl) miss = "depthClipControl";
    else if (!h->shaderInt64) miss = "shaderInt64";
    else if (!h->samplerAnisotropy) miss = "samplerAnisotropy";
    else if (!h->independentBlend) miss = "independentBlend";
    else if (!h->geometryShader) miss = "geometryShader";
    else if (!h->textureCompressionBC) miss = "textureCompressionBC";
    if (miss) { rt_log("vk: required feature missing on %s: %s", g_vk.props.deviceName, miss); free(c); return 0; }
    if (!c->fault.deviceFault) g_vk.has_device_fault = 0;   // vkGetDeviceFaultInfoEXT needs the feature, not just the extension
    g_vk.has_buffer_device_address = c->f12.bufferDeviceAddress;
    g_vk.has_astc_ldr = h->textureCompressionASTC_LDR;
    {
        int layout_ok = has_fbl && has_lr && c->fbl.attachmentFeedbackLoopLayout && c->lr.dynamicRenderingLocalRead;
        g_vk.feedback = layout_ok ? VK_FEEDBACK_LAYOUT : VK_FEEDBACK_RESTART;
        rt_log("vk: feedback loops: %s (attachmentFeedbackLoopLayout %d, dynamicRenderingLocalRead %d)",
               g_vk.feedback == VK_FEEDBACK_LAYOUT ? "feedback layout + by-region barrier"
               : "pass restart + full barrier",
               has_fbl && c->fbl.attachmentFeedbackLoopLayout, has_lr && c->lr.dynamicRenderingLocalRead);
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi[2] = {
        {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = g_vk.gfx_family, .queueCount = 1,
         .pQueuePriorities = &prio}};
    uint32_t nqi = 1;
    if (g_vk.xfer_family != g_vk.gfx_family) {
        qi[1] = (VkDeviceQueueCreateInfo){.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                          .queueFamilyIndex = g_vk.xfer_family, .queueCount = 1, .pQueuePriorities = &prio};
        nqi = 2;
    }
    VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &c->f2, .queueCreateInfoCount = nqi,
                             .pQueueCreateInfos = qi, .enabledExtensionCount = n_ext, .ppEnabledExtensionNames = ext};
    VkResult r = vkCreateDevice(g_vk.phys, &di, NULL, &g_vk.device);
    free(c);
    if (r != VK_SUCCESS) { rt_log("vk: vkCreateDevice failed (%d)", (int)r); return 0; }
    vkGetDeviceQueue(g_vk.device, g_vk.gfx_family, 0, &g_vk.gfx_queue);
    if (g_vk.xfer_family != g_vk.gfx_family) vkGetDeviceQueue(g_vk.device, g_vk.xfer_family, 0, &g_vk.xfer_queue);
    else { g_vk.xfer_queue = g_vk.gfx_queue; g_vk.xfer_shared = 1; }

    VkPhysicalDeviceExternalMemoryHostPropertiesEXT emh = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &emh};
    vkGetPhysicalDeviceProperties2(g_vk.phys, &p2);
    g_vk.min_imported_host_alignment = (uint32_t)emh.minImportedHostPointerAlignment;
    vk_caps_init(&g_vk);
    vk_dlss_init(&g_vk);

    VmaAllocatorCreateInfo ai = {.physicalDevice = g_vk.phys, .device = g_vk.device, .instance = g_vk.instance,
                                 .vulkanApiVersion = VK_API_VERSION_1_3,
                                 .flags = (g_vk.has_buffer_device_address ? VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT : 0) |
                                          (g_vk.has_memory_budget ? VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT : 0),
                                 .pDeviceMemoryCallbacks = vk_headroom_callbacks()};
    if ((r = vmaCreateAllocator(&ai, &g_vk.vma)) != VK_SUCCESS) { rt_log("vk: vmaCreateAllocator failed (%d)", (int)r); return 0; }

    VkSemaphoreTypeCreateInfo st = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                    .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE, .initialValue = 0};
    VkSemaphoreCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &st};
    if ((r = vkCreateSemaphore(g_vk.device, &si, NULL, &g_vk.timeline)) != VK_SUCCESS) {
        rt_log("vk: timeline semaphore creation failed (%d)", (int)r);
        return 0;
    }
    vk_crumbs_init(&g_vk);
    return 1;
}

static void destroy_all(void) {
    gfx_window_release_vk();
    if (g_vk.device) vkDeviceWaitIdle(g_vk.device);
    if (g_vk.device) vk_clear_shutdown(g_vk.device);
    if (g_vk.device) vk_astc_release();
    if (g_vk.device) vk_crumbs_release();
    if (g_vk.timeline) vkDestroySemaphore(g_vk.device, g_vk.timeline, NULL);
    vk_headroom_stop();
    if (g_vk.vma) vmaDestroyAllocator(g_vk.vma);
    if (g_vk.device) vkDestroyDevice(g_vk.device, NULL);
    if (g_vk.instance) vkDestroyInstance(g_vk.instance, NULL);
    memset(&g_vk, 0, sizeof g_vk);
}

int vk_init(void) {
    plat_mutex_lock(&g_lock);
    if (!g_ready) {
        memset(&g_vk, 0, sizeof g_vk);
        if (create_instance() && pick_device() && create_device()) g_ready = 1, vk_mem_flat_setup(), vk_headroom_start();
        else destroy_all();
    }
    int ok = g_ready;
    plat_mutex_unlock(&g_lock);
    return ok;
}

void vk_shutdown(void) {
    plat_mutex_lock(&g_lock);
    if (g_ready) { destroy_all(); g_ready = 0; }
    plat_mutex_unlock(&g_lock);
}

VkCtx *vk_ctx(void) { return g_ready ? &g_vk : NULL; }

static const char *fault_address_type(VkDeviceFaultAddressTypeEXT t) {
    switch (t) {
    case VK_DEVICE_FAULT_ADDRESS_TYPE_NONE_EXT: return "none";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT: return "read invalid";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT: return "write invalid";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_EXT: return "execute invalid";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_UNKNOWN_EXT: return "instruction pointer (unknown)";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_INVALID_EXT: return "instruction pointer (invalid)";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_FAULT_EXT: return "instruction pointer (fault)";
    default: return "unknown type";
    }
}

// The query runs once per process. Threads that hit the loss at the same time wait on the lock until it is logged, so
// none of them reaches its rt_fatal and ends the process before the report is written.
void vk_device_fault_report(VkResult r) {
    static PlatMutex lock = PLAT_MUTEX_INIT;
    static int done;
    if (r != VK_ERROR_DEVICE_LOST) return;
    plat_mutex_lock(&lock);
    if (done) { plat_mutex_unlock(&lock); return; }
    done = 1;
    vk_crumbs_report();
    VkCtx *c = vk_ctx();
    if (c && c->vma) {   // how full each heap was: a fault under memory pressure reads differently
        VmaBudget b[VK_MAX_MEMORY_HEAPS];
        vmaGetHeapBudgets(c->vma, b);
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(c->phys, &mp);
        for (uint32_t i = 0; i < mp.memoryHeapCount; i++)
            rt_log("vk: device lost: heap %u usage %llu MiB of budget %llu MiB (VMA blocks %llu MiB)", i,
                   (unsigned long long)(b[i].usage >> 20), (unsigned long long)(b[i].budget >> 20),
                   (unsigned long long)(b[i].statistics.blockBytes >> 20));
    }
    PFN_vkGetDeviceFaultInfoEXT get =
        c && c->has_device_fault ? (PFN_vkGetDeviceFaultInfoEXT)vkGetDeviceProcAddr(c->device, "vkGetDeviceFaultInfoEXT")
                                 : NULL;
    if (!get) {
        rt_log("vk: device lost; no fault report (VK_EXT_device_fault %s)", c && c->has_device_fault ? "has no entry point"
                                                                                                    : "not enabled");
        plat_mutex_unlock(&lock);
        return;
    }
    VkDeviceFaultCountsEXT n = {.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
    VkResult q = get(c->device, &n, NULL);
    VkDeviceFaultAddressInfoEXT *ai = calloc(n.addressInfoCount + 1, sizeof *ai);
    VkDeviceFaultVendorInfoEXT *vi = calloc(n.vendorInfoCount + 1, sizeof *vi);
    VkDeviceFaultInfoEXT info = {.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT, .pAddressInfos = ai, .pVendorInfos = vi};
    uint64_t binary = n.vendorBinarySize;
    n.vendorBinarySize = 0;   // the vendor binary is not fetched; its size alone is logged
    if (q == VK_SUCCESS && ai && vi) q = get(c->device, &n, &info);
    if (q != VK_SUCCESS && q != VK_INCOMPLETE) {
        rt_log("vk: device lost; vkGetDeviceFaultInfoEXT failed (%d)", (int)q);
    } else {
        rt_log("vk: device fault: \"%s\" (%u address infos, %u vendor infos, vendor binary %llu bytes)", info.description,
               n.addressInfoCount, n.vendorInfoCount, (unsigned long long)binary);
        // The faulting access lies in the precision-aligned range around the reported address.
        for (uint32_t i = 0; i < n.addressInfoCount; i++)
            rt_log("vk: device fault address %u: %s at %#llx, precision %#llx", i, fault_address_type(ai[i].addressType),
                   (unsigned long long)ai[i].reportedAddress, (unsigned long long)ai[i].addressPrecision);
        for (uint32_t i = 0; i < n.vendorInfoCount; i++)
            rt_log("vk: device fault vendor info %u: \"%s\" code %#llx data %#llx", i, vi[i].description,
                   (unsigned long long)vi[i].vendorFaultCode, (unsigned long long)vi[i].vendorFaultData);
    }
    free(ai);
    free(vi);
    plat_mutex_unlock(&lock);
}
