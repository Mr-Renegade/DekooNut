// NVIDIA DLSS Super Resolution in place of the game's upscale (agl NDSR, the tensor dispatch the
// replay drops; vk_exec_draw.c dlss_upscale feeds it). Port of Lichii 65e979ad (zortos293/Lichii, "NVIDIA DLSS Super
// Resolution for Ocarina of Time", GPL-3.0). Lichii reaches DLSS through NVIDIA Streamline, which is
// Windows only. Here it is NGX directly: the DLSS SDK's Linux loader (libnvsdk_ngx.a, linked in; NVIDIA RTX SDK licence,
// headers and library stay outside the repository), the driver's NGX core (libnvidia-ngx.so.1) and the
// model library libnvidia-ngx-dlss.so.<version> from a dlss/ folder beside the executable, else the
// executable's own folder.
// On Windows the loader is nvsdk_ngx_s.lib, a Visual C++ library: vk_dlss_msvc_win.c gives it the dozen Visual C++
// runtime symbols this toolchain lacks. The NGX core is the driver's nvngx.dll, the model library nvngx_dlss.dll.
//
// Preset M for every mode: the mode follows the render/output ratio, LDR colour (no IsHDR, no auto exposure), no jitter,
// motion in render pixels at render size, depth not inverted.
// Anything that fails here (no NGX, no SuperSampling, a failed feature or evaluation) logs why and leaves the stand-in.
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#ifdef _WIN32
#include <windows.h>
#define DLSS_MODEL "nvngx_dlss.dll"
#else
#include <dirent.h>
#include <sys/stat.h>
#define DLSS_MODEL "libnvidia-ngx-dlss.so"
#endif

#include "cfg.h"
#include "host.h"
#include "platform/platform.h"
#include "vk.h"
#include "nvsdk_ngx_vk.h"
#include "nvsdk_ngx_helpers_vk.h"

#define DLSS_APP_ID 0x4f6f544e61ull   // any id: NGX wants one for a non-registered application

static int dlss_on, dlss_broken;
static int dlss_avail;   // an NVIDIA card and the model library: DLSS can run here, on or off by the setting
static NVSDK_NGX_Parameter *dlss_params;
// One feature per render and output size (the game's dynamic resolution has a few levels); none is released while the
// GPU may still use it, so they all live until the device goes. Switching features resets history.
enum { DLSS_FEATURES = 8 };
static struct { uint32_t rw, rh, ow, oh; unsigned preset; NVSDK_NGX_Handle *h; uint64_t used; } F[DLSS_FEATURES];
static int nfeat, last_feat = -1;
static uint64_t nevals;   // evaluations recorded so far; F[].used: the count at a feature's last use

int vk_dlss_on(void) { return dlss_on && !dlss_broken; }
int vk_dlss_available(void) { return dlss_avail; }

// The extensions NGX asks for, as far as the device has them and they are not on already. VK_EXT_buffer_device_address is
// left out: the Vulkan 1.2 bufferDeviceAddress feature is enabled, and the spec forbids both together.
void vk_dlss_device_ext(const char **ext, uint32_t *n, uint32_t cap, const VkExtensionProperties *ep, uint32_t ne) {
    unsigned ni = 0, nd = 0;
    const char **ie = NULL, **de = NULL;
    if (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_RequiredExtensions(&ni, &ie, &nd, &de))) return;
    for (unsigned i = 0; i < nd; i++) {
        if (!strcmp(de[i], "VK_EXT_buffer_device_address")) continue;
        int have = 0, on = 0;
        for (uint32_t k = 0; k < ne; k++) have |= !strcmp(ep[k].extensionName, de[i]);
        for (uint32_t k = 0; k < *n; k++) on |= !strcmp(ext[k], de[i]);
        if (!have) { rt_log("dlss: the device lacks %s", de[i]); continue; }
        if (!on && *n < cap) ext[(*n)++] = de[i];
    }
}
void vk_dlss_instance_ext(const char **ext, uint32_t *n, uint32_t cap) {
    unsigned ni = 0, nd = 0;
    const char **ie = NULL, **de = NULL;
    if (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_RequiredExtensions(&ni, &ie, &nd, &de))) return;
    for (unsigned i = 0; i < ni; i++) {
        int on = 0;
        for (uint32_t k = 0; k < *n; k++) on |= !strcmp(ext[k], ie[i]);
        if (!on && *n < cap) ext[(*n)++] = ie[i];
    }
}

static wchar_t *wide(const char *s) {
#ifdef _WIN32   // paths are UTF-8 here; mbstowcs would follow the C locale
    int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
    wchar_t *w = calloc(n > 0 ? (size_t)n : 1, sizeof *w);
    if (w && n > 0) MultiByteToWideChar(CP_UTF8, 0, s, -1, w, n);
    return w;
#else
    size_t n = strlen(s) + 1;
    wchar_t *w = calloc(n, sizeof *w);
    if (w) mbstowcs(w, s, n);
    return w;
#endif
}

// 1 when dir (beside the executable: its dlss/ folder, else its own folder) holds the model library
// (libnvidia-ngx-dlss.so.<version>; nvngx_dlss.dll on Windows).
#ifdef _WIN32
#define DLSS_SEP "\\"
static int dlss_has_model(const char *d) {
    char path[1100];
    snprintf(path, sizeof path, "%s\\" DLSS_MODEL, d);
    return plat_file_exists(path);
}
#else
#define DLSS_SEP "/"
static int dlss_has_model(const char *d) {
    DIR *dp = opendir(d);
    if (!dp) return 0;
    int found = 0;
    for (struct dirent *de; !found && (de = readdir(dp));) found = !strncmp(de->d_name, DLSS_MODEL, sizeof DLSS_MODEL - 1);
    closedir(dp);
    return found;
}
#endif
static int dlss_dir_beside_exe(char *dir, size_t cap) {
    char exe[1024];
    if (!plat_exe_path(exe, sizeof exe)) return 0;
    char *slash = plat_last_sep(exe);
    if (!slash) return 0;
    *slash = 0;
    snprintf(dir, cap, "%s" DLSS_SEP "dlss", exe);
    if (dlss_has_model(dir)) return 1;
    snprintf(dir, cap, "%s", exe);
    return dlss_has_model(dir);
}

void vk_dlss_init(VkCtx *c) {
    if (c->props.vendorID != 0x10de) { rt_log("dlss: off, not an NVIDIA GPU (vendor %#x): the stand-in stays", c->props.vendorID); return; }
    char dir[1024], logs[1024];
    if (!dlss_dir_beside_exe(dir, sizeof dir)) {
        rt_log("dlss: off, no " DLSS_MODEL " in a dlss folder beside the executable: the stand-in stays");
        return;
    }
    dlss_avail = 1;   // the menu offers the setting from here on, also while it is off
    if (!strcmp(cfg_value("OOT_DLSS"), "off")) { rt_log("dlss: off (the dlss setting): the stand-in stays"); return; }
    rt_log("dlss: model directory %s", dir);
    snprintf(logs, sizeof logs, "%s", "ngx");   // beside the binary (the run's working directory)
#ifdef _WIN32
    plat_file_mkdir_p(logs);
#else
    if (mkdir(logs, 0755) && errno != EEXIST) snprintf(logs, sizeof logs, ".");
#endif
    wchar_t *wdir = wide(dir), *wlogs = wide(logs);
    const wchar_t *paths[1] = {wdir};
    NVSDK_NGX_FeatureCommonInfo fi;
    memset(&fi, 0, sizeof fi);
    fi.PathListInfo.Path = paths;
    fi.PathListInfo.Length = 1;
    fi.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_OFF;
    fi.LoggingInfo.DisableOtherLoggingSinks = false;   // true needs a LoggingCallback: without one Init fails (0xbad00005)
    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init(DLSS_APP_ID, wlogs, c->instance, c->phys, c->device, vkGetInstanceProcAddr,
                                               vkGetDeviceProcAddr, &fi, NVSDK_NGX_Version_API);
    free(wdir), free(wlogs);
    if (NVSDK_NGX_FAILED(r)) { rt_log("dlss: off, NVSDK_NGX_VULKAN_Init failed (%#x): the stand-in stays", (unsigned)r); return; }
    NVSDK_NGX_Parameter *cap = NULL;
    r = NVSDK_NGX_VULKAN_GetCapabilityParameters(&cap);
    if (NVSDK_NGX_FAILED(r) || !cap) { rt_log("dlss: off, no NGX capability parameters (%#x): the stand-in stays", (unsigned)r); return; }
    int avail = 0, init = 0, upd = 0;
    unsigned maj = 0, min = 0;
    NVSDK_NGX_Parameter_GetI(cap, NVSDK_NGX_Parameter_SuperSampling_Available, &avail);
    NVSDK_NGX_Parameter_GetI(cap, NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, &init);
    NVSDK_NGX_Parameter_GetI(cap, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &upd);
    NVSDK_NGX_Parameter_GetUI(cap, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor, &maj);
    NVSDK_NGX_Parameter_GetUI(cap, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor, &min);
    if (!avail) {
        rt_log("dlss: off, NGX has no SuperSampling (init result %#x, needs a newer driver %d, minimum %u.%u; model directory %s): "
               "the stand-in stays", (unsigned)init, upd, maj, min, dir);
        return;
    }
    if (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_AllocateParameters(&dlss_params)) || !dlss_params) {
        rt_log("dlss: off, NVSDK_NGX_VULKAN_AllocateParameters failed: the stand-in stays");
        return;
    }
    dlss_on = 1;
    rt_log("dlss: NVIDIA DLSS Super Resolution on (NGX API %#x, models from %s): it replaces the game's upscale where its inputs "
           "are recognised", (unsigned)NVSDK_NGX_Version_API, dir);
}

// DLSS checks the render size against the mode's range.
static NVSDK_NGX_PerfQuality_Value mode_for(uint32_t rw, uint32_t ow, const char **name) {
    float ratio = (float)rw / (float)ow;
    if (ratio >= 0.99f) return *name = "DLAA", NVSDK_NGX_PerfQuality_Value_DLAA;
    if (ratio >= 0.65f) return *name = "Quality", NVSDK_NGX_PerfQuality_Value_MaxQuality;
    if (ratio >= 0.57f) return *name = "Balanced", NVSDK_NGX_PerfQuality_Value_Balanced;
    if (ratio >= 0.49f) return *name = "Performance", NVSDK_NGX_PerfQuality_Value_MaxPerf;
    return *name = "Ultra Performance", NVSDK_NGX_PerfQuality_Value_UltraPerformance;
}

// Preset M; K with dlss = fast, at about half the GPU time (2.1 against 4.5 ms for 2560x1440 on an RTX 3070).
static unsigned preset(char *letter) {
    *letter = strcmp(cfg_value("OOT_DLSS"), "fast") ? 'M' : 'K';
    return NVSDK_NGX_DLSS_Hint_Render_Preset_J + (unsigned)(*letter - 'J');
}

static int feature_for(VkCommandBuffer cb, uint32_t rw, uint32_t rh, uint32_t ow, uint32_t oh, const char **why) {
    char letter;
    unsigned p = preset(&letter);
    for (int i = 0; i < nfeat; i++)
        if (F[i].rw == rw && F[i].rh == rh && F[i].ow == ow && F[i].oh == oh && F[i].preset == p) return i;
    int slot = nfeat;
    if (nfeat == DLSS_FEATURES) {
        // All 8 in use: the feature unused longest goes, if no evaluation in flight can still use it (120 evaluations,
        // about 2 s; at most a few frames are in flight).
        slot = -1;
        for (int i = 0; i < nfeat; i++)
            if (nevals - F[i].used >= 120 && (slot < 0 || F[i].used < F[slot].used)) slot = i;
        if (slot < 0) return *why = "DLSS has a feature for 8 sizes and presets already", -1;
        rt_log("dlss: feature %ux%u -> %ux%u released (unused for %llu evaluations) for a new size", F[slot].rw, F[slot].rh,
               F[slot].ow, F[slot].oh, (unsigned long long)(nevals - F[slot].used));
        if (F[slot].h) NVSDK_NGX_VULKAN_ReleaseFeature(F[slot].h);
        memset(&F[slot], 0, sizeof F[slot]);   // no size matches it until the new feature is in (creation may fail)
        if (last_feat == slot) last_feat = -1;
    }
    const char *mode_name;
    NVSDK_NGX_PerfQuality_Value mode = mode_for(rw, ow, &mode_name);
    NVSDK_NGX_Parameter_SetUI(dlss_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, p);
    NVSDK_NGX_Parameter_SetUI(dlss_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality, p);
    NVSDK_NGX_Parameter_SetUI(dlss_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced, p);
    NVSDK_NGX_Parameter_SetUI(dlss_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance, p);
    NVSDK_NGX_Parameter_SetUI(dlss_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance, p);
    NVSDK_NGX_Parameter_SetUI(dlss_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraQuality, p);
    NVSDK_NGX_DLSS_Create_Params cp;
    memset(&cp, 0, sizeof cp);
    cp.Feature.InWidth = rw, cp.Feature.InHeight = rh, cp.Feature.InTargetWidth = ow, cp.Feature.InTargetHeight = oh;
    cp.Feature.InPerfQualityValue = mode;
    cp.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;   // LDR, unjittered motion, depth not inverted
    NVSDK_NGX_Handle *h = NULL;
    NVSDK_NGX_Result r = NGX_VULKAN_CREATE_DLSS_EXT(cb, 1, 1, &h, dlss_params, &cp);
    if (NVSDK_NGX_FAILED(r) || !h) {
        static char msg[96];
        snprintf(msg, sizeof msg, "DLSS feature %ux%u -> %ux%u could not be created (%#x)", rw, rh, ow, oh, (unsigned)r);
        rt_log("dlss: %s", msg);
        return *why = msg, -1;
    }
    rt_log("dlss: feature %ux%u -> %ux%u, mode %s, preset %c, LDR", rw, rh, ow, oh, mode_name, letter);
    F[slot] = (typeof(F[0])){rw, rh, ow, oh, p, h, nevals};
    if (slot == nfeat) nfeat++;
    return slot;
}

static NVSDK_NGX_Resource_VK res(const VkDlssImage *i, bool rw) {
    VkImageSubresourceRange range = {i->aspect, 0, 1, 0, 1};
    return NVSDK_NGX_Create_ImageView_Resource_VK(i->view, i->image, range, i->format, i->w, i->h, rw);
}

// Records feature creation (first use of a size) and the evaluation into cb, on the recording thread with the recorder
// drained (vk_exec_draw.c). Inputs in SHADER_READ_ONLY_OPTIMAL, the output in GENERAL. 1 when recorded; 0 with *why set,
// and the caller's stand-in then draws the frame. Three failed evaluations in a row turn DLSS off for the run.
int vk_dlss_evaluate(VkCommandBuffer cb, const VkDlssImage *color, const VkDlssImage *depth, const VkDlssImage *motion,
                     const VkDlssImage *bias, const VkDlssImage *out, const float *jitter, int reset, const char **why) {
    static int fails;
    if (!vk_dlss_on()) return *why = "DLSS is off", 0;
    int f = feature_for(cb, color->w, color->h, out->w, out->h, why);
    if (f < 0) return 0;
    if (f != last_feat) reset = 1;
    NVSDK_NGX_Resource_VK rc = res(color, false), rd = res(depth, false), rm = res(motion, false), ro = res(out, true), rb;
    if (bias) rb = res(bias, false);
    NVSDK_NGX_VK_DLSS_Eval_Params ep;
    memset(&ep, 0, sizeof ep);
    ep.Feature.pInColor = &rc;
    ep.Feature.pInOutput = &ro;
    ep.pInDepth = &rd;
    ep.pInMotionVectors = &rm;
    ep.pInBiasCurrentColorMask = bias ? &rb : NULL;   // 1 = trust the current colour (disocclusion)
    // The sub-pixel shift of this frame's scene draws (render pixels, y down); NULL: unjittered
    ep.InJitterOffsetX = jitter ? jitter[0] : 0.0f, ep.InJitterOffsetY = jitter ? jitter[1] : 0.0f;
    ep.InRenderSubrectDimensions.Width = color->w, ep.InRenderSubrectDimensions.Height = color->h;
    ep.InReset = reset;
    ep.InMVScaleX = ep.InMVScaleY = 1.0f;              // the motion pass writes render pixels
    NVSDK_NGX_Result r = NGX_VULKAN_EVALUATE_DLSS_EXT(cb, F[f].h, dlss_params, &ep);
    if (NVSDK_NGX_FAILED(r)) {
        static char msg[64];
        snprintf(msg, sizeof msg, "DLSS evaluation failed (%#x)", (unsigned)r);
        if (++fails >= 3) { dlss_broken = 1; rt_log("dlss: three evaluations failed in a row: off for this run (%#x)", (unsigned)r); }
        return *why = msg, 0;
    }
    fails = 0;
    last_feat = f;
    F[f].used = ++nevals;
    return 1;
}
