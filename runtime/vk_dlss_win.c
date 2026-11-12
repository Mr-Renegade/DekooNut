// DLSS left out of a Windows build: what tools/build_game.py compiles when it is given no DLSS SDK (--dlss-sdk).
// The game's upscale stays with the stand-in and the present's FSR. The same interface as vk_dlss.c, which is built
// instead when the SDK is there (with vk_dlss_msvc_win.c for NVIDIA's Visual C++ loader library).
#include "host.h"
#include "vk.h"

int vk_dlss_on(void) { return 0; }
int vk_dlss_available(void) { return 0; }
void vk_dlss_instance_ext(const char **ext, uint32_t *n, uint32_t cap) { (void)ext, (void)n, (void)cap; }
void vk_dlss_device_ext(const char **ext, uint32_t *n, uint32_t cap, const VkExtensionProperties *ep, uint32_t ne) {
    (void)ext, (void)n, (void)cap, (void)ep, (void)ne;
}
void vk_dlss_init(VkCtx *c) {
    (void)c;
    rt_log("dlss: off, this build was made without the DLSS SDK: the stand-in stays");
}
int vk_dlss_evaluate(VkCommandBuffer cb, const VkDlssImage *color, const VkDlssImage *depth, const VkDlssImage *motion,
                     const VkDlssImage *bias, const VkDlssImage *out, const float *jitter, int reset, const char **why) {
    (void)cb, (void)color, (void)depth, (void)motion, (void)bias, (void)out, (void)jitter, (void)reset;
    return *why = "DLSS is off", 0;
}
