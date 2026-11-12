// What a bug report needs about the machine (sysinfo.h). The OS calls sit behind platform.h; the GPU and driver come
// from the Vulkan device the game renders on.
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

#include "cfg.h"
#include "host.h"
#include "platform/platform.h"
#include "sysinfo.h"
#include "vk.h"

void sysinfo_cpu_name(char *out, size_t n) {
    out[0] = 0;
#if defined(__x86_64__) || defined(__i386__)
    unsigned r[12];
    if (__get_cpuid_max(0x80000000, NULL) < 0x80000004) return;
    for (unsigned i = 0; i < 3; i++) __get_cpuid(0x80000002 + i, &r[i * 4], &r[i * 4 + 1], &r[i * 4 + 2], &r[i * 4 + 3]);
    char b[49];
    memcpy(b, r, 48);
    b[48] = 0;
    const char *p = b;
    while (*p == ' ') p++;
    snprintf(out, n, "%s", p);
    for (char *e = out + strlen(out); e > out && e[-1] == ' '; e--) e[-1] = 0;
#endif
}

int sysinfo_log_folder(char *out, size_t n) {
    const char *log = rt_log_path();
    if (log) snprintf(out, n, "%s", log);
    else if (!plat_stderr_path(out, n) && !plat_exe_path(out, n)) return 0;
    char *s = strrchr(out, '\\'), *f = strrchr(out, '/');
    if (f > s) s = f;
    if (!s) return 0;
    *s = 0;
    return 1;
}

typedef struct { char *buf; size_t n, at; } Out;

static void put(Out *o, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void put(Out *o, const char *fmt, ...) {
    if (o->at + 1 >= o->n) return;
    va_list ap;
    va_start(ap, fmt);
    int k = vsnprintf(o->buf + o->at, o->n - o->at, fmt, ap);
    va_end(ap);
    if (k > 0) o->at = o->at + (size_t)k < o->n ? o->at + (size_t)k : o->n - 1;
}

static void put_setting(const char *key, const char *value, int from_env, void *ctx) {
    put(ctx, from_env ? "  OOT_%s=%s (environment)\n" : "  %s = %s\n", key, value);
}

int sysinfo_text(char *buf, size_t n) {
    if (!n) return 0;
    Out o = {buf, n, 0};
    buf[0] = 0;
    char s[256];
    put(&o, "DekooNut system info\n");
    put(&o, "OS: %s\n", plat_os_version(s, sizeof s) ? s : "unknown");
    sysinfo_cpu_name(s, sizeof s);
    put(&o, "CPU: %s, %u logical processors\n", s[0] ? s : "unknown", plat_cpu_count());
    float used, total, proc;
    plat_mem_info(&used, &total, &proc);
    if (total > 0) put(&o, "RAM: %.1f GB, %.1f GB in use, the game %.1f GB\n", total / 1024, used / 1024, proc / 1024);
    VkCtx *c = vk_ctx();
    if (c) {
        VkPhysicalDeviceDriverProperties dp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &dp};
        vkGetPhysicalDeviceProperties2(c->phys, &p2);
        const VkPhysicalDeviceProperties *p = &p2.properties;
        uint64_t vram = 0;
        for (uint32_t h = 0; h < c->mem_props.memoryHeapCount; h++)
            if (c->mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) vram += c->mem_props.memoryHeaps[h].size;
        put(&o, "GPU: %s (vendor 0x%04x, device 0x%04x), %.1f GB device-local memory\n", p->deviceName, p->vendorID,
            p->deviceID, vram / 1073741824.0);
        put(&o, "Driver: %s %s, driver version 0x%08x, Vulkan %u.%u.%u\n", dp.driverName, dp.driverInfo,
            p->driverVersion, VK_API_VERSION_MAJOR(p->apiVersion), VK_API_VERSION_MINOR(p->apiVersion),
            VK_API_VERSION_PATCH(p->apiVersion));
    } else {
        put(&o, "GPU: no Vulkan device yet\n");
    }
    put(&o, "Settings with a value:\n");
    size_t before = o.at;
    cfg_each_active(put_setting, &o);
    if (o.at == before) put(&o, "  none\n");
    return (int)o.at;
}
