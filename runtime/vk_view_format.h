// Vulkan color-view compatibility for uncompressed core formats. Depth and compressed
// images cannot be reinterpreted merely because their texel/block byte counts match.
#pragma once
#include <vulkan/vulkan.h>

static inline unsigned vk_color_view_bits(VkFormat f) {
    if (f == VK_FORMAT_R4G4_UNORM_PACK8 || (f >= VK_FORMAT_R8_UNORM && f <= VK_FORMAT_R8_SRGB)) return 8;
    if ((f >= VK_FORMAT_R4G4B4A4_UNORM_PACK16 && f <= VK_FORMAT_A1R5G5B5_UNORM_PACK16) ||
        (f >= VK_FORMAT_R8G8_UNORM && f <= VK_FORMAT_R8G8_SRGB) ||
        (f >= VK_FORMAT_R16_UNORM && f <= VK_FORMAT_R16_SFLOAT) ||
        f == VK_FORMAT_A4R4G4B4_UNORM_PACK16 || f == VK_FORMAT_A4B4G4R4_UNORM_PACK16) return 16;
    if (f >= VK_FORMAT_R8G8B8_UNORM && f <= VK_FORMAT_B8G8R8_SRGB) return 24;
    if ((f >= VK_FORMAT_R8G8B8A8_UNORM && f <= VK_FORMAT_A2B10G10R10_SINT_PACK32) ||
        (f >= VK_FORMAT_R16G16_UNORM && f <= VK_FORMAT_R16G16_SFLOAT) ||
        (f >= VK_FORMAT_R32_UINT && f <= VK_FORMAT_R32_SFLOAT) ||
        f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || f == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32) return 32;
    if (f >= VK_FORMAT_R16G16B16_UNORM && f <= VK_FORMAT_R16G16B16_SFLOAT) return 48;
    if ((f >= VK_FORMAT_R16G16B16A16_UNORM && f <= VK_FORMAT_R16G16B16A16_SFLOAT) ||
        (f >= VK_FORMAT_R32G32_UINT && f <= VK_FORMAT_R32G32_SFLOAT) ||
        (f >= VK_FORMAT_R64_UINT && f <= VK_FORMAT_R64_SFLOAT)) return 64;
    if (f >= VK_FORMAT_R32G32B32_UINT && f <= VK_FORMAT_R32G32B32_SFLOAT) return 96;
    if ((f >= VK_FORMAT_R32G32B32A32_UINT && f <= VK_FORMAT_R32G32B32A32_SFLOAT) ||
        (f >= VK_FORMAT_R64G64_UINT && f <= VK_FORMAT_R64G64_SFLOAT)) return 128;
    if (f >= VK_FORMAT_R64G64B64_UINT && f <= VK_FORMAT_R64G64B64_SFLOAT) return 192;
    if (f >= VK_FORMAT_R64G64B64A64_UINT && f <= VK_FORMAT_R64G64B64A64_SFLOAT) return 256;
    return 0;
}

static inline int vk_color_view_compatible(VkFormat image, VkFormat view) {
    if (image == view) return image != VK_FORMAT_UNDEFINED;
    unsigned bits = vk_color_view_bits(image);
    return bits && bits == vk_color_view_bits(view);
}
