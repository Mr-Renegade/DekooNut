// The checks of a vertex-geometry(-fragment) pipeline's interfaces and device limits before it is built
// (geometry_validate).
#include <stdint.h>

#include "gfx.h"
#include "host.h"
#include "vk.h"
#include "vk_pipeline_int.h"

// Experimental GS accepts a bounded interface: unpacked 32-bit scalar/vectors, the GS outer vertex array and numeric
// built-in blocks, parsed with the same fixup used at upload. The outer GS array is disregarded for matching, and
// component limits include built-ins; device checks use VkPhysicalDeviceLimits.
typedef struct {
    uint32_t op, x, y, type, storage, location, builtin;
    uint8_t component, listed, block, flat, invariant, interpolation, unsupported;
} GeometryId;
typedef struct {
    uint32_t mask, components;
    GfxShaderIo io[32];
    uint8_t invariant[32];
    uint64_t builtin_mask, builtin[64];
    uint64_t standalone_builtins;
    uint8_t block_members, block_invariant, block_builtin[16];
    uint8_t system_inputs; // bit 0: subgroup lane ID, bit 1: subgroup less-than mask; neither is VS-linked
} GeometryInterface;

// These standalone GS inputs are supplied by the implementation, not the VS.
// Match the exact U32 forms emitted by the translator's IoMap, without putting large
// BuiltIn IDs into the per-vertex 64-bit mask or consuming vertex components.
static int geometry_system_input(const GeometryId *id, uint32_t bound, const GeometryId *var,
                                 uint32_t type, uint8_t *seen) {
    uint32_t builtin = var->builtin - 1;
    uint8_t bit = builtin == 41 ? 1 : builtin == 4420 ? 2 : 0;
    if (!bit || (*seen & bit) || var->location || var->component || var->invariant ||
        var->interpolation || var->unsupported || var->block || !type || type >= bound) return 0;
    if (bit == 2) {
        if (id[type].op != 23 || id[type].y != 4) return 0;
        type = id[type].x;
        if (!type || type >= bound) return 0;
    }
    if (id[type].op != 21 || id[type].x != 32 || id[type].y) return 0;
    *seen |= bit;
    return 1;
}

// Reject known subgroup extension families without rejecting unrelated shader
// extensions. The literal must terminate within its instruction, with zero pad.
static int geometry_subgroup_extension(const uint32_t *words, uint32_t count) {
    if (!count) return 0;
    const char *name = (const char *)words;
    size_t bytes = (size_t)count * 4;
    const char *end = memchr(name, 0, bytes);
    if (!end || end == name) return 0;
    for (const char *p = end + 1; p < name + bytes; p++) if (*p) return 0;
    static const char *tokens[] = {"subgroup", "sub_group", "group_non_uniform", "shader_ballot"};
    for (unsigned t = 0; t < sizeof tokens / sizeof *tokens; t++) {
        size_t length = strlen(tokens[t]);
        for (const char *p = name; (size_t)(end - p) >= length; p++) {
            size_t i;
            for (i = 0; i < length; i++) {
                unsigned char c = (unsigned char)p[i];
                if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
                if (c != (unsigned char)tokens[t][i]) break;
            }
            if (i == length) return 0;
        }
    }
    return 1;
}

// Capability support applies even without an instruction; stage support is a separate requirement for actual
// subgroup-scoped group operations (VUID-RuntimeSpirv-None-06343). Admit only the group families emitted here.
static int geometry_subgroups(const GfxShader *s, uint8_t system_inputs) {
    uint64_t words = 0;
    const uint32_t *w = gfx_shader_spirv(s, &words);
    if (!w || words < 5 || !w[3] || w[3] > (1u << 16)) return 0;
    uint32_t bound = w[3], declared = 0, operations = 0;
    GeometryId *id = calloc(bound, sizeof *id);
    int ok = 0;
    if (!id) return 0;
    for (uint64_t i = 5; i < words;) {
        uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (!n || n > words - i) goto done;
        const uint32_t *a = w + i + 1;
        if (op == 17) {
            if (n != 2) goto done;
            if (a[0] >= 61 && a[0] <= 68) declared |= 1u << (a[0] - 61);
            else if (a[0] == 58 || a[0] == 4423 || a[0] == 4431 || a[0] == 5297 ||
                     (a[0] >= 5568 && a[0] <= 5570) || a[0] == 5579 ||
                     (a[0] >= 5696 && a[0] <= 5698)) goto done;
        } else if (op == 10) {
            if (!geometry_subgroup_extension(a, n - 1)) goto done;
        } else if (op == 21 && n == 4 && a[0] < bound) {
            id[a[0]].op = op; id[a[0]].x = a[1]; id[a[0]].y = a[2];
        } else if (op == 43 && n == 4 && a[1] < bound) {
            id[a[1]].op = op; id[a[1]].type = a[0]; id[a[1]].x = a[2];
        } else if (op >= 333 && op <= 366) {
            if (n < 5 || op == 333 || op > 348 || !a[2] || a[2] >= bound) goto done;
            uint32_t expected = (op <= 336 || op == 338 || op == 339 || op == 340 || op == 343 || op == 344) ? 5 : 6;
            if (n != expected) goto done;
            const GeometryId *scope = &id[a[2]];
            if (scope->op != 43 || scope->x != 3 || !scope->type || scope->type >= bound ||
                id[scope->type].op != 21 || id[scope->type].x != 32 || id[scope->type].y) goto done;
            operations |= op <= 336 ? VK_SUBGROUP_FEATURE_VOTE_BIT :
                op <= 344 ? VK_SUBGROUP_FEATURE_BALLOT_BIT : op <= 346 ? VK_SUBGROUP_FEATURE_SHUFFLE_BIT :
                VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT;
        } else if ((op >= 259 && op <= 271) || op == 5296 || op == 224 ||
                   (op >= 4421 && op <= 4422) || (op >= 4428 && op <= 4430) || op == 4432 ||
                   (op >= 5000 && op <= 5007) || (op >= 5571 && op <= 5578) ||
                   (op >= 5580 && op <= 5581) || (op >= 5713 && op <= 5816)) goto done;
        i += n;
    }
    // The specialized GroupNonUniform capabilities imply GroupNonUniform;
    // the translator does not repeat that base declaration in generated modules.
    if (declared) declared |= VK_SUBGROUP_FEATURE_BASIC_BIT;
    uint32_t required = (system_inputs & 1 ? VK_SUBGROUP_FEATURE_BASIC_BIT : 0) |
                        (system_inputs & 2 ? VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT : 0);
    if ((required | operations) & ~declared) goto done;
    // The selected system-input subset also requires reported GS support (a conservative admission policy).
    if ((declared & ~P.subgroup_operations) ||
        ((system_inputs || operations) && !(P.subgroup_stages & VK_SHADER_STAGE_GEOMETRY_BIT))) goto done;
    ok = 1;
done:
    free(id);
    return ok;
}

static int geometry_numeric(const GeometryId *id, uint32_t bound, uint32_t type, int array,
                            uint32_t *components, uint64_t *signature, GfxShaderIo *io) {
    if (!type || type >= bound) return 0;
    const GeometryId *t = &id[type];
    if (array && t->op == 28) {
        if (!t->y || t->y >= bound || id[t->y].op != 43 || !id[t->y].type || id[t->y].type >= bound ||
            id[id[t->y].type].op != 21 || id[id[t->y].type].x != 32 || !id[t->y].x || id[t->y].x > 4096) return 0;
        uint32_t count;
        uint64_t element;
        GfxShaderIo ignored;
        if (!geometry_numeric(id, bound, t->x, 0, &count, &element, &ignored)) return 0;
        *components = count * id[t->y].x;
        *signature = element | (uint64_t)id[t->y].x << 32 | UINT64_C(1) << 63;
        return 1;
    }
    uint32_t count = 1;
    if (t->op == 23) {
        count = t->y;
        if (count < 2 || count > 4 || !t->x || t->x >= bound) return 0;
        t = &id[t->x];
    }
    if (t->op == 21 && t->y > 1) return 0;
    uint32_t kind = t->op == 22 ? 1 : t->op == 21 ? (t->y ? 2 : 3) : t->op == 20 ? 4 : 0;
    if (!kind || (kind != 4 && t->x != 32) || (kind == 4 && count != 1)) return 0;
    *components = count;
    *signature = kind | (uint64_t)(kind == 4 ? 1 : 32) << 8 | (uint64_t)count << 16;
    *io = (GfxShaderIo){(uint8_t)(kind - 1), (uint8_t)count, 0, (uint8_t)(kind == 4 ? 1 : 32)};
    return 1;
}

static int geometry_member(const uint32_t *w, uint64_t words, uint32_t type, uint32_t member,
                           uint32_t *builtin, uint32_t *invariant) {
    int found = 0;
    *invariant = 0;
    for (uint64_t i = 5; i < words;) {
        uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (!n || n > words - i) return 0;
        if (op == 72 && n >= 4 && w[i + 1] == type && w[i + 2] == member) {
            if (w[i + 3] == 11) {
                if (n != 5 || found++ || w[i + 4] >= 64) return 0;
                *builtin = w[i + 4];
            } else if (w[i + 3] == 18 && n == 4) *invariant = 1;
            else if (w[i + 3] != 0 || n != 4) return 0;
        }
        i += n;
    }
    return found == 1;
}

static int geometry_interface(const GfxShader *s, uint32_t storage, uint32_t vertices, GeometryInterface *out) {
    memset(out, 0, sizeof *out);
    uint64_t words = 0, fixed_words = 0;
    const uint32_t *w = gfx_shader_spirv(s, &words);
    if (!w || words < 5 || words > (1u << 20) || !w[3] || w[3] > (1u << 16)) return 0;
    int stage = gfx_shader_stage(s);
    uint32_t *fixed = spirv_fixup_linked(w, words, &fixed_words, stage == GFX_STAGE_VS);
    if (fixed) w = fixed, words = fixed_words;
    uint32_t bound = w[3], entries = 0;
    GeometryId *id = calloc(bound, sizeof *id);
    int ok = 0;
    if (!id) goto done;
    for (uint64_t i = 5; i < words;) {
        uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (!n || n > words - i) goto done;
        const uint32_t *a = w + i + 1;
        if (op == 15) {
            if (n < 4 || entries++ || a[0] != (stage == GFX_STAGE_VS ? 0u : stage == GFX_STAGE_GS ? 3u : 4u)) goto done;
            const char *name = (const char *)(a + 2), *end = memchr(name, 0, (n - 3) * 4);
            if (!end || end - name != 4 || memcmp(name, "main", 4)) goto done;
            uint32_t start = 2 + ((uint32_t)(end - name) + 4) / 4;
            for (uint32_t j = start; j < n - 1; j++) {
                if (!a[j] || a[j] >= bound || id[a[j]].listed) goto done;
                id[a[j]].listed = 1;
            }
        } else if (op == 17 && n == 2) {
            if ((a[0] == 24 && !P.f_gs_point_size) || (a[0] == 32 && !P.f_clip) || (a[0] == 33 && !P.f_cull) ||
                a[0] == 53 || a[0] == 54) goto done;       // transform feedback / geometry streams not supported here
        } else if (op == 220 || op == 221) goto done;    // stream emission needs a different contract
        else if (op == 73 || op == 74 || op == 75 || op == 332) goto done; // indirect decorations are not modeled
        else if (op == 71 && n >= 3 && a[0] < bound) {
            GeometryId *d = &id[a[0]];
            if (a[1] == 30) { if (n != 4 || a[2] >= 32 || d->location) goto done; d->location = a[2] + 1; }
            else if (a[1] == 11) { if (n != 4 || (a[2] >= 64 && a[2] != 4420) || d->builtin) goto done; d->builtin = a[2] + 1; }
            else if (a[1] == 31) d->component = 1;
            else if (a[1] == 2) d->block = 1;
            else if (a[1] == 14) d->flat = d->interpolation = 1;
            else if (a[1] == 13 || a[1] == 16 || a[1] == 17) d->interpolation = 1;
            else if (a[1] == 18) d->invariant = 1;
            else if (a[1] == 15 || a[1] == 5271 || a[1] == 5285) goto done;
            else if (a[1] != 0) d->unsupported = 1;
        } else if ((op == 20 || op == 21 || op == 22 || op == 23 || op == 28 || op == 30 || op == 32) && n >= 2 && a[0] < bound) {
            GeometryId *t = &id[a[0]];
            t->op = op;
            if (op == 30) t->x = (uint32_t)i + 2, t->y = n - 2;
            else if (op == 32) { if (n != 4) goto done; t->x = a[2]; t->y = a[1]; }
            else { if (n >= 3) t->x = a[1]; if (n >= 4) t->y = a[2]; }
        } else if (op == 43 && n == 4 && a[1] < bound) id[a[1]].op = op, id[a[1]].x = a[2], id[a[1]].type = a[0];
        else if (op == 59 && n >= 4 && a[1] < bound) id[a[1]].type = a[0], id[a[1]].storage = a[2];
        i += n;
    }
    if (entries != 1) goto done;
    for (uint32_t v = 1; v < bound; v++) {
        GeometryId *var = &id[v];
        if (!var->listed) continue;
        if (!var->type || var->type >= bound || id[var->type].op != 32 || id[var->type].y != var->storage) goto done;
        if (var->storage != storage) continue;
        uint32_t type = id[var->type].x;
        if (!type || type >= bound || var->component || var->unsupported ||
            (var->builtin && var->interpolation)) goto done;
        if (var->builtin == 42 || var->builtin == 4421) {
            if (stage != GFX_STAGE_GS || storage != 1 ||
                !geometry_system_input(id, bound, var, type, &out->system_inputs)) goto done;
            continue;
        }
        if (stage == GFX_STAGE_GS && storage == 1 && id[type].op == 28) {
            uint32_t length = id[type].y;
            if (!length || length >= bound || id[length].op != 43 || id[length].x != vertices ||
                !id[length].type || id[length].type >= bound || id[id[length].type].op != 21 ||
                id[id[length].type].x != 32) goto done;
            type = id[type].x;
            if (!type || type >= bound) goto done;
        } else if (stage == GFX_STAGE_GS && storage == 1 && var->builtin != 8 && var->builtin != 9) goto done;
        if (id[type].op == 30) {
            if (!id[type].block || id[type].unsupported || id[type].invariant || id[type].interpolation ||
                var->interpolation || var->location || var->builtin ||
                !id[type].y || id[type].y > 16 || out->block_members) goto done;
            out->block_members = (uint8_t)id[type].y;
            out->block_invariant = var->invariant;
            for (uint32_t member = 0; member < id[type].y; member++) {
                uint32_t builtin = 0, invariant = 0, components;
                uint64_t signature;
                GfxShaderIo ignored;
                if (!geometry_member(w, words, type, member, &builtin, &invariant) ||
                    !geometry_numeric(id, bound, w[id[type].x + member], 1, &components, &signature, &ignored) ||
                    (out->builtin_mask & (UINT64_C(1) << builtin))) goto done;
                out->builtin_mask |= UINT64_C(1) << builtin;
                out->block_builtin[member] = (uint8_t)builtin;
                out->builtin[builtin] = signature | (uint64_t)invariant << 62;
                out->components += components;
            }
        } else {
            uint32_t components;
            uint64_t signature;
            GfxShaderIo io;
            if (!geometry_numeric(id, bound, type, var->builtin != 0, &components, &signature, &io)) goto done;
            if (var->location && !var->builtin) {
                uint32_t location = var->location - 1;
                if (io.scalar_bits != 32 || io.type > GFX_TYPE_UINT || (out->mask >> location & 1) ||
                    (stage == GFX_STAGE_FS && io.type != GFX_TYPE_FLOAT && !var->flat)) goto done;
                out->mask |= 1u << location;
                out->io[location] = io;
                out->invariant[location] = var->invariant;
            } else if (var->builtin && !var->location) {
                uint32_t builtin = var->builtin - 1;
                if (out->builtin_mask & (UINT64_C(1) << builtin)) goto done;
                out->builtin_mask |= UINT64_C(1) << builtin;
                out->standalone_builtins |= UINT64_C(1) << builtin;
                out->builtin[builtin] = signature | (uint64_t)var->invariant << 62;
            } else goto done;
            out->components += components;
        }
    }
    GfxShaderIo reflected[32];
    uint32_t mask = storage == 1 ? gfx_shader_inputs(s, reflected) : gfx_shader_outputs(s, reflected);
    if (mask != out->mask) goto done;
    for (unsigned i = 0; i < 32; i++) if (mask >> i & 1)
        if (reflected[i].packed || reflected[i].scalar_bits != 32 || reflected[i].type != out->io[i].type ||
            reflected[i].comps != out->io[i].comps) goto done;
    ok = 1;
done:
    free(id), free(fixed);
    return ok;
}

static int geometry_link(const GeometryInterface *producer, const GeometryInterface *consumer) {
    if (consumer->mask & ~producer->mask) return 0;
    for (unsigned i = 0; i < 32; i++) if (consumer->mask >> i & 1)
        if (memcmp(&producer->io[i], &consumer->io[i], sizeof producer->io[i]) ||
            producer->invariant[i] != consumer->invariant[i]) return 0;
    return 1;
}

static int geometry_locations(uint32_t mask, uint32_t limit) {
    for (unsigned i = 0; i < 32; i++) if ((mask >> i & 1) && i >= limit / 4) return 0;
    return 1;
}

static int geometry_builtin_link(const GeometryInterface *producer, const GeometryInterface *consumer) {
    // Built-in blocks match as structures, not as an unordered subset of their
    // BuiltIn IDs (VUID-RuntimeSpirv-OpVariable-08746). GS outer vertex arrays have
    // already been removed by geometry_interface. Unsupported blocks fail there.
    if (consumer->block_members) {
        if (producer->block_members != consumer->block_members ||
            producer->block_invariant != consumer->block_invariant) return 0;
        for (unsigned i = 0; i < consumer->block_members; i++) {
            unsigned builtin = consumer->block_builtin[i];
            if (producer->block_builtin[i] != builtin || producer->builtin[builtin] != consumer->builtin[builtin]) return 0;
        }
    }
    const uint64_t per_vertex = UINT64_C(1) | UINT64_C(1) << 1 | UINT64_C(1) << 3 | UINT64_C(1) << 4;
    return !(consumer->standalone_builtins & per_vertex & ~producer->standalone_builtins);
}

static int geometry_per_vertex(const GeometryInterface *io, const VkPhysicalDeviceLimits *limits) {
    uint32_t distances = 0;
    const uint64_t scalar = 1 | UINT64_C(32) << 8 | UINT64_C(1) << 16;
    for (unsigned i = 0; i < 5; i++) if (io->builtin_mask >> i & 1) {
        uint64_t signature = io->builtin[i] & ~(UINT64_C(1) << 62); // Invariant does not change the type
        if (i == 0) { if (signature != (1 | UINT64_C(32) << 8 | UINT64_C(4) << 16)) return 0; }
        else if (i == 1) { if (signature != scalar) return 0; }
        else if (i == 3 || i == 4) {
            uint32_t n = (uint32_t)(signature >> 32) & 0x3fffffff;
            if (!(signature >> 63) || (uint32_t)signature != scalar || !n ||
                n > (i == 3 ? limits->maxClipDistances : limits->maxCullDistances)) return 0;
            distances += n;
        } else return 0;
    }
    return distances <= limits->maxCombinedClipAndCullDistances;
}

const char *geometry_validate(const VkPipeState *s, const VkPhysicalDeviceLimits *limits) {
    static const uint8_t input_vertices[] = {0, 1, 2, 4, 3, 6};
    if (!s->vs || !s->gs || gfx_shader_stage(s->vs) != GFX_STAGE_VS ||
        (s->fs && gfx_shader_stage(s->fs) != GFX_STAGE_FS)) return "geometry shader stage mismatch";
    const GfxShaderGeometry *g = gfx_shader_geometry(s->gs);
    if (!P.f_geometry || !g ||
        (g->version != GFX_GEOMETRY_VERSION && g->version != GFX_GEOMETRY_PASSTHROUGH_VERSION) || g->passthrough ||
        !vk_geometry_input(s->prim) || vk_geometry_input(s->prim) != g->input ||
        g->input_vertices != input_vertices[g->input] || g->output < GFX_GEOMETRY_OUTPUT_POINTS ||
        g->output > GFX_GEOMETRY_OUTPUT_TRIANGLE_STRIP) return "geometry input/stage contract differs from draw";
    if (g->version == GFX_GEOMETRY_PASSTHROUGH_VERSION &&
        (!s->fs || g->input != GFX_GEOMETRY_TRIANGLES || g->output != GFX_GEOMETRY_OUTPUT_TRIANGLE_STRIP ||
         g->invocations != 1 || g->max_output_vertices != 3)) return "unsupported portable geometry contract";
    // VUID-VkGraphicsPipelineCreateInfo-shaderTessellationAndGeometryPointSize-08776 requires PointSize on every emitted
    // point, and a declaration is not proof of those writes; polygonModePointSize is not contracted here.
    if (g->output == GFX_GEOMETRY_OUTPUT_POINTS || s->poly.mode == NVN_POLYGON_MODE_POINT)
        return "geometry point-size emission/rasterization is not implemented";
    if (!g->invocations || g->invocations > limits->maxGeometryShaderInvocations || !g->max_output_vertices ||
        g->max_output_vertices > limits->maxGeometryOutputVertices) return "geometry invocation/output-vertex device limit";
    const GfxShader *shaders[] = {s->vs, s->gs};
    for (unsigned j = 0; j < 2; j++) for (int kind = GFX_RES_SSBO; kind < GFX_RES_KINDS; kind++) {
        const GfxShaderRes *res;
        int count = gfx_shader_resources(shaders[j], kind, &res);
        for (int i = 0; i < count; i++)
            if (!P.f_vertex_stores && ((kind == GFX_RES_SSBO && (res[i].flags & 1)) ||
                                      (kind == GFX_RES_IMG && (res[i].flags & 4)))) return "pre-raster stores require vertexPipelineStoresAndAtomics";
    }
    GeometryInterface vo, gi, go, fi;
    if (!geometry_interface(s->vs, 3, 0, &vo)) return "unsupported vertex output interface or missing scalar-width metadata";
    if (!geometry_interface(s->gs, 1, g->input_vertices, &gi)) return "unsupported geometry input interface or missing scalar-width metadata";
    if (!geometry_interface(s->gs, 3, 0, &go)) return "unsupported geometry output interface or missing scalar-width metadata";
    if (!geometry_subgroups(s->gs, gi.system_inputs)) return "unsupported geometry subgroup capability, operation or stage";
    uint64_t per_vertex = UINT64_C(1) | UINT64_C(1) << 1 | UINT64_C(1) << 3 | UINT64_C(1) << 4;
    uint64_t generated = (UINT64_C(1) << 7) | (UINT64_C(1) << 8); // PrimitiveId / InvocationId are not VS outputs
    if ((gi.builtin_mask & ~(per_vertex | generated)) || (go.builtin_mask & ~per_vertex))
        return "geometry built-in outside the supported per-vertex subset";
    const uint64_t scalar_int = 2 | UINT64_C(32) << 8 | UINT64_C(1) << 16;
    for (unsigned i = 7; i <= 8; i++)
        if ((gi.builtin_mask >> i & 1) && gi.builtin[i] != scalar_int) return "geometry invocation/primitive ID type mismatch";
    if (!geometry_per_vertex(&vo, limits) || !geometry_per_vertex(&gi, limits) || !geometry_per_vertex(&go, limits))
        return "geometry per-vertex built-in type/device limit";
    uint64_t distance_components = 0;
    for (unsigned i = 3; i <= 4; i++) {
        if (gi.builtin_mask >> i & 1) distance_components += gi.builtin[i] >> 32 & 0x3fffffff;
        if (go.builtin_mask >> i & 1) distance_components += go.builtin[i] >> 32 & 0x3fffffff;
    }
    if (distance_components > limits->maxCombinedClipAndCullDistances)
        return "geometry combined input/output distance device limit";
    if (!geometry_link(&vo, &gi)) return "vertex-to-geometry varying mismatch";
    if (!geometry_builtin_link(&vo, &gi)) return "vertex-to-geometry built-in block mismatch";
    uint64_t need = gi.builtin_mask & ~generated;
    if ((need & ~vo.builtin_mask) || !(go.builtin_mask & 1)) return "geometry per-vertex built-in mismatch";
    for (unsigned i = 0; i < 64; i++) if ((need >> i & 1) && gi.builtin[i] != vo.builtin[i]) return "geometry built-in type mismatch";
    if (gi.components > limits->maxGeometryInputComponents || go.components > limits->maxGeometryOutputComponents ||
        (uint64_t)go.components * g->max_output_vertices > limits->maxGeometryTotalOutputComponents ||
        vo.components > limits->maxVertexOutputComponents || !geometry_locations(gi.mask, limits->maxGeometryInputComponents) ||
        !geometry_locations(go.mask, limits->maxGeometryOutputComponents) || !geometry_locations(vo.mask, limits->maxVertexOutputComponents))
        return "geometry interface component device limit";
    if (s->fs) {
        if (!geometry_interface(s->fs, 1, 0, &fi)) return "unsupported fragment input interface or missing scalar-width metadata";
        // VUID-VkGraphicsPipelineCreateInfo-pStages-00740: the GS must write PrimitiveId on every path when the FS reads
        // it. This subset admits no such GS output and must not inherit the rasterizer-generated ID.
        if (fi.builtin_mask & (UINT64_C(1) << 7)) return "geometry-to-fragment PrimitiveId output is not implemented";
        if (!geometry_link(&go, &fi)) return "geometry-to-fragment varying mismatch";
        if (fi.components > limits->maxFragmentInputComponents || !geometry_locations(fi.mask, limits->maxFragmentInputComponents))
            return "fragment input component device limit";
    }
    return NULL;
}
