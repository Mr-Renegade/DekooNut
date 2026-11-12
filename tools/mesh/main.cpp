// SPDX-License-Identifier: GPL-3.0-or-later
// Standalone Lichii shader compiler adapter. Private native inputs and outputs belong under build/.
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <vector>
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ampere/program_capture.h"
#include "shader_recompiler/frontend/ampere/program_override.h"
#include "shader_recompiler/frontend/ampere/program_address_map.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"

class MeshEnvironment final : public Shader::Environment {
public:
    std::vector<u64> code;
    struct TextureAnswer { u32 slot, handle, type; };
    struct ConstantAnswer { u32 offset, value; };
    std::vector<TextureAnswer> textures;
    std::vector<ConstantAnswer> constants;
    std::vector<ConstantAnswer> pointer_answers;
    u32 texture_bank=2;
    bool assume_defaults=false;
    explicit MeshEnvironment(const std::vector<unsigned char>& native, const Shader::Ampere::LoweredProgram& lowered,
                             const std::array<u32,3>& registers, bool task, const char* context=nullptr) {
        stage = task ? Shader::Stage::Task : Shader::Stage::Mesh;
        start_address = sizeof(sph);
        mesh_registers = registers;
        is_proprietary_driver=true; // native programs fold their captured inline bank 1, as GraphicsEnvironment does
        if (lowered.required_texture_buffer != ~0U) texture_bank=lowered.required_texture_buffer;
        if (context) {
            std::ifstream input(context);
            std::string magic, raw;
            if (!(input>>magic) || magic!="MTC1") throw std::runtime_error("invalid texture context");
            char kind; u32 slot,type;
            while (input>>kind) {
                if (kind=='P') {
                    u32 bank,offset,mapped;
                    if (!(input>>bank>>offset>>mapped) || bank!=1 || offset%4 || mapped>1)
                        throw std::runtime_error("invalid inline pointer classification");
                    pointer_answers.push_back({offset,mapped});
                    continue;
                }
                if (kind=='C') {
                    u32 bank,offset;
                    if (!(input>>bank>>offset>>raw) || bank!=1 || offset%4)
                        throw std::runtime_error("invalid inline constant context");
                    size_t used=0;auto value=std::stoull(raw,&used,0);
                    if (used!=raw.size() || value>UINT32_MAX) throw std::runtime_error("invalid inline constant word");
                    for (const auto& answer:constants) if (answer.offset==offset) throw std::runtime_error("duplicate inline constant");
                    constants.push_back({offset,static_cast<u32>(value)});
                    continue;
                }
                if (!(input>>slot>>raw>>type) || kind!='T' || slot>=32 || (type!=4 && type!=5) || texture_bank!=2)
                    throw std::runtime_error("texture context supports combined float 2D/array slots in bank 2 only");
                size_t used=0;
                auto handle=std::stoull(raw,&used,0);
                if (used!=raw.size() || handle>UINT32_MAX) throw std::runtime_error("invalid captured texture handle");
                for (const auto& answer:textures) if (answer.slot==slot) throw std::runtime_error("duplicate texture slot");
                textures.push_back({slot,static_cast<u32>(handle),type});
            }
        }
        std::memcpy(&sph, native.data(), sizeof sph);
        code.resize(sizeof(sph) / 8);
        std::memcpy(code.data(), &sph, sizeof sph);
        for (u64 instruction : lowered.instructions) {
            if ((code.size() * 8) % 32 == 0) code.push_back(0);
            code.push_back(instruction);
        }
        code.push_back(0);
        Shader::Ampere::AppendNativeAddressMap(code, lowered.source_instruction_offsets);
    }
    void SetPrimitiveHeader(const unsigned char* header) { std::memcpy(&mesh_primitive_header,header,sizeof mesh_primitive_header); }
    u64 ReadInstruction(u32 address) override {
        if (address % 8 || address / 8 >= code.size()) throw std::runtime_error("instruction outside captured program");
        return code[address / 8];
    }
    bool IsNativeProgram() const override { return true; }
    // The lowering emits extended native register copies. This decoder flag is distinct from
    // IsNativeProgram's resource ABI; leaving it false silently discarded those copies as masked MOVs.
    bool HasNativeInstructions() const override { return true; }
    bool IsCbufStorageBufferPointer(u32 bank,u32 offset) override {
        if (bank!=1) return true;
        for (const auto& answer:pointer_answers) if (answer.offset==offset) return answer.value!=0;
        if (assume_defaults) return false; // an inline word: a small constant, not a storage address
        throw std::runtime_error(fmt::format("inline pointer classification required at {:#x}",offset));
    }
    u32 TranslateProgramAddress(u32 address) const override {
        if (auto target=Shader::Ampere::NativeProgramAddress(code,address)) return *target;
        throw std::runtime_error("indirect branch target outside captured native program");
    }
    u32 ReadCbufValue(u32 bank, u32 offset) override {
        if (bank==1)
            for (const auto& answer:constants) if (answer.offset==offset) return answer.value;
        if (bank==2 && offset>=0x20 && (offset-0x20)%8==0)
            for (const auto& answer:textures) if (answer.slot==(offset-0x20)/8) return answer.handle;
        if (assume_defaults && bank==2 && offset>=0x20 && (offset-0x20)%8==0) {
            // No capture: assume a 2D texture. The runtime compares the real binding's type before each draw and skips a
            // program whose assumption was wrong, so this can only lose a draw.
            textures.push_back({(offset-0x20)/8,0x1000+(offset-0x20)/8,4});
            return textures.back().handle;
        }
        throw std::runtime_error(fmt::format("constant buffer answer required: bank {} offset {:#x}",bank,offset));
    }
    Shader::TextureType ReadTextureType(u32 handle) override {
        for (const auto& answer:textures) if (answer.handle==handle)
            return answer.type==4 ? Shader::TextureType::Color2D : Shader::TextureType::ColorArray2D;
        throw std::runtime_error("texture type answer required");
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override { throw std::runtime_error("texture format answer required"); }
    bool IsTexturePixelFormatInteger(u32 handle) override {
        (void)ReadTextureType(handle); // the captured type 4 is a floating-point sampled 2D view
        return false;
    }
    u32 TextureShape(u32 offset) const {
        for (const auto& answer:textures) if (offset==0x20+answer.slot*8) return answer.type;
        throw std::runtime_error("texture shape answer required");
    }
    u32 ReadViewportTransformState() override { return 1; }
    u32 TextureBoundBuffer() const override { return texture_bank; }
    u32 LocalMemorySize() const override { return static_cast<u32>(sph.LocalMemorySize()) + sph.common3.shader_local_memory_crs_size; }
    u32 SharedMemorySize() const override { return 0; }
    std::array<u32,3> WorkgroupSize() const override { return {1,1,1}; }
    bool HasHLEMacroState() const override { return false; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32,u32) override { return std::nullopt; }
    void Dump(u64,u64) override {}
};

int main(int argc, char** argv) try {
    if (argc != 3 && argc != 7 && argc != 8 && !(argc >= 5 && argc <= 6 && std::string_view(argv[1])=="compile-ctl")) {
        std::cerr << "Usage: oot_mesh_xlate inspect native.bin | compile[-task] native.bin output.spv mesh_A mesh_B mesh_init [textures.context]\n"
                     "       oot_mesh_xlate compile-ctl native.bin native.ctl output.spv [textures.context]\n";
        return 2;
    }
    std::ifstream input(argv[2],std::ios::binary);
    if (!input) throw std::runtime_error("cannot read native input");
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)),{});
    if (std::string_view(argv[1])=="lower") { // any stage, as the runtime lowers it: compute has no SPH
        u32 first; std::memcpy(&first,bytes.data(),4);
        const bool graphics=(first&31)>=1 && (first&31)<=2 && ((first>>5)&31)==4;
        std::vector<Shader::Ampere::Instruction> all;
        Shader::Ampere::ProgramCaptureState state;
        for (size_t off=graphics?0x80:0;off+16<=bytes.size();off+=16) {
            Shader::Ampere::Instruction instruction;
            std::memcpy(instruction.words.data(),bytes.data()+off,16);
            all.push_back(instruction);
            if (state.Observe(instruction,static_cast<u32>(all.size()-1))!=Shader::Ampere::ProgramCaptureState::Result::Continue) break;
        }
        Shader::Ampere::LoweringOptions options;
        if (graphics) {
            const u32 type=(first>>10)&15;
            options.geometry=type==4;
            if (type==2) options.tessellation_stage=Shader::Ampere::TessellationStage::Control;
            else if (type==3) options.tessellation_stage=Shader::Ampere::TessellationStage::Evaluation;
        }
        auto result=Shader::Ampere::LowerToMaxwell(all,options);
        if (result) { fmt::print("lowered {} instructions\n",all.size()); return 0; }
        const auto at=result.error_offset/16;
        fmt::print("at {:#x}: {}",result.error_offset,result.error);
        if (at<all.size()) fmt::print(" [{:08x} {:08x} {:08x} {:08x}]",all[at].words[0],all[at].words[1],all[at].words[2],all[at].words[3]);
        fmt::print("\n");
        return 1;
    }
    if (bytes.size() < 0x90) throw std::runtime_error("native graphics input is too short");
    u32 w0; std::memcpy(&w0,bytes.data(),4);
    if (((w0 & 31) != 1 && (w0 & 31) != 2) || ((w0 >> 5) & 31) != 4) throw std::runtime_error("expected native SPH version 4");
    std::vector<Shader::Ampere::Instruction> native;
    Shader::Ampere::ProgramCaptureState capture;
    bool complete=false;
    for (size_t off=0x80;off+16<=bytes.size();off+=16) {
        Shader::Ampere::Instruction instruction;
        std::memcpy(instruction.words.data(),bytes.data()+off,16);
        native.push_back(instruction);
        auto result=capture.Observe(instruction,static_cast<u32>(native.size()-1));
        if (result==Shader::Ampere::ProgramCaptureState::Result::InvalidTarget) throw std::runtime_error("invalid native branch target");
        if (result==Shader::Ampere::ProgramCaptureState::Result::Complete) { complete=true; break; }
    }
    if (!complete) throw std::runtime_error("native input ends before the complete program was captured");
    auto lowered=Shader::Ampere::LowerToMaxwell(native);
    fmt::print("native {:016x}, SPH type {}, {} instructions: {}\n",Shader::Ampere::NativeProgramKey(native),
               (w0>>10)&15,native.size(),lowered ? "lowered with mesh context" : "lowering rejected");
    if (!lowered) throw std::runtime_error(fmt::format("at {:#x}: {}",lowered.error_offset,lowered.error));
    if (std::string_view(argv[1])=="inspect" && argc==3) return 0;
    const bool from_control=std::string_view(argv[1])=="compile-ctl";
    bool task=std::string_view(argv[1])=="compile-task";
    if (!from_control && ((!task && std::string_view(argv[1])!="compile") || (argc!=7 && argc!=8))) return 2;
    std::array<u32,3> registers;
    const char* output_path=argv[3];
    const char* context_path=argc==8 ? argv[7] : nullptr;
    std::optional<u32> primitive_header;
    if (from_control) {
        // The control section carries what the NVN driver turns into SET_MESH_SHADER_A/B and SET_MESH_INIT_SHADER
        // (facts: nxnative docs/reference/shaders.md, MIT): +0x56 stage, +0x58 threads, +0x5a payload bytes,
        // +0x60/+0x62 most vertices/primitives, +0x64 per-primitive SPH offset when +0x68 bit 7 is set.
        std::ifstream control_input(argv[3],std::ios::binary);
        std::vector<unsigned char> control((std::istreambuf_iterator<char>(control_input)),{});
        if (control.size()<0x6c) throw std::runtime_error("native control input is too short");
        auto u16=[&](size_t at){return static_cast<u32>(control[at])|static_cast<u32>(control[at+1])<<8;};
        const u32 kind=control[0x56];
        if (kind!=6 && kind!=7) throw std::runtime_error("control section is not a task or mesh stage");
        task=kind==6;
        const u32 threads=u16(0x58), lines=(u16(0x5a)+127)/128, vertices=u16(0x60), primitives=u16(0x62);
        if (!threads || threads>63 || lines>511) throw std::runtime_error("control section thread or payload size out of range");
        // +0x5f: the output primitive's vertex count (1 points, 2 lines, 3 triangles). SET_MESH_SHADER_A's topology field
        // takes 1, 2 or 4 for them (patches/mesh-output-topology.patch); the unpatched translator only made triangles, and
        // the two point stages of OoT's Hyrule grass shadow drew streaks from unwritten map bytes (docs/EFFECTS.md).
        // Task stages leave it 0 (no output primitives): triangles, as before.
        const u32 primitive_vertices=task ? 3u : control[0x5f];
        if (primitive_vertices<1 || primitive_vertices>3) throw std::runtime_error("control section output primitive out of range");
        const u32 topology=primitive_vertices==3 ? 4u : primitive_vertices;
        registers={topology|vertices<<4|primitives<<13, threads<<20, threads|lines<<12|lines<<22};
        if (control[0x68]&0x80) {
            u32 at; std::memcpy(&at,control.data()+0x64,4);
            if (at==0 || at+0x50+0x30>bytes.size()) throw std::runtime_error("per-primitive header outside the code area");
            primitive_header=at;
        }
        output_path=argv[4];
        context_path=argc==6 ? argv[5] : nullptr;
    }
    MeshEnvironment env(bytes,lowered,registers,task,context_path);
    if (primitive_header) env.SetPrimitiveHeader(bytes.data()+*primitive_header);
    env.assume_defaults=from_control;
    Shader::ObjectPool<Shader::IR::Inst> instructions;
    Shader::ObjectPool<Shader::IR::Block> blocks;
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow;
    Shader::Maxwell::Flow::CFG cfg(env,flow,env.StartAddress());
    Shader::HostTranslateInfo host;
    host.min_ssbo_alignment=16; host.support_float64=true; host.support_int64=true;
    auto program=Shader::Maxwell::TranslateProgram(instructions,blocks,env,cfg,host);
    fmt::print("mesh output lines: vertex {}, primitive {}, payload words {}\n", program.mesh.vertex_lines.size(),
               program.mesh.primitive_lines.size(), program.mesh.payload_words);
    std::set<u32> driver_offsets;
    for (const auto* block : program.post_order_blocks)
        for (const auto& instruction : block->Instructions()) {
            const auto op = instruction.GetOpcode();
            if (op != Shader::IR::Opcode::GetCbufU32 && op != Shader::IR::Opcode::GetCbufF32 &&
                op != Shader::IR::Opcode::GetCbufU32x2) continue;
            if (instruction.Arg(0).IsImmediate() && instruction.Arg(0).U32() == 0 && instruction.Arg(1).IsImmediate())
                driver_offsets.insert(instruction.Arg(1).U32());
        }
    fmt::print("driver-bank immediate reads:");
    for (u32 offset : driver_offsets) fmt::print(" {:#x}", offset);
    fmt::print("\n");
    Shader::Profile profile;
    profile.supported_spirv=0x00010500;
    profile.unified_descriptor_binding=true;
    profile.support_int64=true;
    profile.min_ssbo_alignment=16;
    // Stage's original 0x7f default predates Task/Mesh. Leaving their bits clear makes the backend return lane 0
    // and scalar votes for every invocation, corrupting the cooperative task/mesh output buffers.
    profile.supported_subgroup_stages=(1u << static_cast<u32>(Shader::Stage::Task)) |
                                      (1u << static_cast<u32>(Shader::Stage::Mesh));
    profile.support_vote=true;
    profile.support_shuffle_relative=true;
    profile.max_subgroup_size=32;
    auto spirv=Shader::Backend::SPIRV::EmitSPIRV(profile,program);
    if (!program.info.image_descriptors.empty() || !program.info.texture_buffer_descriptors.empty() ||
        !program.info.image_buffer_descriptors.empty())
        throw std::runtime_error("standalone binding export does not support storage or buffer images");
    // A texture whose handle the stage reads from a storage buffer (Lichii Info::handle_programs) is exported as an "H"
    // line: its handle program, which the runtime evaluates for each draw (vk_exec_mesh.h mesh_hp_eval).
    const auto by_program=[](const Shader::TextureDescriptor& r){
        return r.handle_program!=~0U && r.handle_program<512 && r.cbuf_index==0 && r.cbuf_offset==0 && !r.shift_left; };
    for (const auto& resource:program.info.texture_descriptors)
        if ((resource.type!=Shader::TextureType::Color2D && resource.type!=Shader::TextureType::ColorArray2D) ||
            resource.is_depth || resource.is_multisample ||
            resource.is_integer || resource.has_secondary || resource.count!=1 ||
            (!by_program(resource) && (resource.cbuf_index!=2 || resource.shift_left ||
            resource.handle_program!=~0U || resource.cbuf_offset<0x20 ||
            resource.cbuf_offset>=0x120 || (resource.cbuf_offset-0x20)%8)) ||
            (by_program(resource) && (resource.handle_program>=program.info.handle_programs.size() ||
            program.info.handle_programs[resource.handle_program].empty() ||
            program.info.handle_programs[resource.handle_program].size()>4096)))
            throw std::runtime_error(fmt::format("standalone binding export requires single combined float 2D/array textures: "
                "type {} depth {} multisample {} integer {} secondary {} bank {} offset {:#x} shift {} count {} handle program {:#x}",
                static_cast<u32>(resource.type),resource.is_depth,resource.is_multisample,resource.is_integer,resource.has_secondary,
                resource.cbuf_index,resource.cbuf_offset,resource.shift_left,resource.count,resource.handle_program));
    std::ofstream output(output_path,std::ios::binary);
    output.write(reinterpret_cast<const char*>(spirv.data()),spirv.size()*4);
    if (!output) throw std::runtime_error("cannot write SPIR-V output");
    std::ofstream metadata(std::string(output_path)+".bindings");
    metadata << "MSH3 " << (task ? 6 : 7) << ' ' << std::hex << Shader::Ampere::NativeProgramKey(native) << std::dec;
    for (u32 reg : registers) metadata << ' ' << reg;
    metadata << ' ' << program.info.constant_buffer_descriptors.size()+program.info.storage_buffers_descriptors.size()+
                      program.info.texture_descriptors.size() << '\n';
    u32 binding=0;
    for (const auto& resource : program.info.constant_buffer_descriptors) {
        metadata << "U " << binding << ' ' << resource.index << ' ' << resource.count << '\n';
        binding+=resource.count;
    }
    for (const auto& resource : program.info.storage_buffers_descriptors) {
        metadata << "S " << binding << ' ' << resource.cbuf_index << ' ' << resource.cbuf_offset << ' '
                 << resource.count << ' ' << resource.is_written << ' ' << resource.indirect_offset << '\n';
        binding+=resource.count;
    }
    for (const auto& resource:program.info.texture_descriptors) {
        if (by_program(resource)) {
            const auto& words=program.info.handle_programs[resource.handle_program];
            metadata << "H " << binding++ << ' ' << (resource.type==Shader::TextureType::ColorArray2D ? 5 : 4) << ' '
                     << words.size();
            for (u32 word:words) metadata << ' ' << word;
            metadata << '\n';
            fmt::print("texture binding {}: handle program {} ({} words):",binding-1,resource.handle_program,words.size());
            for (u32 word:words) fmt::print(" {}",word);
            fmt::print("\n");
            continue;
        }
        metadata << "T " << binding++ << ' ' << resource.cbuf_index << ' ' << resource.cbuf_offset << " 1 "
                 << env.TextureShape(resource.cbuf_offset) << '\n';
    }
    if (!metadata) throw std::runtime_error("cannot write binding metadata");
    fmt::print("{} SPIR-V words, {} vertices, {} primitives\n",spirv.size(),program.mesh.max_vertices,program.mesh.max_primitives);
    return 0;
} catch(const std::exception& error) {
    std::cerr << "mesh-xlate: " << error.what() << '\n';
    return 1;
}
