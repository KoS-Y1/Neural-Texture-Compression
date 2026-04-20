//
// Created by y1 on 2026-04-18.
//

#include "ShaderCompiler.h"

#include <filesystem>

#include <simdjson.h>
#include <slang.h>

#include "Debug.h"
#include "FileSystem.h"

ShaderCompiler::ShaderCompiler() {
    slang::createGlobalSession(m_globalSession.writeRef());

    slang::TargetDesc target{
        .format                      = SLANG_SPIRV,
        .profile                     = m_globalSession->findProfile("spirv_1_6+vulkan_1_4"),
        .forceGLSLScalarBufferLayout = true
    };

    std::vector<slang::CompilerOptionEntry> options{
        {slang::CompilerOptionName::EmitSpirvDirectly,       {slang::CompilerOptionValueKind::Int, 1}},
        {slang::CompilerOptionName::VulkanUseEntryPointName, {slang::CompilerOptionValueKind::Int, 1}},
        {slang::CompilerOptionName::DebugInformation,        {slang::CompilerOptionValueKind::Int, 1}},
    };

    slang::SessionDesc session{
        .targets                  = &target,
        .targetCount              = 1,
        .searchPaths              = &kShaderSearchPath,
        .searchPathCount          = 1,
        .compilerOptionEntries    = options.data(),
        .compilerOptionEntryCount = static_cast<uint32_t>(options.size()),
    };
    m_globalSession->createSession(session, m_session.writeRef());

    auto json = simdjson::padded_string::load(kLoadFile);
    DebugCheckCritical(json.error() == simdjson::SUCCESS, "Failed to load {}", kLoadFile);

    simdjson::ondemand::parser parser;
    auto                       doc = parser.iterate(json);

    for (auto shader: doc["shaders"]) {
        std::string_view sv = shader.get_string();
        std::string      key{sv};
        DebugCheckCritical(Compile(key), "Failed to load {}", key);
    }
}

const uint32_t *ShaderCompiler::GetSpirv(const std::string &filePath) const {
    auto pair = m_spirvs.find(filePath);
    DebugCheckCritical(pair != m_spirvs.end(), "{} does not exist (SPIRV)", filePath);
    return static_cast<const uint32_t *>(pair->second->getBufferPointer());
}

size_t ShaderCompiler::GetSpirvSize(const std::string &filePath) const {
    auto pair = m_spirvs.find(filePath);
    DebugCheckCritical(pair != m_spirvs.end(), "{} does not exist (SPIRV size)", filePath);
    return pair->second->getBufferSize();
}

std::vector<std::pair<std::string, VkShaderStageFlagBits>> ShaderCompiler::GetEntryPoints(const std::string &filePath) const {
    auto pair = m_shaderPrograms.find(filePath);
    DebugCheckCritical(pair != m_shaderPrograms.end(), "{} does not exist (entry point)", filePath);

    slang::ProgramLayout *layout = pair->second->getLayout();

    std::vector<std::pair<std::string, VkShaderStageFlagBits>> entryPoints;
    SlangUInt                                                  entryPointCount = layout->getEntryPointCount();
    for (SlangUInt i = 0; i < entryPointCount; ++i) {
        slang::EntryPointLayout *entryPoint = layout->getEntryPointByIndex(i);
        std::string              name       = entryPoint->getName();
        VkShaderStageFlagBits    stage      = ShaderStageSlangToVulkan(entryPoint->getStage());
        entryPoints.emplace_back(name, stage);
    }
    return entryPoints;
}

bool ShaderCompiler::Compile(const std::string &filePath) {
    DebugInfo("Loading shader from file {}", filePath);
    m_diagnosticMessage.clear();

    std::filesystem::path path{filePath};
    const std::string     fileName    = path.filename().string();
    const std::string     moduleName  = path.stem().string();
    const std::string     slangSource = ntc::Read(filePath);

    Slang::ComPtr<slang::IBlob> diagnostics;

    // Load module
    Slang::ComPtr<slang::IModule> module;
    module = m_session->loadModuleFromSourceString(moduleName.c_str(), fileName.c_str(), slangSource.c_str(), diagnostics.writeRef());
    LogAndAppendDiagnostics(diagnostics);

    if (!module) {
        return false;
    }

    // Entry point shader reflection
    const SlangInt32                               definedEntryPointCount = module->getDefinedEntryPointCount();
    std::vector<Slang::ComPtr<slang::IEntryPoint>> entryPoints(definedEntryPointCount);
    std::vector<slang::IComponentType *>           components;
    components.reserve(definedEntryPointCount + 1);
    components.push_back(module);
    for (SlangInt32 i = 0; i < definedEntryPointCount; ++i) {
        module->getDefinedEntryPoint(i, entryPoints[i].writeRef());
        components.push_back(entryPoints[i]);
    }

    // Compose modules with entry points
    Slang::ComPtr<slang::IComponentType> composedProgram;
    SlangResult                          result = m_session->createCompositeComponentType(
        components.data(),
        static_cast<int64_t>(components.size()),
        composedProgram.writeRef(),
        diagnostics.writeRef()
    );
    LogAndAppendDiagnostics(diagnostics);
    if (SLANG_FAILED(result) || !composedProgram) {
        return false;
    }

    // From composite component to linked program
    Slang::ComPtr<slang::IComponentType> linkedProgram;
    result = composedProgram->link(linkedProgram.writeRef(), diagnostics.writeRef());
    LogAndAppendDiagnostics(diagnostics);
    if (SLANG_FAILED(result) || !linkedProgram) {
        return false;
    }

    // From linked program to SPIR-V
    Slang::ComPtr<ISlangBlob> spirv;
    result = linkedProgram->getTargetCode(0, spirv.writeRef(), diagnostics.writeRef());
    LogAndAppendDiagnostics(diagnostics);
    if (SLANG_FAILED(result) || !spirv) {
        return false;
    }

    m_spirvs.emplace(filePath, spirv);
    m_shaderPrograms.emplace(filePath, linkedProgram);

    return true;
}

void ShaderCompiler::LogAndAppendDiagnostics(slang::IBlob *diagnostics) {
    if (diagnostics) {
        const char *message = static_cast<const char *>(diagnostics->getBufferPointer());
        DebugInfo("\n{}\n", message);

        if (m_diagnosticMessage.empty()) {
            m_diagnosticMessage += '\n';
        }
        m_diagnosticMessage += message;
    }
}

VkShaderStageFlagBits ShaderCompiler::ShaderStageSlangToVulkan(SlangStage stage) {
    switch (stage) {
    case SLANG_STAGE_VERTEX:
        return VK_SHADER_STAGE_VERTEX_BIT;
    case SLANG_STAGE_FRAGMENT:
        return VK_SHADER_STAGE_FRAGMENT_BIT;
    case SLANG_STAGE_GEOMETRY:
        return VK_SHADER_STAGE_GEOMETRY_BIT;
    case SLANG_STAGE_COMPUTE:
        return VK_SHADER_STAGE_COMPUTE_BIT;
    default:
        DebugError("Shader stage not supported!");
        return static_cast<VkShaderStageFlagBits>(0);
    }
}