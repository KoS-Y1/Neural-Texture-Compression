//
// Created by y1 on 2026-04-18.
//

#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <slang-com-ptr.h>
#include <vulkan/vulkan_core.h>

class ShaderCompiler {
public:
    ShaderCompiler();
    ShaderCompiler(const ShaderCompiler &)            = delete;
    ShaderCompiler(ShaderCompiler &&)                 = delete;
    ShaderCompiler &operator=(const ShaderCompiler &) = delete;
    ShaderCompiler &operator=(ShaderCompiler &&)      = delete;

    ~ShaderCompiler() = default;

    [[nodiscard]] const uint32_t                                            *GetSpirv(const std::string &filePath) const;
    [[nodiscard]] size_t                                                     GetSpirvSize(const std::string &filePath) const;
    [[nodiscard]] std::vector<std::pair<std::string, VkShaderStageFlagBits>> GetEntryPoints(const std::string &filePath) const;

private:
    static constexpr const char *kShaderSearchPath{"../runtime/shaders/"};
    static constexpr const char *kLoadFile{"../runtime/shaders/shaders.json"};

    Slang::ComPtr<slang::IGlobalSession> m_globalSession{};
    Slang::ComPtr<slang::ISession>       m_session{};

    std::unordered_map<std::string, Slang::ComPtr<ISlangBlob>>            m_spirvs{};
    std::unordered_map<std::string, Slang::ComPtr<slang::IComponentType>> m_shaderPrograms{};

    std::string m_diagnosticMessage;

    bool Compile(const std::string &filePath);

    void LogAndAppendDiagnostics(slang::IBlob *diagnostics);

    static VkShaderStageFlagBits ShaderStageSlangToVulkan(SlangStage stage);
};