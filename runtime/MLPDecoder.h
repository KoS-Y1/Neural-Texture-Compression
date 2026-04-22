//
// Created by y1 on 2026-04-20.
//

#pragma once

#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "VulkanState.h"

#include "global_io.slang"

enum class NtcActivation : uint8_t {
    None = 0,
    Relu,
    Sigmoid,
};

struct NtcLatentInfo {
    std::string file{};
    uint32_t    width{0};
    uint32_t    height{0};
    uint32_t    channels{0};
    std::string dtype{};
    std::string layout{};
    std::string sampleFormat{};
    uint32_t    sourceBits{0};
};

struct NtcMlpLayerInfo {
    uint32_t      inDim{0};
    uint32_t      outDim{0};
    NtcActivation activation{NtcActivation::None};
    uint64_t      weightOffset{0};
    uint64_t      weightSize{0};
    uint64_t      biasOffset{0};
    uint64_t      biasSize{0};
};

struct NtcMlpInfo {
    std::string                  file{};
    std::string                  dtype{};
    std::string                  weightLayout{};
    uint32_t                     inputDim{0};
    uint32_t                     outputDim{0};
    uint64_t                     totalBytes{0};
    std::vector<NtcMlpLayerInfo> layers{};
};

struct NtcHeader {
    uint32_t      version{0};
    NtcLatentInfo latentHi{};
    NtcLatentInfo latentLo{};
    uint32_t      peNumFreq{0};
    uint32_t      peOutDim{0};
    NtcMlpInfo    mlp{};
};

class MLPDecoder {
public:
    MLPDecoder();

    MLPDecoder(const MLPDecoder &)            = delete;
    MLPDecoder &operator=(const MLPDecoder &) = delete;
    MLPDecoder(MLPDecoder &&)                 = delete;
    MLPDecoder &operator=(MLPDecoder &&)      = delete;

    void Load(VulkanState &state);

    [[nodiscard]] const NtcHeader            &GetHeader() const { return m_header; }
    [[nodiscard]] VkSampler                   GetSampler() const { return m_defaultSampler; }
    [[nodiscard]] const VulkanTexture        &GetLatentHi() const { return m_latentHi; }
    [[nodiscard]] const VulkanTexture        &GetLatentLo() const { return m_latentLo; }
    [[nodiscard]] VkBuffer                    GetMlpBuffer() const { return m_mlpBuffer; }
    [[nodiscard]] const shader_io::MlpParams &GetMlpParams() const { return m_mlpParams; }

    [[nodiscard]] uint32_t             GetOutputResolution() const { return m_outputResolution; }


private:
    static constexpr const char *kMLPDir      = "../assets/export/runtime/";
    static constexpr const char *kMLPLoadFile = "../assets/export/runtime/ntc.json";

    static constexpr uint32_t kCompressionRatio = 4;

    NtcHeader m_header{};

private:
    VkSampler m_defaultSampler{};

    VulkanTexture m_latentHi{};
    VulkanTexture m_latentLo{};

private:
    VkBuffer      m_mlpBuffer{};
    VmaAllocation m_mlpBufferAlloc{};

    shader_io::MlpParams m_mlpParams{};

private:
    int32_t      m_outputResolution{0};
};