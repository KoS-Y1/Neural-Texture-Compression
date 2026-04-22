//
// Created by y1 on 2026-04-20.
//

#include "MLPDecoder.h"

#include <simdjson.h>

#include "Debug.h"
#include "FileSystem.h"

namespace {
NtcHeader LoadHeader(std::string_view path) {
    const std::string pathStr(path);
    auto              json = simdjson::padded_string::load(pathStr);
    DebugCheckCritical(json.error() == simdjson::SUCCESS, "Failed to load NTC header {}", pathStr);

    simdjson::ondemand::parser   parser;
    simdjson::ondemand::document doc = parser.iterate(json);

    NtcHeader header;
    header.version = static_cast<uint32_t>(static_cast<uint64_t>(doc["version"].get_uint64()));

    auto readLatent = [](simdjson::ondemand::object obj) -> NtcLatentInfo {
        NtcLatentInfo info;

        std::string_view sv = obj["file"].get_string();
        info.file           = std::string(sv);
        info.width          = static_cast<uint32_t>(static_cast<uint64_t>(obj["width"].get_uint64()));
        info.height         = static_cast<uint32_t>(static_cast<uint64_t>(obj["height"].get_uint64()));
        info.channels       = static_cast<uint32_t>(static_cast<uint64_t>(obj["channels"].get_uint64()));
        sv                  = obj["layout"].get_string();
        info.layout         = std::string(sv);
        sv                  = obj["sample_format"].get_string();
        info.sampleFormat   = std::string(sv);
        info.sourceBits     = static_cast<uint32_t>(static_cast<uint64_t>(obj["source_bits"].get_uint64()));

        return info;
    };

    header.latentHi = readLatent(doc["latent_hi"].get_object());
    header.latentLo = readLatent(doc["latent_lo"].get_object());

    {
        simdjson::ondemand::object pe = doc["positional_encoder"].get_object();
        header.peNumFreq              = static_cast<uint32_t>(static_cast<uint64_t>(pe["num_freq"].get_uint64()));
        header.peOutDim               = static_cast<uint32_t>(static_cast<uint64_t>(pe["out_dim"].get_uint64()));
    }

    {
        simdjson::ondemand::object mlp = doc["mlp"].get_object();

        std::string_view sv     = mlp["file"].get_string();
        header.mlp.file         = std::string(sv);
        sv                      = mlp["dtype"].get_string();
        header.mlp.dtype        = std::string(sv);
        sv                      = mlp["weight_layout"].get_string();
        header.mlp.weightLayout = std::string(sv);
        header.mlp.inputDim     = static_cast<uint32_t>(static_cast<uint64_t>(mlp["input_dim"].get_uint64()));
        header.mlp.outputDim    = static_cast<uint32_t>(static_cast<uint64_t>(mlp["output_dim"].get_uint64()));
        header.mlp.totalBytes   = static_cast<uint64_t>(mlp["total_bytes"].get_uint64());

        auto parseActivation = [](std::string_view s) -> NtcActivation {
            if (s == "relu") return NtcActivation::Relu;
            if (s == "sigmoid") return NtcActivation::Sigmoid;
            if (s == "none") return NtcActivation::None;
            DebugCheckCritical(false, "Unknown NTC activation '{}'", std::string(s));
            return NtcActivation::None;
        };

        for (auto layer: mlp["layers"]) {
            NtcMlpLayerInfo li;
            li.inDim        = static_cast<uint32_t>(static_cast<uint64_t>(layer["in"].get_uint64()));
            li.outDim       = static_cast<uint32_t>(static_cast<uint64_t>(layer["out"].get_uint64()));
            sv              = layer["activation"].get_string();
            li.activation   = parseActivation(sv);
            li.weightOffset = static_cast<uint64_t>(layer["weight_offset"].get_uint64());
            li.weightSize   = static_cast<uint64_t>(layer["weight_size"].get_uint64());
            li.biasOffset   = static_cast<uint64_t>(layer["bias_offset"].get_uint64());
            li.biasSize     = static_cast<uint64_t>(layer["bias_size"].get_uint64());
            header.mlp.layers.push_back(li);
        }
    }

    return header;
}
} // namespace

MLPDecoder::MLPDecoder()
    : m_header(LoadHeader(kMLPLoadFile)) {
    DebugInfo(
        "Loaded NTC header v{} ({} MLP layers, latent_hi {}x{}x{}, latent_lo {}x{}x{})",
        m_header.version,
        m_header.mlp.layers.size(),
        m_header.latentHi.width,
        m_header.latentHi.height,
        m_header.latentHi.channels,
        m_header.latentLo.width,
        m_header.latentLo.height,
        m_header.latentLo.channels
    );
    m_mlpParams.w0 = m_header.mlp.layers[0].weightOffset;
    m_mlpParams.b0 = m_header.mlp.layers[0].biasOffset;
    m_mlpParams.w1 = m_header.mlp.layers[1].weightOffset;
    m_mlpParams.b1 = m_header.mlp.layers[1].biasOffset;
    m_mlpParams.w2 = m_header.mlp.layers[2].weightOffset;
    m_mlpParams.b2 = m_header.mlp.layers[2].biasOffset;
}

void MLPDecoder::Load(VulkanState &state) {
    // Default sampler
    {
        const VkSamplerCreateInfo infoSampler{
            .sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter    = VK_FILTER_LINEAR,
            .minFilter    = VK_FILTER_LINEAR,
            .mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .maxLod       = VK_LOD_CLAMP_NONE,
        };
        VkResult result = vkCreateSampler(state.GetDevice(), &infoSampler, nullptr, &m_defaultSampler);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to create default sampler");
        state.PushToDeletionQueue([this, &state]() { vkDestroySampler(state.GetDevice(), m_defaultSampler, nullptr); });
    }

    // Latents
    {
        struct TextureUpload {
            std::string    path;
            VkFormat       format;
            VulkanTexture *texture;
            uint32_t       width{0};
            uint32_t       height{0};
        };

        const std::vector<TextureUpload> uploads{
            {kMLPDir + m_header.latentLo.file, VK_FORMAT_R8G8B8A8_UNORM, &m_latentLo, m_header.latentLo.width, m_header.latentLo.height},
            {kMLPDir + m_header.latentHi.file, VK_FORMAT_R8G8B8A8_UNORM, &m_latentHi, m_header.latentHi.width, m_header.latentHi.height},
        };

        for (const TextureUpload &upload: uploads) {
            DebugInfo("Loading latent from file {}", upload.path);
            const std::vector<uint8_t> source = ntc::Read(upload.path);

            const uint32_t     width      = upload.width;
            const uint32_t     height     = upload.height;
            VulkanTexture     *texture    = upload.texture;
            const VkDeviceSize bufferSize = static_cast<VkDeviceSize>(width) * static_cast<VkDeviceSize>(height) * 4;
            DebugCheckCritical(source.size() == bufferSize, "{} size {} does not match expected {}", upload.path, source.size(), bufferSize);

            VkBufferCreateInfo infoStagingBuffer{
                .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size        = bufferSize,
                .usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            };
            VmaAllocationCreateInfo infoStagingAlloc{
                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                .usage = VMA_MEMORY_USAGE_AUTO,
            };
            VkBuffer          stagingBuffer{};
            VmaAllocation     stagingAllocation{};
            VmaAllocationInfo stagingAllocationInfo{};
            VkResult          result = vmaCreateBuffer(
                state.GetAllocator(),
                &infoStagingBuffer,
                &infoStagingAlloc,
                &stagingBuffer,
                &stagingAllocation,
                &stagingAllocationInfo
            );
            DebugCheckCritical(result == VK_SUCCESS, "Failed to create staging buffer for {}", upload.path);
            std::memcpy(stagingAllocationInfo.pMappedData, source.data(), bufferSize);

            VkImageCreateInfo infoImage{
                .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .imageType     = VK_IMAGE_TYPE_2D,
                .format        = upload.format,
                .extent        = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1},
                .mipLevels     = 1,
                .arrayLayers   = 1,
                .samples       = VK_SAMPLE_COUNT_1_BIT,
                .tiling        = VK_IMAGE_TILING_OPTIMAL,
                .usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
            VmaAllocationCreateInfo infoImageAlloc{
                .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            };
            result = vmaCreateImage(state.GetAllocator(), &infoImage, &infoImageAlloc, &texture->image, &texture->allocation, nullptr);
            DebugCheckCritical(result == VK_SUCCESS, "Failed to create image for {}", upload.path);
            state.PushToDeletionQueue([this, texture, &state]() { vmaDestroyImage(state.GetAllocator(), texture->image, texture->allocation); });

            state.ImmediateSubmit([&, texture](VkCommandBuffer commandBuffer) {
                VkImageMemoryBarrier2 barrierToTransfer{
                    .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                    .srcStageMask     = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                    .srcAccessMask    = 0,
                    .dstStageMask     = VK_PIPELINE_STAGE_2_COPY_BIT,
                    .dstAccessMask    = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    .oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    .image            = texture->image,
                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                };
                VkDependencyInfo depToTransfer{
                    .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                    .imageMemoryBarrierCount = 1,
                    .pImageMemoryBarriers    = &barrierToTransfer,
                };
                vkCmdPipelineBarrier2(commandBuffer, &depToTransfer);

                VkBufferImageCopy region{
                    .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                    .imageExtent      = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1},
                };
                vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, texture->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

                VkImageMemoryBarrier2 barrierToShader{
                    .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                    .srcStageMask     = VK_PIPELINE_STAGE_2_COPY_BIT,
                    .srcAccessMask    = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    .dstStageMask     = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    .dstAccessMask    = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                    .oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    .newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    .image            = texture->image,
                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                };
                VkDependencyInfo infoDependency{
                    .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                    .imageMemoryBarrierCount = 1,
                    .pImageMemoryBarriers    = &barrierToShader,
                };
                vkCmdPipelineBarrier2(commandBuffer, &infoDependency);
            });

            vmaDestroyBuffer(state.GetAllocator(), stagingBuffer, stagingAllocation);

            VkImageViewCreateInfo infoView{
                .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image    = texture->image,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format   = upload.format,
                .components =
                    {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY},
                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
            };
            result = vkCreateImageView(state.GetDevice(), &infoView, nullptr, &texture->view);
            DebugCheckCritical(result == VK_SUCCESS, "Failed to create image view for {}", upload.path);
            state.PushToDeletionQueue([this, texture, &state]() { vkDestroyImageView(state.GetDevice(), texture->view, nullptr); });
        }
    }

    // MLP
    {
        const std::string path = kMLPDir + m_header.mlp.file;
        DebugInfo("Loading MLP from file {}", path);
        const std::vector<uint8_t> data       = ntc::Read(path);
        const uint32_t             bufferSize = m_header.mlp.totalBytes;
        VkBufferCreateInfo         infoStagingBuffer{
            .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size        = bufferSize,
            .usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        VmaAllocationCreateInfo infoStagingAlloc{
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO,
        };
        VkBuffer          stagingBuffer{};
        VmaAllocation     stagingAllocation{};
        VmaAllocationInfo stagingAllocationInfo{};
        VkResult          result =
            vmaCreateBuffer(state.GetAllocator(), &infoStagingBuffer, &infoStagingAlloc, &stagingBuffer, &stagingAllocation, &stagingAllocationInfo);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to create helmet vertex staging buffer");
        std::memcpy(stagingAllocationInfo.pMappedData, data.data(), bufferSize);

        VkBufferCreateInfo infoBuffer{
            .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size        = bufferSize,
            .usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        VmaAllocationCreateInfo infoBufferAlloc{
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        };
        result = vmaCreateBuffer(state.GetAllocator(), &infoBuffer, &infoBufferAlloc, &m_mlpBuffer, &m_mlpBufferAlloc, nullptr);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to create MLP storage buffer");
        state.PushToDeletionQueue([&]() { vmaDestroyBuffer(state.GetAllocator(), m_mlpBuffer, m_mlpBufferAlloc); });

        state.ImmediateSubmit([&](VkCommandBuffer commandBuffer) {
            VkBufferCopy region{.size = bufferSize};
            vkCmdCopyBuffer(commandBuffer, stagingBuffer, m_mlpBuffer, 1, &region);
        });

        vmaDestroyBuffer(state.GetAllocator(), stagingBuffer, stagingAllocation);
    }

    m_outputResolution = m_header.latentHi.width * kCompressionRatio;
    DebugInfo("Reconstruct output resolution: {}x{}", m_outputResolution, m_outputResolution);
}