//
// Created by y1 on 2026-04-18.
//

#pragma once

#include <array>
#include <chrono>
#include <deque>
#include <functional>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include "global_io.slang"

struct SDL_Window;

class Camera;
class MLPDecoder;

struct VulkanTexture {
    VkImage       image{};
    VkImageView   view{};
    VmaAllocation allocation{};
};

struct VulkanBuffer {
    VkBuffer                  buffer{};
    VmaAllocation             allocation{};
    [[maybe_unused]] uint32_t vertexCount{};
};

class VulkanState {
public:
    VulkanState()                               = delete;
    VulkanState(const VulkanState &)            = delete;
    VulkanState &operator=(const VulkanState &) = delete;
    VulkanState(VulkanState &&)                 = delete;
    VulkanState &operator=(VulkanState &&)      = delete;

    VulkanState(SDL_Window *window, const Camera &camera);

    ~VulkanState();

    void WaitIdle() const;
    void Run();

    template<class Func>
    void ImmediateSubmit(Func &&func) {
        ResetCommandBuffer(m_immediateCommandBuffer, 0);

        BeginCommandBuffer(m_immediateCommandBuffer, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
        func(m_immediateCommandBuffer);
        EndCommandBuffer(m_immediateCommandBuffer);

        SubmitToQueue(m_immediateCommandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, m_immediateFence, VK_NULL_HANDLE, VK_NULL_HANDLE);

        WaitAndRestFence(m_immediateFence);
    }

    void PushToDeletionQueue(std::function<void()> &&func) { m_deletionQueue.Push(std::move(func)); }

    [[nodiscard]] VkDevice GetDevice() const { return m_device; }

    [[nodiscard]] VmaAllocator GetAllocator() const { return m_allocator; }

public:


private:
    SDL_Window   *m_window{nullptr};
    const Camera &m_camera;

    shader_io::GlobalUniforms    m_globalUniforms{};
    shader_io::ReconstructParams m_reconstructParams{};
    VulkanBuffer                 m_reconstructParamsBuffer{};

    // Device
    static constexpr uint32_t kQueueFamilyIndex{0};

    VkInstance               m_instance{};
    VkDebugUtilsMessengerEXT m_debugUtilsMessenger{};
    VkPhysicalDevice         m_physicalDevice{};
    VkDevice                 m_device{};
    VmaAllocator             m_allocator{};
    VkQueue                  m_queue{};

    // Present
    static constexpr VkFormat         kPresentFormat{VK_FORMAT_B8G8R8A8_SRGB};
    static constexpr VkColorSpaceKHR  kPresentColorSpace{VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    static constexpr VkPresentModeKHR kPresentMode{VK_PRESENT_MODE_FIFO_KHR};
    static constexpr VkFormat         kSceneColorFormat{VK_FORMAT_R16G16B16A16_SFLOAT};
    static constexpr VkFormat         kSceneDepthFormat{VK_FORMAT_D32_SFLOAT};
    static constexpr uint32_t         kMinSwapchainImage{2};
    static constexpr uint32_t         kMaxSwapchainImage{8};
    static constexpr uint32_t         kMaxFramesInFlight{2};

    VkSurfaceKHR                                  m_surface{};
    VkSwapchainKHR                                m_swapchain{};
    std::array<VulkanTexture, kMaxSwapchainImage> m_swapchainTextures{};
    VkExtent2D                                    m_swapchainExtent{};
    uint32_t                                      m_swapchainCount{0};
    uint32_t                                      m_presentImageIndex{0};
    std::array<VkSemaphore, kMaxSwapchainImage>   m_renderSemaphores{};

    std::array<VkCommandBuffer, kMaxFramesInFlight> m_commandBuffers{};
    std::array<VkSemaphore, kMaxFramesInFlight>     m_presentSemaphores{};
    std::array<VkFence, kMaxFramesInFlight>         m_fences{};

    uint32_t m_currentFrameIndex{0};

    // Commands
    static constexpr uint64_t kPointOneSecond = 100000000;

    VkCommandPool   m_commandPool{};
    VkCommandBuffer m_immediateCommandBuffer{};
    VkFence         m_immediateFence{};

    // Profiling
    static constexpr size_t kFrameHistorySize = 128;

    VkQueryPool m_queryPool{};
    float       m_timestampPeriod{1.0f};
    float       m_pbrTime{0.0f};
    float       m_neuralForwardTime{0.0f};
    float       m_computeReconstructTime{0.0f};
    float       m_lastFrameTime{0.0f};

    std::array<uint32_t, kMaxFramesInFlight> m_passModeInFlight{};

    struct PsnrEntry {
        const char *label{nullptr};
        float       psnr{0.0f};
    };

    std::array<PsnrEntry, 5> m_perOutputPsnr{};
    float                    m_overallPsnr{0.0f};

    std::array<float, kFrameHistorySize>           m_frameTimeHistory{};
    uint32_t                                       m_frameHistoryIndex{0};
    uint64_t                                       m_totalFrameCount{0};
    std::chrono::high_resolution_clock::time_point m_lastFrameStart{};


    void InitState();

    static void ResetCommandBuffer(const VkCommandBuffer &commandBuffer, VkCommandBufferResetFlags flags);
    static void BeginCommandBuffer(const VkCommandBuffer &commandBuffer, VkCommandBufferUsageFlags flags);
    static void EndCommandBuffer(const VkCommandBuffer &commandBuffer);

    void WaitAndRestFence(const VkFence &fence, uint64_t timeout = kPointOneSecond) const;
    void SubmitToQueue(
        const VkCommandBuffer &commandBuffer,
        VkPipelineStageFlags   stage,
        const VkFence         &fence,
        const VkSemaphore     &waitSemaphore,
        const VkSemaphore     &signalSemaphore
    ) const;

    [[nodiscard]] VkExtent2D CalcSwapchainExtent(const VkSurfaceCapabilitiesKHR &capabilities) const;


private:
    // Resources
    VkSampler m_defaultSampler{};
    VkSampler m_defaultNearestSampler{};

    std::array<VulkanTexture, kMaxFramesInFlight> m_sceneColors{};
    std::array<VulkanTexture, kMaxFramesInFlight> m_sceneDepths{};

    VulkanBuffer  m_helmetVertexBuffer{};
    VulkanTexture m_helmetAlbedo{};
    VulkanTexture m_helmetAO{};
    VulkanTexture m_helmetEmissive{};
    VulkanTexture m_helmetMetallicRoughness{};
    VulkanTexture m_helmetNormal{};

    VulkanBuffer  m_skyboxVertexBuffer{};
    VulkanTexture m_skyboxTexture{};
    VulkanTexture m_skyboxIrradiance{};
    VulkanTexture m_skyboxSpecular{};
    VulkanTexture m_brdfLut{};

    std::unique_ptr<MLPDecoder> m_mlp{};

    VulkanTexture m_computeOutAlbedo{};
    VulkanTexture m_computeOutNormal{};
    VulkanTexture m_computeOutAO{};
    VulkanTexture m_computeOutMetallicRoughness{};
    VulkanTexture m_computeOutEmissive{};

    // TODO: fragment out

    void InitResources();

private:
    // Pipeline
    struct VulkanPipeline {
        VkPipelineLayout layout{};
        VkPipeline       pipeline{};
    };

    VulkanPipeline        m_forward{};
    VkDescriptorSetLayout m_forwardSetLayout{};

    VulkanPipeline        m_skybox{};
    VkDescriptorSetLayout m_skyboxSetLayout{};

    VulkanPipeline        m_reconstruct{};
    VkDescriptorSetLayout m_reconstructSetLayout{};

    VulkanPipeline        m_forwardNeural{};
    VkDescriptorSetLayout m_forwardNeuralSetLayout{};

    VkDescriptorPool m_descriptorPool{};

    uint32_t m_passMode{0};

    void InitPipelines();

    void ForwardPBR(const VkCommandBuffer &commandBuffer, const VulkanTexture &sceneColor, const VulkanTexture &sceneDepth);
    void Skybox(const VkCommandBuffer &commandBuffer, const VulkanTexture &sceneColor, const VulkanTexture &sceneDepth) const;
    void ForwardNeural(const VkCommandBuffer &commandBuffer, const VulkanTexture &sceneColor, const VulkanTexture &sceneDepth);
    void ImGuiPass(const VkCommandBuffer &commandBuffer, const VulkanTexture &sceneColor) const;
    void ReconstructComputePass();

    void DrawImGuiContent();


private:
    struct DeletionQueue {
        std::deque<std::function<void()>> deletors{};

        void Push(std::function<void()> &&func) { deletors.push_front(std::move(func)); }

        void Flush() {
            for (auto &func: deletors) {
                func();
            }
            deletors.clear();
        }
    };

    DeletionQueue m_deletionQueue{};

private:
    PFN_vkCreateDebugUtilsMessengerEXT  vkCreateDebugUtilsMessengerEXT{};
    PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT{};
    PFN_vkCmdPushDescriptorSetKHR       vkCmdPushDescriptorSetKHR{};
};