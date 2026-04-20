//
// Created by y1 on 2026-04-18.
//

#include "VulkanState.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <vector>

#include <SDL3/SDL_video.h>
#include <SDL3/SDL_vulkan.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>
#include <stb_image.h>

#include "Camera.h"
#include "Debug.h"
#include "FileSystem.h"
#include "ShaderCompiler.h"

static VKAPI_ATTR VkBool32 VKAPI_CALL VulkanDebugCallback(
    [[maybe_unused]] VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    [[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT        messageTypes,
    const VkDebugUtilsMessengerCallbackDataEXT             *pCallbackData,
    [[maybe_unused]] void                                  *pUserData
) {
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        DebugError("Vulkan: {}", pCallbackData->pMessage);
    } else if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        DebugWarning("Vulkan: {}", pCallbackData->pMessage);
    } else if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
        DebugInfo("Vulkan: {}", pCallbackData->pMessage);
    } else {
        DebugVerbose("Vulkan: {}", pCallbackData->pMessage);
    }
    return VK_FALSE;
}

VulkanState::VulkanState(SDL_Window *window, const Camera &camera)
    : m_window(window)
    , m_camera(camera) {
    // Instance
    {
        VkApplicationInfo infoApp{
            .sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName   = "NTCApp",
            .applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),
            .pEngineName        = "NTCEngine",
            .apiVersion         = VK_API_VERSION_1_4
        };

        // Enable validation layer
        std::vector<const char *> layers{
            "VK_LAYER_KHRONOS_validation",
        };

#if defined(NDEBUG)
        constexpr bool enableValidationLayers = false;
#else
        constexpr bool enableValidationLayers = true;
#endif

        uint32_t                  sdlExtensionCount{0};
        const char * const       *sdlExtensions = SDL_Vulkan_GetInstanceExtensions(&sdlExtensionCount);
        std::vector<const char *> extensions{sdlExtensions, sdlExtensionCount + sdlExtensions};
        extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        VkInstanceCreateInfo infoInstance{
            .sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo        = &infoApp,
            .enabledExtensionCount   = static_cast<uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data(),
        };

        if (enableValidationLayers) {
            infoInstance.enabledLayerCount   = static_cast<uint32_t>(layers.size());
            infoInstance.ppEnabledLayerNames = layers.data();
            DebugInfo("Enable Validation Layers");
        }

        VkResult result = vkCreateInstance(&infoInstance, nullptr, &m_instance);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to create VkInstance");
        m_deletionQueue.Push([&]() { vkDestroyInstance(m_instance, nullptr); });
    }

    // Debug messenger
    {
        vkCreateDebugUtilsMessengerEXT =
            reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT"));
        vkDestroyDebugUtilsMessengerEXT =
            reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT"));
        DebugCheckCritical(vkCreateDebugUtilsMessengerEXT != nullptr, "Failed to load vkCreateDebugUtilsMessengerEXT");
        DebugCheckCritical(vkDestroyDebugUtilsMessengerEXT != nullptr, "Failed to load vkDestroyDebugUtilsMessengerEXT");

        const VkDebugUtilsMessengerCreateInfoEXT infoMessenger{
            .sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = VulkanDebugCallback,
        };

        VkResult result = vkCreateDebugUtilsMessengerEXT(m_instance, &infoMessenger, nullptr, &m_debugUtilsMessenger);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to create debug messenger");
        m_deletionQueue.Push([&]() { vkDestroyDebugUtilsMessengerEXT(m_instance, m_debugUtilsMessenger, nullptr); });
    }

    // Physical device
    {
        uint32_t physicalDeviceCount{0};

        VkResult result = vkEnumeratePhysicalDevices(m_instance, &physicalDeviceCount, nullptr);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to enumerate physical devices");
        DebugCheckCritical(physicalDeviceCount > 0, "0 physical device available");

        std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
        result = vkEnumeratePhysicalDevices(m_instance, &physicalDeviceCount, physicalDevices.data());
        DebugCheckCritical(result == VK_SUCCESS, "Failed to enumerate physical devices");

        // Get the physical device with the max api version(usually the best one)
        uint32_t maxApiVersion{0};
        for (VkPhysicalDevice p: physicalDevices) {
            VkPhysicalDeviceProperties properties{0};
            vkGetPhysicalDeviceProperties(p, &properties);

            if (properties.apiVersion > maxApiVersion) {
                m_physicalDevice = p;
                maxApiVersion    = properties.apiVersion;
            }
        }

        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(m_physicalDevice, &properties);
        m_timestampPeriod = properties.limits.timestampPeriod;
        DebugInfo(
            "Selected physical device: {} {}.{}.{}",
            properties.deviceName,
            VK_API_VERSION_MAJOR(properties.apiVersion),
            VK_API_VERSION_MINOR(properties.apiVersion),
            VK_API_VERSION_PATCH(properties.apiVersion)
        );
    }

    // Logical device
    {
        float                   priority{1.0f};
        VkDeviceQueueCreateInfo infoQueue{
            .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = kQueueFamilyIndex,
            .queueCount       = 1,
            .pQueuePriorities = &priority,
        };

        std::vector<const char *> extensions{
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME,
            VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
            VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
        };

        VkPhysicalDeviceVulkan13Features feature13{
            .sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            .pNext            = nullptr,
            .synchronization2 = VK_TRUE,
            .dynamicRendering = VK_TRUE,
        };

        VkPhysicalDeviceFeatures features{};

        VkDeviceCreateInfo infoDevice{
            .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext                   = &feature13,
            .queueCreateInfoCount    = 1,
            .pQueueCreateInfos       = &infoQueue,
            .enabledExtensionCount   = static_cast<uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data(),
            .pEnabledFeatures        = &features,
        };

        VkResult result = vkCreateDevice(m_physicalDevice, &infoDevice, nullptr, &m_device);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to create device");

        vkGetDeviceQueue(m_device, kQueueFamilyIndex, 0, &m_queue);

        vkCmdPushDescriptorSetKHR = reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(vkGetDeviceProcAddr(m_device, "vkCmdPushDescriptorSetKHR"));
        DebugCheckCritical(vkCmdPushDescriptorSetKHR != nullptr, "Failed to load vkCmdPushDescriptorSetKHR");

        m_deletionQueue.Push([&]() { vkDestroyDevice(m_device, nullptr); });
    }

    // Allocator
    {
        VmaAllocatorCreateInfo infoAllocator{
            .physicalDevice   = m_physicalDevice,
            .device           = m_device,
            .instance         = m_instance,
            .vulkanApiVersion = VK_API_VERSION_1_4,
        };
        VkResult result = vmaCreateAllocator(&infoAllocator, &m_allocator);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to create allocator");
        m_deletionQueue.Push([&]() { vmaDestroyAllocator(m_allocator); });
    }

    // Surface
    {
        DebugCheckCritical(SDL_Vulkan_CreateSurface(m_window, m_instance, nullptr, &m_surface), "Failed to create surface");
        m_deletionQueue.Push([&]() { vkDestroySurfaceKHR(m_instance, m_surface, nullptr); });
    }

    // Swapchain
    {
        VkSurfaceCapabilitiesKHR capabilities{};
        VkResult                 result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physicalDevice, m_surface, &capabilities);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to get surface capabilities");

        m_swapchainExtent = CalcSwapchainExtent(capabilities);

        const VkSwapchainCreateInfoKHR infoSwapchain{
            .sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface          = m_surface,
            .minImageCount    = kMinSwapchainImage,
            .imageFormat      = kPresentFormat,
            .imageColorSpace  = kPresentColorSpace,
            .imageExtent      = m_swapchainExtent,
            .imageArrayLayers = 1,
            .imageUsage       = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .preTransform     = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
            .compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode      = kPresentMode,
            .clipped          = VK_FALSE,
            .oldSwapchain     = VK_NULL_HANDLE,
        };
        result = vkCreateSwapchainKHR(m_device, &infoSwapchain, nullptr, &m_swapchain);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to create swapchain");
        m_deletionQueue.Push([&]() { vkDestroySwapchainKHR(m_device, m_swapchain, nullptr); });

        result = vkGetSwapchainImagesKHR(m_device, m_swapchain, &m_swapchainCount, nullptr);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to get swapchain image count");
        DebugCheckCritical(m_swapchainCount <= kMaxSwapchainImage, "Swapchain image count exceeds max");

        std::vector<VkImage> swapchainImages(kMaxSwapchainImage);
        result = vkGetSwapchainImagesKHR(m_device, m_swapchain, &m_swapchainCount, swapchainImages.data());
        DebugCheckCritical(result == VK_SUCCESS, "Failed to get swapchain images");

        VkImageViewCreateInfo infoView{
            .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format   = kPresentFormat,
            .components =
                {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY},
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        };

        for (uint32_t i = 0; i < m_swapchainCount; ++i) {
            m_swapchainTextures[i].image = swapchainImages[i];
            infoView.image               = m_swapchainTextures[i].image;
            result                       = vkCreateImageView(m_device, &infoView, nullptr, &m_swapchainTextures[i].view);
            DebugCheckCritical(result == VK_SUCCESS, "Failed to create swapchain image view #{}", i);
            m_deletionQueue.Push([&, i]() { vkDestroyImageView(m_device, m_swapchainTextures[i].view, nullptr); });
        }

        VkSemaphoreCreateInfo infoSemaphore{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        for (uint32_t i = 0; i < m_swapchainCount; ++i) {
            result = vkCreateSemaphore(m_device, &infoSemaphore, nullptr, &m_renderSemaphores[i]);
            DebugCheckCritical(result == VK_SUCCESS, "Failed to create render semaphore #{}", i);
            m_deletionQueue.Push([&, i]() { vkDestroySemaphore(m_device, m_renderSemaphores[i], nullptr); });
        }
    }

    // Command pool
    {
        const VkCommandPoolCreateInfo infoCommandPool{
            .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = kQueueFamilyIndex,
        };
        VkResult result = vkCreateCommandPool(m_device, &infoCommandPool, nullptr, &m_commandPool);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to create command pool");
        m_deletionQueue.Push([&]() { vkDestroyCommandPool(m_device, m_commandPool, nullptr); });
    }

    // In flight command buffers
    {
        const VkCommandBufferAllocateInfo infoCmdBuffer{
            .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool        = m_commandPool,
            .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = kMaxFramesInFlight,
        };
        VkResult result = vkAllocateCommandBuffers(m_device, &infoCmdBuffer, m_commandBuffers.data());
        DebugCheckCritical(result == VK_SUCCESS, "Failed to allocate command buffers");
        m_deletionQueue.Push([&]() { vkFreeCommandBuffers(m_device, m_commandPool, kMaxFramesInFlight, m_commandBuffers.data()); });
    }

    // In flight fences and semaphores
    {
        VkResult                result = VK_SUCCESS;
        const VkFenceCreateInfo infoFence{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
        };
        const VkSemaphoreCreateInfo infoSemaphore{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
            result = vkCreateFence(m_device, &infoFence, nullptr, &m_fences[i]);
            DebugCheckCritical(result == VK_SUCCESS, "Failed to create fence #{}", i);
            m_deletionQueue.Push([&, i]() { vkDestroyFence(m_device, m_fences[i], nullptr); });

            result = vkCreateSemaphore(m_device, &infoSemaphore, nullptr, &m_presentSemaphores[i]);
            DebugCheckCritical(result == VK_SUCCESS, "Failed to create present semaphore #{}", i);
            m_deletionQueue.Push([&, i]() { vkDestroySemaphore(m_device, m_presentSemaphores[i], nullptr); });
        }
    }

    // Immediate fence
    {
        const VkFenceCreateInfo infoFence{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        };
        VkResult result = vkCreateFence(m_device, &infoFence, nullptr, &m_immediateFence);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to create immediate fence");
        m_deletionQueue.Push([&]() { vkDestroyFence(m_device, m_immediateFence, nullptr); });
    }

    // Immediate command
    {
        const VkCommandBufferAllocateInfo infoCommandBuffer{
            .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool        = m_commandPool,
            .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VkResult result = vkAllocateCommandBuffers(m_device, &infoCommandBuffer, &m_immediateCommandBuffer);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to allocate immediate command buffer");
        m_deletionQueue.Push([&]() { vkFreeCommandBuffers(m_device, m_commandPool, 1, &m_immediateCommandBuffer); });
    }

    // Timestamp query pool
    {
        const VkQueryPoolCreateInfo infoQueryPool{
            .sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType  = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = kMaxFramesInFlight * 2,
        };
        VkResult result = vkCreateQueryPool(m_device, &infoQueryPool, nullptr, &m_queryPool);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to create timestamp query pool");
        m_deletionQueue.Push([&]() { vkDestroyQueryPool(m_device, m_queryPool, nullptr); });
    }

    m_lastFrameStart = std::chrono::high_resolution_clock::now();

    ShaderCompiler shaderCompiler;

    // Default sampler
    {
        const VkSamplerCreateInfo infoSampler{
            .sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter    = VK_FILTER_LINEAR,
            .minFilter    = VK_FILTER_LINEAR,
            .mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .maxLod       = VK_LOD_CLAMP_NONE,
        };
        VkResult result = vkCreateSampler(m_device, &infoSampler, nullptr, &m_defaultSampler);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to create default sampler");
        m_deletionQueue.Push([&]() { vkDestroySampler(m_device, m_defaultSampler, nullptr); });
    }

    // Forward descriptor set layout (push descriptor)
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings(8);
        for (uint32_t i = 0; i < bindings.size(); ++i) {
            bindings[i] = VkDescriptorSetLayoutBinding{
                .binding         = i,
                .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT,
            };
        }
        const VkDescriptorSetLayoutCreateInfo infoSetLayout{
            .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings    = bindings.data(),
        };
        VkResult result = vkCreateDescriptorSetLayout(m_device, &infoSetLayout, nullptr, &m_forwardSetLayout);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to create forward descriptor set layout");
        m_deletionQueue.Push([&]() { vkDestroyDescriptorSetLayout(m_device, m_forwardSetLayout, nullptr); });
    }

    // Skybox descriptor set layout (push descriptor, 1 texture)
    {
        const VkDescriptorSetLayoutBinding binding{
            .binding         = 0,
            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT,
        };
        const VkDescriptorSetLayoutCreateInfo infoSetLayout{
            .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
            .bindingCount = 1,
            .pBindings    = &binding,
        };
        VkResult result = vkCreateDescriptorSetLayout(m_device, &infoSetLayout, nullptr, &m_skyboxSetLayout);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to create skybox descriptor set layout");
        m_deletionQueue.Push([&]() { vkDestroyDescriptorSetLayout(m_device, m_skyboxSetLayout, nullptr); });
    }

    // Forward pipeline
    {
        const VkPushConstantRange pushConstantRange{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset     = 0,
            .size       = sizeof(shader_io::GlobalUniforms),
        };
        const VkPipelineLayoutCreateInfo infoLayout{
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount         = 1,
            .pSetLayouts            = &m_forwardSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pushConstantRange,
        };

        VkResult result = vkCreatePipelineLayout(m_device, &infoLayout, nullptr, &m_forward.layout);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to create forward pipeline layout");
        m_deletionQueue.Push([&]() { vkDestroyPipelineLayout(m_device, m_forward.layout, nullptr); });

        VkPipelineInputAssemblyStateCreateInfo infoInputAssembly{
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .pNext                  = nullptr,
            .flags                  = 0,
            .topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE
        };

        VkPipelineViewportStateCreateInfo infoViewport{
            .sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .pNext         = nullptr,
            .flags         = 0,
            .viewportCount = 1,
            .pViewports    = nullptr,
            .scissorCount  = 1,
            .pScissors     = nullptr,
        };

        VkPipelineRasterizationStateCreateInfo infoRasterization{
            .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .pNext                   = nullptr,
            .flags                   = 0,
            .depthClampEnable        = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode             = VK_POLYGON_MODE_FILL,
            .cullMode                = VK_CULL_MODE_NONE,
            .frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .depthBiasEnable         = VK_FALSE,
            .depthBiasConstantFactor = 0.0f,
            .depthBiasClamp          = 0.0f,
            .depthBiasSlopeFactor    = 0.0f,
            .lineWidth               = 1.0f
        };

        VkPipelineMultisampleStateCreateInfo infoMultisample{
            .sType                 = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .pNext                 = nullptr,
            .flags                 = 0,
            .rasterizationSamples  = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable   = VK_FALSE,
            .minSampleShading      = 0.0f,
            .pSampleMask           = nullptr,
            .alphaToCoverageEnable = VK_FALSE,
            .alphaToOneEnable      = VK_FALSE
        };

        VkPipelineDepthStencilStateCreateInfo infoDepthStencil{
            .sType                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .pNext                 = nullptr,
            .flags                 = 0,
            .depthTestEnable       = VK_TRUE,
            .depthWriteEnable      = VK_TRUE,
            .depthCompareOp        = VK_COMPARE_OP_LESS_OR_EQUAL,
            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable     = VK_FALSE,
            .front                 = {},
            .back                  = {},
            .minDepthBounds        = 0.0f,
            .maxDepthBounds        = 1.0f
        };

        std::vector<VkPipelineColorBlendAttachmentState> blendStates{
            {.blendEnable         = VK_FALSE,
             .srcColorBlendFactor = VK_BLEND_FACTOR_ZERO,
             .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
             .colorBlendOp        = VK_BLEND_OP_ADD,
             .srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
             .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
             .alphaBlendOp        = VK_BLEND_OP_ADD,
             .colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT},
        };

        VkPipelineColorBlendStateCreateInfo infoColorBlend{
            .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .pNext           = nullptr,
            .flags           = 0,
            .logicOpEnable   = VK_FALSE,
            .logicOp         = VK_LOGIC_OP_CLEAR,
            .attachmentCount = static_cast<uint32_t>(blendStates.size()),
            .pAttachments    = blendStates.data(),
        };

        std::vector<VkDynamicState>      dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo infoDynamic{
            .sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .pNext             = nullptr,
            .flags             = 0,
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates    = dynamicStates.data()
        };

        constexpr const char *kForwardDir = "../runtime/shaders/forward.slang";

        VkShaderModuleCreateInfo infoShaderModule{
            .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = shaderCompiler.GetSpirvSize(kForwardDir),
            .pCode    = shaderCompiler.GetSpirv(kForwardDir),
        };
        VkShaderModule shaderModule{};
        result = vkCreateShaderModule(m_device, &infoShaderModule, nullptr, &shaderModule);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to create shader module for {}", kForwardDir);

        std::vector<VkPipelineShaderStageCreateInfo>               shaderStages{};
        std::vector<std::pair<std::string, VkShaderStageFlagBits>> entryPoints = shaderCompiler.GetEntryPoints(kForwardDir);
        shaderStages.reserve(entryPoints.size());
        for (const auto &entryPoint: entryPoints) {
            VkPipelineShaderStageCreateInfo info{
                .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage  = entryPoint.second,
                .module = shaderModule,
                .pName  = entryPoint.first.c_str(),
            };
            shaderStages.push_back(info);
        }

        std::vector<VkFormat> colorFormats{
            kSceneColorFormat,
        };

        VkPipelineRenderingCreateInfo infoRendering{
            .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .viewMask                = 0,
            .colorAttachmentCount    = static_cast<uint32_t>(colorFormats.size()),
            .pColorAttachmentFormats = colorFormats.data(),
            .depthAttachmentFormat   = kSceneDepthFormat,
            .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
        };

        const std::vector<VkVertexInputBindingDescription> vertexInputBindingDescriptions{
            {.binding = 0, .stride = sizeof(ntc::VertexData), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX},
        };
        const std::vector<VkVertexInputAttributeDescription> vertexInputAttributeDescriptions{
            {.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(ntc::VertexData, position)},
            {.location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(ntc::VertexData, normal)  },
            {.location = 2, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(ntc::VertexData, tangent) },
            {.location = 3, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT,    .offset = offsetof(ntc::VertexData, uv)      },
        };

        const VkPipelineVertexInputStateCreateInfo infoVertexInput{
            .sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount   = static_cast<uint32_t>(vertexInputBindingDescriptions.size()),
            .pVertexBindingDescriptions      = vertexInputBindingDescriptions.data(),
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributeDescriptions.size()),
            .pVertexAttributeDescriptions    = vertexInputAttributeDescriptions.data(),
        };

        // We don't support tessellation
        VkGraphicsPipelineCreateInfo infoPipeline{
            .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext               = &infoRendering,
            .flags               = 0,
            .stageCount          = static_cast<uint32_t>(shaderStages.size()),
            .pStages             = shaderStages.data(),
            .pVertexInputState   = &infoVertexInput,
            .pInputAssemblyState = &infoInputAssembly,
            .pTessellationState  = nullptr,
            .pViewportState      = &infoViewport,
            .pRasterizationState = &infoRasterization,
            .pMultisampleState   = &infoMultisample,
            .pDepthStencilState  = &infoDepthStencil,
            .pColorBlendState    = &infoColorBlend,
            .pDynamicState       = &infoDynamic,
            .layout              = m_forward.layout,
            .renderPass          = VK_NULL_HANDLE,
            .subpass             = 0,
            .basePipelineHandle  = VK_NULL_HANDLE,
            .basePipelineIndex   = 0,
        };

        result = vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &infoPipeline, nullptr, &m_forward.pipeline);
        DebugCheck(result == VK_SUCCESS, "Failed to create forward pipeline");
        m_deletionQueue.Push([&]() { vkDestroyPipeline(m_device, m_forward.pipeline, nullptr); });

        vkDestroyShaderModule(m_device, shaderModule, nullptr);
    }

    // Skybox pipeline
    {
        const VkPushConstantRange pushConstantRange{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset     = 0,
            .size       = sizeof(shader_io::GlobalUniforms),
        };
        const VkPipelineLayoutCreateInfo infoLayout{
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount         = 1,
            .pSetLayouts            = &m_skyboxSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pushConstantRange,
        };
        VkResult result = vkCreatePipelineLayout(m_device, &infoLayout, nullptr, &m_skybox.layout);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to create skybox pipeline layout");
        m_deletionQueue.Push([&]() { vkDestroyPipelineLayout(m_device, m_skybox.layout, nullptr); });

        VkPipelineInputAssemblyStateCreateInfo infoInputAssembly{
            .sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
        };
        VkPipelineViewportStateCreateInfo infoViewport{
            .sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .scissorCount  = 1,
        };
        VkPipelineRasterizationStateCreateInfo infoRasterization{
            .sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode    = VK_CULL_MODE_NONE,
            .frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .lineWidth   = 1.0f,
        };
        VkPipelineMultisampleStateCreateInfo infoMultisample{
            .sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        };

        VkPipelineDepthStencilStateCreateInfo infoDepthStencil{
            .sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable  = VK_TRUE,
            .depthWriteEnable = VK_FALSE,
            .depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL,
            .minDepthBounds   = 0.0f,
            .maxDepthBounds   = 1.0f,
        };
        VkPipelineColorBlendAttachmentState blendState{
            .blendEnable    = VK_FALSE,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        };
        VkPipelineColorBlendStateCreateInfo infoColorBlend{
            .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments    = &blendState,
        };
        std::vector<VkDynamicState>      dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo infoDynamic{
            .sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates    = dynamicStates.data(),
        };

        constexpr const char *kSkyboxDir = "../runtime/shaders/skybox.slang";

        VkShaderModuleCreateInfo infoShaderModule{
            .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = shaderCompiler.GetSpirvSize(kSkyboxDir),
            .pCode    = shaderCompiler.GetSpirv(kSkyboxDir),
        };
        VkShaderModule shaderModule{};
        result = vkCreateShaderModule(m_device, &infoShaderModule, nullptr, &shaderModule);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to create shader module for {}", kSkyboxDir);

        std::vector<VkPipelineShaderStageCreateInfo>               shaderStages{};
        std::vector<std::pair<std::string, VkShaderStageFlagBits>> entryPoints = shaderCompiler.GetEntryPoints(kSkyboxDir);
        shaderStages.reserve(entryPoints.size());
        for (const auto &entryPoint: entryPoints) {
            shaderStages.push_back(
                VkPipelineShaderStageCreateInfo{
                    .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage  = entryPoint.second,
                    .module = shaderModule,
                    .pName  = entryPoint.first.c_str(),
                }
            );
        }

        std::vector<VkFormat>         colorFormats{kSceneColorFormat};
        VkPipelineRenderingCreateInfo infoRendering{
            .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .colorAttachmentCount    = static_cast<uint32_t>(colorFormats.size()),
            .pColorAttachmentFormats = colorFormats.data(),
            .depthAttachmentFormat   = kSceneDepthFormat,
            .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
        };

        const std::vector<VkVertexInputBindingDescription> vertexInputBindingDescriptions{
            {.binding = 0, .stride = sizeof(glm::vec3), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX},
        };
        const std::vector<VkVertexInputAttributeDescription> vertexInputAttributeDescriptions{
            {.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = 0},
        };

        const VkPipelineVertexInputStateCreateInfo infoVertexInput{
            .sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount   = static_cast<uint32_t>(vertexInputBindingDescriptions.size()),
            .pVertexBindingDescriptions      = vertexInputBindingDescriptions.data(),
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributeDescriptions.size()),
            .pVertexAttributeDescriptions    = vertexInputAttributeDescriptions.data(),
        };

        VkGraphicsPipelineCreateInfo infoPipeline{
            .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext               = &infoRendering,
            .stageCount          = static_cast<uint32_t>(shaderStages.size()),
            .pStages             = shaderStages.data(),
            .pVertexInputState   = &infoVertexInput,
            .pInputAssemblyState = &infoInputAssembly,
            .pViewportState      = &infoViewport,
            .pRasterizationState = &infoRasterization,
            .pMultisampleState   = &infoMultisample,
            .pDepthStencilState  = &infoDepthStencil,
            .pColorBlendState    = &infoColorBlend,
            .pDynamicState       = &infoDynamic,
            .layout              = m_skybox.layout,
        };

        result = vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &infoPipeline, nullptr, &m_skybox.pipeline);
        DebugCheck(result == VK_SUCCESS, "Failed to create skybox pipeline");
        m_deletionQueue.Push([&]() { vkDestroyPipeline(m_device, m_skybox.pipeline, nullptr); });

        vkDestroyShaderModule(m_device, shaderModule, nullptr);
    }

    // Descriptor pool for ImGui
    {
        const std::vector<VkDescriptorPoolSize> poolSizes{
            {VK_DESCRIPTOR_TYPE_SAMPLER,                1000},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,          1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1000},
            {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,   1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,   1000},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         1000},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
            {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,       1000}
        };

        const VkDescriptorPoolCreateInfo infoPool{
            .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext         = nullptr,
            .flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
            .maxSets       = 1024,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes    = poolSizes.data()
        };
        VkResult result = vkCreateDescriptorPool(m_device, &infoPool, nullptr, &m_descriptorPool);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to create descriptor pool");
        m_deletionQueue.Push([&]() { vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr); });
    }

    // ImGui
    {
        ImGui::CreateContext();

        ImGui_ImplSDL3_InitForVulkan(m_window);

        ImGuiIO &io     = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls

        VkFormat                         colorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
        VkPipelineRenderingCreateInfoKHR infoRendering{
            .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
            .pNext                   = nullptr,
            .viewMask                = 0,
            .colorAttachmentCount    = 1,
            .pColorAttachmentFormats = &colorFormat,
            .depthAttachmentFormat   = VK_FORMAT_UNDEFINED,
            .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
        };

        ImGui_ImplVulkan_InitInfo infoVulkan{
            .ApiVersion                  = VK_API_VERSION_1_4,
            .Instance                    = m_instance,
            .PhysicalDevice              = m_physicalDevice,
            .Device                      = m_device,
            .QueueFamily                 = kQueueFamilyIndex,
            .Queue                       = m_queue,
            .DescriptorPool              = m_descriptorPool,
            .RenderPass                  = VK_NULL_HANDLE,
            .MinImageCount               = kMinSwapchainImage,
            .ImageCount                  = m_swapchainCount,
            .MSAASamples                 = VK_SAMPLE_COUNT_1_BIT,
            .UseDynamicRendering         = true,
            .PipelineRenderingCreateInfo = infoRendering,
        };

        ImGui_ImplVulkan_Init(&infoVulkan);
        ImGui_ImplVulkan_CreateFontsTexture();
    }

    // Scene targets
    {
        for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
            VkImageCreateInfo infoColor{
                .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .imageType     = VK_IMAGE_TYPE_2D,
                .format        = kSceneColorFormat,
                .extent        = {m_swapchainExtent.width, m_swapchainExtent.height, 1},
                .mipLevels     = 1,
                .arrayLayers   = 1,
                .samples       = VK_SAMPLE_COUNT_1_BIT,
                .tiling        = VK_IMAGE_TILING_OPTIMAL,
                .usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
            VmaAllocationCreateInfo infoColorAlloc{
                .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            };
            VkResult result =
                vmaCreateImage(m_allocator, &infoColor, &infoColorAlloc, &m_sceneColors[i].image, &m_sceneColors[i].allocation, nullptr);
            DebugCheckCritical(result == VK_SUCCESS, "Failed to create scene color image #{}", i);
            m_deletionQueue.Push([&, i]() { vmaDestroyImage(m_allocator, m_sceneColors[i].image, m_sceneColors[i].allocation); });

            VkImageViewCreateInfo infoColorView{
                .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image    = m_sceneColors[i].image,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format   = kSceneColorFormat,
                .components =
                    {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY},
                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
            };
            result = vkCreateImageView(m_device, &infoColorView, nullptr, &m_sceneColors[i].view);
            DebugCheckCritical(result == VK_SUCCESS, "Failed to create scene color view #{}", i);
            m_deletionQueue.Push([&, i]() { vkDestroyImageView(m_device, m_sceneColors[i].view, nullptr); });

            VkImageCreateInfo infoDepth{
                .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .imageType     = VK_IMAGE_TYPE_2D,
                .format        = kSceneDepthFormat,
                .extent        = {m_swapchainExtent.width, m_swapchainExtent.height, 1},
                .mipLevels     = 1,
                .arrayLayers   = 1,
                .samples       = VK_SAMPLE_COUNT_1_BIT,
                .tiling        = VK_IMAGE_TILING_OPTIMAL,
                .usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
            VmaAllocationCreateInfo infoDepthAlloc{
                .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            };
            result = vmaCreateImage(m_allocator, &infoDepth, &infoDepthAlloc, &m_sceneDepths[i].image, &m_sceneDepths[i].allocation, nullptr);
            DebugCheckCritical(result == VK_SUCCESS, "Failed to create scene depth image #{}", i);
            m_deletionQueue.Push([&, i]() { vmaDestroyImage(m_allocator, m_sceneDepths[i].image, m_sceneDepths[i].allocation); });

            VkImageViewCreateInfo infoDepthView{
                .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image    = m_sceneDepths[i].image,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format   = kSceneDepthFormat,
                .components =
                    {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY},
                .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
            };
            result = vkCreateImageView(m_device, &infoDepthView, nullptr, &m_sceneDepths[i].view);
            DebugCheckCritical(result == VK_SUCCESS, "Failed to create scene depth view #{}", i);
            m_deletionQueue.Push([&, i]() { vkDestroyImageView(m_device, m_sceneDepths[i].view, nullptr); });
        }
    }

    // Load helmet mesh
    {
        const std::vector<ntc::VertexData> vertices = ntc::LoadMesh("../assets/source/helmet.obj");
        m_helmetVertexBuffer.vertexCount            = static_cast<uint32_t>(vertices.size());
        const VkDeviceSize bufferSize               = vertices.size() * sizeof(ntc::VertexData);

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
        VkResult          result =
            vmaCreateBuffer(m_allocator, &infoStagingBuffer, &infoStagingAlloc, &stagingBuffer, &stagingAllocation, &stagingAllocationInfo);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to create helmet vertex staging buffer");
        std::memcpy(stagingAllocationInfo.pMappedData, vertices.data(), bufferSize);

        VkBufferCreateInfo infoBuffer{
            .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size        = bufferSize,
            .usage       = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        VmaAllocationCreateInfo infoBufferAlloc{
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        };
        result = vmaCreateBuffer(m_allocator, &infoBuffer, &infoBufferAlloc, &m_helmetVertexBuffer.buffer, &m_helmetVertexBuffer.allocation, nullptr);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to create helmet vertex buffer");
        m_deletionQueue.Push([&]() { vmaDestroyBuffer(m_allocator, m_helmetVertexBuffer.buffer, m_helmetVertexBuffer.allocation); });

        ImmediateSubmit([&](VkCommandBuffer commandBuffer) {
            VkBufferCopy region{.size = bufferSize};
            vkCmdCopyBuffer(commandBuffer, stagingBuffer, m_helmetVertexBuffer.buffer, 1, &region);
        });

        vmaDestroyBuffer(m_allocator, stagingBuffer, stagingAllocation);
    }

    // Load skybox mesh
    {
        const std::vector<glm::vec3> skyboxVertices{
            glm::vec3(-1.0f, 1.0f, 1.0f),
            glm::vec3(1.0f, 1.0f, 1.0f),
            glm::vec3(-1.0f, -1.0f, 1.0f),
            glm::vec3(1.0f, -1.0f, 1.0f),
            glm::vec3(1.0f, -1.0f, -1.0f),
            glm::vec3(1.0f, 1.0f, 1.0f),
            glm::vec3(1.0f, 1.0f, -1.0f),
            glm::vec3(-1.0f, 1.0f, 1.0f),
            glm::vec3(-1.0f, 1.0f, -1.0f),
            glm::vec3(-1.0f, -1.0f, 1.0f),
            glm::vec3(-1.0f, -1.0f, -1.0f),
            glm::vec3(1.0f, -1.0f, -1.0f),
            glm::vec3(-1.0f, 1.0f, -1.0f),
            glm::vec3(1.0f, 1.0f, -1.0f)
        };
        m_skyboxVertexBuffer.vertexCount = static_cast<uint32_t>(skyboxVertices.size());
        const VkDeviceSize bufferSize    = skyboxVertices.size() * sizeof(glm::vec3);

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
        VkResult          result =
            vmaCreateBuffer(m_allocator, &infoStagingBuffer, &infoStagingAlloc, &stagingBuffer, &stagingAllocation, &stagingAllocationInfo);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to create helmet vertex staging buffer");
        std::memcpy(stagingAllocationInfo.pMappedData, skyboxVertices.data(), bufferSize);

        VkBufferCreateInfo infoBuffer{
            .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size        = bufferSize,
            .usage       = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        VmaAllocationCreateInfo infoBufferAlloc{
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        };
        result = vmaCreateBuffer(m_allocator, &infoBuffer, &infoBufferAlloc, &m_skyboxVertexBuffer.buffer, &m_skyboxVertexBuffer.allocation, nullptr);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to create helmet vertex buffer");
        m_deletionQueue.Push([&]() { vmaDestroyBuffer(m_allocator, m_skyboxVertexBuffer.buffer, m_skyboxVertexBuffer.allocation); });

        ImmediateSubmit([&](VkCommandBuffer commandBuffer) {
            VkBufferCopy region{.size = bufferSize};
            vkCmdCopyBuffer(commandBuffer, stagingBuffer, m_skyboxVertexBuffer.buffer, 1, &region);
        });

        vmaDestroyBuffer(m_allocator, stagingBuffer, stagingAllocation);
    }

    // Load textures
    {
        struct TextureUpload {
            std::string_view path;
            VkFormat         format;
            VulkanTexture   *texture;
        };

        const std::vector<TextureUpload> uploads{
            {"../assets/source/Default_albedo.jpg",         VK_FORMAT_R8G8B8A8_SRGB,  &m_helmetAlbedo           },
            {"../assets/source/Default_AO.jpg",             VK_FORMAT_R8G8B8A8_UNORM, &m_helmetAO               },
            {"../assets/source/Default_emissive.jpg",       VK_FORMAT_R8G8B8A8_UNORM, &m_helmetEmissive         },
            {"../assets/source/Default_metalRoughness.jpg", VK_FORMAT_R8G8B8A8_UNORM, &m_helmetMetallicRoughness},
            {"../assets/source/Default_normal.jpg",         VK_FORMAT_R8G8B8A8_UNORM, &m_helmetNormal           },
            {"../assets/skybox/skybox.png",                 VK_FORMAT_R8G8B8A8_SRGB,  &m_skyboxTexture          },
            {"../assets/skybox/skybox_irradiance.png",      VK_FORMAT_R8G8B8A8_UNORM, &m_skyboxIrradiance       },
            {"../assets/skybox/skybox_specular.png",        VK_FORMAT_R8G8B8A8_UNORM, &m_skyboxSpecular         },
            {"../assets/skybox/brdf_lut.png",               VK_FORMAT_R8G8B8A8_UNORM, &m_brdfLut                },
        };

        for (const TextureUpload &upload: uploads) {
            int                width{0};
            int                height{0};
            unsigned char     *pixels     = ntc::LoadImage(upload.path, width, height);
            VulkanTexture     *texture    = upload.texture;
            const VkDeviceSize bufferSize = static_cast<VkDeviceSize>(width) * static_cast<VkDeviceSize>(height) * 4;

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
            VkResult          result =
                vmaCreateBuffer(m_allocator, &infoStagingBuffer, &infoStagingAlloc, &stagingBuffer, &stagingAllocation, &stagingAllocationInfo);
            DebugCheckCritical(result == VK_SUCCESS, "Failed to create staging buffer for {}", upload.path);
            std::memcpy(stagingAllocationInfo.pMappedData, pixels, bufferSize);
            stbi_image_free(pixels);

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
            result = vmaCreateImage(m_allocator, &infoImage, &infoImageAlloc, &texture->image, &texture->allocation, nullptr);
            DebugCheckCritical(result == VK_SUCCESS, "Failed to create image for {}", upload.path);
            m_deletionQueue.Push([this, texture]() { vmaDestroyImage(m_allocator, texture->image, texture->allocation); });

            ImmediateSubmit([&](VkCommandBuffer commandBuffer) {
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

            vmaDestroyBuffer(m_allocator, stagingBuffer, stagingAllocation);

            VkImageViewCreateInfo infoView{
                .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image    = texture->image,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format   = upload.format,
                .components =
                    {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY},
                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
            };
            result = vkCreateImageView(m_device, &infoView, nullptr, &texture->view);
            DebugCheckCritical(result == VK_SUCCESS, "Failed to create image view for {}", upload.path);
            m_deletionQueue.Push([this, texture]() { vkDestroyImageView(m_device, texture->view, nullptr); });
        }
    }
}

VulkanState::~VulkanState() {
    WaitIdle();

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    m_deletionQueue.Flush();
}

void VulkanState::WaitIdle() const {
    VkResult result = vkDeviceWaitIdle(m_device);
    DebugCheckCritical(result == VK_SUCCESS, "Failed to wait for device to become idle");
}

void VulkanState::Run() {
    const VkFence         &fence            = m_fences[m_currentFrameIndex];
    const VkSemaphore     &presentSemaphore = m_presentSemaphores[m_currentFrameIndex];
    const VkCommandBuffer &commandBuffer    = m_commandBuffers[m_currentFrameIndex];
    const VulkanTexture   &sceneColor       = m_sceneColors[m_currentFrameIndex];
    const VulkanTexture   &sceneDepth       = m_sceneDepths[m_currentFrameIndex];

    WaitAndRestFence(fence);

    // CPU frame time
    {
        const auto                                     now   = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<float, std::milli> delta = now - m_lastFrameStart;
        m_lastFrameTime                                      = delta.count();
        m_lastFrameStart                                     = now;
        m_frameTimeHistory[m_frameHistoryIndex]              = m_lastFrameTime;
        m_frameHistoryIndex                                  = (m_frameHistoryIndex + 1) % kFrameHistorySize;
    }

    // Read back GPU timestamps from the frame that previously used this in-flight slot
    if (m_totalFrameCount >= kMaxFramesInFlight) {
        uint64_t       timestamps[2]{};
        const VkResult result = vkGetQueryPoolResults(
            m_device,
            m_queryPool,
            m_currentFrameIndex * 2,
            2,
            sizeof(timestamps),
            timestamps,
            sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT
        );
        if (result == VK_SUCCESS) {
            m_pbrTime = static_cast<float>(timestamps[1] - timestamps[0]) * m_timestampPeriod * 1e-6f;
        }
    }

    // ImGui new frame
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
    DrawImGuiContent();
    ImGui::Render();

    // Swapchain acquire
    {
        VkResult result = vkAcquireNextImageKHR(m_device, m_swapchain, kPointOneSecond, presentSemaphore, VK_NULL_HANDLE, &m_presentImageIndex);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to acquire swapchain image");
    }

    const VkSemaphore &renderSemaphore = m_renderSemaphores[m_presentImageIndex];
    const VkImage     &swapchainImage  = m_swapchainTextures[m_presentImageIndex].image;

    ResetCommandBuffer(commandBuffer, 0);
    BeginCommandBuffer(commandBuffer, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    vkCmdResetQueryPool(commandBuffer, m_queryPool, m_currentFrameIndex * 2, 2);

    // Transition scene targets for rendering
    {
        std::vector<VkImageMemoryBarrier2> barriers{
            VkImageMemoryBarrier2{
                                  .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                  .srcStageMask     = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                  .srcAccessMask    = 0,
                                  .dstStageMask     = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                  .dstAccessMask    = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                  .oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED,
                                  .newLayout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                  .image            = sceneColor.image,
                                  .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                                  },
            VkImageMemoryBarrier2{
                                  .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                  .srcStageMask     = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                  .srcAccessMask    = 0,
                                  .dstStageMask     = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                                  .dstAccessMask    = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                  .oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED,
                                  .newLayout        = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                  .image            = sceneDepth.image,
                                  .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
                                  },
        };
        VkDependencyInfo dep{
            .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
            .pImageMemoryBarriers    = barriers.data(),
        };
        vkCmdPipelineBarrier2(commandBuffer, &dep);
    }

    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_queryPool, m_currentFrameIndex * 2);
    ForwardPBR(commandBuffer, sceneColor, sceneDepth);
    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_queryPool, m_currentFrameIndex * 2 + 1);

    // Sync forward PBR writes before skybox reads/writes the same attachments
    {
        std::vector<VkImageMemoryBarrier2> barriers{
            VkImageMemoryBarrier2{
                                  .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                  .srcStageMask     = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                  .srcAccessMask    = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                  .dstStageMask     = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                  .dstAccessMask    = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
                                  .oldLayout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                  .newLayout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                  .image            = sceneColor.image,
                                  .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                                  },
            VkImageMemoryBarrier2{
                                  .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                  .srcStageMask     = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                                  .srcAccessMask    = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                  .dstStageMask     = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                                  .dstAccessMask    = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                                  .oldLayout        = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                  .newLayout        = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                  .image            = sceneDepth.image,
                                  .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
                                  },
        };
        VkDependencyInfo dep{
            .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
            .pImageMemoryBarriers    = barriers.data(),
        };
        vkCmdPipelineBarrier2(commandBuffer, &dep);
    }

    Skybox(commandBuffer, sceneColor, sceneDepth);

    // Sync skybox writes before ImGui writes the color attachment
    {
        VkImageMemoryBarrier2 barrier{
            .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask     = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask    = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .dstStageMask     = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstAccessMask    = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
            .oldLayout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .newLayout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .image            = sceneColor.image,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        };
        VkDependencyInfo dep{
            .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers    = &barrier,
        };
        vkCmdPipelineBarrier2(commandBuffer, &dep);
    }

    ImGuiPass(commandBuffer, sceneColor);

    // Blit scene color to swapchain
    {
        std::vector<VkImageMemoryBarrier2> barriers{
            VkImageMemoryBarrier2{
                                  .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                  .srcStageMask     = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                  .srcAccessMask    = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                  .dstStageMask     = VK_PIPELINE_STAGE_2_BLIT_BIT,
                                  .dstAccessMask    = VK_ACCESS_2_TRANSFER_READ_BIT,
                                  .oldLayout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                  .newLayout        = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  .image            = sceneColor.image,
                                  .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                                  },
            VkImageMemoryBarrier2{
                                  .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                  .srcStageMask     = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                  .srcAccessMask    = 0,
                                  .dstStageMask     = VK_PIPELINE_STAGE_2_BLIT_BIT,
                                  .dstAccessMask    = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                  .oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED,
                                  .newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  .image            = swapchainImage,
                                  .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                                  },
        };
        VkDependencyInfo dep{
            .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
            .pImageMemoryBarriers    = barriers.data(),
        };
        vkCmdPipelineBarrier2(commandBuffer, &dep);

        VkImageBlit blitRegion{
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffsets     = {{0, 0, 0}, {static_cast<int32_t>(m_swapchainExtent.width), static_cast<int32_t>(m_swapchainExtent.height), 1}},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstOffsets     = {{0, 0, 0}, {static_cast<int32_t>(m_swapchainExtent.width), static_cast<int32_t>(m_swapchainExtent.height), 1}},
        };
        vkCmdBlitImage(
            commandBuffer,
            sceneColor.image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            swapchainImage,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &blitRegion,
            VK_FILTER_LINEAR
        );
    }

    // Present barrier
    {
        VkImageMemoryBarrier2 barrierToPresent{
            .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask     = VK_PIPELINE_STAGE_2_BLIT_BIT,
            .srcAccessMask    = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask     = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
            .dstAccessMask    = 0,
            .oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout        = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .image            = swapchainImage,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        };
        VkDependencyInfo depToPresent{
            .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers    = &barrierToPresent,
        };
        vkCmdPipelineBarrier2(commandBuffer, &depToPresent);
    }

    EndCommandBuffer(commandBuffer);
    SubmitToQueue(commandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, fence, presentSemaphore, renderSemaphore);

    // Present
    {
        VkPresentInfoKHR infoPresent{
            .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores    = &renderSemaphore,
            .swapchainCount     = 1,
            .pSwapchains        = &m_swapchain,
            .pImageIndices      = &m_presentImageIndex,
        };
        VkResult result = vkQueuePresentKHR(m_queue, &infoPresent);
        DebugCheckCritical(result == VK_SUCCESS, "Failed to present");
    }

    m_currentFrameIndex = (m_currentFrameIndex + 1) % kMaxFramesInFlight;
    ++m_totalFrameCount;
}

void VulkanState::ResetCommandBuffer(const VkCommandBuffer &commandBuffer, VkCommandBufferResetFlags flags) {
    VkResult result = vkResetCommandBuffer(commandBuffer, flags);
    DebugCheckCritical(result == VK_SUCCESS, "Failed to reset command buffer");
}

void VulkanState::BeginCommandBuffer(const VkCommandBuffer &commandBuffer, VkCommandBufferUsageFlags flags) {
    const VkCommandBufferBeginInfo infoBegin{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = flags,
    };
    VkResult result = vkBeginCommandBuffer(commandBuffer, &infoBegin);
    DebugCheckCritical(result == VK_SUCCESS, "Failed to begin command buffer");
}

void VulkanState::EndCommandBuffer(const VkCommandBuffer &commandBuffer) {
    VkResult result = vkEndCommandBuffer(commandBuffer);
    DebugCheckCritical(result == VK_SUCCESS, "Failed to end command buffer");
}

void VulkanState::WaitAndRestFence(const VkFence &fence, uint64_t timeout) const {
    VkResult result = vkWaitForFences(m_device, 1, &fence, VK_TRUE, timeout);
    DebugCheckCritical(result == VK_SUCCESS, "Failed to wait for fence");

    result = vkResetFences(m_device, 1, &fence);
    DebugCheckCritical(result == VK_SUCCESS, "Failed to reset fence");
}

void VulkanState::SubmitToQueue(
    const VkCommandBuffer &commandBuffer,
    VkPipelineStageFlags   stage,
    const VkFence         &fence,
    const VkSemaphore     &waitSemaphore,
    const VkSemaphore     &signalSemaphore
) const {
    VkSubmitInfo infoSubmit{
        .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pWaitDstStageMask  = &stage,
        .commandBufferCount = 1,
        .pCommandBuffers    = &commandBuffer,
    };

    if (waitSemaphore != VK_NULL_HANDLE) {
        infoSubmit.waitSemaphoreCount = 1;
        infoSubmit.pWaitSemaphores    = &waitSemaphore;
    }
    if (signalSemaphore != VK_NULL_HANDLE) {
        infoSubmit.signalSemaphoreCount = 1;
        infoSubmit.pSignalSemaphores    = &signalSemaphore;
    }

    VkResult result = vkQueueSubmit(m_queue, 1, &infoSubmit, fence);
    DebugCheckCritical(result == VK_SUCCESS, "Failed to submit command buffer");
}

VkExtent2D VulkanState::CalcSwapchainExtent(const VkSurfaceCapabilitiesKHR &capabilities) const {
    if (capabilities.currentExtent.width == std::numeric_limits<uint32_t>::max() &&
        capabilities.currentExtent.height == std::numeric_limits<uint32_t>::max()) {
        int width{0};
        int height{0};
        SDL_GetWindowSize(m_window, &width, &height);
        return {
            std::clamp(static_cast<uint32_t>(width), capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            std::clamp(static_cast<uint32_t>(height), capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
        };
    }
    return capabilities.currentExtent;
}

void VulkanState::ForwardPBR(const VkCommandBuffer &commandBuffer, const VulkanTexture &sceneColor, const VulkanTexture &sceneDepth) {
    VkRenderingAttachmentInfo colorAttachment{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView   = sceneColor.view,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue  = {.color = {{0.0f, 0.0f, 0.0f, 1.0f}}},
    };
    VkRenderingAttachmentInfo depthAttachment{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView   = sceneDepth.view,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue  = {.depthStencil = {1.0f, 0}},
    };
    VkRenderingInfo infoRendering{
        .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea           = {.offset = {0, 0}, .extent = m_swapchainExtent},
        .layerCount           = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments    = &colorAttachment,
        .pDepthAttachment     = &depthAttachment,
    };
    vkCmdBeginRendering(commandBuffer, &infoRendering);

    VkViewport viewport{
        .x        = 0.0f,
        .y        = 0.0f,
        .width    = static_cast<float>(m_swapchainExtent.width),
        .height   = static_cast<float>(m_swapchainExtent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{
        .offset = {0, 0},
        .extent = m_swapchainExtent,
    };
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_forward.pipeline);

    // Push constants
    {
        const glm::mat4 view = m_camera.GetViewMatrix();
        const glm::mat4 proj = m_camera.GetProjectionMatrix();

        m_globalUniforms.view        = view;
        m_globalUniforms.proj        = proj;
        m_globalUniforms.viewProj    = proj * view;
        m_globalUniforms.viewInverse = glm::inverse(view);


        vkCmdPushConstants(
            commandBuffer,
            m_forward.layout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(shader_io::GlobalUniforms),
            &m_globalUniforms
        );
    }

    // Push descriptor sets
    {
        const std::vector<VkImageView> views{
            m_helmetAlbedo.view,
            m_helmetAO.view,
            m_helmetEmissive.view,
            m_helmetMetallicRoughness.view,
            m_helmetNormal.view,
            m_skyboxIrradiance.view,
            m_skyboxSpecular.view,
            m_brdfLut.view,
        };
        std::vector<VkDescriptorImageInfo> imageInfos(views.size());
        std::vector<VkWriteDescriptorSet>  writes(views.size());
        for (uint32_t i = 0; i < writes.size(); ++i) {
            imageInfos[i] = VkDescriptorImageInfo{
                .sampler     = m_defaultSampler,
                .imageView   = views[i],
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };
            writes[i] = VkWriteDescriptorSet{
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstBinding      = i,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo      = &imageInfos[i],
            };
        }
        vkCmdPushDescriptorSetKHR(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_forward.layout,
            0,
            static_cast<uint32_t>(writes.size()),
            writes.data()
        );
    }

    VkDeviceSize vertexOffset{0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &m_helmetVertexBuffer.buffer, &vertexOffset);

    vkCmdDraw(commandBuffer, m_helmetVertexBuffer.vertexCount, 1, 0, 0);

    vkCmdEndRendering(commandBuffer);
}

void VulkanState::Skybox(const VkCommandBuffer &commandBuffer, const VulkanTexture &sceneColor, const VulkanTexture &sceneDepth) const {
    VkRenderingAttachmentInfo colorAttachment{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView   = sceneColor.view,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD,
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
    };
    VkRenderingAttachmentInfo depthAttachment{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView   = sceneDepth.view,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD,
        .storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE,
    };
    VkRenderingInfo infoRendering{
        .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea           = {.offset = {0, 0}, .extent = m_swapchainExtent},
        .layerCount           = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments    = &colorAttachment,
        .pDepthAttachment     = &depthAttachment,
    };
    vkCmdBeginRendering(commandBuffer, &infoRendering);

    VkViewport viewport{
        .x        = 0.0f,
        .y        = 0.0f,
        .width    = static_cast<float>(m_swapchainExtent.width),
        .height   = static_cast<float>(m_swapchainExtent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{
        .offset = {0, 0},
        .extent = m_swapchainExtent,
    };
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_skybox.pipeline);

    vkCmdPushConstants(
        commandBuffer,
        m_skybox.layout,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(shader_io::GlobalUniforms),
        &m_globalUniforms
    );

    const VkDescriptorImageInfo skyboxInfo{
        .sampler     = m_defaultSampler,
        .imageView   = m_skyboxTexture.view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    const VkWriteDescriptorSet skyboxWrite{
        .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding      = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo      = &skyboxInfo,
    };
    vkCmdPushDescriptorSetKHR(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_skybox.layout, 0, 1, &skyboxWrite);

    VkDeviceSize skyboxOffset{0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &m_skyboxVertexBuffer.buffer, &skyboxOffset);

    vkCmdDraw(commandBuffer, m_skyboxVertexBuffer.vertexCount, 1, 0, 0);

    vkCmdEndRendering(commandBuffer);
}

void VulkanState::ImGuiPass(const VkCommandBuffer &commandBuffer, const VulkanTexture &sceneColor) const {
    VkRenderingAttachmentInfo colorAttachment{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView   = sceneColor.view,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD,
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
    };
    VkRenderingInfo infoRendering{
        .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea           = {.offset = {0, 0}, .extent = m_swapchainExtent},
        .layerCount           = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments    = &colorAttachment,
    };
    vkCmdBeginRendering(commandBuffer, &infoRendering);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);

    vkCmdEndRendering(commandBuffer);
}

void VulkanState::DrawImGuiContent() {
    if (ImGui::Begin("Debug Window")) {
        if (ImGui::CollapsingHeader("Overall Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Total Frame Time: %.3f ms", m_lastFrameTime);
            ImGui::Text("FPS: %.1f", m_lastFrameTime > 0.0f ? 1000.0f / m_lastFrameTime : 0.0f);
            ImGui::PlotLines(
                "Frame Time History",
                m_frameTimeHistory.data(),
                static_cast<int>(m_frameTimeHistory.size()),
                static_cast<int>(m_frameHistoryIndex),
                nullptr,
                0.0f,
                33.3f,
                ImVec2(0, 80)
            );
        }

        if (ImGui::CollapsingHeader("Per-Pass Performance (GPU)", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Forward PBR: %.3f ms", m_pbrTime);

            float values[] = {m_pbrTime};
            ImGui::PlotHistogram(
                "Compare (ms)",
                values,
                static_cast<int>(std::size(values)),
                0,
                nullptr,
                0.0f,
                std::max(m_pbrTime * 1.2f, 0.001f),
                ImVec2(0, 80)
            );
        }
    }
    ImGui::End();
}