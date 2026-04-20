cmake_minimum_required(VERSION 3.26)
project(NTC)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(SDL3 CONFIG REQUIRED)
find_package(simdjson CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(slang CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)

add_subdirectory(runtime/shaders/ShaderIO)

add_library(glm_config INTERFACE
        runtime/shaders/ShaderIO/global_io.slang)
target_compile_definitions(glm_config INTERFACE
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        GLM_ENABLE_EXPERIMENTAL
)

add_executable(NTC
        runtime/main.cpp
        runtime/stb_image.c
        runtime/vk_mem_alloc.cpp
        runtime/tiny_obj_loader.cc
        runtime/Window.cpp
        runtime/Window.h
        runtime/Debug.h
        runtime/VulkanState.cpp
        runtime/VulkanState.h
        runtime/FileSystem.cpp
        runtime/FileSystem.h
        runtime/ShaderCompiler.cpp
        runtime/ShaderCompiler.h
        runtime/Camera.cpp
        runtime/Camera.h
)

target_include_directories(NTC PRIVATE ${Stb_INCLUDE_DIR})

target_link_libraries(NTC PRIVATE
        Vulkan::Vulkan
        glm::glm
        imgui::imgui
        SDL3::SDL3
        simdjson::simdjson
        spdlog::spdlog
        GPUOpen::VulkanMemoryAllocator
        slang::slang
        tinyobjloader::tinyobjloader
        Shader
)