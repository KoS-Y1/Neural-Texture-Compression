//
// Created by y1 on 2026-04-18.
//

#include "Window.h"

#include <SDL3/SDL.h>

#include "Camera.h"
#include "Debug.h"
#include "VulkanState.h"

#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>

namespace {
constexpr int kMaxWindowWidth{1600};
constexpr int kMaxWindowHeight{900};

constexpr glm::vec3 kForward{0.0f, 0.0f, -1.0f};
constexpr glm::vec3 kBackward{0.0f, 0.0f, 1.0f};
constexpr glm::vec3 kRight{1.0f, 0.0f, 0.0f};
constexpr glm::vec3 kLeft{-1.0f, 0.0f, 0.0f};
constexpr glm::vec3 kUp{0.0f, 1.0f, 0.0f};
constexpr glm::vec3 kDown{0.0f, -1.0f, 0.0f};
} // namespace

Window::Window() {
    m_window =
        SDL_CreateWindow("VulkanApp", kMaxWindowWidth, kMaxWindowHeight, SDL_WINDOW_HIGH_PIXEL_DENSITY | SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    DebugCheckCritical(m_window, "Failed to create window");
}

void Window::Run() {
    Camera      camera;
    VulkanState state{m_window, camera};

    DebugInfo("SDLWindow running");

    auto processInput = [&camera](const SDL_Event &event) {
        static const std::unordered_map<SDL_Scancode, glm::vec3> cameraMovement{
            {SDL_SCANCODE_W, kForward },
            {SDL_SCANCODE_S, kBackward},
            {SDL_SCANCODE_D, kRight   },
            {SDL_SCANCODE_A, kLeft    },
            {SDL_SCANCODE_Q, kUp      },
            {SDL_SCANCODE_E, kDown    },
        };

        // Camera movement
        if (event.type == SDL_EVENT_KEY_DOWN) {
            const auto pair = cameraMovement.find(event.key.scancode);
            if (pair != cameraMovement.end()) {
                camera.ProcessMovement(pair->second);
            }
        }

        // Camera rotation
        if (event.type == SDL_EVENT_MOUSE_MOTION) {
            SDL_Keymod mod = SDL_GetModState();
            if (mod & SDL_KMOD_SHIFT) {
                glm::vec2 offset{event.motion.xrel, event.motion.yrel};
                camera.ProcessRotation(offset);
            }
        }

        // Camera zoom
        if (event.type == SDL_EVENT_MOUSE_WHEEL) {
            camera.ProcessZoom(event.wheel.y);
        }
    };

    m_running = true;
    while (m_running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
            case SDL_EVENT_QUIT:
                m_running = false;
                break;
            default:
                break;
            }
            processInput(event);
            ImGui_ImplSDL3_ProcessEvent(&event);
        }

        state.Run();
    }

    state.WaitIdle();
    DebugInfo("SDLWindow quitting");
}