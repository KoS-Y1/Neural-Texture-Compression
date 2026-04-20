//
// Created by y1 on 2026-04-18.
//

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include "Debug.h"
#include "Window.h"

int main() {
    DebugCheckCritical(SDL_Init(SDL_INIT_VIDEO), "Failed to init SDL");
    atexit(SDL_Quit);

    DebugCheckCritical(SDL_Vulkan_LoadLibrary(nullptr), "Failed to load SDL_Vulkan_LoadLibrary");
    atexit(SDL_Vulkan_UnloadLibrary);
    Window window;
    window.Run();
}