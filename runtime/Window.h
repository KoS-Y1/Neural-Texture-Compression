//
// Created by y1 on 2026-04-18.
//

#pragma once

struct SDL_Window;

class Window {
public:
    Window();

    Window(const Window &)            = delete;
    Window &operator=(const Window &) = delete;
    Window(Window &&)                 = delete;
    Window &operator=(Window &&)      = delete;

    ~Window() = default;

    void Run();

private:
    SDL_Window *m_window;

    bool m_running{false};
};
