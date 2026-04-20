//
// Created by y1 on 2025-11-14.
//

#pragma once

#include <glm/glm.hpp>

struct CameraConfig;

class Camera {
public:
    static constexpr float kDefaultFov{60.0f};
    static constexpr float kDefaultYaw{-90.0f};
    static constexpr float kDefaultPitch{0.0f};

    Camera();

    Camera(const Camera &)            = delete;
    Camera &operator=(const Camera &) = delete;

    Camera(Camera &&) noexcept            = default;
    Camera &operator=(Camera &&) noexcept = default;

    void ProcessMovement(const glm::vec3 &direction);
    void ProcessRotation(const glm::vec2 &offset);
    void ProcessZoom(float offset);

    void SetRatio(float ratio);
    void SetLocation(const glm::vec3 &location);
    void SetFov(float fov);
    void SetRotation(float yaw, float pitch);

    [[nodiscard]] glm::mat4 GetViewMatrix() const { return m_view; }

    [[nodiscard]] glm::mat4 GetProjectionMatrix() const { return m_projection; }

    [[nodiscard]] glm::vec3 GetLocation() const { return m_eye; }

    [[nodiscard]] float GetYaw() const { return m_yaw; }

    [[nodiscard]] float GetPitch() const { return m_pitch; }

    [[nodiscard]] float GetFov() const { return m_fov; }


private:
    struct CameraFrame {
        glm::vec3 forward;
        glm::vec3 up;
        glm::vec3 right;
    };

    glm::mat4 m_view;
    glm::mat4 m_projection;

    glm::vec3 m_eye;

    float m_yaw;
    float m_pitch;

    float m_fov;
    float m_ratio;

    float m_sensitivity;
    float m_speed;

    void UpdateViewMatrix();
    void UpdateProjectionMatrix();

    CameraFrame CalculateCameraFrame() const;
};
