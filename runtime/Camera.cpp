//
// Created by y1 on 2025-11-14.
//

#include "Camera.h"

#include <algorithm>
#include <glm/detail/type_quat.hpp>
#include <glm/ext/quaternion_trigonometric.hpp>
#include <glm/gtc/quaternion.hpp>

namespace {
constexpr glm::vec3 kWorldUp{0.0f, 1.0f, 0.0f};

constexpr float kNear{0.01f};
constexpr float kFar{10000.0f};

constexpr float kPitchBound{89.0f};

constexpr float kMaxFov{179.0f};
constexpr float kMinFov{0.01f};

constexpr float kRatio{16.0f / 9.0f};

constexpr float kSensitivity{0.1f};
constexpr float kSpeed{0.1f};

} // namespace

Camera::Camera()
    : m_eye(glm::vec3(0.0f, 0.0f, 3.0f))
    , m_yaw(kDefaultYaw)
    , m_pitch(kDefaultPitch)
    , m_fov(kDefaultFov)
    , m_ratio(kRatio)
    , m_sensitivity(kSensitivity)
    , m_speed(kSpeed) {
    UpdateViewMatrix();
    UpdateProjectionMatrix();
}

void Camera::ProcessMovement(const glm::vec3 &direction) {
    CameraFrame frame = CalculateCameraFrame();

    if (direction == glm::vec3(0.0f, 0.0f, -1.0f)) {
        m_eye += frame.forward * m_speed;
    }

    if (direction == glm::vec3(0.0f, 0.0f, 1.0f)) {
        m_eye -= frame.forward * m_speed;
    }

    if (direction == glm::vec3(1.0f, 0.0f, 0.0f)) {
        m_eye += frame.right * m_speed;
    }

    if (direction == glm::vec3(-1.0f, 0.0f, 0.0f)) {
        m_eye -= frame.right * m_speed;
    }

    if (direction == glm::vec3(0.0f, 1.0f, 0.0f)) {
        m_eye += kWorldUp * m_speed;
    }

    if (direction == glm::vec3(0.0f, -1.0f, 0.0f)) {
        m_eye -= kWorldUp * m_speed;
    }

    UpdateViewMatrix();
}

void Camera::ProcessRotation(const glm::vec2 &offset) {
    m_yaw   += offset.x * m_sensitivity;
    m_pitch -= offset.y * m_sensitivity;

    m_pitch = std::clamp(m_pitch, -kPitchBound, kPitchBound);

    m_yaw = glm::mod(m_yaw, 360.f);

    UpdateViewMatrix();
}

void Camera::ProcessZoom(float offset) {
    float fov = m_fov - offset;

    fov   = glm::clamp(fov, kMinFov, kMaxFov);
    m_fov = fov;

    UpdateProjectionMatrix();
}

void Camera::SetRatio(float ratio) {
    m_ratio = ratio;
    UpdateProjectionMatrix();
}

void Camera::SetLocation(const glm::vec3 &location) {
    m_eye = location;
    UpdateViewMatrix();
}

void Camera::SetFov(float fov) {
    m_fov = fov;
    UpdateProjectionMatrix();
}

void Camera::SetRotation(float yaw, float pitch) {
    m_yaw   = yaw;
    m_pitch = pitch;

    UpdateViewMatrix();
}

void Camera::UpdateProjectionMatrix() {
    glm::mat4 projectionMatrix = glm::perspectiveRH_ZO(glm::radians(m_fov), m_ratio, kNear, kFar);

    // Flip the Y axis for Vulkan
    projectionMatrix[1][1] *= -1;

    m_projection = projectionMatrix;
}

void Camera::UpdateViewMatrix() {
    CameraFrame frame = CalculateCameraFrame();
    m_view            = glm::lookAt(m_eye, m_eye + frame.forward, frame.up);
}

Camera::CameraFrame Camera::CalculateCameraFrame() const {
    float yawRadian   = glm::radians(m_yaw);
    float pitchRadian = glm::radians(m_pitch);

    glm::vec3 forward;
    forward.x = cos(yawRadian) * cos(pitchRadian);
    forward.y = sin(pitchRadian);
    forward.z = sin(yawRadian) * cos(pitchRadian);
    forward   = glm::normalize(forward);

    glm::vec3 right = glm::normalize(glm::cross(forward, kWorldUp));

    const glm::vec3 up = glm::normalize(glm::cross(right, forward));

    return {forward, up, right};
}