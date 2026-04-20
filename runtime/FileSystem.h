//
// Created by y1 on 2026-04-18.
//

#pragma once 

#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace ntc {
std::string Read(std::string_view dir);

// Mesh
struct VertexData {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 tangent;
    glm::vec2 uv;
};

std::vector<VertexData> LoadMesh(std::string_view dir);

// Image
unsigned char *LoadImage(std::string_view dir, int &width, int &height);
}
