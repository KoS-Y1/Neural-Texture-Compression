//
// Created by y1 on 2026-04-18.
//

#include "FileSystem.h"

#include <array>
#include <filesystem>
#include <fstream>

#include <stb_image.h>
#include <tiny_obj_loader.h>

#include "Debug.h"

namespace ntc {
std::string Read(std::string_view dir) {
    std::ifstream file(std::filesystem::path{dir}, std::ios::binary | std::ios::ate);
    DebugCheckCritical(file.is_open(), "Failed to open file {}", std::string(dir));

    int64_t     fileSize = file.tellg();
    std::string buf(fileSize, '\0');
    file.seekg(0, std::ios::beg);
    file.read(&buf[0], fileSize);
    file.close();

    return buf;
}

std::vector<VertexData> LoadMesh(std::string_view dir) {
    tinyobj::ObjReader reader;

    DebugCheckCritical(reader.ParseFromFile(std::string(dir)), "Failed to parse obj file {}: {}", std::string(dir), reader.Error());

    if (!reader.Warning().empty()) {
        DebugWarning("Warning parsing obj file {}: {}", std::string(dir), reader.Warning());
    }

    const tinyobj::attrib_t            &attrib    = reader.GetAttrib();
    const std::vector<tinyobj::real_t> &positions = attrib.vertices;
    const std::vector<tinyobj::real_t> &normals   = attrib.normals;
    const std::vector<tinyobj::real_t> &texCoords = attrib.texcoords;

    std::vector<VertexData> vertices;

    // Loop over shapes
    for (const tinyobj::shape_t &shape: reader.GetShapes()) {
        const tinyobj::mesh_t &mesh = shape.mesh;
        DebugInfo("Loading model shape {}", shape.name);

        size_t i = 0;

        // Loop over polygons in the mesh
        for (const size_t f: mesh.num_face_vertices) {
            DebugCheckCritical(f == 3, "Mesh must be triangulated");

            // Calculate tangent space
            std::array<glm::vec3, 3> vPos{};
            std::array<glm::vec2, 3> uvs{};
            for (size_t v = 0; v < f; ++v) {
                const tinyobj::index_t &idx = mesh.indices[i + v];
                DebugCheckCritical(idx.texcoord_index >= 0, "Missing texcoord");

                vPos[v] = glm::vec3{
                    -positions[3 * idx.vertex_index + 0],
                    positions[3 * idx.vertex_index + 1],
                    positions[3 * idx.vertex_index + 2],
                };
                uvs[v] = glm::vec2{
                    texCoords[2 * idx.texcoord_index + 0],
                    texCoords[2 * idx.texcoord_index + 1],
                };
            }

            glm::vec3 deltaPos1 = vPos[1] - vPos[0];
            glm::vec3 deltaPos2 = vPos[2] - vPos[0];
            glm::vec2 deltaUV1  = uvs[1] - uvs[0];
            glm::vec2 deltaUV2  = uvs[2] - uvs[0];
            float     r         = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x);
            glm::vec3 tangent   = (deltaPos1 * deltaUV2.y - deltaPos2 * deltaUV1.y) * r;

            // Loop over vertices in the polygon
            for (size_t v = 0; v < f; ++v) {
                const tinyobj::index_t &idx = mesh.indices[i + v];

                DebugCheckCritical(idx.normal_index >= 0, "Missing normal");
                DebugCheckCritical(idx.texcoord_index >= 0, "Missing texcoord");

                // X axis is flipped because Blender uses right-handed coordinates
                vertices.emplace_back(
                    glm::vec3{
                        -positions[3 * idx.vertex_index + 0],
                        positions[3 * idx.vertex_index + 1],
                        positions[3 * idx.vertex_index + 2],
                    },
                    glm::vec3{
                        -normals[3 * idx.normal_index + 0],
                        normals[3 * idx.normal_index + 1],
                        normals[3 * idx.normal_index + 2],
                    },
                    tangent,
                    glm::vec2{
                        texCoords[2 * idx.texcoord_index + 0],
                        texCoords[2 * idx.texcoord_index + 1],
                    }
                );
            }

            i += f;
        }
    }

    return vertices;
}

unsigned char *LoadImage(std::string_view dir, int &width, int &height) {
    int channels;
    stbi_set_flip_vertically_on_load(true);
    unsigned char *data = stbi_load(std::string(dir).c_str(), &width, &height, &channels, STBI_rgb_alpha);
    DebugCheckCritical(data != nullptr, "Failed to load image {}", dir);

    return data;
}

} // namespace ntc