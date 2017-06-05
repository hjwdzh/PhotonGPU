#ifndef OBJLOADER_H
#define OBJLOADER_H
#include <vector>
#include <glm/glm.hpp>
class Geometry;

bool loadOBJ(
	const char * path, 
	std::vector<glm::vec3> & out_vertices,
	std::vector<glm::vec2> & out_uvs,
	std::vector<glm::vec3> & out_normals,
	bool smooth
);

bool loadScene(
	const char * path, std::vector<Geometry*>& geometries
);

#endif