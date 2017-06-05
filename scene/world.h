#ifndef WORLD_H_
#define WORLD_H_
#include <glm/glm.hpp>
#include "light.h"
#include "geometry.h"

struct InstanceData
{
	float kd, ks;
	glm::vec3 offset;
	glm::vec3 axisX, axisY;
	glm::vec3 scale;
	int s;
	float ka, kr, kf, nr, alpha;
};
class World
{
public:
	// camera
	static glm::vec3 camera_up, camera_lookat, camera;
	// lights
	static Light lights;
	// geometries
	static std::vector<Geometry*> objects;
	// methods
	int num_triangles, num_objects;
	void GenerateGeometries();
	float *vertexBuffer, *normalBuffer, *texBuffer;
	int *indexBuffer;
	InstanceData* materialBuffer;

};

extern World g_world;
#endif