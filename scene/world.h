#ifndef WORLD_H_
#define WORLD_H_
#include <glm/glm.hpp>
#include "light.h"
#include "geometry.h"
#include "../cuda-opengl.h"
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
	static float fov;
	// lights
	static Light lights;
	// geometries
	static std::vector<Geometry*> objects;
	// methods
	int num_triangles, num_objects;
	void GenerateGeometries();

	std::vector<float> vertex_buffer;
	std::vector<float> normal_buffer;
	std::vector<float> tex_buffer;
	std::vector<int> index_buffer;

	std::vector<uchar3> tex_images;
	std::vector<glm::ivec3> tex_offsets;

	std::vector<InstanceData> material;

	std::vector<glm::vec3> causticMap;

	glm::vec3 *vertexBuffer;
	glm::vec3 *normalBuffer;
	glm::vec2 *texBuffer;
	int *indexBuffer;
	InstanceData* materialBuffer;
	
	glm::vec3 *directLightsBuffer;
	glm::vec3 *directLightsColorBuffer;
	glm::vec3 *pointLightsBuffer;
	glm::vec3 *pointLightsColorBuffer;

	uchar3* texImagesBuffer;
	glm::ivec3* texOffsetBuffer;
	glm::vec3* causticBuffer;
	glm::ivec3* causticMapBuffer;
	glm::vec2* causticCoordsBuffer;
};

extern World g_world;
#endif