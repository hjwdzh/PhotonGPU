#ifndef WORLD_H_
#define WORLD_H_
#include <glm/glm.hpp>
#include "light.h"
#include "geometry.h"
#include "../bvh/bvh.h"
#include "../cuda-opengl.h"

#define PATH_DEPTH 10
#define PATH_MAX_DEPTH 3
#define NUM_SAMPLE 1

#define CAUSTIC_X_MIN -12.8f
#define CAUSTIC_MAP_DIS 0.05f
#define SCATTER_RADIUS 0.2f
#define CAUSTIC_W 512

struct InstanceData
{
	float kd, ks;
	int s, bvh_offset;
	float ka, kr, kf, nr, alpha, kt;
	glm::vec3 minPos, maxPos;
};

class World
{
public:
	World()
		: pause(0), rendering_mode(0)
	{}
	int pause;
	int rendering_mode;
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
	uchar3* environmentBuffer;
	glm::ivec3* texOffsetBuffer;

	glm::vec3* causticBuffer;
	glm::ivec3* causticMapBuffer;
	glm::vec2* causticCoordsBuffer;

	glm::vec3* scatterBuffer;
	glm::vec3* scatterPosBuffer;
	float* softShadowBuffer, *softShadowMap;
	std::vector<BVHData> bvhData;

	BVHData* bvhDataBuffer;

	void ProcessScattering();
};

extern World g_world;
#endif