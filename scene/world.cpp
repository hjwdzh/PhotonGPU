#include "world.h"
#include "../cuda-opengl.h"
glm::vec3 World::camera_up;
glm::vec3 World::camera_lookat;
glm::vec3 World::camera;
float World::fov;
Light World::lights;
std::vector<Geometry*> World::objects;

void World::GenerateGeometries()
{
	num_triangles = 0;
	num_objects = objects.size();
	for (int i = 0; i < num_objects; ++i)
		num_triangles += objects[i]->vertex.size() / 3;
	vertex_buffer.resize(num_triangles * 9);
	normal_buffer.resize(num_triangles * 9);
	tex_buffer.resize(num_triangles * 6);
	index_buffer.resize(num_triangles);
	material.resize(num_objects);
	float* v_ptr = vertex_buffer.data(), *n_ptr = normal_buffer.data(), *t_ptr = tex_buffer.data();
	InstanceData* m_ptr = material.data();
	int s = 0;
	for (int i = 0; i < num_objects; ++i) {
		m_ptr->kd = objects[i]->kd;
		m_ptr->ks = objects[i]->ks;
		m_ptr->offset[0] = objects[i]->offset.x;
		m_ptr->offset[1] = objects[i]->offset.y;
		m_ptr->offset[2] = objects[i]->offset.z;
		m_ptr->axisX[0] = objects[i]->x_axis.x;
		m_ptr->axisX[1] = objects[i]->x_axis.y;
		m_ptr->axisX[2] = objects[i]->x_axis.z;
		m_ptr->axisY[0] = objects[i]->y_axis.x;
		m_ptr->axisY[1] = objects[i]->y_axis.y;
		m_ptr->axisY[2] = objects[i]->y_axis.z;
		m_ptr->scale[0] = objects[i]->s.x;
		m_ptr->scale[1] = objects[i]->s.y;
		m_ptr->scale[2] = objects[i]->s.z;
		for (int j = s / 3; j < s / 3 + objects[i]->vertex.size() / 3; ++j)
			index_buffer[j] = i;
		s += objects[i]->vertex.size();
		m_ptr->s = s;
		m_ptr->ka = objects[i]->ka;
		m_ptr->kr = objects[i]->kr;
		m_ptr->kf = objects[i]->kf;
		m_ptr->nr = objects[i]->nr;
		m_ptr->alpha = objects[i]->alpha;
		m_ptr++;

		glm::vec3& x = material[i].offset;
		glm::vec3& axisX = material[i].axisX;
		glm::vec3& axisY = material[i].axisY;
		glm::vec3& axisZ = glm::cross(axisX, axisY);
		glm::vec3& scale = material[i].scale;

		glm::mat4 rotate = glm::mat4(glm::vec4(axisX, 0), glm::vec4(axisY, 0), glm::vec4(axisZ, 0), glm::vec4(0, 0, 0, 1));
		glm::mat4 convert = glm::mat4(glm::vec4(1, 0, 0, 0), glm::vec4(0, 1, 0, 0), glm::vec4(0, 0, 1, 0), glm::vec4(x, 1))
			* rotate
			* glm::mat4(glm::vec4(scale.x, 0, 0, 0), glm::vec4(0, scale.y, 0, 0), glm::vec4(0, 0, scale.z, 0), glm::vec4(0, 0, 0, 1));

		for (auto& v : objects[i]->vertex) {
			glm::vec4 v4(v, 1);
			v4 = convert * v4;
			v = glm::vec3(v4.x, v4.y, v4.z);
		}
		for (auto& n : objects[i]->normal) {
			glm::vec4 n4(n, 0);
			n4 = convert * n4;
			n = glm::normalize(glm::vec3(n4.x, n4.y, n4.z));
		}
		memcpy(v_ptr, objects[i]->vertex.data(), objects[i]->vertex.size() * 3 * sizeof(float));
		v_ptr += objects[i]->vertex.size() * 3;
		memcpy(n_ptr, objects[i]->normal.data(), objects[i]->normal.size() * 3 * sizeof(float));
		n_ptr += objects[i]->normal.size() * 3;

		memcpy(t_ptr, objects[i]->uv.data(), objects[i]->uv.size() * 2 * sizeof(float));
		t_ptr += objects[i]->uv.size() * 2;

	}

	tex_offsets.resize(objects.size());
	int current_offset = 0;
	for (int i = 0; i < objects.size(); ++i) {
		tex_offsets[i] = glm::ivec3(current_offset, objects[i]->texImg.rows, objects[i]->texImg.cols);
		current_offset += objects[i]->texImg.rows * objects[i]->texImg.cols;
	}
	tex_images.resize(current_offset);
	for (int i = 0; i < objects.size(); ++i) {
		memcpy(tex_images.data() + tex_offsets[i].x, objects[i]->texImg.data, sizeof(uchar3) * objects[i]->texImg.rows * objects[i]->texImg.cols);
	}
	cudaMalloc(&vertexBuffer, sizeof(float) * vertex_buffer.size());
	cudaMemcpy(vertexBuffer, vertex_buffer.data(), sizeof(float) * vertex_buffer.size(), cudaMemcpyHostToDevice);
	cudaMalloc(&normalBuffer, sizeof(float) * normal_buffer.size());
	cudaMemcpy(normalBuffer, normal_buffer.data(), sizeof(float) * normal_buffer.size(), cudaMemcpyHostToDevice);
	cudaMalloc(&texBuffer, sizeof(float) * tex_buffer.size());
	cudaMemcpy(texBuffer, tex_buffer.data(), sizeof(float) * tex_buffer.size(), cudaMemcpyHostToDevice);
	cudaMalloc(&indexBuffer, sizeof(int) * index_buffer.size());
	cudaMemcpy(indexBuffer, index_buffer.data(), sizeof(int) * index_buffer.size(), cudaMemcpyHostToDevice);
	cudaMalloc(&materialBuffer, sizeof(InstanceData) * material.size());
	cudaMemcpy(materialBuffer, material.data(), sizeof(InstanceData) * material.size(), cudaMemcpyHostToDevice);

	cudaMalloc(&directLightsBuffer, sizeof(glm::vec3) * lights.direct_light_dir.size());
	cudaMemcpy(directLightsBuffer, lights.direct_light_dir.data(), sizeof(glm::vec3) * lights.direct_light_dir.size(), cudaMemcpyHostToDevice);
	cudaMalloc(&directLightsColorBuffer, sizeof(glm::vec3) * lights.direct_light_color.size());
	cudaMemcpy(directLightsColorBuffer, lights.direct_light_color.data(), sizeof(glm::vec3) * lights.direct_light_color.size(), cudaMemcpyHostToDevice);

	cudaMalloc(&pointLightsBuffer, sizeof(glm::vec3) * lights.point_light_pos.size());
	cudaMemcpy(pointLightsBuffer, lights.point_light_pos.data(), sizeof(glm::vec3) * lights.point_light_pos.size(), cudaMemcpyHostToDevice);
	cudaMalloc(&pointLightsColorBuffer, sizeof(glm::vec3) * lights.point_light_color.size());
	cudaMemcpy(pointLightsColorBuffer, lights.point_light_color.data(), sizeof(glm::vec3) * lights.point_light_color.size(), cudaMemcpyHostToDevice);

	cudaMalloc(&texOffsetBuffer, sizeof(glm::ivec3) * tex_offsets.size());
	cudaMemcpy(texOffsetBuffer, tex_offsets.data(), sizeof(glm::ivec3) * tex_offsets.size(), cudaMemcpyHostToDevice);
	cudaMalloc(&texImagesBuffer, sizeof(uchar3) * tex_images.size());
	cudaMemcpy(texImagesBuffer, tex_images.data(), sizeof(uchar3) * tex_images.size(), cudaMemcpyHostToDevice);

	causticMap.resize(512 * 512);
	cudaMalloc(&causticMapBuffer, sizeof(glm::ivec3) * causticMap.size());
	cudaMalloc(&causticBuffer, sizeof(glm::vec3) * causticMap.size());
	cudaMalloc(&causticCoordsBuffer, sizeof(glm::vec2) * causticMap.size());
}