#include "world.h"
#include "../cuda-opengl.h"
glm::vec3 World::camera_up;
glm::vec3 World::camera_lookat;
glm::vec3 World::camera;
Light World::lights;
std::vector<Geometry*> World::objects;

void World::GenerateGeometries()
{
	num_triangles = 0;
	num_objects = objects.size();
	for (int i = 0; i < num_objects; ++i)
		num_triangles += objects[i]->vertex.size() / 3;
	std::vector<float> vertex_buffer(num_triangles * 9);
	std::vector<float> normal_buffer(num_triangles * 9);
	std::vector<float> tex_buffer(num_triangles * 6);
	std::vector<int> index_buffer(num_triangles);
	std::vector<InstanceData> material(num_objects);
	float* v_ptr = vertex_buffer.data(), *n_ptr = normal_buffer.data(), *t_ptr = tex_buffer.data();
	InstanceData* m_ptr = material.data();
	int s = 0;
	for (int i = 0; i < num_objects; ++i) {
		memcpy(v_ptr, objects[i]->vertex.data(), objects[i]->vertex.size() * 3 * sizeof(float));
		v_ptr += objects[i]->vertex.size() * 3;
		memcpy(n_ptr, objects[i]->normal.data(), objects[i]->normal.size() * 3 * sizeof(float));
		n_ptr += objects[i]->normal.size() * 3;

		memcpy(t_ptr, objects[i]->uv.data(), objects[i]->uv.size() * 2 * sizeof(float));
		t_ptr += objects[i]->uv.size() * 2;
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
}