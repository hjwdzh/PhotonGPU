/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

// Utilities and system includes

#include <helper_cuda.h>
#include "../scene/world.h"

#define NUM_SAMPLE 1

__device__ float clamp(float x, float a, float b)
{
	return max(a, min(b, x));
}

__device__ int clamp(int x, int a, int b)
{
	return max(a, min(b, x));
}

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt(float r, float g, float b)
{
	r = clamp(r, 0.0f, 255.0f);
	g = clamp(g, 0.0f, 255.0f);
	b = clamp(b, 0.0f, 255.0f);
	return (int(b) << 16) | (int(g) << 8) | int(r);
}

__device__ __host__
float rayIntersectsTriangle(glm::vec3 p, glm::vec3 d,
glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, float& u, float& v) {

	glm::vec3 e1 = v1 - v0;
	glm::vec3 e2 = v2 - v0;
	glm::vec3 h = glm::cross(d, e2);
	float a = glm::dot(e1, h);

	if (a > -0.00001 && a < 0.00001)
		return -1;

	float f = 1 / a;
	glm::vec3 s = p - v0;
	u = f * glm::dot(s, h);

	if (u < 0.0 || u > 1.0)
		return -1;

	glm::vec3 q = glm::cross(s, e1);
	v = f * glm::dot(d, q);

	if (v < 0.0 || u + v > 1.0)
		return -1;

	// at this stage we can compute t to find out where
	// the intersection point is on the line
	float t = f * glm::dot(e2, q);

	if (t > 0.00001) // ray intersection
		return t;

	else // this means that there is a line intersection
		// but not a ray intersection
		return -1;
}

__device__ __host__
float tracing(glm::vec3 ray_o_o, glm::vec3 ray_t_o, float shadow, int& tri, int& obj, glm::vec4& hit_point, glm::vec2& uv, glm::vec4& normal,
	InstanceData* instanceData, glm::vec3* vertexBuffer, glm::vec3* normalBuffer, glm::vec2* texBuffer, int num_object) {
	float depth = 1e30;
	obj = -1;
	tri = -1;
	int j = 0;
	for (int k = 0; k < num_object; ++k) {
		glm::vec3& x = instanceData[k].offset;
		glm::vec3& axisX = instanceData[k].axisX;
		glm::vec3& axisY = instanceData[k].axisY;
		glm::vec3& axisZ = glm::cross(axisX, axisY);
		glm::vec3& scale = instanceData[k].scale;

		glm::mat4 rotate = glm::mat4(glm::vec4(axisX, 0), glm::vec4(axisY, 0), glm::vec4(axisZ, 0), glm::vec4(0, 0, 0, 1));
		glm::mat4 convert = glm::mat4(glm::vec4(1, 0, 0, 0), glm::vec4(0, 1, 0, 0), glm::vec4(0, 0, 1, 0), glm::vec4(x, 1))
			* rotate
			* glm::mat4(glm::vec4(scale.x, 0, 0, 0), glm::vec4(0, scale.y, 0, 0), glm::vec4(0, 0, scale.z, 0), glm::vec4(0, 0, 0, 1));
		glm::mat4 inv_convert = glm::mat4(glm::vec4(1 / scale.x, 0, 0, 0), glm::vec4(0, 1 / scale.y, 0, 0), glm::vec4(0, 0, 1 / scale.z, 0), glm::vec4(0, 0, 0, 1))
			* glm::transpose(rotate)
			* glm::mat4(glm::vec4(1, 0, 0, 0), glm::vec4(0, 1, 0, 0), glm::vec4(0, 0, 1, 0), glm::vec4(-x, 1));

		glm::vec4 ray_o4 = (inv_convert * glm::vec4(ray_o_o, 1));
		glm::vec4 ray_t4 = (inv_convert * glm::vec4(ray_t_o, 0));
		glm::vec3 ray_o(ray_o4.x, ray_o4.y, ray_o4.z);
		glm::vec3 ray_t(ray_t4.x, ray_t4.y, ray_t4.z);
		int next_object = instanceData[k].s;
		while (j < next_object) {
			glm::vec3& v1 = vertexBuffer[j];
			glm::vec3& v2 = vertexBuffer[j + 1];
			glm::vec3& v3 = vertexBuffer[j + 2];
			float u, v;
			float t = rayIntersectsTriangle(ray_o, ray_t, v1, v2, v3, u, v);
			if (t > 0 && t < depth) {
				depth = t;
				hit_point = (convert * (ray_o4 + ray_t4 * depth));
				if (shadow >= 0) {
					if (t < shadow) {
						return t;
					}
				}
				else {
					obj = k;
					tri = j;
					glm::vec3& n1 = normalBuffer[j];
					glm::vec3& n2 = normalBuffer[j + 1];
					glm::vec3& n3 = normalBuffer[j + 2];
					normal = (convert * glm::vec4(u * (n2 - n1) + v * (n3 - n1) + n1, 0));
					glm::vec2& uv1 = texBuffer[j];
					glm::vec2& uv2 = texBuffer[j + 1];
					glm::vec2& uv3 = texBuffer[j + 2];
					uv = uv1 + u * (uv2 - uv1) + v * (uv3 - uv1);
				}
			}
			j += 3;
		}
	}
	normal = normalize(normal);
	return depth;
}

__device__ __host__
glm::vec3 lighting(glm::vec3 start_camera, glm::vec3 point, glm::vec3 normal, float tri_index, glm::vec2 uv, float obj_index, glm::vec3 orig_color) {
	/*
	float kd = texture2D(materialSampler, vec2(0.5 / MATERIAL_LEN, (obj_index + 0.5) / num_object)).r;
	float ks = texture2D(materialSampler, vec2(1.5 / MATERIAL_LEN, (obj_index + 0.5) / num_object)).r;
	float ka = texture2D(materialSampler, vec2(16.5 / MATERIAL_LEN, (obj_index + 0.5) / num_object)).r;
	float alpha = texture2D(materialSampler, vec2(20.5 / MATERIAL_LEN, (obj_index + 0.5) / num_object)).r;
	int tex = int(0.1 + texture2D(materialSampler, vec2(2.5 / MATERIAL_LEN, (obj_index + 0.5) / num_object)).r);
	orig_color = texture2D(renderSampler[tex], uv).rgb;
	vec3 color = ka * orig_color * ambient;
	vec3 eye_dir = normalize(start_camera - point);
	float t1, t2;
	vec2 v1;
	vec3 v2, v3;
	for (int i = 0; i < num_direct_light; ++i) {
		float intensity = dot(-direct_lights[i], normal) * dot(eye_dir, normal);
		if (intensity < 0)
			continue;
		float depth = tracing(point, -direct_lights[i], 100, t1, t2, v2, v1, v3);
		if (depth < 1000)
			continue;
		color += intensity * (orig_color * direct_lights_color[i] * kd
			+ clamp(pow(dot(reflect(direct_lights[i], normal), eye_dir), 20), 0, 1) * ks * direct_lights_color[i]);
	}
	for (int i = 0; i < num_point_light; ++i) {
		vec3 dis = point - point_lights[i];
		float len = length(dis);
		float l = 1 / (len * len);
		dis = normalize(dis);
		float intensity = dot(-dis, normal) * dot(eye_dir, normal);
		if (intensity < 0)
			continue;
		float depth = tracing(point, -dis, len, t1, t2, v2, v1, v3);
		if (depth < len)
			continue;
		vec3 para = kd * l * point_lights_color[i];
		color = color + intensity * (orig_color * para
			+ clamp(pow(dot(reflect(dis, normal), eye_dir), alpha), 0, 1) * ks * point_lights_color[i]);
	}
	*/
	return glm::vec3(0, 0, 0);
}


__global__ void
render(unsigned int *g_odata, int imgw, int imgh,
glm::vec3 cam_up, glm::vec3 cam_forward, glm::vec3 right, glm::vec3 cam_pos, float dis_per_pix,
	InstanceData* instanceData, glm::vec3* vertexBuffer, glm::vec3* normalBuffer, glm::vec2* texBuffer,
	int num_object)
{
	extern __shared__ uchar4 sdata[];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;

	glm::vec3 ray_p = cam_pos;
//	uchar4 c4 = make_uchar4((float)x / imgw * 255, (float)y / imgh * 255, 0, 255);
	glm::vec3 color(255, 182, 0);
	glm::vec2 noises[NUM_SAMPLE];
	noises[0] = glm::vec2(0, 0);
	for (int i = 0; i < NUM_SAMPLE; ++i) {
		glm::vec3 ray_d = glm::normalize(cam_forward + (noises[i].x + x - imgw / 2) * dis_per_pix * right + (noises[i].y + y - imgh / 2) * dis_per_pix * cam_up);
		uchar4 c4 = make_uchar4(ray_d.x * 255.0, ray_d.y * 255.0, ray_d.z * 255.0, 255);
		g_odata[y*imgw + x] = rgbToInt(c4.x, c4.y, c4.z);
		int tri_index, obj_index;
		glm::vec4 hit_point;
		glm::vec2 uv;
		glm::vec4 normal;
		glm::vec3 orig_color;
		float depth = tracing(ray_p, ray_d, -1, tri_index, obj_index, hit_point, uv, normal, 
			instanceData, vertexBuffer, normalBuffer, texBuffer, num_object);
		if (depth < 1e20) {
			color = glm::vec3(uv.x * 255, uv.y * 255, 255);
//			color += lighting(ray_p, hit_point, normal, tri_index, uv, obj_index, orig_color);
		}
		else {
			color = glm::vec3(0, 0, 0);
		}
	}
	color /= NUM_SAMPLE;
	uchar4 c4 = make_uchar4(color.r, color.g, color.b, 255);
	g_odata[y*imgw + x] = rgbToInt(c4.x, c4.y, c4.z);
}

extern "C" void
cudaRender(dim3 grid, dim3 block, int sbytes, unsigned int *g_odata, int imgw, int imgh)
{
	float dis_per_pix = tan(World::fov * 0.5 * 3.141592654 / 180.0) / (imgw / 2);
	glm::vec3 right = glm::cross(World::camera_lookat, World::camera_up);
	render << < grid, block, sbytes >> >(g_odata, imgw, imgh,
		World::camera_up, World::camera_lookat, right, World::camera, dis_per_pix,
		g_world.materialBuffer, g_world.vertexBuffer, g_world.normalBuffer, g_world.texBuffer,
		g_world.num_objects);
}