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

#define PATH_DEPTH 10
#define PATH_MAX_DEPTH 3
#define NUM_SAMPLE 1
#define CAUSTIC_X_MIN -12.8
#define CAUSTIC_MAP_DIS 0.05

__device__ __host__ float clamp(float x, float a, float b)
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

__device__ glm::vec3 Inttorgb(int x)
{
	glm::vec3 rgb;
	rgb.b = (x >> 16);
	rgb.g = (x >> 8) & 0xff;
	rgb.r = (x & 0xff);
	return rgb;
}

__device__ __host__
float rayIntersectsTriangle(glm::vec3& p, glm::vec3& d,
glm::vec3& v0, glm::vec3& v1, glm::vec3& v2, float& u, float& v) {

	glm::vec3 e1 = v1 - v0;
	glm::vec3 e2 = v2 - v0;
	glm::vec3 h = glm::cross(d, e2);
	float a = glm::dot(e1, h);

	if (a > -0.00001 && a < 0.00001)
		return -1;

	float f = 1 / a;
	glm::vec3 s = p - v0;
	u = f * glm::dot(s, h);

	if (u < -1e-3 || u > 1 + 1e-3)
		return -1;

	glm::vec3 q = glm::cross(s, e1);
	v = f * glm::dot(d, q);

	if (v < -1e-3 || u + v > 1 + 1e-3)
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
void swaps(float& x, float& y) {
	float t = x;
	x = y, y = t;
}

__device__ __host__
int BoundingBoxIntersect(glm::vec3& ray_o, glm::vec3& ray_t, glm::vec3& minP, glm::vec3& maxP) {
	auto r = ray_t + glm::vec3(1e-6, 1e-6, 1e-6);
	auto rinv = glm::vec3(1 / r.x, 1 / r.y, 1 / r.z);
	float tx1 = (minP.x - ray_o.x)*rinv.x;
	float tx2 = (maxP.x - ray_o.x)*rinv.x;
	float tmin, tmax;
	if (rinv.x > 0)
		tmin = tx1, tmax = tx2;
	else
		tmin = tx2, tmax = tx1;

	float ty1 = (minP.y - ray_o.y)*rinv.y;
	float ty2 = (maxP.y - ray_o.y)*rinv.y;

	if (rinv.y > 0)
		tmin = max(tmin, ty1),
		tmax = min(tmax, ty2);
	else
		tmin = max(tmin, ty2),
		tmax = min(tmax, ty1);

	float tz1 = (minP.z - ray_o.z)*rinv.z;
	float tz2 = (maxP.z - ray_o.z)*rinv.z;

	if (rinv.z > 0)
		tmin = max(tmin, tz1),
		tmax = min(tmax, tz2);
	else
		tmin = max(tmin, tz2),
		tmax = min(tmax, tz1);
	return tmax >= tmin;

}

__device__ __host__
float tracing(glm::vec3& ray_o, glm::vec3& ray_t, float shadow, int& tri, int& obj, glm::vec3& hit_point, glm::vec2& uv, glm::vec3& normal,
InstanceData* instanceData, glm::vec3* vertexBuffer, glm::vec3* normalBuffer, glm::vec2* texBuffer, int num_object) {
	float depth = 1e30;
	obj = -1;
	tri = -1;
	int j = 0;
	for (int k = 0; k < num_object; ++k) {
		int next_object = instanceData[k].s;
		if ((k < 1 || k >= 7) && !BoundingBoxIntersect(ray_o, ray_t, instanceData[k].minPos, instanceData[k].maxPos)) {
			j = next_object;
			continue;
		}
		while (j < next_object) {
			glm::vec3& v1 = vertexBuffer[j];
			glm::vec3& v2 = vertexBuffer[j + 1];
			glm::vec3& v3 = vertexBuffer[j + 2];
			float u, v;
			float t = rayIntersectsTriangle(ray_o, ray_t, v1, v2, v3, u, v);
			if (t > 1e-2 && t < depth) {
				depth = t;
				hit_point = ((ray_o + ray_t * depth));
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
					normal = u * (n2 - n1) + v * (n3 - n1) + n1;
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
glm::vec3 fetchTex(glm::vec2& uv, int objIndex, uchar3* imagesBuffer, glm::ivec3* imageOffsetBuffer)
{
	glm::ivec3& info = imageOffsetBuffer[objIndex];
	int offset = info.x;
	int iy = info.y;
	int ix = info.z;
	float x = ix * uv.x;
	float y = iy * uv.y;
	int lx = x, ly = y;
	int rx = lx + 1, ry = ly + 1;
	float wx = x - lx, wy = y - ly;
	if (lx < 0)
		lx += wx;
	if (ly < 0)
		ly += wy;
	if (rx >= ix)
		rx -= ix;
	if (ry >= iy)
		ry -= iy;
	int ind1 = offset + ly * ix + lx;
	int ind2 = offset + ly * ix + rx;
	int ind3 = offset + ry * ix + lx;
	int ind4 = offset + ry * ix + rx;
	uchar3& c1 = imagesBuffer[ind1];
	uchar3& c2 = imagesBuffer[ind2];
	uchar3& c3 = imagesBuffer[ind3];
	uchar3& c4 = imagesBuffer[ind4];
	float cx = (c1.x * (1 - wx) + c2.x * wx) * (1 - wy) + (c3.x * (1 - wx) + c4.x * wx) * wy;
	float cy = (c1.y * (1 - wx) + c2.y * wx) * (1 - wy) + (c3.y * (1 - wx) + c4.y * wx) * wy;
	float cz = (c1.z * (1 - wx) + c2.z * wx) * (1 - wy) + (c3.z * (1 - wx) + c4.z * wx) * wy;
	return glm::vec3(cz, cy, cx);
}

__device__ __host__
void projectCaustic(glm::vec3& ray_o, glm::vec3& ray_t, glm::vec3 &color,
InstanceData* instanceData, glm::vec3* vertexBuffer, glm::vec3* normalBuffer, glm::vec2* texBuffer, int num_object,
glm::vec3& light, glm::vec2& coords, uchar3* texImages, glm::ivec3* imageOffsets) {
	int tri_index, obj_index;
	glm::vec3 hit_point, normal;
	glm::vec2 uv;
	float depth = tracing(ray_o, ray_t, -1, tri_index, obj_index, hit_point, uv, normal, instanceData, vertexBuffer, normalBuffer, texBuffer, num_object);
	glm::vec3 orig_color = fetchTex(uv, obj_index, texImages, imageOffsets) / 255.0f;
	int steps = 0;
	float intensity = 1;
	while (depth < 1e20 && (instanceData[obj_index].kr != 0 || instanceData[obj_index].kf != 0)) {
		if (instanceData[obj_index].kf != 0) {
			float nr = instanceData[obj_index].nr;
			float cost = glm::dot(normal, ray_t);
			if (cost < 0) {
				nr = 1 / nr;
				cost = -cost;
			}
			else {
				normal = -normal;
			}
			float rootContent = 1 - nr * nr * (1 - cost * cost);
			if (rootContent >= 0) {
				ray_o = glm::vec3(hit_point.x, hit_point.y, hit_point.z);
				ray_t = (nr * cost - sqrt(rootContent)) * normal + nr * ray_t;
				intensity *= instanceData[obj_index].kf;
			}
			else {
				ray_o = glm::vec3(hit_point.x, hit_point.y, hit_point.z);
				ray_t = glm::reflect(ray_t, glm::vec3(normal.x, normal.y, normal.z));
			}
		}
		else if (instanceData[obj_index].kr != 0) {
			ray_o = glm::vec3(hit_point.x, hit_point.y, hit_point.z);
			ray_t = glm::reflect(ray_t, glm::vec3(normal.x, normal.y, normal.z));
			intensity *= instanceData[obj_index].kr;
		}
		steps++;
		if (steps > 2)
			break;
		depth = tracing(ray_o, ray_t, -1, tri_index, obj_index, hit_point, uv, normal, instanceData, vertexBuffer, normalBuffer, texBuffer, num_object);
	}
	if (obj_index == 0 && steps > 0) {
		float x = (hit_point.x - CAUSTIC_X_MIN) / CAUSTIC_MAP_DIS;
		float y = (hit_point.z - CAUSTIC_X_MIN) / CAUSTIC_MAP_DIS;
		coords = glm::vec2(x, y);
		light = intensity * color * orig_color;
	}
	else {
		coords = glm::vec2(-1, -1);
		light = glm::vec3(0, 0, 0);
	}
}


__device__ __host__
glm::vec3 lighting(glm::vec3& start_camera, glm::vec3& point, glm::vec3& normal, int tri_index, glm::vec2& uv, int obj_index,
InstanceData* instanceData, glm::vec3* vertexBuffer, glm::vec3* normalBuffer, glm::vec2* texBuffer, int num_object,
int num_direct_light, glm::vec3* direct_lights, glm::vec3* direct_lights_color,
int num_point_light, glm::vec3* point_lights, glm::vec3* point_lights_color, glm::vec3& ambient,
uchar3* imagesBuffer, glm::ivec3* imageOffsetBuffer, glm::vec3& orig_color, glm::vec3* causticMap) {
	float kd = instanceData[obj_index].kd;
	float ks = instanceData[obj_index].ks;//texture2D(materialSampler, vec2(1.5 / MATERIAL_LEN, (obj_index + 0.5) / num_object)).r;
	float ka = instanceData[obj_index].ka;// texture2D(materialSampler, vec2(16.5 / MATERIAL_LEN, (obj_index + 0.5) / num_object)).r;
	float alpha = instanceData[obj_index].alpha;// texture2D(materialSampler, vec2(20.5 / MATERIAL_LEN, (obj_index + 0.5) / num_object)).r;

	orig_color = fetchTex(uv, obj_index, imagesBuffer, imageOffsetBuffer);
	//	int tex = int(0.1 + texture2D(materialSampler, vec2(2.5 / MATERIAL_LEN, (obj_index + 0.5) / num_object)).r);
	//	orig_color = texture2D(renderSampler[tex], uv).rgb;
	glm::vec3 color = ka * orig_color * ambient;
	glm::vec3 eye_dir = normalize(start_camera - point);
	int t1, t2;
	glm::vec2 v1;
	glm::vec3 v2, v3;
	for (int i = 0; i < num_direct_light; ++i) {
		float intensity = glm::dot(-direct_lights[i], normal) * glm::dot(eye_dir, normal);
		if (intensity < 0)
			continue;
		float depth = tracing(point, -direct_lights[i], 100, t1, t2, v2, v1, v3, instanceData, vertexBuffer, normalBuffer, texBuffer, num_object);
		if (depth < 1000) {
			continue;
		}
		color += intensity * (orig_color * direct_lights_color[i] * kd
			+ clamp((float)pow(glm::dot(glm::reflect(direct_lights[i], normal), eye_dir), alpha), 0.0f, 1.f) * ks * direct_lights_color[i]);
	}
	for (int i = 0; i < num_point_light; ++i) {
		glm::vec3 dis = point - point_lights[i];
		float len = glm::length(dis);
		float l = 1 / (len * len);
		dis = normalize(dis);
		float intensity = glm::dot(-dis, normal) * glm::dot(eye_dir, normal);
		if (intensity < 0)
			continue;
		float depth = tracing(point, -dis, len, t1, t2, v2, v1, v3, instanceData, vertexBuffer, normalBuffer, texBuffer, num_object);
		if (depth < len)
			continue;
		glm::vec3 para = kd * l * point_lights_color[i];
		color = color + intensity * (orig_color * para
			+ clamp((float)pow(dot(reflect(dis, normal), eye_dir), alpha), 0.f, 1.f) * ks * point_lights_color[i]);
	}
	if (obj_index == 0) {
		int rx = (point.x - CAUSTIC_X_MIN) / CAUSTIC_MAP_DIS;
		int ry = (point.z - CAUSTIC_X_MIN) / CAUSTIC_MAP_DIS;
		if (rx < 512 && ry < 512 && rx >= 0 && ry >= 0) {
			glm::vec3 caustic = causticMap[ry * 512 + rx];
			color = color + glm::dot(eye_dir, normal) * kd * caustic;
			color.x = clamp(color.x, 0.0f, 255.f);
			color.y = clamp(color.y, 0.0f, 255.f);
			color.z = clamp(color.z, 0.0f, 255.f);
		}
	}
	return color;
}


__global__ void
render(unsigned int *g_odata, int imgw, int imgh,
glm::vec3 cam_up, glm::vec3 cam_forward, glm::vec3 right, glm::vec3 cam_pos, float dis_per_pix,
InstanceData* instanceData, glm::vec3* vertexBuffer, glm::vec3* normalBuffer, glm::vec2* texBuffer, int num_object,
int num_direct_lights, glm::vec3* direct_lights, glm::vec3* direct_lights_color,
int num_point_lights, glm::vec3* point_lights, glm::vec3* point_lights_color, glm::vec3 ambient,
uchar3* imagesBuffer, glm::ivec3* imageOffsetBuffer,
glm::vec3* causticMap)
{
	extern __shared__ uchar4 sdata[];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;

	glm::vec3 ray_p = cam_pos;
	glm::vec3 ray_d = glm::normalize(cam_forward + (x - imgw / 2) * dis_per_pix * right + (y - imgh / 2) * dis_per_pix * cam_up);
	glm::vec3 color(0, 0, 0);
	int tri_index, obj_index;
	int path_state[PATH_DEPTH];
	int mat_stack[PATH_DEPTH];
	glm::vec3 light_stack[PATH_DEPTH];
	glm::vec3 color_stack[PATH_DEPTH];
	glm::vec3 from_stack[PATH_DEPTH];
	glm::vec3 to_stack[PATH_DEPTH];
	glm::vec3 normal_stack[PATH_DEPTH];
	int node = 0;
	path_state[node] = 0;
	from_stack[node] = ray_p;
	to_stack[node] = ray_d;
	color_stack[node] = glm::vec3(0, 0, 0);
	light_stack[node] = glm::vec3(0, 0, 0);
	float nr;
	int hit_mat = 0;
	glm::vec3 hit_point;
	glm::vec2 uv;
	glm::vec3 normal;
	while (node >= 0) {
		if (path_state[node] == 0) {
			path_state[node] = 1;
			float depth;
			depth = tracing(from_stack[node], to_stack[node], -1, tri_index, obj_index, hit_point, uv, normal, instanceData, vertexBuffer, normalBuffer, texBuffer, num_object);
			if (depth < 1e20) {
				glm::vec3 orig_color;
				light_stack[node] = lighting(from_stack[node], hit_point, normal, tri_index, uv, obj_index, instanceData, vertexBuffer, normalBuffer, texBuffer, num_object,
					num_direct_lights, direct_lights, direct_lights_color, num_point_lights, point_lights, point_lights_color, ambient,
					imagesBuffer, imageOffsetBuffer, orig_color, causticMap);
				color_stack[node] = orig_color;
				normal_stack[node] = normal;
				ray_d = to_stack[node];
				to_stack[node] = hit_point;
				mat_stack[node] = obj_index;
				float kr = instanceData[obj_index].kr;
				if (kr > 0 && node < PATH_DEPTH - 1) {
					color_stack[node] = instanceData[obj_index].kr * color_stack[node];
					node += 1;
					path_state[node] = 0;
					from_stack[node] = hit_point;
					to_stack[node] = ray_d - 2 * glm::dot(ray_d, normal) * normal;
					light_stack[node] = glm::vec3(0, 0, 0);
					continue;
				}
			}
			else {
				path_state[node] = 3;
			}
		}
		if (path_state[node] == 1) {
			path_state[node] = 2;
			obj_index = mat_stack[node];
			float kf = instanceData[obj_index].kf;
			if (kf > 0 && node < PATH_DEPTH - 1) {
				nr = instanceData[obj_index].nr;
				normal = normal_stack[node];
				ray_d = glm::normalize(to_stack[node] - from_stack[node]);
				float cost = glm::dot(normal, ray_d);
				if (cost < 0) {
					nr = 1 / nr;
					cost = -cost;
				}
				else {
					normal = -normal;
				}
				float rootContent = 1 - nr * nr * (1 - cost * cost);
				if (rootContent >= 0) {
					color_stack[node] = instanceData[obj_index].kf * color_stack[node];
					rootContent = sqrt(rootContent);
					node += 1;
					path_state[node] = 0;
					from_stack[node] = to_stack[node - 1];
					to_stack[node] = (nr * cost - rootContent) * normal + nr * ray_d;
					light_stack[node] = glm::vec3(0, 0, 0);
					continue;
				}
				else {
					float kr = 1;
					if (kr > 0 && node < PATH_DEPTH - 1) {
						light_stack[node] = glm::vec3(0, 0, 0);
						node += 1;
						path_state[node] = 0;
						from_stack[node] = to_stack[node - 1];
						to_stack[node] = ray_d - 2 * glm::dot(ray_d, normal) * normal;
						light_stack[node] = glm::vec3(0, 0, 0);
						continue;
					}
					else {
						g_odata[y*imgw + x] = 0;
						return;
					}
				}
			}
		}
		if (path_state[node] == 2) {
			path_state[node] = 3;
			obj_index = mat_stack[node];
			/*float ks = texture2D(materialSampler, vec2(1.5 / MATERIAL_LEN, (obj_index + 0.5) / num_object)).r;
			if (hit_mat < use_path && node < PATH_DEPTH - 1 && ks > 0) {
			normal = normal_stack[node];
			ray_t = normalize(to_stack[node] - from_stack[node]);
			float alpha = texture2D(materialSampler, vec2(20.5 / MATERIAL_LEN, (obj_index + 0.5) / num_object)).r;
			hit_mat += 1;
			node += 1;
			path_state[node] = 0;
			from_stack[node] = to_stack[node - 1];
			to_stack[node] = phoneSample(ray_t, normal, alpha);
			continue;
			}*/
		}
		if (path_state[node] == 3) {
			if (node == 0)
				break;
			int obj_index = mat_stack[node - 1];
			if (path_state[node - 1] == 1) {
				light_stack[node - 1] = (1 - instanceData[obj_index].kr) * light_stack[node - 1]
					+ color_stack[node - 1] * light_stack[node] / 255.0f;
			}
			else
				if (path_state[node - 1] == 2) {
					light_stack[node - 1] = (1 - instanceData[obj_index].kf) * light_stack[node - 1]
						+ color_stack[node - 1] * light_stack[node] / 255.0f;
				}
				else {
					hit_mat -= 1;
					normal = normal_stack[node - 1];
					ray_d = glm::normalize(to_stack[node - 1] - from_stack[node - 1]);
					float alpha = instanceData[obj_index].alpha;
					light_stack[node - 1] = (1 - instanceData[obj_index].ks) * light_stack[node - 1]
						+ instanceData[obj_index].ks * color_stack[node - 1] * light_stack[node] * glm::dot(-ray_d, normal) / 255.0f;
				}
				node -= 1;
		}
	}

	uchar4 c4 = make_uchar4(light_stack[0].r, light_stack[0].g, light_stack[0].b, 255);
	g_odata[y*imgw + x] = rgbToInt(c4.x, c4.y, c4.z);
}

__global__ void
filter(unsigned int *g_odata, int imgw, int imgh) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;
	int id = y * imgw + x;
	if (g_odata[id] == 0) {
		glm::vec3 rgb(0, 0, 0);
		int count = 0;
		for (int dx = -5; dx <= 5; ++dx) {
			for (int dy = -5; dy <= 5; ++dy) {
				int nx = x + dx;
				int ny = y + dy;
				if (nx >= 0 && nx < imgw && ny >= 0 && ny < imgh) {
					int nid = ny * imgw + nx;
					if (g_odata[nid] != 0) {
						count += 1;
						rgb += Inttorgb(g_odata[nid]);
					}
				}
			}
		}
		if (count > 0)
			g_odata[id] = rgbToInt(rgb.r / count, rgb.g / count, rgb.b / count);
		else
			g_odata[id] = rgbToInt(255, 0, 0);
	}
}


__global__ void
ClearCausticMap(glm::ivec3 *g_odata, int imgw, int imgh) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;
	g_odata[y * imgw + x] = glm::ivec3(0, 0, 0);
}

__global__ void
CausticRender(glm::vec3 *causticMap, glm::vec2* cuasticCoords, int imgw, int imgh,
InstanceData* instanceData, glm::vec3* vertexBuffer, glm::vec3* normalBuffer, glm::vec2* texBuffer, int num_object,
glm::vec3 dir, glm::vec3 color, uchar3* texImages, glm::ivec3* imageOffsets) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;
	glm::vec3 point(x * CAUSTIC_MAP_DIS + CAUSTIC_X_MIN, 0, y * CAUSTIC_MAP_DIS + CAUSTIC_X_MIN);
	projectCaustic(point - dir * 1000.0f, dir, color, instanceData, vertexBuffer, normalBuffer, texBuffer, num_object,
		causticMap[y * imgw + x], cuasticCoords[y * imgw + x], texImages, imageOffsets);
}


__global__ void
combineCaustic(unsigned int *g_odata, glm::ivec3* causticMap, int imgw, int imgh,
glm::vec3 cam_up, glm::vec3 cam_forward, glm::vec3 right, glm::vec3 cam_pos, float dis_per_pix,
InstanceData* instanceData, glm::vec3* vertexBuffer, glm::vec3* normalBuffer, glm::vec2* texBuffer, int num_object) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;

	glm::vec3 ray_p = cam_pos;
	glm::vec3 ray_d = glm::normalize(cam_forward + (x - imgw / 2) * dis_per_pix * right + (y - imgh / 2) * dis_per_pix * cam_up);
	glm::vec3 color(0, 0, 0);
	int tri_index, obj_index;

	glm::vec3 hit_point, normal;
	glm::vec2 uv;
	float depth;
	depth = tracing(ray_p, ray_d, -1, tri_index, obj_index, hit_point, uv, normal, instanceData, vertexBuffer, normalBuffer, texBuffer, num_object);
	if (obj_index == 0) {
		int rx = (hit_point.x - CAUSTIC_X_MIN) / CAUSTIC_MAP_DIS;
		int ry = (hit_point.z - CAUSTIC_X_MIN) / CAUSTIC_MAP_DIS;
		if (rx < 512 && ry < 512 && rx >= 0 && ry >= 0) {
			auto& p = causticMap[ry * 512 + rx];
			glm::vec3 np = Inttorgb(g_odata[y * imgw + x]);
			np += p;
			np.x = clamp(np.x, 0.f, 255.f);
			np.y = clamp(np.y, 0.f, 255.f);
			np.z = clamp(np.z, 0.f, 255.f);
			np = glm::vec3(255, 0, 0);
			g_odata[y * imgw + x] = rgbToInt(p.x, p.y, p.z);
		}
	}
}

__global__ void
SplatCaustic(glm::vec3* caustics, glm::vec2* causticCoords, glm::ivec3* causticMaps, int imgw, int imgh) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;
	int ry = y + 1, rx = x + 1;
	if (ry < imgh && rx < imgw) {
		int id[4];
		id[0] = y * imgw + x;
		id[1] = id[0] + 1;
		id[2] = id[0] + imgw;
		id[3] = id[2] + 1;
		float minX = 1e20f, maxX = -1e20f, minY = 1e20f, maxY = -1e20f;
		for (int i = 0; i < 4; ++i) {
			auto& p = causticCoords[id[i]];
			if (causticCoords[id[i]].x < 0)
				return;
			if (p.x < minX)
				minX = p.x;
			if (p.x > maxX)
				maxX = p.x;
			if (p.y < minY)
				minY = p.y;
			if (p.y > maxY)
				maxY = p.y;
		}
		if (maxX - minX > 15 || maxY - minY > 15)
			return;
		int stepX = (maxX - minX) + 1;
		int stepY = (maxY - minY) + 1;
		int steps;
		if (stepX > stepY)
			steps = stepX;
		else
			steps = stepY;
		if (steps == 1)
			steps += 1;
		//		steps *= 2;
		float weight = 255.0 / (steps * steps);
		float stepW = 1.0 / (steps - 1);
		for (int i = 0; i < steps; ++i) {
			for (int j = 0; j < steps; ++j) {
				float wx = stepW * j;
				float wy = stepW * i;
				glm::vec3 interp = (caustics[id[0]] * (1 - wx) + caustics[id[1]] * wx) * (1 - wy)
					+ (caustics[id[2]] * (1 - wx) + caustics[id[3]] * wx) * wy;
				glm::vec2 interp_coords = (causticCoords[id[0]] * (1 - wx) + causticCoords[id[1]] * wx) * (1 - wy)
					+ (causticCoords[id[2]] * (1 - wx) + causticCoords[id[3]] * wx) * wy;
				int nx = interp_coords.x, ny = interp_coords.y;
				if (nx >= 0 && nx < imgw && ny >= 0 && ny < imgh) {
					atomicAdd(&causticMaps[ny * imgw + nx].x, interp.x * weight);
					atomicAdd(&causticMaps[ny * imgw + nx].y, interp.y * weight);
					atomicAdd(&causticMaps[ny * imgw + nx].z, interp.z * weight);
				}
			}
		}
	}
}

__global__ void
FilterCaustic(glm::ivec3* causticMap, glm::vec3* causticBuffer, int imgw, int imgh) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;
	int id = y * imgw + x;
	auto& pix = causticMap[id];
	int temp[3][3] =
	{
		{ 1, 2, 1 },
		{ 2, 4, 2 },
		{ 1, 2, 1 }
	};
	if (pix.x == 0 && pix.y == 0 && pix.z == 0 || true) {
		glm::ivec4 pt;
		for (int py = y - 1; py <= y + 1; ++py) {
			if (py < 0 || py >= imgh)
				continue;
			for (int px = x - 1; px <= x + 1; ++px) {
				if (px < 0 || px >= imgw)
					continue;
				int dy = py - y + 1;
				int dx = px - x + 1;
				auto& p = causticMap[py * imgw + px];
				if (p.x != 0 || p.y != 0 || p.z != 0) {
					pt += glm::ivec4(p, 1) * temp[dy][dx];
				}
			}
		}
		if (pt.w > 0)
			causticBuffer[id] = glm::vec3((float)pt.x / pt.w, (float)pt.y / pt.w, (float)pt.z / pt.w);
		else
			causticBuffer[id] = glm::vec3(0, 0, 0);
	}
	else {
		causticBuffer[id] = glm::vec3(pix.x, pix.y, pix.z);
	}
}

extern "C" void
cudaRender(dim3 grid, dim3 block, int sbytes, unsigned int *g_odata, int imgw, int imgh)
{
	static float count = 1;
	float dis_per_pix = tan(World::fov * 0.5 * 3.141592654 / 180.0) / (imgw / 2);
	glm::vec3 right = glm::cross(World::camera_lookat, World::camera_up);
	ClearCausticMap << < grid, block, sbytes >> >(g_world.causticMapBuffer, imgw, imgh);
	for (int i = 0; i < g_world.lights.direct_light_dir.size(); ++i) {
		CausticRender << < grid, block, sbytes >> > (g_world.causticBuffer, g_world.causticCoordsBuffer, imgw, imgh,
			g_world.materialBuffer, g_world.vertexBuffer, g_world.normalBuffer, g_world.texBuffer, g_world.num_objects,
			g_world.lights.direct_light_dir[i], g_world.lights.direct_light_color[i], g_world.texImagesBuffer, g_world.texOffsetBuffer);
		SplatCaustic << < grid, block, sbytes >> > (g_world.causticBuffer, g_world.causticCoordsBuffer, g_world.causticMapBuffer, imgw, imgh);
		FilterCaustic << < grid, block, sbytes >> > (g_world.causticMapBuffer, g_world.causticBuffer, imgw, imgh);
	}

	render << < grid, block, sbytes >> >(g_odata, imgw, imgh,
		World::camera_up, World::camera_lookat, right, World::camera, dis_per_pix,
		g_world.materialBuffer, g_world.vertexBuffer, g_world.normalBuffer, g_world.texBuffer, g_world.num_objects,
		g_world.lights.direct_light_dir.size(), g_world.directLightsBuffer, g_world.directLightsColorBuffer,
		g_world.lights.point_light_pos.size(), g_world.pointLightsBuffer, g_world.pointLightsColorBuffer, g_world.lights.ambient * count,
		g_world.texImagesBuffer, g_world.texOffsetBuffer,
		g_world.causticBuffer);
	//	filter << < grid, block, sbytes >> >(g_odata, imgw, imgh);
	/*	combineCaustic << < grid, block, sbytes >> >(g_odata, g_world.causticMapBuffer, imgw, imgh,
	World::camera_up, World::camera_lookat, right, World::camera, dis_per_pix,
	g_world.materialBuffer, g_world.vertexBuffer, g_world.normalBuffer, g_world.texBuffer, g_world.num_objects);*/
}