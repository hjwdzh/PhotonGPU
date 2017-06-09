#include <helper_cuda.h>
#include "../scene/world.h"
#include "../bvh/bvh.h"

/* BASIC OPERATIONS */
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

__device__ __host__ glm::vec3 Inttorgb(int x)
{
	glm::vec3 rgb;
	rgb.b = (x >> 16);
	rgb.g = (x >> 8) & 0xff;
	rgb.r = (x & 0xff);
	return rgb;
}

__device__ __host__
glm::vec3 fetchTex(glm::vec2& uv, int objIndex, uchar3* imagesBuffer, glm::ivec3* imageOffsetBuffer)
{
	glm::ivec3& info = imageOffsetBuffer[objIndex];
	int offset = info.x;
	int iy = info.y;
	int ix = info.z;
	float x = uv.x;
	float y = uv.y;
	if (x < 0) 
		x = x - (int)x + 1;
	if (y < 0) 
		y = y - (int)y + 1;
	if (x >= 1) 
		x = x - (int)x;
	if (y >= 1) 
		y = y - (int)y;
	x *= ix;
	y *= iy;
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
glm::vec3 fetchEnvironment(glm::vec3 ray_d, int imgh, int imgw, uchar3* imagesBuffer)
{
	float theta = acos(ray_d.y) / CV_PI * imgh;
	float phi = atan2(ray_d.z, ray_d.x) / CV_PI / 2.0;
	if (phi < 0)
		phi += 1;
	phi += 0.25;
	if (phi > 1)
		phi -= 1;
	phi *= imgw;
	int lx = phi, ly = theta;
	int rx = lx + 1, ry = ly + 1;
	float wx = phi - lx, wy = theta - ly;
	if (rx >= imgw)
		rx -= imgw;
	if (ry >= imgh)
		ry -= imgh;
	int ind1 = ly * imgw + lx;
	int ind2 = ly * imgw + rx;
	int ind3 = ry * imgw + lx;
	int ind4 = ry * imgw + rx;
	uchar3& c1 = imagesBuffer[ind1];
	uchar3& c2 = imagesBuffer[ind2];
	uchar3& c3 = imagesBuffer[ind3];
	uchar3& c4 = imagesBuffer[ind4];
	float cx = (c1.x * (1 - wx) + c2.x * wx) * (1 - wy) + (c3.x * (1 - wx) + c4.x * wx) * wy;
	float cy = (c1.y * (1 - wx) + c2.y * wx) * (1 - wy) + (c3.y * (1 - wx) + c4.y * wx) * wy;
	float cz = (c1.z * (1 - wx) + c2.z * wx) * (1 - wy) + (c3.z * (1 - wx) + c4.z * wx) * wy;
	return glm::vec3(cz, cy, cx);
}

/* Intersections */
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

/* BVH Intersection */
__device__ __host__
int bvh_index(BVHData* bvh_node) {
	return 3 * bvh_node->start_id;
}
__device__ __host__
int bvh_left(BVHData* bvh_node) {
	return bvh_node->left_id;
}
__device__ __host__
int bvh_right(BVHData* bvh_node) {
	return bvh_node->right_id;
}
__device__ __host__
int bvh_parent(BVHData* bvh_node) {
	return bvh_node->parent_id;
}

__device__ __host__
int bvh_dir(BVHData* bvh_node, glm::vec3& ray) {
	int axis = bvh_node->axis;
	if (axis == 0)
		return ray.x > 0;
	if (axis == 1)
		return ray.y > 0;
	return ray.z <= 0;
}

__device__ __host__
float box_intersect(BVHData* bvh_node, glm::vec3& ray_o, glm::vec3& ray_t) {
	float step = 1.0 / 11;
	float tmp = 0;
	float a1, a2, b1, b2, c1, c2;
	a1 = (bvh_node->minCorner.x - ray_o.x);
	a2 = (bvh_node->maxCorner.x - ray_o.x);
	if (ray_t.x < 1e-6 && ray_t.x > -1e-6) {
		if (a1 * a2 > 1e-4)
			return -1;
		a1 = -1e30; a2 = 1e30;
	}
	else {
		a1 /= ray_t.x;
		a2 /= ray_t.x;
	}
	if (a1 > a2) {
		tmp = a1; a1 = a2; a2 = tmp;
	}
	b1 = (bvh_node->minCorner.y - ray_o.y);
	b2 = (bvh_node->maxCorner.y - ray_o.y);
	if (ray_t.y < 1e-6 && ray_t.y > -1e-6) {
		if (b1 * b2 > 1e-4)
			return -1;
		b1 = -1e30; b2 = 1e30;
	}
	else {
		b1 /= ray_t.y;
		b2 /= ray_t.y;
	}
	if (b1 > b2) {
		tmp = b1; b1 = b2; b2 = tmp;
	}
	c1 = (bvh_node->minCorner.z - ray_o.z);
	c2 = (bvh_node->maxCorner.z - ray_o.z);
	if (ray_t.z < 1e-6 && ray_t.z > -1e-6) {
		if (c1 * c2 > 1e-4)
			return -1;
		c1 = -1e30; c2 = 1e30;
	}
	else {
		c1 /= ray_t.z;
		c2 /= ray_t.z;
	}
	if (c1 > c2) {
		tmp = c1; c1 = c2; c2 = tmp;
	}
	float t1, t2;
	t1 = max(a1, max(b1, c1));
	t2 = min(a2, min(b2, c2));

	if (t2 >= t1 && t2 >= 0)
		return (t1 > 0) ? t1 : 0;
	else
		return -1;
}

__device__ __host__
float bvh_intersect(glm::vec3& ray_o, glm::vec3& ray_t, int& index, float& u, float& v, 
	glm::vec3* vertexBuffer, BVHData* bvh) {
	float depth = 1e30;
	index = -1;
	BVHData* bvh_node = bvh;
	BVHData* last_node = 0;
	float u1, v1;
	int t = 0;
	while (bvh_node >= bvh) {
		t += 1;
		if (last_node == 0) {
			float cur_depth = box_intersect(bvh_node, ray_o, ray_t);
			if (cur_depth < 0 || cur_depth > depth) {
				last_node = bvh_node;
				bvh_node = bvh + bvh_parent(bvh_node);
				continue;
			}
			if (bvh_left(bvh_node) < 0) {

				int cur_index = bvh_index(bvh_node);
				cur_depth = rayIntersectsTriangle(ray_o, ray_t, vertexBuffer[cur_index], 
					vertexBuffer[cur_index + 1], vertexBuffer[cur_index + 2], u1, v1);
				if (cur_depth >= 0 && cur_depth < depth) {
					index = cur_index;
					u = u1;
					v = v1;
					depth = cur_depth;
				}
				last_node = bvh_node;
				bvh_node = bvh + bvh_parent(bvh_node);
				continue;
			}
			else {
				last_node = 0;
				if (bvh_dir(bvh_node, ray_t)) {
					bvh_node = bvh + bvh_left(bvh_node);
				}
				else {
					bvh_node = bvh + bvh_right(bvh_node);
				}
			}
		}
		else {
			bool dir = bvh_dir(bvh_node, ray_t);
			BVHData* left_node = bvh + bvh_left(bvh_node);
			BVHData* right_node = bvh + bvh_right(bvh_node);
			if (dir && left_node == last_node) {
				last_node = 0;
				bvh_node = bvh + bvh_right(bvh_node);
			}
			else
				if (!dir && right_node == last_node) {
					last_node = 0;
					bvh_node = bvh + bvh_left(bvh_node);
				}
				else {
					last_node = bvh_node;
					bvh_node = bvh + bvh_parent(bvh_node);
				}
		}
	}
	return depth;
}

/* Tracing Algorithm */
//#define BVH_
__device__ __host__
float tracing(glm::vec3& ray_o, glm::vec3& ray_t, float shadow, int& tri, int& obj, glm::vec3& hit_point, glm::vec2& uv, glm::vec3& normal,
InstanceData* instanceData, glm::vec3* vertexBuffer, glm::vec3* normalBuffer, glm::vec2* texBuffer, int num_object,
BVHData* bvh) {
#ifndef BVH_
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
#else
	float depth = 1e30;
	obj = -1;
	tri = -1;
	for (int k = 0; k < num_object; ++k) {
		int index;
		float u, v;
		float t = bvh_intersect(ray_o, ray_t, index, u, v, vertexBuffer, bvh + instanceData[k].bvh_offset);
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
				tri = index;
				glm::vec3& n1 = normalBuffer[tri];
				glm::vec3& n2 = normalBuffer[tri + 1];
				glm::vec3& n3 = normalBuffer[tri + 2];
				normal = u * (n2 - n1) + v * (n3 - n1) + n1;
				glm::vec2& uv1 = texBuffer[tri];
				glm::vec2& uv2 = texBuffer[tri + 1];
				glm::vec2& uv3 = texBuffer[tri + 2];
				uv = uv1 + u * (uv2 - uv1) + v * (uv3 - uv1);
			}
		}
	}
	normal = normalize(normal);
	return depth;
#endif
}


__device__ __host__
glm::vec3 lighting(glm::vec3& start_camera, glm::vec3& point, glm::vec3& normal, int tri_index, glm::vec2& uv, int obj_index,
InstanceData* instanceData, glm::vec3* vertexBuffer, glm::vec3* normalBuffer, glm::vec2* texBuffer, int num_object,
int num_direct_light, glm::vec3* direct_lights, glm::vec3* direct_lights_color,
int num_point_light, glm::vec3* point_lights, glm::vec3* point_lights_color, glm::vec3& ambient,
uchar3* imagesBuffer, glm::ivec3* imageOffsetBuffer, glm::vec3& orig_color, glm::vec3* causticMap, BVHData* bvh, float depth, uchar3* environment,
glm::vec3& ray_t, glm::vec3* scatterMap, glm::vec3* scatterPosMap, float* shadowMap, int render_mode) {
	float kd = instanceData[obj_index].kd;
	float ks = instanceData[obj_index].ks;//texture2D(materialSampler, vec2(1.5 / MATERIAL_LEN, (obj_index + 0.5) / num_object)).r;
	float ka = instanceData[obj_index].ka;// texture2D(materialSampler, vec2(16.5 / MATERIAL_LEN, (obj_index + 0.5) / num_object)).r;
	float alpha = instanceData[obj_index].alpha;// texture2D(materialSampler, vec2(20.5 / MATERIAL_LEN, (obj_index + 0.5) / num_object)).r;

	if (depth > 1000)
		orig_color = fetchEnvironment(ray_t, 3000, 6000, environment);
	else
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
		float depth = tracing(point, -direct_lights[i], 100, t1, t2, v2, v1, v3, instanceData, vertexBuffer, normalBuffer, texBuffer, num_object, bvh);
		if (obj_index == 0) {
			float rx = (point.x - CAUSTIC_X_MIN) / CAUSTIC_MAP_DIS;
			float ry = (point.z - CAUSTIC_X_MIN) / CAUSTIC_MAP_DIS;
			if (rx >= 0 && ry >= 0 && rx < CAUSTIC_W && ry < CAUSTIC_W) {
				int lx = rx, ly = ry;
				int rrx = lx + 1, rry = ly + 1;
				float wx = rx - lx, wy = ry - ly;
				float shadow1 = shadowMap[ly * CAUSTIC_W + lx];
				float shadow2 = shadowMap[ly * CAUSTIC_W + rrx];
				float shadow3 = shadowMap[rry * CAUSTIC_W + lx];
				float shadow4 = shadowMap[rry * CAUSTIC_W + rrx];
				float shadow = (shadow1 * (1 - wx) + shadow2 * wx) * (1 - wy) + (shadow3 * (1 - wx) + shadow4 * wx) * wy;
				intensity *= shadow;
			} else
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
		float depth = tracing(point, -dis, len, t1, t2, v2, v1, v3, instanceData, vertexBuffer, normalBuffer, texBuffer, num_object, bvh);
		if (depth < len)
			continue;
		glm::vec3 para = kd * l * point_lights_color[i];
		color = color + intensity * (orig_color * para
			+ clamp((float)pow(dot(reflect(dis, normal), eye_dir), alpha), 0.f, 1.f) * ks * point_lights_color[i]);
	}
	if (obj_index == 0) {
		float rx = (point.x - CAUSTIC_X_MIN) / CAUSTIC_MAP_DIS;
		float ry = (point.z - CAUSTIC_X_MIN) / CAUSTIC_MAP_DIS;
		if (rx < CAUSTIC_W - 1 && ry < CAUSTIC_W-1 && rx >= 0 && ry >= 0) {
			int lx = rx, ly = ry;
			int rrx = lx + 1, rry = ly + 1;
			float wx = rx - lx, wy = ry - ly;
			glm::vec3& caustic1 = causticMap[ly * CAUSTIC_W + lx];
			glm::vec3& caustic2 = causticMap[ly * CAUSTIC_W + rrx];
			glm::vec3& caustic3 = causticMap[rry * CAUSTIC_W + lx];
			glm::vec3& caustic4 = causticMap[rry * CAUSTIC_W + rrx];
			glm::vec3 caustic = (caustic1 * (1 - wx) + caustic2 * wx) * (1 - wy) + (caustic3 * (1 - wx) + caustic4 * wx) * wy;
			if (render_mode == 2) {
				color = caustic;
				return color;
			}
			color = color + glm::dot(eye_dir, normal) * kd * caustic;
			float max_v = max(max(color.x, color.y), color.z) / 255.0f;
			if (max_v > 1)
				color /= max_v;
		}
	}
	if (instanceData[obj_index].kt > 1e-3) {
		float radius = 0.4f;
		glm::vec3 lightp = point + -point.y / direct_lights[0].y * direct_lights[0];
		float x = (lightp.x - CAUSTIC_X_MIN) / CAUSTIC_MAP_DIS;
		float y = (lightp.z - CAUSTIC_X_MIN) / CAUSTIC_MAP_DIS;
		int tx = x, ty = y;
		if (tx < 0 || tx >= CAUSTIC_W || ty < 0 || ty >= CAUSTIC_W)
			return glm::vec3(0,0,0);
		int bandwidth = radius / abs(direct_lights[0].y) / CAUSTIC_MAP_DIS / 2;
		glm::vec3 lights(0, 0, 0);
		for (int ly = y - bandwidth; ly <= y + bandwidth; ++ly) {
			for (int lx = x - bandwidth; lx <= x + bandwidth; ++lx) {
				if (ly < 0 || ly >= CAUSTIC_W || lx < 0 || lx >= CAUSTIC_W)
					continue;
				float r = glm::length(point - scatterPosMap[ly * CAUSTIC_W + lx]);
				float weight = exp(-(r*r) / (radius*radius * 2)) * 2.5e-3 / (radius * radius);
				lights += weight * scatterMap[ly * CAUSTIC_W + lx];
			}
		}
		if (render_mode == 5) {
			return (lights * 255.0f);
		}
		lights.x = clamp(lights.x, 0.f, 1.f);
		lights.y = clamp(lights.y, 0.f, 1.f);
		lights.z = clamp(lights.z, 0.f, 1.f);
		color += lights * orig_color;
		color.x = clamp(color.x, 0.0f, 255.f);
		color.y = clamp(color.y, 0.0f, 255.f);
		color.z = clamp(color.z, 0.0f, 255.f);
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
glm::vec3* causticMap, BVHData* bvh, uchar3* environment, glm::vec3* scatterMap, glm::vec3* scatterPos, float* shadowMap, int render_mode)
{
	extern __shared__ uchar4 sdata[];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;

	if (render_mode == 1) {
		g_odata[y * imgw + x] = rgbToInt(causticMap[y * imgw + x].x, causticMap[y * imgw + x].y, causticMap[y * imgw + x].z);
		return;
	}
	if (render_mode == 4) {
		g_odata[y * imgw + x] = rgbToInt(scatterMap[y * imgw + x].x * 255, scatterMap[y * imgw + x].y * 255, scatterMap[y * imgw + x].z * 255);
		return;
	}
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
			depth = tracing(from_stack[node], to_stack[node], -1, tri_index, obj_index, hit_point, uv, normal, instanceData, vertexBuffer, normalBuffer, texBuffer, num_object, bvh);
			if (depth < 1e20) {
				glm::vec3 orig_color;
				light_stack[node] = lighting(from_stack[node], hit_point, normal, tri_index, uv, obj_index, instanceData, vertexBuffer, normalBuffer, texBuffer, num_object,
					num_direct_lights, direct_lights, direct_lights_color, num_point_lights, point_lights, point_lights_color, ambient,
					imagesBuffer, imageOffsetBuffer, orig_color, causticMap, bvh, depth, environment, to_stack[node], scatterMap, scatterPos, shadowMap, render_mode);
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

/* Filtering */
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
			g_odata[id] = rgbToInt(0, 0, 0);
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

/* Caustic Rendering */
__device__ __host__
glm::vec3 projectCaustic(glm::vec3& ray_o, glm::vec3& ray_t, glm::vec3 &color,
InstanceData* instanceData, glm::vec3* vertexBuffer, glm::vec3* normalBuffer, glm::vec2* texBuffer, int num_object,
glm::vec3& light, glm::vec2& coords, uchar3* texImages, glm::ivec3* imageOffsets, BVHData* bvh, glm::vec3& scatterPos, float& softShadow) {
	int tri_index, obj_index;
	glm::vec3 hit_point, normal;
	glm::vec2 uv;
	float depth = tracing(ray_o, ray_t, -1, tri_index, obj_index, hit_point, uv, normal, instanceData, vertexBuffer, normalBuffer, texBuffer, num_object, bvh);
	if (obj_index == 0)
		softShadow = 1;
	else
		softShadow = 0;
	glm::vec3 orig_color = fetchTex(uv, obj_index, texImages, imageOffsets) / 255.0f;
	int steps = 0;
	float intensity = 1;
	while (depth < 1e20 && (instanceData[obj_index].kr > 1e-3 || instanceData[obj_index].kf > 1e-3 || instanceData[obj_index].kt > 1e-3)) {
		if (instanceData[obj_index].kt > 1e-3) {
			float x = (hit_point.x - CAUSTIC_X_MIN) / CAUSTIC_MAP_DIS;
			float y = (hit_point.z - CAUSTIC_X_MIN) / CAUSTIC_MAP_DIS;
			scatterPos = hit_point;
			return color * -glm::dot(normal, ray_t);
		}
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
				intensity *= instanceData[obj_index].kf * 0.6;
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
		depth = tracing(ray_o, ray_t, -1, tri_index, obj_index, hit_point, uv, normal, instanceData, vertexBuffer, normalBuffer, texBuffer, num_object, bvh);
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
	return glm::vec3(0, 0, 0);
}

__global__ void
ClearBuffer(glm::ivec3 *g_odata, glm::vec3* g_light, float* shadow, int imgw, int imgh) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;
	g_odata[y * imgw + x] = glm::ivec3(0, 0, 0);
	g_light[y * imgw + x] = glm::vec3(0, 0, 0);
	shadow[y * imgw + x] = 0;
}

__global__ void
FilterShadow(float* input, float* output, int imgw, int imgh) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;
	int total = 0;
	float shad = 0;
	for (int dy = y - 3; dy <= y + 3; ++dy) {
		for (int dx = x - 3; dx <= x + 3; ++dx) {
			if (dy >= 0 && dy < imgh && dx >= 0 && dx < imgw)
			{
				shad += input[dy * imgw + dx];
				total += 1;
			}
		}
	}
	if (total != 0)
		output[y * imgw + x] = shad / total;
	else
		output[y * imgw + x] = 0;
}

__global__ void
CausticRender(glm::vec3 *causticMap, glm::vec2* cuasticCoords, int imgw, int imgh,
InstanceData* instanceData, glm::vec3* vertexBuffer, glm::vec3* normalBuffer, glm::vec2* texBuffer, int num_object,
glm::vec3 dir, glm::vec3 color, uchar3* texImages, glm::ivec3* imageOffsets, BVHData* bvh, glm::vec3* scatterBuffer, glm::vec3* scatterPos, float* softShadow) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;
	glm::vec3 point(x * CAUSTIC_MAP_DIS + CAUSTIC_X_MIN, 0, y * CAUSTIC_MAP_DIS + CAUSTIC_X_MIN);
	scatterBuffer[y * CAUSTIC_W + x] = projectCaustic(point - dir * 1000.0f, dir, color, instanceData, vertexBuffer, normalBuffer, texBuffer, num_object,
		causticMap[y * CAUSTIC_W + x], cuasticCoords[y * imgw + x], texImages, imageOffsets, bvh, scatterPos[y * CAUSTIC_W + x], softShadow[y * CAUSTIC_W + x]);
}


__global__ void
combineCaustic(unsigned int *g_odata, glm::ivec3* causticMap, int imgw, int imgh,
glm::vec3 cam_up, glm::vec3 cam_forward, glm::vec3 right, glm::vec3 cam_pos, float dis_per_pix,
InstanceData* instanceData, glm::vec3* vertexBuffer, glm::vec3* normalBuffer, glm::vec2* texBuffer, int num_object, BVHData* bvh) {
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
	depth = tracing(ray_p, ray_d, -1, tri_index, obj_index, hit_point, uv, normal, instanceData, vertexBuffer, normalBuffer, texBuffer, num_object, bvh);
	if (obj_index == 0) {
		int rx = (hit_point.x - CAUSTIC_X_MIN) / CAUSTIC_MAP_DIS;
		int ry = (hit_point.z - CAUSTIC_X_MIN) / CAUSTIC_MAP_DIS;
		if (rx < CAUSTIC_W && ry < CAUSTIC_W && rx >= 0 && ry >= 0) {
			auto& p = causticMap[ry * CAUSTIC_W + rx];
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

/* GPU Render Entry */
__global__ void
SplatCaustic(glm::vec3* buffer, glm::vec3* map, int imgw, int imgh) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;
}

extern "C" void
cudaRender(dim3 grid, dim3 block, int sbytes, unsigned int *g_odata, int imgw, int imgh)
{
	static float count = 1;
	float dis_per_pix = tan(World::fov * 0.5 * 3.141592654 / 180.0) / (imgw / 2);
	glm::vec3 right = glm::cross(World::camera_lookat, World::camera_up);
	dim3 grid1(CAUSTIC_W / block.x, CAUSTIC_W / block.y, 1);
	static float angle = 0.0;
	static float angle_dir = 1.0f;
	if (g_world.pause) {
		angle += angle_dir;
	}
	if (angle > 30.0f || angle < -30.0f)
		angle_dir = -angle_dir;
	float rad = angle / 180.0 *	CV_PI;
	glm::mat3 rot(1.0f);
	rot[0][0] = cos(rad);
	rot[0][2] = sin(rad);
	rot[2][0] = -sin(rad);
	rot[2][2] = cos(rad);
	glm::vec3 new_dir = rot * g_world.lights.direct_light_dir[0];
	cudaMemcpy(g_world.directLightsBuffer, &new_dir, sizeof(glm::vec3), cudaMemcpyHostToDevice);
	ClearBuffer << < grid1, block, sbytes >> >(g_world.causticMapBuffer, g_world.scatterBuffer, g_world.softShadowBuffer, CAUSTIC_W, CAUSTIC_W);
	for (int i = 0; i < g_world.lights.direct_light_dir.size(); ++i) {
		CausticRender << < grid1, block, sbytes >> > (g_world.causticBuffer, g_world.causticCoordsBuffer, CAUSTIC_W, CAUSTIC_W,
			g_world.materialBuffer, g_world.vertexBuffer, g_world.normalBuffer, g_world.texBuffer, g_world.num_objects,
			new_dir, g_world.lights.direct_light_color[i], g_world.texImagesBuffer, g_world.texOffsetBuffer, g_world.bvhDataBuffer,
			g_world.scatterBuffer, g_world.scatterPosBuffer, g_world.softShadowBuffer);
		SplatCaustic << < grid1, block, sbytes >> > (g_world.causticBuffer, g_world.causticCoordsBuffer, g_world.causticMapBuffer, CAUSTIC_W, CAUSTIC_W);
		FilterCaustic << < grid1, block, sbytes >> > (g_world.causticMapBuffer, g_world.causticBuffer, CAUSTIC_W, CAUSTIC_W);
		FilterShadow << < grid1, block, sbytes >> > (g_world.softShadowBuffer, g_world.softShadowMap, CAUSTIC_W, CAUSTIC_W);
	}
	
	render << < grid, block, sbytes >> >(g_odata, imgw, imgh,
		World::camera_up, World::camera_lookat, right, World::camera, dis_per_pix,
		g_world.materialBuffer, g_world.vertexBuffer, g_world.normalBuffer, g_world.texBuffer, g_world.num_objects,
		g_world.lights.direct_light_dir.size(), g_world.directLightsBuffer, g_world.directLightsColorBuffer,
		g_world.lights.point_light_pos.size(), g_world.pointLightsBuffer, g_world.pointLightsColorBuffer, g_world.lights.ambient * count,
		g_world.texImagesBuffer, g_world.texOffsetBuffer,
		g_world.causticBuffer, g_world.bvhDataBuffer, g_world.environmentBuffer, g_world.scatterBuffer, g_world.scatterPosBuffer,
		g_world.softShadowMap, g_world.rendering_mode);
	if (g_world.rendering_mode == 0 || g_world.rendering_mode == 3)
		filter << < grid, block, sbytes >> >(g_odata, imgw, imgh);
	printf("%d\n", g_world.rendering_mode);
	/*	combineCaustic << < grid, block, sbytes >> >(g_odata, g_world.causticMapBuffer, imgw, imgh,
	World::camera_up, World::camera_lookat, right, World::camera, dis_per_pix,
	g_world.materialBuffer, g_world.vertexBuffer, g_world.normalBuffer, g_world.texBuffer, g_world.num_objects);*/
}
