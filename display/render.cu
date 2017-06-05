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

__global__ void
render(unsigned int *g_odata, int imgw, int imgh,
glm::vec3 cam_up, glm::vec3 cam_forward, glm::vec3 right, glm::vec3 cam_pos, float dis_per_pix)
{
	extern __shared__ uchar4 sdata[];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;

	glm::vec3 ray_p = cam_pos;
	glm::vec3 ray_d = glm::normalize(cam_forward + (x - imgw / 2) * dis_per_pix * right + (imgh / 2 - y) * dis_per_pix * cam_up);

//	uchar4 c4 = make_uchar4((float)x / imgw * 255, (float)y / imgh * 255, 0, 255);
	uchar4 c4 = make_uchar4(ray_d.x * 255.0, ray_d.y * 255.0, ray_d.z * 255.0, 255);
	g_odata[y*imgw + x] = rgbToInt(c4.x, c4.y, c4.z);
}

extern "C" void
cudaRender(dim3 grid, dim3 block, int sbytes, unsigned int *g_odata, int imgw, int imgh)
{
	float dis_per_pix = tan(World::fov * 0.5 * 3.141592654 / 180.0) / (imgw / 2);
	glm::vec3 right = glm::cross(World::camera_lookat, World::camera_up);
	render << < grid, block, sbytes >> >(g_odata, imgw, imgh,
		World::camera_up, World::camera_lookat, right, World::camera, dis_per_pix);
}