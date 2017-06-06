#include "../cuda-opengl.h"
#include <opencv2/opencv.hpp>
extern "C" void
cudaRender(dim3 grid, dim3 block, int sbytes, unsigned int *g_odata, int imgw, int imgh);

// copy image and process using CUDA
void RenderImage()
{
	// run the Cuda kernel
	unsigned int *out_data;

#ifdef USE_TEXSUBIMAGE2D
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_dest_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&out_data, &num_bytes,
		cuda_pbo_dest_resource));
	//printf("CUDA mapped pointer of pbo_out: May access %ld bytes, expected %d\n", num_bytes, size_tex_data);
#else
	out_data = cuda_dest_resource;
#endif
	// calculate grid size
	dim3 block(16, 16, 1);
	//dim3 block(16, 16, 1);
	dim3 grid(image_width / block.x, image_height / block.y, 1);
	// execute CUDA kernel
	cudaRender(grid, block, 0, out_data, image_width, image_height);
	
	std::vector<int> imgdata(image_height * image_width);
	cudaMemcpy(imgdata.data(), out_data, sizeof(int) * imgdata.size(), cudaMemcpyDeviceToHost);
	
	// CUDA generated data in cuda memory or in a mapped PBO made of BGRA 8 bits
	// 2 solutions, here :
	// - use glTexSubImage2D(), there is the potential to loose performance in possible hidden conversion
	// - map the texture and blit the result thanks to CUDA API
#ifdef USE_TEXSUBIMAGE2D
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_dest_resource, 0));
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_dest);

	glBindTexture(GL_TEXTURE_2D, tex_cudaResult);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
		image_width, image_height,
		GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	SDK_CHECK_ERROR_GL();
	glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
#else
	// We want to copy cuda_dest_resource data to the texture
	// map buffer objects to get CUDA device pointers
	cudaArray *texture_ptr;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_result_resource, 0));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_result_resource, 0, 0));

	int num_texels = image_width * image_height;
	int num_values = num_texels * 4;
	int size_tex_data = sizeof(GLubyte) * num_values;
	checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dest_resource, size_tex_data, cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_result_resource, 0));
#endif
}