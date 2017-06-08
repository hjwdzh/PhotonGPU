#include <opencv2/opencv.hpp>
#include "../scene/world.h"
extern float tracing(glm::vec3& ray_o, glm::vec3& ray_t, float shadow, int& tri, int& obj, glm::vec3& hit_point, glm::vec2& uv, glm::vec3& normal,
	InstanceData* instanceData, glm::vec3* vertexBuffer, glm::vec3* normalBuffer, glm::vec2* texBuffer, int num_object,
	BVHData* bvh);

void testCPU()
{
	int imgw = 512;
	int imgh = 512;
	cv::Mat img = cv::Mat::zeros(imgh, imgw, CV_8UC3);
	for (int i = 0; i < imgh; ++i) {
		printf("%d\n", i);
		for (int j = 0; j < imgw; ++j) {
			float dis_per_pix = tan(World::fov * 0.5 * 3.141592654 / 180.0) / (imgw / 2);
			glm::vec3 right = glm::cross(World::camera_lookat, World::camera_up);
			glm::vec3 ray_p = g_world.camera;
			glm::vec3 ray_d = glm::normalize(g_world.camera_lookat + (j - imgw / 2) * dis_per_pix * right + (i - imgh / 2) * dis_per_pix * g_world.camera_up);
			int tri, obj;
			glm::vec3 hit_point, normal;
			glm::vec2 uv;
			float depth = tracing(ray_p, ray_d, -1, tri, obj, hit_point, uv, normal, g_world.material.data(), (glm::vec3*)g_world.vertex_buffer.data(), 
				(glm::vec3*)g_world.normal_buffer.data(), (glm::vec2*)g_world.tex_buffer.data(), g_world.num_objects, g_world.bvhData.data());
			if (depth < 1e10) {
				img.at<cv::Vec3b>(i, j) = cv::Vec3b(uv.x * 255.0f, uv.y * 255.0f, 255);
			}
		}
	}
	cv::imshow("img", img);
	cv::waitKey();
	cv::imwrite("img.png", img);
	exit(0);
}