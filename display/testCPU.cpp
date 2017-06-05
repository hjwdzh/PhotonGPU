#include "../scene/world.h"
#include <opencv2/opencv.hpp>
extern float tracing(glm::vec3 ray_o_o, glm::vec3 ray_t_o, float shadow, int& tri, int& obj, glm::vec4& hit_point, glm::vec2& uv, glm::vec4& normal,
	InstanceData* instanceData, glm::vec3* vertexBuffer, glm::vec3* normalBuffer, glm::vec2* texBuffer, int num_object);

void testCPU()
{
	int imgw = 512, imgh = 512;
	cv::Mat mask = cv::Mat::zeros(imgh, imgw, CV_8UC3);
	for (int i = 0; i < imgh; ++i) {
		for (int j = 0; j < imgw; ++j) {
			float dis_per_pix = tan(World::fov * 0.5 * 3.141592654 / 180.0) / (imgw / 2);
			glm::vec3 right = glm::cross(World::camera_lookat, World::camera_up);
			glm::vec3 ray_d = glm::normalize(World::camera_lookat + (j - imgw / 2) * dis_per_pix * right + (imgh / 2 - i) * dis_per_pix * World::camera_up);
			glm::vec3 ray_p = World::camera;
			int tri_index, obj_index;
			glm::vec4 hit_point;
			glm::vec2 uv;
			glm::vec4 normal;
			glm::vec3 orig_color;
			float depth = tracing(ray_p, ray_d, -1, tri_index, obj_index, hit_point, uv, normal,
				g_world.material.data(), (glm::vec3*)g_world.vertex_buffer.data(), (glm::vec3*)g_world.normal_buffer.data(), 
				(glm::vec2*)g_world.tex_buffer.data(), g_world.num_objects);
			if (depth < 1e20) {
				mask.at<cv::Vec3b>(i, j) = cv::Vec3b(255, uv.y * 255, uv.x*255);
			}
		}
	}
	cv::imshow("mask", mask);
	cv::waitKey();
	cv::imwrite("mask.png", mask);
	exit(0);
}