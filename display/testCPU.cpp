#include "../scene/world.h"
#include <opencv2/opencv.hpp>
#include "../cuda-opengl.h"
extern float tracing(glm::vec3 ray_o_o, glm::vec3 ray_t_o, float shadow, int& tri, int& obj, glm::vec4& hit_point, glm::vec2& uv, glm::vec4& normal,
	InstanceData* instanceData, glm::vec3* vertexBuffer, glm::vec3* normalBuffer, glm::vec2* texBuffer, int num_object);

glm::vec3 lighting(glm::vec3 start_camera, glm::vec3 point, glm::vec3 normal, int tri_index, glm::vec2 uv, int obj_index,
	InstanceData* instanceData, glm::vec3* vertexBuffer, glm::vec3* normalBuffer, glm::vec2* texBuffer, int num_object,
	int num_direct_light, glm::vec3* direct_lights, glm::vec3* direct_lights_color,
	int num_point_light, glm::vec3* point_lights, glm::vec3* point_lights_color, glm::vec3 ambient,
	uchar3* imagesBuffer, glm::ivec3* imageOffsetBuffer);

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
				glm::vec3 normal3(normal.x, normal.y, normal.z);
				glm::vec3 hit_point3(hit_point.x, hit_point.y, hit_point.z);
				glm::vec3 color = lighting(ray_p, hit_point3, normal3, tri_index, uv, obj_index, 
					g_world.material.data(), (glm::vec3*)g_world.vertex_buffer.data(), (glm::vec3*)g_world.normal_buffer.data(),
					(glm::vec2*)g_world.tex_buffer.data(), g_world.num_objects,
					g_world.lights.direct_light_dir.size(), g_world.lights.direct_light_dir.data(), g_world.lights.direct_light_color.data(),
					g_world.lights.point_light_pos.size(), g_world.lights.point_light_pos.data(), g_world.lights.point_light_color.data(), g_world.lights.ambient,
					g_world.tex_images.data(), g_world.tex_offsets.data());
				color = color;
				mask.at<cv::Vec3b>(i, j) = cv::Vec3b(color.r, color.g, color.b);
			}
		}
	}
	cv::imshow("mask", mask);
	cv::waitKey();
	cv::imwrite("mask.png", mask);
	exit(0);
}