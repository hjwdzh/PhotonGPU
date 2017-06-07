#include "../scene/world.h"
#include <opencv2/opencv.hpp>
#include "../cuda-opengl.h"
#define PATH_DEPTH 5

extern void projectCaustic(glm::vec3 ray_o, glm::vec3 ray_t,
	InstanceData* instanceData, glm::vec3* vertexBuffer, glm::vec3* normalBuffer, glm::vec2* texBuffer, int num_object,
	glm::vec3& light, glm::vec2& coords);

extern float tracing(glm::vec3 ray_o_o, glm::vec3 ray_t_o, float shadow, int& tri, int& obj, glm::vec3& hit_point, glm::vec2& uv, glm::vec3& normal,
	InstanceData* instanceData, glm::vec3* vertexBuffer, glm::vec3* normalBuffer, glm::vec2* texBuffer, int num_object);

glm::vec3 lighting(glm::vec3 start_camera, glm::vec3 point, glm::vec3 normal, int tri_index, glm::vec2 uv, int obj_index,
	InstanceData* instanceData, glm::vec3* vertexBuffer, glm::vec3* normalBuffer, glm::vec2* texBuffer, int num_object,
	int num_direct_light, glm::vec3* direct_lights, glm::vec3* direct_lights_color,
	int num_point_light, glm::vec3* point_lights, glm::vec3* point_lights_color, glm::vec3 ambient,
	uchar3* imagesBuffer, glm::ivec3* imageOffsetBuffer, glm::vec3& orig_color, glm::ivec3* causticMap);
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
			glm::vec3 normal3;
			glm::vec3 hit_point3;
			while (node >= 0) {
				if (path_state[node] == 0) {
					path_state[node] = 1;
					float depth;
					depth = tracing(from_stack[node], to_stack[node], -1, tri_index, obj_index, hit_point, uv, normal, 
						g_world.material.data(), (glm::vec3*)g_world.vertex_buffer.data(), (glm::vec3*)g_world.normal_buffer.data(), 
						(glm::vec2*)g_world.tex_buffer.data(), g_world.num_objects);
					if (depth < 1e20) {
						hit_point3 = glm::vec3(hit_point.x, hit_point.y, hit_point.z);
						normal3 = glm::vec3(normal.x, normal.y, normal.z);
						glm::vec3 orig_color;
						light_stack[node] = lighting(from_stack[node], hit_point3, normal3, tri_index, uv, obj_index, 
							g_world.material.data(), (glm::vec3*)g_world.vertex_buffer.data(), (glm::vec3*)g_world.normal_buffer.data(),
							(glm::vec2*)g_world.tex_buffer.data(), g_world.num_objects,
							g_world.lights.direct_light_dir.size(), g_world.lights.direct_light_dir.data(), g_world.lights.direct_light_color.data(), 
							g_world.lights.point_light_pos.size(), g_world.lights.point_light_pos.data(), g_world.lights.point_light_color.data(), g_world.lights.ambient, 
							g_world.tex_images.data(), g_world.tex_offsets.data(), orig_color, g_world.causticMap.data());
						color_stack[node] = orig_color;
						normal_stack[node] = normal3;
						ray_d = to_stack[node];
						to_stack[node] = hit_point3;
						mat_stack[node] = obj_index;
						float kr = g_world.material[obj_index].kr;
						if (kr > 0 && node < PATH_DEPTH - 1) {
							color_stack[node] = g_world.material[obj_index].kr * color_stack[node];
							node += 1;
							path_state[node] = 0;
							from_stack[node] = hit_point3;
							to_stack[node] = glm::normalize(ray_d - 2 * glm::dot(ray_d, normal3) * normal3);
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
					float kf = g_world.material[obj_index].kf;
					if (kf > 0 && node < PATH_DEPTH - 1) {
						nr = g_world.material[obj_index].nr;
						normal3 = normal_stack[node];
						ray_d = glm::normalize(to_stack[node] - from_stack[node]);
						float cost = glm::dot(normal3, ray_d);
						if (cost < 0) {
							nr = 1 / nr;
							cost = -cost;
						}
						else {
							normal3 = -normal3;
						}
						float rootContent = 1 - nr * nr * (1 - cost * cost);
						if (rootContent >= 0) {
							color_stack[node] = g_world.material[obj_index].kf * color_stack[node];
							rootContent = sqrt(rootContent);
							node += 1;
							path_state[node] = 0;
							from_stack[node] = to_stack[node - 1];
							to_stack[node] = glm::normalize((nr * cost - rootContent) * normal3 + nr * ray_d);
							light_stack[node] = glm::vec3(0, 0, 0);
							continue;
						}
						else {
							float kr = 1;
							if (kr > 0 && node < PATH_DEPTH - 1) {
								node += 1;
								path_state[node] = 0;
								from_stack[node] = to_stack[node - 1];
								to_stack[node] = ray_d - 2 * glm::dot(ray_d, normal3) * normal3;
								light_stack[node] = glm::vec3(0, 0, 0);
								continue;
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
						light_stack[node - 1] += color_stack[node - 1] * light_stack[node] / 255.0f;
					}
					else
						if (path_state[node - 1] == 2) {
							light_stack[node - 1] += color_stack[node - 1] * light_stack[node] / 255.0f;
						}
						else {
							hit_mat -= 1;
							normal3 = normal_stack[node - 1];
							ray_d = glm::normalize(to_stack[node - 1] - from_stack[node - 1]);
							float alpha = g_world.material[obj_index].alpha;
							light_stack[node - 1] += g_world.material[obj_index].ks * color_stack[node - 1] * light_stack[node] * glm::dot(-ray_d, normal3) / 255.0f;
						}
						node -= 1;
				}
			}

			mask.at<cv::Vec3b>(i, j) = cv::Vec3b(light_stack[0].r, light_stack[0].g, light_stack[0].b);
		}
	}
	cv::imshow("mask", mask);
	cv::waitKey();
	cv::imwrite("mask.png", mask);
	exit(0);
}