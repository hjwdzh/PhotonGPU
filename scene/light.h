#ifndef LIGHT_H_
#define LIGHT_H_
#include <glm/glm.hpp>
#include <vector>
class Light {
public:
	std::vector<glm::vec3> point_light_pos;
	std::vector<glm::vec3> point_light_color;
	std::vector<glm::vec3> direct_light_dir;
	std::vector<glm::vec3> direct_light_color;
	glm::vec3 ambient;
};

#endif