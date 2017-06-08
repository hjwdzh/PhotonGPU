#include "world.h"
#include "../cuda-opengl.h"
#include "../bvh/bvh.h"
glm::vec3 World::camera_up;
glm::vec3 World::camera_lookat;
glm::vec3 World::camera;
float World::fov;
Light World::lights;
std::vector<Geometry*> World::objects;

float FetchDepth(glm::vec3& v1, glm::vec3& v2, glm::vec3& v3, glm::vec2 coord) {
	cv::Mat m(3, 3, CV_32F);
	m.at<float>(0, 0) = m.at<float>(0, 1) = m.at<float>(0, 2) = 1;
	m.at<float>(1, 0) = v1.x / 600.0f, m.at<float>(1, 1) = v2.x / 600.0f, m.at<float>(1, 2) = v3.x / 600.0f;
	m.at<float>(2, 0) = v1.y / 600.0f, m.at<float>(2, 1) = v2.y / 600.0f, m.at<float>(2, 2) = v3.y / 600.0f;
	cv::Mat x(3, 1, CV_32F);
	x.at<float>(0, 0) = 1, x.at<float>(1, 0) = coord.x / 600.0f, x.at<float>(2, 0) = coord.y / 600.0f;
	cv::Mat w;
	cv::solve(m, x, w);
	glm::vec3* p = (glm::vec3*)w.data;
	if (p->x < 0 || p->x > 1 || p->y < 0 || p->y > 1 || p->z < 0 || p->z > 1)
		return 1e30;
	return (p->x * v1.z + p->y * v2.z + p->z * v3.z);
}
void fillBottomFlatTriangle(glm::vec3& v1, glm::vec3& v2, glm::vec3& v3, cv::Mat img)
{
	float invslope1 = (v2.x - v1.x) / (v2.y - v1.y);
	float invslope2 = (v3.x - v1.x) / (v3.y - v1.y);
	if (invslope1 > invslope2)
		std::swap(invslope1, invslope2);
	if (!(abs(invslope1) < 1e6))
		return;
	float curx1 = v1.x;
	float curx2 = v1.x;

	for (float scanlineY = v1.y; scanlineY <= v2.y; scanlineY++)
	{
		for (int px = (int)curx1; px <= (int)curx2; ++px) {
			float pt = FetchDepth(v1, v2, v3, glm::vec2(px, scanlineY));
			float& p = img.at<float>((int)scanlineY, px);
			if (p == 0 && pt < 1e20 || pt < p)
				p = pt;
		}
		if (scanlineY + 1 > v2.y) {
			curx1 += invslope1 * (v2.y - scanlineY);
			curx2 += invslope2 * (v2.y - scanlineY);
			for (int px = (int)curx1; px <= (int)curx2; ++px) {
				float pt = FetchDepth(v1, v2, v3, glm::vec2(px, scanlineY));
				float& p = img.at<float>((int)scanlineY, px);
				if (p == 0 && pt < 1e20 || pt < p)
					p = pt;
			}
			break;
		}
		curx1 += invslope1;
		curx2 += invslope2;
	}
}

void fillTopFlatTriangle(glm::vec3& v1, glm::vec3& v2, glm::vec3& v3, cv::Mat img)
{
	float invslope1 = (v3.x - v1.x) / (v3.y - v1.y);
	float invslope2 = (v3.x - v2.x) / (v3.y - v2.y);

	if (invslope1 < invslope2)
		std::swap(invslope1, invslope2);
	if (!(abs(invslope1) < 1e6))
		return;
	float curx1 = v3.x;
	float curx2 = v3.x;

	for (float scanlineY = v3.y; scanlineY > v1.y; scanlineY--)
	{
		for (int px = (int)curx1; px <= (int)curx2; ++px) {
			float pt = FetchDepth(v1, v2, v3, glm::vec2(px, scanlineY));
			float& p = img.at<float>((int)scanlineY, px);
			if (p == 0 && pt < 1e20 || pt < p)
				p = pt;
		}
		if (scanlineY - 1 <= v1.y) {
			curx1 -= invslope1 * (scanlineY - v1.y);
			curx2 -= invslope2 * (scanlineY - v1.y);
			for (int px = (int)curx1; px <= (int)curx2; ++px) {
				float pt = FetchDepth(v1, v2, v3, glm::vec2(px, scanlineY));
				float& p = img.at<float>((int)scanlineY, px);
				if (p == 0 && pt < 1e20 || pt < p)
					p = pt;
			}
			break;
		}
		curx1 -= invslope1;
		curx2 -= invslope2;
	}
}

void fillTriangle(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, cv::Mat img) {
	if (v1.y > v2.y)
		std::swap(v1, v2);
	if (v1.y > v3.y)
		std::swap(v1, v3);
	if (v2.y > v3.y)
		std::swap(v2, v3);
	if (v2.y == v3.y)
		fillBottomFlatTriangle(v1, v2, v3, img);
	else if (v1.y == v2.y)
		fillTopFlatTriangle(v1, v2, v3, img);
	else {
		float wx = (v2.y - v1.y) / (float)(v3.y - v1.y);
		glm::vec3 v4((v1.x + wx * (v3.x - v1.x)), v2.y, v1.z + wx * (v3.z - v1.z));
		fillBottomFlatTriangle(v1, v2, v4, img);
		fillTopFlatTriangle(v2, v4, v3, img);
	}
}

extern float bvh_intersect(glm::vec3& ray_o, glm::vec3& ray_t, int& index, float& u, float& v,
	glm::vec3* vertexBuffer, BVHData* bvh);

void World::ProcessScattering()
{
	printf("%d\n", material.size());
	for (int i = 0; i < material.size(); ++i) {
		if (material[i].kt > 1e-5) {
			cv::Mat lightPos(512, 512, CV_8U);
			cv::Mat directLight = cv::Mat::zeros(512, 512, CV_32F);
			cv::Mat directPos = cv::Mat::zeros(512, 512, CV_32FC3);
			auto dir = lights.direct_light_dir[0];
			for (int ii = 0; ii < 512; ++ii) {
				for (int j = 0; j < 512; ++j) {
					glm::vec3 point(j * CAUSTIC_MAP_DIS + CAUSTIC_X_MIN, 0, ii * CAUSTIC_MAP_DIS + CAUSTIC_X_MIN);
					int index;
					float u, v;
					float depth = bvh_intersect(point - dir * 1000.0f, dir, index, u, v, (glm::vec3*)vertex_buffer.data(), &bvhData[material[i].bvh_offset]);
					if (depth < 1e20) {
						lightPos.at<unsigned char>(ii, j) = 255;
						glm::vec3& n1 = ((glm::vec3*)normal_buffer.data())[index];
						glm::vec3& n2 = ((glm::vec3*)normal_buffer.data())[index + 1];
						glm::vec3& n3 = ((glm::vec3*)normal_buffer.data())[index + 2];
						glm::vec3 normal = u * (n2 - n1) + v * (n3 - n1) + n1;
						directLight.at<float>(ii, j) = glm::dot(-normal, dir);
						glm::vec3 pos = point + dir * (depth - 1000.0f);
						directPos.at<cv::Vec3f>(ii, j) = *(const cv::Vec3f*)&pos;
					}
				}
			}
			cv::Mat s;
			directLight.convertTo(s, CV_8U, 255.0F);
			cv::imshow("direct", s);
			cv::waitKey();
			float minX = 1000, maxX = -1000, minZ = 1000, maxZ = -1000;
			float minY = 1000, maxY = -1000;
			for (auto& v : objects[i]->vertex) {
				minX = std::min(v.x, minX);
				maxX = std::max(v.x, maxX);
				minZ = std::min(-v.y, minZ);
				maxZ = std::max(-v.y, maxZ);
				minY = std::min(v.z, minY);
				maxY = std::max(v.z, maxY);
			}
			float step = std::max(maxX - minX, maxZ - minZ) / 512;
			cv::Mat img = cv::Mat::zeros(600, 600, CV_32F);
			cv::Mat imgLight = cv::Mat::zeros(600, 600, CV_32F);
			minX -= 44 * step;
			minZ -= 44 * step;
			for (int j = 0; j < objects[i]->vertex.size(); j += 3) {
				printf("%d %d\n", j, objects[i]->vertex.size());
				auto& v1 = objects[i]->vertex[j];
				auto& v2 = objects[i]->vertex[j+1];
				auto& v3 = objects[i]->vertex[j+2];
				glm::vec3 p1((v1.x - minX) / step, (-v1.y - minZ) / step, 1);
				glm::vec3 p2((v2.x - minX) / step, (-v2.y - minZ) / step, 1);
				glm::vec3 p3((v3.x - minX) / step, (-v3.y - minZ) / step, 1);
				float tx = glm::length(p2 - p1);
				float ty = glm::length(p3 - p1);
				p1.z = v1.z;
				p2.z = v2.z;
				p3.z = v3.z;
				float radius = 0.2f;
				for (float px = 0; px < tx; px += 0.5) {
					for (float py = 0; py < ty; py += 0.5) {
						float wx = px / tx, wy = py / ty;
						if (wx + wy <= 1) {
							glm::vec3 p = p1 + (p2 - p1) * wx + (p3 - p1) * wy;
							glm::vec3 np = v1 + (v2 - v1) * wx + (v3 - v1) * wy;
							// I am going to compute the light!
							// first locate the light map
							glm::vec3 lightp = np + -np.y / dir.y * dir;
							int mx = p.x, my = p.y;
							if (img.at<float>(my, mx) > p.z)
								img.at<float>(my, mx) = p.z;
							else
								continue;
							float x = (lightp.x - CAUSTIC_X_MIN) / CAUSTIC_MAP_DIS;
							float y = (lightp.z - CAUSTIC_X_MIN) / CAUSTIC_MAP_DIS;
							int tx = x, ty = y;
							if (tx < 0 || tx >= 512 || ty < 0 || ty >= 512 || directLight.at<float>(ty, tx) == 0)
								continue;
							int bandwidth = radius / abs(dir.y) / CAUSTIC_MAP_DIS / 2;
							float lights = 0;
							for (int ly = y - bandwidth; ly <= y + bandwidth; ++ly) {
								for (int lx = x - bandwidth; lx <= x + bandwidth; ++lx) {
									if (ly < 0 || ly >= 512 || lx < 0 || lx >= 512)
										continue;
									float r = glm::length(np - *(glm::vec3*)&directPos.at<cv::Vec3f>(ly, lx));
									float weight = exp(-(r*r) / (radius*radius * 2)) / (1400 * radius * radius);
									lights += weight * directLight.at<float>(ly, lx);
								}
							}
							if (lights > 1000)
								lights = lights;
							imgLight.at<float>(my, mx) = lights;
						}
					}
				}
			}
			double minv, maxv;
			cv::minMaxLoc(imgLight, &minv, &maxv);
			printf("%lf %lf\n", minv, maxv);
			img = (img - minY) / (maxY - minY);
			img.convertTo(img, CV_8U, 255.0f);
			imgLight.convertTo(imgLight, CV_8U, 255.0f);
			cv::imshow("img", imgLight);
//			cv::imshow("img", img);
			cv::imwrite("img.png", imgLight);
			cv::waitKey();
		}
	}
}

void World::GenerateGeometries()
{
	num_triangles = 0;
	num_objects = objects.size();
	for (int i = 0; i < num_objects; ++i)
		num_triangles += objects[i]->vertex.size() / 3;
	vertex_buffer.resize(num_triangles * 9);
	normal_buffer.resize(num_triangles * 9);
	tex_buffer.resize(num_triangles * 6);
	index_buffer.resize(num_triangles);
	material.resize(num_objects);
	float* v_ptr = vertex_buffer.data(), *n_ptr = normal_buffer.data(), *t_ptr = tex_buffer.data();
	InstanceData* m_ptr = material.data();
	int s = 0;
	int bvh_offset = 0;
	for (int i = 0; i < num_objects; ++i) {
		m_ptr->kd = objects[i]->kd;
		m_ptr->ks = objects[i]->ks;
		glm::vec3 x, axisX, axisY, scale;
		x[0] = objects[i]->offset.x;
		x[1] = objects[i]->offset.y;
		x[2] = objects[i]->offset.z;
		axisX[0] = objects[i]->x_axis.x;
		axisX[1] = objects[i]->x_axis.y;
		axisX[2] = objects[i]->x_axis.z;
		axisY[0] = objects[i]->y_axis.x;
		axisY[1] = objects[i]->y_axis.y;
		axisY[2] = objects[i]->y_axis.z;
		scale[0] = objects[i]->s.x;
		scale[1] = objects[i]->s.y;
		scale[2] = objects[i]->s.z;
		int* index_ptr_o = index_buffer.data() + s / 3;
		for (int j = s / 3; j < s / 3 + objects[i]->vertex.size() / 3; ++j)
			index_buffer[j] = i;
		s += objects[i]->vertex.size();
		int* index_ptr = index_buffer.data() + s / 3;
		m_ptr->s = s;
		m_ptr->ka = objects[i]->ka;
		m_ptr->kr = objects[i]->kr;
		m_ptr->kf = objects[i]->kf;
		m_ptr->nr = objects[i]->nr;
		m_ptr->kt = objects[i]->kt;
		m_ptr->alpha = objects[i]->alpha;
		m_ptr->bvh_offset = bvh_offset;
		auto axisZ = glm::cross(axisX, axisY);
		glm::mat4 rotate = glm::mat4(glm::vec4(axisX, 0), glm::vec4(axisY, 0), glm::vec4(axisZ, 0), glm::vec4(0, 0, 0, 1));
		glm::mat4 convert = glm::mat4(glm::vec4(1, 0, 0, 0), glm::vec4(0, 1, 0, 0), glm::vec4(0, 0, 1, 0), glm::vec4(x, 1))
			* rotate
			* glm::mat4(glm::vec4(scale.x, 0, 0, 0), glm::vec4(0, scale.y, 0, 0), glm::vec4(0, 0, scale.z, 0), glm::vec4(0, 0, 0, 1));
		m_ptr->minPos = glm::vec3(1e30, 1e30, 1e30);
		m_ptr->maxPos = glm::vec3(-1e30, -1e30, -1e30);
		for (auto& v : objects[i]->vertex) {
			glm::vec4 v4(v, 1);
			v4 = convert * v4;
			v = glm::vec3(v4.x, v4.y, v4.z);
			for (int i = 0; i < 3; ++i) {
				m_ptr->minPos[i] = std::min(v[i], m_ptr->minPos[i]);
				m_ptr->maxPos[i] = std::max(v[i], m_ptr->maxPos[i]);
			}
		}
		m_ptr->minPos -= glm::vec3(1,1,1);
		m_ptr->maxPos += glm::vec3(1,1,1);
		m_ptr++;

		for (auto& n : objects[i]->normal) {
			glm::vec4 n4(n, 0);
			n4 = convert * n4;
			n = glm::normalize(glm::vec3(n4.x, n4.y, n4.z));
		}

		float* v_ptr_o = v_ptr, *n_ptr_o = n_ptr, *t_ptr_o = t_ptr;
		memcpy(v_ptr, objects[i]->vertex.data(), objects[i]->vertex.size() * 3 * sizeof(float));
		v_ptr += objects[i]->vertex.size() * 3;
		memcpy(n_ptr, objects[i]->normal.data(), objects[i]->normal.size() * 3 * sizeof(float));
		n_ptr += objects[i]->normal.size() * 3;

		memcpy(t_ptr, objects[i]->uv.data(), objects[i]->uv.size() * 2 * sizeof(float));
		t_ptr += objects[i]->uv.size() * 2;

		BVH* bvh = new BVH(v_ptr_o, n_ptr_o, t_ptr_o, index_ptr_o, (v_ptr - v_ptr_o) / 9);
		std::vector<BVHData> bvh_buffer;
		bvh->genBuffer(bvh_buffer, 0, (v_ptr_o - vertex_buffer.data()) / 9);
		bvhData.insert(bvhData.end(), bvh_buffer.begin(), bvh_buffer.end());
		bvh_offset = bvhData.size();
		delete bvh;
	}

	tex_offsets.resize(objects.size());
	int current_offset = 0;
	for (int i = 0; i < objects.size(); ++i) {
		tex_offsets[i] = glm::ivec3(current_offset, objects[i]->texImg.rows, objects[i]->texImg.cols);
		current_offset += objects[i]->texImg.rows * objects[i]->texImg.cols;
	}
	tex_images.resize(current_offset);
	for (int i = 0; i < objects.size(); ++i) {
		memcpy(tex_images.data() + tex_offsets[i].x, objects[i]->texImg.data, sizeof(uchar3) * objects[i]->texImg.rows * objects[i]->texImg.cols);
	}
	cudaMalloc(&vertexBuffer, sizeof(float) * vertex_buffer.size());
	cudaMemcpy(vertexBuffer, vertex_buffer.data(), sizeof(float) * vertex_buffer.size(), cudaMemcpyHostToDevice);
	cudaMalloc(&normalBuffer, sizeof(float) * normal_buffer.size());
	cudaMemcpy(normalBuffer, normal_buffer.data(), sizeof(float) * normal_buffer.size(), cudaMemcpyHostToDevice);
	cudaMalloc(&texBuffer, sizeof(float) * tex_buffer.size());
	cudaMemcpy(texBuffer, tex_buffer.data(), sizeof(float) * tex_buffer.size(), cudaMemcpyHostToDevice);
	cudaMalloc(&indexBuffer, sizeof(int) * index_buffer.size());
	cudaMemcpy(indexBuffer, index_buffer.data(), sizeof(int) * index_buffer.size(), cudaMemcpyHostToDevice);
	cudaMalloc(&materialBuffer, sizeof(InstanceData) * material.size());
	cudaMemcpy(materialBuffer, material.data(), sizeof(InstanceData) * material.size(), cudaMemcpyHostToDevice);

	cudaMalloc(&directLightsBuffer, sizeof(glm::vec3) * lights.direct_light_dir.size());
	cudaMemcpy(directLightsBuffer, lights.direct_light_dir.data(), sizeof(glm::vec3) * lights.direct_light_dir.size(), cudaMemcpyHostToDevice);
	cudaMalloc(&directLightsColorBuffer, sizeof(glm::vec3) * lights.direct_light_color.size());
	cudaMemcpy(directLightsColorBuffer, lights.direct_light_color.data(), sizeof(glm::vec3) * lights.direct_light_color.size(), cudaMemcpyHostToDevice);

	cudaMalloc(&pointLightsBuffer, sizeof(glm::vec3) * lights.point_light_pos.size());
	cudaMemcpy(pointLightsBuffer, lights.point_light_pos.data(), sizeof(glm::vec3) * lights.point_light_pos.size(), cudaMemcpyHostToDevice);
	cudaMalloc(&pointLightsColorBuffer, sizeof(glm::vec3) * lights.point_light_color.size());
	cudaMemcpy(pointLightsColorBuffer, lights.point_light_color.data(), sizeof(glm::vec3) * lights.point_light_color.size(), cudaMemcpyHostToDevice);

	cudaMalloc(&texOffsetBuffer, sizeof(glm::ivec3) * tex_offsets.size());
	cudaMemcpy(texOffsetBuffer, tex_offsets.data(), sizeof(glm::ivec3) * tex_offsets.size(), cudaMemcpyHostToDevice);
	cudaMalloc(&texImagesBuffer, sizeof(uchar3) * tex_images.size());
	cudaMemcpy(texImagesBuffer, tex_images.data(), sizeof(uchar3) * tex_images.size(), cudaMemcpyHostToDevice);

	causticMap.resize(512 * 512);
	cudaMalloc(&causticMapBuffer, sizeof(glm::ivec3) * causticMap.size());
	cudaMalloc(&causticBuffer, sizeof(glm::vec3) * causticMap.size());
	cudaMalloc(&causticCoordsBuffer, sizeof(glm::vec2) * causticMap.size());

	cudaMalloc(&bvhDataBuffer, sizeof(BVHData) * bvhData.size());
	cudaMemcpy(bvhDataBuffer, bvhData.data(), sizeof(BVHData) * bvhData.size(), cudaMemcpyHostToDevice);

	cv::Mat img = cv::imread("tex\\environment.jpg");
	cudaMalloc(&environmentBuffer, sizeof(uchar3) * img.rows * img.cols);
	cudaMemcpy(environmentBuffer, img.data, sizeof(uchar3) * img.cols * img.rows, cudaMemcpyHostToDevice);
	cudaMalloc(&scatterBuffer, sizeof(glm::vec3) * CAUSTIC_W*CAUSTIC_W);
	cudaMalloc(&scatterPosBuffer, sizeof(glm::vec3)*CAUSTIC_W*CAUSTIC_W);
//	ProcessScattering();
}
