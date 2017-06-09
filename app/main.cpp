#include "../cuda-opengl.h"
#include "../loader/objloader.h"
#include "../scene/world.h"

// Shared Library Test Functions
#define MAX_EPSILON 10
#define REFRESH_DELAY     10 //ms

const char *sSDKname = "simpleCUDA2GL";

unsigned int g_TotalErrors = 0;

// CheckFBO/BackBuffer class objects
CheckRender *g_CheckRender = NULL;

////////////////////////////////////////////////////////////////////////////////
// constants / global variables
unsigned int window_width = 512;
unsigned int window_height = 512;
unsigned int image_width = 512;
unsigned int image_height = 512;
int iGLUTWindowHandle = 0;          // handle to the GLUT window

// pbo and fbo variables
#ifdef USE_TEXSUBIMAGE2D
GLuint pbo_dest;
struct cudaGraphicsResource *cuda_pbo_dest_resource;
#else
unsigned int *cuda_dest_resource;
GLuint shDrawTex;  // draws a texture
struct cudaGraphicsResource *cuda_tex_result_resource;
#endif

GLuint fbo_source;
struct cudaGraphicsResource *cuda_tex_screen_resource;

unsigned int size_tex_data;
unsigned int num_texels;
unsigned int num_values;

// (offscreen) render target fbo variables
GLuint tex_screen;      // where we render the image
GLuint tex_cudaResult;  // where we will copy the CUDA result

char *ref_file = NULL;
bool enable_cuda = true;

int   *pArgc = NULL;
char **pArgv = NULL;


// Timer
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;

#ifndef USE_TEXTURE_RGBA8UI
#   pragma message("Note: Using Texture fmt GL_RGBA16F_ARB")
#else
// NOTE: the current issue with regular RGBA8 internal format of textures
// is that HW stores them as BGRA8. Therefore CUDA will see BGRA where users
// expected RGBA8. To prevent this issue, the driver team decided to prevent this to happen
// instead, use RGBA8UI which required the additional work of scaling the fragment shader
// output from 0-1 to 0-255. This is why we have some GLSL code, in this case
#   pragma message("Note: Using Texture RGBA8UI + GLSL for rendering")
#endif
GLuint shDraw;


////////////////////////////////////////////////////////////////////////////////

// Forward declarations
void runStdProgram(int argc, char **argv);
void FreeResource();
void Cleanup(int iExitCode);

// GL functionality
bool initCUDA(int argc, char **argv, bool bUseGL);
bool initGL(int *argc, char **argv);

#ifdef USE_TEXSUBIMAGE2D
void createPBO(GLuint *pbo, struct cudaGraphicsResource **pbo_resource);
void deletePBO(GLuint *pbo);
#endif

void createTextureDst(GLuint *tex_cudaResult, unsigned int size_x, unsigned int size_y);
void deleteTexture(GLuint *tex);

// rendering callbacks
void display();
void idle();
void keyboard(unsigned char key, int x, int y);
void reshape(int w, int h);
void mainMenu(int i);

#ifdef USE_TEXSUBIMAGE2D
////////////////////////////////////////////////////////////////////////////////
//! Create PBO
////////////////////////////////////////////////////////////////////////////////
void
createPBO(GLuint *pbo, struct cudaGraphicsResource **pbo_resource)
{
	// set up vertex data parameter
	num_texels = image_width * image_height;
	num_values = num_texels * 4;
	size_tex_data = sizeof(GLubyte) * num_values;
	void *data = malloc(size_tex_data);

	// create buffer object
	glGenBuffers(1, pbo);
	glBindBuffer(GL_ARRAY_BUFFER, *pbo);
	glBufferData(GL_ARRAY_BUFFER, size_tex_data, data, GL_DYNAMIC_DRAW);
	free(data);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(pbo_resource, *pbo, cudaGraphicsMapFlagsNone));

	SDK_CHECK_ERROR_GL();
}

void
deletePBO(GLuint *pbo)
{
	glDeleteBuffers(1, pbo);
	SDK_CHECK_ERROR_GL();
	*pbo = 0;
}
#endif

const GLenum fbo_targets[] =
{
	GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT,
	GL_COLOR_ATTACHMENT2_EXT, GL_COLOR_ATTACHMENT3_EXT
};

#ifndef USE_TEXSUBIMAGE2D
static const char *glsl_drawtex_vertshader_src =
"void main(void)\n"
"{\n"
"	gl_Position = gl_Vertex;\n"
"	gl_TexCoord[0].xy = gl_MultiTexCoord0.xy;\n"
"}\n";

static const char *glsl_drawtex_fragshader_src =
"#version 130\n"
"uniform usampler2D texImage;\n"
"void main()\n"
"{\n"
"   vec4 c = texture(texImage, gl_TexCoord[0].xy);\n"
"	gl_FragColor = c / 255.0;\n"
"}\n";
#endif

static const char *glsl_draw_fragshader_src =
//WARNING: seems like the gl_FragColor doesn't want to output >1 colors...
//you need version 1.3 so you can define a uvec4 output...
//but MacOSX complains about not supporting 1.3 !!
// for now, the mode where we use RGBA8UI may not work properly for Apple : only RGBA16F works (default)
#if defined(__APPLE__) || defined(MACOSX)
"void main()\n"
"{"
"  gl_FragColor = vec4(gl_Color * 255.0);\n"
"}\n";
#else
"#version 130\n"
"out uvec4 FragColor;\n"
"void main()\n"
"{"
"  FragColor = uvec4(gl_Color.xyz * 255.0, 255.0);\n"
"}\n";
#endif


// display image to the screen as textured quad
void displayImage(GLuint texture)
{
	glBindTexture(GL_TEXTURE_2D, texture);
	glEnable(GL_TEXTURE_2D);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glViewport(0, 0, window_width, window_height);

	// if the texture is a 8 bits UI, scale the fetch with a GLSL shader
#ifndef USE_TEXSUBIMAGE2D
	glUseProgram(shDrawTex);
	GLint id = glGetUniformLocation(shDrawTex, "texImage");
	glUniform1i(id, 0); // texture unit 0 to "texImage"
	SDK_CHECK_ERROR_GL();
#endif

	glBegin(GL_QUADS);
	glTexCoord2f(0.0, 0.0);
	glVertex3f(-1.0, -1.0, 0.5);
	glTexCoord2f(1.0, 0.0);
	glVertex3f(1.0, -1.0, 0.5);
	glTexCoord2f(1.0, 1.0);
	glVertex3f(1.0, 1.0, 0.5);
	glTexCoord2f(0.0, 1.0);
	glVertex3f(-1.0, 1.0, 0.5);
	glEnd();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	glDisable(GL_TEXTURE_2D);

#ifndef USE_TEXSUBIMAGE2D
	glUseProgram(0);
#endif
	SDK_CHECK_ERROR_GL();
}

extern void RenderImage();
////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void
display()
{
	sdkStartTimer(&timer);

	if (enable_cuda)
	{
		RenderImage();
		displayImage(tex_cudaResult);
	}

	// NOTE: I needed to add this call so the timing is consistent.
	// Need to investigate why
	cudaDeviceSynchronize();
	sdkStopTimer(&timer);

	// flip backbuffer
	glutSwapBuffers();

	// If specified, Check rendering against reference,
	if (ref_file && g_CheckRender && g_CheckRender->IsQAReadback())
	{

		static int pass = 0;

		if (pass > 0)
		{
			g_CheckRender->readback(window_width, window_height);
			char currentOutputPPM[256];
			sprintf(currentOutputPPM, "kilt.ppm");
			g_CheckRender->savePPM(currentOutputPPM, true, NULL);

			if (!g_CheckRender->PPMvsPPM(currentOutputPPM, sdkFindFilePath(ref_file, pArgv[0]), MAX_EPSILON, 0.30f))
			{
				g_TotalErrors++;
			}

			Cleanup((g_TotalErrors == 0) ? EXIT_SUCCESS : EXIT_FAILURE);
		}

		pass++;
	}

	// Update fps counter, fps/title display and log
	if (++fpsCount == fpsLimit)
	{
		char cTitle[256];
		float fps = 1000.0f / sdkGetAverageTimerValue(&timer);
		sprintf(cTitle, "PhotonGPU (%d x %d): %.1f fps", window_width, window_height, fps);
		glutSetWindowTitle(cTitle);
		//printf("%s\n", cTitle);
		fpsCount = 0;
		fpsLimit = (int)((fps > 1.0f) ? fps : 1.0f);
		sdkResetTimer(&timer);
	}
}

void timerEvent(int value)
{
	glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}


int mouse_x, mouse_y;
int mouse_state = 0;
glm::vec3 camera, camera_lookat, camera_up;

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void
keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key)
	{
	case (27) :
		Cleanup(EXIT_SUCCESS);
		break;
	case 'p':
		g_world.pause = 1 - g_world.pause;
		break;
	case 'r':
		mouse_state = 1;
		break;
	case ' ':
		enable_cuda ^= 1;
#ifdef USE_TEXTURE_RGBA8UI

		if (enable_cuda)
		{
			glClearColorIuiEXT(128, 128, 128, 255);
		}
		else
		{
			glClearColor(0.5, 0.5, 0.5, 1.0);
		}

#endif
		break;

	}
}

void reshape(int w, int h)
{
	window_width = w;
	window_height = h;
}

void mainMenu(int i)
{
	keyboard((unsigned char)i, 0, 0);
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
void
createTextureDst(GLuint *tex_cudaResult, unsigned int size_x, unsigned int size_y)
{
	// create a texture
	glGenTextures(1, tex_cudaResult);
	glBindTexture(GL_TEXTURE_2D, *tex_cudaResult);

	// set basic parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

#ifdef USE_TEXSUBIMAGE2D
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size_x, size_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	SDK_CHECK_ERROR_GL();
#else
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, size_x, size_y, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);
	SDK_CHECK_ERROR_GL();
	// register this texture with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_tex_result_resource, *tex_cudaResult,
		GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
#endif
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
void
deleteTexture(GLuint *tex)
{
	glDeleteTextures(1, tex);
	SDK_CHECK_ERROR_GL();

	*tex = 0;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
World g_world;
extern void testCPU();
int
main(int argc, char **argv)
{
#if defined(__linux__)
	setenv("DISPLAY", ":0", 0);
#endif
	
	printf("Load Scene...\n");
	loadScene("raytracing.scene", World::objects);
	g_world.GenerateGeometries();
//	testCPU();
	printf("%s Starting...\n\n", argv[0]);
	if (checkCmdLineFlag(argc, (const char **)argv, "file"))
	{

		getCmdLineArgumentString(argc, (const char **)argv, "file", &ref_file);
	}

	pArgc = &argc;
	pArgv = argv;

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if (checkCmdLineFlag(argc, (const char **)argv, "device"))
	{
		printf("[%s]\n", argv[0]);
		printf("   Does not explicitly support -device=n\n");
		printf("   This sample requires OpenGL.  Only -file=<reference> are supported\n");
		printf("exiting...\n");
		exit(EXIT_WAIVED);
	}

	if (ref_file)
	{
		printf("(Test with OpenGL verification)\n");
		runStdProgram(argc, argv);
	}
	else
	{
		printf("(Interactive OpenGL Demo)\n");
		runStdProgram(argc, argv);
	}

	exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
void FreeResource()
{
	sdkDeleteTimer(&timer);

	// unregister this buffer object with CUDA
	//    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_tex_screen_resource));
#ifdef USE_TEXSUBIMAGE2D
	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_dest_resource));
	deletePBO(&pbo_dest);
#else
	cudaFree(cuda_dest_resource);
#endif
	deleteTexture(&tex_screen);
	deleteTexture(&tex_cudaResult);

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();

	if (iGLUTWindowHandle)
	{
		glutDestroyWindow(iGLUTWindowHandle);
	}

	// finalize logs and leave
	printf("simpleCUDA2GL Exiting...\n");
}

void Cleanup(int iExitCode)
{
	FreeResource();
	printf("PPM Images are %s\n", (iExitCode == EXIT_SUCCESS) ? "Matching" : "Not Matching");
	exit(iExitCode);
}


////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
GLuint compileGLSLprogram(const char *vertex_shader_src, const char *fragment_shader_src)
{
	GLuint v, f, p = 0;

	p = glCreateProgram();

	if (vertex_shader_src)
	{
		v = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(v, 1, &vertex_shader_src, NULL);
		glCompileShader(v);

		// check if shader compiled
		GLint compiled = 0;
		glGetShaderiv(v, GL_COMPILE_STATUS, &compiled);

		if (!compiled)
		{
			//#ifdef NV_REPORT_COMPILE_ERRORS
			char temp[256] = "";
			glGetShaderInfoLog(v, 256, NULL, temp);
			printf("Vtx Compile failed:\n%s\n", temp);
			//#endif
			glDeleteShader(v);
			return 0;
		}
		else
		{
			glAttachShader(p, v);
		}
	}

	if (fragment_shader_src)
	{
		f = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(f, 1, &fragment_shader_src, NULL);
		glCompileShader(f);

		// check if shader compiled
		GLint compiled = 0;
		glGetShaderiv(f, GL_COMPILE_STATUS, &compiled);

		if (!compiled)
		{
			//#ifdef NV_REPORT_COMPILE_ERRORS
			char temp[256] = "";
			glGetShaderInfoLog(f, 256, NULL, temp);
			printf("frag Compile failed:\n%s\n", temp);
			//#endif
			glDeleteShader(f);
			return 0;
		}
		else
		{
			glAttachShader(p, f);
		}
	}

	glLinkProgram(p);

	int infologLength = 0;
	int charsWritten = 0;

	glGetProgramiv(p, GL_INFO_LOG_LENGTH, (GLint *)&infologLength);

	if (infologLength > 0)
	{
		char *infoLog = (char *)malloc(infologLength);
		glGetProgramInfoLog(p, infologLength, (GLsizei *)&charsWritten, infoLog);
//		printf("Shader compilation error: %s\n", infoLog);
		free(infoLog);
	}

	return p;
}

////////////////////////////////////////////////////////////////////////////////
//! Allocate the "render target" of CUDA
////////////////////////////////////////////////////////////////////////////////
#ifndef USE_TEXSUBIMAGE2D
void initCUDABuffers()
{
	// set up vertex data parameter
	num_texels = image_width * image_height;
	num_values = num_texels * 4;
	size_tex_data = sizeof(GLubyte) * num_values;
	checkCudaErrors(cudaMalloc((void **)&cuda_dest_resource, size_tex_data));
	//checkCudaErrors(cudaHostAlloc((void**)&cuda_dest_resource, size_tex_data, ));
}
#endif

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
void initGLBuffers()
{
	// create pbo
#ifdef USE_TEXSUBIMAGE2D
	createPBO(&pbo_dest, &cuda_pbo_dest_resource);
#endif
	// create texture that will receive the result of CUDA
	createTextureDst(&tex_cudaResult, image_width, image_height);
	// load shader programs
	shDraw = compileGLSLprogram(NULL, glsl_draw_fragshader_src);

#ifndef USE_TEXSUBIMAGE2D
	shDrawTex = compileGLSLprogram(glsl_drawtex_vertshader_src, glsl_drawtex_fragshader_src);
#endif
	SDK_CHECK_ERROR_GL();
}

void MyMouse(int button, int state, int x, int y)
{
	switch (button)
	{
	case GLUT_LEFT_BUTTON:
		if (state == GLUT_DOWN) {
			mouse_state = 0;
		}
		break;
	case GLUT_RIGHT_BUTTON:
		if (state == GLUT_DOWN) {
			if (mouse_state == 2) {
				g_world.camera = camera;
				g_world.camera_lookat = camera_lookat;
				g_world.camera_up = camera_up;
			}
			mouse_state = 0;
		}
	}
}
void MyMotion(int x, int y) {
	printf("%lf %lf %lf\n", camera_lookat.x, camera_lookat.y, camera_lookat.z);
	if (mouse_state == 1) {
		camera = g_world.camera;
		camera_up = g_world.camera_up;
		camera_lookat = g_world.camera_lookat;
		mouse_state = 2;
		mouse_x = x;
		mouse_y = y;
		return;
	}
	if (mouse_state == 2) {
		float dx = (x - mouse_x) / (float)window_width;
		float dy = (y - mouse_y) / (float)window_width;
		printf("%lf %lf\n", dx, dy);
		double angle = dx * g_world.fov / 180.0 * CV_PI * 2;
		glm::mat3 rot(1.0f);
		rot[0][0] = cos(angle); rot[0][2] = -sin(angle);
		rot[2][0] = sin(angle); rot[2][2] = cos(angle);
		g_world.camera = rot * camera;
		g_world.camera_up = rot * camera_up;
		g_world.camera_lookat = rot * camera_lookat;
		angle = dy * g_world.fov / 180.0 * CV_PI * 2;
		rot = glm::mat3(1.0f);
		rot[1][1] = cos(angle); rot[1][2] = -sin(angle);
		rot[2][1] = sin(angle); rot[2][2] = cos(angle);
		g_world.camera = rot * g_world.camera;
		g_world.camera_up = rot * g_world.camera_up;
		g_world.camera_lookat = rot * g_world.camera_lookat;
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Run standard demo loop with or without GL verification
////////////////////////////////////////////////////////////////////////////////
void
runStdProgram(int argc, char **argv)
{
	// First initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	if (false == initGL(&argc, argv))
	{
		return;
	}

	// Now initialize CUDA context (GL context has been created already)
	initCUDA(argc, argv, true);

	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);

	// register callbacks
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(MyMouse);
	glutPassiveMotionFunc(MyMotion);
	glutReshapeFunc(reshape);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

	
	initGLBuffers();
#ifndef USE_TEXSUBIMAGE2D
	initCUDABuffers();
#endif

	// Creating the Auto-Validation Code
	if (ref_file)
	{
		g_CheckRender = new CheckBackBuffer(window_width, window_height, 4);
		g_CheckRender->setPixelFormat(GL_RGBA);
		g_CheckRender->setExecPath(argv[0]);
		g_CheckRender->EnableQAReadback(true);
	}


	// start rendering mainloop
	glutMainLoop();

	// Normally unused return path
	Cleanup(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize CUDA context
////////////////////////////////////////////////////////////////////////////////
bool
initCUDA(int argc, char **argv, bool bUseGL)
{
	if (bUseGL)
	{
		findCudaGLDevice(argc, (const char **)argv);
	}
	else
	{
		findCudaDevice(argc, (const char **)argv);
	}

	return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool
initGL(int *argc, char **argv)
{
	// Create GL context
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(window_width, window_height);
	iGLUTWindowHandle = glutCreateWindow("CUDA OpenGL post-processing");

	// initialize necessary OpenGL extensions
	glewInit();

	if (!glewIsSupported(
		"GL_VERSION_2_0 "
		"GL_ARB_pixel_buffer_object "
		"GL_EXT_framebuffer_object "
		))
	{
		printf("ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	// default initialization
#ifndef USE_TEXTURE_RGBA8UI
	glClearColor(0.5, 0.5, 0.5, 1.0);
#else
	glClearColorIuiEXT(128, 128, 128, 255);
#endif
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, window_width, window_height);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1f, 10.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glEnable(GL_LIGHT0);
	float red[] = { 1.0f, 0.1f, 0.1f, 1.0f };
	float white[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, red);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 60.0f);

	SDK_CHECK_ERROR_GL();

	return true;
}