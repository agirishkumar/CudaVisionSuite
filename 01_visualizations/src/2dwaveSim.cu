#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>
#include <string>

// Error checking macro for CUDA
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", 
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// Vertex shader source
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out float height;

void main() {
    height = aPos.y;  // Pass height to fragment shader
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
)";

// Fragment shader source
const char* fragmentShaderSource = R"(
#version 330 core
in float height;
out vec4 FragColor;

void main() {
    // Create cyan color with intensity based on height
    float intensity = (height + 1.0) * 0.5;  // Map [-1,1] to [0,1]
    vec3 cyan = vec3(0.0, intensity, intensity);
    FragColor = vec4(cyan, 1.0);
}
)";

// CUDA kernel for wave simulation
__global__ void updateWave(float* vertices, float* u_curr, float* u_prev, float* u_next,
                          int width, int height, float c, float dt_dx2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int idx = y * width + x;
        
        // Update wave equation
        u_next[idx] = 2 * u_curr[idx] - u_prev[idx] + c * c * dt_dx2 * (
            u_curr[idx - 1] + u_curr[idx + 1] +
            u_curr[idx - width] + u_curr[idx + width] -
            4 * u_curr[idx]
        );
        
        // Update vertex positions for rendering
        int vidx = idx * 3;  // Each vertex has 3 components (x,y,z)
        vertices[vidx + 1] = u_next[idx];  // Update Y coordinate (height)
    }
}

class WaveSimulator {
private:
    GLFWwindow* window;
    GLuint VBO, VAO, EBO;
    GLuint shaderProgram;
    
    // Camera parameters
    glm::vec3 cameraPos;
    glm::vec3 cameraFront;
    glm::vec3 cameraUp;
    float cameraSpeed;
    
    // CUDA resources
    float *d_vertices;
    float *d_u_prev, *d_u_curr, *d_u_next;
    cudaGraphicsResource* cuda_vbo_resource;
    
    int width, height;
    float c;  // Wave speed
    float dt, dx;
    float dt_dx2;
    
    GLuint compileShader(const char* source, GLenum type) {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &source, NULL);
        glCompileShader(shader);
        
        // Check compilation status
        int success;
        char infoLog[512];
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 512, NULL, infoLog);
            std::cerr << "Shader compilation failed:\n" << infoLog << std::endl;
            exit(-1);
        }
        return shader;
    }
    
    void initOpenGL() {
        // Initialize GLFW
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW" << std::endl;
            exit(-1);
        }
        
        // Configure GLFW
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        
        // Create window
        window = glfwCreateWindow(800, 600, "Wave Simulation", NULL, NULL);
        if (!window) {
            std::cerr << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            exit(-1);
        }
        glfwMakeContextCurrent(window);
        
        // Initialize GLEW
        if (glewInit() != GLEW_OK) {
            std::cerr << "Failed to initialize GLEW" << std::endl;
            exit(-1);
        }
        
        // Enable depth testing
        glEnable(GL_DEPTH_TEST);
        
        // Create and compile shaders
        GLuint vertexShader = compileShader(vertexShaderSource, GL_VERTEX_SHADER);
        GLuint fragmentShader = compileShader(fragmentShaderSource, GL_FRAGMENT_SHADER);
        
        // Create shader program
        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);
        
        // Check linking status
        int success;
        char infoLog[512];
        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
            std::cerr << "Shader program linking failed:\n" << infoLog << std::endl;
            exit(-1);
        }
        
        // Clean up shaders
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        
        // Initialize camera
        cameraPos = glm::vec3(0.0f, 2.0f, 3.0f);
        cameraFront = glm::vec3(0.0f, -0.5f, -1.0f);
        cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
        cameraSpeed = 0.05f;
    }
    
    void setupVertexData() {
        std::vector<float> vertices;
        std::vector<unsigned int> indices;
        
        // Generate grid vertices
        float xStep = 2.0f / (width - 1);
        float zStep = 2.0f / (height - 1);
        
        for (int z = 0; z < height; z++) {
            for (int x = 0; x < width; x++) {
                vertices.push_back(x * xStep - 1.0f);    // X
                vertices.push_back(0.0f);                // Y (height)
                vertices.push_back(z * zStep - 1.0f);    // Z
            }
        }
        
        // Generate indices for triangle strips
        for (int z = 0; z < height - 1; z++) {
            for (int x = 0; x < width; x++) {
                indices.push_back(z * width + x);
                indices.push_back((z + 1) * width + x);
            }
            // Add degenerate triangles if this isn't the last strip
            if (z < height - 2) {
                indices.push_back((z + 1) * width + (width - 1));
                indices.push_back((z + 1) * width);
            }
        }
        
        // Create buffers
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);
        
        // Bind VAO
        glBindVertexArray(VAO);
        
        // Setup VBO
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float),
                    vertices.data(), GL_DYNAMIC_DRAW);
        
        // Setup EBO
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int),
                    indices.data(), GL_STATIC_DRAW);
        
        // Setup vertex attributes
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        // Register VBO with CUDA
        cudaCheckError(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, VBO, 
                                                  cudaGraphicsMapFlagsWriteDiscard));
    }
    
    void initCUDA() {
        size_t size = width * height * sizeof(float);
        
        // Allocate device memory for wave simulation
        cudaCheckError(cudaMalloc((void**)&d_u_prev, size));
        cudaCheckError(cudaMalloc((void**)&d_u_curr, size));
        cudaCheckError(cudaMalloc((void**)&d_u_next, size));
        
        // Initialize with zero
        cudaCheckError(cudaMemset(d_u_prev, 0, size));
        cudaCheckError(cudaMemset(d_u_curr, 0, size));
        cudaCheckError(cudaMemset(d_u_next, 0, size));
        
        // Create initial perturbation
        std::vector<float> h_init(width * height, 0.0f);
        int cx = width / 2;
        int cy = height / 2;
        h_init[cy * width + cx] = 1.0f;
        
        cudaCheckError(cudaMemcpy(d_u_curr, h_init.data(), size, 
                                cudaMemcpyHostToDevice));
    }

public:
    WaveSimulator(int w, int h, float wave_speed) 
        : width(w), height(h), c(wave_speed) {
        dt = 0.1f;
        dx = 1.0f;
        dt_dx2 = (dt * dt) / (dx * dx);
        
        initOpenGL();
        setupVertexData();
        initCUDA();
    }
    
    void processInput() {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);
            
        // Camera movement
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            cameraPos += cameraSpeed * cameraFront;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            cameraPos -= cameraSpeed * cameraFront;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    }
    
    void run() {
        while (!glfwWindowShouldClose(window)) {
            processInput();
            
            // Map OpenGL buffer for writing from CUDA
            float* dptr;
            cudaCheckError(cudaGraphicsMapResources(1, &cuda_vbo_resource));
            size_t num_bytes;
            cudaCheckError(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes,
                                                             cuda_vbo_resource));
            
            // Launch CUDA kernel
            dim3 threadsPerBlock(16, 16);
            dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16);
            updateWave<<<blocksPerGrid, threadsPerBlock>>>(
                dptr, d_u_curr, d_u_prev, d_u_next, width, height, c, dt_dx2);
            
            // Unmap buffer
            cudaCheckError(cudaGraphicsUnmapResources(1, &cuda_vbo_resource));
            
            // Swap wave buffers
            float* temp = d_u_prev;
            d_u_prev = d_u_curr;
            d_u_curr = d_u_next;
            d_u_next = temp;
            
            // Render
            render();
            
            // Swap buffers and poll events
            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }
    
    void render() {
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Use shader program
        glUseProgram(shaderProgram);
        
        // Update transformation matrices
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        glm::mat4 projection = glm::perspective(glm::radians(45.0f),
                                              800.0f / 600.0f, 0.1f, 100.0f);
        
        // Set uniforms
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"),
                          1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"),
                          1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"),
                          1, GL_FALSE, glm::value_ptr(projection));
        
        // Draw mesh
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLE_STRIP, (width * 2 + 2) * (height - 1) - 2,
                      GL_UNSIGNED_INT, 0);
    }
    
    ~WaveSimulator() {
        // Cleanup CUDA
        cudaGraphicsUnregisterResource(cuda_vbo_resource);
        cudaFree(d_u_prev);
        cudaFree(d_u_curr);
        cudaFree(d_u_next);
        
        // Cleanup OpenGL
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);
        glDeleteProgram(shaderProgram);
        
        glfwTerminate();
    }
}; // Class end

int main() {
    int width = 128;  
    int height = 128;
    float c = 0.5f; 
    
    try {
        WaveSimulator simulator(width, height, c);
        simulator.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}