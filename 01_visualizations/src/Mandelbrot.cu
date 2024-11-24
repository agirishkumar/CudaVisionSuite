#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

// Error checking macro
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", 
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// Kernel for Mandelbrot set computation
__global__ void mandelbrotKernel(int* output, int width, int height, float x_min, float x_max, float y_min, float y_max, int max_iter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < width && idy < height) {
        // Map pixel to complex plane
        float x0 = x_min + idx * (x_max - x_min) / width;
        float y0 = y_min + idy * (y_max - y_min) / height;

        float x = 0.0f, y = 0.0f;
        int iter = 0;

        // Iterate z = z^2 + c
        while (x * x + y * y <= 4.0f && iter < max_iter) {
            float temp = x * x - y * y + x0;
            y = 2.0f * x * y + y0;
            x = temp;
            iter++;
        }

        // Map iteration count to output
        output[idy * width + idx] = iter;
    }
}

void saveImage(const int* data, int width, int height, const char* filename) {
    std::ofstream image(filename);
    image << "P2\n" << width << " " << height << "\n255\n";

    for (int i = 0; i < width * height; i++) {
        int value = (data[i] % 256);
        image << value << " ";
        if (i % width == width - 1) image << "\n";
    }

    image.close();
}

void runMandelbrot(int width, int height, float x_min, float x_max, float y_min, float y_max, int max_iter) {
    size_t size = width * height * sizeof(int);

    // Allocate host memory
    int* h_output = new int[width * height];

    // Allocate device memory
    int* d_output;
    cudaCheckError(cudaMalloc((void**)&d_output, size));

    // Define thread block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    mandelbrotKernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, width, height, x_min, x_max, y_min, y_max, max_iter);
    cudaCheckError(cudaGetLastError());

    // Copy result back to host
    cudaCheckError(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    // Save the result as a PGM image
    saveImage(h_output, width, height, "mandelbrot.pgm");
    std::cout << "Mandelbrot set saved to 'mandelbrot.pgm'\n";

    // Cleanup
    cudaFree(d_output);
    delete[] h_output;
}

int main() {
    int width = 1024;
    int height = 1024;
    float x_min = -2.0f, x_max = 1.0f;
    float y_min = -1.5f, y_max = 1.5f;
    int max_iter = 1000;

    std::cout << "Generating Mandelbrot set...\n";
    runMandelbrot(width, height, x_min, x_max, y_min, y_max, max_iter);
    return 0;
}
