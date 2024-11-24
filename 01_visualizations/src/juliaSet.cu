#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

// CUDA kernel to compute the Julia set
__global__ void juliaKernel(int* output, int width, int height, float x_min, float x_max, float y_min, float y_max, float c_real, float c_imag, int max_iter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < width && idy < height) {
        // Map pixel to complex plane
        float x = x_min + idx * (x_max - x_min) / width;
        float y = y_min + idy * (y_max - y_min) / height;

        int iter = 0;
        while (x * x + y * y <= 4.0f && iter < max_iter) {
            float temp = x * x - y * y + c_real;
            y = 2.0f * x * y + c_imag;
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

void runJuliaSet(int width, int height, float x_min, float x_max, float y_min, float y_max, float c_real, float c_imag, int max_iter) {
    size_t size = width * height * sizeof(int);

    // Allocate host and device memory
    int* h_output = new int[width * height];
    int* d_output;
    cudaMalloc((void**)&d_output, size);

    // Define thread block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    juliaKernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, width, height, x_min, x_max, y_min, y_max, c_real, c_imag, max_iter);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Save the result as a PGM image
    saveImage(h_output, width, height, "julia.pgm");
    std::cout << "Julia set saved to 'julia.pgm'\n";

    // Cleanup
    cudaFree(d_output);
    delete[] h_output;
}

int main() {
    int width = 1024;
    int height = 1024;
    float x_min = -2.0f, x_max = 2.0f;
    float y_min = -2.0f, y_max = 2.0f;
    float c_real = -0.7f, c_imag = 0.27015f;  // Julia set parameter
    int max_iter = 1000;

    std::cout << "Generating Julia set...\n";
    runJuliaSet(width, height, x_min, x_max, y_min, y_max, c_real, c_imag, max_iter);
    return 0;
}
