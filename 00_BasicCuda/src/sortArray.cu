#include <iostream>
#include <cuda_runtime.h>
#include <cmath> // For log2
#include <cstdlib> // For rand()

// Error checking macro
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code)
                  << " " << file << " " << line << std::endl;
        exit(code);
    }
}

__global__ void bitonicSortKernel(float* d_data, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;

    if (ixj > i) {
        if ((i & k) == 0) {
            // Sort ascending
            if (d_data[i] > d_data[ixj]) {
                // Swap elements
                float temp = d_data[i];
                d_data[i] = d_data[ixj];
                d_data[ixj] = temp;
            }
        } else {
            // Sort descending
            if (d_data[i] < d_data[ixj]) {
                // Swap elements
                float temp = d_data[i];
                d_data[i] = d_data[ixj];
                d_data[ixj] = temp;
            }
        }
    }
}

void runBitonicSort(int N) {
    size_t size = N * sizeof(float);

    // Allocate host memory
    float* h_data = new float[N];

    // Initialize host array with random data
    for (int i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float* d_data;
    cudaCheckError(cudaMalloc((void**)&d_data, size));

    // Copy data to device
    cudaCheckError(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    // Set up execution parameters
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Main sorting loop
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicSortKernel<<<blocks, threadsPerBlock>>>(d_data, j, k);
            cudaCheckError(cudaGetLastError());
            cudaCheckError(cudaDeviceSynchronize());
        }
    }

    // Copy result back to host
    cudaCheckError(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // Verify the result
    bool success = true;
    for (int i = 0; i < N - 1; ++i) {
        if (h_data[i] > h_data[i + 1]) {
            success = false;
            std::cerr << "Array not sorted at index " << i << ": "
                      << h_data[i] << " > " << h_data[i + 1] << std::endl;
            break;
        }
    }

    if (success) {
        std::cout << "Array sorted successfully." << std::endl;
    } else {
        std::cout << "Array sorting failed." << std::endl;
    }

    // Cleanup
    cudaFree(d_data);
    delete[] h_data;
}

int main() {
    int N = 1 << 16; // Example size (must be a power of two)
    runBitonicSort(N);
    return 0;
}
