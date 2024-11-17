#include <iostream>
#include <cuda_runtime.h>

// Error checking macro
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code)
                  << " " << file << " " << line << std::endl;
        exit(code);
    }
}

__global__ void arrayRotate(const float* A, float* B, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int new_pos = (i + K) % N; 
        B[new_pos] = A[i];
    }
}

void runArrayRotation(int N, int K) {
    size_t size = N * sizeof(float);

    // Allocate host memory
    float* h_A = new float[N];
    float* h_B = new float[N]; // For rotated array

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float* d_A;
    float* d_B;
    cudaCheckError(cudaMalloc((void**)&d_A, size));
    cudaCheckError(cudaMalloc((void**)&d_B, size));

    // Copy data to device
    cudaCheckError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

    // Setup execution parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    arrayRotate<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N, K);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    // Copy result back to host
    cudaCheckError(cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost));

    // Verify the result
    bool success = true;
    for (int i = 0; i < N; ++i) {
        int expected = (i - K + N) % N; // Inverse operation to find original index
        if (h_B[i] != static_cast<float>(expected)) {
            success = false;
            std::cerr << "Mismatch at index " << i << ": " << h_B[i]
                      << " != " << expected << std::endl;
            break;
        }
    }

    if (success) {
        std::cout << "Array rotation successful." << std::endl;
    } else {
        std::cout << "Array rotation failed." << std::endl;
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    delete[] h_A;
    delete[] h_B;
}

int main() {
    int N = 1 << 20; 
    int K = 12345;   // Number of positions to rotate
    runArrayRotation(N, K);

    return 0;
}

