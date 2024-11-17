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

__global__ void arrayReverse(float* A, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N / 2) {
        float temp = A[i];
        A[i] = A[N - i - 1];
        A[N - i - 1] = temp;
    }
}

void runReverseArray(int N){
    size_t size = N * sizeof(float);

    // Allocate host memory
    float* h_A = new float[N];

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float* d_A;
    cudaCheckError(cudaMalloc((void**)&d_A, size));

    // Copy data to device
    cudaCheckError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

    // Setup execution parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N / 2 + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    arrayReverse<<<blocksPerGrid, threadsPerBlock>>>(d_A, N);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    // Copy result back to host
    cudaCheckError(cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost));

    // Verify the result
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_A[i] != static_cast<float>(N - i - 1)) {
            success = false;
            std::cerr << "Mismatch at index " << i << ": " << h_A[i]
                      << " != " << N - i - 1 << std::endl;
            break;
        }
    }

    if (success) {
        std::cout << "Array reversal successful." << std::endl;
    } else {
        std::cout << "Array reversal failed." << std::endl;
    }

    // Cleanup
    cudaFree(d_A);
    delete[] h_A;
}

int main() {
    int N = 1 << 20; // Example size: 1 million elements
    runReverseArray(N);

    return 0;
}

