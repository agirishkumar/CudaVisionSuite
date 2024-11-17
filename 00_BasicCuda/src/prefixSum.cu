#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code)
                  << " " << file << " " << line << std::endl;
        exit(code);
    }
}

__global__ void prefixSumKernel(float *d_out, const float *d_in, int N) {
    extern __shared__ float temp[]; // Shared memory
    int tid = threadIdx.x;
    int offset = 1;

    // Load input into shared memory
    int ai = tid;
    int bi = tid + (N / 2);
    temp[ai] = d_in[ai];
    temp[bi] = d_in[bi];

    // Up-Sweep (Reduction)
    for (int d = N >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    // Down-Sweep
    if (tid == 0) {
        temp[N - 1] = 0;
    }

    for (int d = 1; d < N; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // Write results to device memory
    d_out[ai] = temp[ai];
    d_out[bi] = temp[bi];
}

void runPrefixSum(int N) {
    size_t size = N * sizeof(float);

    // Allocate host memory
    float* h_in = new float[N];
    float* h_out = new float[N];

    // Initialize input array
    for (int i = 0; i < N; ++i) {
        h_in[i] = 1.0f; // For simplicity
    }

    // Allocate device memory
    float *d_in, *d_out;
    cudaCheckError(cudaMalloc((void**)&d_in, size));
    cudaCheckError(cudaMalloc((void**)&d_out, size));

    // Copy data to device
    cudaCheckError(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = N / 2;
    int blocksPerGrid = 1;
    size_t sharedMemSize = N * sizeof(float);

    prefixSumKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_out, d_in, N);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    // Copy result back to host
    cudaCheckError(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    // Verify result
    bool success = true;
    for (int i = 0; i < N; ++i) {
        float expected = i * 1.0f; // Since all inputs are 1.0f
        if (fabs(h_out[i] - expected) > 1e-5) {
            success = false;
            std::cerr << "Mismatch at index " << i << ": " << h_out[i]
                      << " != " << expected << std::endl;
            break;
        }
    }

    if (success) {
        std::cout << "Prefix sum computed successfully." << std::endl;
    } else {
        std::cout << "Prefix sum computation failed." << std::endl;
    }

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;
}

int main() {
    int N = 1 << 10; // Must be a power of two and fit into shared memory
    runPrefixSum(N);
    return 0;
}
