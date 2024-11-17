#include <iostream>
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

// Kernel for vector dot product using parallel reduction
__global__ void vectorDot(const float* A, const float* B, float* partialSums, int N) {
    __shared__ float sharedMem[256];
    
    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Initialize shared memory
    sharedMem[tid] = 0.0f;
    
    // Calculate partial dot product for this thread
    if (i < N) {
        sharedMem[tid] = A[i] * B[i];
    }
    
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedMem[tid] += sharedMem[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partialSums[blockIdx.x] = sharedMem[0];
    }
}

void runVectorDot(int N) {
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    float* h_A = new float[N];
    float* h_B = new float[N];
    
    // Initialize with smaller numbers to reduce floating-point errors
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i % 100) / 100.0f;  // Values between 0 and 1
        h_B[i] = static_cast<float>((i % 100) * 2) / 100.0f;
    }
    
    // Calculate CPU result
    double cpu_result = 0.0;  // Use double for higher precision
    for (int i = 0; i < N; i++) {
        cpu_result += static_cast<double>(h_A[i]) * static_cast<double>(h_B[i]);
    }
    
    // Allocate device memory
    float *d_A, *d_B;
    cudaCheckError(cudaMalloc((void**)&d_A, size));
    cudaCheckError(cudaMalloc((void**)&d_B, size));
    
    // Copy data to device
    cudaCheckError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    // Setup execution parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Allocate memory for partial sums
    float* d_partialSums;
    float* h_partialSums = new float[blocksPerGrid];
    cudaCheckError(cudaMalloc((void**)&d_partialSums, blocksPerGrid * sizeof(float)));
    
    // Launch kernel
    vectorDot<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_partialSums, N);
    cudaCheckError(cudaGetLastError());
    
    // Get partial results
    cudaCheckError(cudaMemcpy(h_partialSums, d_partialSums, 
                             blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Sum partial results using double precision
    double gpu_result = 0.0;
    for (int i = 0; i < blocksPerGrid; i++) {
        gpu_result += static_cast<double>(h_partialSums[i]);
    }
    
    // Print first few elements and their products
    std::cout << "\nFirst 5 elements:\n";
    for (int i = 0; i < 5; i++) {
        std::cout << "A[" << i << "] = " << h_A[i] 
                 << ", B[" << i << "] = " << h_B[i] 
                 << ", Product = " << (h_A[i] * h_B[i]) << "\n";
    }
    
    // Calculate relative error
    double relative_error = fabs(gpu_result - cpu_result) / cpu_result;
    std::cout << "\nResults:\n";
    std::cout << "GPU Dot Product: " << gpu_result << "\n";
    std::cout << "CPU Dot Product: " << cpu_result << "\n";
    std::cout << "Relative Error: " << relative_error << "\n";
    
    // Use relative error for testing
    if (relative_error < 1e-5) {
        std::cout << "Test PASSED\n";
    } else {
        std::cout << "Test FAILED\n";
    }
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_partialSums);
    delete[] h_A;
    delete[] h_B;
    delete[] h_partialSums;
}

int main() {
    int N = 1 << 20;
    std::cout << "Vector dot product example with " << N << " elements\n";
    runVectorDot(N);
    return 0;
}