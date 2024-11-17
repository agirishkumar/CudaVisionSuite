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

__global__ void normalizeVector(float* A, double magnitude, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        A[i] = A[i] / static_cast<float>(magnitude);
    }
}

__global__ void vectorDot(const float* A, float* partialSums, int N) {
    __shared__ float sharedMem[256];
    
    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Initialize shared memory
    sharedMem[tid] = 0.0f;
    
    // Calculate partial dot product for this thread
    if (i < N) {
        sharedMem[tid] = A[i] * A[i];
    }
    
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedMem[tid] += sharedMem[tid + stride];
        }
        __syncthreads();
    }
    
    // Write the result of this block to the partial sums array
    if (tid == 0) {
        partialSums[blockIdx.x] = sharedMem[0];
    }
}


void runVectorNormalization(int N) {
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    float* h_A = new float[N];
    
    // Initialize with smaller numbers to reduce floating-point errors
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);  
    }
    
    // Calculate CPU result
    double cpu_result = 0.0;  // Use double for higher precision
    for (int i = 0; i < N; i++) {
        cpu_result += static_cast<double>(h_A[i]) * static_cast<double>(h_A[i]);
    }
    
    // Allocate device memory
    float *d_A;
    cudaCheckError(cudaMalloc((void**)&d_A, size));
    
    // Copy data to device
    cudaCheckError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    
    // Setup execution parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Allocate memory for partial sums
    float* d_partialSums;
    float* h_partialSums = new float[blocksPerGrid];
    cudaCheckError(cudaMalloc((void**)&d_partialSums, blocksPerGrid * sizeof(float)));
    
    // Launch dot product kernel
    vectorDot<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_partialSums, N);
    cudaCheckError(cudaGetLastError());
    
    // Get partial results
    cudaCheckError(cudaMemcpy(h_partialSums, d_partialSums, 
                             blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Sum partial results using double precision
    double gpu_result = 0.0;
    for (int i = 0; i < blocksPerGrid; i++) {
        gpu_result += static_cast<double>(h_partialSums[i]);
    }
    
    // Calculate magnitude
    double magnitude = sqrt(gpu_result);
    
    // Normalize the vector
    normalizeVector<<<blocksPerGrid, threadsPerBlock>>>(d_A, magnitude, N);
    cudaCheckError(cudaGetLastError());
    
    // Copy the normalized vector back to host
    cudaCheckError(cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost));
    
    // Verify normalization
    double cpu_normalized_result = 0.0;
    for (int i = 0; i < N; i++) {
        cpu_normalized_result += static_cast<double>(h_A[i]) * static_cast<double>(h_A[i]);
    }
    
    // Print first few elements
    std::cout << "\nFirst 5 elements of normalized vector:\n";
    for (int i = 0; i < 5; i++) {
        std::cout << "A[" << i << "] = " << h_A[i] << "\n";
    }
    
    // Print results
    std::cout << "\nResults:\n";
    std::cout << "Original GPU Dot Product: " << gpu_result << "\n";
    std::cout << "Original CPU Dot Product: " << cpu_result << "\n";
    std::cout << "Magnitude: " << magnitude << "\n";
    std::cout << "Dot Product of Normalized Vector (CPU): " << cpu_normalized_result << "\n";
    
    // Use relative error for testing
    double relative_error = fabs(cpu_normalized_result - 1.0);
    if (relative_error < 1e-5) {
        std::cout << "Normalization Test PASSED\n";
    } else {
        std::cout << "Normalization Test FAILED\n";
    }
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_partialSums);
    delete[] h_A;
    delete[] h_partialSums;
}

int main() {
    int N = 1 << 20;
    std::cout << "Vector normalization example with " << N << " elements\n";
    runVectorNormalization(N);
    return 0;
}
