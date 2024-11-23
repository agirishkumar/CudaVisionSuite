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

// Kernel for matrix multiplication
__global__ void matrixMultiply(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row of C
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column of C

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void runMatrixMultiplication(int N) {
    size_t size = N * N * sizeof(float);
    
    // Allocate and initialize host matrices
    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C = new float[N * N];
    
    for (int i = 0; i < N * N; i++) {
        h_A[i] = static_cast<float>(i % 100);  // Initialize A
        h_B[i] = static_cast<float>((i * 2) % 100);  // Initialize B
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaCheckError(cudaMalloc((void**)&d_A, size));
    cudaCheckError(cudaMalloc((void**)&d_B, size));
    cudaCheckError(cudaMalloc((void**)&d_C, size));
    
    // Copy data to device
    cudaCheckError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    // Define thread block and grid sizes
    dim3 threadsPerBlock(16, 16);  // Block size: 16x16
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);  // Grid size: Enough blocks to cover the matrix
    
    // Launch kernel
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaCheckError(cudaGetLastError());
    
    // Copy result back to host
    cudaCheckError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    
    // Print a small portion of the result for verification
    std::cout << "\nFirst 5x5 block of resulting matrix C:\n";
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << "\n";
    }
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

int main() {
    int N = 1024;  // Matrix dimension (N x N)
    std::cout << "Matrix multiplication example with " << N << "x" << N << " matrices\n";
    runMatrixMultiplication(N);
    return 0;
}
