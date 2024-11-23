# include <iostream>
# include <cuda_runtime.h>

// Error checking macro
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", 
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// Kernel for in-place matrix transposition
__global__ void transposeInPlace(float* matrix, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N && row < col) {
        // Swap elements above the diagonal
        int idx1 = row * N + col;
        int idx2 = col * N + row;
        float temp = matrix[idx1];
        matrix[idx1] = matrix[idx2];
        matrix[idx2] = temp;
    }
}

void runMatrixTranspose(int N){
    size_t size = N*N*sizeof(float);

    // Allocate and initialize the host matrix
    float* h_matrix = new float[N * N];
    for (int i = 0; i < N * N; i++) {
        h_matrix[i] = static_cast<float>(i + 1);  // Initialize with some values
    }

    // Print original matrix
    std::cout << "\nOriginal Matrix:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_matrix[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    // Allocate device memory
    float* d_matrix;
    cudaCheckError(cudaMalloc((void**)&d_matrix, size));

    // Copy matrix to device
    cudaCheckError(cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice));

    // Define thread block and grid sizes
    dim3 threadsPerBlock(16, 16);  // Block size: 16x16
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);  // Grid size

    // Launch the kernel
    transposeInPlace<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, N);
    cudaCheckError(cudaGetLastError());

    // Copy the transposed matrix back to host
    cudaCheckError(cudaMemcpy(h_matrix, d_matrix, size, cudaMemcpyDeviceToHost));

    // Print the transposed matrix
    std::cout << "\nTransposed Matrix:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_matrix[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    // Cleanup
    cudaFree(d_matrix);
    delete[] h_matrix;
}



int main(){
    int N = 4;
    std::cout << "Matrix Transpose example with " << N << "x" << "matrix: \n";
    runMatrixTranspose(N);
    return 0;
}