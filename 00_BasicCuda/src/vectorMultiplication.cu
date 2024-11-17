#include <iostream>
#include <cuda_runtime.h>


// kernel for vector Addition
__global__ void vectorMul(const float* A, const float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] * B[i];
    }
}

void runVectorMul(int N){
    size_t size = N * sizeof(float);

    // Allocate host memory
    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];

    // Initialize host arrays
    for (int i = 0; i< N; i++){
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A , size);
    cudaMalloc((void**)&d_B , size);
    cudaMalloc((void**)&d_C , size);

    // copy data from host to device
    cudaMemcpy( d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy( d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock -1)/threadsPerBlock;
    vectorMul<<<blocksPerGrid, threadsPerBlock>>>(d_A,d_B,d_C,N);

    // Copy results from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify results
    bool passed = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] * h_B[i];
        float diff = fabs(h_C[i] - expected);
        // Use a relative error threshold for larger numbers
        float tolerance = 1e-5 * fabs(expected) + 1e-5;
        
        if (diff > tolerance) {
            std::cerr << "Result verification failed at element " << i << "!\n";
            std::cerr << "Expected: " << expected << ", Got: " << h_C[i] << "\n";
            passed = false;
            break;
        }
    }
    
    // Print some sample results
    std::cout << "\nSample Results (first 5 elements):\n";
    for (int i = 0; i < 5 && i < N; i++) {
        std::cout << h_A[i] << " * " << h_B[i] << " = " << h_C[i] << "\n";
    }
    
    if (passed) {
        std::cout << "\nTest PASSED\n";
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

}

int main(){
    int N = 1 << 20; 
    printf("Vector addition example with %d elements\n", N);
    runVectorMul(N);
    return 0;
}