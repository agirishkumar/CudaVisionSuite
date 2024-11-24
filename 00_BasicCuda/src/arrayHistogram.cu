#include <iostream>
#include <cuda_runtime.h>

#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", 
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// Kernel to compute the histogram using shared memory and atomic operations
__global__ void computeHistogram(const int* data, int* histogram, int dataSize, int binCount) {
    extern __shared__ int localHist[];  // Shared memory for partial histograms

    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Initialize shared memory
    if (tid < binCount) {
        localHist[tid] = 0;
    }
    __syncthreads();

    // Compute partial histogram in shared memory
    if (idx < dataSize) {
        atomicAdd(&localHist[data[idx]], 1);
    }
    __syncthreads();

    // Combine partial histograms into global memory
    if (tid < binCount) {
        atomicAdd(&histogram[tid], localHist[tid]);
    }
}

void runHistogramComputation(int dataSize, int binCount) {
    size_t dataBytes = dataSize * sizeof(int);
    size_t histBytes = binCount * sizeof(int);

    // Allocate and initialize host memory
    int* h_data = new int[dataSize];
    int* h_histogram = new int[binCount]();
    
    // Initialize data with random values in the range [0, binCount-1]
    for (int i = 0; i < dataSize; i++) {
        h_data[i] = rand() % binCount;
    }

    // Allocate device memory
    int *d_data, *d_histogram;
    cudaCheckError(cudaMalloc((void**)&d_data, dataBytes));
    cudaCheckError(cudaMalloc((void**)&d_histogram, histBytes));

    // Copy data to device
    cudaCheckError(cudaMemcpy(d_data, h_data, dataBytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemset(d_histogram, 0, histBytes));  // Initialize histogram on device

    // Define execution parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (dataSize + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    computeHistogram<<<blocksPerGrid, threadsPerBlock, binCount * sizeof(int)>>>(d_data, d_histogram, dataSize, binCount);
    cudaCheckError(cudaGetLastError());

    // Copy histogram back to host
    cudaCheckError(cudaMemcpy(h_histogram, d_histogram, histBytes, cudaMemcpyDeviceToHost));

    // Print a portion of the histogram
    std::cout << "\nHistogram:\n";
    for (int i = 0; i < binCount; i++) {
        std::cout << "Bin " << i << ": " << h_histogram[i] << "\n";
    }

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_histogram);
    delete[] h_data;
    delete[] h_histogram;
}

int main() {
    int dataSize = 1 << 20;  // 1M elements
    int binCount = 256;      // Histogram bins (e.g., for grayscale intensities)

    std::cout << "Histogram computation example with " << dataSize << " data elements and " << binCount << " bins\n";
    runHistogramComputation(dataSize, binCount);
    return 0;
}
