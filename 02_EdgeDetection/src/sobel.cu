#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// Error checking macro
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", 
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// CUDA kernel for Sobel edge detection
__global__ void sobelEdgeDetection(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width - 1 && y < height - 1 && x > 0 && y > 0) {
        // Get the center pixel index
        int idx = y * width + x;
        
        // Sobel operators
        const int Gx[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };
        
        const int Gy[3][3] = {
            {-1, -2, -1},
            { 0,  0,  0},
            { 1,  2,  1}
        };
        
        // Calculate gradients
        float gradX = 0.0f;
        float gradY = 0.0f;
        
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int pixel = input[(y + i) * width + (x + j)];
                gradX += pixel * Gx[i + 1][j + 1];
                gradY += pixel * Gy[i + 1][j + 1];
            }
        }
        
        // Calculate magnitude
        float magnitude = sqrtf(gradX * gradX + gradY * gradY);
        
        // Normalize to 0-255
        magnitude = min(255.0f, max(0.0f, magnitude));
        
        // Store result
        output[idx] = static_cast<unsigned char>(magnitude);
    }
}

class EdgeDetector {
private:
    int width;
    int height;
    unsigned char *d_input;
    unsigned char *d_output;
    
public:
    EdgeDetector(int w, int h) : width(w), height(h) {
        // Allocate device memory
        cudaCheckError(cudaMalloc(&d_input, width * height));
        cudaCheckError(cudaMalloc(&d_output, width * height));
    }
    
    cv::Mat detect(const cv::Mat& input) {
        // Convert input to grayscale if necessary
        cv::Mat gray;
        if (input.channels() == 3) {
            cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = input;
        }
        
        // Copy input to device
        cudaCheckError(cudaMemcpy(d_input, gray.data, width * height,
                                cudaMemcpyHostToDevice));
        
        // Setup grid and block dimensions
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                     (height + blockSize.y - 1) / blockSize.y);
        
        // Launch kernel
        sobelEdgeDetection<<<gridSize, blockSize>>>(
            d_input, d_output, width, height);
        cudaCheckError(cudaGetLastError());
        
        // Create output Mat
        cv::Mat output(height, width, CV_8UC1);
        
        // Copy result back to host
        cudaCheckError(cudaMemcpy(output.data, d_output, width * height,
                                cudaMemcpyDeviceToHost));
        
        return output;
    }
    
    ~EdgeDetector() {
        cudaFree(d_input);
        cudaFree(d_output);
    }
};

int main() {
    // Open video capture
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return -1;
    }
    
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    
    // Create edge detector
    EdgeDetector detector(width, height);
    
    cv::Mat frame, edges;
    while (true) {
        // Capture frame
        cap >> frame;
        if (frame.empty()) break;
        
        // Detect edges
        edges = detector.detect(frame);
        
        // Display results
        cv::imshow("Original", frame);
        cv::imshow("Edges", edges);
        
        // Exit on 'q' press
        if (cv::waitKey(1) == 'q') break;
    }
    
    cap.release();
    cv::destroyAllWindows();
    return 0;
}