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

// Gaussian smoothing kernel
__global__ void gaussianBlur(
    const unsigned char* input,
    float* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width - 2 && y < height - 2 && x > 1 && y > 1) {
        // Gaussian 5x5 kernel
        const float kernel[5][5] = {
            {0.003765, 0.015019, 0.023792, 0.015019, 0.003765},
            {0.015019, 0.059912, 0.094907, 0.059912, 0.015019},
            {0.023792, 0.094907, 0.150342, 0.094907, 0.023792},
            {0.015019, 0.059912, 0.094907, 0.059912, 0.015019},
            {0.003765, 0.015019, 0.023792, 0.015019, 0.003765}
        };
        
        float sum = 0.0f;
        for (int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {
                sum += kernel[i+2][j+2] * input[(y+i) * width + (x+j)];
            }
        }
        output[y * width + x] = sum;
    }
}

// Sobel gradient computation
__global__ void sobelGradient(
    const float* input,
    float* gradientMagnitude,
    float* gradientDirection,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width - 1 && y < height - 1 && x > 0 && y > 0) {
        float gx = 
            input[(y-1)*width + (x+1)] + 2*input[y*width + (x+1)] + input[(y+1)*width + (x+1)] -
            input[(y-1)*width + (x-1)] - 2*input[y*width + (x-1)] - input[(y+1)*width + (x-1)];
            
        float gy = 
            input[(y+1)*width + (x-1)] + 2*input[(y+1)*width + x] + input[(y+1)*width + (x+1)] -
            input[(y-1)*width + (x-1)] - 2*input[(y-1)*width + x] - input[(y-1)*width + (x+1)];
            
        gradientMagnitude[y * width + x] = sqrtf(gx*gx + gy*gy);
        gradientDirection[y * width + x] = atan2f(gy, gx);
    }
}

// Non-maximum suppression
__global__ void nonMaxSuppression(
    const float* magnitude,
    const float* direction,
    unsigned char* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width - 1 && y < height - 1 && x > 0 && y > 0) {
        float angle = direction[y * width + x];
        float mag = magnitude[y * width + x];
        
        // Round angle to nearest 45 degrees
        angle = angle * 180.0f / 3.14159f;
        if (angle < 0) angle += 180;
        
        float mag1 = 0, mag2 = 0;
        
        if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
            mag1 = magnitude[y * width + (x+1)];
            mag2 = magnitude[y * width + (x-1)];
        }
        else if (angle >= 22.5 && angle < 67.5) {
            mag1 = magnitude[(y+1) * width + (x-1)];
            mag2 = magnitude[(y-1) * width + (x+1)];
        }
        else if (angle >= 67.5 && angle < 112.5) {
            mag1 = magnitude[(y+1) * width + x];
            mag2 = magnitude[(y-1) * width + x];
        }
        else if (angle >= 112.5 && angle < 157.5) {
            mag1 = magnitude[(y-1) * width + (x-1)];
            mag2 = magnitude[(y+1) * width + (x+1)];
        }
        
        // Non-maximum suppression
        if (mag >= mag1 && mag >= mag2) {
            output[y * width + x] = (mag > 255) ? 255 : (unsigned char)mag;
        } else {
            output[y * width + x] = 0;
        }
    }
}

// Double thresholding and hysteresis
__global__ void hysteresis(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    unsigned char lowThreshold,
    unsigned char highThreshold
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width - 1 && y < height - 1 && x > 0 && y > 0) {
        int idx = y * width + x;
        unsigned char val = input[idx];
        
        if (val >= highThreshold) {
            output[idx] = 255;
        }
        else if (val < lowThreshold) {
            output[idx] = 0;
        }
        else {
            // Check 8-connected neighbors
            bool hasStrongNeighbor = false;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    if (input[(y+i) * width + (x+j)] >= highThreshold) {
                        hasStrongNeighbor = true;
                        break;
                    }
                }
                if (hasStrongNeighbor) break;
            }
            output[idx] = hasStrongNeighbor ? 255 : 0;
        }
    }
}

class CannyEdgeDetector {
private:
    int width;
    int height;
    unsigned char *d_input;
    float *d_gaussian;
    float *d_magnitude;
    float *d_direction;
    unsigned char *d_suppressed;
    unsigned char *d_output;
    
public:
    CannyEdgeDetector(int w, int h) : width(w), height(h) {
        cudaCheckError(cudaMalloc(&d_input, width * height));
        cudaCheckError(cudaMalloc(&d_gaussian, width * height * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_magnitude, width * height * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_direction, width * height * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_suppressed, width * height));
        cudaCheckError(cudaMalloc(&d_output, width * height));
    }
    
    cv::Mat detect(
        const cv::Mat& input,
        unsigned char lowThreshold = 30,
        unsigned char highThreshold = 90
    ) {
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
        
        // 1. Gaussian blur
        gaussianBlur<<<gridSize, blockSize>>>(
            d_input, d_gaussian, width, height);
        
        // 2. Compute gradients
        sobelGradient<<<gridSize, blockSize>>>(
            d_gaussian, d_magnitude, d_direction, width, height);
        
        // 3. Non-maximum suppression
        nonMaxSuppression<<<gridSize, blockSize>>>(
            d_magnitude, d_direction, d_suppressed, width, height);
        
        // 4. Double thresholding and hysteresis
        hysteresis<<<gridSize, blockSize>>>(
            d_suppressed, d_output, width, height,
            lowThreshold, highThreshold);
        
        // Check for errors
        cudaCheckError(cudaGetLastError());
        
        // Create output Mat
        cv::Mat output(height, width, CV_8UC1);
        
        // Copy result back to host
        cudaCheckError(cudaMemcpy(output.data, d_output, width * height,
                                cudaMemcpyDeviceToHost));
        
        return output;
    }
    
    ~CannyEdgeDetector() {
        cudaFree(d_input);
        cudaFree(d_gaussian);
        cudaFree(d_magnitude);
        cudaFree(d_direction);
        cudaFree(d_suppressed);
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
    CannyEdgeDetector detector(width, height);
    
    cv::Mat frame, edges;
    while (true) {
        // Capture frame
        cap >> frame;
        if (frame.empty()) break;
        
        // Detect edges
        edges = detector.detect(frame, 30, 90);
        
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