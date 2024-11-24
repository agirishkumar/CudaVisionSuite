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

// CUDA kernel for Prewitt edge detection
__global__ void prewittEdgeDetection(
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
        
        // Prewitt operators
        const int Px[3][3] = {
            {-1, 0, 1},
            {-1, 0, 1},
            {-1, 0, 1}
        };
        
        const int Py[3][3] = {
            {-1, -1, -1},
            { 0,  0,  0},
            { 1,  1,  1}
        };
        
        // Calculate gradients
        float gradX = 0.0f;
        float gradY = 0.0f;
        
        #pragma unroll
        for (int i = -1; i <= 1; i++) {
            #pragma unroll
            for (int j = -1; j <= 1; j++) {
                int pixel = input[(y + i) * width + (x + j)];
                gradX += pixel * Px[i + 1][j + 1];
                gradY += pixel * Py[i + 1][j + 1];
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

// Optional: Kernel for computing gradient direction
__global__ void prewittGradientDirection(
    const unsigned char* input,
    float* direction,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width - 1 && y < height - 1 && x > 0 && y > 0) {
        int idx = y * width + x;
        
        // Prewitt operators
        const int Px[3][3] = {
            {-1, 0, 1},
            {-1, 0, 1},
            {-1, 0, 1}
        };
        
        const int Py[3][3] = {
            {-1, -1, -1},
            { 0,  0,  0},
            { 1,  1,  1}
        };
        
        float gradX = 0.0f;
        float gradY = 0.0f;
        
        #pragma unroll
        for (int i = -1; i <= 1; i++) {
            #pragma unroll
            for (int j = -1; j <= 1; j++) {
                int pixel = input[(y + i) * width + (x + j)];
                gradX += pixel * Px[i + 1][j + 1];
                gradY += pixel * Py[i + 1][j + 1];
            }
        }
        
        // Calculate direction in radians
        direction[idx] = atan2f(gradY, gradX);
    }
}

class PrewittEdgeDetector {
private:
    int width;
    int height;
    unsigned char *d_input;
    unsigned char *d_output;
    float *d_direction;  // Optional: for gradient direction
    bool computeDirection;
    
public:
    PrewittEdgeDetector(int w, int h, bool withDirection = false) 
        : width(w), height(h), computeDirection(withDirection) {
        // Allocate device memory
        cudaCheckError(cudaMalloc(&d_input, width * height));
        cudaCheckError(cudaMalloc(&d_output, width * height));
        
        if (computeDirection) {
            cudaCheckError(cudaMalloc(&d_direction, width * height * sizeof(float)));
        }
    }
    
    cv::Mat detect(const cv::Mat& input, cv::Mat* gradientDirection = nullptr) {
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
        
        // Launch edge detection kernel
        prewittEdgeDetection<<<gridSize, blockSize>>>(
            d_input, d_output, width, height);
        cudaCheckError(cudaGetLastError());
        
        // If gradient direction is requested
        if (computeDirection && gradientDirection != nullptr) {
            prewittGradientDirection<<<gridSize, blockSize>>>(
                d_input, d_direction, width, height);
            cudaCheckError(cudaGetLastError());
            
            // Create output Mat for direction
            *gradientDirection = cv::Mat(height, width, CV_32F);
            cudaCheckError(cudaMemcpy(gradientDirection->data, d_direction,
                                    width * height * sizeof(float),
                                    cudaMemcpyDeviceToHost));
        }
        
        // Create output Mat for magnitude
        cv::Mat output(height, width, CV_8UC1);
        
        // Copy result back to host
        cudaCheckError(cudaMemcpy(output.data, d_output, width * height,
                                cudaMemcpyDeviceToHost));
        
        return output;
    }
    
    ~PrewittEdgeDetector() {
        cudaFree(d_input);
        cudaFree(d_output);
        if (computeDirection) {
            cudaFree(d_direction);
        }
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
    
    // Create edge detector (with optional gradient direction computation)
    PrewittEdgeDetector detector(width, height, true);
    
    cv::Mat frame, edges, direction;
    while (true) {
        // Capture frame
        cap >> frame;
        if (frame.empty()) break;
        
        // Detect edges and optionally get gradient direction
        edges = detector.detect(frame, &direction);
        
        // Display results
        cv::imshow("Original", frame);
        cv::imshow("Edges", edges);
        
        // Optionally visualize gradient direction
        if (!direction.empty()) {
            cv::Mat direction_vis;
            // Convert radians to degrees and normalize for visualization
            direction = direction * 180.0 / CV_PI;
            cv::normalize(direction, direction_vis, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            cv::applyColorMap(direction_vis, direction_vis, cv::COLORMAP_JET);
            cv::imshow("Gradient Direction", direction_vis);
        }
        
        // Exit on 'q' press
        if (cv::waitKey(1) == 'q') break;
    }
    
    cap.release();
    cv::destroyAllWindows();
    return 0;
}