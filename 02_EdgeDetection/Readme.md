# CUDA-Accelerated Edge Detection with Sobel Operator

## Overview
This project implements real-time edge detection using the Sobel operator, accelerated with CUDA for parallel processing on NVIDIA GPUs. Edge detection is a fundamental image processing technique that identifies boundaries within an image where brightness changes sharply or has discontinuities.

## Edge Detection Fundamentals

### What are Edges?
Edges in images are areas with strong intensity contrasts – a sharp change in intensity from one pixel to another. They often represent:
- Object boundaries
- Material boundaries
- Changes in surface orientation
- Changes in depth
- Changes in scene illumination

### Mathematical Foundation
An edge can be mathematically defined using the gradient of the image intensity function. For a 2D image function f(x,y), its gradient is a vector:

∇f = [∂f/∂x, ∂f/∂y]

The gradient magnitude indicates how quickly the image intensity is changing:

|∇f| = √[(∂f/∂x)² + (∂f/∂y)²]

The gradient direction is:

θ = arctan(∂f/∂y / ∂f/∂x)

## Sobel Edge Detection

### The Sobel Operator
The Sobel operator consists of two 3×3 kernels, one for detecting changes in the vertical direction (Gy) and another for the horizontal direction (Gx):

```
Gx = [
    [-1  0  1]
    [-2  0  2]
    [-1  0  1]
]

Gy = [
    [-1 -2 -1]
    [ 0  0  0]
    [ 1  2  1]
]
```

### How Sobel Works
1. **Gradient Calculation**:
   - The image is convolved with both Gx and Gy kernels
   - At each pixel (i,j), we compute:
     * Gradient in x: Gx = ∑∑ (kernel_x × image_region)
     * Gradient in y: Gy = ∑∑ (kernel_y × image_region)

2. **Magnitude Computation**:
   - For each pixel, compute the gradient magnitude:
   - G = √(Gx² + Gy²)

3. **Optional Direction Calculation**:
   - The gradient direction can be computed as:
   - θ = arctan(Gy/Gx)

### Advantages of Sobel
- Combines Gaussian smoothing and differentiation
- Reduces noise sensitivity
- Provides both edge strength (magnitude) and direction
- Simple but effective
- Computationally efficient

## CUDA Implementation Details

### Parallelization Strategy
Our implementation parallelizes the Sobel operator by:
1. Assigning each pixel to a CUDA thread
2. Using shared memory for efficient kernel access
3. Computing gradients in parallel for the entire image

### Memory Considerations
- Input image is stored in global memory
- Each thread processes one pixel
- Edge computations are independent, allowing for massive parallelism
- The output magnitude is written directly to global memory

### Performance Optimization
The code includes several optimizations:
- Coalesced memory access patterns
- Shared memory usage for kernel coefficients
- Efficient thread block sizing (16x16)
- Asynchronous memory transfers

## Building and Running

### Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit
- OpenCV 4.x
- CMake 3.10+

### Build Instructions
```bash
make
```

### Usage
```bash
./bin/sobel
```
Press 'q' to quit the application.

## Technical Specifications

### Input
- Real-time video feed or image input
- Grayscale or color (automatically converted to grayscale)
- Any resolution (automatically handled)

### Output
- Edge magnitude map
- Real-time visualization
- Pixel values normalized to 0-255 range

### Performance
- Real-time processing capability
- Frame rate dependent on:
  * Input resolution
  * GPU capabilities
  * System configuration

## Mathematical Details

### Discrete Approximation
The Sobel operator approximates the image gradient using discrete differences:

For a pixel f(x,y):

∂f/∂x ≈ (f(x+1,y-1) + 2f(x+1,y) + f(x+1,y+1)) - (f(x-1,y-1) + 2f(x-1,y) + f(x-1,y+1))

∂f/∂y ≈ (f(x-1,y+1) + 2f(x,y+1) + f(x+1,y+1)) - (f(x-1,y-1) + 2f(x,y-1) + f(x+1,y-1))

### Edge Magnitude Normalization
The final edge magnitude is normalized to fit in the 0-255 range:

normalized_magnitude = min(255, max(0, magnitude))

## References
1. Sobel, I., Feldman, G., "A 3x3 Isotropic Gradient Operator for Image Processing"
2. Gonzalez, R. C., & Woods, R. E., "Digital Image Processing"
3. NVIDIA CUDA Programming Guide

# CUDA-Accelerated Prewitt Edge Detection

## Overview
This project implements real-time edge detection using the Prewitt operator, accelerated with CUDA for parallel processing on NVIDIA GPUs. The Prewitt operator is an alternative to the Sobel operator for edge detection, offering different characteristics in terms of noise sensitivity and computational simplicity.

## Prewitt Edge Detection

### What is the Prewitt Operator?
The Prewitt operator, developed by Judith M. S. Prewitt, is a discrete differentiation operator used for edge detection in image processing. It computes an approximation of the gradient of the image intensity function for edge detection.

### Mathematical Foundation
Like other gradient-based edge detectors, the Prewitt operator uses two 3×3 kernels to calculate approximations of the derivatives:

#### Prewitt Kernels
```
Horizontal Gradient (Px):    Vertical Gradient (Py):
[-1  0  1]                  [-1 -1 -1]
[-1  0  1]                  [ 0  0  0]
[-1  0  1]                  [ 1  1  1]
```

For an image I, the gradients are computed as:
- Gx = Px * I (convolution of image with horizontal kernel)
- Gy = Py * I (convolution of image with vertical kernel)

The gradient magnitude is calculated as:
```
G = √(Gx² + Gy²)
```

The gradient direction is calculated as:
```
θ = arctan(Gy/Gx)
```

### Comparison with Sobel
While similar to the Sobel operator, the Prewitt operator has some distinct characteristics:

#### Prewitt vs Sobel Kernels:
```
Prewitt X:    Sobel X:
[-1  0  1]    [-1  0  1]
[-1  0  1]    [-2  0  2]
[-1  0  1]    [-1  0  1]
```

Key differences:
1. Prewitt uses uniform weights (1) for all non-center elements
2. Sobel gives more weight (2) to the central pixels
3. Prewitt is more sensitive to noise but simpler to compute
4. Sobel provides better noise suppression but slightly higher computational cost

## CUDA Implementation Details

### Parallelization Strategy
The implementation parallelizes the Prewitt operator computation by:
1. Assigning each output pixel to a CUDA thread
2. Processing the 3x3 neighborhood in parallel
3. Computing both gradients simultaneously

### Memory Access Pattern
- Global memory reads: 9 pixels per thread (3x3 neighborhood)
- Global memory writes: 1 pixel per thread (output)
- Optimized using `#pragma unroll` for the convolution loops

### Performance Optimizations
1. Kernel Configuration:
   - Thread blocks: 16x16 threads
   - Grid size: Calculated based on image dimensions
2. Memory optimizations:
   - Coalesced memory access
   - Loop unrolling
   - Efficient thread organization

## Features

### Core Functionality
1. Real-time edge detection
2. Gradient magnitude computation
3. Optional gradient direction calculation
4. Support for both grayscale and color inputs

### Additional Features
1. Real-time visualization
2. Gradient direction visualization using color mapping
3. Automatic handling of different input resolutions
4. Support for webcam input

## Implementation Details

### Class Structure
```cpp
class PrewittEdgeDetector {
    // Handles device memory management
    // Provides high-level interface for edge detection
    // Optional gradient direction computation
}
```

### Key Methods
1. `detect()`: Main edge detection method
2. `prewittEdgeDetection`: CUDA kernel for gradient computation
3. `prewittGradientDirection`: CUDA kernel for direction computation

## Performance Characteristics

### Computational Complexity
- Per pixel: 18 multiplications, 18 additions
- Memory operations: 9 reads, 1 write per pixel
- Additional operations for gradient magnitude calculation

### Memory Requirements
- Input image: width * height bytes
- Output image: width * height bytes
- Optional direction map: width * height * sizeof(float) bytes

## Usage

### Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit
- OpenCV 4.x
- C++ compiler with C++11 support

### Building
```bash
make
```

### Running
```bash
./bin/prewitt
```

### Controls
- 'q': Quit the application
- Real-time display of:
  * Original image
  * Edge detection result
  * Gradient direction (optional)

## Technical Details

### Input Specifications
- Format: RGB or Grayscale
- Resolution: Any (automatically handled)
- Source: Camera feed or image file

### Output
1. Edge Magnitude:
   - 8-bit grayscale image
   - Values normalized to [0, 255]

2. Optional Gradient Direction:
   - 32-bit float values
   - Range: [-π, π] radians
   - Visualized using color mapping

## Mathematical Details

### Gradient Computation
For pixel position (x,y):
```
Gx(x,y) = (I(x+1,y-1) + I(x+1,y) + I(x+1,y+1)) - 
          (I(x-1,y-1) + I(x-1,y) + I(x-1,y+1))

Gy(x,y) = (I(x-1,y+1) + I(x,y+1) + I(x+1,y+1)) - 
          (I(x-1,y-1) + I(x,y-1) + I(x+1,y-1))
```

### Error Analysis
The Prewitt operator provides first-order accuracy in gradient estimation, with error term O(h) where h is the pixel spacing.

## References
1. Prewitt, J.M.S., "Object Enhancement and Extraction"
2. Vernon, D., "Machine Vision: Automated Visual Inspection and Robot Vision"
3. NVIDIA CUDA Programming Guide
4. OpenCV Documentation

