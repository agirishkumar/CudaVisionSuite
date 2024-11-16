# CudaVisionSuite
Repository encompassing a series of computer vision projects implemented in C and CUDA

## CudaVisionSuite Learning Roadmap
1. CUDA Fundamentals (00_BasicCuda)

Memory Management

- Host & Device memory allocation
- Memory transfer operations
- Unified Memory
- Pinned memory


Kernel Programming

- Thread/Block/Grid organization
- Shared memory usage
- Atomic operations
- Warp-level primitives


Performance Optimization

- Memory coalescing
- Bank conflicts
- Occupancy optimization
- Stream processing



2. Image Processing Fundamentals (01_ImageProcessing)

Basic Operations

- Image loading/saving
- Color space conversions (RGB ↔ Grayscale)
- Pixel-wise operations
- Histogram computation


Transformations

- Brightness/Contrast adjustment
- Gamma correction
- Image resizing
- Rotation and affine transforms



3. Edge Detection & Feature Extraction (02_EdgeDetection)

Gradient Operations

- Sobel operator
- Prewitt operator
- Roberts cross operator


Advanced Edge Detection

- Canny edge detector
- Laplacian of Gaussian
- Hough transform


Feature Detection

- SIFT implementation
- SURF implementation
- Corner detection (Harris)



4. Image Filtering & Convolution (03_ImageFiltering)

Basic Filters

- Box blur
- Gaussian blur
- Median filter
- Bilateral filter


Advanced Filtering

- FFT-based filtering
- Wiener filter
- Non-local means denoising



5. Linear Algebra Operations (04_LinearAlgebra)

Basic Operations

- Matrix multiplication
- Matrix transposition
- Vector operations


Advanced Operations

- SVD decomposition
- Eigenvalue computation
- Matrix inversion


Sparse Operations

- Sparse matrix formats
- SpMV operations
- Sparse solvers



6. Computer Vision Algorithms (05_ComputerVision)

Segmentation

- Thresholding techniques
- Region growing
- Watershed algorithm


Object Detection

- Sliding window
- Integral images
- HOG features


3D Vision

- Depth estimation
- Point cloud processing
- Camera calibration



7. Deep Learning Basics (06_DeepLearning)

Neural Network Components

- Layer implementations
- Activation functions
- Loss functions


Basic Architectures

- Fully connected networks
- Basic CNNs
- Pooling layers


Training Infrastructure

- Backpropagation
- Gradient descent optimization
- Mini-batch processing



8. Advanced Deep Learning (07_AdvancedDL)

Modern Architectures

- ResNet implementation
- Transformer blocks
- Attention mechanisms


Training Optimizations

- Mixed precision training
- Multi-GPU training
- Gradient accumulation


Vision Transformers

- ViT implementation
- DETR architecture
- Vision-language models



9. LLM Components (08_LLM)

Core Components

- Tokenization
- Positional encoding
- Multi-head attention


Training & Inference

- KV cache implementation
- Beam search
- Top-k/Top-p sampling


Optimization

- Model parallelism
- Quantization
- Efficient inference