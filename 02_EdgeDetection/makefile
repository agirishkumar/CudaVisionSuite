NVCC := nvcc
NVCC_FLAGS := -O3 -g -G -diag-suppress 611

# OpenCV flags (using pkg-config)
OPENCV_INCLUDES := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS := $(shell pkg-config --libs opencv4)

# CUDA includes
CUDA_INCLUDES := -I/usr/local/cuda/include -I/usr/include/opencv4

# Combined includes
INCLUDES := $(CUDA_INCLUDES) $(OPENCV_INCLUDES)

# Linker options
CUDA_LIBS := -L/usr/local/cuda/lib64
TIFF_LIBS := -L/usr/lib/x86_64-linux-gnu
LINKER_FLAGS := -Xlinker "-rpath,/usr/lib/x86_64-linux-gnu"

# Combined libraries (order matters)
LIBS := $(CUDA_LIBS) $(TIFF_LIBS) -ltiff $(OPENCV_LIBS)

# Directories
SRC_DIR := src
BUILD_DIR := build
BIN_DIR := bin

# Find all CUDA source files
SRCS := $(wildcard $(SRC_DIR)/*.cu)
OBJS := $(SRCS:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)
BINS := $(SRCS:$(SRC_DIR)/%.cu=$(BIN_DIR)/%)

# Create necessary directories
$(shell mkdir -p $(BUILD_DIR))
$(shell mkdir -p $(BIN_DIR))

# Default target
all: $(BINS)

# Compile CUDA source files into object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Link object files into executables
$(BIN_DIR)/%: $(BUILD_DIR)/%.o
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LINKER_FLAGS) $(LIBS)

# Debug target to print variables
print-%:
	@echo '$*=$($*)'

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)