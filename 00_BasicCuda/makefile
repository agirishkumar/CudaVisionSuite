NVCC := nvcc
NVCC_FLAGS := -O3 -g -G

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
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Link object files into executables
$(BIN_DIR)/%: $(BUILD_DIR)/%.o
	$(NVCC) $(NVCC_FLAGS) $< -o $@

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
