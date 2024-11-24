# Compiler and flags
NVCC := nvcc
NVCC_FLAGS := -O3 -g -G -diag-suppress 20012
OPENGL_FLAGS := -lGL -lGLEW -lglfw
INCLUDE_FLAGS := -I/usr/include/glm

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
all: check_dependencies $(BINS)

# Check for required dependencies
check_dependencies:
	@echo "Checking dependencies..."
	@pkg-config --exists glfw3 || (echo "Error: GLFW3 not found" && exit 1)
	@pkg-config --exists glew || (echo "Error: GLEW not found" && exit 1)
	@test -d /usr/include/glm || (echo "Error: GLM not found" && exit 1)

# Compile CUDA source files into object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) -c $< -o $@

# Link object files into executables
$(BIN_DIR)/%: $(BUILD_DIR)/%.o
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(OPENGL_FLAGS)

# Clean build files
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Install dependencies (Ubuntu/Debian)
.PHONY: install_deps
install_deps:
	sudo apt-get update
	sudo apt-get install -y libglfw3-dev libglew-dev libglm-dev

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all           - Build all executables (default)"
	@echo "  clean         - Remove build and bin directories"
	@echo "  install_deps  - Install required dependencies (Ubuntu/Debian)"
	@echo "  help          - Show this help message"