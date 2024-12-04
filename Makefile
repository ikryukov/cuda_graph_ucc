# Compiler and flags
NVCC = nvcc
MPICXX = mpicxx
NVCCFLAGS = -O2 -arch=sm_80
CXXFLAGS = -Wall -O2 -std=c++11

# Paths
CUDA_INC = ${CUDA_HOME}/include
CUDA_LIB = ${CUDA_HOME}/lib64
UCC_INC = ${HPCX_UCC_DIR}/include
UCC_LIB = ${HPCX_UCC_DIR}/lib

# Libraries
LIBS = -lcudart -lucc

# Targets
TARGET = cuda_graph_ucc
SRCS = src/main.cu
CU_SRCS = src/kernels.cu

# Rules
all: $(TARGET)

kernels.o: $(CU_SRCS)
	$(NVCC) $(NVCCFLAGS) -I$(CUDA_INC) -c $< -o $@

main.o: $(SRCS)
	$(NVCC) $(NVCCFLAGS) -I$(CUDA_INC) -I$(UCC_INC) -Xcompiler $(CXXFLAGS) -c $< -o $@

$(TARGET): main.o kernels.o
	$(MPICXX) main.o kernels.o -L$(CUDA_LIB) -L$(UCC_LIB) -o $@ $(LIBS)

clean:
	rm -f $(TARGET) main.o kernels.o
