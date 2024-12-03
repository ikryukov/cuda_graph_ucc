# Compiler and flags
NVCC = nvcc
MPICXX = mpicxx
NVCCFLAGS = -O2 -arch=sm_80
CXXFLAGS = -Wall -O2 -std=c++11

# Paths
CUDA_INC = /usr/local/cuda-12.4/include
CUDA_LIB = /usr/local/cuda-12.4/lib64
UCC_INC = /home/ikryukov/work/ucc/install/include
UCC_LIB = /home/ikryukov/work/ucc/install/lib

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
