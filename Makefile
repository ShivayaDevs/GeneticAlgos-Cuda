CUDA_HOME=/usr/local/cuda-8.0
NVCC=$(CUDA_HOME)/bin/nvcc

CUDA_INCLUDEPATH=$(CUDA_HOME)/include
NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -m64 -Wno-deprecated-gpu-targets -std=c++11
GCC_OPTS=-O3 -m64 -std=c++11


all: cpu gpu Makefile
	echo "Build complete"

cpu: start.cpp
	g++ $(GCC_OPTS) -o cpu start.cpp
	
gpu: gpu.cu
	$(NVCC) -o gpu gpu.cu $(NVCC_OPTS)

clean: 
	rm -f *.o gpu cpu
	find . -type f -name '*.exr' | grep -v memorial | xargs rm -f