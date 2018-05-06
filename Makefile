CUDA_HOME=/usr/local/cuda-8.0
NVCC=$(CUDA_HOME)/bin/nvcc

CUDA_INCLUDEPATH=$(CUDA_HOME)/include
NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -m64 -Wno-deprecated-gpu-targets -std=c++11
GCC_OPTS=-O3 -m64 -std=c++11


all: cpu gpu cocomo_cpu cocomo_gpu Makefile
	echo "Build complete"

cpu: start.cpp
	g++ $(GCC_OPTS) -o cpu start.cpp
	
gpu: gpu.cu
	$(NVCC) -o gpu gpu.cu $(NVCC_OPTS)

cocomo_cpu: cocomo_cpu.cpp
	g++ $(GCC_OPTS) -o cocomo_cpu cocomo_cpu.cpp

cocomo_gpu: cocomo_gpu.cu
	$(NVCC) -o cocomo_gpu cocomo_gpu.cu $(NVCC_OPTS)

clean: 
	rm -f *.o gpu cpu cocomo_gpu cocomo_cpu
	find . -type f -name '*.exr' | grep -v memorial | xargs rm -f