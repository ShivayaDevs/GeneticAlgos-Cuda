CUDA_HOME=/usr/local/cuda-8.0
NVCC=$(CUDA_HOME)/bin/nvcc

CUDA_INCLUDEPATH=$(CUDA_HOME)/include
NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -m64 -Wno-deprecated-gpu-targets
GCC_OPTS=-O3 -m64

major: start.o ga_helper.o Makefile
	$(NVCC) -o major start.o ga_helper.o $(NVCC_OPTS)

start.o: start.cpp ga_helper.h
	g++ -c start.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

ga_helper.o: ga_helper.h ga_helper.cu
	$(NVCC) -c ga_helper.cu $(NVCC_OPTS)

clean: 
	rm -f *.o major
	find . -type f -name '*.exr' | grep -v memorial | xargs rm -f