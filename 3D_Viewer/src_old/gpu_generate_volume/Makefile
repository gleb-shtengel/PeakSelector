# Define where things are:
CUDA_INSTALL_PATH=/usr/local/cuda
IDLDIR=/usr/local/rsi/idl_6.3
CUDA_SDK=/usr/local/NVIDIA_CUDA_SDK
INCL=-I $(CUDA_INSTALL_PATH)/include -I $(CUDA_SDK)/common/inc
IDLINC=-I $(IDLDIR)/external/include
LIBS=-L $(CUDA_INSTALL_PATH)/lib
OPTS=--machine 64 --use_fast_math -O3

dlm : gpu_generate_volume.c generate_volume.cu
	nvcc $(INCL) $(OPTS) -c -shared -Xcompiler "-fPIC" generate_volume.cu
	$(CC) -Wall $(IDLINC) $(LIBS) -shared -fPIC -O3 -o gpu_generate_volume.so gpu_generate_volume.c generate_volume.o -lcudart
	grep -e '^//dlm:' $< | cut -c 8- > gpu_generate_volume.dlm
	rm generate_volume.o

emu : gpu_generate_volume.c generate_volume.cu
	nvcc $(INCL) $(OPTS) -deviceemu -c -shared -Xcompiler "-fPIC" generate_volume.cu
	$(CC) -Wall $(IDLINC) $(LIBS) -shared -fPIC -O3 -o gpu_generate_volume.so gpu_generate_volume.c generate_volume.o -lcudart
	grep -e '^//dlm:' $< | cut -c 8- > gpu_generate_volume.dlm
	rm generate_volume.o
