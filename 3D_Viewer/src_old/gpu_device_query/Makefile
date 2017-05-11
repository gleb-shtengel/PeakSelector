# Define where things are:
CUDA_INSTALL_PATH=/usr/local/cuda
IDLDIR=/usr/local/rsi/idl_6.3
CUDA_SDK=/usr/local/NVIDIA_CUDA_SDK
INCL=-I $(CUDA_INSTALL_PATH)/include -I $(CUDA_SDK)/common/inc
IDLINC=-I $(IDLDIR)/external/include
LIBS=-L $(CUDA_INSTALL_PATH)/lib
OPTS=--machine 64 -O3

dlm : gpu_device_query.c
	$(CC) -Wall $(OPTS) $(IDLINC) $(INCL) $(LIBS) -shared -fPIC -O3 -o gpu_device_query.so gpu_device_query.c -lcudart $(LIBS)
	grep -e '^//dlm:' $< | cut -c 8- > gpu_device_query.dlm

