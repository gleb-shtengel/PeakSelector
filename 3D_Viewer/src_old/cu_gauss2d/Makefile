# Define where things are:
CUDA_INSTALL_PATH=/usr/local/cuda
IDLDIR=/usr/local/itt/idl64
CUDA_SDK=/usr/local/NVIDIA_CUDA_SDK
INCL=-I $(CUDA_INSTALL_PATH)/include -I $(CUDA_SDK)/common/inc
IDLINC=-I $(IDLDIR)/external/include
LIBS=-L $(CUDA_INSTALL_PATH)/lib
OPTS=--machine 64 --use_fast_math -O3 -Xptxas -v

dlm : cu_gauss2d.c gauss2d.cu
	nvcc $(INCL) $(OPTS) -c -shared -Xcompiler "-fPIC" gauss2d.cu
	$(CC) -Wall $(INCL) $(IDLINC) $(LIBS) -shared -fPIC -O3 -o cu_gauss2d.so cu_gauss2d.c gauss2d.o -lcudart
	grep -e '^//dlm:' $< | cut -c 8- > cu_gauss2d.dlm
	rm gauss2d.o gauss2d.linkinfo

emu : cu_gauss2d.c gauss2d.cu
	nvcc $(INCL) $(OPTS) -deviceemu -c -shared -Xcompiler "-fPIC" gauss2d.cu
	$(CC) -Wall $(INCL) $(IDLINC) $(LIBS) -shared -fPIC -O3 -o cu_gauss2d.so cu_gauss2d.c gauss2d.o -lcudart
	grep -e '^//dlm:' $< | cut -c 8- > cu_gauss2d.dlm
	rm gauss2d.o gauss2d.linkinfo
