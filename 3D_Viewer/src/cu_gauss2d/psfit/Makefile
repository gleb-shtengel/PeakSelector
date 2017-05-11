# Define where things are:
CUDA_INSTALL_PATH=/usr/local/cuda
IDLDIR=/usr/local/rsi/idl_6.3
CUDA_SDK=/usr/local/NVIDIA_CUDA_SDK
INCL=-I $(CUDA_INSTALL_PATH)/include -I $(CUDA_SDK)/common/inc
IDLINC=-I $(IDLDIR)/external/include
LIBS=-L $(CUDA_INSTALL_PATH)/lib
OPTS=--machine 64 --use_fast_math -O3


dlm : particle_swarm_fit.c psfit.cu
	nvcc $(INCL) $(OPTS) -c -shared -Xcompiler "-fPIC" psfit.cu
	$(CC) -Wall $(IDLINC) $(LIBS) -shared -fPIC -O3 -o particle_swarm_fit.so particle_swarm_fit.c psfit.o -lcudart
	grep -e '^//dlm:' $< | cut -c 8- > particle_swarm_fit.dlm
	rm psfit.o

emu : particle_swarm_fit.c psfit.cu
	nvcc $(INCL) $(OPTS) -deviceemu -c -shared -Xcompiler "-fPIC" psfit.cu
	$(CC) -Wall $(IDLINC) $(LIBS) -shared -fPIC -O3 -o particle_swarm_fit.so particle_swarm_fit.c psfit.o -lcudart
	grep -e '^//dlm:' $< | cut -c 8- > particle_swarm_fit.dlm
	rm psfit.o

debug : psfit.cu
	nvcc -g $(INCL) $(OPTS) -c psfit.cu -deviceemu
	$(CC) -Wall $(IDLINC) $(LIBS) -shared -o particle_swarm_fit.so particle_swarm_fit.c psfit.o 

