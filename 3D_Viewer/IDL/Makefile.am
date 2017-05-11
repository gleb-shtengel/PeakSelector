# The following macros establish general defaults. They are overridden
# in the "all" rule as needed to satisfy a given platform's requirements.

# Set variables from configure.ac (or associated m4 files) for
# convenience
SHAREDLIBFLAGS = @SHAREDLIBFLAGS@
C_PIC_FLAG = @C_PIC_FLAG@
RSIIDL_LIBBIN = @RSIIDL_LIBBIN@
CUDA_LIBS = @CUDA_LIBS@
EMULATION_LIBS = @EMULATION_LIBS@
GL_LIBS = @GL_LIBS@

if WITH_IDL

IDL_PATH := "+$(srcdir):$(IDL_PATH)"
IDL_DLM_PATH := "+$(srcdir):$(IDL_PATH)"

# Define flags used for compilation
C_FLAGS	= $(SHAREDLIBFLAGS) $(C_PIC_FLAG) -I$(RSIIDL_DIR)/external/include -I@CUDA_INCDIR@ -O3 

# The following is the default entry point. This section will determine 
# what system we are on and call make again with the correct flags set.

all : 
	@echo "OS type detected: "`uname`
	@case `uname` in \
	   "Darwin") make gpulib ;; \
	   "Linux" ) make gpulib ;; \
	   *) echo "This system is not supported" ;; \
       esac


# The following is the entry point for the recursive build

EXTRA_DIST = gpulib.c gpulib.dlm generate_op.sh *.pro *.sed lib idldoc mgunit unit demos

gpulib:
	gcc $(C_FLAGS)  -Bsymbolic -o gpulib.so -g gpulib.c -I../vectorOp -I../physicsOp $(srcdir)/../vectorOp/gpuVectorOp.o $(srcdir)/../physicsOp/gpuPhysicsOp.o -L$(RSIIDL_LIBBIN) -lidl $(CUDA_LIBS) $(EMULATION_LIBS) $(GL_LIBS)
	sh generate_op.sh
	@date

html-local:
	echo In html target, IDL_PATH is $(IDL_PATH)
	idl gpu_make_docs

clean: 
	rm -f gpulib.o gpulib.so gpulib.sl gpulib.a 

check:
	idl gpu_run_test2
	idl gpu_run_unittests

benchmark:
	 idl -e gpu_benchmark

install-exec-local: gpulib
	mkdir -p $(prefix)/IDL/lib
	cp gpulib.so gpulib.dlm lib/* $(prefix)/IDL/lib
	mkdir -p $(prefix)/IDL/docs
	cp -r docs/* $(prefix)/IDL/docs
	mkdir -p $(prefix)/IDL/demos
	cp -r demos/* $(prefix)/IDL/demos

endif
