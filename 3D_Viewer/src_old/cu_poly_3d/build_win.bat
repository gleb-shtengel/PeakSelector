@ECHO OFF

REM  Clean up any previous build
del *.dll
del *.obj

REM  Point to 64-bit version of cl
SET CC="C:\Program Files (x86)\Microsoft Visual Studio 8\VC\bin"

REM  Use this if you include cutil.h - remove if the SDK is not installed
SET CUDA_UTIL_PATH="C:\Program Files (x86)\NVIDIA Corporation\NVIDIA CUDA SDK\common\inc"
SET CUDA_UTIL_LIB_PATH=C:\Program Files (x86)\NVIDIA Corporation\NVIDIA CUDA SDK\common\lib

REM  Point to IDL stuff
SET IDL_DIR=C:\Program Files\ITT\IDL64
SET IDL_LIBDIR=C:\Program Files\ITT\IDL64\bin\bin.x86_64

REM  Build the nvcc obj file
nvcc -m64 -ccbin %CC% -I%CUDA_INC_PATH% -I%CUDA_UTIL_PATH% -O3 --use_fast_math -DWIN32 -c poly_3d.cu

REM  Build the DLM obj file
cl -nologo -I"%IDL_DIR%\external\include" -I"%CUDA_INC_PATH%" -DWIN32 -c cu_poly_3d.c

REM  Now link everything together
link /DLL /OUT:cu_poly_3d.dll /DEF:cu_poly_3d.def "%CUDA_LIB_PATH%\cuda.lib" "%CUDA_LIB_PATH%\cudart.lib" "%CUDA_LIB_PATH%\cufft.lib" "%CUDA_LIB_PATH%\cublas.lib" "%IDL_LIBDIR%\idl.lib" poly_3d.obj cu_poly_3d.obj
