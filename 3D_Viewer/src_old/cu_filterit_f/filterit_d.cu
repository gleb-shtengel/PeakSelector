/*
*  file:  filterit_d.cu
*
*  Double precision host and device code for CUDA filter.
*
*  $Id: //depot/gsg/HHMI/Phase2/src/cu_filter/filter_f.cu#1 $
*  $Date: 2009/03/10 $
*  $Author: rkneusel $
*
*  RTK, 25-Jun-2009
*  Last update:  28-Jul-2009
*/

#include <cuda.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef WIN32
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#endif

//  Number of threads per block
#define THREADS     256

//  Parameter limits
#define MAX_LIMITS    5
#define MAX_PARAMS 1000
#define MAX_PARAM_SPACE  (MAX_PARAMS*MAX_LIMITS)

__device__ __constant__ float d_params[MAX_PARAM_SPACE];
__device__ __constant__ unsigned char d_index[MAX_PARAMS];

double *d_gp;           //  CGroupParams
unsigned char *d_out;  //  Output filter flags

//
//  Device code:
//

/**************************************************************
*  check
*
*  Check a particular element of gp
*
*  row     -  Data values for the current CGroupParams peak
*  col     -  Column number to check
*  nparams -  Number of parameters
*/
__device__ unsigned char check(double *row, int col, int nparams) {
  double d = row[col];
  
  return ((d >= d_params[col]) && (d <= d_params[col+nparams]));
}


/**************************************************************
*  k_filter
*
*  FilterIt kernel
*
*  gp      -  CGroupParams on device
*  out     -  Filter output vector
*  nparams -  Number of parameters
*/
__global__ void k_filter(double *gp, unsigned char *out, int nparams) {
  int tid, i;
  double *row;
  unsigned char ans = 1;
  
  //  Output flag index
  tid = threadIdx.x + blockIdx.x*blockDim.x;

  //  Point to the proper row
  row = &gp[nparams*tid];

  //  Check
  for(i=0; i < nparams; i++) {
    if (d_index[i])
      ans &= check(row, i, nparams);
    __syncthreads();
  }
  
  //  Assign the output value
  out[tid] = ans;
}


//
//  Host code:
//

#ifndef WIN32
/**************************************************************
*  getTime
*
*  Return the system time.
*/
double getTime() {
    struct timeval tv;
    cudaThreadSynchronize();
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000;
}
#endif


/**************************************************************
*  checkError
*
*  Check for a CUDA error on the last call.
*/
void checkError(char *s) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    fprintf(stderr, "ERROR: %s: %s\n", s, cudaGetErrorString(err));
  }
}


/**************************************************************
*  cudaThreadSync
*/
void cudaThreadSync() {
  cudaError_t err = cudaThreadSynchronize();

  if (cudaSuccess != err) {
    fprintf(stderr, "ERROR: Bad sync: %s\n", cudaGetErrorString(err));
  }
}


/**************************************************************
*  cuda_safe_init
*
*  Wait until all threads done.
*/
extern "C" void cuda_safe_init(void) {
  cudaThreadSynchronize();
}


/**************************************************************
*  filterit
*
*  Entry point from IDL DLM code.
*
*/
extern "C" void filterit(double *gp, int nparams, int npeaks, int nlimits, float *params,
                         unsigned char *filterindex, unsigned char *out) {
  int nblocks;

  //  Copy ParamLimits to the device
  cudaMemcpyToSymbol(d_params, params, nparams*nlimits*sizeof(float));
  checkError("copy ParamLimits");
  cudaThreadSync();

  //  Copy filterindex to the device
  cudaMemcpyToSymbol(d_index, filterindex, nparams*sizeof(unsigned char));
  checkError("copy filterindex");
  cudaThreadSync();

  //  Allocate memory for the output filter flags
  nblocks = (int)ceil(npeaks/(float)THREADS);
  cudaMalloc((void **)&d_out, nblocks*THREADS*sizeof(unsigned char));
  checkError("allocate d_out");
  cudaThreadSync();

  //  Allocate memory for CGroupParams aligned properly
  cudaMalloc((void **)&d_gp, nparams*nblocks*THREADS*sizeof(double));
  checkError("allocate CGroupParams");
  cudaThreadSync();

  //  Copy CGroupParams to the device
  cudaMemcpy(d_gp, gp, nparams*npeaks*sizeof(double), cudaMemcpyHostToDevice);
  checkError("copy CGroupParams");
  cudaThreadSync();

  //  Set up the grid and blocks for one image
  dim3 threads(THREADS);
  dim3 blocks(nblocks);

  //
  //  Call the kernel
  //
  k_filter<<< nblocks, THREADS >>>(d_gp, d_out, nparams);
  checkError("kernel call");
  cudaThreadSync();

  //  Copy the output directly to IDL memory
  cudaMemcpy(out, d_out, npeaks*sizeof(unsigned char), cudaMemcpyDeviceToHost);
  checkError("copy output");
  cudaThreadSync();

  //  Clean up device memory
  cudaFree(d_gp);
  checkError("d_gp free");
  cudaFree(d_out);
  checkError("d_out free");
}


#ifndef WIN32
/**************************************************************
*  main
*
*  A simple driver for testing the kernel.  Build with:
*
*  $ nvcc filterit_d.cu -o filterit_d
*
*/
int main(int argc, char *argv[]) {
  double *gp;
  float *p;
  unsigned char *out, *index;
  double s,e;
  
  p = (float *)malloc(45*5*sizeof(double));
  gp = (double *)malloc(45*148623*sizeof(double));
  out = (unsigned char *)malloc(148623);
  index = (unsigned char *)malloc(45);

  memset((void *)index, 0, 45*sizeof(unsigned char));
  memset((void *)index, 1, 10*sizeof(unsigned char)); 

  s = getTime();
  filterit(gp, 45, 148623, 5, p, index, out);
  e = getTime();
  printf("Total runtime = %f\n", e-s);

  free(p);
  free(gp);
  free(out);
  return 0;
}
#endif

/*
*  end filterit_d.cu
*/

