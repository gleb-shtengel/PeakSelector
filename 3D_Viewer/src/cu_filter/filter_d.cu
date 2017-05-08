/*
*  file:  filter_d.cu
*
*  Double precision host and device code for CUDA filter.
*
*  $Id: //depot/gsg/HHMI/Phase2/src/cu_filter/filter_d.cu#1 $
*  $Date: 2009/03/10 $
*  $Author: rkneusel $
*
*  RTK, 09-Mar-2009
*  Last update:  10-Mar-2009
*/

#include <cuda.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>

//  Number of threads per block
#define THREADS     256

//  Parameter limits
__device__ __constant__ float d_params[45*5];

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
*  row  -  Data values for the current CGroupParams peak
*  col  -  Column number to check
*/
__device__ unsigned char check(double *row, int col) {
  double d = row[col];
  
  return ((d >= d_params[col]) && (d <= d_params[col+45]));
}


/**************************************************************
*  k_gpfilter
*
*  GroupFilterIt kernel
*
*  gp      -  CGroupParams on device
*  out     -  Filter output vector
*/
__global__ void k_gpfilter(double *gp, unsigned char *out) {
  int tid;
  double *row;
  unsigned char ans = 1;
  
  //  Output flag index
  tid = threadIdx.x + blockIdx.x*blockDim.x;

  //  Point to the proper row
  row = &gp[45*tid];

  //  Compare the desired values to those in d_params
  ans &= check(row, 9);
  ans &= check(row, 18);
  ans &= check(row, 19);
  ans &= check(row, 20);
  ans &= check(row, 21);
  ans &= check(row, 22);
  ans &= check(row, 23);
  ans &= check(row, 24);
  ans &= check(row, 26);
  ans &= check(row, 37);
  ans &= check(row, 38);
  ans &= check(row, 39);
  ans &= check(row, 40);
  ans &= check(row, 41);
  ans &= check(row, 42);
  
  //  Assign the output value
  out[tid] = ans;
}


/**************************************************************
*  k_filter
*
*  FilterIt kernel
*
*  gp      -  CGroupParams on device
*  out     -  Filter output vector
*/
__global__ void k_filter(double *gp, unsigned char *out) {
  int tid;
  double *row;
  unsigned char ans = 1;
  
  //  Output flag index
  tid = threadIdx.x + blockIdx.x*blockDim.x;

  //  Point to the proper row
  row = &gp[45*tid];

  //  Compare the desired values to those in d_params
  ans &= check(row, 0);
  ans &= check(row, 1);
  ans &= check(row, 2);
  ans &= check(row, 3);
  ans &= check(row, 4);
  ans &= check(row, 5);
  ans &= check(row, 6);
  ans &= check(row, 7);
  ans &= check(row, 8);
  ans &= check(row, 9);
  ans &= check(row, 10);
  ans &= check(row, 11);
  ans &= check(row, 12);
  ans &= check(row, 13);
  ans &= check(row, 14);
  ans &= check(row, 15);
  ans &= check(row, 16);
  ans &= check(row, 17);
  ans &= check(row, 26);
  ans &= check(row, 27);
  ans &= check(row, 28);
  ans &= check(row, 29);
  ans &= check(row, 30);
  ans &= check(row, 31);
  ans &= check(row, 32);
  ans &= check(row, 33);
  ans &= check(row, 34);
  ans &= check(row, 35);
  ans &= check(row, 36);
  
  //  Assign the output value
  out[tid] = ans;
}

//
//  Host code:
//

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


/**************************************************************
*  cuda_error
*
*  Check for a CUDA error on the last call.  If there is one
*  return the code (else cudaSuccess) and set the pointer, if
*  not NULL, to the message text.
*/
void checkError(char *s) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    fprintf(stderr, "ERROR: %s: %s\n", s, cudaGetErrorString(err));
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
*  filter
*
*  Entry point from IDL DLM code.
*
*/
extern "C" void filter(double *gp, int npeaks, float *params,
                       unsigned char *out) {
  int nblocks;
  double s,e;

  s = getTime();

  //  Copy ParamLimits to the device
  cudaMemcpyToSymbol(d_params, params, 45*5*sizeof(float));
  checkError("Copy ParamLimits");
  cudaThreadSynchronize();

  //  Allocate memory for the output filter flags
  nblocks = (int)ceil(npeaks/(float)THREADS);
  cudaMalloc((void **)&d_out, nblocks*THREADS*sizeof(unsigned char));
  checkError("Allocate d_out");
  cudaThreadSynchronize();

  //  Allocate memory for CGroupParams aligned properly
  cudaMalloc((void **)&d_gp, 45*nblocks*THREADS*sizeof(double));

  e = getTime();
  printf("Device and constant memory allocation time = %f\n", e-s);

  //  Copy CGroupParams to the device
  s = getTime();
  cudaMemcpy(d_gp, gp, 45*npeaks*sizeof(double), cudaMemcpyHostToDevice);
  cudaThreadSynchronize();
  e = getTime();
  printf("Copy CGroupParams to device = %f\n", e-s);

  //  Set up the grid and blocks for one image
  dim3 threads(THREADS);
  dim3 blocks(nblocks);

  //
  //  Call the kernel
  //
  s = getTime();
  k_filter<<< nblocks, THREADS >>>(d_gp, d_out);
  e = getTime();
  printf("Kernel call = %f\n", e-s);

  //  Copy the output directly to IDL memory
  s = getTime();
  cudaMemcpy(out, d_out, npeaks*sizeof(unsigned char), cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();
  e = getTime();
  printf("Copy output from device = %f\n", e-s);

  //  Clean up device memory
  cudaFree(d_gp);
  checkError("d_gp free");
  cudaFree(d_out);
  checkError("d_out free");
}


/**************************************************************
*  gpfilter
*
*  Entry point from IDL DLM code.
*
*/
extern "C" void gpfilter(double *gp, int npeaks, float *params,
                         unsigned char *out) {
  int nblocks;
  double s,e;

  s = getTime();

  //  Copy ParamLimits to the device
  cudaMemcpyToSymbol(d_params, params, 45*5*sizeof(float));
  checkError("Copy ParamLimits");
  cudaThreadSynchronize();

  //  Allocate memory for the output filter flags
  nblocks = (int)ceil(npeaks/(float)THREADS);
  cudaMalloc((void **)&d_out, nblocks*THREADS*sizeof(unsigned char));
  checkError("Allocate d_out");
  cudaThreadSynchronize();

  //  Allocate memory for CGroupParams aligned properly
  cudaMalloc((void **)&d_gp, 45*nblocks*THREADS*sizeof(double));

  e = getTime();
  printf("Device and constant memory allocation time = %f\n", e-s);

  //  Copy CGroupParams to the device
  s = getTime();
  cudaMemcpy(d_gp, gp, 45*npeaks*sizeof(double), cudaMemcpyHostToDevice);
  cudaThreadSynchronize();
  e = getTime();
  printf("Copy CGroupParams to device = %f\n", e-s);

  //  Set up the grid and blocks for one image
  dim3 threads(THREADS);
  dim3 blocks(nblocks);

  //
  //  Call the kernel
  //
  s = getTime();
  k_gpfilter<<< nblocks, THREADS >>>(d_gp, d_out);
  e = getTime();
  printf("Kernel call = %f\n", e-s);

  //  Copy the output directly to IDL memory
  s = getTime();
  cudaMemcpy(out, d_out, npeaks*sizeof(unsigned char), cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();
  e = getTime();
  printf("Copy output from device = %f\n", e-s);

  //  Clean up device memory
  cudaFree(d_gp);
  checkError("d_gp free");
  cudaFree(d_out);
  checkError("d_out free");
}


/**************************************************************
*  main
*
*  A simple driver for testing the kernel.  Build with:
*
*  $ nvcc filter.cu -o filter
*
*/
int main(int argc, char *argv[]) {
  double *gp;
  float *p;
  unsigned char *out;
  double s,e;
  
  p = (float *)malloc(45*5*sizeof(float));
  gp = (double *)malloc(45*148623*sizeof(double));
  out = (unsigned char *)malloc(148623);

  s = getTime();
  gpfilter(gp, 148623, p, out);
  e = getTime();
  printf("Total runtime = %f\n", e-s);

  free(p);
  free(gp);
  free(out);
  return 0;
}

/*
*  end filter_d.cu
*/

