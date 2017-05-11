/*
*  file:  poly_3d.cu
*
*  Host and device code for CUDA poly_3d.
*
*  $Id: //depot/gsg/HHMI/Phase2/src/cu_poly_3d/poly_3d.cu#14 $
*  $Date: 2009/11/30 $
*  $Author: rkneusel $
*
*  RTK, 24-Feb-2009
*  Last update:  16-Nov-2009
*/

#include <pthread.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <fcntl.h>
#ifndef WIN32
#include <unistd.h>
#include <sys/time.h>
#endif

#include "poly_3d.h"

//#ifndef WIN32
//#define DEBUG
//#endif

//  Thread stuff
typedef struct thread_data {
    int start;
    int end;
    int nx;
    int ny;
    unsigned short *img;
    unsigned short *out;
    float *P;
    float *Q;
} thread_data_t;

//  x,y to offset
__device__ int stride;
#define IDX(X,Y)  ((X)+(Y)*(stride))

//  Number of threads per block
#define THREADS     256

//  Coefficient arrays
__device__ __constant__ float d_p[MAX_FRAMES*2*2];
__device__ __constant__ float d_q[MAX_FRAMES*2*2];

unsigned short *d_out;  //  Output image stack 
unsigned short *d_img;  //  Input image stack
unsigned short *d_tmp;  //  Single output image

//
//  Device code:
//


/**************************************************************
*  bilinear
*
*  Use bilinear interpolation to get the output image value.
*
*  img     -  source image
*  a, b    -  coordinates to interpolate at
*  nx, ny  -  image dimensions
*/
__device__ unsigned short bilinear(unsigned short *img, float a, float b, int nx, int ny) {
  int x1,x2,y1,y2;
  unsigned short q11, q12, q21, q22;
  float A,B,C,D;
  unsigned short ans;

  //  Set the stride for IDX
  stride = nx;
  
  //  Get the coords around (a,b)
  x1 = (int)floor(a);
  y1 = (int)floor(b);
  
  //  Check for out of bounds
  if ((x1 < 0) || (y1 < 0) || (x1 > nx-2) || (y1 > ny-2)) {
    x1 = (x1 < 0) ? 0 : x1;
    x1 = (x1 > nx-1) ? nx-1 : x1;
    y1 = (y1 < 0) ? 0 : y1;
    y1 = (y1 > ny-1) ? ny-1 : y1;
    ans = img[IDX(x1,y1)];
  } else {
    x2 = x1 + 1;
    y2 = y1 + 1;
    
    //  Get the image values at the coordinates above
    q11 = img[IDX(x1,y1)];
    q12 = img[IDX(x1,y2)];
    q21 = img[IDX(x2,y1)];
    q22 = img[IDX(x2,y2)];
    
    A = x2-a;
    B = y2-b;
    C = a-x1;
    D = b-y1;
    
    ans = (unsigned short)(0.5+(A*B*q11 + C*B*q21 + A*D*q12 + C*D*q22));
  }
  __syncthreads();

  return ans;
}


/**************************************************************
*  poly2d
*
*  Interpolate a single image.
*
*  img     -  input image (nx,ny)
*  out     -  output image pointer
*  nx, ny  -  image dimensions
*  mp      -  index into d_q and d_p
*/
__global__ void poly2d(unsigned short *img, unsigned short *out, int nx, int ny, int mp) {
  int x,y;
  float a,b;
  
  //  index into the image
  int tid = threadIdx.x + blockIdx.x*blockDim.x;

  //  Convert tid to x,y (col,row) indices
  x = tid / nx;
  y = tid - nx*x;

  //  Calculate a(x,y) and b(x,y), indices into original image
  a = d_p[mp+0] + d_p[mp+1]*x + d_p[mp+2]*y + d_p[mp+3]*x*y;
  b = d_q[mp+0] + d_q[mp+1]*x + d_q[mp+2]*y + d_q[mp+3]*x*y;

  //  Calculate interpolated image value
  out[tid] = bilinear(img, a, b, nx, ny);
}


//
//  Host code:
//

#ifdef DEBUG
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
*  poly_warp_3d
*
*  Thread entry point
*/
void *poly_warp_3d(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    int nframes, nblocks, nx, ny, nelem;
    int ip, mp, k;
    unsigned short *img, *out;
#ifdef DEBUG
    double s,e;
#endif

    //  Input and output pointers
    img = data->img;
    out = data->out;
    
    //  Set the card to use
    cudaSetDevice((data->start != 0));

    //  Copy P and Q to the device
    nframes = data->end - data->start + 1;

    cudaMemcpyToSymbol(d_p, &(data->P)[data->start], nframes*2*2*sizeof(float));
    checkError("Copy P to d_p");
    cudaThreadSynchronize();

    cudaMemcpyToSymbol(d_q, &(data->Q)[data->start], nframes*2*2*sizeof(float));
    checkError("Copy Q to d_q");
    cudaThreadSynchronize();

    //  Allocate memory for the image stack
    nx = data->nx;
    ny = data->ny;
    nelem = nx*ny*nframes;
    cudaMalloc((void **)&d_img, nelem*sizeof(unsigned short));
    checkError("Allocate d_img");
    cudaThreadSynchronize();

    //  Allocate memory for the output image stack
    cudaMalloc((void **)&d_out, nelem*sizeof(unsigned short));
    checkError("Allocate d_out");
    cudaThreadSynchronize();

    //  Temporary output image
    nblocks = (int)ceil((nx*ny)/(float)THREADS);
    cudaMalloc((void **)&d_tmp, nblocks*THREADS*sizeof(unsigned short));
    checkError("Allocate d_tmp");
    cudaThreadSynchronize();

    //  Copy the input images to the device
    cudaMemcpy(d_img, &img[data->start], nelem*sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaThreadSynchronize();

    //  Set up the grid and blocks for one image
    dim3 threads(THREADS);
    dim3 blocks(nblocks);

    //
    //  Loop over all images in the stack
    //
#ifdef DEBUG
    s = getTime();
#endif
    for(k=0; k < nframes; k++) {
        //  Interpolate the current frame
        ip = nx*ny*k;  // current image index
        mp = 2*2*k;    // current P,Q matrix index
        poly2d<<< blocks, threads >>>(&d_img[ip], d_tmp, nx, ny, mp);
        checkError("poly2d call");
        cudaThreadSynchronize();
        
        //  Copy the new image to the output image stack
        cudaMemcpy(&d_out[ip], d_tmp, nx*ny*sizeof(unsigned short),
                   cudaMemcpyDeviceToDevice);
        checkError("Device to device memcpy");
        cudaThreadSynchronize();
    }
#ifdef DEBUG
    e = getTime();
    printf("Kernel call = %f\n", e-s);
#endif

  //  Copy the output image directly to IDL memory
#ifdef DEBUG
    s = getTime();
#endif
    cudaMemcpy(&out[data->start], d_out, nelem*sizeof(unsigned short), cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
#ifdef DEBUG
    e = getTime();
    printf("Copy image stack from device = %f\n", e-s);
#endif

    //  Clean up device memory
    cudaFree(d_img);
    checkError("d_img free");
    cudaFree(d_out);
    checkError("d_out free");
    cudaFree(d_tmp);
    checkError("d_tmp free");

    return 0;
}


/**************************************************************
*  poly_3d
*
*  Entry point from IDL DLM code.
*
*  img, nx, ny, nz  -  input image and dimensions
*  P, Q             -  polynomial coeff, 2x2xnz assumed
*  out              -  output image, same dims as img
*/
extern "C" void poly_3d(unsigned short *img, int nx, int ny, int nframes,
                        float *P, float *Q,
                        unsigned short *out) {
  pthread_t thing1, thing2;
  pthread_attr_t attr;
  thread_data_t data1, data2;
  void *status;
  int rc, ncards;
#ifdef DEBUG
  double s,e;
#endif

#ifdef DEBUG
  s = getTime();
#endif

  //  How many cards in the system?
  cudaGetDeviceCount(&ncards);

  //  Ensure threads are joinable
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  //  Set up pointers to divide the input image stack into two parts
  data1.start = 0;
  data2.start = nframes / 2;
  data1.end = data2.start - 1;
  data2.end = nframes - 1;
  data1.nx = data2.nx = nx;
  data1.ny = data2.ny = ny;
  data1.img = data2.img = img;
  data1.out = data2.out = out;
  data1.P = data2.P = P;
  data1.Q = data2.Q = Q;

  //  Adjust if only one card
  if (ncards == 1) {
    data1.end = nframes - 1;
  }

  //  Create the threads, each one processing one part of the image stack
  if ((rc=pthread_create(&thing1, &attr, poly_warp_3d, (void *)&data1))) {
    printf("Error: Unable to create thread: %d\n", rc);
  }
  
  if (ncards > 1) {
    if ((rc=pthread_create(&thing2, &attr, poly_warp_3d, (void *)&data2))) {
      printf("Error: Unable to create thread: %d\n", rc);
    }
  }

  //  Wait for the threads to finish
  pthread_attr_destroy(&attr);
  pthread_join(thing1, &status);
  
  if (ncards > 1) {
    pthread_join(thing2, &status);
  }

#ifdef DEBUG
  e = getTime();
  printf("Run time = %f\n", e-s);
#endif
}

#ifdef DEBUG
/**************************************************************
*  main
*
*  A simple driver for testing the kernel.  Build with:
*
*  $ nvcc poly_3d.cu -o poly_3d
*
*/
int main(int argc, char *argv[]) {
  unsigned short *img, *out;
  float *p, *q;
  double s,e;

  img = (unsigned short *)malloc(512*512*100*sizeof(unsigned short));
  out = (unsigned short *)malloc(512*512*100*sizeof(unsigned short));
  p = (float *)malloc(2*2*100*sizeof(float));
  q = (float *)malloc(2*2*100*sizeof(float));

  s = getTime();
  poly_3d(img, 512, 512, 100, p, q, out);
  e = getTime();
  printf("Run time = %f\n", e-s);

  return 0;
}
#endif

/*
*  end poly_3d.cu
*/

