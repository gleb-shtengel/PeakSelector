/*
*  file:  poly3d.cu
*
*  Host and device code for CUDA poly3d.
*
*  $Id: //depot/gsg/HHMI/Phase2/src/cu_poly3d/poly3d.cu#6 $
*  $Date: 2009/07/07 $
*  $Author: rkneusel $
*
*  RTK, 23-Nov-2009
*  Last update:  02-Dec-2009
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

#include "poly3d.h"

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
    int ncards;
    int thing;
} thread_data_t;

/*  
*  Grid size is Nx128 blocks with 256 blocks per thread, therefore,
*  allocate to a multiple of 128*256 = 32768.
*/
#define GRID_SIZE  32768
#define GRID_Y       128
#define THREAD_X     256

unsigned short *d_out;  //  Output image stack 
unsigned short *d_img;  //  Input image stack
float *d_P, *d_Q;       //  P and Q 

//  x,y to offset
__device__ int stride;
#define IDX(X,Y)  ((X)+(Y)*(stride))


//
//  Device code:
//

/**************************************************************
*  idx2coord
*
*  Change a sequential index into the x,y,z position of the 
*  corresponding voxel center.
*/
__device__ void idx2coord(int xdim, int ydim, int xy, int idx,
                  int *i, int *j, int *k) {
  int w;

  /*  Map idx -> i,j,k, voxel indices  */
  *k = idx / xy;
  w = idx - xy*(*k);
  *j = w / xdim;
  *i = w - xdim*(*j);
}


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
*  k_poly3d
*
*  Interpolate a single voxel.
*
*  img     -  input image stack
*  out     -  output image stack
*  nx, ny  -  image dimensions
*  nframes -  number of images in the stack
*  d_P     -  2x2 P matrix
*  d_Q     -  2x3 Q matrix
*  maxZ    -  z upper bound
*/
__global__ void k_poly3d(unsigned short *img, unsigned short *out, 
                         int nx, int ny, float *d_P, float *d_Q, int maxZ, int thing) {
    int tid,x,y,z,mp;
    float a,b;
    
    //  Index into the input image stack
    tid = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;

    //  Change into x,y,z coordinates
    idx2coord(nx, ny, nx*ny, tid, &x, &y, &z);
 
    //  Check if in bounds
    if (z > maxZ) {
        return;
    }
    __syncthreads();

    //  Calculate a(x,y) and b(x,y), indices into original image
    mp = 2*2*z;
    a = d_P[mp+0] + d_P[mp+1]*x + d_P[mp+2]*y + d_P[mp+3]*x*y;
    b = d_Q[mp+0] + d_Q[mp+1]*x + d_Q[mp+2]*y + d_Q[mp+3]*x*y;

    //  Calculate interpolated image value
    out[tid] = bilinear(&img[nx*ny*z], a, b, nx, ny);
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
    unsigned short *img, *out;
    int nalloc, npq;
#ifdef DEBUG
    double s,e;
#endif

    //  Input and output pointers
    img = data->img;
    out = data->out;
    
    //  Set the card to use
    cudaThreadExit();
    cudaSetDevice(data->thing);
    checkError("set device error");
    cudaThreadSynchronize();

    /*  Determine the number of elements to allocate for the volume  */
    nframes = data->end - data->start + 1;
    nx = data->nx;
    ny = data->ny;
    nelem = nx*ny*nframes;
    nalloc = (int)(GRID_SIZE*ceil(nelem/(float)GRID_SIZE));
    npq = (int)(GRID_SIZE*ceil((4*nframes)/(float)GRID_SIZE));
    nblocks = nalloc/GRID_SIZE;

    /*  Allocate memory on the device */
#ifdef DEBUG
    printf("%d: nelem = %d\n", data->thing,  nelem);
    printf("%d: nalloc= %d\n", data->thing,  nalloc);
    printf("%d: npq = %d\n", data->thing,  npq);
    printf("%d: nblocks= %d\n", data->thing,  nblocks);
    printf("%d: GRID_SIZE= %d\n", data->thing,  GRID_SIZE);
    printf("%d: nx,ny = %d, %d\n", data->thing,  nx,ny);
    printf("%d: nframes = %d\n", data->thing, nframes);
    printf("%d: data->start = %d\n", data->thing, data->start);
    printf("%d: data->end = %d\n", data->thing, data->end);
    s = getTime();
#endif
    cudaMalloc((void **)&d_img, nalloc*sizeof(unsigned short));      //  input image stack
    checkError("Image stack");
    cudaThreadSynchronize();
    cudaMalloc((void **)&d_out, nalloc*sizeof(unsigned short));      //  output image stack
    checkError("Output stack");
    cudaThreadSynchronize();
    cudaMalloc((void **)&d_P, npq*sizeof(float));  //  P array (2x2 for each image)
    checkError("P stack");
    cudaThreadSynchronize();
    cudaMalloc((void **)&d_Q, npq*sizeof(float));  //  Q array (2x2 for each image)
    checkError("Q stack");
    cudaThreadSynchronize();
#ifdef DEBUG
    e = getTime();
    printf("%d: Memory allocation = %f\n", data->thing,  e-s);
#endif

    /*  Copy the data to the device  */
#ifdef DEBUG
    s = getTime();
#endif
    cudaMemcpy(d_P, &(data->P)[data->start*4], nframes*2*2*sizeof(float), cudaMemcpyHostToDevice);
    checkError("Copy to d_P");
    cudaThreadSynchronize();
    cudaMemcpy(d_Q, &(data->Q)[data->start*4], nframes*2*2*sizeof(float), cudaMemcpyHostToDevice);
    checkError("Copy to d_Q");
    cudaThreadSynchronize();
    cudaMemcpy(d_img, &img[data->start*nx*ny], nelem*sizeof(unsigned short), cudaMemcpyHostToDevice);
    checkError("Copy to d_img");
    cudaThreadSynchronize();
#ifdef DEBUG
    e = getTime();
    printf("%d: Copy data to the card = %f\n", data->thing,  e-s);
#endif
    //  Set up the grid 
    dim3 threads(THREAD_X);
    dim3 grid(nblocks, GRID_Y);

#ifdef DEBUG
    s = getTime();
#endif
    //  Call the kernel
    k_poly3d<<< grid, threads >>>(d_img, d_out, nx, ny, d_P, d_Q, data->end, data->thing);
    checkError("Kernel launch error");
    cudaThreadSynchronize();
#ifdef DEBUG
    e = getTime();
    printf("%d: kernel run time = %f\n", data->thing,  e-s);
#endif

    //  Copy the output image back
#ifdef DEBUG
    s = getTime();
#endif
    cudaMemcpy(&out[data->start*nx*ny], d_out, nelem*sizeof(unsigned short), cudaMemcpyDeviceToHost);
    checkError("output copy error");
    cudaThreadSynchronize();
#ifdef DEBUG
    e = getTime();
    printf("%d: Copy output to IDL = %f\n", data->thing,  e-s);
#endif

    //  Clean up
    cudaFree(d_img);
    checkError("d_img clean up");
    cudaFree(d_out);
    checkError("d_out clean up");
    cudaFree(d_P);
    checkError("d_P clean up");
    cudaFree(d_Q);
    checkError("d_Q clean up");

    cudaThreadExit();
    return 0;
}


/**************************************************************
*  pickCards
*
*  Check for Tesla cards and choose the first two, if any.
*/
void pickCards(int *ncards, int *card1, int *card2) {
    struct cudaDeviceProp prop;
    int i;

    cudaGetDeviceCount(ncards);

    if (*ncards == 0) {
        *card1 = *card2 = -1;
        return;
    }
    if (*ncards == 1) {
        *card1 = 0;
        *card2 = -1;
        return;
    }

    *card1 = *card2 = -1;

    //  Look for two Tesla cards
    for(i=0; i < *ncards; i++) {
        cudaGetDeviceProperties(&prop, i);
        if (strncmp(prop.name, "Tesla", 5) == 0) {
            if (*card1 == -1) {
                *card1 = i;
            } else {
                *card2 = i;
                return;  // found two, return now
            }
        }
    }

    //  Look for two cards of any kind
    for(i=0; i < *ncards; i++) {
        cudaGetDeviceProperties(&prop, i);
        if (*card1 == -1) {
            *card1 = i;
        } else {
            *card2 = i;
            return;
        }
    }
}


/**************************************************************
*  poly3d
*
*  Entry point from IDL DLM code.
*
*  img, nx, ny, nz  -  input image and dimensions
*  P, Q             -  polynomial coeff, 2x2xnz assumed
*  out              -  output image, same dims as img
*/
extern "C" void poly3d(unsigned short *img, int nx, int ny, int nframes,
                        float *P, float *Q,
                        unsigned short *out,
                        int c1, int c2) {
    pthread_t thing1, thing2;
    pthread_attr_t attr;
    thread_data_t data1, data2;
    int rc, ncards, card1, card2;
#ifdef DEBUG
    double s,e;
#endif

#ifdef DEBUG
    s = getTime();
#endif

    //  Decide which cards to use
    if (c1 == -1) {
        pickCards(&ncards, &card1, &card2);
    } else {
        card1 = c1;
        card2 = c2;
        ncards = (c1 != -1) + (c2 != -1);
    }

#ifdef DEBUG
    printf("CU card1 is %d\n", card1);
    printf("CU card2 is %d\n", card2);
    printf("CU ncards is %d\n", ncards);
    printf("CU c1 is %d\n", c1);
    printf("CU c2 is %d\n", c2);
#endif

    //  Ensure threads are joinable, if we will use them
    if (ncards > 1) {
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    }

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
    data1.ncards = data2.ncards = ncards;
    data1.thing = card1;
    data2.thing = card2;

    if (ncards == 1) {
        //  One card, no threads
        data1.end = nframes - 1;
        poly_warp_3d((void *)&data1);
    } else {
        //  Create the threads, each one processing one part of the image stack
        if ((rc=pthread_create(&thing1, &attr, poly_warp_3d, (void *)&data1))) {
            printf("Error: Unable to create thread 1: %d\n", rc);
        }
#ifdef DEBUG
    printf("Thing 1 started\n");
#endif
        if ((rc=pthread_create(&thing2, &attr, poly_warp_3d, (void *)&data2))) {
            printf("Error: Unable to create thread 2: %d\n", rc);
        }
#ifdef DEBUG
    printf("Thing 2 started\n");
#endif

        //  Wait for the threads to finish
        pthread_attr_destroy(&attr);
        pthread_join(thing1, NULL);
        pthread_join(thing2, NULL);
#ifdef DEBUG
        printf("threads done\n");
#endif        
    }

#ifdef DEBUG
    e = getTime();
    printf("Run time = %f\n", e-s);
#endif
}

/*
*  end poly3d.cu
*/

