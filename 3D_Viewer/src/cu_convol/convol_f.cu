/*
*  file:  convol_f.cu
*
*  Host and device code for CUDA convol.
*
*  RTK, 18-Dec-2009
*  Last update:  23-Dec-2009
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

#include "convol_f.h"

//  Thread stuff
typedef struct thread_data {
    int start;
    int end;
    int nx;
    int ny;
    float *img;
    float *out;
    float *kernel;
    int nk;
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

float *d_out_f;  //  Output image stack 
float *d_img_f;  //  Input image stack

__device__ __constant__ float d_k_f[MAX_NK*MAX_NK];  //  kernel

//  x,y to offset
__device__ int stride;
__device__ int mpixel;
#define IDX(X,Y)  ((X)+(Y)*(stride))

//
//  Device code:
//

/**************************************************************
*  ix
*
*  Lookup an image value, zero if indices out of bounds.
*  Assumes stride already set to ncol.
*
*  img - pointer to image
*  a   - column
*  b   - row
*/
__device__ float ix(float *img, int a, int b) {
    int idx = a + b*stride;
    float ans;

    ans = ((idx < 0) || (idx >= mpixel)) ? 0.0 : img[idx];
    __syncthreads();
    return ans;
}


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
*  convolve_f
*
*  Do the actual convolution
*
*  im     -  source image
*  x,y    -  pixel to convolve
*  nx,ny  -  image dimensions
*  nk     -  kernel size
*/
__device__ float convolve_f(float *im, int x, int y, int nx, int ny, int nk) {
    float *p = d_k_f;
    float ans = 0.0;
    int i,j;

    stride = nx;
    mpixel = nx*ny;

    for(j=-nk/2; j <= nk/2; j++) {
        for(i=-nk/2; i <= nk/2; i++) {
            ans += ix(im,x+i,y+j)*(*p++);
        }
    }

    return ans;
}


/**************************************************************
*  k_convol_f
*
*  Convolve a single pixel
*
*  img     -  input image stack
*  out     -  output image stack
*  nx, ny  -  image dimensions
*  nk      -  convolution kernel size
*  maxZ    -  z upper bound
*  thing   -  parameters
*/
__global__ void k_convol_f(float *img, float *out, 
                           int nx, int ny, int nk, int maxZ, int thing) {
    int tid,x,y,z;
    
    //  Index into the input image stack
    tid = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;

    //  Change into x,y,z coordinates
    idx2coord(nx, ny, nx*ny, tid, &x, &y, &z);
 
    //  Check if in bounds
    if (z > maxZ) {
        return;
    }
    __syncthreads();

    //  Calculate the new pixel value
    out[tid] = convolve_f(&img[nx*ny*z], x, y, nx, ny, nk);
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
*  convol_warp_f
*
*  Thread entry point
*/
void *convol_warp_f(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    int nframes, nblocks, nx, ny, nelem;
    float *img, *out;
    int nalloc;
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
    nblocks = nalloc/GRID_SIZE;

    /*  Allocate memory on the device */
#ifdef DEBUG
    printf("%d: nelem = %d\n", data->thing,  nelem);
    printf("%d: nalloc= %d\n", data->thing,  nalloc);
    printf("%d: nblocks= %d\n", data->thing,  nblocks);
    printf("%d: GRID_SIZE= %d\n", data->thing,  GRID_SIZE);
    printf("%d: nx,ny = %d, %d\n", data->thing,  nx,ny);
    printf("%d: nframes = %d\n", data->thing, nframes);
    printf("%d: data->start = %d\n", data->thing, data->start);
    printf("%d: data->end = %d\n", data->thing, data->end);
    s = getTime();
#endif
    cudaMalloc((void **)&d_img_f, nalloc*sizeof(float));      //  input image stack
    checkError("Image stack");
    cudaThreadSynchronize();
    cudaMalloc((void **)&d_out_f, nalloc*sizeof(float));      //  output image stack
    checkError("Output stack");
    cudaThreadSynchronize();
#ifdef DEBUG
    e = getTime();
    printf("%d: Memory allocation = %f\n", data->thing,  e-s);
#endif

    /*  Copy the data to the device  */
#ifdef DEBUG
    s = getTime();
#endif
    cudaMemcpyToSymbol(d_k_f, data->kernel, (data->nk)*(data->nk)*sizeof(float));
    checkError("Unable to copy kernel to constant memory");
    cudaThreadSynchronize();

    cudaMemcpy(d_img_f, &img[data->start*nx*ny], nelem*sizeof(float), cudaMemcpyHostToDevice);
    checkError("Copy to d_img_f");
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
    k_convol_f<<< grid, threads >>>(d_img_f, d_out_f, nx, ny, data->nk, data->end, data->thing);
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
    cudaMemcpy(&out[data->start*nx*ny], d_out_f, nelem*sizeof(float), cudaMemcpyDeviceToHost);
    checkError("output copy error");
    cudaThreadSynchronize();
#ifdef DEBUG
    e = getTime();
    printf("%d: Copy output to IDL = %f\n", data->thing,  e-s);
#endif

    //  Clean up
    cudaFree(d_img_f);
    checkError("d_img clean up");
    cudaFree(d_out_f);
    checkError("d_out clean up");

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
*  convol_f
*
*  Entry point from IDL DLM code.
*
*  img, nx, ny, nz  -  input image and dimensions
*  kernel, nk       -  kernel and size (assumed square)
*  out              -  output image, same dims as img
*  c1, c2           -  CUDA card numbers or -1
*/
extern "C" void convol_f(float *imgs, int nx, int ny, int nz,
                         float *kernel, int nk,
                         float *out, int c1, int c2) {
    pthread_t thing1, thing2;
    pthread_attr_t attr;
    thread_data_t data1, data2;
    int rc, ncards, card1=0, card2=-1;
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
    }
    ncards = (card1 != -1) + (card2 != -1);

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
    data2.start = nz / 2;
    data1.end = data2.start - 1;
    data2.end = nz - 1;
    data1.nx = data2.nx = nx;
    data1.ny = data2.ny = ny;
    data1.img = data2.img = imgs;
    data1.out = data2.out = out;
    data1.kernel = data2.kernel = kernel;
    data1.nk = data2.nk = nk;
    data1.ncards = data2.ncards = ncards;
    data1.thing = card1;
    data2.thing = card2;

    if (ncards == 1) {
        //  One card, no threads
        data1.end = nz - 1;
        convol_warp_f((void *)&data1);
    } else {
        //  Create the threads, each one processing one part of the image stack
        if ((rc=pthread_create(&thing1, &attr, convol_warp_f, (void *)&data1))) {
            printf("Error: Unable to create thread 1: %d\n", rc);
        }
#ifdef DEBUG
    printf("Thing 1 started\n");
#endif
        if ((rc=pthread_create(&thing2, &attr, convol_warp_f, (void *)&data2))) {
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
*  end convol_f.cu
*/

