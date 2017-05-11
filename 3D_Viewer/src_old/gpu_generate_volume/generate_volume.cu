/*
*  file:  generate_volume.cu
*
*  Host and device code for the GPU volume generation.
*
*  RTK, 14-Oct-2008
*  Last update:  27-Oct-2008
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

/*  The number of data elements for each peak  */
#define PEAK_STEP  6

/*  (2*pi)^(3/2)  */
#define W 15.749609918

/*  Cutoff for calling expf()  */
#define EXP_CUTOFF  40.0

/*  
*  Grid size is Nx128 blocks with 256 blocks per thread, therefore,
*  allocate to a multiple of 128*256 = 32768.
*/
#define GRID_SIZE  32768
#define GRID_Y       128
#define THREAD_X     256

//
//  Memory:
//
float *d_vol;       //  Volume on the device
float *d_peaks;     //  Peaks (positions and sigmas)

//  The current peak
__device__ __shared__ float s_peaks[6];

//  A block's worth of results
__device__ __shared__ float s_vol[256];

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
                  float zscale, float df, float *xx, float *yy, float *zz, 
                  float xlow, float ylow, float zlow) {
  int i,j,k,w;

  /*  Map idx -> i,j,k, voxel indices  */
  k = idx / xy;
  w = idx - xy*k;
  j = w / xdim;
  i = w - xdim*j;

  /*  Map i,j,k -> xx,yy,zz, voxel center position (nm)  */
  *xx = xlow + (i + 0.5) * df;
  *yy = ylow + (j + 0.5) * df;
  *zz = zlow + (k + 0.5) * (df/zscale);
}


/**************************************************************
*  generage_sum
*
*  Generate the volume data using the sum.
*/
__global__ void generate_sum(
    float *vol, int xdim, int ydim, int zdim, int nelem, 
    float *peaks, int npeaks, float zscale, float df,
    float xlow, float ylow, float zlow) {
  int tid, i, xy;
  float x,y,z, sx,sy,sz, amp, xx,yy,zz;
  float ssx,ssy,ssz, v;
  float e;

  /*  Size of each 2D plane  */
  xy = xdim*ydim;

  /*  Thread id, used as the index into the volume array  */
  //tid = threadIdx.x + blockIdx.x*blockDim.x;
  tid = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;

  /*  Loop over all peaks  */
  v = 0.0;
  
  /*  Get the position of this volume element  */
  idx2coord(xdim, ydim, xy, tid, zscale, df, &xx, &yy, &zz,
            xlow, ylow, zlow);

  for(i=0; i < PEAK_STEP*npeaks; i+=PEAK_STEP) {
    //
    //  Use the first 6 threads to preload from global memory
    //  so that the remaining 250 threads can use fast shared memory.
    //
    if (threadIdx.x < 6) {
      s_peaks[threadIdx.x] = peaks[i + threadIdx.x];
    }
    __syncthreads();

    x = s_peaks[0];                         //  x position (nm)
    y = s_peaks[1];                         //  y position (nm)
    z = s_peaks[2];                         //  z position (nm)
    sx = s_peaks[3];                        //  x standard deviation
    sy = s_peaks[4];                        //  y standard deviation
    sz = s_peaks[5];                        //  z standard deviation
    ssx = sx*sx;                            //  x variance
    ssy = sy*sy;                            //  y variance
    ssz = sz*sz*(zscale/df)*(zscale/df);    //  z variance

    /*  Add the contribution to this voxel from the current peak  */
    amp = 1.0/(W*(sx/df)*(sy/df)*(sz/df));
    amp = (amp > 1.0) ? 1.0 : amp;
    e = ((xx-x)*(xx-x)/ssx)+((yy-y)*(yy-y)/ssy)+((zz-z)*(zz-z)/ssz);
    if (e < EXP_CUTOFF)  {
      v += amp*expf(-0.5*e);
    }
    __syncthreads();  // this keeps all threads together at the 
                      // end of the loop
  }

  /*  Set the voxel value  */
  s_vol[threadIdx.x] = v;
  
  //
  //  If the last thread, update global memory.  This is a little
  //  faster than letting each thread update global memory directly.
  //
  __syncthreads();
  if (threadIdx.x == 0) {
    for(i=0; i < 256; i++) {
      vol[tid+i] = s_vol[i];
    }
  }
  __syncthreads();
}


/**************************************************************
*  generage_env
*
*  Generate the volume data using the envelope.
*/
__global__ void generate_env(
    float *vol, int xdim, int ydim, int zdim, int nelem, 
    float *peaks, int npeaks, float zscale, float df,
    float xlow, float ylow, float zlow) {
  int tid, i, xy;
  float x,y,z, sx,sy,sz, amp, xx,yy,zz;
  float ssx,ssy,ssz, v,t;
  float e;

  /*  Size of each 2D plane  */
  xy = xdim*ydim;

  /*  Thread id, used as the index into the volume array  */
  tid = threadIdx.x + blockIdx.x*blockDim.x;

  /*  Loop over all peaks  */
  v = 0.0;
  
  /*  Get the position of this volume element  */
  idx2coord(xdim, ydim, xy, tid, zscale, df, &xx, &yy, &zz,
            xlow, ylow, zlow);

  for(i=0; i < PEAK_STEP*npeaks; i+=PEAK_STEP) {
    //
    //  Use the first 6 threads to preload from global memory
    //  so that the remaining 250 threads can use fast shared memory.
    //
    if (threadIdx.x < 6) {
      s_peaks[threadIdx.x] = peaks[i + threadIdx.x];
    }
    __syncthreads();

    x = s_peaks[0];                         //  x position (nm)
    y = s_peaks[1];                         //  y position (nm)
    z = s_peaks[2];                         //  z position (nm)
    sx = s_peaks[3];                        //  x standard deviation
    sy = s_peaks[4];                        //  y standard deviation
    sz = s_peaks[5];                        //  z standard deviation
    ssx = sx*sx;                            //  x variance
    ssy = sy*sy;                            //  y variance
    ssz = sz*sz*(zscale/df)*(zscale/df);    //  z variance

    /*  Add the contribution to this voxel from the current peak  */
    amp = 1.0/(W*(sx/df)*(sy/df)*(sz/df));
    amp = (amp > 1.0) ? 1.0 : amp;
    e = ((xx-x)*(xx-x)/ssx)+((yy-y)*(yy-y)/ssy)+((zz-z)*(zz-z)/ssz);
    if (e < EXP_CUTOFF)  {
       t = amp*expf(-0.5*e);
       v = (t > v) ? t : v;
    }
    __syncthreads();  // this keeps all threads together at the 
                      // end of the loop
  }

  /*  Set the voxel value  */
  s_vol[threadIdx.x] = v;
  
  //
  //  If the last thread, update global memory.  This is a little
  //  faster than letting each thread update global memory directly.
  //
  __syncthreads();
  if (threadIdx.x == 0) {
    for(i=0; i < 256; i++) {
      vol[tid+i] = s_vol[i];
    }
  }
  __syncthreads();
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
*  generate_volume
*
*  Entry point from IDL DLM code.
*/
extern "C" void generate_volume(
    float *vol, int xdim, int ydim, int zdim, int nelem,
    float *peaks, int npeaks,
    float zscale, float df,
    float x_low, float x_high, 
    float y_low, float y_high, 
    float z_low, float z_high, int envelope) {
  int nalloc, nblocks;

  /*  Determine the number of elements to allocate for the volume  */
  nalloc = (int)(GRID_SIZE*ceil(nelem/(float)GRID_SIZE));
  nblocks = nalloc/GRID_SIZE;

  /*  Allocate memory on the device for the volume and peaks  */
  cudaMalloc((void **)&d_vol, nalloc*sizeof(float));
  checkError("nalloc call");
  cudaMalloc((void **)&d_peaks, npeaks*6*sizeof(float));
  checkError("npeaks call");

  /*  Copy the peak data to the device  */
  cudaThreadSynchronize();
  cudaMemcpy(d_peaks, peaks, npeaks*6*sizeof(float), cudaMemcpyHostToDevice);
  checkError("d_peaks = peaks call");
  
  /*  
  *  Set up the threads.  The number of elements in the volume determines
  *  the number of threads.
  */
  dim3 threads(THREAD_X,1);
  dim3 grid(nblocks, GRID_Y);

  /*
  *  Call the kernel to do the calculation.
  */
  if (envelope) {
    generate_env<<< grid, threads >>>(d_vol, xdim, ydim, zdim, nelem, 
      d_peaks, npeaks, zscale, df, x_low, y_low, z_low);
      checkError("generate_env");
  } else {
    generate_sum<<< grid, threads >>>(d_vol, xdim, ydim, zdim, nelem, 
      d_peaks, npeaks, zscale, df, x_low, y_low, z_low);
    checkError("generate_sum");
  }
  cudaThreadSynchronize();
  
  /*  
  *  Copy the updated volume data back to the host.  This will update the
  *  IDL variable data directly.
  */
  cudaMemcpy(vol, d_vol, nelem*sizeof(float), cudaMemcpyDeviceToHost);
  checkError("vol = d_vol call");
  cudaThreadSynchronize();

  /*  Clean up  */
  cudaFree(d_vol);
  checkError("d_vol free");
  cudaFree(d_peaks);
  checkError("d_peaks free");
}


/**************************************************************
*  main
*
*  A simple driver for testing the kernel.  Build with:
*
*  $ nvcc generate_volume.cu -o generate_volume
*
*/
int main(int argc, char *argv[]) {
  FILE *f;
  float *peaks, *p;
  float *vol;
  int xdim,ydim,zdim,nelem,npeaks;
  int fsize, fid;
  float zscale, df;
  float x_low, x_high, y_low, y_high, z_low, z_high;
  struct stat stats;
  double st,et;

  //  Show usage
  if (argc == 1) {
    printf("\n%s <file>\n\n", argv[0]);
    printf("where:\n\n");
    printf("  <file>      =  name of the peaks file\n\n");
    return 0;
  }

  //  Get the size of the input file
  fid = open(argv[1], O_RDONLY);
  fstat(fid, &stats);
  fsize = stats.st_size;
  close(fid);

  //  Define memory for the peaks
  p = (float *)malloc(fsize);
  if (!p)
    return 1;
    
  //  Load the peaks
  f = fopen(argv[1],"r");
  fread((void *)p, fsize, 1, f);
  fclose(f);
  
  //  Extract the necessary parameters
  zscale = p[0];
  df     = p[1];
  x_low  = p[2];
  x_high = p[3];
  y_low  = p[4];
  y_high = p[5];
  z_low  = p[6];
  z_high = p[7];
  xdim   = (int)p[8];
  ydim   = (int)p[9];
  zdim   = (int)p[10];
  nelem  = xdim*ydim*zdim;
  npeaks = (int)p[11];
  peaks  = &p[12];

  //  Print file info
  printf("\n");
  printf("zscale           = %f\n", zscale);
  printf("df               = %f\n", df);
  printf("x_low, x_high    = %f, %f\n", x_low, x_high);
  printf("y_low, y_high    = %f, %f\n", y_low, y_high);
  printf("z_low, z_high    = %f, %f\n", z_low, z_high);
  printf("xdim, ydim, zdim = %d, %d, %d\n", xdim, ydim, zdim); 
  printf("npeaks           = %d\n", npeaks);
  printf("peaks 0 = %f,%f,%f (%f,%f,%f)\n\n", peaks[0], peaks[1],
         peaks[2], peaks[3], peaks[4], peaks[5]);

  //  Define the volume and set to zero
  vol = (float *)malloc(nelem*sizeof(float));
  if (!vol)
    return 1;
  memset((void *)vol, 0, nelem*sizeof(float));

  //  Generate the volume
  cuda_safe_init();
  printf("Calling kernel...\n");  fflush(stdout);
  st = getTime();
  generate_volume(vol, xdim, ydim, zdim, nelem, peaks, npeaks, zscale,
                  df, x_low, x_high, y_low, y_high, z_low, z_high, 0);
  et = getTime();
  checkError("kernel call");
  printf("... done, time = %f sec\n", et-st);  fflush(stdout);

  //  Write the volume to disk
  f = fopen("volume.dat","w");
  if (!f)
    return 1;
  fwrite((void *)vol, nelem*sizeof(float), 1, f);
  fclose(f);

  //  Clean up
  free(p);
  free(vol);
  
  return 0;
}

/*
*  end generate_volume.cu
*/

