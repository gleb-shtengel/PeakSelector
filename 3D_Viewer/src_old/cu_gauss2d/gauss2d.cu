/*
*  file:  gauss2d.cu
*
*  Fit a stack of 11x11 unsigned short images to a 2D Gaussian
*  using a particle swarm approach on the GPU.
*
*  RTK, 21-Sep-2009
*  Last update:  25-Nov-2009
*
*/

#define DEBUG

#include "gauss2d.h"

//  Thread stuff
typedef struct thread_data {
    unsigned int start;
    unsigned int end;
    int nseeds;
    int imax;
    unsigned short *imgs;
    float *out;
    float *constraints;
    unsigned int *seeds;
    int uncertainties;
    int card;
    int ncards;
} thread_data_t;


//  Pointers to memory allocated on the device (kept on the host)
float *d_gb;              //  global bests - output parameters
unsigned short *d_imgs;   //  input images
unsigned int *d_seeds;    //  PRNG seeds

//
//  Device code:
//

//  Allocate space in constant memory for the fixed x,y position arrays
__device__ __constant__ unsigned char d_xp[NUM_POINTS];
__device__ __constant__ unsigned char d_yp[NUM_POINTS];

//  Constraints, filled in by host code
__device__ __constant__ float d_constraints[2*NUM_PARAMETERS];

//  Global best kept in shared memory to be used in common
//  by one swarm
__device__ __shared__ float s_gb[NUM_PARAMETERS+1];
__device__ __shared__ float s_xb[(NUM_PARAMETERS+1)*PARTICLES_PER_IMAGE];
__device__ __shared__ float s_img[NUM_POINTS];

//  Last five global best positions
__device__ __shared__ float s_sb[5*NUM_PARAMETERS];


//
//  Kernel code:
//

/**************************************************************
*  rnd
*
*  Park-Miller MINSTD PRNG with seed supplied by thread.
*/
__device__ float rnd(unsigned int *seed) {
    *seed = 16807*(*seed) % 2147483647;
    //return (float)(*seed) / 2147483648.0;
    return 4.6566128730773926e-10 * (float)(*seed);
}


/**************************************************************
*  update_best_list
*
*  Update the list of the last five global best positions
*/
__device__ void update_best_list() {
    int i,j;

    for(j=3; j > -1; j--)
        for(i=0; i < NUM_PARAMETERS; i++)
            s_sb[(j+1)*NUM_PARAMETERS+i] = s_sb[j*NUM_PARAMETERS+i];

    s_sb[0] = s_gb[0];
    s_sb[1] = s_gb[1];
    s_sb[2] = s_gb[2];
    s_sb[3] = s_gb[3];
    s_sb[4] = s_gb[4];
    s_sb[5] = s_gb[5];
}


/**************************************************************
*  update_sigmas
*
*  Update the sigmas for the current image
*/
__device__ void update_sigmas(float *gp) {
    float m, sd;
    unsigned char i,k;

    for(i=0; i < NUM_PARAMETERS; i++) {
        m = 0.0;
        for(k=0; k < 5; k++) {
            m += s_sb[k*NUM_PARAMETERS+i];
        }
        m /= 5.0;
        sd = 0.0;
        for(k=0; k < 5; k++) {
            sd += (s_sb[k*NUM_PARAMETERS+i] - m)*(s_sb[k*NUM_PARAMETERS+i] - m);
        }
        gp[i] = sqrtf(sd/5.0);
    }
}


/**************************************************************
*  k_particle
*
*  One particle, each block is a swarm
*/
__global__ void k_particle(unsigned short *imgs,   //  Pointer to stack of images (function z values)
                           unsigned int *seeds,    //  Pointer to particle random numbers
                           int imax,               //  Number of iterations to do
                           float *gp,              //  Pointer to output parameter store
                           unsigned char unc) {    //  Will use k_sigma if true
    int i,j,k;
    float chi;
    float zfit, t,a,b;
    unsigned int seed;

    //  Use explicitly named parameters instead of an array because nvcc insists on
    //  putting arrays in slow linear memory.
    float x0,x1,x2,x3,x4,x5;  //  position
    float v0,v1,v2,v3,v4,v5;  //  velocity

    //  Initial global best chi-square
    if (threadIdx.x == 0) {
        s_gb[6] = 1e38;
    }
    __syncthreads();
    
    //  Set the PRNG seed
    seed = seeds[threadIdx.x + blockIdx.x*gridDim.y + blockIdx.y];

    //  Copy image data to shared memory
    if (threadIdx.x < NUM_POINTS) {
        s_img[threadIdx.x] = (float)(&imgs[IMGX*IMGY*(blockIdx.x*gridDim.y+blockIdx.y)])[threadIdx.x];
    }
    __syncthreads();

    //
    //  Initialize
    //

    //  Initial velocity is zero
    v0 = v1 = v2 = v3 = v4 = v5 = 0.0;

    //  Set x to a random value within the given constraints
    x0 = d_constraints[ 0] + rnd(&seed)*(d_constraints[ 1]-d_constraints[ 0]);
    x1 = d_constraints[ 2] + rnd(&seed)*(d_constraints[ 3]-d_constraints[ 2]);
    x2 = d_constraints[ 4] + rnd(&seed)*(d_constraints[ 5]-d_constraints[ 4]);
    x3 = d_constraints[ 6] + rnd(&seed)*(d_constraints[ 7]-d_constraints[ 6]);
    x4 = d_constraints[ 8] + rnd(&seed)*(d_constraints[ 9]-d_constraints[ 8]);
    x5 = d_constraints[10] + rnd(&seed)*(d_constraints[11]-d_constraints[10]);

    //  This is also the current best position
    k = (NUM_PARAMETERS+1)*threadIdx.x;
    s_xb[k  ] = x0;
    s_xb[k+1] = x1;
    s_xb[k+2] = x2;
    s_xb[k+3] = x3;
    s_xb[k+4] = x4;
    s_xb[k+5] = x5;
    __syncthreads();

    //  Calculate the chisq for the current x position which is also
    //  the particle best chi-square
    chi = 0.0;
    for(j=0; j < NUM_POINTS; j++) {
         a = (d_xp[j] - x4) / x2;
         a *= a;
         b = (d_yp[j] - x5) / x3;
         b *= b;
         zfit = x0 + x1*expf(-0.5*(a+b));
         t = (zfit - s_img[j]);
         chi += t*t;
    }
    s_xb[k+6] = (double)chi / (float)(NUM_POINTS - NUM_PARAMETERS);

    //  Set the initial swarm best - thread 0 only
    __syncthreads();
    if (threadIdx.x == 0) {
        for(i=0; i < PARTICLES_PER_IMAGE; i++) {
            k = (NUM_PARAMETERS+1)*i;
            if (s_xb[k+6] < s_gb[6]) {
                s_gb[0] = s_xb[k];
                s_gb[1] = s_xb[k+1];
                s_gb[2] = s_xb[k+2];
                s_gb[3] = s_xb[k+3];
                s_gb[4] = s_xb[k+4];
                s_gb[5] = s_xb[k+5];
                s_gb[6] = s_xb[k+6];
                if (!unc) {
                    update_best_list();
                }
            }
        }
    }
    __syncthreads();

    //
    //  Iterate
    //
    for(i=0; i < imax; i++) {

        //  Update velocity with max of +/-VMAX for any one component
        v0 = W*v0 + C1*rnd(&seed)*(s_xb[(NUM_PARAMETERS+1)*threadIdx.x+0] - x0) + 
                    C2*rnd(&seed)*(s_gb[0] - x0);
        v1 = W*v1 + C1*rnd(&seed)*(s_xb[(NUM_PARAMETERS+1)*threadIdx.x+1] - x1) + 
                    C2*rnd(&seed)*(s_gb[1] - x1);
        v2 = W*v2 + C1*rnd(&seed)*(s_xb[(NUM_PARAMETERS+1)*threadIdx.x+2] - x2) + 
                    C2*rnd(&seed)*(s_gb[2] - x2);
        v3 = W*v3 + C1*rnd(&seed)*(s_xb[(NUM_PARAMETERS+1)*threadIdx.x+3] - x3) + 
                    C2*rnd(&seed)*(s_gb[3] - x3);
        v4 = W*v4 + C1*rnd(&seed)*(s_xb[(NUM_PARAMETERS+1)*threadIdx.x+4] - x4) + 
                    C2*rnd(&seed)*(s_gb[4] - x4);
        v5 = W*v5 + C1*rnd(&seed)*(s_xb[(NUM_PARAMETERS+1)*threadIdx.x+5] - x5) + 
                    C2*rnd(&seed)*(s_gb[5] - x5);

        v0 = (v0 < -VMAX) ? -VMAX : (v0 > VMAX) ? VMAX : v0;
        v1 = (v1 < -VMAX) ? -VMAX : (v1 > VMAX) ? VMAX : v1;
        v2 = (v2 < -VMAX) ? -VMAX : (v2 > VMAX) ? VMAX : v2;
        v3 = (v3 < -VMAX) ? -VMAX : (v3 > VMAX) ? VMAX : v3;
        v4 = (v4 < -VMAX) ? -VMAX : (v4 > VMAX) ? VMAX : v4;
        v5 = (v5 < -VMAX) ? -VMAX : (v5 > VMAX) ? VMAX : v5;
        __syncthreads();

        //  Update position and apply constraints
        x0 += v0;
        x1 += v1;
        x2 += v2;
        x3 += v3;
        x4 += v4;
        x5 += v5;

        x0 = (x0 < d_constraints[ 0]) ? d_constraints[ 0] : (x0 > d_constraints[ 1]) ? d_constraints[ 1] : x0;
        x1 = (x1 < d_constraints[ 2]) ? d_constraints[ 2] : (x1 > d_constraints[ 3]) ? d_constraints[ 3] : x1;
        x2 = (x2 < d_constraints[ 4]) ? d_constraints[ 4] : (x2 > d_constraints[ 5]) ? d_constraints[ 5] : x2;
        x3 = (x3 < d_constraints[ 6]) ? d_constraints[ 6] : (x3 > d_constraints[ 7]) ? d_constraints[ 7] : x3;
        x4 = (x4 < d_constraints[ 8]) ? d_constraints[ 8] : (x4 > d_constraints[ 9]) ? d_constraints[ 9] : x4;
        x5 = (x5 < d_constraints[10]) ? d_constraints[10] : (x5 > d_constraints[11]) ? d_constraints[11] : x5;

        //  Calculate chisq for current position
        chi = 0.0;
        for(j=0; j < NUM_POINTS; j++) {
             a = (d_xp[j] - x4) / x2;
             a *= a;
             b = (d_yp[j] - x5) / x3;
             b *= b;
             zfit = x0 + x1*expf(-0.5*(a+b));
             t = (zfit - s_img[j]);
             chi += t*t;
        }
        chi /= (float)(NUM_POINTS - NUM_PARAMETERS);

        //  Update particle best, if necessary
        k = (NUM_PARAMETERS+1)*threadIdx.x;
        if (chi < s_xb[k+6]) {
            s_xb[k  ] = x0;        
            s_xb[k+1] = x1;        
            s_xb[k+2] = x2;        
            s_xb[k+3] = x3;        
            s_xb[k+4] = x4;        
            s_xb[k+5] = x5;        
            s_xb[k+6] = chi;        
        }

        //  Update the global best for all particles (threads)
        __syncthreads();
        if (threadIdx.x == 0) {
            for(j=0; j < PARTICLES_PER_IMAGE; j++) {
                k = (NUM_PARAMETERS+1)*j;
                if (s_xb[k+6] < s_gb[6]) {
                    s_gb[0] = s_xb[k];
                    s_gb[1] = s_xb[k+1];
                    s_gb[2] = s_xb[k+2];
                    s_gb[3] = s_xb[k+3];
                    s_gb[4] = s_xb[k+4];
                    s_gb[5] = s_xb[k+5];
                    s_gb[6] = s_xb[k+6];
                    if (!unc) {
                        update_best_list();
                    }
                }
            }
        }
        __syncthreads();
    }

    //
    //  Swarm search complete, copy the swarm best back to global memory
    //  (0..5 - parameters, 6 - global best reduced chi-square)
    //
    if (threadIdx.x < 7) {
        gp[(2*NUM_PARAMETERS+1)*(blockIdx.x*gridDim.y + blockIdx.y) + threadIdx.x] = s_gb[threadIdx.x];
    }
    __syncthreads();
    if ((threadIdx.x == 0) && (!unc)) {
        update_sigmas(&gp[(2*NUM_PARAMETERS+1)*(blockIdx.x*gridDim.y + blockIdx.y) + 7]);
    }
    __syncthreads();
}


/**************************************************************
*  k_sigmas
*
*  Calculate parameter uncertainties ala Bevington 11-36.
*/
__global__ void k_sigma(float *gp, unsigned short *imgs, int nimgs) {
  float *p, *s;
  unsigned short *z;
  float u, u2, g2, g3, g4, g5;
  float t0,t1,t2;
  int i;
  //float D = 2.0 / (121-6);  

  //  Index into parameter memory
  //j = blockIdx.x*gridDim.y + blockIdx.y;
  i = (2*NUM_PARAMETERS+1)*(blockIdx.x*gridDim.y + blockIdx.y);
  p = &gp[i];    //  parameters for the current image, 0..5
  s = &gp[i+7];  //  parameter uncertainties here

  //  Index into image memory (z values)
  i = (IMGX*IMGY)*(blockIdx.x*gridDim.y + blockIdx.y);
  z = &imgs[i];

  //  Zero output values
  s[1] = s[2] = s[3] = s[4] = s[5] = 0.0;

  //  s[0] is trivial case
  s[0] = 121;

  //  Calculate s[1..5]
  //
  //  All the temporaries are necessary to avoid using powf() which has poor
  //  precision and pow() which eats up too many registers.
  //
  for(i=0; i < NUM_POINTS; i++) {
    //arg = -0.5*(((x-p[4])/p[2])^2+((y-p[5])/p[3])^2);
    t0 = (d_xp[i]-p[4])/p[2];
    t1 = (d_yp[i]-p[5])/p[3];
    u = expf(-0.5*(t0*t0+t1*t1));
    u2 = u*u;
    //g2 = u*(p[1]*(x-p[4])^4/p[2]^6 - 3.0*p[1]*(x-p[4])^2/p[2]^4);
    //g3 = u*(p[1]*(y-p[5])^4/p[3]^6 - 3.0*p[1]*(y-p[5])^2/p[3]^4);
    //g4 = u*(p[1]*(x-p[4])^2/p[2]^4 - p[1]/p[2]^2);
    //g5 = u*(p[1]*(y-p[5])^2/p[3]^4 - p[1]/p[3]^2);
    t0 = d_xp[i]-p[4]; t1 = p[2]; t2 = (t0*t0*t0*t0)/(t1*t1*t1*t1*t1*t1);
    g2 = u*p[1]*(t2 - 3.0*(t0*t0)/(t1*t1*t1*t1));
    g4 = u*p[1]*((t0*t0)/(t1*t1*t1*t1) - 1.0/(t1*t1));

    t0 = d_yp[i]-p[5]; t1 = p[3]; t2 = (t0*t0*t0*t0)/(t1*t1*t1*t1*t1*t1);
    g3 = u*p[1]*(t2 - 3.0*(t0*t0)/(t1*t1*t1*t1));
    g5 = u*p[1]*((t0*t0)/(t1*t1*t1*t1) - 1.0/(t1*t1));

    s[1] += u2;
    //s[2] = (p[1]*(x-p[4])^2/p[2]^3)^2*u2 - z*g2 + (p[0]+p[1]*u)*g2;
    //s[3] = (p[1]*(y-p[5])^2/p[3]^3)^2*u2 - z*g3 + (p[0]+p[1]*u)*g3;
    //s[4] = (p[1]*(x-p[4])/p[2]^2)^2*u2 - z*g4 + (p[0]+p[1]*u)*g4;  
    //s[5] = (p[1]*(y-p[5])/p[3]^2)^2*u2 - z*g5 + (p[0]+p[1]*u)*g5;  
    t0 = d_xp[i]-p[4]; t1 = p[2]; t2 = p[1]*(t0*t0)/(t1*t1*t1);
    s[2] += t2*t2*u2 - z[i]*g2 + (p[0]+p[1]*u)*g2;
    t2 = p[1]*t0/(t1*t1);
    s[4] += t2*t2*u2 - z[i]*g4 + (p[0]+p[1]*u)*g4;

    t0 = d_yp[i]-p[5]; t1 = p[3]; t2 = p[1]*(t0*t0)/(t1*t1*t1);
    s[3] += t2*t2*u2 - z[i]*g3 + (p[0]+p[1]*u)*g3;
    t2 = p[1]*t0/(t1*t1);
    s[5] += t2*t2*u2 - z[i]*g5 + (p[0]+p[1]*u)*g5;
  }

  //  Calculate final value
  for(i=0; i < 6; i++) {
    //s[i] = sqrtf(2.0/(D*s[i]));
    s[i] = sqrtf(115.0/s[i]);
  }
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
*  gauss2d_warp
*
*  Thread entry point
*/
void *gauss2d_warp(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    unsigned char x[NUM_POINTS], y[NUM_POINTS];
    int i,j,k, nimgs;
    unsigned short *imgs = (unsigned short *)NULL;
    float *out = (float *)NULL;
    unsigned int *seeds = (unsigned int *)NULL;
    int rows;    //  number of rows in the grid x COLS
#ifdef DEBUG
    double sss,eee;
#endif

    //  Setup pointers
    imgs = data->imgs;
    out = data->out;
    seeds = data->seeds;

    //  Determine the grid size
    nimgs = data->end - data->start + 1;
    rows = nimgs/COLS + 1;

#ifdef DEBUG
    printf("%d: data->start = %d\n", data->card, data->start);
    printf("%d: data->end = %d\n", data->card, data->end);
    printf("%d: data->imax = %d\n", data->card, data->imax);
    printf("%d: nimgs, rows = %d, %d\n", data->card, nimgs, rows);

    sss = getTime();
#endif

    //  Set the card to use
    cudaThreadExit();
    cudaSetDevice(data->card);
    checkError("set device error");
    cudaThreadSynchronize();

    //  Set up the x,y arrays and copy to constant memory
    for(i=0; i < NUM_POINTS; i++)
        x[i] = i % IMGX;
    for(i=0, k=0; i < IMGY; i++)
        for(j=0; j < IMGX; j++)
            y[k++] = i;

    cudaMemcpyToSymbol(d_xp, x, NUM_POINTS*sizeof(unsigned char));
    checkError("Unable to copy x array to constant memory");
    cudaMemcpyToSymbol(d_yp, y, NUM_POINTS*sizeof(unsigned char));
    checkError("Unable to copy y array to constant memory");

    //  Copy the constraints to constant memory
    cudaMemcpyToSymbol(d_constraints, data->constraints, 2*NUM_PARAMETERS*sizeof(float));
    checkError("Unable to copy constraints to constant memory");
    cudaThreadSynchronize();

#ifdef DEBUG
    eee = getTime();
    printf("%d: Create x,y arrays and put constraints on GPU = %f\n", data->card, eee-sss);

    sss = getTime();
#endif
    //  Create space for the images and copy them to the GPU
    //  allowing for the extra garbage images for the grid
#ifdef DEBUG
    printf("%d: IMGX*IMGY*rows*COLS*sizeof(unsigned short) = %d\n", data->card, IMGX*IMGY*rows*COLS*sizeof(unsigned short));
    printf("%d: IMGX*IMGY*nimgs*sizeof(unsigned short) = %d\n", data->card, IMGX*IMGY*nimgs*sizeof(unsigned short));
#endif
    cudaMalloc((void **)&d_imgs, IMGX*IMGY*rows*COLS*sizeof(unsigned short));
    checkError("Unable to reserve space for the input images");
    cudaThreadSynchronize();
    cudaMemcpy(d_imgs, &imgs[data->start*IMGX*IMGY], 
                IMGX*IMGY*nimgs*sizeof(unsigned short), cudaMemcpyHostToDevice);
    checkError("Unable to copy the images to the device");

#ifdef DEBUG
    eee = getTime();
    printf("%d: Allocate and copy image data to GPU = %f\n", data->card, eee-sss);

    sss = getTime();
#endif

    //  Copy the PRNG seeds to the GPU with bogus seeds for the garbage images
    cudaMalloc((void **)&d_seeds, rows*COLS*PARTICLES_PER_IMAGE*sizeof(unsigned int));
    checkError("Unable to reserve space for the PRNG seeds");
    cudaThreadSynchronize();
    cudaMemcpy(d_seeds, &seeds[data->start*PARTICLES_PER_IMAGE], 
                nimgs*PARTICLES_PER_IMAGE*sizeof(unsigned int), cudaMemcpyHostToDevice);
    checkError("Unable to copy the seeds to the device");

#ifdef DEBUG
    eee = getTime();
    printf("%d: Copy the PRNG seeds to the GPU = %f\n", data->card, eee-sss);
#endif
    //  Create the output parameter array 
    cudaMalloc((void **)&d_gb, (2*NUM_PARAMETERS+1)*rows*COLS*sizeof(float));
    checkError("Unable to reserve space for the output parameters");
    cudaThreadSynchronize();
    
    //  Calculate the fit parameters
#ifdef DEBUG
    sss = getTime();
#endif
    dim3 grid(rows, COLS);
    k_particle<<< grid, THREADS_PER_BLOCK >>>(d_imgs, d_seeds, 
        (data->imax > 4) ? data->imax : 5, d_gb, data->uncertainties);
    cudaThreadSynchronize();
#ifdef DEBUG
    eee = getTime();
#endif
    checkError("k_particle kernel error");
#ifdef DEBUG
    printf("%d: Time to run the kernel = %f\n", data->card, eee-sss);
#endif

    //  Calculate uncertainties
    if (data->uncertainties) {
#ifdef DEBUG
      sss = getTime();
#endif
      k_sigma<<< grid, THREADS_PER_BLOCK >>>(d_gb, d_imgs, nimgs);
      cudaThreadSynchronize();
      checkError("k_sigma kernel error");
#ifdef DEBUG
      eee = getTime();
      printf("%d: Time to run uncertainties kernel = %f\n", data->card, eee-sss);
#endif
    }
    
    //  Copy the parameters out of global memory ignoring the garbage parameters
#ifdef DEBUG
    sss = getTime();
#endif
    cudaMemcpy(&out[data->start*(2*NUM_PARAMETERS+1)], d_gb, 
                (2*NUM_PARAMETERS+1)*nimgs*sizeof(float), cudaMemcpyDeviceToHost);
    checkError("Unable to copy output parameters from the device");
    cudaThreadSynchronize();
#ifdef DEBUG
    eee = getTime();
    printf("%d: Time to copy output back to host = %f\n", data->card, eee-sss);
#endif
    cudaFree(d_gb);
    checkError("Unable to free output parameters on the device");
    cudaFree(d_seeds);
    checkError("Unable to free PRNG seeds on the device");
    cudaFree(d_imgs);
    checkError("Unable to free input images on the device");

    return 0;
}


/**************************************************************
*  gauss2d
*
*  Entry point from IDL DLM code.
*/
extern "C" void gauss2d(unsigned short *imgs, int nimgs,   //  input images, assumed 11x11
                        unsigned int *seeds, int nseeds,   //  particle random number seeds
                        float *constraints,                //  particle constraints
                        int imax,                          //  number of iterations to perform
                        float *out,                        //  output fit parameters plus chi-square
                        int uncertainties) {               //  if true, calculate uncertainties
    pthread_t thing1, thing2;
    pthread_attr_t attr;
    thread_data_t data1, data2;
    int ncards, card1, card2;
    int rc;
#ifdef DEBUG
    double s,e;

    s = getTime();
#endif

    //  Decide which cards to use
    pickCards(&ncards, &card1, &card2);

    //  Set up the thread data
    data1.start = 0;
    data2.start = nimgs / 2;
    data1.end = data2.start - 1;
    data2.end = nimgs - 1;
    data1.nseeds = data2.nseeds = nseeds;
    data1.seeds = data2.seeds = seeds;
    data1.imgs = data2.imgs = imgs;
    data1.out = data2.out = out;
    data1.constraints = data2.constraints = constraints;
    data1.imax = data2.imax = imax;
    data1.uncertainties = data2.uncertainties = uncertainties;
    data1.card = card1;
    data2.card = card2;
    data1.ncards = data2.ncards = ncards;

    //  Ensure threads are joinable, if we will use them
    if (ncards > 1) {
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    }

    if (ncards == 1) {
        //  One card, no threads
        data1.end = nimgs - 1;
        cudaSetDevice(1);
        gauss2d_warp((void *)&data1);
    } else {
        //  Create the threads, each one processing one part of the image stack
        if ((rc=pthread_create(&thing1, &attr, gauss2d_warp, (void *)&data1))) {
            printf("Error: Unable to create thread 1: %d\n", rc);
        }
#ifdef DEBUG
    printf("Thing 1 started\n");
#endif
        if ((rc=pthread_create(&thing2, &attr, gauss2d_warp, (void *)&data2))) {
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

//  end gauss2d.cu

