/*
*  file:  psfit.cu
*
*  Host and device code for the particle swarm curve fit.
*
*  RTK, 07-Oct-2008
*  Last update:  10-Oct-2008
*/

#include <cuda.h>
#include <math.h>
#include <time.h>
#include <stdio.h>

//
//  Memory:
//

#define MAX_SAMPLES  5400
#define MAX_PARAMS     25

/*  x, y, and w vectors  */
__device__ __constant__ float d_x[MAX_SAMPLES];
__device__ __constant__ float d_y[MAX_SAMPLES];
__device__ __constant__ float d_w[MAX_SAMPLES];
float *h_x;
float *h_y;
float *h_w;

/*  Parameters - device constant memory  */
__device__ __constant__ float d_params[4];
float h_params[4];

/*  Global best and function value  */
__device__ __constant__ float d_global[MAX_PARAMS+1];
float *h_global;

/*  Particle positions  */
float *d_particles;  
float *h_particles;

/*  Particle velocities  */
float *d_velocities;
float *h_velocities;

/*  Random vectors  */
float *d_r1;  
float *d_r2;  
float *h_r1;
float *h_r2;

/*  Particle best positions  */
float *d_best;
float *h_best;

//
//  Device code:
//

/**************************************************************
*
*  psfit_function
*
*  Device code to implement the function to be fit.  The user
*  must supply this function.
*
*  Args:
*    x - sample value
*   *p - vector of parameters (current particle position)
*
*  Returns:
*    A scalar value.  The function value for that x position using
*    the current particle position for the parameters.  This value is
*    compared to the given y value for the given x.
*/
__device__ float psfit_function(float x, float *p) {
  return p[0]*sinf(p[1]*x) + p[2]*expf(-(x-p[3])*(x-p[3])/p[4]) + p[5];
}


/**************************************************************
*  step
*
*  Step the particles once.
*/
__global__ void step(float *d_r1, float *d_r2, float *d_particles, 
                     float *d_velocities, float *d_best,
                     int nparams, int nsamp, int nparticles) {
  float f,w,c1,c2,*r1,*r2,*x,*xb,*v;
  float s,yf;
  int i, tid;

  /*  Get parameters  */
  tid = threadIdx.x + blockIdx.x*blockDim.x;

  r1 = &d_r1[nparams*tid];
  r2 = &d_r2[nparams*tid];
  x  = &d_particles[nparams*tid];
  xb = &d_best[(nparams+1)*tid];
  v  = &d_velocities[nparams*tid];
  w  = d_params[0];
  c1 = d_params[1];
  c2 = d_params[2];
  s  = d_params[3];  //  sum of the weights, d_w

  /*  Calculate a new position and velocity  */
  for(i=0; i < nparams; i++) {
    x[i] = x[i] + v[i];
    v[i] = w*v[i] + c1*r1[i]*(xb[1+i]-x[i]) + c2*r2[i]*(d_global[1+i]-x[i]);
    if (v[i] < -10.0)  v[i] = -10.0;
    if (v[i] > +10.0)  v[i] = +10.0;
  }

  /*  Calculate the function value at the new position  */
  f = 0.0;
  for(i=0; i < nsamp; i++) {
    yf = psfit_function(d_x[i], x);
    f += d_w[i]*(yf - d_y[i])*(yf - d_y[i]);
  }
  f = sqrt(f/s);

  /*  Check for new personal best  */
  if (f < xb[0]) {
    xb[0] = f;
    for(i=0; i < nparams; i++) {
      xb[1+i] = x[i];
    }
  }
}


//
//  Host code:
//

float h_function(float x, float *p) {
  return p[0]*sin(p[1]*x) + p[2]*exp(-(x-p[3])*(x-p[3])/p[4]) + p[5];
}

/**************************************************************
*  h_step
*
*  Step the particles once.
*/
void h_step(float *h_r1, float *h_r2, float *h_particles, 
                     float *h_velocities, float *h_best, int nparams,
                     int nsamp, int nparticles) {
  float f,w,c1,c2,*r1,*r2,*x,*xb,*v;
  float s,yf;
  int i, tid;

  printf("in h_step\n");

  /*  Get parameters  */
  for(tid=0; tid < nparticles; tid++) {
      r1 = &h_r1[nparams*tid];
      r2 = &h_r2[nparams*tid];
      x  = &h_particles[nparams*tid];
      xb = &h_best[(nparams+1)*tid];
      v  = &h_velocities[nparams*tid];
      w  = h_params[0];
      c1 = h_params[1];
      c2 = h_params[2];
      s  = h_params[3];  //  sum of the weights, h_w

      /*  Calculate a new position and velocity  */
      for(i=0; i < nparams; i++) {
        x[i] = x[i] + v[i];
        v[i] = w*v[i] + c1*r1[i]*(xb[1+i]-x[i]) + c2*r2[i]*(h_global[1+i]-x[i]);
      }

      /*  Calculate the function value at the new position  */
      f = 0.0;
      for(i=0; i < nsamp; i++) {
        yf = h_function(h_x[i], x);
        f += h_w[i]*(yf - h_y[i])*(yf - h_y[i]);
      }
      f /= s;

      /*  Check for new personal best  */
      if (f < xb[0]) {
        xb[0] = f;
        for(i=0; i < nparams; i++) {
          xb[1+i] = x[i];
        }
      }
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
*  randfp
*
*  Randfp float, [0..1)
*/
float randfp() {
  float r = ((float)rand())/((float)RAND_MAX+1.0);
  return r;
}


/**************************************************************
*  random_value
*
*  A random value in the given range.
*/
float random_value(float min, float max) {
  return min + randfp()*(max-min);
}


/**************************************************************
*  assign_random_vectors
*/
void assign_random_vectors(int np, float *r1, float *r2, int nparams) {
  int i;
  float *p = r1;
  float *q = r2;
  
  for(i=0; i < nparams*np; i++) {
    *p++ = randfp();
    *q++ = randfp();
  }
}


/**************************************************************
*  update_global_best
*
*  Scan the particle bests to find the new global best.
*/
void update_global_best(int np, int nparams, float *h_best, float *h_global) {
  int i,j;
  int bsize = nparams+1;

  for(i=0; i < np; i++) {
    if (h_best[bsize*i] < h_global[0]) {
      h_global[0] = h_best[bsize*i];
      for(j=0; j < nparams; j++) {
        h_global[1+j] = h_best[bsize*i+1+j];
      }
    }
  }

  cudaMemcpyToSymbol(d_global, h_global, bsize*sizeof(float));
}


/**************************************************************
*  psfit
*
*  Entry point from IDL DLM code.
*/
extern "C" void psfit(float *x, float *y, float *w, int nsamp, int nparams, 
                      int npart, int imax, float tol, float *params) {
  int i, j, nparticles, nBlocks;
  
  /*  Seed the random number generator  */
  srand(time(NULL));
  
  /*  Set the actual number of particles and blocks to use  */
  nparticles = 256*(npart/256 + 1);
  nparticles = (nparticles > 65535*256) ? 65535*256 : nparticles;
  nBlocks = nparticles/256;
  dim3 threads(256,1);
  dim3 grid(nBlocks,1);

  /*  Copy x, y, and w to device memory  */
  cudaMemcpyToSymbol(d_x, x, nsamp*sizeof(float));
  cudaMemcpyToSymbol(d_y, y, nsamp*sizeof(float));
  cudaMemcpyToSymbol(d_w, w, nsamp*sizeof(float));
  h_x = x;
  h_y = y;
  h_w = w;
  
  /*  Particle positions  */
  cudaMalloc((void **)&d_particles, nparticles*nparams*sizeof(float));
  h_particles = (float *)malloc(nparticles*nparams*sizeof(float));

  /*  Particle velocities  */
  cudaMalloc((void**)&d_velocities, nparticles*nparams*sizeof(float));
  h_velocities = (float *)malloc(nparticles*nparams*sizeof(float));

  /*  Random vectors  */
  cudaMalloc((void **)&d_r1, nparticles*nparams*sizeof(float));
  cudaMalloc((void **)&d_r2, nparticles*nparams*sizeof(float));
  h_r1 = (float *)malloc(nparticles*nparams*sizeof(float));
  h_r2 = (float *)malloc(nparticles*nparams*sizeof(float));

  /*  Particle best positions and function value  */
  cudaMalloc((void **)&d_best, nparticles*(nparams+1)*sizeof(float));
  h_best = (float *)malloc(nparticles*(nparams+1)*sizeof(float));

  /*  Global best position and function value  */
  //cudaMalloc((void **)&d_global, (nparams+1)*sizeof(float));
  h_global = (float *)malloc((nparams+1)*sizeof(float));

  /*  Set up initial positions and velocities  */
  for(i=0; i < nparticles; i++) {

    /*  Initial velocity always zero  */
    for(j=0; j < nparams; j++) {
      h_velocities[nparams*i + j] = 0.0;
    }

    /*  Initial position */
    for(j=0; j < nparams; j++) {
      h_particles[nparams*i + j] = random_value(0.0, 1.0);
    }
    
    /*  Initial "best" - save the function call on the CPU  */
    h_best[(nparams+1)*i] = (float)1e12;
    for(j=0; j < nparams; j++) {
      h_best[(nparams+1)*i + 1 + j] = h_particles[nparams*i + j];
    }
  }
  cudaMemcpy(d_particles, h_particles, nparticles*nparams*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_velocities, h_velocities, nparticles*nparams*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_best, h_best, nparticles*(nparams+1)*sizeof(float), cudaMemcpyHostToDevice);

  /*  Pick an initial best  */
  h_global[0] = (float)1e18;
  for(j=0; j < nparams; j++) {
    h_global[1+j] = h_particles[j];
  }
  cudaMemcpyToSymbol(d_global, h_global, (nparams+1)*sizeof(float));

  /*  Set w, c1, c2  */
  h_params[0] = 0.9;
  h_params[1] = 2.0;
  h_params[2] = 2.0;
  h_params[3] = 0.0;
  for(i=0; i < nsamp; i++) {
    h_params[3] += w[i];
  }
  cudaMemcpyToSymbol(d_params, h_params, 4*sizeof(float));

  /*
  *  Loop through the iterations, convergence test coming later.
  */
  for(i=0; i < imax; i++) {

    /*  Set up the random vectors for this step  */
    assign_random_vectors(nparticles, h_r1, h_r2, nparams);
    cudaMemcpy(d_r1, h_r1, nparticles*nparams*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r2, h_r2, nparticles*nparams*sizeof(float), cudaMemcpyHostToDevice);

    /*  Step all the particles on the GPU  */
    step<<< grid, threads >>>(d_r1, d_r2, d_particles, d_velocities,
                              d_best, nparams, nsamp, nparticles);
    cudaThreadSynchronize();
    //h_step(h_r1, h_r2, h_particles, h_velocities, h_best, nparams, nsamp, nparticles);

    /*  Read back all the current best positions  */
    cudaMemcpy(h_best, d_best, nparticles*(nparams+1)*sizeof(float), cudaMemcpyDeviceToHost);

    /*  Find the smallest and update the global best in constant memory, and
        keep a local copy  */
    update_global_best(nparticles, nparams, h_best, h_global);

    /*  Check if tolerance met  */
    if (h_global[0] < tol) {
      printf("tolerance met, exiting\n");
      break;
    }
  }
  
  /*  Copy the global best position to the output array  */
  for(i=0; i < nparams; i++) {
    params[i] = h_global[1+i];
  }

  /*  Clean up  */
  cudaThreadSynchronize();

  cudaFree(d_particles);
  cudaFree(d_velocities);
  cudaFree(d_r1);
  cudaFree(d_r2);
  cudaFree(d_best);
  
  free(h_particles);
  free(h_velocities);
  free(h_r1);
  free(h_r2);
  free(h_best);
  free(h_global);
}


/**************************************************************
*  main
*
*  Driver for testing outside of IDL.
*/
int main() {
  float x[100];
  float y[100];
  float w[100];
  float p[6];
  int i;

  //  Sample function, y = 2*x^2 + 3*x + 4
  for(i=0; i < 100; i++) {
    x[i] = -10 + 20.0*i/100.0;
    //y[i] = 2*x[i]*x[i] + 3*x[i] + 4;
    y[i] = 2*sin(3.3*x[i]) + 30*exp(-(x[i]-5.5)*(x[i]-5.5)/6.6) + 15.6;
    w[i] = 1.0;
  }

  //  Call do the fit
  psfit(x,y,w,100,6,3000,2000,1e-26,p);

  //  Display the results
  printf("p[0] = %f\n", p[0]);
  printf("p[1] = %f\n", p[1]);
  printf("p[2] = %f\n", p[2]);
  printf("p[3] = %f\n", p[3]);
  printf("p[4] = %f\n", p[4]);
  printf("p[5] = %f\n", p[5]);
  
  return 0;
}

/*
*  end psfit.cu
*/

