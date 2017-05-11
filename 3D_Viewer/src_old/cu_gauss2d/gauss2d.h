/*
*  file:  gauss2d.h
*
*  Constants for gauss2d.cu
*
*  RTK, 29-Sep-2009
*  Last update:  24-Nov-2009
*/

#ifndef GAUSS2D_H
#define GAUSS2D_H

#include "cuda_runtime_api.h"
#include <stdio.h>
#include <stdlib.h>

#ifndef WIN32
#include <pthread.h>
#include <time.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#endif

//
//  Constants - N.B. implicitly assumes 11x11 input images
//

//  Maximum number of images to be fit at one time:
//      (11*11*2 + 256*4 + 7*4)*nimg <= MAX_GLOBAL_RAM
//          ^        ^      ^
//          |        |      +--  6 parameters + reduced chi-square for each image
//          |        +---------  256 unsigned int random number seeds per image
//          +------------------  11x11 unsigned short image dimensions
//
#ifdef TESLA_C1060
#define MAX_IMAGES   3319140
#else
#define MAX_IMAGES    829582
#endif

//  Number of columns in the grid
#define COLS  128

//  Tests show these values should work well when the search is constrained
#define THREADS_PER_BLOCK        256                     //  number of threads per block
#define PARTICLES_PER_IMAGE      THREADS_PER_BLOCK       //  tests show these values should work well

//  Fitting y = p0 + p1*exp(-0.5*(((x-p4)/p2)^2 + ((y-p5)/p3)^2))
#define NUM_PARAMETERS             6                     

//  For x, x_best, v, and function value at x_best
#define BYTES_PER_PARTICLE   (4*NUM_PARAMETERS*3+4)
#define BYTES_PER_IMAGE      (BYTES_PER_PARTICLE*PARTICLES_PER_IMAGE)

//  Number of input points in an 11x11 image
#define NUM_POINTS               121
#define IMGX                      11
#define IMGY                      11

//  Fixed swarm parameters
#define W       0.7
#define C1      1.0
#define C2      1.0
#define VMAX  1000.0

#endif

