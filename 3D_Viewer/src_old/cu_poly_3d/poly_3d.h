//
//  file:  poly_3d.h
//
//  Common header between cu_poly_3d.c and poly_3d.cu
//
//  RTK, 27-Feb-2009
//  Last update:  03-Mar-2009
//
///////////////////////////////////////////////////////////////

#ifndef POLY_3D_H
#define POLY_3D_H

//  Maximum number of frames - scale to fit in GPU memory
//  but watch that P and Q still fit in constant memory as well
#define MAX_FRAMES  2000
#define PQ_MEMORY   (2*2*MAX_FRAMES)

#endif

