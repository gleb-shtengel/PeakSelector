/*
*  file:  cu_poly_3d.c
*
*  DLM code for GPU poly_3d.
*
*  $Id: //depot/gsg/HHMI/Phase2/src/cu_poly_3d/cu_poly_3d.c#13 $
*  $Date: 2009/12/18 $
*  $Author: rkneusel $
*
*  RTK, 24-Feb-2009
*  Last update:  16-Nov-2009
*/

#include "cuda_runtime_api.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "idl_export.h"

#include "poly_3d.h"

/*  
*  Kernel and CUDA functions  
*/
extern void poly_3d(unsigned short *img, int nx, int ny, int nframes,
                    float *P, float *Q, unsigned short *out);
extern void cuda_safe_init(void);

/*
*  Message block definitions
*/
static IDL_MSG_BLOCK msg_block;  

#define POLY_3D_NO_MEMORY             0
#define POLY_3D_NOT_SCALAR           -1
#define POLY_3D_NOT_NUMERIC          -2
#define POLY_3D_UNEQUAL_LENGTHS      -3
#define POLY_3D_NOT_ARRAY            -4
#define POLY_3D_MUST_BE_FLOAT        -5
#define POLY_3D_MUST_BE_INT          -6
#define POLY_3D_MUST_BE_3D           -7
#define POLY_3D_UNEQUAL_FRAMES       -8
#define POLY_3D_TOO_MANY_FRAMES      -9

static IDL_MSG_DEF msg_arr[] = {  
  {"POLY_3D_NO_MEMORY", "%NUnable to allocate memory"},
  {"POLY_3D_NOT_SCALAR", "%NNot a scalar"},
  {"POLY_3D_NOT_NUMERIC", "%NNot numeric"},
  {"POLY_3D_UNEQUAL_LENGTHS", "%NVector lengths are not equal"},
  {"POLY_3D_NOT_ARRAY", "%NNot an array"},
  {"POLY_3D_MUST_BE_FLOAT", "%NMust be of type float"},
  {"POLY_3D_MUST_BE_INT", "%NMust be of type long"},
  {"POLY_3D_MUST_BE_3D", "%NMust be 3D"},
  {"POLY_3D_UNEQUAL_FRAMES", "%NThe number of frames must match"},
  {"POLY_3D_TOO_MANY_FRAMES", "%NToo many frames"},
};


/**************************************************************
*  is_scalar
*
*  Returns TRUE if the IDL variable is a scalar.
*/
static char is_scalar(IDL_VPTR v) {

  return (!(v->flags & IDL_V_ARR));
}


/**************************************************************
*  is_numeric
*
*  Returns TRUE if the IDL variable is a numeric type (not complex).
*
*/
static char is_numeric(IDL_VPTR v) {

  if ((v->type == IDL_TYP_BYTE)   ||
      (v->type == IDL_TYP_INT)    ||
      (v->type == IDL_TYP_LONG)   ||
      (v->type == IDL_TYP_FLOAT)  ||
      (v->type == IDL_TYP_DOUBLE) ||
      (v->type == IDL_TYP_UINT)   ||
      (v->type == IDL_TYP_ULONG)  ||
      (v->type == IDL_TYP_LONG64) ||
      (v->type == IDL_TYP_ULONG64))
    return TRUE;
    
  return FALSE;
}


/**************************************************************
*  freeHostCB
*/
void freeHostCB(UCHAR *p) {
  cudaFreeHost(p);
}


/**************************************************************
*  idl_poly3d
*/
static IDL_VPTR idl_poly3d(int argc, IDL_VPTR *argv) {
  IDL_VPTR ans;
  unsigned short *img;
  unsigned int totalMem;
  UCHAR *out;
  IDL_MEMINT dims[3];
  int xdim, ydim, nframes, ncards, segsize, nseg, nelem, k, idx, n;
  int max_frames;
  float *P, *Q;
  struct cudaDeviceProp info;

  /*  1st argument is the input image  */
  if (is_scalar(argv[0])) {
    IDL_MessageFromBlock(msg_block, POLY_3D_NOT_ARRAY, IDL_MSG_LONGJMP);
  }
  if (!is_numeric(argv[0])) {
    IDL_MessageFromBlock(msg_block, POLY_3D_NOT_NUMERIC, IDL_MSG_LONGJMP);
  }
  if (argv[0]->type != IDL_TYP_INT) {
    IDL_MessageFromBlock(msg_block, POLY_3D_MUST_BE_INT, IDL_MSG_LONGJMP);
  }
  if (argv[0]->value.arr->n_dim != 3) {
    IDL_MessageFromBlock(msg_block, POLY_3D_MUST_BE_3D, IDL_MSG_LONGJMP);
  }
  img = (unsigned short *)argv[0]->value.arr->data;
  xdim = (int)argv[0]->value.arr->dim[0];
  ydim = (int)argv[0]->value.arr->dim[1];
  nframes = (int)argv[0]->value.arr->dim[2];

  /*  2nd argument is the P stack of 2x2 matrices  */
  if (is_scalar(argv[1])) {
    IDL_MessageFromBlock(msg_block, POLY_3D_NOT_ARRAY, IDL_MSG_LONGJMP);
  }
  if (!is_numeric(argv[1])) {
    IDL_MessageFromBlock(msg_block, POLY_3D_NOT_NUMERIC, IDL_MSG_LONGJMP);
  }
  if (argv[1]->type != IDL_TYP_FLOAT) {
    IDL_MessageFromBlock(msg_block, POLY_3D_MUST_BE_FLOAT, IDL_MSG_LONGJMP);
  }
  if (argv[1]->value.arr->n_dim != 3) {
    IDL_MessageFromBlock(msg_block, POLY_3D_MUST_BE_3D, IDL_MSG_LONGJMP);
  }
  P = (float *)argv[1]->value.arr->data;
  
  /*  3rd argument is the Q 2x2 matrix  */
  if (is_scalar(argv[2])) {
    IDL_MessageFromBlock(msg_block, POLY_3D_NOT_ARRAY, IDL_MSG_LONGJMP);
  }
  if (!is_numeric(argv[2])) {
    IDL_MessageFromBlock(msg_block, POLY_3D_NOT_NUMERIC, IDL_MSG_LONGJMP);
  }
  if (argv[2]->type != IDL_TYP_FLOAT) {
    IDL_MessageFromBlock(msg_block, POLY_3D_MUST_BE_FLOAT, IDL_MSG_LONGJMP);
  }
  if (argv[2]->value.arr->n_dim != 3) {
    IDL_MessageFromBlock(msg_block, POLY_3D_MUST_BE_3D, IDL_MSG_LONGJMP);
  }
  Q = (float *)argv[2]->value.arr->data;

  //  Number of frames must match for all arguments
  if ((nframes != (int)argv[1]->value.arr->dim[2]) ||
      (nframes != (int)argv[2]->value.arr->dim[2])) {
    IDL_MessageFromBlock(msg_block, POLY_3D_UNEQUAL_FRAMES, IDL_MSG_LONGJMP);
  }

  //  Get the number of cards in the system - this determines the number of
  //  frames that can be processed at once.  If 1 => MAX_FRAMES, if > 1 
  //  => 2*MAX_FRAMES
  cudaGetDeviceCount(&ncards);
  ncards = (ncards > 2) ? 2 : ncards;

  /*
  *  Make the output image, same size as the input
  */
  //  Pinned memory not good here - cudaMallocHost((void **)&out, xdim*ydim*nframes*sizeof(unsigned short));
  out = (unsigned short *)malloc(xdim*ydim*nframes*sizeof(unsigned short));
  
  dims[0] = xdim;
  dims[1] = ydim;
  dims[2] = nframes;
  ans = IDL_ImportArray(3, dims, IDL_TYP_UINT, out,
            //(IDL_ARRAY_FREE_CB)freeHostCB, NULL);
            (IDL_ARRAY_FREE_CB)NULL, NULL);

  /*
  *  Call the kernel repeatedly until all frames processed
  */
  cudaGetDeviceProperties(&info,0);
  totalMem = info.totalGlobalMem; //(unsigned int)floor(0.95*info.totalGlobalMem);  
  nelem = xdim*ydim;
  max_frames = (totalMem - 2*nelem - 512) / (8*nelem);
  max_frames = (max_frames > MAX_FRAMES) ? MAX_FRAMES : max_frames;
  segsize = ncards*max_frames;
  nseg = 1 + nframes/segsize;

  for(k=0; k < nseg; k++) {
    idx = k*segsize;
    n = nframes - k*segsize;
    n = (n > segsize) ? segsize : n;

    if (n != 0) {
      poly_3d(img+nelem*idx, xdim, ydim, n, P+2*2*idx, Q+2*2*idx, 
              (unsigned short *)out+nelem*idx);
    }
  }

  //  Output already filled in
  return ans;
}


/*************************************************************
*  IDL_Load
*
*  Tell IDL what functions are to be added to the system.
*/
#ifndef WIN32
#define IDL_CDECL
#endif
#ifdef WIN32
__declspec(dllexport)
#endif
int IDL_CDECL IDL_Load(void) {

  /*  Procedure table  */
  static IDL_SYSFUN_DEF2 function_addr[] = {  
    {{(IDL_FUN_RET) idl_poly3d}, "CU_POLY_3D", 3, 3, 0, 0},
  };  

  /*  Error messages  */
  if (!(msg_block = IDL_MessageDefineBlock("POLY3D", IDL_CARRAY_ELTS(msg_arr), msg_arr))) {
    return IDL_FALSE;  
  }

  /*  Initialize CUDA when the module is loaded  */
  cuda_safe_init();

  return IDL_SysRtnAdd(function_addr, TRUE, IDL_CARRAY_ELTS(function_addr)); 
}

/*  Module description - make pulls this out to create the DLM file  */
//dlm: MODULE CU_POLY_3D
//dlm: DESCRIPTION CUDA poly_3d, HHMI
//dlm: VERSION 1.0
//dlm: FUNCTION CU_POLY_3D 3 3

/*
*  end cu_poly_3d.c
*/

