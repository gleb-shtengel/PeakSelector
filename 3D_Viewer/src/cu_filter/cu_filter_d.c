/*
*  file:  cu_filter_d.c
*
*  Double precision DLM code for GPU GroupFilterIt and FilterIt.
*
*  $Id: //depot/gsg/HHMI/Phase2/src/cu_filter/cu_filter_d.c#1 $
*  $Date: 2009/03/10 $
*  $Author: rkneusel $
*
*  RTK, 06-Mar-2009
*  Last update:  10-Mar-2009
*/

#include "cuda_runtime_api.h"
#include <stdio.h>
#include <stdlib.h>
#include "idl_export.h"

/*  
*  Kernel and CUDA functions  
*/
extern void gpfilter(double *gp, int npeaks, float *params, 
                     unsigned char *out);
extern void filter(double *gp, int npeaks, float *params, 
                   unsigned char *out);
extern void cuda_safe_init(void);

/*
*  Message block definitions
*/
static IDL_MSG_BLOCK msg_block;  

#define CU_FILTER_NO_MEMORY             0
#define CU_FILTER_NOT_SCALAR           -1
#define CU_FILTER_NOT_NUMERIC          -2
#define CU_FILTER_UNEQUAL_LENGTHS      -3
#define CU_FILTER_NOT_ARRAY            -4
#define CU_FILTER_MUST_BE_FLOAT        -5
#define CU_FILTER_MUST_BE_INT          -6
#define CU_FILTER_MUST_BE_3D           -7
#define CU_FILTER_UNEQUAL_FRAMES       -8
#define CU_FILTER_TOO_MANY_FRAMES      -9

static IDL_MSG_DEF msg_arr[] = {  
  {"CU_FILTER_NO_MEMORY", "%NUnable to allocate memory"},
  {"CU_FILTER_NOT_SCALAR", "%NNot a scalar"},
  {"CU_FILTER_NOT_NUMERIC", "%NNot numeric"},
  {"CU_FILTER_UNEQUAL_LENGTHS", "%NVector lengths are not equal"},
  {"CU_FILTER_NOT_ARRAY", "%NNot an array"},
  {"CU_FILTER_MUST_BE_FLOAT", "%NMust be of type float"},
  {"CU_FILTER_MUST_BE_INT", "%NMust be of type long"},
  {"CU_FILTER_MUST_BE_3D", "%NMust be 3D"},
  {"CU_FILTER_UNEQUAL_FRAMES", "%NThe number of frames must match"},
  {"CU_FILTER_TOO_MANY_FRAMES", "%NToo many frames"},
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
*  idl_gpfilter
*/
static IDL_VPTR idl_gpfilter(int argc, IDL_VPTR *argv) {
  IDL_VPTR ans;
  double *gp;
  float *params;
  UCHAR *out;
  IDL_MEMINT dims[1];
  int npeaks;

  /*  1st argument is CGroupParams  */
  if (is_scalar(argv[0])) {
    IDL_MessageFromBlock(msg_block, CU_FILTER_NOT_ARRAY, IDL_MSG_LONGJMP);
  }
  if (!is_numeric(argv[0])) {
    IDL_MessageFromBlock(msg_block, CU_FILTER_NOT_NUMERIC, IDL_MSG_LONGJMP);
  }
  if (argv[0]->type != IDL_TYP_DOUBLE) {
    IDL_MessageFromBlock(msg_block, CU_FILTER_MUST_BE_FLOAT, IDL_MSG_LONGJMP);
  }
  gp = (double *)argv[0]->value.arr->data;
  npeaks = (int)argv[0]->value.arr->dim[1];

  /*  2nd argument is ParamLimits  */
  if (is_scalar(argv[1])) {
    IDL_MessageFromBlock(msg_block, CU_FILTER_NOT_ARRAY, IDL_MSG_LONGJMP);
  }
  if (!is_numeric(argv[1])) {
    IDL_MessageFromBlock(msg_block, CU_FILTER_NOT_NUMERIC, IDL_MSG_LONGJMP);
  }
  if (argv[1]->type != IDL_TYP_FLOAT) {
    IDL_MessageFromBlock(msg_block, CU_FILTER_MUST_BE_FLOAT, IDL_MSG_LONGJMP);
  }
  params = (float *)argv[1]->value.arr->data;

  /*
  *  Make the output image, same size as the input
  */
  cudaMallocHost((void **)&out, npeaks*sizeof(unsigned char));
  
  dims[0] = npeaks;
  ans = IDL_ImportArray(1, dims, IDL_TYP_BYTE, out,
            (IDL_ARRAY_FREE_CB)freeHostCB, NULL);

  /*
  *  Call the kernel
  */
  gpfilter(gp, npeaks, params, (unsigned char *)out);
  return ans;
}


/**************************************************************
*  idl_filter
*/
static IDL_VPTR idl_filter(int argc, IDL_VPTR *argv) {
  IDL_VPTR ans;
  double *gp;
  float *params;
  UCHAR *out;
  IDL_MEMINT dims[1];
  int npeaks;

  /*  1st argument is CGroupParams  */
  if (is_scalar(argv[0])) {
    IDL_MessageFromBlock(msg_block, CU_FILTER_NOT_ARRAY, IDL_MSG_LONGJMP);
  }
  if (!is_numeric(argv[0])) {
    IDL_MessageFromBlock(msg_block, CU_FILTER_NOT_NUMERIC, IDL_MSG_LONGJMP);
  }
  if (argv[0]->type != IDL_TYP_DOUBLE) {
    IDL_MessageFromBlock(msg_block, CU_FILTER_MUST_BE_FLOAT, IDL_MSG_LONGJMP);
  }
  gp = (double *)argv[0]->value.arr->data;
  npeaks = (int)argv[0]->value.arr->dim[1];

  /*  2nd argument is ParamLimits  */
  if (is_scalar(argv[1])) {
    IDL_MessageFromBlock(msg_block, CU_FILTER_NOT_ARRAY, IDL_MSG_LONGJMP);
  }
  if (!is_numeric(argv[1])) {
    IDL_MessageFromBlock(msg_block, CU_FILTER_NOT_NUMERIC, IDL_MSG_LONGJMP);
  }
  if (argv[1]->type != IDL_TYP_FLOAT) {
    IDL_MessageFromBlock(msg_block, CU_FILTER_MUST_BE_FLOAT, IDL_MSG_LONGJMP);
  }
  params = (float *)argv[1]->value.arr->data;

  /*
  *  Make the output image, same size as the input
  */
  cudaMallocHost((void **)&out, npeaks*sizeof(unsigned char));
  
  dims[0] = npeaks;
  ans = IDL_ImportArray(1, dims, IDL_TYP_BYTE, out,
            (IDL_ARRAY_FREE_CB)freeHostCB, NULL);

  /*
  *  Call the kernel
  */
  filter(gp, npeaks, params, (unsigned char *)out);
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
    {{(IDL_FUN_RET) idl_gpfilter}, "CU_GROUPFILTERIT_D", 2, 2, 0, 0},
    {{(IDL_FUN_RET) idl_filter}, "CU_FILTERIT_D", 2, 2, 0, 0},
  };  

  /*  Error messages  */
  if (!(msg_block = IDL_MessageDefineBlock("FILTER_D", IDL_CARRAY_ELTS(msg_arr), msg_arr))) {
    return IDL_FALSE;  
  }

  /*  Initialize CUDA when the module is loaded  */
  cuda_safe_init();

  return IDL_SysRtnAdd(function_addr, TRUE, IDL_CARRAY_ELTS(function_addr)); 
}

/*  Module description - make pulls this out to create the DLM file  */
//dlm: MODULE CU_FILTER_D
//dlm: DESCRIPTION CUDA GroupFilterIt and FilterIt (double), HHMI
//dlm: VERSION 1.0
//dlm: FUNCTION CU_GROUPFILTERIT_D 2 2
//dlm: FUNCTION CU_FILTERIT_D 2 2

/*
*  end cu_filter_d.c
*/

