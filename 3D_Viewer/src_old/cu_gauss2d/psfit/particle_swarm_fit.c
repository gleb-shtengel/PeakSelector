/*
*  file:  particle_swarm_fit.c
*
*  DLM code for GPU version of the particle swarm curve fit.
*
*  RTK, 07-Oct-2008
*  Last update:  09-Oct-2008
*/

#include <stdio.h>
#include <stdlib.h>
#include "idl_export.h"

/*  
*  Kernel and CUDA functions  
*/
extern void psfit(float *x, float *y, float *w, int nsamp, int nparams, 
                  int npart, int imax, float tol, float *params);
extern void cuda_safe_init(void);

/*
*  Message block definitions
*/
static IDL_MSG_BLOCK msg_block;  

#define PSF_NO_MEMORY             0
#define PSF_NOT_SCALAR           -1
#define PSF_NOT_NUMERIC          -2
#define PSF_UNEQUAL_LENGTHS      -3
#define PSF_NOT_ARRAY            -4
#define PSF_MUST_BE_FLOAT        -5

static IDL_MSG_DEF msg_arr[] = {  
  {"PSO_NO_MEMORY", "%NUnable to allocate memory"},
  {"PSO_NOT_SCALAR", "%NNot a scalar"},
  {"PSO_NOT_NUMERIC", "%NNot numeric"},
  {"PSF_UNEQUAL_LENGTHS", "%NVector lengths are not equal"},
  {"PSF_NOT_ARRAY", "%NNot an array"},
  {"PSF_MUST_BE_FLOAT", "%NMust be of type float"},
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


/*************************************************************
*  get_numeric_value
*
*  Return the numeric value as a double.
*/
static double get_numeric_value(IDL_VPTR v) {

  switch (v->type) {
    case IDL_TYP_BYTE:    { return (double)v->value.c;     break; }
    case IDL_TYP_INT:     { return (double)v->value.i;     break; }
    case IDL_TYP_LONG:    { return (double)v->value.l;     break; }
    case IDL_TYP_FLOAT:   { return (double)v->value.f;     break; }
    case IDL_TYP_DOUBLE:  { return (double)v->value.d;     break; }
    case IDL_TYP_UINT:    { return (double)v->value.ui;    break; }
    case IDL_TYP_ULONG:   { return (double)v->value.ul;    break; }
    case IDL_TYP_LONG64:  { return (double)v->value.l64;   break; }
#ifdef WIN32
    case IDL_TYP_ULONG64: { return (double)(long)v->value.ul64; break; }
#else
    case IDL_TYP_ULONG64: { return (double)v->value.ul64;  break; }
#endif
    default: return (double)-1.0;
  }
}


/**************************************************************
*  idl_psf
*/
static IDL_VPTR idl_psf(int argc, IDL_VPTR *argv) {
  IDL_VPTR ans;
  IDL_MEMINT d;
  int i, nparams, npart, imax, nsamp;
  float tol;
  float *x, *y, *w, *params, *p;

  /*  1st argument is the x vector  */
  if (is_scalar(argv[0])) {
    IDL_MessageFromBlock(msg_block, PSF_NOT_ARRAY, IDL_MSG_LONGJMP);
  }
  if (!is_numeric(argv[0])) {
    IDL_MessageFromBlock(msg_block, PSF_NOT_NUMERIC, IDL_MSG_LONGJMP);
  }
  if (argv[0]->type != 4) {
    IDL_MessageFromBlock(msg_block, PSF_MUST_BE_FLOAT, IDL_MSG_LONGJMP);
  }

  nsamp = (int)argv[0]->value.arr->n_elts;
  x = (float *)malloc(nsamp*sizeof(float));
  if (!x) {
    IDL_MessageFromBlock(msg_block, PSF_NO_MEMORY, IDL_MSG_LONGJMP);
  }
  memcpy((void*)x, (void*)argv[0]->value.arr->data, nsamp*sizeof(float));

  /*  2nd argument is the y vector  */
  if (is_scalar(argv[1])) {
    IDL_MessageFromBlock(msg_block, PSF_NOT_ARRAY, IDL_MSG_LONGJMP);
  }
  if (!is_numeric(argv[1])) {
    IDL_MessageFromBlock(msg_block, PSF_NOT_NUMERIC, IDL_MSG_LONGJMP);
  }
  if (argv[1]->type != 4) {
    IDL_MessageFromBlock(msg_block, PSF_MUST_BE_FLOAT, IDL_MSG_LONGJMP);
  }
  if (nsamp != argv[1]->value.arr->n_elts) {
    IDL_MessageFromBlock(msg_block, PSF_UNEQUAL_LENGTHS, IDL_MSG_LONGJMP);
  }
  y = (float *)malloc(nsamp*sizeof(float));
  if (!y) {
    IDL_MessageFromBlock(msg_block, PSF_NO_MEMORY, IDL_MSG_LONGJMP);
  }
  memcpy((void*)y, (void*)argv[1]->value.arr->data, nsamp*sizeof(float));

  /*  3rd argument is the w vector  */
  if (is_scalar(argv[2])) {
    IDL_MessageFromBlock(msg_block, PSF_NOT_ARRAY, IDL_MSG_LONGJMP);
  }
  if (!is_numeric(argv[2])) {
    IDL_MessageFromBlock(msg_block, PSF_NOT_NUMERIC, IDL_MSG_LONGJMP);
  }
  if (argv[2]->type != 4) {
    IDL_MessageFromBlock(msg_block, PSF_MUST_BE_FLOAT, IDL_MSG_LONGJMP);
  }
  if (nsamp != argv[2]->value.arr->n_elts) {
    IDL_MessageFromBlock(msg_block, PSF_UNEQUAL_LENGTHS, IDL_MSG_LONGJMP);
  }
  w = (float *)malloc(nsamp*sizeof(float));
  if (!w) {
    IDL_MessageFromBlock(msg_block, PSF_NO_MEMORY, IDL_MSG_LONGJMP);
  }
  memcpy((void*)w, (void*)argv[2]->value.arr->data, nsamp*sizeof(float));

  /*  4th argument is the number of fit parameters  */
  if (!is_scalar(argv[3])) {
    IDL_MessageFromBlock(msg_block, PSF_NOT_SCALAR, IDL_MSG_LONGJMP);
  }
  if (!is_numeric(argv[3])) {
    IDL_MessageFromBlock(msg_block, PSF_NOT_NUMERIC, IDL_MSG_LONGJMP);
  }
  nparams = (int)get_numeric_value(argv[3]);

  /*  5th argument is the number of particles  */
  if (!is_scalar(argv[4])) {
    IDL_MessageFromBlock(msg_block, PSF_NOT_SCALAR, IDL_MSG_LONGJMP);
  }
  if (!is_numeric(argv[4])) {
    IDL_MessageFromBlock(msg_block, PSF_NOT_NUMERIC, IDL_MSG_LONGJMP);
  }
  npart = (int)get_numeric_value(argv[4]);

  /*  6th argument is the iteration limit  */
  if (!is_scalar(argv[5])) {
    IDL_MessageFromBlock(msg_block, PSF_NOT_SCALAR, IDL_MSG_LONGJMP);
  }
  if (!is_numeric(argv[5])) {
    IDL_MessageFromBlock(msg_block, PSF_NOT_NUMERIC, IDL_MSG_LONGJMP);
  }
  imax = (int)get_numeric_value(argv[5]);

  /*  7th argument is the tolerance  */
  if (!is_scalar(argv[6])) {
    IDL_MessageFromBlock(msg_block, PSF_NOT_SCALAR, IDL_MSG_LONGJMP);
  }
  if (!is_numeric(argv[6])) {
    IDL_MessageFromBlock(msg_block, PSF_NOT_NUMERIC, IDL_MSG_LONGJMP);
  }
  tol = (float)get_numeric_value(argv[6]);
  
  /*  Output fit parameter values  */
  params = (float *)malloc(nparams*sizeof(float));
  if (!params) {
    IDL_MessageFromBlock(msg_block, PSF_NO_MEMORY, IDL_MSG_LONGJMP);
  }

  /*
  *  All data values read, call the kernel.
  */
  psfit(x, y, w, nsamp, nparams, npart, imax, tol, params);

  /*  Return the parameter values  */
  d = (IDL_MEMINT)nparams;
  IDL_MakeTempArray(IDL_TYP_FLOAT, 1, &d, IDL_ARR_INI_NOP, &ans);
  p = (float *)ans->value.arr->data;
  for(i=0; i < nparams; i++)
    *p++ = params[i];
  
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

  /*  Function table  */
  static IDL_SYSFUN_DEF2 function_addr[] = {
    {{(IDL_FUN_RET) idl_psf}, "PARTICLE_SWARM_FIT", 7, 7, 0, 0},
  };

  /*  Error messages  */
  if (!(msg_block = IDL_MessageDefineBlock("PSF", IDL_CARRAY_ELTS(msg_arr), msg_arr))) {
    return IDL_FALSE;  
  }

  /*  Initialize CUDA when the module is loaded  */
  cuda_safe_init();

  return IDL_SysRtnAdd(function_addr, TRUE, IDL_CARRAY_ELTS(function_addr));
}

/*  Module description - make pulls this out to create particle_swarm_search.dlm  */
//dlm: MODULE PARTICLE_SWARM_FIT
//dlm: DESCRIPTION GPU based particle swarm curve fit
//dlm: VERSION 1.0
//dlm: FUNCTION PARTICLE_SWARM_FIT 7 7

/*
*  end particle_swarm_fit.c
*/

