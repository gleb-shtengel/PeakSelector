/*
*  file:  gpu_generate_volume.c
*
*  DLM code for GPU volume generation.
*
*  RTK, 14-Oct-2008
*  Last update:  21-Oct-2008
*/

#include <stdio.h>
#include <stdlib.h>
#include "idl_export.h"

/*  
*  Kernel and CUDA functions  
*/
extern void generate_volume(
    float *vol, int xdim, int ydim, int zdim, int nelem,
    float *peaks, int npeaks,
    float zscale, float df,
    float x_low, float x_high, 
    float y_low, float y_high, 
    float z_low, float z_high,
    int envelope);
extern void cuda_safe_init(void);

/*
*  Message block definitions
*/
static IDL_MSG_BLOCK msg_block;  

#define VOL_GEN_NO_MEMORY             0
#define VOL_GEN_NOT_SCALAR           -1
#define VOL_GEN_NOT_NUMERIC          -2
#define VOL_GEN_UNEQUAL_LENGTHS      -3
#define VOL_GEN_NOT_ARRAY            -4
#define VOL_GEN_MUST_BE_FLOAT        -5
#define VOL_GEN_MUST_BE_3D           -6
#define VOL_GEN_MUST_BE_MULT_6       -7

static IDL_MSG_DEF msg_arr[] = {  
  {"VOL_GEN_NO_MEMORY", "%NUnable to allocate memory"},
  {"VOL_GEN_NOT_SCALAR", "%NNot a scalar"},
  {"VOL_GEN_NOT_NUMERIC", "%NNot numeric"},
  {"VOL_GEN_UNEQUAL_LENGTHS", "%NVector lengths are not equal"},
  {"VOL_GEN_NOT_ARRAY", "%NNot an array"},
  {"VOL_GEN_MUST_BE_FLOAT", "%NMust be of type float"},
  {"VOL_GEN_MUST_BE_3D", "%NMust be 3D array"},
  {"VOL_GEN_MUST_BE_MULT_6", "%NMust be a multiple of six"},
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
*  idl_volgen
*/
static void idl_volgen(int argc, IDL_VPTR *argv) {
  int npeaks, doEnvelope;
  int xdim, ydim, zdim, nelem;
  float x_low, x_high, y_low, y_high, z_low, z_high;
  float *peaks = NULL;
  float *vol = NULL;
  float zscale, df, *p = NULL;

  /*  1st argument is the output volume array  */
  if (is_scalar(argv[0])) {
    IDL_MessageFromBlock(msg_block, VOL_GEN_NOT_ARRAY, IDL_MSG_LONGJMP);
  }
  if (!is_numeric(argv[0])) {
    IDL_MessageFromBlock(msg_block, VOL_GEN_NOT_NUMERIC, IDL_MSG_LONGJMP);
  }
  if (argv[0]->type != 4) {
    IDL_MessageFromBlock(msg_block, VOL_GEN_MUST_BE_FLOAT, IDL_MSG_LONGJMP);
  }
  if (argv[0]->value.arr->n_dim != 3) {
    IDL_MessageFromBlock(msg_block, VOL_GEN_MUST_BE_3D, IDL_MSG_LONGJMP);
  }
  vol = (float *)argv[0]->value.arr->data;
  nelem = (int)argv[0]->value.arr->n_elts;
  xdim = (int)argv[0]->value.arr->dim[0];
  ydim = (int)argv[0]->value.arr->dim[1];
  zdim = (int)argv[0]->value.arr->dim[2];

  /*  2nd argument is the peaks vector  */
  if (is_scalar(argv[1])) {
    IDL_MessageFromBlock(msg_block, VOL_GEN_NOT_ARRAY, IDL_MSG_LONGJMP);
  }
  if (!is_numeric(argv[1])) {
    IDL_MessageFromBlock(msg_block, VOL_GEN_NOT_NUMERIC, IDL_MSG_LONGJMP);
  }
  if (argv[1]->type != 4) {
    IDL_MessageFromBlock(msg_block, VOL_GEN_MUST_BE_FLOAT, IDL_MSG_LONGJMP);
  }
  if ((argv[1]->value.arr->n_elts % 6) != 0) {
    IDL_MessageFromBlock(msg_block, VOL_GEN_MUST_BE_MULT_6, IDL_MSG_LONGJMP);
  }
  npeaks = (int)argv[1]->value.arr->n_elts / 6;
  peaks = (float *)argv[1]->value.arr->data;
  
  /*  3rd argument is the z scale factor  */
  if (!is_scalar(argv[2])) {
    IDL_MessageFromBlock(msg_block, VOL_GEN_NOT_SCALAR, IDL_MSG_LONGJMP);
  }
  if (!is_numeric(argv[2])) {
    IDL_MessageFromBlock(msg_block, VOL_GEN_NOT_NUMERIC, IDL_MSG_LONGJMP);
  }
  if (argv[2]->type != 4) {
    IDL_MessageFromBlock(msg_block, VOL_GEN_MUST_BE_FLOAT, IDL_MSG_LONGJMP);
  }
  zscale = (float)get_numeric_value(argv[2]);

  /*  4th argument is the distance scale factor  */
  if (!is_scalar(argv[3])) {
    IDL_MessageFromBlock(msg_block, VOL_GEN_NOT_SCALAR, IDL_MSG_LONGJMP);
  }
  if (!is_numeric(argv[3])) {
    IDL_MessageFromBlock(msg_block, VOL_GEN_NOT_NUMERIC, IDL_MSG_LONGJMP);
  }
  df = (float)get_numeric_value(argv[3]);

  /*  5th argument is whether or not to do an envelope or a sum  */
  if (!is_scalar(argv[4])) {
    IDL_MessageFromBlock(msg_block, VOL_GEN_NOT_SCALAR, IDL_MSG_LONGJMP);
  }
  if (!is_numeric(argv[4])) {
    IDL_MessageFromBlock(msg_block, VOL_GEN_NOT_NUMERIC, IDL_MSG_LONGJMP);
  }
  doEnvelope = (int)get_numeric_value(argv[4]);

  /*  6th argument is the x range (nm)  */
  if (is_scalar(argv[5])) {
    IDL_MessageFromBlock(msg_block, VOL_GEN_NOT_ARRAY, IDL_MSG_LONGJMP);
  }
  if (!is_numeric(argv[5])) {
    IDL_MessageFromBlock(msg_block, VOL_GEN_NOT_NUMERIC, IDL_MSG_LONGJMP);
  }
  if (argv[5]->type != 4) {
    IDL_MessageFromBlock(msg_block, VOL_GEN_MUST_BE_FLOAT, IDL_MSG_LONGJMP);
  }
  p = (float *)argv[5]->value.arr->data;
  x_low = p[0];
  x_high = p[1];

  /*  7th argument is the y range (nm)  */
  if (is_scalar(argv[6])) {
    IDL_MessageFromBlock(msg_block, VOL_GEN_NOT_ARRAY, IDL_MSG_LONGJMP);
  }
  if (!is_numeric(argv[6])) {
    IDL_MessageFromBlock(msg_block, VOL_GEN_NOT_NUMERIC, IDL_MSG_LONGJMP);
  }
  if (argv[6]->type != 4) {
    IDL_MessageFromBlock(msg_block, VOL_GEN_MUST_BE_FLOAT, IDL_MSG_LONGJMP);
  }
  p = (float *)argv[6]->value.arr->data;
  y_low = p[0];
  y_high = p[1];
  
  /*  8th argument is the z range (nm)  */
  if (is_scalar(argv[7])) {
    IDL_MessageFromBlock(msg_block, VOL_GEN_NOT_ARRAY, IDL_MSG_LONGJMP);
  }
  if (!is_numeric(argv[7])) {
    IDL_MessageFromBlock(msg_block, VOL_GEN_NOT_NUMERIC, IDL_MSG_LONGJMP);
  }
  if (argv[7]->type != 4) {
    IDL_MessageFromBlock(msg_block, VOL_GEN_MUST_BE_FLOAT, IDL_MSG_LONGJMP);
  }
  p = (float *)argv[7]->value.arr->data;
  z_low = p[0];
  z_high = p[1];

  /*
  *  All data values read, call the kernel
  */
  generate_volume(vol, 
                  xdim, ydim, zdim, nelem,
                  peaks, npeaks,
                  zscale, df,
                  x_low, x_high, y_low, y_high, z_low, z_high,
                  doEnvelope);

  /*  
  *  Nothing to do since the IDL variable memory will be updated
  *  directly.
  */
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
  static IDL_SYSFUN_DEF2 procedure_addr[] = {  
    {{(IDL_SYSRTN_GENERIC) idl_volgen}, "GPU_GENERATE_VOLUME", 8, 8, 0, 0},
  };  

  /*  Error messages  */
  if (!(msg_block = IDL_MessageDefineBlock("VOL_GEN", IDL_CARRAY_ELTS(msg_arr), msg_arr))) {
    return IDL_FALSE;  
  }

  /*  Initialize CUDA when the module is loaded  */
  cuda_safe_init();

  return IDL_SysRtnAdd(procedure_addr, FALSE, IDL_CARRAY_ELTS(procedure_addr)); 
}

/*  Module description - make pulls this out to create particle_swarm_search.dlm  */
//dlm: MODULE GPU_GENERATE_VOLUME
//dlm: DESCRIPTION GPU based volume generation, HHMI
//dlm: VERSION 1.0
//dlm: PROCEDURE GPU_GENERATE_VOLUME 8 8

/*
*  end gpu_generate_volume.c
*/

