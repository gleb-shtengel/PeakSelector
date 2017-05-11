/*
*  file:  gpu_device_query.c
*
*  Query the system for stats on CUDA enabled GPUs.
*
*  RTK, 13-Oct-2008
*  Last update:  13-Oct-2008
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <cuda_runtime_api.h>
#include "idl_export.h"

#define ARRLEN(arr) (sizeof(arr)/sizeof(arr[0]))

/*
*  Message block definitions
*/
static IDL_MSG_BLOCK msg_block;  

#define GDQ_NO_MEMORY             0
#define GDQ_NOT_SCALAR           -1
#define GDQ_NOT_NUMERIC          -2
#define GDQ_NOT_STRING           -3
#define GDQ_NO_PROP              -4
#define GDQ_NO_DEVICE            -5

static IDL_MSG_DEF msg_arr[] = {  
  {"GDQ_NO_MEMORY", "%NUnable to allocate memory"},
  {"GDQ_NOT_SCALAR", "%NNot a scalar"},
  {"GDQ_NOT_NUMERIC", "%NNot numeric"},
  {"GDQ_NOT_STRING", "%NNot a string"},
  {"GDQ_NO_PROP", "%NNo such property"},
  {"GDQ_NO_DEVICE", "%NNo such device"},
};

/*  Property names  */
static char *names[] = {     //  struct cudaDeviceProp {
  "DEVICE_NAME",             //    char name[256];
  "GLOBAL_MEMORY",           //    size_t totalGlobalMem;
  "SHARED_MEMORY",           //    size_t sharedMemPerBlock;
  "REGISTERS",               //    int regsPerBlock;
  "WARP_SIZE",               //    int warpSize;
  "MEMORY_PITCH",            //    size_t memPitch;
  "THREADS_PER_BLOCK",       //    int maxThreadsPerBlock;
  "THREADS_X",               //    int maxThreadsDim[3];
  "THREADS_Y",               //
  "THREADS_Z",               //
  "GRID_X",                  //    int maxGridSize[3];
  "GRID_Y",                  //
  "GRID_Z",                  //
  "CONSTANT_MEMORY",         //    size_t totalConstMem;
  "MAJOR_REVISION",          //    int major;
  "MINOR_REVISION",          //    int minor;
  "CLOCK_RATE",              //    int clock_rate;  // in kiloHertz
  "TEXTURE_ALIGNMENT",       //    size_t textureAlignment;
  "DEVICE_OVERLAP",          //    int deviceOverlap;
  "MULTIPROCESSOR_COUNT"     //    multiProcessorCount;
};                           //  };

/*  Device count  */
int gDevices = 0;

/*  Device properties  */
#define MAX_DEVICES  4
struct cudaDeviceProp gProperties[MAX_DEVICES];

/**************************************************************
*  query_devices
*
*  Query for CUDA enabled devices and load the properties
*  of those found.
*/
static void query_devices() {
  int i;
   
  /*  Device count  */
  cudaGetDeviceCount(&gDevices);

  /*  Clear device properties  */
  memset((void*)gProperties, 0, MAX_DEVICES*sizeof(struct cudaDeviceProp));

  /*  Read device properties  */
  for(i=0; i < gDevices; i++) {
    cudaGetDeviceProperties(&gProperties[i], i);
  }
}


/**************************************************************
*  is_scalar
*
*  Returns TRUE if the IDL variable is a scalar.
*/
static char is_scalar(IDL_VPTR v) {

  return (!(v->flags & IDL_V_ARR));
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
*  upcase
*
*  Make the given string uppercase.
*/
static char *upcase(char *s) {
  char *t = s;
  
  for(; *t != '\0'; t++)
    *t = toupper(*t);
  return s;
}


/**************************************************************
*  idl_dev_count
*/
static IDL_VPTR idl_dev_count(int argc, IDL_VPTR *argv) {
  return IDL_GettmpMEMINT((IDL_MEMINT)gDevices);
}


/**************************************************************
*  idl_dev_prop
*/
static IDL_VPTR idl_dev_prop(int argc, IDL_VPTR *argv) {
  IDL_VPTR ans = NULL;
  char *p = NULL;
  int i, device;

  /*  First argument must be a scalar string  */
  if (!is_scalar(argv[0])) {
    IDL_MessageFromBlock(msg_block, GDQ_NOT_SCALAR, IDL_MSG_LONGJMP);
  }
  if (argv[0]->type != IDL_TYP_STRING) {
    IDL_MessageFromBlock(msg_block, GDQ_NOT_STRING, IDL_MSG_LONGJMP);
  }
  p = upcase(IDL_STRING_STR(&argv[0]->value.str));

  /*  Second argument must be an integer  */
  if (!is_scalar(argv[1])) { 
    IDL_MessageFromBlock(msg_block, GDQ_NOT_SCALAR, IDL_MSG_LONGJMP);
  }
  device = (int)get_numeric_value(argv[1]);

  /*  The device number must be valid  */
  if ((device < 0) || (device >= gDevices))  {
    IDL_MessageFromBlock(msg_block, GDQ_NO_DEVICE, IDL_MSG_LONGJMP);
  }

  /*  Look for a property with the given name  */
  for(i=0; i < ARRLEN(names); i++)
    if (strcmp((const char *)p, (const char *)names[i]) == 0)
      break;
  
  if (i == ARRLEN(names))
    IDL_MessageFromBlock(msg_block, GDQ_NO_PROP, IDL_MSG_LONGJMP);

  /*  Get the requested property  */
  if (i == 0) {
    /*  Device name, a string  */
    ans = IDL_Gettmp();
    ans->type = IDL_TYP_STRING;
    IDL_StrStore(&(ans->value.str), gProperties[device].name);
  } else {
    /*  All other properties are integers  */
    switch (i) {
      case  1:
        ans = IDL_GettmpMEMINT(gProperties[device].totalGlobalMem);
        break;
      case  2:
        ans = IDL_GettmpMEMINT(gProperties[device].sharedMemPerBlock);
        break;
      case  3:
        ans = IDL_GettmpMEMINT(gProperties[device].regsPerBlock);
        break;
      case  4:
        ans = IDL_GettmpMEMINT(gProperties[device].warpSize);
        break;
      case  5:
        ans = IDL_GettmpMEMINT(gProperties[device].memPitch);
        break;
      case  6:
        ans = IDL_GettmpMEMINT(gProperties[device].maxThreadsPerBlock);
        break;
      case  7:
        ans = IDL_GettmpMEMINT(gProperties[device].maxThreadsDim[0]);
        break;
      case  8:
        ans = IDL_GettmpMEMINT(gProperties[device].maxThreadsDim[1]);
        break;
      case  9:
        ans = IDL_GettmpMEMINT(gProperties[device].maxThreadsDim[2]);
        break;
      case  10:
        ans = IDL_GettmpMEMINT(gProperties[device].maxGridSize[0]);
        break;
      case  11:
        ans = IDL_GettmpMEMINT(gProperties[device].maxGridSize[1]);
        break;
      case  12:
        ans = IDL_GettmpMEMINT(gProperties[device].maxGridSize[2]);
        break;
      case  13:
        ans = IDL_GettmpMEMINT(gProperties[device].totalConstMem);
        break;
      case  14:
        ans = IDL_GettmpMEMINT(gProperties[device].major);
        break;
      case  15:
        ans = IDL_GettmpMEMINT(gProperties[device].minor);
        break;
      case  16:
        ans = IDL_GettmpMEMINT(gProperties[device].clockRate);
        break;
      case  17:
        ans = IDL_GettmpMEMINT(gProperties[device].textureAlignment);
        break;
      case  18:
        ans = IDL_GettmpMEMINT(gProperties[device].deviceOverlap);
        break;
      case  19:
        ans = IDL_GettmpMEMINT(gProperties[device].multiProcessorCount);
        break;
    }
  }

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
    {{(IDL_FUN_RET) idl_dev_count}, "GPU_DEVICE_COUNT", 0, 0, 0, 0},
    {{(IDL_FUN_RET) idl_dev_prop}, "GPU_DEVICE_PROPERTY", 2, 2, 0, 0},
  };

  /*  Error messages  */
  if (!(msg_block = IDL_MessageDefineBlock("GDQ", IDL_CARRAY_ELTS(msg_arr), msg_arr))) {
    return IDL_FALSE;  
  }

  /*  Query devices when the DLM loads  */
  query_devices();

  return IDL_SysRtnAdd(function_addr, TRUE, IDL_CARRAY_ELTS(function_addr));
}

/*  Module description - make pulls this out to create particle_swarm_search.dlm  */
//dlm: MODULE GPU_DEVICE_QUERY
//dlm: DESCRIPTION Query the system for CUDA enabled devices
//dlm: VERSION 1.0
//dlm: FUNCTION GPU_DEVICE_COUNT 0 0
//dlm: FUNCTION GPU_DEVICE_PROPERTY 2 2

/*
*  end gpu_device_query.c
*/

