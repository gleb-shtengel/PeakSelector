/*
*  File:  particle_swarm.c
*  
*  A particle swarm engine.  Implements the canonical and
*  repulsive particle swarm algorithms.
*
*  Author:  R. Kneusel, ITT VIS, rkneusel@ittvis.com
*  Date:  02-Apr-2007
*  Last Update:  14-Sep-2009
*
*  Copyright 2007, ITT VIS.  All Rights Reserved.
*/

/* Includes */
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <idl_export.h>

#ifndef WIN32
#define IDL_CDECL
#endif

/* Extra IDL stuff not in idl_export.h */
void IDL_CDECL IDL_CallRoutineByString(char *routName, IDL_VPTR *pvResult, 
                                       int argc, IDL_VPTR *argv, char *argk, int bStepOver);

int IDL_CDECL IDL_ObjCallMethodByString(char *methName, IDL_HVID obj, IDL_VPTR *pvResult,
                                        int argc, IDL_VPTR *argv, char *argk);
                                                   
#define ARRLEN(arr) (sizeof(arr)/sizeof(arr[0]))

/*
*  Message block definitions:
*/
static IDL_MSG_BLOCK msg_block;  

#define PARTICLE_SWARM_NO_MEMORY                   0
#define PARTICLE_SWARM_STRING_REQUIRED            -1
#define PARTICLE_SWARM_NO_SUCH_PROP               -2
#define PARTICLE_SWARM_MUST_CONFIGURE             -3
#define PARTICLE_SWARM_SCALAR_NUMBER_REQUIRED     -4
#define PARTICLE_SWARM_ILLEGAL_TYPE               -5
#define PARTICLE_SWARM_ILLEGAL_VALUE              -6
#define PARTICLE_SWARM_NO_FUNCTION                -7

static IDL_MSG_DEF msg_arr[] = {  
  {"PARTICLE_SWARM_NO_MEMORY", "%NNo memory available."},
  {"PARTICLE_SWARM_STRING_REQUIRED", "%NA scalar string argument is required."},
  {"PARTICLE_SWARM_NO_SUCH_PROP", "%NNo such property."},
  {"PARTICLE_SWARM_MUST_CONFIGURE", "%NThe engine must be configured before use (see PS_CONFIGURE)"},
  {"PARTICLE_SWARM_SCALAR_NUMBER_REQUIRED","%NA numeric scalar argument is required."},
  {"PARTICLE_SWARM_ILLEGAL_TYPE", "%NIllegal argument type."},
  {"PARTICLE_SWARM_ILLEGAL_VALUE","%NIllegal argument value."},
  {"PARTICLE_SWARM_NO_FUNCTION", "%NNo function or method name supplied."},
};


/**************************************************************
*  Property names
*/
static char *getprop_names[] = {
  "W","C1","C2","FUNCTION_NAME","METHOD_NAME", "OBJECT", "PARTICLES",
  "DOMAIN", "MAX_ITERATIONS", "TOLERANCE", "POSITIONS", "VELOCITIES",
  "PARTICLE_BEST", "GLOBAL_BEST", "DIMENSIONS", "INITIAL_POSITIONS",
  "INITIAL_VELOCITIES", "ITERATIONS_DONE", "MAX_VELOCITY",
  "GLOBAL_COUNT", "CONSTRAIN_TO_DOMAIN", "C3", "REPULSIVE",
  "MAXIMIZE"
};

static char *setprop_names[] = {
  "W","C1","C2","FUNCTION_NAME", "METHOD_NAME", "OBJECT", 
  "DOMAIN", "MAX_ITERATIONS", "TOLERANCE", "INITIAL_POSITIONS",
  "INITIAL_VELOCITIES", "MAX_VELOCITY", 
  "CONSTRAIN_TO_DOMAIN", "C3", "REPULSIVE", "MAXIMIZE"
};


/**************************************************************
*  Properties
*/
typedef struct properties {
  IDL_LONG m;      //  number of dimensions
  double w[2];     //  inertial constant - decreases with each step, [start, end]
  double wm;       //  slope of w line
  double c1;       //  cognitive constant
  double c2;       //  social constant
  double c3;       //  repulsive constant
  IDL_LONG n;      //  number of particles
  double *domain;  //  domain, low, high (2 x m matrix)
  char constrain;  //  if set, constrain search to the given domain
  char repulsive;  //  if set, use repulsive optimization, otherwise use global
  char maximize;   //  if set, maximize not minimize
  double *xi;      //  initial particle positions (m x n matrix)
  double *x;       //  current particle positions (m x n matrix)
  double *vi;      //  initial particle velocities (m x n matrix)
  double *v;       //  current particle velocities (m x n matrix)
  double *b;       //  particle personal best positions (m x n matrix)
  double *bf;      //  particle personal best function values (n element vector)
  double *g;       //  global best position (m element vector)
  double gf;       //  global best function value
  double *gs;      //  global best sigma values (m element vector)
  char calcgs;     //  flag, calculate gs if true
  double vmax;     //  maximum velocity (applied per component)
  IDL_LONG imax;   //  maximum number of iterations
  IDL_LONG i;      //  current iteration number
  IDL_LONG gtol;   //  global best tolerance (quit if g not changed in this many iterations)
  IDL_LONG gcount; //  counts number of iterations where g has not changed
  char ftype;      //  1=function call, 0=method call on an object
  char *func;      //  name of the IDL objective function
  char *meth;      //  name of the IDL objective method
  IDL_HVID obj;    //  an IDL object reference which supports the method
} PropertiesType;

static PropertiesType properties;
static char configured;

/*
*  Hybrid Tausworthe PRNG.  Period ~2^121.
*/
static unsigned int z1 = 0xff32422;
static unsigned int z2 = 0xee03202;
static unsigned int z3 = 0xcc23423;
static unsigned int z4 = 0x1235;     // use as seed value

static unsigned int TausStep(unsigned *z, int S1, int S2, int S3, unsigned int M)
{
  unsigned int b = ((*z << S1) ^ *z) << S2;
  *z = (((*z & M) << S3) ^ b);
  return *z;
}

static unsigned int LCGStep(unsigned int *z, unsigned int A, unsigned int C)
{
  *z = (A*(*z)+C);
  return *z;
}

static double HybridTaus()
{
  return (double)(TausStep(&z1, 13, 19, 12, 4294967294UL) ^
          TausStep(&z2, 2, 25, 4, 4294967288UL) ^
          TausStep(&z3, 3, 11, 17, 4294967280UL) ^
          LCGStep(&z4, 1664525, 1013904223UL)) / (double)4294967296.0;
}


/**************************************************************
*  randomu
*
*  Return a random number, [0,1).
*/
static double randomu(void) {
  return HybridTaus();
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
*  is_zero
*
*  True (1) if the double array given is all zero.
*/
static char is_zero(double *p, int n) {
  IDL_MEMINT i;
  
  for(i=0; i < n; i++)
    if (*p++ != 0.0)
      return 0;
  return 1;
}


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


/*************************************************************
*  get_tmp_double
*
*  Return a new temporary IDL variable of type double.
*/
static IDL_VPTR get_tmp_double(double n) {
  IDL_VPTR ans;
  ans = IDL_Gettmp();
  ans->type = IDL_TYP_DOUBLE;
  ans->value.d = n;
  return ans;
}


/*************************************************************
*  get_tmp_string
*
*  Make a temporary IDL variable from a string.
*/
static IDL_VPTR get_tmp_string(char *s) {
  IDL_VPTR ans;
  
  ans = IDL_Gettmp();
  ans->type = IDL_TYP_STRING;
  IDL_StrStore(&(ans->value.str), s);
  return ans;
}


/*************************************************************
*  get_tmp_w
*
*  Get the current W vector.
*/
static IDL_VPTR get_tmp_w(double *w) {
  IDL_VPTR ans;
  IDL_MEMINT d = 2;
  double *p;
  
  IDL_MakeTempArray(IDL_TYP_DOUBLE, 1, &d, IDL_ARR_INI_NOP, &ans);
  p = (double *)ans->value.arr->data;
  p[0] = w[0];
  p[1] = w[1];
  
  return ans;
}


/*************************************************************
*  get_tmp_object
*
*  Get the current object reference.
*/
static IDL_VPTR get_tmp_object(IDL_HVID obj) {
  IDL_VPTR ans;
  ans = IDL_Gettmp();
  ans->type = IDL_TYP_OBJREF;
  ans->value.hvid = obj;  
  return ans;
}


/*************************************************************
*  get_tmp_domain
*
*  Get the current domain, if any.
*/
static IDL_VPTR get_tmp_domain(void) {
  IDL_VPTR ans;
  double *p, *q;
  IDL_MEMINT i, d[2];
  
  /*  Return -1 if not set  */
  if (!properties.domain)
    return IDL_GettmpMEMINT(-1);
  
  /*  Output array  */
  d[0] = 2;
  d[1] = (IDL_MEMINT)properties.m;
  IDL_MakeTempArray(IDL_TYP_DOUBLE, 2, d, IDL_ARR_INI_NOP, &ans);
  p = (double *)ans->value.arr->data;
  q = properties.domain;
  
  for(i=0; i < 2*properties.m; i++)
    *p++ = *q++;
    
  return ans;
}


/*************************************************************
*  get_tmp_mxn
*
*  Get an m x n array.
*/
static IDL_VPTR get_tmp_mxn(double *prop) {
  IDL_VPTR ans;
  double *p, *q;
  IDL_MEMINT i, d[2];
  
  /*  Return -1 if not set  */
  if (!prop)
    return IDL_GettmpMEMINT(-1);
  
  /*  Output array  */
  d[0] = (IDL_MEMINT)properties.m;
  d[1] = (IDL_MEMINT)properties.n;
  IDL_MakeTempArray(IDL_TYP_DOUBLE, 2, d, IDL_ARR_INI_NOP, &ans);
  p = (double *)ans->value.arr->data;
  q = prop;
  
  for(i=0; i < properties.m*properties.n; i++)
    *p++ = *q++;
    
  return ans;
}


/*************************************************************
*  get_tmp_global
*
*  Global best position.
*/
static IDL_VPTR get_tmp_global() {
  IDL_VPTR ans;
  double *p, *q;
  IDL_MEMINT i, d;
  
  /*  Return -1 if not set  */
  if (!properties.g)
    return IDL_GettmpMEMINT(-1);
    
  d = (IDL_MEMINT)properties.m;
  IDL_MakeTempArray(IDL_TYP_DOUBLE, 1, &d, IDL_ARR_INI_NOP, &ans);
  p = (double *)ans->value.arr->data;
  q = properties.g;
  
  for(i=0; i < properties.m; i++)
    *p++ = *q++;
    
  return ans;
}


/*************************************************************
*  idl_getprop
*
*  Return a property value.
*/
static IDL_VPTR idl_getprop(int argc, IDL_VPTR *argv) {
  IDL_VPTR ans;
  char *p;
  IDL_MEMINT i;
  
  /*  Must have configured the engine first  */
  if (!configured)
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_MUST_CONFIGURE, IDL_MSG_LONGJMP);
  
  /*  Argument must be a scalar string  */
  if ((!is_scalar(argv[0])) || (argv[0]->type != IDL_TYP_STRING))
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_STRING_REQUIRED, IDL_MSG_LONGJMP);
    
  /*  Find it in the list of properties  */
  p = upcase(IDL_STRING_STR(&argv[0]->value.str));

  for(i=0; i < ARRLEN(getprop_names); i++)
    if (strcmp((const char *)p, (const char *)getprop_names[i]) == 0)
      break;
  
  if (i == ARRLEN(getprop_names))
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_NO_SUCH_PROP, IDL_MSG_LONGJMP);

  /*  Retrieve the property and return  */
  switch (i) {
    case  0: {  ans = get_tmp_w(properties.w);                       break;  }  //  W
    case  1: {  ans = get_tmp_double(properties.c1);                 break;  }  //  C1
    case  2: {  ans = get_tmp_double(properties.c2);                 break;  }  //  C2
    case  3: {  ans = get_tmp_string(properties.func);               break;  }  //  FUNCTION_NAME
    case  4: {  ans = get_tmp_string(properties.meth);               break;  }  //  METHOD_NAME
    case  5: {  ans = get_tmp_object(properties.obj);                break;  }  //  OBJECT
    case  6: {  ans = IDL_GettmpLong(properties.n);                  break;  }  //  PARTICLES
    case  7: {  ans = get_tmp_domain();                              break;  }  //  DOMAIN
    case  8: {  ans = IDL_GettmpLong(properties.imax);               break;  }  //  MAX_ITERATIONS
    case  9: {  ans = IDL_GettmpLong(properties.gtol);               break;  }  //  TOLERANCE
    case 10: {  ans = get_tmp_mxn(properties.x);                     break;  }  //  POSITIONS
    case 11: {  ans = get_tmp_mxn(properties.v);                     break;  }  //  VELOCITIES
    case 12: {  ans = get_tmp_mxn(properties.b);                     break;  }  //  PARTICLE_BEST
    case 13: {  ans = get_tmp_global();                              break;  }  //  GLOBAL_BEST
    case 14: {  ans = IDL_GettmpLong(properties.m);                  break;  }  //  DIMENSIONS
    case 15: {  ans = get_tmp_mxn(properties.xi);                    break;  }  //  INITIAL_POSITIONS
    case 16: {  ans = get_tmp_mxn(properties.vi);                    break;  }  //  INITIAL_VELOCITIES
    case 17: {  ans = IDL_GettmpLong(properties.i);                  break;  }  //  ITERATIONS_DONE
    case 18: {  ans = get_tmp_double(properties.vmax);               break;  }  //  MAX_VELOCITY
    case 19: {  ans = IDL_GettmpLong(properties.gcount);             break;  }  //  GLOBAL_COUNT
    case 20: {  ans = IDL_GettmpInt((IDL_INT)properties.constrain);  break;  }  //  CONSTRAIN_TO_DOMAIN
    case 21: {  ans = get_tmp_double(properties.c3);                 break;  }  //  C3
    case 22: {  ans = IDL_GettmpInt((IDL_INT)properties.repulsive);  break;  }  //  REPULSIVE
    case 23: {  ans = IDL_GettmpInt((IDL_INT)properties.maximize);   break;  }  //  MAXIMIZE
    default: { break; }
  }
  
  return ans;
}


/*************************************************************
*  idl_configure
*
*  Configure for a given dimension and number of particles.
*/
static void idl_configure(int argc, IDL_VPTR *argv) {
  IDL_LONG m,n;  

  /*  Get the dimensions and number of particles  */
  if ((!is_scalar(argv[0])) || (!is_numeric(argv[0])))
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_SCALAR_NUMBER_REQUIRED, IDL_MSG_LONGJMP);
  if ((!is_scalar(argv[1])) || (!is_numeric(argv[1])))
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_SCALAR_NUMBER_REQUIRED, IDL_MSG_LONGJMP);
  m = properties.m = (IDL_LONG)get_numeric_value(argv[0]);
  n = properties.n = (IDL_LONG)get_numeric_value(argv[1]);
  
  /*  Values must make sense  */
  if (m < 1)
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_ILLEGAL_VALUE, IDL_MSG_LONGJMP);
  if (n < 2)
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_ILLEGAL_VALUE, IDL_MSG_LONGJMP);
  
  /*  Reconfigure memory  */
  if (properties.domain) {
    free(properties.domain);
    properties.domain = NULL;
  }
  if (properties.xi) {
    free(properties.xi);
    properties.xi = NULL;
  }
  if (properties.x) {
    free(properties.x);
    properties.x = NULL;
  }
  if (properties.vi) {
    free(properties.vi);
    properties.vi = NULL;
  }
  if (properties.v) {
    free(properties.v);
    properties.v = NULL;
  }
  if (properties.b) {
    free(properties.b);
    properties.b = NULL;
  }
  if (properties.bf) {
    free(properties.bf);
    properties.bf = NULL;
  }
  if (properties.g) {
    free(properties.g);
    properties.g = NULL;
  }
  if (properties.func) {
    free(properties.func);
    properties.func = NULL;
  }
  if (properties.meth) {
    free(properties.meth);
    properties.meth = NULL;
  }

  properties.domain = (double *)malloc(2*m*sizeof(double));
  if (!properties.domain)  
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_NO_MEMORY, IDL_MSG_LONGJMP);
  memset((void*)properties.domain, 0, 2*m*sizeof(double));
  
  properties.xi     = (double *)malloc(m*n*sizeof(double));
  if (!properties.xi)
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_NO_MEMORY, IDL_MSG_LONGJMP);
  memset((void*)properties.xi, 0, m*n*sizeof(double));
    
  properties.x      = (double *)malloc(m*n*sizeof(double));
  if (!properties.x)
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_NO_MEMORY, IDL_MSG_LONGJMP);
  memset((void*)properties.x, 0, m*n*sizeof(double));
  
  properties.vi     = (double *)malloc(m*n*sizeof(double));
  if (!properties.vi)
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_NO_MEMORY, IDL_MSG_LONGJMP);
  memset((void*)properties.vi, 0, m*n*sizeof(double));
  
  properties.v      = (double *)malloc(m*n*sizeof(double));
  if (!properties.v)
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_NO_MEMORY, IDL_MSG_LONGJMP);
  memset((void*)properties.v, 0, m*n*sizeof(double));
  
  properties.b      = (double *)malloc(m*n*sizeof(double));
  if (!properties.b)
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_NO_MEMORY, IDL_MSG_LONGJMP);
  memset((void*)properties.b, 0, m*n*sizeof(double));
  
  properties.bf     = (double *)malloc(n*sizeof(double));
  if (!properties.bf)
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_NO_MEMORY, IDL_MSG_LONGJMP);
  memset((void*)properties.bf, 0, n*sizeof(double));
  
  properties.g      = (double *)malloc(m*sizeof(double));
  if (!properties.g)
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_NO_MEMORY, IDL_MSG_LONGJMP);
  memset((void*)properties.g, 0, m*sizeof(double));
  
  /*  Set default values for some properties.  */
  properties.ftype     = 1;
  properties.c1        = 2.0;
  properties.c2        = 2.0;
  properties.c3        = 2.0;
  properties.w[0]      = 0.9;
  properties.w[1]      = 0.9;
  properties.wm        = 0;
  properties.imax      = 25;
  properties.gtol      = 10;
  properties.vmax      = 1e8;
  properties.constrain = 0;
  properties.repulsive = 0;
  properties.maximize  = 0;
  
  /*  Indicate configured  */
  configured = 1;   
}


/*************************************************************
*  is_true
*
*  Return TRUE if the given value should be considered
*  as true.
*/
static char is_true(IDL_VPTR arg) {
  
  switch (arg->type) {
  
    /*  All simple numbers  */
    case IDL_TYP_BYTE:  case IDL_TYP_INT:  case IDL_TYP_LONG:
    case IDL_TYP_FLOAT: case IDL_TYP_DOUBLE:  case IDL_TYP_UINT:
    case IDL_TYP_ULONG: case IDL_TYP_LONG64: case IDL_TYP_ULONG64: {
      if (is_scalar(arg)) {
        return ((long)get_numeric_value(arg) != 0);
      } else {
        switch (arg->type) {
          case 1:  return (*(UCHAR *)arg->value.arr->data       != 0);
          case 2:  return (*(IDL_INT *)arg->value.arr->data     != 0);
          case 3:  return (*(IDL_LONG *)arg->value.arr->data    != 0);
          case 4:  return (*(float *)arg->value.arr->data       != 0);
          case 5:  return (*(double *)arg->value.arr->data      != 0);
          case 12: return (*(IDL_UINT *)arg->value.arr->data    != 0);
          case 13: return (*(IDL_ULONG *)arg->value.arr->data   != 0);
          case 14: return (*(IDL_LONG64 *)arg->value.arr->data  != 0);
          case 15: return (*(IDL_ULONG64 *)arg->value.arr->data != 0);
        }
      }
    }
    
    /*  Complex numbers  */
    case IDL_TYP_COMPLEX: {
      if (is_scalar(arg)) {
        float r,i,*p;
        p = (float*)&(arg->value.cmp);
        r = p[0];
        i = p[1];
        return (sqrt(r*r+i*i) != 0.0);
      } else {
        float r,i,*p;
        p = (float *)arg->value.arr->data;
        r = p[0];
        i = p[1];
        return (sqrt(r*r+i*i) != 0.0);
      }      
    }
    case IDL_TYP_DCOMPLEX: {
      if (is_scalar(arg)) {
        double r,i,*p;
        p = (double*)&(arg->value.dcmp);
        r = p[0];
        i = p[1];
        return (sqrt(r*r+i*i) != 0.0);
      } else {
        double r,i,*p;
        p = (double *)arg->value.arr->data;
        r = p[0];
        i = p[1];
        return (sqrt(r*r+i*i) != 0.0);
      }
    }
    
    /*  Strings  */
    case IDL_TYP_STRING: {
      if (is_scalar(arg)) {
        char *s = IDL_VarGetString(arg);
        return (s[0] != '\0' );
      } else {
        char *s = IDL_STRING_STR((IDL_STRING *)arg->value.arr->data);
        return (s[0] != '\0');
      }      
    }
    
    /*  Structures, pointers and object references are always true  */
    case IDL_TYP_STRUCT: 
    case IDL_TYP_PTR:
    case IDL_TYP_OBJREF: {
      return 1;
    }
    
    /*  Undefined  */
    default: { 
      return 0; 
    }
  }
}


/*************************************************************
*  set_long
*
*  Set an IDL_LONG property.
*/
static void set_long(IDL_VPTR arg, IDL_LONG *prop) {

  if ((!is_scalar(arg)) || (!is_numeric(arg)))
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_ILLEGAL_TYPE, IDL_MSG_LONGJMP);
  *prop = (IDL_LONG)get_numeric_value(arg);
}


/*************************************************************
*  set_object
*
*  Set a the object reference.
*/
static void set_object(IDL_VPTR arg) {

  if (!is_scalar(arg))
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_ILLEGAL_TYPE, IDL_MSG_LONGJMP);
  if (arg->type != IDL_TYP_OBJREF)
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_ILLEGAL_TYPE, IDL_MSG_LONGJMP);
  properties.obj = arg->value.hvid;
}


/*************************************************************
*  set_double
*
*  Set a double property.
*/
static void set_double(IDL_VPTR arg, double *prop) {

  if ((!is_scalar(arg)) || (!is_numeric(arg)))
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_ILLEGAL_TYPE, IDL_MSG_LONGJMP);

  *prop = get_numeric_value(arg);
}


/*************************************************************
*  get_tmp_double_copy
*
*  Get a double version of a numeric array.
*/
static IDL_VPTR get_tmp_double_copy(IDL_VPTR arg) {
  UCHAR flags;
  IDL_VPTR ans;
  
  flags = arg->flags;
  arg->flags &= ~IDL_V_TEMP;
  IDL_CallRoutineByString("DOUBLE", &ans, 1, &arg, 0, 0);
  arg->flags = flags;
  return ans;
}


/*************************************************************
*  set_w
*
*  Set the inertial parameter.
*/
static void set_w(IDL_VPTR arg) {
  IDL_VPTR t;
  double *p;
  
  /*  Must be numeric  */
  if (!is_numeric(arg))
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_ILLEGAL_TYPE, IDL_MSG_LONGJMP);
    
  /*  Set the W values  */
  if (is_scalar(arg)) {
    double w = get_numeric_value(arg);
    properties.w[0] = w;
    properties.w[1] = w;
    properties.wm = 0.0;
  } else {
    /*  Must have two elements  */
    if (arg->value.arr->n_elts != 2)
      IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_ILLEGAL_TYPE, IDL_MSG_LONGJMP);
    
    if (arg->type == IDL_TYP_DOUBLE) {
      p = (double *)arg->value.arr->data;
      properties.w[0] = p[0];
      properties.w[1] = p[1];      
    } else {
      t = get_tmp_double_copy(arg);
      p = (double *)t->value.arr->data;
      properties.w[0] = p[0];
      properties.w[1] = p[1];
      IDL_Deltmp(t);
    }
    
    if (properties.w[1] > properties.w[0]) {
      double q = properties.w[0];
      properties.w[0] = properties.w[1];
      properties.w[1] = q;
    }
    if (properties.imax == 0) {
      properties.wm = 0;
      properties.w[1] = properties.w[0];
    } else {
      properties.wm = (properties.w[1] - properties.w[0]) / (double)properties.imax;
    }
  }
}


/*************************************************************
*  set_string
*
*  Set a string.
*/
static void set_string(IDL_VPTR arg, char **s) {
  char *t;
  
  if ((arg->type != IDL_TYP_STRING) || (!is_scalar(arg)))
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_ILLEGAL_TYPE, IDL_MSG_LONGJMP);
  t = IDL_STRING_STR(&(arg->value.str));

  if (*s)
    free(*s);
  *s = (char *)malloc(strlen(t)+1);
  if (!*s)
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_NO_MEMORY, IDL_MSG_LONGJMP);
  strcpy(*s,t);
}


/*************************************************************
*  set_matrix
*
*  Set an m x n matrix.
*/
static void set_matrix(IDL_VPTR arg, double **p) {
  IDL_MEMINT i, *d;
  double *v;
  IDL_VPTR t;
  
  /*  Validate  */
  if ((!is_numeric(arg)) || (is_scalar(arg)))
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_ILLEGAL_TYPE, IDL_MSG_LONGJMP);
  if (arg->value.arr->n_dim != 2)
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_ILLEGAL_TYPE, IDL_MSG_LONGJMP);
  d = (IDL_MEMINT *)arg->value.arr->dim;
  if ((d[0] != properties.m) || (d[1] != properties.n))
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_ILLEGAL_TYPE, IDL_MSG_LONGJMP);
  
  /*  Allocate memory  */
  if (*p)
    free(*p);
  *p = (double *)malloc(properties.m*properties.n*sizeof(double));
  if (!*p)
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_NO_MEMORY, IDL_MSG_LONGJMP);
    
  /*  Get a double copy of the input and copy to memory  */
  if (arg->type == IDL_TYP_DOUBLE) {
    v = (double *)arg->value.arr->data;
    for(i=0; i < properties.m*properties.n; i++)
      (*p)[i] = *v++;  
  } else {
    t = get_tmp_double_copy(arg);
    v = (double *)t->value.arr->data;
    for(i=0; i < properties.m*properties.n; i++)
      (*p)[i] = *v++;
    IDL_Deltmp(t);
  }
}


/*************************************************************
*  set_domain
*
*  Set the domain.
*/
static void set_domain(IDL_VPTR arg) {
  IDL_MEMINT i, *d, n;
  double *v, *p;
  IDL_VPTR t;
  
  /*  Initial validation  */
  if ((!is_numeric(arg)) || (is_scalar(arg)))
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_ILLEGAL_TYPE, IDL_MSG_LONGJMP);

  if (properties.domain)
    free(properties.domain);
    
  /*  Fully specified or only one dimension?  */
  n = arg->value.arr->n_elts;
  if (n == 2) {
    double lo, hi;
    
    p = properties.domain = (double *)malloc(2*properties.m*sizeof(double));
    if (!p)
      IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_NO_MEMORY, IDL_MSG_LONGJMP);
      
    if (arg->type = IDL_TYP_DOUBLE) {
      lo = ((double *)arg->value.arr->data)[0];
      hi = ((double *)arg->value.arr->data)[1];
    } else {
      t = get_tmp_double_copy(arg);
      lo = ((double *)t->value.arr->data)[0];
      hi = ((double *)t->value.arr->data)[1];
      IDL_Deltmp(t);
    }
    
    for(i=0; i < 2*properties.m-1; i+=2) {
      p[i] = lo;
      p[i+1] = hi;
    }
  } else {
    if (n != 2*properties.m)
      IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_ILLEGAL_VALUE, IDL_MSG_LONGJMP);
    
    p = properties.domain = (double *)malloc(2*properties.m*sizeof(double));
    if (!p)
      IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_NO_MEMORY, IDL_MSG_LONGJMP);
    
    if (arg->type == IDL_TYP_DOUBLE) {
      v = (double *)arg->value.arr->data;
      for(i=0; i < 2*properties.m; i++) {
        *p++ = *v++;      
      }
    } else {
      t = get_tmp_double_copy(arg);
      v = (double *)t->value.arr->data;
      for(i=0; i < 2*properties.m; i++)
        *p++ = *v++;
      IDL_Deltmp(t);
    }
  }
}


/*************************************************************
*  idl_setprop
*
*  Set a property value.
*/
static void idl_setprop(int argc, IDL_VPTR *argv) {
  char *p;
  IDL_MEMINT i;
  
  /*  Must have configured the engine first  */
  if (!configured)
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_MUST_CONFIGURE, IDL_MSG_LONGJMP);
  
  /*  Argument must be a scalar string  */
  if ((!is_scalar(argv[0])) || (argv[0]->type != IDL_TYP_STRING))
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_STRING_REQUIRED, IDL_MSG_LONGJMP);
    
  /*  Find it in the list of properties  */
  p = upcase(IDL_STRING_STR(&argv[0]->value.str));

  for(i=0; i < ARRLEN(setprop_names); i++)
    if (strcmp((const char *)p, (const char *)setprop_names[i]) == 0)
      break;
  
  if (i == ARRLEN(setprop_names))
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_NO_SUCH_PROP, IDL_MSG_LONGJMP);

  /*  Retrieve the property and return  */
  switch (i) {
    case  0: { set_w(argv[1]);                            break; }  //  W
    case  1: { set_double(argv[1], &properties.c1);       break; }  //  C1
    case  2: { set_double(argv[1], &properties.c2);       break; }  //  C2
    case  3: { 
      set_string(argv[1], &properties.func);  
      properties.ftype = 1;  
      upcase(properties.func);
      break; }  //  FUNCTION_NAME
    case  4: { 
      set_string(argv[1], &properties.meth);  
      properties.ftype = 0;  
      upcase(properties.meth);
      break; }  //  METHOD_NAME
    case  5: { set_object(argv[1]);                       break; }  //  OBJECT
    case  6: { set_domain(argv[1]);                       break; }  //  DOMAIN
    case  7: { set_long(argv[1], &properties.imax);       break; }  //  MAX_ITERATIONS
    case  8: { set_long(argv[1], &properties.gtol);       break; }  //  TOLERANCE
    case  9: { set_matrix(argv[1], &properties.xi);       break; }  //  INITIAL_POSITIONS
    case 10: { set_matrix(argv[1], &properties.vi);       break; }  //  INITIAL_VELOCITIES
    case 11: { set_double(argv[1], &properties.vmax);     break; }  //  MAX_VELOCITY
    case 12: { properties.constrain = is_true(argv[1]);   break; }  //  CONSTRAIN_TO_DOMAIN
    case 13: { set_double(argv[1], &properties.c3);       break; }  //  C3
    case 14: { properties.repulsive = is_true(argv[1]);   break; }  //  REPULSIVE
    case 15: { properties.maximize = is_true(argv[1]);    break; }  //  MAXIMIZE
    default: { break; }
  }
}


/*************************************************************
*  random_vector_in_domain
*
*  Set the input to properties.m random values in the domain.
*/
static void random_vector_in_domain(double *p) {
  double *d = properties.domain;
  IDL_MEMINT i;
  
  for(i=0; i < properties.m; i++)
    p[i] = d[2*i] + randomu()*(d[2*i+1]-d[2*i]);
}


/*************************************************************
*  global_update
*
*  Update the particle positions and velocities according to
*  the canonical algorithm.
*/
static void global_update(void) {
  IDL_VPTR arg, ans;
  char update;
  IDL_MEMINT i,j, dim;
  double *x, *q, *v, *b, *g, *xx=NULL, *vv=NULL, *arr;
  double w, bf, r1, r2;
  int m;
  
  /*  Local copies of position and velocity  */
  xx = (double *)malloc(properties.m*sizeof(double));
  if (!xx)
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_NO_MEMORY, IDL_MSG_LONGJMP);
  vv = (double *)malloc(properties.m*sizeof(double));
  if (!vv)
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_NO_MEMORY, IDL_MSG_LONGJMP);
  
  /*  Dimensionality  */
  m = (int)properties.m;
    
  /*  Track if global position updated in this step  */
  update = 0;
  
  /*  Compute the inertial weight for this step  */
  w = properties.wm*properties.i + (properties.w)[0];
  
  /*  A temporary array to hold argument values  */    
  dim = (IDL_MEMINT)m;
  IDL_MakeTempArray(IDL_TYP_DOUBLE, 1, &dim, IDL_ARR_INI_NOP, &arg);
  arr = (double *)arg->value.arr->data;
  arg->flags &= ~IDL_V_TEMP;
  
  /*  Look at each particle  */
  for(i=0; i < properties.n; i++) {
    int k;
    
    /*  Offset into 2D arrays for this particle  */
    k = i*m;
    
    /*  Update this particle's position  */
    x = properties.x;
    v = properties.v;
    for(j=0; j < m; j++)
      xx[j] = x[k+j] + v[k+j];
     
    /*  Constrain to domain, if set  */
    if (properties.constrain) {
      for(j=0; j < m; j++) {
        q = &((properties.domain)[2*j]);
        if (xx[j] < q[0])
          xx[j] = q[0];
        if (xx[j] > q[1])
          xx[j] = q[1];
      }
    }
    
    /*  Update this particle's velocity vector  */
    v = properties.v;
    g = properties.g;
    b = properties.b;
    r1 = properties.c1*randomu();
    r2 = properties.c2*randomu();
    for(j=0; j < m; j++)
      vv[j] = w*v[k+j] + r1*(b[k+j]-xx[j]) + r2*(g[j]-xx[j]);
    
    /*  Constrain velocity components to Vmax  */
    for(j=0; j < m; j++) {
      if (vv[j] > properties.vmax)
        vv[j] = properties.vmax;
      if (vv[j] < -properties.vmax)
        vv[j] = -properties.vmax;
    }
    
    /*  Update the properties values  */
    x = properties.x;
    v = properties.v;
    for(j=0; j < m; j++) {
      x[k+j] = xx[j];
      v[k+j] = vv[j];
    }
    
    /*  Update best positions  */
    for(j=0; j < m; j++)
      arr[j] = xx[j];
      
    /*  Call the function or method  */
    if (properties.ftype) {
      IDL_CallRoutineByString(properties.func, &ans, 1, &arg, 0, 0);
    } else {
      IDL_ObjCallMethodByString(properties.meth, properties.obj, &ans, 1, &arg, NULL);
    }
      
    /*  Pull out the return value  */
    bf = get_numeric_value(ans);
    IDL_Deltmp(ans);
    
    /*  Decide if we update or not  */
    if (properties.maximize) {
      if (bf > (properties.bf)[i]) {
        b = properties.b;
        for(j=0; j < m; j++)
          b[k+j] = xx[j];
        (properties.bf)[i] = bf;
      }
      if ((properties.bf)[i] > properties.gf) {
        g = properties.g;
        b = properties.b;
        for(j=0; j < m; j++)
          g[j] = b[k+j];
        properties.gf = (properties.bf)[i];
        update = 1;
      }
    } else {
      if (bf < (properties.bf)[i]) {
        b = properties.b;
        for(j=0; j < m; j++)
          b[k+j] = xx[j];
        (properties.bf)[i] = bf;
      }
      if ((properties.bf)[i] < properties.gf) {
        g = properties.g;
        b = properties.b;
        for(j=0; j < m; j++)
          g[j] = b[k+j];
        properties.gf = (properties.bf)[i];
        update = 1;
      }    
    }
  }

  /*  Release temporary argument array  */
  arg->flags |= IDL_V_TEMP;
  IDL_Deltmp(arg);
  
  /*  If no update, bump count, else reset count  */
  if (!update)
    properties.gcount++;
  else
    properties.gcount = 0;
    
  /*  Count this iteration  */
  properties.i++;
  
  /*  Clean up  */
  free(xx);
  free(vv);
}


/*************************************************************
*  repulsive_update
*
*  Update the particle positions and velocities according to
*  the repulsive algorithm.
*/
static void repulsive_update(void) {
  IDL_VPTR arg, ans;
  char update;
  IDL_MEMINT i,j, dim;
  double *x, *q, *v, *b, *g, *xx=NULL, *vv=NULL, *arr;
  double w, bf, r1, r2, r3;
  int m, rb, rv;
  
  /*  Local copies of position and velocity  */
  xx = (double *)malloc(properties.m*sizeof(double));
  if (!xx)
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_NO_MEMORY, IDL_MSG_LONGJMP);
  vv = (double *)malloc(properties.m*sizeof(double));
  if (!vv)
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_NO_MEMORY, IDL_MSG_LONGJMP);
    
  /*  Dimensionality  */
  m = (int)properties.m;
    
  /*  Track if global position updated in this step  */
  update = 0;
  
  /*  Compute the inertial weight for this step  */
  w = properties.wm*properties.i + (properties.w)[0];
  
  /*  A temporary array to hold argument values  */    
  dim = (IDL_MEMINT)m;
  IDL_MakeTempArray(IDL_TYP_DOUBLE, 1, &dim, IDL_ARR_INI_NOP, &arg);
  arr = (double *)arg->value.arr->data;
  arg->flags &= ~IDL_V_TEMP;
  
  /*  Look at each particle  */
  for(i=0; i < properties.n; i++) {
    int k;
    
    /*  Offset into 2D arrays for this particle  */
    k = i*m;
    
    /*  Update this particle's position  */
    x = properties.x;
    v = properties.v;
    for(j=0; j < m; j++)
      xx[j] = x[k+j] + v[k+j];
     
    /*  Constrain to domain, if set  */
    if (properties.constrain) {
      for(j=0; j < m; j++) {
        q = &((properties.domain)[2*j]);
        if (xx[j] < q[0])
          xx[j] = q[0];
        if (xx[j] > q[1])
          xx[j] = q[1];
      }
    }

#if 0    
    ;  A randomly chosen best position
    repeat begin
      n = fix(self.n * randomu(seed))
    endrep until (n ne i)
    y = reform((*self.b)[*,n])

    ;  A randomly chosen velocity
    repeat begin
      n = fix(self.n * randomu(seed))
    endrep until (n ne i)
    z = reform((*self.v)[*,n])

    ;  Update the velocity
    v = w*v + c1*r1*(b-x) + c2*r2*w*(y-x) + c3*r3*w*z
#endif
    
    /*  A randomly chosen best position & velocity  */
    do {
      rb = (int)(properties.n*randomu());
    } while (rb==i);
    do {
      rv = (int)(properties.n*randomu());
    } while (rv==i);

    /*  Update this particle's velocity vector  */
    v = properties.v;
    b = properties.b;
    r1 = properties.c1*randomu();
    r2 = properties.c2*randomu();
    r3 = properties.c3*randomu();
    for(j=0; j < m; j++)
      vv[j] = w*v[k+j] + r1*(b[k+j]-xx[j]) + r2*w*(b[rb+j]-xx[j]) + r3*w*v[rv+j];
    
    /*  Constrain velocity components to Vmax  */
    for(j=0; j < m; j++) {
      if (vv[j] > properties.vmax)
        vv[j] = properties.vmax;
      if (vv[j] < -properties.vmax)
        vv[j] = -properties.vmax;
    }
    
    /*  Update the properties values  */
    x = properties.x;
    v = properties.v;
    for(j=0; j < m; j++) {
      x[k+j] = xx[j];
      v[k+j] = vv[j];
    }
    
    /*  Update best positions  */
    for(j=0; j < m; j++)
      arr[j] = xx[j];
      
    /*  Call the function or method  */
    if (properties.ftype) {
      IDL_CallRoutineByString(properties.func, &ans, 1, &arg, 0, 0);
    } else {
      IDL_ObjCallMethodByString(properties.meth, properties.obj, &ans, 1, &arg, NULL);
    }
      
    /*  Pull out the return value  */
    bf = get_numeric_value(ans);
    IDL_Deltmp(ans);
    
    /*  Decide if we update or not  */
    if (properties.maximize) {
      if (bf > (properties.bf)[i]) {
        b = properties.b;
        for(j=0; j < m; j++)
          b[k+j] = xx[j];
        (properties.bf)[i] = bf;
      }
      if ((properties.bf)[i] > properties.gf) {
        g = properties.g;
        b = properties.b;
        for(j=0; j < m; j++)
          g[j] = b[k+j];
        properties.gf = (properties.bf)[i];
        update = 1;
      }
    } else {
      if (bf < (properties.bf)[i]) {
        b = properties.b;
        for(j=0; j < m; j++)
          b[k+j] = xx[j];
        (properties.bf)[i] = bf;
      }
      if ((properties.bf)[i] < properties.gf) {
        g = properties.g;
        b = properties.b;
        for(j=0; j < m; j++)
          g[j] = b[k+j];
        properties.gf = (properties.bf)[i];
        update = 1;
      }    
    }
  }

  /*  Release temporary argument array  */
  arg->flags |= IDL_V_TEMP;
  IDL_Deltmp(arg);
  
  /*  If no update, bump count, else reset count  */
  if (!update)
    properties.gcount++;
  else
    properties.gcount = 0;
    
  /*  Count this iteration  */
  properties.i++;
  
  /*  Clean up  */
  free(xx);
  free(vv);
}


/*************************************************************
*  idl_initialize
*
*  Set up to run an optimization.
*/
static void idl_initialize(int argc, IDL_VPTR *argv) {
  double *p, *q, *b;
  IDL_MEMINT i,j;
  
  /*  
  *  Must be configured.
  */
  if (!configured)
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_MUST_CONFIGURE, IDL_MSG_LONGJMP);  

  /*
  *  Must have defined a function or method name.
  */
  if (properties.ftype) {
    if (properties.func == NULL)
      IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_NO_FUNCTION, IDL_MSG_LONGJMP);
  } else {
    if (properties.meth == NULL)
      IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_NO_FUNCTION, IDL_MSG_LONGJMP);  
  }
    
  /*  
  *  Reset iteration and g count.  
  */
  properties.i = properties.gcount = 0;

  /*  
  *  Reset initial positions.
  */
  p = properties.x;
  if (!is_zero(properties.xi, properties.m*properties.n)) {
    /*  User-supplied initial positions  */
    q = properties.xi;
    for(i=0; i < properties.m*properties.n; i++)
      *p++ = *q++;
  } else {
    /*  Randomly selected */
    if (is_zero(properties.domain, 2*properties.m)) {
      /*  No domain given  */
      for(i=0; i < properties.m*properties.n; i++)
        *p++ = randomu();
    } else {
      /*  Restrict to given domain  */
      for(i=0; i < properties.n; i++)
        random_vector_in_domain(&p[i*properties.m]);
    }
  }
  
  /* 
  *  Reset initial velocities.
  */
  p = properties.v;
  if (!is_zero(properties.vi, properties.m*properties.n)) {
    q = properties.vi;
    for(i=0; i < properties.m*properties.n; i++)
      *p++ = *q++;
  } else {
    /*  Set all to zero  */
    memset((void*)p, 0, properties.m*properties.n);
  }
  
  /*
  *  Set up initial particle best positions and values.
  */
  p = properties.x;
  b = properties.b;
  q = properties.bf;
  
  for(i=0; i < properties.n; i++) {
    IDL_VPTR ans;
    IDL_VPTR arg;
    IDL_MEMINT d=properties.m;
    double *v;
  
    /*  Initial position is the current best position  */
    for(j=0; j < properties.m; j++)
      b[i*properties.m+j] = p[i*properties.m+j];
            
    /*  Define the argument to the function  */
    IDL_MakeTempArray(IDL_TYP_DOUBLE, 1, &d, IDL_ARR_INI_NOP, &arg);
    v = (double *)arg->value.arr->data;
    for(j=0; j < properties.m; j++)
      v[j] = p[i*properties.m+j];
      
    /*  Call the function or method  */
    if (properties.ftype) {
      IDL_CallRoutineByString(properties.func, &ans, 1, &arg, 0, 0);
    } else {
      IDL_ObjCallMethodByString(properties.meth, properties.obj, &ans, 1, &arg, NULL);
    }
      
    /*  Pull out the return value  */
    q[i] = get_numeric_value(ans);
    IDL_Deltmp(ans);
    IDL_Deltmp(arg);
  }

  /*
  *  Find the initial global best.
  */
  properties.gf = (properties.bf)[0];
  for(j=0; j < properties.m; j++)
    (properties.g)[j] = (properties.b)[j];
    
  for(i=1; i < properties.n; i++) {
    if ((properties.bf)[i] > properties.gf) {
      properties.gf = (properties.bf)[i];
      for(j=0; j < properties.m; j++)
        (properties.g)[j] = (properties.b)[i*properties.m+j];
    }
  }  
}


/*************************************************************
*  check_if_done
*/
static char check_if_done(void) {

  /*  Done with iterating?  */
  if (properties.i == properties.imax)
    return 1;
    
  /*  Tolerance met?  */
  if (properties.gcount >= properties.gtol)
    return 1;
    
  /*  Nope, keep iterating  */
  return 0;
}


/*************************************************************
*  idl_done
*
*  Check if ending conditions have been met.
*/
static IDL_VPTR idl_done(int argc, IDL_VPTR *argv) {
  
  /*  Must be configured  */
  if (!configured)
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_MUST_CONFIGURE, IDL_MSG_LONGJMP);  
    
  return IDL_GettmpInt(check_if_done());  
}


/*************************************************************
*  idl_step
*
*  Do one step.
*/
static void idl_step(int argc, IDL_VPTR *argv) {

  /*  Must be configured  */
  if (!configured)
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_MUST_CONFIGURE, IDL_MSG_LONGJMP);  

  if (properties.repulsive)
    repulsive_update();
  else
    global_update();
}


/*************************************************************
*  idl_optimize
*
*  Run an optimization.
*/
static IDL_VPTR idl_optimize(int argc, IDL_VPTR *argv, char *argk) {
  IDL_VPTR ans;
  IDL_MEMINT dim,i;
  
  typedef struct {  
    IDL_KW_RESULT_FIRST_FIELD; 
    IDL_VPTR error;
    int error_present;
  } KW_RESULT; 
   
  static IDL_KW_PAR kw_pars[] = {  
    { "ERROR",  IDL_TYP_UNDEF, 1, IDL_KW_OUT, IDL_KW_OFFSETOF(error_present),  IDL_KW_OFFSETOF(error)  },  
    { NULL }  
  };  
  
  KW_RESULT kw;  

  /*  Process the keywords  */
  IDL_KWProcessByOffset(argc, argv, argk, kw_pars, (IDL_VPTR *) 0, 1, &kw); 

  /*  Must be configured  */
  if (!configured)
    IDL_MessageFromBlock(msg_block, PARTICLE_SWARM_MUST_CONFIGURE, IDL_MSG_LONGJMP);  

  /*  Empty error string  */
  if (kw.error_present)
    IDL_VarCopy(IDL_StrToSTRING(""), kw.error);
    
  /*  Set things up  */
  idl_initialize(0, NULL);
  
  /*  Iterate until done  */
  do {
    if (properties.repulsive)
      repulsive_update();
    else
      global_update();
  } while(!check_if_done());
  
  /*  Didn't "converge", call this an error  */
  if ((properties.i >= properties.imax) && (kw.error_present))
    IDL_VarCopy(IDL_StrToSTRING("Failed to met the tolerance condition."), kw.error);
  
  /*  Return the current best position  */
  dim = (IDL_MEMINT)properties.m;
  IDL_MakeTempArray(IDL_TYP_DOUBLE, 1, &dim, IDL_ARR_INI_NOP, &ans);
  for(i=0; i < properties.m; i++)
    ((double *)(ans->value.arr->data))[i] = (properties.g)[i];
  
  return ans;
}


/*************************************************************
*  IDL_Load
*
*  Tell IDL what functions are to be added to the system.
*/
#ifdef WIN32
__declspec(dllexport)
#endif
int IDL_CDECL IDL_Load(void) {

  /*  Function table  */
  static IDL_SYSFUN_DEF2 function_addr[] = {
    { (IDL_FUN_RET) idl_getprop, "PS_GETPROPERTY", 1, 1, 0, 0},  
    { (IDL_FUN_RET) idl_done, "PS_DONE", 0, 0, 0, 0},
    { (IDL_FUN_RET) idl_optimize, "PS_OPTIMIZE", 0, 0, IDL_SYSFUN_DEF_F_KEYWORDS, 0},
  };

  /*  Procedure table  */
  static IDL_SYSFUN_DEF2 procedure_addr[] = {  
    { (IDL_SYSRTN_GENERIC) idl_configure, "PS_CONFIGURE", 2, 2, 0, 0},
    { (IDL_SYSRTN_GENERIC) idl_setprop, "PS_SETPROPERTY", 2, 2, 0, 0},
    { (IDL_SYSRTN_GENERIC) idl_initialize, "PS_INITIALIZE", 0, 0, 0, 0},
    { (IDL_SYSRTN_GENERIC) idl_step, "PS_STEP", 0, 0, 0, 0},
  };  

  if (!(msg_block = IDL_MessageDefineBlock("PARTICLE_SWARM", IDL_CARRAY_ELTS(msg_arr), msg_arr))) {
    return IDL_FALSE;  
  }
 
  /*  Module initialization  */
  configured = 0;
  properties.domain = properties.xi = properties.x = properties.vi = properties.v =
  properties.b = properties.bf = properties.g = NULL;  
  properties.func = properties.meth = NULL;
  properties.obj = (IDL_HVID)NULL;
  
  /*  Seed PRNG using current time  */
  z4 = (unsigned int)time(NULL);

  /*  Tell IDL about the new routines  */
  return IDL_SysRtnAdd(function_addr,  TRUE,  IDL_CARRAY_ELTS(function_addr))
         && IDL_SysRtnAdd(procedure_addr, FALSE, IDL_CARRAY_ELTS(procedure_addr));
}
