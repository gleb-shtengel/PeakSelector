/*
*****************************************************************************
**
** gpulib.c
**
** driver/wrapper code to access GPU functions from within IDL.
** 
** Copyright (C) 2008 Tech-X Corporation. All rights reserved.
**
** This file is part of GPULib.
**
** This file may be distributed under the terms of the GNU Affero General Public
** License (AGPL). This file may be distributed and/or modified under the
** terms of the GNU Affero General Public License version 3 as published by the
** Free Software Foundation.
**
** This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
** WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
**
** Licensees holding valid Tech-X commercial licenses may use this file
** in accordance with the Tech-X Commercial License Agreement provided
** with the Software.
**
** See http://gpulib.txcorp.com/ or email sales@txcorp.com for more information.
**
** This work was in part funded by NASA SBIR Phase II Grant #NNG06CA13C.
**
*****************************************************************************
*/


#ifdef WIN32
#include <windows.h>
#endif


#include <stdio.h>
#include "idl_export.h"
#include "driver_types.h"
#include "cublas.h"
#include "cufft.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "gpuVectorOp.h"
//#include "gpuPhysicsOp.h"
#include "gpuMT.h"

/* prototype for some IDL functions */
IDL_VPTR get_IDL_long(long l);
int IDL_Load( void );

/*
 * some helper functions for sending data back to IDL
 */

IDL_VPTR get_IDL_long(long l){
 IDL_VPTR idl_l;
 if( (idl_l = IDL_Gettmp()) == (IDL_VPTR) NULL )
      IDL_Message(IDL_M_NAMED_GENERIC,IDL_MSG_LONGJMP,
             "Could not create temporary variable");
 idl_l->type    = IDL_TYP_LONG;
 idl_l->value.l = l;

 return idl_l;
}

IDL_VPTR get_IDL_float(float f){
 IDL_VPTR idl_f;

 if( (idl_f = IDL_Gettmp()) == (IDL_VPTR) NULL )
      IDL_Message(IDL_M_NAMED_GENERIC,IDL_MSG_LONGJMP,
             "Could not create temporary variable");
 idl_f->type    = IDL_TYP_FLOAT;
 idl_f->value.f = f;

 return idl_f;
}

IDL_VPTR get_IDL_complex(cuComplex c){
 IDL_VPTR idl_c;
 if( (idl_c = IDL_Gettmp()) == (IDL_VPTR) NULL )
      IDL_Message(IDL_M_NAMED_GENERIC,IDL_MSG_LONGJMP,
             "Could not create temporary variable");
 idl_c->type    = IDL_TYP_COMPLEX;
 idl_c->value.cmp.r = c.x;
 idl_c->value.cmp.i = c.y;

 return idl_c;
}



/*** 
 * the following macro is used to generate the IDL 
 * accessible wrappers for the GPU kernels.
 * Note: the function names still have to be registered
 * with IDL_Load.
 */

/*
static IDL_VPTR IDL_gpuSqrtF  (int argc, IDL_VPTR *argv)  
{                                                      
  cudaError_t err;                                     
  int n =  argv[0]->value.l;                           
  float * in  = (float*) argv[1]->value.ptrint;          
  float * res = (float*) argv[2]->value.ptrint;          
                                                       
  int rr = gpuSqrtF  (n, in, res);                  
                                                       
  return get_IDL_long( (long) rr);                     
}
*/

#define VEC_FUNC_UNARY(NAME, TYPE)                     \
                                                       \
static IDL_VPTR IDL_gpu ## NAME  (int argc, IDL_VPTR *argv)  \
{                                                      \
  cudaError_t err;                                     \
  int n =  argv[0]->value.l;                           \
  TYPE * in  = (TYPE*) argv[1]->value.ptrint;          \
  TYPE * res = (TYPE*) argv[2]->value.ptrint;          \
                                                       \
  int rr = gpu ## NAME  (n, in, res);                  \
                                                       \
  return get_IDL_long( (long) rr);                     \
} 

#define VEC_FUNC_UNARY_ALL(NAME)                      \
  VEC_FUNC_UNARY(NAME##F, float)                      \
  VEC_FUNC_UNARY(NAME##D, double)                     \
  VEC_FUNC_UNARY(NAME##C, float2)                     \
  VEC_FUNC_UNARY(NAME##Z, double2)

VEC_FUNC_UNARY_ALL(Sqrt)
VEC_FUNC_UNARY_ALL(Exp)
VEC_FUNC_UNARY_ALL(Exp2)
VEC_FUNC_UNARY_ALL(Exp10)
VEC_FUNC_UNARY_ALL(Log)
VEC_FUNC_UNARY_ALL(Log2)
VEC_FUNC_UNARY_ALL(Log10)
VEC_FUNC_UNARY_ALL(Log1p)
VEC_FUNC_UNARY_ALL(Sin)
VEC_FUNC_UNARY_ALL(Cos)
VEC_FUNC_UNARY_ALL(Tan)
VEC_FUNC_UNARY_ALL(Asin)
VEC_FUNC_UNARY_ALL(Acos)
VEC_FUNC_UNARY_ALL(Atan)

VEC_FUNC_UNARY_ALL(Erf)
VEC_FUNC_UNARY_ALL(Lgamma)
VEC_FUNC_UNARY_ALL(Tgamma)
VEC_FUNC_UNARY_ALL(Logb)
VEC_FUNC_UNARY_ALL(Trunc)
VEC_FUNC_UNARY_ALL(Round)
VEC_FUNC_UNARY_ALL(Rint)
VEC_FUNC_UNARY_ALL(Nearbyint)

VEC_FUNC_UNARY_ALL(Ceil)
VEC_FUNC_UNARY_ALL(Floor)
VEC_FUNC_UNARY_ALL(Lrint)
VEC_FUNC_UNARY_ALL(Lround)
VEC_FUNC_UNARY_ALL(Signbit)
VEC_FUNC_UNARY_ALL(Isinf)
VEC_FUNC_UNARY_ALL(Isnan)
VEC_FUNC_UNARY_ALL(Isfinite)
VEC_FUNC_UNARY_ALL(Fabs)

/***
 * the following macro is used to generate the IDL
 * accessible wrappers for the GPU kernels with affine
 * transforms.
 * Note: the function names still have to be registered
 * with IDL_Load.
 */
#define VEC_FUNC_UNARY_AT(NAME, TYPE, IDL_MEMBER)      \
                                                       \
                                                       \
static IDL_VPTR IDL_gpu ## NAME ## AT(int argc, IDL_VPTR *argv)  \
{                                                      \
  cudaError_t err;                                     \
  int      n =  argv[0]->value.l;                      \
  TYPE   a1  =  *((TYPE*)&(argv[1]->value.IDL_MEMBER));\
  TYPE   a2  =  *((TYPE*)&(argv[2]->value.IDL_MEMBER));\
  TYPE * in  = (TYPE*) argv[3]->value.ptrint;          \
  TYPE   a3  =  *((TYPE*)&(argv[4]->value.IDL_MEMBER));\
  TYPE   a4  =  *((TYPE*)&(argv[5]->value.IDL_MEMBER));\
  TYPE * res = (TYPE*) argv[6]->value.ptrint;          \
                                                       \
  int rr = gpu ## NAME ## AT(n, a1, a2, in, a3, a4,res);\
                                                       \
  return get_IDL_long( (long) rr);                     \
}


#define VEC_FUNC_UNARY_AT_ALL(NAME)                   \
  VEC_FUNC_UNARY_AT(NAME##F, float, f)                \
  VEC_FUNC_UNARY_AT(NAME##D, double, d)               \
  VEC_FUNC_UNARY_AT(NAME##C, float2, cmp)             \
  VEC_FUNC_UNARY_AT(NAME##Z, double2, dcmp)


VEC_FUNC_UNARY_AT_ALL(Sqrt)
VEC_FUNC_UNARY_AT_ALL(Exp)
VEC_FUNC_UNARY_AT_ALL(Exp2)
VEC_FUNC_UNARY_AT_ALL(Exp10)
VEC_FUNC_UNARY_AT_ALL(Log)
VEC_FUNC_UNARY_AT_ALL(Log2)
VEC_FUNC_UNARY_AT_ALL(Log10)
VEC_FUNC_UNARY_AT_ALL(Log1p)
VEC_FUNC_UNARY_AT_ALL(Sin)
VEC_FUNC_UNARY_AT_ALL(Cos)
VEC_FUNC_UNARY_AT_ALL(Tan)
VEC_FUNC_UNARY_AT_ALL(Asin)
VEC_FUNC_UNARY_AT_ALL(Acos)
VEC_FUNC_UNARY_AT_ALL(Atan)
VEC_FUNC_UNARY_AT_ALL(Erf)
VEC_FUNC_UNARY_AT_ALL(Lgamma)
VEC_FUNC_UNARY_AT_ALL(Tgamma)
VEC_FUNC_UNARY_AT_ALL(Logb)
VEC_FUNC_UNARY_AT_ALL(Trunc)
VEC_FUNC_UNARY_AT_ALL(Round)
VEC_FUNC_UNARY_AT_ALL(Rint)
VEC_FUNC_UNARY_AT_ALL(Nearbyint)
VEC_FUNC_UNARY_AT_ALL(Ceil)
VEC_FUNC_UNARY_AT_ALL(Floor)
VEC_FUNC_UNARY_AT_ALL(Lrint)
VEC_FUNC_UNARY_AT_ALL(Lround)
VEC_FUNC_UNARY_AT_ALL(Signbit)
VEC_FUNC_UNARY_AT_ALL(Isinf)
VEC_FUNC_UNARY_AT_ALL(Isnan)
VEC_FUNC_UNARY_AT_ALL(Isfinite)
VEC_FUNC_UNARY_AT_ALL(Fabs)

/***
 * the following macro is used to generate the IDL
 * accessible wrappers for the GPU kernels of binary operators
 */

#define VEC_FUNC_BINARY(NAME, TYPE, RES_TYPE)        \
                                                     \
                                                     \
static IDL_VPTR IDL_gpu ## NAME (int argc, IDL_VPTR *argv)  \
{                                                    \
  cudaError_t err;                                   \
  int n =  argv[0]->value.l;                         \
  TYPE * ina = (TYPE*) argv[1]->value.ptrint;        \
  TYPE * inb = (TYPE*) argv[2]->value.ptrint;        \
  RES_TYPE * res = (RES_TYPE*) argv[3]->value.ptrint;\
                                                     \
  int rr = gpu ## NAME (n, ina, inb, res);           \
                                                     \
  return get_IDL_long( (long) rr);                   \
} 


#define VEC_FUNC_BINARY_ALL(NAME)                   \
  VEC_FUNC_BINARY(NAME##F, float, float)            \
  VEC_FUNC_BINARY(NAME##D, double, double)          \
  VEC_FUNC_BINARY(NAME##C, float2, float2)          \
  VEC_FUNC_BINARY(NAME##Z, double2, double2)


VEC_FUNC_BINARY_ALL(Add)
VEC_FUNC_BINARY_ALL(Sub)
VEC_FUNC_BINARY_ALL(Mult)
VEC_FUNC_BINARY_ALL(Div)

#define VEC_FUNC_RELATIONAL_ALL(NAME)               \
  VEC_FUNC_BINARY(NAME##F, float, float)            \
  VEC_FUNC_BINARY(NAME##D, double, float)           \
  VEC_FUNC_BINARY(NAME##C, float2, float)           \
  VEC_FUNC_BINARY(NAME##Z, double2, float)

VEC_FUNC_RELATIONAL_ALL(Eq)
VEC_FUNC_RELATIONAL_ALL(Neq)
VEC_FUNC_RELATIONAL_ALL(Lt)
VEC_FUNC_RELATIONAL_ALL(Gt)
VEC_FUNC_RELATIONAL_ALL(LtEq)
VEC_FUNC_RELATIONAL_ALL(GtEq)

/***
 * the following macro is used to generate the IDL
 * accessible wrappers for the GPU kernels of binary operators
 */
#define VEC_FUNC_BINARY_AT(NAME, TYPE, RES_TYPE, IDL_MEMBER)     \
                                                                 \
static IDL_VPTR IDL_gpu ## NAME ## AT(int argc, IDL_VPTR *argv)  \
{                                                                \
  cudaError_t err;                                               \
  int  n =  argv[0]->value.l;                                    \
  TYPE   a1  = *((TYPE*)&(argv[1]->value.IDL_MEMBER));           \
  TYPE * ina = (TYPE*) argv[2]->value.ptrint;                    \
  TYPE   a2  = *((TYPE*)&(argv[3]->value.IDL_MEMBER));           \
  TYPE * inb = (TYPE*) argv[4]->value.ptrint;                    \
  TYPE   a3  = *((TYPE*)&(argv[5]->value.IDL_MEMBER));           \
  RES_TYPE * res = (RES_TYPE*) argv[6]->value.ptrint;            \
                                                                 \
  int rr = gpu ## NAME ## AT(n, a1, ina, a2, inb, a3, res);      \
                                                                 \
  return get_IDL_long( (long) rr);                               \
}

#define VEC_FUNC_BINARY_AT_ALL(NAME)                             \
  VEC_FUNC_BINARY_AT(NAME##F, float, float, f)                   \
  VEC_FUNC_BINARY_AT(NAME##D, double, double, d)                 \
  VEC_FUNC_BINARY_AT(NAME##C, float2, float2, cmp)               \
  VEC_FUNC_BINARY_AT(NAME##Z, double2, double2, dcmp)

VEC_FUNC_BINARY_AT_ALL(Add)
VEC_FUNC_BINARY_AT_ALL(Sub)
VEC_FUNC_BINARY_AT_ALL(Mult)
VEC_FUNC_BINARY_AT_ALL(Div)

#define VEC_FUNC_RELATIONAL_AT_ALL(NAME)                        \
  VEC_FUNC_BINARY_AT(NAME##F, float, float, f)                  \
  VEC_FUNC_BINARY_AT(NAME##D, double, float, d)                 \
  VEC_FUNC_BINARY_AT(NAME##C, float2, float, cmp)               \
  VEC_FUNC_BINARY_AT(NAME##Z, double2, float, dcmp)

VEC_FUNC_RELATIONAL_AT_ALL(Eq)
VEC_FUNC_RELATIONAL_AT_ALL(Neq)
VEC_FUNC_RELATIONAL_AT_ALL(Lt)
VEC_FUNC_RELATIONAL_AT_ALL(Gt)
VEC_FUNC_RELATIONAL_AT_ALL(LtEq)
VEC_FUNC_RELATIONAL_AT_ALL(GtEq)

/***
 * the following macro is used to generate the IDL
 * accessible wrappers for the GPU kernels of type cast operators
 */
#define CAST_OP(NAME, TYPE, RES_TYPE)        \
                                                    \
static IDL_VPTR IDL_gpu ## NAME (int argc, IDL_VPTR *argv)  \
{                                                    \
  cudaError_t err;                                   \
  int  n =  argv[0]->value.l;                        \
  TYPE * in = (TYPE*) argv[1]->value.ptrint;         \
  RES_TYPE * res = (RES_TYPE*) argv[2]->value.ptrint;\
                                                     \
  int rr = gpu ## NAME (n, in, res);                 \
                                                     \
  return get_IDL_long( (long) rr);                   \
}

CAST_OP(FloatToDouble, float, double)
CAST_OP(DoubleToFloat, double, float)

CAST_OP(FloatToComplexReal, float, float2)
CAST_OP(ComplexRealToFloat, float2, float)
CAST_OP(FloatToComplexImag, float, float2)
CAST_OP(ComplexImagToFloat, float2, float)

CAST_OP(FloatToDcomplexReal, float, double2)
CAST_OP(DcomplexRealToFloat, double2, float)
CAST_OP(FloatToDcomplexImag, float, double2)
CAST_OP(DcomplexImagToFloat, double2, float)

CAST_OP(DoubleToComplexReal, double, float2)
CAST_OP(ComplexRealToDouble, float2, double)
CAST_OP(DoubleToComplexImag, double, float2)
CAST_OP(ComplexImagToDouble, float2, double)

CAST_OP(DoubleToDcomplexReal, double, double2)
CAST_OP(DcomplexRealToDouble, double2, double)
CAST_OP(DoubleToDcomplexImag, double, double2)
CAST_OP(DcomplexImagToDouble, double2, double)

#undef CAST_OP


 

 /****************************************************************/
 /*                                                              */
 /*                            CUBLAS                            */
 /*                                                              */
 /****************************************************************/

/* cublasInit() */
static IDL_VPTR IDL_cublasInit(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;

  stat =   cublasInit();

  return get_IDL_long( stat);
}

/* cublasShutdown() */
static IDL_VPTR IDL_cublasShutdown(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;

  stat =   cublasShutdown();

  return get_IDL_long( stat);
}

/* cublasGetError() */
static IDL_VPTR IDL_cublasGetError(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;

  stat =   cublasGetError();

  return get_IDL_long( stat);
}


/* cublasAlloc() */
static IDL_VPTR IDL_cublasAlloc(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  long n = (long) argv[0]->value.l;
  long elemSize = (long) argv[1]->value.l;
  void* devicePtr;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);

  cublasAlloc(n, elemSize, &devicePtr);
  
  // clear return variable
  IDL_StoreScalarZero(argv[2], IDL_TYP_LONG);

  argv[2]->type = IDL_TYP_LONG;
  argv[2]->value.ptrint = (long) devicePtr;

  stat =   cublasGetError();

  return get_IDL_long( stat);
}

/* cublasFree() */
static IDL_VPTR IDL_cublasFree(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  void* devicePtr = (void*) argv[0]->value.ptrint;
  IDL_ENSURE_SIMPLE(argv[0]);

  cublasFree(devicePtr);
  stat =   cublasGetError();

  return get_IDL_long( stat);
}

/* cublasSetVector() */
static IDL_VPTR IDL_cublasSetVector(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  long n = argv[0]->value.l;
  long elemSize = (long) argv[1]->value.l;
  void* x = (void*) argv[2]->value.arr->data;
  long incx = argv[3]->value.l;
  void* y = (void*) argv[4]->value.ptrint;
  long incy = (long) argv[5]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_ARRAY(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);

  cublasSetVector(n, elemSize, x, incx, y, incy);

  stat =   cublasGetError();

  return get_IDL_long( stat);
}

/* cublasGetVector() */
static IDL_VPTR IDL_cublasGetVector(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  long n = argv[0]->value.l;
  long elemSize = (long) argv[1]->value.l;
  void* x = (void*) argv[2]->value.ptrint;
  long incx = argv[3]->value.l;
  void* y = (void*) argv[4]->value.arr->data;
  long incy = argv[5]->value.l;


  // cudaMemcpy(y, x, n * elemSize, cudaMemcpyDeviceToHost);

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_ARRAY(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);

  cublasGetVector(n, elemSize, x, incx, y, incy);

  stat =   cublasGetError();

  return get_IDL_long( stat);
}

/* cublasSetMatrix() */
static IDL_VPTR IDL_cublasSetMatrix(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  long rows = argv[0]->value.l;
  long cols = argv[1]->value.l;
  long elemSize = argv[2]->value.l;
  void* A = (void*) argv[3]->value.arr->data;
  long lda = argv[4]->value.l;
  void* B = (void*) argv[5]->value.ptrint;
  long ldb = argv[6]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_ARRAY(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);
  IDL_ENSURE_SIMPLE(argv[6]);

  cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb);

  stat =   cublasGetError();

  return get_IDL_long( stat);
}

/* cublasGetMatrix() */
static IDL_VPTR IDL_cublasGetMatrix(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  long rows = argv[0]->value.l;
  long cols = argv[1]->value.l;
  long elemSize = (long) argv[2]->value.l;
  void* A = (void*) argv[3]->value.ptrint;
  long lda = argv[4]->value.l;
  void* B = (void*) argv[5]->value.arr->data;
  long ldb = argv[6]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);
  IDL_ENSURE_ARRAY(argv[5]);
  IDL_ENSURE_SIMPLE(argv[6]);

  cublasGetMatrix(rows, cols, elemSize, A, lda, B, ldb);

  stat =   cublasGetError();

  return get_IDL_long( stat);
}

/*  ********************************************************** */
/*  ****    BLAS LEVEL 1 Routines                         **** */
/*  ********************************************************** */

/* cublasIsamax() */
static IDL_VPTR IDL_cublasIsamax(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  long n = argv[0]->value.l;
  float* x = (float*) argv[1]->value.ptrint;
  long incx = argv[2]->value.l;
  long indx;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);

  indx = cublasIsamax(n, x, incx);

  return get_IDL_long( indx);
}

/* cublasSasum() */
static IDL_VPTR IDL_cublasSasum(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  long n = (long) argv[0]->value.l;
  float* x = (float*) argv[1]->value.ptrint;
  long incx = (long) argv[2]->value.l;
  float sum;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);

  sum = cublasSasum(n, x, incx);

  return get_IDL_float( sum);
}

/* cublasSaxpy() */
static IDL_VPTR IDL_cublasSaxpy(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  long n = argv[0]->value.l;
  float alpha = (float) argv[1]->value.f;
  float* x = (float*) argv[2]->value.ptrint;
  long incx = argv[3]->value.l;
  float* y = (float*) argv[4]->value.ptrint;
  long incy = argv[5]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);

  cublasSaxpy(n, alpha, x, incx, y, incy);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}

/* cublasScopy() */
static IDL_VPTR IDL_cublasScopy(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  long n = argv[0]->value.l;
  float* x = (float*) argv[1]->value.ptrint;
  long incx = argv[2]->value.l;
  float* y = (float*) argv[3]->value.ptrint;
  long incy = argv[4]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);

  cublasScopy(n, x, incx, y, incy);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}

/* cublasSdot() */
static IDL_VPTR IDL_cublasSdot(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  long n = (long) argv[0]->value.l;
  float* x = (float*) argv[1]->value.ptrint;
  long incx = (long) argv[2]->value.l;
  float* y = (float*) argv[3]->value.ptrint;
  long incy = (long) argv[4]->value.l;
  float dot;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);

  dot = cublasSdot(n, x, incx, y, incy);
  return get_IDL_float( dot);
}

/* cublasSnrm2 */
static IDL_VPTR IDL_cublasSnrm2(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  long n = (long) argv[0]->value.l;
  float* x = (float*) argv[1]->value.ptrint;
  long incx = (long) argv[2]->value.l;
  float nrm2;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);

  nrm2 = cublasSnrm2(n, x, incx);
  return get_IDL_float( nrm2);
}

/* cublasSrot() */
static IDL_VPTR IDL_cublasSrot(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  long n = (long) argv[0]->value.l;
  float* x = (float*) argv[1]->value.ptrint;
  long incx = (long) argv[2]->value.l;
  float* y = (float*) argv[3]->value.ptrint;
  long incy = (long) argv[4]->value.l;
  float sc = (float) argv[5]->value.f;
  float ss = (float) argv[6]->value.f;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);
  IDL_ENSURE_SIMPLE(argv[6]);

  cublasSrot(n, x, incx, y, incy, sc, ss);

  stat =   cublasGetError();
  return get_IDL_long( stat);
}


/* cublasSrotg() */
static IDL_VPTR IDL_cublasSrotg(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  float sa =  argv[0]->value.f;
  float sb =  argv[1]->value.f;
  float sc =  argv[2]->value.f;
  float ss =  argv[3]->value.f;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);

  cublasSrotg(&sa, &sb, &sc, &ss);
  
  argv[0]->value.f = sa;
  argv[1]->value.f = sb;
  argv[2]->value.f = sc;
  argv[3]->value.f = ss;

  stat =   cublasGetError();
  return get_IDL_long( stat);
}

/* cublasSrotm() */
static IDL_VPTR IDL_cublasSrotm(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  long n = (long) argv[0]->value.l;
  float* x = (float*) argv[1]->value.ptrint;
  long incx = (long) argv[2]->value.l;
  float* y = (float*) argv[3]->value.ptrint;
  long incy = (long) argv[4]->value.l;
  float* sparam = (float*) argv[5]->value.arr->data;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);
  IDL_ENSURE_ARRAY(argv[5]);

  cublasSrotm(n, x, incx, y, incy, sparam);

  stat =   cublasGetError();
  return get_IDL_long( stat);
}

/* cublasSrotmg() */
static IDL_VPTR IDL_cublasSrotmg(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  float sd1 = argv[0]->value.f;
  float sd2 = argv[1]->value.f;
  float sx1 = argv[2]->value.f;
  float sy1 = argv[3]->value.f;
  float* sparam = (float*) argv[4]->value.arr->data;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_ARRAY(argv[4]);

  cublasSrotmg(&sd1, &sd2, &sx1, &sy1, sparam);

  argv[0]->value.f = sd1;
  argv[1]->value.f = sd2;
  argv[2]->value.f = sx1;

  stat =   0; /* this function does not set an error */
  return get_IDL_long( stat);
}

/* cublasSscal() */
static IDL_VPTR IDL_cublasSscal(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  long n = (long) argv[0]->value.l;
  float alpha = (float) argv[1]->value.f;
  float* x = (float*) argv[2]->value.ptrint;
  long incx = (long) argv[3]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);

  cublasSscal(n, alpha, x, incx);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}


/* cublasSswap() */
static IDL_VPTR IDL_cublasSswap(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  long n = (long) argv[0]->value.l;
  float* x = (float*) argv[1]->value.ptrint;
  long incx = (long) argv[2]->value.l;
  float* y = (float*) argv[3]->value.ptrint;
  long incy = (long) argv[4]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);

  cublasSswap(n, x, incx, y, incy);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}

/*  ********************************************************** */
/*  ****    BLAS LEVEL 1 Routines  (complex)              **** */
/*  ********************************************************** */

/* cublasCaxpy() */
static IDL_VPTR IDL_cublasCaxpy(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  long n = (long) argv[0]->value.l;
  IDL_COMPLEX alpha_idl =  argv[1]->value.cmp;
  cuComplex* x = (cuComplex*) argv[2]->value.ptrint;
  long incx = (long) argv[3]->value.l;
  cuComplex* y = (cuComplex*) argv[4]->value.ptrint;
  long incy = (long) argv[5]->value.l;
  cuComplex alpha;
  
  alpha.x = alpha_idl.r;
  alpha.y = alpha_idl.i;
 
  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);

  cublasCaxpy(n, alpha, x, incx, y, incy);

  stat =   cublasGetError();
  return get_IDL_long( stat);

} 

/* cublasCcopy() */
static IDL_VPTR IDL_cublasCcopy(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  long n = (long) argv[0]->value.l;
  cuComplex* x = (cuComplex*) argv[1]->value.ptrint;
  long incx = (long) argv[2]->value.l;
  cuComplex* y = (cuComplex*) argv[3]->value.ptrint;
  long incy = (long) argv[4]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);

  cublasCcopy(n, x, incx, y, incy);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}

/* cublasCdotu() */
static IDL_VPTR IDL_cublasCdotu(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  long n = (long) argv[0]->value.l;
  cuComplex* x = (cuComplex*) argv[1]->value.ptrint;
  long incx = (long) argv[2]->value.l;
  cuComplex* y = (cuComplex*) argv[3]->value.ptrint;
  long incy = (long) argv[4]->value.l;
  cuComplex dot;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);

  dot = cublasCdotu(n, x, incx, y, incy);
  return get_IDL_complex( dot);
}


/* cublasCscal() */
static IDL_VPTR IDL_cublasCscal(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  long n = (long) argv[0]->value.l;
  IDL_COMPLEX alpha_idl =  argv[1]->value.cmp;
  cuComplex* x = (cuComplex*) argv[2]->value.ptrint;
  long incx = (long) argv[3]->value.l;
  cuComplex alpha;

  alpha.x = alpha_idl.r;
  alpha.y = alpha_idl.i;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);

  cublasCscal(n, alpha, x, incx);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}

/* cublasCsscal() */
static IDL_VPTR IDL_cublasCsscal(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  long n = (long) argv[0]->value.l;
  float alpha = (float) argv[1]->value.f;
  cuComplex* x = (cuComplex*) argv[2]->value.ptrint;
  long incx = (long) argv[3]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);

  cublasCsscal(n, alpha, x, incx);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}


/* cublasCswap() */
static IDL_VPTR IDL_cublasCswap(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  long n = (long) argv[0]->value.l;
  cuComplex* x = (cuComplex*) argv[1]->value.ptrint;
  long incx = (long) argv[2]->value.l;
  cuComplex* y = (cuComplex*) argv[3]->value.ptrint;
  long incy = (long) argv[4]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);

  cublasCswap(n, x, incx, y, incy);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}


/* cublasScasum() */
static IDL_VPTR IDL_cublasScasum(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  long n = (long) argv[0]->value.l;
  cuComplex* x = (cuComplex*) argv[1]->value.ptrint;
  long incx = (long) argv[2]->value.l;
  float sum;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);

  sum = cublasScasum(n, x, incx);

  return get_IDL_float( sum);
}

/*  ********************************************************** */
/*  ****    BLAS LEVEL 2 Routines                         **** */
/*  ********************************************************** */

/* cublasSgbmv() */
static IDL_VPTR IDL_cublasSgbmv(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  char trans = (char) argv[0]->value.str.s[0];
  long m = (long) argv[1]->value.l;
  long n = (long) argv[2]->value.l;
  long kl = (long) argv[3]->value.l;
  long ku = (long) argv[4]->value.l;
  float alpha = (float) argv[5]->value.f;
  float* A = (float*) argv[6]->value.ptrint;
  long lda = (long) argv[7]->value.l;
  float* x = (float*) argv[8]->value.ptrint;
  long incx = (long) argv[9]->value.l;
  float beta = (float) argv[10]->value.f;
  float* y = (float*) argv[11]->value.ptrint;
  long incy = (long) argv[12]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);
  IDL_ENSURE_SIMPLE(argv[6]);
  IDL_ENSURE_SIMPLE(argv[7]);
  IDL_ENSURE_SIMPLE(argv[8]);
  IDL_ENSURE_SIMPLE(argv[9]);
  IDL_ENSURE_SIMPLE(argv[10]);
  IDL_ENSURE_SIMPLE(argv[11]);
  IDL_ENSURE_SIMPLE(argv[12]);

  cublasSgbmv(trans, m, n, kl, ku, 
            alpha, A, lda, 
            x, incx, beta, y, incy);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}

/* cublasSgemv() */
static IDL_VPTR IDL_cublasSgemv(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  char trans = (char) argv[0]->value.str.s[0];
  long m = (long) argv[1]->value.l;
  long n = (long) argv[2]->value.l;
  float alpha = (float) argv[3]->value.f;
  float* A = (float*) argv[4]->value.ptrint;
  long lda = (long) argv[5]->value.l;
  float* x = (float*) argv[6]->value.ptrint;
  long incx = (long) argv[7]->value.l;
  float beta = (float) argv[8]->value.f;
  float* y = (float*) argv[9]->value.ptrint;
  long incy = (long) argv[10]->value.l;


  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);
  IDL_ENSURE_SIMPLE(argv[6]);
  IDL_ENSURE_SIMPLE(argv[7]);
  IDL_ENSURE_SIMPLE(argv[8]);
  IDL_ENSURE_SIMPLE(argv[9]);
  IDL_ENSURE_SIMPLE(argv[10]);

  cublasSgemv(trans, m, n, alpha, 
              A, lda, x, incx, beta, y, incy);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}

/* cublasSger() */
static IDL_VPTR IDL_cublasSger(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  long m = (long) argv[0]->value.l;
  long n = (long) argv[1]->value.l;
  float alpha = (float) argv[2]->value.f;
  float* x = (float*) argv[3]->value.ptrint;
  long incx = (long) argv[4]->value.l;
  float* y = (float*) argv[5]->value.ptrint;
  long incy = (long) argv[6]->value.l;
  float* A = (float*) argv[7]->value.ptrint;
  long lda = (long) argv[8]->value.l;


  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);
  IDL_ENSURE_SIMPLE(argv[6]);
  IDL_ENSURE_SIMPLE(argv[7]);
  IDL_ENSURE_SIMPLE(argv[8]);

  cublasSger(m, n, alpha, x, incx, y, incy, A, lda);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}

/* cublasSsbmv() */
static IDL_VPTR IDL_cublasSsbmv(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  char uplo = (char) argv[0]->value.str.s[0];
  long n = (long) argv[1]->value.l;
  long k = (long) argv[2]->value.l;
  float alpha = (float) argv[3]->value.f;
  float* A = (float*) argv[4]->value.ptrint;
  long lda = (long) argv[5]->value.l;
  float* x = (float*) argv[6]->value.ptrint;
  long incx = (long) argv[7]->value.l;
  float beta = (float) argv[8]->value.f;
  float* y = (float*) argv[9]->value.ptrint;
  long incy = (long) argv[10]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);
  IDL_ENSURE_SIMPLE(argv[6]);
  IDL_ENSURE_SIMPLE(argv[7]);
  IDL_ENSURE_SIMPLE(argv[8]);
  IDL_ENSURE_SIMPLE(argv[9]);
  IDL_ENSURE_SIMPLE(argv[10]);

  cublasSsbmv(uplo, n, k, alpha, 
           A, lda, x, 
          incx, beta, y, incy);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}

/* cublasSspmv() */
static IDL_VPTR IDL_cublasSspmv(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  char uplo = (char) argv[0]->value.str.s[0];
  long n = (long) argv[1]->value.l;
  float alpha = (float) argv[2]->value.f;
  float* AP = (float*) argv[3]->value.ptrint;
  float* x = (float*) argv[4]->value.ptrint;
  long incx = (long) argv[5]->value.l;
  float beta = (float) argv[6]->value.f;
  float* y = (float*) argv[7]->value.ptrint;
  long incy = (long) argv[8]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);
  IDL_ENSURE_SIMPLE(argv[6]);
  IDL_ENSURE_SIMPLE(argv[7]);
  IDL_ENSURE_SIMPLE(argv[8]);

  cublasSspmv(uplo, n,  alpha, 
           AP, x, 
          incx, beta, y, incy);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}

/* cublasSspr() */
static IDL_VPTR IDL_cublasSspr(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  char uplo = (char) argv[0]->value.str.s[0];
  long n = (long) argv[1]->value.l;
  float alpha = (float) argv[2]->value.f;
  float* x = (float*) argv[3]->value.ptrint;
  long incx = (long) argv[4]->value.l;
  float* AP = (float*) argv[5]->value.ptrint;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);

  cublasSspr(uplo, n,  alpha, x, incx, AP);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}

/* cublasSspr2() */
static IDL_VPTR IDL_cublasSspr2(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  char uplo = (char) argv[0]->value.str.s[0];
  long n = (long) argv[1]->value.l;
  float alpha = (float) argv[2]->value.f;
  float* x = (float*) argv[3]->value.ptrint;
  long incx = (long) argv[4]->value.l;
  float* y = (float*) argv[5]->value.ptrint;
  long incy = (long) argv[6]->value.l;
  float* AP = (float*) argv[7]->value.ptrint;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);
  IDL_ENSURE_SIMPLE(argv[6]);
  IDL_ENSURE_SIMPLE(argv[7]);

  cublasSspr2(uplo, n,  alpha, x, incx, y, incy, AP);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}

/* cublasSsymv() */
static IDL_VPTR IDL_cublasSsymv(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  char uplo = (char) argv[0]->value.str.s[0];
  long n = (long) argv[1]->value.l;
  float alpha = (float) argv[2]->value.f;
  float* A = (float*) argv[3]->value.ptrint;
  long lda = (long) argv[4]->value.l;
  float* x = (float*) argv[5]->value.ptrint;
  long incx = (long) argv[6]->value.l;
  float beta = (float) argv[7]->value.f;
  float* y = (float*) argv[8]->value.ptrint;
  long incy = (long) argv[9]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);
  IDL_ENSURE_SIMPLE(argv[6]);
  IDL_ENSURE_SIMPLE(argv[7]);
  IDL_ENSURE_SIMPLE(argv[8]);
  IDL_ENSURE_SIMPLE(argv[9]);

  cublasSsymv(uplo, n,  alpha, A, lda, x, incx, beta, y, incy);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}

/* cublasSsyr() */
static IDL_VPTR IDL_cublasSsyr(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  char uplo = (char) argv[0]->value.str.s[0];
  long n = (long) argv[1]->value.l;
  float alpha = (float) argv[2]->value.f;
  float* x = (float*) argv[3]->value.ptrint;
  long incx = (long) argv[4]->value.l;
  float* A = (float*) argv[5]->value.ptrint;
  long lda = (long) argv[6]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);
  IDL_ENSURE_SIMPLE(argv[6]);

  cublasSsyr(uplo, n,  alpha, x, incx, A, lda);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}

/* cublasSsyr2() */
static IDL_VPTR IDL_cublasSsyr2(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  char uplo = (char) argv[0]->value.str.s[0];
  long n = (long) argv[1]->value.l;
  float alpha = (float) argv[2]->value.f;
  float* x = (float*) argv[3]->value.ptrint;
  long incx = (long) argv[4]->value.l;
  float* y = (float*) argv[5]->value.ptrint;
  long incy = (long) argv[6]->value.l;
  float* A = (float*) argv[7]->value.ptrint;
  long lda = (long) argv[8]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);
  IDL_ENSURE_SIMPLE(argv[6]);
  IDL_ENSURE_SIMPLE(argv[7]);
  IDL_ENSURE_SIMPLE(argv[8]);

  cublasSsyr2(uplo, n, alpha, x, incx, y, incy,  A, lda);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}

/* cublasStbmv() */
static IDL_VPTR IDL_cublasStbmv(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  char uplo = (char) argv[0]->value.str.s[0];
  char trans = (char) argv[1]->value.str.s[0];
  char diag = (char) argv[2]->value.str.s[0];
  long n = (long) argv[3]->value.l;
  long k = (long) argv[4]->value.l;
  float* A = (float*) argv[5]->value.ptrint;
  long lda = (long) argv[6]->value.l;
  float* x = (float*) argv[7]->value.ptrint;
  long incx = (long) argv[8]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);
  IDL_ENSURE_SIMPLE(argv[6]);
  IDL_ENSURE_SIMPLE(argv[7]);
  IDL_ENSURE_SIMPLE(argv[8]);

  cublasStbmv(uplo, trans, diag, n, k, A, lda, x, incx);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}

/* cublasStbsv() */
static IDL_VPTR IDL_cublasStbsv(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  char uplo = (char) argv[0]->value.str.s[0];
  char trans = (char) argv[1]->value.str.s[0];
  char diag = (char) argv[2]->value.str.s[0];
  long n = (long) argv[3]->value.l;
  long k = (long) argv[4]->value.l;
  float* A = (float*) argv[5]->value.ptrint;
  long lda = (long) argv[6]->value.l;
  float* x = (float*) argv[7]->value.ptrint;
  long incx = (long) argv[8]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);
  IDL_ENSURE_SIMPLE(argv[6]);
  IDL_ENSURE_SIMPLE(argv[7]);
  IDL_ENSURE_SIMPLE(argv[8]);

  cublasStbsv(uplo, trans, diag, n, k, A, lda, x, incx);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}

/* cublasStpmv() */
static IDL_VPTR IDL_cublasStpmv(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  char uplo = (char) argv[0]->value.str.s[0];
  char trans = (char) argv[1]->value.str.s[0];
  char diag = (char) argv[2]->value.str.s[0];
  long n = (long) argv[3]->value.l;
  float* AP= (float*) argv[4]->value.ptrint;
  float* x = (float*) argv[5]->value.ptrint;
  long incx = (long) argv[6]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);
  IDL_ENSURE_SIMPLE(argv[6]);

  cublasStpmv(uplo, trans, diag, n, AP, x, incx);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}

/* cublasStpsv() */
static IDL_VPTR IDL_cublasStpsv(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  char uplo = (char) argv[0]->value.str.s[0];
  char trans = (char) argv[1]->value.str.s[0];
  char diag = (char) argv[2]->value.str.s[0];
  long n = (long) argv[3]->value.l;
  float* AP= (float*) argv[4]->value.ptrint;
  float* x = (float*) argv[5]->value.ptrint;
  long incx = (long) argv[6]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);
  IDL_ENSURE_SIMPLE(argv[6]);

  cublasStpsv(uplo, trans, diag, n, AP, x, incx);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}

/* cublasStrmv() */
static IDL_VPTR IDL_cublasStrmv(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  char uplo = (char) argv[0]->value.str.s[0];
  char trans = (char) argv[1]->value.str.s[0];
  char diag = (char) argv[2]->value.str.s[0];
  long n = (long) argv[3]->value.l;
  float* A= (float*) argv[4]->value.ptrint;
  long lda = (long) argv[5]->value.l;
  float* x = (float*) argv[6]->value.ptrint;
  long incx = (long) argv[7]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);
  IDL_ENSURE_SIMPLE(argv[6]);
  IDL_ENSURE_SIMPLE(argv[7]);

  cublasStrmv(uplo, trans, diag, n, A, lda, x, incx);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}

/* cublasStrsv() */
static IDL_VPTR IDL_cublasStrsv(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  char uplo = (char) argv[0]->value.str.s[0];
  char trans = (char) argv[1]->value.str.s[0];
  char diag = (char) argv[2]->value.str.s[0];
  long n = (long) argv[3]->value.l;
  float* A= (float*) argv[4]->value.ptrint;
  long lda = (long) argv[5]->value.l;
  float* x = (float*) argv[6]->value.ptrint;
  long incx = (long) argv[7]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);
  IDL_ENSURE_SIMPLE(argv[6]);
  IDL_ENSURE_SIMPLE(argv[7]);

  cublasStrsv(uplo, trans, diag, n, A, lda, x, incx);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}

/*  ********************************************************** */
/*  ****    BLAS LEVEL 3 Routines                         **** */
/*  ********************************************************** */

/* cublasSgemm() */
static IDL_VPTR IDL_cublasSgemm(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  char transa = (char) argv[0]->value.str.s[0];
  char transb = (char) argv[1]->value.str.s[0];
  long m = (long) argv[2]->value.l;
  long n = (long) argv[3]->value.l;
  long k = (long) argv[4]->value.l;
  float alpha= (float) argv[5]->value.f;
  float* A= (float*) argv[6]->value.ptrint;
  long lda = (long) argv[7]->value.l;
  float* B= (float*) argv[8]->value.ptrint;
  long ldb = (long) argv[9]->value.l;
  float beta= (float) argv[10]->value.f;
  float* C= (float*) argv[11]->value.ptrint;
  long ldc = (long) argv[12]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);
  IDL_ENSURE_SIMPLE(argv[6]);
  IDL_ENSURE_SIMPLE(argv[7]);
  IDL_ENSURE_SIMPLE(argv[8]);
  IDL_ENSURE_SIMPLE(argv[9]);
  IDL_ENSURE_SIMPLE(argv[10]);
  IDL_ENSURE_SIMPLE(argv[11]);
  IDL_ENSURE_SIMPLE(argv[12]);

  cublasSgemm(transa, transb, m, n, k, alpha, A, lda, 
             B, ldb, beta, C, ldc);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}

/* cublasSsymm() */
static IDL_VPTR IDL_cublasSsymm(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  char side = (char) argv[0]->value.str.s[0];
  char uplo = (char) argv[1]->value.str.s[0];
  long m = (long) argv[2]->value.l;
  long n = (long) argv[3]->value.l;
  float alpha= (float) argv[4]->value.f;
  float* A= (float*) argv[5]->value.ptrint;
  long lda = (long) argv[6]->value.l;
  float* B= (float*) argv[7]->value.ptrint;
  long ldb = (long) argv[8]->value.l;
  float beta= (float) argv[9]->value.f;
  float* C= (float*) argv[10]->value.ptrint;
  long ldc = (long) argv[11]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);
  IDL_ENSURE_SIMPLE(argv[6]);
  IDL_ENSURE_SIMPLE(argv[7]);
  IDL_ENSURE_SIMPLE(argv[8]);
  IDL_ENSURE_SIMPLE(argv[9]);
  IDL_ENSURE_SIMPLE(argv[10]);
  IDL_ENSURE_SIMPLE(argv[11]);

  cublasSsymm(side, uplo, m, n, alpha, A, lda, 
             B, ldb, beta, C, ldc);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}

/* cublasSsyrk() */
static IDL_VPTR IDL_cublasSsyrk(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  char uplo = (char) argv[0]->value.str.s[0];
  char trans = (char) argv[1]->value.str.s[0];
  long n = (long) argv[2]->value.l;
  long k = (long) argv[3]->value.l;
  float alpha= (float) argv[4]->value.f;
  float* A= (float*) argv[5]->value.ptrint;
  long lda = (long) argv[6]->value.l;
  float beta= (float) argv[7]->value.f;
  float* C= (float*) argv[8]->value.ptrint;
  long ldc = (long) argv[9]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);
  IDL_ENSURE_SIMPLE(argv[6]);
  IDL_ENSURE_SIMPLE(argv[7]);
  IDL_ENSURE_SIMPLE(argv[8]);
  IDL_ENSURE_SIMPLE(argv[9]);

  cublasSsyrk(uplo, trans,  n, k, alpha, A, lda, 
             beta, C, ldc);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}

/* cublasSsyr2k() */
static IDL_VPTR IDL_cublasSsyr2k(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  char uplo = (char) argv[0]->value.str.s[0];
  char trans = (char) argv[1]->value.str.s[0];
  long n = (long) argv[2]->value.l;
  long k = (long) argv[3]->value.l;
  float alpha= (float) argv[4]->value.f;
  float* A= (float*) argv[5]->value.ptrint;
  long lda = (long) argv[6]->value.l;
  float* B= (float*) argv[7]->value.ptrint;
  long ldb = (long) argv[8]->value.l;
  float beta= (float) argv[9]->value.f;
  float* C= (float*) argv[10]->value.ptrint;
  long ldc = (long) argv[11]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);
  IDL_ENSURE_SIMPLE(argv[6]);
  IDL_ENSURE_SIMPLE(argv[7]);
  IDL_ENSURE_SIMPLE(argv[8]);
  IDL_ENSURE_SIMPLE(argv[9]);
  IDL_ENSURE_SIMPLE(argv[10]);
  IDL_ENSURE_SIMPLE(argv[11]);

  cublasSsyr2k(uplo, trans,  n, k, alpha, A, lda, 
              B, ldb, beta, C, ldc);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}

/* cublasStrmm() */
static IDL_VPTR IDL_cublasStrmm(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  char side = (char) argv[0]->value.str.s[0];
  char uplo = (char) argv[1]->value.str.s[0];
  char transa = (char) argv[2]->value.str.s[0];
  char diag = (char) argv[3]->value.str.s[0];
  long m = (long) argv[4]->value.l;
  long n = (long) argv[5]->value.l;
  float alpha= (float) argv[6]->value.f;
  float* A= (float*) argv[7]->value.ptrint;
  long lda = (long) argv[8]->value.l;
  float* B= (float*) argv[9]->value.ptrint;
  long ldb = (long) argv[10]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);
  IDL_ENSURE_SIMPLE(argv[6]);
  IDL_ENSURE_SIMPLE(argv[7]);
  IDL_ENSURE_SIMPLE(argv[8]);
  IDL_ENSURE_SIMPLE(argv[9]);
  IDL_ENSURE_SIMPLE(argv[10]);

  cublasStrmm(side, uplo, transa, diag, m, n,  alpha, A, lda, 
              B, ldb);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}

/* cublasStrsm() */
static IDL_VPTR IDL_cublasStrsm(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  char side = (char) argv[0]->value.str.s[0];
  char uplo = (char) argv[1]->value.str.s[0];
  char transa = (char) argv[2]->value.str.s[0];
  char diag = (char) argv[3]->value.str.s[0];
  long m = (long) argv[4]->value.l;
  long n = (long) argv[5]->value.l;
  float alpha= (float) argv[6]->value.f;
  float* A= (float*) argv[7]->value.ptrint;
  long lda = (long) argv[8]->value.l;
  float* B= (float*) argv[9]->value.ptrint;
  long ldb = (long) argv[10]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);
  IDL_ENSURE_SIMPLE(argv[6]);
  IDL_ENSURE_SIMPLE(argv[7]);
  IDL_ENSURE_SIMPLE(argv[8]);
  IDL_ENSURE_SIMPLE(argv[9]);
  IDL_ENSURE_SIMPLE(argv[10]);

  cublasStrsm(side, uplo, transa, diag, m, n,  alpha, A, lda, 
              B, ldb);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}



/*  ********************************************************** */
/*  ****    BLAS LEVEL 3 Routines  (Complex)              **** */
/*  ********************************************************** */

/* cublasCgemm() */
static IDL_VPTR IDL_cublasCgemm(int argc, IDL_VPTR *argv)
{
  cublasStatus stat;
  char transa = (char) argv[0]->value.str.s[0];
  char transb = (char) argv[1]->value.str.s[0];
  long m = (long) argv[2]->value.l;
  long n = (long) argv[3]->value.l;
  long k = (long) argv[4]->value.l;
  IDL_COMPLEX alpha_idl = argv[5]->value.cmp;
  cuComplex* A= (cuComplex*) argv[6]->value.ptrint;
  long lda = (long) argv[7]->value.l;
  cuComplex* B= (cuComplex*) argv[8]->value.ptrint;
  long ldb = (long) argv[9]->value.l;
  IDL_COMPLEX beta_idl = argv[10]->value.cmp;
  cuComplex* C= (cuComplex*) argv[11]->value.ptrint;
  long ldc = (long) argv[12]->value.l;

  cuComplex alpha;
  cuComplex beta;
  
  alpha.x = alpha_idl.r;
  alpha.y = alpha_idl.i;

  beta.x = beta_idl.r;
  beta.y = beta_idl.i;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);
  IDL_ENSURE_SIMPLE(argv[5]);
  IDL_ENSURE_SIMPLE(argv[6]);
  IDL_ENSURE_SIMPLE(argv[7]);
  IDL_ENSURE_SIMPLE(argv[8]);
  IDL_ENSURE_SIMPLE(argv[9]);
  IDL_ENSURE_SIMPLE(argv[10]);
  IDL_ENSURE_SIMPLE(argv[11]);
  IDL_ENSURE_SIMPLE(argv[12]);

  cublasCgemm(transa, transb, m, n, k, alpha, A, lda, 
             B, ldb, beta, C, ldc);

  stat =   cublasGetError();
  return get_IDL_long( stat);

}


/*  ********************************************************** */
/*  ****                                                  **** */
/*  ****    CUDA FFT Routines                             **** */
/*  ****                                                  **** */
/*  ********************************************************** */
/* cufftPlan1d() */
static IDL_VPTR IDL_cufftPlan1d(int argc, IDL_VPTR *argv)
{
  cufftResult stat;
  long plan    = (long) argv[0]->value.l;
  long nx      = (long) argv[1]->value.l;
  long fftType = (long) argv[2]->value.l;
  long batch   = (long) argv[3]->value.l;
  
  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);

  stat = cufftPlan1d( (cufftHandle*) &(argv[0]->value.ptrint), 
                      nx, fftType, batch);
  return get_IDL_long( stat);
}

/* cufftPlan2d() */
static IDL_VPTR IDL_cufftPlan2d(int argc, IDL_VPTR *argv)
{
  cufftResult stat;
  //long plan    = (long) argv[0]->value.l;
  long nx      = (long) argv[1]->value.l;
  long ny      = (long) argv[2]->value.l;
  long fftType = (long) argv[3]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);

  stat = cufftPlan2d( (cufftHandle*) &(argv[0]->value.ptrint),
                      nx, ny, (int)fftType);
  return get_IDL_long( stat);
}

/* cufftPlan3d() */
static IDL_VPTR IDL_cufftPlan3d(int argc, IDL_VPTR *argv)
{
  cufftResult stat;
  long plan = (long) argv[0]->value.l;
  long nx   = (long) argv[1]->value.l;
  long ny   = (long) argv[2]->value.l;
  long nz   = (long) argv[3]->value.l;
  long fftType = (long) argv[4]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);
  IDL_ENSURE_SIMPLE(argv[3]);
  IDL_ENSURE_SIMPLE(argv[4]);

  stat = cufftPlan3d( (cufftHandle*) &(argv[0]->value.ptrint),
                      nx, ny, nz, fftType);
  return get_IDL_long( stat);
}

/* cufftDestroy() */
static IDL_VPTR IDL_cufftDestroy(int argc, IDL_VPTR *argv)
{
  cufftResult stat;
  long plan = (long) argv[0]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);

  stat = cufftDestroy( (cufftHandle) plan);
  return get_IDL_long( stat);
}

/* cufftExecC2C() */
static IDL_VPTR IDL_cufftExecC2C(int argc, IDL_VPTR *argv)
{
  cufftResult stat;
  long    plan = (long) argv[0]->value.ul;
  cufftComplex* idata = (cufftComplex*) argv[1]->value.ptrint;
  cufftComplex* odata = (cufftComplex*) argv[2]->value.ptrint;
  long  direction = (long) argv[3]->value.l;

  IDL_ENSURE_SIMPLE(argv[0]);  
  IDL_ENSURE_SIMPLE(argv[1]);  
  IDL_ENSURE_SIMPLE(argv[2]);  
  IDL_ENSURE_SIMPLE(argv[3]);  

  stat = cufftExecC2C( plan, idata, odata, direction);
  return get_IDL_long( stat);
}

/* cufftExecR2C() */
static IDL_VPTR IDL_cufftExecR2C(int argc, IDL_VPTR *argv)
{
  cufftResult stat;
  long    plan = (long) argv[0]->value.ul;
  float* idata = (float*) argv[1]->value.ptrint;
  cufftComplex* odata = (cufftComplex*) argv[2]->value.ptrint;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);

  stat = cufftExecR2C( plan, idata, odata);
  return get_IDL_long( stat);
}

/* cufftExecC2R() */
static IDL_VPTR IDL_cufftExecC2R(int argc, IDL_VPTR *argv)
{
  cufftResult stat;
  long    plan = (long) argv[0]->value.ul;
  cufftComplex* idata = (cufftComplex*) argv[1]->value.ptrint;
  float* odata = (float*) argv[2]->value.ptrint;

  IDL_ENSURE_SIMPLE(argv[0]);
  IDL_ENSURE_SIMPLE(argv[1]);
  IDL_ENSURE_SIMPLE(argv[2]);

  stat = cufftExecC2R( plan, idata, odata);
  return get_IDL_long( stat);
}


 /****************************************************************/
 /*                                                              */
 /*                       CUDA RUNTIME API                       */
 /*                                                              */
 /****************************************************************/


/*  ********************************************************** */
/*  ****    CUDA Runtime Device Management                **** */
/*  ********************************************************** */

/* cudaGetDeviceCount() */
static IDL_VPTR IDL_cudaGetDeviceCount(int argc, IDL_VPTR *argv)
{
  cudaError_t err;
  int count;

  err = cudaGetDeviceCount( &count);

  IDL_StoreScalarZero(argv[0], IDL_TYP_LONG);
  argv[0]->value.l = count;

  return get_IDL_long( (long) err);
}

/* cudaGetDeviceProperties */
static IDL_VPTR IDL_cudaGetDeviceProperties(int argc, IDL_VPTR *argv)
{
  cudaError_t err;
  struct cudaDeviceProp prop;
  static IDL_MEMINT one  = 1;
  static IDL_MEMINT dim3d[] = {1,3};
  static IDL_MEMINT dim40[] = {1,40};

  static IDL_STRUCT_TAG_DEF s_tags[] = {
    {"NAME",               0, (void *) IDL_TYP_STRING},
    {"TOTALGLOBALMEM",     0, (void *) IDL_TYP_PTRINT},
    {"SHAREDMEMPERBLOCK",  0, (void *) IDL_TYP_PTRINT},
    {"REGSPERBLOCK",       0, (void *) IDL_TYP_LONG},
    {"WARPSIZE",           0, (void *) IDL_TYP_LONG},
    {"MEMPITCH",           0, (void *) IDL_TYP_PTRINT}, 
    {"MAXTHREADSPERBLOCK", 0, (void *) IDL_TYP_LONG}, 
    {"MAXTHREADSDIM",  dim3d, (void *) IDL_TYP_LONG}, 
    {"MAXGRIDSIZE",    dim3d, (void *) IDL_TYP_LONG},
    {"CLOCKRATE",          0, (void *) IDL_TYP_LONG}, 
    {"TOTALCONSTMEM",      0, (void *) IDL_TYP_PTRINT},
    {"MAJOR",              0, (void *) IDL_TYP_LONG}, 
    {"MINOR",              0, (void *) IDL_TYP_LONG},
    {"TEXTUREALIGNMENT",   0, (void *) IDL_TYP_PTRINT}, 
    {"DEVICEOVERLAP",      0, (void *) IDL_TYP_LONG}, 
    {"MULTIPROCESSORCOUNT",0, (void *) IDL_TYP_LONG}, 
    {"__CUDARESERVED", dim40, (void *) IDL_TYP_LONG}, 
    { 0 }
  };

  typedef struct data_struct {
    IDL_STRING name;
    IDL_PTRINT  totalGlobalMem;
    IDL_PTRINT  sharedMemPerBlock;
    IDL_LONG    regsPerBlock;
    IDL_LONG    warpSize;
    IDL_PTRINT  memPitch;
    IDL_LONG    maxThreadsPerBlock;
    IDL_LONG    maxThreadsDim[3];
    IDL_LONG    maxGridSize[3];
    IDL_LONG    clockRate;
    IDL_PTRINT  totalConstMem;
    IDL_LONG    major;
    IDL_LONG    minor;
    IDL_PTRINT  textureAlignment;
    IDL_LONG    deviceOverlap;
    IDL_LONG    multiProcessorCount;
    IDL_LONG    __cudaReserved[40];
  } PROP_STRUCT;
  
  static PROP_STRUCT s_data;
  void   *s;
  IDL_VPTR v;
  int i;
  int dev;
  // clear return variable
  IDL_StoreScalarZero(argv[0], IDL_TYP_LONG);

  dev = (int) argv[1]->value.l;

  err = cudaGetDeviceProperties(&prop, dev);

  /* copy the entire struct, except for the name  */
  bcopy(&prop.totalGlobalMem, &s_data.totalGlobalMem, 
             sizeof(prop)-sizeof(prop.name));
   

  IDL_StrDelete(&s_data.name, 1);
  IDL_StrStore(&s_data.name, prop.name);

  /* create the structure definition */
  s = IDL_MakeStruct("cudaDeviceProp", s_tags);
  v = IDL_ImportArray(1, &one, IDL_TYP_STRUCT, 
			    (UCHAR *) &s_data, 0, s);

  IDL_VarCopy(v, argv[0]);
  return get_IDL_long( err);

}

/* cudaChoseDevice() */
static IDL_VPTR IDL_cudaChoseDevice(int argc, IDL_VPTR *argv)
{
  cudaError_t err;

  // Chose device is currently not implemented
  // stay tuned..
  err = (cudaError_t) -1;

  return get_IDL_long( (long) err);
}



/* cudaSetDevice() */
static IDL_VPTR IDL_cudaSetDevice(int argc, IDL_VPTR *argv)
{
  cudaError_t err;
  int dev =  argv[0]->value.l;

  err = cudaSetDevice( dev);

  return get_IDL_long( (long) err);
}


/* cudaGetDevice() */
static IDL_VPTR IDL_cudaGetDevice(int argc, IDL_VPTR *argv)
{
  cudaError_t err;
  int dev;

  err = cudaGetDevice( &dev);

  IDL_StoreScalarZero(argv[0], IDL_TYP_LONG);
  argv[0]->value.l = dev;

  return get_IDL_long( (long) err);
}



/*  ********************************************************** */
/*  ****    CUDA Runtime Thread Management                **** */
/*  ********************************************************** */

/* cudaThreadExit() */
static IDL_VPTR IDL_cudaThreadSynchronize(int argc, IDL_VPTR *argv)
{
  cudaError_t err;

  err = cudaThreadSynchronize();

  return get_IDL_long( (long) err);
}

 
/* cudaThreadExit() */
static IDL_VPTR IDL_cudaThreadExit(int argc, IDL_VPTR *argv)
{
  cudaError_t err;

  err = cudaThreadExit();

  return get_IDL_long( (long) err);
}

/*  ********************************************************** */
/*  ****    CUDA Runtime Memory Management                **** */
/*  ********************************************************** */

/* cudaMalloc() */
static IDL_VPTR IDL_cudaMalloc(int argc, IDL_VPTR *argv)
{
  cudaError_t err;  
  size_t count =  argv[1]->value.ul;
  
  IDL_StoreScalarZero(argv[0], IDL_TYP_ULONG);

  err = cudaMalloc((void**) &(argv[0]->value.ptrint), count);

  return get_IDL_long( (long) err);
}

/* cudaMallocPitch() */
static IDL_VPTR IDL_cudaMallocPitch(int argc, IDL_VPTR *argv)
{
  cudaError_t err;  
  size_t widthInBytes =  argv[2]->value.ul;
  size_t height       =  argv[3]->value.ul;

  IDL_StoreScalarZero(argv[0], IDL_TYP_ULONG);
  IDL_StoreScalarZero(argv[1], IDL_TYP_ULONG);
  
  
  err = cudaMallocPitch((void**) &argv[0]->value.ptrint, 
		        (size_t *)  &argv[1]->value.ptrint, 
                         widthInBytes, height);

  return get_IDL_long( (long) err);
}

/* cudaFree() */
static IDL_VPTR IDL_cudaFree(int argc, IDL_VPTR *argv)
{
  cudaError_t err;  
  
  err = cudaFree((void*) argv[0]->value.ptrint);

  return get_IDL_long( (long) err);
}


/* cudaMallocArray() */
static IDL_VPTR IDL_cudaMallocArray(int argc, IDL_VPTR *argv)
{
  cudaError_t err;
  struct cudaChannelFormatDesc desc;
  size_t width  =  argv[2]->value.ul;
  size_t height =  argv[3]->value.ul;

  desc.x = ((IDL_LONG *)argv[1]->value.arr->data)[0];
  desc.y = ((IDL_LONG *)argv[1]->value.arr->data)[1];
  desc.z = ((IDL_LONG *)argv[1]->value.arr->data)[2];
  desc.w = ((IDL_LONG *)argv[1]->value.arr->data)[3];
  desc.f = (enum cudaChannelFormatKind) 
                ((IDL_LONG *)argv[1]->value.arr->data)[4];

  err = cudaMallocArray((struct cudaArray**) &argv[0]->value.l, 
			&desc, width, height);

  return get_IDL_long( (long) err);
}

/* cudaFreeArray() */
static IDL_VPTR IDL_cudaFreeArray(int argc, IDL_VPTR *argv)
{
  cudaError_t err;
  struct cudaArray*  array =  (struct cudaArray*) argv[0]->value.ptrint;

  err = cudaFreeArray( array );

  return get_IDL_long( (long) err);
}

/* cudaMallocHost() */
static IDL_VPTR IDL_cudaMallocHost(int argc, IDL_VPTR *argv)
{
  cudaError_t err;  
  size_t count =  argv[1]->value.ul;

  IDL_StoreScalarZero(argv[0], IDL_TYP_ULONG);

  err = cudaMalloc((void**) &argv[0]->value.l, count);

  return get_IDL_long( (long) err);
}

/* cudaFreeHost() */
static IDL_VPTR IDL_cudaFreeHost(int argc, IDL_VPTR *argv)
{
  cudaError_t err;  
  
  err = cudaFree((void*) argv[0]->value.ptrint);

  return get_IDL_long( (long) err);
}

/* cudaMemset() */
static IDL_VPTR IDL_cudaMemset(int argc, IDL_VPTR *argv)
{
  cudaError_t err;  
  void * devPtr = (void*) argv[0]->value.ptrint;
  int    value  = argv[1]->value.l;
  size_t count  = argv[2]->value.ul;

  err = cudaMemset( devPtr, value, count);

  return get_IDL_long( (long) err);
}

/* cudaMemset2D() */
static IDL_VPTR IDL_cudaMemset2D(int argc, IDL_VPTR *argv)
{
  cudaError_t err;  
  void * dstPtr = (void*) argv[0]->value.ptrint;
  size_t pitch  = argv[1]->value.ul;
  int    value  = argv[2]->value.l;
  size_t width  = argv[3]->value.ul;
  size_t height = argv[4]->value.ul;

  err = cudaMemset2D( dstPtr, pitch, value, width, height);

  return get_IDL_long( (long) err);
}

/* cudaMemcpy() */
static IDL_VPTR IDL_cudaMemcpy(int argc, IDL_VPTR *argv)
{
  cudaError_t err;  
  void * dst = (void*) argv[0]->value.ptrint;
  void * src = (void*) argv[1]->value.ptrint;
  size_t count = argv[2]->value.ul;
  enum cudaMemcpyKind kind = (enum cudaMemcpyKind) argv[3]->value.ul;

  err = cudaMemcpy( dst, src, count, kind);

  return get_IDL_long( (long) err);
}


/* cudaMemcpy2D() */
static IDL_VPTR IDL_cudaMemcpy2D(int argc, IDL_VPTR *argv)
{
  cudaError_t err;
  void * dst    = (void*) argv[0]->value.ptrint;
  size_t dpitch = (size_t) argv[1]->value.l;
  void * src    = (void*) argv[2]->value.ptrint;
  size_t spitch = (size_t) argv[3]->value.l;
  size_t width  = argv[4]->value.ul;
  size_t height = argv[5]->value.ul;
  enum cudaMemcpyKind kind = (enum cudaMemcpyKind) argv[6]->value.ul;

  err = cudaMemcpy2D( dst, dpitch, src, spitch, width, height, kind);

  return get_IDL_long( (long) err);
}


/* cudaGLRegisterBufferObject */
static IDL_VPTR IDL_cudaGLRegisterBufferObject(int argc, IDL_VPTR *argv)
{
  cudaError_t err;
  GLuint bufferObj = (GLuint) argv[0]->value.ul;

  err = cudaGLRegisterBufferObject( bufferObj);

  return get_IDL_long( (long) err);
}

/* cudaGLMapBufferObject */
static IDL_VPTR IDL_cudaGLMapBufferObject(int argc, IDL_VPTR *argv)
{
  cudaError_t err;
  GLuint bufferObj = (GLuint) argv[1]->value.ul;

  err = cudaGLMapBufferObject( (void**) &argv[0]->value.l, bufferObj);

  return get_IDL_long( (long) err);
}

/* cudaGLUnmapBufferObject */
static IDL_VPTR IDL_cudaGLUnmapBufferObject(int argc, IDL_VPTR *argv)
{
  cudaError_t err;
  GLuint bufferObj = (GLuint) argv[0]->value.ul;

  err = cudaGLUnmapBufferObject( bufferObj);

  return get_IDL_long( (long) err);
}

/* cudaGLUregisterBufferObject */
static IDL_VPTR IDL_cudaGLUnregisterBufferObject(int argc, IDL_VPTR *argv)
{
  cudaError_t err;
  GLuint bufferObj = (GLuint) argv[0]->value.ul;

  err = cudaGLUnregisterBufferObject( bufferObj);

  return get_IDL_long( (long) err);
}






/****************************************************************/
/*                                                              */
/*              GPU versions of IDL intrinsics                  */
/*                                                              */
/****************************************************************/

static IDL_VPTR IDL_gpuInterpolate1DF(int argc, IDL_VPTR *argv)
{
  size_t np =  argv[0]->value.ul;
  float * p = (float*) argv[1]->value.ptrint;
  size_t nx =  argv[2]->value.ul;
  float * x = (float*) argv[3]->value.ptrint;
  float * res = (float*) argv[4]->value.ptrint;

  long err = gpuInterpolate1DF(np, p, nx, x, res);

  return get_IDL_long( err);
}


static IDL_VPTR IDL_gpuInterpolate2DF(int argc, IDL_VPTR *argv)
{
  long npx =  argv[0]->value.l;
  long npy =  argv[1]->value.l;
  float * p = (float*) argv[2]->value.ptrint;
  long nx =  argv[3]->value.l;
  float * x = (float*) argv[4]->value.ptrint;
  float * y = (float*) argv[5]->value.ptrint;
  float * res = (float*) argv[6]->value.ptrint;

  long err = gpuInterpolate2DF(npx, npy, p, nx, x, y, res);

  return get_IDL_long( err);
}

static IDL_VPTR IDL_gpuFindgenF(int argc, IDL_VPTR *argv)
{
  size_t nx =  argv[0]->value.ul;
  //float *res = (float*) argv[1]->value.l;
  float *res = (float*) argv[1]->value.ptrint;

  long err = gpuIndexF(nx, res);

  return get_IDL_long( err);
}

static IDL_VPTR IDL_gpuDindgenD(int argc, IDL_VPTR *argv)
{
  size_t nx =  argv[0]->value.ul;
  //double *res = (double*) argv[1]->value.l;
  double *res = (double*) argv[1]->value.ptrint;

  long err = gpuIndexD(nx, res);

  return get_IDL_long( err);
}

static IDL_VPTR IDL_gpuCindgenC(int argc, IDL_VPTR *argv)
{
  size_t nx =  argv[0]->value.ul;
  float2 *res = (float2*) argv[1]->value.ptrint;

  long err = gpuIndexC(nx, res);

  return get_IDL_long( err);
}

IDL_VPTR IDL_gpuDCindgenZ(int argc, IDL_VPTR *argv)
{
  size_t nx =  argv[0]->value.ul;
  double2 *res = (double2*) argv[1]->value.ptrint;

  long err = gpuIndexZ(nx, res);

  return get_IDL_long( err);
}



static IDL_VPTR IDL_gpuTotalF(int argc, IDL_VPTR *argv)
{
  size_t nx =  argv[0]->value.ul;
  float * p = (float*) argv[1]->value.ptrint;
  float res = (float) argv[2]->value.f;

  long err = gpuTotalF(nx, p, &(argv[2]->value.f));

  return get_IDL_long( err);

}

static IDL_VPTR IDL_gpuMinF(int argc, IDL_VPTR *argv)
{
  size_t nx =  argv[0]->value.ul;
  float * p = (float*) argv[1]->value.ptrint;
  float res = (float) argv[2]->value.f;
  int pos   = (int) argv[3]->value.l;

  long err = gpuMinMaxF(nx, p, (float*) &(argv[2]->value.f), (int*) &(argv[3]->value.l), -1);

  return get_IDL_long( err);
}

static IDL_VPTR IDL_gpuMaxF(int argc, IDL_VPTR *argv)
{
  size_t nx =  argv[0]->value.ul;
  float * p = (float*) argv[1]->value.ptrint;
  float res = (float) argv[2]->value.f;
  int pos   = (int) argv[3]->value.l;

  long err = gpuMinMaxF(nx, p, (float *) &(argv[2]->value.f), (int *)&(argv[3]->value.l), 1);

  return get_IDL_long( err);
}

static IDL_VPTR IDL_gpuCongrid1DF(int argc, IDL_VPTR *argv)
{
  size_t npx =  argv[0]->value.ul;
  float *  p = (float*) argv[1]->value.ptrint;
  size_t  nx =  argv[2]->value.ul;
  float * res = (float*) argv[3]->value.ptrint;
  int  interp = (int) argv[4]->value.l;

  long err = gpuCongrid1DF(npx, p, nx, res, interp);

  return get_IDL_long( err);
}


static IDL_VPTR IDL_gpuCongrid2DF(int argc, IDL_VPTR *argv)
{
  long npx =  argv[0]->value.ul;
  long npy =  argv[1]->value.ul;
  float *  p = (float*) argv[2]->value.ptrint;
  long  nx =  argv[3]->value.ul;
  long  ny =  argv[4]->value.ul;
  float * res = (float*) argv[5]->value.ptrint;
  int  interp = (int) argv[6]->value.l;
  
  int err = gpuCongrid2DF(npx, npy, p, nx, ny, res, interp);

  return get_IDL_long( err);
}

static IDL_VPTR IDL_gpuSubscriptF(int argc, IDL_VPTR *argv)
{
  size_t  nx =  argv[0]->value.ul;
  float *  x = (float*) argv[1]->value.ptrint;
  float *  y = (float*) argv[2]->value.ptrint;
  float * res = (float*) argv[3]->value.ptrint;

  long  err = gpuSubscriptF(nx, x, y, res);

  return get_IDL_long( err);
}

static IDL_VPTR IDL_gpuSubscriptLHSF(int argc, IDL_VPTR *argv)
{
  size_t  nx =  argv[0]->value.ul;
  float *  x = (float*) argv[1]->value.ptrint;
  float *  y = (float*) argv[2]->value.ptrint;
  float * res = (float*) argv[3]->value.ptrint;

  long err = gpuSubscriptLHSF(nx, x, y, res);

  return get_IDL_long( err);
}



static IDL_VPTR IDL_gpuPrefixSumF(int argc, IDL_VPTR *argv)
{
  size_t nx  =  argv[0]->value.ul;
  float *x   = (float*) argv[1]->value.ptrint;
  float *res = (float*) argv[2]->value.ptrint;

  long err = gpuPrefixSumF(nx, x, res);

  return get_IDL_long( err);
}


static IDL_VPTR IDL_gpuWhereF(int argc, IDL_VPTR *argv)
{
  int nx  =  argv[0]->value.ul;
  float *x   = (float*) argv[1]->value.ptrint;
  float *res = (float*) argv[2]->value.ptrint;
  float lastX = (float) argv[3]->value.f;

  long err = gpuWhereF(nx, x, res, lastX);

  return get_IDL_long( err);
}

/****************************************************************/
/*                                                              */
/*********** routines from gpuMT                    *************/
/*                                                              */
/****************************************************************/

static IDL_VPTR IDL_gpuMTF(int argc, IDL_VPTR *argv)
{
  cudaError_t err;
  size_t nx =  argv[0]->value.ul;
  float *res = (float*) argv[1]->value.ptrint;

  int rr = gpuMTF(nx, res);

  err = 0;
  return get_IDL_long( (long) err);
}

static IDL_VPTR IDL_gpuBoxMullerF(int argc, IDL_VPTR *argv)
{
  cudaError_t err;
  size_t nx =  argv[0]->value.ul;
  float *res = (float*) argv[1]->value.ptrint;

  int rr = gpuBoxMullerF(nx, res);

  err = 0;
  return get_IDL_long( (long) err);
}


static IDL_VPTR IDL_gpuSeedMTF(int argc, IDL_VPTR *argv)
{
  cudaError_t err;
  size_t seed =  argv[0]->value.ul;

  int rr = gpuSeedMTF(seed);

  err = 0;
  return get_IDL_long( (long) err);
}

/****************************************************************/
/*                                                              */
/*********** routines from gpuPhysicsOP             *************/
/*                                                              */

static IDL_VPTR IDL_gpuPoisson(int argc, IDL_VPTR *argv)
{
  cudaError_t err;
  size_t n =  argv[0]->value.ul;
  float * b = (float*) argv[1]->value.ptrint;
  float * c = (float*) argv[2]->value.ptrint;
  float * res = (float*) argv[3]->value.ptrint;
  
  int rr = gpuPoisson(n, b, c, res);

  err = 0;
  return get_IDL_long( (long) err);
}


static IDL_VPTR IDL_gpuBrmBremcrossF(int argc, IDL_VPTR *argv)
{
  cudaError_t err;
  size_t nx  =  argv[0]->value.ul;
  float* eel = (float*) argv[1]->value.ptrint;
  float* eph = (float*) argv[2]->value.ptrint;
  float   z  = (float) argv[3]->value.f;
  float* cross = (float*) argv[4]->value.ptrint;

  int rr = gpuBrmBremcrossF(nx, eel, eph, z, cross);

  err = 0;
  return get_IDL_long( (long) err);
}

static IDL_VPTR IDL_gpuBrmFinnerF(int argc, IDL_VPTR *argv)
{
  cudaError_t err;
  size_t nx  =  argv[0]->value.ul;
  float* eel = (float*) argv[1]->value.ptrint;
  float* eph = (float*) argv[2]->value.ptrint;
  float   z  = (float) argv[3]->value.f;
  float* finner = (float*) argv[4]->value.ptrint;

  int rr = gpuBrmFinnerF(nx, eel, eph, z, finner);

  err = 0;
  return get_IDL_long( (long) err);
}

static IDL_VPTR IDL_gpuGauLegF(int argc, IDL_VPTR *argv)
{
  cudaError_t err;
  size_t nx  =  argv[0]->value.ul;
  float* x1 = (float*) argv[1]->value.ptrint;
  float* x2 = (float*) argv[2]->value.ptrint;
  size_t    n =  argv[3]->value.ul;
  float* x  = (float*) argv[4]->value.ptrint;
  float* w  = (float*) argv[5]->value.ptrint;

  int rr = gpuGauLegF(nx, x1, x2, n, x, w);

  err = 0;
  return get_IDL_long( (long) err);
}



/****************************************************************/


 /****************************************************************/
 /*                                                              */
 /*                       IDL registration                       */
 /*                                                              */
 /****************************************************************/


int IDL_Load(void)
{
  /*
   * These tables contain information on the functions and procedures
   * that make up the GPUMODULE DLM. The information contained in these
   * tables must be identical to that contained in testmodule.dlm.
   */
  static IDL_SYSFUN_DEF2 function_addr[] = {
    { IDL_cublasInit,     "CUBLASINIT",     0, 0, 0, 0},
    { IDL_cublasShutdown, "CUBLASSHUTDOWN", 0, 0, 0, 0},
    { IDL_cublasGetError, "CUBLASGETERROR", 0, 0, 0, 0},
    { IDL_cublasAlloc,    "CUBLASALLOC",    1, 3, 0, 0},
    { IDL_cublasFree,     "CUBLASFREE",     1, 1, 0, 0},
    { IDL_cublasSetVector,"CUBLASSETVECTOR",6, 6, 0, 0},
    { IDL_cublasGetVector,"CUBLASGETVECTOR",6, 6, 0, 0},
    { IDL_cublasSetMatrix,"CUBLASSETMATRIX",7, 7, 0, 0},
    { IDL_cublasGetMatrix,"CUBLASGETMATRIX",7, 7, 0, 0},
/*  BLAS 1 */
    { IDL_cublasIsamax,   "CUBLASISAMAX",   3, 3, 0, 0},
    { IDL_cublasSasum,    "CUBLASSASUM",    3, 3, 0, 0},
    { IDL_cublasSaxpy,    "CUBLASSAXPY",    6, 6, 0, 0},
    { IDL_cublasScopy,    "CUBLASSCOPY",    5, 5, 0, 0},
    { IDL_cublasSdot,     "CUBLASSDOT",     5, 5, 0, 0},
    { IDL_cublasSnrm2,    "CUBLASSNRM2",    3, 3, 0, 0},
    { IDL_cublasSrot,     "CUBLASSROT",     7, 7, 0, 0},
    { IDL_cublasSrotg,    "CUBLASSROTG",    4, 4, 0, 0},
    { IDL_cublasSrotm,    "CUBLASSROTM",    6, 6, 0, 0},
    { IDL_cublasSrotmg,   "CUBLASSROTMG",   5, 5, 0, 0},
    { IDL_cublasSscal,    "CUBLASSCAL",     4, 4, 0, 0},
    { IDL_cublasSswap,    "CUBLASSSWAP",    5, 5, 0, 0},
/*  BLAS 1 Complex */
    { IDL_cublasCaxpy,    "CUBLASCAXPY",    6, 6, 0, 0},
    { IDL_cublasCcopy,    "CUBLASCCOPY",    5, 5, 0, 0},
    { IDL_cublasCdotu,    "CUBLASCDOTU",    5, 5, 0, 0},
    { IDL_cublasCscal,    "CUBLASCSCAL",    4, 4, 0, 0},
    { IDL_cublasCsscal,   "CUBLASCSSCAL",   4, 4, 0, 0},
    { IDL_cublasCswap,    "CUBLASCSWAP",    5, 5, 0, 0},
    { IDL_cublasScasum,   "CUBLASSCASUM",   3, 3, 0, 0},
/*  BLAS 2  */
    { IDL_cublasSgbmv,   "CUBLASSGBMV",   13, 13, 0, 0},
    { IDL_cublasSgemv,   "CUBLASSGEMV",   11, 11, 0, 0},
    { IDL_cublasSger,    "CUBLASSGER",    9, 9, 0, 0},
    { IDL_cublasSsbmv,   "CUBLASSSBMV",   11, 11, 0, 0},
    { IDL_cublasSspmv,   "CUBLASSSPMV",   9, 9, 0, 0},
    { IDL_cublasSspr,    "CUBLASSSPR",    6, 6, 0, 0},
    { IDL_cublasSspr2,   "CUBLASSSPR2",   8, 8, 0, 0},
    { IDL_cublasSsymv,   "CUBLASSSYMV",   10, 10, 0, 0},
    { IDL_cublasSsyr,    "CUBLASSSYR",    7, 7, 0, 0},
    { IDL_cublasSsyr2,   "CUBLASSSYR2",   9, 9, 0, 0},
    { IDL_cublasStbmv,   "CUBLASSTBMV",   9, 9, 0, 0},
    { IDL_cublasStbsv,   "CUBLASSTBSV",   9, 9, 0, 0},
    { IDL_cublasStpmv,   "CUBLASSTPMV",   7, 7, 0, 0},
    { IDL_cublasStpsv,   "CUBLASSTPSV",   7, 7, 0, 0},
    { IDL_cublasStrmv,   "CUBLASSTRMV",   8, 8, 0, 0},
    { IDL_cublasStrsv,   "CUBLASSTRSV",   8, 8, 0, 0},
/*  BLAS 3  */
    { IDL_cublasSgemm,   "CUBLASSGEMM",   13, 13, 0, 0},
    { IDL_cublasSsymm,   "CUBLASSSYMM",   12, 12, 0, 0},
    { IDL_cublasSsyrk,   "CUBLASSSYRK",   10, 10, 0, 0},
    { IDL_cublasSsyr2k,  "CUBLASSSYR2K",  12, 12, 0, 0},
    { IDL_cublasStrmm,   "CUBLASSTRMM",   11, 11, 0, 0},
    { IDL_cublasStrsm,   "CUBLASSTRSM",   11, 11, 0, 0},
/*  BLAS 3 Complex */
    { IDL_cublasCgemm,   "CUBLASGEMM",    13, 13, 0, 0},


/*  FFT */
    { IDL_cufftPlan1d,   "CUFFTPLAN1D",   4, 4, 0, 0},
    { IDL_cufftPlan2d,   "CUFFTPLAN2D",   4, 4, 0, 0},
    { IDL_cufftPlan3d,   "CUFFTPLAN3D",   5, 5, 0, 0},
    { IDL_cufftDestroy,  "CUFFTDESTROY",  1, 1, 0, 0},
    { IDL_cufftExecC2C,  "CUFFTEXECC2C",  4, 4, 0, 0},
    { IDL_cufftExecR2C,  "CUFFTEXECR2C",  3, 3, 0, 0},
    { IDL_cufftExecC2R,  "CUFFTEXECC2R",  3, 3, 0, 0},


/*  CUDA Runtime Device Management */
    { IDL_cudaGetDeviceCount,       "CUDAGETDEVICECOUNT",      1, 1, 0, 0},
    { IDL_cudaGetDeviceProperties,  "CUDAGETDEVICEPROPERTIES", 2, 2, 0, 0},
    { IDL_cudaChoseDevice,          "CUDACHOSEDEVICE",         0, 0, 0, 0},
    { IDL_cudaSetDevice,            "CUDASETDEVICE",           1, 1, 0, 0},
    { IDL_cudaGetDevice,            "CUDAGETDEVICE",           1, 1, 0, 0},

/*  CUDA Runtime Thread Management */
    { IDL_cudaThreadSynchronize,    "CUDATHREADSYNCHRONIZE",   0, 0, 0, 0},
    { IDL_cudaThreadExit,           "CUDATHREADEXIT",          0, 0, 0, 0},

/*  CUDA Runtime Memory Management */
    { IDL_cudaMalloc,               "CUDAMALLOC",              2, 2, 0, 0},
    { IDL_cudaMallocPitch,          "CUDAMALLOCPITCH",         4, 4, 0, 0},
    { IDL_cudaFree,                 "CUDAFREE",                1, 1, 0, 0},
    { IDL_cudaMallocArray,          "CUDAMALLOCARRAY",         4, 4, 0, 0},
    { IDL_cudaFreeArray,            "CUDAFREEARRAY",           1, 1, 0, 0},
    { IDL_cudaMallocHost,           "CUDAMALLOCHOST",          2, 2, 0, 0},
    { IDL_cudaFreeHost,             "CUDAFREEHOST",            1, 1, 0, 0},
    { IDL_cudaMemset,               "CUDAMEMSET",              3, 3, 0, 0},
    { IDL_cudaMemset2D,             "CUDAMEMSET2D",            5, 5, 0, 0},
    { IDL_cudaMemcpy,               "CUDAMEMCPY",              4, 4, 0, 0},
    { IDL_cudaMemcpy2D,             "CUDAMEMCPY2D",            7, 7, 0, 0},

    /*  CUDA GL interoperability */
    { IDL_cudaGLRegisterBufferObject,"CUDAGLREGISTERBUFFEROBJECT",1,1,0,0},
    { IDL_cudaGLMapBufferObject,     "CUDAGLMAPBUFFEROBJECT",     2,2,0,0},
    { IDL_cudaGLUnmapBufferObject,   "CUDAGLUNMAPBUFFEROBJECT",   1,1,0,0},
    { IDL_cudaGLUnregisterBufferObject,
                                    "CUDAGLUNREGISTERBUFFEROBJECT",1,1,0,0},

    /*  gpuVectorOp */

   /* gpuUnaryOp operations with multi-type functionality */
    //{ IDL_gpuSqrtF,   "GPUSQRTF",   3,3,0,0},

    #define MAKE_INSTANCE(NAME,NAME2) { IDL_gpu##NAME, #NAME2, 3, 3, 0, 0},
    #define MAKE_INSTANCE_ALL(NAME,NAME2)      \
    MAKE_INSTANCE(NAME ## F,GPU ## NAME2 ## F) \
    MAKE_INSTANCE(NAME ## D,GPU ## NAME2 ## D) \
    MAKE_INSTANCE(NAME ## C,GPU ## NAME2 ## C) \
    MAKE_INSTANCE(NAME ## Z,GPU ## NAME2 ## Z)

    MAKE_INSTANCE_ALL(Sqrt, SQRT)
    MAKE_INSTANCE_ALL(Exp, EXP)
    MAKE_INSTANCE_ALL(Exp2, EXP2)
    MAKE_INSTANCE_ALL(Exp10, EXP10)
    MAKE_INSTANCE_ALL(Log, LOG)
    MAKE_INSTANCE_ALL(Log2, LOG2)
    MAKE_INSTANCE_ALL(Log10, LOG10)
    MAKE_INSTANCE_ALL(Log1p, LOG1P)
    MAKE_INSTANCE_ALL(Sin, SIN)
    MAKE_INSTANCE_ALL(Cos, COS)
    MAKE_INSTANCE_ALL(Tan, TAN)
    MAKE_INSTANCE_ALL(Asin, ASIN)
    MAKE_INSTANCE_ALL(Acos, ACOS)
    MAKE_INSTANCE_ALL(Atan, ATAN)

    MAKE_INSTANCE_ALL(Erf, ERF)
    MAKE_INSTANCE_ALL(Lgamma, LGAMMA)
    MAKE_INSTANCE_ALL(Tgamma, TGAMMA)
    MAKE_INSTANCE_ALL(Logb, LOGB)
    MAKE_INSTANCE_ALL(Trunc, TRUNC)
    MAKE_INSTANCE_ALL(Round, ROUND)
    MAKE_INSTANCE_ALL(Rint, RINT)
    MAKE_INSTANCE_ALL(Nearbyint, NEARBYINT)

    MAKE_INSTANCE_ALL(Ceil, CEIL)
    MAKE_INSTANCE_ALL(Floor, FLOOR)
    MAKE_INSTANCE_ALL(Lrint, LRINT)
    MAKE_INSTANCE_ALL(Lround, LROUND)
    MAKE_INSTANCE_ALL(Signbit, SIGNBIT)
    MAKE_INSTANCE_ALL(Isinf, ISINF)
    MAKE_INSTANCE_ALL(Isnan, ISNAN)
    MAKE_INSTANCE_ALL(Isfinite, ISFINITE)
    MAKE_INSTANCE_ALL(Fabs, FABS)

#undef MAKE_INSTANCE
#undef MAKE_INSTANCE_ALL

#define MAKE_INSTANCE_AT(NAME,NAME2) { IDL_gpu##NAME, #NAME2, 7, 7, 0, 0},
#define MAKE_INSTANCE_AT_ALL(NAME,NAME2)          \
    MAKE_INSTANCE_AT(NAME ## FAT,GPU ## NAME2 ## FAT) \
    MAKE_INSTANCE_AT(NAME ## DAT,GPU ## NAME2 ## DAT) \
    MAKE_INSTANCE_AT(NAME ## CAT,GPU ## NAME2 ## CAT) \
    MAKE_INSTANCE_AT(NAME ## ZAT,GPU ## NAME2 ## ZAT)

    MAKE_INSTANCE_AT_ALL(Sqrt, SQRT)
    MAKE_INSTANCE_AT_ALL(Exp, EXP)
    MAKE_INSTANCE_AT_ALL(Exp2, EXP2)
    MAKE_INSTANCE_AT_ALL(Exp10, EXP10)
    MAKE_INSTANCE_AT_ALL(Log, LOG)
    MAKE_INSTANCE_AT_ALL(Log2, LOG2)
    MAKE_INSTANCE_AT_ALL(Log10, LOG10)
    MAKE_INSTANCE_AT_ALL(Log1p, LOG1P)
    MAKE_INSTANCE_AT_ALL(Sin, SIN)
    MAKE_INSTANCE_AT_ALL(Cos, COS)
    MAKE_INSTANCE_AT_ALL(Tan, TAN)
    MAKE_INSTANCE_AT_ALL(Asin, ASIN)
    MAKE_INSTANCE_AT_ALL(Acos, ACOS)
    MAKE_INSTANCE_AT_ALL(Atan, ATAN)
    MAKE_INSTANCE_AT_ALL(Erf, ERF)
    MAKE_INSTANCE_AT_ALL(Lgamma, LGAMMA)
    MAKE_INSTANCE_AT_ALL(Tgamma, TGAMMA)
    MAKE_INSTANCE_AT_ALL(Logb, LOGB)
    MAKE_INSTANCE_AT_ALL(Trunc, TRUNC)
    MAKE_INSTANCE_AT_ALL(Round, ROUND)
    MAKE_INSTANCE_AT_ALL(Rint, RINT)
    MAKE_INSTANCE_AT_ALL(Nearbyint, NEARBYINT)
    MAKE_INSTANCE_AT_ALL(Ceil, CEIL)
    MAKE_INSTANCE_AT_ALL(Floor, FLOOR)
    MAKE_INSTANCE_AT_ALL(Lrint, LRINT)
    MAKE_INSTANCE_AT_ALL(Lround, LROUND)
    MAKE_INSTANCE_AT_ALL(Signbit, SIGNBIT)
    MAKE_INSTANCE_AT_ALL(Isinf, ISINF)
    MAKE_INSTANCE_AT_ALL(Isnan, ISNAN)
    MAKE_INSTANCE_AT_ALL(Isfinite, ISFINITE)
    MAKE_INSTANCE_AT_ALL(Fabs, FABS)
#undef MAKE_INSTANCE_AT
#undef MAKE_INSTANCE_AT_ALL

    /* gpuVectorOp binary operations with multi-type functionality */
#define MAKE_INSTANCE(NAME,NAME2) { IDL_gpu##NAME, #NAME2, 4, 4, 0, 0},
#define MAKE_INSTANCE_ALL(NAME,NAME2)      \
    MAKE_INSTANCE(NAME ## F,GPU ## NAME2 ## F) \
    MAKE_INSTANCE(NAME ## D,GPU ## NAME2 ## D) \
    MAKE_INSTANCE(NAME ## C,GPU ## NAME2 ## C) \
    MAKE_INSTANCE(NAME ## Z,GPU ## NAME2 ## Z)

    MAKE_INSTANCE_ALL(Add,  ADD)
    MAKE_INSTANCE_ALL(Sub,  SUB)
    MAKE_INSTANCE_ALL(Mult, MULT)
    MAKE_INSTANCE_ALL(Div,  DIV)

    MAKE_INSTANCE_ALL(Eq,   EQ)
    MAKE_INSTANCE_ALL(Neq,  NEQ)
    MAKE_INSTANCE_ALL(Lt,   LT)
    MAKE_INSTANCE_ALL(Gt,   GT)
    MAKE_INSTANCE_ALL(LtEq, LTEQ)
    MAKE_INSTANCE_ALL(GtEq, GTEQ)

#undef MAKE_INSTANCE
#undef MAKE_INSTANCE_ALL

#define MAKE_INSTANCE_AT(NAME,NAME2) { IDL_gpu##NAME, #NAME2, 7, 7, 0, 0},
#define MAKE_INSTANCE_AT_ALL(NAME,NAME2)          \
    MAKE_INSTANCE_AT(NAME ## FAT,GPU ## NAME2 ## FAT) \
    MAKE_INSTANCE_AT(NAME ## DAT,GPU ## NAME2 ## DAT) \
    MAKE_INSTANCE_AT(NAME ## CAT,GPU ## NAME2 ## CAT) \
    MAKE_INSTANCE_AT(NAME ## ZAT,GPU ## NAME2 ## ZAT) 

    MAKE_INSTANCE_AT_ALL(Add,ADD)
    MAKE_INSTANCE_AT_ALL(Sub,SUB)
    MAKE_INSTANCE_AT_ALL(Mult,MULT)
    MAKE_INSTANCE_AT_ALL(Div,DIV)
    MAKE_INSTANCE_AT_ALL(Eq, EQ)
    MAKE_INSTANCE_AT_ALL(Neq, NEQ)
    MAKE_INSTANCE_AT_ALL(Lt, LT)
    MAKE_INSTANCE_AT_ALL(Gt, GT)
    MAKE_INSTANCE_AT_ALL(LtEq, LTEQ)
    MAKE_INSTANCE_AT_ALL(GtEq, GTEQ)
#undef MAKE_INSTANCE_AT
#undef MAKE_INSTANCE_AT_ALL

#define MAKE_INSTANCE(NAME,NAME2) { IDL_gpu##NAME, #NAME2, 3, 3, 0, 0},
#define MAKE_INSTANCE_ALL(NAME,NAME2) MAKE_INSTANCE(NAME, GPU##NAME2)

    MAKE_INSTANCE_ALL(FloatToDouble, FLOATTODOUBLE)
    MAKE_INSTANCE_ALL(DoubleToFloat, DOUBLETOFLOAT)

    MAKE_INSTANCE_ALL(FloatToComplexReal, FLOATTOCOMPLEXREAL)
    MAKE_INSTANCE_ALL(ComplexRealToFloat, COMPLEXREALTOFLOAT)
    MAKE_INSTANCE_ALL(FloatToComplexImag, FLOATTOCOMPLEXIMAG)
    MAKE_INSTANCE_ALL(ComplexImagToFloat, COMPLEXIMAGTOFLOAT)

    MAKE_INSTANCE_ALL(FloatToDcomplexReal, FLOATTODCOMPLEXREAL)
    MAKE_INSTANCE_ALL(DcomplexRealToFloat, DCOMPLEXREALTOFLOAT)
    MAKE_INSTANCE_ALL(FloatToDcomplexImag, FLOATTODCOMPLEXIMAG) 
    MAKE_INSTANCE_ALL(DcomplexImagToFloat, DCOMPLEXIMAGTOFLOAT)

    MAKE_INSTANCE_ALL(DoubleToComplexReal, DOUBLETOCOMPLEXREAL)
    MAKE_INSTANCE_ALL(ComplexRealToDouble, COMPLEXREALTODOUBLE)
    MAKE_INSTANCE_ALL(DoubleToComplexImag, DOUBLETOCOMPLEXIMAG)
    MAKE_INSTANCE_ALL(ComplexImagToDouble, COMPLEXIMAGTODOUBLE)

    MAKE_INSTANCE_ALL(DoubleToDcomplexReal, DOUBLETODCOMPLEXREAL)
    MAKE_INSTANCE_ALL(DcomplexRealToDouble, DCOMPLEXREALTODOUBLE)
    MAKE_INSTANCE_ALL(DoubleToDcomplexImag, DOUBLETODCOMPLEXIMAG)
    MAKE_INSTANCE_ALL(DcomplexImagToDouble, DCOMPLEXIMAGTODOUBLE)

#undef MAKE_INSTANCE
#undef MAKE_INSTANCE_ALL

    /* mainly IDL functions */
    { IDL_gpuInterpolate1DF,        "GPUINTERPOLATE1DF",      5, 5, 0, 0},
    { IDL_gpuInterpolate2DF,        "GPUINTERPOLATE2DF",      7, 7, 0, 0},
    { IDL_gpuFindgenF,              "GPUFINDGENF",            2, 2, 0, 0},
    { IDL_gpuDindgenD,              "GPUDINDGEND",            2, 2, 0, 0},
    { IDL_gpuCindgenC,              "GPUCINDGENC",            2, 2, 0, 0},
    { IDL_gpuDCindgenZ,             "GPUDCINDGENZ",           2, 2, 0, 0},
    { IDL_gpuTotalF,                "GPUTOTALF",              3, 3, 0, 0},
    { IDL_gpuMinF,                  "GPUMINF",                4, 4, 0, 0},
    { IDL_gpuMaxF,                  "GPUMAXF",                4, 4, 0, 0},
    { IDL_gpuCongrid1DF,            "GPUCONGRID1DF",          5, 5, 0, 0},
    { IDL_gpuCongrid2DF,            "GPUCONGRID2DF",          7, 7, 0, 0},
    { IDL_gpuSubscriptF,            "GPUSUBSCRIPTF",          4, 4, 0, 0},
    { IDL_gpuSubscriptLHSF,         "GPUSUBSCRIPTLHSF",       4, 4, 0, 0},
    { IDL_gpuPrefixSumF,            "GPUPREFIXSUMF",          3, 3, 0, 0},
    { IDL_gpuWhereF,                "GPUWHEREF",              4, 4, 0, 0},

    { IDL_gpuMTF,                   "GPUMTF",                 2, 2, 0, 0},
    { IDL_gpuBoxMullerF,            "GPUBOXMULLERF",          2, 2, 0, 0},
    { IDL_gpuSeedMTF,               "GPUSEEDMTF",             1, 1, 0, 0},
    /* the following are actually from gpuPhysicsOp */
//    { IDL_gpuPoisson,               "GPUPOISSON",             4, 4, 0, 0},
//    { IDL_gpuBrmBremcrossF,         "GPUBRMBREMCROSSF",       5, 5, 0, 0},
//    { IDL_gpuBrmFinnerF,            "GPUBRMFINNERF",          5, 5, 0, 0},
//    { IDL_gpuGauLegF,               "GPUGAULEGF",             6, 6, 0, 0},
  };

  /*
   * Register our routine. The routines must be specified exactly the same
   * as in cudamodule.dlm.
   */
  return IDL_SysRtnAdd(function_addr, TRUE, IDL_CARRAY_ELTS(function_addr));
}
