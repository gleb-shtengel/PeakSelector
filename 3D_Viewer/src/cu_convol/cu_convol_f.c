/*
*  file:  cu_convol_f.c
*
*  DLM code for GPU convolution.
*
*  RTK, 18-Dec-2009
*  Last update:  18-Dec-2009
*/

#include "cuda_runtime_api.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "idl_export.h"

#include "convol_f.h"

/*  
*  Kernel and CUDA functions  
*/
extern void convol_f(float *imgs, int nx, int ny, int nz,
                     float *kernel, int nk,
                     float *out, int c1, int c2);
extern void cuda_safe_init(void);

/*
*  Message block definitions
*/
static IDL_MSG_BLOCK msg_block;  

#define CONVOL_F_NO_MEMORY             0
#define CONVOL_F_NOT_SCALAR           -1
#define CONVOL_F_NOT_NUMERIC          -2
#define CONVOL_F_UNEQUAL_LENGTHS      -3
#define CONVOL_F_NOT_ARRAY            -4
#define CONVOL_F_MUST_BE_FLOAT        -5
#define CONVOL_F_MUST_BE_INT          -6
#define CONVOL_F_MUST_BE_3D           -7
#define CONVOL_F_UNEQUAL_FRAMES       -8
#define CONVOL_F_TOO_MANY_FRAMES      -9
#define CONVOL_F_CARDS_MUST_BE_UNIQUE -10
#define CONVOL_F_MUST_BE_2D           -11
#define CONVOL_F_SQUARE_KERNEL        -12

static IDL_MSG_DEF msg_arr[] = {  
  {"CONVOL_F_NO_MEMORY", "%NUnable to allocate memory"},
  {"CONVOL_F_NOT_SCALAR", "%NNot a scalar"},
  {"CONVOL_F_NOT_NUMERIC", "%NNot numeric"},
  {"CONVOL_F_UNEQUAL_LENGTHS", "%NVector lengths are not equal"},
  {"CONVOL_F_NOT_ARRAY", "%NNot an array"},
  {"CONVOL_F_MUST_BE_FLOAT", "%NMust be of type float"},
  {"CONVOL_F_MUST_BE_INT", "%NMust be of type long"},
  {"CONVOL_F_MUST_BE_3D", "%NMust be 3D"},
  {"CONVOL_F_UNEQUAL_FRAMES", "%NThe number of frames must match"},
  {"CONVOL_F_TOO_MANY_FRAMES", "%NToo many frames"},
  {"CONVOL_F_CARDS_MUST_BE_UNIQUE", "%NCARD1 and CARD2 cannot both be the same"},
  {"CONVOL_F_MUST_BE_2D", "%NMust be 2D"},
  {"CONVOL_F_SQUARE_KERNEL", "%NKernel must be square"},
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
*  pickCards
*
*  Check for Tesla cards and choose the first two, if any.
*/
void pickCards(int *ncards, int *card1, int *card2) {
    struct cudaDeviceProp prop;
    int i;

    cudaGetDeviceCount(ncards);

    if (*ncards == 0) {
        *card1 = *card2 = -1;
        return;
    }
    if (*ncards == 1) {
        *card1 = 0;
        *card2 = -1;
        return;
    }

    *card1 = *card2 = -1;

    //  Look for two Tesla cards
    for(i=0; i < *ncards; i++) {
        cudaGetDeviceProperties(&prop, i);
        if (strncmp(prop.name, "Tesla", 5) == 0) {
            if (*card1 == -1) {
                *card1 = i;
            } else {
                *card2 = i;
                return;  // found two, return now
            }
        }
    }

    //  Look for two cards of any kind
    for(i=0; i < *ncards; i++) {
        cudaGetDeviceProperties(&prop, i);
        if (*card1 == -1) {
            *card1 = i;
        } else {
            *card2 = i;
            return;
        }
    }
}


/**************************************************************
*  idl_convol_f
*/
static IDL_VPTR idl_convol_f(int argc, IDL_VPTR *argv, char *argk) {
  IDL_VPTR ans;
  float *img;
  unsigned int totalMem, a;
  UCHAR *out;
  IDL_MEMINT dims[3];
  int xdim, ydim, nframes, segsize, nseg, nelem, k, idx, n;
  int max_frames, card1, card2, ntesla, nk, ncards;
  float *kernel;
  struct cudaDeviceProp info;

    typedef struct {
        IDL_KW_RESULT_FIRST_FIELD;
        char card1;
        int card1_present;
        char card2;
        int card2_present;
    } KW_RESULT;

    static IDL_KW_PAR kw_pars[] = {
        {"CARD1", IDL_TYP_LONG, 1, 0, IDL_KW_OFFSETOF(card1_present), IDL_KW_OFFSETOF(card1) },
        {"CARD2", IDL_TYP_LONG, 1, 0, IDL_KW_OFFSETOF(card2_present), IDL_KW_OFFSETOF(card2) },
        { NULL }
    };

    KW_RESULT kw;
    
    //  Process keywords
    IDL_KWProcessByOffset(argc, argv, argk, kw_pars, (IDL_VPTR *)0, 1, &kw);

    //  Set up the cards to use
    card1 = (kw.card1_present) ? kw.card1 : -1;
    card2 = (kw.card2_present) ? kw.card2 : -1;

    if ((card1 != -1) && (card2 != -1) && (card1 == card2)) {
        IDL_MessageFromBlock(msg_block, CONVOL_F_CARDS_MUST_BE_UNIQUE, IDL_MSG_LONGJMP);
    }

#ifdef DEBUG
    printf("Card 1 is %d\n", card1);
    printf("Card 2 is %d\n", card2);
#endif

  if ((card1 == -1) && (card2 == -1)) {
    pickCards(&ncards, &card1, &card2);
  }

  /*  1st argument is the input image  */
  if (is_scalar(argv[0])) {
    IDL_MessageFromBlock(msg_block, CONVOL_F_NOT_ARRAY, IDL_MSG_LONGJMP);
  }
  if (!is_numeric(argv[0])) {
    IDL_MessageFromBlock(msg_block, CONVOL_F_NOT_NUMERIC, IDL_MSG_LONGJMP);
  }
  if (argv[0]->type != IDL_TYP_FLOAT) {
    IDL_MessageFromBlock(msg_block, CONVOL_F_MUST_BE_FLOAT, IDL_MSG_LONGJMP);
  }
  if (argv[0]->value.arr->n_dim != 3) {
    IDL_MessageFromBlock(msg_block, CONVOL_F_MUST_BE_3D, IDL_MSG_LONGJMP);
  }
  img = (float *)argv[0]->value.arr->data;
  xdim = (int)argv[0]->value.arr->dim[0];
  ydim = (int)argv[0]->value.arr->dim[1];
  nframes = (int)argv[0]->value.arr->dim[2];

  /*  2nd argument is the kernel  */
  if (is_scalar(argv[1])) {
    IDL_MessageFromBlock(msg_block, CONVOL_F_NOT_ARRAY, IDL_MSG_LONGJMP);
  }
  if (!is_numeric(argv[1])) {
    IDL_MessageFromBlock(msg_block, CONVOL_F_NOT_NUMERIC, IDL_MSG_LONGJMP);
  }
  if (argv[1]->type != IDL_TYP_FLOAT) {
    IDL_MessageFromBlock(msg_block, CONVOL_F_MUST_BE_FLOAT, IDL_MSG_LONGJMP);
  }
  if (argv[1]->value.arr->n_dim != 2) {
    IDL_MessageFromBlock(msg_block, CONVOL_F_MUST_BE_2D, IDL_MSG_LONGJMP);
  }
  kernel = (float *)argv[1]->value.arr->data;
  nk = (int)argv[1]->value.arr->dim[0];
  if (argv[1]->value.arr->dim[1] != nk) {
    IDL_MessageFromBlock(msg_block, CONVOL_F_SQUARE_KERNEL, IDL_MSG_LONGJMP);
  }

  /*
  *  Make the output image, same size as the input
  */
  out = (UCHAR *)malloc(xdim*ydim*nframes*sizeof(float));
  
  dims[0] = xdim;
  dims[1] = ydim;
  dims[2] = nframes;
  ans = IDL_ImportArray(3, dims, IDL_TYP_FLOAT, out,
            (IDL_ARRAY_FREE_CB)freeHostCB, NULL);

  /*
  *  Call the kernel repeatedly until all frames processed
  */
  ntesla = (card1 != -1) + (card2 != -1);

  if (ntesla == 1) {
    cudaGetDeviceProperties(&info, card1);
    totalMem = (unsigned int)floor(0.85*(double)info.totalGlobalMem);
  } else {
    cudaGetDeviceProperties(&info, card1);
    a = info.totalGlobalMem;
    cudaGetDeviceProperties(&info, card2);
    totalMem = (a < info.totalGlobalMem) ? a : info.totalGlobalMem;
    totalMem = (unsigned int)floor(0.85*(double)totalMem);
  }

  nelem = xdim*ydim;
  max_frames = totalMem / (8*nelem);  // max frames per card
  segsize = ntesla*max_frames;        // scale by number of cards (1 or 2)
  nseg = 1 + nframes/segsize;         // number of segments required to process all frames

#ifdef DEBUG
printf("info.totalGlobalMem = %15.0f\n", (double)info.totalGlobalMem);
printf("totalMem            = %15.0f\n", (double)totalMem);
printf("max_frames          = %d\n", max_frames);
printf("segsize             = %d\n", segsize);
printf("nseg                = %d\n", nseg);
#endif

  for(k=0; k < nseg; k++) {
    idx = k*segsize;
    n = nframes - k*segsize;
    n = (n > segsize) ? segsize : n;

    if (n != 0) {
      convol_f(img+nelem*idx, xdim, ydim, n, kernel, nk,
               (float *)out+nelem*idx, card1, card2);
    }
  }

  //  Output already filled in, just return
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
    {{(IDL_FUN_RET) idl_convol_f}, "CU_CONVOL_F", 2, 2, IDL_SYSFUN_DEF_F_KEYWORDS, 0},
  };  

  /*  Error messages  */
  if (!(msg_block = IDL_MessageDefineBlock("CONVOL_F", IDL_CARRAY_ELTS(msg_arr), msg_arr))) {
    return IDL_FALSE;  
  }

  /*  Initialize CUDA when the module is loaded  */
  cuda_safe_init();

  return IDL_SysRtnAdd(function_addr, TRUE, IDL_CARRAY_ELTS(function_addr)); 
}

/*  Module description - make pulls this out to create the DLM file  */
//dlm: MODULE CU_CONVOL_F
//dlm: DESCRIPTION CUDA CONVOLUTION, HHMI
//dlm: VERSION 1.0
//dlm: FUNCTION CU_CONVOL_F 2 2 KEYWORDS

/*
*  end cu_convol_f.c
*/

