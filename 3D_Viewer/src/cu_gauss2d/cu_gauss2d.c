/*
*  file:  cu_gauss2d.c
*
*  GPU based 2D Gaussian fitting.
*
*  RTK, 24-Sep-2009
*  Last update:  02-Dec-2009
*/

#define DEBUG

#include <math.h>
#include "gauss2d.h"
#include "idl_export.h"

/*  
*  Kernel and CUDA functions  
*/
extern void gauss2d(unsigned short *imgs, int nimgs,   //  input images, assumed 11x11
                    unsigned int *seeds, int nseeds,   //  particle random number seeds
                    float *constraints,                //  particle constraints
                    int imax,                          //  number of iterations to perform
                    float *out,                        //  output fit parameters plus chi-square
                    int uncertainties,                 //  if true, calculate uncertainties
                    int c1, int c2);                   //  CUDA cards to use or -1

/*
*  Message block definitions
*/
static IDL_MSG_BLOCK msg_block;  

#define CU_GAUSS2D_NO_MEMORY             0
#define CU_GAUSS2D_NOT_SCALAR           -1
#define CU_GAUSS2D_NOT_NUMERIC          -2
#define CU_GAUSS2D_UNEQUAL_LENGTHS      -3
#define CU_GAUSS2D_MUST_BE_SCALAR       -4
#define CU_GAUSS2D_MUST_BE_UINT         -5
#define CU_GAUSS2D_MUST_BE_INT          -6
#define CU_GAUSS2D_NOT_ARRAY            -7
#define CU_GAUSS2D_MUST_BE_FLOAT        -8
#define CU_GAUSS2D_TOO_MANY_FRAMES      -9
#define CU_GAUSS2D_MUST_BE_BYTE        -10
#define CU_GAUSS2D_TOO_MANY_PARAMS     -11
#define CU_GAUSS2D_NO_MATCH            -12
#define CU_GAUSS2D_CARDS_MUST_BE_UNIQUE -13

static IDL_MSG_DEF msg_arr[] = {  
  {"CU_GAUSS2D_NO_MEMORY", "%NUnable to allocate memory"},
  {"CU_GAUSS2D_NOT_SCALAR", "%NNot a scalar"},
  {"CU_GAUSS2D_NOT_NUMERIC", "%NNot numeric"},
  {"CU_GAUSS2D_UNEQUAL_LENGTHS", "%NVector lengths are not equal"},
  {"CU_GAUSS2D_MUST_BE_SCALAR", "%NMust be a scalar"},
  {"CU_GAUSS2D_MUST_BE_UINT", "%NMust be of type unsigned int (type 12)"},
  {"CU_GAUSS2D_MUST_BE_INT", "%NMust be of type long"},
  {"CU_GAUSS2D_NOT_ARRAY", "%NNot an array"},
  {"CU_GAUSS2D_MUST_BE_FLOAT", "%NMust be of type float"},
  {"CU_GAUSS2D_TOO_MANY_FRAMES", "%NToo many frames"},
  {"CU_GAUSS2D_MUST_BE_BYTE", "%NMust be of type byte"},
  {"CU_GAUSS2D_TOO_MANY_PARAMS", "%NParameter space exceeds constant memory limit"},
  {"CU_GAUSS2D_NO_MATCH", "%NThe filter index length must match the number of parameters"},
  {"CU_GAUSS2D_CARDS_MUST_BE_UNIQUE", "%NCARD1 and CARD2 cannot both be the same"},
};

#ifndef WIN32
/**************************************************************
*  getTime
*
*  Return the system time.
*/
double getTime() {
    struct timeval tv;
    cudaThreadSynchronize();
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000;
}
#endif


/**************************************************************
*  Integer hybrid Tausworthe generator
*/

unsigned int z1 = 0xff32422;
unsigned int z2 = 0xee03202;
unsigned int z3 = 0xcc23423;
unsigned int z4 = 0x1235;  // can update this as the seed

unsigned int TausStep(unsigned *z, int S1, int S2, int S3, unsigned int M)
{
    unsigned int b = ((*z << S1) ^ *z) << S2;
    *z = (((*z & M) << S3) ^ b);
    return *z;
}

unsigned int LCGStep(unsigned int *z, unsigned int A, unsigned int C)
{
    *z = (A*(*z)+C);
    return *z;
}

unsigned int HybridTaus()
{
    return TausStep(&z1, 13, 19, 12, 4294967294UL) ^
           TausStep(&z2, 2, 25, 4, 4294967288UL) ^
           TausStep(&z3, 3, 11, 17, 4294967280UL) ^
           LCGStep(&z4, 1664525, 1013904223UL);
}


/**************************************************************
*  Park-Miller LCG
*/
void rnd(unsigned int *seed) {
    *seed = 16807*(*seed) % 2147483647;
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


/**************************************************************
*  freeHostCB
*/
static void freeHostCB(UCHAR *p) {
    cudaFreeHost(p);
}


/**************************************************************
*  generate_seeds
*/
static void generate_seeds(unsigned int *seeds, int nseeds) {
    int i;

#ifndef WIN32
    srand((unsigned int)time(NULL));
#endif

    //  Set up the Tausworthe and LCG seeds
    z1 = rand();
    z1 = (z1 < 128) ? z1+1001 : z1;
    z2 = rand();
    z2 = (z2 < 128) ? z2+1002 : z2;
    z3 = rand();
    z3 = (z3 < 128) ? z3+1003 : z3;
    z4 = rand();

    //  Now define a seed for each particle
    for(i=0; i < nseeds; i++)
        seeds[i] = HybridTaus();
}


/**************************************************************
*  generate_seeds_stepped
*
*  Use seeds from the MINSTD sequence separated by 60,000.
*  (more efficient methods of getting the stepped seeds exist!)
*/
static void generate_seeds_stepped(unsigned int *seeds, int nseeds) {
    unsigned int seed = 1;
    int i,k;

    for(i=0; i < nseeds; i++) {
        seeds[i] = seed;
        for(k=0; k < 60000; k++)
            rnd(&seed);
    }
}


#if 0
/**************************************************************
*  calculate_uncertainties
*
*  Calculate the fit parameter uncertainties ala Bevington 11-36.
*/
static void calculate_uncertainties(float *gp, int nimgs) {
  float *p, *s;
  float arg, u, u2, g2, g3, g4, g5;
  int i,x,y,k;
  float D = 2.0 / (121-6);  

  for(k=0; k < nimgs; k++) {
    p = &gp[(6+6+1)*k];    //  parameters for the current image, 0..5
    s = &gp[(6+6+1)*k+7];  //  parameter uncertainties here

    //  Zero output values
    memset((void *)p, 6*sizeof(float), 0);

    //  s[0] is trivial case
    s[0] = 121;

    //  Calculate s[1..5]
    for(y=0; y < 11; y++) {
      for(x=0; x < 11; x++) {
        //arg = -0.5*(((x-p[4])/p[2])^2+((y-p[5])/p[3])^2);
        arg = -0.5*(pow((x-p[4])/p[2],2)+pow((y-p[5])/p[3],2));
        arg = (arg < -50.0) ? -50.0 : arg;
        arg = (arg > +50.0) ? +50.0 : arg;

        u = exp(arg);
        u2 = u*u;
        //g2 = u*(p[1]*(x-p[4])^4/p[2]^6 - 3.0*p[1]*(x-p[4])^2/p[2]^4);
        //g3 = u*(p[1]*(y-p[5])^4/p[3]^6 - 3.0*p[1]*(y-p[5])^2/p[3]^4);
        //g4 = u*(p[1]*(x-p[4])^2/p[2]^4 - p[1]/p[2]^2);
        //g5 = u*(p[1]*(y-p[5])^2/p[3]^4 - p[1]/p[3]^2);
        g2 = u*(p[1]*pow(x-p[4],4)/pow(p[2],6) - 3.0*p[1]*pow(x-p[4],2)/pow(p[2],4));
        g3 = u*(p[1]*pow(y-p[5],4)/pow(p[3],6) - 3.0*p[1]*pow(y-p[5],2)/pow(p[3],4));
        g4 = u*(p[1]*pow(x-p[4],2)/pow(p[2],4) - p[1]/pow(p[2],2));
        g5 = u*(p[1]*pow(y-p[5],2)/pow(p[3],4) - p[1]/pow(p[3],2));

        s[1] += u2;
        //s[2] = (p[1]*(x-p[4])^2/p[2]^3)^2*u2 - y*g2 + (p[0]+p[1]*u)*g2;
        //s[3] = (p[1]*(y-p[5])^2/p[3]^3)^2*u2 - y*g3 + (p[0]+p[1]*u)*g3;
        //s[4] = (p[1]*(x-p[4])/p[2]^2)^2*u2 - y*g4 + (p[0]+p[1]*u)*g4;  
        //s[5] = (p[1]*(y-p[5])/p[3]^2)^2*u2 - y*g5 + (p[0]+p[1]*u)*g5;  
        s[2] += (pow(p[1]*pow(x-p[4],2)/pow(p[2],3),2)*u2 - y*g2 + (p[0]+p[1]*u)*g2);
        s[3] += (pow(p[1]*pow(y-p[5],2)/pow(p[3],3),2)*u2 - y*g3 + (p[0]+p[1]*u)*g3);
        s[4] += (pow(p[1]*(x-p[4])/pow(p[2],2),2)*u2 - y*g4 + (p[0]+p[1]*u)*g4);  
        s[5] += (pow(p[1]*(y-p[5])/pow(p[3],2),2)*u2 - y*g5 + (p[0]+p[1]*u)*g5);  
      }
    }

    //  Calculate final value
    for(i=0; i < 7; i++) {
      s[i] = sqrt(2.0/fabs(D*s[i]));
    }
  }
}
#endif


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
*  calc_num_segments
*
*  Determine the number of segments the source images must
*  be broken up into in order to fit on the card.
*/
int calc_num_segments(int nimgs, int *segsize, int c1, int c2) {
    struct cudaDeviceProp prop;
    size_t tot_mem, scale, sys_mem;
    int segs, ncards;
    size_t totalGlobalMem, a;
    int card1, card2;

    if (c1 == -1) {
        pickCards(&ncards, &card1, &card2);
    } else {
        card1 = c1;
        card2 = c2;
    }

    ncards = (card1 != -1) + (card2 != -1);

    if (ncards == 1) {
        cudaGetDeviceProperties(&prop, card1);
        totalGlobalMem = prop.totalGlobalMem;
    } else {
        cudaGetDeviceProperties(&prop, card1);
        a = prop.totalGlobalMem;
        cudaGetDeviceProperties(&prop, card2);
        totalGlobalMem = (a < prop.totalGlobalMem) ? a : prop.totalGlobalMem;
    }

    scale = 11*11*sizeof(unsigned short) + 
            PARTICLES_PER_IMAGE*sizeof(unsigned int) +
            (6+6+1)*sizeof(float);
    tot_mem = nimgs * scale;
    sys_mem = (size_t)(0.9*totalGlobalMem) * ncards;
    segs = (int)(tot_mem/sys_mem)+1;
    *segsize = sys_mem / scale;
    return segs;
}


/**************************************************************
*  idl_gauss2d
*/
static IDL_VPTR idl_gauss2d(int argc, IDL_VPTR *argv, char *argk) {
    IDL_VPTR ans;
    int imax, n, i, idx;
    IDL_MEMINT dims;
    unsigned short *imgs;
    unsigned int *seeds;
    float *out, *constraints;
    int nimgs, nseeds, uncertainties = 1, nsegs, segsize;
    int card1, card2;
    double sss,eee,sss1,eee1;

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
    
    sss1 = getTime();

    //  Process keywords
    IDL_KWProcessByOffset(argc, argv, argk, kw_pars, (IDL_VPTR *)0, 1, &kw);

    //  Set up the cards to use
    card1 = (kw.card1_present) ? kw.card1 : -1;
    card2 = (kw.card2_present) ? kw.card2 : -1;

    if ((card1 != -1) && (card2 != -1) && (card1 == card2)) {
        IDL_MessageFromBlock(msg_block, CU_GAUSS2D_CARDS_MUST_BE_UNIQUE, IDL_MSG_LONGJMP);
    }

#ifdef DEBUG
    printf("Card 1 is %d\n", card1);
    printf("Card 2 is %d\n", card2);
#endif
    
    //  1st argument is the stack of images
    if (is_scalar(argv[0])) {
        IDL_MessageFromBlock(msg_block, CU_GAUSS2D_NOT_ARRAY, IDL_MSG_LONGJMP);
    }
    if (!is_numeric(argv[0])) {
        IDL_MessageFromBlock(msg_block, CU_GAUSS2D_NOT_NUMERIC, IDL_MSG_LONGJMP);
    }
    if (argv[0]->type != IDL_TYP_UINT) {
        IDL_MessageFromBlock(msg_block, CU_GAUSS2D_MUST_BE_UINT, IDL_MSG_LONGJMP);
    }
    imgs = (unsigned short *)argv[0]->value.arr->data;
    nimgs = (int)argv[0]->value.arr->dim[2];
  
    //  2nd argument is the number of iterations
    if (!is_scalar(argv[1])) {
        IDL_MessageFromBlock(msg_block, CU_GAUSS2D_MUST_BE_SCALAR, IDL_MSG_LONGJMP);
    }
    if (!is_numeric(argv[1])) {
        IDL_MessageFromBlock(msg_block, CU_GAUSS2D_NOT_NUMERIC, IDL_MSG_LONGJMP);
    }
    imax = (int)get_numeric_value(argv[1]);
  
    //  3rd argument is the constraints
    if (is_scalar(argv[2])) {
        IDL_MessageFromBlock(msg_block, CU_GAUSS2D_NOT_ARRAY, IDL_MSG_LONGJMP);
    }
    if (!is_numeric(argv[2])) {
        IDL_MessageFromBlock(msg_block, CU_GAUSS2D_NOT_NUMERIC, IDL_MSG_LONGJMP);
    }
    if (argv[2]->type != IDL_TYP_FLOAT) {
        IDL_MessageFromBlock(msg_block, CU_GAUSS2D_MUST_BE_FLOAT, IDL_MSG_LONGJMP);
    }
    constraints = (float *)argv[2]->value.arr->data;
  
    //  4th argument is the uncertainties flag
    if (!is_scalar(argv[3])) {
        IDL_MessageFromBlock(msg_block, CU_GAUSS2D_MUST_BE_SCALAR, IDL_MSG_LONGJMP);
    }
    if (!is_numeric(argv[3])) {
        IDL_MessageFromBlock(msg_block, CU_GAUSS2D_NOT_NUMERIC, IDL_MSG_LONGJMP);
    }
    uncertainties = ((int)get_numeric_value(argv[3]) != 0);
  
    //  Make the output parameter array
    sss = getTime();
    //cudaMallocHost((void **)&out, (6+6+1)*nimgs*sizeof(float));
    out = (float *)malloc((6+6+1)*nimgs*sizeof(float));
    dims = (6+6+1)*nimgs;
    //ans = IDL_ImportArray(1, &dims, IDL_TYP_FLOAT, (UCHAR *)out, (IDL_ARRAY_FREE_CB)freeHostCB, NULL);
    ans = IDL_ImportArray(1, &dims, IDL_TYP_FLOAT, (UCHAR *)out, (IDL_ARRAY_FREE_CB)NULL, NULL);
    eee = getTime();
#ifdef DEBUG
    printf("Creating output array in IDL = %f\n", eee-sss);
#endif

    //  Create the PRNG seeds
    sss = getTime();
    nseeds = PARTICLES_PER_IMAGE*nimgs;  // one seed per particle per image
    seeds = (unsigned int *)malloc(nseeds*sizeof(unsigned int));
    if (!seeds) {
        IDL_MessageFromBlock(msg_block, CU_GAUSS2D_NO_MEMORY, IDL_MSG_LONGJMP);
    }
  
    generate_seeds(seeds, nseeds);
    eee = getTime();
#ifdef DEBUG
    printf("Generate RNG seeds = %f\n", eee-sss);
#endif

    //  Call the kernel
    nsegs = calc_num_segments(nimgs, &segsize, card1, card2);

#ifdef DEBUG
    printf("nsegs = %d\n", nsegs);
    printf("segsize = %d\n", segsize);
#endif
    
    if (nsegs == 1) {
        gauss2d(imgs, nimgs, seeds, nseeds, constraints, imax, out, uncertainties,
                card1, card2);
    } else {
        for(i=0; i < nsegs; i++) {
            idx = 11*11*i*segsize;
            n = nimgs - i*segsize;
            n = (n > segsize) ? segsize : n;
#ifdef DEBUG
            printf("n images = %d\n", n);
#endif
            if (n != 0) {
                gauss2d(&imgs[idx], n, &seeds[i*segsize], n, constraints, imax, &out[(6+6+1)*i*segsize], 
                        uncertainties, card1, card2);
            }
        }
    }
  
    //  Clean up and return
    free(seeds);
    eee1 = getTime();
#ifdef DEBUG
    printf("cu_gauss2d call time = %f\n", eee1-sss1);
#endif
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
        {{(IDL_FUN_RET) idl_gauss2d}, "CU_GAUSS2D", 4, 4, IDL_SYSFUN_DEF_F_KEYWORDS, 0},
    };  

    /*  Error messages  */
    if (!(msg_block = IDL_MessageDefineBlock("GAUSS2D", IDL_CARRAY_ELTS(msg_arr), msg_arr))) {
        return IDL_FALSE;  
    }

    return IDL_SysRtnAdd(function_addr, TRUE, IDL_CARRAY_ELTS(function_addr)); 
}

/*  Module description - make pulls this out to create the DLM file  */
//dlm: MODULE CU_GAUSS2D
//dlm: DESCRIPTION GPU-based 2D Gaussian fitting, HHMI
//dlm: VERSION 1.0
//dlm: FUNCTION CU_GAUSS2D 4 4 KEYWORDS

/*
*  end cu_gauss2d.c
*/

