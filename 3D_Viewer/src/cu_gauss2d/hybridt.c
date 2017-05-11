/*
*  Hybrid Tausworthe LCG RNG.
*
*  Estimated period is 2^121
*/

#include <stdio.h>

unsigned int z1 = 0xff32422;
unsigned int z2 = 0xee03202;
unsigned int z3 = 0xcc23423;
unsigned int z4 = 0x1235;

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

double HybridTaus()
{
  return (double)(TausStep(&z1, 13, 19, 12, 4294967294UL) ^
          TausStep(&z2, 2, 25, 4, 4294967288UL) ^
          TausStep(&z3, 3, 11, 17, 4294967280UL) ^
          LCGStep(&z4, 1664525, 1013904223UL)) / (double)4294967296.0;
}

int main() {
    int i;

    for(i=0; i < 100; i++)
        printf("%f ", HybridTaus());
    printf("\n");

    return 0;
}

