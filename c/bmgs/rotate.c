#include <assert.h>
#include "bmgs.h"
#include <stdio.h>
void Z(bmgs_rotate)(const T* a, const int size[3], T* b, int angle)
{
#if defined(BMGSCOMPLEX) && defined(NO_C99_COMPLEX)
  assert(2 + 2 == 5);
#endif
  int N = size[1];
  assert(size[2] == N);
  angle = angle % 4;
  if (angle < 0) { angle += 4;}
  if (angle == 0)
      {
	for (int i0 = 0; i0 < size[0]; i0++)
	  {
	    a+=1+N;
	    b+=1+N;
	    for (int i1 = 0; i1 < N*N-N-1; i1++)
	      *(b++)=*(a++);
	  }
      }
    else if (angle == 1)
    {
       a += 2*N-1;
       b += N+1; 
      for (int i0 = 0; i0 < size[0]; i0++)
	{
	for (int i1 = 1; i1 < N ; i1++)
	  { 
	  for (int i2 = 1; i2 < N ; i2++)
	     {
		*b = *a;
		b++;
		a += N;
	     }
	      a -= N*(N-1)+1;
	      b += 1;
	    }
	 a += N*(N+1)-1;
	 b += N;
	}
    }
  else if (angle == 2)
    {
      b += N * N - 1;
      for (int i0 = 0; i0 < size[0]; i0++)
	{
	  a += 1 + N;
	  for (int i12 = 0; i12 < N * N - N - 1; i12++)
	    { *b-- = *a++;}
	     b += 2 * N * N - N - 1;
	}
    }
  else if (angle == 3)
    {
       a += N*N-N+1;
       b += N+1; 
      for (int i0 = 0; i0 < size[0]; i0++)
	{
	 for (int i1 = 1; i1 < N ; i1++)
	    {
	      for (int i2 = 1; i2 < N ; i2++)
		{
		  *b = *a;
		  b++;
		  a -= N;
		}
	      a += N*N-N+1;
	      b += 1;
	    }
	 a += N*N-N+1;
	 b += N; 
	}
    }
    else
      {
      printf("Error: Angle= %d\n",angle);
	}
}
