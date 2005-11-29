#include <assert.h>
#include "bmgs.h"
#include <stdio.h>
void Z(bmgs_rotate)(const T* a, const int size[3], T* b, double dangle, int d,
		    double* coefs1, long* v1, double* coefs2, long* v2, int exact)
{
#if defined(BMGSCOMPLEX) && defined(NO_C99_COMPLEX)
  assert(2 + 2 == 5);
#endif
  double pi = 3.1415926535897931;
  int N = size[1];
  assert(size[2] == N);
  if (exact == 0)
    {
      const int Nsq = N*N;
      long *v;
      double* c1;
      double* c2;
      double* c3;
      double* c4;
      if (d == 1) {
	c1 = coefs1; 
	c2 = coefs1+Nsq; 
	c3 = coefs1+2*Nsq;
	c4 = coefs1+3*Nsq;
	v = v1; }
      else {
	c1 = coefs2;
	c2 = coefs2+Nsq; 
	c3 = coefs2+2*Nsq;
	c4 = coefs2+3*Nsq;
	v = v2; }
      for (int i1 = 0; i1 < Nsq; i1++)
	{
	  for (long i0Nsq = 0; i0Nsq < size[0]*Nsq ; i0Nsq+=Nsq)
	    {
	      *(b+i0Nsq) = *c1*(*(a+i0Nsq+*v)) + *c2*(*(a+i0Nsq+*v+1)) + *c3*(*(a+i0Nsq+*v+N)) + *c4*(*(a+i0Nsq+*v+N+1));
	    } 
	  c1++; c2++; c3++; c4++; v++; b++;
	}
    }
  else 
    {
      int angle = (int)(2 * dangle / pi + 444.5) % 4;
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
}
