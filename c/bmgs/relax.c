#include "bmgs.h"

void bmgs_relax(const bmgsstencil* s, double* a, double* b, const double* src)
     /* Gauss-Seidel relaxation for the equation operator a = src, solution is given in b */ 
{

  a += (s->j[0] + s->j[1] + s->j[2]) / 2;
  for (int i0 = 0; i0 < s->n[0]; i0++)
    {
      for (int i1 = 0; i1 < s->n[1]; i1++)
	{
	  for (int i2 = 0; i2 < s->n[2]; i2++)
	    {
	      double x = 0.0;
	      for (int c = 1; c < s->ncoefs; c++)
		x += a[s->offsets[c]] * s->coefs[c];
	      *b++ = (*src - x)/s->coefs[0];
	      *a++ = (*src - x)/s->coefs[0];
	      src++;
	    }
	  a += s->j[2];
	}
      a += s->j[1];
    }
}
