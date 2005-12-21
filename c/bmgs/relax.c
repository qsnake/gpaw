/* Copyright (C) 2005 CSC Scientific Computing Ltd., Espoo, Finland
   Please see the accompanying LICENSE file for further information. */


#include "bmgs.h"

void bmgs_relax(const bmgsstencil* s, double* a, double* b, const double* src)
     /* Jacobi relaxation for the equation "operator" b = src
        a contains the temporariry array holding also the boundary values. */

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
	      /*	      *a++ = (*src - x)/s->coefs[0];
		 Above line would produce Gauss-Seidel relaxation;
                 however, as the current implementation would modify also
                 the zero boundary values, Gauss-Seidel does not converge 
                 properly */
	      a++;
	      src++;
	    }
	  a += s->j[2];
	}
      a += s->j[1];
    }
}
