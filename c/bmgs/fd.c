#include "bmgs.h"

void Z(bmgs_fd)(const bmgsstencil* s, const T* a, T* b)
{
  a += (s->j[0] + s->j[1] + s->j[2]) / 2;
  for (int i0 = 0; i0 < s->n[0]; i0++)
    {
      for (int i1 = 0; i1 < s->n[1]; i1++)
	{
	  for (int i2 = 0; i2 < s->n[2]; i2++)
	    {
#if defined(BMGSCOMPLEX) && defined(NO_C99_COMPLEX)
	      double x = 0.0;
	      double y = 0.0;
	      for (int c = 0; c < s->ncoefs; c++)
		{
		  x += a[s->offsets[c]].r * s->coefs[c];
		  y += a[s->offsets[c]].i * s->coefs[c];
		}
	      (*b).r = x;
	      (*b).i = y;
	      b++;
#else
	      T x = 0.0;
	      for (int c = 0; c < s->ncoefs; c++)
		x += a[s->offsets[c]] * s->coefs[c];
	      *b++ = x;
#endif
	      a++;
	    }
	  a += s->j[2];
	}
      a += s->j[1];
    }
}
