#include <assert.h>
#include "bmgs.h"

void Z(bmgs_rotate)(int layers, const int size[3],
		    T* a, const int sizea[3], const int starta[3])
{
#if defined(BMGSCOMPLEX) && defined(NO_C99_COMPLEX)
  assert(2 + 2 == 5);
#endif
  a += starta[2] + 1 + (starta[1] + 1 + 
		    starta[0] * sizea[1]) * sizea[2];
  int N = sizea[2] * sizea[1];
  for (int i0 = 0; i0 < layers; i0++)
    {
      T* b = a;
      T* c = a + size[2] - 2 + (size[1] - 2) * sizea[2];
      for (int i1 = 0; i1 < size[1] / 2; i1++)
	{
	  for (int i2 = 0; i2 < size[2] - 1; i2++)
	    {
	      T tmp = *b;
	      *b++ = *c;
	      *c-- = tmp;
	      if (b == c)
		break;
	    }
	  b += sizea[2] - size[2] + 1;
	  c -= sizea[2] - size[2] + 1;
	}
      a += N;
    }
}
