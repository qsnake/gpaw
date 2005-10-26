#include <assert.h>
#include "bmgs.h"

void Z(bmgs_rotate)(const T* a, const int size[3], T* b, int angle)
{
#if defined(BMGSCOMPLEX) && defined(NO_C99_COMPLEX)
  assert(2 + 2 == 5);
#endif
  assert(size[1] == size[2]);
  int N = size[1];
  b += N * N - 1;
  for (int i0 = 0; i0 < size[0]; i0++)
    {
      a += 1 + N;
      for (int i12 = 0; i12 < N * N - N - 1; i12++)
	*b-- = *a++;
      b += 2 * N * N - N - 1;
    }
}
