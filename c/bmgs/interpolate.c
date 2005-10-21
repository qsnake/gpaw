#include "bmgs.h"

#ifdef K

void IP1D(const T* a, int n, int m, T* b)
{
  a += K / 2 - 1;
  for (int j = 0; j < m; j++)
    {
      T* c = b;
      for (int i = 0; i < n; i++)
	{
#if defined(BMGSCOMPLEX) && defined(NO_C99_COMPLEX)
	  c[0].r = a[0].r;
	  c[0].i = a[0].i;
	  if (K == 2)
	    {
	      c[m].r = 0.5 * (a[0].r + a[1].r);
	      c[m].i = 0.5 * (a[0].i + a[1].i);
	    }
	  else if (K == 4)
            {
	      c[m].r = ( 0.5625 * (a[ 0].r + a[1].r) +
			 -0.0625 * (a[-1].r + a[2].r));
	      c[m].i = ( 0.5625 * (a[ 0].i + a[1].i) +
			 -0.0625 * (a[-1].i + a[2].i));
	    }
	  else
	    {
	      c[m].r = ( 0.58593750 * (a[ 0].r + a[1].r) +
			 -0.09765625 * (a[-1].r + a[2].r) +
			 0.01171875 * (a[-2].r + a[3].r));
	      c[m].i = ( 0.58593750 * (a[ 0].i + a[1].i) +
			 -0.09765625 * (a[-1].i + a[2].i) +
			 0.01171875 * (a[-2].i + a[3].i));
	    }
#else
	  c[0] = a[0];
	  if (K == 2)
            c[m] = 0.5 * (a[0] + a[1]);
	  else if (K == 4)
            c[m] = ( 0.5625 * (a[ 0] + a[1]) +
		    -0.0625 * (a[-1] + a[2]));
	  else
            c[m] = ( 0.58593750 * (a[ 0] + a[1]) +
		    -0.09765625 * (a[-1] + a[2]) +
		     0.01171875 * (a[-2] + a[3]));
#endif
	  a++;
	  c += 2 * m;
	}
      a += K - 1;
      b++;
    }
}

#else
#  define K 2
#    define P 2
#    define IP1D Z(bmgs_interpolate1D2_2)
#    include "interpolate.c"
#    undef P
#    undef IP1D
#  undef K
#  define K 4
#    define P 2
#    define IP1D Z(bmgs_interpolate1D4_2)
#    include "interpolate.c"
#    undef P
#    undef IP1D
#  undef K
#  define K 6
#    define P 2
#    define IP1D Z(bmgs_interpolate1D6_2)
#    include "interpolate.c"
#    undef P
#    undef IP1D
#  undef K

void Z(bmgs_interpolate)(int k, int p, 
			 const T* a, const int size[3], T* b, T* w)
{
  void (*ip)(const T*, int, int, T*);
  if (k == 2)
    ip = Z(bmgs_interpolate1D2_2);
  else if (k == 4)
    ip = Z(bmgs_interpolate1D4_2);
  else
    ip = Z(bmgs_interpolate1D6_2);
  int e = k - 1;
  ip(a, (size[2] - e), size[0] * size[1], b);
  ip(b, (size[1] - e), size[0] * (size[2] - e) * p, w);
  ip(w, (size[0] - e), (size[1] - e) * (size[2] - e) * p * p, b);
}



#endif
