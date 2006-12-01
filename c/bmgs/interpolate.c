#include "bmgs.h"

#ifdef K

void IP1D(const T* a, int n, int m, T* b, int skip[2])
{
  a += K / 2 - 1;
  for (int j = 0; j < m; j++)
    {
      T* c = b;
      for (int i = 0; i < n; i++)
	{
#if defined(BMGSCOMPLEX) && defined(NO_C99_COMPLEX)
	  c[0].r=a[0].r;c[0].i=a[0].i;if(K==2){c[m].r=0.5*(a[0].r+a[1].r);c[m].i=0.5*(a[0].i+a[1].i);}else if(K==4){c[m].r=(0.5625*(a[0].r+a[1].r)+-0.0625*(a[-1].r+a[2].r));c[m].i=(0.5625*(a[0].i+a[1].i)+-0.0625*(a[-1].i+a[2].i));}else{c[m].r=(0.58593750*(a[0].r+a[1].r)+-0.09765625*(a[-1].r+a[2].r)+0.01171875*(a[-2].r+a[3].r));c[m].i=(0.58593750*(a[0].i+a[1].i)+-0.09765625*(a[-1].i+a[2].i)+0.01171875*(a[-2].i+a[3].i));
	}
#else
	  if (i == 0 && skip[0])
	    c -= m;
	  else
	    c[0] = a[0];

	  if (i == n - 1 && skip[1])
	    c -= m;
	  else
	    {
	      if (K == 2)
		c[m] = 0.5 * (a[0] + a[1]);
	      else if (K == 4)
		c[m] = ( 0.5625 * (a[ 0] + a[1]) +
			 -0.0625 * (a[-1] + a[2]));
	      else
		c[m] = ( 0.58593750 * (a[ 0] + a[1]) +
			 -0.09765625 * (a[-1] + a[2]) +
			 0.01171875 * (a[-2] + a[3]));
	    }
#endif
	  a++;
	  c += 2 * m;
	}
      a += K - 1 - skip[1];
      b++;
    }
}

#else
#  define K 2
#  define IP1D Z(bmgs_interpolate1D2)
#  include "interpolate.c"
#  undef IP1D
#  undef K
#  define K 4
#  define IP1D Z(bmgs_interpolate1D4)
#  include "interpolate.c"
#  undef IP1D
#  undef K
#  define K 6
#  define IP1D Z(bmgs_interpolate1D6)
#  include "interpolate.c"
#  undef IP1D
#  undef K

void Z(bmgs_interpolate)(int k, int skip[3][2], 
			 const T* a, const int size[3], T* b, T* w)
{
  void (*ip)(const T*, int, int, T*, int[2]);
  if (k == 2)
    ip = Z(bmgs_interpolate1D2);
  else if (k == 4)
    ip = Z(bmgs_interpolate1D4);
  else
    ip = Z(bmgs_interpolate1D6);

  int e = k - 1;

  ip(a, size[2] - e + skip[2][1],
     size[0] *
     size[1],
     b, skip[2]);
  ip(b, size[1] - e + skip[1][1], 
     size[0] * 
     ((size[2] - e) * 2 - skip[2][0] + skip[2][1]),
     w, skip[1]);
  ip(w, size[0] - e + skip[0][1], 
     ((size[1] - e) * 2 - skip[1][0] + skip[1][1]) * 
     ((size[2] - e) * 2 - skip[2][0] + skip[2][1]),
     b, skip[0]);
}



#endif
