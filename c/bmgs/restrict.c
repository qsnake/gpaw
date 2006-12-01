#include "bmgs.h"

#ifdef K

void RST1D(const T* a, int n, int m, T* b)
{
  a += K - 1;
  for (int j = 0; j < m; j++)
    {
      T* c = b;
      for (int i = 0; i < n; i++)
	{
#if defined(BMGSCOMPLEX) && defined(NO_C99_COMPLEX)
	  if(K==2){c[0].r=0.5*(a[0].r+0.5*(a[1].r+a[-1].r));c[0].i=0.5*(a[0].i+0.5*(a[1].i+a[-1].i));}else if(K==4){c[0].r=0.5*(a[0].r+0.5625*(a[1].r+a[-1].r)+-0.0625*(a[3].r+a[-3].r));c[0].i=0.5*(a[0].i+0.5625*(a[1].i+a[-1].i)+-0.0625*(a[3].i+a[-3].i));}else if(K==6){c[0].r=0.5*(a[0].r+0.58593750*(a[1].r+a[-1].r)+-0.09765625*(a[3].r+a[-3].r)+0.01171875*(a[5].r+a[-5].r));c[0].i=0.5*(a[0].i+0.58593750*(a[1].i+a[-1].i)+-0.09765625*(a[3].i+a[-3].i)+0.01171875*(a[5].i+a[-5].i));}
#else
	  if      (K == 2)
	    c[0] = 0.5 * (a[0] + 
			  0.5 * (a[1] + a[-1]));

	  else if (K == 4)
	    c[0] = 0.5 * (a[0] + 
			   0.5625 * (a[1] + a[-1]) +
			  -0.0625 * (a[3] + a[-3]));

	  else if (K == 6)
	    c[0] = 0.5 * (a[0] + 
			   0.58593750 * (a[1] + a[-1]) +
			  -0.09765625 * (a[3] + a[-3]) +
			   0.01171875 * (a[5] + a[-5]));
#endif
	  a += 2;
	  c += m;
	}
      a += K * 2 - 3;
      b++;
    }
}

#else
#  define K 2
#  define RST1D Z(bmgs_restrict1D2)
#  include "restrict.c"
#  undef RST1D
#  undef K
#  define K 4
#  define RST1D Z(bmgs_restrict1D4)
#  include "restrict.c"
#  undef RST1D
#  undef K
#  define K 6
#  define RST1D Z(bmgs_restrict1D6)
#  include "restrict.c"
#  undef RST1D
#  undef K

void Z(bmgs_restrict)(int k, T* a, const int n[3], T* b, T* w)
{
  void (*plg)(const T*, int, int, T*);

  if (k == 2)
    plg = Z(bmgs_restrict1D2);
  else if (k == 4)
    plg = Z(bmgs_restrict1D4);
  else
    plg = Z(bmgs_restrict1D6);

  int e = k * 2 - 3;
  plg(a, (n[2] - e) / 2, n[0] * n[1], w);
  plg(w, (n[1] - e) / 2, n[0] * (n[2] - e) / 2, a);
  plg(a, (n[0] - e) / 2, (n[1] - e) * (n[2] - e) / 4, b);
}

#endif
