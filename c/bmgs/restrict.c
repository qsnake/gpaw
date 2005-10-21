#include "bmgs.h"

#ifdef K

void RST1D(const T* a, int n, int m, T* b)
{
  a += K * P / 2 - 1;
  for (int j = 0; j < m; j++)
    {
      T* c = b;
      for (int i = 0; i < n; i++)
	{
#if defined(BMGSCOMPLEX) && defined(NO_C99_COMPLEX)
	  if      (K == 2 && P == 2)
	    {
	      c[0].r = 0.5 * (a[0].r + 
			      0.5 * (a[1].r + a[-1].r));
	      c[0].i = 0.5 * (a[0].i + 
			      0.5 * (a[1].i + a[-1].i));
	  }
	  else if (K == 2 && P == 5)
	    {
	      c[0].r = 0.2 * (a[0].r +
			      0.8 * (a[-1].r + a[1].r) +
			      0.6 * (a[-2].r + a[2].r) +
			      0.4 * (a[-3].r + a[3].r) +
			      0.2 * (a[-4].r + a[4].r));
	      c[0].i = 0.2 * (a[0].i +
			      0.8 * (a[-1].i + a[1].i) +
			      0.6 * (a[-2].i + a[2].i) +
			      0.4 * (a[-3].i + a[3].i) +
			      0.2 * (a[-4].i + a[4].i));
	    }
	  else if (K == 4 && P == 2)
	    {
	      c[0].r = 0.5 * (a[0].r + 
			      0.5625 * (a[1].r + a[-1].r) +
			      -0.0625 * (a[3].r + a[-3].r));
	      c[0].i = 0.5 * (a[0].i + 
			      0.5625 * (a[1].i + a[-1].i) +
			      -0.0625 * (a[3].i + a[-3].i));
	    }
	  else if (K == 4 && P == 5)
	    {
	      c[0].r = 0.2 * (a[0].r +
			      0.864 * (a[-1].r + a[1].r) +
			      0.672 * (a[-2].r + a[2].r) +
			      0.448 * (a[-3].r + a[3].r) +
			      0.216 * (a[-4].r + a[4].r) +
			      -0.048 * (a[-6].r + a[6].r) +
			      -0.064 * (a[-7].r + a[7].r) +
			      -0.056 * (a[-8].r + a[8].r) +
			      -0.032 * (a[-9].r + a[9].r));
	      c[0].i = 0.2 * (a[0].i +
			      0.864 * (a[-1].i + a[1].i) +
			      0.672 * (a[-2].i + a[2].i) +
			     0.448 * (a[-3].i + a[3].i) +
			      0.216 * (a[-4].i + a[4].i) +
			      -0.048 * (a[-6].i + a[6].i) +
			      -0.064 * (a[-7].i + a[7].i) +
			      -0.056 * (a[-8].i + a[8].i) +
			      -0.032 * (a[-9].i + a[9].i));
	    }
	  else if (K == 6 && P == 2)
	    {
	      c[0].r = 0.5 * (a[0].r + 
			      0.58593750 * (a[1].r + a[-1].r) +
			      -0.09765625 * (a[3].r + a[-3].r) +
			      0.01171875 * (a[5].r + a[-5].r));
	      c[0].i = 0.5 * (a[0].i + 
			      0.58593750 * (a[1].i + a[-1].i) +
			      -0.09765625 * (a[3].i + a[-3].i) +
			      0.01171875 * (a[5].i + a[-5].i));
	    }
	  else if (K == 6 && P == 5)
	    {
	      c[0].r = 0.2 * (a[0].r +
			      0.887040 * (a[1].r + a[-1].r) +
			      0.698880 * (a[2].r + a[-2].r) +
			      0.465920 * (a[3].r + a[-3].r) +
			      0.221760 * (a[4].r + a[-4].r) +
			      -0.073920 * (a[6].r + a[-6].r) +
			      -0.099840 * (a[7].r + a[-7].r) +
			      -0.087360 * (a[8].r + a[-8].r) +
			      -0.049280 * (a[9].r + a[-9].r) + 
			      0.008064 * (a[11].r + a[-11].r) +
			      0.011648 * (a[12].r + a[-12].r) +
			      0.010752 * (a[13].r + a[-13].r) +
			      0.006336 * (a[14].r + a[-14].r));
	      c[0].i = 0.2 * (a[0].i +
			      0.887040 * (a[1].i + a[-1].i) +
			      0.698880 * (a[2].i + a[-2].i) +
			      0.465920 * (a[3].i + a[-3].i) +
			      0.221760 * (a[4].i + a[-4].i) +
			      -0.073920 * (a[6].i + a[-6].i) +
			      -0.099840 * (a[7].i + a[-7].i) +
			      -0.087360 * (a[8].i + a[-8].i) +
			      -0.049280 * (a[9].i + a[-9].i) + 
			      0.008064 * (a[11].i + a[-11].i) +
			      0.011648 * (a[12].i + a[-12].i) +
			      0.010752 * (a[13].i + a[-13].i) +
			      0.006336 * (a[14].i + a[-14].i));
	    }
#else
	  if      (K == 2 && P == 2)
	    c[0] = 0.5 * (a[0] + 
			  0.5 * (a[1] + a[-1]));

	  else if (K == 2 && P == 5)
	    c[0] = 0.2 * (a[0] +
			  0.8 * (a[-1] + a[1]) +
			  0.6 * (a[-2] + a[2]) +
			  0.4 * (a[-3] + a[3]) +
			  0.2 * (a[-4] + a[4]));

	  else if (K == 4 && P == 2)
	    c[0] = 0.5 * (a[0] + 
			   0.5625 * (a[1] + a[-1]) +
			  -0.0625 * (a[3] + a[-3]));

	  else if (K == 4 && P == 5)
	    c[0] = 0.2 * (a[0] +
			   0.864 * (a[-1] + a[1]) +
			   0.672 * (a[-2] + a[2]) +
			   0.448 * (a[-3] + a[3]) +
			   0.216 * (a[-4] + a[4]) +
			  -0.048 * (a[-6] + a[6]) +
			  -0.064 * (a[-7] + a[7]) +
			  -0.056 * (a[-8] + a[8]) +
			  -0.032 * (a[-9] + a[9]));

	  else if (K == 6 && P == 2)
	    c[0] = 0.5 * (a[0] + 
			   0.58593750 * (a[1] + a[-1]) +
			  -0.09765625 * (a[3] + a[-3]) +
			   0.01171875 * (a[5] + a[-5]));

	  else if (K == 6 && P == 5)
	    c[0] = 0.2 * (a[0] +
			   0.887040 * (a[1] + a[-1]) +
			   0.698880 * (a[2] + a[-2]) +
			   0.465920 * (a[3] + a[-3]) +
			   0.221760 * (a[4] + a[-4]) +
			  -0.073920 * (a[6] + a[-6]) +
			  -0.099840 * (a[7] + a[-7]) +
			  -0.087360 * (a[8] + a[-8]) +
			  -0.049280 * (a[9] + a[-9]) + 
			   0.008064 * (a[11] + a[-11]) +
			   0.011648 * (a[12] + a[-12]) +
			   0.010752 * (a[13] + a[-13]) +
			   0.006336 * (a[14] + a[-14]));
#endif
	  a += P;
	  c += m;
	}
      a += K * P - P - 1;
      b++;
    }
}

#else
#  define K 2
#    define P 2
#    define RST1D Z(bmgs_restrict1D2_2)
#    include "restrict.c"
#    undef RST1D
#    undef P
#    define P 5
#    define RST1D Z(bmgs_restrict1D2_5)
#    include "restrict.c"
#    undef RST1D
#    undef P
#  undef K
#  define K 4
#    define P 2
#    define RST1D Z(bmgs_restrict1D4_2)
#    include "restrict.c"
#    undef RST1D
#    undef P
#    define P 5
#    define RST1D Z(bmgs_restrict1D4_5)
#    include "restrict.c"
#    undef RST1D
#    undef P
#  undef K
#  define K 6
#    define P 2
#    define RST1D Z(bmgs_restrict1D6_2)
#    include "restrict.c"
#    undef RST1D
#    undef P
#    define P 5
#    define RST1D Z(bmgs_restrict1D6_5)
#    include "restrict.c"
#    undef RST1D
#    undef P
#  undef K

void Z(bmgs_restrict)(int k, int p, T* a, const int n[3], T* b, T* w)
{
  void (*plg)(const T*, int, int, T*);
  if (k == 2)
    {
      if (p == 2)
	plg = Z(bmgs_restrict1D2_2);
      else
	plg = Z(bmgs_restrict1D2_5);
    }
  else if (k == 4)
    {
      if (p == 2)
	plg = Z(bmgs_restrict1D4_2);
      else
	plg = Z(bmgs_restrict1D4_5);
    }
  else
    {
      if (p == 2)
	plg = Z(bmgs_restrict1D6_2);
      else
	plg = Z(bmgs_restrict1D6_5);
    }
  int e = k * p - p - 1;
  plg(a, (n[2] - e) / p, n[0] * n[1], w);
  plg(w, (n[1] - e) / p, n[0] * (n[2] - e) / p, a);
  plg(a, (n[0] - e) / p, (n[1] - e) * (n[2] - e) / (p * p), b);
}

#endif
