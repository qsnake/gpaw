#include "bmgs.h"

#ifdef GPAW_OMP
  #include <omp.h>
#endif

#ifdef K

void RST1D(const T* a, int n, int m, T* b)
{
  a += K - 1;

  #ifdef GPAW_OMP
    #pragma omp parallel for
  #endif
  for (int j = 0; j < m; j++)
    {
      const T* aa = a + j * (n * 2 + K * 2 - 3);
      T* bb = b + j;

      for (int i = 0; i < n; i++)
        {
        #if defined(BMGSCOMPLEX) && defined(NO_C99_COMPLEX)
          if(K==2)
            {
              bb[0].r=0.5*(aa[0].r+0.5*(aa[1].r+aa[-1].r));
              bb[0].i=0.5*(aa[0].i+0.5*(aa[1].i+aa[-1].i));
            }
          else if(K==4)
            {
              bb[0].r=0.5*(aa[0].r+0.5625*(aa[1].r+aa[-1].r)+-0.0625*(aa[3].r+aa[-3].r));
              bb[0].i=0.5*(aa[0].i+0.5625*(aa[1].i+aa[-1].i)+-0.0625*(aa[3].i+aa[-3].i));
            }
          else if(K==6)
            {
              bb[0].r=0.5*(aa[0].r+0.58593750*(aa[1].r+aa[-1].r)+-0.09765625*(aa[3].r+aa[-3].r)+0.01171875*(aa[5].r+aa[-5].r));
              bb[0].i=0.5*(aa[0].i+0.58593750*(aa[1].i+aa[-1].i)+-0.09765625*(aa[3].i+aa[-3].i)+0.01171875*(aa[5].i+aa[-5].i));
            }
        #else
          if      (K == 2)
            bb[0] = 0.5 * (aa[0] +
              0.5 * (aa[1] + aa[-1]));

          else if (K == 4)
            bb[0] = 0.5 * (aa[0] +
               0.5625 * (aa[1] + aa[-1]) +
              -0.0625 * (aa[3] + aa[-3]));

          else if (K == 6)
            bb[0] = 0.5 * (aa[0] +
               0.58593750 * (aa[1] + aa[-1]) +
              -0.09765625 * (aa[3] + aa[-3]) +
               0.01171875 * (aa[5] + aa[-5]));
          #endif
          aa += 2;
          bb += m;
        }
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
