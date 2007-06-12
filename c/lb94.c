#include <math.h>
#include "xc.h"
#include <stdio.h>

/* Correction to LDA to resemble asymptotic -1/r potential               */
/* from [LB94] = van Leeuwen and Baerends, Phys.Rev.A vol 49 (1994) 2421 */
double lb94_correction(const xc_parameters* par,
		       double n,   /* I: density                  */
		       double rs,  /* I: rs = (3/(4 \pi n))^(1/3) */
 		       double a2,  /* I: |\nabla n|^2             */
		       double* dedrs, /* not touched */
		       double* deda2) /* not touched */
{
  /* (4*pi/3.)**(1./3.) = 1.6119919540164696 */
  double rhom13 = 1.6119919540164696 * rs; /* n^(-1/3) */
  double x = sqrt(a2) * (rhom13*rhom13*rhom13*rhom13);
  // printf("<lb94_correction> n=%g,rs=%g,a2=%g,x=%g\n",n,rs,a2,x);
  /* constant as given in [LB94] */
  double beta = 0.05;
  double corr = -beta* x*x / ( rhom13 * ( 1. + 3.*beta*x*asinh(x) ) );
  //  printf("<lb94_correction> rho1/3=%g,corr=%g\n",1./rhom13,corr);
  return corr;
}
