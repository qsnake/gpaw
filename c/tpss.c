#include <math.h>
#include "xc.h"
#define cons  2.87123400019 /* 3*(3*Pi^2)^(2/3)/10 = Cf*/
#define p 1.66666666667     /* 5/3 */
#define cons2 24.  /* 8 *3 constant in 1/3 Tw */

void tpssfx(double *n, double *g, double *t, double *fxu, double *dfxudn,
	       double *dfxudg, double *dfxudtau, int iexc);
void tpssfc(double *nu, double *nd, double *guu, double *gdd, double *gud,
	    double *tu, double *td, double *fc, double *dfxudnu, 
	    double *dfxudnd, double *dfxudguu, double *dfxudgdd,
	    double *dfxudgud, double *dfxudtu,double *dfxudtd);


/* Local part of tpss */
/* for testing self-concistency called with local tau variable */
/* tau = tf + 1/3 tW */
/* other existing functionals proposed are 1, 1/5 and 1/9 */ 
/* use with PBE setups and do not apply to atom H alone */

double atpss_exchange(double n, double a2, double* dexdn, double* deda2)
{
  double e,tau,dfxdn,dfxdg,dfxdt,dedn;
  double dtdn = - a2 / (cons2 * n * n) + cons * p * pow(n,p-1.);
  double dtdg = 1. / cons2 * n;
  tau = a2 / (cons2 * n) + cons * pow(n,p);
  tpssfx(&n,&a2,&tau,&e,&dfxdn,&dfxdg,&dfxdt,0);
  dedn = dfxdn + dfxdt * dtdn;
  *dexdn = dedn ;
  *deda2 = dfxdg + dfxdt*dtdg;
  return e;
}

double atpss_correlation(double na, double  nb, double aa2, double ab2,
			 double a2, bool spinpol, double* decdna, 
			 double* decdnb, double* decdaa2, double* decdab2, 
			 double* decdgab)
{    
  double e;
  double dfcdna, dfcdgaa, dfcdta;
  double dfcdnb, dfcdgbb, dfcdtb;
  double dfcdgab;
  
  double gaa = aa2 ;
  double gbb = ab2;
  double gab = (a2 - aa2 - ab2) / 2.;
  double ta = gaa / (cons2 * na) + cons * pow(na,p);
  double tb = gbb / (cons2 * nb) + cons * pow(nb,p);
  double dtdna = - gaa / (cons2 * na * na) + cons * p * pow(na,p-1.);
  double dtdnb = - gbb / (cons2 * nb * nb) + cons * p * pow(nb,p-1.);
  double dtdgaa = 1. / cons2 * na;
  double dtdgbb = 1. / cons2 * na;


 if (spinpol == 0)
   {
     tpssfc(&na,&na,&gaa,&gaa,&gaa,&ta,&ta,&e,&dfcdna, &dfcdnb, 
	    &dfcdgaa, &dfcdgbb, &dfcdgab,&dfcdta,&dfcdtb);
     *decdna = dfcdna + dfcdta * dtdna;
     *decdaa2 =  dfcdgaa + dfcdta * dtdgaa;
     *decdgab = dfcdgab ;

   }
 else
   {
     tpssfc(&na,&nb,&gaa,&gbb,&gab,&ta,&tb,&e,&dfcdna, &dfcdnb, 
	    &dfcdgaa, &dfcdgbb, &dfcdgab,&dfcdta,&dfcdtb);
     *decdna = dfcdna + dfcdta * dtdna;
     *decdnb = dfcdnb + dfcdtb * dtdnb;
     *decdaa2 = dfcdgaa + dfcdta * dtdgaa ;
     *decdab2 = dfcdgbb + dfcdtb * dtdgbb ;
     *decdgab = dfcdgab ;
   }
 return e;
}
