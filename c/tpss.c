#include <math.h>
#include "xc_gpaw.h"
#define cons 0.0*2.87123400019 /* a*3*(3*Pi^2)^(2/3)/10 = a*Cf */ 
#define p 1.66666666667     /* 5/3 */
#define cons2 1.0*8.  /* b*8 constant in Tw */

void tpssfx(double *n, double *g, double *t, double *fxu, double *dfxudn,
	       double *dfxudg, double *dfxudtau, int iexc);
void tpssfc(double *nu, double *nd, double *guu, double *gdd, double *gud,
	    double *tu, double *td, double *fc, double *dfxudnu, 
	    double *dfxudnd, double *dfxudguu, double *dfxudgdd,
	    double *dfxudgud, double *dfxudtu,double *dfxudtd);


/* Local part of tpss */
/* for testing self-concistency and convergence called with local */
/* local  tau variable */
/* tau = a*tf + b*tW */
/* other existing functionals proposed are a=1 and b=1, 1/5 and 1/9 */ 
/* use with PBE setups, for atom H a=0.0 and b=1.0 is exact (default) */

double atpss_exchange(double n, double a2, double tau, 
		      double* dexdn, double* deda2, double* dedtaua)
{
  double e,dfxdn,dfxdg,dfxdt,dedn;
  double dtdn = - a2 / (cons2 * n * n) + 2 * cons * p * pow(n,p-1.)* pow(0.5,p); 
  double dtdg = 1. / cons2 * n;
  if (tau == -1.0)
    tau = 2* (a2 / (cons2 * n *2) + cons * pow(n,p) * pow(0.5,p));
  tpssfx(&n,&a2,&tau,&e,&dfxdn,&dfxdg,&dfxdt,0);
  dedn = dfxdn + dfxdt * dtdn;
  *dexdn = dedn ;
  *deda2 = dfxdg + dfxdt*dtdg;
  return e;
}

double atpss_correlation(double na, double  nb, double aa2, double ab2,
			 double a2, double ta, double tb, 
			 bool spinpol, double* decdna, 
			 double* decdnb, double* decdaa2, double* decdab2, 
			 double* decdgab,double* dedtaua,double* dedtaub)
{    
  double e;
  double dfcdna, dfcdgaa, dfcdta;
  double dfcdnb, dfcdgbb, dfcdtb;
  double dfcdgab;
  
  double gaa = aa2 ;
  double gbb = ab2;
  double gab = (a2 - aa2 - ab2) / 2.;
  if (ta == -1.0)
    {
      ta = gaa / (cons2 * na) + cons * pow(na,p);
      tb = gbb / (cons2 * nb) + cons * pow(nb,p);
    }
  double dtdna = - gaa / (cons2 * na * na) + cons * p * pow(na,p-1.);
  double dtdnb = - gbb / (cons2 * nb * nb) + cons * p * pow(nb,p-1.);
  double dtdgaa = 1. / cons2 * na;
  double dtdgbb = 1. / cons2 * nb;


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

double tpss_exchange(double n, double a2, double tau, 
		     double* dexdn, double* deda2, double* dedtaua)
{
  double e,dfxdn,dfxdg,dfxdt,dedn;

  tpssfx(&n,&a2,&tau,&e,&dfxdn,&dfxdg,&dfxdt,0);
  dedn = dfxdn ;
  *dexdn = dedn ;
  *deda2 = dfxdg ;
  *dedtaua = dfxdt;
  return e;
}

double tpss_correlation(double na, double  nb, double aa2, double ab2,
			 double a2, double ta, double tb, 
			 bool spinpol, double* decdna, 
			 double* decdnb, double* decdaa2, double* decdab2, 
			double* decdgab,double* dedtaua,double* dedtaub)
{    
  double e;
  double dfcdna, dfcdgaa, dfcdta;
  double dfcdnb, dfcdgbb, dfcdtb;
  double dfcdgab;
  
  double gaa = aa2 ;
  double gbb = ab2;
  double gab = (a2 - aa2 - ab2) / 2.;

 if (spinpol == 0)
   {
     tpssfc(&na,&na,&gaa,&gaa,&gaa,&ta,&ta,&e,&dfcdna, &dfcdnb, 
	    &dfcdgaa, &dfcdgbb, &dfcdgab,&dfcdta,&dfcdtb);
     *decdna = dfcdna ;
     *decdaa2 =  dfcdgaa ;
     *decdgab = dfcdgab ;
     *dedtaua = dfcdta ;
   }
 else
   {
     tpssfc(&na,&nb,&gaa,&gbb,&gab,&ta,&tb,&e,&dfcdna, &dfcdnb, 
	    &dfcdgaa, &dfcdgbb, &dfcdgab,&dfcdta,&dfcdtb);
     *decdna = dfcdna ;
     *decdnb = dfcdnb ;
     *decdaa2 = dfcdgaa ;
     *decdab2 = dfcdgbb ;
     *decdgab = dfcdgab ;
     *dedtaua = dfcdta ;
     *dedtaub = dfcdtb ;
   }
 return e;
}
