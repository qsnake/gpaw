#include <math.h>
#include "xc.h"

double rpbe_exchange(const xc_parameters* par,
		     double n, double rs, double a2,
                     double* dedrs, double* deda2)
{
  double e = C1 / rs;
  if (par->rel)
    {
      double betaR = 1.0 / (rs * 71.4);
      double eta = sqrt(1.0 + betaR * betaR);
      double ksi = eta + betaR;
      double x = (betaR * eta - log(ksi)) / (betaR * betaR);
      double phi = 1.0 - 1.5 * x * x;
      double dphidbetaR = 3.0 * x / betaR * (2.0 * x - 
                                             (eta + betaR * betaR / eta -
                                              (1.0 + betaR / eta) / ksi) / 
                                             betaR);
      *dedrs = -e / rs * phi - e * betaR / rs * dphidbetaR;
      e *= phi;
    }
  else
    *dedrs = -e / rs;
  if (par->gga)
    {
      double c = C2 * rs / n;
      c *= c;
      double s2 = a2 * c;
      double x = exp(-MU * s2 / 0.804);
      double Fx = 1.0 + 0.804 * (1 - x);
      double dFxds2 = MU * x;
      double ds2drs = 8.0 * c * a2 / rs;
      *dedrs = *dedrs * Fx + e * dFxds2 * ds2drs;
      *deda2 = e * dFxds2 * c;
      e *= Fx;
    }
  return e;
}
