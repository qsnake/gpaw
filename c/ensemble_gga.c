#include <math.h>
#include "xc.h"

double ensemble_exchange(const xc_parameters* par,
			 double n, double rs, double a2,
			 double* dedrs, double* deda2)
{
  double e = C1 / rs;
  *dedrs = -e / rs;
  double Fx = 1.0;
  double dFxds2 = 0.0;
  double c = C2 * rs / n;
  if (par->i == 9)
    Fx = 0.0;
  else if (par->i > 0)
    {
      c *= c;
      double s0 = par->s0;
      double s2 = a2 * c;
      double sps0 = sqrt(s2) + s0;
      double x = s2 * s0 * s0 / (sps0 * sps0);
      double y = 1.0;
      for(int i = 1; i < par->i; i++)
	y *= x;
      Fx = x * y;
      dFxds2 = par->i * s0 * s0 * s0 * y / (sps0 * sps0 * sps0);
    }
  double ds2drs = 8.0 * c * a2 / rs;
  *dedrs = *dedrs * Fx + e * dFxds2 * ds2drs;
  *deda2 = e * dFxds2 * c;
  e *= Fx;
  return e;
}


double pade_exchange(const xc_parameters* par,
		     double n, double rs, double a2,
		     double* dedrs, double* deda2)
{
  double e = C1 / rs;
  *dedrs = -e / rs;
  double c = C2 * rs / n;
  c *= c;
  double s2 = a2 * c;
  const double* p = par->pade;
  double Fx = p[0] + p[1] * s2;
  double dFxds2 = p[1];
  double ds2drs = 8.0 * c * a2 / rs;
  *dedrs = *dedrs * Fx + e * dFxds2 * ds2drs;
  *deda2 = e * dFxds2 * c;
  e *= Fx;
  return e;
}
