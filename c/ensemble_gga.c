/*  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2007-2009  CAMd
 *  Please see the accompanying LICENSE file for further information. */

#include <math.h>
#include "xc_gpaw.h"

double bee1_exchange(const xc_parameters* par,
		     double n, double rs, double a2,
		     double* dedrs, double* deda2)
{
  double e = C1 / rs;
  *dedrs = -e / rs;
  double c = C2 * rs / n;
  c *= c;
  double s2 = a2 * c;
  double kappa = 0.804;
  double x1 = MU / kappa;
  double x2 = x1 * s2;
  double Fx = 0.0;
  double dFxds2 = 0.0;
  for (int i = 0; i < par->nparameters; i++)
    {
      double x3 = x2 * par->parameters[i];
      if (x3 < 100.0)
        {
          double x4 = exp(x3);
          double x5 = 1.0 / (1.0 + x2 * x4);
          double coef = par->parameters[i + par->nparameters];
          Fx -= coef * (kappa * x5 - 1.0 - kappa);
          dFxds2 += kappa * coef * x5 * x5 * (1.0 + x3) * x1 * x4;
        }
    }
  double ds2drs = 8.0 * c * a2 / rs;
  *dedrs = *dedrs * Fx + e * dFxds2 * ds2drs;
  *deda2 = e * dFxds2 * c;
  e *= Fx;
  return e;
}
