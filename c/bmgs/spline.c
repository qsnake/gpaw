#include <malloc.h>
#include <math.h>
#include "bmgs.h"


static const double Ys = 0.28209479177387814;
static const double Yp = 0.48860251190291992;
static const double Yd1 = 1.0925484305920792;
static const double Yd2 = 0.54627421529603959;
static const double Yd3 = 0.94617469575756008;
static const double Yd4 = 0.31539156525252005;


bmgsspline bmgs_spline(int l, double dr, int nbins, double* f)
{
  double c = 3.0 / (dr * dr);
  double* f2 = (double*)malloc((nbins + 1) * sizeof(double));
  double* u = (double*)malloc(nbins * sizeof(double));
  f2[0] = -0.5;
  u[0] = (f[1] - f[0]) * c;
  for (int b = 1; b < nbins; b++)
    {
      double p = 0.5 * f2[b - 1] + 2.0;
      f2[b] = -0.5 / p;
      u[b] = ((f[b + 1] - 2.0 * f[b] + f[b - 1]) * c - 0.5 * u[b - 1]) / p;
    }
  f2[nbins] = ((f[nbins - 1] * c - 0.5 * u[nbins - 1]) /
               (0.5 * f2[nbins - 1] + 1.0));
  for (int b = nbins - 1; b >= 0; b--)
    f2[b] = f2[b] * f2[b + 1] + u[b];
  double* data = (double*)malloc(4 * (nbins + 1) * sizeof(double));
  bmgsspline spline = {l, dr, nbins, data};
  for (int b = 0; b < nbins; b++)
    {
      *data++ = f[b];
      *data++ = (f[b + 1] - f[b]) / dr - (f2[b] / 3 + f2[b + 1] / 6) * dr;
      *data++ = 0.5 * f2[b];
      *data++ = (f2[b + 1] - f2[b]) / (6 * dr);
    }
  data[0] = 0.0;
  data[1] = 0.0;
  data[2] = 0.0;
  data[3] = 0.0;
  free(u);
  free(f2);
  return spline;
}


double bmgs_splinevalue(const bmgsspline* spline, double r)
{
  int b = r / spline->dr;
  if (b >= spline->nbins)
    return 0.0;
  double u = r - b * spline->dr;
  double* s = spline->data + 4 * b;
  return  s[0] + u * (s[1] + u * (s[2] + u * s[3]));
}


void bmgs_deletespline(bmgsspline* spline)
{
  free(spline->data);
}


void bmgs_radial1(const bmgsspline* spline, 
		  const int n[3], const double C[3],
		  const double h[3],
		  int* b, double* d)
{
  int nbins = spline->nbins;
  double dr = spline->dr;
  double x = C[0];
  for (int i0 = 0; i0 < n[0]; i0++)
    {
      double xx = x * x;
      double y = C[1];
      for (int i1 = 0; i1 < n[1]; i1++)
	{
	  double xxpyy = xx + y * y;
	  double z = C[2];
	  for (int i2 = 0; i2 < n[2]; i2++)
	    {
	      double r = sqrt(xxpyy + z * z);
	      int j = r / dr;
	      if (j < nbins)
		{
		  *b++ = j;
		  *d++ = r - j * dr;
		}
	      else
		{
		  *b++ = nbins;
		  *d++ = 0.0;
		}
	      z += h[2];
	    }
	  y += h[1];
	}
      x += h[0];
    }
}


void bmgs_radial2(const bmgsspline* spline, const int n[3],
		  const int* b, const double* d, 
		  double* f, double* g)
{
  double dr = spline->dr;
  for (int q = 0; q < n[0] * n[1] * n[2]; q++)
    {
      int j = b[q];
      const double* s = spline->data + 4 * j;
      double u = d[q];
      f[q] = s[0] + u * (s[1] + u * (s[2] + u * s[3]));
      if (g != 0)
	{
	  if (j == 0)
	    g[q] = 2.0 * s[2] + u * 3.0 * s[3];
	  else
	    g[q] = (s[1] + u * (2.0 * s[2] + u * 3.0 * s[3])) / (j * dr + u);
	}
    }
}


void bmgs_radial3(const bmgsspline* spline, int m, 
		  const int n[3], 
		  const double C[3],
		  const double h[3],
		  const double* f, double* a)
{
  int l = spline->l;
  if (l == 0)
    for (int q = 0; q < n[0] * n[1] * n[2]; q++)
      a[q] = Ys * f[q];
  else if (l == 1)
    {
      int q = 0;
      double x = C[0];
      for (int i0 = 0; i0 < n[0]; i0++)
	{
	  double y = C[1];
	  for (int i1 = 0; i1 < n[1]; i1++)
	    {
	      double z = C[2];
	      for (int i2 = 0; i2 < n[2]; i2++, q++)
		{
		  if (m == 0)
		    a[q] = Yp * x * f[q];
		  else if (m == 1)
		    a[q] = Yp * y * f[q];
		  else
		    a[q] = Yp * z * f[q];
		  z += h[2];
		}
	      y += h[1];
	    }
	  x += h[0];
	}
    }
  else
    {
      int q = 0;
      double x = C[0];
      for (int i0 = 0; i0 < n[0]; i0++)
	{
	  double y = C[1];
	  for (int i1 = 0; i1 < n[1]; i1++)
	    {
	      double z = C[2];
	      for (int i2 = 0; i2 < n[2]; i2++, q++)
		{
		  if (m == 0)
		    a[q] = Yd1 * x * y * f[q];
		  else if (m == 1)
		    a[q] = Yd1 * y * z * f[q];
		  else if (m == 2)
		    a[q] = Yd1 * x * z * f[q];
		  else if (m == 3)
		    a[q] = f[q] * Yd2 * (x * x - y * y);
		  else
		    a[q] = f[q] * (Yd3 * z * z - 
				   Yd4 * (x * x + y * y + z * z));
		  z += h[2];
		}
	      y += h[1];
	    }
	  x += h[0];
	}
    }
}

void bmgs_radiald3(const bmgsspline* spline, int m, 
		  const int n[3], 
		  const double C[3],
		  const double h[3],
		  const double* f, const double* g, double* a)
{
  int q = 0;
  int l = spline->l;
  if (l == 0)
    {
      double x = C[0];
      for (int i0 = 0; i0 < n[0]; i0++)
	{
	  double y = C[1];
	  for (int i1 = 0; i1 < n[1]; i1++)
	    {
	      double z = C[2];
	      for (int i2 = 0; i2 < n[2]; i2++, q++)
		{
		  if (m == 0)
		    a[q] = Ys * x * g[q];
		  else if (m == 1)
		    a[q] = Ys * y * g[q];
		  else
		    a[q] = Ys * z * g[q];
		  z += h[2];
		}
	      y += h[1];
	    }
	  x += h[0];
	}
    }
  else if (l == 1)
    {
      double x = C[0];
      for (int i0 = 0; i0 < n[0]; i0++)
	{
	  double y = C[1];
	  for (int i1 = 0; i1 < n[1]; i1++)
	    {
	      double z = C[2];
	      for (int i2 = 0; i2 < n[2]; i2++, q++)
		{
		  if (m == 0)
		    a[q] = Yp * (g[q] * x * x + f[q]);
		  else if (m == 1)
		    a[q] = g[q] * Yp * x * y;
		  else if (m == 2)
		    a[q] = g[q] * Yp * x * z;
		  else if (m == 3)
		    a[q] = Yp * (g[q] * y * y + f[q]);
		  else if (m == 4)
		    a[q] = g[q] * Yp * y * z;
		  else
		    a[q] = Yp * (g[q] * z * z + f[q]);
		  z += h[2];
		}
	      y += h[1];
	    }
	  x += h[0];
	}
    }
  else
    {
      double x = C[0];
      for (int i0 = 0; i0 < n[0]; i0++)
	{
	  double y = C[1];
	  for (int i1 = 0; i1 < n[1]; i1++)
	    {
	      double z = C[2];
	      for (int i2 = 0; i2 < n[2]; i2++, q++)
		{
		  double r2 = x * x + y * y + z * z;
		  // x:
 		  if (m == 0)
		    a[q] = Yd1 * (g[q] * x * y * x + f[q] * y);
		  else if (m == 1)
		    a[q] = Yd1 * g[q] * y * z * x;
		  else if (m == 2)
		    a[q] = Yd1 * (g[q] * z * x * x + f[q] * z);
		  else if (m == 3)
		    a[q] = Yd2 * (g[q] * (x * x - y * y) * x 
				  + f[q] * 2.0 * x);
		  else if (m == 4)
		    a[q] = (Yd3 * g[q] * z * z * x 
			    - Yd4 * (g[q] * r2 * x + f[q] * 2.0 * x));
		  // y:
		  else if (m == 5)
		    a[q] = Yd1 * (g[q] * x * y * y + f[q] * x);
		  else if (m == 6)
		    a[q] = Yd1 * (g[q] * y * z * y + f[q] * z);
		  // skipping xyz
		  else if (m == 7)
		    a[q] = Yd2 * (g[q] * (x * x - y * y) * y 
				  - f[q] * 2.0 * y);
		  else if (m == 8)
		    a[q] = (Yd3 * g[q] * z * z * y 
			    - Yd4 * (g[q] * r2 * y + f[q] * 2.0 * y));
		  // z:
		  // skipping xyz
		  else if (m == 9)
		    a[q] = Yd1 * (g[q] * y * z * z + f[q] * y);
		  else if (m == 10)
		    a[q] = Yd1 * (g[q] * x * z * z + f[q] * x);
		  else if (m == 11)
		    a[q] = Yd2 * g[q] * (x * x - y * y) * z;
		  else
		    a[q] = (Yd3 * g[q] * z * z * z 
			    - Yd4 * (g[q] * r2 * z - f[q] * 4.0 * z));
		  z += h[2];
		}
	      y += h[1];
	    }
	  x += h[0];
	}
    }
}
