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

//Computer generated code! Hands off!
    
// inserts values of f(r) r^l Y_lm(theta, phi) in elements of input array 'a'
void bmgs_radial3(const bmgsspline* spline, int m, 
		  const int n[3], 
		  const double C[3],
		  const double h[3],
		  const double* f, double* a)
{
  int l = spline->l;
  if (l == 0)
    for (int q = 0; q < n[0] * n[1] * n[2]; q++)
      a[q] = 0.28209479177387814 * f[q];
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
                  if (m == -1)
                    a[q] = f[q] * 0.488602511903 * y;
                  else if (m == 0)
                    a[q] = f[q] * 0.488602511903 * z;
                  else
                    a[q] = f[q] * 0.488602511903 * x;
                  z += h[2];
		}
	      y += h[1];
	    }
	  x += h[0];
	}
    }
  else if (l == 2)
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
                  double r2 = x*x+y*y+z*z;
                  if (m == -2)
                    a[q] = f[q] * 1.09254843059 * x*y;
                  else if (m == -1)
                    a[q] = f[q] * 1.09254843059 * y*z;
                  else if (m == 0)
                    a[q] = f[q] * 0.315391565253 * (3*z*z-r2);
                  else if (m == 1)
                    a[q] = f[q] * 1.09254843059 * x*z;
                  else
                    a[q] = f[q] * 0.546274215296 * (x*x-y*y);
                  z += h[2];
		}
	      y += h[1];
	    }
	  x += h[0];
	}
    }
  else
    assert(0 == 1);
}


// insert values of
// d( f(r) * r^l Y_l^m )                           d( r^l Y_l^m )
// --------------------- = g(r) q r^l Y_l^m + f(r) --------------
//        dq                                             dq
// where q={x, y, z} and g(r) = 1/r*(df/dr)
void bmgs_radiald3(const bmgsspline* spline, int m, int c, 
		  const int n[3], 
		  const double C[3],
		  const double h[3],
		  const double* f, const double* g, double* a)
{
  int l = spline->l;
  // x
  if (c == 0 && l == 0)
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
                    a[q] = g[q] * 0.282094791774 * x;
                  z += h[2];
		}
	      y += h[1];
	    }
	  x += h[0];
	}
    }
  else if (c == 0 && l == 1)
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
                  if (m == -1)
                    a[q] = g[q] * 0.488602511903 * x*y;
                  else if (m == 0)
                    a[q] = g[q] * 0.488602511903 * x*z;
                  else
                    a[q] = g[q] * 0.488602511903 * x*x + f[q] * 0.488602511903;
                  z += h[2];
		}
	      y += h[1];
	    }
	  x += h[0];
	}
    }
  else if (c == 0 && l == 2)
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
                  double r2 = x*x+y*y+z*z;
                  if (m == -2)
                    a[q] = g[q] * 1.09254843059 * x*x*y + f[q] * 1.09254843059 * y;
                  else if (m == -1)
                    a[q] = g[q] * 1.09254843059 * x*y*z;
                  else if (m == 0)
                    a[q] = g[q] * 0.315391565253 * (3*x*z*z-x*r2) + f[q] * 0.630783130505 * -x;
                  else if (m == 1)
                    a[q] = g[q] * 1.09254843059 * x*x*z + f[q] * 1.09254843059 * z;
                  else
                    a[q] = g[q] * 0.546274215296 * (x*x*x-x*y*y) + f[q] * 1.09254843059 * x;
                  z += h[2];
		}
	      y += h[1];
	    }
	  x += h[0];
	}
    }
  // y
  else if (c == 1 && l == 0)
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
                    a[q] = g[q] * 0.282094791774 * y;
                  z += h[2];
		}
	      y += h[1];
	    }
	  x += h[0];
	}
    }
  else if (c == 1 && l == 1)
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
                  if (m == -1)
                    a[q] = g[q] * 0.488602511903 * y*y + f[q] * 0.488602511903;
                  else if (m == 0)
                    a[q] = g[q] * 0.488602511903 * y*z;
                  else
                    a[q] = g[q] * 0.488602511903 * x*y;
                  z += h[2];
		}
	      y += h[1];
	    }
	  x += h[0];
	}
    }
  else if (c == 1 && l == 2)
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
                  double r2 = x*x+y*y+z*z;
                  if (m == -2)
                    a[q] = g[q] * 1.09254843059 * x*y*y + f[q] * 1.09254843059 * x;
                  else if (m == -1)
                    a[q] = g[q] * 1.09254843059 * y*y*z + f[q] * 1.09254843059 * z;
                  else if (m == 0)
                    a[q] = g[q] * 0.315391565253 * (-y*r2+3*y*z*z) + f[q] * 0.630783130505 * -y;
                  else if (m == 1)
                    a[q] = g[q] * 1.09254843059 * x*y*z;
                  else
                    a[q] = g[q] * 0.546274215296 * (-y*y*y+x*x*y) + f[q] * 1.09254843059 * -y;
                  z += h[2];
		}
	      y += h[1];
	    }
	  x += h[0];
	}
    }
  // z
  else if (c == 2 && l == 0)
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
                    a[q] = g[q] * 0.282094791774 * z;
                  z += h[2];
		}
	      y += h[1];
	    }
	  x += h[0];
	}
    }
  else if (c == 2 && l == 1)
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
                  if (m == -1)
                    a[q] = g[q] * 0.488602511903 * y*z;
                  else if (m == 0)
                    a[q] = g[q] * 0.488602511903 * z*z + f[q] * 0.488602511903;
                  else
                    a[q] = g[q] * 0.488602511903 * x*z;
                  z += h[2];
		}
	      y += h[1];
	    }
	  x += h[0];
	}
    }
  else if (c == 2 && l == 2)
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
                  double r2 = x*x+y*y+z*z;
                  if (m == -2)
                    a[q] = g[q] * 1.09254843059 * x*y*z;
                  else if (m == -1)
                    a[q] = g[q] * 1.09254843059 * y*z*z + f[q] * 1.09254843059 * y;
                  else if (m == 0)
                    a[q] = g[q] * 0.315391565253 * (3*z*z*z-z*r2) + f[q] * 1.26156626101 * z;
                  else if (m == 1)
                    a[q] = g[q] * 1.09254843059 * x*z*z + f[q] * 1.09254843059 * x;
                  else
                    a[q] = g[q] * 0.546274215296 * (x*x*z-y*y*z);
                  z += h[2];
		}
	      y += h[1];
	    }
	  x += h[0];
	}
    }
  else
      assert(0 == 1);
}

