#include <Python.h>
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "extensions.h"
#include <math.h>


// returns the distance between two 3d double vectors
double distance(double *a, double *b)
{
  double sum = 0;
  double diff;
  for (int c = 0; c < 3; c++) {
    diff = a[c] - b[c];
    sum += diff*diff;
  }
  return sqrt(sum);
} 

/* elementwise multiply and add result to another vector
 *
 * c[i] += a[i] * b[i] ,  for i = every element in the vectors
 */
PyObject* elementwise_multiply_add(PyObject *self, PyObject *args)
{
  PyArrayObject* aa;
  PyArrayObject* bb;
  PyArrayObject* cc;
  if (!PyArg_ParseTuple(args, "OOO", &aa, &bb, &cc)) 
    return NULL;
  const double* const a = DOUBLEP(aa);
  const double* const b = DOUBLEP(bb);
  double* const c = DOUBLEP(cc);
  int n = 1;
  for (int d = 0; d < aa->nd; d++)
    n *= aa->dimensions[d];
  for (int i = 0; i < n; i++)
    {
      c[i] += a[i] * b[i];
    }
  Py_RETURN_NONE;
}

/* vdot
 * 
 * If a and b are input vectors,
 * a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + ...
 * is returned.
 */
PyObject* utilities_vdot(PyObject *self, PyObject *args)
{
  PyArrayObject* aa;
  PyArrayObject* bb;
  if (!PyArg_ParseTuple(args, "OO", &aa, &bb)) 
    return NULL;
  const double* const a = DOUBLEP(aa);
  const double* const b = DOUBLEP(bb);
  double sum = 0.0;
  int n = 1;
  for (int d = 0; d < aa->nd; d++)
    n *= aa->dimensions[d];
  for (int i = 0; i < n; i++)
    {
      sum += a[i] * b[i];
    }
  return PyFloat_FromDouble(sum);
}

/* vdot
 * 
 * If a is the input vector,
 * a[0]*a[0] + a[1]*a[1] + a[2]*a[2] + ...
 * is returned.
 */
PyObject* utilities_vdot_self(PyObject *self, PyObject *args)
{
  PyArrayObject* aa;
  if (!PyArg_ParseTuple(args, "O", &aa)) 
    return NULL;
  const double* const a = DOUBLEP(aa);
  double sum = 0.0;
  int n = 1;
  for (int d = 0; d < aa->nd; d++)
    n *= aa->dimensions[d];
  for (int i = 0; i < n; i++)
    {
      sum += a[i] * a[i];
    }
  return PyFloat_FromDouble(sum);
}

PyObject* errorfunction(PyObject *self, PyObject *args)
{
  double x;
  if (!PyArg_ParseTuple(args, "d", &x)) 
    return NULL;

  return Py_BuildValue("d", erf(x));
}

PyObject* unpack(PyObject *self, PyObject *args)
{
  PyArrayObject* ap;
  PyArrayObject* a;
  if (!PyArg_ParseTuple(args, "OO", &ap, &a)) 
    return NULL;
  int n = a->dimensions[0];
  double* datap = DOUBLEP(ap);
  double* data = DOUBLEP(a);
  for (int r = 0; r < n; r++)
    for (int c = r; c < n; c++)
      {
        double d = *datap++;
        data[c + r * n] = d;
        data[r + c * n] = d;
      }
  Py_RETURN_NONE;
}

PyObject* unpack_complex(PyObject *self, PyObject *args)
{
  PyArrayObject* ap;
  PyArrayObject* a;
  if (!PyArg_ParseTuple(args, "OO", &ap, &a)) 
    return NULL;
  int n = a->dimensions[0];
  double_complex* datap = COMPLEXP(ap);
  double_complex* data = COMPLEXP(a);
  for (int r = 0; r < n; r++)
    for (int c = r; c < n; c++)
      {
        double_complex d = *datap++;
#if defined(NO_C99_COMPLEX)	
	data[c + r * n].r = d.r;
	data[c + r * n].i = d.i;
        data[r + c * n].r = d.r;
        data[r + c * n].i = -d.i;
#else
        data[c + r * n] = d;
        data[r + c * n] = conj(d);
#endif
      }
  Py_RETURN_NONE;
}

PyObject* hartree(PyObject *self, PyObject *args)
{
  int l;
  PyArrayObject* nrdr_array;
  double b;
  int N;
  PyArrayObject* vr_array;
  if (!PyArg_ParseTuple(args, "iOdiO", &l, &nrdr_array, &b, &N, &vr_array)) 
    return NULL;
  const int M = nrdr_array->dimensions[0];
  const double* nrdr = DOUBLEP(nrdr_array);
  double* vr = DOUBLEP(vr_array);
  double p = 0.0;
  double q = 0.0;
  for (int g = M - 1; g > 0; g--)
    {
      double r = b * g / (N - g);
      double rl = pow(r, l);
      double dp = nrdr[g] / rl;
      double rlp1 = rl * r;
      double dq = nrdr[g] * rlp1;
      vr[g] = (p + 0.5 * dp) * rlp1 - (q + 0.5 * dq) / rl;
      p += dp;
      q += dq;
    }
  vr[0] = 0.0;
  double f = 4.0 * M_PI / (2 * l + 1);
  for (int g = 1; g < M; g++)
    {
      double r = b * g / (N - g);
      vr[g] = f * (vr[g] + q / pow(r, l));
    }
  Py_RETURN_NONE;
}

PyObject* localize(PyObject *self, PyObject *args)
{
  PyArrayObject* Z_nnc;
  PyArrayObject* U_nn;
  if (!PyArg_ParseTuple(args, "OO", &Z_nnc, &U_nn)) 
    return NULL;

#if defined(NO_C99_COMPLEX)	
  Py_RETURN_NONE;
#else

  int n = U_nn->dimensions[0];
  double complex (*Z)[n][3] = (double complex (*)[n][3])COMPLEXP(Z_nnc);
  double (*U)[n] = (double (*)[n])DOUBLEP(U_nn);

  double value = 0.0;
  for (int a = 0; a < n; a++)
    {
      for (int b = a + 1; b < n; b++)
	{
	  double complex* Zaa = Z[a][a];
	  double complex* Zab = Z[a][b];
	  double complex* Zbb = Z[b][b];
	  double x = 0.0;
	  double y = 0.0;
	  for (int c = 0; c < 3; c++)
	    {
	      x += (0.25 * creal(Zbb[c] * conj(Zbb[c])) +
		    0.25 * creal(Zaa[c] * conj(Zaa[c])) -
		    0.5 * creal(Zaa[c] * conj(Zbb[c])) -
		    creal(Zab[c] * conj(Zab[c])));
	      y += creal((Zaa[c] - Zbb[c]) * conj(Zab[c]));
	    }
	  double t = 0.25 * atan2(y, x);
	  double C = cos(t);
	  double S = sin(t);
	  for (int i = 0; i < a; i++)
	    for (int c = 0; c < 3; c++)
	      {
		double complex Ziac = Z[i][a][c];
		Z[i][a][c] = C * Ziac + S * Z[i][b][c];
		Z[i][b][c] = C * Z[i][b][c] - S * Ziac;
	      }
	  for (int c = 0; c < 3; c++)
	    {
	      double complex Zaac = Zaa[c];
	      double complex Zabc = Zab[c];
	      double complex Zbbc = Zbb[c];
	      Zaa[c] = C * C * Zaac + 2 * C * S * Zabc + S * S * Zbbc;
	      Zbb[c] = C * C * Zbbc - 2 * C * S * Zabc + S * S * Zaac;
	      Zab[c] = S * C * (Zbbc - Zaac) + (C * C - S * S) * Zabc;
	    }
	  for (int i = a + 1; i < b; i++)
	    for (int c = 0; c < 3; c++)
	      {
		double complex Zaic = Z[a][i][c];
		Z[a][i][c] = C * Zaic + S * Z[i][b][c];
		Z[i][b][c] = C * Z[i][b][c] - S * Zaic;
	      }
	  for (int i = b + 1; i < n; i++)
	    for (int c = 0; c < 3; c++)
	      {
		double complex Zaic = Z[a][i][c];
		Z[a][i][c] = C * Zaic + S * Z[b][i][c];
		Z[b][i][c] = C * Z[b][i][c] - S * Zaic;
	      }
	  for (int i = 0; i < n; i++)
	    {
	      double Uia = U[i][a];
	      U[i][a] = C * Uia + S * U[i][b];
	      U[i][b] = C * U[i][b] - S * Uia;
	    }
	}
      double complex* Zaa = Z[a][a];
      for (int c = 0; c < 3; c++)
	value += creal(Zaa[c] * conj(Zaa[c]));
    }
  return Py_BuildValue("d", value);
#endif
}

PyObject* swap_arrays(PyObject *self, PyObject *args)
{
  PyArrayObject* a;
  PyArrayObject* b;
  if (!PyArg_ParseTuple(args, "OO", &a, &b)) 
    return NULL;

  void *tmp = (void*) a->data;
  a->data = b->data;
  b->data = tmp;
    
  Py_RETURN_NONE;
}

PyObject* spherical_harmonics(PyObject *self, PyObject *args)
{
  int l;
  PyArrayObject* R_obj_c;
  PyArrayObject* Y_obj_m;
  if (!PyArg_ParseTuple(args, "iOO", &l, &R_obj_c, &Y_obj_m)) 
    return NULL;

  double* R_c = DOUBLEP(R_obj_c);
  double* Y_m = DOUBLEP(Y_obj_m);

  if (l == 0)
      Y_m[0] = 0.28209479177387814;
  else
    {
      double x = R_c[0];
      double y = R_c[1];
      double z = R_c[2];
      if (l == 1)
	{
	  Y_m[0] = 0.48860251190291992 * y;
	  Y_m[1] = 0.48860251190291992 * z;
	  Y_m[2] = 0.48860251190291992 * x;
	}
      else
 	{
	  double r2 = x*x+y*y+z*z;
	  if (l == 2)
	    {		    
	      Y_m[0] = 1.0925484305920792 * x*y;
	      Y_m[1] = 1.0925484305920792 * y*z;
	      Y_m[2] = 0.31539156525252005 * (3*z*z-r2);
	      Y_m[3] = 1.0925484305920792 * x*z;
	      Y_m[4] = 0.54627421529603959 * (x*x-y*y);
	    }
	  else if (l == 3)
	    {
	      Y_m[0] = 0.59004358992664352 * (-y*y*y+3*x*x*y);
	      Y_m[1] = 2.8906114426405538 * x*y*z;
	      Y_m[2] = 0.45704579946446577 * (-y*r2+5*y*z*z);
	      Y_m[3] = 0.3731763325901154 * (5*z*z*z-3*z*r2);
	      Y_m[4] = 0.45704579946446577 * (5*x*z*z-x*r2);
	      Y_m[5] = 1.4453057213202769 * (x*x*z-y*y*z);
	      Y_m[6] = 0.59004358992664352 * (x*x*x-3*x*y*y);
	    }
	  else if (l == 4)
	    {
	      Y_m[0] = 2.5033429417967046 * (x*x*x*y-x*y*y*y);
	      Y_m[1] = 1.7701307697799307 * (-y*y*y*z+3*x*x*y*z);
	      Y_m[2] = 0.94617469575756008 * (-x*y*r2+7*x*y*z*z);
	      Y_m[3] = 0.66904654355728921 * (-3*y*z*r2+7*y*z*z*z);
	      Y_m[4] = 0.10578554691520431 * (-30*z*z*r2+3*r2*r2+35*z*z*z*z);
	      Y_m[5] = 0.66904654355728921 * (7*x*z*z*z-3*x*z*r2);
	      Y_m[6] = 0.47308734787878004 * (-x*x*r2+7*x*x*z*z+y*y*r2-7*y*y*z*z);
	      Y_m[7] = 1.7701307697799307 * (x*x*x*z-3*x*y*y*z);
	      Y_m[8] = 0.62583573544917614 * (-6*x*x*y*y+x*x*x*x+y*y*y*y);
	    }
	  else if (l == 5)
	    {
	      Y_m[0] = 0.65638205684017015 * (y*y*y*y*y+5*x*x*x*x*y-10*x*x*y*y*y);
	      Y_m[1] = 8.3026492595241645 * (x*x*x*y*z-x*y*y*y*z);
	      Y_m[2] = 0.48923829943525038 * (y*y*y*r2-9*y*y*y*z*z-3*x*x*y*r2+27*x*x*y*z*z);
	      Y_m[3] = 4.7935367849733241 * (3*x*y*z*z*z-x*y*z*r2);
	      Y_m[4] = 0.45294665119569694 * (-14*y*z*z*r2+y*r2*r2+21*y*z*z*z*z);
	      Y_m[5] = 0.1169503224534236 * (63*z*z*z*z*z+15*z*r2*r2-70*z*z*z*r2);
	      Y_m[6] = 0.45294665119569694 * (x*r2*r2-14*x*z*z*r2+21*x*z*z*z*z);
	      Y_m[7] = 2.3967683924866621 * (-3*y*y*z*z*z+y*y*z*r2+3*x*x*z*z*z-x*x*z*r2);
	      Y_m[8] = 0.48923829943525038 * (9*x*x*x*z*z-27*x*y*y*z*z-x*x*x*r2+3*x*y*y*r2);
	      Y_m[9] = 2.0756623148810411 * (y*y*y*y*z-6*x*x*y*y*z+x*x*x*x*z);
	      Y_m[10] = 0.65638205684017015 * (-10*x*x*x*y*y+5*x*y*y*y*y+x*x*x*x*x);
	    }
	  else if (l == 6)
	    {
	      Y_m[0] = 1.3663682103838286 * (-10*x*x*x*y*y*y+3*x*x*x*x*x*y+3*x*y*y*y*y*y);
	      Y_m[1] = 2.3666191622317521 * (y*y*y*y*y*z-10*x*x*y*y*y*z+5*x*x*x*x*y*z);
	      Y_m[2] = 2.0182596029148967 * (-x*x*x*y*r2+x*y*y*y*r2-11*x*y*y*y*z*z+11*x*x*x*y*z*z);
	      Y_m[3] = 0.92120525951492349 * (-11*y*y*y*z*z*z-9*x*x*y*z*r2+33*x*x*y*z*z*z+3*y*y*y*z*r2);
	      Y_m[4] =0.92120525951492349 * (x*y*r2*r2+33*x*y*z*z*z*z-18*x*y*z*z*r2); 
	      Y_m[5] = 0.58262136251873142 * (5*y*z*r2*r2+33*y*z*z*z*z*z-30*y*z*z*z*r2);
	      Y_m[6] = 0.063569202267628425 * (231*z*z*z*z*z*z-5*r2*r2*r2+105*z*z*r2*r2-315*z*z*z*z*r2);
	      Y_m[7] = 0.58262136251873142 * (-30*x*z*z*z*r2+33*x*z*z*z*z*z+5*x*z*r2*r2);
	      Y_m[8] = 0.46060262975746175 * (33*x*x*z*z*z*z+x*x*r2*r2-y*y*r2*r2-18*x*x*z*z*r2+18*y*y*z*z*r2-33*y*y*z*z*z*z); 
	      Y_m[9] = 0.92120525951492349 * (-3*x*x*x*z*r2-33*x*y*y*z*z*z+9*x*y*y*z*r2+11*x*x*x*z*z*z);
	      Y_m[10] = 0.50456490072872417 * (11*y*y*y*y*z*z-66*x*x*y*y*z*z-x*x*x*x*r2+6*x*x*y*y*r2+11*x*x*x*x*z*z-y*y*y*y*r2);
	      Y_m[11] = 2.3666191622317521 * (5*x*y*y*y*y*z+x*x*x*x*x*z-10*x*x*x*y*y*z); 
	      Y_m[12] = 0.6831841051919143 * (x*x*x*x*x*x+15*x*x*y*y*y*y-15*x*x*x*x*y*y-y*y*y*y*y*y);
	    }
	}
    }
  Py_RETURN_NONE;
}
