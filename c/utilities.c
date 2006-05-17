#include <Python.h>
#define NO_IMPORT_ARRAY
#include <Numeric/arrayobject.h>
#include "extensions.h"
#include <math.h>


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
