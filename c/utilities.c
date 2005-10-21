#include "Python.h"
#define NO_IMPORT_ARRAY
#include <Numeric/arrayobject.h>
#include "extensions.h"

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
