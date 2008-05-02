#include "spline.h"

static void spline_dealloc(SplineObject *xp)
{
  bmgs_deletespline(&xp->spline);
  PyObject_DEL(xp);
}

static PyObject * spline_get_cutoff(SplineObject *self, PyObject *args)
{
  return Py_BuildValue("d", self->spline.dr * self->spline.nbins);
}

static PyObject * spline_get_angular_momentum_number(SplineObject *self,
						     PyObject *args)
{
  return Py_BuildValue("i", self->spline.l);
}

static PyObject * spline_get_value_and_derivative(SplineObject *obj, 
						  PyObject *args,
						  PyObject *kwargs)
{
  double r;
  if (!PyArg_ParseTuple(args, "d", &r))
    return NULL;  
  double f;
  double dfdr;
  bmgs_get_value_and_derivative(&obj->spline, r, &f, &dfdr);
  return Py_BuildValue("(dd)", f, dfdr);
}

static PyMethodDef spline_methods[] = {
    {"get_cutoff",
     (PyCFunction)spline_get_cutoff, METH_VARARGS, 0},
    {"get_angular_momentum_number", 
     (PyCFunction)spline_get_angular_momentum_number, METH_VARARGS, 0},
    {"get_value_and_derivative", 
     (PyCFunction)spline_get_value_and_derivative, METH_VARARGS, 0},
    {NULL, NULL, 0, NULL}
};

static PyObject* spline_get_attr(PyObject *obj, char *name)
{
    return Py_FindMethod(spline_methods, obj, name);
}

static PyObject * spline_call(SplineObject *obj, PyObject *args,
                              PyObject *kwargs)
{
  double r;
  if (!PyArg_ParseTuple(args, "d", &r))
    return NULL;  
  return Py_BuildValue("d", bmgs_splinevalue(&obj->spline, r));
}

static PyTypeObject SplineType = {
  PyObject_HEAD_INIT(&PyType_Type) 0,
  "Spline",
  sizeof(SplineObject), 0,
  (destructor)spline_dealloc, 0,
  spline_get_attr, 0, 0, 0, 0, 0, 0, 0,
  (ternaryfunc)spline_call
};

PyObject * NewSplineObject(PyObject *self, PyObject *args)
{
  int l;
  double rcut;
  PyArrayObject* farray;
  if (!PyArg_ParseTuple(args, "idO", &l, &rcut, &farray)) 
    return NULL;
  SplineObject *spline = PyObject_NEW(SplineObject, &SplineType);
  if (spline == NULL)
    return NULL;
  int nbins = farray->dimensions[0] - 1;
  double dr = rcut / nbins;
  spline->spline = bmgs_spline(l, dr, nbins, DOUBLEP(farray));
  return (PyObject*)spline;
}
