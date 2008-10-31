#include <Python.h>

PyObject* compiled_WITH_SL(PyObject *self, PyObject *args)
{
     if (!PyArg_ParseTuple(args, ""))
          return NULL;

     int sl = 0;

#ifdef GPAW_WITH_SL
     sl = 1;
#endif

     return Py_BuildValue("i", sl);
}
