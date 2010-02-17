/*  Copyright (C) 2010 CAMd
 *  Please see the accompanying LICENSE file for further information. */

#include "extensions.h"

//
// Apply symmetry operation op_cc to a and add result to b:
// 
//     =_       _
//   b(Ug) += a(g),
// 
// where:
// 
//   =                         _T
//   U     = op_cc[c1, c2] and g = (g0, g1, g2).
//    c1,c2
//
PyObject* symmetrize(PyObject *self, PyObject *args)
{
  PyArrayObject* a_g_obj;
  PyArrayObject* b_g_obj;
  PyArrayObject* op_cc_obj;
  if (!PyArg_ParseTuple(args, "OOO", &a_g_obj, &b_g_obj, &op_cc_obj)) 
    return NULL;

  const double* a_g = (const double*)a_g_obj->data;
  double* b_g = (double*)b_g_obj->data;
  const long* C = (const long*)op_cc_obj->data;
  int ng0 = a_g_obj->dimensions[0];
  int ng1 = a_g_obj->dimensions[1];
  int ng2 = a_g_obj->dimensions[2];
  for (int g0 = 0; g0 < ng0; g0++)
    for (int g1 = 0; g1 < ng1; g1++)
      for (int g2 = 0; g2 < ng2; g2++) {
        int p0 = ((C[0] * g0 + C[3] * g1 + C[6] * g2) % ng0 + ng0) % ng0;
        int p1 = ((C[1] * g0 + C[4] * g1 + C[7] * g2) % ng1 + ng1) % ng1;
        int p2 = ((C[2] * g0 + C[5] * g1 + C[8] * g2) % ng2 + ng2) % ng2;
        b_g[(p0 * ng1 + p1) * ng2 + p2] += *a_g++;
      }
  Py_RETURN_NONE;
}
