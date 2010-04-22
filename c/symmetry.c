/*  Copyright (C) 2010 CAMd
 *  Please see the accompanying LICENSE file for further information. */
#include "extensions.h"

//
// Apply symmetry operation op_cc to a and add result to b:
// 
//     =T_       _
//   b(U g) += a(g),
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

  const long* C = (const long*)op_cc_obj->data;
  int ng0 = a_g_obj->dimensions[0];
  int ng1 = a_g_obj->dimensions[1];
  int ng2 = a_g_obj->dimensions[2];

  const double* a_g = (const double*)a_g_obj->data;
  double* b_g = (double*)b_g_obj->data;
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


PyObject* symmetrize_wavefunction(PyObject *self, PyObject *args)
{
  PyArrayObject* a_g_obj;
  PyArrayObject* b_g_obj;
  PyArrayObject* op_cc_obj;
  PyArrayObject* kpt0_obj;
  PyArrayObject* kpt1_obj;

  if (!PyArg_ParseTuple(args, "OOOOO", &a_g_obj, &b_g_obj, &op_cc_obj, &kpt0_obj, &kpt1_obj)) 
    return NULL;

  const long* C = (const long*)op_cc_obj->data;
  const double* kpt0 = (const double*) kpt0_obj->data;
  const double* kpt1 = (const double*) kpt1_obj->data;
  int ng0 = a_g_obj->dimensions[0];
  int ng1 = a_g_obj->dimensions[1];
  int ng2 = a_g_obj->dimensions[2];

  const double complex* a_g = (const double complex*)a_g_obj->data;
  double complex* b_g = (double complex*)b_g_obj->data;

  for (int g0 = 0; g0 < ng0; g0++)
    for (int g1 = 0; g1 < ng1; g1++)
      for (int g2 = 0; g2 < ng2; g2++) {
        int p0 = ((C[0] * g0 + C[3] * g1 + C[6] * g2) % ng0 + ng0) % ng0;
        int p1 = ((C[1] * g0 + C[4] * g1 + C[7] * g2) % ng1 + ng1) % ng1;
        int p2 = ((C[2] * g0 + C[5] * g1 + C[8] * g2) % ng2 + ng2) % ng2;

	double complex phase = cexp( I * 2. * M_PI * 
          (  kpt1[0]/ng0*p0 + kpt1[1]/ng1*p1 + kpt1[2]/ng2*p2 
           - kpt0[0]/ng0*g0 - kpt0[1]/ng1*g1 - kpt0[2]/ng2*g2) );
	b_g[(p0 * ng1 + p1) * ng2 + p2] += (*a_g * phase);
        a_g++;

      }

  Py_RETURN_NONE;
}
