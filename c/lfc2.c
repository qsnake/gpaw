/*  Copyright (C) 2010 CAMd
 *  Please see the accompanying LICENSE file for further information. */

#include "extensions.h"
#include "spline.h"
#include "lfc.h"


PyObject* second_derivative(LFCObject *lfc, PyObject *args)
{
  const PyArrayObject* a_G_obj;
  PyArrayObject* c_Mvv_obj;
  PyArrayObject* h_cv_obj;
  PyArrayObject* n_c_obj;
  PyObject* spline_M_obj;
  PyArrayObject* beg_c_obj;
  PyArrayObject* pos_Wc_obj;

  if (!PyArg_ParseTuple(args, "OOOOOOO", &a_G_obj, &c_Mvv_obj,
                        &h_cv_obj, &n_c_obj,
                        &spline_M_obj, &beg_c_obj,
                        &pos_Wc_obj))
    return NULL; 

  const double* h_cv = (const double*)h_cv_obj->data;
  const long* n_c = (const long*)n_c_obj->data;
  const double (*pos_Wc)[3] = (const double (*)[3])pos_Wc_obj->data;

  long* beg_c = LONGP(beg_c_obj);

  const double Y00dv = lfc->dv / sqrt(4.0 * M_PI);
  const double* a_G = (const double*)a_G_obj->data;
  double* c_Mvv = (double*)c_Mvv_obj->data;
  GRID_LOOP_START(lfc, -1) {
    // In one grid loop iteration, only i2 changes.
    int i2 = Ga % n_c[2] + beg_c[2];
    int i1 = (Ga / n_c[2]) % n_c[1] + beg_c[1];
    int i0 = Ga / (n_c[2] * n_c[1]) + beg_c[0];
    double xG = h_cv[0] * i0 + h_cv[3] * i1 + h_cv[6] * i2;
    double yG = h_cv[1] * i0 + h_cv[4] * i1 + h_cv[7] * i2;
    double zG = h_cv[2] * i0 + h_cv[5] * i1 + h_cv[8] * i2;
    for (int G = Ga; G < Gb; G++) {
      for (int i = 0; i < ni; i++) {
        LFVolume* vol = volume_i + i;
        int M = vol->M;
        double* c_vv = c_Mvv + 9 * M;
        const bmgsspline* spline = (const bmgsspline*) \
          &((const SplineObject*)PyList_GetItem(spline_M_obj, M))->spline;
          
        double x = xG - pos_Wc[vol->W][0];
        double y = yG - pos_Wc[vol->W][1];
        double z = zG - pos_Wc[vol->W][2];
        double r2 = x * x + y * y + z * z;
        double r = sqrt(r2);
        int bin = r / spline->dr;
        assert(bin <= spline->nbins);
        double* s = spline->data + 4 * bin;
        double u = r - bin * spline->dr;
        double dfdror;
        if (bin == 0)
          dfdror = 2.0 * s[2] + 3.0 * s[3] * r;
        else
          dfdror = (s[1] + u * (2.0 * s[2] + u * 3.0 * s[3])) / r;
        double a = a_G[G] * Y00dv;
        dfdror *= a;
        c_vv[0] += dfdror;
        c_vv[4] += dfdror;
        c_vv[8] += dfdror;
        if (r > 1e-15) {
          double b = ((2.0 * s[2] + 6.0 * s[3] * u) * a - dfdror) / r2;
          c_vv[0] += b * x * x;
          c_vv[1] += b * x * y;
          c_vv[2] += b * x * z;
          c_vv[3] += b * y * x;
          c_vv[4] += b * y * y;
          c_vv[5] += b * y * z;
          c_vv[6] += b * z * x;
          c_vv[7] += b * z * y;
          c_vv[8] += b * z * z;
        }
      }
      xG += h_cv[6];
      yG += h_cv[7];
      zG += h_cv[8];
    }
  }
  GRID_LOOP_STOP(lfc, -1);
  Py_RETURN_NONE;
}
