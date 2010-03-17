/*  Copyright (C) 2010 CAMd
 *  Please see the accompanying LICENSE file for further information. */

#include "extensions.h"
#include "spline.h"
#include "bmgs/spherical_harmonics.h"
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

PyObject* add_derivative(LFCObject *lfc, PyObject *args)
{
  // Coefficients for the lfc's
  const PyArrayObject* c_xM_obj;
  // Array 
  PyArrayObject* a_xG_obj;
  PyArrayObject* h_cv_obj;
  PyArrayObject* n_c_obj;
  PyObject* spline_M_obj;
  PyArrayObject* beg_c_obj;
  PyArrayObject* pos_Wc_obj;
  // Atom index
  int a;
  // Cartesian coordinate
  int v;
  // k-point index
  int q;

  if (!PyArg_ParseTuple(args, "OOOOOOOiii", &c_xM_obj, &a_xG_obj,
                        &h_cv_obj, &n_c_obj, &spline_M_obj, &beg_c_obj,
                        &pos_Wc_obj, &a, &v, &q))
    return NULL;

  // Number of dimensions
  int nd = a_xG_obj->nd;
  // Array with lengths of array dimensions
  npy_intp* dims = a_xG_obj->dimensions;
  // Number of extra dimensions
  int nx = PyArray_MultiplyList(dims, nd - 3);
  // Number of grid points
  int nG = PyArray_MultiplyList(dims + nd - 3, 3);
  // Number of lfc's 
  int nM = c_xM_obj->dimensions[c_xM_obj->nd - 1];

  const double* h_cv = (const double*)h_cv_obj->data;
  const long* n_c = (const long*)n_c_obj->data;
  const double (*pos_Wc)[3] = (const double (*)[3])pos_Wc_obj->data;

  long* beg_c = LONGP(beg_c_obj);

  if (!lfc->bloch_boundary_conditions) {
    const double* c_M = (const double*)c_xM_obj->data;
    double* a_G = (double*)a_xG_obj->data;
    for (int x = 0; x < nx; x++) {
      GRID_LOOP_START(lfc, -1) {

        // In one grid loop iteration, only i2 changes.
        int i2 = Ga % n_c[2] + beg_c[2];
        int i1 = (Ga / n_c[2]) % n_c[1] + beg_c[1];
        int i0 = Ga / (n_c[2] * n_c[1]) + beg_c[0];
        // Grid point position
        double xG = h_cv[0] * i0 + h_cv[3] * i1 + h_cv[6] * i2;
        double yG = h_cv[1] * i0 + h_cv[4] * i1 + h_cv[7] * i2;
        double zG = h_cv[2] * i0 + h_cv[5] * i1 + h_cv[8] * i2;
        // Loop over grid points in current stride
        for (int G = Ga; G < Gb; G++) {
          // Loop over volumes at current grid point
          for (int i = 0; i < ni; i++) {

            LFVolume* vol = volume_i + i;
            int M = vol->M;
            // Check that the volume belongs to the atom in consideration later
            int W = vol->W;
            int nm = vol->nm;
            int l = (nm - 1) / 2;

            const bmgsspline* spline = (const bmgsspline*)              \
              &((const SplineObject*)PyList_GetItem(spline_M_obj, M))->spline;
              
            double x = xG - pos_Wc[W][0];
            double y = yG - pos_Wc[W][1];
            double z = zG - pos_Wc[W][2];
            double R_c[] = {x, y, z};
            double r2 = x * x + y * y + z * z;
            double r = sqrt(r2);
            double f;
            double dfdr;

            bmgs_get_value_and_derivative(spline, r, &f, &dfdr);
            
            // First contribution: f * d(r^l * Y)/dv
            double fdrlYdx_m[nm];
            if (v == 0)
              spherical_harmonics_derivative_x(l, f, x, y, z, r2, fdrlYdx_m);
            else if (v == 1)
              spherical_harmonics_derivative_y(l, f, x, y, z, r2, fdrlYdx_m);
            else
              spherical_harmonics_derivative_z(l, f, x, y, z, r2, fdrlYdx_m);

            for (int m = 0; m < nm; m++)
              a_G[G] += fdrlYdx_m[m] * c_M[M + m];

            // Second contribution: r^(l-1) * Y * df/dr * R_v
            if (r > 1e-15) {
              double rlm1Ydfdr_m[nm]; // r^(l-1) * Y * df/dr
              double rm1dfdr = 1. / r * dfdr;
              spherical_harmonics(l, rm1dfdr, x, y, z, r2, rlm1Ydfdr_m);
              for (int m = 0; m < nm; m++)
                  a_G[G] += rlm1Ydfdr_m[m] * R_c[v] * c_M[M + m];
            }
          }
          // Update coordinates of current grid point
          xG += h_cv[6];
          yG += h_cv[7];
          zG += h_cv[8];
        }
      }
      GRID_LOOP_STOP(lfc, -1);
      c_M += nM;
      a_G += nG;
    }
  }
  else {
    const double complex* c_M = (const double complex*)c_xM_obj->data;
    double complex* a_G = (double complex*)a_xG_obj->data;
    for (int x = 0; x < nx; x++) {
      GRID_LOOP_START(lfc, q) {

        // In one grid loop iteration, only i2 changes.
        int i2 = Ga % n_c[2] + beg_c[2];
        int i1 = (Ga / n_c[2]) % n_c[1] + beg_c[1];
        int i0 = Ga / (n_c[2] * n_c[1]) + beg_c[0];
        // Grid point position
        double xG = h_cv[0] * i0 + h_cv[3] * i1 + h_cv[6] * i2;
        double yG = h_cv[1] * i0 + h_cv[4] * i1 + h_cv[7] * i2;
        double zG = h_cv[2] * i0 + h_cv[5] * i1 + h_cv[8] * i2;
        // Loop over grid points in current stride
        for (int G = Ga; G < Gb; G++) {
          // Loop over volumes at current grid point
          for (int i = 0; i < ni; i++) {
            // Phase of volume
            double complex conjphase = conj(phase_i[i]);
            LFVolume* vol = volume_i + i;
            int M = vol->M;
            // Check that the volume belongs to the atom in consideration later
            int W = vol->W;
            int nm = vol->nm;
            int l = (nm - 1) / 2;

            const bmgsspline* spline = (const bmgsspline*)              \
              &((const SplineObject*)PyList_GetItem(spline_M_obj, M))->spline;

            double x = xG - pos_Wc[W][0];
            double y = yG - pos_Wc[W][1];
            double z = zG - pos_Wc[W][2];
            double R_c[] = {x, y, z};
            double r2 = x * x + y * y + z * z;
            double r = sqrt(r2);
            double f;
            double dfdr;

            bmgs_get_value_and_derivative(spline, r, &f, &dfdr);

            // First contribution: f * d(r^l * Y)/dv
            double fdrlYdx_m[nm];
            if (v == 0)
              spherical_harmonics_derivative_x(l, f, x, y, z, r2, fdrlYdx_m);
            else if (v == 1)
              spherical_harmonics_derivative_y(l, f, x, y, z, r2, fdrlYdx_m);
            else
              spherical_harmonics_derivative_z(l, f, x, y, z, r2, fdrlYdx_m);

            for (int m = 0; m < nm; m++)
              a_G[G] += fdrlYdx_m[m] * c_M[M + m] * conjphase;

            // Second contribution: r^(l-1) * Y * df/dr * R_v
            if (r > 1e-15) {
              double rlm1Ydfdr_m[nm]; // r^(l-1) * Y * df/dr
              double rm1dfdr = 1. / r * dfdr;
              spherical_harmonics(l, rm1dfdr, x, y, z, r2, rlm1Ydfdr_m);
              for (int m = 0; m < nm; m++)
                  a_G[G] += rlm1Ydfdr_m[m] * R_c[v] * c_M[M + m] * conjphase;
            }
          }
          // Update coordinates of current grid point
          xG += h_cv[6];
          yG += h_cv[7];
          zG += h_cv[8];
        }
      }
      GRID_LOOP_STOP(lfc, q);
      c_M += nM;
      a_G += nG;
    }
  }
  Py_RETURN_NONE;
}
