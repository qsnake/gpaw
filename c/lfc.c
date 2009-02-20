#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "spline.h"
#include "lfc.h"
#include "bmgs/spherical_harmonics.h"
#include "bmgs/bmgs.h"


static void lfc_dealloc(LFCObject *self)
{
  if (self->bloch_boundary_conditions)
    free(self->phase_i);
  free(self->volume_i);
  free(self->work_gm);
  free(self->ngm_W);
  free(self->i_W);
  free(self->volume_W);
  PyObject_DEL(self);
}

PyObject* calculate_potential_matrix(LFCObject *self, PyObject *args);
PyObject* integrate(LFCObject *self, PyObject *args);
PyObject* construct_density(LFCObject *self, PyObject *args);
PyObject* construct_density1(LFCObject *self, PyObject *args);
PyObject* lcao_to_grid(LFCObject *self, PyObject *args);
PyObject* calculate_potential_matrix_derivative(LFCObject *self, 
                                                PyObject *args);

static PyMethodDef lfc_methods[] = {
    {"calculate_potential_matrix",
     (PyCFunction)calculate_potential_matrix, METH_VARARGS, 0},
    {"integrate",
     (PyCFunction)integrate, METH_VARARGS, 0},
    {"construct_density",
     (PyCFunction)construct_density, METH_VARARGS, 0},
    {"construct_density1",
     (PyCFunction)construct_density1, METH_VARARGS, 0},
    {"lcao_to_grid",
     (PyCFunction)lcao_to_grid, METH_VARARGS, 0},
    {"calculate_potential_matrix_derivative",
     (PyCFunction)calculate_potential_matrix_derivative, METH_VARARGS, 0},
#ifdef PARALLEL
    {"broadcast",
     (PyCFunction)localized_functions_broadcast, METH_VARARGS, 0},
#endif
    {NULL, NULL, 0, NULL}
};

static PyObject* lfc_getattr(PyObject *obj, char *name)
{
  return Py_FindMethod(lfc_methods, obj, name);
}

static PyTypeObject LFCType = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,
  "LocalizedFunctionsCollection",
  sizeof(LFCObject),
  0,
  (destructor)lfc_dealloc,
  0,
  lfc_getattr
};

PyObject * NewLFCObject(PyObject *obj, PyObject *args)
{
  PyObject* A_Wgm_obj;
  const PyArrayObject* M_W_obj;
  const PyArrayObject* G_B_obj;
  const PyArrayObject* W_B_obj;
  double dv;
  const PyArrayObject* phase_kW_obj;

  if (!PyArg_ParseTuple(args, "OOOOdO",
                        &A_Wgm_obj, &M_W_obj, &G_B_obj, &W_B_obj, &dv,
                        &phase_kW_obj))
    return NULL; 

  LFCObject *self = PyObject_NEW(LFCObject, &LFCType);
  if (self == NULL)
    return NULL;

  self->dv = dv;

  const int* M_W = (const int*)M_W_obj->data;
  self->G_B = (int*)G_B_obj->data;
  self->W_B = (int*)W_B_obj->data;

  if (phase_kW_obj->dimensions[0] > 0) {
    self->bloch_boundary_conditions = true;
    self->phase_kW = (double complex*)phase_kW_obj->data;
  }
  else {
    self->bloch_boundary_conditions = false;
  }

  int nB = G_B_obj->dimensions[0];
  int nW = PyList_Size(A_Wgm_obj);

  self->nW = nW;
  self->nB = nB;

  int nimax = 0;
  int ngmax = 0;
  int ni = 0;
  int Ga = 0;
  for (int B = 0; B < nB; B++) {
    int Gb = self->G_B[B];
    int nG = Gb - Ga;
    if (ni > 0 && nG > ngmax)
      ngmax = nG;
    if (self->W_B[B] >= 0)
      ni += 1;
    else {
      if (ni > nimax)
        nimax = ni;
      ni--;
    }
    Ga = Gb;
  }
  assert(ni == 0);
  
  self->volume_W = GPAW_MALLOC(LFVolume, nW);
  self->i_W = GPAW_MALLOC(int, nW);
  self->ngm_W = GPAW_MALLOC(int, nW);

  int nmmax = 0;
  for (int W = 0; W < nW; W++) {
    const PyArrayObject* A_gm_obj = \
      (const PyArrayObject*)PyList_GetItem(A_Wgm_obj, W);
    LFVolume* volume = &self->volume_W[W];
    volume->A_gm = (const double*)A_gm_obj->data;
    self->ngm_W[W] = A_gm_obj->dimensions[0] * A_gm_obj->dimensions[1];
    volume->nm = A_gm_obj->dimensions[1];
    volume->M = M_W[W];
    volume->W = W;
    if (volume->nm > nmmax)
      nmmax = volume->nm;
  }

  self->work_gm = GPAW_MALLOC(double, ngmax * nmmax);
  self->volume_i = GPAW_MALLOC(LFVolume, nimax);
  if (self->bloch_boundary_conditions)
    self->phase_i = GPAW_MALLOC(complex double, nimax);
  
  return (PyObject*)self;
}

PyObject* calculate_potential_matrix(LFCObject *lfc, PyObject *args)
{
  const PyArrayObject* vt_G_obj;
  PyArrayObject* Vt_MM_obj;
  int k;

  if (!PyArg_ParseTuple(args, "OOi", &vt_G_obj, &Vt_MM_obj, &k))
    return NULL; 

  const double* vt_G = (const double*)vt_G_obj->data;

  int nM = Vt_MM_obj->dimensions[0];
  double* work_gm = lfc->work_gm;

  if (!lfc->bloch_boundary_conditions) {
    double* Vt_MM = (double*)Vt_MM_obj->data;
    GRID_LOOP_START(lfc, -1) {
      for (int i1 = 0; i1 < ni; i1++) {
	LFVolume* v1 = volume_i + i1;
	int M1 = v1->M;
	int nm1 = v1->nm;
	int gm1 = 0;
	for (int G = Ga; G < Gb; G++)
	  for (int m1 = 0; m1 < nm1; m1++, gm1++){
            //printf("A %f\n", v1->A_gm[gm1]);
            //assert(v1->A_gm[gm1] != 0.0);
	    lfc->work_gm[gm1] = vt_G[G] * v1->A_gm[gm1];
          }
              
	for (int i2 = 0; i2 < ni; i2++) {
	  LFVolume* v2 = volume_i + i2;
	  int M2 = v2->M;
	  if (M1 >= M2) {
	    int nm2 = v2->nm;
	    double* Vt_mm = Vt_MM + M1 * nM + M2;
	    for (int g = 0; g < nG; g++)
	      for (int m1 = 0; m1 < nm1; m1++)
		for (int m2 = 0; m2 < nm2; m2++)
		  Vt_mm[m2 + m1 * nM] += (v2->A_gm[g * nm2 + m2] * 
					  work_gm[g * nm1 + m1] *
					  lfc->dv);
	  }
	}
      }
    }
    GRID_LOOP_STOP(lfc, -1);
  }
  else {
    complex double* Vt_MM = (complex double*)Vt_MM_obj->data;
    GRID_LOOP_START(lfc, k) {
      for (int i1 = 0; i1 < ni; i1++) {
	LFVolume* v1 = volume_i + i1;
	int M1 = v1->M;
	int nm1 = v1->nm;
	int gm1 = 0;
	for (int G = Ga; G < Gb; G++) {
	  for (int m1 = 0; m1 < nm1; m1++, gm1++) {
	    lfc->work_gm[gm1] = vt_G[G] * v1->A_gm[gm1];
	  }
	}
	
	for (int i2 = 0; i2 < ni; i2++) {
	  LFVolume* v2 = volume_i + i2;
	  int M2 = v2->M;
	  if (M1 >= M2) {
	    int nm2 = v2->nm;
	    double complex phase = (conj(phase_i[i1]) * phase_i[i2] * lfc->dv);
	    double complex* Vt_mm = Vt_MM + M1 * nM + M2;
	    for (int g = 0; g < nG; g++) {
	      for (int m1 = 0; m1 < nm1; m1++) {
		complex double wphase = work_gm[g * nm1 + m1] * phase;
		for (int m2 = 0; m2 < nm2; m2++) {
		  Vt_mm[m2 + m1 * nM] += (v2->A_gm[g * nm2 + m2] * wphase);
		}
	      }
	    }
	  }
	}
      }
    }
    GRID_LOOP_STOP(lfc, k);
  }
  Py_RETURN_NONE;
}

PyObject* integrate(LFCObject *lfc, PyObject *args)
{
  const PyArrayObject* a_G_obj;
  PyArrayObject* c_M_obj;
  int k;

  if (!PyArg_ParseTuple(args, "OOi", &a_G_obj, &c_M_obj, &k))
    return NULL; 

  const double* a_G = (const double*)a_G_obj->data;

  if (!lfc->bloch_boundary_conditions)
    {
      double* c_M = (double*)c_M_obj->data;
      GRID_LOOP_START(lfc, -1)
        {
          for (int i = 0; i < ni; i++)
            {
              LFVolume* v = volume_i + i;
              for (int gm = 0, G = Ga; G < Gb; G++)
                for (int m = 0; m < v->nm; m++, gm++)
		  c_M[v->M + m] += a_G[G] * v->A_gm[gm] * lfc->dv;
            }
        }
      GRID_LOOP_STOP(lfc, -1);
    }
  else
    {
      complex double* c_M = (complex double*)c_M_obj->data;
      GRID_LOOP_START(lfc, k)
        {
          for (int i = 0; i < ni; i++)
            {
              LFVolume* v = volume_i + i;
	      double complex phase = phase_i[i] * lfc->dv;
              for (int gm = 0, G = Ga; G < Gb; G++)
                for (int m = 0; m < v->nm; m++, gm++)
		  c_M[v->M + m] += a_G[G] * v->A_gm[gm] * phase;
            }
        }
      GRID_LOOP_STOP(lfc, k);
    }
  Py_RETURN_NONE;
}

PyObject* construct_density(LFCObject *lfc, PyObject *args)
{
  const PyArrayObject* rho_MM_obj;
  PyArrayObject* nt_G_obj;
  int k;

  if (!PyArg_ParseTuple(args, "OOi", &rho_MM_obj, &nt_G_obj, &k))
    return NULL; 
  
  double* nt_G = (double*)nt_G_obj->data;
  
  int nM = rho_MM_obj->dimensions[0];
  
  double* work_gm = lfc->work_gm;

  if (!lfc->bloch_boundary_conditions) {
    const double* rho_MM = (const double*)rho_MM_obj->data;
    GRID_LOOP_START(lfc, -1) {
      for (int i1 = 0; i1 < ni; i1++) {
	LFVolume* v1 = volume_i + i1;
	int M1 = v1->M;
	int nm1 = v1->nm;
	memset(work_gm, 0, nG * nm1 * sizeof(double));
	double factor = 1.0;
	for (int i2 = i1; i2 < ni; i2++) {
	  LFVolume* v2 = volume_i + i2;
	  int M2 = v2->M;
	  int nm2 = v2->nm;
	  const double* rho_mm = rho_MM + M1 * nM + M2;
	  for (int g = 0; g < nG; g++)
	    for (int m2 = 0; m2 < nm2; m2++)
	      for (int m1 = 0; m1 < nm1; m1++)
		work_gm[m1 + g * nm1] += (v2->A_gm[g * nm2 + m2] * 
					  rho_mm[m2 + m1 * nM] *
					  factor);
	  factor = 2.0;
	}
	int gm1 = 0;
	for (int G = Ga; G < Gb; G++)
	  {
	    double nt = 0.0;
	    for (int m1 = 0; m1 < nm1; m1++, gm1++)
	      nt += v1->A_gm[gm1] * work_gm[gm1];
	    nt_G[G] += nt;
	  }
      }
    }
    GRID_LOOP_STOP(lfc, -1);
  }
  else {
    const double complex* rho_MM = (const double complex*)rho_MM_obj->data;
    GRID_LOOP_START(lfc, k) {
      for (int i1 = 0; i1 < ni; i1++) {
	LFVolume* v1 = volume_i + i1;
	int M1 = v1->M;
	int nm1 = v1->nm;
	memset(work_gm, 0, nG * nm1 * sizeof(double));
	double complex factor = 1.0;
	for (int i2 = i1; i2 < ni; i2++) {
	  if (i2 > i1)
	    factor = 2.0 * phase_i[i1] * conj(phase_i[i2]);
	  
	  double rfactor = creal(factor);
	  double ifactor = cimag(factor);
	  
	  LFVolume* v2 = volume_i + i2;
	  int M2 = v2->M;
	  int nm2 = v2->nm;
	  const double complex* rho_mm = rho_MM + M1 * nM + M2;
	  for (int g = 0; g < nG; g++) {
	    for (int m2 = 0; m2 < nm2; m2++) {
	      for (int m1 = 0; m1 < nm1; m1++) {
		complex double rho = rho_mm[m2 + m1 * nM];
		double rrho = creal(rho);
		double irho = cimag(rho);
		double x = rfactor * rrho - ifactor * irho;
		work_gm[m1 + g * nm1] += (v2->A_gm[g * nm2 + m2] * x);
		//creal(rho_mm[m2 + m1 * nM] *
		//factor));
	      }
	    }
	  }
	}
	int gm1 = 0;
	for (int G = Ga; G < Gb; G++) {
	  double nt = 0.0;
	  for (int m1 = 0; m1 < nm1; m1++, gm1++) {
	    nt += v1->A_gm[gm1] * work_gm[gm1];
	  }
	  nt_G[G] += nt;
	}
      }
    }
    GRID_LOOP_STOP(lfc, k);
  }
  Py_RETURN_NONE;
}

PyObject* construct_density1(LFCObject *lfc, PyObject *args)
{
  const PyArrayObject* f_M_obj;
  PyArrayObject* nt_G_obj;
  
  if (!PyArg_ParseTuple(args, "OO", &f_M_obj, &nt_G_obj))
    return NULL; 
  
  const double* f_M = (const double*)f_M_obj->data;
  double* nt_G = (double*)nt_G_obj->data;

  GRID_LOOP_START(lfc, -1) {
    for (int i = 0; i < ni; i++) {
      LFVolume* v = volume_i + i;
      for (int gm = 0, G = Ga; G < Gb; G++) {
	for (int m = 0; m < v->nm; m++, gm++) {
	  nt_G[G] += v->A_gm[gm] * v->A_gm[gm] * f_M[v->M + m];
	}
      }
    }
  }
  GRID_LOOP_STOP(lfc, -1);
  Py_RETURN_NONE;
}

PyObject* lcao_to_grid(LFCObject *lfc, PyObject *args)
{
  const PyArrayObject* c_M_obj;
  PyArrayObject* psit_G_obj;
  int k;

  if (!PyArg_ParseTuple(args, "OOi", &c_M_obj, &psit_G_obj, &k))
    return NULL; 
  
  if (!lfc->bloch_boundary_conditions) {
    const double* c_M = (const double*)c_M_obj->data;
    double* psit_G = (double*)psit_G_obj->data;
    GRID_LOOP_START(lfc, -1) {
      for (int i = 0; i < ni; i++) {
	LFVolume* v = volume_i + i;
	for (int gm = 0, G = Ga; G < Gb; G++) {
	  for (int m = 0; m < v->nm; m++, gm++) {
	    psit_G[G] += v->A_gm[gm] * c_M[v->M + m];
	  }
	}
      }
    }
    GRID_LOOP_STOP(lfc, -1);
  }
  else {
    const double complex* c_M = (const double complex*)c_M_obj->data;
    double complex* psit_G = (double complex*)psit_G_obj->data;
    GRID_LOOP_START(lfc, k) {
      for (int i = 0; i < ni; i++) {
	LFVolume* v = volume_i + i;
	for (int gm = 0, G = Ga; G < Gb; G++) {
	  for (int m = 0; m < v->nm; m++, gm++) {
	    psit_G[G] += v->A_gm[gm] * c_M[v->M + m] * conj(phase_i[i]);
	  }
	}
      }
    }
    GRID_LOOP_STOP(lfc, k);
  }
  Py_RETURN_NONE;
}

PyObject* spline_to_grid(PyObject *self, PyObject *args)
{
  const SplineObject* spline_obj;
  PyArrayObject* beg_c_obj;
  PyArrayObject* end_c_obj;
  PyArrayObject* pos_v_obj;
  PyArrayObject* h_cv_obj;
  PyArrayObject* n_c_obj;
  PyArrayObject* gdcorner_c_obj;
  if (!PyArg_ParseTuple(args, "OOOOOOO", &spline_obj,
                        &beg_c_obj, &end_c_obj, &pos_v_obj, &h_cv_obj,
                        &n_c_obj, &gdcorner_c_obj))
    return NULL; 

  const bmgsspline* spline = (const bmgsspline*)(&(spline_obj->spline));
  long* beg_c = LONGP(beg_c_obj);
  long* end_c = LONGP(end_c_obj);
  double* pos_v = DOUBLEP(pos_v_obj);
  double* h_cv = DOUBLEP(h_cv_obj);
  long* n_c = LONGP(n_c_obj);
  long* gdcorner_c = LONGP(gdcorner_c_obj);

  int l = spline_obj->spline.l;
  int nm = 2 * l + 1;
  double rcut = spline->dr * spline->nbins;

  int ngmax = ((end_c[0] - beg_c[0]) *
               (end_c[1] - beg_c[1]) *
               (end_c[2] - beg_c[2]));
  double* A_gm = GPAW_MALLOC(double, ngmax * nm);
  
  int nBmax = ((end_c[0] - beg_c[0]) *
               (end_c[1] - beg_c[1]));
  int* G_B = GPAW_MALLOC(int, 2 * nBmax);

  int nB = 0;
  int ngm = 0;
  int G = -gdcorner_c[2] + n_c[2] * (beg_c[1] - gdcorner_c[1] + n_c[1] 
                    * (beg_c[0] - gdcorner_c[0]));

  for (int g0 = beg_c[0]; g0 < end_c[0]; g0++) {
    for (int g1 = beg_c[1]; g1 < end_c[1]; g1++) {
      int g2_beg = -1; // function boundary coordinates
      int g2_end = -1;
      for (int g2 = beg_c[2]; g2 < end_c[2]; g2++) {
	double x = h_cv[0] * g0 + h_cv[3] * g1 + h_cv[6] * g2 - pos_v[0];
	double y = h_cv[1] * g0 + h_cv[4] * g1 + h_cv[7] * g2 - pos_v[1];
	double z = h_cv[2] * g0 + h_cv[5] * g1 + h_cv[8] * g2 - pos_v[2];
	double r2 = x * x + y * y + z * z;
	double r = sqrt(r2);
	if (r < rcut) {
	  if (g2_beg < 0)
	    g2_beg = g2; // found boundary
	  g2_end = g2;
	  double A = bmgs_splinevalue(spline, r);
	  double* p = A_gm + ngm;
	  
	  spherical_harmonics(l, A, x, y, z, r2, p);
	  
	  ngm += nm;
	}
      }
      if (g2_end >= 0) {
	g2_end++;
	G_B[nB++] = G + g2_beg;
	G_B[nB++] = G + g2_end;
      }
      G += n_c[2];
    }
    G += n_c[2] * (n_c[1] - end_c[1] + beg_c[1]);
  }
  npy_intp gm_dims[2] = {ngm / (2 * l + 1), 2 * l + 1};
  PyArrayObject* A_gm_obj = (PyArrayObject*)PyArray_SimpleNew(2, gm_dims, 
                                                              NPY_DOUBLE);
  
  memcpy(A_gm_obj->data, A_gm, ngm * sizeof(double));
  free(A_gm);
  
  npy_intp B_dims[1] = {nB};
  PyArrayObject* G_B_obj = (PyArrayObject*)PyArray_SimpleNew(1, B_dims,
                                                             NPY_INT);
  memcpy(G_B_obj->data, G_B, nB * sizeof(int));
  free(G_B);
  
  return Py_BuildValue("(OO)", A_gm_obj, G_B_obj);
}


// Horrible copy-paste of calculate_potential_matrix
// Surely it must be possible to find a way to actually reuse code
// Maybe some kind of preprocessor thing
PyObject* calculate_potential_matrix_derivative(LFCObject *lfc, PyObject *args)
{
  const PyArrayObject* vt_G_obj;
  PyArrayObject* DVt_MMc_obj;
  PyArrayObject* h_cv_obj;
  PyArrayObject* n_c_obj;
  int k;
  PyArrayObject* spline_obj_M_obj;
  PyArrayObject* beg_c_obj;
  PyArrayObject* pos_Wc_obj;

  if (!PyArg_ParseTuple(args, "OOOOiOOO", &vt_G_obj, &DVt_MMc_obj, 
			&h_cv_obj, &n_c_obj, &k,
                        &spline_obj_M_obj, &beg_c_obj,
                        &pos_Wc_obj))
    return NULL;

  const double* vt_G = (const double*)vt_G_obj->data;
  const double* h_cv = (const double*)h_cv_obj->data;
  const long* n_c = (const long*)n_c_obj->data;
  const SplineObject** spline_obj_M = \
    (const SplineObject**)spline_obj_M_obj->data;
  const double (*pos_Wc)[3] = (const double (*)[3])pos_Wc_obj->data;

  long* beg_c = LONGP(beg_c_obj);
  int nM = DVt_MMc_obj->dimensions[0];
  double* work_gm = lfc->work_gm;

  if (!lfc->bloch_boundary_conditions) {
    double* DVt_MMc = (double*)DVt_MMc_obj->data;
    for (int c = 0; c < 3; c++) {
      GRID_LOOP_START(lfc, -1) {
        // In one grid loop iteration, only z changes.
        int iza = Ga % n_c[2] + beg_c[2];
        int iy = (Ga / n_c[2]) % n_c[1] + beg_c[1];
        int ix = Ga / (n_c[2] * n_c[1]) + beg_c[0];
        int iz = iza;

        //assert(Ga == ((ix - beg_c[0]) * n_c[1] + (iy - beg_c[1])) 
        //       * n_c[2] + iza - beg_c[2]);

        for (int i1 = 0; i1 < ni; i1++) {
          iz = iza;
          LFVolume* v1 = volume_i + i1;
          int M1 = v1->M;
          const SplineObject* spline_obj = spline_obj_M[M1];
          const bmgsspline* spline = \
            (const bmgsspline*)(&(spline_obj->spline));
          
          int nm1 = v1->nm;
          int l = (nm1 - 1) / 2;
          //assert(2 * l + 1 == nm1);
          //assert(spline_obj->spline.l == l);

          int gm1 = 0;
          for (int G = Ga; G < Gb; G++, iz++) {
            double xG = h_cv[0] * ix + h_cv[3] * iy + h_cv[6] * iz;
            double yG = h_cv[1] * ix + h_cv[4] * iy + h_cv[7] * iz;
            double zG = h_cv[2] * ix + h_cv[5] * iy + h_cv[8] * iz;

            double x = xG - pos_Wc[v1->W][0];
            double y = yG - pos_Wc[v1->W][1];
            double z = zG - pos_Wc[v1->W][2];

            double R_c[] = {x, y, z};
            
            double r2 = x * x + y * y + z * z;
            double r = sqrt(r2);
            double invr;
            if(r > 1e-15) {
              invr = 1.0 / r;
            }
            else {
              invr = 0.0;
            }
            //assert(G == ((ix - beg_c[0]) * n_c[1] + 
            //             (iy - beg_c[1])) * n_c[2] + iz - beg_c[2]);

            double f;
            double dfdr;
            bmgs_get_value_and_derivative(spline, r, &f, &dfdr);
            //assert (r <= spline->dr * spline->nbins); // important

            double fdYdc_m[nm1];
            double rlYdfdr_m[nm1];
            double test_fY_m[nm1];

            switch(c) {
            case 0:
              spherical_harmonics_derivative_x(l, f, x, y, z, r2, fdYdc_m);
              break;
            case 1:
              spherical_harmonics_derivative_y(l, f, x, y, z, r2, fdYdc_m);
              break;
            case 2:
              spherical_harmonics_derivative_z(l, f, x, y, z, r2, fdYdc_m);
              break;
            }
            spherical_harmonics(l, dfdr, x, y, z, r2, rlYdfdr_m);
            spherical_harmonics(l, f, x, y, z, r2, test_fY_m);

            for (int m1 = 0; m1 < nm1; m1++, gm1++) {
              //assert(abs(test_fY_m[m1] - v1->A_gm[gm1]) < 1e-10);
              //if(l == 0){
              //  assert(fdYdc_m[m1] == 0);
              //}
              lfc->work_gm[gm1] = vt_G[G] * (fdYdc_m[m1] +
                                             rlYdfdr_m[m1] * R_c[c] * invr);
            }            
          } // end loop over G
          for (int i2 = 0; i2 < ni; i2++) {
            LFVolume* v2 = volume_i + i2;
            int M2 = v2->M;
            //if (M1 >= M2) { // XXX // Matrix is hermitian
            if (1){
              int nm2 = v2->nm;
              double* DVt_mmc = DVt_MMc + (M1 * nM + M2) * 3;
              for (int g = 0; g < nG; g++) {
                for (int m1 = 0; m1 < nm1; m1++) {
                  for (int m2 = 0; m2 < nm2; m2++) {
                    double A2 = v2->A_gm[g * nm2 + m2];
                    double A1 = work_gm[g * nm1 + m1];
                    DVt_mmc[3 * (m2 + m1 * nM) + c] += A2 * A1 * lfc->dv;
                  }
                }
              }
            }
          } // i2 loop
        } // G loop
      } // i1 loop
      GRID_LOOP_STOP(lfc, -1);
    } // c loop

  }
  else {
    complex double* DVt_MMc = (complex double*)DVt_MMc_obj->data;
    for (int c = 0; c < 3; c++) {
      GRID_LOOP_START(lfc, k) {
        // In one grid loop iteration, only z changes.
        int iza = Ga % n_c[2] + beg_c[0];
        int iy = (Ga / n_c[2]) % n_c[1] + beg_c[1];
        int ix = Ga / (n_c[2] * n_c[1]) + beg_c[2];
        int iz = iza;

        for (int i1 = 0; i1 < ni; i1++) {
          iz = iza;
          LFVolume* v1 = volume_i + i1;
          int M1 = v1->M;
          const SplineObject* spline_obj = spline_obj_M[M1];
          const bmgsspline* spline = \
            (const bmgsspline*)(&(spline_obj->spline));
          
          int nm1 = v1->nm;
          int l = (nm1 - 1) / 2;
          //assert(2 * l + 1 == nm1);
          //assert(spline_obj->spline.l == l);

          int gm1 = 0;
          for (int G = Ga; G < Gb; G++, iz++) {
            double xG = h_cv[0] * ix + h_cv[3] * iy + h_cv[6] * iz;
            double yG = h_cv[1] * ix + h_cv[4] * iy + h_cv[7] * iz;
            double zG = h_cv[2] * ix + h_cv[5] * iy + h_cv[8] * iz;

            double x = xG - pos_Wc[v1->W][0];
            double y = yG - pos_Wc[v1->W][1];
            double z = zG - pos_Wc[v1->W][2];

            double R_c[] = {x, y, z};
            
            double r2 = x * x + y * y + z * z;
            double r = sqrt(r2);
            double invr;
            if(r > 1e-15) {
              invr = 1.0 / r;
            }
            else {
              invr = 0.0;
            }
            double f;
            double dfdr;
            bmgs_get_value_and_derivative(spline, r, &f, &dfdr);
            //assert (r <= spline->dr * spline->nbins);

            double fdYdc_m[nm1];
            double rlYdfdr_m[nm1];
            double test_fY_m[nm1];

            switch(c) {
            case 0:
              spherical_harmonics_derivative_x(l, f, x, y, z, r2, fdYdc_m);
              break;
            case 1:
              spherical_harmonics_derivative_y(l, f, x, y, z, r2, fdYdc_m);
              break;
            case 2:
              spherical_harmonics_derivative_z(l, f, x, y, z, r2, fdYdc_m);
              break;
            }
            spherical_harmonics(l, dfdr, x, y, z, r2, rlYdfdr_m);
            spherical_harmonics(l, f, x, y, z, r2, test_fY_m);

            for (int m1 = 0; m1 < nm1; m1++, gm1++) {
              assert(abs(test_fY_m[m1] - v1->A_gm[gm1]) < 1e-10);
              //if(l == 0){
              //  assert(fdYdc_m[m1] == 0);
              //}
              lfc->work_gm[gm1] = vt_G[G] * (fdYdc_m[m1] +
                                             rlYdfdr_m[m1] * R_c[c] * invr);
            }            
          } // end loop over G
	
          for (int i2 = 0; i2 < ni; i2++) {
            LFVolume* v2 = volume_i + i2;
            int M2 = v2->M;
            if(1) { //if (M1 >= M2) { // XXX Matrix is hermitian
              int nm2 = v2->nm;
              double complex phase = (conj(phase_i[i1]) * phase_i[i2] 
                                      * lfc->dv);
              double complex* DVt_mmc = DVt_MMc + (M1 * nM + M2) * 3;
              for (int g = 0; g < nG; g++) {
                for (int m1 = 0; m1 < nm1; m1++) {
                  complex double wphase = work_gm[g * nm1 + m1] * phase;
                  for (int m2 = 0; m2 < nm2; m2++) {
                    DVt_mmc[3 * (m2 + m1 * nM) + c] += (v2->A_gm[g * nm2 + m2] 
                                                        * wphase);
                  }
                }
              }
            }
          } // i2 loop
        } // G loop
      } // i1 loop
      GRID_LOOP_STOP(lfc, k);
    } // c loop
  }
  Py_RETURN_NONE;
}

