#include <Python.h>
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <assert.h>
#include "xc_gpaw.h"
#include "extensions.h"
#include "libxc/src/xc.h"

//
// This module uses http://www.tddft.org/programs/octopus/wiki/index.php/Libxc
//
// Design follows http://www.cse.scitech.ac.uk/ccg/dft/design.html
//
//
// Definitions of variables used in the spin-compensated case
// (names follow from 'gpaw' tradition):
//
//          __  2
// a2    = |\/n|
//
//
// E = n * e; (e = energy per particle)
//
//         de
// dedrs = ---
//         dr
//           s
//
//            de
// deda2 = ---------
//            __  2
//         d(|\/n| )
//
//         dE
// dEdn =  --- = v
//         dn
//
//            dE
// dEda2 = ---------
//            __  2
//         d(|\/n| )
//

typedef struct {
  int family; /* used very often, so declared redundantly here */
  /* needed choice between different functional types during initialization */
  xc_lda_type lda_func;
  xc_gga_type gga_func;
  xc_mgga_type mgga_func;
  xc_lca_type lca_func;
} functionals_type;

typedef struct
{
  PyObject_HEAD
  void (*get_point_xc)(functionals_type *func, double point[5], double *e, double der[5]);
  void (*get_point_x)(functionals_type *func, double point[5], double *e, double der[5]);
  void (*get_point_c)(functionals_type *func, double point[5], double *e, double der[5]);
  functionals_type xc_functional;
  functionals_type x_functional;
  functionals_type c_functional;
  int nspin; /* must be common to x and c, so declared redundantly here */
  double hybrid;
} lxcXCFunctionalObject;

/* a general call for an LDA functional */
void get_point_lda(functionals_type *func, double point[5], double *e, double der[5])
{
/*   xc_lda_vxc(&(func->lda_func), &(point[0]), e, &(der[0])); */
  xc_lda_vxc(&(func->lda_func), point, e, der);
}

/* a general call for a GGA functional */
void get_point_gga(functionals_type *func, double point[5], double *e, double der[5])
{
  xc_gga(&(func->gga_func), &(point[0]), &(point[2]),
	 e, &(der[0]), &(der[2]));
}

/* a general call for a MGGA functional */
void get_point_mgga(functionals_type *func, double point[5], double *e, double der[5])
{
  printf("get_point_mgga: MGGA not implemented yet\n");
  exit(1);
/*   xc_mgga(&(func->mgga_func), &(point[0]), &(point[2]), */
/*	  e, &(der[0]), &(der[2])); */
}

static void lxcXCFunctional_dealloc(lxcXCFunctionalObject *self)
{
  /* destroy xc functional */
  switch(self->xc_functional.family)
    {
    case XC_FAMILY_GGA:
      xc_gga_end(&(self->xc_functional.gga_func));
      break;
    case XC_FAMILY_LCA:
      /* xc_lca_end(&(self->xc_functional.lca_func)); */ /* MDTMP - does not exist in libx! */
      break;
/*     default: */
/*       printf("lxcXCFunctional_dealloc: cannot destroy nonexisting %d xc functional\n", self->xc_functional.family); */
    }
  /* destroy x functional */
  switch(self->x_functional.family)
    {
    case XC_FAMILY_LDA:
/*       xc_lda_end(&(self->x_functional.lda_func)); */
      break;
    case XC_FAMILY_GGA:
      xc_gga_end(&(self->x_functional.gga_func));
      break;
    case XC_FAMILY_MGGA:
      xc_mgga_end(&(self->x_functional.mgga_func));
      break;
/*     default: */
/*       printf("lxcXCFunctional_dealloc: cannot destroy nonexisting %d x functional\n", self->x_functional.family); */
    }
  /* destroy c functional */
  switch(self->c_functional.family)
    {
    case XC_FAMILY_LDA:
      xc_lda_end(&(self->c_functional.lda_func));
      break;
    case XC_FAMILY_GGA:
      xc_gga_end(&(self->c_functional.gga_func));
      break;
    case XC_FAMILY_MGGA:
      xc_mgga_end(&(self->c_functional.mgga_func));
      break;
/*     default: */
/*       printf("lxcXCFunctional_dealloc: cannot destroy nonexisting %d c functional\n", self->c_functional.family); */
    }
  PyObject_DEL(self);
}

static PyObject*
lxcXCFunctional_is_gga(lxcXCFunctionalObject *self, PyObject *args)
{
  int success = 0; /* assume functional is not GGA */

  /* any of xc x c can be gga */
  if (self->xc_functional.family == XC_FAMILY_GGA) success = XC_FAMILY_GGA;
  if (self->x_functional.family == XC_FAMILY_GGA) success = XC_FAMILY_GGA;
  if (self->c_functional.family == XC_FAMILY_GGA) success = XC_FAMILY_GGA;

  return Py_BuildValue("i", success);
}

static PyObject*
lxcXCFunctional_is_mgga(lxcXCFunctionalObject *self, PyObject *args)
{
  int success = 0; /* assume functional is not MGGA */

  /* not implemented yet */
  /* any of xc x c can be mgga */
/*   if (self->xc_functional.family == XC_FAMILY_MGGA) success = XC_FAMILY_MGGA; */
/*   if (self->x_functional.family == XC_FAMILY_MGGA) success = XC_FAMILY_MGGA; */
/*   if (self->c_functional.family == XC_FAMILY_MGGA) success = XC_FAMILY_MGGA; */

  return Py_BuildValue("i", success);
}

static PyObject*
lxcXCFunctional_CalculateSpinPaired(lxcXCFunctionalObject *self, PyObject *args)
{
  PyArrayObject* e_array;
  PyArrayObject* n_array;
  PyArrayObject* v_array;
  PyArrayObject* a2_array = 0;
  PyArrayObject* dEda2_array = 0;
  if (!PyArg_ParseTuple(args, "OOO|OO", &e_array, &n_array, &v_array,
			&a2_array, &dEda2_array))
    return NULL;

  /* find nspin */
  int nspin = self->nspin;

  assert(nspin == XC_UNPOLARIZED); /* we are spinpaired */

  /* assert (self->hybrid == 0.0); */ /* MDTMP - not implemented yet */

  int ng = e_array->dimensions[0]; /* number of grid points */

  double* e_g = DOUBLEP(e_array); /* e on the grid */
  const double* n_g = DOUBLEP(n_array); /* density on the grid */
  double* v_g = DOUBLEP(v_array); /* v on the grid */

  const double* a2_g = 0; /* a2 on the grid */
  double* dEda2_g = 0; /* dEda2 on the grid */

  if ((self->x_functional.family == XC_FAMILY_GGA) ||
      (self->c_functional.family == XC_FAMILY_GGA))
    {
      a2_g = DOUBLEP(a2_array);
      dEda2_g = DOUBLEP(dEda2_array);
    }

  assert (self->xc_functional.family == XC_FAMILY_UNKNOWN); /* MDTMP not implemented */

  /* find x functional */
  switch(self->x_functional.family)
    {
    case XC_FAMILY_LDA:
      self->get_point_x = get_point_lda;
      break;
    case XC_FAMILY_GGA:
      self->get_point_x = get_point_gga;
      break;
    case XC_FAMILY_MGGA:
      self->get_point_x = get_point_mgga;
      break;
/*     default: */
/*       printf("lxcXCFunctional_CalculateSpinPaired: exchange functional '%d' not found\n", */
/*	     self->x_functional.family); */
    }
  /* find c functional */
  switch(self->c_functional.family)
    {
    case XC_FAMILY_LDA:
      self->get_point_c = get_point_lda;
      break;
    case XC_FAMILY_GGA:
      self->get_point_c = get_point_gga;
      break;
    case XC_FAMILY_MGGA:
      self->get_point_c = get_point_mgga;
      break;
/*     default: */
/*       printf("lxcXCFunctional_CalculateSpinPaired: correlation functional '%d' not found\n", */
/*	     self->c_functional.family); */
    }
  /* ################################################################ */
  for (int g = 0; g < ng; g++)
    {
      double n = n_g[g];
      if (n < NMIN)
	n = NMIN;
      double a2 = 0.0; /* initialize for lda */
      if ((self->x_functional.family == XC_FAMILY_GGA) ||
	  (self->c_functional.family == XC_FAMILY_GGA))
	{
	  a2 = a2_g[g];
	}
      double dExdn = 0.0;
      double dExda2 = 0.0;
      double ex  = 0.0;
      double dEcdn = 0.0;
      double dEcda2 = 0.0;
      double ec = 0.0;

      double point[5]; /* generalized point */
      // from http://www.tddft.org/programs/octopus/wiki/index.php/Libxc:manual
      // rhoa rhob sigmaaa sigmaab sigmabb
      // \sigma[0] = \nabla n_\uparrow \cdot \nabla n_\uparrow \qquad
      // \sigma[1] = \nabla n_\uparrow \cdot \nabla n_\downarrow \qquad
      // \sigma[2] = \nabla n_\downarrow \cdot \nabla n_\downarrow \qquad
      double derivative_x[5]; /* generalized potential */
      // vrhoa vrhob vsigmaaa vsigmaab vsigmabb
      // {\rm vrho}_{\alpha} = \frac{\partial E}{\partial n_\alpha} \qquad
      // {\rm vsigma}_{\alpha} = \frac{\partial E}{\partial \sigma_\alpha}
      double derivative_c[5]; /* generalized potential */
      for(int j=0; j<5; j++) {
	point[j] = derivative_x[j] = derivative_c[j] = 0.0;
      }
      point[0] = n;
      point[2] = a2;

      /* calculate exchange */
      if (self->x_functional.family != XC_FAMILY_UNKNOWN) {
	self->get_point_x(&(self->x_functional), point, &ex, derivative_x);
	dExdn = derivative_x[0];
	dExda2 = derivative_x[2];
      }
      /* calculate correlation */
      if (self->c_functional.family != XC_FAMILY_UNKNOWN) {
	self->get_point_c(&(self->c_functional), point, &ec, derivative_c);
	dEcdn = derivative_c[0];
	dEcda2 = derivative_c[2];
      }
      if ((self->x_functional.family == XC_FAMILY_GGA) ||
	  (self->c_functional.family == XC_FAMILY_GGA))
	{
	  dEda2_g[g] = dExda2 + dEcda2;
	}
      double h1 = 1.0 - self->hybrid;
      e_g[g] = n* (h1 * ex + ec);
      v_g[g] += h1 * dExdn + dEcdn;
    }
  Py_RETURN_NONE;
}

static PyObject*
lxcXCFunctional_CalculateSpinPolarized(lxcXCFunctionalObject *self, PyObject *args)
{
  PyArrayObject* e;
  PyArrayObject* na;
  PyArrayObject* va;
  PyArrayObject* nb;
  PyArrayObject* vb;
  PyArrayObject* a2 = 0;
  PyArrayObject* aa2 = 0;
  PyArrayObject* ab2 = 0;
  PyArrayObject* dEda2 = 0;
  PyArrayObject* dEdaa2 = 0;
  PyArrayObject* dEdab2 = 0;
  if (!PyArg_ParseTuple(args, "OOOOO|OOOOOO", &e, &na, &va, &nb, &vb,
                        &a2, &aa2, &ab2, &dEda2, &dEdaa2, &dEdab2))
    return NULL;

  /* find nspin */
  int nspin = self->nspin;

  assert(nspin == XC_POLARIZED); /* we are spinpolarized */

  /* assert (self->hybrid == 0.0); */ /* MDTMP - not implemented yet */

  int ng = e->dimensions[0];  /* number of grid points */

  double* e_g = DOUBLEP(e); /* e on the grid */
  const double* na_g = DOUBLEP(na); /* alpha density on the grid */
  double* va_g = DOUBLEP(va); /* alpha v on the grid */
  const double* nb_g = DOUBLEP(nb); /* beta density on the grid */
  double* vb_g = DOUBLEP(vb); /* beta v on the grid */

  const double* a2_g = 0; /* sigmaab on the grid */
  const double* aa2_g = 0; /* sigmaaa on the grid */
  const double* ab2_g = 0; /* sigmabb on the grid */
  double* dEda2_g = 0; /* dEdsigmaab on the grid */
  double* dEdaa2_g = 0; /* dEdsigmaaa on the grid */
  double* dEdab2_g = 0; /* dEdsigmabb on the grid */

  if ((self->x_functional.family == XC_FAMILY_GGA) ||
      (self->c_functional.family == XC_FAMILY_GGA))
    {
      a2_g = DOUBLEP(a2);
      aa2_g = DOUBLEP(aa2);
      ab2_g = DOUBLEP(ab2);
      dEda2_g = DOUBLEP(dEda2);
      dEdaa2_g = DOUBLEP(dEdaa2);
      dEdab2_g = DOUBLEP(dEdab2);
    }

  assert (self->xc_functional.family == XC_FAMILY_UNKNOWN); /* MDTMP not implemented */

  /* find x functional */
  switch(self->x_functional.family)
    {
    case XC_FAMILY_LDA:
      self->get_point_x = get_point_lda;
      break;
    case XC_FAMILY_GGA:
      self->get_point_x = get_point_gga;
      break;
    case XC_FAMILY_MGGA:
      self->get_point_x = get_point_mgga;
      break;
/*     default: */
/*       printf("lxcXCFunctional_CalculateSpinPolarized: exchange functional '%d' not found\n", */
/*	     self->x_functional.family); */
    }
  /* find c functional */
  switch(self->c_functional.family)
    {
    case XC_FAMILY_LDA:
      self->get_point_c = get_point_lda;
      break;
    case XC_FAMILY_GGA:
      self->get_point_c = get_point_gga;
      break;
    case XC_FAMILY_MGGA:
      self->get_point_c = get_point_mgga;
      break;
/*     default: */
/*       printf("lxcXCFunctional_CalculateSpinPolarized: correlation functional '%d' not found\n", */
/*	     self->c_functional.family); */
    }
  /* ################################################################ */
  for (int g = 0; g < ng; g++)
    {
      double na = na_g[g];
      if (na < NMIN)
        na = NMIN;
      double sigma0 = 0.0; /* initialize for lda */
      double sigma1 = 0.0; /* initialize for lda */
      double sigma2 = 0.0; /* initialize for lda */
      if ((self->x_functional.family == XC_FAMILY_GGA) ||
	  (self->c_functional.family == XC_FAMILY_GGA))
	{
	  sigma0 = aa2_g[g];
	  sigma2 = ab2_g[g];
	  sigma1 = (a2_g[g] - (sigma0 + sigma2))/2.0;
	}
      double nb = nb_g[g];
      if (nb < NMIN)
        nb = NMIN;
      double n = na + nb;
      double dExdna = 0.0;
      double dExdsigma0 = 0.0;
      double dExdnb = 0.0;
      double dExdsigma2 = 0.0;
      double ex = 0.0;
      double dExdsigma1 = 0.0;

      double dEcdna = 0.0;
      double dEcdsigma0 = 0.0;
      double dEcdnb = 0.0;
      double dEcdsigma2 = 0.0;
      double ec = 0.0;
      double dEcdsigma1 = 0.0;

      double point[5]; /* generalized point */
      // from http://www.tddft.org/programs/octopus/wiki/index.php/Libxc:manual
      // rhoa rhob sigmaaa sigmaab sigmabb
      // \sigma[0] = \nabla n_\uparrow \cdot \nabla n_\uparrow \qquad
      // \sigma[1] = \nabla n_\uparrow \cdot \nabla n_\downarrow \qquad
      // \sigma[2] = \nabla n_\downarrow \cdot \nabla n_\downarrow \qquad
      double derivative_x[5]; /* generalized potential */
      // vrhoa vrhob vsigmaaa vsigmaab vsigmabb
      // {\rm vrho}_{\alpha} = \frac{\partial E}{\partial n_\alpha} \qquad
      // {\rm vsigma}_{\alpha} = \frac{\partial E}{\partial \sigma_\alpha}
      double derivative_c[5]; /* generalized potential */
      for(int j=0; j<5; j++) {
	point[j] = derivative_x[j] = derivative_c[j] = 0.0;
      }

      point[0] = na;
      point[1] = nb;
      point[2] = sigma0;
      point[3] = sigma1;
      point[4] = sigma2;

      /* calculate exchange */
      if (self->x_functional.family != XC_FAMILY_UNKNOWN) {
	self->get_point_x(&(self->x_functional), point, &ex, derivative_x);
	dExdna = derivative_x[0];
	dExdnb = derivative_x[1];
	dExdsigma0 = derivative_x[2];
	dExdsigma1 = derivative_x[3];
	dExdsigma2 = derivative_x[4];
      }
      /* calculate correlation */
      if (self->c_functional.family != XC_FAMILY_UNKNOWN) {
	self->get_point_c(&(self->c_functional), point, &ec, derivative_c);
	dEcdna = derivative_c[0];
	dEcdnb = derivative_c[1];
	dEcdsigma0 = derivative_c[2];
	dEcdsigma1 = derivative_c[3];
	dEcdsigma2 = derivative_c[4];
      }

      if ((self->x_functional.family == XC_FAMILY_GGA) ||
	  (self->c_functional.family == XC_FAMILY_GGA))
	{
          dEdaa2_g[g] = dExdsigma0 + dEcdsigma0;
          dEdab2_g[g] = dExdsigma2 + dEcdsigma2;
          dEda2_g[g] = dExdsigma1 + dEcdsigma1;
	}
      double h1 = 1.0 - self->hybrid;
      e_g[g] = n* (h1 * ex + ec);
      va_g[g] += (h1 * dExdna + dEcdna);
      vb_g[g] += (h1 * dExdnb + dEcdnb);
    }
  Py_RETURN_NONE;
}

static PyObject*
lxcXCFunctional_XCEnergy(lxcXCFunctionalObject *self, PyObject *args)
{
     double na = 0.0;
     double nb = 0.0;
     double sigma0 = 0.0;
     double sigma1 = 0.0;
     double sigma2 = 0.0;
     if (!PyArg_ParseTuple(args, "ddddd", &na, &nb,
                           &sigma0, &sigma1, &sigma2))
          return NULL;

     /* find nspin */
     // int nspin = self->nspin;

     /* assert (self->hybrid == 0.0); */ /* MDTMP - not implemented yet */

     assert (self->xc_functional.family == XC_FAMILY_UNKNOWN); /* MDTMP not implemented */

     double exc = 0.0; /* output */
     double ex = 0.0; /* output */
     double ec = 0.0; /* output */

     /* find x functional */
     switch(self->x_functional.family)
     {
     case XC_FAMILY_LDA:
          self->get_point_x = get_point_lda;
          break;
     case XC_FAMILY_GGA:
          self->get_point_x = get_point_gga;
          break;
     case XC_FAMILY_MGGA:
          self->get_point_x = get_point_mgga;
          break;
/*     default: */
/*       printf("lxcXCFunctional_CalculateSpinPolarized: exchange functional '%d' not found\n", */
/*	     self->x_functional.family); */
     }
     /* find c functional */
     switch(self->c_functional.family)
     {
     case XC_FAMILY_LDA:
          self->get_point_c = get_point_lda;
          break;
     case XC_FAMILY_GGA:
          self->get_point_c = get_point_gga;
          break;
     case XC_FAMILY_MGGA:
          self->get_point_c = get_point_mgga;
          break;
/*     default: */
/*       printf("lxcXCFunctional_CalculateSpinPolarized: correlation functional '%d' not found\n", */
/*	     self->c_functional.family); */
     }
     /* ################################################################ */

     if (na < NMIN)
          na = NMIN;
     if (nb < NMIN)
          nb = NMIN;

     double n = na + nb;

     double dExcdna = 0.0;
     double dExcdsigma0 = 0.0;
     double dExcdnb = 0.0;
     double dExcdsigma2 = 0.0;
     double dExcdsigma1 = 0.0;

     double dExdna = 0.0;
     double dExdsigma0 = 0.0;
     double dExdnb = 0.0;
     double dExdsigma2 = 0.0;
     double dExdsigma1 = 0.0;

     double dEcdna = 0.0;
     double dEcdsigma0 = 0.0;
     double dEcdnb = 0.0;
     double dEcdsigma2 = 0.0;
     double dEcdsigma1 = 0.0;

     double point[5]; /* generalized point */
     // from http://www.tddft.org/programs/octopus/wiki/index.php/Libxc:manual
     // rhoa rhob sigmaaa sigmaab sigmabb
     // \sigma[0] = \nabla n_\uparrow \cdot \nabla n_\uparrow \qquad
     // \sigma[1] = \nabla n_\uparrow \cdot \nabla n_\downarrow \qquad
     // \sigma[2] = \nabla n_\downarrow \cdot \nabla n_\downarrow \qquad
     double derivative_xc[5]; /* generalized potential */
     double derivative_x[5]; /* generalized potential */
     // vrhoa vrhob vsigmaaa vsigmaab vsigmabb
     // {\rm vrho}_{\alpha} = \frac{\partial E}{\partial n_\alpha} \qquad
     // {\rm vsigma}_{\alpha} = \frac{\partial E}{\partial \sigma_\alpha}
     double derivative_c[5]; /* generalized potential */
     for(int j=0; j<5; j++) {
          point[j] =
               derivative_xc[j] = derivative_x[j] = derivative_c[j] = 0.0;
     }

     point[0] = na;
     point[1] = nb;
     point[2] = sigma0;
     point[3] = sigma1;
     point[4] = sigma2;

     /* calculate exchange */
     if (self->x_functional.family != XC_FAMILY_UNKNOWN) {
          self->get_point_x(&(self->x_functional), point, &ex, derivative_x);
          dExdna = derivative_x[0];
          dExdnb = derivative_x[1];
          dExdsigma0 = derivative_x[2];
          dExdsigma1 = derivative_x[3];
          dExdsigma2 = derivative_x[4];
     }
     /* calculate correlation */
     if (self->c_functional.family != XC_FAMILY_UNKNOWN) {
          self->get_point_c(&(self->c_functional), point, &ec, derivative_c);
          dEcdna = derivative_c[0];
          dEcdnb = derivative_c[1];
          dEcdsigma0 = derivative_c[2];
          dEcdsigma1 = derivative_c[3];
          dEcdsigma2 = derivative_c[4];
     }

     // MDTMP: temporary for xc functional
     exc = ex + ec;
     for(int j=0; j<5; j++) {
          derivative_xc[j] = derivative_x[j] + derivative_c[j];
     }

     return Py_BuildValue("dddddddddddddddddd", n*exc, n*ex, n*ec,
                          derivative_xc[0], derivative_xc[1],
                          derivative_xc[2], derivative_xc[3], derivative_xc[4],
                          derivative_x[0], derivative_x[1],
                          derivative_x[2], derivative_x[3], derivative_x[4],
                          derivative_c[0], derivative_c[1],
                          derivative_c[2], derivative_c[3], derivative_c[4]);
}

static PyMethodDef lxcXCFunctional_Methods[] = {
     {"is_gga",
      (PyCFunction)lxcXCFunctional_is_gga, METH_VARARGS, 0},
     {"is_mgga",
      (PyCFunction)lxcXCFunctional_is_mgga, METH_VARARGS, 0},
     {"calculate_spinpaired",
      (PyCFunction)lxcXCFunctional_CalculateSpinPaired, METH_VARARGS, 0},
     {"calculate_spinpolarized",
      (PyCFunction)lxcXCFunctional_CalculateSpinPolarized, METH_VARARGS, 0},
     {"calculate_xcenergy",
      (PyCFunction)lxcXCFunctional_XCEnergy, METH_VARARGS, 0},
     {NULL, NULL, 0, NULL}
};

static PyObject* lxcXCFunctional_getattr(PyObject *obj, char *name)
{
     return Py_FindMethod(lxcXCFunctional_Methods, obj, name);
}

static PyTypeObject lxcXCFunctionalType = {
     PyObject_HEAD_INIT(&PyType_Type)
     0,
     "lxcXCFunctional",
     sizeof(lxcXCFunctionalObject),
     0,
     (destructor)lxcXCFunctional_dealloc,
     0,
     lxcXCFunctional_getattr
};

PyObject * NewlxcXCFunctionalObject(PyObject *obj, PyObject *args)
{
  int xc, x, c; /* functionals identifier number */
  int nspin; /* XC_UNPOLARIZED or XC_POLARIZED  */
  double hybrid = 0.0;

  if (!PyArg_ParseTuple(args, "iiii|d", &xc, &x, &c, &nspin, &hybrid))
    return NULL;
/*   printf("<NewlxcXCFunctionalObject> type=%d %d %d %d %f\n", */
/*	 xc, x, c, nspin, hybrid); */

  /* checking if the numbers xc x c are valid is done at python level */

  lxcXCFunctionalObject *self = PyObject_NEW(lxcXCFunctionalObject,
					     &lxcXCFunctionalType);

  if (self == NULL)
    return NULL;

  assert(nspin==XC_UNPOLARIZED || nspin==XC_POLARIZED);
  self->nspin = nspin; /* must be common to x and c, so declared redundantly */

  /* assert(hybrid==0.0) */;  /* MDTMP - not implemented yet */
  self->hybrid = hybrid;

  /* initialize xc functional */
  assert (xc == XC_FAMILY_UNKNOWN); /* MDTMP not implemented */
  self->xc_functional.family = xc;

  /* initialize x functional */
  self->x_functional.family = xc_family_from_id(x);
  switch(self->x_functional.family)
    {
    case XC_FAMILY_LDA:
      if(x == XC_LDA_X)
	xc_lda_x_init(&(self->x_functional.lda_func),
		      nspin, 3, XC_NON_RELATIVISTIC);
      else
	xc_lda_init(&(self->x_functional.lda_func), x, nspin);
      break;
    case XC_FAMILY_GGA:
      xc_gga_init(&(self->x_functional.gga_func), x, nspin);
      break;
    case XC_FAMILY_MGGA:
      xc_mgga_init(&(self->x_functional.mgga_func), x, nspin);
      break;
/*     default: */
/*       printf("NewlxcXCFunctionalObject: exchange functional '%d' not found\n", x); */
      /* exit(1); */
    }
  /* initialize c functional */
  self->c_functional.family = xc_family_from_id(c);
  switch(self->c_functional.family)
    {
    case XC_FAMILY_LDA:
      xc_lda_init(&(self->c_functional.lda_func), c, nspin);
      break;
    case XC_FAMILY_GGA:
      xc_gga_init(&(self->c_functional.gga_func), c, nspin);
      break;
    case XC_FAMILY_MGGA:
      xc_mgga_init(&(self->c_functional.mgga_func), c, nspin);
      break;
/*     default: */
/*       printf("NewlxcXCFunctionalObject: correlation functional '%d' not found\n", c); */
      /* exit(1); */
    }

/*   printf("NewlxcXCFunctionalObject family=%d %d %d\n", self->xc_functional.family, self->x_functional.family, self->c_functional.family); */

  assert (self->x_functional.family != 4); /* MDTMP not implemented */
  assert (self->c_functional.family != 4); /* MDTMP not implemented */
  assert (self->x_functional.family != 8); /* MDTMP not implemented */
  assert (self->c_functional.family != 8); /* MDTMP not implemented */
  assert (self->x_functional.family != 16); /* MDTMP not implemented */
  assert (self->c_functional.family != 16); /* MDTMP not implemented */

  return (PyObject*)self;
}
