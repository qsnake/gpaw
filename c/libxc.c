/*  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2007-2008  CAMd
 *  Please see the accompanying LICENSE file for further information. */

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
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
//            dE
// dEdtau = ---------
//           d(tau)
//
//                                  2
//                                 d E
// fxc_{\alpha \beta} =  -----------------------
//                        dn_{\alpha}dn_{\beta}
//


typedef struct {
  int family; /* used very often, so declared redundantly here */
  /* needed choice between different functional types during initialization */
  xc_lda_type lda_func;
  xc_gga_type gga_func;
  xc_mgga_type mgga_func;
  xc_lca_type lca_func;
  xc_hyb_gga_type hyb_gga_func;
} functionals_type;

typedef struct
{
  PyObject_HEAD
  /* see http://www.tddft.org/programs/octopus/wiki/index.php/Libxc:manual#Evaluation */
  /* exchange-correlation energy and derivatives */
  void (*get_vxc_xc)(functionals_type *func, double point[7], double *e, double der[7]);
  void (*get_vxc_x)(functionals_type *func, double point[7], double *e, double der[7]);
  void (*get_vxc_c)(functionals_type *func, double point[7], double *e, double der[7]);
  /* exchange-correlation energy second derivatives */
  void (*get_fxc_xc)(functionals_type *func, double point[7], double der[5][5]);
  void (*get_fxc_x)(functionals_type *func, double point[7], double der[5][5]);
  void (*get_fxc_c)(functionals_type *func, double point[7], double der[5][5]);
  /* exchange-correlation energy second derivatives - finite difference */
  void (*get_fxc_fd_xc)(functionals_type *func, double point[7], double der[5][5]);
  void (*get_fxc_fd_x)(functionals_type *func, double point[7], double der[5][5]);
  void (*get_fxc_fd_c)(functionals_type *func, double point[7], double der[5][5]);
  functionals_type xc_functional;
  functionals_type x_functional;
  functionals_type c_functional;
  int nspin; /* must be common to x and c, so declared redundantly here */
} lxcXCFunctionalObject;

void XC(lda_fxc_fd)(const XC(lda_type) *p, const FLOAT *rho, FLOAT *fxc);

/* a general call for an LDA functional */
void get_vxc_lda(functionals_type *func, double point[7], double *e, double der[7])
{
/*   XC(lda_vxc)(&(func->lda_func), &(point[0]), e, &(der[0])); */
  XC(lda_vxc)(&(func->lda_func), point, e, der);
}

/* a general call for a GGA functional */
void get_vxc_gga(functionals_type *func, double point[7], double *e, double der[7])
{
  switch(func->family)
    {
    case XC_FAMILY_GGA:
      XC(gga_vxc)(&(func->gga_func), &(point[0]), &(point[2]),
                  e, &(der[0]), &(der[2]));
      break;
    case XC_FAMILY_HYB_GGA:
      XC(hyb_gga)(&(func->hyb_gga_func), &(point[0]), &(point[2]),
                  e, &(der[0]), &(der[2]));
      break;
    }
}

/* a general call for a MGGA functional */
void get_vxc_mgga(functionals_type *func, double point[7], double *e, double der[7])
{

  XC(mgga)(&(func->mgga_func), &(point[0]), &(point[2]), &(point[5]),
           e, &(der[0]), &(der[2]),  &(der[5]));
}

/* a general call for an LDA functional */
void get_fxc_lda(functionals_type *func, double point[7], double der[5][5])
{
  double v2rho2[3], v2rhosigma[6], v2sigma2[6];

  for(int i=0; i<3; i++) v2rho2[i] = 0.0;
  for(int i=0; i<6; i++){
    v2rhosigma[i] = 0.0;
    v2sigma2[i]    = 0.0;
  }
  XC(lda_fxc)(&(func->lda_func), point, v2rho2);
  der[0][0] = v2rho2[0];
  der[0][1] = der[1][0] = v2rho2[1];
  der[1][1] = v2rho2[2];
  der[0][2] = der[2][0] = v2rhosigma[0];
  der[0][3] = der[3][0] = v2rhosigma[1];
  der[0][4] = der[4][0] = v2rhosigma[2];
  der[1][2] = der[2][1] = v2rhosigma[3];
  der[1][3] = der[3][1] = v2rhosigma[4];
  der[1][4] = der[4][1] = v2rhosigma[5];
  der[2][2] = v2sigma2[0];
  der[2][3] = der[3][2] = v2sigma2[1];
  der[2][4] = der[4][2] = v2sigma2[2];
  der[3][3] = v2sigma2[3];
  der[3][4] = der[4][3] = v2sigma2[4];
  der[4][4] = v2sigma2[5];
}

/* a general call for a GGA functional */
void get_fxc_gga(functionals_type *func, double point[7], double der[5][5])
{
  double v2rho2[3], v2rhosigma[6], v2sigma2[6];

  for(int i=0; i<3; i++) v2rho2[i] = 0.0;
  for(int i=0; i<6; i++){
    v2rhosigma[i] = 0.0;
    v2sigma2[i]    = 0.0;
  }
  XC(gga_fxc)(&(func->gga_func), &(point[0]), &(point[2]),
              v2rho2, v2rhosigma, v2sigma2);
  der[0][0] = v2rho2[0];
  der[0][1] = der[1][0] = v2rho2[1];
  der[1][1] = v2rho2[2];
  der[0][2] = der[2][0] = v2rhosigma[0];
  der[0][3] = der[3][0] = v2rhosigma[1];
  der[0][4] = der[4][0] = v2rhosigma[2];
  der[1][2] = der[2][1] = v2rhosigma[3];
  der[1][3] = der[3][1] = v2rhosigma[4];
  der[1][4] = der[4][1] = v2rhosigma[5];
  der[2][2] = v2sigma2[0];
  der[2][3] = der[3][2] = v2sigma2[1];
  der[2][4] = der[4][2] = v2sigma2[2];
  der[3][3] = v2sigma2[3];
  der[3][4] = der[4][3] = v2sigma2[4];
  der[4][4] = v2sigma2[5];
}

/* a general call for an LDA functional - finite difference */
void get_fxc_fd_lda(functionals_type *func, double point[7], double der[5][5])
{
  double v2rho2[3], v2rhosigma[6], v2sigma2[6];

  for(int i=0; i<3; i++) v2rho2[i] = 0.0;
  for(int i=0; i<6; i++){
    v2rhosigma[i] = 0.0;
    v2sigma2[i]    = 0.0;
  }
  XC(lda_fxc_fd)(&(func->lda_func), point, v2rho2);
  der[0][0] = v2rho2[0];
  der[0][1] = der[1][0] = v2rho2[1];
  der[1][1] = v2rho2[2];
  der[0][2] = der[2][0] = v2rhosigma[0];
  der[0][3] = der[3][0] = v2rhosigma[1];
  der[0][4] = der[4][0] = v2rhosigma[2];
  der[1][2] = der[2][1] = v2rhosigma[3];
  der[1][3] = der[3][1] = v2rhosigma[4];
  der[1][4] = der[4][1] = v2rhosigma[5];
  der[2][2] = v2sigma2[0];
  der[2][3] = der[3][2] = v2sigma2[1];
  der[2][4] = der[4][2] = v2sigma2[2];
  der[3][3] = v2sigma2[3];
  der[3][4] = der[4][3] = v2sigma2[4];
  der[4][4] = v2sigma2[5];
}

// finite difference calculation of second functional derivative
// stolen from libxc/testsuite/xc-consistency.c

double get_point(functionals_type *func, double point[7], double *e, double der[5], int which)
{
  switch(func->family)
    {
    case XC_FAMILY_LDA:
      xc_lda_vxc(&(func->lda_func), &(point[0]), e, &(der[0]));
      break;
    case XC_FAMILY_GGA:
      xc_gga_vxc(&(func->gga_func), &(point[0]), &(point[2]),
                 e, &(der[0]), &(der[2]));
      break;
    case XC_FAMILY_HYB_GGA:
      xc_hyb_gga(&(func->hyb_gga_func), &(point[0]), &(point[2]),
                 e, &(der[0]), &(der[2]));
      break;
    }

  if(which == 0)
    return (*e)*(point[0] + point[1]);
  else
    return der[which-1];
}

void first_derivative(functionals_type *func, double point[7], double der[5], int which,
                      int nspin)
{
  int i;

  for(i=0; i<5; i++){
    const double delta = 5e-10;

    double dd, p[5], v[5];
    int j;

    if(nspin==1 && (i!=0 && i!=2)){
      der[i] = 0.0;
      continue;
    }

    dd = point[i]*delta;
    if(dd < delta) dd = delta;

    for(j=0; j<5; j++) p[j] = point[j];

    if(point[i]>=3.0*dd){ /* centered difference */
      double e, em1, em2, ep1, ep2;

      p[i] = point[i] + dd;
      ep1 = get_point(func, p, &e, v, which);

      p[i] = point[i] + 2*dd;
      ep2 = get_point(func, p, &e, v, which);

      p[i] = point[i] - dd;  /* backward point */
      em1 = get_point(func, p, &e, v, which);

      p[i] = point[i] - 2*dd;  /* backward point */
      em2 = get_point(func, p, &e, v, which);

      der[i]  = 1.0/2.0*(ep1 - em1);
      der[i] += 1.0/12.0*(em2 - 2*em1 + 2*ep1 - ep2);

      der[i] /= dd;

    }else{  /* we use a 5 point forward difference */
      double e, e1, e2, e3, e4, e5;

      p[i] = point[i];
      e1 = get_point(func, p, &e, v, which);

      p[i] = point[i] + dd;
      e2 = get_point(func, p, &e, v, which);

      p[i] = point[i] + 2.0*dd;
      e3 = get_point(func, p, &e, v, which);

      p[i] = point[i] + 3.0*dd;
      e4 = get_point(func, p, &e, v, which);

      p[i] = point[i] + 4.0*dd;
      e5 = get_point(func, p, &e, v, which);

      der[i]  =          (-e1 + e2);
      der[i] -=  1.0/2.0*( e1 - 2*e2 + e3);
      der[i] +=  1.0/3.0*(-e1 + 3*e2 - 3*e3 + e4);
      der[i] -=  1.0/4.0*( e1 - 4*e2 + 6*e3 - 4*e4 + e5);

      der[i] /= dd;
    }
  }
}

void first_derivative_spinpaired(functionals_type *func, double point[7], double der[5],
                                 int which)
{
  first_derivative(func, point, der, which, XC_UNPOLARIZED);
}

void first_derivative_spinpolarized(functionals_type *func, double point[7], double der[5],
                                    int which)
{
  first_derivative(func, point, der, which, XC_POLARIZED);
}

void second_derivatives_spinpaired(functionals_type *func, double point[7], double der[5][5])
{
  int i;

  for(i=0; i<5; i++){
    first_derivative_spinpaired(func, point, der[i], i+1);
  }
}

void second_derivatives_spinpolarized(functionals_type *func, double point[7], double der[5][5])
{
  int i;

  for(i=0; i<5; i++){
    first_derivative_spinpolarized(func, point, der[i], i+1);
  }
}

/* a general call for a functional - finite difference */
void get_fxc_fd_spinpaired(functionals_type *func, double point[7], double der[5][5])
{
  second_derivatives_spinpaired(func, point, der);
}

/* a general call for a functional - finite difference */
void get_fxc_fd_spinpolarized(functionals_type *func, double point[7], double der[5][5])
{
  second_derivatives_spinpolarized(func, point, der);
}

static void lxcXCFunctional_dealloc(lxcXCFunctionalObject *self)
{
  /* destroy xc functional */
  switch(self->xc_functional.family)
    {
    case XC_FAMILY_GGA:
      XC(gga_end)(&(self->xc_functional.gga_func));
      break;
    case XC_FAMILY_HYB_GGA:
      XC(hyb_gga_end)(&(self->xc_functional.hyb_gga_func));
      break;
    case XC_FAMILY_MGGA:
      XC(mgga_end)(&(self->xc_functional.mgga_func));
      break;

    case XC_FAMILY_LCA:
      /* XC(lca_end)(&(self->xc_functional.lca_func)); */ /* MDTMP - does not exist in libx! */
      break;
/*     default: */
/*       printf("lxcXCFunctional_dealloc: cannot destroy nonexisting %d xc functional\n", self->xc_functional.family); */
    }
  /* destroy x functional */
  switch(self->x_functional.family)
    {
    case XC_FAMILY_LDA:
      /*XC(lda_end)(&(self->x_functional.lda_func)); */
      break;
    case XC_FAMILY_GGA:
      XC(gga_end)(&(self->x_functional.gga_func));
      break;
    case XC_FAMILY_HYB_GGA:
      XC(hyb_gga_end)(&(self->x_functional.hyb_gga_func));
      break;
    case XC_FAMILY_MGGA:
      XC(mgga_end)(&(self->x_functional.mgga_func));
      break;
/*     default: */
/*       printf("lxcXCFunctional_dealloc: cannot destroy nonexisting %d x functional\n", self->x_functional.family); */
    }
  /* destroy c functional */
  switch(self->c_functional.family)
    {
    case XC_FAMILY_LDA:
      /*      XC(lda_end)(&(self->c_functional.lda_func)); */
      break;
    case XC_FAMILY_GGA:
      XC(gga_end)(&(self->c_functional.gga_func));
      break;
    case XC_FAMILY_HYB_GGA:
      XC(hyb_gga_end)(&(self->c_functional.hyb_gga_func));
      break;
    case XC_FAMILY_MGGA:
      XC(mgga_end)(&(self->c_functional.mgga_func));
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
lxcXCFunctional_is_hyb_gga(lxcXCFunctionalObject *self, PyObject *args)
{
  int success = 0; /* assume functional is not HYB_GGA */

  /* any of xc x c can be hyb gga */
  if (self->xc_functional.family == XC_FAMILY_HYB_GGA) success = XC_FAMILY_HYB_GGA;
  if (self->x_functional.family == XC_FAMILY_HYB_GGA) success = XC_FAMILY_HYB_GGA;
  if (self->c_functional.family == XC_FAMILY_HYB_GGA) success = XC_FAMILY_HYB_GGA;

  return Py_BuildValue("i", success);
}

static PyObject*
lxcXCFunctional_is_mgga(lxcXCFunctionalObject *self, PyObject *args)
{
  int success = 0; /* assume functional is not MGGA */

  /* any of xc x c can be mgga */
  if (self->xc_functional.family == XC_FAMILY_MGGA) success = XC_FAMILY_MGGA;
  if (self->x_functional.family == XC_FAMILY_MGGA) success = XC_FAMILY_MGGA;
  if (self->c_functional.family == XC_FAMILY_MGGA) success = XC_FAMILY_MGGA;

  return Py_BuildValue("i", success);
}

static PyObject*
lxcXCFunctional_CalculateSpinPaired(lxcXCFunctionalObject *self, PyObject *args)
{
  PyArrayObject* e_array;             /* energy per particle*/
  PyArrayObject* n_array;             /* rho */
  PyArrayObject* v_array;             /* dE/drho */
  PyArrayObject* a2_array = 0;         /* |nabla rho|^2*/
  PyArrayObject* dEda2_array = 0;      /* dE/d|nabla rho|^2 */
  PyArrayObject* tau_array = 0;        /* tau*/
  PyArrayObject* dEdtau_array = 0;     /* dE/dtau */
  if (!PyArg_ParseTuple(args, "OOO|OOOO", &e_array, &n_array, &v_array, /* object | optional objects*/
			&a2_array, &dEda2_array, &tau_array, &dEdtau_array))
    return NULL;

  /* find nspin */
  int nspin = self->nspin;

  assert(nspin == XC_UNPOLARIZED); /* we are spinpaired */

  int ng = e_array->dimensions[0]; /* number of grid points */

  double* e_g = DOUBLEP(e_array); /* e on the grid */
  const double* n_g = DOUBLEP(n_array); /* density on the grid */
  double* v_g = DOUBLEP(v_array); /* v on the grid */

  const double* a2_g = 0; /* a2 on the grid */
  double* tau_g = 0;      /* tau on the grid */
  double* dEda2_g = 0;    /* dEda2 on the grid */
  double* dEdtau_g= 0;    /* dEdt on the grid */

  if (((self->x_functional.family == XC_FAMILY_GGA) ||
       (self->c_functional.family == XC_FAMILY_GGA))
      ||
      ((self->x_functional.family == XC_FAMILY_HYB_GGA) ||
       (self->c_functional.family == XC_FAMILY_HYB_GGA)))
    {
      a2_g = DOUBLEP(a2_array);
      dEda2_g = DOUBLEP(dEda2_array);
    }

  if ((self->x_functional.family == XC_FAMILY_MGGA) ||
      (self->c_functional.family == XC_FAMILY_MGGA))
    {
      a2_g = DOUBLEP(a2_array);
      tau_g = DOUBLEP(tau_array);
      dEda2_g = DOUBLEP(dEda2_array);
      dEdtau_g = DOUBLEP(dEdtau_array);
    }

  assert (self->xc_functional.family == XC_FAMILY_UNKNOWN); /* MDTMP not implemented */

  /* find x functional */
  switch(self->x_functional.family)
    {
    case XC_FAMILY_LDA:
      self->get_vxc_x = get_vxc_lda;
      break;
    case XC_FAMILY_GGA:
      self->get_vxc_x = get_vxc_gga;
      break;
    case XC_FAMILY_HYB_GGA:
      self->get_vxc_x = get_vxc_gga;
      break;
    case XC_FAMILY_MGGA:
      self->get_vxc_x = get_vxc_mgga;
      break;
/*     default: */
/*       printf("lxcXCFunctional_CalculateSpinPaired: exchange functional '%d' not found\n", */
/*	     self->x_functional.family); */
    }
  /* find c functional */
  switch(self->c_functional.family)
    {
    case XC_FAMILY_LDA:
      self->get_vxc_c = get_vxc_lda;
      break;
    case XC_FAMILY_GGA:
      self->get_vxc_c = get_vxc_gga;
      break;
    case XC_FAMILY_HYB_GGA:
      self->get_vxc_c = get_vxc_gga;
      break;
    case XC_FAMILY_MGGA:
      self->get_vxc_c = get_vxc_mgga;
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
      if (((self->x_functional.family == XC_FAMILY_GGA) ||
           (self->c_functional.family == XC_FAMILY_GGA))
          ||
          ((self->x_functional.family == XC_FAMILY_HYB_GGA) ||
           (self->c_functional.family == XC_FAMILY_HYB_GGA)))
	{
	  a2 = a2_g[g];
        }
      double tau = 0.0;
      if ((self->x_functional.family == XC_FAMILY_MGGA) ||
          (self->c_functional.family == XC_FAMILY_MGGA))
        {
          a2 = a2_g[g];
          tau = tau_g[g];
        }
      double dExdn = 0.0;
      double dExda2 = 0.0;
      double ex  = 0.0;
      double dEcdn = 0.0;
      double dEcda2 = 0.0;
      double ec = 0.0;
      double dExdtau=0.0;
      double dEcdtau=0.0;

      double point[7]; /* generalized point */
      // from http://www.tddft.org/programs/octopus/wiki/index.php/Libxc:manual
      // rhoa rhob sigmaaa sigmaab sigmabb taua taub
      // \sigma[0] = \nabla n_\uparrow \cdot \nabla n_\uparrow \qquad
      // \sigma[1] = \nabla n_\uparrow \cdot \nabla n_\downarrow \qquad
      // \sigma[2] = \nabla n_\downarrow \cdot \nabla n_\downarrow \qquad

      double derivative_x[7]; /* generalized potential */
      double derivative_c[7]; /* generalized potential */
      // vrhoa vrhob vsigmaaa vsigmaab vsigmabb dedtaua dedtaub
      // {\rm vrho}_{\alpha} = \frac{\partial E}{\partial n_\alpha} \qquad
      // {\rm vsigma}_{\alpha} = \frac{\partial E}{\partial \sigma_\alpha}
      // {\rm dedtau}_{\alpha} = \frac{\partial E}{\partial \tau_\alpha}

      for(int j=0; j<7; j++)
        {
          point[j] = derivative_x[j] = derivative_c[j] = 0.0;
        }

      point[0] = n;   /* -> rho */
      point[2] = a2;  /* -> sigma */
      point[5] = tau; /* -> tau */

      /* calculate exchange */
      if (self->x_functional.family != XC_FAMILY_UNKNOWN) {
        self->get_vxc_x(&(self->x_functional), point, &ex, derivative_x);
        dExdn = derivative_x[0];
        dExda2 = derivative_x[2];
        dExdtau = derivative_x[5];
        if (self->c_functional.family == XC_FAMILY_HYB_GGA)
        {
          // MDTMP - a hack: HYB_GGA handle h1 internally in c_functional
          dExdn = 0.0;
          dExda2 = 0.0;
          dExdtau = 0.0;
        }
      }
      /* calculate correlation */
      if (self->c_functional.family != XC_FAMILY_UNKNOWN) {
        self->get_vxc_c(&(self->c_functional), point, &ec, derivative_c);
        dEcdn = derivative_c[0];
        dEcda2 = derivative_c[2];
        dEcdtau = derivative_c[5];
      }
      if (((self->x_functional.family == XC_FAMILY_GGA) ||
           (self->c_functional.family == XC_FAMILY_GGA))
          ||
          ((self->x_functional.family == XC_FAMILY_HYB_GGA) ||
           (self->c_functional.family == XC_FAMILY_HYB_GGA)))
        {
          dEda2_g[g] = dExda2 + dEcda2;
        }
      if ((self->x_functional.family == XC_FAMILY_MGGA) ||
	  (self->c_functional.family == XC_FAMILY_MGGA))
	{
	  dEda2_g[g] = dExda2 + dEcda2;
	  dEdtau_g[g] = dExdtau + dEcdtau;
	}
      e_g[g] = n * (ex + ec);
      v_g[g] += dExdn + dEcdn;
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
  PyArrayObject* taua = 0;          /* taua*/
  PyArrayObject* taub = 0 ;         /* taub*/
  PyArrayObject* dEdtaua = 0;       /* dE/dtaua */
  PyArrayObject* dEdtaub = 0;       /* dE/dtaub */
  if (!PyArg_ParseTuple(args, "OOOOO|OOOOOOOOOOOOOOO", &e, &na, &va, &nb, &vb,
                        &a2, &aa2, &ab2, &dEda2, &dEdaa2, &dEdab2,
                        &taua, &taub, &dEdtaua, &dEdtaub))
    return NULL;

  /* find nspin */
  int nspin = self->nspin;

  assert(nspin == XC_POLARIZED); /* we are spinpolarized */

  int ng = e->dimensions[0];  /* number of grid points */

  double* e_g = DOUBLEP(e); /* e on the grid */
  const double* na_g = DOUBLEP(na); /* alpha density on the grid */
  double* va_g = DOUBLEP(va); /* alpha v on the grid */
  const double* nb_g = DOUBLEP(nb); /* beta density on the grid */
  double* vb_g = DOUBLEP(vb); /* beta v on the grid */

  const double* a2_g = 0; /* sigmaab on the grid */
  const double* aa2_g = 0; /* sigmaaa on the grid */
  const double* ab2_g = 0; /* sigmabb on the grid */
  double* taua_g = 0;       /* taua on the grid */
  double* taub_g = 0;       /* taub on the grid */
  double* dEda2_g = 0; /* dEdsigmaab on the grid */
  double* dEdaa2_g = 0; /* dEdsigmaaa on the grid */
  double* dEdab2_g = 0; /* dEdsigmabb on the grid */
  double* dEdtaua_g = 0; /* dEdta on the grid */
  double* dEdtaub_g = 0; /* dEdtb on the grid */

  if (((self->x_functional.family == XC_FAMILY_GGA) ||
       (self->c_functional.family == XC_FAMILY_GGA))
      ||
      ((self->x_functional.family == XC_FAMILY_HYB_GGA) ||
       (self->c_functional.family == XC_FAMILY_HYB_GGA)))
    {
      a2_g = DOUBLEP(a2);
      aa2_g = DOUBLEP(aa2);
      ab2_g = DOUBLEP(ab2);
      dEda2_g = DOUBLEP(dEda2);
      dEdaa2_g = DOUBLEP(dEdaa2);
      dEdab2_g = DOUBLEP(dEdab2);
    }

  if ((self->x_functional.family == XC_FAMILY_MGGA) ||
      (self->c_functional.family == XC_FAMILY_MGGA))
    {
      a2_g = DOUBLEP(a2);
      aa2_g = DOUBLEP(aa2);
      ab2_g = DOUBLEP(ab2);
      taua_g = DOUBLEP(taua);
      taub_g = DOUBLEP(taub);
      dEda2_g = DOUBLEP(dEda2);
      dEdaa2_g = DOUBLEP(dEdaa2);
      dEdab2_g = DOUBLEP(dEdab2);
      dEdtaua_g = DOUBLEP(dEdtaua);
      dEdtaub_g = DOUBLEP(dEdtaub);
    }
  assert (self->xc_functional.family == XC_FAMILY_UNKNOWN); /* MDTMP not implemented */

  /* find x functional */
  switch(self->x_functional.family)
    {
    case XC_FAMILY_LDA:
      self->get_vxc_x = get_vxc_lda;
      break;
    case XC_FAMILY_GGA:
      self->get_vxc_x = get_vxc_gga;
      break;
    case XC_FAMILY_HYB_GGA:
      self->get_vxc_x = get_vxc_gga;
      break;
    case XC_FAMILY_MGGA:
      self->get_vxc_x = get_vxc_mgga;
      break;
/*     default: */
/*       printf("lxcXCFunctional_CalculateSpinPolarized: exchange functional '%d' not found\n", */
/*	     self->x_functional.family); */
    }
  /* find c functional */
  switch(self->c_functional.family)
    {
    case XC_FAMILY_LDA:
      self->get_vxc_c = get_vxc_lda;
      break;
    case XC_FAMILY_GGA:
      self->get_vxc_c = get_vxc_gga;
      break;
    case XC_FAMILY_HYB_GGA:
      self->get_vxc_c = get_vxc_gga;
      break;
    case XC_FAMILY_MGGA:
      self->get_vxc_c = get_vxc_mgga;
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
    double taua = 0.0, taub =0.0;
    double dExdtaua = 0.0;
    double dExdtaub = 0.0;
    double dEcdtaua = 0.0;
    double dEcdtaub = 0.0;

    if (((self->x_functional.family == XC_FAMILY_GGA) ||
         (self->c_functional.family == XC_FAMILY_GGA))
        ||
        ((self->x_functional.family == XC_FAMILY_HYB_GGA) ||
         (self->c_functional.family == XC_FAMILY_HYB_GGA)))
      {
        sigma0 = a2_g[g];
        sigma1 = aa2_g[g];
        sigma2 = ab2_g[g];
      }
    double nb = nb_g[g];
    if (nb < NMIN)
      nb = NMIN;
    if ((self->x_functional.family == XC_FAMILY_MGGA) ||
        (self->c_functional.family == XC_FAMILY_MGGA))
      {
        sigma0 = a2_g[g];
        sigma1 = aa2_g[g];
        sigma2 = ab2_g[g];
        taua = taua_g[g];
        taub = taub_g[g];
      }
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

    double point[7]; /* generalized point */
    // from http://www.tddft.org/programs/octopus/wiki/index.php/Libxc:manual
    // rhoa rhob sigmaaa sigmaab sigmabb taua taub
    // \sigma[0] = \nabla n_\uparrow \cdot \nabla n_\uparrow \qquad
    // \sigma[1] = \nabla n_\uparrow \cdot \nabla n_\downarrow \qquad
    // \sigma[2] = \nabla n_\downarrow \cdot \nabla n_\downarrow \qquad
    double derivative_x[7]; /* generalized potential */
    double derivative_c[7]; /* generalized potential */
    // vrhoa vrhob vsigmaaa vsigmaab vsigmabb dedtaua dedtaub
    // {\rm vrho}_{\alpha} = \frac{\partial E}{\partial n_\alpha} \qquad
    // {\rm vsigma}_{\alpha} = \frac{\partial E}{\partial \sigma_\alpha}
    // {\rm dedtau}_{\alpha} = \frac{\partial E}{\partial \tau_\alpha}

    for(int j=0; j<7; j++)
      {
        point[j] = derivative_x[j] = derivative_c[j]= 0.0;
      }

    point[0] = na;
    point[1] = nb;
    point[2] = sigma0;
    point[3] = sigma1;
    point[4] = sigma2;
    point[5] = taua;
    point[6] = taub;

    /* calculate exchange */
    if (self->x_functional.family != XC_FAMILY_UNKNOWN) {
      self->get_vxc_x(&(self->x_functional), point, &ex, derivative_x);
      dExdna = derivative_x[0];
      dExdnb = derivative_x[1];
      dExdsigma0 = derivative_x[2];
      dExdsigma1 = derivative_x[3];
      dExdsigma2 = derivative_x[4];
      dExdtaua = derivative_x[5];
      dExdtaub = derivative_x[6];
      if (self->c_functional.family == XC_FAMILY_HYB_GGA)
        {
          // MDTMP - a hack: HYB_GGA handle h1 internally in c_functional
          dExdna = 0.0;
          dExdnb = 0.0;
          dExdsigma0 = 0.0;
          dExdsigma1 = 0.0;
          dExdsigma2 = 0.0;
          dExdtaua = 0.0;
          dExdtaub = 0.0;
        }
    }
    /* calculate correlation */
    if (self->c_functional.family != XC_FAMILY_UNKNOWN) {
      self->get_vxc_c(&(self->c_functional), point, &ec, derivative_c);
      dEcdna = derivative_c[0];
      dEcdnb = derivative_c[1];
      dEcdsigma0 = derivative_c[2];
      dEcdsigma1 = derivative_c[3];
      dEcdsigma2 = derivative_c[4];
      dEcdtaua = derivative_c[5];
      dEcdtaub = derivative_c[6];
    }

    if (((self->x_functional.family == XC_FAMILY_GGA) ||
         (self->c_functional.family == XC_FAMILY_GGA))
        ||
        ((self->x_functional.family == XC_FAMILY_HYB_GGA) ||
         (self->c_functional.family == XC_FAMILY_HYB_GGA)))
      {
        dEdaa2_g[g] = dExdsigma1 + dEcdsigma1;
        dEdab2_g[g] = dExdsigma2 + dEcdsigma2;
        dEda2_g[g] = dExdsigma0 + dEcdsigma0;
      }
    if ((self->x_functional.family == XC_FAMILY_MGGA) ||
        (self->c_functional.family == XC_FAMILY_MGGA))
      {
        dEdaa2_g[g] = dExdsigma1 + dEcdsigma1;
        dEdab2_g[g] = dExdsigma2 + dEcdsigma2;
        dEda2_g[g] = dExdsigma0 + dEcdsigma0;
        dEdtaua_g[g] = dExdtaua + dEcdtaua;
        dEdtaub_g[g] = dExdtaub + dEcdtaub;
      }
    e_g[g] = n* (ex + ec);
    va_g[g] += dExdna + dEcdna;
    vb_g[g] += dExdnb + dEcdnb;
  }
  Py_RETURN_NONE;
}

static PyObject*
lxcXCFunctional_CalculateFXCSpinPaired(lxcXCFunctionalObject *self, PyObject *args)
{
  PyArrayObject* n_array;              /* rho */
  PyArrayObject* v2rho2_array;         /* d2E/drho2 */
  PyArrayObject* a2_array = 0;         /* |nabla rho|^2*/
  PyArrayObject* v2rhosigma_array = 0; /* d2E/drhod|nabla rho|^2 */
  PyArrayObject* v2sigma2_array = 0;   /* d2E/drhod|nabla rho|^2 */
  if (!PyArg_ParseTuple(args, "OO|OOO", &n_array, &v2rho2_array, /* object | optional objects*/
                        &a2_array, &v2rhosigma_array, &v2sigma2_array))
    return NULL;

  /* find nspin */
  int nspin = self->nspin;

  assert(nspin == XC_UNPOLARIZED); /* we are spinpaired */

  int ng = n_array->dimensions[0]; /* number of grid points */

  const double* n_g = DOUBLEP(n_array); /* density on the grid */
  double* v2rho2_g = DOUBLEP(v2rho2_array); /* v on the grid */

  const double* a2_g = 0; /* a2 on the grid */
  double* v2rhosigma_g = 0;    /* d2Ednda2 on the grid */
  double* v2sigma2_g = 0;    /* d2Eda2da2 on the grid */

  if (((self->x_functional.family == XC_FAMILY_GGA) ||
       (self->c_functional.family == XC_FAMILY_GGA))
      ||
      ((self->x_functional.family == XC_FAMILY_HYB_GGA) ||
       (self->c_functional.family == XC_FAMILY_HYB_GGA)))
    {
      a2_g = DOUBLEP(a2_array);
      v2rhosigma_g = DOUBLEP(v2rhosigma_array);
      v2sigma2_g = DOUBLEP(v2sigma2_array);
    }

  //assert (self->x_functional.family != XC_FAMILY_MGGA); /* MDTMP - not implemented yet */

  //assert (self->c_functional.family != XC_FAMILY_MGGA); /* MDTMP - not implemented yet */

  assert (self->xc_functional.family == XC_FAMILY_UNKNOWN); /* MDTMP not implemented */

  /* find x functional */
  switch(self->x_functional.family)
    {
    case XC_FAMILY_LDA:
      self->get_fxc_x = get_fxc_lda;
      break;
    case XC_FAMILY_GGA:
      self->get_fxc_x = get_fxc_gga;
      break;
/*     default: */
/*       printf("lxcXCFunctional_CalculateSpinPaired: exchange functional '%d' not found\n", */
/*	     self->x_functional.family); */
    }
  /* find c functional */
  switch(self->c_functional.family)
    {
    case XC_FAMILY_LDA:
      self->get_fxc_c = get_fxc_lda;
      break;
    case XC_FAMILY_GGA:
      self->get_fxc_c = get_fxc_gga;
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
      if (((self->x_functional.family == XC_FAMILY_GGA) ||
           (self->c_functional.family == XC_FAMILY_GGA))
          ||
          ((self->x_functional.family == XC_FAMILY_HYB_GGA) ||
           (self->c_functional.family == XC_FAMILY_HYB_GGA)))
        {
          a2 = a2_g[g];
        }

      double point[7]; /* generalized point */
      // from http://www.tddft.org/programs/octopus/wiki/index.php/Libxc:manual
      // rhoa rhob sigmaaa sigmaab sigmabb taua taub
      // \sigma[0] = \nabla n_\uparrow \cdot \nabla n_\uparrow \qquad
      // \sigma[1] = \nabla n_\uparrow \cdot \nabla n_\downarrow \qquad
      // \sigma[2] = \nabla n_\downarrow \cdot \nabla n_\downarrow \qquad

      double derivative_x[5][5]; /* generalized derivative */
      double derivative_c[5][5]; /* generalized derivative */
      double v2rho2_x[3];
      double v2rhosigma_x[6];
      double v2sigma2_x[6];
      double v2rho2_c[3];
      double v2rhosigma_c[6];
      double v2sigma2_c[6];
      // one that uses this: please add description of spin derivative order notation
      // (see c/libxc/src/gga_perdew.c) MDTMP

      for(int i=0; i<3; i++) v2rho2_x[i] = 0.0;
      for(int i=0; i<3; i++) v2rho2_c[i] = 0.0;
      for(int i=0; i<6; i++){
        v2rhosigma_x[i] = 0.0;
        v2sigma2_x[i]    = 0.0;
        v2rhosigma_c[i] = 0.0;
        v2sigma2_c[i]    = 0.0;
      }

      for(int j=0; j<7; j++)
        {
          point[j] = 0.0;
        }

      for(int i=0; i<7; i++)
        {
          for(int j=0; j<7; j++)
            {
              derivative_x[i][j] = derivative_c[i][j] = 0.0;
            }
        }

      point[0] = n;   /* -> rho */
      point[2] = a2;  /* -> sigma */

      /* calculate exchange */
      if (self->x_functional.family != XC_FAMILY_UNKNOWN) {
        self->get_fxc_x(&(self->x_functional), point, derivative_x);
        //printf("fxc_x '%f'\n", derivative_x[0][0]); // MDTMP
        v2rho2_x[0] = derivative_x[0][0];
        v2rho2_x[1] = derivative_x[0][1]; // XC_POLARIZED
        v2rho2_x[2] = derivative_x[1][1]; // XC_POLARIZED
        //printf("fxc_x '%f'\n", derivative_x[0][2]); // MDTMP
        v2rhosigma_x[0] = derivative_x[0][2];
        v2rhosigma_x[1] = derivative_x[0][3]; // XC_POLARIZED
        v2rhosigma_x[2] = derivative_x[0][4]; // XC_POLARIZED
        v2rhosigma_x[3] = derivative_x[1][2]; // XC_POLARIZED
        v2rhosigma_x[4] = derivative_x[1][3]; // XC_POLARIZED
        v2rhosigma_x[5] = derivative_x[1][4]; // XC_POLARIZED
        //printf("fxc_x '%f'\n", derivative_x[2][2]); // MDTMP
        v2sigma2_x[0] = derivative_x[2][2]; /* aa_aa */
        v2sigma2_x[1] = derivative_x[2][3]; // XC_POLARIZED /* aa_ab */
        v2sigma2_x[2] = derivative_x[2][4]; // XC_POLARIZED /* aa_bb */
        v2sigma2_x[3] = derivative_x[3][3]; // XC_POLARIZED /* ab_ab */
        v2sigma2_x[4] = derivative_x[3][4]; // XC_POLARIZED /* ab_bb */
        v2sigma2_x[5] = derivative_x[4][4]; // XC_POLARIZED /* bb_bb */
      }
      /* calculate correlation */
      if (self->c_functional.family != XC_FAMILY_UNKNOWN) {
        self->get_fxc_c(&(self->c_functional), point, derivative_c);
        v2rho2_c[0] = derivative_c[0][0];
        v2rho2_c[1] = derivative_c[0][1]; // XC_POLARIZED
        v2rho2_c[2] = derivative_c[1][1]; // XC_POLARIZED
        v2rhosigma_c[0] = derivative_c[0][2];
        v2rhosigma_c[1] = derivative_c[0][3]; // XC_POLARIZED
        v2rhosigma_c[2] = derivative_c[0][4]; // XC_POLARIZED
        v2rhosigma_c[3] = derivative_c[1][2]; // XC_POLARIZED
        v2rhosigma_c[4] = derivative_c[1][3]; // XC_POLARIZED
        v2rhosigma_c[5] = derivative_c[1][4]; // XC_POLARIZED
        v2sigma2_c[0] = derivative_c[2][2]; /* aa_aa */
        v2sigma2_c[1] = derivative_c[2][3]; // XC_POLARIZED /* aa_ab */
        v2sigma2_c[2] = derivative_c[2][4]; // XC_POLARIZED /* aa_bb */
        v2sigma2_c[3] = derivative_c[3][3]; // XC_POLARIZED /* ab_ab */
        v2sigma2_c[4] = derivative_c[3][4]; // XC_POLARIZED /* ab_bb */
        v2sigma2_c[5] = derivative_c[4][4]; // XC_POLARIZED /* bb_bb */
      }
      if (((self->x_functional.family == XC_FAMILY_GGA) ||
           (self->c_functional.family == XC_FAMILY_GGA))
          ||
          ((self->x_functional.family == XC_FAMILY_HYB_GGA) ||
           (self->c_functional.family == XC_FAMILY_HYB_GGA)))
        {
          v2rhosigma_g[g] = v2rhosigma_x[0] + v2rhosigma_c[0];
          v2sigma2_g[g] = v2sigma2_x[0] + v2sigma2_c[0];
        }
      v2rho2_g[g] += v2rho2_x[0] + v2rho2_c[0];
    }
  Py_RETURN_NONE;
}

static PyObject*
lxcXCFunctional_CalculateFXC_FD_SpinPaired(lxcXCFunctionalObject *self, PyObject *args)
{
  PyArrayObject* n_array;              /* rho */
  PyArrayObject* v2rho2_array;         /* d2E/drho2 */
  PyArrayObject* a2_array = 0;         /* |nabla rho|^2*/
  PyArrayObject* v2rhosigma_array = 0; /* d2E/drhod|nabla rho|^2 */
  PyArrayObject* v2sigma2_array = 0;   /* d2E/drhod|nabla rho|^2 */
  if (!PyArg_ParseTuple(args, "OO|OOO", &n_array, &v2rho2_array, /* object | optional objects*/
                        &a2_array, &v2rhosigma_array, &v2sigma2_array))
    return NULL;

  /* find nspin */
  int nspin = self->nspin;

  assert(nspin == XC_UNPOLARIZED); /* we are spinpaired */

  int ng = n_array->dimensions[0]; /* number of grid points */

  const double* n_g = DOUBLEP(n_array); /* density on the grid */
  double* v2rho2_g = DOUBLEP(v2rho2_array); /* v on the grid */

  const double* a2_g = 0; /* a2 on the grid */
  double* v2rhosigma_g = 0;    /* d2Ednda2 on the grid */
  double* v2sigma2_g = 0;    /* d2Eda2da2 on the grid */

  if (((self->x_functional.family == XC_FAMILY_GGA) ||
       (self->c_functional.family == XC_FAMILY_GGA))
      ||
      ((self->x_functional.family == XC_FAMILY_HYB_GGA) ||
       (self->c_functional.family == XC_FAMILY_HYB_GGA)))
    {
      a2_g = DOUBLEP(a2_array);
      v2rhosigma_g = DOUBLEP(v2rhosigma_array);
      v2sigma2_g = DOUBLEP(v2sigma2_array);
    }

  //assert (self->x_functional.family != XC_FAMILY_GGA); /* MDTMP - not implemented yet */

  //assert (self->c_functional.family != XC_FAMILY_GGA); /* MDTMP - not implemented yet */

  assert (self->x_functional.family != XC_FAMILY_MGGA); /* MDTMP - not implemented yet */

  assert (self->c_functional.family != XC_FAMILY_MGGA); /* MDTMP - not implemented yet */

  assert (self->xc_functional.family == XC_FAMILY_UNKNOWN); /* MDTMP not implemented */

  /* find x functional */
  self->get_fxc_x = get_fxc_fd_spinpaired;
  /* find c functional */
  self->get_fxc_c = get_fxc_fd_spinpaired;
  /* ################################################################ */
  for (int g = 0; g < ng; g++)
    {
      double n = n_g[g];
      if (n < NMIN)
        n = NMIN;
      double a2 = 0.0; /* initialize for lda */
      if (((self->x_functional.family == XC_FAMILY_GGA) ||
           (self->c_functional.family == XC_FAMILY_GGA))
          ||
          ((self->x_functional.family == XC_FAMILY_HYB_GGA) ||
           (self->c_functional.family == XC_FAMILY_HYB_GGA)))
        {
          a2 = a2_g[g];
        }

      double point[7]; /* generalized point */
      // from http://www.tddft.org/programs/octopus/wiki/index.php/Libxc:manual
      // rhoa rhob sigmaaa sigmaab sigmabb taua taub
      // \sigma[0] = \nabla n_\uparrow \cdot \nabla n_\uparrow \qquad
      // \sigma[1] = \nabla n_\uparrow \cdot \nabla n_\downarrow \qquad
      // \sigma[2] = \nabla n_\downarrow \cdot \nabla n_\downarrow \qquad

      double derivative_x[5][5]; /* generalized derivative */
      double derivative_c[5][5]; /* generalized derivative */
      double v2rho2_x[3];
      double v2rhosigma_x[6];
      double v2sigma2_x[6];
      double v2rho2_c[3];
      double v2rhosigma_c[6];
      double v2sigma2_c[6];
      // one that uses this: please add description of spin derivative order notation
      // (see c/libxc/src/gga_perdew.c) MDTMP

      for(int i=0; i<3; i++) v2rho2_x[i] = 0.0;
      for(int i=0; i<3; i++) v2rho2_c[i] = 0.0;
      for(int i=0; i<6; i++){
        v2rhosigma_x[i] = 0.0;
        v2sigma2_x[i]    = 0.0;
        v2rhosigma_c[i] = 0.0;
        v2sigma2_c[i]    = 0.0;
      }

      for(int j=0; j<7; j++)
        {
          point[j] = 0.0;
        }

      for(int i=0; i<7; i++)
        {
          for(int j=0; j<7; j++)
            {
              derivative_x[i][j] = derivative_c[i][j] = 0.0;
            }
        }

      point[0] = n;   /* -> rho */
      point[2] = a2;  /* -> sigma */

      /* calculate exchange */
      if (self->x_functional.family != XC_FAMILY_UNKNOWN) {
        self->get_fxc_x(&(self->x_functional), point, derivative_x);
        //printf("fxc_fd_x '%f'\n", derivative_x[0][0]); // MDTMP
        v2rho2_x[0] = derivative_x[0][0];
        v2rho2_x[1] = derivative_x[0][1]; // XC_POLARIZED
        v2rho2_x[2] = derivative_x[1][1]; // XC_POLARIZED
        //printf("fxc_fd_x '%f'\n", derivative_x[0][2]); // MDTMP
        v2rhosigma_x[0] = derivative_x[0][2];
        v2rhosigma_x[1] = derivative_x[0][3]; // XC_POLARIZED
        v2rhosigma_x[2] = derivative_x[0][4]; // XC_POLARIZED
        v2rhosigma_x[3] = derivative_x[1][2]; // XC_POLARIZED
        v2rhosigma_x[4] = derivative_x[1][3]; // XC_POLARIZED
        v2rhosigma_x[5] = derivative_x[1][4]; // XC_POLARIZED
        //printf("fxc_fd_x '%f'\n", derivative_x[2][2]); // MDTMP
        v2sigma2_x[0] = derivative_x[2][2]; /* aa_aa */
        v2sigma2_x[1] = derivative_x[2][3]; // XC_POLARIZED /* aa_ab */
        v2sigma2_x[2] = derivative_x[2][4]; // XC_POLARIZED /* aa_bb */
        v2sigma2_x[3] = derivative_x[3][3]; // XC_POLARIZED /* ab_ab */
        v2sigma2_x[4] = derivative_x[3][4]; // XC_POLARIZED /* ab_bb */
        v2sigma2_x[5] = derivative_x[4][4]; // XC_POLARIZED /* bb_bb */
      }
      /* calculate correlation */
      if (self->c_functional.family != XC_FAMILY_UNKNOWN) {
        self->get_fxc_c(&(self->c_functional), point, derivative_c);
        v2rho2_c[0] = derivative_c[0][0];
        v2rho2_c[1] = derivative_c[0][1]; // XC_POLARIZED
        v2rho2_c[2] = derivative_c[1][1]; // XC_POLARIZED
        v2rhosigma_c[0] = derivative_c[0][2];
        v2rhosigma_c[1] = derivative_c[0][3]; // XC_POLARIZED
        v2rhosigma_c[2] = derivative_c[0][4]; // XC_POLARIZED
        v2rhosigma_c[3] = derivative_c[1][2]; // XC_POLARIZED
        v2rhosigma_c[4] = derivative_c[1][3]; // XC_POLARIZED
        v2rhosigma_c[5] = derivative_c[1][4]; // XC_POLARIZED
        v2sigma2_c[0] = derivative_c[2][2]; /* aa_aa */
        v2sigma2_c[1] = derivative_c[2][3]; // XC_POLARIZED /* aa_ab */
        v2sigma2_c[2] = derivative_c[2][4]; // XC_POLARIZED /* aa_bb */
        v2sigma2_c[3] = derivative_c[3][3]; // XC_POLARIZED /* ab_ab */
        v2sigma2_c[4] = derivative_c[3][4]; // XC_POLARIZED /* ab_bb */
        v2sigma2_c[5] = derivative_c[4][4]; // XC_POLARIZED /* bb_bb */
      }
      if (((self->x_functional.family == XC_FAMILY_GGA) ||
           (self->c_functional.family == XC_FAMILY_GGA))
          ||
          ((self->x_functional.family == XC_FAMILY_HYB_GGA) ||
           (self->c_functional.family == XC_FAMILY_HYB_GGA)))
        {
          v2rhosigma_g[g] = v2rhosigma_x[0] + v2rhosigma_c[0];
          v2sigma2_g[g] = v2sigma2_x[0] + v2sigma2_c[0];
        }
      v2rho2_g[g] += v2rho2_x[0] + v2rho2_c[0];
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
  double taua = 0.0;
  double taub = 0.0;
  if (!PyArg_ParseTuple(args, "ddddd|dd", &na, &nb,
                        &sigma0, &sigma1, &sigma2, &taua, &taub))
    return NULL;

  /* find nspin */
  // int nspin = self->nspin;

  assert (self->xc_functional.family == XC_FAMILY_UNKNOWN); /* MDTMP not implemented */

  double exc = 0.0; /* output */
  double ex = 0.0; /* output */
  double ec = 0.0; /* output */

  /* find x functional */
  switch(self->x_functional.family)
    {
    case XC_FAMILY_LDA:
      self->get_vxc_x = get_vxc_lda;
      break;
    case XC_FAMILY_GGA:
      self->get_vxc_x = get_vxc_gga;
      break;
    case XC_FAMILY_HYB_GGA:
      self->get_vxc_x = get_vxc_gga;
      break;
    case XC_FAMILY_MGGA:
      self->get_vxc_x = get_vxc_mgga;
      break;
/*     default: */
/*       printf("lxcXCFunctional_CalculateSpinPolarized: exchange functional '%d' not found\n", */
/*	     self->x_functional.family); */
    }
     /* find c functional */
  switch(self->c_functional.family)
    {
    case XC_FAMILY_LDA:
      self->get_vxc_c = get_vxc_lda;
      break;
    case XC_FAMILY_GGA:
      self->get_vxc_c = get_vxc_gga;
      break;
    case XC_FAMILY_HYB_GGA:
      self->get_vxc_c = get_vxc_gga;
      break;
    case XC_FAMILY_MGGA:
      self->get_vxc_c = get_vxc_mgga;
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

  double dExdna = 0.0;
  double dExdsigma0 = 0.0;
  double dExdnb = 0.0;
  double dExdsigma2 = 0.0;
  double dExdsigma1 = 0.0;
  double dExdtaua = 0.0 ; /* dex/dtaua */
  double dExdtaub= 0.0 ;  /* dex/dtaub */

  double dEcdna = 0.0;
  double dEcdsigma0 = 0.0;
  double dEcdnb = 0.0;
  double dEcdsigma2 = 0.0;
  double dEcdsigma1 = 0.0;
  double dEcdtaua = 0.0; /* de_corr/dtau */
  double dEcdtaub = 0.0; /* de_corr/dtau */

  double point[7]; /* generalized point */
  // from http://www.tddft.org/programs/octopus/wiki/index.php/Libxc:manual
  // rhoa rhob sigmaaa sigmaab sigmabb taua taub
  // \sigma[0] = \nabla n_\uparrow \cdot \nabla n_\uparrow \qquad
  // \sigma[1] = \nabla n_\uparrow \cdot \nabla n_\downarrow \qquad
  // \sigma[2] = \nabla n_\downarrow \cdot \nabla n_\downarrow \qquad
  double derivative_xc[7]; /* generalized potential */
  double derivative_x[7]; /* generalized potential */
  double derivative_c[7]; /* generalized potential */
  // vrhoa vrhob vsigmaaa vsigmaab vsigmabb
  // {\rm vrho}_{\alpha} = \frac{\partial E}{\partial n_\alpha} \qquad
  // {\rm vsigma}_{\alpha} = \frac{\partial E}{\partial \sigma_\alpha}

  for(int j=0; j<7; j++) {
    point[j] =derivative_xc[j] = derivative_x[j] = derivative_c[j]= 0.0;
  }

  point[0] = na;
  point[1] = nb;
  point[2] = sigma0;
  point[3] = sigma1;
  point[4] = sigma2;
  point[5] = taua;
  point[6] = taub;


  /* calculate exchange */
  if (self->x_functional.family != XC_FAMILY_UNKNOWN) {
    self->get_vxc_x(&(self->x_functional), point, &ex, derivative_x);
    dExdna = derivative_x[0];
    dExdnb = derivative_x[1];
    dExdsigma0 = derivative_x[2];
    dExdsigma1 = derivative_x[3];
    dExdsigma2 = derivative_x[4];
    dExdtaua   = derivative_x[5];
    dExdtaub   = derivative_x[6];
    if (self->c_functional.family == XC_FAMILY_HYB_GGA)
      {
        // MDTMP - a hack: HYB_GGA handle h1 internally in c_functional
        derivative_x[0] = 0.0;
        derivative_x[1] = 0.0;
        derivative_x[2] = 0.0;
        derivative_x[3] = 0.0;
        derivative_x[4] = 0.0;
        derivative_x[5] = 0.0;
        derivative_x[6] = 0.0;
      }
  }
  /* calculate correlation */
  if (self->c_functional.family != XC_FAMILY_UNKNOWN) {
    self->get_vxc_c(&(self->c_functional), point, &ec, derivative_c);
    dEcdna = derivative_c[0];
    dEcdnb = derivative_c[1];
    dEcdsigma0 = derivative_c[2];
    dEcdsigma1 = derivative_c[3];
    dEcdsigma2 = derivative_c[4];
    dEcdtaua   = derivative_c[5];
    dEcdtaub   = derivative_c[6];
  }

  // MDTMP: temporary for xc functional
  exc = ex + ec;
  for(int j=0; j<7; j++) {
    derivative_xc[j] = derivative_x[j] + derivative_c[j];
  }

  return Py_BuildValue("dddddddddddddddddddddddd", n*exc, n*ex, n*ec,
                       derivative_xc[0], derivative_xc[1],
                       derivative_xc[2], derivative_xc[3], derivative_xc[4],
                       derivative_xc[5], derivative_xc[6],
                       derivative_x[0], derivative_x[1],
                       derivative_x[2], derivative_x[3], derivative_x[4],
                       derivative_x[5], derivative_x[6],
                       derivative_c[0], derivative_c[1],
                       derivative_c[2], derivative_c[3], derivative_c[4],
                       derivative_c[5], derivative_c[6]);
}

static PyMethodDef lxcXCFunctional_Methods[] = {
  {"is_gga",
   (PyCFunction)lxcXCFunctional_is_gga, METH_VARARGS, 0},
  {"is_hyb_gga",
   (PyCFunction)lxcXCFunctional_is_hyb_gga, METH_VARARGS, 0},
  {"is_mgga",
   (PyCFunction)lxcXCFunctional_is_mgga, METH_VARARGS, 0},
  {"calculate_spinpaired",
   (PyCFunction)lxcXCFunctional_CalculateSpinPaired, METH_VARARGS, 0},
  {"calculate_spinpolarized",
   (PyCFunction)lxcXCFunctional_CalculateSpinPolarized, METH_VARARGS, 0},
  {"calculate_fxc_spinpaired",
   (PyCFunction)lxcXCFunctional_CalculateFXCSpinPaired, METH_VARARGS, 0},
  {"calculate_fxc_fd_spinpaired",
   (PyCFunction)lxcXCFunctional_CalculateFXC_FD_SpinPaired, METH_VARARGS, 0},
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

  if (!PyArg_ParseTuple(args, "iiii", &xc, &x, &c, &nspin))
    return NULL;

  /* checking if the numbers xc x c are valid is done at python level */

  lxcXCFunctionalObject *self = PyObject_NEW(lxcXCFunctionalObject,
					     &lxcXCFunctionalType);

  if (self == NULL)
    return NULL;

  assert(nspin==XC_UNPOLARIZED || nspin==XC_POLARIZED);
  self->nspin = nspin; /* must be common to x and c, so declared redundantly */

  /* initialize xc functional */
  assert (xc == XC_FAMILY_UNKNOWN); /* MDTMP not implemented */
  self->xc_functional.family = xc;

  /* initialize x functional */
  self->x_functional.family = xc_family_from_id(x);
  switch(self->x_functional.family)
    {
    case XC_FAMILY_LDA:
      if(x == XC_LDA_X)
	XC(lda_x_init)(&(self->x_functional.lda_func),
                       nspin, 3, XC_NON_RELATIVISTIC);
      else
	XC(lda_init)(&(self->x_functional.lda_func), x, nspin);
      break;
    case XC_FAMILY_GGA:
      XC(gga_init)(&(self->x_functional.gga_func), x, nspin);
      break;
    case XC_FAMILY_HYB_GGA:
      XC(hyb_gga_init)(&(self->x_functional.hyb_gga_func), x, nspin);
      break;
    case XC_FAMILY_MGGA:
      XC(mgga_init)(&(self->x_functional.mgga_func), x, nspin);
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
      XC(lda_init)(&(self->c_functional.lda_func), c, nspin);
      break;
    case XC_FAMILY_GGA:
      XC(gga_init)(&(self->c_functional.gga_func), c, nspin);
      break;
    case XC_FAMILY_HYB_GGA:
      XC(hyb_gga_init)(&(self->c_functional.hyb_gga_func), c, nspin);
      break;
    case XC_FAMILY_MGGA:
      XC(mgga_init)(&(self->c_functional.mgga_func), c, nspin);
      break;
/*     default: */
/*       printf("NewlxcXCFunctionalObject: correlation functional '%d' not found\n", c); */
      /* exit(1); */
    }

/*   printf("NewlxcXCFunctionalObject family=%d %d %d\n", self->xc_functional.family, self->x_functional.family, self->c_functional.family); */

/*
  assert (self->x_functional.family != 4);
  assert (self->c_functional.family != 4);*/
  assert (self->x_functional.family != 8); /* MDTMP not implemented */
  assert (self->c_functional.family != 8); /* MDTMP not implemented */
  assert (self->x_functional.family != 16); /* MDTMP not implemented */
  assert (self->c_functional.family != 16); /* MDTMP not implemented */

  return (PyObject*)self;
}
