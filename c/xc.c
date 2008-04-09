#include <Python.h>
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "xc_gpaw.h"
#include "extensions.h"

//
//          __  2
// a2    = |\/n|
//
//         dE
// dedrs = ---
//         dr
//           s
//
//            dE
// deda2 = ---------
//            __  2
//         d(|\/n| )
//

double pbe_exchange(const xc_parameters* par,
		    double n, double rs, double a2,
		    double* dedrs, double* deda2);
double pbe_correlation(double n, double rs, double zeta, double a2, 
		       bool gga, bool spinpol,
		       double* dedrs, double* dedzeta, double* deda2);
double pw91_exchange(const xc_parameters* par,
		     double n, double rs, double a2,
		     double* dedrs, double* deda2);
double pw91_correlation(double n, double rs, double zeta, double a2, 
			bool gga, bool spinpol,
			double* dedrs, double* dedzeta, double* deda2);
double rpbe_exchange(const xc_parameters* par,
		     double n, double rs, double a2,
		     double* dedrs, double* deda2);
double ensemble_exchange(const xc_parameters* par,
			 double n, double rs, double a2,
			 double* dedrs, double* deda2);
double pade_exchange(const xc_parameters* par,
		     double n, double rs, double a2,
		     double* dedrs, double* deda2);
double bee1_exchange(const xc_parameters* par,
		     double n, double rs, double a2,
		     double* dedrs, double* deda2);
double lb94_correction(const xc_parameters* par,
		       double n, double rs, double a2,
		       double* dedrs, double* deda2);

double zero_exchange(const xc_parameters* par,
		     double n, double rs, double a2,
		     double* dedrs, double* deda2)
{
  *dedrs = 0.0;
  if (par->gga)
    *deda2 = 0.0;
  return 0.0;
}

double zero_correlation(double n, double rs, double zeta, double a2, 
			bool gga, bool spinpol,
			double* dedrs, double* dedzeta, double* deda2)
{
  *dedrs = 0.0;
  if (spinpol)
    *dedzeta = 0.0;
  if (gga)
    *deda2 = 0.0;
  return 0.0;
}

typedef struct 
{
  PyObject_HEAD
  double (*exchange)(const xc_parameters* par,
		     double n, double rs, double a2,
		     double* dedrs, double* deda2);
  double (*correlation)(double n, double rs, double zeta, double a2, 
			bool gga, bool spinpol,
			double* dedrs, double* dedzeta, double* deda2);
  double (*correction)(const xc_parameters* par,
		       double n, double rs, double a2,
		       double* dedrs, double* deda2);
  xc_parameters par;
} XCFunctionalObject;

static void XCFunctional_dealloc(XCFunctionalObject *self)
{
  PyObject_DEL(self);
}

static PyObject* 
XCFunctional_CalculateSpinPaired(XCFunctionalObject *self, PyObject *args)
{
  PyArrayObject* e_array;
  PyArrayObject* n_array;
  PyArrayObject* v_array;
  PyArrayObject* a2_array = 0;
  PyArrayObject* deda2_array = 0;
  if (!PyArg_ParseTuple(args, "OOO|OO", &e_array, &n_array, &v_array,
			&a2_array, &deda2_array))
    return NULL;

  int ng = e_array->dimensions[0];
  const xc_parameters* par = &self->par;

  double* e_g = DOUBLEP(e_array);
  const double* n_g = DOUBLEP(n_array);
  double* v_g = DOUBLEP(v_array);

  const double* a2_g = 0;
  double* deda2_g = 0;
  if (par->gga)
    {
      a2_g = DOUBLEP(a2_array);
      deda2_g = DOUBLEP(deda2_array);
    }

  /* h1 is the weight for the exchange part of the energy */
  double h1 = 1.0 - par->hybrid;
  /* h1v is the weight for the exchange part of the potential */
  double h1v = h1;
  if(self->correction) { 
    /* we assume this to be LB, increase of the exchange part, see 
       Schipper et al, J.Chem.Phys. 112, 1344 (2000) */
    h1v += par->pade[0] - 1.0;
  }

  for (int g = 0; g < ng; g++)
    {
      double n = n_g[g];
      if (n < NMIN)
        n = NMIN;
      double rs = pow(C0I / n, THIRD);
      double dexdrs;
      double dexda2;
      double ex;
      double decdrs;
      double decda2;
      double ec;
      if (par->gga)
        {
          double a2 = a2_g[g];
          ex = self->exchange(par, n, rs, a2, &dexdrs, &dexda2);
          ec = self->correlation(n, rs, 0.0, a2, 1, 0, &decdrs, 0, &decda2);
          deda2_g[g] = n * (dexda2 + decda2);
        }
      else
        {
          ex = self->exchange(par, n, rs, 0.0, &dexdrs, 0);
          ec = self->correlation(n, rs, 0.0, 0.0, 0, 0, &decdrs, 0, 0);
        }
      e_g[g] = n * (h1 * ex + ec);
      v_g[g] += h1v * ex + ec - rs * (h1v * dexdrs + decdrs) / 3.0;
      if(self->correction) {
	if(!par->gga) { /* we need to load the GGA arrays */
	  a2_g = DOUBLEP(a2_array);
	  deda2_g = DOUBLEP(deda2_array);
	}
	v_g[g] += self->correction(par, n, rs, a2_g[g], &dexdrs, &dexda2);
	deda2_g[g] = 0.0; /* avoid correction used in python gga code */
      }
    }
  Py_RETURN_NONE;
}

static PyObject* 
XCFunctional_CalculateSpinPolarized(XCFunctionalObject *self, PyObject *args)
{
  PyArrayObject* e;
  PyArrayObject* na;
  PyArrayObject* va;
  PyArrayObject* nb;
  PyArrayObject* vb;
  PyArrayObject* a2 = 0;
  PyArrayObject* aa2 = 0;
  PyArrayObject* ab2 = 0;
  PyArrayObject* deda2 = 0;
  PyArrayObject* dedaa2 = 0;
  PyArrayObject* dedab2 = 0;
  if (!PyArg_ParseTuple(args, "OOOOO|OOOOOO", &e, &na, &va, &nb, &vb,
                        &a2, &aa2, &ab2, &deda2, &dedaa2, &dedab2))
    return NULL;

  int ng = e->dimensions[0];
  double* e_g = DOUBLEP(e);
  const double* na_g = DOUBLEP(na);
  double* va_g = DOUBLEP(va);
  const double* nb_g = DOUBLEP(nb);
  double* vb_g = DOUBLEP(vb);

  const double* a2_g = 0;
  const double* aa2_g = 0;
  const double* ab2_g = 0;
  double* deda2_g = 0;
  double* dedaa2_g = 0;
  double* dedab2_g = 0;
  const xc_parameters* par = &self->par;
  if (par->gga)
    {
      a2_g = DOUBLEP(a2);
      aa2_g = DOUBLEP(aa2);
      ab2_g = DOUBLEP(ab2);
      deda2_g = DOUBLEP(deda2);
      dedaa2_g = DOUBLEP(dedaa2);
      dedab2_g = DOUBLEP(dedab2);
    }

  for (int g = 0; g < ng; g++)
    {
      double na = 2.0 * na_g[g];
      if (na < NMIN)
        na = NMIN;
      double rsa = pow(C0I / na, THIRD);
      double nb = 2.0 * nb_g[g];
      if (nb < NMIN)
        nb = NMIN;
      double rsb = pow(C0I / nb, THIRD);
      double n = 0.5 * (na + nb);
      double rs = pow(C0I / n, THIRD);
      double zeta = 0.5 * (na - nb) / n;
      double dexadrs;
      double dexada2;
      double exa;
      double dexbdrs;
      double dexbda2;
      double exb;
      double decdrs;
      double decdzeta;
      double decda2;
      double ec;
      if (par->gga)
        {
          exa = self->exchange(par, na, rsa, 4.0 * aa2_g[g],
			       &dexadrs, &dexada2);
          exb = self->exchange(par, nb, rsb, 4.0 * ab2_g[g],
			       &dexbdrs, &dexbda2);
          ec = self->correlation(n, rs, zeta, a2_g[g], 1, 1, 
				 &decdrs, &decdzeta, &decda2);
          dedaa2_g[g] = na * dexada2;
          dedab2_g[g] = nb * dexbda2;
          deda2_g[g] = n * decda2;
        }
      else
        {
          exa = self->exchange(par, na, rsa, 0.0, &dexadrs, 0);
          exb = self->exchange(par, nb, rsb, 0.0, &dexbdrs, 0);
          ec = self->correlation(n, rs, zeta, 0.0, 0, 1, 
				 &decdrs, &decdzeta, 0);
        }
      double h1 = 1.0 - par->hybrid;
      e_g[g] = 0.5 * h1 * (na * exa + nb * exb) + n * ec;
      va_g[g] += (h1 * exa + ec - (h1 * rsa * dexadrs + rs * decdrs) / 3.0 -
                  (zeta - 1.0) * decdzeta);
      vb_g[g] += (h1 * exb + ec - (h1 * rsb * dexbdrs + rs * decdrs) / 3.0 -
                  (zeta + 1.0) * decdzeta);
    }
  Py_RETURN_NONE;
}

static PyObject* 
XCFunctional_exchange(XCFunctionalObject *self, PyObject *args)
{
  double rs;
  double a2;
  if (!PyArg_ParseTuple(args, "dd", &rs, &a2))
    return NULL;

  double dedrs;
  double deda2;
  double n = 1.0 / (C0 * rs * rs * rs);
  double ex = self->exchange(&self->par, n, rs, a2, &dedrs, &deda2);
  if (!(self->par.gga))
    deda2 = 0.0;
  return Py_BuildValue("ddd", ex, dedrs, deda2); 
}

static PyObject* 
XCFunctional_correlation(XCFunctionalObject *self, PyObject *args)
{
  double rs;
  double zeta;
  double a2;
  if (!PyArg_ParseTuple(args, "ddd", &rs, &zeta, &a2)) 
    return NULL;

  double dedrs;
  double dedzeta;
  double deda2;
  double n = 1.0 / (C0 * rs * rs * rs);
  double ec = self->correlation(n, rs, zeta, a2, self->par.gga, 1,
				&dedrs, &dedzeta, &deda2);
  if (!(self->par.gga))
    deda2 = 0.0;
  return Py_BuildValue("dddd", ec, dedrs, dedzeta, deda2); 
}

static PyObject* 
XCFunctional_correlation0(XCFunctionalObject *self, PyObject *args)
{
  double rs;
  double a2;
  if (!PyArg_ParseTuple(args, "dd", &rs, &a2)) 
    return NULL;

  double dedrs;
  double dedzeta;
  double deda2;
  double n = 1.0 / (C0 * rs * rs * rs);
  double ec = self->correlation(n, rs, 0.0, a2, self->par.gga, 0,
				&dedrs, &dedzeta, &deda2);
  if (!(self->par.gga))
    deda2 = 0.0;
  return Py_BuildValue("dddd", ec, dedrs, 0.0, deda2); 
}

static PyMethodDef XCFunctional_Methods[] = {
    {"calculate_spinpaired", 
     (PyCFunction)XCFunctional_CalculateSpinPaired, METH_VARARGS, 0},
    {"calculate_spinpolarized", 
     (PyCFunction)XCFunctional_CalculateSpinPolarized, METH_VARARGS, 0},
    {"exchange", (PyCFunction)XCFunctional_exchange, METH_VARARGS, 0},
    {"correlation", (PyCFunction)XCFunctional_correlation, METH_VARARGS, 0},
    {"correlation0", (PyCFunction)XCFunctional_correlation0, METH_VARARGS, 0},
    {NULL, NULL, 0, NULL}
};

static PyObject* XCFunctional_getattr(PyObject *obj, char *name)
{
    return Py_FindMethod(XCFunctional_Methods, obj, name);
}

static PyTypeObject XCFunctionalType = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,
  "XCFunctional",
  sizeof(XCFunctionalObject),
  0,
  (destructor)XCFunctional_dealloc,
  0,
  XCFunctional_getattr
};

PyObject * NewXCFunctionalObject(PyObject *obj, PyObject *args)
{
  int type;
  int gga;
  double s0 = 1.0;
  int i = -1;
  PyArrayObject* padearray = 0;
  if (!PyArg_ParseTuple(args, "ii|diO", &type, &gga, &s0, &i, &padearray))
    return NULL;

  XCFunctionalObject *self = PyObject_NEW(XCFunctionalObject,
					  &XCFunctionalType);
  if (self == NULL)
    return NULL;

  self->par.gga = gga;
  self->par.hybrid = 0.0;

  self->correlation = pbe_correlation;
  self->correction = NULL;

  if (type == 2)
    {
      // RPBE
      self->exchange = rpbe_exchange;
    }
  else if (type == 3)
    {
      self->exchange = ensemble_exchange;
      self->par.i = i;
      self->par.s0 = s0;
    }
  else if (type == 4)
    {
      // PBE0
      self->exchange = pbe_exchange;
      self->par.hybrid = 0.25;
      self->par.kappa = 0.804;
    }
  else if (type == 5)
    {
      // PADE
      self->exchange = pade_exchange;
      int n = padearray->dimensions[0];
      double* p = DOUBLEP(padearray);
      for (int i = 0; i < n; i++)
	self->par.pade[i] = p[i];
    }
  else if (type == 7)
    {
      // LDAc
      self->exchange = zero_exchange;
    }
  else if (type == 8)
    {
      // revPBEx
      self->par.kappa = 1.245; 
      self->exchange = pbe_exchange;
      self->correlation = zero_correlation;
    }
  else if (type == 12)
    {
      // RPBEx
      self->exchange = rpbe_exchange;
      self->correlation = zero_correlation;
    }
  else if (type == 14)
    {
      // PW91
      self->exchange = pw91_exchange;
      self->correlation = pw91_correlation;
    }
  else if (type == 11)
    {
      // LDAx
      self->exchange = pbe_exchange;
      self->correlation = zero_correlation;
    }
  else if (type == 17)
    {
      // LB94
      self->par.gga = 0; /* hide gga */
      self->exchange = pbe_exchange;
      self->correction = lb94_correction;
      int n = padearray->dimensions[0];
      assert(n == 2);
      double* p = DOUBLEP(padearray);
      for (int i = 0; i < n; i++) {
	printf("-------- p[%d]=%g\n", i, p[i]);
	self->par.pade[i] = p[i];
      }
      
    }
  else if (type == 18)
    {
      // BEE1
      self->exchange = bee1_exchange;
      int n = padearray->dimensions[0];
      assert(n == 1);
      double* p = DOUBLEP(padearray);
      for (int i = 0; i < n; i++)
	self->par.pade[i] = p[i];
    }
  else
    {
      if (type == 1)
	// revPBE
        self->par.kappa = 1.245; 
      else
	// PBE
	self->par.kappa = 0.804;

      self->exchange = pbe_exchange;
    }
  return (PyObject*)self;
}
