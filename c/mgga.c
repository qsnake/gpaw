#include <Python.h>
#define NO_IMPORT_ARRAY
#include <Numeric/arrayobject.h>
#include "mgga.h"
#include "extensions.h"

//          __  2
// a2    = |\/n|
//          __   2
// aa2   = |\/na|
//          __   __
// gab   =  \/na.\/nb
//
//
//  OUTPUT SPINPAIRED:
//
//  v = dexc/dna
//
// deda2 = 1/2 dexc/daa2 
//        
//        
//  OUTPUT SPIN POLARIZED:
//
// va = dexc/dna
//
// dedaa2 = 1/2  dexc/daa2 - 1/4 dexc/dgab
//
// deda2 = 1/2 dexc/dgab
//

double atpss_exchange(double n, double a2, double tau,
		     double* dedn, double* deda2);
double atpss_correlation(double na, double nb, double aa2,
			 double ab2, double a2, double taua, double taub,
			 bool spinpol, 
			 double* dedna, double* dednb, double* dedaa2,
			 double* dedab2, double* dedgud);

typedef struct 
{
  PyObject_HEAD
  double (*exchange)(double n, double a2, double tau,
		     double* dedn, double* deda2);
  double (*correlation)(double nu, double nd, double aa2,
			double ab2, double gud, double taua, double taub,
			bool spinpol, 
 			double* dednu, double* dednd, double* dedaa2,
			double* dedab2, double* dedgud); 
  xc_parameters par;
} MGGAFunctionalObject;

static void MGGAFunctional_dealloc(MGGAFunctionalObject *self)
{
  PyObject_DEL(self);
}

static PyObject* 
MGGAFunctional_CalculateSpinPaired(MGGAFunctionalObject *self, PyObject *args)
{
  PyArrayObject* e_array;
  PyArrayObject* n_array;
  PyArrayObject* v_array;
  PyArrayObject* a2_array = 0;
  PyArrayObject* deda2_array = 0;
  PyArrayObject* tau_array = 0;
  if (!PyArg_ParseTuple(args, "OOOOO|O", &e_array, &n_array, &v_array,
			&a2_array, &deda2_array, &tau_array))
    return NULL;

  int ng = e_array->dimensions[0];
  const xc_parameters* par = &self->par; 

  double* e_g = DOUBLEP(e_array);
  const double* n_g = DOUBLEP(n_array);
  double* v_g = DOUBLEP(v_array);

  const double* a2_g = DOUBLEP(a2_array);
  double* deda2_g = DOUBLEP(deda2_array);

  double* tau_g;
  if (par->mgga)
    tau_g = DOUBLEP(tau_array);


  for (int g = 0; g < ng; g++)
    {
      double n = n_g[g];
      if (n < NMIN)
        n = NMIN;
      double a2 = a2_g[g];
      if (a2 < 4 * NMIN)
	a2 = 4 * NMIN;
      double dexdn;  /*dex/d(2*na)*/
      double dexda2; /*dex/d(4*aa2)*/
      double ex;
      double decdna;
      double decdaa2;
      double decdgab;
      double ec;
      double temp;
      double tau = -1.0; 
      if (par->mgga)
	{
	  if (tau_g[g] != -1.0)
	    {
	      tau = tau_g[g];  
	      if (tau < a2 / (16.* n))
		{
		  tau = a2/ (16. * n);
		}
	    }
	}

      ex = self->exchange(n, a2, 2*tau, &dexdn, &dexda2);
      ec = self->correlation(n / 2, 0, a2 / 4 , 0, 0, tau, tau, 0, &decdna, 
			      &temp, &decdaa2, &temp, &decdgab); 

      deda2_g[g] =  dexda2 + 0.5 * decdaa2 + 0.25 * decdgab;
      e_g[g] =  ex +  ec;
      v_g[g] +=  decdna +  dexdn;

      }  
  Py_RETURN_NONE;
}

static PyObject* 
MGGAFunctional_CalculateSpinPolarized(MGGAFunctionalObject *self, PyObject *args)
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
  PyArrayObject* taua = 0;
  PyArrayObject* taub = 0;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOOO|OO", &e, &na, &va, &nb, &vb,
                        &a2, &aa2, &ab2, &deda2, &dedaa2, &dedab2,
			&taua, &taub))
    return NULL;

  const xc_parameters* par = &self->par; 
  int ng = e->dimensions[0];
  double* e_g = DOUBLEP(e);
  const double* na_g = DOUBLEP(na);
  double* va_g = DOUBLEP(va);
  const double* nb_g = DOUBLEP(nb);
  double* vb_g = DOUBLEP(vb);

  const double* a2_g = DOUBLEP(a2);
  const double* aa2_g = DOUBLEP(aa2);
  const double* ab2_g = DOUBLEP(ab2);
  double* deda2_g = DOUBLEP(deda2);
  double* dedaa2_g = DOUBLEP(dedaa2);
  double* dedab2_g = DOUBLEP(dedab2);
  double* taua_g ;
  double* taub_g ;
  if (par->mgga)
    {
      taua_g = DOUBLEP(taua);
      taub_g = DOUBLEP(taub);
    }

  for (int g = 0; g < ng; g++)
    {
      double na = 2.0 * na_g[g];
      if (na < 2 * NMIN)
        na = 2 * NMIN;
      double nb = 2.0 * nb_g[g];
      if (nb < 2 * NMIN)
        nb = 2 * NMIN;
      double aa2 = aa2_g[g];
      if (aa2 < NMIN)
        aa2 = NMIN;
      double ab2 = ab2_g[g];
      if (ab2 < NMIN)
        ab2 = NMIN;
      double a2 = a2_g[g];
      if (a2 < 4 * NMIN)
        a2 = 4 * NMIN;
      double dexdna;  /*dex/d(2*na)*/
      double dexada2; /*dex/d(4*aa2)*/
      double exa;
      double dexdnb;
      double dexbda2;
      double exb;
      double decdna;
      double decdnb;
      double decdaa2;
      double decdab2;
      double decdgab;
      double ec;
      double taua = -1.0;
      double taub = -1.0;
      if (par->mgga)
	{
	  if (taua_g[g] != -1.0)
	    {
	      taua = taua_g[g];
	      taub = taub_g[g];  
	      if (taua < aa2 / (4.* na))
		{
		  taua = aa2 / (4.* na);
		}
	      if (taub < ab2 / (4.* nb))
		{
		  taub = ab2 / (4.* nb);
		}
	    }
	}


      exa = self->exchange(na, 4.0 * aa2, 2 * taua, &dexdna, &dexada2);
      exb = self->exchange(nb, 4.0 * ab2, 2 * taub, &dexdnb, &dexbda2);
      ec = self->correlation(na / 2, nb / 2, aa2, ab2, a2, taua, taub, 
			     1, &decdna,
			     &decdnb, &decdaa2, &decdab2, &decdgab);

      dedaa2_g[g] = dexada2 + 0.5 * decdaa2 - 0.25 * decdgab;
      dedab2_g[g] = dexbda2 + 0.5 * decdab2 - 0.25 * decdgab;
      deda2_g[g] = 0.5 * decdgab;

      e_g[g] = 0.5 * (exa + exb) + ec;
      va_g[g] +=  dexdna + decdna ;
      vb_g[g] +=  dexdnb + decdnb;
    }
  Py_RETURN_NONE;
}

static PyObject*
MGGAFunctional_exchange(MGGAFunctionalObject *self, PyObject *args)
{
  return NULL;
}

static PyObject*
MGGAFunctional_correlation(MGGAFunctionalObject *self, PyObject *args)
{
  return NULL;
}

static PyMethodDef MGGAFunctional_Methods[] = {
    {"calculate_spinpaired", 
     (PyCFunction)MGGAFunctional_CalculateSpinPaired, METH_VARARGS, 0},
    {"calculate_spinpolarized", 
     (PyCFunction)MGGAFunctional_CalculateSpinPolarized, METH_VARARGS, 0},
    {"exchange", (PyCFunction)MGGAFunctional_exchange, METH_VARARGS, 0},
    {"correlation", (PyCFunction)MGGAFunctional_correlation, METH_VARARGS, 0},
    {NULL, NULL, 0, NULL}
};

static PyObject* MGGAFunctional_getattr(PyObject *obj, char *name)
{
    return Py_FindMethod(MGGAFunctional_Methods, obj, name);
}

static PyTypeObject MGGAFunctionalType = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,
  "MGGAFunctional",
  sizeof(MGGAFunctionalObject),
  0,
  (destructor)MGGAFunctional_dealloc,
  0,
  MGGAFunctional_getattr
};

PyObject * NewMGGAFunctionalObject(PyObject *obj, PyObject *args)
{
  int type;
  int mgga;

  if (!PyArg_ParseTuple(args, "i|i", &type, &mgga))
    return NULL;

  MGGAFunctionalObject *self = PyObject_NEW(MGGAFunctionalObject,
					  &MGGAFunctionalType);
  if (self == NULL)
    return NULL;
  self->par.mgga = mgga;

  if (type == 9)
    {
      //TPSS local
      self->exchange = atpss_exchange;
      self->correlation = atpss_correlation;
    }

  return (PyObject*)self;
}
