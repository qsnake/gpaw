#include <Python.h>
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
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
// dedtau = dexc/dtaua
//        
//  OUTPUT SPIN POLARIZED:
//
// va = dexc/dna
//
// dedaa2 = 1/2  dexc/daa2 - 1/4 dexc/dgab
//
// deda2 = 1/2 dexc/dgab
//
// detaua = dexc/dtaua

double atpss_exchange(double n, double a2, double tau,
		      double* dedn, double* deda2, double* detaua);
double atpss_correlation(double na, double nb, double aa2,
			 double ab2, double a2, double taua, double taub,
			 bool spinpol, 
			 double* dedna, double* dednb, double* dedaa2,
			 double* dedab2, double* dedgud,double* detaua,
			 double* detaub);
double tpss_exchange(double n, double a2, double tau, double* dedn, 
		     double* deda2, double* detaua);
double tpss_correlation(double na, double nb, double aa2,
			double ab2, double a2, double taua, double taub,
			bool spinpol, 
			double* dedna, double* dednb, double* dedaa2,
			double* dedab2, double* dedgud,double* detaua,
			double* detaub);

typedef struct 
{
  PyObject_HEAD
  double (*exchange)(double n, double a2, double tau,
		     double* dedn, double* deda2, double* dedtaua);
  double (*correlation)(double nu, double nd, double aa2,
			double ab2, double gud, double taua, double taub,
			bool spinpol, 
 			double* dednu, double* dednd, double* dedaa2,
			double* dedab2, double* dedgud,double* detaua,
			double* detaub); 
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
  PyArrayObject* dedtaua_array = 0;
  PyArrayObject* tau_array = 0;
  if (!PyArg_ParseTuple(args, "OOOOOOO", &e_array, &n_array, &v_array,
			&a2_array, &deda2_array, &tau_array,&dedtaua_array))
    return NULL;

  int ng = e_array->dimensions[0];
  const xc_parameters* par = &self->par; 

  double* e_g = DOUBLEP(e_array);
  const double* n_g = DOUBLEP(n_array);
  double* v_g = DOUBLEP(v_array);

  const double* a2_g = DOUBLEP(a2_array);
  double* deda2_g = DOUBLEP(deda2_array);
  double* dedtaua_g = DOUBLEP(dedtaua_array);
  double* tau_g;
  tau_g = DOUBLEP(tau_array);


  for (int g = 0; g < ng; g++)
    {
      double n = n_g[g];
      if (n < NMIN) 
	n = NMIN; 
      double a2 = a2_g[g];
      if (a2 < NMIN2) 
 	a2 = NMIN2; 
      double dexdn;  /*dex/d(2*na)*/
      double dexda2; /*dex/d(4*aa2)*/
      double ex;
      double decdna;
      double decdaa2;
      double decdgab;
      double ec;
      double temp;
      double dexdtaua;
      double decdtaua;
      double tau = tau_g[g]; 
      if (par->mgga)
	{
	  tau = a2/ (8. * n);
	  tau_g[g] = tau;
	}
      if (tau < a2/ (8. * n))
	{
	  tau = a2/ (8. * n);
	  tau_g[g] = tau;
	} 
      ex = self->exchange(n, a2, tau, &dexdn, &dexda2, &dexdtaua);
      ec = self->correlation(n / 2., 0, a2 / 4. , 0, 0, 0.5 * tau, 0.5 * tau,
			     0, &decdna, 
			     &temp, &decdaa2, &temp, &decdgab,&decdtaua,&temp); 

      deda2_g[g] =  dexda2 + 0.5 * decdaa2 + 0.25 * decdgab;
      e_g[g] =  ex +  ec;
      v_g[g] +=  decdna +  dexdn;
      dedtaua_g[g] = dexdtaua + decdtaua;
      }  
  Py_RETURN_NONE;
}

static PyObject* 
MGGAFunctional_CalculateSpinPolarized(MGGAFunctionalObject *self, PyObject *args)
{
  PyArrayObject* e_array;
  PyArrayObject* na_array;
  PyArrayObject* va_array;
  PyArrayObject* nb_array;
  PyArrayObject* vb_array;
  PyArrayObject* a2_array = 0;
  PyArrayObject* aa2_array = 0;
  PyArrayObject* ab2_array = 0;
  PyArrayObject* deda2_array = 0;
  PyArrayObject* dedaa2_array = 0;
  PyArrayObject* dedab2_array = 0;
  PyArrayObject* taua_array = 0;
  PyArrayObject* taub_array = 0;
  PyArrayObject* dedtaua_array = 0;
  PyArrayObject* dedtaub_array = 0;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOOO", &e_array, &na_array, &va_array
			, &nb_array, &vb_array, &a2_array, &aa2_array,
			&ab2_array, &deda2_array, &dedaa2_array, &dedab2_array,
			&taua_array, &taub_array, &dedtaua_array,
			&dedtaub_array))
    return NULL;

  const xc_parameters* par = &self->par; 
  int ng = e_array->dimensions[0];
  double* e_g = DOUBLEP(e_array);
  const double* na_g = DOUBLEP(na_array);
  double* va_g = DOUBLEP(va_array);
  const double* nb_g = DOUBLEP(nb_array);
  double* vb_g = DOUBLEP(vb_array);

  const double* a2_g = DOUBLEP(a2_array);
  const double* aa2_g = DOUBLEP(aa2_array);
  const double* ab2_g = DOUBLEP(ab2_array);
  double* deda2_g = DOUBLEP(deda2_array);
  double* dedaa2_g = DOUBLEP(dedaa2_array);
  double* dedab2_g = DOUBLEP(dedab2_array);
  double* taua_g ;
  double* taub_g ;
  taua_g = DOUBLEP(taua_array);
  taub_g = DOUBLEP(taub_array);
  double* dedtaua_g = DOUBLEP(dedtaua_array);
  double* dedtaub_g = DOUBLEP(dedtaub_array);
    
  for (int g = 0; g < ng; g++)
    {
      double na = 2.0 * na_g[g];
      if (na <  NMIN) 
	na =  NMIN; 
      double nb = 2.0 * nb_g[g]; 
      if (nb <  NMIN) 
	nb = NMIN; 
      double aa2 = aa2_g[g];  
      double ab2 = ab2_g[g];
      double a2 = a2_g[g];   
      if (aa2 < NMIN2) 
	aa2 = NMIN2; 
      if (ab2 < NMIN2) 
	ab2 = NMIN2; 
      if (a2 < NMIN2) 
	a2 = NMIN2; 
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
      double taua = taua_g[g];
      double taub = taub_g[g];
      double dexdtaua;
      double dexdtaub;
      double decdtaua;
      double decdtaub;
      if (par->mgga)
	{
	  taua = aa2 / (4.* na);
	  taub = ab2 / (4.* nb);
	  taua_g[g] = taua;
	  taub_g[g] = taub;
	}
      if (taua < aa2 / (4.* na))
	{
	  taua_g[g] =  aa2 / (4.* na);
	  taua = aa2 / (4.* na);
	}
      
      if (taub < ab2 / (4.* nb))
	{
	  taub_g[g] = ab2 / (4.* nb);
	  taub = ab2 / (4.* nb);
	}
      
      exa = self->exchange(na, 4.*aa2, 2.*taua, &dexdna, &dexada2, 
			   &dexdtaua);
      exb = self->exchange(nb, 4.*ab2, 2.*taub, &dexdnb, &dexbda2, &dexdtaub);
      ec = self->correlation(0.5 * na, 0.5 * nb, aa2, ab2, a2, taua, taub,
			     1, &decdna,
			     &decdnb, &decdaa2, &decdab2, &decdgab,
			     &decdtaua,&decdtaub);

      dedaa2_g[g] = dexada2 + 0.5 * decdaa2 - 0.25 * decdgab;
      dedab2_g[g] = dexbda2 + 0.5 * decdab2 - 0.25 * decdgab;
      deda2_g[g] = 0.5 * decdgab;
      dedtaua_g[g] = dexdtaua+decdtaua;
      dedtaub_g[g] = dexdtaub+decdtaub;

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
  /* mgga is True if tau is the weiszacker term*/
  self->par.mgga = mgga;

  if (mgga)
    {
      //TPSS local
      self->exchange = atpss_exchange;
      self->correlation = atpss_correlation;
    }
  else
    {
      //TPSS non local
      self->exchange = tpss_exchange;
      self->correlation = tpss_correlation;
    }

  return (PyObject*)self;
}
