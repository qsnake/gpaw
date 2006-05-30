/* second derivatives of the exchange-correlation energy 
 */
#include <Python.h>
#include "extensions.h"

double d2ecdnsdnt_(double*,double*,const int*,const int *); 
double d2exdnsdnt_(double*,double*,const int*,const int *); 
double d2ecdrho2_u__(double*);
double d2exdn2_(double);

/* unpolarised */
PyObject* d2Excdn2(PyObject *self, PyObject *args)
{
  PyArrayObject* den; /* density */
  PyArrayObject* res; /* derivative */
  
  if (!PyArg_ParseTuple(args, "OO", &den, &res)) 
    return NULL;

  int n = den->dimensions[0];
/*   printf("<d2Excdn2> nd=%d\n",den->nd); */
  for (int d = 1; d < den->nd; d++)
    n *= den->dimensions[d];
  double *denp = DOUBLEP(den);
  double *resp = DOUBLEP(res);
  
  for (int i=0; i<n;i++) {
    resp[i] = d2exdn2_(denp[i])
      + d2ecdrho2_u__(denp+i);
  }
  
  Py_RETURN_NONE;
}

/* polarised */
PyObject* d2Excdnsdnt(PyObject *self, PyObject *args)
{
  PyArrayObject* dup; /* "up" density */
  PyArrayObject* ddn; /* "down" density */
  int is;             /* i spin (0,1) */
  int ks;             /* k spin (0,1) */
  PyArrayObject* res; /* derivative */
 
/*   printf("<d2Excdnsdnt> is=%p ks=%p\n",is,ks); */
  if (!PyArg_ParseTuple(args, "OOiiO", &dup, &ddn, &is, &ks, &res)) 
    return NULL;
/*   printf("<d2Excdnsdnt> args passed\n"); */
/*   printf("<d2Excdnsdnt> is=%d ks=%d\n",is,ks); */

  int n = dup->dimensions[0];
  for (int d = 1; d < dup->nd; d++)
    n *= dup->dimensions[d];
/*   printf("<d2Excdnsdnt> n=%d\n",n); */
  double *dupp = DOUBLEP(dup);
  double *ddnp = DOUBLEP(ddn);
/*   printf("<d2Excdnsdnt> dupp=%p ddnp=%p\n",dupp,ddnp); */
  
  double *resp = DOUBLEP(res);

/*   int iis = *is */
/*   int iks = *ks; */
/*   printf("<d2Excdnsdnt> is=%i ks=%i\n",iis,iks); */

  for (int i=0; i<n;i++) {
    resp[i] = d2exdnsdnt_(dupp+i,ddnp+i,&is,&ks)
      + d2ecdnsdnt_(dupp+i,ddnp+i,&is,&ks);
/*     printf("<d2Excdnsdnt> i=%d dup=%g ddn=%g resp=%g\n",i, */
/* 	   dupp[i],ddnp[i],resp[i]); */
  }

   Py_RETURN_NONE;
 }

