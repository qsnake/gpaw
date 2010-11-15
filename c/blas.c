/*  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2007-2009  CAMd
 *  Copyright (C) 2007       CSC - IT Center for Science Ltd.
 *  Please see the accompanying LICENSE file for further information. */

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "extensions.h"

#ifdef GPAW_NO_UNDERSCORE_BLAS
#  define dscal_  dscal
#  define zscal_  zscal
#  define daxpy_  daxpy
#  define zaxpy_  zaxpy
#  define dsyrk_  dsyrk
#  define zherk_  zherk
#  define dsyr2k_ dsyr2k
#  define zher2k_ zher2k
#  define dgemm_  dgemm
#  define zgemm_  zgemm
#  define dgemv_  dgemv
#  define zgemv_  zgemv
#  define ddot_   ddot
#endif

void dscal_(int*n, double* alpha, double* x, int* incx);

void zscal_(int*n, void* alpha, void* x, int* incx);

void daxpy_(int* n, double* alpha,
	    double* x, int *incx,
	    double* y, int *incy);
void zaxpy_(int* n, void* alpha,
	    void* x, int *incx,
	    void* y, int *incy);

void dsyrk_(char *uplo, char *trans, int *n, int *k,
	    double *alpha, double *a, int *lda, double *beta,
	    double *c, int *ldc);
void zherk_(char *uplo, char *trans, int *n, int *k,
	    double *alpha, void *a, int *lda,
	    double *beta,
	    void *c, int *ldc);
void dsyr2k_(char *uplo, char *trans, int *n, int *k,
	     double *alpha, double *a, int *lda,
	     double *b, int *ldb, double *beta,
	     double *c, int *ldc);
void zher2k_(char *uplo, char *trans, int *n, int *k,
	     void *alpha, void *a, int *lda,
	     void *b, int *ldb, double *beta,
	     void *c, int *ldc);
void dgemm_(char *transa, char *transb, int *m, int * n,
	    int *k, double *alpha, double *a, int *lda,
	    double *b, int *ldb, double *beta,
	    double *c, int *ldc);
void zgemm_(char *transa, char *transb, int *m, int * n,
	    int *k, void *alpha, void *a, int *lda,
	    void *b, int *ldb, void *beta,
	    void *c, int *ldc);
void dgemv_(char *trans, int *m, int * n,
	    double *alpha, double *a, int *lda,
	    double *x, int *incx, double *beta,
	    double *y, int *incy);
void zgemv_(char *trans, int *m, int * n,
	    void *alpha, void *a, int *lda,
	    void *x, int *incx, void *beta,
	    void *y, int *incy);
double ddot_(int *n, void *dx, int *incx, void *dy, int *incy);

PyObject* scal(PyObject *self, PyObject *args)
{
  Py_complex alpha;
  PyArrayObject* x;
  if (!PyArg_ParseTuple(args, "DO", &alpha, &x))
    return NULL;
  int n = x->dimensions[0];
  for (int d = 1; d < x->nd; d++)
    n *= x->dimensions[d];
  int incx = 1;

  if (x->descr->type_num == PyArray_DOUBLE)
    dscal_(&n, &(alpha.real), DOUBLEP(x), &incx);
  else
    zscal_(&n, &alpha, (void*)COMPLEXP(x), &incx);

  Py_RETURN_NONE;
}

PyObject* gemm(PyObject *self, PyObject *args)
{
  Py_complex alpha;
  PyArrayObject* a;
  PyArrayObject* b;
  Py_complex beta;
  PyArrayObject* c;
  char transa = 'n';
  if (!PyArg_ParseTuple(args, "DOODO|c", &alpha, &a, &b, &beta, &c, &transa))
    return NULL;
  int m, k, lda, ldb, ldc;
  if (transa == 'n')
    {
      m = a->dimensions[1];
      for (int i = 2; i < a->nd; i++)
	m *= a->dimensions[i];
      k = a->dimensions[0];
      lda = MAX(1, m);
      ldb = b->strides[0] / b->strides[1];
      ldc = c->strides[0] / c->strides[c->nd - 1];
    }
  else
    {
      k = a->dimensions[1];
      for (int i = 2; i < a->nd; i++)
	k *= a->dimensions[i];
      m = a->dimensions[0];
      lda = MAX(1, k);
      ldb = lda;
      ldc = c->strides[0] / c->strides[1];
    }
  int n = b->dimensions[0];
  if (a->descr->type_num == PyArray_DOUBLE)
    dgemm_(&transa, "n", &m, &n, &k,
           &(alpha.real),
           DOUBLEP(a), &lda,
           DOUBLEP(b), &ldb,
           &(beta.real),
           DOUBLEP(c), &ldc);
  else
    zgemm_(&transa, "n", &m, &n, &k,
           &alpha,
           (void*)COMPLEXP(a), &lda,
           (void*)COMPLEXP(b), &ldb,
           &beta,
           (void*)COMPLEXP(c), &ldc);
  Py_RETURN_NONE;
}


PyObject* gemv(PyObject *self, PyObject *args)
{
  Py_complex alpha;
  PyArrayObject* a;
  PyArrayObject* x;
  Py_complex beta;
  PyArrayObject* y;
  char trans = 't';
  if (!PyArg_ParseTuple(args, "DOODO|c", &alpha, &a, &x, &beta, &y, &trans))
    return NULL;

  int m, n, lda, itemsize, incx, incy;

  if (trans == 'n')
    {
      m = a->dimensions[1];
      for (int i = 2; i < a->nd; i++)
	m *= a->dimensions[i];
      n = a->dimensions[0];
      lda = m;
    }
  else
    {
      n = a->dimensions[0];
      for (int i = 1; i < a->nd-1; i++)
	n *= a->dimensions[i];
      m = a->dimensions[a->nd-1];
      lda = m;
    }

  if (a->descr->type_num == PyArray_DOUBLE)
    itemsize = sizeof(double);
  else
    itemsize = sizeof(double_complex);

  incx = x->strides[0]/itemsize;
  incy = 1;

  if (a->descr->type_num == PyArray_DOUBLE)
    dgemv_(&trans, &m, &n,
           &(alpha.real),
           DOUBLEP(a), &lda,
           DOUBLEP(x), &incx,
           &(beta.real),
           DOUBLEP(y), &incy);
  else
    zgemv_(&trans, &m, &n,
           &alpha,
           (void*)COMPLEXP(a), &lda,
           (void*)COMPLEXP(x), &incx,
           &beta,
           (void*)COMPLEXP(y), &incy);
  Py_RETURN_NONE;
}


PyObject* axpy(PyObject *self, PyObject *args)
{
  Py_complex alpha;
  PyArrayObject* x;
  PyArrayObject* y;
  if (!PyArg_ParseTuple(args, "DOO", &alpha, &x, &y))
    return NULL;
  int n = x->dimensions[0];
  for (int d = 1; d < x->nd; d++)
    n *= x->dimensions[d];
  int incx = 1;
  int incy = 1;
  if (x->descr->type_num == PyArray_DOUBLE)
    daxpy_(&n, &(alpha.real),
           DOUBLEP(x), &incx,
           DOUBLEP(y), &incy);
  else
    zaxpy_(&n, &alpha,
           (void*)COMPLEXP(x), &incx,
           (void*)COMPLEXP(y), &incy);
  Py_RETURN_NONE;
}

PyObject* rk(PyObject *self, PyObject *args)
{
  double alpha;
  PyArrayObject* a;
  double beta;
  PyArrayObject* c;
  if (!PyArg_ParseTuple(args, "dOdO", &alpha, &a, &beta, &c))
    return NULL;
  int n = a->dimensions[0];
  int k = a->dimensions[1];
  for (int d = 2; d < a->nd; d++)
    k *= a->dimensions[d];
  int ldc = c->strides[0] / c->strides[1];
  if (a->descr->type_num == PyArray_DOUBLE)
    dsyrk_("u", "t", &n, &k,
           &alpha, DOUBLEP(a), &k, &beta,
           DOUBLEP(c), &ldc);
  else
    zherk_("u", "c", &n, &k,
           &alpha, (void*)COMPLEXP(a), &k, &beta,
           (void*)COMPLEXP(c), &ldc);
  Py_RETURN_NONE;
}

PyObject* r2k(PyObject *self, PyObject *args)
{
  Py_complex alpha;
  PyArrayObject* a;
  PyArrayObject* b;
  double beta;
  PyArrayObject* c;
  if (!PyArg_ParseTuple(args, "DOOdO", &alpha, &a, &b, &beta, &c))
    return NULL;
  int n = a->dimensions[0];
  int k = a->dimensions[1];
  for (int d = 2; d < a->nd; d++)
    k *= a->dimensions[d];
  int ldc = c->strides[0] / c->strides[1];
  if (a->descr->type_num == PyArray_DOUBLE)
    dsyr2k_("u", "t", &n, &k,
            (double*)(&alpha), DOUBLEP(a), &k,
            DOUBLEP(b), &k, &beta,
            DOUBLEP(c), &ldc);
  else
    zher2k_("u", "c", &n, &k,
            (void*)(&alpha), (void*)COMPLEXP(a), &k,
            (void*)COMPLEXP(b), &k, &beta,
            (void*)COMPLEXP(c), &ldc);
  Py_RETURN_NONE;
}

PyObject* dotc(PyObject *self, PyObject *args)
{
  PyArrayObject* a;
  PyArrayObject* b;
  if (!PyArg_ParseTuple(args, "OO", &a, &b))
    return NULL;
  int n = a->dimensions[0];
  for (int i = 1; i < a->nd; i++)
    n *= a->dimensions[i];
  int incx = 1;
  int incy = 1;
  if (a->descr->type_num == PyArray_DOUBLE)
    {
      double result;
      result = ddot_(&n, (void*)DOUBLEP(a),
	     &incx, (void*)DOUBLEP(b), &incy);
      return PyFloat_FromDouble(result);
    }
  else
    {
      double_complex* ap = COMPLEXP(a);
      double_complex* bp = COMPLEXP(b);
      double_complex z = 0.0;
      for (int i = 0; i < n; i++)
	z += conj(ap[i]) * bp[i];
      return PyComplex_FromDoubles(creal(z), cimag(z));
    }
}


PyObject* dotu(PyObject *self, PyObject *args)
{
  PyArrayObject* a;
  PyArrayObject* b;
  if (!PyArg_ParseTuple(args, "OO", &a, &b))
    return NULL;
  int n = a->dimensions[0];
  for (int i = 1; i < a->nd; i++)
    n *= a->dimensions[i];
  int incx = 1;
  int incy = 1;
  if (a->descr->type_num == PyArray_DOUBLE)
    {
      double result;
      result = ddot_(&n, (void*)DOUBLEP(a),
	     &incx, (void*)DOUBLEP(b), &incy);
      return PyFloat_FromDouble(result);
    }
  else
    {
      double_complex* ap = COMPLEXP(a);
      double_complex* bp = COMPLEXP(b);
      double_complex z = 0.0;
      for (int i = 0; i < n; i++)
	z += ap[i] * bp[i];
      return PyComplex_FromDoubles(creal(z), cimag(z));
    }
}

PyObject* multi_dotu(PyObject *self, PyObject *args)
{
  PyArrayObject* a;
  PyArrayObject* b;
  PyArrayObject* c;
  if (!PyArg_ParseTuple(args, "OOO", &a, &b, &c)) 
    return NULL;
  int n0 = a->dimensions[0];
  int n = a->dimensions[1];
  for (int i = 2; i < a->nd; i++)
    n *= a->dimensions[i];
  int incx = 1;
  int incy = 1;
  if (a->descr->type_num == PyArray_DOUBLE)
    {
      double *ap = DOUBLEP(a);
      double *bp = DOUBLEP(b);
      double *cp = DOUBLEP(c);

      for (int i = 0; i < n0; i++)
	{
	  cp[i] = ddot_(&n, (void*)ap, 
	     &incx, (void*)bp, &incy);
	  ap += n;
	  bp += n;
	}
    }
  else
    {
      double_complex* ap = COMPLEXP(a);
      double_complex* bp = COMPLEXP(b);
      double_complex* cp = COMPLEXP(c);
      for (int i = 0; i < n0; i++)
	{
	  cp[i] = 0.0;
	  for (int j = 0; j < n; j++)
	      cp[i] += ap[j] * bp[j];
	  ap += n;
	  bp += n;
	}
    }
  Py_RETURN_NONE;
}

PyObject* multi_axpy(PyObject *self, PyObject *args)
{
  PyArrayObject* alpha;
  PyArrayObject* x;
  PyArrayObject* y;
  if (!PyArg_ParseTuple(args, "OOO", &alpha, &x, &y)) 
    return NULL;
  int n0 = x->dimensions[0];
  int n = x->dimensions[1];
  for (int d = 2; d < x->nd; d++)
    n *= x->dimensions[d];
  int incx = 1;
  int incy = 1;

   if (alpha->descr->type_num == PyArray_DOUBLE)
    {
      if (x->descr->type_num == PyArray_CDOUBLE)
	n *= 2;
      double *ap = DOUBLEP(alpha);
      double *xp = DOUBLEP(x);
      double *yp = DOUBLEP(y);
      for (int i = 0; i < n0; i++)
	{
	  daxpy_(&n, &ap[i], 
		 (void*)xp, &incx,
		 (void*)yp, &incy);
	  xp += n;
	  yp += n;
	}
    }
  else
    {
      double_complex *ap = COMPLEXP(alpha);
      double_complex *xp = COMPLEXP(x);
      double_complex *yp = COMPLEXP(y);
      for (int i = 0; i < n0; i++)
	{
	  zaxpy_(&n, (void*)(&ap[i]), 
		 (void*)xp, &incx,
		 (void*)yp, &incy);
	  xp += n;
	  yp += n;
	}
    }
  Py_RETURN_NONE;
}
