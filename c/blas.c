#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "extensions.h"

#ifdef GPAW_AIX
#  define daxpy_ daxpy
#  define zaxpy_ zaxpy
#  define dsyrk_ dsyrk
#  define zherk_ zherk
#  define dsyr2k_ dsyr2k
#  define zher2k_ zher2k
#  define dgemm_ dgemm
#  define zgemm_ zgemm
#  define dgemv_ dgemv
#  define zgemv_ zgemv
#  define ddot_ ddot
#endif

void daxpy_(int* n, double* alpha,
	    double* x, int *incx, 
	    double* y, int *incy);
void zaxpy_(int* n, void* alpha,
	    void* x, int *incx, 
	    void* y, int *incy);
int dsyrk_(char *uplo, char *trans, int *n, int *k, 
	   double *alpha, double *a, int *lda, double *beta, 
	   double *c, int *ldc);
int zherk_(char *uplo, char *trans, int *n, int *k, 
	   double *alpha, void *a, int *lda,
	   double *beta, 
	   void *c, int *ldc);
int dsyr2k_(char *uplo, char *trans, int *n, int *k, 
	    double *alpha, double *a, int *lda, 
	    double *b, int *ldb, double *beta, 
	    double *c, int *ldc);
int zher2k_(char *uplo, char *trans, int *n, int *k, 
	    void *alpha, void *a, int *lda,
	    void *b, int *ldb, double *beta, 
	    void *c, int *ldc);
int dgemm_(char *transa, char *transb, int *m, int * n,
	   int *k, double *alpha, double *a, int *lda, 
	   double *b, int *ldb, double *beta, 
	   double *c, int *ldc);
int zgemm_(char *transa, char *transb, int *m, int * n,
	   int *k, void *alpha, void *a, int *lda, 
	   void *b, int *ldb, void *beta,
	   void *c, int *ldc);
int dgemv_(char *trans, int *m, int * n,
	   double *alpha, double *a, int *lda,
	   double *x, int *incx, double *beta,
	   double *y, int *incy);
int zgemv_(char *trans, int *m, int * n,
	   void *alpha, void *a, int *lda,
	   void *x, int *incx, void *beta,
	   void *y, int *incy);
double ddot_(int *n, void *dx, int *incx, void *dy, int *incy);

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
      lda = m;
      ldb = b->strides[0] / b->strides[1];
      ldc = c->strides[0] / c->strides[c->nd - 1];
    } 
  else
    {
      k = a->dimensions[1];
      for (int i = 2; i < a->nd; i++)
	k *= a->dimensions[i];
      m = a->dimensions[0];
      lda = k;
      ldb = k;
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
  PyObject* alpha;
  PyArrayObject* x;
  PyArrayObject* y;
  if (!PyArg_ParseTuple(args, "OOO", &alpha, &x, &y)) 
    return NULL;
  int n = x->dimensions[0];
  for (int d = 1; d < x->nd; d++)
    n *= x->dimensions[d];
  int incx = 1;
  int incy = 1;
  if (PyFloat_Check(alpha))
    {
      if (x->descr->type_num == PyArray_CDOUBLE)
	n *= 2;
      PyFloatObject* palpha = (PyFloatObject*)alpha;
      daxpy_(&n, &(palpha->ob_fval), 
            DOUBLEP(x), &incx,
            DOUBLEP(y), &incy);
    }
  else
    {
      PyComplexObject* palpha = (PyComplexObject*)alpha;
      zaxpy_(&n, (void*)(&(palpha->cval)), 
             (void*)COMPLEXP(x), &incx,
             (void*)COMPLEXP(y), &incy);
    }
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
    dsyrk_("u", "c", &n, &k, 
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
    dsyr2k_("u", "c", &n, &k, 
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
#ifndef NO_C99_COMPLEX
      double_complex z = 0.0;
      for (int i = 0; i < n; i++)
	z += conj(ap[i]) * bp[i];
      return PyComplex_FromDoubles(creal(z), cimag(z));
#else
      double_complex z = {0.0, 0.0};
      for (int i = 0; i < n; i++)
        {
          z.r += ap[i].r * bp[i].r + ap[i].i * bp[i].i;
          z.i += ap[i].r * bp[i].i - ap[i].i * bp[i].r;
        }
      return PyComplex_FromDoubles(z.r, z.i);
#endif
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
#ifndef NO_C99_COMPLEX
      double_complex z = 0.0;
      for (int i = 0; i < n; i++)
	z += ap[i] * bp[i];
      return PyComplex_FromDoubles(creal(z), cimag(z));
#else
      double_complex z = {0.0, 0.0};
      for (int i = 0; i < n; i++)
        {
          z.r += ap[i].r * bp[i].r - ap[i].i * bp[i].i;
          z.i += ap[i].r * bp[i].i + ap[i].i * bp[i].r;
        }
      return PyComplex_FromDoubles(z.r, z.i);
#endif
    }
}
