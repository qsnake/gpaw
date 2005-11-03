#include <Python.h>
#define NO_IMPORT_ARRAY
#include <Numeric/arrayobject.h>
#include "extensions.h"

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

PyObject* gemm(PyObject *self, PyObject *args)
{
  Py_complex alpha;
  PyArrayObject* a;
  PyArrayObject* b;
  Py_complex beta;
  PyArrayObject* c;
  if (!PyArg_ParseTuple(args, "DOODO", &alpha, &a, &b, &beta, &c)) 
    return NULL;
  int m = a->dimensions[1];
  for (int i = 2; i < a->nd; i++)
    m *= a->dimensions[i];
  int k = a->dimensions[0];
  int n = b->dimensions[0];
  char transa = 'n';
  char transb = 'n';
  if (a->descr->type_num == PyArray_DOUBLE)
    dgemm_(&transa, &transb, &m, &n, &k, 
           &(alpha.real),
           DOUBLEP(a), &m, 
           DOUBLEP(b), &k,
           &(beta.real), 
           DOUBLEP(c), &m);
  else
    zgemm_(&transa, &transb, &m, &n, &k, 
           &alpha,
           (void*)COMPLEXP(a), &m, 
           (void*)COMPLEXP(b), &k,
           &beta, 
           (void*)COMPLEXP(c), &m);
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
  char uplo = 'u';
  char trans = 'c';
  int n = a->dimensions[0];
  int k = a->dimensions[1];
  for (int d = 2; d < a->nd; d++)
    k *= a->dimensions[d];
  if (a->descr->type_num == PyArray_DOUBLE)
    dsyrk_(&uplo, &trans, &n, &k, 
           &alpha, DOUBLEP(a), &k, &beta,
           DOUBLEP(c), &n);
  else
    zherk_(&uplo, &trans, &n, &k, 
           &alpha, (void*)COMPLEXP(a), &k, &beta,
           (void*)COMPLEXP(c), &n);
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
  char uplo = 'u';
  char trans = 'c';
  int n = a->dimensions[0];
  int k = a->dimensions[1];
  for (int d = 2; d < a->nd; d++)
    k *= a->dimensions[d];
  if (a->descr->type_num == PyArray_DOUBLE)
    dsyr2k_(&uplo, &trans, &n, &k, 
            (double*)(&alpha), DOUBLEP(a), &k, 
            DOUBLEP(b), &k, &beta,
            DOUBLEP(c), &n);
  else
    zher2k_(&uplo, &trans, &n, &k, 
            (void*)(&alpha), (void*)COMPLEXP(a), &k, 
            (void*)COMPLEXP(b), &k, &beta,
            (void*)COMPLEXP(c), &n);
  Py_RETURN_NONE;
}
