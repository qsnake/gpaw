#include <Python.h>
#define NO_IMPORT_ARRAY
#include <Numeric/arrayobject.h>
#include "extensions.h"

#ifdef GPAW_AIX
#  define dsyev_ dsyev
#  define zhegv_ zhegv
#  define zheev_ zheev
#endif

int dsyev_(char *jobz, char *uplo, int *n, double *
	   a, int *lda, double *w, double *work, int *lwork, 
	   int *info);
int zheev_(char *jobz, char *uplo, int *n, 
	   void *a, int *lda, double *w, void *work, 
	   int *lwork, double *rwork, int *lrwork, int *info);
int dsygv_(int *itype, char *jobz, char *uplo, int *
	   n, double *a, int *lda, double *b, int *ldb, 
	   double *w, double *work, int *lwork, int *info);
int zhegv_(int *itype, char *jobz, char *uplo, int *
	   n, void *a, int *lda, void *b, int *ldb, 
	   double *w, void *work, int *lwork,
	   double *rwork,
	   int *lrwork, int *info);
int dpotrf_(char *uplo, int *n, double *a, int *
	    lda, int *info);
int dpotri_(char *uplo, int *n, double *a, int *
	    lda, int *info);
int zpotrf_(char *uplo, int *n, void *a, 
	    int *lda, int *info);
int zpotri_(char *uplo, int *n, void *a, 
	    int *lda, int *info);

PyObject* diagonalize(PyObject *self, PyObject *args)
{
  PyArrayObject* a;
  PyArrayObject* w;
  PyArrayObject* b = 0;
  if (!PyArg_ParseTuple(args, "OO|O", &a, &w, &b)) 
    return NULL;
  int n = a->dimensions[0];
  int lda = n;
  int ldb = n;
  int itype = 1;
  int info = 0;
  if (a->descr->type_num == PyArray_DOUBLE)
    {
      int lwork = 3 * n + 1;
      double* work = (double*)malloc(lwork * sizeof(double));
      if (b == 0)
        dsyev_("V", "U", &n, DOUBLEP(a), &lda,
               DOUBLEP(w), work, &lwork, &info);
      else
        dsygv_(&itype, "V", "U", &n, DOUBLEP(a), &lda,
                DOUBLEP(b), &ldb, DOUBLEP(w), 
                work, &lwork, &info);
      free(work);
    }
  else
    {
      int lwork = 2 * n + 1;
      int lrwork = 3 * n + 1;
      void* work = malloc(lwork * 2 * sizeof(double));
      double* rwork = (double*)malloc(lrwork * sizeof(double));
      if (b == 0)
        zheev_("V", "U", &n, (void*)COMPLEXP(a), &lda,
               DOUBLEP(w), 
               work, &lwork, rwork, &lrwork, &info);
      else
        zhegv_(&itype, "V", "U", &n, (void*)COMPLEXP(a), &lda,
                (void*)COMPLEXP(b), &lda,
                DOUBLEP(w), 
                work, &lwork, rwork, &lrwork, &info);
      free(work);
      free(rwork);
    }
  return Py_BuildValue("i", info);
}
