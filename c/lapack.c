#include <Python.h>
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "extensions.h"

#ifdef GPAW_AIX
#  define dsyev_ dsyev
#  define zheev_ zheev
#  define dsygv_ dsygv
#  define zhegv_ zhegv
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
int dgeev_(char *jovl, char *jobvr, int *n, double *a, int *lda,
	   double *wr, double *wl, 
	   double *vl, int *ldvl, double *vr, int *ldvr,
	   double *work, int *lwork, int *info);
			        
int dtrtri_(char *uplo,char *diag, int *n, void *a, 
	    int *lda, int *info );
int ztrtri_(char *uplo,char *diag, int *n, void *a, 
	    int *lda, int *info );

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
      double* work = GPAW_MALLOC(double, lwork);
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
      void* work = (void*)GPAW_MALLOC(double_complex, lwork);
      double* rwork = GPAW_MALLOC(double, lrwork);
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

PyObject* inverse_cholesky(PyObject *self, PyObject *args)
{
  PyArrayObject* a;
  if (!PyArg_ParseTuple(args, "O", &a)) 
    return NULL;
  int n = a->dimensions[0];
  int lda = n;
  int info = 0;

  if (a->descr->type_num == PyArray_DOUBLE)
    {
      dpotrf_("U", &n, (void*)DOUBLEP(a), &lda, &info);
      if (info == 0)
	{
	  dtrtri_("U", "N", &n, (void*)DOUBLEP(a), &lda, &info);
	  if (info == 0)
	    {
	      /* Make sure that the other diagonal is zero */
	      double* ap = DOUBLEP(a);
	      ap++;
	      for (int i = 0; i < n - 1; i++)
		{
		  memset(ap, 0, (n-1-i) * sizeof(double));
		  ap += n + 1;
		}
	    }
	}
    }
  else
    {
      zpotrf_("U", &n, (void*)COMPLEXP(a), &lda, &info);
      if (info == 0)
	{
	  ztrtri_("U", "N", &n, (void*)DOUBLEP(a), &lda, &info);
	  if (info == 0)
	    {
	      /* Make sure that lower diagonal is zero */
	      double_complex* ap = COMPLEXP(a);
	      ap++;
	      for (int i = 0; i < n - 1; i++)
		{
		  memset(ap, 0, (n-1-i) * sizeof(double_complex));
		  ap += n + 1;
		}
	    }
	}
    }
  return Py_BuildValue("i", info);
}

void swap(double *a, double *b) {
  double tmp=*b;
  *b = *a;
  *a = tmp;
}
void transpose(double *A, int n) {
  int i, j;
  int in=0;
  for(i=0;i<n-1;i++) {
    for(j=i+1;j<n;j++)
      swap(A+in+j,A+j*n+i);
    in+=n;
  }
}
void print(double *A, int n) {
  int i,j;
  for(i=0;i<n;i++) {
    if(i) printf(" (");
    else printf("((");
    for(j=0;j<n;j++) {
      printf(" %g",A[n*i+j]);
    }
    if(i==n-1) printf("))\n");
    else printf(")\n");
  }
}
PyObject* right_eigenvectors(PyObject *self, PyObject *args)
/* Return eigenvalues and right eigenvectors of a
 * nonsymmetric eigenvalue problem
 */
{
  PyArrayObject* A;
  PyArrayObject* v; /* eigenvectors */
  PyArrayObject* w; /* eigenvalues */
  if (!PyArg_ParseTuple(args, "OOO", &A, &w, &v)) 
    return NULL;
  int n = A->dimensions[0];
  int lda = n;
  int info = 0;
  if (A->descr->type_num == PyArray_DOUBLE)
    {
      int lwork = -1;
      double* work = GPAW_MALLOC(double, 1);
      double* wr = GPAW_MALLOC(double, n);
      double* wi = GPAW_MALLOC(double, n);
      int ldvl = 1;
      int ldvr = n;
      double* vl = 0;
      int i;
      /* get size of work needed */
      dgeev_("No eigenvectors left", "Vectors right", 
	     &n, DOUBLEP(A), &lda, wr, wi, 
	     vl, &ldvl, DOUBLEP(v), &ldvr, work, &lwork, &info);
      lwork = (int) work[0];
      free(work);
      work = GPAW_MALLOC(double, lwork);

      transpose(DOUBLEP(A),n); /* transform to Fortran form */
      dgeev_("No eigenvectors left", "Vectors right", 
	     &n, DOUBLEP(A), &lda, wr, wi, 
	     vl, &ldvl, DOUBLEP(v), &ldvr, work, &lwork, &info);

      for(i=0;i<n;i++) {
	if(wi[i]!=0.)
	  printf("<diagonalize_nonsymmetric> dgeev i=%d,wi[i]=%g\n",
		 i,wi[i]);
	DOUBLEP(w)[i]=wr[i];
      }
      free(wr);
      free(wi);
      free(work);
    }
  return Py_BuildValue("i", info);
}
