#include "spline.h"
#include <malloc.h>
#include <stdlib.h>
#ifdef PARALLEL
#  include <mpi.h>
#else
   typedef int* MPI_Request; // !!!!!!!???????????
   typedef int* MPI_Comm;
#  define MPI_COMM_NULL 0
#  define MPI_Comm_rank(comm, rank) *(rank) = 0, 0
#  define MPI_Bcast(buff, count, datatype, root, comm) 0
#endif

#include "mympi.h"

int dgemm_(char *transa, char *transb, int *m, int * n,
	   int *k, double *alpha, double *a, int *lda, 
	   double *b, int *ldb, double *beta, 
	   double *c, int *ldc);
int dgemv_(char *trans, int *m, int * n,
	   double *alpha, double *a, int *lda, 
	   double *x, int *incx, double *beta, 
	   double *y, int *incy);

typedef struct 
{
  PyObject_HEAD
  double dv;
  int size[3];
  int start[3];
  int size0[3];
  int ng;
  int ng0;
  int nf;
  int nfd;
  double* f;
  double* fd;
  double* w;
} LocalizedFunctionsObject;

static void localized_functions_dealloc(LocalizedFunctionsObject *self)
{
  free(self->f);
  free(self->w);
  PyObject_DEL(self);
}

static PyObject * localized_functions_integrate(LocalizedFunctionsObject *self,
                                              PyObject *args)
{
  const PyArrayObject* aa;
  PyArrayObject* bb;
  int derivatives = 0;
  if (!PyArg_ParseTuple(args, "OO|i", &aa, &bb, &derivatives))
    return NULL;

  const double* a = DOUBLEP(aa);
  double* b = DOUBLEP(bb);
  int na = 1;
  for (int d = 0; d < aa->nd - 3; d++)
    na *= aa->dimensions[d];
  int nf = (derivatives ? self->nfd : self->nf);
  double* f = (derivatives ? self->fd : self->f);
  double* w = self->w;
  int ng = self->ng;
  int ng0 = self->ng0;

  if (aa->descr->type_num == PyArray_DOUBLE)
    for (int n = 0; n < na; n++)
      {
	bmgs_cut(a, self->size, self->start, w, self->size0);
	double zero = 0.0;
	int inc = 1;
	dgemv_("t", &ng0, &nf, &self->dv, f, &ng0, w, &inc, &zero, b, &inc);

	a += ng;
	b += nf;
      }
  else
    for (int n = 0; n < na; n++)
      {
	bmgs_cutz((const double_complex*)a, self->size, self->start, 
		  (double_complex*)w, self->size0);
	double zero = 0.0;
	int inc = 2;
	dgemm_("n", "n", &inc, &nf, &ng0, &self->dv, w, &inc, f, &ng0,
	       &zero, b, &inc);

	a += 2 * ng;
	b += 2 * nf;
      }
  Py_RETURN_NONE;
}

static PyObject * localized_functions_add(LocalizedFunctionsObject *self,
					  PyObject *args)
{
  PyArrayObject* cc;
  PyArrayObject* aa;
  if (!PyArg_ParseTuple(args, "OO", &cc, &aa))
    return NULL;

  double* c = DOUBLEP(cc);
  double* a = DOUBLEP(aa);
  int na = 1;
  for (int d = 0; d < aa->nd - 3; d++)
    na *= aa->dimensions[d];
  int ng = self->ng;
  int ng0 = self->ng0;
  int nf = self->nf;
  double* f = self->f;
  double* w = self->w;

  if (aa->descr->type_num == PyArray_DOUBLE)
    for (int n = 0; n < na; n++)
      {
	double zero = 0.0;
	double one = 1.0;
	int inc = 1;
	dgemv_("n", &ng0, &nf, &one, f, &ng0, c, &inc, &zero, w, &inc);
	bmgs_pastep(w, self->size0, a, self->size, self->start);
	a += ng;
	c += nf;
      }
  else
    for (int n = 0; n < na; n++)
      {
	double zero = 0.0;
	double one = 1.0;
	int inc = 2;
	dgemm_("n", "t", &inc, &ng0, &nf, &one, c, &inc, f, &ng0,
	       &zero, w, &inc);
	bmgs_pastepz((const double_complex*)w, self->size0, 
		    (double_complex*)a, self->size, self->start);
	a += 2 * ng;
	c += 2 * nf;
      }
  Py_RETURN_NONE;
}

static PyObject * localized_functions_add_density(LocalizedFunctionsObject*
						  self,
						  PyObject *args)
{
  PyArrayObject* dd;
  PyArrayObject* oo;
  if (!PyArg_ParseTuple(args, "OO", &dd, &oo))
    return NULL;

  const double* o = DOUBLEP(oo);
  double* d = DOUBLEP(dd);
  int nf = self->nf;
  int ng0 = self->ng0;
  const double* f = self->f;
  double* w = self->w;

  memset(w, 0, ng0 * sizeof(double));
  for (int i = 0; i < nf; i++)
    for (int n = 0; n < ng0; n++)
      {
	double g = *f++;
	w[n] += o[i] * g * g;
      }
  bmgs_pastep(w, self->size0, d, self->size, self->start);
  Py_RETURN_NONE;
}

#ifdef PARALLEL
static PyObject * localized_functions_broadcast(LocalizedFunctionsObject*
						self,
						PyObject *args)
{
  PyObject* comm_obj;
  int root;
  if (!PyArg_ParseTuple(args, "Oi", &comm_obj, &root))
    return NULL;

  MPI_Comm comm = ((MPIObject*)comm_obj)->comm;
  MPI_Bcast(self->f, self->ng0 * (self->nf + self->nfd),
	    MPI_DOUBLE, root, comm);
  Py_RETURN_NONE;
}
#endif

static PyMethodDef localized_functions_methods[] = {
    {"integrate",
     (PyCFunction)localized_functions_integrate, METH_VARARGS, 0},
    {"add",
     (PyCFunction)localized_functions_add, METH_VARARGS, 0},
    {"add_density",
     (PyCFunction)localized_functions_add_density, METH_VARARGS, 0},
#ifdef PARALLEL
    {"broadcast",
     (PyCFunction)localized_functions_broadcast, METH_VARARGS, 0},
#endif
    {NULL, NULL, 0, NULL}
};

static PyObject* localized_functions_getattr(PyObject *obj, char *name)
{
    return Py_FindMethod(localized_functions_methods, obj, name);
}

static PyTypeObject LocalizedFunctionsType = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,
  "LocalizedFunctions",
  sizeof(LocalizedFunctionsObject),
  0,
  (destructor)localized_functions_dealloc,
  0,
  localized_functions_getattr
};

PyObject * NewLocalizedFunctionsObject(PyObject *obj, PyObject *args)
{
  PyObject* radials;
  PyArrayObject* size0_array;
  PyArrayObject* size_array;
  PyArrayObject* start_array;
  PyArrayObject* h_array;
  PyArrayObject* C_array;
  int pp;
  int kk;
  int real;
  int forces;
  int compute;
  if (!PyArg_ParseTuple(args, "OOOOOOiiiii", &radials, 
			&size0_array, &size_array,
                        &start_array, &h_array, &C_array,
                        &pp, &kk, &real, &forces, &compute))
    return NULL;

  LocalizedFunctionsObject *self = PyObject_NEW(LocalizedFunctionsObject,
						&LocalizedFunctionsType);
  if (self == NULL)
    return NULL;

  const long* size0 = LONGP(size0_array);
  const long* size = LONGP(size_array);
  const long* start = LONGP(start_array);
  const double* h = DOUBLEP(h_array);
  const double* C = DOUBLEP(C_array);
  self->dv = h[0] * h[1] * h[2];
  int ng = size[0] * size[1] * size[2];
  int ng0 = size0[0] * size0[1] * size0[2];
  self->ng = ng;
  self->ng0 = ng0;
  for (int i = 0; i < 3; i++)
    {
      self->size0[i] = size0[i];
      self->size[i] = size[i];
      self->start[i] = start[i];
    } 
  int nf = 0;
  int nfd = 0;
  int nbins;
  double dr;
  for (int j = 0; j < PyList_Size(radials); j++)
    {
      const bmgsspline* spline = 
	&(((SplineObject*)PyList_GetItem(radials, j))->spline);
      int l = spline->l;
      assert(l <= 2);
      if (j == 0)
	{
	  nbins = spline->nbins;
	  dr = spline->dr;
	}
      else
	{
	  assert(spline->nbins == nbins);
	  assert(spline->dr == dr);
	}
      nf += (2 * l + 1); 
      if (forces)
        nfd += 3 + l * (1 + 2 * l); 
    }
  self->nf = nf;
  self->nfd = nfd;
  self->f = (double*)malloc((nf + nfd) * ng0 * sizeof(double));
  if (forces)
    self->fd = self->f + nf * ng0;
  else
    self->fd = 0;

  int ndouble = (real ? 1 : 2);
  self->w = (double*)malloc(ng0 * ndouble * sizeof(double));

  if (compute)
    {
      if (pp == 1)
	{
	  int* bin = (int*)malloc(ng0 * sizeof(int));
	  double* d = (double*)malloc(ng0 * sizeof(double));
	  double* f0 = (double*)malloc(ng0 * sizeof(double));
	  double* fd0 = 0;
	  if (forces)
	    fd0 = (double*)malloc(ng0 * sizeof(double));
	  
	  double* a = self->f;
	  double* ad = self->fd;
	  for (int j = 0; j < PyList_Size(radials); j++)
	    {
	      const bmgsspline* spline = 
		&(((SplineObject*)PyList_GetItem(radials, j))->spline);
	      if (j == 0)
		bmgs_radial1(spline, self->size0, C, h, bin, d);
	      bmgs_radial2(spline, self->size0, bin, d, f0, fd0);
	      int l = spline->l;
	      for (int k = 0; k < 2 * l + 1; k++)
		{
		  bmgs_radial3(spline, k, self->size0, C, h, f0, a);
		  a += ng0;
		}
	      if (forces)
		for (int k = 0; k < 3 + l * (2 * l + 1); k++)
		  {
		    bmgs_radiald3(spline, k, self->size0, C, h, f0, fd0, ad);
		    ad += ng0;
		  }
	    }
	  if (forces)
	    free(fd0);
	  free(f0);
	  free(d);
	  free(bin);
	}
      else
	{
	  int size2[3];
	  double h2[3];
	  double C2[3];
	  int ng02 = 1;
	  for (int i = 0; i < 3; i++)
	    {
	      size2[i] = pp * size0[i] + pp * (kk - 1) - 1;
	      h2[i] = h[i] / pp;
	      C2[i] = C[i] - h2[i] * (pp * kk / 2 - 1);
	      ng02 *= size2[i];
	    }
	  int* bin = (int*)malloc(ng02 * sizeof(int));
	  double* d = (double*)malloc(ng02 * sizeof(double));
	  double* f0 = (double*)malloc(ng02 * sizeof(double));
	  double* f1 = (double*)malloc(ng02 * sizeof(double));
	  double* f2 = (double*)malloc(size2[0] * size2[1] * size0[2] *
				       sizeof(double));
	  double* fd0 = 0;
	  
	  if (forces)
	    fd0 = (double*)malloc(ng02 * sizeof(double));
	  
	  double* a = self->f;
	  double* ad = self->fd;
	  for (int j = 0; j < PyList_Size(radials); j++)
	    {
	      const bmgsspline* spline = 
		&(((SplineObject*)PyList_GetItem(radials, j))->spline);
	      if (j == 0)
		bmgs_radial1(spline, size2, C2, h2, bin, d);
	      bmgs_radial2(spline, size2, bin, d, f0, fd0);
	      int l = spline->l;
	      for (int k = 0; k < 2 * l + 1; k++)
		{
		  bmgs_radial3(spline, k, size2, C2, h2, f0, f1);
		  bmgs_restrict(kk, pp, f1, size2, a, f2);
		  a += ng0;
		}
	      if (forces)
		for (int k = 0; k < 3 + l * (2 * l + 1); k++)
		  {
		    bmgs_radiald3(spline, k, size2, C2, h2, f0, fd0, f1);
		    bmgs_restrict(kk, pp, f1, size2, ad, f2);
		    ad += ng0;
		  }
	    }
	  if (forces)
	    free(fd0);
	  free(f2);
	  free(f1);
	  free(f0);
	  free(d);
	  free(bin);
	}
    }
  return (PyObject*)self;
}
