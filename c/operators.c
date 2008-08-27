//*** The apply operator and some associate structors are imple-  ***//
//*** mented in two version: a original version and a speciel     ***//
//*** OpenMP version. By default the original version will        ***//
//*** be used, but it's possible to use the OpenMP version        ***//
//*** by compiling gpaw with the macro GPAW_OMP defined and       ***//
//*** and the compile/link option "-fopenmp".                     ***//
//*** Author of the optimized OpenMP code:                        ***//
//*** Mads R. B. Kristensen - madsbk@diku.dk                      ***//

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "extensions.h"
#include "bc.h"
#include "mympi.h"
#include "bmgs/bmgs.h"
#ifdef GPAW_OMP
  #include <omp.h>
#endif

#ifdef GPAW_ASYNC
  #define GPAW_ASYNC_D 3
#else
  #define GPAW_ASYNC_D 1
#endif

typedef struct
{
  PyObject_HEAD
  bmgsstencil stencil;
  boundary_conditions* bc;
  MPI_Request recvreq[2];
  MPI_Request sendreq[2];
  double* buf;
  double* sendbuf;
  double* recvbuf;
} OperatorObject;

static void Operator_dealloc(OperatorObject *self)
{
  free(self->bc);
  free(self->buf);
  free(self->sendbuf);
  free(self->recvbuf);
  PyObject_DEL(self);
}

static PyObject * Operator_relax(OperatorObject *self,
                                 PyObject *args)
{
  int relax_method;
  PyArrayObject* func;
  PyArrayObject* source;
  double w = 1.0;
  int nrelax;
  if (!PyArg_ParseTuple(args, "iOOi|d", &relax_method, &func, &source, &nrelax, &w))
    return NULL;

  const boundary_conditions* bc = self->bc;

  double* fun = DOUBLEP(func);
  const double* src = DOUBLEP(source);
  const double_complex* ph;

  ph = 0;

  for (int n = 0; n < nrelax; n++ )
    {
      for (int i = 0; i < 3; i++)
        {
          bc_unpack1(bc, fun, self->buf, i,
               self->recvreq, self->sendreq,
               self->recvbuf, self->sendbuf, ph + 2 * i);
          bc_unpack2(bc, self->buf, i,
               self->recvreq, self->sendreq, self->recvbuf);
        }
      bmgs_relax(relax_method, &self->stencil, self->buf, fun, src, w);
    }
  Py_RETURN_NONE;
}

static PyObject * Operator_apply(OperatorObject *self,
                                 PyObject *args)
{
  PyArrayObject* input;
  PyArrayObject* output;
  PyArrayObject* phases = 0;
  if (!PyArg_ParseTuple(args, "OO|O", &input, &output, &phases))
    return NULL;

  int nin = 1;
  if (input->nd == 4)
    nin = input->dimensions[0];

  const boundary_conditions* bc = self->bc;
  const int* size1 = bc->size1;
  const int* size2 = bc->size2;
  int ng = bc->ndouble * size1[0] * size1[1] * size1[2];
  int ng2 = bc->ndouble * size2[0] * size2[1] * size2[2];

  const double* inn = DOUBLEP(input);
  double* outt = DOUBLEP(output);
  const double_complex* ph;

  bool real = (input->descr->type_num == PyArray_DOUBLE);

  if (real)
    ph = 0;
  else
    ph = COMPLEXP(phases);

  #ifdef GPAW_OMP
    #pragma omp parallel for
  #endif
  for (int n = 0; n < nin; n++)
    {
      const double* in = inn + n * ng;
      double* out = outt + n * ng;
    #ifdef GPAW_OMP
      int thd = omp_get_thread_num();
    #else
      int thd = 0;
    #endif

    double* sendbuf = self->sendbuf + thd * bc->maxsend * GPAW_ASYNC_D;
    double* recvbuf = self->recvbuf + thd * bc->maxrecv * GPAW_ASYNC_D;
    double* buf = self->buf + thd * ng2;
    MPI_Request recvreq[2 * GPAW_ASYNC_D];
    MPI_Request sendreq[2 * GPAW_ASYNC_D];

    #ifndef GPAW_ASYNC
      if (1)
    #else
      if (bc->cfd == 0)
    #endif
        {
          for (int i = 0; i < 3; i++)
            {
              bc_unpack1(bc, in, buf, i,
                         recvreq, sendreq,
                         recvbuf, sendbuf, ph + 2 * i);

              bc_unpack2(bc, buf, i,
                         recvreq, sendreq, recvbuf);
            }
        }
      else
        {
          for (int i = 0; i < 3; i++)
            {

              bc_unpack1(bc, in, buf, i,
                         recvreq + i * 2, sendreq + i * 2,
                         recvbuf + i * bc->maxrecv,
                         sendbuf + i * bc->maxsend, ph + 2 * i);
            }
          for (int i = 0; i < 3; i++)
            {
              bc_unpack2(bc, buf, i,
                         recvreq + i * 2, sendreq + i * 2,
                         recvbuf + i * bc->maxrecv);
            }
        }
      if (real)
        bmgs_fd(&self->stencil, buf, out);
      else
        bmgs_fdz(&self->stencil, (const double_complex*) buf,
                                 (double_complex*)out);
    }
  Py_RETURN_NONE;
}


static PyObject * Operator_get_diagonal_element(OperatorObject *self,
                                              PyObject *args)
{
  if (!PyArg_ParseTuple(args, ""))
    return NULL;

  const bmgsstencil* s = &self->stencil;
  double d = 0.0;
  for (int n = 0; n < s->ncoefs; n++)
    if (s->offsets[n] == 0)
      d = s->coefs[n];

  return Py_BuildValue("d", d);
}


static PyMethodDef Operator_Methods[] = {
    {"apply",
     (PyCFunction)Operator_apply, METH_VARARGS, NULL},
    {"relax",
     (PyCFunction)Operator_relax, METH_VARARGS, NULL},
    {"get_diagonal_element",
     (PyCFunction)Operator_get_diagonal_element, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}

};


static PyObject* Operator_getattr(PyObject *obj, char *name)
{
    return Py_FindMethod(Operator_Methods, obj, name);
}

static PyTypeObject OperatorType = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,
  "Operator",
  sizeof(OperatorObject),
  0,
  (destructor)Operator_dealloc,
  0,
  Operator_getattr
};

PyObject * NewOperatorObject(PyObject *obj, PyObject *args)
{
  PyArrayObject* coefs;
  PyArrayObject* offsets;
  PyArrayObject* size;
  int range;
  PyArrayObject* neighbors;
  int real;
  PyObject* comm_obj;
  int cfd;
  if (!PyArg_ParseTuple(args, "OOOiOiOi",
                        &coefs, &offsets, &size, &range, &neighbors,
                        &real, &comm_obj, &cfd))
    return NULL;

  OperatorObject *self = PyObject_NEW(OperatorObject, &OperatorType);
  if (self == NULL)
    return NULL;

  self->stencil = bmgs_stencil(coefs->dimensions[0], DOUBLEP(coefs),
                               LONGP(offsets), range, LONGP(size));

  const long (*nb)[2] = (const long (*)[2])LONGP(neighbors);
  const long padding[3][2] = {{range, range},
                             {range, range},
                             {range, range}};

  MPI_Comm comm = MPI_COMM_NULL;
  if (comm_obj != Py_None)
    comm = ((MPIObject*)comm_obj)->comm;

  self->bc = bc_init(LONGP(size), padding, padding, nb, comm, real, cfd);

  const int* size2 = self->bc->size2;

#ifndef GPAW_OMP
  self->buf = GPAW_MALLOC(double, size2[0] * size2[1] * size2[2] *
                          self->bc->ndouble);
  self->sendbuf = GPAW_MALLOC(double, self->bc->maxsend * GPAW_ASYNC_D);
  self->recvbuf = GPAW_MALLOC(double, self->bc->maxrecv * GPAW_ASYNC_D);
#else
  //We need a buffer per OpenMP Thread.
  self->buf = GPAW_MALLOC(double, size2[0] * size2[1] * size2[2] *
                          self->bc->ndouble * omp_get_max_threads());
  self->sendbuf = GPAW_MALLOC(double, self->bc->maxsend *
                              omp_get_max_threads() * GPAW_ASYNC_D);
  self->recvbuf = GPAW_MALLOC(double, self->bc->maxrecv *
                              omp_get_max_threads() * GPAW_ASYNC_D);
#endif
  return (PyObject*)self;
}
