//*** The apply operator and some associate structors are imple-  ***//
//*** mented in two version: a original version and a speciel     ***//
//*** Blue Gene version. By default the original version will     ***//
//*** be used, but it's possible to use the Blue Gene version     ***//
//*** by compiling gpaw with the macro BLUEGENE defined.          ***//
//*** Author of the optimized Blue Gene code:                     ***//
//*** Mads R. B. Kristensen - madsbk@diku.dk                      ***//

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <pthread.h>
#include "extensions.h"
#include "bc.h"
#include "mympi.h"
#include "bmgs/bmgs.h"

typedef struct
{
  PyObject_HEAD
  bmgsstencil stencil;
  boundary_conditions* bc;
#ifndef BLUEGENE
  MPI_Request recvreq[2];
  MPI_Request sendreq[2];
#else
  //We need 2 revc and send request per Blue Gene kernel e.g. 4.
  MPI_Request recvreq[2 * NUM_OF_THREADS];
  MPI_Request sendreq[2 * NUM_OF_THREADS];
#endif
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
#ifndef BLUEGENE
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
  int ng = bc->ndouble * size1[0] * size1[1] * size1[2];

  const double* in = DOUBLEP(input);
  double* out = DOUBLEP(output);
  const double_complex* ph;

  bool real = (input->descr->type_num == PyArray_DOUBLE);

  if (real)
    ph = 0;
  else
    ph = COMPLEXP(phases);

  for (int n = 0; n < nin; n++)
    {
      for (int i = 0; i < 3; i++)
        {
          bc_unpack1(bc, in, self->buf, i,
               self->recvreq, self->sendreq,
               self->recvbuf, self->sendbuf, ph + 2 * i);
          bc_unpack2(bc, self->buf, i,
               self->recvreq, self->sendreq, self->recvbuf);
        }
      if (real)
        bmgs_fd(&self->stencil, self->buf, out);
      else
        bmgs_fdz(&self->stencil, (const double_complex*)self->buf,
                                 (double_complex*)out);

      in += ng;
      out += ng;
    }
  Py_RETURN_NONE;
}

#else

typedef struct
{
  OperatorObject *self;
  boundary_conditions *bc;
  double *in, *out;
  int nin, ng, thd_id;
  bool real;
  double_complex *ph;
  pthread_barrier_t *barrier;
} fd_worker_t;

void* fd_worker(void* args)
{
  fd_worker_t* arg = args;
  OperatorObject *self = arg->self;
  int thd_id = arg->thd_id;
  int nin = arg->nin;

  double *in = arg->in + (thd_id * arg->ng * (nin / NUM_OF_THREADS));
  double *out = arg->out + (thd_id * arg->ng * (nin / NUM_OF_THREADS));

  double *buf = self->buf + thd_id;
  double *sendbuf = self->sendbuf + thd_id;
  double *recvbuf = self->recvbuf + thd_id;
  MPI_Request *sendreq = self->sendreq + (thd_id * 2);
  MPI_Request *recvreq = self->recvreq + (thd_id * 2);

  int nmax = (thd_id == NUM_OF_THREADS - 1) ? nin : (nin / NUM_OF_THREADS) * (thd_id + 1);
  for (int n = (nin / NUM_OF_THREADS) * thd_id; n < nmax; n++)
  {
    for (int i = 0; i < 3; i++)
      {
        bc_unpack1(arg->bc, in, buf, i,
                   recvreq, sendreq,
                   recvbuf, sendbuf, arg->ph + 2 * i);
        bc_unpack2(arg->bc, buf, i,
                   recvreq, sendreq, recvbuf);
      }
    if (arg->real)
      bmgs_fd(&arg->self->stencil, buf, out);
    else
      bmgs_fdz(&arg->self->stencil, (const double_complex*)buf,
                                    (double_complex*)out);
    in += arg->ng;
    out += arg->ng;
  }
  return NULL;
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

  boundary_conditions* bc = self->bc;
  const int* size1 = bc->size1;
  int ng = bc->ndouble * size1[0] * size1[1] * size1[2];

  double* in = DOUBLEP(input);
  double* out = DOUBLEP(output);
  double_complex* ph;

  bool real = (input->descr->type_num == PyArray_DOUBLE);

  if (real)
    ph = 0;
  else
    ph = COMPLEXP(phases);

  // Worker-thread handlers
  pthread_t fd_threads[NUM_OF_THREADS];
  fd_worker_t* fd_arg = malloc(NUM_OF_THREADS * sizeof(fd_worker_t));

  for (int i = 0; i < NUM_OF_THREADS; i++)
    {
      (fd_arg + i)->thd_id = i;
      (fd_arg + i)->self = self;
      (fd_arg + i)->bc = bc;
      (fd_arg + i)->in = in;
      (fd_arg + i)->out = out;
      (fd_arg + i)->nin = nin;
      (fd_arg + i)->real = real;
      (fd_arg + i)->ng = ng;
      (fd_arg + i)->ph = ph;
      pthread_create(&fd_threads[i], NULL, &fd_worker, (fd_arg + i));
    }
  for (int i = 0; i < NUM_OF_THREADS; i++)
    pthread_join(fd_threads[i], NULL);
  free(fd_arg);

  Py_RETURN_NONE;
}
#endif

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
#ifndef BLUEGENE
  self->buf = GPAW_MALLOC(double, size2[0] * size2[1] * size2[2] *
        self->bc->ndouble);
  self->sendbuf = GPAW_MALLOC(double, self->bc->maxsend);
  self->recvbuf = GPAW_MALLOC(double, self->bc->maxrecv);
#else
  //We need a buffer per Blue Gene kernel e.g. 4.
  self->buf = GPAW_MALLOC(double, size2[0] * size2[1] * size2[2] *
        self->bc->ndouble * NUM_OF_THREADS);
  self->sendbuf = GPAW_MALLOC(double, self->bc->maxsend * NUM_OF_THREADS);
  self->recvbuf = GPAW_MALLOC(double, self->bc->maxrecv * NUM_OF_THREADS);
#endif
  return (PyObject*)self;
}
