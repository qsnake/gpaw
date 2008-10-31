#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "extensions.h"
#include "bc.h"
#include "mympi.h"
#include "bmgs/bmgs.h"

#ifdef GPAW_ASYNC
  #define GPAW_ASYNC_D 3
#else
  #define GPAW_ASYNC_D 1
#endif

typedef struct
{
  PyObject_HEAD
  boundary_conditions* bc;
  int p;
  int k;
  bool interpolate;
  MPI_Request recvreq[2];
  MPI_Request sendreq[2];
  int skip[3][2];
  double* buf;
  double* buf2;
  double* sendbuf;
  double* recvbuf;
} TransformerObject;

static void Transformer_dealloc(TransformerObject *self)
{
  free(self->bc);
  free(self->buf);
  free(self->buf2);
  free(self->sendbuf);
  free(self->recvbuf);
  PyObject_DEL(self);
}

struct transapply_args{
  int thread_id;
  TransformerObject *self;
  int ng;
  int ng2;
  int nin;
  int nthds;
  const double* in;
  double* out;
  int real;
  const double_complex* ph;
};

void *transapply_worker(void *threadarg)
{
  struct transapply_args *args = (struct transapply_args *) threadarg;
  boundary_conditions* bc = args->self->bc;
  TransformerObject *self = args->self;
  double* sendbuf = self->sendbuf + args->thread_id * bc->maxsend * GPAW_ASYNC_D;
  double* recvbuf = self->recvbuf + args->thread_id * bc->maxrecv * GPAW_ASYNC_D;
  double* buf = self->buf + args->thread_id * args->ng2;
  double* buf2 = self->buf2 + args->thread_id * args->ng2 * 16;
  MPI_Request recvreq[2 * GPAW_ASYNC_D];
  MPI_Request sendreq[2 * GPAW_ASYNC_D];

  int chunksize = args->nin / args->nthds;
  if (!chunksize)
    chunksize = 1;
  int nstart = args->thread_id * chunksize;
  if (nstart >= args->nin)
    return NULL;
  int nend = nstart + chunksize;
  if (nend > args->nin)
    nend = args->nin;

  for (int n = nstart; n < nend; n++)
    {
      const double* in = args->in + n * args->ng;
      double* out = args->out + n * args->ng * 8;
      for (int i = 0; i < 3; i++)
        {
          bc_unpack1(bc, in, buf, i,
                     recvreq, sendreq,
                     recvbuf, sendbuf, args->ph + 2 * i,
                     args->thread_id, 1);
          bc_unpack2(bc, buf, i,
                     recvreq, sendreq, recvbuf, 1);
        }
      if (args->real)
        {
          if (self->interpolate)
            bmgs_interpolate(self->k, self->skip, buf, bc->size2,
                             out, buf2);
          else
            bmgs_restrict(self->k, self->buf, bc->size2,
                          out, buf2);
        }
      else
        {
          if (self->interpolate)
            bmgs_interpolatez(self->k, self->skip, (double_complex*)buf,
                              bc->size2, (double_complex*)out,
                              (double_complex*) buf2);
          else
            bmgs_restrictz(self->k, (double_complex*) buf,
                           bc->size2, (double_complex*)out,
                           (double_complex*) buf2);
        }
    }
  return NULL;
}



static PyObject* Transformer_apply(TransformerObject *self, PyObject *args)
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
  const int* size2 = bc->size2;
  int ng = bc->ndouble * size1[0] * size1[1] * size1[2];
  int ng2 = bc->ndouble * size2[0] * size2[1] * size2[2];

  const double* in = DOUBLEP(input);
  double* out = DOUBLEP(output);
  bool real = (input->descr->type_num == PyArray_DOUBLE);
  const double_complex* ph = (real ? 0 : COMPLEXP(phases));

  int nthds = 1;
#ifdef GPAW_OMP
  if (getenv("OMP_NUM_THREADS") != NULL)
    nthds = atoi(getenv("OMP_NUM_THREADS"));
#endif
  struct transapply_args *wargs = GPAW_MALLOC(struct transapply_args, nthds);
  pthread_t *thds = GPAW_MALLOC(pthread_t, nthds);

  for(int i=0; i < nthds; i++)
    {
      (wargs+i)->thread_id = i;
      (wargs+i)->nthds = nthds;
      (wargs+i)->self = self;
      (wargs+i)->ng = ng;
      (wargs+i)->ng2 = ng2;
      (wargs+i)->nin = nin;
      (wargs+i)->in = in;
      (wargs+i)->out = out;
      (wargs+i)->real = real;
      (wargs+i)->ph = ph;
    }

#ifdef GPAW_OMP
  for(int i=1; i < nthds; i++)
    pthread_create(thds + i, NULL, transapply_worker, (void*) (wargs+i));
#endif
  transapply_worker(wargs);
#ifdef GPAW_OMP
  for(int i=1; i < nthds; i++)
    pthread_join(*(thds+i), NULL);
#endif
  free(wargs);
  free(thds);

  Py_RETURN_NONE;
}

static PyMethodDef Transformer_Methods[] = {
    {"apply", (PyCFunction)Transformer_apply, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyObject* Transformer_getattr(PyObject *obj, char *name)
{
    return Py_FindMethod(Transformer_Methods, obj, name);
}

static PyTypeObject TransformerType = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,
  "Transformer",
  sizeof(TransformerObject),
  0,
  (destructor)Transformer_dealloc,
  0,
  Transformer_getattr
};

PyObject * NewTransformerObject(PyObject *obj, PyObject *args)
{
  PyArrayObject* size;
  int k;
  PyArrayObject* paddings;
  PyArrayObject* npaddings;
  PyArrayObject* skip;
  PyArrayObject* neighbors;
  int real;
  PyObject* comm_obj;
  int interpolate;
  if (!PyArg_ParseTuple(args, "OiOOOOiOi",
                        &size, &k, &paddings, &npaddings, &skip,
                        &neighbors, &real, &comm_obj,
                        &interpolate))
    return NULL;

  TransformerObject* self = PyObject_NEW(TransformerObject, &TransformerType);
  if (self == NULL)
    return NULL;

  self->k = k;
  self->interpolate = interpolate;

  MPI_Comm comm = MPI_COMM_NULL;
  if (comm_obj != Py_None)
    comm = ((MPIObject*)comm_obj)->comm;

  const long (*nb)[2] = (const long (*)[2])LONGP(neighbors);
  const long (*pad)[2] = (const long (*)[2])LONGP(paddings);
  const long (*npad)[2] = (const long (*)[2])LONGP(npaddings);
  const long (*skp)[2] = (const long (*)[2])LONGP(skip);
  self->bc = bc_init(LONGP(size), pad, npad, nb, comm, real, 0);
  //const int* size1 = self->bc->size1;
  const int* size2 = self->bc->size2;

  for (int c = 0; c < 3; c++)
    for (int d = 0; d < 2; d++)
      self->skip[c][d] = (int)skp[c][d];

#ifndef GPAW_OMP
  self->buf = GPAW_MALLOC(double, size2[0] * size2[1] * size2[2] *
                          self->bc->ndouble);
  if (interpolate)
    // Much larger than necessary!  I don't have the energy right now to
    // estimate the minimum size of buf2!
    self->buf2 = GPAW_MALLOC(double, 16 * size2[0] * size2[1] * size2[2] *
                             self->bc->ndouble);
  else
    self->buf2 = GPAW_MALLOC(double, size2[0] * size2[1] *
                             //size1[2] / 2 *
                             (size2[2] - 2 * k + 3) / 2 *
                             self->bc->ndouble);
  self->sendbuf = GPAW_MALLOC(double, self->bc->maxsend);
  self->recvbuf = GPAW_MALLOC(double, self->bc->maxrecv);
#else
  int nthds = 1;
  if (getenv("OMP_NUM_THREADS") != NULL)
    nthds = atoi(getenv("OMP_NUM_THREADS"));
  self->buf = GPAW_MALLOC(double, size2[0] * size2[1] * size2[2] *
                          self->bc->ndouble * nthds);
  if (interpolate)
    // Much larger than necessary!  I don't have the energy right now to
    // estimate the minimum size of buf2!
    self->buf2 = GPAW_MALLOC(double, 16 * size2[0] * size2[1] * size2[2] *
                             self->bc->ndouble * nthds);
  else
    self->buf2 = GPAW_MALLOC(double, size2[0] * size2[1] *
                             //size1[2] / 2 *
                             (size2[2] - 2 * k + 3) / 2 *
                             self->bc->ndouble * nthds);
  self->sendbuf = GPAW_MALLOC(double, self->bc->maxsend * nthds);
  self->recvbuf = GPAW_MALLOC(double, self->bc->maxrecv * nthds);
#endif
  return (PyObject*)self;
}
