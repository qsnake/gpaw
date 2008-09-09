#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "extensions.h"
#include "bc.h"
#include "mympi.h"
#include "bmgs/bmgs.h"

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

static PyObject* Transformer_apply(TransformerObject *self, PyObject *args)
{
  PyArrayObject* input;
  PyArrayObject* output;
  PyArrayObject* phases = 0;
  if (!PyArg_ParseTuple(args, "OO|O", &input, &output, &phases))
    return NULL;

  const double* in = DOUBLEP(input);
  double* out = DOUBLEP(output);
  bool real = (input->descr->type_num == PyArray_DOUBLE);
  const double_complex* ph = (real ? 0 : COMPLEXP(phases));

  const boundary_conditions* bc = self->bc;
#ifndef GPAW_OMP
  if (1)
#else
  if (bc->cfd == 0)
#endif
    {
      for (int i = 0; i < 3; i++)
        {
          bc_unpack1(bc, in, self->buf, i,
                     self->recvreq, self->sendreq,
                     self->recvbuf, self->sendbuf, ph + 2 * i, 0);
          bc_unpack2(bc, self->buf, i,
                     self->recvreq, self->sendreq, self->recvbuf);
        }
    }
  else
    {
      for (int i = 0; i < 3; i++)
        {
          MPI_Request recvreq[2];
          MPI_Request sendreq[2];
          double* sendbuf = self->sendbuf + i * bc->maxsend;
          double* recvbuf = self->recvbuf + i * bc->maxrecv;
          bc_unpack1(bc, in, self->buf, i,
                     recvreq, sendreq,
                     recvbuf, sendbuf, ph + 2 * i, 0);
          bc_unpack2(bc, self->buf, i,
                     recvreq, sendreq, recvbuf);
        }
    }
  if (real)
    {
      if (self->interpolate)
        bmgs_interpolate(self->k, self->skip, self->buf, bc->size2,
                         out, self->buf2);
      else
        bmgs_restrict(self->k, self->buf, bc->size2,
                      out, self->buf2);
    }
  else
    {
      if (self->interpolate)
        bmgs_interpolatez(self->k, self->skip, (double_complex*)self->buf,
                          bc->size2, (double_complex*)out,
                          (double_complex*)self->buf2);
      else
        bmgs_restrictz(self->k, (double_complex*)self->buf,
                       bc->size2, (double_complex*)out,
                       (double_complex*)self->buf2);
    }

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
#ifndef GPAW_OMP
  self->sendbuf = GPAW_MALLOC(double, self->bc->maxsend);
  self->recvbuf = GPAW_MALLOC(double, self->bc->maxrecv);
#else
  int nthds = 1;
  if (getenv("OMP_NUM_THREADS") != NULL)
    nthds = atoi(getenv("OMP_NUM_THREADS"));
  self->sendbuf = GPAW_MALLOC(double, self->bc->maxsend *
                              nthds);
  self->recvbuf = GPAW_MALLOC(double, self->bc->maxrecv *
                              nthds);
#endif
  return (PyObject*)self;
}
