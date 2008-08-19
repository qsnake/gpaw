#include <Python.h>
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "extensions.h"
#include "bc.h"
#include "mympi.h"
#include "bmgs/bmgs.h"

typedef struct 
{
  PyObject_HEAD
  bmgsstencil stencil;
  boundary_conditions* bc;
  MPI_Request recvreq[6];
  MPI_Request sendreq[6];
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
  int ng = bc->ndouble * size1[0] * size1[1] * size1[2];

  const double* in = DOUBLEP(input);
  double* out = DOUBLEP(output);
  const double_complex* ph;

  bool real = (input->descr->type_num == PyArray_DOUBLE);

  if (real)
    ph = 0;
  else
    ph = COMPLEXP(phases);

#ifndef GPAW_ASYNC
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
#else
  for (int n = 0; n < nin; n++)
    {
      double* pre_buf = self->buf + n%2;
      double* cur_buf = self->buf + (n+1)%2;
      MPI_Request* pre_recvreq = self->recvreq;
      MPI_Request* cur_recvreq = self->recvreq;
      MPI_Request* pre_sendreq = self->sendreq;
      MPI_Request* cur_sendreq = self->sendreq;
      double* pre_recvbuf = self->recvbuf;
      double* cur_recvbuf = self->recvbuf;
      double* sendbuf = self->sendbuf;

      for (int i = 0; i < 3; i++)
	{
	  bc_unpack1(bc, in, self->buf, i, 
		     self->recvreq + 2*i, self->sendreq + 2*i, 
		     self->recvbuf + bc->maxrecv*i, self->sendbuf + bc->maxsend*i, ph + 2 * i);
          bc_unpack2(bc, self->buf, i,
                     self->recvreq + 2*i, self->sendreq + 2*i, self->recvbuf + bc->maxrecv*i);
	}
  /*    for (int i = 0; i < 3; i++)
        {
          bc_unpack2(bc, self->buf, i,
                     self->recvreq + 2*i, self->sendreq + 2*i, self->recvbuf + bc->maxrecv*i);
        }*/
      if (real)
	bmgs_fd(&self->stencil, self->buf, out);
      else
	bmgs_fdz(&self->stencil, (const double_complex*)self->buf, 
		 (double_complex*)out);

      in += ng;
      out += ng;
    }
#endif
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
#ifndef GPAW_ASYNC
  self->buf = GPAW_MALLOC(double, size2[0] * size2[1] * size2[2] *
                          self->bc->ndouble);
  self->sendbuf = GPAW_MALLOC(double, self->bc->maxsend);
  self->recvbuf = GPAW_MALLOC(double, self->bc->maxrecv);
#else
  self->buf = GPAW_MALLOC(double, size2[0] * size2[1] * size2[2] * 
			  self->bc->ndouble);
  self->sendbuf = GPAW_MALLOC(double, self->bc->maxsend * 3);
  self->recvbuf = GPAW_MALLOC(double, self->bc->maxrecv * 3);
#endif
  return (PyObject*)self;
}
