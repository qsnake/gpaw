#include <Python.h>
#define NO_IMPORT_ARRAY
#include <Numeric/arrayobject.h>
#include "extensions.h"
#include "bmgs/bmgs.h"
#include "bc.h"
#include "mympi.h"

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

      // XXX
      int start[3] = {0, 0, 0};
      int size[3] = {size1[0], size1[1], size1[2]};
      for (int i = 0; i < 3; i++)
	if (bc->zero[i])
	  {
	    size[i] = 1;
	    if (real)
	      bmgs_zero(out, size1, start, size);
	    else
	      bmgs_zeroz((double_complex*)out, size1, start, size);
	    size[i] = size1[i];
	  }

      in += ng;
      out += ng;
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
  int angle;
  if (!PyArg_ParseTuple(args, "OOOiOiOii", 
                        &coefs, &offsets, &size, &range, &neighbors,
			&real, &comm_obj, &cfd, &angle))
    return NULL;

  OperatorObject *self = PyObject_NEW(OperatorObject, &OperatorType);
  if (self == NULL)
    return NULL;

  self->stencil = bmgs_stencil(coefs->dimensions[0], DOUBLEP(coefs),
			       INTP(offsets), range, INTP(size));

  const int (*nb)[2] = (const int (*)[2])INTP(neighbors);
  int padding[2] = {range, range};

  MPI_Comm comm = MPI_COMM_NULL;
  if (comm_obj != Py_None)
    comm = ((MPIObject*)comm_obj)->comm;

  self->bc = bc_init(INTP(size), padding, nb, comm, real, cfd, angle);

  const int* size2 = self->bc->size2;
  self->buf = (double*)malloc(size2[0] * size2[1] * size2[2] * 
			      self->bc->ndouble * sizeof(double));
  self->sendbuf = (double*)malloc(self->bc->maxsend * sizeof(double));
  self->recvbuf = (double*)malloc(self->bc->maxrecv * sizeof(double));
  return (PyObject*)self;
}
