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
  boundary_conditions* bc;
  int p;
  int k;
  bool interpolate;
  MPI_Request recvreq[2];
  MPI_Request sendreq[2];
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
  const double_complex* ph;
  bool real = (input->descr->type_num == PyArray_DOUBLE);
  if (real)
    ph = 0;
  else
    ph = COMPLEXP(phases);

  const boundary_conditions* bc = self->bc;
  for (int i = 0; i < 3; i++)
    {
      bc_unpack1(bc, in, self->buf, i, 
		 self->recvreq, self->sendreq, 
		 self->recvbuf, self->sendbuf, ph + 2 * i);
      bc_unpack2(bc, self->buf, i, 
		 self->recvreq, self->sendreq, self->recvbuf);
     }

  if (real)
    {
      if (self->interpolate)
	bmgs_interpolate(self->k, self->p, self->buf, bc->size2,
			 out, self->buf2);
      else
	bmgs_restrict(self->k, self->p, self->buf, bc->size2,
		      out, self->buf2);
    }
  else
    {
      if (self->interpolate)
	bmgs_interpolatez(self->k, self->p, (double_complex*)self->buf,
			  bc->size2, (double_complex*)out, 
			  (double_complex*)self->buf2);
      else
	bmgs_restrictz(self->k, self->p, (double_complex*)self->buf,
		       bc->size2, (double_complex*)out,
		       (double_complex*)self->buf2);
    }

  // XXX
  int start[3] = {0, 0, 0};
  int size[3];
  int size3[3];
  for (int i = 0; i < 3; i++)
    {
      if (self->interpolate)
	size[i] = bc->size1[i] * self->p;
      else
	size[i] = bc->size1[i] / self->p;
      size3[i] = size[i];
    }

  for (int i = 0; i < 3; i++)
    if (bc->zero[i])
      {
	size[i] = 1;
	if (real)
	  bmgs_zero(out, size3, start, size);
	else
	  bmgs_zeroz((double_complex*)out, size3, start, size);
	size[i] = size3[i];
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
  int p;
  int k;
  PyArrayObject* neighbors;
  int real;
  PyObject* comm_obj;
  int interpolate;
  int angle;
  if (!PyArg_ParseTuple(args, "OiiOiOii", 
                        &size, &p, &k, &neighbors, &real, &comm_obj,
			&interpolate, &angle))
    return NULL;

  TransformerObject* self = PyObject_NEW(TransformerObject, &TransformerType);
  if (self == NULL)
    return NULL;

  self->k = k;
  self->p = p;
  self->interpolate = interpolate;

  MPI_Comm comm = MPI_COMM_NULL;
  if (comm_obj != Py_None)
    comm = ((MPIObject*)comm_obj)->comm;

  const int (*nb)[2] = (const int (*)[2])INTP(neighbors);
  int padding[2] = {k * p / 2 - 1, k * p / 2 - p};
  if (interpolate)
    {
      padding[0] = k / 2 - 1;
      padding[1] = k / 2;
    }
  self->bc = bc_init(INTP(size), padding, nb, comm, real, 0, 
		     angle);
  const int* size1 = self->bc->size1;
  const int* size2 = self->bc->size2;

  self->buf = (double*)malloc(size2[0] * size2[1] * size2[2] * 
			      self->bc->ndouble * sizeof(double));
  if (interpolate)
    self->buf2 = (double*)malloc(size2[0] * size1[1] * size1[2] * p * p *
				 self->bc->ndouble * sizeof(double));
  else
    self->buf2 = (double*)malloc(size2[0] * size2[1] * size1[2] / p * 
				 self->bc->ndouble * sizeof(double));
  self->sendbuf = (double*)malloc(self->bc->maxsend * sizeof(double));
  self->recvbuf = (double*)malloc(self->bc->maxrecv * sizeof(double));
  return (PyObject*)self;
}
