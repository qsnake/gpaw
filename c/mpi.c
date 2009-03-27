#include <Python.h>
#ifdef PARALLEL
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <mpi.h>
#include "extensions.h"
#include <structmember.h>
#include "mympi.h"

// Check that array is well-behaved and contains data that can be sent.
#define CHK_ARRAY(a) if ((a) == NULL || !PyArray_Check(a)		\
			 || !PyArray_ISCARRAY(a) || !PyArray_ISNUMBER(a)) { \
    PyErr_SetString(PyExc_TypeError,					\
		    "Not a proper NumPy array for MPI communication."); \
    return NULL; }

// Check that two arrays have the same type, and the size of the
// second is a given multiple of the size of the first
#define CHK_ARRAYS(a,b,n)						\
  if ((PyArray_TYPE(a) != PyArray_TYPE(b))				\
      || (PyArray_SIZE(b) != PyArray_SIZE(a) * n)) {			\
    PyErr_SetString(PyExc_ValueError,					\
		    "Incompatible array types or sizes.");		\
      return NULL; }

// Check that a processor number is valid
#define CHK_PROC(n) if (n < 0 || n >= self->size) {\
    PyErr_SetString(PyExc_ValueError, "Invalid processor number.");	\
    return NULL; }

// Check that a processor number is valid or is -1
#define CHK_PROC_DEF(n) if (n < -1 || n >= self->size) {\
    PyErr_SetString(PyExc_ValueError, "Invalid processor number.");	\
    return NULL; }

// Check that a processor number is valid and is not this processor
#define CHK_OTHER_PROC(n) if (n < 0 || n >= self->size || n == self->rank) { \
    PyErr_SetString(PyExc_ValueError, "Invalid processor number.");	\
    return NULL; }

// Poor mans MPI request object, so we can store a reference to the buffer,
// preventing its early deallocation.  This should be replaced by a real
// (opaque) python object, so we can detect waiting multiple times on the same
// object.
typedef struct {
  MPI_Request rq;
  PyObject *buffer;
} mpi_request;

static void mpi_dealloc(MPIObject *obj)
{
  if (obj->comm != MPI_COMM_WORLD)
    {
      MPI_Comm_free(&(obj->comm));
      Py_XDECREF(obj->parent);
    }
  PyObject_DEL(obj);
}

static PyObject * mpi_receive(MPIObject *self, PyObject *args, PyObject *kwargs)
{
  PyObject* a;
  int src;
  int tag = 123;
  int block = 1;
  static char *kwlist[] = {"a", "src", "tag", "block", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|ii:receive", kwlist,
				   &a, &src, &tag, &block))
    return NULL;
  CHK_ARRAY(a);
  CHK_OTHER_PROC(src);
  int n = PyArray_DESCR(a)->elsize;
  for (int d = 0; d < PyArray_NDIM(a); d++)
    n *= PyArray_DIM(a, d);
  if (block)
    {
      MPI_Recv(PyArray_BYTES(a), n, MPI_BYTE, src, tag, self->comm,
	       MPI_STATUS_IGNORE);
      Py_RETURN_NONE;
    }
  else
    {
      mpi_request req;
      req.buffer = a;
      Py_INCREF(req.buffer);
      MPI_Irecv(PyArray_BYTES(a), n, MPI_BYTE, src, tag, self->comm, &(req.rq));
      return Py_BuildValue("s#", &req, sizeof(req));
    }
}

static PyObject * mpi_send(MPIObject *self, PyObject *args, PyObject *kwargs)
{
  PyObject* a;
  int dest;
  int tag = 123;
  int block = 1;
  static char *kwlist[] = {"a", "dest", "tag", "block", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|ii:send", kwlist,
				   &a, &dest, &tag, &block))
    return NULL;
  CHK_ARRAY(a);
  CHK_OTHER_PROC(dest);
  int n = PyArray_DESCR(a)->elsize;
  for (int d = 0; d < PyArray_NDIM(a); d++)
    n *= PyArray_DIM(a,d);
  if (block)
    {
      MPI_Send(PyArray_BYTES(a), n, MPI_BYTE, dest, tag, self->comm);
      Py_RETURN_NONE;
    }
  else
    {
      mpi_request req;
      req.buffer = a;
      Py_INCREF(a);
      MPI_Isend(PyArray_BYTES(a), n, MPI_BYTE, dest, tag, self->comm,
		&(req.rq));
      return Py_BuildValue("s#", &req, sizeof(req));
    }
}

static PyObject * mpi_ssend(MPIObject *self, PyObject *args, PyObject *kwargs)
{
  PyObject* a;
  int dest;
  int tag = 123;
  static char *kwlist[] = {"a", "dest", "tag", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|i:send", kwlist,
				   &a, &dest, &tag))
    return NULL;
  CHK_ARRAY(a);
  CHK_OTHER_PROC(dest);
  int n = PyArray_DESCR(a)->elsize;
  for (int d = 0; d < PyArray_NDIM(a); d++)
    n *= PyArray_DIM(a,d);
  MPI_Ssend(PyArray_BYTES(a), n, MPI_BYTE, dest, tag, self->comm);
  Py_RETURN_NONE;
}


static PyObject * mpi_name(MPIObject *self, PyObject *noargs)
{
  char name[MPI_MAX_PROCESSOR_NAME];
  int resultlen;
  MPI_Get_processor_name(name, &resultlen);
  return Py_BuildValue("s#", name, resultlen);
}

static PyObject * mpi_abort(MPIObject *self, PyObject *args)
{
  int errcode;
  if (!PyArg_ParseTuple(args, "i:abort", &errcode))
    return NULL;
  MPI_Abort(self->comm, errcode);
  Py_RETURN_NONE;
}

static PyObject * mpi_barrier(MPIObject *self)
{
  MPI_Barrier(self->comm);
  Py_RETURN_NONE;
}


static PyObject * mpi_wait(MPIObject *self, PyObject *args)
{
  mpi_request* s;
  int n;
  if (!PyArg_ParseTuple(args, "s#:wait", &s, &n))
    return NULL;
  if (n != sizeof(mpi_request))
    {
      PyErr_SetString(PyExc_TypeError, "Invalid MPI request object.");
      return NULL;
    }
  MPI_Wait(&(s->rq), MPI_STATUS_IGNORE); // Can this change the Python string?
  Py_DECREF(s->buffer);
  Py_RETURN_NONE;
}

static MPI_Datatype get_mpi_datatype(PyObject *a)
{
  int n = PyArray_DESCR(a)->elsize;
  if (PyArray_ISCOMPLEX(a))
    n = n/2;

  switch(PyArray_TYPE(a))
    {
      // Floating point numbers including complex numbers
    case NPY_DOUBLE:
    case NPY_CDOUBLE:
      assert(sizeof(double) == n);
      return MPI_DOUBLE;
    case NPY_FLOAT:
    case NPY_CFLOAT:
      assert(sizeof(float) == n);
      return MPI_FLOAT;
    case NPY_LONGDOUBLE:
    case NPY_CLONGDOUBLE:
       assert(sizeof(long double) == n);
      return MPI_LONG_DOUBLE;
      // Signed integer types
    case NPY_BYTE:
      assert(sizeof(char) == n);
      return MPI_CHAR;
    case NPY_SHORT:
      assert(sizeof(short) == n);
      return MPI_SHORT;
    case NPY_INT:
      assert(sizeof(int) == n);
      return MPI_INT;
    case NPY_LONG:
      assert(sizeof(long) == n);
      return MPI_LONG;
      // Unsigned integer types
    case NPY_BOOL:
    case NPY_UBYTE:
      assert(sizeof(unsigned char) == n);
      return MPI_UNSIGNED_CHAR;
    case NPY_USHORT:
      assert(sizeof(unsigned short) == n);
      return MPI_UNSIGNED_SHORT;
    case NPY_UINT:
      assert(sizeof(unsigned) == n);
      return MPI_UNSIGNED;
    case NPY_ULONG:
      assert(sizeof(unsigned long) == n);
      return MPI_UNSIGNED_LONG;
    }
  // If we reach this point none of the cases worked out.
  PyErr_SetString(PyExc_ValueError, "Cannot communicate data of this type.");
  return 0;
}

static PyObject * mpi_reduce(MPIObject *self, PyObject *args, PyObject *kwargs,
			     MPI_Op operation, int allowcomplex)
{
  PyObject* obj;
  int root = -1;
  static char *kwlist[] = {"a", "root", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|i:reduce", kwlist,
				   &obj, &root))
    return NULL;
  CHK_PROC_DEF(root);
  if (PyFloat_Check(obj))
    {
      double din = PyFloat_AS_DOUBLE(obj);
      double dout;
      if (root == -1)
        MPI_Allreduce(&din, &dout, 1, MPI_DOUBLE, operation, self->comm);
      else
        MPI_Reduce(&din, &dout, 1, MPI_DOUBLE, operation, root, self->comm);
      return PyFloat_FromDouble(dout);
    }
  if (PyInt_Check(obj))
    {
      long din = PyInt_AS_LONG(obj);
      long dout;
      if (root == -1)
        MPI_Allreduce(&din, &dout, 1, MPI_LONG, operation, self->comm);
      else
        MPI_Reduce(&din, &dout, 1, MPI_LONG, operation, root, self->comm);
      return PyInt_FromLong(dout);
    }
  else if (PyComplex_Check(obj) && allowcomplex)
    {
      double din[2];
      double dout[2];
      din[0] = PyComplex_RealAsDouble(obj);
      din[1] = PyComplex_ImagAsDouble(obj);
      if (root == -1)
        MPI_Allreduce(&din, &dout, 2, MPI_DOUBLE, MPI_SUM, self->comm);
      else
        MPI_Reduce(&din, &dout, 2, MPI_DOUBLE, MPI_SUM, root, self->comm);
      return PyComplex_FromDoubles(dout[0], dout[1]);
    }
  else if (PyComplex_Check(obj))
    {
      PyErr_SetString(PyExc_ValueError,
		      "Operation not allowed on complex numbers");
      return NULL;
    }   
  else   // It should be an array
    {
      int n;
      int elemsize;
      MPI_Datatype datatype;
      CHK_ARRAY(obj);
      datatype = get_mpi_datatype(obj);
      if (datatype == 0)
	return NULL;
      n = PyArray_SIZE(obj);
      elemsize = PyArray_DESCR(obj)->elsize;
      if (PyArray_ISCOMPLEX(obj))
	{
	  if (allowcomplex)
	    {
	      n *= 2;
	      elemsize /= 2;
	    }
	  else
	    {
	      PyErr_SetString(PyExc_ValueError,
			      "Operation not allowed on complex numbers");
	      return NULL;
	    }
	}
      if (root == -1)
        {
          char* b = GPAW_MALLOC(char, n * elemsize);
          // XXX Use MPI_IN_PLACE!!
          MPI_Allreduce(PyArray_BYTES(obj), b, n, datatype, operation,
			self->comm);
	  assert(PyArray_NBYTES(obj) == n * elemsize);
          memcpy(PyArray_BYTES(obj), b, n * elemsize);
          free(b);
        }
      else
        {
          char* b = 0;
          int rank;
          MPI_Comm_rank(self->comm, &rank);
#ifdef GPAW_BGP
          b = GPAW_MALLOC(char, n * elemsize); // bug on BGP
#else
          if (rank == root)
               b = GPAW_MALLOC(char, n * elemsize);
#endif
          // XXX Use MPI_IN_PLACE!!
          MPI_Reduce(PyArray_BYTES(obj), b, n, datatype, operation, root,
		     self->comm);
          if (rank == root)
            {
	      assert(PyArray_NBYTES(obj) == n * elemsize);
              memcpy(PyArray_BYTES(obj), b, n * elemsize);
            }
#ifdef GPAW_BGP
          free(b); // bug on BGP
#else
          if (rank == root)
               free(b);
#endif
        }
      Py_RETURN_NONE;
    }
}

static PyObject * mpi_sum(MPIObject *self, PyObject *args, PyObject *kwargs)
{
  return mpi_reduce(self, args, kwargs, MPI_SUM, 1);
}

static PyObject * mpi_product(MPIObject *self, PyObject *args, PyObject *kwargs)
{
  // No complex numbers as that would give separate products of
  // real and imaginary parts.
  return mpi_reduce(self, args, kwargs,  MPI_PROD, 0); 
}

static PyObject * mpi_max(MPIObject *self, PyObject *args, PyObject *kwargs)
{
  return mpi_reduce(self, args,  kwargs, MPI_MAX, 0);
}

static PyObject * mpi_min(MPIObject *self, PyObject *args, PyObject *kwargs)
{
  return mpi_reduce(self, args,  kwargs, MPI_MIN, 0);
}

static PyObject * mpi_scatter(MPIObject *self, PyObject *args)
{
  PyObject* sendobj;
  PyObject* recvobj;
  int root;
  if (!PyArg_ParseTuple(args, "OOi:scatter", &sendobj, &recvobj, &root))
    return NULL;
  CHK_ARRAY(sendobj);
  CHK_ARRAY(recvobj);
  CHK_PROC(root);
  CHK_ARRAYS(recvobj, sendobj, self->size); // size(send) = size(recv)*Ncpu
  int n = PyArray_DESCR(recvobj)->elsize;
  for (int d = 0; d < PyArray_NDIM(recvobj); d++)
    n *= PyArray_DIM(recvobj,d);
  MPI_Scatter(PyArray_BYTES(sendobj), n, MPI_BYTE, PyArray_BYTES(recvobj),
	      n, MPI_BYTE, root, self->comm);
  Py_RETURN_NONE;
}



static PyObject * mpi_allgather(MPIObject *self, PyObject *args)
{
  PyArrayObject* a;
  PyArrayObject* b;
  if (!PyArg_ParseTuple(args, "OO:allgather", &a, &b))
    return NULL;
  CHK_ARRAY(a);
  CHK_ARRAY(b);
  CHK_ARRAYS(a, b, self->size);
  int n = PyArray_DESCR(a)->elsize;
  for (int d = 0; d < PyArray_NDIM(a); d++)
    n *= PyArray_DIM(a,d);
  // What about endianness???? 
  MPI_Allgather(PyArray_BYTES(a), n, MPI_BYTE, PyArray_BYTES(b), n,
		MPI_BYTE, self->comm);
  Py_RETURN_NONE;
}

static PyObject * mpi_gather(MPIObject *self, PyObject *args)
{
  PyObject* a;
  int root;
  PyObject* b = 0;
  if (!PyArg_ParseTuple(args, "Oi|O", &a, &root, &b))
    return NULL;
  CHK_ARRAY(a);
  CHK_PROC(root);
  if (root == self->rank)
    {
      CHK_ARRAY(b);
      CHK_ARRAYS(a, b, self->size);
    }
  else if (b != Py_None && b != NULL)
    {
      fprintf(stderr, "******** Root=%d\n", root);
      PyErr_SetString(PyExc_ValueError,	 
		      "mpi_gather: b array should not be given on non-root processors.");
      return NULL;
    }
  int n = PyArray_DESCR(a)->elsize;
  for (int d = 0; d < PyArray_NDIM(a); d++)
    n *= PyArray_DIM(a,d);
  if (root != self->rank)
    MPI_Gather(PyArray_BYTES(a), n, MPI_BYTE, 0, n, MPI_BYTE, root, self->comm);
  else
    MPI_Gather(PyArray_BYTES(a), n, MPI_BYTE, PyArray_BYTES(b), n, MPI_BYTE, root, self->comm);
  Py_RETURN_NONE;
}

static PyObject * mpi_broadcast(MPIObject *self, PyObject *args)
{
  PyObject* buf;
  int root;
  if (!PyArg_ParseTuple(args, "Oi:broadcast", &buf, &root))
    return NULL;
  CHK_ARRAY(buf);
  CHK_PROC(root);
  int n = PyArray_DESCR(buf)->elsize;
  for (int d = 0; d < PyArray_NDIM(buf); d++)
    n *= PyArray_DIM(buf,d);
  MPI_Bcast(PyArray_BYTES(buf), n, MPI_BYTE, root, self->comm);
  Py_RETURN_NONE;
}

static PyObject * mpi_cart_create(MPIObject *self, PyObject *args)
{
  int dimx, dimy, dimz;
  int periodic = 0;

  if (!PyArg_ParseTuple(args, "iii|i:cart_create", &dimx,
                        &dimy, &dimz, &periodic))
    return NULL;

  int dims[3] = {dimx, dimy, dimz};
  int periods[3] = {periodic, periodic, periodic};
  MPI_Comm comm_new;
  MPI_Cart_create(self->comm, 3, dims, periods, 1, &comm_new);
  return Py_BuildValue("s#", &comm_new, sizeof(comm_new));
  // This looks wrong!  Shouldn't it return a new communincator object?
}

#ifdef GPAW_WITH_SL
#include "scalapack.c"
#endif

// Forward declaration of MPI_Communicator because it needs MPIType
// that needs MPI_getattr that needs MPI_Methods that need
// MPI_Communicator that need ...
static PyObject * MPICommunicator(MPIObject *self, PyObject *args);

static PyMethodDef mpi_methods[] = {
    {"receive",          (PyCFunction)mpi_receive,
     METH_VARARGS|METH_KEYWORDS,
     "receive(a, src, tag=123, block=1) receives array a from src."},
    {"send",             (PyCFunction)mpi_send,
     METH_VARARGS|METH_KEYWORDS,
     "send(a, dest, tag=123, block=1) sends array a to dest."},
    {"ssend",             (PyCFunction)mpi_ssend,
     METH_VARARGS|METH_KEYWORDS,
     "ssend(a, dest, tag=123) synchronously sends array a to dest."},
    {"abort",            (PyCFunction)mpi_abort,        METH_VARARGS,
     "abort(errcode) aborts all MPI tasks."},
    {"name",             (PyCFunction)mpi_name,         METH_NOARGS,
     "name() returns the name of the processor node."},
    {"barrier",          (PyCFunction)mpi_barrier,      METH_VARARGS,
     "barrier() synchronizes all MPI tasks"},
    {"wait",             (PyCFunction)mpi_wait,         METH_VARARGS,
     "wait(request) waits for a nonblocking communication to complete."},
    {"sum",              (PyCFunction)mpi_sum,
     METH_VARARGS|METH_KEYWORDS,
     "sum(a, root=-1) sums arrays, result on all tasks unless root is given."},
    {"product",          (PyCFunction)mpi_product,
     METH_VARARGS|METH_KEYWORDS,
     "product(a, root=-1) multiplies arrays, result on all tasks unless root is given."},
    {"max",              (PyCFunction)mpi_max,
     METH_VARARGS|METH_KEYWORDS,
     "max(a, root=-1) maximum of arrays, result on all tasks unless root is given."},
    {"min",              (PyCFunction)mpi_min,
     METH_VARARGS|METH_KEYWORDS,
     "min(a, root=-1) minimum of arrays, result on all tasks unless root is given."},
    {"scatter",          (PyCFunction)mpi_scatter,      METH_VARARGS,
     "scatter(src, target, root) distributes array from root task."},
    {"gather",           (PyCFunction)mpi_gather,       METH_VARARGS,
     "gather(src, root, target=None) gathers data from all tasks on root task."},
    {"all_gather",       (PyCFunction)mpi_allgather,    METH_VARARGS,
     "all_gather(src, target) gathers data from all tasks on all tasks."},
    {"broadcast",        (PyCFunction)mpi_broadcast,    METH_VARARGS,
     "broadcast(buffer, root) Broadcast data in-place from root task."},
#ifdef GPAW_WITH_SL
    {"diagonalize",      (PyCFunction)diagonalize,      METH_VARARGS, 0},
    {"inverse_cholesky", (PyCFunction)inverse_cholesky, METH_VARARGS, 0},
#endif
    {"new_communicator", (PyCFunction)MPICommunicator,  METH_VARARGS,
     "new_communicator(ranks) creates a new communicator."},
    {"cart_create",      (PyCFunction)mpi_cart_create,  METH_VARARGS,
     "cart_create(nx, ny, nz, periodic=0) creates cartesian grid (BROKEN?)"},
    {0, 0, 0, 0}
};

static PyMemberDef mpi_members[] = {
  {"size", T_INT, offsetof(MPIObject, size), 0, "Number of processors"},
  {"rank", T_INT, offsetof(MPIObject, rank), 0, "Number of this processor"},
  {0, 0, 0, 0, 0}  /* Sentinel */
};

// __new__
static PyObject *NewMPIObject(PyTypeObject* type, PyObject *args, PyObject *kwds)
{
  static char *kwlist[] = {NULL};
  MPIObject* self;
  
  if (! PyArg_ParseTupleAndKeywords(args, kwds, "", kwlist))
    return NULL;

  self = (MPIObject *) type->tp_alloc(type, 0);
  if (self == NULL)
    return NULL;
  
  MPI_Comm_size(MPI_COMM_WORLD, &(self->size));
  MPI_Comm_rank(MPI_COMM_WORLD, &(self->rank));
  self->comm = MPI_COMM_WORLD;
  self->parent = NULL;

  return (PyObject *) self;
}

// __init__ does nothing.
static int InitMPIObject(MPIObject* self, PyObject *args, PyObject *kwds)
{
  static char *kwlist[] = {NULL};
  
  if (! PyArg_ParseTupleAndKeywords(args, kwds, "", kwlist))
    return -1;
  
  return 0;
}


PyTypeObject MPIType = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                         /*ob_size*/
  "MPI",             /*tp_name*/
  sizeof(MPIObject),             /*tp_basicsize*/
  0,                         /*tp_itemsize*/
  (destructor)mpi_dealloc, /*tp_dealloc*/
  0,                         /*tp_print*/
  0,                         /*tp_getattr*/
  0,                         /*tp_setattr*/
  0,                         /*tp_compare*/
  0,                         /*tp_repr*/
  0,                         /*tp_as_number*/
  0,                         /*tp_as_sequence*/
  0,                         /*tp_as_mapping*/
  0,                         /*tp_hash */
  0,                         /*tp_call*/
  0,                         /*tp_str*/
  0,                         /*tp_getattro*/
  0,                         /*tp_setattro*/
  0,                         /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
  "MPI object",           /* tp_doc */
  0,                   /* tp_traverse */
  0,                   /* tp_clear */
  0,                   /* tp_richcompare */
  0,                   /* tp_weaklistoffset */
  0,                   /* tp_iter */
  0,                   /* tp_iternext */
  mpi_methods,             /* tp_methods */
  mpi_members,
  0,                         /* tp_getset */
  0,                         /* tp_base */
  0,                         /* tp_dict */
  0,                         /* tp_descr_get */
  0,                         /* tp_descr_set */
  0,                         /* tp_dictoffset */
  (initproc)InitMPIObject,      /* tp_init */
  0,                         /* tp_alloc */
  NewMPIObject,                 /* tp_new */
};


static PyObject * MPICommunicator(MPIObject *self, PyObject *args)
{
  PyObject* ranks;
  if (!PyArg_ParseTuple(args, "O", &ranks))
    return NULL;
  CHK_ARRAY(ranks);
  if (PyArray_NDIM(ranks) != 1 || PyArray_TYPE(ranks) != NPY_LONG)
    {
      PyErr_SetString(PyExc_TypeError,
		      "ranks must be a onedimensional array of ints.");
      return NULL;
    }
  PyObject *iranks;
  int n = PyArray_DIM(ranks, 0);
  iranks = PyArray_Cast((PyArrayObject*) ranks, NPY_INT);
  if (iranks == NULL)
    return NULL;
  // Check that all ranks make sense
  for (int i = 0; i < n; i++)
    {
      int *x = PyArray_GETPTR1(iranks, i);
      if (*x < 0 || *x >= self->size)
	{
	  Py_DECREF(iranks);
	  PyErr_SetString(PyExc_ValueError, "invalid rank");
	  return NULL;
	}
      for (int j = 0; j < i; j++)
	{
	  int *y = PyArray_GETPTR1(iranks, j);
	  if (*y == *x)
	    {
	      Py_DECREF(iranks);
	      PyErr_SetString(PyExc_ValueError, "duplicate rank");
	      return NULL;
	    }
	}
    }
  MPI_Group group;
  MPI_Comm_group(self->comm, &group);
  MPI_Group newgroup;
  MPI_Group_incl(group, n, (int *) PyArray_BYTES(iranks), &newgroup);
  Py_DECREF(iranks);
  MPI_Comm comm;
  MPI_Comm_create(self->comm, newgroup, &comm);
  MPI_Group_free(&newgroup);
  MPI_Group_free(&group);
  if (comm == MPI_COMM_NULL)
    {
      Py_RETURN_NONE;
    }
  else
    {
      MPIObject *obj = PyObject_NEW(MPIObject, &MPIType);
      if (obj == NULL)
        return NULL;
      MPI_Comm_size(comm, &(obj->size));
      MPI_Comm_rank(comm, &(obj->rank));
      obj->comm = comm;
      // Make sure that MPI_COMM_WORLD is kept alive til the end (we
      // don't want MPI_Finalize to be called before MPI_Comm_free):
      Py_INCREF(self);
      obj->parent = (PyObject*)self;
      return (PyObject*)obj;
    }
}
#endif //PARALLEL
