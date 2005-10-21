typedef struct
{
  PyObject_HEAD
  int size;
  int rank;
  MPI_Comm comm;
  PyObject* parent;
} MPIObject;

