/*
 *  Copyright (C) 2010       CSC - IT Center for Science Ltd.
 *  Please see the accompanying LICENSE file for further information. */

/* Routines needed for using parallel IO in combination with h5py */

#ifdef HDF5
#include <Python.h>
#include <hdf5.h>
#include <mpi.h>
#include "mympi.h"


extern void inith5(void);
extern void  inith5e(void);
extern void  inith5f(void);
extern void  inith5g(void);
extern void  inith5s(void);
extern void  inith5t(void);
extern void  inith5d(void);
extern void  inith5a(void);
extern void  inith5p(void);
extern void  inith5z(void);
extern void  inith5i(void);
extern void  inith5r(void);
extern void inith5fd(void);
extern void initutils(void);
extern void  inith5o(void);
extern void  inith5l(void);

void init_h5py()
{
  // Add h5py init functions to the list of builtin modules

  struct _inittab _h5py_Inittab[] = {
        {"h5", inith5},
        {"h5e", inith5e},
        {"h5f", inith5f},
        {"h5g", inith5g},
        {"h5s", inith5s},
        {"h5t", inith5t},
        {"h5d", inith5d},
        {"h5a", inith5a},
        {"h5p", inith5p},
        {"h5z", inith5z},
        {"h5i", inith5i},
        {"h5r", inith5r},
        {"h5fd", inith5fd},
        {"utils", initutils},
        {"h5o", inith5o},
        {"h5l", inith5l},
// Sentinel
        {0, 0}
  };
  PyImport_ExtendInittab(_h5py_Inittab);

}

PyObject* set_fapl_mpio(PyObject *self, PyObject *args)
{
  PyObject *comm_obj;
  int plist_id;
  if (!PyArg_ParseTuple(args, "iO", &plist_id, &comm_obj))
    return NULL;

  MPI_Comm comm = MPI_COMM_NULL;
  MPI_Info info = MPI_INFO_NULL;
  if (comm_obj != Py_None)
    {
      comm = ((MPIObject*)comm_obj)->comm;
      int nprocs;
      MPI_Comm_size(comm, &nprocs);
      char tmp[20];
      MPI_Info_create(&info);
      sprintf(tmp,"%d", nprocs);
      MPI_Info_set(info,"cb_nodes",tmp);
    }
  H5Pset_fapl_mpio(plist_id, comm, info);
  Py_RETURN_NONE;
}

PyObject* set_dxpl_mpio(PyObject *self, PyObject *args)
{
  int plist_id;
  if (!PyArg_ParseTuple(args, "i", &plist_id))
    return NULL;

  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
  Py_RETURN_NONE;
}
#endif

