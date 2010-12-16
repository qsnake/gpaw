/*  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2007-2009  CAMd
 *  Copyright (C) 2007-2010  CSC - IT Center for Science Ltd.
 *  Please see the accompanying LICENSE file for further information. */

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#include <numpy/arrayobject.h>

#ifdef GPAW_HPM
void HPM_Init(void);
void HPM_Start(char *);
void HPM_Stop(char *);
void HPM_Print(void);
void HPM_Print_Flops(void);
PyObject* ibm_hpm_start(PyObject *self, PyObject *args);
PyObject* ibm_hpm_stop(PyObject *self, PyObject *args);
#endif

#ifdef GPAW_PAPI
#include <papi.h>
#endif

#ifdef GPAW_CRAYPAT
#include <pat_api.h>
PyObject* craypat_region_begin(PyObject *self, PyObject *args);
PyObject* craypat_region_end(PyObject *self, PyObject *args);
#endif


PyObject* symmetrize(PyObject *self, PyObject *args);
PyObject* symmetrize_wavefunction(PyObject *self, PyObject *args);
PyObject* scal(PyObject *self, PyObject *args);
PyObject* gemm(PyObject *self, PyObject *args);
PyObject* gemv(PyObject *self, PyObject *args);
PyObject* axpy(PyObject *self, PyObject *args);
PyObject* d2Excdnsdnt(PyObject *self, PyObject *args);
PyObject* d2Excdn2(PyObject *self, PyObject *args);
PyObject* rk(PyObject *self, PyObject *args);
PyObject* r2k(PyObject *self, PyObject *args);
PyObject* dotc(PyObject *self, PyObject *args);
PyObject* dotu(PyObject *self, PyObject *args);
PyObject* multi_dotu(PyObject *self, PyObject *args);
PyObject* multi_axpy(PyObject *self, PyObject *args);
PyObject* diagonalize(PyObject *self, PyObject *args);
PyObject* diagonalize_mr3(PyObject *self, PyObject *args);
PyObject* general_diagonalize(PyObject *self, PyObject *args);
PyObject* inverse_cholesky(PyObject *self, PyObject *args);
PyObject* inverse_symmetric(PyObject *self, PyObject *args);
PyObject* inverse_general(PyObject *self, PyObject *args);
PyObject* linear_solve_band(PyObject *self, PyObject *args);
PyObject* linear_solve_tridiag(PyObject *self, PyObject *args);
PyObject* right_eigenvectors(PyObject *self, PyObject *args);
PyObject* NewLocalizedFunctionsObject(PyObject *self, PyObject *args);
PyObject* NewOperatorObject(PyObject *self, PyObject *args);
PyObject* NewSplineObject(PyObject *self, PyObject *args);
PyObject* NewTransformerObject(PyObject *self, PyObject *args);
PyObject* pc_potential(PyObject *self, PyObject *args);
PyObject* pc_potential_value(PyObject *self, PyObject *args);
PyObject* elementwise_multiply_add(PyObject *self, PyObject *args);
PyObject* utilities_gaussian_wave(PyObject *self, PyObject *args);
PyObject* utilities_vdot(PyObject *self, PyObject *args);
PyObject* utilities_vdot_self(PyObject *self, PyObject *args);
PyObject* errorfunction(PyObject *self, PyObject *args);
PyObject* cerf(PyObject *self, PyObject *args);
PyObject* unpack(PyObject *self, PyObject *args);
PyObject* unpack_complex(PyObject *self, PyObject *args);
PyObject* hartree(PyObject *self, PyObject *args);
PyObject* localize(PyObject *self, PyObject *args);
PyObject* NewXCFunctionalObject(PyObject *self, PyObject *args);
PyObject* NewlxcXCFunctionalObject(PyObject *self, PyObject *args);
PyObject* exterior_electron_density_region(PyObject *self, PyObject *args);
PyObject* plane_wave_grid(PyObject *self, PyObject *args);
PyObject* overlap(PyObject *self, PyObject *args);
PyObject* vdw(PyObject *self, PyObject *args);
PyObject* vdw2(PyObject *self, PyObject *args);
PyObject* swap_arrays(PyObject *self, PyObject *args);
PyObject* spherical_harmonics(PyObject *self, PyObject *args);
PyObject* spline_to_grid(PyObject *self, PyObject *args);
PyObject* NewLFCObject(PyObject *self, PyObject *args);
PyObject* compiled_WITH_SL(PyObject *self, PyObject *args);
#if defined(GPAW_WITH_SL) && defined(PARALLEL)
PyObject* new_blacs_context(PyObject *self, PyObject *args);
PyObject* get_blacs_gridinfo(PyObject* self, PyObject *args);
PyObject* get_blacs_local_shape(PyObject* self, PyObject *args);
PyObject* blacs_destroy(PyObject *self, PyObject *args);
PyObject* scalapack_set(PyObject *self, PyObject *args);
PyObject* scalapack_redist(PyObject *self, PyObject *args);
PyObject* scalapack_diagonalize_dc(PyObject *self, PyObject *args);
PyObject* scalapack_diagonalize_ex(PyObject *self, PyObject *args);
#ifdef GPAW_MR3
PyObject* scalapack_diagonalize_mr3(PyObject *self, PyObject *args);
#endif
PyObject* scalapack_general_diagonalize_dc(PyObject *self, PyObject *args);
PyObject* scalapack_general_diagonalize_ex(PyObject *self, PyObject *args);
#ifdef GPAW_MR3
PyObject* scalapack_general_diagonalize_mr3(PyObject *self, PyObject *args);
#endif 
PyObject* scalapack_inverse_cholesky(PyObject *self, PyObject *args);
PyObject* pblas_gemm(PyObject *self, PyObject *args);
PyObject* pblas_gemv(PyObject *self, PyObject *args);
PyObject* pblas_r2k(PyObject *self, PyObject *args);
PyObject* pblas_rk(PyObject *self, PyObject *args);
#endif

// Moving least squares interpolation
PyObject* mlsqr(PyObject *self, PyObject *args); 

// Parallel HDF5
#ifdef HDF5
void init_h5py();
PyObject* set_fapl_mpio(PyObject *self, PyObject *args);
PyObject* set_dxpl_mpio(PyObject *self, PyObject *args);
#endif

// IO wrappers
#ifdef IO_WRAPPERS
void init_io_wrappers();
#endif
PyObject* Py_enable_io_wrappers(PyObject *self, PyObject *args);
PyObject* Py_disable_io_wrappers(PyObject *self, PyObject *args);

static PyMethodDef functions[] = {
  {"symmetrize", symmetrize, METH_VARARGS, 0},
  {"symmetrize_wavefunction", symmetrize_wavefunction, METH_VARARGS, 0},
  {"scal", scal, METH_VARARGS, 0},
  {"gemm", gemm, METH_VARARGS, 0},
  {"gemv", gemv, METH_VARARGS, 0},
  {"axpy", axpy, METH_VARARGS, 0},
  {"d2Excdnsdnt", d2Excdnsdnt, METH_VARARGS, 0},
  {"d2Excdn2", d2Excdn2, METH_VARARGS, 0},
  {"rk",  rk,  METH_VARARGS, 0},
  {"r2k", r2k, METH_VARARGS, 0},
  {"dotc", dotc, METH_VARARGS, 0},
  {"dotu", dotu, METH_VARARGS, 0},
  {"multi_dotu", multi_dotu, METH_VARARGS, 0},
  {"multi_axpy", multi_axpy, METH_VARARGS, 0},
  {"diagonalize", diagonalize, METH_VARARGS, 0},
  {"diagonalize_mr3", diagonalize_mr3, METH_VARARGS, 0},
  {"general_diagonalize", general_diagonalize, METH_VARARGS, 0},
  {"inverse_cholesky", inverse_cholesky, METH_VARARGS, 0},
  {"inverse_symmetric", inverse_symmetric, METH_VARARGS, 0},
  {"inverse_general", inverse_general, METH_VARARGS, 0},
  {"linear_solve_band", linear_solve_band, METH_VARARGS, 0},
  {"linear_solve_tridiag", linear_solve_tridiag, METH_VARARGS, 0},
  {"right_eigenvectors", right_eigenvectors, METH_VARARGS, 0},
  {"LocalizedFunctions", NewLocalizedFunctionsObject, METH_VARARGS, 0},
  {"Operator", NewOperatorObject, METH_VARARGS, 0},
  {"Spline", NewSplineObject, METH_VARARGS, 0},
  {"Transformer", NewTransformerObject, METH_VARARGS, 0},
  {"elementwise_multiply_add", elementwise_multiply_add, METH_VARARGS, 0},
  {"utilities_gaussian_wave", utilities_gaussian_wave, METH_VARARGS, 0},
  {"utilities_vdot", utilities_vdot, METH_VARARGS, 0},
  {"utilities_vdot_self", utilities_vdot_self, METH_VARARGS, 0},
  {"eed_region", exterior_electron_density_region, METH_VARARGS, 0},
  {"plane_wave_grid", plane_wave_grid, METH_VARARGS, 0},
  {"erf",        errorfunction,        METH_VARARGS, 0},
  {"cerf",       cerf,        METH_VARARGS, 0},
  {"unpack",       unpack,           METH_VARARGS, 0},
  {"unpack_complex",       unpack_complex,           METH_VARARGS, 0},
  {"hartree",        hartree,        METH_VARARGS, 0},
  {"localize",       localize,        METH_VARARGS, 0},
  {"XCFunctional",    NewXCFunctionalObject,    METH_VARARGS, 0},
  /*  {"MGGAFunctional",    NewMGGAFunctionalObject,    METH_VARARGS, 0},*/
  {"lxcXCFunctional",    NewlxcXCFunctionalObject,    METH_VARARGS, 0},
  {"overlap",       overlap,        METH_VARARGS, 0},
  {"vdw", vdw, METH_VARARGS, 0},
  {"vdw2", vdw2, METH_VARARGS, 0},
  {"swap", swap_arrays, METH_VARARGS, 0},
  {"spherical_harmonics", spherical_harmonics, METH_VARARGS, 0},
  {"compiled_with_sl", compiled_WITH_SL, METH_VARARGS, 0},
  {"pc_potential", pc_potential, METH_VARARGS, 0},
  {"pc_potential_value", pc_potential_value, METH_VARARGS, 0},
  {"spline_to_grid", spline_to_grid, METH_VARARGS, 0},
  {"LFC", NewLFCObject, METH_VARARGS, 0},
  /*
  {"calculate_potential_matrix", calculate_potential_matrix, METH_VARARGS, 0},
  {"construct_density", construct_density, METH_VARARGS, 0},
  {"construct_density1", construct_density1, METH_VARARGS, 0},
  */
#if defined(GPAW_WITH_SL) && defined(PARALLEL)
  {"new_blacs_context", new_blacs_context, METH_VARARGS, NULL},
  {"get_blacs_gridinfo", get_blacs_gridinfo, METH_VARARGS, NULL},
  {"get_blacs_local_shape", get_blacs_local_shape, METH_VARARGS, NULL},
  {"blacs_destroy",     blacs_destroy,      METH_VARARGS, 0},
  {"scalapack_set", scalapack_set, METH_VARARGS, 0}, 
  {"scalapack_redist",      scalapack_redist,     METH_VARARGS, 0},
  {"scalapack_diagonalize_dc", scalapack_diagonalize_dc, METH_VARARGS, 0}, 
  {"scalapack_diagonalize_ex", scalapack_diagonalize_ex, METH_VARARGS, 0},
#ifdef GPAW_MR3
  {"scalapack_diagonalize_mr3", scalapack_diagonalize_mr3, METH_VARARGS, 0},
#endif // GPAW_MR3
  {"scalapack_general_diagonalize_dc", 
   scalapack_general_diagonalize_dc, METH_VARARGS, 0},
  {"scalapack_general_diagonalize_ex", 
   scalapack_general_diagonalize_ex, METH_VARARGS, 0},
#ifdef GPAW_MR3
  {"scalapack_general_diagonalize_mr3",
   scalapack_general_diagonalize_mr3, METH_VARARGS, 0},
#endif // GPAW_MR3
  {"scalapack_inverse_cholesky", scalapack_inverse_cholesky, METH_VARARGS, 0},
  {"pblas_gemm", pblas_gemm, METH_VARARGS, 0},
  {"pblas_gemv", pblas_gemv, METH_VARARGS, 0},
  {"pblas_r2k", pblas_r2k, METH_VARARGS, 0},
  {"pblas_rk", pblas_rk, METH_VARARGS, 0},
#endif // GPAW_WITH_SL && PARALLEL
#ifdef GPAW_HPM
  {"hpm_start", ibm_hpm_start, METH_VARARGS, 0},
  {"hpm_stop", ibm_hpm_stop, METH_VARARGS, 0},
#endif // GPAW_HPM
#ifdef GPAW_CRAYPAT
  {"craypat_region_begin", craypat_region_begin, METH_VARARGS, 0},
  {"craypat_region_end", craypat_region_end, METH_VARARGS, 0},
#endif // GPAW_CRAYPAT
  {"mlsqr", mlsqr, METH_VARARGS, 0}, 
#ifdef HDF5
  {"h5_set_fapl_mpio", set_fapl_mpio, METH_VARARGS, 0}, 
  {"h5_set_dxpl_mpio", set_dxpl_mpio, METH_VARARGS, 0}, 
#endif // HDF5
  {"enable_io_wrappers", Py_enable_io_wrappers, METH_VARARGS, 0},
  {"disable_io_wrappers", Py_disable_io_wrappers, METH_VARARGS, 0},
  {0, 0, 0, 0}
};

#ifdef PARALLEL
extern PyTypeObject MPIType;
#endif

#ifndef GPAW_INTERPRETER
PyMODINIT_FUNC init_gpaw(void)
{
#ifdef PARALLEL
  if (PyType_Ready(&MPIType) < 0)
    return;
#endif

  PyObject* m = Py_InitModule3("_gpaw", functions,
             "C-extension for GPAW\n\n...\n");
  if (m == NULL)
    return;

#ifdef PARALLEL
  Py_INCREF(&MPIType);
  PyModule_AddObject(m, "Communicator", (PyObject *)&MPIType);
#endif

  import_array();
}
#endif

#ifdef NO_SOCKET
/*dummy socket module for systems which do not support sockets */
PyMODINIT_FUNC initsocket(void)
{
  Py_InitModule("socket", NULL);
  return;
}
#endif

#ifdef GPAW_INTERPRETER
extern DL_EXPORT(int) Py_Main(int, char **);

#include <mpi.h>

int
main(int argc, char **argv)
{
  int status;

#ifdef GPAW_CRAYPAT
  PAT_region_begin(1, "C-Initializations");
#endif

#ifndef GPAW_OMP
  MPI_Init(&argc, &argv);
#else
  int granted;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &granted);
  if(granted != MPI_THREAD_MULTIPLE) exit(1);
#endif // GPAW_OMP

#ifdef GPAW_PAPI
  float rtime, ptime, mflops;
  long_long flpops;
  PAPI_flops( &rtime, &ptime, &flpops, &mflops );
#endif

#ifdef IO_WRAPPERS
  init_io_wrappers();
#endif

#ifdef GPAW_MPI_MAP
  int tag = 99;
  int myid, numprocs, i, procnamesize;
  char procname[MPI_MAX_PROCESSOR_NAME];
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs );
  MPI_Comm_rank(MPI_COMM_WORLD, &myid );
  MPI_Get_processor_name(procname, &procnamesize);
  if (myid > 0) {
      MPI_Send(&procnamesize, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
      MPI_Send(procname, procnamesize, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
  }
  else {
      printf("MPI_COMM_SIZE is %d \n", numprocs);
      printf("%s \n", procname);
      
      for (i = 1; i < numprocs; ++i) {
	  MPI_Recv(&procnamesize, 1, MPI_INT, i, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(procname, procnamesize, MPI_CHAR, i, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  printf("%s \n", procname);
      }
  }
#endif // GPAW_MPI_MAP

#ifdef GPAW_HPM
  HPM_Init();
  HPM_Start("GPAW");
#endif 


#ifdef GPAW_MPI_DEBUG
  // Default Errhandler is MPI_ERRORS_ARE_FATAL
  MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
#endif

#ifdef HDF5
  init_h5py();
#endif

  Py_Initialize();

#ifdef NO_SOCKET
  initsocket();
#endif

  if (PyType_Ready(&MPIType) < 0)
    return -1;

  PyObject* m = Py_InitModule3("_gpaw", functions,
             "C-extension for GPAW\n\n...\n");
  if (m == NULL)
    return -1;

  Py_INCREF(&MPIType);
  PyModule_AddObject(m, "Communicator", (PyObject *)&MPIType);
  import_array1(-1);
  MPI_Barrier(MPI_COMM_WORLD);
#ifdef GPAW_CRAYPAT
  PAT_region_end(1);
#endif
  status = Py_Main(argc, argv);
#ifdef GPAW_HPM
  HPM_Stop("GPAW");
  HPM_Print();
  HPM_Print_Flops();
#endif
#ifdef GPAW_PAPI
  // print out some statistics
  int myid, numprocs;
  long_long sum_flpops;
  float ave_mflops;
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs );
  MPI_Comm_rank(MPI_COMM_WORLD, &myid );
  PAPI_flops( &rtime, &ptime, &flpops, &mflops );
  MPI_Reduce(&mflops, &ave_mflops, 1, MPI_FLOAT, MPI_SUM, 0, 
	     MPI_COMM_WORLD);
  ave_mflops /= numprocs;
  MPI_Reduce(&flpops, &sum_flpops, 1, MPI_LONG_LONG, MPI_SUM, 0, 
	     MPI_COMM_WORLD);
  if (myid == 0)
    {
      printf("GPAW: Information from PAPI counters\n");
      printf("GPAW: Total time: %10.2f s\n", rtime);
      printf("GPAW: Floating point operations: %lld \n", sum_flpops);
      printf("GPAW: FLOP rate (GFLOP/s):  ave %10.3f  total %10.3f\n",
	     ave_mflops / 1000.0 , sum_flpops / rtime / 1.e9);
    }
#endif      
  MPI_Finalize();
  return status;
}
#endif // GPAW_INTERPRETER
