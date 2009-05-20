#ifdef PARALLEL
#include <Python.h>
#ifdef GPAW_WITH_SL
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <mpi.h>
#include "extensions.h"
#include <structmember.h>
#include "mympi.h"

// BLACS
#define BLOCK_CYCLIC_2D 1
#ifdef GPAW_MKL
#define   Cblacs_gridexit_ Cblacs_gridexit
#define   Cblacs_gridinfo_ Cblacs_gridinfo
#define   Cblacs_gridinit_ Cblacs_gridinit
#define   Cblacs_pinfo_    Cblacs_pinfo
#define   Csys2blacs_handle_ Csys2blacs_handle
#endif

void Cblacs_gridexit_(int ConTxt);

void Cblacs_gridinfo_(int ConTxt, int* nprow, int* npcol,
                      int* myrow, int* mycol);

void Cblacs_gridinit_(int* ConTxt, char* order, int nprow, int npcol);

void Cblacs_pinfo_(int* mypnum, int* nprocs);

int Csys2blacs_handle_(MPI_Comm SysCtxt);
// End of BLACS

// ScaLAPACK
#ifdef GPAW_AIX
#define   descinit_   descinit
#define   numroc_     numroc
#define   pdlamch_    pdlamch

#define   pdpotrf_  pdpotrf
#define   pzpotrf_  pzpotrf
#define   pdtrtri_  pdtrtri
#define   pztrtri_  pztrtri

#define   pdsyevd_  pdsyevd
#define   pdsygvx_  pdsygvx
#endif

#ifdef GPAW_MKL
#define   Cpdgemr2d_   Cpdgemr2d
#define   Cpdgemr2do_  Cpdgemr2do
#endif

// tools
void descinit_(int* desc, int* m, int* n, int* mb, int* nb, int* irsrc,
               int* icsrc, int* ictxt,
               int* lld, int* info);

int numroc_(int* n, int* nb, int* iproc, int* isrcproc, int* nprocs);

void Cpdgemr2d_(int m, int n, double* a, int ia, int ja, int* desca,
               double* b, int ib, int jb, int* descb, int gcontext);

void Cpdgemr2do_(int m, int n, double* a, int ia, int ja, int* desca,
               double* b, int ib, int jb, int* descb);

double pdlamch_(int* ictxt, char* cmach);

// cholesky
void pdpotrf_(char* uplo, int* n, double* a, int* ia, int* ja, int* desca,
              int* info);

void pdtrtri_(char* uplo, char* diag, int* n, double* a, int *ia, int* ja,
              int* desca, int* info);

// diagonalization
void pdsyevd_(char* jobz, char* uplo, int* n, double* a, int* ia, int* ja,
              int* desca, double* w,
              double* z, int* iz, int* jz, int* descz, double* work,
              int* lwork, int* iwork, int* liwork, int* info);

void pdsygvx_(int* ibtype, char* jobz, char* range, char* uplo, int* n,
              double* a, int* ia, int* ja,
              int* desca, double* b, int *ib, int* jb, int* descb,
              double* vl, double* vu, int* il, int* iu,
              double* abstol, int* m, int* nz, double* w, double* orfac,
              double* z, int* iz, int* jz, int* descz,
              double* work, int* lwork, int* iwork, int* liwork, int* ifail,
              int* iclustr, double* gap, int* info);


PyObject* blacs_create(PyObject *self, PyObject *args)
{
  PyObject*  comm_obj;     // communicator
  char order='R';
  int row_order = 1;
  int m, n, nprow, npcol, mb, nb, lld;
  int nprocs;
  int ConTxt = -1;
  int iam = 0;
  int rsrc = 0;
  int csrc = 0;
  int myrow = -1;
  int mycol = -1;
  int desc[9];

  npy_intp desc_dims[1] = {9};
  PyArrayObject* desc_obj = (PyArrayObject*)PyArray_SimpleNew(1, desc_dims, NPY_INT);

  if (!PyArg_ParseTuple(args, "Oiiiiii|i", &comm_obj, &m, &n, &nprow, &npcol, &mb, &nb, &row_order))
    return NULL;

  // False, gives us C-column order. Not clear when this would be beneficial, but we
  // do not hard-code R-row order for the sake of generality.
  if (!row_order) order = 'C';


  if (comm_obj == Py_None)
    {
      // SPECIAL CASE: Rank is not part of this communicator
      // ScaLAPACK documentation here is vague. It was emperically determined that
      // the values of desc[1]-desc[5] are important for use with pdgemr2d routines.
      // (otherwise, ScaLAPACK core dumps). PBLAS requires desc[0] == 1 | 2, even
      // for an inactive context.
      desc[0] = BLOCK_CYCLIC_2D;
      desc[1] = -1; // Tells BLACS to ignore me.
      desc[2] = m;
      desc[3] = n;
      desc[4] = mb;
      desc[5] = nb;
      desc[6] = 0;
      desc[7] = 0;
      desc[8] = 0;
    }
  else
    {
      // Create blacs grid on this communicator
      MPI_Comm comm = ((MPIObject*)comm_obj)->comm;

      // Get my id and nprocs. This is for debugging purposes only
      Cblacs_pinfo_(&iam, &nprocs);
      MPI_Comm_size(comm, &nprocs);
      // printf("iam=%d,nprocs=%d\n",iam,nprocs);

      // Create blacs grid on this communicator continued
      ConTxt = Csys2blacs_handle(comm);
      Cblacs_gridinit_(&ConTxt, &order, nprow, npcol);
      // printf("ConTxt=%d,nprow=%d,npcol=%d\n",ConTxt,nprow,npcol);
      Cblacs_gridinfo_(ConTxt, &nprow, &npcol, &myrow, &mycol);

      lld = numroc_(&m, &mb, &myrow, &rsrc, &nprow);
      desc[0] = BLOCK_CYCLIC_2D;
      desc[1] = ConTxt;
      desc[2] = m;
      desc[3] = n;
      desc[4] = mb;
      desc[5] = nb;
      desc[6] = 0;
      desc[7] = 0;
      desc[8] = MAX(1, lld);
    }
  memcpy(desc_obj->data, desc, 9*sizeof(int));

  return (PyObject*)desc_obj;
}

PyObject* blacs_destroy(PyObject *self, PyObject *args)
{
    PyArrayObject* adesc; //blacs descriptor

    if (!PyArg_ParseTuple(args, "O", &adesc))
      return NULL;

    int a_ConTxt = INTP(adesc)[1];

    if (a_ConTxt != -1) Cblacs_gridexit_(a_ConTxt);

    Py_RETURN_NONE;
}

PyObject* scalapack_redist(PyObject *self, PyObject *args)
{
    PyArrayObject* a_obj; //source matrix
    PyArrayObject* adesc; //source descriptor
    PyArrayObject* bdesc; //destination descriptor
    PyArrayObject* cdesc = 0; //intermediate descriptor, must encompass adesc + bdesc
    int m = 0;
    int n = 0;
    static int one = 1;

    if (!PyArg_ParseTuple(args, "OOO|Oii", &a_obj, &adesc, &bdesc, &cdesc, &m, &n))
      return NULL;

    // adesc
    int a_mycol = -1;
    int a_myrow = -1;
    int a_nprow, a_npcol; 
    int a_ConTxt = INTP(adesc)[1];
    int a_m = INTP(adesc)[2];
    int a_n = INTP(adesc)[3];
    int a_mb = INTP(adesc)[4];
    int a_nb = INTP(adesc)[5];
    int a_rsrc = INTP(adesc)[6];
    int a_csrc = INTP(adesc)[7];

    // If m and n not specified, redistribute all rows and columns of a.
    if ((m == 0) | (n == 0))
      {
        m = a_m;
        n = a_n;	  
      }

    // bdesc
    int b_mycol = -1;
    int b_myrow = -1;
    int b_nprow, b_npcol; 
    int b_ConTxt = INTP(bdesc)[1];
    int b_m = INTP(bdesc)[2];
    int b_n = INTP(bdesc)[3];
    int b_mb = INTP(bdesc)[4];
    int b_nb = INTP(bdesc)[5];
    int b_rsrc = INTP(bdesc)[6];
    int b_csrc = INTP(bdesc)[7];

    // Get adesc and bdesc grid info
    Cblacs_gridinfo_(a_ConTxt, &a_nprow, &a_npcol,&a_myrow, &a_mycol);
    
    Cblacs_gridinfo_(b_ConTxt, &b_nprow, &b_npcol,&b_myrow, &b_mycol);

    // It appears that the memory requirements for Cpdgemr2do are non-trivial.
    // Consideer A_loc, B_loc to be the local piece of the global array. Then
    // to perform this operation you will need an extra A_loc, B_loc worth of
    // memory. Hence, for --state-parallelization=4 the memory savings are
    // about nbands-by-nbands is exactly 1/4*(nbands-by-nbands). For --state-
    // parallelization=8 it is about 3/4*(nbands-by-nbands). For --state-
    // parallelization=B it is about (B-2)/B*(nbands-by-nbands).

    int b_locM = numroc_(&b_m, &b_mb, &b_myrow, &b_rsrc, &b_nprow);
    int b_locN = numroc_(&b_n, &b_nb, &b_mycol, &b_csrc, &b_npcol);
    
    if ((b_locM < 0) | (b_locN < 0))
      { 
          b_locM = 0;
          b_locN = 0;
      }

    // Make Fortran contiguous array, ScaLAPACK requires Fortran order arrays!
    // Note there are some times when you can get away with C order arrays.                                          
    // Most notable example is a symmetric matrix stored on a square ConTxt.
    // PyArray_UpdateFlags(b_obj,NPY_F_CONTIGUOUS);
    npy_intp b_dims[2] = {b_locM, b_locN};
    PyArrayObject* b_obj = (PyArrayObject*)PyArray_EMPTY(2, b_dims, NPY_DOUBLE,
                                                            NPY_F_CONTIGUOUS);

    // This should work for redistributing a_obj unto b_obj regardless of whether
    // the ConTxt are overlapping. Cpdgemr2do is undocumented but can be understood
    // by looking at the scalapack-1.8.0/REDIST/SRC/pdgemr.c. Cpdgemr2do creates
    // another ConTxt which encompasses MPI_COMM_WORLD. It is used as an intermediary
    // for copying between a_ConTxt and b_ConTxt. It then calls Cpdgemr2d which
    // performs the actual redistribution.
    if (cdesc != 0)
      {
        int c_ConTxt = INTP(cdesc)[1];
        if (c_ConTxt != -1)
          {
            Cpdgemr2d_(m, n, DOUBLEP(a_obj), one, one, INTP(adesc), DOUBLEP(b_obj), 
                                               one, one, INTP(bdesc), c_ConTxt);
          }
      }
    else
      {
        Cpdgemr2do_(m, n, DOUBLEP(a_obj), one, one, INTP(adesc), DOUBLEP(b_obj), 
                                                        one, one, INTP(bdesc));
      }

    // Note that we choose to return Py_None, instead of an empty array.
    if ((b_locM == 0) | (b_locN == 0))
      {
        Py_DECREF(b_obj);
        Py_RETURN_NONE;
      }

    PyObject* value = Py_BuildValue("O",b_obj);
    Py_DECREF(b_obj);
    return value;
}

PyObject* scalapack_diagonalize_dc(PyObject *self, PyObject *args)
{
    // Standard Driver for Divide and Conquer algorithm
    // Computes all eigenvalues and eigenvectors
 
    PyArrayObject* a_obj; // symmetric matrix
    PyArrayObject* adesc; // symmetric matrix description vector
    int z_mycol = -1;
    int z_myrow = -1;
    int z_nprow, z_npcol;
    int z_type, z_ConTxt, z_m, z_n, z_mb, z_nb, z_rsrc, z_csrc, z_lld;
    int zdesc[9];
    static int one = 1;

    char jobz = 'V'; // eigenvectors also
    char uplo = 'U'; // work with upper

    if (!PyArg_ParseTuple(args, "OO", &a_obj, &adesc))
      return NULL;

    // adesc
    int a_type   = INTP(adesc)[0];
    int a_ConTxt = INTP(adesc)[1];
    int a_m      = INTP(adesc)[2];
    int a_n      = INTP(adesc)[3];
    int a_mb     = INTP(adesc)[4];
    int a_nb     = INTP(adesc)[5];
    int a_rsrc   = INTP(adesc)[6];
    int a_csrc   = INTP(adesc)[7];
    int a_lld    = INTP(adesc)[8];

    // Note that A is symmetric, so n = a_m = a_n;
    // We do not test for that here.
    int n = a_n;

    // zdesc = adesc
    // This is generally not required, as long as the 
    // alignment properties are satisfied, see pdsyevd.f
    // In the context of GPAW, don't see why zdesc would
    // not be equal to adesc so I am just hard-coding it in.
    z_type   = a_type;
    z_ConTxt = a_ConTxt;
    z_m      = a_m;
    z_n      = a_n;
    z_mb     = a_mb;
    z_nb     = a_nb;
    z_rsrc   = a_rsrc;
    z_csrc   = a_csrc;
    z_lld    = a_lld;
    zdesc[0] = z_type;
    zdesc[1] = z_ConTxt;
    zdesc[2] = z_m;
    zdesc[3] = z_n;
    zdesc[4] = z_mb;
    zdesc[5] = z_nb;
    zdesc[6] = z_rsrc;
    zdesc[7] = z_csrc;
    zdesc[8] = z_lld;

    Cblacs_gridinfo_(z_ConTxt, &z_nprow, &z_npcol,&z_myrow, &z_mycol);

    if (z_ConTxt != -1)
      {
        // z_locM, z_locN should not be negative or zero
        int z_locM = numroc_(&z_m, &z_mb, &z_myrow, &z_rsrc, &z_nprow);
        int z_locN = numroc_(&z_n, &z_nb, &z_mycol, &z_csrc, &z_npcol);
    
        // Eigenvectors
        npy_intp z_dims[2] = {z_locM, z_locN};
        PyArrayObject* z_obj = (PyArrayObject*)PyArray_EMPTY(2, z_dims, NPY_DOUBLE, 
                                                                NPY_F_CONTIGUOUS);

        // Eigenvalues, since w_obj is 1D-array NPY_F_CONTIGUOUS not needed here
        npy_intp w_dims[1] = {n};
        PyArrayObject* w_obj = (PyArrayObject*)PyArray_SimpleNew(1, w_dims, NPY_DOUBLE);

        // Query part, need to find the optimal size of a number of work arrays
	int info;
        double* work;
        work = GPAW_MALLOC(double, 1);
        int querylwork = -1;
        int* iwork;
        iwork = GPAW_MALLOC(int, 1);
        int queryliwork = -1;
        pdsyevd_(&jobz, &uplo, &n, DOUBLEP(a_obj), &one, &one, INTP(adesc), 
                 DOUBLEP(w_obj), DOUBLEP(z_obj), &one, &one, zdesc, 
                 work, &querylwork, iwork, &queryliwork, &info);
        // printf("query info = %d\n", info);
	// Computation part
        int lwork = (int)(work[0]+0.1); // Give extra space to avoid complaint from PDORMTR
        free(work);
        int liwork = (int)iwork[0];
        free(iwork);
        work = GPAW_MALLOC(double, lwork);
        iwork = GPAW_MALLOC(int, liwork);
        pdsyevd_(&jobz, &uplo, &n, DOUBLEP(a_obj), &one, &one, INTP(adesc), 
                 DOUBLEP(w_obj), DOUBLEP(z_obj), &one, &one, zdesc, 
                 work, &lwork, iwork, &liwork, &info);
        // printf("computation info = %d\n", info);
        free(work);
	free(iwork);
        PyObject* values = Py_BuildValue("(OO)", w_obj, z_obj);
        Py_DECREF(w_obj);
        Py_DECREF(z_obj);
        return values;
      }
    else
      {
	return Py_BuildValue("(OO)", Py_None, Py_None);
      }
}

PyObject* scalapack_general_diagonalize(PyObject *self, PyObject *args)
{
    // Expert Driver for QR algorithm
    // Computes *all* eigenvalues and eigenvectors
 
    PyArrayObject* a_obj; // Hamiltonian matrix
    PyArrayObject* b_obj; // overlap matrix
    PyArrayObject* adesc; // Hamintonian matrix descriptor
    int ibtype  =  1; // Solve H*psi = lambda*S*psi
    int z_mycol = -1;
    int z_myrow = -1;
    int z_nprow, z_npcol;
    int z_type, z_ConTxt, z_m, z_n, z_mb, z_nb, z_rsrc, z_csrc, z_lld;
    int zdesc[9];
    int il, iu;  // not used when range = 'A' or 'V'
    int eigvalm, nz;
    static int one = 1;
 
    double vl, vu; // not used when range = 'A' or 'I'

    char jobz = 'V'; // eigenvectors also
    char range = 'A'; // all eigenvalues
    char uplo = 'U'; // work with upper
    char cmach = 'U'; // most orthogonal eigenvectors    
    // char cmach = 'S'; // most acccurate eigenvalues

    if (!PyArg_ParseTuple(args, "OOO", &a_obj, &b_obj, &adesc))
      return NULL;

    // adesc,
    // bdesc = adesc
    // This is generally not required, as long as the
    // alignment properties are satisfied, see pdsygvx.f
    // In the context of GPAW, don't see why bdesc would
    // not be equal to adesc so I am just hard-coding it in.
    int a_type   = INTP(adesc)[0];
    int a_ConTxt = INTP(adesc)[1];
    int a_m      = INTP(adesc)[2];
    int a_n      = INTP(adesc)[3];
    int a_mb     = INTP(adesc)[4];
    int a_nb     = INTP(adesc)[5];
    int a_rsrc   = INTP(adesc)[6];
    int a_csrc   = INTP(adesc)[7];
    int a_lld    = INTP(adesc)[8];

    // Note that A is symmetric, so n = a_m = a_n;
    // We do not test for that here.
    int n = a_n;

    // zdesc = adesc
    // This is generally not required, as long as the 
    // alignment properties are satisfied, see pdsygvx.f
    // In the context of GPAW, don't see why zdesc would
    // not be equal to adesc so I am just hard-coding it in.
    z_type   = a_type;
    z_ConTxt = a_ConTxt;
    z_m      = a_m;
    z_n      = a_n;
    z_mb     = a_mb;
    z_nb     = a_nb;
    z_rsrc   = a_rsrc;
    z_csrc   = a_csrc;
    z_lld    = a_lld;
    zdesc[0] = z_type;
    zdesc[1] = z_ConTxt;
    zdesc[2] = z_m;
    zdesc[3] = z_n;
    zdesc[4] = z_mb;
    zdesc[5] = z_nb;
    zdesc[6] = z_rsrc;
    zdesc[7] = z_csrc;
    zdesc[8] = z_lld;

    // bdesc = adesc

    Cblacs_gridinfo_(z_ConTxt, &z_nprow, &z_npcol, &z_myrow, &z_mycol);

    if (z_ConTxt != -1)
      {
        // Convergence tolerance                                                         
        double abstol = pdlamch_(&z_ConTxt, &cmach); // most orthogonal eigenvectors     
        // double abstol = 2.0*pdlamch_(&z_ConTxt, &cmach); // most accurate eigenvalues            
        double orfac = -1.0;

        // z_locM, z_locN should not be negative or zero
        int z_locM = numroc_(&z_m, &z_mb, &z_myrow, &z_rsrc, &z_nprow);
        int z_locN = numroc_(&z_n, &z_nb, &z_mycol, &z_csrc, &z_npcol);

        // Eigenvectors
        npy_intp z_dims[2] = {z_locM, z_locN};
        PyArrayObject* z_obj = (PyArrayObject*)PyArray_EMPTY(2, z_dims, NPY_DOUBLE,
                                                                NPY_F_CONTIGUOUS);

        // Eigenvalues, since w_obj is 1D-array NPY_F_CONTIGUOUS not needed here
        npy_intp w_dims[1] = {n};
        PyArrayObject* w_obj = (PyArrayObject*)PyArray_SimpleNew(1, w_dims, NPY_DOUBLE);

        // Query part, need to find the optimal size of a number of work arrays
        int info;
        int *ifail;
        ifail = GPAW_MALLOC(int, n);
        int *iclustr;
        iclustr = GPAW_MALLOC(int, 2*z_nprow*z_npcol);
        double  *gap;
        gap = GPAW_MALLOC(double, z_nprow*z_npcol);
        double* work;
        work = GPAW_MALLOC(double, 3);
        int querylwork = -1;
        int* iwork;
        iwork = GPAW_MALLOC(int, 1);
        int queryliwork = -1;
        pdsygvx_(&ibtype, &jobz, &range, &uplo, &n, DOUBLEP(a_obj), &one, &one, 
                 INTP(adesc), DOUBLEP(b_obj), &one, &one, INTP(adesc), &vl, &vu, 
                 &il, &iu, &abstol, &eigvalm, &nz, DOUBLEP(w_obj), &orfac,
                 DOUBLEP(z_obj), &one, &one, zdesc, work, &querylwork, iwork, 
                 &queryliwork, ifail, iclustr, gap, &info);
        // printf("query info = %d\n", info);
	// Computation part
        int lwork = (int)work[0];
        lwork = lwork + (n-1)*n;
        free(work);
        int liwork = (int)iwork[0];
        free(iwork);
        work  = GPAW_MALLOC(double, lwork);
        iwork = GPAW_MALLOC(int, liwork);
        pdsygvx_(&ibtype, &jobz, &range, &uplo, &n, DOUBLEP(a_obj), &one, &one, 
                 INTP(adesc), DOUBLEP(b_obj), &one, &one, INTP(adesc), &vl, &vu, 
                 &il, &iu, &abstol, &eigvalm, &nz, DOUBLEP(w_obj), &orfac,
                 DOUBLEP(z_obj), &one, &one, zdesc, work, &lwork, iwork, 
                 &liwork, ifail, iclustr, gap, &info);
        // printf("computation info = %d\n", info);                
        free(work);
        free(iwork);
        free(gap);
        free(iclustr);
        free(ifail);

        PyObject* values = Py_BuildValue("(OO)", w_obj, z_obj);
        Py_DECREF(w_obj);
        Py_DECREF(z_obj);
        return values;
      }
    else
      {
	return Py_BuildValue("(OO)", Py_None, Py_None);
      }
}

PyObject* scalapack_inverse_cholesky(PyObject *self, PyObject *args)
{
    // Cholesky plus inverse of triangular matrix
 
    PyArrayObject* a_obj; // overlap matrix
    PyArrayObject* adesc; // symmetric matrix description vector
    int a_mycol = -1;
    int a_myrow = -1;
    int a_nprow, a_npcol;
    int info;
    static int one = 1;

    char diag = 'N'; // non-unit triangular
    char uplo = 'U'; // work with upper

    if (!PyArg_ParseTuple(args, "OO", &a_obj, &adesc))
      return NULL;

    // adesc
    int a_ConTxt = INTP(adesc)[1];
    int a_m      = INTP(adesc)[2];
    int a_n      = INTP(adesc)[3];

    // Note that A is symmetric, so n = a_m = a_n;
    // We do not test for that here.
    int n = a_n;

    Cblacs_gridinfo_(a_ConTxt, &a_nprow, &a_npcol,&a_myrow, &a_mycol);

    if (a_ConTxt != -1)
      { 
          pdpotrf_(&uplo, &n, DOUBLEP(a_obj), &one, &one, INTP(adesc), &info);

          pdtrtri_(&uplo, &diag, &n, DOUBLEP(a_obj), &one, &one, INTP(adesc), &info);
      }
    Py_RETURN_NONE;
}

#endif
#endif // PARALLEL

