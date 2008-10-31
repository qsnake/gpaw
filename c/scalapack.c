// BLACS
#ifdef GPAW_MKL
#define   Cblacs_barrier_  Cblacs_barrier
#define   Cblacs_exit_     Cblacs_exit
#define   Cblacs_get_      Cblacs_get
#define   Cblacs_gridexit_ Cblacs_gridexit
#define   Cblacs_gridinfo_ Cblacs_gridinfo
#define   Cblacs_gridinit_ Cblacs_gridinit
#define   Cblacs_pinfo_    Cblacs_pinfo
#define   Cblacs_pnum_     Cblacs_pnum
#define   Cblacs_setup_    Cblacs_setup
#endif

#ifdef GPAW_AIX
#define   dgebr2d_  dgebr2d
#define   dgebs2d_  dgebs2d
#define   zgebr2d_  zgebr2d
#define   zgebs2d_  zgebs2d
#endif

void Cblacs_barrier_(int ConTxt, char *scope);

void Cblacs_exit_(int NotDone);

void Cblacs_get_(int ConTxt, int what, int* val);

void Cblacs_gridexit_(int ConTxt);

void Cblacs_gridinfo_(int ConTxt, int *nprow, int *npcol,
              int *myrow, int *mycol);

void Cblacs_gridinit_(int *ConTxt, char* order, int nprow, int npcol);

void Cblacs_pinfo_(int *mypnum, int *nprocs);

int Cblacs_pnum_(int ConTxt, int prow, int pcol);

void Cblacs_setup_(int *mypnum, int *nprocs);

void dgebr2d_(int *ConTxt, char* scope, char* top, int *m, int *n,
              double *A, int *lda, int *rsrc, int *csrc);

void dgebs2d_(int *ConTxt, char* scope, char* top, int *m, int *n,
              double *A, int *lda);

void zgebr2d_(int *ConTxt, char* scope, char* top, int *m, int *n,
              double *A, int *lda, int *rsrc, int *csrc);

void zgebs2d_(int *ConTxt, char* scope, char* top, int *m, int *n,
              double *A, int *lda);
// End of BLACS

// ScaLapack

#ifdef GPAW_AIX
#define   descinit_  descinit
#define   numroc_  numroc
#define   pdelset_  pdelset
#define   pzelset_  pzelset
#define   pdgemr2d_  pdgemr2d
#define   pdlamch_  pdlamch

#define   pdpotrf_  pdpotrf
#define   pdpotri_  pdpotri
#define   pzpotrf_  pzpotrf
#define   pzpotri_  pzpotri
#define   pdtrtri_  pdtrtri
#define   pztrtri_  pztrtri

#define   pdsyevd_  pdsyevd
#define   pdsyev_  pdsyev
#define   pdsyevx_  pdsyevx
#define   pdsygvx_  pdsygvx
#define   pzheev_  pzheev
#define   sl_init_  sl_init
#endif

#ifdef GPAW_MKL
#define   Cpdgemr2d_  Cpdgemr2d
#endif

void descinit_(int* desc, int* m, int* n, int* mb, int* nb, int* irsrc,
               int* icsrc, int* ictxt,
               int* lld, int* info);

int numroc_(int* n, int* nb, int* iproc, int* isrcproc, int* nprocs);

void pdelset_(double* a,int* ia,int* ja,int* desca,double* alpha);

//void pdelset2_(double* alpha,double* a,int* ia,int* ja,int* desca,double* beta);

void pzelset_(void* a,int* ia,int* ja,int* desca,void* alpha);

int pdgemr2d_(int* m, int*n, double* a, int* ia, int* ja, int* desca,
              double* b, int* ib, int* jb, int* descb, int* ctxt);

double pdlamch_(int* ictxt, char* cmach);

void Cpdgemr2d_(int m, int n, double *A, int IA, int JA, int *descA,
               double *B, int IB, int JB, int *descB, int gcontext);

// cholesky
void pdpotrf_(char *uplo, int *n, double* a, int *ia, int* ja, int* desca,
              int *info);
void pdpotri_(char *uplo, int *n, double* a, int *ia, int* ja, int* desca,
              int *info);

void pzpotrf_(char *uplo, int *n, void* a, int *ia, int* ja, int* desca,
              int *info);
void pzpotri_(char *uplo, int *n, void* a, int *ia, int* ja, int* desca,
              int *info);


void pdtrtri_(char *uplo, char *diag, int *n, double* a, int *ia, int* ja,
              int* desca, int *info);
void pztrtri_(char *uplo, char *diag, int *n, void* a, int *ia, int* ja,
              int* desca, int *info);


void pdsyevd_(char *jobz, char *uplo, int *n, double* a, int *ia, int* ja,
              int* desca, double *w,
              double* z, int *iz, int* jz, int* descz, double *work,
              int *lwork, int *iwork, int *liwork, int *info);

void pdsyev_(char *jobz, char *uplo, int *n, double* a, int *ia, int* ja,
             int* desca, double *w,
             double* z, int *iz, int* jz, int* descz, double *work,
             int *lwork, int *info);

void pdsyevx_(char *jobz, char *range, char *uplo, int *n, double* a,
              int *ia, int* ja, int* desca, double* vl,
              double* vu, int* il, int* iu, double* abstol, int* m, int* nz,
              double* w, double* orfac, double* z, int *iz,
              int* jz, int* descz, double *work, int *lwork, int *iwork,
              int *liwork, int *ifail, int *iclustr, double* gap, int *info);

void pdsygvx_(int *ibtype, char *jobz, char *range, char *uplo, int *n,
              double* a, int *ia, int* ja,
              int* desca, double* b, int *ib, int* jb, int* descb,
              double* vl, double* vu, int* il, int* iu,
              double* abstol, int* m, int* nz, double* w, double* orfac,
              double* z, int *iz, int* jz, int* descz,
              double *work, int *lwork, int *iwork, int *liwork, int *ifail,
              int *iclustr, double* gap, int *info);

void pzheev_(char *jobz, char *uplo, int *n, double* a, int *ia, int* ja,
             int* desca, double *w, double* z, int *iz, int* jz,
             int* descz, double *work, int *lwork, double *rwork,
             int *lrwork, int *info);

void sl_init_(int* ictxt, int* nprow, int* npcol);
// End of ScaLapack


static PyObject* diagonalize(MPIObject *self, PyObject *args)
{
  static int minusone = -1;
  static int zero = 0;
  static int one = 1;

  PyArrayObject* a; // symmetric/hermitian matrix
  PyArrayObject* w; // eigenvalues of `a` in ascending order
  PyArrayObject* b = 0; // symmetric/hermitian positive definite: a*v=b*v*w
  // the number of rows in the process grid
  // over which the matrix is distributed
  int nprow = 1;
  // the number of columns in the process grid
  // over which the matrix is distributed
  int npcol = 1;
  // the size of the blocks the distributed matrix is split into
  // (applies to both rows and columns)
  int mb = 32;
  int root = -1;
  root = 0;
  if (!PyArg_ParseTuple(args, "OO|iiiiO", &a, &w,
                        &nprow, &npcol, &mb, &root, &b))
    return NULL;
  //printf("mb %d\n", mb);
  int nb = mb;
  int n = a->dimensions[0]; // dimension of a
  int m = n;
  int lda = n;
  int ldb = n;
  int ibtype = 1;
  int info = 0;
  int iam = -1;

  int ConTxt;
  // initialize the grid

  //int nprocs = nprow*npcol;
  // Get starting information
  //printf("iam %d, nprocs %d\n", iam, nprocs);
  //Cblacs_pinfo_(&iam, &nprocs);
  //printf("iam %d, nprocs %d\n", iam, nprocs);

  //Cblacs_get_(minusone, zero, &ConTxt);
  //printf("nprow %d, npcol %d\n", nprow, npcol);
  //char lolo = 'R'; // All grid
  //Cblacs_gridinit_(&ConTxt, &lolo, nprow, npcol);

  sl_init_(&ConTxt, &nprow, &npcol);
  // get information back about the grid
  int myrow = -1;
  int mycol = -1;
  Cblacs_gridinfo_(ConTxt, &nprow, &npcol, &myrow, &mycol);

  //int rank;
  int pnum;
  //MPI_Comm_rank(self->comm, &rank);

  char TOP = ' '; // ?
  char scope = 'A'; // All grid

  int rsrc = 0;
  int csrc = 0;

  //printf("outside root: %4d, rank: %4d, pnum: %d, nprow: %d, npcol: %d, myrow: %4d, mycol: %4d, ConTxt: %d\n", root, rank, pnum, nprow, npcol, myrow, mycol, ConTxt);
  if (myrow != -1 && mycol != -1)
    {
      // Returns the system process number of the process in the process grid
      pnum = Cblacs_pnum_(ConTxt, myrow, mycol);
      //printf("root: %4d, rank: %4d, pnum: %d, nprow: %d, npcol: %d, myrow: %4d, mycol: %4d, ConTxt: %d\n", root, rank, pnum, nprow, npcol, myrow, mycol, ConTxt);

      // build the descriptor
      int desc0[9];
      descinit_(desc0, &m, &n, &m, &n, &rsrc, &csrc, &ConTxt, &m, &info);
      //for(int i1 = one; i1 <= 9; ++i1) {
      //     desc0[i1] = 0;
      //}
      //desc0[2] = -1;
      // distribute the full matrix to the process grid
      if (pnum == root)
        {
          //for(int i1 = one; i1 < m+one; ++i1) {
          //for(int i2 = one; i2 < n+one; ++i2) {
          //DOUBLEP(a)[(i1-one)*n+i2-one] = 10.0*i1 + i2;
          //printf("a(%d, %d) = %f\n",i1, i2, DOUBLEP(a)[(i1-one)*n+i2-one]);
          //}
          //}
          if (b == 0)
            {
                 //printf("C diagonalize ScaLapack\n");
            }
          else
            {
                 //printf("C diagonalize general ScaLapack\n");
              dgebs2d_(&ConTxt,&scope,&TOP,&m,&n,DOUBLEP(b),&lda);
            }
          // http://icl.cs.utk.edu/lapack-forum/viewtopic.php?p=87&
          // distribute the matrix
          // Uncomment for PDSYEV:
          //printf("run %d\n", 1);
          dgebs2d_(&ConTxt,&scope,&TOP,&m,&n,DOUBLEP(a),&lda);
          //printf("run %d\n", 2);
          // Uncomment for PZHEEV:
          //zgebs2d_(&ConTxt,&scope,&TOP,&m,&n,DOUBLEP(a),&lda);
        }
      else
        {
          if (b != 0)
            dgebr2d_(&ConTxt,&scope,&TOP,&m,&n,DOUBLEP(b),&lda,&rsrc,&csrc);
          // receive the matrix
          // Uncomment for PDSYEV:
          //printf("run %d\n", 3);
          dgebr2d_(&ConTxt,&scope,&TOP,&m,&n,DOUBLEP(a),&lda,&rsrc,&csrc);
          //printf("run %d\n", 4);
          // Uncomment for PZHEEV:
          //zgebr2d_(&ConTxt,&scope,&TOP,&m,&n,DOUBLEP(a),&lda,&rsrc,&csrc);
        }

      int desc[9];

      // get the size of the distributed matrix
      int locM = numroc_(&m, &mb, &myrow, &rsrc, &nprow);
      int locN = numroc_(&n, &nb, &mycol, &csrc, &npcol);
      // allocate the distributed matrices
      double* mata = GPAW_MALLOC(double, locM*locN);
      double* matb;
      if (b != 0)
        matb = GPAW_MALLOC(double, locM*locN);
      // allocate the distributed matrix of eigenvectors
      double* z = GPAW_MALLOC(double, locM*locN);
      //          for(int i1 = zero; i1 < locM; ++i1) {
      //               for(int i2 = zero; i2 < locN; ++i2) {
      ////          for(int i1 = zero; i1 < m; ++i1) {
      ////               for(int i2 = zero; i2 < n; ++i2) {
      //                    mata[i1*locM+i2] = 0.0;
      //               }
      //          }

      int lld = MAX(1,locM);

      // build the descriptor
      descinit_(desc, &m, &n, &mb, &nb, &rsrc, &csrc, &ConTxt, &lld, &info);

      //for(int i1 = one; i1 < m+one; ++i1) {
      //printf("w(%d) = %f\n",i1, DOUBLEP(w)[i1]);
      //     for(int i2 = one; i2 < n+one; ++i2) {
      //printf("pnum %d, before a(%d, %d) = %f\n",pnum, i1, i2, DOUBLEP(a)[(i1-one)*n+i2-one]);
      //     }
      //}

      //printf("run %d\n", 5);
      // build the distributed matrices
      for(int i1 = one; i1 < m+one; ++i1)
        for(int i2 = one; i2 < n+one; ++i2)
          {
            // Uncomment for PDSYEV:
            // http://icl.cs.utk.edu/lapack-forum/viewtopic.php?t=321&sid=f2ac0a3c06c66d74a2fbd65c222ffdb0
            pdelset_(mata,&i1,&i2,desc,&DOUBLEP(a)[(i1-one)*n+i2-one]);
            // Uncomment for PZHEEV:
            //pzelset_(mata,&i1,&i2,desc,a(i1-one,i2-one));
          }
      if (b != 0)
        for(int i1 = one; i1 < m+one; ++i1)
          for(int i2 = one; i2 < n+one; ++i2)
            {
              // Uncomment for PDSYEV:
              // http://icl.cs.utk.edu/lapack-forum/viewtopic.php?t=321&sid=f2ac0a3c06c66d74a2fbd65c222ffdb0
              pdelset_(matb,&i1,&i2,desc,&DOUBLEP(b)[(i1-one)*n+i2-one]);
              // Uncomment for PZHEEV:
              //pzelset_(matb,&i1,&i2,desc,b(i1-one,i2-one));
            }

      //Cblacs_barrier_(ConTxt, &scope);

      //printf("run %d\n", 6);
      char jobz = 'V'; // eigenvectors also
      char range = 'A'; // all eiganvalues
      char uplo = 'U'; // work with upper

      double vl, vu;
      int il, iu;

      char cmach = 'U';

      double abstol = 2.0 * pdlamch_(&ConTxt, &cmach);

      int eigvalm, nz;

      double orfac = -1.0;
      //double orfac = 0.001;

      int* ifail;
      ifail = GPAW_MALLOC(int, m);

      int* iclustr;
      iclustr = GPAW_MALLOC(int, 2*nprow*npcol);

      double* gap;
      gap = GPAW_MALLOC(double, nprow*npcol);


      double* work;
      work = GPAW_MALLOC(double, 3);
      int querylwork = -1;
      int* iwork;
      iwork = GPAW_MALLOC(int, 1);
      int queryliwork = 1;
      if (b != 0) queryliwork = -1;
      //queryliwork = -1;

      //printf("run %d\n", 7);
      if (b == 0)
        {
          //pdsyevx_(&jobz, &range, &uplo, &n, mata, &one, &one, desc, &vl,
          //         &vu, &il, &iu, &abstol, &eigvalm, &nz, DOUBLEP(w), &orfac, z, &one,
          //         &one, desc, work, &querylwork, iwork, &queryliwork, ifail, iclustr, gap, &info);
          pdsyevd_(&jobz, &uplo, &n, mata, &one, &one, desc, DOUBLEP(w),
                   z, &one, &one, desc,
                   work, &querylwork, iwork, &queryliwork, &info);
          //pdsyev_(&jobz, &uplo, &m, mata, &one, &one, desc, DOUBLEP(w),
          //        z, &one, &one, desc, work, &querylwork, &info);
        }
      else
        pdsygvx_(&ibtype, &jobz, &range, &uplo, &n, mata, &one, &one,
                 desc, matb, &one, &one, desc, &vl, &vu, &il, &iu,
                 &abstol, &eigvalm, &nz, DOUBLEP(w), &orfac, z, &one, &one, desc,
                 work, &querylwork, iwork, &queryliwork, ifail, iclustr, gap, &info);

      int lwork = (int)work[0];
      if (b != 0)
        lwork = lwork + (n-1)*n;
      //printf("lwork %d\n", lwork);
      free(work);
      int liwork = (int)iwork[0];
      //printf("liwork %d\n", liwork);
      free(iwork);

      //printf("run %d\n", 8);
      work = GPAW_MALLOC(double, lwork);
      iwork = GPAW_MALLOC(int, liwork);
      //printf("run %d\n", 9);
      //          printf("myrow: %4d, mycol: %4d, lwork: %d\n", myrow, mycol, lwork);

      double t0, t1, time = 0.0;

      /* Timer accuracy test */

      t0 = MPI_Wtime();
      t1 = MPI_Wtime();

      while (t1 == t0)
        t1 = MPI_Wtime();

      //          if (pnum == root)
      //               printf("Timer accuracy of ~%f micro secs\n\n", (t1 - t0) * 1e6);

      int runs = 1;

      t0 = MPI_Wtime();
      if (b == 0)
        {
          //pdsyevx_(&jobz, &range, &uplo, &n, mata, &one, &one, desc, &vl,
          //         &vu, &il, &iu, &abstol, &eigvalm, &nz, DOUBLEP(w), &orfac, z, &one,
          //         &one, desc, work, &lwork, iwork, &liwork, ifail, iclustr, gap, &info);
          pdsyevd_(&jobz, &uplo, &n, mata, &one, &one, desc, DOUBLEP(w),
                   z, &one, &one, desc,
                   work, &lwork, iwork, &liwork, &info);
          //pdsyev_(&jobz, &uplo, &m, mata, &one, &one, desc, DOUBLEP(w),
          //        z, &one, &one, desc, work, &lwork, &info);
        }
      else
        pdsygvx_(&ibtype, &jobz, &range, &uplo, &n, mata, &one, &one,
                 desc, matb, &one, &one, desc, &vl, &vu, &il, &iu,
                 &abstol, &eigvalm, &nz, DOUBLEP(w), &orfac, z, &one, &one, desc,
                 work, &lwork, iwork, &liwork, ifail, iclustr, gap, &info);
      t1 = MPI_Wtime();
      time = time + (t1 - t0);

      free(work);
      free(iwork);

      //for(int i1 = one; i1 < locM+one; ++i1) {
      //printf("w(%d) = %f\n",i1, DOUBLEP(w)[i1]);
      //for(int i2 = one; i2 < locN+one; ++i2) {
      //z[(i1-one)*locN+i2-one] = mata[(i1-one)*locN+i2-one];
      //printf("pnum %d, locM %d, locN %d, mata(%d, %d) = %f\n",pnum, locM, locN, i1, i2, mata[(i1-one)*locN+i2-one]);
      //}
      //}
      //if (pnum == 0) {
      //     printf("pnum %d, %f\n", pnum, z[(1-one)*locM+1-one]);
      //     z[(1-one)*locM+1-one] = 0.0;
      //}
      //if (pnum == 1) {
      //     printf("pnum %d, %f\n", pnum, z[(1-one)*locM+1-one]);
      //     z[(1-one)*locN+1-one] = 0.0;
      //}

      //printf("run %d\n", 10);
      //printf("info diagonalize, pnum %d, %d\n", info, pnum);

      //Cblacs_barrier_(ConTxt, &scope);

      // pdgemr2d_(&n, &n, z, &one, &one, desc, DOUBLEP(a), &one, &one, desc0, &ConTxt);
      Cpdgemr2d_(m, n, z, one, one, desc, DOUBLEP(a), one, one, desc0, ConTxt);

      //for(int i1 = one; i1 < m+one; ++i1) {
      //printf("w(%d) = %f\n",i1, DOUBLEP(w)[i1]);
      //for(int i2 = one; i2 < n+one; ++i2) {
      //printf("pnum %d, after a(%d, %d) = %f\n",pnum, i1, i2, DOUBLEP(a)[(i1-one)*n+i2-one]);
      //}
      //}

      //printf("after Cpdgemr2d, pnum %d\n", pnum);
      //Cblacs_barrier_(ConTxt, &scope);

      //          if (pnum == root) {
      //               for(int i1 = zero; i1 < m; ++i1) {
      //                    printf("w(%d) = %f\n",i1, DOUBLEP(w)[i1]);
      //                    for(int i2 = zero; i2 < n; ++i2) {
      //printf("zz(%d, %d) = %f\n",i1, i2, zz[(i1+one)*(m-1)+(i2+one)]);
      //printf("a(%d, %d) = %f\n",i1, i2, DOUBLEP(a)[i1*m+i2]);
      //                         DOUBLEP(a)[i1*m+i2] = zz[(i1+one)*(m-1)+(i2+one)];
      //                    }
      //               }
      //          }

      //          printf("N: %d, nprow: %d, npcol: %d, MB: %d, NB: %d, myrow: %4d, mycol: %4d, Time: %f\n",
      //                 n, nprow, npcol, mb, nb, myrow, mycol,
      //                 time / runs);

      free(gap);
      free(iclustr);
      free(ifail);

      free(z);
      free(mata);
      if (b != 0)
        free(matb);
      //printf("run %d\n", 12);
      // clean up the grid
      Cblacs_gridexit_(ConTxt);
    }
  //Cblacs_exit_(zero);
  //printf("run %d\n", 13);
  //          if (a->descr->type_num == PyArray_DOUBLE)
  //          {
  /*           int lwork = 3 * n + 1; */
  /*           double* work = GPAW_MALLOC(double, lwork); */
  /*           if (b == 0) */
  /*                dsyev_("V", "U", &n, DOUBLEP(a), &lda, */
  /*                       DOUBLEP(w), work, &lwork, &info); */
  /*           free(work); */
  //          }
  //     }
  //printf("info %d, pnum %d\n", info, pnum);
  //Py_RETURN_NONE;
  return Py_BuildValue("i", info);
}

#include "sl_inverse_cholesky.c"
