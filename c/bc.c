// Copyright (C) 2003  CAMP
// Please see the accompanying LICENSE file for further information.

#include <string.h>
#include <assert.h>
#include "bc.h"
#include "extensions.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef GPAW_OMP
 #include <omp.h>
#endif

//By defining GPAW_ASYNC you will use non-blocking mpi calls
//and not the blocking mpi calls dictated by the GPAW_AIX
#ifdef GPAW_ASYNC
#undef GPAW_AIX
#define GPAW_REAIX
#endif

boundary_conditions* bc_init(const long size1[3],
           const long padding[3][2],
           const long npadding[3][2],
           const long neighbors[3][2],
           MPI_Comm comm, bool real, bool cfd)
{
  boundary_conditions* bc = GPAW_MALLOC(boundary_conditions, 1);

  for (int i = 0; i < 3; i++)
    {
      bc->size1[i] = size1[i];
      bc->size2[i] = size1[i] + padding[i][0] + padding[i][1];
      bc->padding[i] = padding[i][0];
    }

  bc->comm = comm;
  bc->ndouble = (real ? 1 : 2);

  int rank = 0;
  if (comm != MPI_COMM_NULL)
    MPI_Comm_rank(comm, &rank);

  int start[3];
  int size[3];
  for (int i = 0; i < 3; i++)
    {
      start[i] = padding[i][0];
      size[i] = size1[i];
    }

  for (int i = 0; i < 3; i++)
    {
      int n = bc->ndouble;
      for (int j = 0; j < 3; j++)
      if (j != i)
        n *= size[j];

      for (int d = 0; d < 2; d++)
        {
          int ds = npadding[i][d];
          int dr = padding[i][d];
          for (int j = 0; j < 3; j++)
            {
              bc->sendstart[i][d][j] = start[j];
              bc->sendsize[i][d][j] = size[j];
              bc->recvstart[i][d][j] = start[j];
              bc->recvsize[i][d][j] = size[j];
            }
          if (d == 0)
            {
              bc->sendstart[i][d][i] = dr;
              bc->recvstart[i][d][i] = 0;
            }
          else
            {
              bc->sendstart[i][d][i] = padding[i][0] + size1[i] - ds;
              bc->recvstart[i][d][i] = padding[i][0] + size1[i];
            }
          bc->sendsize[i][d][i] = ds;
          bc->recvsize[i][d][i] = dr;

          bc->sendproc[i][d] = DO_NOTHING;
          bc->recvproc[i][d] = DO_NOTHING;
          bc->nsend[i][d] = 0;
          bc->nrecv[i][d] = 0;

          int p = neighbors[i][d];
          if (p == rank)
            {
              if (ds > 0)
                bc->sendproc[i][d] = COPY_DATA;
              if (dr > 0)
                bc->recvproc[i][d] = COPY_DATA;
            }
          else if (p >= 0)
            {
              // Communication required:
              if (ds > 0)
                {
                  bc->sendproc[i][d] = p;
                  bc->nsend[i][d] = n * ds;
                }
                    if (dr > 0)
                {
                  bc->recvproc[i][d] = p;
                  bc->nrecv[i][d] = n * dr;
                }
            }
        }

      if (cfd == 0)
        {
          start[i] = 0;
          size[i] = bc->size2[i];
        }
      // If the two neighboring processors along the
      // i'th axis are the same, then we join the two communications
      // into one:
      bc->rjoin[i] = ((bc->recvproc[i][0] == bc->recvproc[i][1]) &&
          bc->recvproc[i][0] >= 0);
      bc->sjoin[i] = ((bc->sendproc[i][0] == bc->sendproc[i][1]) &&
          bc->sendproc[i][0] >= 0);
    }

  bc->maxsend = 0;
  bc->maxrecv = 0;
  for (int i = 0; i < 3; i++)
    {
      int n = bc->nsend[i][0] + bc->nsend[i][1];
      if (n > bc->maxsend)
        bc->maxsend = n;
      n = bc->nrecv[i][0] + bc->nrecv[i][1];
      if (n > bc->maxrecv)
        bc->maxrecv = n;
    }

  return bc;
}


void bc_unpack1(const boundary_conditions* bc,
                const double* a1, double* a2, int i,
                MPI_Request recvreq[2],
                MPI_Request sendreq[2],
                double* rbuf, double* sbuf,
                const double_complex phases[2])
{
  bool real = (bc->ndouble == 1);
#ifdef PARALLEL

  #ifdef GPAW_OMP
    int thd = omp_get_thread_num();
  #else
    int thd = 0;
  #endif


  // Start receiving.
  for (int d = 0; d < 2; d++)
    {
      int p = bc->recvproc[i][d];
      if (p >= 0)
        {
          if (bc->rjoin[i])
            {
              if (d == 0)
                {
                  int count = bc->nrecv[i][0] + bc->nrecv[i][1];
                  MPI_Irecv(rbuf, count, MPI_DOUBLE, p, 10 * thd + 1000 * i + 100000,
                            bc->comm, &recvreq[0]);
                }
            }
          else
            {
              int count = bc->nrecv[i][d];
              MPI_Irecv(rbuf, count, MPI_DOUBLE, p, d + 10 * thd + 1000 * i,
                        bc->comm, &recvreq[d]);
              rbuf += bc->nrecv[i][d];
            }
        }
    }
  // Prepare send-buffers and start sending:
  double* sbuf0 = sbuf;
  for (int d = 0; d < 2; d++)
    {
      sendreq[d] = 0;
      int p = bc->sendproc[i][d];
      if (p >= 0)
        {
          const double* a;
          const int* start = bc->sendstart[i][d];
          const int* sizea;
          const int* size = bc->sendsize[i][d];
          if (i == 0)
            {
              const int* p = bc->padding;
              int start0[3] = {start[0] - p[0],
                   start[1] - p[1],
                   start[2] - p[2]};
              a = a1;
              start = start0;
              sizea = bc->size1;
            }
          else
            {
              a = a2;
              sizea = bc->size2;
            }

          if (real)
            bmgs_cut(a, sizea, start, sbuf, size);
          else
            bmgs_cutmz((const double_complex*)a, sizea, start,
                      (double_complex*)sbuf, size, phases[d]);
          if (bc->sjoin[i])
            {
              if (d == 1)
                {
                  int count = bc->nsend[i][0] + bc->nsend[i][1];
            #ifdef GPAW_AIX
                  MPI_Send(sbuf0, count, MPI_DOUBLE, p, 10 * thd + 1000 * i + 100000,
                           bc->comm);
            #else
                  MPI_Isend(sbuf0, count, MPI_DOUBLE, p, 10 * thd + 1000 * i + 100000,
                            bc->comm, &sendreq[0]);
            #endif
                }
            }
          else
            {
              int count = bc->nsend[i][d];
            #ifdef GPAW_AIX
              MPI_Send(sbuf, count, MPI_DOUBLE, p, 1 - d + 10 * thd + 1000 * i,
                       bc->comm);
            #else
              MPI_Isend(sbuf, count, MPI_DOUBLE, p, 1 - d + 10 * thd + 1000 * i,
                        bc->comm, &sendreq[d]);
            #endif
            }
          sbuf += bc->nsend[i][d];
        }
    }
#endif // Parallel
  // Copy data:
  if (i == 0)
    {
      memset(a2, 0, (bc->size2[0] * bc->size2[1] * bc->size2[2] *
         bc->ndouble * sizeof(double))); // XXXXXXXXXXXXXXXXX

      /*      assert(bc->sendstart[0][0][0] == bc->padding);
      assert(bc->sendstart[0][0][1] == bc->padding);
      assert(bc->sendstart[0][0][2] == bc->padding);*/

      if (real)
        bmgs_paste(a1, bc->size1, a2, bc->size2, bc->sendstart[0][0]);
      else
        bmgs_pastez((const double_complex*)a1, bc->size1,
                  (double_complex*)a2, bc->size2, bc->sendstart[0][0]);
    }

  // Copy data for periodic boundary conditions:
  for (int d = 0; d < 2; d++)
    if (bc->sendproc[i][d] == COPY_DATA)
      {
        if (real)
          bmgs_translate(a2, bc->size2, bc->sendsize[i][d],
             bc->sendstart[i][d], bc->recvstart[i][1 - d]);
        else
          bmgs_translatemz((double_complex*)a2, bc->size2,
               bc->sendsize[i][d],
               bc->sendstart[i][d], bc->recvstart[i][1 - d],
                   phases[d]);
      }
}


void bc_unpack2(const boundary_conditions* bc,
    double* a2, int i,
    MPI_Request recvreq[2],
    MPI_Request sendreq[2],
    double* rbuf)
{
#ifdef PARALLEL
  // Store data from receive-buffer:
  bool real = (bc->ndouble == 1);
  double* rbuf0 = rbuf;
  for (int d = 0; d < 2; d++)
    if (bc->recvproc[i][d] >= 0)
      {
  if (bc->rjoin[i])
    {
      if (d == 0)
        {
          MPI_Wait(&recvreq[0], MPI_STATUS_IGNORE);
          rbuf += bc->nrecv[i][1];
        }
      else
        rbuf = rbuf0;
    }
  else
    MPI_Wait(&recvreq[d], MPI_STATUS_IGNORE);

  if (real)
    bmgs_paste(rbuf, bc->recvsize[i][d],
         a2, bc->size2, bc->recvstart[i][d]);
  else
    bmgs_pastez((const double_complex*)rbuf, bc->recvsize[i][d],
          (double_complex*)a2, bc->size2, bc->recvstart[i][d]);
  rbuf += bc->nrecv[i][d];
      }
#ifndef GPAW_AIX
  // This does not work on the ibm with gcc!  We do a blocking send instead.
  for (int d = 0; d < 2; d++)
    if (sendreq[d] != 0)
      MPI_Wait(&sendreq[d], MPI_STATUS_IGNORE);
#endif
#endif // PARALLEL
}

//Remember to redefine GPAW_AIX
#ifdef GPAW_REAIX
#define GPAW_AIX
#endif

