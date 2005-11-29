// Copyright (C) 2003  CAMP
// Please see the accompanying LICENSE file for further information.

#include <string.h>
#include <assert.h>
#include "bc.h"
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

boundary_conditions* bc_init(const long size1[3], const int padding[2], 
			     const long neighbors[3][2],
			     MPI_Comm comm, bool real, bool cfd)
{
  boundary_conditions* bc = 
    (boundary_conditions*)malloc(sizeof(boundary_conditions));

  for (int i = 0; i < 3; i++)
    {
      bc->size1[i] = size1[i];
      bc->size2[i] = size1[i] + padding[0] + padding[1]; 
      bc->zero[i] = (neighbors[i][0] < 0);
    }

  bc->comm = comm;
  bc->ndouble = (real ? 1 : 2);
  bc->padding = padding[0];

  int rank = 0;
  if (comm != MPI_COMM_NULL)
    MPI_Comm_rank(comm, &rank);

  int start[3];
  int size[3];
  for (int i = 0; i < 3; i++)
    {
      start[i] = padding[0];
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
	  int ds = padding[1 - d];
	  int dr = padding[d];
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
	      bc->sendstart[i][d][i] = size1[i];
	      bc->recvstart[i][d][i] = size1[i] + ds;
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
      bc->join[i] = ((bc->recvproc[i][0] == bc->recvproc[i][1]) && 
		     bc->recvproc[i][0] >= 0);
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

  bc->angle = 0.0;
  bc->rotbuf = 0;

  return bc;
}


void bc_set_rotation(boundary_conditions* bc,
		     double angle, double* coefs, long* offsets, int exact)
{
  bc->angle = angle;
  bc->rotcoefs = coefs;
  bc->rotoffsets = offsets;
  bc->exact = exact;
  int s0 = bc->sendsize[0][0][0];
  if (bc->sendsize[0][1][0] > s0)
    s0 = bc->sendsize[0][1][0];
  bc->rotbuf = (double*)malloc(s0 * bc->size1[1] * bc->size1[2] *
				   bc->ndouble * sizeof(double));
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
  // Start receiving.  
  for (int d = 0; d < 2; d++)
    {
      int p = bc->recvproc[i][d];
      if (p >= 0)
	{
	  if (bc->join[i])
	    {
	      if (d == 0)
		{
		  int count = bc->nrecv[i][0] + bc->nrecv[i][1];
		  MPI_Irecv(rbuf, count, MPI_DOUBLE, p, 20,
			    bc->comm, &recvreq[0]);
		}
	    }
	  else
	    {
	      int count = bc->nrecv[i][d];
	      MPI_Irecv(rbuf, count, MPI_DOUBLE, p, 10 + d, bc->comm,
			&recvreq[d]);
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
	      int p = bc->padding;
	      int start0[3] = {start[0] - p, start[1] - p, start[2] - p};
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

	  if (bc->join[i])
	    {
	      if (d == 1)
		{
		  int count = bc->nsend[i][0] + bc->nsend[i][1];
#ifdef GRIDPAW_AIX
		  MPI_Send(sbuf0, count, MPI_DOUBLE, p, 20, bc->comm);
#else
		  MPI_Isend(sbuf0, count, MPI_DOUBLE, p, 20,
			    bc->comm, &sendreq[0]);
#endif
		}
	    }
	  else
 	    {
	      int count = bc->nsend[i][d];
#ifdef GRIDPAW_AIX
	      MPI_Send(sbuf, count, MPI_DOUBLE, p, 11 - d, bc->comm);
#else
	      MPI_Isend(sbuf, count, MPI_DOUBLE, p, 11 - d,
			bc->comm, &sendreq[d]);
#endif
	    }
          sbuf += bc->nsend[i][d];
	}
    }
#endif // PARALLEL
  
  // Copy data:
  if (i == 0)
    {
      memset(a2, 0, (bc->size2[0] * bc->size2[1] * bc->size2[2] *
		     bc->ndouble * sizeof(double))); // XXXXXXXXXXXXXXXXX
      
      assert(bc->sendstart[0][0][0] == bc->padding);
      assert(bc->sendstart[0][0][1] == bc->padding);
      assert(bc->sendstart[0][0][2] == bc->padding);
      
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
	if (bc->angle == 0.0 || i != 0)
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
	else
	  {
	    int p = bc->padding;
	    if (real)
	      {
		int nn = (bc->sendsize[i][d][0] * 
			  bc->sendsize[i][d][1] * 
			  bc->sendsize[i][d][2]);
		double* c = bc->rotbuf;
		for (int n = 0; n < nn; n++)
		  c[n] = 0.0;
		bmgs_rotate(a1 + (bc->sendstart[i][d][0] - p) *
			    bc->size1[1] * bc->size1[2], bc->sendsize[i][d],
			    bc->rotbuf, bc->angle * (2 * d - 1),
			    bc->rotcoefs, bc->rotoffsets, bc->exact);
		bmgs_paste(bc->rotbuf, bc->sendsize[i][d], 
			   a2, bc->size2, bc->recvstart[i][1 - d]);
	      }
	    else
	      {
		int nn = (bc->sendsize[i][d][0] * 
			  bc->sendsize[i][d][1] * 
			  bc->sendsize[i][d][2]);
		double_complex* c = (double_complex*)bc->rotbuf;
		for (int n = 0; n < nn; n++) {
#ifdef NO_C99_COMPLEX		  
		  c[n].r = 0.0;
		  c[n].i = 0.0;
#else
		  c[n] = 0.0;	  
#endif
		}
		bmgs_rotatez(((double_complex*)a1) + 
			     (bc->sendstart[i][d][0] - p) *
			     bc->size1[1] * bc->size1[2], 
			     bc->sendsize[i][d],
			     (double_complex*)bc->rotbuf, 
			     bc->angle * (2 * d - 1),
			     bc->rotcoefs, bc->rotoffsets, bc->exact);
		for (int n = 0; n < nn; n++) {
#ifdef NO_C99_COMPLEX		  
		  c[n].r = c[n].r*phases[d].r-c[n].i*phases[d].i;
		  c[n].i = c[n].r*phases[d].i+c[n].i*phases[d].r;
#else
		  c[n] *= phases[d];
#endif
		}
		bmgs_pastez((double_complex*)bc->rotbuf, bc->sendsize[i][d], 
			    (double_complex*)a2, bc->size2,
			    bc->recvstart[i][1 - d]);
	      }
	  }	  
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
	if (bc->join[i])
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
#ifndef GRIDPAW_AIX
  // This does not work on the ibm!  We do a blocking send instead.
  for (int d = 0; d < 2; d++)
    if (sendreq[d] != 0)
      MPI_Wait(&sendreq[d], MPI_STATUS_IGNORE);
#endif
#endif // PARALLEL
}
