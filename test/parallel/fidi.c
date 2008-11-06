#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <mpi.h>

// Compile with:
//
//   mpicc -std=c99 -O2 fidi.c
//
// or similar.  Run like this:
//
//   mpirun -np 8 ./a.out 2 2 2
//
// This will do a calculation with 100 arrays of size 96x96x96 grid
// points distributed over 2x2x2 processors, and the calculation is
// repeated 10 times.
//
// The finite diference formula used is:
//
//            2 
//            __
//           \                    1       6
//   f''(x) = )  c    f(x + n h) --- + O(h ),
//           /__  |n|              2
//                                h
//           n=-2
//
// where
//
//        2          4        1
//   c  = -,  c  = - -,  c  = --.
//    0   5    1     3    2   12
//

void fd(const double* a, double* b, double* w,
	double h,
	int Nx, int Ny, int Nz
	, int rank, 
	int rxm, int rxp, int rym, int ryp, int rzm, int rzp);

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  assert(argc == 4);
  // Domain decomposition:
  int Px = atoi(argv[1]);
  int Py = atoi(argv[2]);
  int Pz = atoi(argv[3]);
  assert(Px * Py * Pz == size);

  // Size of global arrays, number of arrays, and number of repeats:
  int N = 96;
  int M = 100;
  int R = 10;
  //N = 24; M = 3; R = 200;

  // Size of local arrays:
  int Nx = N / Px;
  int Ny = N / Py;
  int Nz = N / Pz;
  assert(Nx * Px == N);
  assert(Ny * Py == N);
  assert(Nz * Pz == N);

  // Find the position of this processor:
  int nx = rank / (Py * Pz);
  int ny = (rank - nx * Py * Pz) / Pz;
  int nz = rank % Pz;

  // Find the ranks of the 6 neighboring processors:
  int rxm = nz + Pz * (ny + Py * ((nx - 1 + Px) % Px));
  int rxp = nz + Pz * (ny + Py * ((nx + 1) % Px));
  int rym = nz + Pz * ((ny - 1 + Py) % Py + Py * nx);
  int ryp = nz + Pz * ((ny + 1) % Py + Py * nx);
  int rzm = (nz - 1 + Pz) % Pz + Pz * (ny + Py * nx);
  int rzp = (nz + 1) % Pz + Pz * (ny + Py * nx);
  
  int NNN = Nx * Ny * Nz;
  // Input and output arrays:
  double* a = (double*)malloc(M * NNN * sizeof(double));
  double* b = (double*)malloc(M * NNN * sizeof(double));
  
  // Initialize input arrays:
  int n = 0;
  for (int m = 0; m < M; m++)
    for (int x = nx * Nx; x < (nx + 1) * Nx; x++)
      for (int y = ny * Ny; y < (ny + 1) * Ny; y++)
	for (int z = nz * Nz; z < (nz + 1) * Nz; z++, n++)
	  a[n] = x + y * y + z * z * z;

  // Work array used by fd():
  int K = (Nx + 4) * (Ny + 4) * (Nz + 4);
  double* w = (double*)malloc(K * sizeof(double));

  double h = 0.1;  // grid spacing

  double t0 = clock();
  for (int r = 0; r < R; r++)
    for (int m = 0; m < M; m++)
      fd(a + m * NNN, b + m * NNN, w,
	 h, 
	 Nx, Ny, Nz,
	 rank, rxm, rxp, rym, ryp, rzm, rzp);
  double t =  clock() - t0;

  printf("Rank: %4d, Time: %f, GFLOPS: %f\n", rank, t / CLOCKS_PER_SEC,
	 19.0e-9 * R * M * NNN * CLOCKS_PER_SEC / t);

  double s = 0.0;
  for (int n = 0; n < NNN; n++)
    s += b[n];
  double sum;
  MPI_Reduce(&s, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0)
    printf("Sum:    %f\n", sum);

  MPI_Finalize();
  return 0;
}

void cp(const double* a, 
	 int ix, int iy, int iz, int jx, int jy, int jz,
	 int Ax, int Ay, int Az,
	 double* b, int kx, int ky, int kz, int Bx, int By, int Bz)
{
  // Copy array a[ix:jx, iy:jy, iz:jz] to array b[kx:kx+jx-ix,
  // ky:ky+jy-iy, kz:kz+jz-iz]. Arrays a and b have sizes Ax*Ay*Az and
  // Bx*By*Bz respectively.

  for (int x = ix; x < jx; x++)
    for (int y = iy; y < jy; y++)
      for (int z = iz; z < jz; z++)
	b[z - iz + kz + Bz * (y - iy + ky + By * (x - ix + kx))] =
	  a[z + Az * (y + Ay * x)];
}

void fd(const double* a, double* b, double* w, 
	double h,
	int Nx, int Ny, int Nz,
	int rank, 
	int rxm, int rxp, int rym, int ryp, int rzm, int rzp)
{
  // Apply finite difference operation to array a of size Nx*Ny*Nz and
  // place result in array b of size Nx*Ny*Nz.  Use periodic boundary
  // conditions.  rank is our rank, rxm is the rank of the processor
  // in the minus x-direction, rxp is the rank of the processor in the
  // plus x-direction, and so on ... . h is the grid spacing.

  int Kx = Nx + 4;
  int Ky = Ny + 4;
  int Kz = Nz + 4;

  MPI_Request reqm;
  MPI_Request reqp;
  
  if (rxm == rank)
    {
      cp(a, 0,      0, 0, 2,  Ny, Nz, Nx, Ny, Nz, w, Nx + 2, 2, 2, Kx, Ky, Kz);
      cp(a, Nx - 2, 0, 0, Nx, Ny, Nz, Nx, Ny, Nz, w, 0,      2, 2, Kx, Ky, Kz);
    }
  else
    {
      // Use output array for send and receive buffers:
      int NN = Ny * Nz;
      double* sendm = b;
      double* sendp = b + 2 * NN;
      double* recvm = b + 4 * NN;
      double* recvp = b + 6 * NN;
      assert(Nx >= 8);
      MPI_Irecv(recvm, 2 * NN, MPI_DOUBLE, rxm, 3, MPI_COMM_WORLD, &reqm);
      MPI_Irecv(recvp, 2 * NN, MPI_DOUBLE, rxp, 4, MPI_COMM_WORLD, &reqp);
      cp(a, 0,      0, 0, 2,  Ny, Nz, Nx, Ny, Nz, sendm, 0, 0, 0, 2, Ny, Nz);
      cp(a, Nx - 2, 0, 0, Nx, Ny, Nz, Nx, Ny, Nz, sendp, 0, 0, 0, 2, Ny, Nz);
      MPI_Send(sendm, 2 * NN, MPI_DOUBLE, rxm, 4, MPI_COMM_WORLD);
      MPI_Send(sendp, 2 * NN, MPI_DOUBLE, rxp, 3, MPI_COMM_WORLD);
      MPI_Wait(&reqm, MPI_STATUS_IGNORE);
      MPI_Wait(&reqp, MPI_STATUS_IGNORE);
      cp(recvm, 0, 0, 0, 2, Ny, Nz, 2, Ny, Nz, w, 0,      2, 2, Kx, Ky, Kz);
      cp(recvp, 0, 0, 0, 2, Ny, Nz, 2, Ny, Nz, w, Nx + 2, 2, 2, Kx, Ky, Kz);
    }

  if (rym == rank)
    {
      cp(a, 0, 0,     0,  Nx, 2,  Nz, Nx, Ny, Nz, w, 2, Ny + 2, 2, Kx, Ky, Kz);
      cp(a, 0, Ny - 2, 0, Nx, Ny, Nz, Nx, Ny, Nz, w, 2, 0,      2, Kx, Ky, Kz);
    }
  else
    {
      int NN = Nx * Nz;
      double* sendm = b;
      double* sendp = b + 2 * NN;
      double* recvm = b + 4 * NN;
      double* recvp = b + 6 * NN;
      assert(Ny >= 8);
      MPI_Irecv(recvm, 2 * NN, MPI_DOUBLE, rym, 5, MPI_COMM_WORLD, &reqm);
      MPI_Irecv(recvp, 2 * NN, MPI_DOUBLE, ryp, 6, MPI_COMM_WORLD, &reqp);
      cp(a, 0, 0,      0, Nx,  2, Nz, Nx, Ny, Nz, sendm, 0, 0, 0, Nx, 2, Nz);
      cp(a, 0, Ny - 2, 0, Nx, Ny, Nz, Nx, Ny, Nz, sendp, 0, 0, 0, Nx, 2, Nz);
      MPI_Send(sendm, 2 * NN, MPI_DOUBLE, rym, 6, MPI_COMM_WORLD);
      MPI_Send(sendp, 2 * NN, MPI_DOUBLE, ryp, 5, MPI_COMM_WORLD);
      MPI_Wait(&reqm, MPI_STATUS_IGNORE);
      MPI_Wait(&reqp, MPI_STATUS_IGNORE);
      cp(recvm, 0, 0, 0, Nx, 2, Nz, Nx, 2, Nz, w, 2, 0,      2, Kx, Ky, Kz);
      cp(recvp, 0, 0, 0, Nx, 2, Nz, Nx, 2, Nz, w, 2, Ny + 2, 2, Kx, Ky, Kz);
    }

  if (rzm == rank)
    {
      cp(a, 0, 0, 0,      Nx, Ny, 2,  Nx, Ny, Nz, w, 2, 2, Nz + 2, Kx, Ky, Kz);
      cp(a, 0, 0, Nz - 2, Nx, Ny, Nz, Nx, Ny, Nz, w, 2, 2, 0,      Kx, Ky, Kz);
    }
  else
    {
      int NN = Nx * Ny;
      double* sendm = b;
      double* sendp = b + 2 * NN;
      double* recvm = b + 4 * NN;
      double* recvp = b + 6 * NN;
      assert(Nz >= 8);
      MPI_Irecv(recvm, 2 * NN, MPI_DOUBLE, rzm, 7, MPI_COMM_WORLD, &reqm);
      MPI_Irecv(recvp, 2 * NN, MPI_DOUBLE, rzp, 8, MPI_COMM_WORLD, &reqp);
      cp(a, 0, 0, 0,      Nx, Ny, 2,  Nx, Ny, Nz, sendm, 0, 0, 0, Nx, Ny, 2);
      cp(a, 0, 0, Nz - 2, Nx, Ny, Nz, Nx, Ny, Nz, sendp, 0, 0, 0, Nx, Ny, 2);
      MPI_Send(sendm, 2 * NN, MPI_DOUBLE, rzm, 8, MPI_COMM_WORLD);
      MPI_Send(sendp, 2 * NN, MPI_DOUBLE, rzp, 7, MPI_COMM_WORLD);
      MPI_Wait(&reqm, MPI_STATUS_IGNORE);
      MPI_Wait(&reqp, MPI_STATUS_IGNORE);
      cp(recvm, 0, 0, 0, Nx, Ny, 2, Nx, Ny, 2, w, 2, 2, 0,      Kx, Ky, Kz);
      cp(recvp, 0, 0, 0, Nx, Ny, 2, Nx, Ny, 2, w, 2, 2, Nz + 2, Kx, Ky, Kz);
    }

  cp(a, 0, 0, 0, Nx, Ny, Nz, Nx, Ny, Nz, w, 2, 2, 2, Kx, Ky, Kz);

  double c0 = -7.5 / (h * h);
  double c1 = 4.0 / 3.0 / (h * h);
  double c2 = -1.0 / 12.0 / (h * h);
  int m = 0;
  for (int nx = 0; nx < Nx; nx++)
    for (int ny = 0; ny < Ny; ny++)
      for (int nz = 0; nz < Nz; nz++, m++)
	{
	  int n = nz + 2 + Kz * (ny + 2 + Ky * (nx + 2));
	  double d = c0 * w[n];
	  d += c1 * (w[n - 1] + w[n + 1]);
	  d += c1 * (w[n - Kz] + w[n + Kz]);
	  d += c1 * (w[n - Ky * Kz] + w[n + Ky * Kz]);
	  d += c2 * (w[n - 2] + w[n + 2]);
	  d += c2 * (w[n - 2 * Kz] + w[n + 2 * Kz]);
	  d += c2 * (w[n - 2 * Ky * Kz] + w[n + 2 * Ky * Kz]);
	  b[m] = d;
	}
}
