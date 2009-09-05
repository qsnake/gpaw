#include <Python.h>
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "extensions.h"

#ifdef GPAW_AIX
#  define dgels_ dgels
#endif

// Predefine dgels function of lapack
int dgels_(char* trans, int *m, int *n, int *nrhs, double* a, int *lda, double* b, int *ldb, double* work, int* lwork, int *info);


// Perform a moving linear least squares interpolation to arrays
// Input arguments:
// order: order of polynomial used (1 or 2)
// cutoff: the cutoff of weight (in grid points)
// coords: scaled coords [0,1] for interpolation
// N_c: number of grid points
// beg_c: first grid point
// data: the array used
// target: the results are stored in this array
PyObject* mlsqr(PyObject *self, PyObject *args)
{
  // The order of interpolation
  unsigned char order = -1;

  // The cutoff for moving least squares
  double cutoff = -1;

  // The coordinates for interpolation: array of size (3, N)
  PyArrayObject* coords = 0;

  // Number of grid points
  PyArrayObject* N_c = 0;

  // Beginning of grid
  PyArrayObject* beg_c = 0;

  // The 3d-data to be interpolated: array of size (X, Y, Z)
  PyArrayObject* data;

  // The interpolation target: array of size (N,)
  PyArrayObject* target = 0;

  if (!PyArg_ParseTuple(args, "BdOOOOO", &order, &cutoff, &coords, &N_c, &beg_c, &data, &target))
    {
      return NULL;
    }

  //printf( "Performing moving linear least squares interpolation\n" );
  //printf( "Using polynomial order %d\n", order);
  //printf( "Source data size %d %d %d\n", data->dimensions[0], data->dimensions[1], data->dimensions[2]);

  int coeffs = -1;

  if (order == 1)
    {
      coeffs = 4;
    }
  if (order == 2)
    {
      coeffs = 10;
      // 1 x y z xy yz zx xx yy zz
    }

  int points = coords->dimensions[0];
  //printf( "Interpolating %d points\n", points);

  double* coord_nc = DOUBLEP(coords);
  double* grid_points = DOUBLEP(N_c);
  double* grid_start = DOUBLEP(beg_c);
  double* target_n = DOUBLEP(target);
  double* data_g = DOUBLEP(data);


  // TODO: Calculate fit
  const int sizex = ceil(cutoff)+1;
  const int sizey = ceil(cutoff)+1;
  const int sizez = ceil(cutoff)+1;

  // Allocate X-matrix and b-vector
  int source_points = (2*sizex+1)*(2*sizey+1)*(2*sizez+1);
  double* X = GPAW_MALLOC(double, coeffs*source_points);
  double* b = GPAW_MALLOC(double, source_points);
  double* work = GPAW_MALLOC(double, coeffs*source_points);

  // The multipliers for each dimension
  int ldx = data->dimensions[1]*data->dimensions[2];
  int ldy = data->dimensions[2];
  int ldz = 1;

  // For each point to be interpolated
  for (int p=0; p< points; p++)
    {
      double x = (*coord_nc++)*grid_points[0] - grid_start[0];
      double y = (*coord_nc++)*grid_points[1] - grid_start[0];
      double z = (*coord_nc++)*grid_points[2] - grid_start[0];

      // The grid center point
      int cx2 = round(x);
      int cy2 = round(y);
      int cz2 = round(z);

      // Scaled to grid
      int cx = cx2 % data->dimensions[0];
      int cy = cy2 % data->dimensions[1];
      int cz = cz2 % data->dimensions[2];

      double* i_X = X;
      double* i_b = b;
      //printf("at point %d", p);
      // For each point to take into account
      for (int dx=-sizex;dx<=sizex;dx++)
	for (int dy=-sizey;dy<=sizey;dy++)
	  for (int dz=-sizez;dz<=sizez;dz++)
	    {
	      // Normalized distance from center
	      double d = sqrt(dx*dx+dy*dy+dz*dz) / cutoff;
	      double w = 0.0;
	      if (d < 1)
	      {
	         w = (1-d)*(1-d);
	         w*=w;
	         w*=(4*d+1);
	      }
	      //double w = exp(-d*d);

	      // Coordinates centered on x,y,z
	      double sx = (cx2 + dx) - x;
	      double sy = (cy2 + dy) - y;
	      double sz = (cz2 + dz) - z;

	      *i_X++ = w*1.0;
	      *i_X++ = w*sx;
	      *i_X++ = w*sy;
	      *i_X++ = w*sz;

	      if (order == 2)
		{
		  *i_X++ = w*sx*sy;
		  *i_X++ = w*sy*sz;
		  *i_X++ = w*sz*sx;
		  *i_X++ = w*sx*sx;
		  *i_X++ = w*sy*sy;
		  *i_X++ = w*sz*sz;
		}
	      *i_b++ = w*data_g[ (cx+dx) % data->dimensions[0] * ldx +
				 (cy+dy) % data->dimensions[1] * ldy +
				 (cz+dz) % data->dimensions[2] * ldz ];
	    }

      int info = 0;
      int rhs = 1;
      int worksize = coeffs*source_points;
      int ldb = source_points;
      dgels_("T",
	    &coeffs,              // ...times 4.
	    &source_points,  // lhs is of size sourcepoints...
	    &rhs,            // one rhs.
	    X,               // provide lhs
	    &coeffs,         // Leading dimension of X
	    b,               // provide rhs
	    &ldb,            // Leading dimension of b
	    work,            // work array (and output)
	    &worksize,       // the size of work array
	    &info);          // info
      if (info != 0)
	printf("WARNING: dgels returned %d!", info);

      // Evaluate the polynomial
      // Due to centered coordinates, it's just the constant term
      double value = b[0];

      *target_n++ = value;

      //Nearest neighbour
      //double value = data_g[ cx*data->dimensions[1]*data->dimensions[2] + cy*data->dimensions[2] + cz ];
      //printf("%.5f" , value);
    }

  free(work);
  free(b);
  free(X);
  Py_RETURN_NONE;
}
