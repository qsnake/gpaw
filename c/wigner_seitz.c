#include "extensions.h"
#include <stdlib.h>

// returns the squared distance between a 3d double vector
// and a 3d int vector
double distance3d2_di(double *a, int *b)
{
  double sum = 0;
  double diff;
  for (int c = 0; c < 3; c++) {
    diff = a[c] - (double)b[c];
    sum += diff*diff;
  }
  return sum;
} 

PyObject *wigner_seitz_grid(PyObject *self, PyObject *args)
{
  PyArrayObject* ai;
  PyArrayObject* aatom_c;
  PyArrayObject* beg_c;
  PyArrayObject* end_c;
  if (!PyArg_ParseTuple(args, "OOOO", &ai, &aatom_c, &beg_c, &end_c))
    return NULL;

  long *aindex = LONGP(ai);
  int natoms = aatom_c->dimensions[0];
  double *atom_c = DOUBLEP(aatom_c);
  long *beg = LONGP(beg_c);
  long *end = LONGP(end_c);

  int n[3], pos[3], ij;
  for (int c = 0; c < 3; c++) { n[c] = end[c] - beg[c]; }
  // go over all points and indicate the nearest atom
  for (int i = 0; i < n[0]; i++) {
    pos[0] = beg[0] + i;
    for (int j = 0; j < n[1]; j++) {
      pos[1] = beg[1] + j;
      ij = (i*n[1] + j)*n[2];
      for (int k = 0; k < n[2]; k++) {
	pos[2] = beg[2] + k;
	double dmin = 1.e99;
	for (int a=0; a < natoms; a++) {
	  double d = distance3d2_di(atom_c + a*3, pos);
/* 	  printf("apos=(%g,%g,%g)\n",(atom_c + a*3)[0], */
/* 		 (atom_c + a*3)[1],(atom_c + a*3)[2]); */
	  if (d < dmin) {
	    aindex[ij + k] = (long) a;
	    dmin = d;
	  }
	}
      }
    }
  }

  Py_RETURN_NONE;
}

