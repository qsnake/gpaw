#include "extensions.h"
#include <stdlib.h>

double distance(double *a, double *b);

PyObject *pc_potential(PyObject *self, PyObject *args)
{
  PyArrayObject* poti;
  PyArrayObject* pc_ci;
  PyArrayObject* beg_c;
  PyArrayObject* end_c;
  PyArrayObject* hh_c;
  PyArrayObject* qi;
  if (!PyArg_ParseTuple(args, "OOOOOO", &poti, &pc_ci, &qi,
			&beg_c, &end_c, &hh_c))
    return NULL;

  double *pot = DOUBLEP(poti);
  int npc = pc_ci->dimensions[0];
  double *pc_c = DOUBLEP(pc_ci);
  long *beg = LONGP(beg_c);
  long *end = LONGP(end_c);
  double *h_c = DOUBLEP(hh_c);
  double *q = DOUBLEP(qi);

  int n[3], ij;
  double pos[3];
  for (int c = 0; c < 3; c++) { n[c] = end[c] - beg[c]; }
  // loop over all points
  for (int i = 0; i < n[0]; i++) {
    pos[0] = (beg[0] + i) * h_c[0];
    for (int j = 0; j < n[1]; j++) {
      pos[1] = (beg[1] + j) * h_c[1];
      ij = (i*n[1] + j)*n[2];
      for (int k = 0; k < n[2]; k++) {
	pos[2] = (beg[2] + k) * h_c[2];
	// loop over all atoms
	double V = 0.0;
	for (int a=0; a < npc; a++) {
	  double d = distance(pc_c + a*3, pos);
	  V -= q[a] / d;
	}
	pot[ij + k] = V;
      }
    }
  }

  Py_RETURN_NONE;
}

