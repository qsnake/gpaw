#include <Python.h>
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "extensions.h"

/* pseudo electron localisation function (ELF) as defined in
   Becke and Edgecombe, J. Chem. Phys., vol 92 (1990) 5397 */
PyObject* elf(PyObject *self, PyObject *args)
{
  PyArrayObject* n;   /* density */
  PyArrayObject* gn2; /* gradient of the density squared */
  PyArrayObject* tau; /* kinetic energy density */
  double ncut;        /* min. density cutoff */
  PyArrayObject* elf; 
  if (!PyArg_ParseTuple(args, "OOOdO", &n, &gn2, &tau, &ncut, &elf)) 
    return NULL; 

  /* get number of values in the array */
  int ng = 1;
  for (int i = 0; i<n->nd; i++) { ng *= n->dimensions[i]; }

  double* n_g = DOUBLEP(n);
  double* gn2_g = DOUBLEP(gn2);
  double* tau_g = DOUBLEP(tau);
  double* elf_g = DOUBLEP(elf);
  for (int g = 0; g < ng; g++) {
    if( n_g[g] > ncut ) {
      /* uniform electron gas value of D, Becke eq. (13) */
      /* 3./5.*(6*pi)**(2./3.) = 4.2496386096005443 */
      double D0 = 4.2496386096005443 * pow(n_g[g],5./3.);
      /* note: the definition of tau in Becke misses the factor 1/2 */
      double D = 2.*tau_g[g] - .25*gn2_g[g]/n_g[g];
      double chi = D/D0;
      elf_g[g] = 1./(1.+chi*chi);
    } else {
      elf_g[g] = 0;
    }
  }
  Py_RETURN_NONE;
}
