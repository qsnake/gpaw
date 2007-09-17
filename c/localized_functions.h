#include <Python.h>

typedef struct 
{
  PyObject_HEAD
  double dv;
  int size[3];
  int start[3];
  int size0[3];
  int ng;
  int ng0;
  int nf;
  int nfd;
  double* f;
  double* fd;
  double* w;
} LocalizedFunctionsObject;

