#include "extensions.h"
#include <bmgs.h>

typedef struct 
{
  PyObject_HEAD
  bmgsspline spline;
} SplineObject;
