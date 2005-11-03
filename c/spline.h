#include "extensions.h"
#include "bmgs/bmgs.h"

typedef struct 
{
  PyObject_HEAD
  bmgsspline spline;
} SplineObject;
