#include <Python.h>
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <malloc.h>

#ifndef DOUBLECOMPLEXDEFINED
#  define DOUBLECOMPLEXDEFINED 1
#  ifdef NO_C99_COMPLEX
     typedef struct { double r, i; } double_complex;
#  else
#    include <complex.h>
     typedef double complex double_complex;
#  endif
#endif

#if PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION < 4
#  define Py_RETURN_NONE return Py_INCREF(Py_None), Py_None
#endif

#ifndef NO_C99_COMPLEX
#define INLINE inline
#else
#define INLINE 
#endif

static INLINE void* gpaw_malloc(int n)
{
  void* p = malloc(n);
  assert(p != NULL);
  return p;
}

#ifdef GPAW_AIX
#define GPAW_MALLOC(T, n) ((T*)malloc((n) * sizeof(T)))
#else
#define GPAW_MALLOC(T, n) ((T*)gpaw_malloc((n) * sizeof(T)))
#endif
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define LONGP(a) ((long*)((a)->data))
#define DOUBLEP(a) ((double*)((a)->data))
#define COMPLEXP(a) ((double_complex*)((a)->data))
