#ifndef H_EXTENSIONS
#define H_EXTENSIONS


#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <malloc.h>

/* If strict ANSI, then some useful macros are not defined */
#if defined(__STRICT_ANSI__)
# define M_PI           3.14159265358979323846  /* pi */
#endif

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

#ifdef GPAW_BGP
#define GPAW_MALLOC(T, n) (gpaw_malloc((n) * sizeof(T)))
#else
#ifdef GPAW_AIX
#define GPAW_MALLOC(T, n) (malloc((n) * sizeof(T)))
#else
#define GPAW_MALLOC(T, n) (gpaw_malloc((n) * sizeof(T)))
#endif
#endif
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define INTP(a) ((int*) ((a)->data))
#define LONGP(a) ((long*)((a)->data))
#define DOUBLEP(a) ((double*)((a)->data))
#define COMPLEXP(a) ((double_complex*)((a)->data))

#endif //H_EXTENSIONS
