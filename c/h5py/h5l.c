/* Generated by Cython 0.11.2 on Tue Jun 15 02:58:45 2010 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "structmember.h"
#ifndef Py_PYTHON_H
    #error Python headers needed to compile C extensions, please install development version of Python.
#endif
#ifndef PY_LONG_LONG
  #define PY_LONG_LONG LONG_LONG
#endif
#ifndef DL_EXPORT
  #define DL_EXPORT(t) t
#endif
#if PY_VERSION_HEX < 0x02040000
  #define METH_COEXIST 0
  #define PyDict_CheckExact(op) (Py_TYPE(op) == &PyDict_Type)
#endif
#if PY_VERSION_HEX < 0x02050000
  typedef int Py_ssize_t;
  #define PY_SSIZE_T_MAX INT_MAX
  #define PY_SSIZE_T_MIN INT_MIN
  #define PY_FORMAT_SIZE_T ""
  #define PyInt_FromSsize_t(z) PyInt_FromLong(z)
  #define PyInt_AsSsize_t(o)   PyInt_AsLong(o)
  #define PyNumber_Index(o)    PyNumber_Int(o)
  #define PyIndex_Check(o)     PyNumber_Check(o)
#endif
#if PY_VERSION_HEX < 0x02060000
  #define Py_REFCNT(ob) (((PyObject*)(ob))->ob_refcnt)
  #define Py_TYPE(ob)   (((PyObject*)(ob))->ob_type)
  #define Py_SIZE(ob)   (((PyVarObject*)(ob))->ob_size)
  #define PyVarObject_HEAD_INIT(type, size) \
          PyObject_HEAD_INIT(type) size,
  #define PyType_Modified(t)

  typedef struct {
       void *buf;
       PyObject *obj;
       Py_ssize_t len;
       Py_ssize_t itemsize;
       int readonly;
       int ndim;
       char *format;
       Py_ssize_t *shape;
       Py_ssize_t *strides;
       Py_ssize_t *suboffsets;
       void *internal;
  } Py_buffer;

  #define PyBUF_SIMPLE 0
  #define PyBUF_WRITABLE 0x0001
  #define PyBUF_FORMAT 0x0004
  #define PyBUF_ND 0x0008
  #define PyBUF_STRIDES (0x0010 | PyBUF_ND)
  #define PyBUF_C_CONTIGUOUS (0x0020 | PyBUF_STRIDES)
  #define PyBUF_F_CONTIGUOUS (0x0040 | PyBUF_STRIDES)
  #define PyBUF_ANY_CONTIGUOUS (0x0080 | PyBUF_STRIDES)
  #define PyBUF_INDIRECT (0x0100 | PyBUF_STRIDES)

#endif
#if PY_MAJOR_VERSION < 3
  #define __Pyx_BUILTIN_MODULE_NAME "__builtin__"
#else
  #define __Pyx_BUILTIN_MODULE_NAME "builtins"
#endif
#if PY_MAJOR_VERSION >= 3
  #define Py_TPFLAGS_CHECKTYPES 0
  #define Py_TPFLAGS_HAVE_INDEX 0
#endif
#if (PY_VERSION_HEX < 0x02060000) || (PY_MAJOR_VERSION >= 3)
  #define Py_TPFLAGS_HAVE_NEWBUFFER 0
#endif
#if PY_MAJOR_VERSION >= 3
  #define PyBaseString_Type            PyUnicode_Type
  #define PyString_Type                PyBytes_Type
  #define PyString_CheckExact          PyBytes_CheckExact
  #define PyInt_Type                   PyLong_Type
  #define PyInt_Check(op)              PyLong_Check(op)
  #define PyInt_CheckExact(op)         PyLong_CheckExact(op)
  #define PyInt_FromString             PyLong_FromString
  #define PyInt_FromUnicode            PyLong_FromUnicode
  #define PyInt_FromLong               PyLong_FromLong
  #define PyInt_FromSize_t             PyLong_FromSize_t
  #define PyInt_FromSsize_t            PyLong_FromSsize_t
  #define PyInt_AsLong                 PyLong_AsLong
  #define PyInt_AS_LONG                PyLong_AS_LONG
  #define PyInt_AsSsize_t              PyLong_AsSsize_t
  #define PyInt_AsUnsignedLongMask     PyLong_AsUnsignedLongMask
  #define PyInt_AsUnsignedLongLongMask PyLong_AsUnsignedLongLongMask
  #define __Pyx_PyNumber_Divide(x,y)         PyNumber_TrueDivide(x,y)
#else
  #define __Pyx_PyNumber_Divide(x,y)         PyNumber_Divide(x,y)
  #define PyBytes_Type                 PyString_Type
#endif
#if PY_MAJOR_VERSION >= 3
  #define PyMethod_New(func, self, klass) PyInstanceMethod_New(func)
#endif
#if !defined(WIN32) && !defined(MS_WINDOWS)
  #ifndef __stdcall
    #define __stdcall
  #endif
  #ifndef __cdecl
    #define __cdecl
  #endif
  #ifndef __fastcall
    #define __fastcall
  #endif
#else
  #define _USE_MATH_DEFINES
#endif
#if PY_VERSION_HEX < 0x02050000
  #define __Pyx_GetAttrString(o,n)   PyObject_GetAttrString((o),((char *)(n)))
  #define __Pyx_SetAttrString(o,n,a) PyObject_SetAttrString((o),((char *)(n)),(a))
  #define __Pyx_DelAttrString(o,n)   PyObject_DelAttrString((o),((char *)(n)))
#else
  #define __Pyx_GetAttrString(o,n)   PyObject_GetAttrString((o),(n))
  #define __Pyx_SetAttrString(o,n,a) PyObject_SetAttrString((o),(n),(a))
  #define __Pyx_DelAttrString(o,n)   PyObject_DelAttrString((o),(n))
#endif
#if PY_VERSION_HEX < 0x02050000
  #define __Pyx_NAMESTR(n) ((char *)(n))
  #define __Pyx_DOCSTR(n)  ((char *)(n))
#else
  #define __Pyx_NAMESTR(n) (n)
  #define __Pyx_DOCSTR(n)  (n)
#endif
#ifdef __cplusplus
#define __PYX_EXTERN_C extern "C"
#else
#define __PYX_EXTERN_C extern
#endif
#include <math.h>
#define __PYX_HAVE_API__h5py__h5l
#include "stdlib.h"
#include "string.h"
#include "time.h"
#include "unistd.h"
#include "stdint.h"
#include "compat.h"
#include "lzf_filter.h"
#include "hdf5.h"
#define __PYX_USE_C99_COMPLEX defined(_Complex_I)


#ifdef __GNUC__
#define INLINE __inline__
#elif _WIN32
#define INLINE __inline
#else
#define INLINE 
#endif

typedef struct {PyObject **p; char *s; long n; char is_unicode; char intern; char is_identifier;} __Pyx_StringTabEntry; /*proto*/



static int __pyx_skip_dispatch = 0;


/* Type Conversion Predeclarations */

#if PY_MAJOR_VERSION < 3
#define __Pyx_PyBytes_FromString          PyString_FromString
#define __Pyx_PyBytes_FromStringAndSize   PyString_FromStringAndSize
#define __Pyx_PyBytes_AsString            PyString_AsString
#else
#define __Pyx_PyBytes_FromString          PyBytes_FromString
#define __Pyx_PyBytes_FromStringAndSize   PyBytes_FromStringAndSize
#define __Pyx_PyBytes_AsString            PyBytes_AsString
#endif

#define __Pyx_PyBool_FromLong(b) ((b) ? (Py_INCREF(Py_True), Py_True) : (Py_INCREF(Py_False), Py_False))
static INLINE int __Pyx_PyObject_IsTrue(PyObject*);
static INLINE PyObject* __Pyx_PyNumber_Int(PyObject* x);

#if !defined(T_PYSSIZET)
#if PY_VERSION_HEX < 0x02050000
#define T_PYSSIZET T_INT
#elif !defined(T_LONGLONG)
#define T_PYSSIZET \
        ((sizeof(Py_ssize_t) == sizeof(int))  ? T_INT  : \
        ((sizeof(Py_ssize_t) == sizeof(long)) ? T_LONG : -1))
#else
#define T_PYSSIZET \
        ((sizeof(Py_ssize_t) == sizeof(int))          ? T_INT      : \
        ((sizeof(Py_ssize_t) == sizeof(long))         ? T_LONG     : \
        ((sizeof(Py_ssize_t) == sizeof(PY_LONG_LONG)) ? T_LONGLONG : -1)))
#endif
#endif

#if !defined(T_SIZET)
#if !defined(T_ULONGLONG)
#define T_SIZET \
        ((sizeof(size_t) == sizeof(unsigned int))  ? T_UINT  : \
        ((sizeof(size_t) == sizeof(unsigned long)) ? T_ULONG : -1))
#else
#define T_SIZET \
        ((sizeof(size_t) == sizeof(unsigned int))          ? T_UINT      : \
        ((sizeof(size_t) == sizeof(unsigned long))         ? T_ULONG     : \
        ((sizeof(size_t) == sizeof(unsigned PY_LONG_LONG)) ? T_ULONGLONG : -1)))
#endif
#endif

static INLINE Py_ssize_t __Pyx_PyIndex_AsSsize_t(PyObject*);
static INLINE PyObject * __Pyx_PyInt_FromSize_t(size_t);
static INLINE size_t __Pyx_PyInt_AsSize_t(PyObject*);

#define __pyx_PyFloat_AsDouble(x) (PyFloat_CheckExact(x) ? PyFloat_AS_DOUBLE(x) : PyFloat_AsDouble(x))


#ifdef __GNUC__
/* Test for GCC > 2.95 */
#if __GNUC__ > 2 ||               (__GNUC__ == 2 && (__GNUC_MINOR__ > 95)) 
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else /* __GNUC__ > 2 ... */
#define likely(x)   (x)
#define unlikely(x) (x)
#endif /* __GNUC__ > 2 ... */
#else /* __GNUC__ */
#define likely(x)   (x)
#define unlikely(x) (x)
#endif /* __GNUC__ */
    
static PyObject *__pyx_m;
static PyObject *__pyx_b;
static PyObject *__pyx_empty_tuple;
static int __pyx_lineno;
static int __pyx_clineno = 0;
static const char * __pyx_cfilenm= __FILE__;
static const char *__pyx_filename;
static const char **__pyx_f;


#ifdef CYTHON_REFNANNY
typedef struct {
  void (*INCREF)(void*, PyObject*, int);
  void (*DECREF)(void*, PyObject*, int);
  void (*GOTREF)(void*, PyObject*, int);
  void (*GIVEREF)(void*, PyObject*, int);
  void* (*NewContext)(const char*, int, const char*);
  void (*FinishContext)(void**);
} __Pyx_RefnannyAPIStruct;
static __Pyx_RefnannyAPIStruct *__Pyx_Refnanny = NULL;
#define __Pyx_ImportRefcountAPI(name)   (__Pyx_RefnannyAPIStruct *) PyCObject_Import((char *)name, (char *)"RefnannyAPI")
#define __Pyx_INCREF(r) __Pyx_Refnanny->INCREF(__pyx_refchk, (PyObject *)(r), __LINE__)
#define __Pyx_DECREF(r) __Pyx_Refnanny->DECREF(__pyx_refchk, (PyObject *)(r), __LINE__)
#define __Pyx_GOTREF(r) __Pyx_Refnanny->GOTREF(__pyx_refchk, (PyObject *)(r), __LINE__)
#define __Pyx_GIVEREF(r) __Pyx_Refnanny->GIVEREF(__pyx_refchk, (PyObject *)(r), __LINE__)
#define __Pyx_XDECREF(r) if((r) == NULL) ; else __Pyx_DECREF(r)
#define __Pyx_SetupRefcountContext(name)   void* __pyx_refchk = __Pyx_Refnanny->NewContext((name), __LINE__, __FILE__)
#define __Pyx_FinishRefcountContext()   __Pyx_Refnanny->FinishContext(&__pyx_refchk)
#else
#define __Pyx_INCREF(r) Py_INCREF(r)
#define __Pyx_DECREF(r) Py_DECREF(r)
#define __Pyx_GOTREF(r)
#define __Pyx_GIVEREF(r)
#define __Pyx_XDECREF(r) Py_XDECREF(r)
#define __Pyx_SetupRefcountContext(name)
#define __Pyx_FinishRefcountContext()
#endif /* CYTHON_REFNANNY */
#define __Pyx_XGIVEREF(r) if((r) == NULL) ; else __Pyx_GIVEREF(r)
#define __Pyx_XGOTREF(r) if((r) == NULL) ; else __Pyx_GOTREF(r)

static INLINE int __Pyx_StrEq(const char *, const char *); /*proto*/

static INLINE unsigned char __Pyx_PyInt_AsUnsignedChar(PyObject *);

static INLINE unsigned short __Pyx_PyInt_AsUnsignedShort(PyObject *);

static INLINE unsigned int __Pyx_PyInt_AsUnsignedInt(PyObject *);

static INLINE char __Pyx_PyInt_AsChar(PyObject *);

static INLINE short __Pyx_PyInt_AsShort(PyObject *);

static INLINE int __Pyx_PyInt_AsInt(PyObject *);

static INLINE signed char __Pyx_PyInt_AsSignedChar(PyObject *);

static INLINE signed short __Pyx_PyInt_AsSignedShort(PyObject *);

static INLINE signed int __Pyx_PyInt_AsSignedInt(PyObject *);

static INLINE unsigned long __Pyx_PyInt_AsUnsignedLong(PyObject *);

static INLINE unsigned PY_LONG_LONG __Pyx_PyInt_AsUnsignedLongLong(PyObject *);

static INLINE long __Pyx_PyInt_AsLong(PyObject *);

static INLINE PY_LONG_LONG __Pyx_PyInt_AsLongLong(PyObject *);

static INLINE signed long __Pyx_PyInt_AsSignedLong(PyObject *);

static INLINE signed PY_LONG_LONG __Pyx_PyInt_AsSignedLongLong(PyObject *);

static void __Pyx_AddTraceback(const char *funcname); /*proto*/

static int __Pyx_InitStrings(__Pyx_StringTabEntry *t); /*proto*/

/* Type declarations */
/* Module declarations from h5py.h5l */

#define __Pyx_MODULE_NAME "h5l"
int __pyx_module_is_main_h5py__h5l = 0;

/* Implementation of h5py.h5l */
static char __pyx_k___main__[] = "__main__";
static PyObject *__pyx_kp___main__;

static struct PyMethodDef __pyx_methods[] = {
  {0, 0, 0, 0}
};

static void __pyx_init_filenames(void); /*proto*/

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef __pyx_moduledef = {
    PyModuleDef_HEAD_INIT,
    __Pyx_NAMESTR("h5l"),
    0, /* m_doc */
    -1, /* m_size */
    __pyx_methods /* m_methods */,
    NULL, /* m_reload */
    NULL, /* m_traverse */
    NULL, /* m_clear */
    NULL /* m_free */
};
#endif

static __Pyx_StringTabEntry __pyx_string_tab[] = {
  {&__pyx_kp___main__, __pyx_k___main__, sizeof(__pyx_k___main__), 1, 1, 1},
  {0, 0, 0, 0, 0, 0}
};
static int __Pyx_InitCachedBuiltins(void) {
  return 0;
  return -1;
}

static int __Pyx_InitGlobals(void) {
  if (__Pyx_InitStrings(__pyx_string_tab) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1; __pyx_clineno = __LINE__; goto __pyx_L1_error;};
  return 0;
  __pyx_L1_error:;
  return -1;
}

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC inith5l(void); /*proto*/
PyMODINIT_FUNC inith5l(void)
#else
PyMODINIT_FUNC PyInit_h5l(void); /*proto*/
PyMODINIT_FUNC PyInit_h5l(void)
#endif
{
  #ifdef CYTHON_REFNANNY
  void* __pyx_refchk = NULL;
  __Pyx_Refnanny = __Pyx_ImportRefcountAPI("refnanny");
  if (!__Pyx_Refnanny) {
      PyErr_Clear();
      __Pyx_Refnanny = __Pyx_ImportRefcountAPI("Cython.Runtime.refnanny");
      if (!__Pyx_Refnanny)
          Py_FatalError("failed to import refnanny module");
  }
  __pyx_refchk = __Pyx_Refnanny->NewContext("PyMODINIT_FUNC PyInit_h5l(void)", __LINE__, __FILE__);
  #endif
  __pyx_empty_tuple = PyTuple_New(0); if (unlikely(!__pyx_empty_tuple)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  /*--- Library function declarations ---*/
  __pyx_init_filenames();
  /*--- Threads initialization code ---*/
  #if defined(__PYX_FORCE_INIT_THREADS) && __PYX_FORCE_INIT_THREADS
  #ifdef WITH_THREAD /* Python build with threading support? */
  PyEval_InitThreads();
  #endif
  #endif
  /*--- Initialize various global constants etc. ---*/
  if (unlikely(__Pyx_InitGlobals() < 0)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  /*--- Module creation code ---*/
  #if PY_MAJOR_VERSION < 3
  __pyx_m = Py_InitModule4(__Pyx_NAMESTR("h5l"), __pyx_methods, 0, 0, PYTHON_API_VERSION);
  #else
  __pyx_m = PyModule_Create(&__pyx_moduledef);
  #endif
  if (!__pyx_m) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1; __pyx_clineno = __LINE__; goto __pyx_L1_error;};
  #if PY_MAJOR_VERSION < 3
  Py_INCREF(__pyx_m);
  #endif
  __pyx_b = PyImport_AddModule(__Pyx_NAMESTR(__Pyx_BUILTIN_MODULE_NAME));
  if (!__pyx_b) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1; __pyx_clineno = __LINE__; goto __pyx_L1_error;};
  if (__Pyx_SetAttrString(__pyx_m, "__builtins__", __pyx_b) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1; __pyx_clineno = __LINE__; goto __pyx_L1_error;};
  if (__pyx_module_is_main_h5py__h5l) {
    if (__Pyx_SetAttrString(__pyx_m, "__name__", __pyx_kp___main__) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1; __pyx_clineno = __LINE__; goto __pyx_L1_error;};
  }
  /*--- Builtin init code ---*/
  if (unlikely(__Pyx_InitCachedBuiltins() < 0)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_skip_dispatch = 0;
  /*--- Global init code ---*/
  /*--- Function export code ---*/
  /*--- Type init code ---*/
  /*--- Type import code ---*/
  /*--- Function import code ---*/
  /*--- Execution code ---*/

  /* "/media/storage/Documents/Work/Examproject/Devel/branch_gpaw/c/h5py/h5l.pxd":1
 * #+             # <<<<<<<<<<<<<<
 * #
 * # This file is part of h5py, a low-level Python interface to the HDF5 library.
 */
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_AddTraceback("h5l");
  Py_DECREF(__pyx_m); __pyx_m = 0;
  __pyx_L0:;
  __Pyx_FinishRefcountContext();
  #if PY_MAJOR_VERSION < 3
  return;
  #else
  return __pyx_m;
  #endif
}

static const char *__pyx_filenames[] = {
  "h5l.pyx",
};

/* Runtime support code */

static void __pyx_init_filenames(void) {
  __pyx_f = __pyx_filenames;
}

static INLINE int __Pyx_StrEq(const char *s1, const char *s2) {
     while (*s1 != '\0' && *s1 == *s2) { s1++; s2++; }
     return *s1 == *s2;
}

static INLINE unsigned char __Pyx_PyInt_AsUnsignedChar(PyObject* x) {
    if (sizeof(unsigned char) < sizeof(long)) {
        long val = __Pyx_PyInt_AsLong(x);
        if (unlikely(val != (long)(unsigned char)val)) {
            if (unlikely(val == -1 && PyErr_Occurred()))
                return (unsigned char)-1;
            if (unlikely(val < 0)) {
                PyErr_SetString(PyExc_OverflowError,
                                "can't convert negative value to unsigned char");
                return (unsigned char)-1;
            }
            PyErr_SetString(PyExc_OverflowError,
                           "value too large to convert to unsigned char");
            return (unsigned char)-1;
        }
        return (unsigned char)val;
    }
    return (unsigned char)__Pyx_PyInt_AsUnsignedLong(x);
}

static INLINE unsigned short __Pyx_PyInt_AsUnsignedShort(PyObject* x) {
    if (sizeof(unsigned short) < sizeof(long)) {
        long val = __Pyx_PyInt_AsLong(x);
        if (unlikely(val != (long)(unsigned short)val)) {
            if (unlikely(val == -1 && PyErr_Occurred()))
                return (unsigned short)-1;
            if (unlikely(val < 0)) {
                PyErr_SetString(PyExc_OverflowError,
                                "can't convert negative value to unsigned short");
                return (unsigned short)-1;
            }
            PyErr_SetString(PyExc_OverflowError,
                           "value too large to convert to unsigned short");
            return (unsigned short)-1;
        }
        return (unsigned short)val;
    }
    return (unsigned short)__Pyx_PyInt_AsUnsignedLong(x);
}

static INLINE unsigned int __Pyx_PyInt_AsUnsignedInt(PyObject* x) {
    if (sizeof(unsigned int) < sizeof(long)) {
        long val = __Pyx_PyInt_AsLong(x);
        if (unlikely(val != (long)(unsigned int)val)) {
            if (unlikely(val == -1 && PyErr_Occurred()))
                return (unsigned int)-1;
            if (unlikely(val < 0)) {
                PyErr_SetString(PyExc_OverflowError,
                                "can't convert negative value to unsigned int");
                return (unsigned int)-1;
            }
            PyErr_SetString(PyExc_OverflowError,
                           "value too large to convert to unsigned int");
            return (unsigned int)-1;
        }
        return (unsigned int)val;
    }
    return (unsigned int)__Pyx_PyInt_AsUnsignedLong(x);
}

static INLINE char __Pyx_PyInt_AsChar(PyObject* x) {
    if (sizeof(char) < sizeof(long)) {
        long val = __Pyx_PyInt_AsLong(x);
        if (unlikely(val != (long)(char)val)) {
            if (unlikely(val == -1 && PyErr_Occurred()))
                return (char)-1;
            PyErr_SetString(PyExc_OverflowError,
                           "value too large to convert to char");
            return (char)-1;
        }
        return (char)val;
    }
    return (char)__Pyx_PyInt_AsLong(x);
}

static INLINE short __Pyx_PyInt_AsShort(PyObject* x) {
    if (sizeof(short) < sizeof(long)) {
        long val = __Pyx_PyInt_AsLong(x);
        if (unlikely(val != (long)(short)val)) {
            if (unlikely(val == -1 && PyErr_Occurred()))
                return (short)-1;
            PyErr_SetString(PyExc_OverflowError,
                           "value too large to convert to short");
            return (short)-1;
        }
        return (short)val;
    }
    return (short)__Pyx_PyInt_AsLong(x);
}

static INLINE int __Pyx_PyInt_AsInt(PyObject* x) {
    if (sizeof(int) < sizeof(long)) {
        long val = __Pyx_PyInt_AsLong(x);
        if (unlikely(val != (long)(int)val)) {
            if (unlikely(val == -1 && PyErr_Occurred()))
                return (int)-1;
            PyErr_SetString(PyExc_OverflowError,
                           "value too large to convert to int");
            return (int)-1;
        }
        return (int)val;
    }
    return (int)__Pyx_PyInt_AsLong(x);
}

static INLINE signed char __Pyx_PyInt_AsSignedChar(PyObject* x) {
    if (sizeof(signed char) < sizeof(long)) {
        long val = __Pyx_PyInt_AsLong(x);
        if (unlikely(val != (long)(signed char)val)) {
            if (unlikely(val == -1 && PyErr_Occurred()))
                return (signed char)-1;
            PyErr_SetString(PyExc_OverflowError,
                           "value too large to convert to signed char");
            return (signed char)-1;
        }
        return (signed char)val;
    }
    return (signed char)__Pyx_PyInt_AsSignedLong(x);
}

static INLINE signed short __Pyx_PyInt_AsSignedShort(PyObject* x) {
    if (sizeof(signed short) < sizeof(long)) {
        long val = __Pyx_PyInt_AsLong(x);
        if (unlikely(val != (long)(signed short)val)) {
            if (unlikely(val == -1 && PyErr_Occurred()))
                return (signed short)-1;
            PyErr_SetString(PyExc_OverflowError,
                           "value too large to convert to signed short");
            return (signed short)-1;
        }
        return (signed short)val;
    }
    return (signed short)__Pyx_PyInt_AsSignedLong(x);
}

static INLINE signed int __Pyx_PyInt_AsSignedInt(PyObject* x) {
    if (sizeof(signed int) < sizeof(long)) {
        long val = __Pyx_PyInt_AsLong(x);
        if (unlikely(val != (long)(signed int)val)) {
            if (unlikely(val == -1 && PyErr_Occurred()))
                return (signed int)-1;
            PyErr_SetString(PyExc_OverflowError,
                           "value too large to convert to signed int");
            return (signed int)-1;
        }
        return (signed int)val;
    }
    return (signed int)__Pyx_PyInt_AsSignedLong(x);
}

static INLINE unsigned long __Pyx_PyInt_AsUnsignedLong(PyObject* x) {
#if PY_VERSION_HEX < 0x03000000
    if (likely(PyInt_CheckExact(x) || PyInt_Check(x))) {
        long val = PyInt_AS_LONG(x);
        if (unlikely(val < 0)) {
            PyErr_SetString(PyExc_OverflowError,
                            "can't convert negative value to unsigned long");
            return (unsigned long)-1;
        }
        return (unsigned long)val;
    } else
#endif
    if (likely(PyLong_CheckExact(x) || PyLong_Check(x))) {
        if (unlikely(Py_SIZE(x) < 0)) {
            PyErr_SetString(PyExc_OverflowError,
                            "can't convert negative value to unsigned long");
            return (unsigned long)-1;
        }
        return PyLong_AsUnsignedLong(x);
    } else {
        unsigned long val;
        PyObject *tmp = __Pyx_PyNumber_Int(x);
        if (!tmp) return (unsigned long)-1;
        val = __Pyx_PyInt_AsUnsignedLong(tmp);
        Py_DECREF(tmp);
        return val;
    }
}

static INLINE unsigned PY_LONG_LONG __Pyx_PyInt_AsUnsignedLongLong(PyObject* x) {
#if PY_VERSION_HEX < 0x03000000
    if (likely(PyInt_CheckExact(x) || PyInt_Check(x))) {
        long val = PyInt_AS_LONG(x);
        if (unlikely(val < 0)) {
            PyErr_SetString(PyExc_OverflowError,
                            "can't convert negative value to unsigned PY_LONG_LONG");
            return (unsigned PY_LONG_LONG)-1;
        }
        return (unsigned PY_LONG_LONG)val;
    } else
#endif
    if (likely(PyLong_CheckExact(x) || PyLong_Check(x))) {
        if (unlikely(Py_SIZE(x) < 0)) {
            PyErr_SetString(PyExc_OverflowError,
                            "can't convert negative value to unsigned PY_LONG_LONG");
            return (unsigned PY_LONG_LONG)-1;
        }
        return PyLong_AsUnsignedLongLong(x);
    } else {
        unsigned PY_LONG_LONG val;
        PyObject *tmp = __Pyx_PyNumber_Int(x);
        if (!tmp) return (unsigned PY_LONG_LONG)-1;
        val = __Pyx_PyInt_AsUnsignedLongLong(tmp);
        Py_DECREF(tmp);
        return val;
    }
}

static INLINE long __Pyx_PyInt_AsLong(PyObject* x) {
#if PY_VERSION_HEX < 0x03000000
    if (likely(PyInt_CheckExact(x) || PyInt_Check(x))) {
        long val = PyInt_AS_LONG(x);
        return (long)val;
    } else
#endif
    if (likely(PyLong_CheckExact(x) || PyLong_Check(x))) {
        return PyLong_AsLong(x);
    } else {
        long val;
        PyObject *tmp = __Pyx_PyNumber_Int(x);
        if (!tmp) return (long)-1;
        val = __Pyx_PyInt_AsLong(tmp);
        Py_DECREF(tmp);
        return val;
    }
}

static INLINE PY_LONG_LONG __Pyx_PyInt_AsLongLong(PyObject* x) {
#if PY_VERSION_HEX < 0x03000000
    if (likely(PyInt_CheckExact(x) || PyInt_Check(x))) {
        long val = PyInt_AS_LONG(x);
        return (PY_LONG_LONG)val;
    } else
#endif
    if (likely(PyLong_CheckExact(x) || PyLong_Check(x))) {
        return PyLong_AsLongLong(x);
    } else {
        PY_LONG_LONG val;
        PyObject *tmp = __Pyx_PyNumber_Int(x);
        if (!tmp) return (PY_LONG_LONG)-1;
        val = __Pyx_PyInt_AsLongLong(tmp);
        Py_DECREF(tmp);
        return val;
    }
}

static INLINE signed long __Pyx_PyInt_AsSignedLong(PyObject* x) {
#if PY_VERSION_HEX < 0x03000000
    if (likely(PyInt_CheckExact(x) || PyInt_Check(x))) {
        long val = PyInt_AS_LONG(x);
        return (signed long)val;
    } else
#endif
    if (likely(PyLong_CheckExact(x) || PyLong_Check(x))) {
        return PyLong_AsLong(x);
    } else {
        signed long val;
        PyObject *tmp = __Pyx_PyNumber_Int(x);
        if (!tmp) return (signed long)-1;
        val = __Pyx_PyInt_AsSignedLong(tmp);
        Py_DECREF(tmp);
        return val;
    }
}

static INLINE signed PY_LONG_LONG __Pyx_PyInt_AsSignedLongLong(PyObject* x) {
#if PY_VERSION_HEX < 0x03000000
    if (likely(PyInt_CheckExact(x) || PyInt_Check(x))) {
        long val = PyInt_AS_LONG(x);
        return (signed PY_LONG_LONG)val;
    } else
#endif
    if (likely(PyLong_CheckExact(x) || PyLong_Check(x))) {
        return PyLong_AsLongLong(x);
    } else {
        signed PY_LONG_LONG val;
        PyObject *tmp = __Pyx_PyNumber_Int(x);
        if (!tmp) return (signed PY_LONG_LONG)-1;
        val = __Pyx_PyInt_AsSignedLongLong(tmp);
        Py_DECREF(tmp);
        return val;
    }
}

#include "compile.h"
#include "frameobject.h"
#include "traceback.h"

static void __Pyx_AddTraceback(const char *funcname) {
    PyObject *py_srcfile = 0;
    PyObject *py_funcname = 0;
    PyObject *py_globals = 0;
    PyObject *empty_string = 0;
    PyCodeObject *py_code = 0;
    PyFrameObject *py_frame = 0;

    #if PY_MAJOR_VERSION < 3
    py_srcfile = PyString_FromString(__pyx_filename);
    #else
    py_srcfile = PyUnicode_FromString(__pyx_filename);
    #endif
    if (!py_srcfile) goto bad;
    if (__pyx_clineno) {
        #if PY_MAJOR_VERSION < 3
        py_funcname = PyString_FromFormat( "%s (%s:%d)", funcname, __pyx_cfilenm, __pyx_clineno);
        #else
        py_funcname = PyUnicode_FromFormat( "%s (%s:%d)", funcname, __pyx_cfilenm, __pyx_clineno);
        #endif
    }
    else {
        #if PY_MAJOR_VERSION < 3
        py_funcname = PyString_FromString(funcname);
        #else
        py_funcname = PyUnicode_FromString(funcname);
        #endif
    }
    if (!py_funcname) goto bad;
    py_globals = PyModule_GetDict(__pyx_m);
    if (!py_globals) goto bad;
    #if PY_MAJOR_VERSION < 3
    empty_string = PyString_FromStringAndSize("", 0);
    #else
    empty_string = PyBytes_FromStringAndSize("", 0);
    #endif
    if (!empty_string) goto bad;
    py_code = PyCode_New(
        0,            /*int argcount,*/
        #if PY_MAJOR_VERSION >= 3
        0,            /*int kwonlyargcount,*/
        #endif
        0,            /*int nlocals,*/
        0,            /*int stacksize,*/
        0,            /*int flags,*/
        empty_string, /*PyObject *code,*/
        __pyx_empty_tuple,  /*PyObject *consts,*/
        __pyx_empty_tuple,  /*PyObject *names,*/
        __pyx_empty_tuple,  /*PyObject *varnames,*/
        __pyx_empty_tuple,  /*PyObject *freevars,*/
        __pyx_empty_tuple,  /*PyObject *cellvars,*/
        py_srcfile,   /*PyObject *filename,*/
        py_funcname,  /*PyObject *name,*/
        __pyx_lineno,   /*int firstlineno,*/
        empty_string  /*PyObject *lnotab*/
    );
    if (!py_code) goto bad;
    py_frame = PyFrame_New(
        PyThreadState_GET(), /*PyThreadState *tstate,*/
        py_code,             /*PyCodeObject *code,*/
        py_globals,          /*PyObject *globals,*/
        0                    /*PyObject *locals*/
    );
    if (!py_frame) goto bad;
    py_frame->f_lineno = __pyx_lineno;
    PyTraceBack_Here(py_frame);
bad:
    Py_XDECREF(py_srcfile);
    Py_XDECREF(py_funcname);
    Py_XDECREF(empty_string);
    Py_XDECREF(py_code);
    Py_XDECREF(py_frame);
}

static int __Pyx_InitStrings(__Pyx_StringTabEntry *t) {
    while (t->p) {
        #if PY_MAJOR_VERSION < 3
        if (t->is_unicode && (!t->is_identifier)) {
            *t->p = PyUnicode_DecodeUTF8(t->s, t->n - 1, NULL);
        } else if (t->intern) {
            *t->p = PyString_InternFromString(t->s);
        } else {
            *t->p = PyString_FromStringAndSize(t->s, t->n - 1);
        }
        #else  /* Python 3+ has unicode identifiers */
        if (t->is_identifier || (t->is_unicode && t->intern)) {
            *t->p = PyUnicode_InternFromString(t->s);
        } else if (t->is_unicode) {
            *t->p = PyUnicode_FromStringAndSize(t->s, t->n - 1);
        } else {
            *t->p = PyBytes_FromStringAndSize(t->s, t->n - 1);
        }
        #endif
        if (!*t->p)
            return -1;
        ++t;
    }
    return 0;
}

/* Type Conversion Functions */

static INLINE int __Pyx_PyObject_IsTrue(PyObject* x) {
   if (x == Py_True) return 1;
   else if ((x == Py_False) | (x == Py_None)) return 0;
   else return PyObject_IsTrue(x);
}

static INLINE PyObject* __Pyx_PyNumber_Int(PyObject* x) {
  PyNumberMethods *m;
  const char *name = NULL;
  PyObject *res = NULL;
#if PY_VERSION_HEX < 0x03000000
  if (PyInt_Check(x) || PyLong_Check(x))
#else
  if (PyLong_Check(x))
#endif
    return Py_INCREF(x), x;
  m = Py_TYPE(x)->tp_as_number;
#if PY_VERSION_HEX < 0x03000000
  if (m && m->nb_int) {
    name = "int";
    res = PyNumber_Int(x);
  }
  else if (m && m->nb_long) {
    name = "long";
    res = PyNumber_Long(x);
  }
#else
  if (m && m->nb_int) {
    name = "int";
    res = PyNumber_Long(x);
  }
#endif
  if (res) {
#if PY_VERSION_HEX < 0x03000000
    if (!PyInt_Check(res) && !PyLong_Check(res)) {
#else
    if (!PyLong_Check(res)) {
#endif
      PyErr_Format(PyExc_TypeError,
                   "__%s__ returned non-%s (type %.200s)",
                   name, name, Py_TYPE(res)->tp_name);
      Py_DECREF(res);
      return NULL;
    }
  }
  else if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError,
                    "an integer is required");
  }
  return res;
}

static INLINE Py_ssize_t __Pyx_PyIndex_AsSsize_t(PyObject* b) {
  Py_ssize_t ival;
  PyObject* x = PyNumber_Index(b);
  if (!x) return -1;
  ival = PyInt_AsSsize_t(x);
  Py_DECREF(x);
  return ival;
}

static INLINE PyObject * __Pyx_PyInt_FromSize_t(size_t ival) {
#if PY_VERSION_HEX < 0x02050000
   if (ival <= LONG_MAX)
       return PyInt_FromLong((long)ival);
   else {
       unsigned char *bytes = (unsigned char *) &ival;
       int one = 1; int little = (int)*(unsigned char*)&one;
       return _PyLong_FromByteArray(bytes, sizeof(size_t), little, 0);
   }
#else
   return PyInt_FromSize_t(ival);
#endif
}

static INLINE size_t __Pyx_PyInt_AsSize_t(PyObject* x) {
   unsigned PY_LONG_LONG val = __Pyx_PyInt_AsUnsignedLongLong(x);
   if (unlikely(val == (unsigned PY_LONG_LONG)-1 && PyErr_Occurred())) {
       return (size_t)-1;
   } else if (unlikely(val != (unsigned PY_LONG_LONG)(size_t)val)) {
       PyErr_SetString(PyExc_OverflowError,
                       "value too large to convert to size_t");
       return (size_t)-1;
   }
   return (size_t)val;
}


