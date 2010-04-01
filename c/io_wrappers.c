/*  Copyright (C) 2010       CSC - IT Center for Science Ltd.
 *  Please see the accompanying LICENSE file for further information. */

#include "extensions.h"
#ifdef PARALLEL
#ifdef IO_WRAPPERS
#include "io_wrappers.h"
#include <mpi.h>
#include <stdio.h>
#include <sys/stat.h>
#include <Python.h>

#define MASTER 0
#define MAX_FILES FOPEN_MAX // 5000 Maximum number of files to open in parallel

static int rank = MASTER; 
static int enabled = 0;

static FILE *parallel_fps[MAX_FILES];
static int current_fp = 0;
static FILE *fp_dev_null;

// Initialize wrapper stuff
void init_io_wrappers() 
{
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  for (int i=0; i < MAX_FILES; i++)
    parallel_fps[i] = -1;
  fp_dev_null = __real_fopen64("/dev/null", "rb");
  enabled = 1;
}

// switching wrapping on and off
void enable_io_wrappers()
{
  enabled = 1;
}

void disable_io_wrappers()
{
  enabled = 0;
}

// Utility function to check if the file pointer is "parallel"
int check_fp(FILE *fp)
{
  for (int i=current_fp-1; i >=0; i--)
    if ( fp == parallel_fps[i] )
      return i+1;

  return 0;
}


// File open, close, lock, etc.
FILE* __wrap_fopen(const char *filename, const char *modes)
{
  FILE *fp;
  int fp_is_null;
  // Wrap only in read mode
  if ( modes[0] == 'r' && enabled )
    {
#ifdef IO_DEBUG
      printf("Opening: %d %s\n", rank, filename);
#endif
      if (rank == MASTER )
	{
	  fp = __real_fopen(filename, modes);
	  // NULL information is needed also in other ranks
          fp_is_null = ((fp == NULL) ? 1 : 0);
	  MPI_Bcast(&fp_is_null, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	}
      else
	{
	  //	{ 
	  MPI_Bcast(&fp_is_null, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
          if ( fp_is_null)
	    fp = NULL;
	  else
	    fp = fp_dev_null;
	} 
      // Store the "parallel" file pointer
      if (fp != NULL)
	{
	  parallel_fps[current_fp] = fp;
	  current_fp++;
	  if (current_fp == MAX_FILES)
	    {
	      printf("Too many open files\n");
	      MPI_Abort(MPI_COMM_WORLD, -1);
	    }
	}
      MPI_Barrier(MPI_COMM_WORLD);
    }
  else
    // Write mode, all processes can participate
    fp = __real_fopen(filename, modes);
  return fp;
}

FILE* __wrap_fopen64(const char *filename, const char *modes)
{
  FILE *fp;
  int fp_is_null;
  // Wrap only in read mode
  if ( modes[0] == 'r' && enabled)
    {
#ifdef IO_DEBUG
      printf("Opening: %d %s\n", rank, filename);
#endif
      if (rank == MASTER )
	{
	  fp = __real_fopen64(filename, modes);
	  // NULL information is needed also in other ranks
          fp_is_null = ((fp == NULL) ? 1 : 0);
	  MPI_Bcast(&fp_is_null, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	}
      else
	{
	  //	{ 
	  MPI_Bcast(&fp_is_null, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
          if ( fp_is_null)
	    fp = NULL;
	  else
	    fp = fp_dev_null;
	} 
      // Store the "parallel" file pointer
      if (fp != NULL)
	{
	  parallel_fps[current_fp] = fp;
	  current_fp++;
	  if (current_fp == MAX_FILES)
	    {
	      printf("Too many open files\n");
	      MPI_Abort(MPI_COMM_WORLD, -1);
	    }
	}
      MPI_Barrier(MPI_COMM_WORLD);
    }
  else
    // Write mode, all processes can participate
    fp = __real_fopen64(filename, modes);
  return fp;
}

int  __wrap_fclose(FILE *fp)
{
  int x;
  int i = check_fp(fp);
  if ( i == current_fp && i > 0 )
    current_fp--;
  if ( ! i || (rank == MASTER) ) 
    {
      x = __real_fclose(fp);
    }

  // TODO: Should one return the true return value of fclose?
  return 0;
}

void  __wrap_setbuf(FILE *fp, char *buf)
{
  if ( ! check_fp(fp) || (rank == MASTER) ) 
      __real_setbuf(fp, buf);
}

int  __wrap_setvbuf(FILE *fp, char *buf, int type, size_t size)
{
  int x;
  if (check_fp(fp)) 
    {
      if (rank == MASTER) 
	{
 	  x = __real_setvbuf(fp, buf, type, size);
	  MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
	}
      else
	MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
    }
  else
    x = __real_setvbuf(fp, buf, type, size);
  return x;
}

int  __wrap_flockfile(FILE *fp)
{
  if ( ! check_fp(fp) || (rank == MASTER) ) 
      __real_flockfile(fp);
  return 0;
}

int  __wrap_funlockfile(FILE *fp)
{
  if ( ! check_fp(fp) || (rank == MASTER) ) 
      __real_funlockfile(fp);
  return 0;
}

int __wrap_ferror(FILE* fp)
{
  int x;
  if (check_fp(fp)) 
    {
      if (rank == MASTER) 
	{
	  x = __real_ferror(fp);
	  MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
	}
      else
	MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
    }
  else
    x = __real_ferror(fp);
   return x;
}

int __wrap_feof(FILE* fp)
{
  int x;
  if (check_fp(fp)) 
    {
      if (rank == MASTER) 
	{
	  x = __real_feof(fp);
	  MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
	}
      else
	MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
    }
  else
    x = __real_feof(fp);
   return x;
}

void  __wrap_clearerr(FILE *fp)
{
  if ( ! check_fp(fp) || (rank == MASTER) ) 
      __real_clearerr(fp);
}

// File positioning etc.
int __wrap_fseek(FILE *fp, long offset, int origin)
{
  int x;
  if (check_fp(fp)) 
    {
      if (rank == MASTER) 
	{
	  x = __real_fseek(fp, offset, origin);
	  MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
	}
      else
	MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
    }
  else
    x = __real_fseek(fp, offset, origin);
  return x;
}

void __wrap_rewind(FILE *fp)
{
  if (! check_fp(fp) || (rank == MASTER)) 
      __real_rewind(fp);
}


int __wrap_ungetc(int c, FILE* fp)
{
   int x;
   if (enabled) 
     if (rank == MASTER) 
       {
	 x =__real_ungetc(c, fp);
	 MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
       }
     else
       MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
   else
     x =__real_ungetc(c, fp);
   return x;
}

int __wrap_fflush(FILE *fp)
{
  int x;
  if (check_fp(fp)) 
    {
      if (rank == MASTER) 
	{
	  x = __real_fflush(fp);
	  MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
	}
      else
	MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
    }
  else
    x = __real_fflush(fp);
   return x;
}

int __wrap_fgetpos ( FILE * fp, fpos_t * pos )
{
  int x;
  if (enabled) 
    if (rank == MASTER) 
      {
	x = __real_fgetpos(fp, pos);
	MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
      }
    else
      MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
  else
    x = __real_fgetpos(fp, pos);
  return x;
}

int __wrap_fsetpos ( FILE * fp, const fpos_t * pos )
  {
    int x;
    if (enabled) 
      if (rank == MASTER) 
	{
	  x = __real_fsetpos(fp, pos);
	  MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
	}
      else
	MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
    else
      x = __real_fsetpos(fp, pos);
  return x;
  }

long int __wrap_ftell ( FILE * fp )
  {
    long x;
    if (enabled) 
      if (rank == MASTER) 
	{
	  x = __real_ftell(fp);
	  MPI_Bcast(&x, 1, MPI_LONG, MASTER, MPI_COMM_WORLD); 
	}
      else
	MPI_Bcast(&x, 1, MPI_LONG, MASTER, MPI_COMM_WORLD); 
    else
      x = __real_ftell(fp);
    return x;
  }

// Read functions
int __wrap__IO_getc(FILE *fp)
{
  // printf("getc %d\n", rank);
  int x;
  if (enabled) 
    if (rank == MASTER )
      {
	x = __real__IO_getc(fp);
	MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
      }
    else
      {
	MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
      }
  else
    x = __real__IO_getc(fp);
  return x;
}

int __wrap_getc_unlocked(FILE *fp)
{
  int x;
  if (enabled) 
    if (rank == MASTER )
      {
	x = __real_getc_unlocked(fp);
	MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
      }
    else
      {
	MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
      }
  else
    x = __real_getc_unlocked(fp);
  return x;
}

int __wrap_fread(void *ptr, size_t size, size_t n, FILE* fp)
{
   // printf("read %d\n", rank);
   // Is it OK to use just int for the size of data read?
   int x;
   if (enabled) 
     if (rank == MASTER) 
       {
	 x = __real_fread(ptr, size, n, fp);
	 MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	 MPI_Bcast(ptr, x*size, MPI_BYTE, MASTER, MPI_COMM_WORLD); 
       }
     else
       {
	 MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	 MPI_Bcast(ptr, x*size, MPI_BYTE, MASTER, MPI_COMM_WORLD); 
       }
   else
     x = __real_fread(ptr, size, n, fp);
   return x;
}
              
char *__wrap_fgets(char *str, int num, FILE* fp)
{
  char* s;
  int s_is_null=0;
   if (enabled) 
     if (rank == MASTER) 
       {
	 s = __real_fgets(str, num, fp);
	 if (s==NULL)
	   {
	     s_is_null = 1;
	     MPI_Bcast(&s_is_null, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
	   }
	 else
	   {
	     MPI_Bcast(&s_is_null, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
	     MPI_Bcast(s, num, MPI_BYTE, MASTER, MPI_COMM_WORLD); 
	   }
       }
     else
       {
         MPI_Bcast(&s_is_null, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
         if (s_is_null == 1)
           s = NULL;
         else
           MPI_Bcast(s, num, MPI_BYTE, MASTER, MPI_COMM_WORLD); 
       }    
   else   
     s = __real_fgets(str, num, fp);
   return s;
}

int __wrap_fgetc ( FILE * fp )
  {
  int x;
  if (enabled) 
    if (rank == MASTER )
      {
	x = __real_fgetc(fp);
	MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
      }
    else
      MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  else
    x = __real_fgetc(fp);
  return x;
}

int __wrap_fstat(int fildes, struct stat *buf)
{
 // printf("fstat %d\n", rank);
 int size = sizeof(struct stat);
 if (enabled) 
   if (rank == MASTER) 
     {
       __real_fstat(fildes, buf);
       MPI_Bcast(buf, size, MPI_BYTE, MASTER, MPI_COMM_WORLD); 
     }
   else
     MPI_Bcast(buf, size, MPI_BYTE, MASTER, MPI_COMM_WORLD); 
 else
   __real_fstat(fildes, buf);

 // TODO should one return the true return value?
 return 0;
}

int __wrap_fstat64(int fildes, struct stat *buf)
{
 // printf("fstat64 %d\n", rank);
 int size = sizeof(struct stat);
 if (enabled) 
   if (rank == MASTER) 
     {
       __real_fstat(fildes, buf);
       MPI_Bcast(buf, size, MPI_BYTE, MASTER, MPI_COMM_WORLD); 
     }
   else
     MPI_Bcast(buf, size, MPI_BYTE, MASTER, MPI_COMM_WORLD); 
 else
   __real_fstat(fildes, buf);
 return 0;
}

/* fileno is not actually needed
int __wrap_fileno( FILE *fp )
{
  int x;
  if (check_fp(fp)) 
    {
      if (rank == MASTER) 
	{
	  x = __real_fileno(fp);
	  MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
	}
      else
	MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
    }
  else
    x = __real_fileno(fp);
  return x;
  }
*/


// Write functions
/*
int __wrap_fputc ( int character, FILE * fp )
  {
    int x;
    if (rank == MASTER) 
      {
	x =  __real_fputc(character, fp);
	MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
      }
    else
      MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 

    return x;
  }
      
int __wrap_fputs ( const char * str, FILE * fp )
  {
    int x;
    if (rank == MASTER) 
      {
	x = __real_fputs(str, fp);
	MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
      }
    else
      MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 

    return x;
  }

int __wrap__IO_putc ( int character, FILE * fp )
  {
    int x;
    if (rank == MASTER) 
      {
	x = __real__IO_putc(character, fp);
	MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
      }
    else
      MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
    return x;
  }

size_t __wrap_fwrite ( const void * ptr, size_t size, size_t count, FILE * fp )
{
    int x;
    if (rank == MASTER) 
    {
       x = __real_fwrite(ptr, size, count, fp);
	MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
       }
     else
	MPI_Bcast(&x, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
    return x;
  }
*/
#endif
#endif

// Python interfaces
PyObject* Py_enable_io_wrappers(PyObject *self, PyObject *args)
{
#ifdef PARALLEL
#ifdef IO_WRAPPERS
  enabled = 1;
#endif
#endif
  Py_RETURN_NONE;
}

PyObject* Py_disable_io_wrappers(PyObject *self, PyObject *args)
{
#ifdef PARALLEL
#ifdef IO_WRAPPERS
  enabled = 0;
#endif
#endif
  Py_RETURN_NONE;
}
