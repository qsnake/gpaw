# Makefile for building a custom python-interpreter which has 
# all the gpaw-extensions as well as Numeric-extensions statically build in
# 
# It is recommended that python distutils is used whenever possible, 
# this Makefile should be used only when distutils cannot be used.
#
# The example flags correspond to the Cray Catamount operating system
#

CC=cc -target=catamount -c9x -fast
CFLAGS=-I/wrk/jenkovaa/include/python2.4 -DGPAW_INTERPRETER -DPARALLEL -DSTATIC_NUMERIC -DNO_SOCKET
LIBDIR=${HOME}/lib
LDFLAGS=-L$(LIBDIR)  -lpython2.4 -lnumpy24.2 -ldl -lz -lexpat

ARCH := $(shell uname -p)
OBJDIR = build/temp.$(ARCH)
DO_IT := $(shell mkdir -p $(OBJDIR))

OBJS=$(OBJDIR)/bc.o $(OBJDIR)/lapack.o $(OBJDIR)/transformers.o \
     $(OBJDIR)/blas.o $(OBJDIR)/localized_functions.o $(OBJDIR)/rpbe.o \
     $(OBJDIR)/utilities.o $(OBJDIR)/d2Excdn2.o $(OBJDIR)/mpi.o \
     $(OBJDIR)/spline.o $(OBJDIR)/vasiliev02prb.o $(OBJDIR)/elf.o \
     $(OBJDIR)/operators.o $(OBJDIR)/tpss.o $(OBJDIR)/tpss_ec.o \
     $(OBJDIR)/tpss_ex.o $(OBJDIR)/xc.o $(OBJDIR)/plt.o \
     $(OBJDIR)/ensemble_gga.o $(OBJDIR)/pbe.o $(OBJDIR)/pw91.o \
     $(OBJDIR)/bmgs.o $(OBJDIR)/_gpaw.o


gpaw-python: $(OBJS)
	$(CC) -o gpaw-python $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: c/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/%.o: c/bmgs/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJDIR)/*.o
	rm -f gpaw-python
