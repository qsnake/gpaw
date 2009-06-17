.. _louhi:

============
louhi.csc.fi
============

Here you find information about the the system
`<http://raketti.csc.fi/english/research/Computing_services/computing/servers/louhi>`_.

The current operating system in Cray XT4 compute nodes, Compute Linux
Environment (CLE) has some limitations, most notably it does not
support shared libraries. In order to use python in CLE some
modifications to the standard python are needed. Before installing a
special python, there are two packages which are needed by GPAW, but
which are not included in the python distribution. Installation of
expat_ and zlib_ should succee with a standard ``./configure; make;
make install;`` procedure.

.. _expat: http://expat.sourceforge.net/
.. _zlib: http://www.zlib.net/  

Next, one can proceed with the actual python installation. The
following instructions are tested with python 2.5.1, and it is assumed
that one is working in the top level of python source
directory. First, one should create a special dynamic loader for
correct resolution of namespaces. Save the file
:svn:`~doc/install/Cray/dynload_redstorm.c`
in the :file:`Python/` directory::

  /* This module provides the simulation of dynamic loading in Red Storm */

  #include "Python.h"
  #include "importdl.h"

  const struct filedescr _PyImport_DynLoadFiletab[] = {
    {".a", "rb", C_EXTENSION},
    {0, 0}
  };

  extern struct _inittab _PyImport_Inittab[];

  dl_funcptr _PyImport_GetDynLoadFunc(const char *fqname, const char *shortname,
                                      const char *pathname, FILE *fp)
  {
    struct _inittab *tab = _PyImport_Inittab;
    while (tab->name && strcmp(shortname, tab->name)) tab++;

    return tab->initfunc;
  }

Then, one should remove ``sharemods`` from ``all:`` target in
:file:`Makefile.pre.in` and set the correct C compiler and flags,
e.g.::

 setenv CC cc
 setenv OPT '-fastsse'

You should be now ready to run :file:`configure`::

  ./configure --prefix=<install_path> SO=.a DYNLOADFILE=dynload_redstorm.o MACHDEP=redstorm --host=x86_64-unknown-linux-gnu --disable-sockets --disable-ssl --enable-static --disable-shared --without-threads

Now, one should specify which modules will be statically linked in to
the python interpreter by editing :file:`Modules/Setup`. An example can be
loaded here :svn:`~doc/install/Cray/Setup`.
Note that at this point all numpy related stuff
in the example should be commented out. Finally, in order to use
``distutils`` for building extensions the following function should be
added to the end of :file:`Lib/distutils/unixccompiler.py` so that instead
of shared libraries static ones are created

.. literalinclude:: linkforshared.py

If copy-pasting the above code block, be sure to have the correct indentation (four whitespaces before ``def link_for_shared...``), or download the whole  
file: :svn:`~doc/install/Cray/unixccompiler.py`

You should be now ready to run ``make`` and ``make install`` and have
a working python interpreter.

Next, one can use the newly created interpreter for installing
``numpy``. Switch to the ``numpy`` source directory and install it
normally::

  <your_new_python> setup.py install >& install.log

The C-extensions of numpy have to be still added to the python
interpreter. Grep :file:`install.log`::

  grep 'Append to Setup' install.log

and add the correct lines to the :file:`Modules/Setup` in the python
source tree. Switch to the python source directory and run ``make``
and ``make install`` again to get interpreter with builtin numpy.

Final step is naturaly to compile GPAW. Only thing is to specify
``numpy``, ``expat`` and ``zlib`` libraries in :file:`customize.py`
then :ref:`compile GPAW <installationguide>` as usual. Here is an example
of :file:`customize.py`, modify according your own directory
structures:

.. literalinclude:: customize.py

Now you should be ready for massively parallel calculations, a sample
job file would be::

  #!/bin/csh
  #
  #PBS -N jobname
  #PBS -l walltime=24:00
  #PBS -l mppwidth=512

  cd /wrk/my_workdir
  # set the environment variables
  SETENV PYTHONPATH ...

  aprun -n 512 /path_to_gpaw_bin/gpaw-python input.py

In order to use a preinstalled version of gpaw one can give the
command ``module load gpaw`` which sets all the correct environment
variables (:envvar:`PYTHONPATH`, :envvar:`GPAW_SETUP_PATH`, ...)

