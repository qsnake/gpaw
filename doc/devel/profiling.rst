.. _profiling:

=========
Profiling
=========

profile
=======

Python has a ``profile`` module to help you find the places in the code where the time is spent.

Let's say you have a script
(`CH4.py <https://svn.fysik.dtu.dk/projects/gpaw/trunk/test/CH4.py>`_)
that you want to run through the profiler.  This is what you do:

>>> import profile
>>> profile.run('import CH4', 'prof')

This will run your script and generate a profile in the file ``prof``.  You can also generate the profile by inserting a line like this in your script::

  ...
  import profile
  profile.run('atoms.GetPotentialEnergy()', 'prof')
  ...

To analyse the results, you do this::

 >>> import pstats
 >>> pstats.Stats('prof').strip_dirs().sort_stats('time').print_stats(20)
 Tue Oct 14 19:08:54 2008    prof

         1093215 function calls (1091618 primitive calls) in 37.430 CPU seconds

   Ordered by: internal time
   List reduced from 1318 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    37074   10.310    0.000   10.310    0.000 :0(calculate_spinpaired)
     1659    4.780    0.003    4.780    0.003 :0(relax)
   167331    3.990    0.000    3.990    0.000 :0(dot)
     7559    3.440    0.000    3.440    0.000 :0(apply)
      370    2.730    0.007   17.090    0.046 xc_correction.py:130(calculate_energy_and_derivatives)
    37000    0.780    0.000    9.650    0.000 xc_functional.py:657(get_energy_and_potential_spinpaired)
    37074    0.720    0.000   12.990    0.000 xc_functional.py:346(calculate_spinpaired)
      ...
      ...

The list shows the 20 functions where the most time is spent.  Check the pstats_ documentation if you want to do more fancy things.

.. _pstats: http://docs.python.org/lib/module-profile.html


.. tip::

   Since the ``profile`` module does not time calls to C-code, it
   is a good idea to run the code in debug mode - this will wrap
   calls to C-code in Python functions::

     $ python script.py --debug

tau
===

`TAU Performance System <http://www.cs.uoregon.edu/research/tau/>`_
is a portable profiling and tracing toolkit for performance analysis
of parallel programs written in Fortran, C, C++, Java, Python.

Installation and configuration
------------------------------

TAU requires `Program Database Toolkit <http://www.cs.uoregon.edu/research/pdt/>`_ to produce profiling data. Follow the vendor installation instructions
for ``pdtoolkit`` or, on an RPM-based system
(El4/EL5 and FC8/FC9 are currently supported), build RPM following
`<https://wiki.fysik.dtu.dk/niflheim/Cluster_software_-_RPMS?action=show#pdtoolkit>`_. **Note**: If you want to use only TAU's
``paraprof`` and/or ``perfexplorer``
for analysing profile data made elsewhere - skip the installation of
``pdtoolkit``.

Follow the vendor installation instructions for ``tau`` or, on an RPM-based
system (El4/EL5 and FC8/FC9 are currently supported), build RPM following
`<https://wiki.fysik.dtu.dk/niflheim/Cluster_software_-_RPMS?action=show#tau>`_.
Then, configure the environment by running first ``perfdmf_configure``.
Choose the default answer for most of the questions, the only important
points are to provide a reasonable path to the database directory
(the database may grow to GB's), and ignore the password question::

 Please enter the path to the database directory.
 (/home/camp/dulak/.ParaProf/perfdmf):/scratch/dulak/perfdmf
 Please enter the database username.
 ():
 Store the database password in CLEAR TEXT in your configuration file? (y/n):y
 Please enter the database password:

Similarly, run ``perfexplorer_configure`` letting the default settings.

Generating profile data
-----------------------

You need to compile a special version of gpaw.

Simply include the following into ``customize.py`` and run ``python setup.py build_ext``::

 import tau
 tau_path = tau.__file__[0:tau.__file__.find('lib')]
 tau_make = tau_path+'lib/Makefile.tau-mpi-python-pdt'
 tau_options='-optShared '
 mpicompiler = 'tau_cc.sh -tau_options='+tau_options+' -tau_makefile='+tau_make
 mpilinker = mpicompiler
 compiler = mpicompiler

 extra_link_args += ['-Wl,-rpath='+tau_path+'lib/']

 To obtain the profiler data run the following ``wrapper.py``::

 import tau

 def OurMain():
     import CH4;

 tau.run('OurMain()')

, e.g., for two processes::

 mpirun -np 2 gpaw-python wrapper.py

This will generate `profile.?.?.?` files, convert
these files into a ppk (ParaProf Packed Profile) file with::

 paraprof --pack CH4.ppk

You should be able to quickly view the profiler data with::

 paraprof CH4.ppk

Analysing profile data
----------------------

Now, assuming you have an ppk (ParaProf Packed Profile) file ready,
run ``paraprof`` and choose the following from the menu
``Default -> add application -> add experiment -> add trial -> Type ParaProf Packed Profile``.

``paraprof`` allows you to investigate profiler data for a single run (trial).
Repeat the previous step (adding a trial) for parallel runs
with increasing number of processes, exit ``paraprof`` (derby database
format can be accessed by only one program at a time), and run
``perfexplorer`` to investigate the strong scaling of your application.
