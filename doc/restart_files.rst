.. _restart_files:

=============
Restart files
=============

Writing restart files
=====================

Use ``calc.write('xyz.gpw')`` or ``calc.write('xyz.gpw', mode='all')``
to include also the wave functions.

You can register an automatic call to the ``write`` method, every
``n``'th iteration like this::

  calc.attach(calc.write, n, 'xyz.gpw')

or::

  calc.attach(calc.write, n, 'xyz.gpw', mode='all')


Reading restart files
=====================

The calculation can be read from file like this::

  calc = GPAW('xyz.gpw')

or this::

  atoms, calc = restart('xyz.gpw')



GPAW's native file format
=========================

The purpose of this format is to reduce the GPAW codes dependencies on
all sorts of funny io-libraries.  The ``.gpw`` files are simple
tar-files containing binary files and an XML file::

  $ tar -tf xyz.gpw
  AtomicNumbers
  CartesianPositions
  ...
  ...
  OccupationNumbers
  PseudoElectronDensity
  PseudoPotential
  info.xml

The file ``info.xml`` has information about array sizes, types,
endianness, parameters, and more.  Try::

  $ tar -f xyz.gpw -xO info.xml

To remove the wave functions from a ``.gpw`` file, do this::

  $ tar -f xyz.gpw --delete PseudoWaveFunctions



Writing to separate files
=========================

In case of large files it is a good idea to write the wave functions
into seperate files, this can be done in the following way::

  calc.attach(calc.write, n, 'xyz.gpw', mode='gpw:wfs_tmp/psit_Gs%dk%dn%d')

which uses the 'wfs_tmp/psit_Gs%dk%dn%d.gpw' % (s,k,n), where s=spin,
k=k point and n=band number to write out the wave functions.  The
directory 'wfs_tmp' is created automatically if needed. Note: The
shorthand mode='gpw' has the same meaning as
mode='gpw:psit_Gs%dk%dn%d'.

In case that you have written the wave functions to separate files, you can read them via::

  calc = GPAW('xyz.gpw')
  calc.read_wave_functions(mode='gpw:wfs_tmp/psit_Gs%dk%dn%d')

where the syntax for mode is the same as for writing the wave functions.
