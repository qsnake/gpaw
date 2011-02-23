.. _SnowLeopard:

========================
Snow Leopard
========================

Follow the instructions here for installing MacPorts and all supporting packages:
`<http://wiki.alcf.anl.gov/naromero>`_

This build of GPAW will use MPICH2 1.2, Python 2.7.x and NumPy 1.5.x.

Use the following customize file :svn:`customize_snowleopard_macports.py`

There have been some problems with vecLib library which is linked automatically. Disabling this is straigtforward.  Here is a diff :svn:`config.disable_vecLib.py.diff`

Then build::

  python setup.py build_ext --customize=customize_snowleopard_macports.py



