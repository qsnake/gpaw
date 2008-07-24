.. _plot_wave_functions:

=======================
Plotting wave functions
=======================

-----------------------------
Creating a wave function file
-----------------------------

The following script will do a calculation for a CO
molecule and save the wave functions in a file (`CO.gpw`).

script_

.. _script: inline:CO.py

---------------------------------
Creating wave function cube files
---------------------------------

You can get seperate cube files (the format used by Gaussian) for each wavefunction with the script:

cubescript_

.. _cubescript: inline:CO2cube.py

The script produced the files CO_0.cube .. CO_5.cube, which might be viewed using for example `gOpenMol <http://www.csc.fi/gopenmol/>`_ or `VMD <http://www.ks.uiuc.edu/Research/vmd/>`_. 


Creating cube to plt files (gOpenMol)
-----------------------------------------

The cube files can be transformed to plt format using the program g94cub2pl from the gOpenMol utilities.

--------------------------------
Creating wave function plt files
--------------------------------

One can write out the wave functions in the very compact (binary) `gOpenMol <http://www.csc.fi/gopenmol/>`_ plt format directly:

pltscript_

.. _pltscript: inline:CO2plt.py
