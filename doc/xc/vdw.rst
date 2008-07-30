===
VDW
===


------------
Introduction
------------

The vdW-DF exchange correlation functional has been implemented following the scheme in reference [#vdW-DF]_
The vdW-DF exchange correlation functional implementation requires a table of phi, which can be downloaded here phi_. Download this an unpack it in a folder name e.g VDW and refer to it as  setenv VDW /home/user/Phi/. The provided interaction kernal phi has been calculated using the scripts which can be downloaded here: makephi_


---------------------------------
Doing a van der Waals calculation
---------------------------------

A vdW calculation is done in the following way. Assuming that we have a input .gpw file

>>> from ase import *
>>> from gpaw import *
>>> from ase.units import *
>>> calc = Calculator(input.gpw, txt=None)
>>> density = calc.get_all_electron_density()
>>> density = density*Bohr**3 
>>> vdw = VanDerWaals(density, calc.finegd,calc,'revPBE', ncoarsen=0)
>>> 
>>> totalenergy = calc.get_potential_energy()
>>> 
>>> corecorrected_nonlocal_correlation, nonlocal_correlation = vdw.get_energy(repeat=None, ncut=0.0005)
>>> 
>>> energy = totalenergy + corecorrected_nonlocal_correlation

The user should check thoroughly that the grid spacing and ncut is converged.

Parameters
-----------

===============  ==========  ===================  ===============================
keyword          type        default value        description
===============  ==========  ===================  ===============================
``ncut``          ``float``  ``0.0005``           Lower bound on density
``ncoarsen``      ``str``    ``0``                Coarsening of the density grid
``xcname``        ``str``    ``'revPBE'``         XC-functional
``gd``                                            Grid descriptor object
``density``                                       Density array 
``calc``                     ``None``             Calculator object
===============  ==========  ===================  ===============================



Methods
-------------

============================  ==========  ==========================  
keyword                       type        description  
============================  ==========  ========================== 
``get_kernel_plot``                        ``Returns plot of the kernel``
``get_energy``                ``tupple``   ``Returns the energy``               
``get_e_xc_LDA``                           ``LDA xc energy on grid``
``get_e_c_LDA``                            ``LDA c energy on grid``                   
``get_e_x_LDA``                            ``LDA x energy on grid``                   
``get_q0``                                 ``q0 on the grid``
``get_phitab_from_1darrays``               ``Reads phi from table``
``get_c6``                                 ``Calculates C6 coefficients``
============================  ==========  ==========================  



.. [#vdW-DF]    M. Dion, H. Rydberg, E. Schroder, D.C. Langreth, and B. I. Lundqvist. 
                Van der Waals density functional for general geometries. 
                Physical Review Letters, 92 (24):246401. 2004

.. _phi: ../_static/phi.dat

.. _makephi: ../_static/makephi.tar.gz
