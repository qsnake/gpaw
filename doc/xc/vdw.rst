.. section-numbering::

.. |H2| replace:: H\ :sub:`2`
.. |Gamma|  unicode:: U+00393 .. GREEK CAPITAL LETTER GAMMA
.. |angst|  unicode:: U+0212B .. ANGSTROM SIGN
.. |times|  unicode:: U+000D7 .. MULTIPLICATION SIGN
.. |simeq|  unicode:: U+02243 .. ASYMPTOTICALLY EQUAL TO



------------
Introduction
------------

The vdW-DF exchange correlation functional has been implemented following the scheme in reference [#vdW-DF]_


---------------------------------
Doing a van der Waals calculation
---------------------------------


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

============================  ==========  ===================  
keyword                       type        description  
============================  ==========  ===================  
``get_prl_plot``                  
``get_energy``                ``tupple``                  
``get_e_xc_LDA`` 
``get_e_xc_LDA_c``                                            
``get_e_x_LDA``                                       
``get_q0``                                 
``get_phitab_from_1darrays`` 
``get_c6_coarse`` 
============================  ==========  ===================  



.. [#vdW-DF]    M. Dion, H. Rydberg, E. Schroder, D.C. Langreth, and B. I. Lundqvist. 
                Van der Waals density functional for general geometries. 
                Physical Review Letters, 92 (24):246401. 2004
