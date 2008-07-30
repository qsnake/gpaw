.. _Li: http://wiki.fysik.dtu.dk/stuff/test-atom/Li.PBE/page.html
.. _C: http://wiki.fysik.dtu.dk/stuff/test-atom/C.PBE/page.html
.. _Na: http://wiki.fysik.dtu.dk/stuff/test-atom/Na.PBE/page.html
.. _Al: http://wiki.fysik.dtu.dk/stuff/test-atom/Al.PBE/page.html
.. _Si: http://wiki.fysik.dtu.dk/stuff/test-atom/Si.PBE/page.html
.. _K: http://wiki.fysik.dtu.dk/stuff/test-atom/K.PBE/page.html
.. _Ca: http://wiki.fysik.dtu.dk/stuff/test-atom/Ca.PBE/page.html
.. _V: http://wiki.fysik.dtu.dk/stuff/test-atom/V.PBE/page.html
.. _Fe: http://wiki.fysik.dtu.dk/stuff/test-atom/Fe.PBE/page.html
.. _Ni: http://wiki.fysik.dtu.dk/stuff/test-atom/Ni.PBE/page.html
.. _Cu: http://wiki.fysik.dtu.dk/stuff/test-atom/Cu.PBE/page.html
.. _Rb: http://wiki.fysik.dtu.dk/stuff/test-atom/Rb.PBE/page.html
.. _Nb: http://wiki.fysik.dtu.dk/stuff/test-atom/Nb.PBE/page.html
.. _Mo: http://wiki.fysik.dtu.dk/stuff/test-atom/Mo.PBE/page.html
.. _Rh: http://wiki.fysik.dtu.dk/stuff/test-atom/Rh.PBE/page.html
.. _Pd: http://wiki.fysik.dtu.dk/stuff/test-atom/Pd.PBE/page.html
.. _Ag: http://wiki.fysik.dtu.dk/stuff/test-atom/Ag.PBE/page.html
.. _Pt: http://wiki.fysik.dtu.dk/stuff/test-atom/Pt.PBE/page.html
.. _Au: http://wiki.fysik.dtu.dk/stuff/test-atom/Au.PBE/page.html
.. _Ba: http://wiki.fysik.dtu.dk/stuff/test-atom/Ba.PBE/page.html

.. contents::

A setup is to the PAW method what a pseudo-potential is to the
pseudo-potential method.  All available setups are contained in this
tar-file: gpaw-setups-0.4.2039.tar.gz_.  Install them as described in the
:ref:`installationguide`.  The setups are stored as compressed pawxml_
files.

See also `NIST Atomic Reference Data`_ and `Computational Chemistry Comparison and Benchmark DataBase`_.  Link to `Dacapo pseudo potentials`_.

.. _NIST Atomic Reference Data: http://physics.nist.gov/PhysRefData/DFTdata/Tables/ptable.html
.. _Computational Chemistry Comparison and Benchmark DataBase: http://srdata.nist.gov/cccbdb/
.. _gpaw-setups-0.4.2039.tar.gz: http://wiki.fysik.dtu.dk/stuff/gpaw-setups-0.4.2039.tar.gz
.. _pawxml: http://wiki.fysik.dtu.dk/stuff/pawxml/pawxml.xhtml
.. _Dacapo pseudo potentials: https://wiki.fysik.dtu.dk/dacapo/Pseudopotential_Library

Summary of bulk tests: svn revision 1389
========================================

For details - see the periodic table above.

Figure shows relative percentage errors
of the equilibrium lattice constant calculated with GPAW using PBE, compared to very accurate, state-of-the-art, PBE calculations.

XXX .. figure:: http://wiki.fysik.dtu.dk/stuff/test-atom-1389/lattice_constant_percentage_error_PBE.png

Figure shows relative percentage errors
of the bulk modulus calculated with GPAW using PBE, compared to very accurate, state-of-the-art, PBE calculations.

XXX .. figure:: http://wiki.fysik.dtu.dk/stuff/test-atom-1389/bulk_modulus_percentage_error_PBE.png

Generating your own setups
==========================

You can generate atom data for all atoms using the ``gpaw-setup``
program in the ``tools`` directory.  For a He LDA-setup, you do this::

  $ gpaw-setup -f LDA He

Type ``gpaw-setup --help`` for more help.
