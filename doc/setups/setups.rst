.. _setups:

=================
Atomic PAW Setups
=================

A setup is to the PAW method what a pseudo-potential is to the
pseudo-potential method.  All available setups are contained in this
tar-file: gpaw-setups-0.6.6300.tar.gz_.  Install them as described in the
:ref:`installationguide`.  The setups are stored as compressed pawxml_
files.


Periodic table
==============

=== === === === === === === === === === === === === === === === === ===
H_                                                                  He_
Li_ Be_                                         B_  C_  N_  O_  F_  Ne_ 
Na_ Mg_                                         Al_ Si_ P_  S_  Cl_ Ar_  
K_  Ca_ Sc_ Ti_ V_  Cr_ Mn_ Fe_ Co_ Ni_ Cu_ Zn_ Ga_ Ge_ As_ Se_ Br_ Kr_
Rb_ Sr_ Y   Zr_ Nb_ Mo_ Tc  Ru_ Rh_ Pd_ Ag_ Cd_ In  Sn_ Sb  Te  I_  Xe 
Cs_ Ba_ La_ Hf  Ta_ W_  Re  Os_ Ir_ Pt_ Au_ Hg  Tl  Pb_ Bi_ Po  At  Rn 
=== === === === === === === === === === === === === === === === === ===

See also `NIST Atomic Reference Data`_, `Computational Chemistry
Comparison and Benchmark DataBase`_, `Dacapo pseudo potentials`_, and
`Vasp pseudo potentials`_.

.. _gpaw-setups-0.6.6300.tar.gz: http://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-0.6.6300.tar.gz
.. _pawxml: http://wiki.fysik.dtu.dk/gpaw-files/pawxml/pawxml.xhtml
.. _NIST Atomic Reference Data: http://physics.nist.gov/PhysRefData/DFTdata/Tables/ptable.html
.. _Computational Chemistry Comparison and Benchmark DataBase: http://srdata.nist.gov/cccbdb/
.. _Dacapo pseudo potentials: https://wiki.fysik.dtu.dk/dacapo/Pseudopotential_Library
.. _Vasp pseudo potentials: http://cms.mpi.univie.ac.at/vasp/vasp/Pseudopotentials_supplied_with_VASP_package.html


.. toctree::
   :maxdepth: 2

   molecule_tests
   bulk_tests
   generation_of_setups


.. from gpaw.atom.generator import parameters
   for s in parameters:
       print '.. %3s: %2s.html' % ('_' + s, s)

.. _Ni: Ni.html
.. _Pd: Pd.html
.. _Pt: Pt.html
.. _Ru: Ru.html
.. _Na: Na.html
.. _Nb: Nb.html
.. _Mg: Mg.html
.. _Li: Li.html
.. _Pb: Pb.html
.. _Rb: Rb.html
.. _Ti: Ti.html
.. _Te: Te.html
.. _Rh: Rh.html
.. _Ta: Ta.html
.. _Be: Be.html
.. _Ba: Ba.html
.. _Bi: Bi.html
.. _La: La.html
.. _Si: Si.html
.. _As: As.html
.. _Fe: Fe.html
.. _Br: Br.html
.. _Sr: Sr.html
.. _Mo: Mo.html
.. _He: He.html
..  _C:  C.html
..  _B:  B.html
..  _F:  F.html
..  _I:  I.html
..  _H:  H.html
..  _K:  K.html
.. _Mn: Mn.html
..  _O:  O.html
.. _Ne: Ne.html
..  _P:  P.html
..  _S:  S.html
.. _Kr: Kr.html
..  _W:  W.html
..  _V:  V.html
.. _Sc: Sc.html
..  _N:  N.html
.. _Os: Os.html
.. _Se: Se.html
.. _Zn: Zn.html
.. _Co: Co.html
.. _Ag: Ag.html
.. _Cl: Cl.html
.. _Ca: Ca.html
.. _Ir: Ir.html
.. _Al: Al.html
.. _Cd: Cd.html
.. _Ge: Ge.html
.. _Ar: Ar.html
.. _Au: Au.html
.. _Zr: Zr.html
.. _Ga: Ga.html
.. _In: In.html
.. _Cs: Cs.html
.. _Cr: Cr.html
.. _Cu: Cu.html
.. _Sn: Sn.html

.. toctree::
   :glob:
   :maxdepth: 1

   [A-Z]*
