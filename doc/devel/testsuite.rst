.. _testsuite:

==========
Test suite
==========

Test results from last night
============================

::


   pbe-pw91.py ... (0.014s) ok
   xcfunc.py ... (0.014s) ok
   xc.py ... (0.014s) ok
   gp2.py ... (0.017s) ok
   lapack.py ... (0.014s) ok
   gradient.py ... (0.014s) ok
   lf.py ... (0.019s) ok
   non-periodic.py ... (0.019s) ok
   lxc_xc.py ... (0.022s) ok
   transformations.py ... (0.027s) ok
   Gauss.py ... (0.023s) ok
   denom_int.py ... (0.024s) ok
   setups.py ... (0.012s) ok
   poisson.py ... (0.056s) ok
   cluster.py ... (0.028s) ok
   integral4.py ... (0.065s) ok
   cg2.py ... (0.018s) ok
   XC2.py ... (0.101s) ok
   d2Excdn2.py ... (0.017s) ok
   XC2Spin.py ... (0.195s) ok
   multipoletest.py ... (0.550s) ok
   coulomb.py ... (1.203s) ok
   ase3k.py ... (1.434s) ok
   mixer.py ... (1.662s) ok
   proton.py ... (1.603s) ok
   timing.py ... (1.703s) ok
   restart.py ... (1.773s) ok
   gauss_func.py ... (3.027s) ok
   xcatom.py ... (3.101s) ok
   wfs_io.py ... (2.937s) ok
   ylexpand.py ... (3.955s) ok
   nonselfconsistentLDA.py ... (4.560s) ok
   bee1.py ... (5.294s) ok
   gga-atom.py ... (6.092s) ok
   revPBE.py ... (6.817s) ok
   nonselfconsistent.py ... (7.577s) ok
   bulk.py ... (8.908s) ok
   spinpol.py ... (7.666s) ok
   refine.py ... (7.602s) ok
   bulk-lcao.py ... (8.292s) ok
   stdout.py ... (8.092s) ok
   restart2.py ... (8.174s) ok
   hydrogen.py ... (8.816s) ok
   H-force.py ... (9.799s) ok
   plt.py ... (9.545s) ok
   h2o-xas.py ... (12.277s) ok
   degeneracy.py ... (13.085s) ok
   davidson.py ... (15.575s) ok
   cg.py ... (16.930s) ok
   ldos.py ... (21.090s) ok
   h2o-xas-recursion.py ... (27.060s) ok
   atomize.py ... (22.122s) ok
   wannier-ethylene.py ... (23.200s) ok
   lrtddft.py ... (23.641s) ok
   CH4.py ... (28.601s) ok
   gllb2.py ... (25.798s) ok
   apmb.py ... (34.154s) ok
   relax.py ... (35.538s) ok
   fixmom.py ... (47.353s) ok
   si-xas.py ... (52.666s) ok
   revPBE_Li.py ... (40.447s) ok
   lxc_xcatom.py ... (44.545s) ok
   exx_coarse.py ... (43.923s) ok
   2Al.py ... (56.307s) ok
   8Si.py ... (70.034s) ok
   dscf_test.py ... (59.826s) ok
   lcao-h2o.py ... (47.879s) ok
   IP-oxygen.py ... (63.010s) ok
   generatesetups.py ... (68.525s) ok
   aedensity.py ... (80.255s) ok
   Cu.py ... (103.048s) ok
   exx.py ... (97.603s) ok
   H2Al110.py ... (221.729s) ok
   ltt.py ... (192.789s) ok
   ae-calculation.py ... (203.873s) ok
   lb.py ... (33.357s) ok
   
   ----------------------------------------------------------------------
   Ran 76 tests in 1957.195s
   
   OK


Coverage
========


Test-suite does not cover all of the code!
Number of missing lines:

 =================================  =====
 Module                             Lines
 =================================  =====
 `xc_correction`_                   533
 `xas`_                             191
 `atom.generator`_                  190
 `xc_functional`_                   153
 `atom.all_electron`_               152
 `utilities.dos`_                   114
 `dscf`_                            106
 `lcao.overlap`_                    104
 `nucleus`_                         104
 `utilities.tools`_                 101
 `localized_functions`_             100
 `utilities.memory`_                100
 `coulomb`_                         99
 `setup`_                           99
 `mpi.__init__`_                    91
 `exx`_                             88
 `lrtddft.omega_matrix`_            87
 `lrtddft.apmb`_                    85
 `pawextra`_                        81
 `paw`_                             80
 `basis_data`_                      79
 `aseinterface`_                    71
 `grid_descriptor`_                 62
 `utilities.__init__`_              61
 `wannier`_                         61
 `utilities.lapack`_                58
 `__init__`_                        52
 `atom.configurations`_             52
 `sphere`_                          52
 `domain`_                          51
 `io.__init__`_                     51
 `lrtddft.__init__`_                51
 `eigensolvers.rmm_diis2`_          45
 `io.plt`_                          41
 `lcao.hamiltonian`_                41
 `libxc`_                           40
 `poisson`_                         40
 `utilities.vector`_                37
 `utilities.timing`_                36
 `preconditioner`_                  34
 `density`_                         31
 `utilities.blas`_                  28
 `Function1D`_                      26
 `atom.filter`_                     25
 `gllb.gllb`_                       24
 `gaunt`_                           23
 `io.xyz`_                          23
 `output`_                          23
 `transformers`_                    21
 `occupations`_                     20
 `gllb.gllb1d`_                     19
 `operators`_                       19
 `cluster`_                         18
 `eigensolvers.rmm_diis`_           17
 `spherical_harmonics`_             17
 `lrtddft.kssingle`_                13
 `gauss`_                           12
 `kpoint`_                          12
 `rotation`_                        11
 `symmetry`_                        11
 `analyse.expandyl`_                10
 `mixer`_                           10
 `io.tar`_                          9
 `lrtddft.excitation`_              9
 `pair_potential`_                  9
 `hamiltonian`_                     8
 `setup_data`_                      7
 `eigensolvers.eigensolver`_        6
 `gllb.nonlocalfunctionalfactory`_  5
 `overlap`_                         5
 `eigensolvers.cg`_                 4
 `eigensolvers.davidson`_           3
 `utilities.complex`_               3
 `brillouin`_                       2
 `pair_density`_                    2
 `lcao.eigensolver`_                1
 `utilities.cg`_                    1
 `utilities.gauss`_                 1
 =================================  =====

.. _xc_correction: http://wiki.fysik.dtu.dk/stuff/xc_correction.cover
.. _xas: http://wiki.fysik.dtu.dk/stuff/xas.cover
.. _atom.generator: http://wiki.fysik.dtu.dk/stuff/atom.generator.cover
.. _xc_functional: http://wiki.fysik.dtu.dk/stuff/xc_functional.cover
.. _atom.all_electron: http://wiki.fysik.dtu.dk/stuff/atom.all_electron.cover
.. _utilities.dos: http://wiki.fysik.dtu.dk/stuff/utilities.dos.cover
.. _dscf: http://wiki.fysik.dtu.dk/stuff/dscf.cover
.. _lcao.overlap: http://wiki.fysik.dtu.dk/stuff/lcao.overlap.cover
.. _nucleus: http://wiki.fysik.dtu.dk/stuff/nucleus.cover
.. _utilities.tools: http://wiki.fysik.dtu.dk/stuff/utilities.tools.cover
.. _localized_functions: http://wiki.fysik.dtu.dk/stuff/localized_functions.cover
.. _utilities.memory: http://wiki.fysik.dtu.dk/stuff/utilities.memory.cover
.. _coulomb: http://wiki.fysik.dtu.dk/stuff/coulomb.cover
.. _setup: http://wiki.fysik.dtu.dk/stuff/setup.cover
.. _mpi.__init__: http://wiki.fysik.dtu.dk/stuff/mpi.__init__.cover
.. _exx: http://wiki.fysik.dtu.dk/stuff/exx.cover
.. _lrtddft.omega_matrix: http://wiki.fysik.dtu.dk/stuff/lrtddft.omega_matrix.cover
.. _lrtddft.apmb: http://wiki.fysik.dtu.dk/stuff/lrtddft.apmb.cover
.. _pawextra: http://wiki.fysik.dtu.dk/stuff/pawextra.cover
.. _paw: http://wiki.fysik.dtu.dk/stuff/paw.cover
.. _basis_data: http://wiki.fysik.dtu.dk/stuff/basis_data.cover
.. _aseinterface: http://wiki.fysik.dtu.dk/stuff/aseinterface.cover
.. _grid_descriptor: http://wiki.fysik.dtu.dk/stuff/grid_descriptor.cover
.. _utilities.__init__: http://wiki.fysik.dtu.dk/stuff/utilities.__init__.cover
.. _wannier: http://wiki.fysik.dtu.dk/stuff/wannier.cover
.. _utilities.lapack: http://wiki.fysik.dtu.dk/stuff/utilities.lapack.cover
.. ___init__: http://wiki.fysik.dtu.dk/stuff/__init__.cover
.. _atom.configurations: http://wiki.fysik.dtu.dk/stuff/atom.configurations.cover
.. _sphere: http://wiki.fysik.dtu.dk/stuff/sphere.cover
.. _domain: http://wiki.fysik.dtu.dk/stuff/domain.cover
.. _io.__init__: http://wiki.fysik.dtu.dk/stuff/io.__init__.cover
.. _lrtddft.__init__: http://wiki.fysik.dtu.dk/stuff/lrtddft.__init__.cover
.. _eigensolvers.rmm_diis2: http://wiki.fysik.dtu.dk/stuff/eigensolvers.rmm_diis2.cover
.. _io.plt: http://wiki.fysik.dtu.dk/stuff/io.plt.cover
.. _lcao.hamiltonian: http://wiki.fysik.dtu.dk/stuff/lcao.hamiltonian.cover
.. _libxc: http://wiki.fysik.dtu.dk/stuff/libxc.cover
.. _poisson: http://wiki.fysik.dtu.dk/stuff/poisson.cover
.. _utilities.vector: http://wiki.fysik.dtu.dk/stuff/utilities.vector.cover
.. _utilities.timing: http://wiki.fysik.dtu.dk/stuff/utilities.timing.cover
.. _preconditioner: http://wiki.fysik.dtu.dk/stuff/preconditioner.cover
.. _density: http://wiki.fysik.dtu.dk/stuff/density.cover
.. _utilities.blas: http://wiki.fysik.dtu.dk/stuff/utilities.blas.cover
.. _Function1D: http://wiki.fysik.dtu.dk/stuff/Function1D.cover
.. _atom.filter: http://wiki.fysik.dtu.dk/stuff/atom.filter.cover
.. _gllb.gllb: http://wiki.fysik.dtu.dk/stuff/gllb.gllb.cover
.. _gaunt: http://wiki.fysik.dtu.dk/stuff/gaunt.cover
.. _io.xyz: http://wiki.fysik.dtu.dk/stuff/io.xyz.cover
.. _output: http://wiki.fysik.dtu.dk/stuff/output.cover
.. _transformers: http://wiki.fysik.dtu.dk/stuff/transformers.cover
.. _occupations: http://wiki.fysik.dtu.dk/stuff/occupations.cover
.. _gllb.gllb1d: http://wiki.fysik.dtu.dk/stuff/gllb.gllb1d.cover
.. _operators: http://wiki.fysik.dtu.dk/stuff/operators.cover
.. _cluster: http://wiki.fysik.dtu.dk/stuff/cluster.cover
.. _eigensolvers.rmm_diis: http://wiki.fysik.dtu.dk/stuff/eigensolvers.rmm_diis.cover
.. _spherical_harmonics: http://wiki.fysik.dtu.dk/stuff/spherical_harmonics.cover
.. _lrtddft.kssingle: http://wiki.fysik.dtu.dk/stuff/lrtddft.kssingle.cover
.. _gauss: http://wiki.fysik.dtu.dk/stuff/gauss.cover
.. _kpoint: http://wiki.fysik.dtu.dk/stuff/kpoint.cover
.. _rotation: http://wiki.fysik.dtu.dk/stuff/rotation.cover
.. _symmetry: http://wiki.fysik.dtu.dk/stuff/symmetry.cover
.. _analyse.expandyl: http://wiki.fysik.dtu.dk/stuff/analyse.expandyl.cover
.. _mixer: http://wiki.fysik.dtu.dk/stuff/mixer.cover
.. _io.tar: http://wiki.fysik.dtu.dk/stuff/io.tar.cover
.. _lrtddft.excitation: http://wiki.fysik.dtu.dk/stuff/lrtddft.excitation.cover
.. _pair_potential: http://wiki.fysik.dtu.dk/stuff/pair_potential.cover
.. _hamiltonian: http://wiki.fysik.dtu.dk/stuff/hamiltonian.cover
.. _setup_data: http://wiki.fysik.dtu.dk/stuff/setup_data.cover
.. _eigensolvers.eigensolver: http://wiki.fysik.dtu.dk/stuff/eigensolvers.eigensolver.cover
.. _gllb.nonlocalfunctionalfactory: http://wiki.fysik.dtu.dk/stuff/gllb.nonlocalfunctionalfactory.cover
.. _overlap: http://wiki.fysik.dtu.dk/stuff/overlap.cover
.. _eigensolvers.cg: http://wiki.fysik.dtu.dk/stuff/eigensolvers.cg.cover
.. _eigensolvers.davidson: http://wiki.fysik.dtu.dk/stuff/eigensolvers.davidson.cover
.. _utilities.complex: http://wiki.fysik.dtu.dk/stuff/utilities.complex.cover
.. _brillouin: http://wiki.fysik.dtu.dk/stuff/brillouin.cover
.. _pair_density: http://wiki.fysik.dtu.dk/stuff/pair_density.cover
.. _lcao.eigensolver: http://wiki.fysik.dtu.dk/stuff/lcao.eigensolver.cover
.. _utilities.cg: http://wiki.fysik.dtu.dk/stuff/utilities.cg.cover
.. _utilities.gauss: http://wiki.fysik.dtu.dk/stuff/utilities.gauss.cover

(coverage-test is performed Sunday nights only)
