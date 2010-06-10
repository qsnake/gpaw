.. _literature:

----------
Literature
----------


Links to guides and manual pages
--------------------------------

* The GPAW calculator :ref:`manual`

* The :ref:`devel` pages

* The :ref:`guide for developers <developersguide>`

* The code :ref:`overview`

* The :ref:`features_and_algorithms` used in the code


.. _special_topics:

Specialized information
-----------------------

Here is a list of specific advanced topics and functionalities of the
GPAW calculator:

.. toctree::
   :maxdepth: 2
   
   toc-special


.. _literature_reports_presentations_and_theses:

Reports, presentations, and theses using gpaw
---------------------------------------------

* A short note on the basics of PAW: `paw note`_

* A master thesis on the inclusion of non-local exact exchange in the
  PAW formalism, and the implementation in gpaw: `exact exchange`_

* A master thesis on the inclusion of a localized basis in the PAW
  formalism, plus implementation and test results in GPAW: `lcao`_

* A master thesis on the inclusion of localized basis sets in the PAW
  formalism, focusing on basis set generation and force calculations:
  `localized basis sets`_

* A course report on a project involving the optimization of the
  setups (equivalent of pseudopotentials) in gpaw: `setup
  optimization`_

* Slides from a talk about PAW: `introduction to PAW slides`_

* Slides from a talk about GPAW development: `gpaw for developers`_

* Slides from a mini symposium during early development stage: `early gpaw`_

.. _paw note: ../paw_note.pdf
.. _exact exchange: ../_static/rostgaard_master.pdf
.. _lcao: ../_static/marco_master.pdf
.. _localized basis sets: ../_static/askhl_master.pdf
.. _setup optimization: ../_static/askhl_10302_report.pdf
.. _introduction to PAW slides: ../_static/mortensen_paw.pdf
.. _gpaw for developers: ../_static/mortensen_gpaw-dev.pdf
.. _early gpaw: ../_static/mortensen_mini2003talk.pdf


.. _paw_papers:

Articles on the PAW formalism
-----------------------------

The original article introducing the PAW formalism:
   | P. E. Blöchl
   | `Projector augmented-wave method`__
   | Physical Review B, Vol. **50**, 17953, 1994

   __ http://dx.doi.org/10.1103/PhysRevB.50.17953

A different formulation of PAW by Kresse and Joubert designed to make the transistion from USPP to PAW easy.
  | G. Kresse and D. Joubert
  | `From ultrasoft pseudopotentials to the projector augmented-wave method`__
  | Physical Review B, Vol. **59**, 1758, 1999

  __ http://dx.doi.org/10.1103/PhysRevB.59.1758

A second, more pedagogical, article on PAW by Blöchl and co-workers.
  | P. E. Blöchl, C. J. Först, and J. Schimpl
  | `Projector Augmented Wave Method: ab-initio molecular dynamics with full wave functions`__
  | Bulletin of Materials Science, Vol. **26**, 33, 2003

  __ http://www.ias.ac.in/matersci/



.. _gpaw_publications:

Publications using the gpaw code
--------------------------------

.. image:: publications.png
   :width: 650

.. If the first author is A. Einstein, then remember to use
   \A. Einstein so that we don't start an enumerated list (A, B, C,
   ...).

   The date should be the publication date


1) The first article introducing the gpaw project:
   
   \J. J. Mortensen, L. B. Hansen, and K. W. Jacobsen

   `Real-space grid implementation of the projector augmented wave method`__
   
   Physical Review B, Vol. **71**, 035109 (2005)

   __ http://dx.doi.org/10.1103/PhysRevB.71.035109

   .. 21 January 2005

#) A description of a statistical approach to the exchange-correlation
   energy in DFT:

   \J. J. Mortensen, K. Kaasbjerg, S. L. Frederiksen, J. K. Nørskov,
   J. P. Sethna, and K. W. Jacobsen

   `Bayesian Error Estimation in Density-Functional Theory`__
  
   Physical Review Letters, Vol. **95**, 216401 (2005)

   __ http://dx.doi.org/10.1103/PhysRevLett.95.216401

   .. 15 November 2005


#) First article related to ligand protected gold clusters:

   \J. Akola, M. Walter, R. L. Whetten, H. Häkkinen, and H. Grönbeck

   `On the structure of Thiolate-Protected Au25`__
  
   Journal of the American Chemical Society, Vol. **130**, 3756-3757 (2008)

   __ http://dx.doi.org/10.1021/ja800594p

   .. 6 March 2008


#) The article describing the time-dependent DFT implementations in
   gpaw:

   \M. Walter, H. Häkkinen, L. Lehtovaara, M. Puska, J. Enkovaara,
   C. Rostgaard, and J. J. Mortensen

   `Time-dependent density-functional theory in the projector
   augmented-wave method`__

   Journal of Chemical Physics, Vol. **128**, 244101 (2008)

   __ http://dx.doi.org/10.1063/1.2943138

   .. 23 June 2008


#) Second article related to ligand protected gold clusters:
   
   \M. Walter, J. Akola, O. Lopez-Acevedo, P.D. Jadzinsky, G. Calero,
   C.J. Ackerson, R.L. Whetten, H. Grönbeck, and H. Häkkinen

   `A unified view of ligand-protected gold clusters as superatom complexes`__
   
   Proceedings of the National Academy of Sciences, Vol. **105**,
   9157-9162 (2008) 
 
   __ http://www.pnas.org/cgi/content/abstract/0801001105v1

   .. 1 July 2008

#) Description of the delta SCF method implemented in GPAW for
   determination of excited-state energy surfaces:

   Jeppe Gavnholt, Thomas Olsen, Mads Engelund, and Jakob Schiotz

   `Delta self-consistent field method to obtain potential energy
   surfaces of excited molecules on surfaces`__

   Physical Review B, Vol. **78**, 075441 (2008)

   __ http://dx.doi.org/10.1103/PhysRevB.78.075441

   .. 27 August 2008


#) GPAW applied to the study of graphene edges:

   Pekka Koskinen, Sami Malola, and Hannu Häkkinen

   `Self-passivating edge reconstructions of graphene`__

   Physical Review Letters, Vol. **101**, 115502 (2008)

   __ http://dx.doi.org/10.1103/PhysRevLett.101.115502

   .. 10 September 2008


#) Application of delta SCF method, for making predictions on
   hot-electron assisted chemistry:

   Thomas Olsen, Jeppe Gavnholt, and Jakob Schiotz
  
   `Hot-electron-mediated desorption rates calculated from
   excited-state potential energy surfaces`__

   Physical Review B, Vol. **79**, 035403 (2009)

   __ http://dx.doi.org/10.1103/PhysRevB.79.035403

   .. 6 January 2009 


#) A DFT study of a large thiolate protected gold cluster with 144 Au
   atoms and 60 thiolates:

   Olga Lopez-Acevedo, Jaakko Akola, Robert L. Whetten, Henrik
   Grönbeck, and Hannu Häkkinen

   `Structure and Bonding in the Ubiquitous Icosahedral Metallic Gold
   Cluster Au144(SR)60`__

   The Journal of Physical Chemistry C, in press (2009)

   __ http://dx.doi.org/10.1021/jp8115098

   .. 16 January 2009


#) Sami Malola, Hannu Häkkinen and Pekka Koskinen

   `Gold in graphene: In-plane adsorption and diffusion`__

   Appl. Phys. Lett. **94**, 043106 (2009)

   __ http://dx.doi.org/10.1063/1.3075216 
   
   .. 26 January 2009

#) A study of gold cluster stability on a rutile TiO\ :sub:`2`
   surface, and CO adsorbed on such clusters:

   Georg K. H. Madsen and Bjørk Hammer

   `Effect of subsurface Ti-interstitials on the bonding of small gold
   clusters on rutile TiO_2 (110)`__

   Journal of Chemical Physics, **130**, 044704 (2009)

   __ http://dx.doi.org/10.1063/1.3055419 

   .. 26 January 2009

#) Interpretation of STM images with DFT calculations:

   \F. Yin, J. Akola, P. Koskinen, M. Manninen, and R. E. Palmer

   `Bright Beaches of Nanoscale Potassium Islands on Graphite in STM
   Imaging`__
  
   Physical Review Letters, Vol. **102**, 106102 (2009)

   __ http://dx.doi.org/10.1103/PhysRevLett.102.106102

   .. 13 March 2009

#) Poul Georg Moses, Jens J. Mortensen, Bengt I. Lundqvist and Jens
   K. Nørskov

   `Density functional study of the adsorption and van der Waals
   binding of aromatic and conjugated compounds on the basal plane of
   MoS2`__

   \J. Chem. Phys. **130**, 104709 (2009)

   __ http://dx.doi.org/10.1063/1.3086040 

   .. 14 March 2009

#) Jeppe Gavnholt, Angel Rubio, Thomas Olsen, Kristian S. Thygesen and
   Jakob Schiøtz

   `Hot-electron-assisted femtochemistry at surfaces: A time-dependent
   density functional theory approach`__

   Phys. Rev. B **79**, 195405 (2009)

   __ http://dx.doi.org/10.1103/PhysRevB.79.195405

   .. 6 May 2009  

#) X. Lin, N. Nilius, H.-J. Freund, M. Walter, P. Frondelius,
   K. Honkala, and H. Häkkinen

   `Quantum well states in two-dimensional gold clusters on MgO thin films`__  

   Physical Review Letters, Vol. **102**, 206801 (2009)

   __ http://dx.doi.org/10.1103/PhysRevLett.102.206801

   .. 5 June 2009

#) Katarzyna A. Kacprzak, Lauri Lehtovaara, Jaakko Akola, Olga
   Lopez-Acevedo and Hannu Häkkinen

   `A density functional investigation of thiolate-protected bimetal
   PdAu24(SR)(18)(z) clusters: doping the superatom complex`__

   Physical Chemistry Chemical Physics **11**, 7123-7129 (2009)

   __ http://dx.doi.org/10.1039/b904491d

   .. 11 June 2009

#) The effect of frustrated rotations in HEFatS is calculated using
   the delta SCF method

   Thomas Olsen

   `Inelastic scattering in a local polaron model with quadratic
   coupling to bosons`__

   Physical Review B, Vol. **79**, 235414 (2009)

   __ http://dx.doi.org/10.1103/PhysRevB.79.235414

   .. 12 June 2009 

#) \A. K. Kelkkanen, B. I. Lundqvist, J. K. Nørskov

   `Density functional for van der Waals forces accounts for hydrogen
   bond in benchmark set of water hexamers`__

   Journal of Chemical Physics **131**, 046102 (2009)

   __ http://dx.doi.org/10.1063/1.3193462

   .. 29 July 2009

#) Olga Lopez-Acevedo, Jyri Rintala, Suvi Virtanen, Cristina Femoni,
   Cristina Tiozzo, Henrik Grönbeck, Mika Pettersson and Hannu Häkkinen

   `Characterization of Iron-Carbonyl-Protected Gold Clusters`__

   Journal of the American Chemical Society, Vol. **131**, 12573 (2009)

   __ http://dx.doi.org/10.1021/ja905182g

   .. 14 August 2009

#) Michael Walter and Michael Moseler

   `Ligand protected gold alloy clusters: doping the superatom`__

   Journal of Physical Chemistry C **113**, 15834 (2009)

   __ http://pubs.acs.org/doi/abs/10.1021/jp9023298

   .. 17 August 2009

#) Engelbert Redel, Michael Walter, Ralf Thomann, Christian Vollmer, 
   Laith Hussein, Harald Scherer, Michael Krüger and Christoph Janiak

   `Synthesis, stabilization, functionalization and DFT calculations
   of gold nanoparticles in fluorous phases (PTFE and ILs)`__

   Chem. Eur. J. **15**, 10047 (2009)

   __ http://www3.interscience.wiley.com/journal/122564324/abstract

   .. 20 August 2009

#) \A. H. Larsen, M. Vanin, J. J. Mortensen, K. S. Thygesen,
   K. W. Jacobsen

   `Localized atomic basis set in the projector augmented wave method`__

   Physical Review B, Vol. **80**, 195112 (2009)

   __ http://dx.doi.org/10.1103/PhysRevB.80.195112

   .. 18 November 2009

#) Jiří Klimeš, David R. Bowler, and Angelos Michaelides

   `Chemical accuracy for the van der Waals density functional`__

   __ http://arxiv.org/abs/0910.0438

   .. 10 December 2009

#) Thomas Olsen and Jakob Schiøtz

   `Origin of Power Laws for Reactions at Metal Surfaces Mediated by Hot Electrons`__  

   Physical Review Letters, Vol. **103**, 238301 (2009)

   __ http://dx.doi.org/10.1103/PhysRevLett.103.238301

   .. 4 December 2009

#) Carsten Rostgaard, Karsten W. Jacobsen, and Kristian S. Thygesen

   `Fully self-consistent GW calculations for molecules`__

   Physical Review B, Vol. **81**, 085103 (2010)

   __ http://dx.doi.org/10.1103/PhysRevB.81.085103

   .. 3 February 2010

#) \J. Wellendorff, A. Kelkkanen, J. J. Mortensen, B. Lundqvist, and T. Bligaard

   `RPBE-vdW Description of Benzene Adsorption on Au(111)`__

   Topics in Catalysis (2010)

   __ http://dx.doi.org/10.1007/s11244-010-9443-6

   .. 4 February 2010

#) \M. Leetmaa, M. P. Ljungberg, A. Lyubartsev, A. Nilsson and
   L. G. M. Pettersson

   `Theoretical approximations to X-ray absorption spectroscopy of
   liquid water and ice`__

   Journal of Electron Spectroscopy and Related Phenomena,
   Vol. **177**, 135-157, (2010)

   __ http://dx.doi.org/10.1016/j.elspec.2010.02.004

   .. 16 February 2010

#) Thomas Olsen and Jakob Schiøtz

   `Vibrationally mediated control of single-electron transmission in
   weakly coupled molecule-metal junctions`__  

   Physical Review B, Vol. **81**, 115443 (2010)

   __ http://dx.doi.org/10.1103/PhysRevB.81.115443

   .. 23 March 2010

#) Olga Lopez-Acevedo, Katarzyna A. Kacprzak, Jaakko Akola and Hannu
   Häkkinen

   `Quantum size effects in ambient CO oxidation catalysed by
   ligand-protected gold clusters`__

   Nature Chemistry **2**, 329 - 334 (2010) 

   __ http://dx.doi.org/10.1038/nchem.589

   .. 24 March 2010

#) \J. F. Parker, K. A. Kacprzak, O. Lopez-Acevedo, H. Häkkinen,
   R. W. Murray

   `An experimental and density-functional theory analysis of serial
   introductions of electron-withdrawing ligands into the ligand shell
   of a thiolate-protected Au25 nanoparticle`__

   J. Phys. Chem. C 114, 8276 (2010)

   __ http://dx.doi.org/10.1021/jp101265v

   .. 13 April 2010

#) \J. Akola, K. A. Kacprzak, O. Lopez-Acevedo, M. Walter, H. Grönbeck and 
   H. Häkkinen

   `Materials from thiolate-protected Au25 superatoms: Dimers and 
   crystals`__

   J. Phys. Chem C ASAP

   __ http://dx.doi.org/10.1021/jp1015438

   .. 27 April 2010

#) Mikkel Strange, Olga Lopez-Acevedo, and Hannu Häkkinen

   `Oligomeric Gold−Thiolate Units Define the Properties of the 
   Molecular Junction between Gold and Benzene Dithiols`__
   
   Journal of Physical Chemistry Letters **1**, 1528 (2010)
 
   __ http://pubs.acs.org/doi/abs/10.1021/jz1002988
  
   .. 28 April 2010

#) Toyli Anniyev, Hirohito Ogasawara, Mathias P. Ljungberg, Kjartan
   T. Wikfeldt, Janay B. MacNaughton, Lars-Åke Näslund, Uwe Bergmann,
   Shirlaine Koh, Peter Strasser, Lars G. M. Pettersson and Anders
   Nilsson

   `Complementarity between high-energy photoelectron and L-edge
   spectroscopy for probing the electronic structure of 5d transition
   metal catalysts`__

   Phys. Chem. Chem. Phys., **12**, 5694 - 5700, (2010)

   __ http://dx.doi.org/10.1039/b926414k

   .. 4 May 2010

#) Olga Lopez-Acevedo, H. Tsunoyama, T. Tsukuda, H. Häkkinen and
   C. M. Aikens

   `Chirality and electronic structure of the thiolate-protected Au38 
   nanocluster`__

   J. Am. Chem. Soc. ASAP article

   __ http://dx.doi.org/10.1021/ja102934q

   .. 25 May 2010

#) J. Enkovaara, C. Rostgaard, J. J. Mortensen, J. Chen, M. Dulak,
   L. Ferrighi, J. Gavnholt, C. Glinsvad, V. Haikola, H. A. Hansen,
   H. H. Kristoffersen, M. Kuisma, A. H. Larsen, L. Lehtovaara,
   M. Ljungberg, O. Lopez-Acevedo, P. G. Moses, J. Ojanen, T. Olsen,
   V. Petzold, N. A. Romero, J. Stausholm, M. Strange, G. A. Tritsaris,
   M. Vanin, M. Walter, B. Hammer, H. Häkkinen, G. K. H. Madsen,
   R. M. Nieminen, J. K. Nørskov, M. Puska, T. T. Rantala,
   J. Schiøtz, K. S. Thygesen, and K. W. Jacobsen   

   `Electronic structure calculations with GPAW: a real-space
   implementation of the projector augmented-wave method`__ 

   J. Phys.: Condens. Matter **22**, 253202 (2010)

   __ http://stacks.iop.org/0953-8984/22/253202

   .. 8 June 2010