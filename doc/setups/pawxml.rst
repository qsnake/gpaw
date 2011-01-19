.. _pawxml:

==================================
Atomic setups for PAW calculations
==================================

.. default-role:: math

------------
Introduction
------------


This page contains information about the PAW-XML data format for the
atomic setups necessary for doing projector-augmented wave
calculations\ [#Blo94]_.  We use the term *setup* because the PAW
method is not a pseudopotential method.

An example XML file for nitrogen PAW setup using LDA can be seen
here: N.LDA_.


.. _N.LDA: N.LDA


.. note::
   Hartree atomic units are used in the XML file (`\hbar = m = e = 1`).



---------------------
What defines a setup?
---------------------

The following quantities defines a minimum PAW setup (the notation
from Ref. [#Blo03]_ is used here):


  ============================  ======================================
  Quantity                      Description
  ============================  ======================================
  `Z`                           atomic number
  `E_\text{XC}[n]`              exchange-correlation functional
  `E^\text{kin}_c`              kinetic energy of the core electrons
  `g_{\ell m}(\mathbf{r})`      shape function for compensation charge
  `n_c(r)`                      all-electron core density
  `\tilde{n}_c(r)`              pseudo electron core density
  `\tilde{n}_v(r)`              pseudo electron valence density
  `\bar{v}(r)`                  zero potential
  `\phi_i(\mathbf{r})`          all-electron partial waves
  `\tilde{\phi}_i(\mathbf{r})`  pseudo partial waves
  `\tilde{p}_i(\mathbf{r})`     projector functions
  `\Delta E^\text{kin}_{ij}`    kinetic energy differences
  ============================  ======================================

  Contents of a PAW setup (in order of appearance).



-----------------------------
Specification of the elements
-----------------------------

An element looks like this::

  <name> ... </name>

or for an empty element::

  <name/>

.. tip::
   An XML-tutorial can be found here_

   .. _here: http://www.w3schools.com/xml/default.asp







The header
----------

The first two lines should look like this::

  <?xml version="1.0"?>
  <paw_setup version="0.6">

The first line must be present in all XML files.  Everything else is put
inside an element with name ``paw_setup``, and this element has an
attribute called ``version``.  We are currently at version 0.6.


A comment
---------

It is recommended to put a comment giving the units and a link to this
web page::

  <!-- Nitrogen setup for the Projector Augmented Wave method. -->
  <!-- Units: Hartree and Bohr radii.                          -->
  <!-- http://www.where.org/paw_setup.html                     -->



The ``atom`` element
--------------------

::

    <atom symbol="N" Z="7" core="2" valence="5"/>

The ``atom`` element has attributes ``symbol``, ``Z``, ``core`` and
``valence`` (chemical symbol, atomic number, number of core electrons and
number of valence electrons).



Exchange-correlation
--------------------

::

    <xc_functional type="LDA", name="PW"/>

The ``xc_functional`` element defines the exchange-correlation
functional used for generating the setup, and we currently have the
following possibilities for LDA type functionals:

  ========  ===============================
  ``name``  reference
  ========  ===============================
  ``W``     Wigner\ [#Wig38]_ 
  ``HL``    Hedin, Lundqvist\ [#Hed71]_ 
  ``GL``    Gunnarson, Lundqvist\ [#Gun76]_ 
  ``VWN``   Vosko, Wilk, Nusair\ [#Vos80]_ 
  ``PZ``    Perdew, Zunger\ [#Per81]_ 
  ``PW``    Perdew, Wang\ [#Per92]_
  ========  ===============================



For ``type="GGA"`` we have these possibilities:

  ==========  =============================================
  ``name``    reference
  ==========  =============================================
  ``BLYP``    Becke\ [#Bec88]_ + Lee, Yang, Parr\ [#Lee88]_
  ``PW91``    Perdew, Wang\ [#Per92b]_
  ``PBE``     Perdew, Burke, Ernzerhof\ [#Per96]_
  ``revPBE``  Zhang, Yang\ [#Zha98]_
  ``RPBE``    Hammer, Hansen, Nørskov\ [#Ham99]_
  ==========  =============================================



Generator
---------

::

  <generator type="scalar-relativistic" name="MyGenerator-2.0">
    Frozen core: [He]
  </generator>


This element contains *character data* describing in words how the
setup was generated.  The ``type`` attribute must be one of:
``non-relativistic``, ``scalar-relativistic`` or ``relativistic``.



Energies
--------

::

  <ae_energy kinetic="53.777460" xc="-6.127751"
             electrostatic="-101.690410" total="-54.040701"/>
  <core_energy kinetic="43.529213"/>

The kinetic energy of the core electrons,
`E^\text{kin}_c`, is used in the PAW method.  The other
energies are convenient to have for testing purposes and can also be
useful for checking the quality of the underlying atomic calculation.



Valence states
--------------

::

  <valence_states>
    <state n="2" l="0" f="2"  rc="1.10" e="-0.6766" id="N-2s"/>
    <state n="2" l="1" f="3"  rc="1.10" e="-0.2660" id="N-2p"/>
    <state       l="0"        rc="1.10" e=" 0.3234" id="N-s1"/>
    <state       l="1"        rc="1.10" e=" 0.7340" id="N-p1"/>
    <state       l="2"        rc="1.10" e=" 0.0000" id="N-d1"/>
  </valence_states>

The ``valence_states`` element contains several ``state`` elements.
For this setup, the first two lines describe bound eigenstates with
occupation numbers and principal quantum numbers.  Notice, that the
three additional unbound states should have no ``f`` and ``n``
attributes.  In this way, we know that only the first two bound states
(with ``f`` and ``n`` attributes) should be used for constructing an
initial guess for the wave functions.



Radial grids
------------

There can be one or more definitions of radial grids::

  <radial_grid eq="r=a*i/(n-i)" a="0.400000" n="300" istart="0" iend="299" id="g1"/>

This defines one radial grid as:

.. math::

    r_i = \frac{ai}{n-i},

where `i` runs from 0 to 299.  All functions (densities,
potentials, ...) that use this grid are given as 300 numbers defining
the radial part of the function.  The radial part of the function must
be multiplied by a spherical harmonics: `f_{\ell m}(\mathbf{r}) =
f_\ell(r) Y_{\ell m}(\theta, \phi)`.

Each radial grid has a unique id::

  <radial_grid eq="r=a*exp(d*i)" a="1.056e-4" d="0.05"
               istart="0" iend="249" id="log"/>
  <radial_grid eq="r=d*i" d="0.01" istart="0" iend="99" id="lin"/>

and each numerical function must refer to one of these ids::

  <function grid="lin">
    ... ... ...
  </function>

In this example, the ``function`` element should contain 100 numbers
(`i = 0, ..., 99`).  Each number must be separated by a ``<newline>``
character or by one or more ``<tab>``'s or ``<space>``'s (no commas).
For numbers with scientific nutation, use this format: ``1.23456e-5``
or ``1.23456E-5`` and not ``1.23456D-5``.

There are currently five types of radial grids:

  =====================  ========================
  ``eq``                 parameters
  =====================  ========================
  ``r=a*exp(d*i))``      ``a`` and ``d``
  ``r=a*i/(n-i)``        ``a`` and ``n``
  ``r=a*(exp(d*i)-1)``   ``a`` and ``d``
  ``r=d*i``              ``d``
  ``r=(i/n+a)^5/a-a^4``  ``a`` and ``n``
  =====================  ========================

  Types of radial grids.  Note that the ``eq`` attribute is
  not meant to be parsed - only recognized.

The ``istart`` and ``iend`` attributes indicating the range of `i`
should always be present.



Shape function for the compensation charge
------------------------------------------

The compensation charge for an atom is expanded using the multipole
moments `Q_{\ell m}`:

.. math::

  \sum_{\ell m} Q_{\ell m} \tilde{g}_\ell(r) Y_{\ell m}(\theta, \phi),

where `g_\ell(r) \propto r^\ell k(r)`, and `k(r)` is the shape
function defined in the file like this::

  <shape_function type="gauss" rc="3.478505426185e-01"/>

Here, Gaussians are used, `\exp(-(r/r_c)^2)`, as the
basis for the compensation charges as described in Ref. [#Blo94]_.

Another choice would be ``type="bessel" rc="..."``, where a sum of two
Bessel functions is used\ [#Kre99]_.  Other possibilities are

.. math::

  \left(\frac{\sin(\pi r/r_c)}{\pi r/r_c}\right)^2

as described in Ref. [#Hol01]_, and `\exp(-(r/r_c)^\lambda)`.

  ==========  ===================
  ``type``    parameters
  ==========  ===================
  ``gauss``   ``rc``
  ``bessel``  ``rc``
  ``sinc``    ``rc``
  ``exp``     ``rc`` and ``lamb``
  ==========  ===================

There is also the possibility that the shape function is given in
numerical form.  In that case, the shape function defined inside the
``shape_function`` element::

  <shape_function grid="...">
    ... ...
  </shape_function>

The ``grid`` attribute refers to one of the radial grids defined
earlier.




Radial functions
----------------

Continuing, we have now reached the all-electron core density::

  <ae_core_density grid="g1">
     6.801207147443e+02 6.801207147443e+02 6.665042896724e+02
     ... ...
  </ae_core_density>
  <pseudo_core_density grid="g1">
     ...
  </pseudo_core_density>
  <pseudo_valence_density grid="g1">
     ...
  </pseudo_valence_density>
  <zero_potential grid="g1">
     ...
  </zero_potential>

The numbers inside the ``ae_core_density`` element defines the radial
part of `n_c(\mathbf{r})`.  The radial part must be multiplied by
`Y_{00} = (4\pi)^{-1/2}` to get the full all-electron core density
(which should integrate to the number of core electrons).  The pseudo
core density, the pseudo valence density and the zero potential,
`\bar{v}`, are defined similarly.
 
The ``ae_partial_wave``, ``pseudo_partial_wave`` and
``projector_function`` elements contain the radial parts of the
`\phi_i(\mathbf{r})`, `\tilde{\phi}_i(\mathbf{r})` and
`\tilde{p}_i(\mathbf{r})` functions for the ``state``\ s listed in
the ``valence_states`` element above (five states in the nitrogen
example).  All functions must have an attribute ``state="..."``
referring to one of the states listed in the ``valence_states``
element::

  <ae_partial_wave state="N-2s" grid="g1">
    -8.178800366898029e+00 -8.178246914143839e+00 -8.177654917302689e+00
    ... ...
  </ae_partial_wave>
  <pseudo_partial_wave state="N-2s" grid="g1">
    ...
  </pseudo_partial_wave>
  <projector_function state="N-2s" grid="g1">
    ...
  </projector_function>
  <ae_partial_wave state="N-2p" grid="g1">
    ...
  </ae_partial_wave>
  ...
  ...



Kinetic energy differences
--------------------------

::

    <kinetic_energy_differences>
       1.744042161013e+00 0.000000000000e+00 2.730637956456e+00
       ...
    <kinetic_energy_differences>
  </paw_setup>

This element contains the symmetric `\Delta E^\text{kin}_{ij}` matrix:

.. math::

  \Delta E^\text{kin}_{ij} = \langle \phi_i | \hat{T} | \phi_j \rangle 
  - \langle \tilde{\phi}_i | \hat{T} | \tilde{\phi}_j \rangle 

where `\hat{T}` is the kinetic energy operator used by the
generator.  With `n` states, we have an `n \times n` matrix
listed as `n^2` numbers.





------------------------------
The Kresse-Joubert formulation
------------------------------

The Kresse-Joubert formulation of the PAW method\ [#Kre99]_ is very
similar to the original formulation of Blöchl\ [#Blo94]_.
However, the Kresse-Joubert formulation does not use `\bar{v}`
directly, but indirectly through the local ionic pseudopotential,
`v_H[\tilde{n}_{Zc}]`.  Therefore, the following
transformation is necessary:

.. math::

  v_H[\tilde{n}_{Zc}] = v_H[\tilde{n}_c +
  (N_c - Z - \tilde{N}_c) g_{00} Y_{00}] + \bar{v} +
  v_{xc}[\tilde{n}_v + \tilde{n}_c] -
  v_{xc}[\tilde{n}_v + \tilde{n}_c + (N_v - \tilde{N}_v - \tilde{N}_c) g_{00} Y_{00}]

where `N_c` is the number of core electrons, `N_v` is the number of
valence electrons, `\tilde{N}_c` is the number of electrons contained
in the pseudo core density and `\tilde{N}_v` is the number of
electrons contained in the pseudo valence density.  The Hartree
potential from the density `n` is defined as:

.. math::

   v_H[n](r_1) = 4\pi \int_0^\infty r_2^2 dr_2 \frac{n(r_2)}{r_>},

where `r_>` is the larger of `r_1` and `r_2`.


.. note::
   In the Kresse-Joubert formulation, the symbol `\tilde{n}` is used
   for what we here call `\tilde{n}_v` and in the Blöchl formulation,
   we have `\tilde{n} = \tilde{n}_c + \tilde{n}_v`.


It is also possible to add an element
``kresse_joubert_local_ionic_pseudopotential`` that contains the
`v_H[\tilde{n}_{Zc}](r)` function directly, so that no conversion is
necessary::

  <kresse_joubert_local_ionic_pseudopotential grid="log">
     ...
  </kresse_joubert_local_ionic_pseudopotential>


---------------------
How to use the setups
---------------------

Most likely, the radial functions will be needed on some other type of
radial grid than the one used in the setup.  The idea is that one
should read in the radial functions and then transform them to the
radial grids used by the specific implementation.  After the
transformation, some sort of normalization may be necessary.







-----------------------------
Plotting the radial functions
-----------------------------

The first 10-20 lines of the XML-setups, should be pretty much
human readable, and should give an overview of what kind of setup it
is and how it was generated.  The remaining part of the files contain
numerical data for all the radial functions.  To get an overview of
these functions, you can extract that data with the `<pawxml.py>`__
program and then pass it on to your favorite plotting tool.

.. note::
   The ``pawxml.py`` program is very primitive and is only included in
   order to demonstrates how to parse XML using SAX
   from a Python program.  Parsing XML from Fortran_ or C code with
   SAX should be similar.


.. _Fortran: http://lcdx00.wm.lc.ehu.es/ag/xml/index.html




Usage
-----

It works like this::

  $ pawxml.py [options] setup[.gz]

Options:

==================================  =======================================
``--version``                       Show program's version number and exit.
``-h, --help``                      Show this help message and exit.
``-x <name>, --extract=<name>``     Function to extract.
``-s<channel>, --state=<channel>``  Select valence state.
``-l, --list``                      List valence states
==================================  =======================================



Examples::

  [~]$ pawxml.py -x pseudo_core_density N.LDA | xmgrace -
  [~]$ pawxml.py -x ae_partial_wave -s N2p N.LDA > N.ae.2p 
  [~]$ pawxml.py -x pseudo_partial_wave -s N2p N.LDA > N.ps.2p 
  [~]$ xmgrace N.??.2p




----------
References
----------



.. [#Blo03]  P. E. Blöchl, C. J. Forst and J. Schimpl,
             Projector augmented wave method: Ab initio molecular
             dynamics with full wave functions,
	     *Bulletin of Materials Science* **26**, 33-41 (2003)
.. [#Kre99]  G. Kresse and D. Joubert,
             Form ultrasoft pseudopotentials to the projector 
             augmented-wave method,
	     *Phys. Rev. B* **59**, 1758-1775 (1999)
.. [#Hol01]  N. A. W. Holzwarth, A. R. Tackett, and G. E. Matthews,
             A Projector Augmented Wave (PAW) code for electronics
             structure calculations: Part I *atompaw* for generating
             atom-centered functions, 
             *Computer Physics Communications* **135**, 329-347 (2001)
.. [#Blo94]  P. E. Blöchl, 
             Projector augmented-wave method,
             *Phys. Rev. B* **50**, 17953-19979 (1994)
.. [#Van90]  D. Vanderbilt,
             Soft Self-Consistent Pseudopotentials in a Generalized
             Eigenvalue Formalism,
             *Phys. Rev. B* **41** (Rapid Communications), 7892 (1990)
.. [#Per92]  J. P. Perdew and Y. Wang,
             Accurate and simple analytic representation of the
             electron-gas correlation energy
             *Phys. Rev. B* **45**, 13244-13249 (1992)
.. [#Per96]  J. P. Perdew, K. Burke, and M. Ernzerhof,
             Generalized Gradient Approximation Made Simple,
             *Phys. Rev. Lett.* **77**, 3865 (1996)
.. [#Zha98]  Y. Zhang and W. Yang,
             Comment on "Generalized Gradient Approximation Made Simple",
             *Phys. Rev. Lett.* **80**, 890 (1998)
.. [#Ham99]  B. Hammer, L. B. Hansen and J. K. Nørskov,
             Improved adsorption energetics within density-functional
             theory using revised Perdew-Burke-Ernzerhof functionals,
             *Phys. Rev. B* **59**, 7413 (1999)
.. [#Per81]  J. P. Perdew and A. Zunger, 
             Self-interaction correction to density-functional
             approximations for many-electron systems,
             *Phys. Rev. B* **23**, 5048 (1981)
.. [#Hed71]  L. Hedin and B. I. Lundqvist,
             Explicit local exchange-correlation potentials,
             *J. Phys. C* **4**, 2064 (1971)
.. [#Wig38]  E. P. Wigner,
             Effects of the electron interaction on the energy levels
             of electrons in metals,
             *Trans. Faraday Soc.* **34**, 678 (1938)
.. [#Per92b] J. P. Perdew and Y.Wang,
             Atoms, molecules, solids, and surfaces: Applications
             of the generalized gradient approximation for exchange
             and correlation,
             *Phys. Rev. B* **46**, 6671 (1992)
.. [#Vos80]  S. H. Vosko, L. Wilk and M. Nusair,
             Accurate spin-dependent electron liquid correlation
             energies for local spin density calculations: a critical
             analysis,
             *Can. J. Phys.* **58**, 1200 (1980)
.. [#Gun76]  O. Gunnarson and B. I. Lundqvist,
             Exchange and correlation in atoms, molecules, and solids
             by the spin-density-functional formalism,
             *Phys. Rev. B* **13** ,4274 (1976)
.. [#Bec88]  A. D. Becke,
             Density-functional exchange-energy approximation with
             correct asymptotic behavior,
             *Phys. Rev. A* **38**, 3098 (1988)
.. [#Lee88]  C. Lee, W. Yang and R. G. Parr,
             Development of the Colle-Salvetti correlation-energy
             formula into a functional of the electron density,
             *Phys. Rev. B* **37**, 785 (1988)
