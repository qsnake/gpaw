.. _lcao:

.. default-role:: math


=========
LCAO Mode
=========

.. highlight:: bash

In the LCAO mode the Kohn-Sham wave functions `\psi_n` are expanded
onto a set of atomic-like orbitals `\Phi_{nlm}`

.. math::

 \psi_n = \sum_N c_{\mu n} \Phi_{\mu}

The atomic orbitals are constructed as products of numerical radial
functions and spherical harmonics

.. math::

  \Phi_{nlm}(\mathbf{r}) = \varphi_{nl}(r^a) Y_{lm}(\hat{\mathbf{r}}^a)

where `r^a = \mathbf{r-R}^a` is the position of nucleus `a`.

Some detailed informatiom can be found in the master theses `1`_ and `2`_   

.. _1: ../_static/askhl_master.pdf
.. _2: ../_static/marco_master.pdf

Basis-set generation
--------------------

In order to perform a LCAO calculation, a basis-set must be generated
for every element in your system. This can be done by using the
:command:`gpaw-basis` tool, located in your ``\gpaw\tools\``
directory. For example, typing::

  $ gpaw-basis H

will generate the basis-set file :file:`H.dzp.basis` for the Hydrogen
atom with default parameters. Note that :file:`dzp` stands for
``double zeta polarized`` which is the default basis-set type. The
basis-set should be placed in the same directory as the GPAW
setups. For a complete list of the parameters do::

  $ gpaw-basis --help


Running a calculation
---------------------

In order to run a LCAO calculation, the ``lcao`` mode and a basis-set
should be set in the calculator::

  >>> calc=GPAW(mode='lcao',
  >>>           basis='dzp',
  >>>           ...)
 
The calculator can then be used in the usual way. 


Example
-------

The following example will relax a water molecule using the LCAO
calculator. The ``QuasiNewton`` minimizer will use the forces
calculated using the localized basis set.

.. literalinclude:: lcao_h2o.py
