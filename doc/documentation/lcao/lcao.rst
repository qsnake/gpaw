.. _lcao:

.. default-role:: math


=========
LCAO Mode
=========

.. highlight:: bash

In the LCAO mode the Kohn-Sham pseudo wave functions `\tilde{\psi}_n`
are expanded onto a set of atomic-like orbitals `\Phi_{nlm}`, in the
same spirit as the SIESTA method [Siesta]_ :

.. math::

 \tilde{\psi}_n = \sum_N c_{\mu n} \Phi_{\mu}

The atomic orbitals are constructed as products of numerical radial
functions and spherical harmonics

.. math::

  \Phi_{nlm}(\mathbf{r}) = \varphi_{nl}(r^a) Y_{lm}(\hat{\mathbf{r}}^a)

where `r^a = \mathbf{r-R}^a` is the position of nucleus `a`.

In this approximation the variational parameters are the coefficients
`c_{\mu n}` rather than the real space wave function. The eigenvalue
problem then becomes

.. math::

 \sum_\nu H_{\mu\nu} c_{\nu n}   = \sum_{\nu} S_{\mu\nu} c_{\nu n} \epsilon_n

which can be solved by directly diagonalization of the Hamiltonian in
the basis of the atomic orbitals.


Some detailed informatiom can be found in the master theses `1`_ and `2`_.  

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

  >>> calc = GPAW(mode='lcao',
  >>>             basis='dzp',
  >>>             ...)
 
The calculator can then be used in the usual way. 


Example
-------

The following example will relax a water molecule using the LCAO
calculator. The ``QuasiNewton`` minimizer will use the forces
calculated using the localized basis set.

.. literalinclude:: lcao_h2o.py

It is possible to switch to the Finite Difference mode and further
relax the structure simply by doing::

  >>> calc.set(mode='fd')
  >>> dyn.run(fmax=0.05)



.. [Siesta] J.M. Soler et al.,
   J. Phys. Cond. Matter 14, 2745-2779 (2002) 

