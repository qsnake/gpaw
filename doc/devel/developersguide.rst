.. _developersguide:

========================
Developers guide to GPAW
========================

.. default-role:: math

XXX Update page to new GPAW style (after guc merge)

This page goes through the most important equations of a PAW
calculation and has references to the code.  It is a good idea to have
:ref:`the_big_picture` in front of you when reading this page.

For special topics, look at these pages:

* :ref:`eigenvalues_of_core_states`
* :ref:`generation_of_setups`
* :ref:`electrostatic_potential`
* Initial wave functions and densities (todo)
* Finding the ground state (todo)
* :ref:`densitymix`
* ...


Wave functions
==============

The central quantities in a PAW calculation are the pseudo
wave-functions, `\tilde{\psi}_{\sigma\mathbf{k}n}(\mathbf{r})`, from which
the all-electron wave functions can be obtained:

.. math::

  \psi_{\sigma\mathbf{k}n}(\mathbf{r}) =
  \tilde{\psi}_{\sigma\mathbf{k}n}(\mathbf{r}) +
  \sum_a \sum_i 
  [\phi_i^a(\mathbf{r} - \mathbf{R}^a) -
   \tilde{\phi}_i^a(\mathbf{r} - \mathbf{R}^a)]
  \langle\tilde{p}_i^a | \tilde{\psi}_{\sigma\mathbf{k}n} \rangle,

where

.. math::

  \langle\tilde{p}_i^a | \tilde{\psi}_{\sigma\mathbf{k}n} \rangle =
  \int d\mathbf{r}
  \tilde{p}_i^a(\mathbf{r} - \mathbf{R}^a) \tilde{\psi}_{\sigma\mathbf{k}n}(\mathbf{r}).

Here, `a` is the atom number, `\mathbf{R}^a` is the position of atom
number `a` and `\tilde{p}_i^a`, `\tilde{\phi}_i^a` and `\phi_i^a` are
the projector functions, pseudo partial waves, and all-electron
partial waves respectively, of the atoms.

See :ref:`overview_array_naming` for more information on the naming of
arrays.  Note that ``spos_c`` gives the position of the atom in scaled
coordinates in the range [0:1[ (relative to the unit cell).


Note, that in the code, ``i`` refers to `n`, `\ell` and `m` quantum
numbers, and ``j`` refers to `n` and `\ell` only (see
:ref:`overview_array_naming`).  So, to put an atom-centered function
like `\tilde{p}_{n\ell m}^a(\mathbf{r})` on the 3D grid, you need both
the radial part `\tilde{p}_{n\ell}^a(r)` (one of the splines in
``paw.nucleus[a].setup.pt_j``) and a spherical harmonics `Y_{\ell
m}(\theta,\phi)`.  Putting radial functions times spherical harmonics
on a grid is done by the :epydoc:`create_localized_functions
<gpaw.localized_functions>` function.  The result of the function is a
:epydoc:`gpaw.localized_functions.LocFuncs` object
(``paw.nuclei[a].pt_i``).

.. _orthogonality:

The pseudo wave-functions are othonormalized like this:
 
.. math::

  \langle \psi_{\sigma\mathbf{k}n} | 
          \psi_{\sigma\mathbf{k}m} \rangle =
  \langle \tilde{\psi}_{\sigma\mathbf{k}n} | \hat{O} |
          \tilde{\psi}_{\sigma\mathbf{k}m} \rangle =
  \delta_{nm},

where the overlap operator looks like:

.. math::

  \hat{O} = 1 +
    \sum_a \sum_{i_1 i_2} |\tilde{p}_{i_1}^a\rangle
    \Delta O_{i_1 i_2}^a \langle\tilde{p}_{i_2}^a|.

The constants `\Delta O_{i_1 i_2}^a` are found in
``nuclei[a].setup.O_ii`` (``ndarray``).  Someone should rename
``setup.O_ii`` to ``setup.dO_ii``.

.. math::

  \Delta O_{i_1 i_2}^a =
  \int d\mathbf{r}
  [\phi_{i_1}^a(\mathbf{r})\phi_{i_2}^a(\mathbf{r}) -
   \tilde{\phi}_{i_1}^a(\mathbf{r})\tilde{\phi}_{i_2}^a(\mathbf{r})].

See also :epydoc:`gpaw.setup.Setup`,
and :epydoc:`gpaw.spline.Spline`.


.. _density:

Densities
=========

From the pseudo wave-functions, the pseudo electron spin-densities can be
constructed:

.. math::

  \tilde{n}_\sigma(\mathbf{r}) = 
  \frac{1}{N_s} \sum_{s=1}^{N_s}
  \hat{S}_s \left [
  \sum_{n\mathbf{k}} f_{n\mathbf{k}\sigma}
  |\tilde{\psi}_{n\mathbf{k}\sigma}(\mathbf{r})|^2 +
  \frac{1}{2} \sum_a \tilde{n}_c^a(|\mathbf{r}-\mathbf{R}^a|) \right ].

Here, `\hat{S}_s` is one of the `N_s` symmetry operators of the system
(see :epydoc:`gpaw.symmetry.Symmetry`), `f_{n\mathbf{k}\sigma}` are the occupation numbers (adding up to the number of valence elctrons), and
`\tilde{n}_c^a(r)` is the pseudo core density for atom number `a`.

The all-electron spin-densities are given as:

.. math::

  n_\sigma(\mathbf{r}) = \tilde{n}_\sigma(\mathbf{r}) +
  \sum_a [n_\sigma^a(\mathbf{r} - \mathbf{R}^a) -
          \tilde{n}_\sigma^a(\mathbf{r} - \mathbf{R}^a)],

where

.. math::

  n_\sigma^a(\mathbf{r}) =
  \sum_{i_1 i_2} D_{\sigma i_1 i_2}^a
  \phi_{i_1}^a(\mathbf{r})\phi_{i_2}^a(\mathbf{r}) +
  \frac{1}{2} n_c^a(r),

.. math::

  \tilde{n}_\sigma^a(\mathbf{r}) =
  \sum_{i_1 i_2} D_{\sigma i_1 i_2}^a
  \tilde{\phi}_{i_1}^a(\mathbf{r})\tilde{\phi}_{i_2}^a(\mathbf{r}) +
  \frac{1}{2} \tilde{n}_c^a(r),

are atom centered expansions, and 

.. math::

  D_{\sigma i_1 i_2}^a =
  \sum_{n\mathbf{k}}
  \langle \tilde{\psi}_{\sigma\mathbf{k}n} | \tilde{p}_{i_1}^a \rangle
   f_{n\mathbf{k}\sigma}
  \langle \tilde{p}_{i_2}^a | \tilde{\psi}_{\sigma\mathbf{k}n} \rangle

is an atomic spin-density matrix, which must be symmetrized the same
way as the pseudo electron spin-densities.

.. list-table::

   * - formula
     - object
     - type
   * - `\hat{S}_s`
     - ``paw.symmetry``
     - :epydoc:`gpaw.symmetry.Symmetry`
   * - `\tilde{n}_\sigma`
     - ``paw.density.nt_sG`` and ``paw.density.nt_sg``
     - ``ndarray``
   * - `\tilde{n}=\sum_\sigma\tilde{n}_\sigma`
     - ``paw.density.nt_g``
     - ``ndarray``
   * - `\tilde{n}_c^a(r)`
     - ``setup.nct``
     - :epydoc:`gpaw.spline.Spline`
   * - `\tilde{n}_c^a(\mathbf{r}-\mathbf{R}^a)`
     - ``nuclei[a].nct``
     - :epydoc:`gpaw.localized_functions.LocFuncs`
   * - `f_{\sigma\mathbf{k}n}`
     - ``paw.kpt_u[u].f_n``
     - ``ndarray``
   * - `D_{\sigma i_1 i_2}^a`
     - ``nuclei[a].D_sp``
     - ``ndarray``

From the all-electron and pseudo electron densities we can now construct
corresponding total all-electron and pseudo charge densities:

.. math::

  \rho(\mathbf{r}) = \sum_\sigma n_\sigma(\mathbf{r}) +
  \sum_a Z^a(\mathbf{r} - \mathbf{R}^a),

.. math::

  \tilde{\rho}(\mathbf{r}) = \sum_\sigma \tilde{n}_\sigma(\mathbf{r}) +
  \sum_a \tilde{Z}^a(\mathbf{r} - \mathbf{R}^a).

If `\mathbb{Z}^a` is the atomic number of atom number `a`, then
`Z^a(\mathbf{r})=-\mathbb{Z}^a\delta(\mathbf{r})` (we count the electrons as
positive charge and the protons as negative charge).  The compensation charges are given as:

.. math::

  \tilde{Z}^a(\mathbf{r}) = 
  \sum_{\ell=0}^{\ell_{\text{max}}} \sum_{m=-\ell}^\ell
   Q_{\ell m}^a \hat{g}_{\ell m}^a(\mathbf{r}) =
  \sum_{\ell=0}^{\ell_{\text{max}}} \sum_{m=-\ell}^\ell
   Q_{\ell m}^a \hat{g}_\ell^a(r) Y_{\ell m}(\theta,\phi),

where `\hat{g}_\ell^a(r)\propto r^\ell\exp(-\alpha^a r^2)` are
Gaussians.  The compensation charges should make sure that the two atom
centered densities `\rho^a=\sum_\sigma n_\sigma^a + Z^a` and `\tilde{\rho}^a=\sum_\sigma
\tilde{n}_\sigma^a + \tilde{Z}^a` have identical multipole expansions
outside the augmentation sphere.  This gives the following equation
for `Q_L^a`:

.. math::

  Q_L^a = \sum_{i_1 i_2} \Delta_{i_1 i_2 L}^a 
  \sum_\sigma D_{\sigma i_1 i_2}^a +
  \Delta_0^a \delta_{\ell,0},

where

.. math::

  \Delta_{i_1 i_2 L}^a = 
  \int d\mathbf{r} Y_L(\hat{\mathbf{r}}) r^\ell
  [\phi_{i_1}^a(\mathbf{r})\phi_{i_2}^a(\mathbf{r}) -
   \tilde{\phi}_{i_1}^a(\mathbf{r})\tilde{\phi}_{i_2}^a(\mathbf{r})],

.. math::

  \Delta_0^a =
  \int d\mathbf{r} Y_{00}(\hat{\mathbf{r}})
  [-\mathbb{Z}^a \delta(\mathbf{r}) + n_c^a(\mathbf{r}) - \tilde{n}_c^a(\mathbf{r})].


.. list-table::

   * - formula
     - object
     - type
   * - `\tilde{\rho}`
     - ``paw.density.rhot_g``
     - ``ndarray``
   * - `\mathbb{Z}^a`
     - ``setup.Z``
     - ``int``
   * - `\Delta_{i_1 i_2 L}^a`
     - ``setup.Delta_pL``
     - ``ndarray``
   * - `\Delta_0^a`
     - ``setup.Delta0``
     - ``float``
   * - `\hat{g}_\ell^a(r)`
     - ``setup.ghat_l``
     - List of :epydoc:`gpaw.spline.Spline`\ s
   * - `\hat{g}_L^a(\mathbf{r}-\mathbf{R}^a)`
     - ``nuclei[a].ghat_L``
     - :epydoc:`gpaw.localized_functions.LocFuncs`
   * - `Q_L^a`
     - ``nuclei[a].Q_L``
     - ``ndarray``


.. _developersguide_total_energy:

The total energy
================

The total PAW energy is composed of a smooth part evaluated using
pseudo quantities on the 3D grid, plus corrections for each atom
evaluated on radial grids inside the augmentation spheres:
`E=\tilde{E}+\sum_a(E^a - \tilde{E}^a)`.

.. math::

  \tilde{E} &= -\frac{1}{2} \sum_{\sigma\mathbf{k}n} f_{\sigma\mathbf{k}n}
  \int d\mathbf{r}
  \tilde{\psi}_{\sigma\mathbf{k}n}(\mathbf{r})
  \nabla^2 \tilde{\psi}_{\sigma\mathbf{k}n}(\mathbf{r}) +
  \frac{1}{2}\int d\mathbf{r}d\mathbf{r}'
  \frac{\tilde{\rho}(\mathbf{r})\tilde{\rho}(\mathbf{r}')}
       {|\mathbf{r}-\mathbf{r}'|} \\ &\quad+
  \sum_\sigma\sum_a\int d\mathbf{r}\tilde{n}_\sigma(\mathbf{r})
  \bar{v}^a(|\mathbf{r}-\mathbf{R}^a|) +
  E_{\text{xc}}[\tilde{n}_\uparrow, \tilde{n}_\downarrow]
  %
  %.. math::
  %
  \\
  E^a &= -\frac{1}{2} 2\sum_i^{\text{core}} 
  \int d\mathbf{r}
  \phi_i^a(\mathbf{r})
  \nabla^2 \phi_i^a(\mathbf{r})
  -\frac{1}{2} \sum_\sigma \sum_{i_1 i_2} D_{\sigma i_1 i_2}^a
  \int d\mathbf{r}
  \phi_{i_1}^a(\mathbf{r})
  \nabla^2 \phi_{i_2}^a(\mathbf{r}) \\ &\quad+
  \frac{1}{2}\int d\mathbf{r}d\mathbf{r}'
  \frac{\rho^a(\mathbf{r})\rho^a(\mathbf{r}')}
       {|\mathbf{r}-\mathbf{r}'|} +
  E_{\text{xc}}[n^a_\uparrow, n^a_\downarrow]
  %
  %.. math::
  %
  \\
  \tilde{E}^a &= -\frac{1}{2} \sum_\sigma\sum_{i_1 i_2} D_{\sigma i_1 i_2}^a
  \int d\mathbf{r}
  \tilde{\phi}_{i_1}^a(\mathbf{r})
  \nabla^2 \tilde{\phi}_{i_2}^a(\mathbf{r}) +
  \frac{1}{2}\int d\mathbf{r}d\mathbf{r}'
  \frac{\tilde{\rho}^a(\mathbf{r})\tilde{\rho}^a(\mathbf{r}')}
       {|\mathbf{r}-\mathbf{r}'|} \\ &\quad+
  \sum_\sigma \int d\mathbf{r}\tilde{n}^a_\sigma(\mathbf{r})
  \bar{v}^a(r) +
  E_{\text{xc}}[\tilde{n}^a_\uparrow, \tilde{n}^a_\downarrow]

In the last two equations, the integrations are limited to inside the
augmentation spheres only.

The electrostatic energy part of `\tilde{E}` is calculated as
`\frac{1}{2}\int
d\mathbf{r}\tilde{v}_H(\mathbf{r})\tilde{\rho}(\mathbf{r})`, where the
Hartree potential is found by solving Poissons equation:
`\nabla^2 \tilde{v}_H(\mathbf{r})=-4\pi\tilde{\rho}(\mathbf{r})` (see
:epydoc:`gpaw.poisson.PoissonSolver`).
