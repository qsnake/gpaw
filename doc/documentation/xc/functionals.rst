.. _xc_functionals:

====================================
Exchange and correlation functionals
====================================

.. default-role:: math


.. index:: libxc


Libxc
=====

We used the functionals from libxc_.  ...



Calculation of GGA potential
============================


In libxc_ we have (see also "Standard subroutine calls" on ccg_dft_design_)
`\sigma_0=\sigma_{\uparrow\uparrow}`,
`\sigma_1=\sigma_{\uparrow\downarrow}` and
`\sigma_2=\sigma_{\downarrow\downarrow}` with

.. math::

  \sigma_{ij} = \mathbf{\nabla}n_i \cdot \mathbf{\nabla}n_j


.. _libxc: http://www.tddft.org/programs/octopus/wiki/index.php/Libxc

.. _ccg_dft_design: http://www.cse.scitech.ac.uk/ccg/dft/design.html


Uniform 3D grid
---------------

We use a finite-difference stencil to calculate the gradients:

.. math::

  \mathbf{\nabla}n_g = \sum_{g'} \mathbf{D}_{gg'} n_{g'}.

The `x`-component of `\mathbf{D}_{gg'}` will be non-zero only when `g`
and `g'` grid points are neighbors in the `x`-direction, where the
values will be `1/(2h)` when `g'` is to the right of `g` and `-1/(2h)`
when `g'` is to the left of `g`.  Similar story for the `y` and `z`
components.

Let's look at the spin-`k` XC potential from the energy expression
`\sum_g\epsilon(\sigma_{ijg})`:

.. math::

  v_{kg} = \sum_{g'} \frac{\partial \epsilon(\sigma_{ijg'})}{\partial n_{kg}}
  = \sum_{g'} 
  \frac{\partial \epsilon(\sigma_{ijg'})}{\partial \sigma_{ijg'}}
  \frac{\partial \sigma_{ijg'}}{\partial n_{kg}}

Using `v_{ijg}=\partial \epsilon(\sigma_{ijg})/\partial \sigma_{ijg}`,
`\mathbf{D}_{gg'}=-\mathbf{D}_{g'g}` and

.. math::

  \frac{\partial \sigma_{ijg'}}{\partial n_{kg}} =
  (\delta_{jk} \mathbf{D}_{g'g} \cdot \mathbf{\nabla}n_{ig'} +
   \delta_{ik} \mathbf{D}_{g'g} \cdot \mathbf{\nabla}n_{jg'}),

we get:

.. math::

  v_{kg} = -\sum_{g'} \mathbf{D}_{gg'} \cdot
  (v_{ijg'} [\delta_{jk} \mathbf{\nabla}n_{ig'} +
             \delta_{ik}  \mathbf{\nabla}n_{jg'}]).


The potentials from the general energy expression
`\sum_g\epsilon(\sigma_{0g}, \sigma_{1g}, \sigma_{2g})` will be:

.. math::

  v_{\uparrow g} = -\sum_{g'} \mathbf{D}_{gg'} \cdot
  (2v_{\uparrow\uparrow g'} \mathbf{\nabla}n_{\uparrow g'} +
   v_{\uparrow\downarrow g'} \mathbf{\nabla}n_{\downarrow g'})

and

.. math::

  v_{\downarrow g} = -\sum_{g'} \mathbf{D}_{gg'} \cdot
  (2v_{\downarrow\downarrow g'} \mathbf{\nabla}n_{\downarrow g'} +
   v_{\uparrow\downarrow g'} \mathbf{\nabla}n_{\uparrow g'}).



Radial grid
-----------

Same story ...
