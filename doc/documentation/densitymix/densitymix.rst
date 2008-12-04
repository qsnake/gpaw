.. _densitymix:

.. default-role:: math

==============
Density Mixing
==============

The density is updated using Pulay-mixing [#Pulay1980]_, [#Kresse1996]_.
Convergence is improved by an optimized metric
`\hat{M}` for calculation of scalar products in the mixing scheme,
`\langle A | B \rangle _s = \langle A | \hat{M} | B \rangle`, where
`\langle \rangle _s` is the scalar product with the special metric and
`\langle \rangle` is the usual scalar product.  The metric is based on
the rationale that contributions for small wave vectors are more
important than contributions for large wave
vectors [#Kresse1996]_.

It has been found [#Kresse1996]_ that the metric

.. math::

  \hat{M} = \sum_q | q \rangle f_q \langle q |, \quad f_q =
  1 + \frac{w}{q^2}

is particularly usefull (`w` is a suitably choosen weight).

This is easy to apply in plane wave codes, as it is local in reciprocal space.
Expressed in real space, this metric is

.. math::

  \hat{M} = \sum_{R R'} | R \rangle f(R' - R) \langle R' |, \quad f(R) =
  \sum_q f_q e^{i q R}

As this is fully nonlocal in real space, it would be very costly to apply.
Instead we use a semilocal stencil with only 3rd nearest neighbors:

.. math::

  f(R) = \begin{cases}
  1 + w/8 & R = 0 \\
  w / 16 & R = \text{nearest neighbor dist.} \\
  w / 32 & R = \text{2nd nearest neighbor dist.} \\
  w / 64 & R = \text{3rd nearest neighbor dist.} \\
  0 & \text{otherwise}
  \end{cases}

which correspond to the reciprocal space metric

.. math::

  f_q = 1 + \frac{w}{8} (1 + \cos q_x + \cos q_y + \cos q_z +
  \cos q_x \cos q_y + \cos q_y \cos q_z + \cos q_x \cos q_z +
  \cos q_x \cos q_y \cos q_z)

With the nice property that it is a monotonously decaying function
from `f_q = w + 1` at `q = 0` to `f_q = 1` anywhere at the zone
boundary in reciprocal space.

A comparison of the two metrics is displayed in the figure below

.. image:: metric.png

.. [#Pulay1980] Pulay, Chem. Phys. Let. **73**, 393 (1980)
.. [#Kresse1996] Kresse, Phys. Rev. B **54**, 11169 (1996)

.. default-role::
