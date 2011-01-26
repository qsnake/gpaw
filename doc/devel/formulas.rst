========
Formulas
========

.. default-role:: math


Coulomb
=======

.. math::

    \frac{1}{|\br-\br'|} =
    \sum_\ell \sum_{m=-\ell}^\ell
    \frac{4\pi}{2\ell+1}
    \frac{r_<^\ell}{r_>^{\ell+1}}
    Y_{\ell m}^*(\hat\br) Y_{\ell m}(\hat\br')

or

.. math::

    \frac{1}{r} = \int d\mathbf{G}\frac{4\pi}{G^2}
    e^{i\mathbf{G}\cdot\br}.


Gaussians
=========

.. math:: n(r) = (\alpha/\pi)^{3/2} e^{-\alpha r^2},

.. math:: \int_0^\infty 4\pi r^2 dr n(r) = 1

Its Fourrier transform is:

.. math::

    n(k) = \int d\br e^{i\mathbf{k}\cdot\br} n(r) =
    \int_0^\infty 4\pi r^2 dr \frac{\sin(kr)}{kr} n(r) =
    e^{-k^2/(4a)}.

With `\nabla^2 v=4\pi n`, we get the potential:

.. math:: v(r) = -\frac{\text{erf}(\sqrt\alpha r)}{r},

and the energy:

.. math::

    \frac12 \int_0^\infty 4\pi r^2 dr n(r) v(r) =
    \sqrt{\frac{\alpha}{2\pi}}.


Hydrogen
========

The 1s orbital:

.. math:: \psi_{\text{1s}}(r) = 2Y_{00} e^{-r},

and the density is:

.. math:: n(r) = |\psi_{\text{1s}}(r)|^2 = e^{-2r}/\pi.

