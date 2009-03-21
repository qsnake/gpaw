.. default-role:: math

============
XAS examples
============

Schematic illustration of XAS (from [Nil04]_):

.. figure:: xas.png
   :width: 400 px
  
Excitation energies can be calculated as eigenvalue differences `\epsilon_n-\epsilon_{1s}`, where the eigenvalues are taken from *one* transition potential calculation with half a core-hole removed.

The oscillator strengths are proportional to 
`|\langle \phi_{1s}| \mathbf{r} | \psi_n \rangle|^2`, where the one-center expansion of `\psi_n` for the core-hole atom can be used.

Calculated oxygen 1s XA spectra for isolated water molecule for different box sizes and experimental data [Myn02]_:

.. |i1| image:: xas_H2O.png
        :width: 550 px
.. |i2| image:: xas_exp.png
        :width: 330 px

|i1| |i2|

Liquid water (32 molecules) XAS for two different oxygen atoms:

.. figure:: xas_32H2O.png

Todo:

* Calculate eigenvalue of core-hole.
* Use spin-polarized core-hole.
* XES.

.. [Nil04] *Chemical bonding on surfaces probed by X-ray emission
   spectroscopy and density functional theory*, A. Nilsson and
   L. G. M. Pettersson, Surf. Sci. Rep. 55 (2004) 49-167
.. [Myn02] S. Myneni, Y. Luo, L. Naslund, M. Cavalleri, L. Ojamae,
   H. Ogasawara, A. Palmenschikov, P. Wernet, P. Vaterlaein, C. Heske,
   Z. Hussain, L. Pettersson and A. Nilsson, 
   J. Phys.: Condens. Matter, 14 (2002) 213-219
