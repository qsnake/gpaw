===============
STM simulations
===============

Scanning Tunneling Microscopy (STM) is a widely used experimental
technique. STM maps out a convolution of the geometric and electronic
structure of a given surface and it is often difficult if not
impossiple to intrepret STM images without the aid of theoretical
tools.

We will use GPAW to simulate an STM image.  Start by
doing a Al(100) surface with hydrogen adsorbed in the ontop site:
HAl100.py_.  This will produce a `gpw` file containing the wave
functions that are needed for calculating local density of states.

The STM image can be calculated with the stm.py_ script::

  $ python stm.py HAl100.gpw

Try the following:

* clean slab without hydrogen
* different number of layers
* different number of **k**-points
* different current


.. _HAl100.py : wiki:SVN:examples/stm/HAl100.py
.. _stm.py : wiki:SVN:examples/stm/stm.py
