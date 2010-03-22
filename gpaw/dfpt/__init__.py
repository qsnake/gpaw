"""This package contains an implementation of density functional perturbation
theory (DFPT^1). In DFPT the first-order variation in the density due to a
static perturbation in the external potential is calculated self-consistently
from a set of coupled equations for the 1) variation of the effective
potential, 2) variation of the wave functions, and 3) the resulting variation
of the density. The variation of the wave functions can be determined from a
set of linear equations, the so-called Sternheimer equation. Since only the
projection on the unoccupied state manifold is required to determine the
corresponding density variation, only the occupied wave functions/states of the
unperturbed system are needed. This is in contrast to the standard textbook
expression for the first-order variation of the wave function which involves
the full spectrum of the unpertured system.

The first-order variation of the density with respect to different
perturbations can be used to obtain various physical quantities of the
system. This package includes calculators for: 

  1) Phonons in periodic systems
     - perturbation is a lattice distortion with a given q-vector
  2) Born effective charges
     - perturbation is a constant electric field
  3) Dielectric constant
     - perturbation is a constant electric field

References
----------
1) Rev. Mod. Phys. 73, 515 (2001)


Notes
-----
1) When using the ``derivative`` method of the ``lfc`` class to calculate
integrals between derivatives of localized functions wrt atomic displacements
and functions defined on the grid, the signs of the calculated integrals come
out incorrectly (see doc string of the ``derivative`` method). All files of
this ``dfpt`` package follows this rule for correcting the sign error::

    X_niv = -1 * X_aniv[a] ,

i.e. correct the sign when extracting from the dictionary.

"""

# __version__ = "0.1"

# Import things specified in the __all__ attributes
from gpaw.dfpt.phononcalculator import *
from gpaw.dfpt.linearresponse import *
from gpaw.dfpt.phononperturbation import *

# Set the behavior of from gpaw.dfpt import *
import gpaw.dfpt.phononcalculator
import gpaw.dfpt.linearresponse
import gpaw.dfpt.phononperturbation

__all__ = []
__all__.extend(phononcalculator.__all__)
__all__.extend(linearresponse.__all__)
__all__.extend(phononperturbation.__all__)
