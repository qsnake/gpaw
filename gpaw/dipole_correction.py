import numpy as np

from gpaw.utilities import erf


class DipoleCorrectionPoissonSolver:
    """Dipole-correcting wrapper around another PoissonSolver."""
    def __init__(self, poissonsolver, direction):
        self.corrector = DipoleCorrection(direction)
        self.poissonsolver = poissonsolver

    def get_method(self):
        description = self.poissonsolver.get_method()
        return '%s with %s-axis dipole correction' % (description,
                                                      'xyz'[self.corrector.c])
    def get_stencil(self):
        return self.poissonsolver.get_stencil()

    def set_grid_descriptor(self, gd):
        self.poissonsolver.set_grid_descriptor(gd)

    def initialize(self):
        self.poissonsolver.initialize()

    def solve(self, phi, rho, **kwargs):
        gd = self.poissonsolver.gd
        drho, dphi = self.corrector.get_dipole_correction(gd, rho)
        iters = self.poissonsolver.solve(phi, rho + drho, **kwargs)
        phi += dphi
        return iters

    def estimate_memory(self, mem):
        self.poissonsolver.estimate_memory(mem)


class DipoleCorrection:
    def __init__(self, direction):
        self.c = direction

    def get_dipole_correction(self, gd, rhot_g):
        """Get dipole corrections to charge and potential.

        Returns arrays drhot_g and dphit_g such that if rhot_g has the
        potential phit_g, then rhot_g + drhot_g has the potential
        phit_g + dphit_g, where dphit_g is an error function.
        
        The error function is chosen so as to be largely constant at the
        cell boundaries and beyond.
        """
        # This implementation is not particularly economical memory-wise
        if not gd.orthogonal:
            raise ValueError('Dipole correction requires orthorhombic cell')
        
        c = self.c
        moment = gd.calculate_dipole_moment(rhot_g)[c]
        if abs(moment) < 1e-12:
            return gd.zeros(), gd.zeros()

        r_g = gd.get_grid_point_coordinates()[c]
        cellsize = abs(gd.cell_cv[c, c])
        sr_g = 2.0 / cellsize * r_g - 1.0 # sr ~ 'scaled r'
        alpha = 12.0 # should perhaps be variable
        drho_g = sr_g * np.exp(-alpha * sr_g**2)
        moment2 = gd.calculate_dipole_moment(drho_g)[c]
        factor = -moment / moment2
        drho_g *= factor
        phifactor = factor * (np.pi / alpha)**1.5 * cellsize**2 / 4.0
        dphi_g = -phifactor * erf(sr_g * np.sqrt(alpha))
        return drho_g, dphi_g
