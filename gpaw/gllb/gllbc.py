from gpaw.gllb.gllb import GLLBFunctional
from gpaw.gllb import construct_density1D, SMALL_NUMBER
import Numeric as num


class GLLBCFunctional(GLLBFunctional):
    """GLLB + P86 correlation.

       -I think that there exists some justification to add this
       correlation potential. I will try to discuss it later. Mikael

    """

    def __init__(self, lumo=False):
        GLLBFunctional.__init__(self, lumo=lumo)
        self.correlation_part = None
        self.correlation_part1D = None

    def ensure_P86(self):
        # Create the P86 correlation functional for "GLLB-Correlation part"
        if self.correlation_part == None:
            from gpaw.xc_functional import XCFunctional
            self.correlation_part = XCFunctional("None-C_P86", 1)
            self.initialization_ready = True

    def ensure_radialP86(self, gd):
        self.ensure_P86()
        if self.correlation_part1D == None:
            from gpaw.xc_functional import XCFunctional, XCRadialGrid
            self.correlation_part1D = XCRadialGrid(self.correlation_part, gd)

    def get_slater_part(self, info_s, v_sg, e_g):

        GLLBFunctional.get_slater_part(self, info_s, v_sg, e_g)

        if len(info_s)>1:
            raise "Spin polarized correlation not supported in GLLB (yet!)"

        self.ensure_P86()

        deg = self.nspins

        # Go through all spin densities
        for s, (v_g, info) in enumerate(zip(v_sg, info_s)):
            # Calculate the correlation potential.
            self.tempvxc_g[:] = 0.0
            self.correlation_part.calculate_spinpaired(self.tempe_g, deg*info['n_g'], self.tempvxc_g,
                                                  a2_g = deg*deg*info['a2_g'], deda2_g = self.vt_g)


            # Add it to the total potential
            v_g[:] += self.tempvxc_g
            e_g[:] += self.tempe_g.flat

    def get_slater_part_paw_correction(self, rgd, n_g, a2_g, v_g, pseudo = True, ndenom_g=None):

        Exc = GLLBFunctional.get_slater_part_paw_correction(self, rgd, n_g, a2_g, v_g, pseudo, ndenom_g)

        if ndenom_g == None:
            ndenom_g = n_g

        # TODO: This method needs more arguments to support arbitary slater part
        self.ensure_P86()

        N = len(n_g)
        # TODO: Allocate these only once
        vtemp_g = num.zeros(N, num.Float)
        etemp_g = num.zeros(N, num.Float)
        deda2temp_g = num.zeros(N, num.Float)

        self.correlation_part.calculate_spinpaired(etemp_g, n_g, vtemp_g, a2_g, deda2temp_g)

        # Grr...
        etemp_g[:] = num.where(abs(n_g) < SMALL_NUMBER, 0, etemp_g)

        v_g[:] += vtemp_g

        Exc +=num.sum(etemp_g * rgd.dv_g)

        return Exc

    def get_non_local_energy_and_potential1D(self, gd, u_j, f_j, e_j, l_j, v_xc,
                                             njcore=None, density=None, vbar=False):
        Exc = GLLBFunctional.get_non_local_energy_and_potential1D(self, gd, u_j, f_j, e_j, l_j, v_xc,
                                                                  njcore, density, vbar)
        if njcore is not None:
            return Exc

        # Construct the density if required
        if density == None:
            n_g = construct_density1D(gd, u_j, f_j)
        else:
            n_g = density

        # Do we have already XCRadialGrid object, if not, create one
        self.ensure_radialP86(gd)

        v_g = n_g.copy()
        v_g[:] = 0.0
        v_g2 = n_g.copy()
        v_g2[:] = 0.0

        e_g = n_g.copy()
        e_g[:] = 0.0

        # Calculate B88-energy density
        self.correlation_part1D.get_energy_and_potential_spinpaired(n_g, v_g, e_g=e_g)

        # Add the correlation potential
        v_xc[:] += v_g

        Exc += num.dot(e_g, gd.dv_g)

        return Exc











