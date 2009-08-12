"""SIC stuff - work in progress!"""

import numpy as np
import ase.units as units

from gpaw.utilities import pack

def sic(calc, n, s=0):
    """Calculate self interaction energy.

    calc: Calculator object
        Calculator object containing wave functions.
    n: int
        Band index.
    s: int
        Spin index.  Defaults to 0.
        
    **Currently only Coulomb part!**
    """

    kpt = calc.wfs.kpt_u[s]
    psit_G = kpt.psit_nG[n]
    nt_G = psit_G**2

    setups = calc.wfs.setups
    density = calc.density

    I = 0.0
    D_aii = {}
    Q_aL = {}
    for a, P_ni in kpt.P_ani.items():
        P_i = P_ni[n]
        D_aii[a] = np.outer(P_i, P_i)
        D_p = pack(D_aii[a])
        Q_aL[a] = np.dot(D_p, setups[a].Delta_pL)
        # Add atomic corrections to integral
        I += np.dot(D_p, np.dot(setups[a].M_pp, D_p))

    Nt = density.gd.integrate(nt_G)
    nt_g = density.finegd.empty()
    density.interpolator.apply(nt_G, nt_g)
    Ntfine = density.finegd.integrate(nt_g)
    nt_g *= Nt / Ntfine
    density.ghat.add(nt_g, Q_aL)

    # Add coulomb energy of compensated pseudo densities to integral
    psolver = calc.hamiltonian.poisson
    v_g = density.finegd.zeros()
    psolver.solve(v_g, nt_g, charge=1, zero_initial_phi=True)
    I += 0.5 * density.finegd.integrate(nt_g * v_g)

    return I * units.Hartree
