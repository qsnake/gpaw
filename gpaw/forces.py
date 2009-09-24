import numpy as np


class ForceCalculator:
    def __init__(self, timer):
        self.timer = timer
        self.reset()
        
    def reset(self):
        self.F_av = None

    def calculate(self, wfs, density, hamiltonian):
        """Return the atomic forces."""

        if wfs.gd.is_non_orthogonal():
            raise NotImplementedError
        
        if self.F_av is not None:
            return self.F_av

        natoms = len(wfs.setups)
        self.F_av = np.zeros((natoms, 3))

        vt_sG = hamiltonian.vt_sG
        if len(vt_sG) == 2:
            vt_G = 0.5 * (vt_sG[0] + vt_sG[1])
        else:
            vt_G = vt_sG[0]

        # Force from projector functions (and basis set):
        wfs.calculate_forces(hamiltonian, self.F_av)

        wfs.band_comm.sum(self.F_av)
        wfs.kpt_comm.sum(self.F_av)
        
        # Force from compensation charges:
        dF_aLv = density.ghat.dict(derivative=True)
        density.ghat.derivative(hamiltonian.vHt_g, dF_aLv)
        for a, dF_Lv in dF_aLv.items():
            self.F_av[a] += np.dot(density.Q_aL[a], dF_Lv)

        # Force from smooth core charge:
        dF_av = density.nct.dict(derivative=True)
        density.nct.derivative(vt_G, dF_av)
        for a, dF_v in dF_av.items():
            self.F_av[a] += dF_v[0]

        # Force from zero potential:
        dF_av = hamiltonian.vbar.dict(derivative=True)
        hamiltonian.vbar.derivative(density.nt_g, dF_av)
        for a, dF_v in dF_av.items():
            self.F_av[a] += dF_v[0]

        wfs.gd.comm.sum(self.F_av)

        # Add non-local contributions:
        for kpt in wfs.kpt_u:
            self.F_av += hamiltonian.xcfunc.get_non_local_force(kpt)
    
        if wfs.symmetry:
            self.F_av = wfs.symmetry.symmetrize_forces(self.F_av)

        return self.F_av
