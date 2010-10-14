"""This module implements the calculation of the electron-phonon couplings."""

__all__ = ["ElectronPhononCoupling"]

import numpy as np
import numpy.fft as fft

from gpaw.mpi import serial_comm

class ElectronPhononCoupling:
    """..."""

    def __init__(self, atoms, gd, kd, calc=None, dmatrix=None, dtype=float):
        """...

        Parameters
        ----------
        atoms: ...
            Atoms in the unit cell.
        gd: GridDescriptor
            Descriptor for the grid on which the derivatives of the effective
            potential are calculated (most likely the coarse grid in the
            ground-state calculation).
        kd: KPointDescriptor
            Descriptor for the q-vector grid on which the derivatives of the
            effective potential are calculated. This will be the same as the
            one used in the ``PhononCalculator`` to obtain the phonons.
        calc: Calculator
            Calculator for a ground-state calculation.
        dmatrix: DynamicalMatrix
            The dynamical matrix. Used to obtain the phonon polarization
            vectors which appears in the coupling elements.
    
        """

        self.atoms = atoms
        self.gd = gd
        self.kd = kd
        self.calc = calc
        self.dmatrix = dmatrix
        self.dtype = dtype
        
        # List for effective-potential derivatives
        self.v1_eff_qavG = []
        self.v1_eff_kavG = None

    def __getstate__(self): 
        """Method used to pickle an instance of this object.

        Bound method attributes cannot be pickled and must therefore be deleted
        before an instance is dumped to file.

        """

        # Get state of object and take care of troublesome attributes
        state = dict(self.__dict__)
        state.pop('calc')
        # state['kd'].__dict__['comm'] = serial_comm
        # state['atoms'].__dict__.pop('calc')

        return state
    
    def set_calculator(self, calc):
        """Set ground-state calculator."""
        
        self.calc = calc

    def collect(self):
        """Collect derivatives from slaves."""

        N = self.atoms.get_number_of_atoms()
        n_c = tuple(self.gd.n_c)
        mynks = self.kd.mynks
        
        # Collect from slaves
        if self.kd.comm.rank == 0:
            # Global array
            self.v1_eff_kavG = self.gd.empty(n=(self.kd.nibzkpts, N, 3),
                                             dtype=self.dtype)
            v1_eff_qavG = np.asarray(self.v1_eff_qavG)
            v1_eff_qavG.shape = (mynks, N, 3) + n_c
            uslice = self.kd.get_slice()
            self.v1_eff_kavG[uslice] = v1_eff_qavG
            
            for slave_rank in range(1, self.kd.comm.size):
                uslice = self.kd.get_slice(rank=slave_rank)
                nks = uslice.stop - uslice.start
                v1_eff_qavG = self.gd.empty(n=(nks, N, 3), dtype=self.dtype)
                self.kd.comm.receive(v1_eff_qavG, slave_rank, tag=123)
                self.v1_eff_kavG[uslice] = v1_eff_qavG
        else:
            v1_eff_qavG = np.asarray(self.v1_eff_qavG)
            v1_eff_qavG.shape = (mynks, N, 3) + n_c
            self.kd.comm.send(v1_eff_qavG, 0, tag=123)
            
    def evaluate_integrals(self, kpts=None, bands=[4,5]):
        """Calculate matrix elements of the potential derivatives.

        Start by evaluating the integrals between the already calculated
        wave-functions.
        
        Parameters
        ----------
        kpts: ???
            ...
        bands: list
            List of band indices to include.

        """

        assert self.calc is not None, "Calculator not set."

        # 1) Do a single-shot calculation to get the required wave-functions
        ## self.calc.set(fixdensity=True,
        ##               kpts=kpts,
        ##               convergence=dict(nbands=8),
        ##               # basis='dzp',
        ##               eigensolver='cg',
        ##               usesymm=None)
        
        # Electronic k-point descriptor
        kd_e = self.calc.wfs.kd
        # Phonon k-point descriptor
        kd_p = self.kd

        # Grid descriptor
        gd = self.calc.wfs.gd

        # Array for matrix elements
        n_q = kd_p.nbzkpts
        n_k = kd_e.nbzkpts
        n_a = self.atoms.get_number_of_atoms()
        n_bands = len(bands)
        g_qavknn = np.empty((n_q, n_a, 3, n_k, n_bands, n_bands),
                            dtype=self.dtype)
        
        for q, q_c in enumerate(kd_p.bzk_kc):
            
            kplusq_k  = kd_e.find_k_plus_q(q_c)

            for a, atom in enumerate(self.atoms):

                for v in [0, 1, 2]:

                    v1_eff_G = self.v1_eff_kavG[q, a, v]
                    
                    for k, k_c in enumerate(kd_e.bzk_kc):

                        kplusq = kplusq_k[k]

                        kpt = self.calc.wfs.kpt_u[k]
                        kplusqpt = self.calc.wfs.kpt_u[kplusq]

                        psit_k_nG = kpt.psit_nG[:][bands]
                        psit_kplusq_nG = kplusqpt.psit_nG[:][bands]

                        a_n = psit_kplusq_nG.conj() * v1_eff_G

                        for n, psit_G in enumerate(psit_k_nG):
                            
                            g_n = gd.integrate(a_n * psit_G)

                            g_qavknn[q, a, v, k, n] = g_n

        return g_qavknn
            


        
