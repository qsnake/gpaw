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
        state['kd'].__dict__['comm'] = serial_comm
        state['atoms'].__dict__.pop('calc')

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
            self.v1_eff_kavG = self.gd.empty(n=(self.kd.nibzkpts, 3 * N),
                                             dtype=self.dtype)
            v1_eff_qavG = np.asarray(self.v1_eff_qavG)
            v1_eff_qavG.shape = (mynks, 3*N) + n_c
            uslice = self.kd.get_slice()
            self.v1_eff_kavG[uslice] = v1_eff_qavG
            
            for slave_rank in range(1, self.kd.comm.size):
                uslice = self.kd.get_slice(rank=slave_rank)
                nks = uslice.stop - uslice.start
                v1_eff_qavG = self.gd.empty(n=(nks, 3*N), dtype=self.dtype)
                self.kd.comm.receive(v1_eff_qavG, slave_rank, tag=123)
                self.v1_eff_kavG[uslice] = v1_eff_qavG
        else:
            v1_eff_qavG = np.asarray(self.v1_eff_qavG)
            v1_eff_qavG.shape = (mynks, 3*N) + n_c
            self.kd.comm.send(v1_eff_qavG, 0, tag=123)
            
