# Copyright (c) 2007 Lauri Lehtovaara

"""This module implements a class for (true) time-dependent density 
functional theory calculations.

"""


__docformat__ = "restructuredtext en"


import sys

import numpy as npy

from ase.units import Bohr, Hartree

from gpaw.paw import PAW
#from gpaw.pawextra import PAWExtra

from gpaw.mpi import rank

from gpaw.preconditioner import Preconditioner

from gpaw.tddft.bicgstab import BiCGStab
from gpaw.tddft.propagators import \
    ExplicitCrankNicolson, \
    SemiImplicitCrankNicolson, \
    AbsorptionKick
from gpaw.tddft.tdopers import \
    TimeDependentHamiltonian, \
    TimeDependentOverlap, \
    TimeDependentDensity, \
    AbsorptionKickHamiltonian

# Where to put these?
# vvvvvvvvv
class DummyMixer:
    def mix(self, nt_sG):
        pass

# T^-1
# Bad preconditioner
class KineticEnergyPreconditioner:
    def __init__(self, gd, kin, dtype):
        self.preconditioner = Preconditioner(gd, kin, dtype)

    def apply(self, kpt, psi, psin):
        psin[:] = self.preconditioner(psi, kpt.phase_cd, None, None)

# S^-1
# Not too good preconditioner, might be useful in big system
class InverseOverlapPreconditioner:
    def __init__(self, overlap):
        self.overlap = overlap

    def apply(self, kpt, psi, psin):
        self.overlap.apply_inverse(psi, psin, kpt)
# ^^^^^^^^^^


###########################
# Main class
###########################
class TDDFT(PAW):
    """ Time-dependent density functional theory
    
    This class is the core class of the time-dependent density functional 
    theory implementation and is the only class which user has to use.
    """
    
    def __init__( self, ground_state_file, td_potential = None,
                  propagator='SICN', solver='BiCGStab', tolerance=1e-15 ):
        """Create TDDFT-object.
        
        Parameters:
        -----------
        ground_state_file: string
            File name for the ground state data
        td_potential: class, optional
            Function class for the time-dependent potential. It must have method
            'strength(time)' which returns the strength of the linear potential
            to each direction as a vector of three floats.
        propagator:  {'SICN', 'ECN'}, optional
            Name of the propagator the name of the time propagator
        solver: {'BiCGStab'}, optional
            Name of the iterative linear equations solver 
        tolerance: float
            Tolerance for the linear solver
            Note: Use about ???10^-3 - 10^-4??? tighter tolerance for PAW.

        """

        # Set units to ASE units
        self.a0 = Bohr
        self.Ha = Hartree

        # Initialize paw-object
        PAW.__init__(self,ground_state_file)

        # Initialize wavefunctions and density 
        # (necessary after restarting from file)
        for nucleus in self.nuclei:
            nucleus.ready = False
        self.set_positions()
        self.initialize_wave_functions()
        self.density.update_pseudo_charge()

        # Don't be too strict
        self.density.charge_eps = 1e-6

        # Convert PAW-object to complex
        self.totype(complex);

        # No density mixing
        self.density.mixer = DummyMixer()

        # Set initial time
        self.time = 0.

        # Time-dependent variables and operators
        self.td_potential = td_potential
        self.td_hamiltonian = \
            TimeDependentHamiltonian( self.pt_nuclei,
                                      self.hamiltonian,
                                      td_potential )
        self.td_overlap = TimeDependentOverlap(self.overlap)
        self.td_density = TimeDependentDensity(self)

        # Solver for linear equations
        if solver is 'BiCGStab':
            self.solver = BiCGStab( gd=self.gd, timer=self.timer, 
                                    tolerance=tolerance )
        else:
            raise RuntimeError( 'Error in TDDFT: Solver %s not supported. '
                                'Only BiCGStab is currently supported.' 
                                % (solver) )

        # Preconditioner
        # No preconditioner as none good found
        self.preconditioner = None

        # Time propagator
        if propagator is 'ECN':
            self.propagator = \
                ExplicitCrankNicolson( self.td_density,
                                       self.td_hamiltonian,
                                       self.td_overlap,
                                       self.solver,
                                       self.preconditioner,
                                       self.gd,
                                       self.timer )
        elif propagator is 'SICN':
            self.propagator = \
                SemiImplicitCrankNicolson( self.td_density,
                                           self.td_hamiltonian,
                                           self.td_overlap,
                                           self.solver,
                                           self.preconditioner,
                                           self.gd,
                                           self.timer )
        else:
            raise RuntimeError( 'Error in TDDFT:' +
                                'Time propagator %s not supported. '
                                % (propagator) )
        

    def propagate(self, time_step = 1.0, iterations=10000, 
                  dipole_moment_file = None, 
                  restart_file = None, dump_interval = 1000):
        """Propagates wavefunctions.
        
        Parameters:
        time_step: float
            Time step in attoseconds (10^-18 s)
        iterations: integer
            Iterations
        dipole_moment_file: string, optional
            Name of the data file where to the time-dependent dipole 
            moment is saved
        restart_file: string, optional
            Name of the restart file
        dump_interval: integer
            After how many iterations restart data is dumped
        
        """
        # Convert to atomic units
        time_step = time_step / 24.1888

        if dipole_moment_file is not None:
            dm_file = file(dipole_moment_file,'w')

        for i in range(iterations):
            # print something
            if rank == 0:
                if i % 100 == 0:
                    print ''
                    print i, ' iterations done. Current time is ', self.time * 24.1888, ' as.'
                elif i % 10 == 0:
                    print '.',
                    sys.stdout.flush()

            # write dipole moment
            if dipole_moment_file is not None:
                dm = self.finegd.calculate_dipole_moment(self.density.rhot_g)
                if rank == 0:
                    line = repr(self.time).rjust(20) + '  '
                    line = line + repr(dm[0]).rjust(20) + '  '
                    line = line + repr(dm[1]).rjust(20) + '  '
                    line = line + repr(dm[2]).rjust(20) + '\n'
                    dm_file.write(line)
                    dm_file.flush()

            # propagate
            self.propagator.propagate(self.kpt_u, self.time, time_step)
            self.time += time_step

            # restart data
            if restart_file is not None and ( (i+1) % dump_interval == 0 ):
                if rank == 0:
                    self.write(restart_file, 'all')
                    
        # close dipole moment file
        if dipole_moment_file is not None:
            dm_file.close()

        print ''


    def photoabsortion_spectrum(self, dipole_moment_file, spectrum_file, fwhm = 0.2, delta_omega = 0.01, omega_max = 50.0):
        """ Calculates photoabsorption spectrum from the time-dependent
        dipole moment.
        
        Parameters:
        dipole_moment_file: string
            Name of the time-dependent dipole moment file from which
            the specturm is calculated
        spectrum_file: string
            Name of the spectrum file
        fwhm: float
            Full width at half maximum for peaks
        delta_omega: float
            Energy resolution in electron volts, eV
        omega_max: float
            Maximum excitation energy
        """
        print 'Method "photoabsortion_spectrum(self, dipole_moment_file, spectrum_file)" not implemented yet.'
        dm_file = file(dipole_moment_file, 'r')
        dm_file.close()
        pass

    # exp(ip.r) psi
    def absorption_kick(self, strength = [0.0,0.0,1e-4]):
        """ Delta absoprtion kick for photoabsorption spectrum.
        
        """
        if rank == 0:
            print 'Delta kick: ', strength

        abs_kick = \
            AbsorptionKick( AbsorptionKickHamiltonian( self.pt_nuclei,
                                                       npy.array(strength, 
                                                                 dtype=float) ),
                            self.td_overlap, self.solver, self.gd, self.timer )
        abs_kick.kick(self.kpt_u)
