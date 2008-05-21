# Written by Lauri Lehtovaara, 2007

"""This module implements a class for (true) time-dependent density 
functional theory calculations.

"""

import sys

import numpy as npy

from ase.units import Bohr, Hartree

from gpaw.paw import PAW
#from gpaw.pawextra import PAWExtra
from gpaw.mixer import BaseMixer

from gpaw.mpi import rank

from gpaw.preconditioner import Preconditioner


from gpaw.tddft.bicgstab import BiCGStab
from gpaw.tddft.cscg import CSCG
from gpaw.tddft.propagators import \
    ExplicitCrankNicolson, \
    SemiImplicitCrankNicolson, \
    SemiImplicitTaylorExponential, \
    SemiImplicitKrylovExponential, \
    AbsorptionKick
from gpaw.tddft.tdopers import \
    TimeDependentHamiltonian, \
    TimeDependentOverlap, \
    TimeDependentDensity, \
    AbsorptionKickHamiltonian

# Where to put these?
# vvvvvvvvv
class DummyMixer(BaseMixer):
    """Dummy mixer for TDDFT, i.e., it does not mix."""
    def mix(self, nt_sG):
        pass

# T^-1
# Bad preconditioner
class KineticEnergyPreconditioner:
    def __init__(self, gd, kin, dtype):
        self.preconditioner = Preconditioner(gd, kin, dtype)

    def apply(self, kpt, psi, psin):
        for i in range(len(psi)):
            psin[i][:] = self.preconditioner(psi[i], kpt.phase_cd, None, None)

# S^-1
class InverseOverlapPreconditioner:
    """Preconditioner for TDDFT.""" 
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
                  propagator='SICN', solver='CSCG', tolerance=1e-8 ):
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
        solver: {'CSCG','BiCGStab'}, optional
            Name of the iterative linear equations solver 
        tolerance: float
            Tolerance for the linear solver

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


        # Convert PAW-object to complex
        self.totype(complex);


        self.text('')
        self.text('')
        self.text('------------------------------------------')
        self.text('  Time-propagation TDDFT                  ')
        self.text('------------------------------------------')
        self.text('')


        # No density mixing
        self.density.mixer = DummyMixer()

        # Set initial time
        self.time = 0.

        # Don't be too strict
        self.density.charge_eps = 1e-5
        self.text('Charge epsilon: ', self.density.charge_eps)

        # Time-dependent variables and operators
        self.td_potential = td_potential
        self.td_hamiltonian = \
            TimeDependentHamiltonian( self.pt_nuclei,
                                      self.hamiltonian,
                                      td_potential )
        self.td_overlap = TimeDependentOverlap(self.overlap)
        self.td_density = TimeDependentDensity(self)


        # Solver for linear equations
        self.text('Solver: ', solver)
        if solver is 'BiCGStab':
            self.solver = BiCGStab( gd=self.gd, timer=self.timer,
                                    tolerance=tolerance )

        elif solver is 'CSCG':
            self.solver = CSCG( gd=self.gd, timer=self.timer,
                                tolerance=tolerance )
        else:
            raise RuntimeError( 'Error in TDDFT: Solver %s not supported. '
                                'Only BiCGStab is currently supported.' 
                                % (solver) )

        # Preconditioner
        # No preconditioner as none good found
        self.text('Preconditioner: ', 'None')
        self.preconditioner = None
        #self.preconditioner = InverseOverlapPreconditioner(self.overlap)
        #self.preconditioner = KineticEnergyPreconditioner(self.gd, self.td_hamiltonian.hamiltonian.kin, npy.complex)

        # Time propagator
        self.text('Propagator: ', propagator)
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
        elif propagator is 'SITE':
            self.propagator = \
                SemiImplicitTaylorExponential( self.td_density,
                                               self.td_hamiltonian,
                                               self.td_overlap,
                                               self.solver,
                                               self.preconditioner,
                                               degree = 4,
                                               gd = self.gd,
                                               timer = self.timer )
        elif propagator is 'SIKE':
            self.propagator = \
                SemiImplicitKrylovExponential( self.td_density,
                                               self.td_hamiltonian,
                                               self.td_overlap,
                                               self.solver,
                                               self.preconditioner,
                                               degree = 4,
                                               gd = self.gd,
                                               timer = self.timer )
        elif propagator is 'SIKE5':
            self.propagator = \
                SemiImplicitKrylovExponential( self.td_density,
                                               self.td_hamiltonian,
                                               self.td_overlap,
                                               self.solver,
                                               self.preconditioner,
                                               degree = 5,
                                               gd = self.gd,
                                               timer = self.timer )
        elif propagator is 'SIKE6':
            self.propagator = \
                SemiImplicitKrylovExponential( self.td_density,
                                               self.td_hamiltonian,
                                               self.td_overlap,
                                               self.solver,
                                               self.preconditioner,
                                               degree = 6,
                                               gd = self.gd,
                                               timer = self.timer )
        else:
            raise RuntimeError( 'Error in TDDFT:' +
                                'Time propagator %s not supported. '
                                % (propagator) )

        if rank == 0:
            self.text('States per processor = ', self.nmybands)


    def propagate(self, time_step, iterations,
                  dipole_moment_file = None,
                  restart_file = None, dump_interval = 1000):
        """Propagates wavefunctions.
        
        Parameters
        ----------
        time_step: float
            Time step in attoseconds (10^-18 s), e.g., 1.0 or 4.0
        iterations: integer
            Iterations, e.g., 10 000 as / 1.0 as = 10 000
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
            if rank == 0:
                dm_file = file(dipole_moment_file,'w')

        for i in range(iterations):
            # print something
            if rank == 0:
                if i % 100 == 0:
                    print ''
                    print i, ' iterations done. Current time is ', \
                        self.time * 24.1888, ' as.'
                elif i % 10 == 0:
                    print '.',
                    sys.stdout.flush()

            # write dipole moment
            if dipole_moment_file is not None:
                dm = self.finegd.calculate_dipole_moment(self.density.rhot_g)
                norm = self.finegd.integrate(self.density.rhot_g)
                if rank == 0:
                    line = '%20.8lf %20.8le %22.12le %22.12le %22.12le\n' \
                        % (self.time, norm, dm[0], dm[1], dm[2])
                    dm_file.write(line)
                    dm_file.flush()

            # propagate
            self.propagator.propagate(self.kpt_u, self.time, time_step)
            self.time += time_step

            # restart data
            if restart_file is not None and ( (i+1) % dump_interval == 0 ):
                #self.write(restart_file, 'all')
                if rank == 0:
                    print 'Warning: Writing restart files in TDDFT does not work yet.'
                    print 'Continuing without writing restart file.'

        # close dipole moment file
        if dipole_moment_file is not None:
            if rank == 0:
                dm_file.close()

        print ''


    # exp(ip.r) psi
    def absorption_kick(self, kick_strength):
        """ Delta absoprtion kick for photoabsorption spectrum.

        Parameters
        ----------
        kick_strength: [float, float, float]
            Strength of the kick, e.g., [0.0, 0.0, 1e-3]
        
        """
        if rank == 0:
            self.text('Delta kick: ', kick_strength)

        abs_kick = \
            AbsorptionKick( AbsorptionKickHamiltonian( self.pt_nuclei,
                                                       npy.array(kick_strength,
                                                                 dtype=float) ),
                            self.td_overlap, self.solver, None,
                            self.gd, self.timer )
        abs_kick.kick(self.kpt_u)


    def photoabsorption_spectrum(dipole_moment_file, spectrum_file, kick_strength, fwhm = 0.5, delta_omega = 0.05, max_energy = 50.0):
        """ Calculates photoabsorption spectrum from the time-dependent
        dipole moment.
        
        Parameters
        ----------
        dipole_moment_file: string
            Name of the time-dependent dipole moment file from which
            the specturm is calculated
        spectrum_file: string
            Name of the spectrum file
        kick_strength: [float, float, float]
            Strength of the kick, e.g., [0.0, 0.0, 1e-3]
        fwhm: float
            Full width at half maximum for peaks in eV
        delta_omega: float
            Energy resolution in eV
        max_energy: float
            Maximum excitation energy in eV
        """

        if rank == 0:
            strength = npy.array(kick_strength, dtype=float)

            self.text( 'Calculating photoabsorption spectrum from file "',
                       dipole_moment_file, '".' )

            f_file = file(spectrum_file, 'w')
            dm_file = file(dipole_moment_file, 'r')
            lines = dm_file.readlines()
            n = len(lines)
            dm = npy.zeros((n,3),dtype=float)
            time = npy.zeros((n,),dtype=float)
            for i in range(n):
                data = lines[i].split()
                time[i] = float(data[0])
                dm[i,0] = float(data[2])
                dm[i,1] = float(data[3])
                dm[i,2] = float(data[4])
            dm_file.close()

            t = time - time[0]
            dt = time[1] - time[0]
            dm[:] = dm - dm[0]
            nw = int(max_energy / delta_omega)
            dw = delta_omega / 27.211
            # f(w) = Nw exp(-w^2/2sigma^2)
            sigma = fwhm / 27.211 / (2.* npy.sqrt(2.* npy.log(2.0)))
            # f(t) = Nt exp(-t^2/2gamma^2)
            gamma = 1.0 / sigma
            kick_magnitude = npy.sum(strength**2)

            # write comment line
            f_file.write('# FWHM = %lf eV = %lf Hartree <=> sigma = %lf eV\n' % (fwhm, fwhm/27.211, sigma*27.211))

            # alpha = 2/(2*pi) / eps int dt sin(omega t) exp(-t^2/(2gamma^2))
            #                                * ( dm(t) - dm(0) )
            alpha = 0
            for i in range(nw):
                w = i * dw
                # x
                alphax = npy.sum( npy.sin(t * w) 
                                 * npy.exp(-t**2 / (2.0*gamma**2)) * dm[:,0] )
                alphax *= \
                    2 * dt / (2*npy.pi) / kick_magnitude * strength[0]
                # y
                alphay = npy.sum( npy.sin(t * w) 
                                 * npy.exp(-t**2 / (2.0*gamma**2)) * dm[:,1] )
                alphay *= \
                    2 * dt / (2*npy.pi) / kick_magnitude * strength[1]
                # z
                alphaz = npy.sum( npy.sin(t * w) 
                                 * npy.exp(-t**2 / (2.0*gamma**2)) * dm[:,2] )
                alphaz *= \
                    2 * dt / (2*npy.pi) / kick_magnitude * strength[2]

                # f = 2 * omega * alpha
                line = '%10.6lf %20.10le %20.10le %20.10le\n' \
                    % ( w*27.211, 
                       2*w*alphax / 27.211, 
                        2*w*alphay / 27.211,
                        2*w*alphaz / 27.211 )
                f_file.write(line)

                if (i % 100) == 0:
                    print '.',
                    sys.stdout.flush()

            print ''
            f_file.close()

            self.text( 'Calculated photoabsorption spectrum saved to file "',
                       spectrum_file, '".' )

    # Make static method
    photoabsorption_spectrum=staticmethod(photoabsorption_spectrum)

        
        
