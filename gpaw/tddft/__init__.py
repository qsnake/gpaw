# Written by Lauri Lehtovaara, 2007

"""This module implements a class for (true) time-dependent density
functional theory calculations.

"""

import sys
import time
from math import log

import numpy as np
from ase.units import Bohr, Hartree

import gpaw.io
from gpaw.aseinterface import GPAW
from gpaw.mixer import DummyMixer
from gpaw.version import version
from gpaw.preconditioner import Preconditioner
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.tddft.units import attosec_to_autime, autime_to_attosec, \
                             eV_to_aufrequency, aufrequency_to_eV
from gpaw.tddft.utils import MultiBlas
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

# T^-1
# Bad preconditioner
class KineticEnergyPreconditioner:
    def __init__(self, gd, kin, dtype):
        self.preconditioner = Preconditioner(gd, kin, dtype)
        self.preconditioner.allocate()

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
class TDDFT(GPAW):
    """Time-dependent density functional theory calculation based on GPAW.
    
    This class is the core class of the time-dependent density functional
    theory implementation and is the only class which a user has to use.
    """
    
    def __init__(self, ground_state_file=None, txt='-', td_potential=None,
                 propagator='SICN', solver='CSCG', tolerance=1e-8):
        """Create TDDFT-object.
        
        Parameters:
        -----------
        ground_state_file: string
            File name for the ground state data
        td_potential: class, optional
            Function class for the time-dependent potential. Must have a method
            'strength(time)' which returns the strength of the linear potential
            to each direction as a vector of three floats.
        propagator:  {'SICN', 'ECN', 'SITE', 'SIKE4', 'SIKE5', 'SIKE6'}, optional
            Name of the time propagator for the Kohn-Sham wavefunctions
        solver: {'CSCG','BiCGStab'}, optional
            Name of the iterative linear equations solver for time propagation
        tolerance: float
            Tolerance for the linear solver

        """

        if ground_state_file is None:
            raise RuntimeError('TDDFT calculation has to start from converged '
                               'ground or excited state restart file')

        # Set initial time
        self.time = 0.0

        # Prepare for dipole moment file handle
        self.dm_file = None

        # Initialize paw-object without density mixing
        GPAW.__init__(self, ground_state_file, txt=txt, mixer=DummyMixer())

        # Paw-object has no ``niter`` counter in this branch TODO!
        self.niter = 0

        # Initialize wavefunctions and density 
        # (necessary after restarting from file)
        self.set_positions()
        #self.density.update_pseudo_charge() #TODO done in density.update() ?

        # Don't be too strict
        self.density.charge_eps = 1e-5

        wfs = self.wfs
        self.rank = wfs.world.rank
        
        # Convert PAW-object to complex
        if wfs.dtype == float:
            wfs.dtype = complex
            from gpaw.operators import Laplace
            nn = self.input_parameters.stencils[0]
            wfs.kin = Laplace(wfs.gd, -0.5, nn, complex)
            wfs.pt = LFC(wfs.gd, [setup.pt_j for setup in wfs.setups],
                         self.kpt_comm, dtype=complex)
            
            self.set_positions()

            # Wave functions
            for kpt in wfs.kpt_u:
                kpt.psit_nG = np.array(kpt.psit_nG[:], complex)
        else:
            self.set_positions()

        self.text('')
        self.text('')
        self.text('------------------------------------------')
        self.text('  Time-propagation TDDFT                  ')
        self.text('------------------------------------------')
        self.text('')

        self.text('Charge epsilon: ', self.density.charge_eps)

        # Time-dependent variables and operators
        self.td_potential = td_potential
        self.td_hamiltonian = TimeDependentHamiltonian(self.wfs,
                                  self.hamiltonian, td_potential)
        self.td_overlap = TimeDependentOverlap(self.wfs)
        self.td_density = TimeDependentDensity(self)

        # Solver for linear equations
        self.text('Solver: ', solver)
        if solver is 'BiCGStab':
            self.solver = BiCGStab(gd=self.gd, timer=self.timer,
                                   tolerance=tolerance)
        elif solver is 'CSCG':
            self.solver = CSCG(gd=self.gd, timer=self.timer,
                               tolerance=tolerance)
        else:
            raise RuntimeError('Solver %s not supported.' % solver)

        # Preconditioner
        # No preconditioner as none good found
        self.text('Preconditioner: ', 'None')
        self.preconditioner = None #TODO! check out SSOR preconditioning
        #self.preconditioner = InverseOverlapPreconditioner(self.overlap)
        #self.preconditioner = KineticEnergyPreconditioner(self.gd, self.td_hamiltonian.hamiltonian.kin, np.complex)

        # Time propagator
        self.text('Propagator: ', propagator)
        if propagator is 'ECN':
            self.propagator = ExplicitCrankNicolson(self.td_density,
                self.td_hamiltonian, self.td_overlap, self.solver,
                self.preconditioner, self.gd, self.timer)
        elif propagator is 'SICN':
            self.propagator = SemiImplicitCrankNicolson(self.td_density,
                self.td_hamiltonian, self.td_overlap, self.solver,
                self.preconditioner, self.gd, self.timer)
        elif propagator in ['SITE4', 'SITE']:
            self.propagator = SemiImplicitTaylorExponential(self.td_density,
                self.td_hamiltonian, self.td_overlap, self.solver,
                self.preconditioner, self.gd, self.timer, degree = 4)
        elif propagator in ['SIKE4', 'SIKE']:
            self.propagator = SemiImplicitKrylovExponential(self.td_density,
                self.td_hamiltonian, self.td_overlap, self.solver,
                self.preconditioner, self.gd, self.timer, degree = 4)
        elif propagator is 'SIKE5':
            self.propagator = SemiImplicitKrylovExponential(self.td_density,
                self.td_hamiltonian, self.td_overlap, self.solver,
                self.preconditioner, self.gd, self.timer, degree = 5)
        elif propagator is 'SIKE6':
            self.propagator = SemiImplicitKrylovExponential(self.td_density,
                self.td_hamiltonian, self.td_overlap, self.solver, 
                self.preconditioner, self.gd, self.timer, degree = 6)
        else:
            raise RuntimeError('Time propagator %s not supported.' % propagator)

        if self.rank == 0:
            if wfs.kpt_comm.size > 1:
                if wfs.nspins == 2:
                    self.text('Parallelization Over Spin')

                if self.gd.comm.size > 1:
                    self.text('Using Domain Decomposition: %d x %d x %d' %
                              tuple(self.gd.parsize_c))

                if wfs.band_comm.size > 1:
                    self.text('Parallelization Over bands on %d Processors' %
                              wfs.band_comm.size)
            self.text('States per processor = ', wfs.mynbands)

        self.hpsit = None
        self.eps_tmp = None
        self.mblas = MultiBlas(self.gd)

        self.kick_strength = np.array([0.0,0.0,0.0], dtype=float)

    def read(self, reader):
        assert reader.has_array('PseudoWaveFunctions')
        GPAW.read(self, reader)

    def propagate(self, time_step, iterations, dipole_moment_file=None,
                  restart_file=None, dump_interval=100):
        """Propagates wavefunctions.
        
        Parameters
        ----------
        time_step: float
            Time step in attoseconds (10^-18 s), e.g., 4.0 or 8.0
        iterations: integer
            Iterations, e.g., 20 000 as / 4.0 as = 5000
        dipole_moment_file: string, optional
            Name of the data file where to the time-dependent dipole
            moment is saved
        restart_file: string, optional
            Name of the restart file
        dump_interval: integer
            After how many iterations restart data is dumped
        
        """

        if self.rank == 0:
            self.text()
            self.text('Starting time: %7.2f as'
                      % (self.time * autime_to_attosec))
            self.text('Time step:     %7.2f as' % time_step)
            header = """\
                        Simulation      Total        log10     Iterations:
             Time          time         Energy       Norm      Propagator"""
            self.text()
            self.text(header)


        # Convert to atomic units
        time_step = time_step * attosec_to_autime
        
        if dipole_moment_file is not None:
            self.initialize_dipole_moment_file(dipole_moment_file)

        niterpropagator = 0

        while self.niter<iterations:
            norm = self.finegd.integrate(self.density.rhot_g)

            # Write dipole moment at every iteration
            if dipole_moment_file is not None:
                self.update_dipole_moment_file(norm)

            if self.niter % 10 == 0:
                # print output (energy etc.) every 10th iteration 
                #print '.',
                #sys.stdout.flush()
                # Calculate and print total energy here 
                # self.Eband = sum_i <psi_i|H|psi_j>
                # !!!!
                self.td_overlap.update()
                self.td_density.update()
                self.td_hamiltonian.update(self.td_density.get_density(),
                                           self.time)

                kpt_u = self.wfs.kpt_u
                if self.hpsit is None:
                    self.hpsit = self.gd.zeros(len(kpt_u[0].psit_nG),
                                               dtype=complex)
                if self.eps_tmp is None:
                    self.eps_tmp = np.zeros(len(kpt_u[0].eps_n),
                                             dtype=complex)

                for kpt in kpt_u:
                    self.td_hamiltonian.apply(kpt, kpt.psit_nG, self.hpsit,
                                              calculate_P_ani=False)
                    self.mblas.multi_zdotc(self.eps_tmp, kpt.psit_nG,
                                           self.hpsit, len(kpt_u[0].psit_nG))
                    self.eps_tmp *= self.gd.dv
                    # print 'Eps_n = ', self.eps_tmp
                    kpt.eps_n[:] = self.eps_tmp.real

                self.occupations.calculate_band_energy(kpt_u)

                H = self.td_hamiltonian.hamiltonian

                # Nonlocal
                xcfunc = H.xc.xcfunc
                self.Enlxc = xcfunc.get_non_local_energy()
                self.Enlkin = xcfunc.get_non_local_kinetic_corrections()

                # PAW
                self.Ekin = H.Ekin0 + self.occupations.Eband + self.Enlkin
                self.Epot = H.Epot
                self.Eext = H.Eext
                self.Ebar = H.Ebar
                self.Exc = H.Exc + self.Enlxc
                self.Etot = self.Ekin + self.Epot + self.Ebar + self.Exc

                T = time.localtime()
                if self.rank == 0:
                    iter_text = """iter: %3d  %02d:%02d:%02d %11.2f\
   %13.6f %9.1f %10d"""
                    self.text(iter_text % 
                              (self.niter, T[3], T[4], T[5],
                               self.time * autime_to_attosec,
                               self.Etot, log(abs(norm)+1e-16)/log(10),
                               niterpropagator))

                    self.txt.flush()


            # Propagate the Kohn-Shame wavefunctions a single timestep
            niterpropagator = self.propagator.propagate(self.wfs.kpt_u,
                                  self.time, time_step)
            self.time += time_step
            self.niter += 1

            # Call registered callback functions
            self.call_observers(self.niter)

            # Write restart data
            if restart_file is not None and self.niter % dump_interval == 0:
                self.write(restart_file, 'all')
                if self.rank == 0:
                    print 'Wrote restart file.'
                    print self.niter, ' iterations done. Current time is ', \
                        self.time * autime_to_attosec, ' as.' 
                    # print 'Warning: Writing restart files in TDDFT does not work yet.'
                    # print 'Continuing without writing restart file.'

        # Write final results and close dipole moment file
        if dipole_moment_file is not None:
            #TODO final iteration is propagated, but nothing is updated
            #norm = self.finegd.integrate(self.density.rhot_g)
            #self.finalize_dipole_moment_file(norm)
            self.finalize_dipole_moment_file()

        # Call registered callback functions
        self.call_observers(self.niter, final=True)

        if restart_file is not None:
            self.write(restart_file, 'all')

    def initialize_dipole_moment_file(self, dipole_moment_file):
        if self.rank == 0:
            if self.dm_file is not None and not self.dm_file.closed:
                raise RuntimeError('Dipole moment file is already open')

            if self.time == 0.0:
                mode = 'w'
            else:
                # We probably continue from restart
                mode = 'a'

            self.dm_file = file(dipole_moment_file, mode)
            line = '# Kick = [%22.12le, %22.12le, %22.12le]\n' \
                % (self.kick_strength[0], self.kick_strength[1], self.kick_strength[2])
            self.dm_file.write(line)
            line = '# %15s %15s %22s %22s %22s\n' \
                    % ('time', 'norm', 'dmx', 'dmy', 'dmz')
            self.dm_file.write(line)
            self.dm_file.flush()

    def update_dipole_moment_file(self, norm):
        dm = self.finegd.calculate_dipole_moment(self.density.rhot_g)

        if self.rank == 0:
            line = '%20.8lf %20.8le %22.12le %22.12le %22.12le\n' \
                % (self.time, norm, dm[0], dm[1], dm[2])
            self.dm_file.write(line)
            self.dm_file.flush()

    def finalize_dipole_moment_file(self, norm=None):
        if norm is not None:
            self.update_dipole_moment_file(norm)

        if self.rank == 0:
            self.dm_file.close()
            self.dm_file = None

    # exp(ip.r) psi
    def absorption_kick(self, kick_strength):
        """Delta absoprtion kick for photoabsorption spectrum.

        Parameters
        ----------
        kick_strength: [float, float, float]
            Strength of the kick, e.g., [0.0, 0.0, 1e-3]
        
        """
        if self.rank == 0:
            self.text('Delta kick = ', kick_strength)

        self.kick_strength = np.array(kick_strength)

        abs_kick_hamiltonian = AbsorptionKickHamiltonian(self.wfs, self.atoms,
                                   np.array(kick_strength, float))
        abs_kick = AbsorptionKick(self.wfs, abs_kick_hamiltonian,
                                  self.td_overlap, self.solver,
                                  self.preconditioner, self.gd, self.timer)
        abs_kick.kick(self.wfs.kpt_u)

    def __del__(self):
        """Destructor"""
        GPAW.__del__(self)


from gpaw.mpi import world

# Function for calculating photoabsorption spectrum
#def photoabsorption_spectrum(dipole_moment_file, spectrum_file, fwhm = 0.5, delta_omega = 0.05, max_energy = 50.0):
def photoabsorption_spectrum(dipole_moment_file, spectrum_file,
                             folding='Gauss', width=0.2123,
                             e_min=0.0, e_max=30.0, delta_e=0.05):
    """Calculates photoabsorption spectrum from the time-dependent
    dipole moment.
    
    Parameters
    ----------
    dipole_moment_file: string
        Name of the time-dependent dipole moment file from which
        the specturm is calculated
    spectrum_file: string
        Name of the spectrum file
    folding: 'Gauss' or 'Lorentz'
        Whether to use Gaussian or Lorentzian folding
    width: float
        Width of the Gaussian (sigma) or Lorentzian (Gamma)
        Gaussian =     1/(sigma sqrt(2pi)) exp(-(1/2)(omega/sigma)^2)
        Lonrentzian =  (1/pi) (1/2) Gamma / [omega^2 + ((1/2) Gamma)^2]
    e_min: float
        Minimum energy shown in the spectrum (eV)
    e_max: float
        Maxiumum energy shown in the spectrum (eV)
    delta_e: float
        Energy resolution (eV)
    

    """


#    kick_strength: [float, float, float]
#        Strength of the kick, e.g., [0.0, 0.0, 1e-3]
#    fwhm: float
#        Full width at half maximum for peaks in eV
#    delta_omega: float
#        Energy resolution in eV
#    max_energy: float
#        Maximum excitation energy in eV

    if folding != 'Gauss':
        raise RuntimeError('Error in photoabsorption_spectrum: '
                           'Only Gaussian folding is currently supported.')
    
    
    if world.rank == 0:
        print 'Calculating photoabsorption spectrum from file "%s"' % dipole_moment_file

        f_file = file(spectrum_file, 'w')
        dm_file = file(dipole_moment_file, 'r')
        lines = dm_file.readlines()
        dm_file.close()
        # Read kick strength
        columns = lines[0].split('[')
        columns = columns[1].split(']')
        columns = columns[0].split(',')
        kick_strength = np.array([eval(columns[0]),eval(columns[1]),eval(columns[2])], dtype=float)
        strength = np.array(kick_strength, dtype=float)
        # Remove first two lines
        lines.pop(0)
        lines.pop(0)
        print 'Using kick strength = ', strength
        # Continue with dipole moment data
        n = len(lines)
        dm = np.zeros((n,3),dtype=float)
        time = np.zeros((n,),dtype=float)
        for i in range(n):
            data = lines[i].split()
            time[i] = float(data[0])
            dm[i,0] = float(data[2])
            dm[i,1] = float(data[3])
            dm[i,2] = float(data[4])

        t = time - time[0]
        dt = time[1] - time[0]
        dm[:] = dm - dm[0]
        nw = int(e_max / delta_e)
        dw = delta_e * eV_to_aufrequency
        # f(w) = Nw exp(-w^2/2sigma^2)
        #sigma = fwhm / Hartree / (2.* np.sqrt(2.* np.log(2.0)))
        # f(t) = Nt exp(-t^2*sigma^2/2)
        sigma = width * eV_to_aufrequency
        fwhm = sigma * (2.* np.sqrt(2.* np.log(2.0)))
        kick_magnitude = np.sum(strength**2)

        # write comment line
        f_file.write('# Photoabsorption spectrum from real-time propagation\n')
        f_file.write('# GPAW version: ' + str(version) + '\n')
        f_file.write('# Total time = %lf fs, Time step = %lf as\n' \
            % (n * dt * autime_to_attosec/1000.0, \
               dt * autime_to_attosec))
        f_file.write('# Kick = [%lf,%lf,%lf]\n' % (kick_strength[0], kick_strength[1], kick_strength[2]))
        f_file.write('# %sian folding, Width = %lf eV = %lf Hartree <=> FWHM = %lf eV\n' % (folding, sigma*aufrequency_to_eV, sigma, fwhm*aufrequency_to_eV))

        f_file.write('#  om (eV) %14s%20s%20s\n' % ('Sx', 'Sy', 'Sz'))
        # alpha = 2/(2*pi) / eps int dt sin(omega t) exp(-t^2*sigma^2/2)
        #                                * ( dm(t) - dm(0) )
        alpha = 0
        for i in range(nw):
            w = i * dw
            # x
            alphax = np.sum( np.sin(t * w) 
                              * np.exp(-t**2*sigma**2/2.0) * dm[:,0] )
            alphax *= \
                2 * dt / (2*np.pi) / kick_magnitude * strength[0]
            # y
            alphay = np.sum( np.sin(t * w) 
                              * np.exp(-t**2*sigma**2/2.0) * dm[:,1] )
            alphay *= \
                2 * dt / (2*np.pi) / kick_magnitude * strength[1]
            # z
            alphaz = np.sum( np.sin(t * w) 
                              * np.exp(-t**2*sigma**2/2.0) * dm[:,2] )
            alphaz *= \
                2 * dt / (2*np.pi) / kick_magnitude * strength[2]

            # f = 2 * omega * alpha
            line = '%10.6lf %20.10le %20.10le %20.10le\n' \
                % ( w*aufrequency_to_eV, 
                    2*w*alphax / Hartree, 
                    2*w*alphay / Hartree,
                    2*w*alphaz / Hartree )
            f_file.write(line)

            if (i % 100) == 0:
                print '.',
                sys.stdout.flush()
                
        print ''
        f_file.close()
        
        print ('Calculated photoabsorption spectrum saved to file "%s"' % spectrum_file)

            
    # Make static method
    # photoabsorption_spectrum=staticmethod(photoabsorption_spectrum)
        
