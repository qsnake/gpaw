# Written by Lauri Lehtovaara, 2007

"""This module implements a class for (true) time-dependent density 
functional theory calculations.

"""

import sys
import time
from math import log

import numpy as npy

from ase.units import Bohr, Hartree

from gpaw.paw import PAW
#from gpaw.pawextra import PAWExtra
from gpaw.mixer import BaseMixer

from gpaw.mpi import rank
from gpaw.version import version

from gpaw.preconditioner import Preconditioner


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
    
    def __init__( self, ground_state_file=None, txt='-', td_potential = None,
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

        if ground_state_file is None:
            raise RuntimeError('TD calculation has to start from ground state restart file')

        # Set units to ASE units
        self.a0 = Bohr
        self.Ha = Hartree
        self.attosec_to_autime = 1/24.188843265
        self.autime_to_attosec = 24.188843265


        # Set initial time
        self.time = 0.0

        # Initialize paw-object
        PAW.__init__(self,ground_state_file, txt=txt)

        # Initialize wavefunctions and density 
        # (necessary after restarting from file)
        self.density.charge_eps = 1e-5
        for nucleus in self.nuclei:
            nucleus.ready = False
        self.set_positions()
        self.initialize_wave_functions()
        # Don't be too strict
        self.density.update_pseudo_charge()

        # Convert PAW-object to complex
        if self.dtype == float:
            self.totype(complex);


        self.text('')
        self.text('')
        self.text('------------------------------------------')
        self.text('  Time-propagation TDDFT                  ')
        self.text('------------------------------------------')
        self.text('')


        # No density mixing
        self.density.mixer = DummyMixer()

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
            if self.kpt_comm.size > 1:
                if self.nspins == 2:
                    self.text('Parallelization Over Spin')

                domain = self.domain
                if domain.comm.size > 1:
                    self.text('Using Domain Decomposition: %d x %d x %d' %
                              tuple(domain.parsize_c))

                if self.band_comm.size > 1:
                    self.text('Parallelization Over bands on %d Processors' %
                              self.band_comm.size)
            self.text('States per processor = ', self.nmybands)

        self.hpsit = None
        self.eps_tmp = None
        self.mblas = MultiBlas(self.gd)

        self.kick_strength = npy.array([0.0,0.0,0.0], dtype=float)


    def propagate(self, time_step, iterations,
                  dipole_moment_file = None,
                  restart_file = None, dump_interval = 100):
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

        if rank == 0:
            self.text()
            self.text('Starting time: %7.2f as'
                      % (self.time * self.autime_to_attosec))
            self.text('Time step:     %7.2f as' % time_step)
            header = """\
                        Simulation      Total        log10     Iterations:
             Time          time         Energy       Norm      Propagator"""
            self.text()
            self.text(header)


        # Convert to atomic units
        time_step = time_step * self.attosec_to_autime
        
        if dipole_moment_file is not None:
            if rank == 0:
                if self.time == 0.0:
                    mode = 'w'
                else:
                    # We probably continue from restart
                    mode = 'a'
                dm_file = file(dipole_moment_file, mode)
                line = '# Kick = [%22.12le, %22.12le, %22.12le]\n' \
                    % (self.kick_strength[0], self.kick_strength[1], self.kick_strength[2])
                dm_file.write(line)
                line = '# %15s %15s %22s %22s %22s\n' \
                        % ('time', 'norm', 'dmx', 'dmy', 'dmz')
                dm_file.write(line)
                dm_file.flush()

        niterpropagator = 0
        for i in range(iterations):
            # write dipole moment at every iteration
            if dipole_moment_file is not None:
                dm = self.finegd.calculate_dipole_moment(self.density.rhot_g)
                norm = self.finegd.integrate(self.density.rhot_g)
                if rank == 0:
                    line = '%20.8lf %20.8le %22.12le %22.12le %22.12le\n' \
                        % (self.time, norm, dm[0], dm[1], dm[2])
                    dm_file.write(line)
                    dm_file.flush()

            if i % 10 == 0:
                # print output (energy etc.) every 10th iteration 
                #print '.',
                #sys.stdout.flush()
                # Calculate and print total energy here 
                # self.Eband = sum_i <psi_i|H|psi_j>
                # !!!!
                H = self.td_hamiltonian.hamiltonian
                self.td_density.update()
                self.td_hamiltonian.update( self.td_density.get_density(), 
                                            self.time )
                self.td_overlap.update()

                if self.hpsit is None:
                    self.hpsit = self.gd.zeros( len(self.kpt_u[0].psit_nG), 
                                                dtype=complex )
                if self.eps_tmp is None:
                    self.eps_tmp = npy.zeros( len(self.kpt_u[0].eps_n), 
                                              dtype=complex )

                for kpt in self.kpt_u:
                    self.td_hamiltonian.apply(kpt, kpt.psit_nG, self.hpsit)
                    self.mblas.multi_zdotc(self.eps_tmp, kpt.psit_nG, self.hpsit, len(self.kpt_u[0].psit_nG)) 
                    self.eps_tmp *= self.gd.dv
                    #print 'Eps_n = ', self.eps_tmp
                    kpt.eps_n = self.eps_tmp.real

                self.occupation.calculate_band_energy(self.kpt_u)
                # Nonlocal
                xcfunc = H.xc.xcfunc
                self.Enlxc = xcfunc.get_non_local_energy()
                self.Enlkin = xcfunc.get_non_local_kinetic_corrections()
                # PAW
                self.Ekin = H.Ekin + self.occupation.Eband + self.Enlkin
                self.Epot = H.Epot
                self.Eext = H.Eext
                self.Ebar = H.Ebar
                self.Exc = H.Exc + self.Enlxc
                self.Etot = self.Ekin + self.Epot + self.Ebar + self.Exc
                T = time.localtime()
                if rank == 0:
                    iter_text = """iter: %3d  %02d:%02d:%02d %11.2f\
   %13.6f %9.1f %10d"""
                    self.text(iter_text % 
                              (i, T[3], T[4], T[5],
                               self.time * self.autime_to_attosec,
                               self.Etot, log(abs(norm))/log(10),
                               niterpropagator))

                    self.txt.flush()


            # propagate
            niterpropagator = self.propagator.propagate(self.kpt_u, self.time,
                                                        time_step)
            self.time += time_step


            # restart data
            if restart_file is not None and ( (i+1) % dump_interval == 0 ):
                self.write(restart_file, 'all')
                if rank == 0:
                    print 'Wrote restart file.'
                    print i, ' iterations done. Current time is ', \
                        self.time * self.autime_to_attosec, ' as.' 
                    # print 'Warning: Writing restart files in TDDFT does not work yet.'
                    # print 'Continuing without writing restart file.'

        # close dipole moment file
        if dipole_moment_file is not None:
            if rank == 0:
                dm_file.close()

        if restart_file is not None:
            self.write(restart_file, 'all')

    # exp(ip.r) psi
    def absorption_kick(self, kick_strength):
        """ Delta absoprtion kick for photoabsorption spectrum.

        Parameters
        ----------
        kick_strength: [float, float, float]
            Strength of the kick, e.g., [0.0, 0.0, 1e-3]
        
        """
        if rank == 0:
            self.text('Delta kick = ', kick_strength)
        self.kick_strength = npy.array(kick_strength)

        abs_kick = \
            AbsorptionKick( AbsorptionKickHamiltonian( self.pt_nuclei,
                                                       npy.array(kick_strength,
                                                                 dtype=float) ),
                            self.td_overlap, self.solver, None,
                            self.gd, self.timer )
        abs_kick.kick(self.kpt_u)

    def __del__(self):
        """Destructor"""
        PAW.__del__(self)

# Function for calculating photoabsorption spectrum
#def photoabsorption_spectrum(dipole_moment_file, spectrum_file, fwhm = 0.5, delta_omega = 0.05, max_energy = 50.0):
def photoabsorption_spectrum(dipole_moment_file, spectrum_file, folding='Gauss', width=0.2123, e_min=0.0, e_max=30.0, delta_e=0.05):
    """ Calculates photoabsorption spectrum from the time-dependent
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
        raise RuntimeError( 'Error in photoabsorption_spectrum: '
                            'Only Gaussian folding is currently supported.' )
    
    
    if rank == 0:
        print ('Calculating photoabsorption spectrum from file "%s"' % dipole_moment_file)

        f_file = file(spectrum_file, 'w')
        dm_file = file(dipole_moment_file, 'r')
        lines = dm_file.readlines()
        dm_file.close()
        # Read kick strength
        columns = lines[0].split('[')
        columns = columns[1].split(']')
        columns = columns[0].split(',')
        kick_strength = npy.array([eval(columns[0]),eval(columns[1]),eval(columns[2])], dtype=float)
        strength = npy.array(kick_strength, dtype=float)
        # Remove first two lines
        lines.pop(0)
        lines.pop(0)
        print 'Using kick strength = ', strength
        # Continue with dipole moment data
        n = len(lines)
        dm = npy.zeros((n,3),dtype=float)
        time = npy.zeros((n,),dtype=float)
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
        dw = delta_e / 27.211
        # f(w) = Nw exp(-w^2/2sigma^2)
        #sigma = fwhm / 27.211 / (2.* npy.sqrt(2.* npy.log(2.0)))
        # f(t) = Nt exp(-t^2/2gamma^2)
        #gamma = 1.0 / sigma
        sigma = width/27.211
        gamma = 1.0 / sigma
        fwhm = sigma * (2.* npy.sqrt(2.* npy.log(2.0)))
        kick_magnitude = npy.sum(strength**2)

        # write comment line
        f_file.write('# Photoabsorption spectrum from real-time propagation\n')
        f_file.write('# GPAW version: ' + str(version) + '\n')
        f_file.write('# Total time = %lf fs, Time step = %lf as\n' % (n * dt * 24.1888/1000.0, dt *  24.1888))
        f_file.write('# Kick = [%lf,%lf,%lf]\n' % (kick_strength[0], kick_strength[1], kick_strength[2]))
        f_file.write('# %sian folding, Width = %lf eV = %lf Hartree <=> FWHM = %lf eV\n' % (folding, sigma*27.211, sigma, fwhm*27.211))

        f_file.write('#  om (eV) %14s%20s%20s\n' % ('Sx', 'Sy', 'Sz'))
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
        
        print ('Calculated photoabsorption spectrum saved to file "%s"' % spectrum_file)

            
    # Make static method
    # photoabsorption_spectrum=staticmethod(photoabsorption_spectrum)
        
