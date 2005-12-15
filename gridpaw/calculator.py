# pylint: disable-msg=W0142,C0103,E0201

# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.


"""ASE-calculator interface."""


import os
import sys
import tempfile
import time
import weakref

import Numeric as num
from ASE.Units import units, Convert
try:
    from ASE.Trajectories.NetCDFTrajectory import NetCDFTrajectory
except ImportError:
    print "No Netcdf installed"
from ASE.Utilities.MonkhorstPack import MonkhorstPack

from gridpaw.utilities import DownTheDrain, check_unit_cell
from gridpaw.mpi_paw import MPIPaw
from gridpaw.startup import create_paw_object
from gridpaw.version import version
import gridpaw.utilities.timing as timing
import gridpaw


class Calculator:
    """This is the ASE-calculator frontend for doing a PAW calculation.

    The calculator object controlls a paw object that does the actual
    work.  The paw object can run in serial or in parallel, the
    calculator interface will allways be the same."""

    # Default values for all possible keyword parameters:
    parameters = {'nbands': None,
                  'xc': 'LDA',
                  'kpts': None,
                  'spinpol': None,
                  'gpts': None,
                  'h': None,
                  'charge': 0,
                  'usesymm': True,
                  'width': None,
                  'mix': 0.25,
                  'old': 3,
                  'hund': False,
                  'lmax': 0,
                  'fixdensity': False,
                  'idiotproof': True,
                  'tolerance': 1.0e-9,
                  'out': '-',
                  'hosts': None,
                  'parsize': None,
                  'softgauss': True,
                  'order': 5,
                  'onohirose': 5,
                  }
    
    def __init__(self, **kwargs):
        """ASE-calculator interface.

        The following parameters can be used: `nbands`, `xc`, `kpts`,
        `spinpol`, `gpts`, `h`, `charge`, `usesymm`, `width`, `mix`,
        `old`, `hund`, `lmax`, `fixdensity`, `idiotproof`,
        `tolerance`, `out`, `hosts`, `parsize`, `softgauss`, `order`
        and `onohirose`.  If you don't specify any parameters, you
        will get:

        Defaults: neutrally charged, LDA, gamma-point calculation, a
        reasonable grid-spacing, zero Kelvin elctronic temperature,
        and the number of bands will be half the number of valence
        electrons plus 3 extra bands.  The calculation will be
        spin-polarized if and only if one or more of the atoms have
        non-zero magnetic moments.  Text output will be written to
        standard output.

        For a non-gamma point calculation, the electronic temperature
        will be 0.1 eV (energies are extrapolated to zero Kelvin) and
        all symmetries will be used to reduce the number of
        **k**-points,"""

        self.t0 = time.time()

        self.paw = None

        # Set default parameters and adjust with user parameters:
        self.Set(**Calculator.parameters)
        self.Set(**kwargs)

        out = self.out
        print >> out
        print >> out, '   _                    '
        print >> out, '  | |  ___ ___ ___ _ _  '
        print >> out, ' |   ||_  |  _|  _| | | '
        print >> out, '  | |_| . |_  |_  | | | '
        print >> out, '  |___|___|___|___|___| '
        print >> out
        print >> out, '  _  _  _           '
        print >> out, ' (_||_)(_|\/\/   -  ', version
        print >> out, '  _||               '
        print >> out

        uname = os.uname()
        print >> out, 'User:', os.getenv('USER') + '@' + uname[1]
        print >> out, 'Date:', time.asctime()
        print >> out, 'Arch:', uname[4]
        print >> out, 'Pid: ', os.getpid()
        print >> out

        lengthunit = units.GetLengthUnit()
        energyunit = units.GetEnergyUnit()
        print >> out, 'units:', lengthunit, 'and', energyunit
        self.a0 = Convert(1, 'Bohr', lengthunit)
        self.Ha = Convert(1, 'Hartree', energyunit)

        self.reset()

        self.tempfile = None

        self.parallel_cputime = 0.0

    def reset(self):
        """Delete PAW-object."""
        self.stop_paw()
        self.restart_file = None     # ??????
        self.pos_ac = None
        self.cell_cc = None
        self.periodic_c = None
        self.Z_a = None

    def set_out(self, out):
        """Set the stream for text output.

        If `out` is not a stream-object, then it must be one of:

        ``None``:
          Throw output away.
        ``'-'``:
          Use standard-output (``sys.stduot``).
        A filename:
          open a new file.
        """
        
        if out is None:
            out = DownTheDrain()
        elif out == '-':
            out = sys.stdout
        elif type(out) is str:
            out = open(out, 'w')
        self.out = out

    def set_kpts(self, bzk_kc):
        """Set the scaled k-points. 
        
        ``kpts`` should be an array of scaled k-points."""
        
        if bzk_kc is None:
            bzk_kc = (1, 1, 1)
        if type(bzk_kc[0]) is int:
            bzk_kc = MonkhorstPack(bzk_kc)
        self.bzk_kc = num.array(bzk_kc)
        self.reset()
     
    def update(self):
        """Update PAW calculaton if needed."""
        atoms = self.atoms()

        if self.paw is not None and self.lastcount == atoms.GetCount():
            # Nothing to do:
            return

        if (self.paw is None or
            atoms.GetAtomicNumbers() != self.Z_a or
            atoms.GetUnitCell() != self.cell_cc or
            atoms.GetBoundaryConditions() != self.periodic_c):
            # Drastic changes:
            self.initialize_paw_object()
            self.find_ground_state()
        else:
            # Something else has changed:
            if (atoms.GetCartesianPositions() != self.pos_ac):
                # It was the positions:
                self.find_ground_state()
            else:
                # It was something that we don't care about - like
                # velocities, masses, ...
                pass
    
    def initialize_paw_object(self):
        """Initialize PAW-object."""
        atoms = self.atoms()

        pos_ac = atoms.GetCartesianPositions()
        Z_a = atoms.GetAtomicNumbers()
        cell_cc = num.array(atoms.GetUnitCell())
        periodic_c = atoms.GetBoundaryConditions()
        try:
            angle = atoms.GetRotationAngle()
        except AttributeError:
            angle = None
	
        # Check that the cell is orthorhombic:
        check_unit_cell(cell_cc)
        # Get the diagonal:
        cell_c = num.diagonal(cell_cc)

        magmoms = [atom.GetMagneticMoment() for atom in atoms]

        # Get rid of the old calculator before the new one is created:
        self.stop_paw()

        # Maybe parsize has been set by command line argument
        # --gridpaw-domain-decomposition? (see __init__.py)
        if gridpaw.parsize is not None:
            # Yes, it was:
            self.parsize = gridpaw.parsize
            
        args = [self.out,
                self.a0, self.Ha,
                pos_ac, Z_a, magmoms, cell_c, periodic_c, angle,
                self.h, self.gpts, self.xc,
                self.nbands, self.spinpol, self.width,
                self.bzk_kc,
                self.softgauss,
                self.order,
                self.usesymm,
                self.mix,
                self.old,
                self.fixdensity,
                self.idiotproof,
                self.hund,
                self.lmax,
                self.onohirose,
                self.tolerance,
                self.parsize,
                self.restart_file,
                ]

        if gridpaw.hosts is not None:
            # The hosts have been set by one of the command line arguments
            # --gridpaw-hosts or --gridpaw-hostfile (see __init__.py):
            self.hosts = gridpaw.hosts

        if self.hosts is None:
            if os.environ.has_key('PBS_NODEFILE'):
                # This job was submitted to the PBS queing system.  Get
                # the hosts from the PBS_NODEFILE environment variable:
                self.hosts = os.environ['PBS_NODEFILE']
                
                if len(open(self.hosts).readlines()) == 1:
                    # Only one node - don't do a parallel calculation:
                    self.hosts = None
            elif os.environ.has_key('NSLOTS'):
                # This job was submitted to the Grid Engine queing system:
                self.hosts = int(os.environ['NSLOTS'])
            elif os.environ.has_key('LOADL_PROCESSOR_LIST'):
                self.hosts = 'dummy file-name'

        if type(self.hosts) is int:
            if self.hosts == 1:
                # Only one node - don't do a parallel calculation:
                self.hosts = None
            else:
                self.hosts = [os.uname()[1]] * self.hosts
            
        if type(self.hosts) is list:
            # We need the hosts in a file:
            self.tempfile = tempfile.mktemp()
            f = open(self.tempfile, 'w')
            for host in self.hosts:
                print >> f, host
            f.close()
            self.hosts = self.tempfile
            # (self.tempfile is removed in Calculator.__del__)

        # What kind of calculation should we do?
        if self.hosts is None:
            # Serial:
            self.paw = create_paw_object(*args)
        else:
            # Parallel:
            self.paw = MPIPaw(self.hosts, *args)
            
    def find_ground_state(self):
        """Tell PAW-object to start iterating ..."""
        atoms = self.atoms()
        pos_ac = atoms.GetCartesianPositions()
        Z_a = atoms.GetAtomicNumbers()
        cell_cc = atoms.GetUnitCell()
        periodic_c = atoms.GetBoundaryConditions()
        try:
            angle = atoms.GetRotationAngle()
        except AttributeError:
            angle = None
	
        # Check that the cell is orthorhombic:
        check_unit_cell(cell_cc)

        self.paw.find_ground_state(pos_ac, num.diagonal(cell_cc), angle)
        
        # Save the state of the atoms:
        self.lastcount = atoms.GetCount()
        self.pos_ac = pos_ac
        self.cell_cc = cell_cc
        self.periodic_c = periodic_c
        self.angle = angle
        self.Z_a = Z_a

        timing.update()

    def stop_paw(self):
        """Delete PAW-object."""
        if isinstance(self.paw, MPIPaw):
            # Stop old MPI calculation and get total CPU time for all CPUs:
            self.parallel_cputime += self.paw.stop()
        self.paw = None
        
    def __del__(self):
        """Destructor:  Write timing output before closing."""
        if self.tempfile is not None:
            # Delete hosts file:
            os.remove(self.tempfile)

        self.stop_paw()
        
        # Get CPU time:
        c = self.parallel_cputime + timing.clock()
                
        if c > 1.0e99:
            print >> self.out, 'cputime : unknown!'
        else:
            print >> self.out, 'cputime : %f' % c

        print >> self.out, 'walltime: %f' % (time.time() - self.t0)
        print >> self.out, 'date    :', time.asctime()

    #####################
    ## User interface: ##
    #####################
    def Set(self, **kwargs):
        """Set keyword parameters.

        Works like this:

        >>> calc.Set(out='stuff.txt')
        >>> calc.Set(nbands=24, spinpol=True)

        """

        for name, value in kwargs.items():
            method_name = 'set_' + name
            if hasattr(self, method_name):
                getattr(self, method_name)(value)
            else:
                if name not in self.parameters:
                    raise RuntimeError('Unknown keyword: ' + name)
                setattr(self, name, value)
        self.reset()
            
    def GetReferenceEnergy(self):
        """Get reference energy for all-electron atoms."""
        return self.paw.Eref * self.Ha

    def GetEnsembleCoefficients(self):
        """Get BEE ensemble coefficients.

        See The ASE manual_ for details.

        .. _manual: https://wiki.fysik.dtu.dk/ase/Utilities
                    #bayesian-error-estimate-bee
        """

        E = self.GetPotentialEnergy()
        E0 = self.GetXCDifference('XC-9-1.0')
        coefs = (E + E0,
                 self.GetXCDifference('XC-0-1.0') - E0,
                 self.GetXCDifference('XC-1-1.0') - E0,
                 self.GetXCDifference('XC-2-1.0') - E0)
        print >> self.out, 'BEE: (%.9f, %.9f, %.9f, %.9f)' % coefs
        return num.array(coefs)

    def GetXCDifference(self, xcname):
        """Calculate non-seflconsistent XC-energy difference."""
        self.update()
        return self.paw.get_xc_difference(xcname)

    def Write(self, filename):
        """Write current state to a netCDF file."""
        traj = NetCDFTrajectory(filename, self.atoms())

        # Write the atoms:
        traj.Update()

        # Dig out the netCDF file:
        nc = traj.nc

        nc.history = 'gridpaw restart file'
        nc.version = version

        traj.Close()
        self.paw.write_netcdf(filename)

    def GetNumberOfIterations(self):
        """Return the number of SCF iterations."""
        return self.paw.niter

    ####################
    ## ASE interface: ##
    ####################
    def GetPotentialEnergy(self, force_consistent=False):
        """Return the energy for the current state of the ListOfAtoms."""
        self.update()
        return self.paw.get_total_energy(force_consistent)

    def GetCartesianForces(self):
        """Return the forces for the current state of the ListOfAtoms."""
        self.update()
        return self.paw.get_cartesian_forces()
      
    def GetStress(self):
        """Return the stress for the current state of the ListOfAtoms."""
        raise NotImplementedError

    def _SetListOfAtoms(self, atoms):
        """Make a weak reference to the ListOfAtoms."""
        self.lastcount = -1
        self.atoms = weakref.ref(atoms)
        self.stop_paw()

    def GetNumberOfBands(self):
        """Return the number of bands."""
        return self.nbands 
  
    def GetXCFunctional(self):
        """Return the XC-functional identifier.
        
        'LDA', 'PBE', ..."""
        
        return self.xc 
 
    def GetBZKPoints(self):
        """Return the k-points."""
        return self.bzk_kc
 
    def GetSpinPolarized(self):
        """Is it a spin-polarized calculation?"""
        return self.paw.nspins == 2
    
    def GetIBZKPoints(self):
        """Return k-points in the irreducible part of the Brillouin zone."""
        return self.paw.get_ibz_kpoints()

    # Alternative name:
    GetKPoints = GetIBZKPoints
 
    def GetExactExchange(self,decompose = False):
        from gridpaw.exx import get_exact_exchange as exx
        paw = self.paw
        paw.timer.start('exx')
        exx = exx(paw.wf, paw.nuclei, paw.gd, decompose)
        paw.timer.stop('exx')
        return exx * self.Ha
    
    def GetXCEnergy(self):
        return self.paw.Exc * self.Ha

    def GetIBZKPointWeights(self):
        """Weights of the k-points. 
        
        The sum of all weights is one."""
        
        return self.weights

    def GetDensityArray(self):
        """Return pseudo-density array."""
        return self.paw.get_density_array()

    def GetWaveFunctionArray(self, band=0, kpt=0, spin=0):
        """Return pseudo-wave-function array."""
        return self.paw.get_wave_function_array(band, kpt, spin)

    def GetWannierLocalizationMatrix(self, G_I, **args):
        """Calculate integrals for maximally localized Wannier functions."""
        c = G_I.index(1)
        return self.paw.get_wannier_integrals(c)

    def GetMagneticMoment(self):
        """Return the magnetic moment."""
        return self.paw.magmom

    def GetFermiLevel(self):
        """Return the Fermi-level."""
        return self.paw.get_fermi_level()

    def GetElectronicStates(self):
        """Return electronic-state object."""
        from ASE.Utilities.ElectronicStates import ElectronicStates
        self.Write('tmp27.nc')
        return ElectronicStates('tmp27.nc')
    
    # @staticmethod  # (Python 2.4 style)
    def ReadAtoms(filename, **overruling_kwargs):
        """Read state from a netCDF file."""
        traj = NetCDFTrajectory(filename)
        atoms = traj.GetListOfAtoms()
        nc = traj.nc
        vars = nc.variables
        dims = nc.dimensions

        kwargs = {'nbands':      dims['nbands'],
                  'xc':          nc.XCFunctional,
                  'kpts':        num.array(vars['BZKPoints']),
                  'spinpol':     (dims['nspins'] == 2),
                  'gpts':        (dims['ngptsx'],
                                  dims['ngptsy'],
                                  dims['ngptsz']),
                  'usesymm':     nc.UseSymmetry[0],
                  'width':       nc.FermiWidth[0],
                  'mix':         nc.Mix[0],
                  'old':         nc.Old[0],
                  'lmax':        nc.MaximumAngularMomentum[0],
                  'fixdensity':  nc.FixDensity[0],
                  'idiotproof':  nc.IdiotProof[0],
                  'tolerance':   nc.Tolerance[0]}
        
        kwargs.update(overruling_kwargs)
        calc = Calculator(**kwargs)

        # Get the forces from the old calculation:
        F_ac = atoms.GetCartesianForces()
        
        atoms.SetCalculator(calc)

        # Wave functions and other stuff will be read from 'filename'
        # later, when requiered:
        calc.restart_file = filename
        calc.initialize_paw_object()
        calc.paw.set_forces(F_ac)
        calc.lastcount = atoms.GetCount()
        return atoms

    # Make ReadAtoms a static method:
    ReadAtoms = staticmethod(ReadAtoms)

    def GetListOfAtoms(self):
        return self.atoms()
 
