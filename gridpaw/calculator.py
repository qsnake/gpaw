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
if os.uname()[4] == 'i686':
    from ASE.Trajectories.NetCDFTrajectory import NetCDFTrajectory
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
        `spinpol`, `gpts`, `h`, `usesymm`, `width`, `mix`, `old`,
        `hund`, `lmax`, `fixdensity`, `idiotproof`, `tolerance`,
        `out`, `hosts`, `parsize`, `softgauss`, `order` and
        `onohirose`.  If you don't specify any parameters, you will get:

        Defaults: LDA, gamma-point calculation, a reasonable
        grid-spacing, zero Kelvin elctronic temperature, and the
        number of bands will be half the number of valence electrons
        plus 3 extra bands.  The calculation will be spin-polarized if
        and only if one or more of the atoms have non-zero magnetic
        moments.  Text output will be written to standard output.

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
        print >> out, '  _  _ o _| _  _           '
        print >> out, ' (_||  |(_||_)(_|\/\/   -  ', version
        print >> out, '  _|       |               '
        print >> out

        uname = os.uname()
        print >> out, 'User:', os.getenv('USER') + '@' + uname[1]
        print >> out, 'Date:', time.asctime()
        print >> out, 'Arch:', uname[4]
        print >> out, 'Pid: ', os.getpid()

        self.reset()

        self.tempfile = None

        self.parallel_cputime = 0.0

    def reset(self):
        self.stop_paw()
        self.restart_file = None     # ??????
        self.positions = None
        self.cell = None
        self.bc = None
        self.numbers = None

    def set_out(self, out):
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
     
    def update_energy_and_forces(self):
        atoms = self.atoms()

        if self.paw is not None and self.lastcount == atoms.GetCount():
            # Nothing to do:
            return

        if (self.paw is None or
            atoms.GetAtomicNumbers() != self.numbers or
            atoms.GetUnitCell() != self.cell or
            atoms.GetBoundaryConditions() != self.bc):
            # Drastic changes:
            self.initialize()
            self.calculate()
        else:
            # Something else has changed:
##             if (atoms.GetUnitCell() != self.cell or
##                 atoms.GetCartesianPositions() != self.positions):
            if (atoms.GetCartesianPositions() != self.positions):
                # It was the positions:
                self.calculate()
            else:
                # It was something that we don't care about - like
                # velocities, masses, ...
                pass
    
    def initialize(self):
        atoms = self.atoms()

        lengthunit = units.GetLengthUnit()
        energyunit = units.GetEnergyUnit()
        print >> self.out, 'units:', lengthunit, 'and', energyunit
        a0 = Convert(1, 'Bohr', lengthunit)
        Ha = Convert(1, 'Hartree', energyunit)

        positions = atoms.GetCartesianPositions()
        numbers = atoms.GetAtomicNumbers()
        cell = num.array(atoms.GetUnitCell())
        bc = atoms.GetBoundaryConditions()
        try:
            angle = atoms.GetRotationAngle()
        except AttributeError:
            angle = None
	
        # Check that the cell is orthorhombic:
        check_unit_cell(cell)
        # Get the diagonal:
        cell = num.diagonal(cell)

        magmoms = [atom.GetMagneticMoment() for atom in atoms]

        # Get rid of the old calculator before the new one is created:
        self.stop_paw()

        # Maybe parsize has been set by command line argument
        # --gridpaw-domain-decomposition? (see __init__.py)
        if gridpaw.parsize is not None:
            # Yes, it was:
            self.parsize = gridpaw.parsize
            
        args = [self.out,
                a0, Ha,
                positions, numbers, magmoms, cell, bc, angle,
                self.h, self.gpts, self.xc,
                self.nbands, self.spinpol, self.width,
                self.bzk_kc,
                self.softgauss,
                self.order,
                self.usesymm,
                self.mix, self.old,
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
            # (self.tempfile is unlinked in Calculator.__del__)

        # What kind of calculation should we do?
        if self.hosts is None:
            # Serial:
            self.paw = create_paw_object(*args)
        else:
            # Parallel:
            self.paw = MPIPaw(self.hosts, *args)
            
    def calculate(self):
        atoms = self.atoms()
        positions = atoms.GetCartesianPositions()
        numbers = atoms.GetAtomicNumbers()
        cell = atoms.GetUnitCell()
        bc = atoms.GetBoundaryConditions()
        try:
            angle = atoms.GetRotationAngle()
        except AttributeError:
            angle = None
	
        # Check that the cell is orthorhombic:
        check_unit_cell(cell)

        self.paw.calculate(positions, num.diagonal(cell), angle)
        
        # Save the state of the atoms:
        self.lastcount = atoms.GetCount()
        self.positions = positions
        self.cell = cell
        self.bc = bc
        self.angle = angle
        self.numbers = numbers

        timing.update()

    def stop_paw(self):
        if isinstance(self.paw, MPIPaw):
            # Stop old MPI calculation and get total CPU time for all CPUs:
            self.parallel_cputime += self.paw.stop()
        self.paw = None
        
    def __del__(self):
        if self.tempfile is not None:
            # Delete hosts file:
            os.unlink(self.tempfile)

        self.stop_paw()
        
        # Get CPU time:
        c = self.parallel_cputime + timing.clock()
                
        if c > 1.0e99:
            print >> self.out, 'cputime : unknown!'
        else:
            print >> self.out, 'cputime : %f' % c

        print >> self.out, 'walltime: %f' % (time.time() - self.t0)
        print >> self.out, 'date    :', time.asctime()

    ###################
    # User interface: #
    ###################
    def Set(self, **kwargs):
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
        return self.paw.get_reference_energy()

    def GetEnsembleCoefficients(self):
        E = self.GetPotentialEnergy()
        E0 = self.GetXCDifference('XC-9-1.0')
        coefs = (E + E0,
                 self.GetXCDifference('XC-0-1.0') - E0,
                 self.GetXCDifference('XC-1-1.0') - E0,
                 self.GetXCDifference('XC-2-1.0') - E0)
        print >> self.out, 'ensemble: (%.9f, %.9f, %.9f, %.9f)' % coefs
        return num.array(coefs)

    def GetXCDifference(self, xcname):
        self.update_energy_and_forces()
        return self.paw.get_xc_difference(xcname)

    def Write(self, filename):
        traj = NetCDFTrajectory(filename, self.atoms())

        # Write the atoms:
        traj.Update()

        # Dig out the netCDF file:
        nc = traj.nc

        nc.history = 'gridpaw restart file'
        nc.version = version

        traj.Close()
        self.paw.write_netcdf(filename)

    def GetGGAHistogram(self, smax=10.0, nbins=200):
        return self.paw.get_gga_histogram(smax, nbins)

    def GetNumberOfIterations(self):
        return self.paw.get_number_of_iterations()

    ##################
    # ASE interface: #
    ##################
    def GetPotentialEnergy(self, force_consistent=False):
        """Return the energy for the current state of the ListOfAtoms."""
        self.update_energy_and_forces()
        return self.paw.get_potential_energy(force_consistent)

    def GetCartesianForces(self):
        """Return the forces for the current state of the ListOfAtoms."""
        self.update_energy_and_forces()
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
        return self.paw.get_spinpol()
    
    def GetIBZKPoints(self):
        """Return k-points in the irreducible part of the Brillouin zone."""
        return self.paw.get_ibz_kpoints()

    def GetExactExchange(self):
        paw = self.paw
        paw.timer.start('exx')
        exx = paw.wf.exx(paw.nuclei, paw.gd)
        paw.timer.stop('exx')
        return exx*paw.Ha
    
    def GetXCEnergy(self):
        return self.paw.Exc*self.paw.Ha

    # Alternative name:
    GetKPoints = GetIBZKPoints
 
    def GetIBZKPointWeights(self):
        """Weights of the k-points. 
        
        The sum of all weights is one."""
        
        return self.weights

    def GetDensityArray(self):
        return self.paw.get_density_array()

    def GetWaveFunctionArray(self, band=0, kpt=0, spin=0):
        return self.paw.get_wave_function_array(band, kpt, spin)

    def GetWannierLocalizationMatrix(self, G_I=None, nbands=None, spin=None,
                                     kpoint=0, nextkpoint=0, dirG=0):
        c = G_I.index(1)
        return self.paw.get_wannier_integral(c)

    def GetMagneticMoment(self):
        return self.paw.get_magnetic_moment()

    def GetFermiLevel(self):
        return self.paw.get_fermi_level()

    def GetElectronicStates(self):
        from ASE.Utilities.ElectronicStates import ElectronicStates
        self.Write('tmp27.nc')
        return ElectronicStates('tmp27.nc')
    
    # @staticmethod
    def ReadAtoms(filename, **overruling_kwargs):
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
        
        atoms.SetCalculator(calc)

        # Wave functions and other stuff will be read from 'filename'
        # later, when requiered:
        calc.restart_file = filename
        calc.initialize()
        calc.lastcount = atoms.GetCount()
        return atoms

    # Make ReadAtoms a static method:
    ReadAtoms = staticmethod(ReadAtoms)
