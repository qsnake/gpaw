.. _timepropagation:

======================
Time-propagation TDDFT
======================


------------
Ground state
------------

To obtain the ground state for TDDFT, one has to just do a standard ground state 
with a larger simulation box. A proper distance from any atom to edge of the 
simulation box is problem dependent, but a minimum reasonable value is around
6 Ångströms and recommended between 8-10 Ång. In TDDFT, one can use larger 
grid spacing than for geometry optimization. For example, if you use h=0.25
for geometry optimization, try h=0.3 for TDDFT. This saves a lot of time. 

A good way to start is to use too small box (vacuum=6.0), too large grid 
spacing (h=0.35), and too large time step (dt=16.0). Then repeat the simulation
with better parameter values and compare. Probably lowest peaks are already 
pretty good, and far beyond the ionization limit, in the continuum, the spectrum 
is not going to converge anyway. The first run takes only fraction of 
the time of the second run.

For a parallel-over-states TDDFT calculation, you must choose the number 
of states so, that these can be distributed equally to processors. For 
example, if you have 79 occupied states and you want to use 8 processes 
in parallelization over states, add one unoccupied state to get 80 states 
in total.


Ground state example::

  # Standard magic
  from ase import *
  from gpaw import *
  
  # Beryllium atom
  atoms = Atoms( symbols = 'Be', 
                 positions = [(0,0,0)],
                 pbc = False )
  
  # Add 6.0 ang vacuum around the atom
  atoms.center(vacuum=6.0)
  
  # Create GPAW calculator
  calc = Calculator(nbands=1, h=0.3)
  # Attach calculator to atoms
  atoms.set_calculator(calc)
  
  # Calculate the ground state
  energy = atoms.get_potential_energy()
  
  # Save the ground state
  calc.write('be_gs.gpw', 'all')



--------------------------------
Optical photoabsorption spectrum
--------------------------------

Optical photoabsorption spectrum can be obtained by applying a weak 
delta pulse of dipole electric field, and then letting the system evolve
freely. Time-step around 4.0-8.0 attoseconds is reasonable. Total simulation
time should be few tens of picoseconds depending on the desired resolution.
Typically in experiments, the spherically averaged spectrum is measured.
To obtain this, one must repeat the time-propagation to each Cartesian 
direction and average over them.

Example::

  from gpaw.tddft import *
  
  time_step = 8.0                  # 1 attoseconds = 0.041341 autime
  iterations = 2500                # 2500 x 8 as => 20 fs
  kick_strength = [0.0,0.0,1e-3]   # Kick to z-direction
  
  # Read ground state
  td_atoms = TDDFT('be_gs.gpw')
  
  # Kick with a delta pulse to z-direction
  td_atoms.absorption_kick(kick_strength=kick_strength)
  
  # Propagate, save the time-dependent dipole moment to 'be_dm.dat',
  # and use 'be_td.gpw' as restart file
  td_atoms.propagate(time_step, iterations, 'be_dm.dat', 'be_td.gpw')
  
  # Calculate photoabsorption spectrum and write it to 'be_spectrum_z.dat'
  photoabsorption_spectrum('be_dm.dat', 'be_spectrum_z.dat')


Keywords for TDDFT:

===================== =============== ============== =====================================
Keyword               Type            Default        Description
===================== =============== ============== =====================================
``ground_state_file`` ``string``                     Name of the ground state file
``td_potential``      ``TDPotential`` ``None``       Time-dependent external potential
``propagator``        ``string``      ``'SICN'``     Time-propagator
``solver``            ``string``      ``'CSCG'``     Linear equation solver
``tolerance``         ``float``       ``1e-8``       Tolerance for linear solver
===================== =============== ============== =====================================

Keywords for absorption_kick:

================== =============== ================== =====================================
Keyword            Type            Default            Description
================== =============== ================== =====================================
``kick_strength``  ``float[3]``    ``[0,0,1e-3]``     Kick strength
================== =============== ================== =====================================

Keywords for propagate:

====================== =========== =========== ================================================
Keyword                Type        Default     Description
====================== =========== =========== ================================================
``time_step``          ``float``               Time step in attoseconds (1 autime = 24.188 as)
``iterations``         ``integer``             Iterations
``dipole_moment_file`` ``string``  ``None``    Name of the dipole moment file
``restart_file``       ``string``  ``None``    Name of the restart file
``dump_interal``       ``integer`` ``500``     How often restart file is written
====================== =========== =========== ================================================

Keywords for photoabsorption_spectrum:

====================== ============ ============== ===============================================
Keyword                Type         Default        Description
====================== ============ ============== ===============================================
``dipole_moment_file`` ``string``                  Name of the dipole moment file
``spectrum_file``      ``string``                  Name of the spectrum file
``folding``            ``string``   ``Gauss``      Gaussian folding (or Lorentzian in future)
``width``              ``float``    ``0.2123``     Width of the Gaussian/Lorentzian (in eV)
``e_min``              ``float``    ``0.0``        Lowest energy shown in spectrum (in eV)
``e_max``              ``float``    ``3.0``        Highest energy shown in spectrum (in eV)
``delta_e``            ``float``    ``0.05``       Resolution of energy in spectrum (in eV)
====================== ============ ============== ===============================================
