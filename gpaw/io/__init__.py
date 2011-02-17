import os
import os.path

try:
    from ase.units import AUT # requires rev1839 or later
except ImportError:
    from ase.units import second, alpha, _hbar, _me, _c
    AUT = second * _hbar / (alpha**2 * _me * _c**2)
    del second, alpha, _hbar, _me, _c

from ase.units import Bohr, Hartree
from ase.data import atomic_names
from ase.atoms import Atoms
import numpy as np

import gpaw.mpi as mpi
import os,time,tempfile

def open(filename, mode='r', comm=mpi.world):
    if filename.endswith('.nc'):
        import gpaw.io.netcdf as io
    elif filename.endswith('.db'):
        import gpaw.io.cmr_io as io
    elif filename.endswith('.hdf5'):
        import gpaw.io.hdf5 as io
    else:
        if not filename.endswith('.gpw'):
            filename += '.gpw'
        import gpaw.io.tar as io

    if mode == 'r':
        return io.Reader(filename, comm)
    elif mode == 'w':
        return io.Writer(filename, comm)
    else:
        raise ValueError("Illegal mode!  Use 'r' or 'w'.")

def wave_function_name_template(mode):
    try:
        ftype, template = mode.split(':')
    except:
        ftype = mode
        template = 'wfs/psit_Gs%dk%dn%d'
    return ftype, template

def write(paw, filename, mode, cmr_params=None, **kwargs):
    """Write state to file.
    
    The `mode` argument should be one of:

    ``''``:
      Don't write the wave functions.
    ``'all'``:
      Write also the wave functions to the file.
    ``'nc'`` or ``'gpw'``:
      Write wave functions as separate files (the default filenames
      are ``'psit_Gs%dk%dn%d.nc' % (s, k, n)`` for ``'nc'``, where
      ``s``, ``k`` and ``n`` are spin, **k**-point and band indices). XXX
    ``'nc:mywfs/psit_Gs%dk%dn%d'``:
      Defines the filenames to be ``'mywfs/psit_Gs%dk%dn%d' % (s, k, n)``.
      The directory ``mywfs`` is created if not present. XXX

    cmr_params specifies the parameters that should be used for CMR.
    (Computational Materials Repository)

    Please note: mode argument is ignored by for CMR.
    """

    wfs = paw.wfs
    scf = paw.scf
    density = paw.density
    hamiltonian = paw.hamiltonian

    world = paw.wfs.world
    domain_comm = wfs.gd.comm
    kpt_comm = wfs.kpt_comm
    band_comm = wfs.band_comm

    master = (world.rank == 0)

    atoms = paw.atoms
    natoms = len(atoms)

    magmom_a = paw.get_magnetic_moments()

    hdf5 = filename.endswith('.hdf5')

    if master or hdf5:
        w = open(filename, 'w', world)
        
        w['history'] = 'GPAW restart file'
        w['version'] = '0.8'
        w['lengthunit'] = 'Bohr'
        w['energyunit'] = 'Hartree'

        try:
            tag_a = atoms.get_tags()
            if tag_a is None:
                raise KeyError
        except KeyError:
            tag_a = np.zeros(natoms, int)

        w.dimension('natoms', natoms)
        w.dimension('3', 3)

        w.add('AtomicNumbers', ('natoms',),
              atoms.get_atomic_numbers(), units=(0, 0, 0))
        w.add('CartesianPositions', ('natoms', '3'),
              atoms.get_positions() / Bohr, units=(1, 0, 0))
        w.add('MagneticMoments', ('natoms',), magmom_a, units=(0, 0, 0))
        w.add('Tags', ('natoms',), tag_a, units=(0, 0, 0))
        w.add('BoundaryConditions', ('3',), atoms.get_pbc(), units=(0, 0, 0))
        w.add('UnitCell', ('3', '3'), atoms.get_cell() / Bohr, units=(1, 0, 0))

        if atoms.get_velocities() is not None:
            w.add('CartesianVelocities', ('natoms', '3'),
                  atoms.get_velocities() * AUT / Bohr, units=(1, 0, -1))

        w.add('PotentialEnergy', (), hamiltonian.Etot + 0.5 * hamiltonian.S,
              units=(0, 1, 0))
        if paw.forces.F_av is not None:
            w.add('CartesianForces', ('natoms', '3'), paw.forces.F_av,
                  units=(-1, 1, 0))

        # Write the k-points:
        if wfs.kd.N_c is not None:
            w.add('NBZKPoints', ('3'), wfs.kd.N_c)
        w.dimension('nbzkpts', len(wfs.bzk_kc))
        w.dimension('nibzkpts', len(wfs.ibzk_kc))
        w.add('BZKPoints', ('nbzkpts', '3'), wfs.bzk_kc)
        w.add('IBZKPoints', ('nibzkpts', '3'), wfs.ibzk_kc)
        w.add('IBZKPointWeights', ('nibzkpts',), wfs.weight_k)

        # Create dimensions for varioius netCDF variables:
        ng = wfs.gd.get_size_of_global_array()
        w.dimension('ngptsx', ng[0])
        w.dimension('ngptsy', ng[1])
        w.dimension('ngptsz', ng[2])
        ng = density.finegd.get_size_of_global_array()
        w.dimension('nfinegptsx', ng[0])
        w.dimension('nfinegptsy', ng[1])
        w.dimension('nfinegptsz', ng[2])
        w.dimension('nspins', wfs.nspins)
        w.dimension('nbands', wfs.nbands)
        nproj = sum([setup.ni for setup in wfs.setups])
        nadm = sum([setup.ni * (setup.ni + 1) // 2 for setup in wfs.setups])
        w.dimension('nproj', nproj)
        w.dimension('nadm', nadm)

        p = paw.input_parameters
        # Write various parameters:
        (w['KohnShamStencil'],
         w['InterpolationStencil']) = p['stencils']
        w['PoissonStencil'] = paw.hamiltonian.poisson.get_stencil()
        w['XCFunctional'] = paw.hamiltonian.xc.name
        w['Charge'] = p['charge']
        w['FixMagneticMoment'] = paw.occupations.fixmagmom
        w['UseSymmetry'] = p['usesymm']
        w['Converged'] = scf.converged
        w['FermiWidth'] = paw.occupations.width
        w['MixClass'] = density.mixer.__class__.__name__
        w['MixBeta'] = density.mixer.beta
        w['MixOld'] = density.mixer.nmaxold
        w['MixWeight'] = density.mixer.weight
        w['MaximumAngularMomentum'] = p.lmax
        w['SoftGauss'] = False
        w['FixDensity'] = p.fixdensity
        w['DensityConvergenceCriterion'] = p['convergence']['density']
        w['EnergyConvergenceCriterion'] = p['convergence']['energy'] / Hartree
        w['EigenstatesConvergenceCriterion'] = p['convergence']['eigenstates']
        w['NumberOfBandsToConverge'] = p['convergence']['bands']
        w['Ekin'] = hamiltonian.Ekin
        w['Epot'] = hamiltonian.Epot
        w['Ebar'] = hamiltonian.Ebar
        w['Eext'] = hamiltonian.Eext
        w['Exc'] = hamiltonian.Exc
        w['S'] = hamiltonian.S
        try:
            if paw.occupations.fixmagmom:
                w['FermiLevel'] = paw.occupations.get_fermi_levels_mean()
                w['FermiSplit'] = paw.occupations.get_fermi_splitting()
            else:
                w['FermiLevel'] = paw.occupations.get_fermi_level()
        except ValueError:
            # Zero temperature calculation - don't write Fermi level:
            pass

        # write errors
        w['DensityError'] = scf.density_error
        w['EnergyError'] = scf.energy_error
        w['EigenstateError'] = scf.eigenstates_error

        if wfs.dtype == float:
            w['DataType'] = 'Float'
        else:
            w['DataType'] = 'Complex'

        # Try to write time and kick strength in time-propagation TDDFT:
        for attr, name in [('time', 'Time'), ('niter', 'TimeSteps'), \
                           ('kick_strength', 'AbsorptionKick')]:
            if hasattr(paw, attr):
                value = getattr(paw, attr)
                if isinstance(value, np.ndarray):
                    w.add(name, ('3',), value)
                else:
                    w[name] = value

        w['Mode'] = p.mode
        
        # Write fingerprint (md5-digest) for all setups:
        for setup in wfs.setups.setups.values():
            key = atomic_names[setup.Z] + 'Fingerprint'
            if setup.type != 'paw':
                key += '(%s)' % setup.type
            w[key] = setup.fingerprint

        setup_types = p['setups']
        if isinstance(setup_types, str):
            setup_types = {None: setup_types}
        for key, value in setup_types.items():
            if not isinstance(value, str):
                # Setups which are not strings are assumed to be
                # runtime-dependent and should *not* be saved.  We'll
                # just discard the whole dictionary
                setup_types = None
                break
        w['SetupTypes'] = repr(setup_types)

        basis = p['basis'] # And similarly for basis sets
        if isinstance(basis, dict):
            for key, value in basis.items():
                if not isinstance(value, str):
                    basis = None
        w['BasisSet'] = repr(basis)

        dtype = {float: float, complex: complex}[wfs.dtype]
    else:
        w = None

    # Write projections:
    if master or hdf5:
        w.add('Projections', ('nspins', 'nibzkpts', 'nbands', 'nproj'),
              dtype=dtype)
    for s in range(wfs.nspins):
        for k in range(wfs.nibzkpts):
            all_P_ni = wfs.collect_projections(k, s)
            if master:
                w.fill(all_P_ni, s, k)

    # Write atomic density matrices and non-local part of hamiltonian:
    if master:
        all_D_sp = np.empty((wfs.nspins, nadm))
        all_H_sp = np.empty((wfs.nspins, nadm))
        p1 = 0
        for a in range(natoms):
            ni = wfs.setups[a].ni
            nii = ni * (ni + 1) // 2
            if a in density.D_asp:
                D_sp = density.D_asp[a]
                dH_sp = hamiltonian.dH_asp[a]
            else:
                D_sp = np.empty((wfs.nspins, nii))
                domain_comm.receive(D_sp, wfs.rank_a[a], 207)
                dH_sp = np.empty((wfs.nspins, nii))
                domain_comm.receive(dH_sp, wfs.rank_a[a], 2071)
            p2 = p1 + nii
            all_D_sp[:, p1:p2] = D_sp
            all_H_sp[:, p1:p2] = dH_sp
            p1 = p2
        assert p2 == nadm

    elif kpt_comm.rank == 0 and band_comm.rank == 0:
        for a in range(natoms):
            if a in density.D_asp:
                domain_comm.send(density.D_asp[a], 0, 207)
                domain_comm.send(hamiltonian.dH_asp[a], 0, 2071)

    if master or hdf5:
        w.add('AtomicDensityMatrices', ('nspins', 'nadm'), dtype=float)
    if master:
        w.fill(all_D_sp)
    if master or hdf5:
        w.add('NonLocalPartOfHamiltonian', ('nspins', 'nadm'), dtype=float)
    if master:
        w.fill(all_H_sp)

    # Write the eigenvalues and occupation numbers:
    for name, var in [('Eigenvalues', 'eps_n'), ('OccupationNumbers', 'f_n')]:
        if master or hdf5:
            w.add(name, ('nspins', 'nibzkpts', 'nbands'), dtype=float)
        for s in range(wfs.nspins):
            for k in range(wfs.nibzkpts):
                a_n = wfs.collect_array(var, k, s)
                if master:
                    w.fill(a_n, s, k)

    # Attempt to read the number of delta-scf orbitals:
    if hasattr(paw.occupations, 'norbitals'):
        norbitals = paw.occupations.norbitals
    else:
        norbitals = None

    # Write the linear expansion coefficients for Delta SCF:
    if mode == 'all' and norbitals is not None:
        if master or hdf5:
            w.dimension('norbitals', norbitals)
            w.add('LinearExpansionOccupations', ('nspins',
                  'nibzkpts', 'norbitals'), dtype=float)
        for s in range(wfs.nspins):
            for k in range(wfs.nibzkpts):
                ne_o = wfs.collect_auxiliary('ne_o', k, s, shape=norbitals)
                if master:
                    w.fill(ne_o, s, k)

        if master or hdf5:
            w.add('LinearExpansionCoefficients', ('nspins',
                  'nibzkpts', 'norbitals', 'nbands'), dtype=complex)
        for s in range(wfs.nspins):
            for k in range(wfs.nibzkpts):
                for o in range(norbitals):
                    c_n = wfs.collect_array('c_on', k, s, subset=o)
                    if master:
                        w.fill(c_n, s, k, o)

    # Write the pseudodensity on the coarse grid:
    if master or hdf5:
        w.add('PseudoElectronDensity',
              ('nspins', 'ngptsx', 'ngptsy', 'ngptsz'), dtype=float)

    for s in range(wfs.nspins):
        if hdf5:
            do_write = (kpt_comm.rank == 0)
            indices = [s,] +  wfs.gd.get_slice()
            w.fill(density.nt_sG[s], parallel=True, write=do_write,
                   *indices)
        elif kpt_comm.rank == 0:
            nt_sG = wfs.gd.collect(density.nt_sG[s])
            if master:
                w.fill(nt_sG, s)

    # Write the pseudopotential on the coarse grid:
    if master or hdf5:
        w.add('PseudoPotential',
              ('nspins', 'ngptsx', 'ngptsy', 'ngptsz'), dtype=float)

    for s in range(wfs.nspins):
        if hdf5:
            do_write = (kpt_comm.rank == 0)
            indices = [s,] + wfs.gd.get_slice()
            w.fill(hamiltonian.vt_sG[s], parallel=True, write=do_write, 
                   *indices)
        elif kpt_comm.rank == 0:
            vt_sG = wfs.gd.collect(hamiltonian.vt_sG[s])
            if master:
                w.fill(vt_sG, s)

    hamiltonian.xc.write(w, natoms)

    if mode == 'all':
        wfs.write_wave_functions(w)
    elif mode != '':
        # Write the wave functions as seperate files

        # check if we need subdirs and have to create them
        ftype, template = wave_function_name_template(mode)
        dirname = os.path.dirname(template)
        if dirname:
            if master and not os.path.isdir(dirname):
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                else:
                    raise RuntimeError("Can't create subdir " + dirname)
        else:
            dirname = '.'
        # the slaves have to wait until the directory is created
        world.barrier()
        print >> paw.txt, 'Writing wave functions to', dirname,\
              'using mode=', mode

        ngd = wfs.gd.get_size_of_global_array()
        for s in range(wfs.nspins):
            for k in range(wfs.nibzkpts):
                for n in range(wfs.nbands):
                    psit_G = wfs.get_wave_function_array(n, k, s)
                    if master:
                        fname = template % (s, k, n) + '.' + ftype
                        wpsi = open(fname, 'w')
                        wpsi.dimension('1', 1)
                        wpsi.dimension('ngptsx', ngd[0])
                        wpsi.dimension('ngptsy', ngd[1])
                        wpsi.dimension('ngptsz', ngd[2])
                        wpsi.add('PseudoWaveFunction',
                                 ('1', 'ngptsx', 'ngptsy', 'ngptsz'),
                                 dtype=dtype)
                        wpsi.fill(psit_G)
                        wpsi.close()

    db = False
    if filename.endswith('.db'):
        if master:
            w.write_additional_db_params(cmr_params=cmr_params)
    elif cmr_params is not None and 'db' in cmr_params:
        db = cmr_params['db']

    if master or hdf5:
        # Close the file here to ensure that the last wave function is
        # written to disk:
        w.close()

    # We don't want the slaves to start reading before the master has
    # finished writing:
    world.barrier()

   # Creates a db file for CMR, if requested
    if db and not filename.endswith('.db'):
        #Write a db copy to the database
        write(paw, '.db', mode='', cmr_params=cmr_params, **kwargs)


def read(paw, reader):
    r = reader
    wfs = paw.wfs
    density = paw.density
    density.allocate()
    hamiltonian = paw.hamiltonian
    hamiltonian.allocate()
    natoms = len(paw.atoms)

    world = paw.wfs.world
    domain_comm = wfs.gd.comm
    kpt_comm = wfs.kpt_comm
    band_comm = wfs.band_comm

    version = r['version']

    hdf5 = hasattr(r, 'hdf5_reader')

    # Verify setup fingerprints and count projectors and atomic matrices:
    for setup in wfs.setups.setups.values():
        try:
            key = atomic_names[setup.Z] + 'Fingerprint'
            if setup.type != 'paw':
                key += '(%s)' % setup.type
            if setup.fingerprint != r[key]:
                str = 'Setup for %s (%s) not compatible with restart file.' \
                    % (setup.symbol, setup.filename)
                if paw.input_parameters['idiotproof']:
                    raise RuntimeError(str)
                else:
                    paw.warn(str)
        except (AttributeError, KeyError):
            str = 'Fingerprint of setup for %s (%s) not in restart file.' \
                % (setup.symbol, setup.filename)
            if paw.input_parameters['idiotproof']:
                raise RuntimeError(str)
            else:
                paw.warn(str)
    nproj = sum([setup.ni for setup in wfs.setups])
    nadm = sum([setup.ni * (setup.ni + 1) // 2 for setup in wfs.setups])

    # Verify dimensions for minimally required netCDF variables:
    ng = wfs.gd.get_size_of_global_array()
    nfg = density.finegd.get_size_of_global_array()
    shapes = {'ngptsx': ng[0],
              'ngptsy': ng[1],
              'ngptsz': ng[2],
              'nspins': wfs.nspins,
              'nproj' : nproj,
              'nadm'  : nadm}
    for name,dim in shapes.items():
        if r.dimension(name) != dim:
            raise ValueError('shape mismatch: expected %s=%d' % (name,dim))

    # Read pseudoelectron density on the coarse grid
    # and distribute out to nodes:
    nt_sG = wfs.gd.empty(density.nspins)
    if hdf5:
        indices = [slice(0, density.nspins),] + wfs.gd.get_slice()
        nt_sG[:] = r.get('PseudoElectronDensity', *indices)
    else:
        for s in range(density.nspins):
            wfs.gd.distribute(r.get('PseudoElectronDensity', s), nt_sG[s])

    # Read atomic density matrices
    D_asp = {}
    density.rank_a = np.zeros(natoms, int)
    if domain_comm.rank == 0:
        D_asp = read_atomic_matrices(r, 'AtomicDensityMatrices', wfs.setups)
    density.initialize_directly_from_arrays(nt_sG, D_asp)

    # Read pseudo potential on the coarse grid
    # and distribute out to nodes:
    if version > 0.3:
        hamiltonian.vt_sG = wfs.gd.empty(hamiltonian.nspins)
        if hdf5:
            indices = [slice(0, hamiltonian.nspins), ] + wfs.gd.get_slice()
            hamiltonian.vt_sG[:] = r.get('PseudoPotential', *indices)
        else:
            for s in range(hamiltonian.nspins):
                wfs.gd.distribute(r.get('PseudoPotential', s),
                                  hamiltonian.vt_sG[s])

    # Read non-local part of hamiltonian
    hamiltonian.dH_asp = {}
    hamiltonian.rank_a = np.zeros(natoms, int)

    if domain_comm.rank == 0 and version > 0.3:
        hamiltonian.dH_asp = read_atomic_matrices(r, \
            'NonLocalPartOfHamiltonian', wfs.setups)

    hamiltonian.Ekin = r['Ekin']
    hamiltonian.Epot = r['Epot']
    hamiltonian.Ebar = r['Ebar']
    try:
        hamiltonian.Eext = r['Eext']
    except (AttributeError, KeyError):
        hamiltonian.Eext = 0.0
    hamiltonian.Exc = r['Exc']
    hamiltonian.S = r['S']
    hamiltonian.Etot = r.get('PotentialEnergy') - 0.5 * hamiltonian.S

    wfs.rank_a = np.zeros(natoms, int)

    if version > 0.3:
        density_error = r['DensityError']
        if density_error is not None:
            density.mixer.set_charge_sloshing(density_error)
        Etot = hamiltonian.Etot
        energy_error = r['EnergyError']
        if energy_error is not None:
            paw.scf.energies = [Etot, Etot + energy_error, Etot]
    else:
        paw.scf.converged = r['Converged']

    if version > 0.6:
        if paw.occupations.fixmagmom:
            if 'FermiLevel' in r.get_parameters():
                paw.occupations.set_fermi_levels_mean(r['FermiLevel'])
            if 'FermiSplit' in r.get_parameters():
                paw.occupations.set_fermi_splitting(r['FermiSplit'])
        else:
            if 'FermiLevel' in r.get_parameters():
                paw.occupations.set_fermi_level(r['FermiLevel'])
    else:
        if not paw.input_parameters.fixmom and 'FermiLevel' in r.get_parameters():
            paw.occupations.set_fermi_level(r['FermiLevel'])


    #paw.occupations.magmom = paw.atoms.get_initial_magnetic_moments().sum()
    
    # Try to read the current time and kick strength in time-propagation TDDFT:
    for attr, name in [('time', 'Time'), ('niter', 'TimeSteps'), \
                       ('kick_strength', 'AbsorptionKick')]:
        if hasattr(paw, attr):
            try:
                if r.has_array(name):
                    value = r.get(name)
                else:
                    value = r[name]
                setattr(paw, attr, value)
            except KeyError:
                pass

    # Try to read the number of Delta SCF orbitals
    try:
        norbitals = r.dimension('norbitals')
        paw.occupations.norbitals = norbitals
    except (AttributeError, KeyError):
        norbitals = None

    # Wave functions and eigenvalues:
    dtype = r['DataType']
    if dtype == 'Float' and paw.input_parameters['dtype']!=complex:
        wfs.dtype = float
    else:
        wfs.dtype = complex
        
    nibzkpts = r.dimension('nibzkpts')
    nbands = r.dimension('nbands')
    nslice = wfs.bd.get_slice()

    if (nibzkpts == len(wfs.ibzk_kc) and
        nbands == band_comm.size * wfs.mynbands):

        # Verify that symmetries for for k-point reduction hasn't changed:
        assert np.abs(r.get('IBZKPoints')-wfs.kd.ibzk_kc).max() < 1e-12
        assert np.abs(r.get('IBZKPointWeights')-wfs.kd.weight_k).max() < 1e-12

        for kpt in wfs.kpt_u:
            # Eigenvalues and occupation numbers:
            k = kpt.k
            s = kpt.s
            eps_n = r.get('Eigenvalues', s, k)
            f_n = r.get('OccupationNumbers', s, k)
            kpt.eps_n = eps_n[nslice].copy()
            kpt.f_n = f_n[nslice].copy()

            if norbitals is not None:
                kpt.ne_o = np.empty(norbitals, dtype=float)
                kpt.c_on = np.empty((norbitals, wfs.mynbands), dtype=complex)
                for o in range(norbitals):
                    kpt.ne_o[o] = r.get('LinearExpansionOccupations',  s, k, o)
                    c_n = r.get('LinearExpansionCoefficients', s, k, o)
                    kpt.c_on[o,:] = c_n[nslice]

        if version > 0.3:
            wfs.eigensolver.error = r['EigenstateError']

        if (r.has_array('PseudoWaveFunctions') and
            paw.input_parameters.mode == 'fd'):
            
            if band_comm.size == 1 and not hdf5:
                # We may not be able to keep all the wave
                # functions in memory - so psit_nG will be a special type of
                # array that is really just a reference to a file:
                for kpt in wfs.kpt_u:
                    kpt.psit_nG = r.get_reference('PseudoWaveFunctions',
                                                  kpt.s, kpt.k)

            else:
                for kpt in wfs.kpt_u:
                    kpt.psit_nG = wfs.gd.empty(wfs.mynbands, wfs.dtype)
                    if hdf5:
                        indices = [kpt.s, kpt.k]
                        indices.append(wfs.bd.get_slice())
                        indices += wfs.gd.get_slice()
                        kpt.psit_nG[:] = r.get('PseudoWaveFunctions', *indices)
                    else:
                        # Read band by band to save memory
                        for myn, psit_G in enumerate(kpt.psit_nG):
                            n = wfs.bd.global_index(myn)
                            if domain_comm.rank == 0:
                                big_psit_G = np.array(
                                    r.get('PseudoWaveFunctions',
                                          kpt.s, kpt.k, n),
                                    wfs.dtype)
                            else:
                                big_psit_G = None
                            wfs.gd.distribute(big_psit_G, psit_G)

        if (r.has_array('WaveFunctionCoefficients') and
            paw.input_parameters.mode == 'lcao'):
            wfs.read_coefficients(r)

        for u, kpt in enumerate(wfs.kpt_u):
            P_ni = r.get('Projections', kpt.s, kpt.k)
            i1 = 0
            kpt.P_ani = {}
            for a, setup in enumerate(wfs.setups):
                i2 = i1 + setup.ni
                if domain_comm.rank == 0:
                    kpt.P_ani[a] = np.array(P_ni[nslice, i1:i2], wfs.dtype)
                i1 = i2

    # Manage mode change:
    paw.scf.check_convergence(density, wfs.eigensolver)
    newmode =  paw.input_parameters.mode
    try:
        oldmode = r['Mode']
    except (AttributeError, KeyError):
        oldmode = 'fd' # This is an old gpw file from before lcao existed
        
    if newmode == 'lcao':
        spos_ac = paw.atoms.get_scaled_positions() % 1.0
        wfs.load_lazily(hamiltonian, spos_ac)

    if newmode != oldmode:
        paw.scf.reset()

    # Get the forces from the old calculation:
    if r.has_array('CartesianForces'):
        paw.forces.F_av = r.get('CartesianForces')
    else:
        paw.forces.reset()

    hamiltonian.xc.read(r)


def read_atoms(reader):
    if isinstance(reader, str):
        reader = open(filename, 'r')

    positions = reader.get('CartesianPositions') * Bohr
    numbers = reader.get('AtomicNumbers')
    cell = reader.get('UnitCell') * Bohr
    pbc = reader.get('BoundaryConditions')
    tags = reader.get('Tags')
    magmoms = reader.get('MagneticMoments')

    atoms = Atoms(positions=positions,
                  numbers=numbers,
                  cell=cell,
                  pbc=pbc)

    if tags.any():
        atoms.set_tags(tags)
    if magmoms.any():
        atoms.set_initial_magnetic_moments(magmoms)

    if reader.has_array('CartesianVelocities'):
        velocities = reader.get('CartesianVelocities') * Bohr / AUT
        atoms.set_velocities(velocities)

    return atoms

def read_atomic_matrices(reader, key, setups):
    all_M_sp = reader.get(key)
    M_asp = {}
    p1 = 0
    for a, setup in enumerate(setups):
        ni = setup.ni
        p2 = p1 + ni * (ni + 1) // 2
        M_asp[a] = all_M_sp[:, p1:p2].copy()
        p1 = p2
    return M_asp

def read_wave_function(gd, s, k, n, mode):
    """Read the wave function for spin s, kpoint k and index n
    from a sperate file. The filename is determined from the mode
    in the same way as in write() (see above)"""

    ftype, template = wave_function_name_template(mode)
    fname = template % (s,k,n) + '.'+ftype
##    print 'fname=', fname

    i = gd.get_slice()
    r = open(fname, 'r')
    psit_G = r.get('PseudoWaveFunction', 0)[i]
    r.close()
    return psit_G
