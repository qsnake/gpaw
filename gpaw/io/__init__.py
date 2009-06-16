import gpaw.db
import os
import os.path


from ase.units import Bohr, Hartree
from ase.data import atomic_names
from ase.atoms import Atoms
import numpy as npy

import gpaw.mpi as mpi
import os,time,tempfile


def open(filename, mode='r'):
    if filename.endswith('.nc'):
        import gpaw.io.netcdf as io
    elif filename.endswith('.db'):
        import gpaw.db.gpaw_ReadWriter as io
    else:
        if not filename.endswith('.gpw'):
            filename += '.gpw'
        import gpaw.io.tar as io

    if mode == 'r':
        return io.Reader(filename)
    elif mode == 'w':
        return io.Writer(filename)
    else:
        raise ValueError("Illegal mode!  Use 'r' or 'w'.")

def wave_function_name_template(mode):
    try:
        ftype, template = mode.split(':')
    except:
        ftype = mode
        template = 'wfs/psit_Gs%dk%dn%d'
    return ftype, template

def write(paw, filename, mode, db=True, private="660", **kwargs):
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
    
    Please note: mode argument is ignored by ``*.db`` files

    The `db` argument:
        if True a copy of the results is automatically written to the location
        specified in gpaw.db.db_path, IF that path exists!

    The `private` argument:
       unix file access rights (i.e. 700 or ug+rwx) for the db file

       private is only applicable to ``*.db`` files.

    The `kwargs` can be any keyword-parameter (only supported with
    ``*.db`` files).
    
    The following are commonly used arguments:

    desc:
        A short description of the calculation.
    db_path:
        The path to the user-database which will be a directory where
        the output is stored. (The filename is automatically created.)
    keywords:
        A list of keywords to identify the calculation.
        A good practise is to identify calculations that belong
        together with the same keyword.
    """

    wfs = paw.wfs
    scf = paw.scf
    hamiltonian = paw.hamiltonian

    world = paw.wfs.world
    domain_comm = wfs.gd.comm
    kpt_comm = wfs.kpt_comm
    band_comm = wfs.band_comm

    master = (world.rank == 0)

    atoms = paw.atoms
    natoms = len(atoms)

    magmom_a = paw.get_magnetic_moments()

    if master:
        w = open(filename, 'w')

        w['history'] = 'GPAW restart file'
        w['version'] = '0.7'
        w['lengthunit'] = 'Bohr'
        w['energyunit'] = 'Hartree'

        if filename.endswith(".db"):
           w.write_additional_db_params(**kwargs)

        try:
            tag_a = atoms.get_tags()
            if tag_a is None:
                raise KeyError
        except KeyError:
            tag_a = npy.zeros(natoms, int)

        w.dimension('natoms', natoms)
        w.dimension('3', 3)

        w.add('AtomicNumbers', ('natoms',),
              atoms.get_atomic_numbers(), units=(0, 0))
        w.add('CartesianPositions', ('natoms', '3'),
              atoms.get_positions() / Bohr, units=(1, 0))
        w.add('MagneticMoments', ('natoms',), magmom_a, units=(0, 0))
        w.add('Tags', ('natoms',), tag_a, units=(0, 0))
        w.add('BoundaryConditions', ('3',), atoms.get_pbc(), units=(0, 0))
        w.add('UnitCell', ('3', '3'), atoms.get_cell() / Bohr, units=(1, 0))

        w.add('PotentialEnergy', (), hamiltonian.Etot + 0.5 * hamiltonian.S,
              units=(0, 1))
        if paw.forces.F_av is not None:
            w.add('CartesianForces', ('natoms', '3'), paw.forces.F_av,
                  units=(-1, 1))

        # Write the k-points:
        w.dimension('nbzkpts', len(wfs.bzk_kc))
        w.dimension('nibzkpts', len(wfs.ibzk_kc))
        w.add('BZKPoints', ('nbzkpts', '3'), wfs.bzk_kc)
        w.add('IBZKPoints', ('nibzkpts', '3'), wfs.ibzk_kc)
        w.add('IBZKPointWeights', ('nibzkpts',), wfs.weight_k)

        # Create dimensions for varioius netCDF variables:
        ng = paw.gd.get_size_of_global_array()
        w.dimension('ngptsx', ng[0])
        w.dimension('ngptsy', ng[1])
        w.dimension('ngptsz', ng[2])
        ng = paw.finegd.get_size_of_global_array()
        w.dimension('nfinegptsx', ng[0])
        w.dimension('nfinegptsy', ng[1])
        w.dimension('nfinegptsz', ng[2])
        w.dimension('nspins', wfs.nspins)
        w.dimension('nbands', wfs.nbands)

        nproj = 0
        nadm = 0
        for setup in wfs.setups:
            ni = setup.ni
            nproj += ni
            nadm += ni * (ni + 1) // 2
        w.dimension('nproj', nproj)
        w.dimension('nadm', nadm)

        p = paw.input_parameters
        # Write various parameters:
        (w['KohnShamStencil'],
         w['InterpolationStencil']) = p['stencils']
        w['PoissonStencil'] = paw.hamiltonian.poisson.nn
        w['XCFunctional'] = paw.hamiltonian.xcfunc.get_name()
        w['Charge'] = p['charge']
        w['FixMagneticMoment'] = p.fixmom
        w['UseSymmetry'] = p['usesymm']
        w['Converged'] = scf.converged
        w['FermiWidth'] = paw.occupations.kT
        w['BasisSet'] = p['basis']
        w['MixClass'] = paw.density.mixer.__class__.__name__
        w['MixBeta'] = paw.density.mixer.beta
        w['MixOld'] = paw.density.mixer.nmaxold
        w['MixMetric'] = paw.density.mixer.metric_type
        w['MixWeight'] = paw.density.mixer.weight
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
            w['FermiLevel'] = paw.occupations.get_fermi_level()
        except NotImplementedError:
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
                if isinstance(value, npy.ndarray):
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
        w['SetupTypes'] = repr(setup_types)

        dtype = {float: float, complex: complex}[wfs.dtype]

    # Write projections:
    if master:
        w.add('Projections', ('nspins', 'nibzkpts', 'nbands', 'nproj'),
              dtype=dtype)
    for s in range(wfs.nspins):
        for k in range(wfs.nibzkpts):
            all_P_ni = wfs.collect_projections(k, s)
            if master:
                w.fill(all_P_ni)

    # Write atomic density matrices and non-local part of hamiltonian:
    if master:
        all_D_sp = npy.empty((wfs.nspins, nadm))
        all_H_sp = npy.empty((wfs.nspins, nadm))
        p1 = 0
        for a in range(natoms):
            ni = wfs.setups[a].ni
            nii = ni * (ni + 1) // 2
            if a in paw.density.D_asp:
                D_sp = paw.density.D_asp[a]
                dH_sp = paw.hamiltonian.dH_asp[a]
            else:
                D_sp = npy.empty((wfs.nspins, nii))
                domain_comm.receive(D_sp, wfs.rank_a[a], 207)
                dH_sp = npy.empty((wfs.nspins, nii))
                domain_comm.receive(dH_sp, wfs.rank_a[a], 2071)
            p2 = p1 + nii
            all_D_sp[:, p1:p2] = D_sp
            all_H_sp[:, p1:p2] = dH_sp
            p1 = p2
        assert p2 == nadm
        w.add('AtomicDensityMatrices', ('nspins', 'nadm'), all_D_sp)
        w.add('NonLocalPartOfHamiltonian', ('nspins', 'nadm'), all_H_sp)

    elif kpt_comm.rank == 0 and band_comm.rank == 0:
        for a in range(natoms):
            if a in paw.density.D_asp:
                domain_comm.send(paw.density.D_asp[a], 0, 207)
                domain_comm.send(paw.hamiltonian.dH_asp[a], 0, 2071)

    # Write the eigenvalues and occupation numbers:
    for name, var in [('Eigenvalues', 'eps_n'), ('OccupationNumbers', 'f_n')]:
        if master:
            w.add(name, ('nspins', 'nibzkpts', 'nbands'), dtype=float)
        for s in range(wfs.nspins):
            for k in range(wfs.nibzkpts):
                a_n = wfs.collect_array(var, k, s)
                if master:
                    w.fill(a_n)

    # Attempt to read the number of delta-scf orbitals:
    if hasattr(paw.occupations,'norbitals'):
        norbitals = paw.occupations.norbitals
    else:
        norbitals = None

    # Write the linear expansion coefficients for Delta SCF:
    if mode == 'all' and norbitals is not None:
        if master:
            w.dimension('norbitals', norbitals)
            w.add('LinearExpansionOccupations', ('nspins',
                  'nibzkpts', 'norbitals'), dtype=float)
        for s in range(wfs.nspins):
            for k in range(wfs.nibzkpts):
                ne_o = wfs.collect_auxiliary('ne_o', k, s, shape=norbitals)
                if master:
                    w.fill(ne_o)

        if master:
            w.add('LinearExpansionCoefficients', ('nspins',
                  'nibzkpts', 'norbitals', 'nbands'), dtype=complex)
        for s in range(wfs.nspins):
            for k in range(wfs.nibzkpts):
                for o in range(norbitals):
                    c_n = wfs.collect_array('c_on', k, s, subset=o, dtype=complex)
                    if master:
                        w.fill(c_n)

    # Write the pseudodensity on the coarse grid:
    if master:
        w.add('PseudoElectronDensity',
              ('nspins', 'ngptsx', 'ngptsy', 'ngptsz'), dtype=float)
    if kpt_comm.rank == 0:
        for s in range(wfs.nspins):
            nt_sG = wfs.gd.collect(paw.density.nt_sG[s])
            if master:
                w.fill(nt_sG)

    # Write the pseudopotential on the coarse grid:
    if master:
        w.add('PseudoPotential',
              ('nspins', 'ngptsx', 'ngptsy', 'ngptsz'), dtype=float)
    if kpt_comm.rank == 0:
        for s in range(wfs.nspins):
            vt_sG = wfs.gd.collect(paw.hamiltonian.vt_sG[s])

            if master:
                w.fill(vt_sG)

    # Write GLLB-releated stuff
    if paw.hamiltonian.xcfunc.gllb:
        paw.hamiltonian.xcfunc.xc.write(w)

    if mode == 'all':
        # Write the wave functions:
        if master:
            w.add('PseudoWaveFunctions', ('nspins', 'nibzkpts', 'nbands',
                                          'ngptsx', 'ngptsy', 'ngptsz'),
                  dtype=dtype)

        for s in range(wfs.nspins):
            for k in range(wfs.nibzkpts):
                for n in range(wfs.nbands):
                    psit_G = wfs.get_wave_function_array(n, k, s)
                    if master: 
                        w.fill(psit_G)
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

        ngd = paw.gd.get_size_of_global_array()
        for s in range(wfs.nspins):
            for k in range(wfs.nibzkpts):
                for n in range(wfs.nbands):
                    psit_G = wfs.get_wave_function_array(n, k, s)
                    if master:
                        fname = template % (s,k,n) + '.'+ftype
                        wpsi = open(fname,'w')
                        wpsi.dimension('1', 1)
                        wpsi.dimension('ngptsx', ngd[0])
                        wpsi.dimension('ngptsy', ngd[1])
                        wpsi.dimension('ngptsz', ngd[2])
                        wpsi.add('PseudoWaveFunction',
                                 ('1','ngptsx', 'ngptsy', 'ngptsz'),
                                 dtype=dtype)
                        wpsi.fill(psit_G)
                        wpsi.close()

    if master and filename.endswith(".db"):
       # Set the private flag for the db copy
       w.set_db_copy_settings(db, private)

    if master:
        # Close the file here to ensure that the last wave function is
        # written to disk:
        w.close()

    # We don't want the slaves to start reading before the master has
    # finished writing:
    world.barrier()

   # Creates a db file
    if db and not filename.endswith(".db"):
        #Write a db copy to the database
        tmp = tempfile.gettempdir()+"/"
        fname  = tmp+"gpaw.db"

        if 0:#master:
            while os.path.exists(fname):
                fname = tmp+str(time.time())+".db"

        write(paw, fname, mode='', db=True, private=private, **kwargs)

        if master:
            try:
                os.remove(fname)
            except:
                pass


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

    for setup in wfs.setups.setups.values():
        try:
            key = atomic_names[setup.Z] + 'Fingerprint'
            if setup.type != 'paw':
                key += '(%s)' % setup.type
            fp = r[key]
        except (AttributeError, KeyError):
            break
        if setup.fingerprint != fp:
            str = 'Setup for %s (%s) not compatible with restart file.' % \
                  (setup.symbol, setup.filename)
            if paw.input_parameters['idiotproof']:
                raise RuntimeError(str)
            else:
                paw.warn(str)
            
    # Read pseudoelectron density on the coarse grid
    # and distribute out to nodes:
    nt_sG = paw.gd.empty(density.nspins)
    for s in range(density.nspins):
        paw.gd.distribute(r.get('PseudoElectronDensity', s),
                          nt_sG[s])

    # Read atomic density matrices
    D_asp = {}
    density.rank_a = npy.zeros(natoms, int)
    if domain_comm.rank == 0:
        D_asp = read_atomic_matrices(r, 'AtomicDensityMatrices',
                                     wfs.setups)
    
    density.initialize_directly_from_arrays(nt_sG, D_asp)


    # Read pseudo potential on the coarse grid
    # and distribute out to nodes:    
    if version > 0.3:
        hamiltonian.vt_sG = paw.gd.empty(hamiltonian.nspins)
        for s in range(hamiltonian.nspins): 
            paw.gd.distribute(r.get('PseudoPotential', s),
                              hamiltonian.vt_sG[s])

    # Read non-local part of hamiltonian
    hamiltonian.dH_asp = {}
    hamiltonian.rank_a = npy.zeros(natoms, int)

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

    # Read GLLB-releated stuff
    if hamiltonian.xcfunc.gllb:
        hamiltonian.xcfunc.xc.read(r)

    wfs.rank_a = npy.zeros(natoms, int)

    if version > 0.3:
        paw.scf.converged = r['Converged']
        density_error = r['DensityError']
        if density_error is not None:
            density.mixer.set_charge_sloshing(density_error)
        Etot = hamiltonian.Etot
        energy_error = r['EnergyError']
        if energy_error is not None:
            paw.scf.energies = [Etot, Etot + energy_error, Etot]
    else:
        paw.scf.converged = True
        
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
    nibzkpts = r.dimension('nibzkpts')
    nbands = r.dimension('nbands')
    nslice = wfs.bd.get_slice()

    if (nibzkpts == len(wfs.ibzk_kc) and
        nbands == band_comm.size * wfs.mynbands):
        for kpt in wfs.kpt_u:
            # Eigenvalues and occupation numbers:
            k = kpt.k
            s = kpt.s
            eps_n = r.get('Eigenvalues', s, k)
            f_n = r.get('OccupationNumbers', s, k)
            kpt.eps_n = eps_n[nslice].copy()
            kpt.f_n = f_n[nslice].copy()

            if norbitals is not None:
                kpt.ne_o = npy.empty(norbitals, dtype=float)
                kpt.c_on = npy.empty((norbitals, wfs.mynbands), dtype=complex)
                for o in range(norbitals):
                    kpt.ne_o[o] = r.get('LinearExpansionOccupations',  s, k, o)
                    c_n = r.get('LinearExpansionCoefficients', s, k, o)
                    kpt.c_on[o,:] = c_n[nslice]

        if r.has_array('PseudoWaveFunctions'):
            if version > 0.3:
                wfs.eigensolver.error = r['EigenstateError']
            if band_comm.size == 1:
                # We may not be able to keep all the wave
                # functions in memory - so psit_nG will be a special type of
                # array that is really just a reference to a file:
                for kpt in wfs.kpt_u:
                    kpt.psit_nG = r.get_reference('PseudoWaveFunctions',
                                                  kpt.s, kpt.k)
            else:
                for kpt in wfs.kpt_u:
                    # Read band by band to save memory
                    kpt.psit_nG = wfs.gd.empty(wfs.mynbands, wfs.dtype)
                    for myn, psit_G in enumerate(kpt.psit_nG):
                        n = wfs.bd.global_index(myn)
                        if domain_comm.rank == 0:
                            big_psit_G = npy.array(r.get('PseudoWaveFunctions',
                                               kpt.s, kpt.k, n), wfs.dtype)
                        else:
                            big_psit_G = None
                        wfs.gd.distribute(big_psit_G, psit_G)

        for u, kpt in enumerate(wfs.kpt_u):
            P_ni = r.get('Projections', kpt.s, kpt.k)
            i1 = 0
            kpt.P_ani = {}
            for a, setup in enumerate(wfs.setups):
                i2 = i1 + setup.ni
                if domain_comm.rank == 0:
                    kpt.P_ani[a] = P_ni[nslice, i1:i2].copy()
                i1 = i2

    try:
        if r['Mode'] == 'lcao':
            spos_ac = paw.atoms.get_scaled_positions()
            paw.wfs.load_lazily(hamiltonian, spos_ac)
    except(AttributeError, KeyError):
        pass

    # Get the forces from the old calculation:
    if r.has_array('CartesianForces'):
        paw.forces.F_av = r.get('CartesianForces')
    else:
        paw.forces.reset()

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
##    print "fname=",fname

    i = gd.get_slice()
    r = open(fname, 'r')
    psit_G = r.get('PseudoWaveFunction', 0)[i]
    r.close()
    return psit_G
