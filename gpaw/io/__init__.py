import os
import os.path


from ase.units import Bohr, Hartree
from ase.data import atomic_names
from ase.atoms import Atoms
import numpy as npy

import gpaw.mpi as mpi


def open(filename, mode='r'):
    if filename.endswith('.nc'):
        import gpaw.io.netcdf as io
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

def write(paw, filename, mode):
    """Write state to file.
    
    The `mode` argument should be one of:

    ``''``:
      Don't write the wave functions.
    ``'all'``:
      Write also the wave functions to the file.
    ``'nc'`` or ``'gpw'``:
      Write wave functions as seperate files (the default filenames
      are ``'psit_Gs%dk%dn%d.nc' % (s, k, n)`` for ``'nc'``, where
      ``s``, ``k`` and ``n`` are spin, **k**-point and band indices). XXX
    ``'nc:mywfs/psit_Gs%dk%dn%d'``:
      Defines the filenames to be ``'mywfs/psit_Gs%dk%dn%d' % (s, k, n)``.
      The directory ``mywfs`` is created if not present. XXX
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
            nadm += ni * (ni + 1) / 2
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
        #w['BasisSet'] = p['basis']
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
        # In time propagation, write current time
        if hasattr(paw, 'time'):
            w['Time'] = paw.time

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
        # write projections
        w.add('Projections', ('nspins', 'nibzkpts', 'nbands', 'nproj'),
              dtype=dtype)

        mynu = len(wfs.kpt_u)
        all_P_ni = npy.empty((wfs.nbands, nproj), wfs.dtype)
        nstride = band_comm.size
        for kpt_rank in range(kpt_comm.size):
            for u in range(mynu):
                P_ani = wfs.kpt_u[u].P_ani
                for band_rank in range(band_comm.size):
                    i = 0
                    for a in range(natoms):
                        ni = wfs.setups[a].ni
                        if (kpt_rank == 0 and band_rank == 0 and a in P_ani):
                            P_ni = P_ani[a]
                        else:
                            P_ni = npy.empty((wfs.mynbands, ni), wfs.dtype)
                            world_rank = (wfs.rank_a[a] +
                                          kpt_rank * domain_comm.size *
                                          band_comm.size +
                                          band_rank * domain_comm.size)
                            world.receive(P_ni, world_rank, 300 + a)
                        all_P_ni[band_rank::nstride, i:i + ni] = P_ni
                        i += ni
                w.fill(all_P_ni)
        assert i == nproj

    # else is slave
    else:
        for kpt in wfs.kpt_u:
            P_ani = kpt.P_ani
            for a in range(natoms):
                if a in P_ani:
                    world.send(P_ani[a], 0, 300 + a)

    # Write atomic density matrices and non-local part of hamiltonian:
    if master:
        all_D_sp = npy.empty((wfs.nspins, nadm))
        all_H_sp = npy.empty((wfs.nspins, nadm))
        p1 = 0
        for a in range(natoms):
            ni = wfs.setups[a].ni
            nii = ni * (ni + 1) / 2
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

    nibzkpts = len(wfs.ibzk_kc)
    # Write the eigenvalues and occupation numbers:
    for name, var in [('Eigenvalues', 'eps_n'), ('OccupationNumbers', 'f_n')]:
        if master:
            w.add(name, ('nspins', 'nibzkpts', 'nbands'), dtype=float)
        for s in range(wfs.nspins):
            for k in range(nibzkpts):
                a_n = wfs.collect_array(var, k, s)
                if master:
                    w.fill(a_n)

    # Write the pseudodensity on the coarse grid:
    if master:
        w.add('PseudoElectronDensity',
              ('nspins', 'ngptsx', 'ngptsy', 'ngptsz'), dtype=float)
    if kpt_comm.rank == 0:
        for s in range(wfs.nspins):
            nt_sG = wfs.gd.collect(paw.density.nt_sG[s])
            if master:
                w.fill(nt_sG)

    # Write the pseudpotential on the coarse grid:
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
            for k in range(nibzkpts):
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
            for k in range(nibzkpts):
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

    if master:
        # Close the file here to ensure that the last wave function is
        # written to disk:
        w.close()

    # We don't want the slaves to start reading before the master has
    # finished writing:
    world.barrier()


def read(paw, reader):
    r = reader
    wfs = paw.wfs
    hamiltonian = paw.hamiltonian

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
            
    # Read pseudoelectron density pseudo potential on the coarse grid
    # and distribute out to nodes:
    paw.density.nt_sG = wfs.gd.empty(wfs.nspins)
    for s in range(wfs.nspins):
        paw.gd.distribute(r.get('PseudoElectronDensity', s),
                          paw.density.nt_sG[s])

    
    if version > 0.3:
        paw.hamiltonian.vt_sG = wfs.gd.empty(wfs.nspins)
        for s in range(wfs.nspins): 
            paw.gd.distribute(r.get('PseudoPotential', s),
                              paw.hamiltonian.vt_sG[s])

    # Read atomic density matrices and non-local part of hamiltonian:
    D_sp = r.get('AtomicDensityMatrices')
    if version > 0.3:
        dH_sp = r.get('NonLocalPartOfHamiltonian')
    paw.density.D_asp = {}
    paw.hamiltonian.dH_asp = {}
    natoms = len(paw.atoms)
    wfs.rank_a = npy.zeros(natoms, int)
    paw.density.rank_a = npy.zeros(natoms, int)
    p1 = 0
    for a, setup in enumerate(wfs.setups):
        ni = setup.ni
        p2 = p1 + ni * (ni + 1) // 2
        if domain_comm.rank == 0:
            paw.density.D_asp[a] = D_sp[:, p1:p2].copy()
            if version > 0.3:
                paw.hamiltonian.dH_asp[a] = dH_sp[:, p1:p2].copy()
        p1 = p2

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

    if version > 0.3:
        paw.scf.converged = r['Converged']
        density_error = r['DensityError']
        if density_error is not None:
            paw.density.mixer.set_charge_sloshing(density_error)
        Etot = hamiltonian.Etot
        energy_error = r['EnergyError']
        if energy_error is not None:
            paw.scf.energies = [Etot, Etot + energy_error, Etot]
        wfs.eigensolver.error = r['EigenstateError']
    else:
        paw.scf.converged = True
        
    if not paw.input_parameters.fixmom and 'FermiLevel' in r.parameters:
        paw.occupations.set_fermi_level(r['FermiLevel'])

    #paw.occupations.magmom = paw.atoms.get_initial_magnetic_moments().sum()
    
    # Try to read the current time in time-propagation:
    if hasattr(paw, 'time'):
        try:
            paw.time = r['Time']
        except KeyError:
            pass
        
    # Wave functions and eigenvalues:
    nibzkpts = r.dims['nibzkpts']
    nbands = r.dims['nbands']

    if (nibzkpts == len(wfs.ibzk_kc) and
        nbands == band_comm.size * wfs.mynbands):
        for kpt in wfs.kpt_u:
            # Eigenvalues and occupation numbers:
            k = kpt.k
            s = kpt.s
            eps_n = r.get('Eigenvalues', s, k)
            f_n = r.get('OccupationNumbers', s, k)
            n0 = band_comm.rank
            nstride = band_comm.size
            kpt.eps_n = eps_n[n0::nstride].copy()
            kpt.f_n = f_n[n0::nstride].copy()
        
        if r.has_array('PseudoWaveFunctions'):
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
                        n = band_comm.rank + myn * band_comm.size
                        if domain_comm.rank == 0:
                            big_psit_G = npy.array(r.get('PseudoWaveFunctions',
                                               kpt.s, kpt.k, n), wfs.dtype)
                        else:
                            big_psit_G = None
                        wfs.gd.distribute(big_psit_G, psit_G)

        for u, kpt in enumerate(wfs.kpt_u):
            P_ni = r.get('Projections', kpt.s, kpt.k)
            i1 = 0
            n0 = band_comm.rank
            nstride = band_comm.size
            kpt.P_ani = {}
            for a, setup in enumerate(wfs.setups):
                i2 = i1 + setup.ni
                if domain_comm.rank == 0:
                    kpt.P_ani[a] = P_ni[n0::nstride, i1:i2].copy()
                i1 = i2

    # Get the forces from the old calculation:
    if r.has_array('CartesianForces'):
        paw.forces.F_av = r.get('CartesianForces')
    else:
        paw.forces.reset()

    # Read GLLB-releated stuff
    if paw.hamiltonian.xcfunc.gllb:
        paw.hamiltonian.xcfunc.xc.read(r)

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
