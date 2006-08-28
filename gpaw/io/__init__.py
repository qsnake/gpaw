import os

import Numeric as num

import gpaw.mpi as mpi
from gpaw.version import version


MASTER = 0

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


def write(paw, filename, pos_ac, magmom_a, tag_a, mode):
    wf = paw.wf

    paw.get_cartesian_forces()
    
    if mpi.rank == MASTER:
        w = open(filename, 'w')

        w['history'] = 'gpaw restart file'
        w['version'] = version
        w['lengthunit'] = 'Bohr'
        w['energyunit'] = 'Hartree'
        
        w.dimension('natoms', len(paw.nuclei))
        w.dimension('3', 3)

        w.add('AtomicNumbers', ('natoms',),
              [nucleus.setup.Z for nucleus in paw.nuclei], units=(0, 0))
        w.add('CartesianPositions', ('natoms', '3'), pos_ac,
              units=(1, 0))
        w.add('MagneticMoments', ('natoms',), magmom_a, units=(0, 0))
        w.add('Tags', ('natoms',), tag_a, units=(0, 0))
        w.add('BoundaryConditions', ('3',), paw.domain.periodic_c,
              units=(0, 0))
        cell_cc = num.zeros((3, 3), num.Float)
        cell_cc.flat[::4] = paw.domain.cell_c  # fill in the diagonal
        w.add('UnitCell', ('3', '3'), cell_cc, units=(1, 0))

        w.add('PotentialEnergy', (), paw.Etot + 0.5 * paw.S,
              units=(0, 1))
        w.add('CartesianForces', ('natoms', '3'), paw.F_ac, units=(-1, 1))
        
        # Write the k-points:
        w.dimension('nbzkpts', len(wf.bzk_kc))
        w.dimension('nibzkpts', wf.nkpts)
        w.add('BZKPoints', ('nbzkpts', '3'), wf.bzk_kc)
        w.add('IBZKPoints', ('nibzkpts', '3'), wf.ibzk_kc)
        w.add('IBZKPointWeights', ('nibzkpts',), wf.weights_k)

        # Create dimensions for varioius netCDF variables:
        N_c = paw.gd.N_c
        w.dimension('ngptsx', N_c[0])
        w.dimension('ngptsy', N_c[1])
        w.dimension('ngptsz', N_c[2])
        ng = paw.finegd.N_c
        w.dimension('nfinegptsx', ng[0])
        w.dimension('nfinegptsy', ng[1])
        w.dimension('nfinegptsz', ng[2])
        w.dimension('nspins', wf.nspins)
        w.dimension('nbands', wf.nbands)
        
        nproj = 0
        nadm = 0
        for nucleus in paw.nuclei:
            ni = nucleus.get_number_of_partial_waves()
            nproj += ni
            nadm += ni * (ni + 1) / 2
        w.dimension('nproj', nproj)
        w.dimension('nadm', nadm)

        # Write various parameters:
        w['XCFunctional'] = paw.xcfunc.get_xc_name()
        w['UseSymmetry'] = paw.usesymm
        w['FermiWidth'] = wf.occupation.kT
        w['Mix'] = paw.mixer.beta
        w['Old'] = paw.mixer.nmaxold
        w['MaximumAngularMomentum'] = paw.nuclei[0].setup.lmax
        w['SoftGauss'] = paw.nuclei[0].setup.softgauss
        w['FixDensity'] = paw.fixdensity
        w['IdiotProof'] = paw.idiotproof
        w['Tolerance'] = paw.tolerance
        w['Ekin'] = paw.Ekin
        w['Epot'] = paw.Epot
        w['Ebar'] = paw.Ebar
        w['Exc'] = paw.Exc
        w['S'] = paw.S
        epsF = wf.occupation.get_fermi_level()
        if epsF is None:
            epsF = 100.0
        w['FermiLevel'] = epsF

        typecode = {num.Float: float, num.Complex: complex}[wf.typecode]
        # write projections
        w.add('Projections', ('nspins', 'nibzkpts', 'nbands', 'nproj'),
              typecode=typecode)
              
    if mpi.rank == MASTER:
        all_P_uni = num.zeros((wf.nmyu, wf.nbands, nproj), wf.typecode)
        for kpt_rank in range(wf.kpt_comm.size):
            i = 0
            for nucleus in paw.nuclei:
                ni = nucleus.get_number_of_partial_waves()
                if kpt_rank == MASTER and nucleus.in_this_domain:
                    P_uni = nucleus.P_uni
                else:
                    P_uni = num.zeros((wf.nmyu, wf.nbands, ni), wf.typecode)
                    world_rank = nucleus.rank + kpt_rank * paw.domain.comm.size
                    mpi.world.receive(P_uni, world_rank, 300)

                all_P_uni[:, :, i:i + ni] = P_uni
                i += ni

            for u in range(wf.nmyu):
                w.fill(all_P_uni[u])
        assert i == nproj
    else:
        for nucleus in paw.my_nuclei:
            mpi.world.send(nucleus.P_uni, MASTER, 300)

    # Write atomic density matrices:
    if mpi.rank == MASTER:
        all_D_sp = num.zeros((wf.nspins, nadm), num.Float)
        q1 = 0
        for nucleus in paw.nuclei:
            ni = nucleus.get_number_of_partial_waves()
            np = ni * (ni + 1) / 2
            if nucleus.in_this_domain:
                D_sp = nucleus.D_sp
            else:
                D_sp = num.zeros((wf.nspins, np), num.Float)
                paw.domain.comm.receive(D_sp, nucleus.rank, 207)
            q2 = q1 + np
            all_D_sp[:, q1:q1+np] = D_sp
            q1 = q2
        assert q2 == nadm
        w.add('AtomicDensityMatrices', ('nspins', 'nadm'), all_D_sp)
        
    elif wf.kpt_comm.rank == MASTER:
        for nucleus in paw.my_nuclei:
            paw.domain.comm.send(nucleus.D_sp, MASTER, 207)

    # Write the eigenvalues:
    if mpi.rank == MASTER:
        w.add('Eigenvalues', ('nspins', 'nibzkpts', 'nbands'), typecode=float)
        for kpt_rank in range(wf.kpt_comm.size):
            for u in range(wf.nmyu):
                s, k = divmod(u + kpt_rank * wf.nmyu, wf.nkpts)
                if kpt_rank == MASTER:
                    eps_n = wf.kpt_u[u].eps_n
                else:
                    eps_n = num.zeros(wf.nbands, num.Float)
                    wf.kpt_comm.receive(eps_n, kpt_rank, 4300)
                w.fill(eps_n)
    elif paw.domain.comm.rank == MASTER:
        for kpt in wf.kpt_u:
            wf.kpt_comm.send(kpt.eps_n, MASTER, 4300)

    # Write the occupation numbers:
    if mpi.rank == MASTER:
        w.add('OccupationNumbers', ('nspins', 'nibzkpts', 'nbands'),
              typecode=float)
        for kpt_rank in range(wf.kpt_comm.size):
            for u in range(wf.nmyu):
                s, k = divmod(u + kpt_rank * wf.nmyu, wf.nkpts)
                if kpt_rank == MASTER:
                    f_n = wf.kpt_u[u].f_n
                else:
                    f_n = num.zeros(wf.nbands, num.Float)
                    wf.kpt_comm.receive(f_n, kpt_rank, 4300)
                w.fill(f_n)
    elif paw.domain.comm.rank == MASTER:
        for kpt in wf.kpt_u:
            wf.kpt_comm.send(kpt.f_n, MASTER, 4300)

    # Write the pseudodensity on the coarse grid
    if mpi.rank == MASTER:
        w.add('PseudoElectronDensity',
              ('nspins', 'ngptsx', 'ngptsy', 'ngptsz'), typecode=float)

    if wf.kpt_comm.rank == MASTER:
        for s in range(wf.nspins):
            nt_sG = paw.gd.collect(paw.nt_sG[s])
            if mpi.rank == MASTER:
                w.fill(nt_sG)

    if mode == 'all':
        # Write the wave functions:
        if mpi.rank == MASTER:
            w.add('PseudoWaveFunctions', ('nspins', 'nibzkpts', 'nbands',
                                          'ngptsx', 'ngptsy', 'ngptsz'),
                  typecode=typecode)

        for s in range(wf.nspins):
            for k in range(wf.nkpts):
                for n in range(wf.nbands):
                    psit_G = paw.get_wave_function_array(n, k, s)
                    if mpi.rank == MASTER: 
                        w.fill(psit_G)
                    
    if mpi.rank == MASTER:
        # Add sync here to ensure that the last wave function is
        # written to disk:
        w.close()


def read(paw, filename):
    wf = paw.wf
    
    r = open(filename, 'r')
    
    for kpt in wf.kpt_u:
        kpt.allocate(wf.nbands)
        
    # Wave functions:
    if r.has_array('PseudoWaveFunctions'):
        if mpi.parallel:
            # Slice of the global array for this domain:
            i = [slice(b, e) for b, e in zip(paw.gd.beg0_c, paw.gd.end_c)]

            for kpt in wf.kpt_u:
                kpt.psit_nG = paw.gd.new_array(wf.nbands, wf.typecode)
                kpt.Htpsit_nG = paw.gd.new_array(wf.nbands, wf.typecode)
                # Read band by band to save memory
                for n, psit_G in enumerate(kpt.psit_nG):
                    psit_G[:] = r.get('PseudoWaveFunctions',
                                      kpt.s, kpt.k, n)[i]
        else:
            # Serial calculation.  We may not be able to keep all the wave
            # functions in memory - so psit_nG will be a special type of
            # array that is really just a reference to a file:
            for kpt in wf.kpt_u:
                kpt.psit_nG = r.get_reference('PseudoWaveFunctions',
                                              kpt.s, kpt.k)
    
    # Eigenvalues and occupation
    for kpt in wf.kpt_u:
        k = kpt.k
        s = kpt.s
        kpt.eps_n[:] = r.get('Eigenvalues', s, k)
        kpt.f_n[:] = r.get('OccupationNumbers', s, k)

    paw.Ekin = r['Ekin']
    paw.Epot = r['Epot']
    paw.Ebar = r['Ebar']
    paw.Exc = r['Exc']
    paw.S = r['S']
    paw.Etot = r.get('PotentialEnergy') - 0.5 * paw.S

    wf.occupation.set_fermi_level(r['FermiLevel'])
    
    # Read pseudoelectron density on the coarse grid and
    # distribute out to nodes
    for s in range(paw.nspins): 
        paw.gd.distribute(r.get('PseudoElectronDensity', s), paw.nt_sG[s])

    # Transfer the density to the fine grid:
    for s in range(paw.nspins):
        paw.interpolate(paw.nt_sG[s], paw.nt_sg[s])

    for u, kpt in enumerate(wf.kpt_u):
        P_ni = r.get('Projections', kpt.s, kpt.k)
        i1 = 0
        for nucleus in paw.nuclei:
            i2 = i1 + nucleus.get_number_of_partial_waves()
            if nucleus.in_this_domain:
                nucleus.P_uni[u] = P_ni[:, i1:i2]
            i1 = i2

    # Read atomic density matrices:
    D_sp = r.get('AtomicDensityMatrices')
    p1 = 0
    for nucleus in paw.nuclei:
        ni = nucleus.get_number_of_partial_waves()
        p2 = p1 + ni * (ni + 1) / 2
        if nucleus.in_this_domain:
            nucleus.D_sp[:] = D_sp[:, p1:p2]
        p1 = p2
