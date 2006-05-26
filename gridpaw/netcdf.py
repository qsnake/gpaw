import os

import Numeric as num

try:
    import Scientific.IO.NetCDF as NetCDF
except ImportError:
    print "No netcdf installed"
from ASE.ChemicalElements.symbol import symbols

import gridpaw.utilities.mpi as mpi


NOT_INITIALIZED = -1
NOTHING = 0
COMPENSATION_CHARGE = 1
PROJECTOR_FUNCTION = 2
EVERYTHING = 3

MASTER = 0


def read_netcdf(paw, filename):
    wf = paw.wf
    a0 = paw.a0
    Ha = paw.Ha

    nc = NetCDF.NetCDFFile(filename, 'r')
    
    # Units ???? XXXX
    realvalued = (not nc.dimensions.has_key('two'))
    vars = nc.variables

    required_netcdf_variables = ['PseudoWaveFunctions',
                                 'OccupationNumbers',
                                 'PotentialEnergy',
                                 'PseudoElectronDensity',
                                 'Projections' ]

    # wavefunctions: All processors keeps a reference to the netcdf variable 
    psit_unG = vars['PseudoWaveFunctions']
    for kpt in wf.kpt_u:
        kpt.allocate(wf.nbands)
        kpt.psit_nG = NetCDFWaveFunction(psit_unG, kpt.s, kpt.k,
                                         scale=a0**1.5,
                                         cmplx=not realvalued)
    
    # eigenvalues and occupation
    eps_skn = vars['Eigenvalues']
    f_skn = vars['OccupationNumbers']
    for kpt in wf.kpt_u:
        k = kpt.k
        s = kpt.s
        kpt.eps_n[:] = eps_skn[s, k]
        kpt.f_n[:] = f_skn[s, k]

    paw.Etot = vars['PotentialEnergy'][0] / Ha
    paw.Ekin = nc.Ekin[0] / Ha
    paw.Epot = nc.Epot[0] / Ha
    paw.Ebar = nc.Ebar[0] / Ha
    paw.Exc = nc.Exc[0] / Ha
    paw.S = nc.S[0] / Ha
    
    # Read pseudoelectron density on the coarse grid and
    # distribute out to nodes
    for s in range(paw.nspins): 
        paw.gd.distribute(vars['PseudoElectronDensity'][:][s] * a0**3,
                          paw.nt_sG[s])

    # Transfer the density to the fine grid:
    for s in range(paw.nspins):
        paw.interpolate(paw.nt_sG[s], paw.nt_sg[s])

    P_skni = vars['Projections']
    i = 0
    for nucleus in paw.nuclei:
        ni = nucleus.get_number_of_partial_waves()
        if nucleus.in_this_domain:
            P_uni = nucleus.P_uni
            for u, kpt in enumerate(wf.kpt_u):
                k = kpt.k
                s = kpt.s
                if realvalued:
                    P_uni[u] = P_skni[s, k, :, i:i + ni]
                else:
                    P_uni[u].real = P_skni[s, k, :, i:i + ni, 0]
                    P_uni[u].real = P_skni[s, k, :, i:i + ni, 0]
        i += ni

    assert i == nc.dimensions['nproj']

    # Read atomic density matrices:
    D_sq = vars['AtomicDensityMatrices']
    q1 = 0
    for nucleus in paw.nuclei:
        if nucleus.in_this_domain:
            D_sp = nucleus.D_sp
            q2 = q1 + D_sp.shape[1]
            D_sp[:] = D_sq[:, q1:q2]
            q1 = q2
        else:
            ni = nucleus.get_number_of_partial_waves()
            np = ni * (ni + 1) / 2
            q2 = q1 + np
            q1 = q2
            
    assert q2 == nc.dimensions['nadm']



def write_netcdf(paw, filename):
    wf = paw.wf
    a0 = paw.a0
    Ha = paw.Ha
    if mpi.rank == MASTER:
        nc = NetCDF.NetCDFFile(filename, 'a')
        # Write the k-points:
        nc.createDimension('nbzkpts', len(wf.bzk_kc))
        nc.createDimension('nibzkpts', wf.nkpts)
        var = nc.createVariable('BZKPoints', num.Float, ('nbzkpts', '3'))
        var[:] = wf.bzk_kc
        var = nc.createVariable('IBZKPoints', num.Float, ('nibzkpts', '3'))
        var[:] = wf.ibzk_kc
        var = nc.createVariable('IBZKPointWeights', num.Float,
                                ('nibzkpts',))
        var[:] = wf.weights_k

        # Create dimensions for varioius netCDF variables:
        N_c = paw.gd.N_c
        nc.createDimension('ngptsx', N_c[0])
        nc.createDimension('ngptsy', N_c[1])
        nc.createDimension('ngptsz', N_c[2])
        ng = paw.finegd.N_c
        nc.createDimension('nfinegptsx', ng[0])
        nc.createDimension('nfinegptsy', ng[1])
        nc.createDimension('nfinegptsz', ng[2])
        nc.createDimension('nspins', wf.nspins)
        nc.createDimension('nbands', wf.nbands)
        nproj = 0
        nadm = 0
        for nucleus in paw.nuclei:
            ni = nucleus.get_number_of_partial_waves()
            nproj += ni
            nadm += ni * (ni + 1) / 2
        nc.createDimension('nproj', nproj)
        nc.createDimension('nadm', nadm)

        realvalued = (wf.typecode == num.Float)

        if not realvalued:
            nc.createDimension('two', 2)

        # Write various parameters:
        nc.XCFunctional = paw.xcfunc.get_xc_name()
        nc.UseSymmetry = [paw.usesymm]
        nc.FermiWidth = [wf.occupation.kT * Ha]
        nc.Mix = [paw.mixer.beta]
        nc.Old = [paw.mixer.nmaxold]
        nc.MaximumAngularMomentum = [paw.nuclei[0].setup.lmax]
        nc.SoftGauss = [paw.nuclei[0].setup.softgauss]
        nc.FixDensity = [paw.fixdensity]
        nc.IdiotProof = [paw.idiotproof]
        nc.Tolerance = [paw.tolerance]
        nc.Ekin = [paw.Ekin * Ha]
        nc.Epot = [paw.Epot * Ha]
        nc.Ebar = [paw.Ebar * Ha]
        nc.Exc = [paw.Exc * Ha]
        nc.S = [paw.S * Ha]
        epsF = wf.occupation.get_fermi_level()
        if epsF is None:
            epsF = 100.0
        nc.FermiLevel = [epsF * Ha]


    # write projections
    if mpi.rank == MASTER: 

        if realvalued:
            var = nc.createVariable('Projections', num.Float,
                                    ('nspins', 'nibzkpts',
                                     'nbands', 'nproj'))
        else:
            var = nc.createVariable('Projections', num.Float,
                                    ('nspins', 'nibzkpts',
                                     'nbands', 'nproj', 'two'))
              
    if mpi.rank == MASTER:
        i = 0
        for nucleus in paw.nuclei:
            ni = nucleus.get_number_of_partial_waves()
            for kpt_rank in range(wf.kpt_comm.size):
                if kpt_rank == MASTER and nucleus.in_this_domain:
                    P_uni = nucleus.P_uni
                else:
                    P_uni = num.zeros((wf.nmyu, wf.nbands, ni), wf.typecode)
                    world_rank = nucleus.rank + kpt_rank * paw.domain.comm.size
                    mpi.world.receive(P_uni, world_rank, 300)

                for u in range(wf.nmyu):
                    k, s = divmod(u + kpt_rank * wf.nmyu, wf.nspins)
                    if realvalued:
                        var[s, k, :, i:i + ni] = P_uni[u]
                    else:
                        var[s, k, :, i:i + ni, 0] = P_uni[u].real
                        var[s, k, :, i:i + ni, 1] = P_uni[u].imag
            i += ni
        assert i == nproj
    else:
        for nucleus in paw.my_nuclei:
            mpi.world.send(nucleus.P_uni, MASTER, 300)

    if mpi.rank == MASTER:
        # Write atomic density matrices:
        var = nc.createVariable('AtomicDensityMatrices', num.Float,
                                ('nspins', 'nadm'))

    if mpi.rank == MASTER:
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
            var[:, q1:q1+np] = D_sp[:]
            q1 = q2
        assert q2 == nadm

    elif wf.kpt_comm.rank == MASTER:
        for nucleus in paw.my_nuclei:
            paw.domain.comm.send(nucleus.D_sp, MASTER, 207)

    if mpi.rank == MASTER:
        # Write the eigenvalues and the occupation numbers:
        vare = nc.createVariable('Eigenvalues', num.Float,
                                ('nspins', 'nibzkpts', 'nbands'))
        varf = nc.createVariable('OccupationNumbers', num.Float,
                                ('nspins', 'nibzkpts', 'nbands'))

    if mpi.rank == MASTER:
        for kpt_rank in range(wf.kpt_comm.size):
            for u in range(wf.nmyu):
                k, s = divmod(u + kpt_rank * wf.nmyu, wf.nspins)
                if kpt_rank == MASTER:
                    eps_n = wf.kpt_u[u].eps_n
                    f_n = wf.kpt_u[u].f_n
                else:
                    eps_n = num.zeros(wf.nbands, num.Float)
                    f_n = num.zeros(wf.nbands, num.Float)
                    wf.kpt_comm.receive(eps_n, kpt_rank, 4300)
                    wf.kpt_comm.receive(f_n, kpt_rank, 4301)
                vare[s, k, :] = eps_n * Ha
                varf[s, k, :] = f_n
    elif paw.domain.comm.rank == MASTER:
        for kpt in wf.kpt_u:
            wf.kpt_comm.send(kpt.eps_n, MASTER, 4300)
            wf.kpt_comm.send(kpt.f_n, MASTER, 4301)

    # Write the pseudodensity on the coarse grid
    if mpi.rank==MASTER:
        var = nc.createVariable('PseudoElectronDensity', num.Float,
                                ('nspins', 'ngptsx', 'ngptsy', 'ngptsz'))

    for s in range(wf.nspins):
        nt_sG = paw.gd.collect(paw.nt_sG[s])
        if mpi.rank == MASTER:
            var[s] = nt_sG / a0**3

    # Write the wave functions:
    if mpi.rank == MASTER:
        if realvalued:
            var = nc.createVariable('PseudoWaveFunctions', num.Float,
                                    ('nspins', 'nibzkpts', 'nbands',
                                     'ngptsx', 'ngptsy', 'ngptsz'))
        else:
            var = nc.createVariable('PseudoWaveFunctions', num.Float,
                                    ('nspins', 'nibzkpts', 'nbands',
                                     'ngptsx', 'ngptsy', 'ngptsz', 'two'))
    for s in range(wf.nspins):
        for k in range(wf.nkpts):
            for n in range(wf.nbands):
                a_G = paw.get_wave_function_array(n, k, s)
                if mpi.rank == MASTER: 
                    if realvalued:
                        var[s, k, n] = a_G
                    else:
                        var[s, k, n, :, :, :, 0] = a_G.real
                        var[s, k, n, :, :, :, 1] = a_G.imag

    if mpi.rank == MASTER:
        # Add sync here to ensure that the last wave function is
        # written to disk:
        nc.sync()


class NetCDFWaveFunction:
    def __init__(self, psit_unG, s, k, scale, cmplx):
        self.psit_unG = psit_unG
        self.u = (s, k)
        self.scale = scale
        self.cmplx = cmplx
        netcdfshape = self.psit_unG.shape
        if self.cmplx:
            self.shape = netcdfshape[2:-1]
        else:
            self.shape = netcdfshape[2:]

    def __len__(self):
        return self.shape[0]
            
    def __getitem__(self, index):
        if type(index) is not tuple:
            index = (index,)
        if self.cmplx:
            try: 
                a = self.psit_unG[self.u + index]
            except IOError: 
                raise IndexError
            w = num.zeros(a.shape[:-1], num.Complex)
            w.real = a[..., 0]
            w.imag = a[..., 1]
            w *= self.scale
            return w
        else:
            try: 
                return self.scale * self.psit_unG[self.u + index]
            except IOError:
                raise IndexError
