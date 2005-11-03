import os

import Numeric as num
if os.uname()[4] == 'i686':
    import Scientific.IO.NetCDF as NetCDF
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

    if mpi.parallel:
        raise NotImplementedError('Not implemented')

    nc = NetCDF.NetCDFFile(filename, 'r')
    
    # Units ???? XXXX
    realvalued = (not nc.dimensions.has_key('two'))
    vars = nc.variables

    if vars.has_key('PseudoWaveFunctions'):
        psit_unG = vars['PseudoWaveFunctions']
        f_un = vars['OccupationNumbers']
        n = 0 
        for ns in range(wf.nspins):
            for nk in range(wf.nkpts): 

                kpt = wf.kpts[n]
                kpt.allocate(wf.nbands, wavefunctions=False)
                kpt.psit_nG = NetCDFWaveFunction(psit_unG, ns, nk,
                                                 scale=a0**1.5,
                                                 cmplx=not realvalued)
                kpt.f_n[:] = f_un[ns, nk]

                n += 1

    # Read projections:
    for nucleus in paw.nuclei:
        nucleus.allocate(wf.nspins, wf.nmykpts, wf.nbands)
    P_skni = vars['Projections']
    for ns in range(wf.nspins): 
        for nk in range(wf.nkpts):
            i = 0
            for nucleus in paw.nuclei:
                P = nucleus.P_uni
                ni = P.shape[2]  
                if realvalued:
                    P[ns,:,: ] = P_skni[ns, nk,:,i:i + ni]
                else:
                    P.real[ns,:,:] = P_skni[ns, nk, :,i:i + ni, 0]
                    P.imag[ns,:,:] = P_skni[ns, nk, :,i:i + ni, 1]
                i += ni
            assert i == nc.dimensions['nproj']

    # Read atomic density matrices:
    D_sq = vars['AtomicDensityMatrices']
    q1 = 0
    for nucleus in paw.nuclei:
        D_sp = nucleus.D_sp
        q2 = q1 + D_sp.shape[1]
        D_sp[:] = D_sq[:, q1:q2]
        q1 = q2
    assert q2 == nc.dimensions['nadm']

    paw.Etot = vars['PotentialEnergy'][0] / Ha
    paw.Ekin = nc.Ekin[0] / Ha
    paw.Epot = nc.Epot[0] / Ha
    paw.Ebar = nc.Ebar[0] / Ha
    paw.Exc = nc.Exc[0] / Ha
    paw.S = nc.S[0] / Ha

    paw.nt_sg[:] = vars['PseudoElectronDensity'][:] * a0**3

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
                                    ('nspins', 'nibzkpst',
                                     'nbands', 'nproj', 'two'))

    # master (domain_comm 0) collects projections
    i = 0
    for nucleus in paw.nuclei:
        P_uni = paw.get_nucleus_P_uni(nucleus)
        if mpi.rank == MASTER:
            ni = P_uni.shape[2]
            P_uni.shape = (wf.nspins, wf.nkpts, wf.nbands, ni)
            if realvalued:
                var[:, :, :, i:i + ni] = P_uni
            else:
                var[:, :, :, i:i + ni, 0] = P_uni.real
                var[:, :, :, i:i + ni, 1] = P_uni.imag
            i += ni
    assert i == nproj


    if mpi.rank == MASTER:
        # Write atomic density matrices:
        var = nc.createVariable('AtomicDensityMatrices', num.Float,
                                ('nspins', 'nadm'))
        q1 = 0
        for nucleus in paw.nuclei:
            if nucleus.domain_overlap == EVERYTHING:
                D_sp = nucleus.D_sp
                q2 = q1 + D_sp.shape[1]
                var[:, q1:q2] = D_sp
            else:
                ni = nucleus.get_number_of_partial_waves()
                np = ni * (ni + 1) / 2
                q2 = q1 + np
            q1 = q2
        assert q2 == nadm

        # Write the eigenvalues:
        var = nc.createVariable('Eigenvalues', num.Float,
                                ('nspins', 'nibzkpts', 'nbands'))
        u = 0
        for s in range(wf.nspins):
            for k in range(wf.nkpts):
                var[s, k] = wf.kpts[u].eps_n * Ha
                u += 1

        # Write the occupation numbers:
        var = nc.createVariable('OccupationNumbers', num.Float,
                                ('nspins', 'nibzkpts', 'nbands'))
        u = 0
        for s in range(wf.nspins):
            for k in range(wf.nkpts):
                var[s, k] = wf.kpts[u].f_n
                u += 1

        var = nc.createVariable('PseudoElectronDensity', num.Float,
                                ('nspins',
                                 'nfinegptsx', 'nfinegptsy', 'nfinegptsz'))

    # Write the pseudodensity:
    nt_sg = paw.gd.collect(paw.nt_sg)
    if mpi.rank == MASTER:
        var[:] = nt_sg / a0**3

    # Write the wavefunctions:
    if mpi.rank == MASTER:
        if realvalued:
            var = nc.createVariable('PseudoWaveFunctions', num.Float,
                                    ('nspins', 'nibzkpts', 'nbands',
                                     'ngptsx', 'ngptsy', 'ngptsz'))
        else:
            var = nc.createVariable('PseudoWaveFunctions', num.Float,
                                    ('nspins', 'nibzkpts', 'nbands',
                                     'ngptsx', 'ngptsy', 'ngptsz', 'two'))
    c = 1.0 / a0**1.5
    u = 0
    for s in range(wf.nspins):
        for k in range(wf.nkpts):
            for n, psit_G in enumerate(wf.kpts[u].psit_nG):
                a_G = paw.gd.collect(psit_G)
                if mpi.rank == MASTER:
                    if realvalued:
                        var[s, k, n] = c * a_G
                    else:
                        var[s, k, n, :, :, :, 0] = c * a_G.real
                        var[s, k, n, :, :, :, 1] = c * a_G.imag
            u += 1

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
        
    def __getitem__(self, index):
        if type(index) is not tuple:
            index = (index,)
        if self.cmplx:
            a = self.psit_unG[self.u + index]
            w = num.zeros(a.shape[:-1], num.Complex)
            w.real = a[..., 0]
            w.imag = a[..., 1]
            w *= self.scale
            return w
        else:
            return self.scale * self.psit_unG[self.u + index]
