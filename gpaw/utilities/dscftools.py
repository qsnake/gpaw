
import numpy as npy
import gpaw.mpi as mpi

from gpaw.utilities.blas import axpy

"""
def dscf_find_lumo(paw,band):

    # http://trac.fysik.dtu.dk/projects/gpaw/browser/trunk/doc/documentation/dscf/lumo.py?format=raw

    assert band in [5,6]

    #Find band corresponding to lumo
    lumo = paw.get_pseudo_wave_function(band=band, kpt=0, spin=0)
    lumo = npy.reshape(lumo, -1)

    wf1_k = [paw.get_pseudo_wave_function(band=5, kpt=k, spin=0) for k in range(paw.wfs.nibzkpts)]
    wf2_k = [paw.get_pseudo_wave_function(band=6, kpt=k, spin=0) for k in range(paw.wfs.nibzkpts)]

    band_k = []
    for k in range(paw.wfs.nibzkpts):
        wf1 = npy.reshape(wf1_k[k], -1)
        wf2 = npy.reshape(wf2_k[k], -1)
        p1 = npy.abs(npy.dot(wf1, lumo))
        p2 = npy.abs(npy.dot(wf2, lumo))

        if p1 > p2:
            band_k.append(5)
        else:
            band_k.append(6)

    return band_k
"""

# -------------------------------------------------------------------

def mpi_debug(text):
    if isinstance(text,list):
        for t in text:
            mpi_debug(t)
    else:
        print 'mpi.rank=%d, %s' % (mpi.rank,text)

# -------------------------------------------------------------------

def dscf_find_atoms(atoms,symbol):
    chemsyms = atoms.get_chemical_symbols()
    return npy.where(map(lambda s: s==symbol,chemsyms))[0]

# -------------------------------------------------------------------

def dscf_find_bands(paw,bands,data=None,debug=False):
    """Entirely serial, but works regardless of parallelization. DOES NOT WORK WITH DOMAIN-DECOMPOSITION IN GPAW v0.5.2725 """ #TODO!

    if data is None:
        data = range(len(bands))
    else:
        assert len(data)==len(bands), 'Length mismatch.'

    """
    if allspins:
        raise NotImplementedError, 'Currently only the spin-down case is considered...' #TODO!

    # Extract wave functions for each band and k-point
    wf_knG = []
    for k in range(paw.wfs.nkpts):
        wf_knG.append([paw.get_pseudo_wave_function(band=n,kpt=k,spin=0).ravel() for n in bands]) #paw.get_pseudo fails with domain-decomposition from tar-file

    # Extract wave function for each band of the Gamma point
    gamma_nG = wf_knG[0]

    if debug: mpi_debug('wf_knG='+str(wf_knG))

    band_kn = []
    data_kn = []

    for k in range(paw.wfs.nibzkpts):
        band_n = []
        data_n = []

        for n in range(len(bands)):
            # Find the band for this k-point which corresponds to bands[n] of the Gamma point
            wf = wf_knG[k][n]
            p = npy.argmax([npy.abs(npy.dot(wf,gamma_nG[m])) for m in range(len(bands))])
            band_n.append(bands[p])
            data_n.append(datas[p])

        band_kn.append(band_n)
        data_kn.append(data_n)

    """

    k0 = 0 #TODO find kpt with lowest kpt.k_c (closest to gamma point)

    gamma_siG = []
    for s in range(paw.wfs.nspins):
        gamma_siG.append([paw.get_pseudo_wave_function(band=n,kpt=k0,spin=s).ravel() for n in bands]) #TODO! paw.get_pseudo fails with domain-decomposition from tar-file

    band_ui = []
    data_ui = []

    for u,kpt in enumerate(paw.wfs.kpt_u):
        band_i = []
        data_i = []

        for (i,n) in enumerate(bands):
            if kpt.k == k0:
                wf = gamma_siG[kpt.s][i]
            else:
                wf = paw.get_pseudo_wave_function(band=n,kpt=kpt.k,spin=kpt.s).ravel()

            overlaps = [npy.abs(npy.dot(wf,gamma_siG[kpt.s][i])) for i in range(len(bands))]
            if debug: mpi_debug('u=%d, i=%d, band=%d, overlaps=%s' % (u,i,n,str(overlaps)))
            p = npy.argmax(overlaps)
            band_i.append(bands[p])
            data_i.append(data[p])

        assert len(npy.unique(band_i))==len(npy.sort(band_i)), 'Non-unique band range' #TODO!

        band_ui.append(band_i)
        data_ui.append(data_i)

    return (band_ui,data_ui,)

# -------------------------------------------------------------------

def dscf_linear_combination(paw, molecule, bands, coefficients, debug=False):
    """Full parallelization over k-point - grid-decomposition parallelization needs heavy testing.""" #TODO!

    if debug: dumpkey = mpi.world.size == 1 and 'serial' or 'mpi'

    (band_ui,coeff_ui,) = dscf_find_bands(paw,bands,coefficients)

    #if debug: mpi_debug('nkpts=%d, ni=%d, len(paw.wfs.kpt_u)=%d' % (paw.wfs.nibzkpts,paw.wfs.setups[0].ni, len(paw.wfs.kpt_u)))
    if debug: mpi_debug('band_ui=%s, coeff_ui=%s' % (str(band_ui),str(coeff_ui)))
    if debug: mpi_debug(['u=%d, k=%d, s=%d, paw.wfs.kpt_comm.rank=%d, k_c=%s' % (u,kpt.k,kpt.s,paw.wfs.kpt_comm.rank,paw.wfs.ibzk_kc[kpt.k]) for u,kpt in enumerate(paw.wfs.kpt_u)])

    #P_aui = npy.zeros((len(molecule),len(paw.wfs.kpt_u),paw.wfs.setups[0].ni),dtype=complex)
    P_aui = {}
    for m,a in enumerate(molecule):
        if debug: mpi_debug('a=%d, paw.wfs.nibzkpts=%d, len(paw.wfs.kpt_u)=%d, paw.wfs.setups[%d].ni=%d' % (a,paw.wfs.nibzkpts,len(paw.wfs.kpt_u),a,paw.wfs.setups[a].ni))
        P_aui[m] = npy.zeros((len(paw.wfs.kpt_u),paw.wfs.setups[a].ni),dtype=complex)

    for u,kpt in enumerate(paw.wfs.kpt_u):

        band_i = band_ui[u]
        coeff_i = coeff_ui[u]

        #if debug: mpi_debug(['paw.wfs.kpt_u[%d].P_ani[:,%d,:].shape=%s' % (u,n,str(paw.wfs.kpt_u[u].P_ani[:,n,:].shape)) for n in bands])

        for m,a in enumerate(molecule):
            """
            if debug:
                for n in bands:
                    print 'mpi.rank=%d, paw.nuclei[%d].P_uni[:,%d,:].shape=' % (mpi.rank,a,n), paw.nuclei[a].P_uni[:,n,:].shape

                print 'mpi.rank=%d, test.shape=' % mpi.rank, npy.sum([c*paw.nuclei[a].P_uni[:,n,:] for (c,n) in zip(coefficients,bands)],axis=0).shape
            """

            #P_aui[m] += npy.sum([c*paw.nuclei[a].P_uni[:,n,:] for (c,n) in zip(coefficients,bands)],axis=0)

            #if paw.nuclei[a].in_this_domain: #TODO what happened to this one in guc?
            if True:
                P_aui[m][u,:] += npy.sum([c*kpt.P_ani[a][n,:] for (c,n) in zip(coeff_i,band_i)],axis=0)

                if debug: kpt.P_ani[a][:,:].dump('dscf_tool_P_ani_a%01d_k%01ds%01d_%s%02d.pickle' % (a,kpt.k,kpt.s,dumpkey,mpi.rank))

    #paw.gd.comm.sum(P_aui) #TODO HUH?!

    if debug: P_aui.dump('dscf_tool_P_aui_%s%02d.pickle' % (dumpkey,mpi.rank))

    """
    if debug and mpi.rank == 0:
        print 'P_aui.shape=',P_aui.shape

        for (a,P_ui) in enumerate(P_aui):
            print 'P_aui[%d].shape=' % a,P_ui.shape

        print 'P_aui=',P_aui

        print 'gd.Nc=',paw.gd.N_c
    """

    if debug: mpi_debug('P_aui.shape='+str(P_aui.shape))

    #wf_u = [npy.sum([c*paw.wfs.kpt_u[u].psit_nG[n] for (c,n) in zip(coefficients,bands)],axis=0) for u in range(0,len(paw.wfs.kpt_u))]
    #wf_u = npy.zeros((paw.wfs.nibzkpts,paw.gd.N_c[0]-1,paw.gd.N_c[1]-1,paw.gd.N_c[2]-1))#,dtype=complex)
    wf_u = paw.gd.zeros(len(paw.wfs.kpt_u),dtype=complex)

    gd_slice = paw.gd.get_slice()

    if debug: mpi_debug('gd_slice='+str(gd_slice))

    for u,kpt in enumerate(paw.wfs.kpt_u):
        if debug: mpi_debug('u=%d, k=%d, s=%d, paw.wfs.kpt_comm.rank=%d, paw.wfs.kpt_comm.rank=%d, gd.shape=%s, psit.shape=%s' % (u,kpt.k,kpt.s,paw.wfs.kpt_comm.rank,paw.wfs.kpt_comm.rank,str(wf_u[0].shape),str(npy.array(kpt.psit_nG[0])[gd_slice].shape)))

        #wf_u[u] += npy.sum([c*npy.array(kpt.psit_nG[n])[gd_slice] for (c,n) in zip(coefficients,bands)],axis=0)

        band_i = band_ui[u]
        coeff_i = coeff_ui[u]
        wf_u[u] += npy.sum([c*npy.array(kpt.psit_nG[n])[gd_slice] for (c,n) in zip(coeff_i,band_i)],axis=0)

    #paw.gd.comm.sum(wf_u)

    if debug: mpi_debug('|wf_u|^2=%s' % str([npy.sum(npy.abs(wf.flatten())**2) for wf in wf_u]))

    """
    if debug and mpi.rank == 0:
        print 'wf_u.shape=',wf_u.shape

        for (u,wf) in enumerate(wf_u):
            print 'wf[%d].shape=' % u,wf.shape

        for (u,wf) in enumerate(wf_u):
            print 'wf_u[%d].shape=' % u,wf.shape
    """

    return (P_aui,wf_u,)

# -------------------------------------------------------------------

def dscf_decompose_occupations(ft_mn):
    # We define a Hermitian matrix with elements f[m,n] = c[m]*c[n].conj()
    # and rewrite the coefficients in polar form as c[n] = r[n]*exp(1j*v[n]):
    # f[m,n] = r[m]*exp(1j*v[m])*r[n]*exp(-1j*v[n]) = r[m]*r[n]*exp(1j*(v[m]-v[n]))

    # Assuming c[i] is positive real for some i, i.e. v[i] = 0, then
    # c[m] = f[m,i]/r[i] = r[m]*exp(1j*v[m]) , where r[i] = sqrt(f[i,i])
    i = npy.argmax(npy.abs(ft_mn.diagonal()))
    c_n = ft_mn[:,i]/npy.abs(ft_mn[i,i])**0.5

    if (npy.abs(ft_mn-npy.outer(c_n,c_n.conj()))>1e-12).any():
        raise RuntimeError('Hermitian matrix cannot be decomposed')

    # Note that c_n is only defined up to an arbitrary phase factor
    return c_n

# -------------------------------------------------------------------

"""
def dscf_reconstruct_orbital(paw, c_un, mol):

    nkpts = len(paw.wfs.kpt_u)
    f_u = npy.zeros(nkpts,dtype=float)
    wf_u = paw.gd.zeros(nkpts,dtype=complex)

    P_aui = {}
    for a in mol:
        P_aui[a] = npy.zeros((nkpts,paw.wfs.setups[a].ni) dtype=complex)

    for c_n,kpt in zip(c_un, wfs.kpt_u):
        f = npy.dot(c_n, c_n.conj())

        wf = npy.zeros_like(kpt.psit_nG, dtype=complex)

        for n,psit_G in enumerate(kpt.psit_nG):
            #wf += c/f**0.5*psit_G
            axpy(c_n[n] / f**0.5, psit_G, wf)

        wf_u.append(wf)

        for a in mol:
            #for n,P_i in enumerate(kpt.P_ani[a]):
            #    P_ani[a][n,:] += c_n[n]/f**0.5*P_i
            P_aui[a][u,:] += npy.sum(c_n[:,npy.newaxis] / f**0.5 * kpt.P_ani[a], axis=0)

    return (f_u,wf_u,P_aui)
"""


def dscf_reconstruct_orbitals_k_point(paw, norbitals, mol, kpt):

    f_o = npy.zeros(norbitals, dtype=float)
    eps_o = npy.zeros(norbitals, dtype=float)
    wf_oG = paw.gd.zeros(norbitals, dtype=complex)

    P_aoi = {}
    for a in mol:
        P_aoi[a] = npy.zeros((norbitals,paw.wfs.setups[a].ni), dtype=complex)

    for o, c_n in enumerate(kpt.c_on):
        f = npy.dot(c_n.conj(), c_n)

        for n, psit_G in enumerate(npy.asarray(kpt.psit_nG)):
            wf_oG[o] += c_n[n] / f**0.5 * psit_G
            #axpy(c_n[n] / f**0.5, psit_G, wf_oG[o,:])

        for a, P_oi in P_aoi.items():
            #for n,P_i in enumerate(kpt.P_ani[a]):
            #    P_aoi[a][o,:] += c_n[n]/f**0.5*P_i
            P_oi[o] += npy.sum(c_n[:,npy.newaxis] / f**0.5 * kpt.P_ani[a], axis=0)

        f_o[o] = f
        eps_o[o] = npy.dot(npy.abs(c_n)**2 / f, kpt.eps_n)

    return (f_o, eps_o, wf_oG, P_aoi,)


def dscf_collapse_orbitals(paw, norbitals, nmaxbands, mol):

    for kpt in paw.wfs.kpt_u:
        (f_o, eps_o, wf_oG, P_aoi,) = dscf_reconstruct_orbitals_k_point(paw, norbitals, mol, kpt)

        assert abs(f_o-1)<1e-9
        f_o = kpt.ne_o

        #unocc_bands = npy.argwhere(abs(kpt.f_n)<1e-12).ravel()
        neworder = npy.argsort(npy.hstack((kpt.eps_n, eps_o)))[:nmaxbands]

        kpt.f_n = npy.hstack((kpt.f_n, f_o))[neworder]
        kpt.eps_n = npy.hstack((kpt.eps_n, eps_o))[neworder]

        for a, P_ni in kpt.P_ani.items():
            assert a in mol #XXX
            kpt.P_ani[a] = npy.vstack((P_ni, P_aoi[a]))[neworder,:]

        #shape = (nmaxbands,) + psit_nG_old.shape[1:]
        #kpt.psit_nG = npy.empty(shape, dtype=paw.wfs.dtype)
        psit_nG_old = kpt.psit_nG
        kpt.psit_nG = paw.wfs.gd.empty(nmaxbands, dtype=paw.wfs.dtype)

        for m, n in enumerate(neworder):
            if n < len(psit_nG_old):
                kpt.psit_nG[m] = psit_nG_old[n]
            else:
                o = n - len(psit_nG_old)
                kpt.psit_nG[m] = wf_oG[o]

        del kpt.ne_o, kpt.c_on

    del paw.occupations.norbitals

    par = paw.input_parameters

    if 'convergence' in par:
        conv = par['convergence']
        if 'bands' in conv:
            conv['bands'] = min(nmaxbands,conv['bands'])

    paw.wfs.mynbands = nmaxbands
    paw.wfs.nbands = nmaxbands

