import numpy as np
from pylab import *
from math import sqrt, pi
from ase import Atoms, Atom
from ase.units import Bohr, Hartree
from gpaw import GPAW
from gpaw.spherical_harmonics import Y


def solve_Schrodinger(kpts, bcell, a, E_p, Vpppi):
    """Solve 2*2 TB hamiltonian of pi band of graphene.

    Refer to G. Grosso, Solid state Physics, P153
    """

    try:
        nkpt = kpts.shape[0]
    except:
        nkpt = len(kpts)
    H = np.zeros((2,2), dtype=complex)
    H[0,0] = H[1,1] = E_p # Hartree
    F = np.zeros(nkpt, dtype=complex)
    E1 = np.zeros(nkpt)
    E2 = np.zeros(nkpt)
    C1 = np.zeros((2, nkpt), dtype=complex)
    C2 = np.zeros((2, nkpt), dtype=complex)

    for ik in range(nkpt):
        kx = np.dot(kpts[ik], bcell[:,0])
        ky = np.dot(kpts[ik], bcell[:,1])
        F[ik] = 1. + 2. * np.cos(kx*a/2) * np.exp(-1j * sqrt(3)*ky*a/2)
        H[0,1] = Vpppi * F[ik]
        H[1,0] = H[0,1].conj()
        w, v = np.linalg.eig(H)
        if w[1] >= w[0]:
            E1[ik] = w[0]
            C1[:,ik] = v[:, 0]
            E2[ik] = w[1]
            C2[:,ik] = v[:, 1]
        else:
            E1[ik] = w[1]
            C1[:,ik] = v[:, 1]
            E2[ik] = w[0]
            C2[:,ik] = v[:, 0]
    
    return E1, E2, C1, C2


def get_bandstructure(bcell, a, E_p, Vpppi):
    """Get bandstructure of graphene: Gamma-K-M."""

    nkpt1 = 20
    nkpt2 = 10
    nkpt3 = 18
    
    # Gamma: (0,0,0)
    # K    : (2/3, 1/3, 0)
    # M    : (0.5, 0,   0)
    
    kpts1 = [(2./3.*k/(nkpt1-1),  1./3.*k/(nkpt1-1), 0) for k in range(nkpt1)]
    kpts2 = [(2./3.-(2./3.-0.5)*k/(nkpt2-1),  1./3.-1./3.*k/(nkpt2-1), 0) for k in range(1,nkpt2)]
    kpts3 = [(0.5-0.5*k/(nkpt3-1),  0., 0.) for k in range(1,nkpt3)]
    
    kpts = kpts1 + kpts2 + kpts3

    E1, E2, C1, C2 = solve_Schrodinger(kpts, bcell, a, E_p, Vpppi)

    nkpt = len(kpts)
#    plot(np.arange(nkpt), E1 * Hartree)
#    plot(np.arange(nkpt), E2 * Hartree)
#    show()


def get_pz_orbital(calc):
    """Get Carbon p_z orbital on 3D grid by lcao_to_grid method."""

    bfs_a = [setup.phit_j for setup in calc.wfs.setups]
    
    from gpaw.lfc import BasisFunctions
    bfs = BasisFunctions(calc.wfs.gd, bfs_a, calc.wfs.kpt_comm, cut=True)
    spos_ac = calc.atoms.get_scaled_positions()
    bfs.set_positions(spos_ac)
    
    orb_MG = calc.wfs.gd.zeros(8)
    C_M = np.identity(8)
    bfs.lcao_to_grid(C_M, orb_MG,q=-1)

    phi_aG = [orb_MG[2], orb_MG[6]]

    # Plot orbitals
#    r = calc.wfs.gd.get_grid_point_coordinates()
#    nG = calc.get_number_of_grid_points()
#    x = np.zeros((nG[0], nG[2]))
#    y = np.zeros((nG[0], nG[2]))
#    z = np.zeros((nG[0], nG[2]))
#    for i in range(nG[0]):
#        x[i, :] = r[1, i, i, :]
#        y[i, :] = r[2, i, i, :]
#        z[i,:] = (orb_MG[2,i,i,:] + orb_MG[6,i,i,:]) / sqrt(2)
#
#    contourf(x,y,z)
#    colorbar()
#    show()
#
    return phi_aG


def generate_pz_orbital():
    """Get Carbon p_z orbital on 1D radial grid."""

    from gpaw.basis_data import Basis
    basis = Basis('C', 'sz')
    
    # get p orbital
    p_orb = basis.bf_j[1]
    phi_g = p_orb.phit_g
    r_g = np.linspace(0., p_orb.rc, p_orb.ng)

    # Normalization and plot
#    dr = r_g[1] - r_g[0]
#    norm = (rphit_g**2).sum() * dr
#    phi_g = phi_g / sqrt(norm)
#    rphi_g = phi_g * r_g
#    plot(r_g, rphit_g,'-k')
#    plot(r_g, rphi_g ,'-r')
#    show()

    return r_g, phi_g


def pz_orbital_to_GRID(phi_g, h_cv, nG, r, R_a, dr):
    """Tranform Carbon p_z orbital from 1D radial to 3D grid."""

    # wavefunction on the grid
    phi_aG = {}

    dv = np.abs(np.dot(h_cv[0],np.cross(h_cv[1],h_cv[2])))
    ng = phi_g.shape[0]
    for ia in range(2):
        phi_aG[ia] = np.zeros(nG)
        for i in range(nG[0]):
            for j in range(nG[1]):
                for k in range(nG[2]):
                    tmp = r[:,i,j,k] - R_a[ia] 
                    rindex = np.int(sqrt(np.inner(tmp, tmp))/ dr)
                    if rindex >= ng:
                        radial_phi = 0.
                    else:
                        radial_phi = phi_g[rindex]
                    # L**2 + m = 2 for p_z component
                    phi_aG[ia][i,j,k] = radial_phi * Y(2, tmp[0], tmp[1], tmp[2]) 

#        Normalized wavefunction on the grid
        norm = (phi_aG[ia]**2).sum() * dv
        phi_aG[ia] /= sqrt(norm)

# Plot yz plane of phi_aG
#    x = np.zeros((nG[0], nG[2]))
#    y = np.zeros((nG[0], nG[2]))
#    z1 = np.zeros((nG[0], nG[2]))
#    z2 = np.zeros((nG[0], nG[2]))
#    for i in range(nG[0]):
#        assert r[0, i,i,:].all() == 0.
#        x[i, :] = r[1, i, i, :]
#        y[i, :] = r[2, i, i, :]
#        z1[i, :] = phi_aG[0][i, i, :] 
#        z2[i, :] = phi_aG[1][i, i, :]
#    figure()
#    subplot(111, aspect='equal')
#    contourf(x,y,z1)
#    colorbar()
#
#    subplot(212)
#    contourf(x,y,z2)
#    show()
#
    return phi_aG


def hilbert_transform(Nw, NwS, dw, eta, specfunc_w):

    from gpaw.utilities.blas import gemmdot
    tmp_ww = np.zeros((Nw, NwS), dtype=complex)

    for iw in range(Nw):
        w = iw * dw
        for jw in range(NwS):
            ww = jw * dw 
            tmp_ww[iw, jw] = 1. / (w - ww + 1j*eta) - 1. / (w + ww + 1j*eta)

    chi0_w = gemmdot(tmp_ww, specfunc_w, beta = 0.)
    return chi0_w * dw


def plot_psiG(psi_G, r):

    # Plot yz plane of phi_aG
    nG = r.shape[1:]
    x = np.zeros((nG[0], nG[2]))
    y = np.zeros((nG[0], nG[2]))
    z = np.zeros((nG[0], nG[2]))
    for i in range(nG[0]):
        assert r[0, i,i,:].all() == 0.
        x[i, :] = r[1, i, i, :]
        y[i, :] = r[2, i, i, :]
        z[i, :] = psi_G[i, i, :] 

    figure()
    contourf(x,y,z)
    clim(-0.3,0.3)
    colorbar()


def calculate_RPA_dielectric_function(gd, f_nk, E_nk, psi_nk, kq, qq, vol, HilbertTrans=True):

    wmax = 30. / Hartree
    wcut = wmax + 5. / Hartree
    dw = 0.1 / Hartree
    Nw = int(wmax  / dw) + 1
    NwS = int(wcut  / dw) + 1
    sigma = 1e-5
    eta = 0.2 / Hartree

    chi0_w = np.zeros(Nw, dtype=complex)
    specfunc_w = np.zeros(NwS,dtype=complex)

    nband = f_nk.shape[0]
    nkpt = f_nk.shape[1]

    r = gd.get_grid_point_coordinates()
    qr = np.inner(qq, r.T).T
    expqr = np.exp(-1j * qr)

    from gpaw.response.grid_chi import CHI
    chi = CHI()

    for k in range(nkpt):
        rho_nn = np.zeros((nband, nband), dtype=complex)
        for n in range(nband):
            psi1_G = psi_nk[n, k].conj() * expqr
            for m in range(nband):
                if np.abs(f_nk[n, k] - f_nk[m, kq[k]]) > 1e-8:
                    rho_nn[n, m] = gd.integrate(psi1_G * psi_nk[m, kq[k]])

        if not HilbertTrans:
            C_nn = np.zeros((nband, nband), dtype=complex)
            for iw in range(Nw):
                w = iw * dw
                for n in range(nband):
                    for m in range(nband):
                        if np.abs(f_nk[n, k] - f_nk[m, kq[k]]) > 1e-8:
                            C_nn[n, m] = (f_nk[n, k] - f_nk[m, kq[k]]) / (
                                w + E_nk[n, k] - E_nk[m, kq[k]] + 1j * eta)
                                          
                chi0_w[iw] += (rho_nn * C_nn * rho_nn.conj()).sum()
        else:
            for n in range(nband):
                for m in range(nband):
                    focc = f_nk[n, k] - f_nk[m, kq[k]]
                    if focc > 1e-8:
                        w0 = E_nk[m, kq[k]] - E_nk[n, k]
                        tmp = focc * rho_nn[n, m] * rho_nn[n, m].conj()

                        deltaw = chi.delta_function(w0, dw, NwS, sigma)
                        for iw in range(NwS):
                            if deltaw[iw] > 1e-8:
                                specfunc_w[iw] += tmp * deltaw[iw]
        print 'finished kpoint', k
        
    if HilbertTrans:
        chi0_w = hilbert_transform(Nw, NwS, dw, eta, specfunc_w)

    eRPA_w =  1. - 4 * pi / np.inner(qq, qq) * chi0_w /  vol

    w = np.arange(Nw)*dw*Hartree

    return w, eRPA_w


def Ab_calc(calc):
    """Get eigen-energies and wavefunctions from Ab-calculation."""

    nband = calc.get_number_of_bands()
    nkpt = calc.get_ibz_k_points().shape[0]
    nG = calc.get_number_of_grid_points()
    
    E_nk = np.array([calc.get_eigenvalues(kpt=k) for k in range(nkpt)]).T / Hartree
    f_nk = np.array([calc.get_occupation_numbers(kpt=k) for k in range(nkpt)]).T 
    psi_nkG = np.zeros((nband, nkpt, nG[0], nG[1], nG[2]),dtype=complex)

    for n in range(nband):
        for k in range(nkpt):
            psi_nkG[n, k] = calc.wfs.kpt_u[k].psit_nG[n]

    return E_nk, f_nk, psi_nkG


def TB_calc(calc, E_p, Vpppi, localization_factor):
    """Get eigen-energies and wavefunctions from tight-binding calculation."""
    
    r_g, phi_g = generate_pz_orbital()
    dr = (r_g[1] - r_g[0]) * localization_factor 
    h_cv = calc.wfs.gd.h_cv
    r = calc.wfs.gd.get_grid_point_coordinates()
    R_a = calc.atoms.positions / Bohr
    dR_a = R_a[1] - R_a[0]
    a = sqrt(3) * sqrt(np.inner(dR_a, dR_a))
    nG = calc.get_number_of_grid_points()
    bzkpt_kG = calc.get_ibz_k_points()
    nkpt = bzkpt_kG.shape[0]
    acell = calc.atoms.cell / Bohr
    bcell = np.linalg.inv(acell.T) * 2. * pi
    
    phi_aG = pz_orbital_to_GRID(phi_g, h_cv, nG, r, R_a, dr)
#    phi_aG = get_pz_orbital(calc)
    
#    get_bandstructure(bcell, a, E_p, Vpppi)
    
    E1, E2, C1, C2 = solve_Schrodinger(bzkpt_kG, bcell, a, E_p, Vpppi)
    
    nband = 2
    E_nk = np.array([E1, E2])
    f_nk = np.zeros((nband, nkpt))
    f_nk[0] = 2. / nkpt

    psi_nkG = np.zeros((nband, nkpt, nG[0], nG[1], nG[2]),dtype=complex)
    
    C_nMk = np.array([C1, C2])
    for n in range(nband):
        for k in range(nkpt):
            psi_nkG[n, k] = C_nMk[n, 0, k] * phi_aG[0] + C_nMk[n, 1, k] * phi_aG[1]

            # Normalize ? 
#            norm = gd.integrate(psi2_nkG[n, k].conj() * psi2_nkG[n, k])
#            psi2_nkG[n, k] /= sqrt(norm)

    return E_nk, f_nk, psi_nkG
            

def initialize(filename, q=None):
    
    calc = GPAW(filename)
    
    bzkpt_kG = calc.get_ibz_k_points()
    nkpt = bzkpt_kG.shape[0]
    
    from ase.dft.kpoints import get_monkhorst_shape
    nkptxyz = get_monkhorst_shape(bzkpt_kG)
    
    acell = calc.atoms.cell / Bohr
    bcell = np.linalg.inv(acell.T) * 2. * pi
    vol = np.abs(np.dot(acell[0],np.cross(acell[1],acell[2])))

    if q is None:
        q = np.array([1./nkptxyz[0], 0., 0.])
    qq = np.dot(q, bcell)
    
    # get k+q
    from gpaw.response.grid_chi import CHI
    chi = CHI()
    chi.nkpt = nkpt
    chi.nkptxyz = nkptxyz
    kq = chi.find_kq(bzkpt_kG, q)

    return calc, kq, qq, vol



