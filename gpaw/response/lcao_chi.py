from math import pi, sqrt
from os.path import isfile
from scipy.special import sph_jn
import numpy as np
from ase.units import Hartree, Bohr
from ase.data import chemical_symbols

from gpaw.xc_functional import XCFunctional
from gpaw.lcao.pwf2 import LCAOwrap
from gpaw.utilities.blas import gemmdot
from gpaw.utilities import unpack
from gpaw.lfc import BasisFunctions
from gpaw import GPAW

from gpaw.gaunt import gaunt as G_LLL
from gpaw.spherical_harmonics import Y
from gpaw.setup_data import SetupData
from gpaw.setup import Setup

class CHI:
    def __init__(self):
        self.xc = 'LDA'
        self.nspin = 1

    def periodic(self, calc, q, wcut, wmin, wmax, dw, eta):

        bzkpt_kG = calc.get_bz_k_points()
        self.nkpt = bzkpt_kG.shape[0]

        self.nband = calc.wfs.nbands
        self.nvalence = calc.wfs.nvalence
        assert calc.wfs.nspins == 1
        
        self.acell = calc.atoms.cell / Bohr
        self.get_primitive_cell()
        
        # obtain eigenvalues, occupations
        e_kn = np.array([calc.get_eigenvalues(kpt=k) for k in range(self.nkpt)])
        f_kn = np.array([calc.get_occupation_numbers(kpt=k) for k in range(self.nkpt)])

        self.nG = calc.get_number_of_grid_points()
        self.nG0 = self.nG[0] * self.nG[1] * self.nG[2]
        print 'grid size', self.nG

        gd = calc.wfs.gd

        # obtain LCAO coefficient and add phase
        # C_knM *= e{-i k. R_a}
        if calc.wfs.kpt_u[0].C_nM is not None:
            print 'calculating renormalized C_knM'
            C_knM = np.array([kpt.C_nM.copy() for kpt in calc.wfs.kpt_u])
            pos_a = calc.get_atoms().positions / Bohr
            m_a = calc.wfs.basis_functions.M_a
            for a in calc.wfs.basis_functions.my_atom_indices:
                m1 = m_a[a]
                m2 = m1+ calc.wfs.setups[a].niAO
                for ik in range(self.nkpt):
                    kk =  np.array([np.inner(bzkpt_kG[ik], self.bcell[:,i]) for i in range(3)])
                    C_knM[ik,:,m1:m2] *= np.exp(-1j * np.dot(kk, pos_a[a]))
            np.savez('C_knM.npz',C=C_knM)
        else:
            foo = np.load('C_knM.npz')
            C_knM = foo['C']
            assert C_knM.shape[0] == self.nkpt and (
                   C_knM.shape[1] == self.nband)

        
        # get P_aMi
        P_aMi  = self.get_P_aMi(calc)

        self.nLCAO = C_knM.shape[2]

        # obtain LCAO orbitals
        spos_ac = calc.atoms.get_scaled_positions()
        orb_MG = self.get_orbitals(calc, spos_ac)

        # evaluate Phi_mu(r)[1 - e{-ik.(r-R_a)}] to see how large the error is
        if calc.wfs.kpt_u[0].C_nM is not None:
            self.h_c = h_c = calc.wfs.gd.h_c
            pos_a = calc.get_atoms().positions / Bohr
            m_a = calc.wfs.basis_functions.M_a
            for a in calc.wfs.basis_functions.my_atom_indices:
                m1 = m_a[a]
                m2 = m1+ calc.wfs.setups[a].niAO
                for ik in range(self.nkpt):
                    kk =  np.array([np.inner(bzkpt_kG[ik], self.bcell[:,i]) for i in range(3)])
                    for i in range(self.nG[0]):
                        for j in range(self.nG[1]):
                            for k in range(self.nG[2]):
                                tmp = np.array([i*h_c[0], j*h_c[1], k*h_c[2]])
                                kr[i,j,k] = np.inner(kk, tmp)
                
                    orb_MG[m1:m2] *= 1. - np.exp(-1j * kr)
                    for mu in range(m1, m2):
                        print ik, mu, calc.wfs.gd.integrate(orb_MG[mu])            

        
        # Check the orthonormalization of wfs
        setups = calc.wfs.setups
        rho_MM = np.zeros((self.nLCAO, self.nLCAO))

        phi_ii = {}
        for a in range(len(setups)):
            phi_p = setups[a].Delta_pL[:,0].copy()
            phi_ii[a] = unpack(phi_p) * sqrt(4*pi)

        for mu in range(self.nLCAO):
            for nu in range(self.nLCAO):
                rho_MM[mu, nu] = gd.integrate(orb_MG[mu].conj() * orb_MG[nu])
                for a in range(len(setups)):
                    P_ii = np.outer(P_aMi[a][mu].conj(), P_aMi[a][nu])
                    rho_MM[mu, nu] += (P_ii * phi_ii[a]).sum()
        for k in range(self.nkpt):
            for n in range(self.nband):
                for m in range(self.nband):
                    rho = (np.outer(C_knM[k, n].conj(), C_knM[k, m]) * rho_MM).sum()
                    #if n == m and np.abs(rho-1) > 1e-10:
                        #print 'after PAW', (k, n, m), rho
                    #if n != m and np.abs(rho) > 1e-10:
                        #print 'after PAW', (k, n, m), rho
                        

        # Construte k and k + q index
        self.q = q
        kq = self.find_kq(bzkpt_kG, q)

        self.h_c = h_c = calc.wfs.gd.h_c
        Li = np.array([3, 1, 2])
        d_nn = np.zeros((self.nband, self.nband, 3))
        qr = np.zeros(self.nG)

        # construct q.r
        qq = np.array([np.inner(self.q, self.bcell[:,i]) for i in range(3)])
        for i in range(self.nG[0]):
            for j in range(self.nG[1]):
                for k in range(self.nG[2]):
                    tmp = np.array([i*h_c[0], j*h_c[1], k*h_c[2]])
                    qr[i,j,k] = np.inner(qq, tmp)

        # unit conversion
        e_kn = e_kn / Hartree
        wcut = wcut / Hartree
        wmin = 0 #wmin / Hartree
        wmax = wmax / Hartree
        self.dw = dw / Hartree
        self.Nw = int((wmax - wmin) / self.dw) + 1
        eta = eta / Hartree

        setups = calc.wfs.setups
        chi0_w = np.zeros(self.Nw, dtype = complex)
        rho_MM = np.zeros((self.nLCAO, self.nLCAO), dtype=complex)

        # calculate <phi_i | e**(-iq.r) | phi_j>
        phi_ii = {}
        R_a = calc.atoms.positions / Bohr
        for a, id in enumerate(setups.id_a):
                Z, type, basis = id
                if not phi_ii.has_key(Z):
                    phi_ii[Z] = ( self.two_phi_planewave_integrals(Z)
                                  * np.exp(-1j * np.inner(qq, R_a[a])) )
                    
        # calculate rho_MM
        for mu in range(self.nLCAO):
            for nu in range(self.nLCAO):
                rho_MM[mu, nu] = gd.integrate(orb_MG[mu].conj() * orb_MG[nu])
                for a, id in enumerate(setups.id_a):
                    Z, type, basis = id
                    P_ii = np.outer(P_aMi[a][mu].conj(), P_aMi[a][nu])
                    rho_MM[mu, nu] += ( (P_ii * phi_ii[Z]).sum() * 
                                      np.exp(-1j * np.inner(qq, R_a[a])) )

        # calculate chi0
        rho_nn = np.zeros((self.nband, self.nband), dtype=complex)
        for k in range(self.nkpt):
            for n in range(self.nband):
                for m in range(self.nband):
                    rho_nn[n, m] = (np.outer(C_knM[k, n].conj(), C_knM[k, m]) * rho_MM).sum()
                                        
            # construct (f_nk - f_n'k+q) / (w + e_nk - e_n'k+q + ieta )
            C_nn = np.zeros((self.nband, self.nband), dtype=complex)

            for iw in range(self.Nw):
                w = iw * self.dw
                for n in range(self.nband):
                    for m in range(self.nband):
                        C_nn[n, m] = (f_kn[k, n] - f_kn[kq[k], m]) / (
                             w + e_kn[k, n] - e_kn[kq[k], m] + 1j * eta)
                # get chi0(G=0,G'=0,w)    
                chi0_w[iw] += (rho_nn * C_nn * rho_nn.conj()).sum()

        epsilonRPA = np.zeros(self.Nw)
        for iw in range(self.Nw):
            w = iw * self.dw
            epsilonRPA[iw] = (1 - 4 * pi / np.inner(qq, qq) * np.imag(chi0_w[iw])) * w / (2 * pi**2)

        
        N = epsilonRPA.sum() * self.dw
        print 'sum rule:'
        print 'N = ', N, (N - self.nvalence) / self.nvalence * 100, '% error'

        f = open('Absorption','w')
        for iw in range(self.Nw):
            print >> f, iw * self.dw * Hartree, epsilonRPA[iw]

#        import pylab as pl
#        pl.plot(epsilonRPA)
#        pl.show()
        
        return

    def get_orbitals(self, calc, spos_ac):
        if self.nkpt > 1:
            bfs_a = [setup.phit_j for setup in calc.wfs.setups]
            gd = calc.wfs.gd
            bfs = BasisFunctions(gd, bfs_a, calc.wfs.kpt_comm, cut=True)
            bfs.set_positions(spos_ac)
    
            orb_MG = gd.zeros(self.nLCAO)
            C_M = np.identity(self.nLCAO)
            bfs.lcao_to_grid(C_M, orb_MG, q=-1)
        else:  # for Gamma point calculation, wrapper is correct
            from gpaw.lcao.pwf2 import LCAOwrap
            wrapper = LCAOwrap(calc)
            orb_MG = wrapper.get_orbitals()

#        for mu in range(self.nLCAO):
#            for nu in range(self.nLCAO):
#                print calc.wfs.gd.integrate(orb_MG[mu].conj() * orb_MG[nu])
        return orb_MG


    def get_P_aMi(self, calc):
        """Calculate P_aMi without k-dependence. """

        if self.nkpt > 1 or calc.wfs.P_aqMi is None:
            a = calc.get_atoms()
            calc = GPAW(mode='lcao', basis='dzp')
            a.set_calculator(calc)
            calc.initialize(a)
            calc.set_positions(a)

        assert calc.wfs.P_aqMi is not None
        P_aMi = []
        for a, P_qMi in calc.wfs.P_aqMi.items():
            P_aMi.append(P_qMi[0])
        
        return P_aMi
    
    def two_phi_planewave_integrals(self, Z):

        # Create setup for a certain specie
        xcfunc = XCFunctional('LDA',nspins=1)
        symbol = chemical_symbols[Z]
        data = SetupData(symbol,'LDA')
        s = Setup(data,xcfunc,lmax=2)

        # radial grid stuff
        ng = s.ng
        g = np.arange(ng, dtype=float)
        r_g = s.beta * g / (ng - g) 
        dr_g = s.beta * ng / (ng - g)**2
        r2dr_g = r_g **2 * dr_g
        gcut2 = s.gcut2
            
        # Obtain the phi_j and phit_j
        phi_jg = []
        phit_jg = []
        
        for (phi_g, phit_g) in zip(s.data.phi_jg, s.data.phit_jg):
            phi_g = phi_g.copy()
            phit_g = phit_g.copy()
            phi_g[gcut2:] = phit_g[gcut2:] = 0.
            phi_jg.append(phi_g)
            phit_jg.append(phit_g)

        # Construct L (l**2 + m) and j (nl) index
        L_i = []
        j_i = []
        lmax = 0 
        for j, l in enumerate(s.l_j):
            for m in range(2 * l + 1):
                L_i.append(l**2 + m)
                j_i.append(j)
                if l > lmax:
                    lmax = l
        ni = len(L_i)
        lmax = 2 * lmax + 1

        # Initialize        
        R_jj = np.zeros((s.nj, s.nj))
        R_ii = np.zeros((ni, ni))
        phi_ii = np.zeros((ni, ni), dtype=complex)
        j_lg = np.zeros((lmax, ng))
   
        # Store (phi_j1 * phi_j2 - phit_j1 * phit_j2 ) for further use
        tmp_jjg = np.zeros((s.nj, s.nj, ng))
        for j1 in range(s.nj):
            for j2 in range(s.nj): 
                tmp_jjg[j1, j2] = phi_jg[j1] * phi_jg[j2] - phit_jg[j1] * phit_jg[j2]

        qq = np.array([np.inner(self.q, self.bcell[:,i]) for i in range(3)])
        q = np.sqrt(np.inner(qq, qq)) # calculate length of q+G
        
        # Calculating spherical bessel function
        for ri in range(ng):
            j_lg[:,ri] = sph_jn(lmax - 1,  q*r_g[ri])[0]

        for li in range(lmax):
            # Radial part 
            for j1 in range(s.nj):
                for j2 in range(s.nj): 
                    R_jj[j1, j2] = np.dot(r2dr_g, tmp_jjg[j1, j2] * j_lg[li])

            for mi in range(2 * li + 1):
                # Angular part
                for i1 in range(ni):
                    L1 = L_i[i1]
                    j1 = j_i[i1]
                    for i2 in range(ni):
                        L2 = L_i[i2]
                        j2 = j_i[i2]
                        R_ii[i1, i2] =  G_LLL[L1, L2, li**2+mi]  * R_jj[j1, j2]

                phi_ii += R_ii * Y(li**2 + mi, qq[0], qq[1], qq[2]) * (-1j)**li

        phi_ii *= 4 * pi

        return phi_ii

    def sph_jn(self, n, z):
        """Calcuate spherical Bessel function.
   
        The spehrical Bessel function for the first three orders are::

                    sinz               3       sinz   3cosz
            j (z) = ---- ,   j (z) = (--- -1 ) ---- - -----
             0       z        2       z^2       z      z^2  
                            
                    sinz   cosz           15     6  sinz    15      cosz
            j (z) = ---- - ----, j (z) = (--- - ---)----  -(--- - 1)----
             1       z^2    z     3       z^3    z   z      z^2      z  
        """

        if n > 3:
            raise ValueError(
         'Spherical bessel function with n > 3 not implemented yet!')
        sph_n = np.zeros(4)
        if z == 0.:
            sph_n[0] = 1.
            sph_n[1:] = 0.
        else:
            tmp1 = np.sin(z) / z
            tmp2 = np.cos(z) / z
            sph_n[0] = tmp1
            sph_n[1] = tmp1 / z - tmp2
            sph_n[2] = (3./z**2 -1.) * tmp1 - 3./z * tmp2
            if n == 3:
                sph_n[3] = (15./z**3 - 6./z) * tmp1 - (15./z**2 -1.) * tmp2
                
        return sph_n[0:n+1]



    def find_kq(self, bzkpt_kG, q):
        """Find the index of k+q for all kpoints in BZ."""

        found = False
        kq = np.zeros(self.nkpt, dtype=int)

        for k in range(self.nkpt):
            # bzkpt(k,:) + q = bzkpt(kq,:)
            tmp1 = bzkpt_kG[k] + q
            for dim in range(3):
                if tmp1[dim] > 0.5: # 
                    tmp1[dim] -= 1.
                elif tmp1[dim] < -0.5:
                    tmp1[dim] += 1.

            for kk in range(self.nkpt):
                tmp2 = bzkpt_kG[kk]
                if (abs(tmp1-tmp2)).sum() < 1e-8:
                    kq[k] = kk
                    found = True
                    break
            if not found:
                raise ValueError('k+q not found')
        return kq

    def get_primitive_cell(self):

        a = self.acell

        self.vol = np.dot(a[0],np.cross(a[1],a[2]))
        self.BZvol = (2. * pi)**3 / self.vol

        b = np.zeros_like(a)
        b[0] = np.cross(a[1], a[2])
        b[1] = np.cross(a[2], a[0])
        b[2] = np.cross(a[0], a[1])
        self.bcell = 2. * pi * b / self.vol

        return


    def finite(self, calc, q, wcut, wmin, wmax, dw, eta):

        self.nband = calc.wfs.nbands
        self.nkpt = 1 
        self.nvalence = calc.wfs.nvalence
                
        # obtain eigenvalues, occupations
        e_n = calc.get_eigenvalues(kpt=0)
        f_n = calc.get_occupation_numbers(kpt=0)

        # obtain pseudo wfs
        assert calc.wfs.nspins == 1
        
        from gpaw.lcao.pwf2 import LCAOwrap
        wrapper = LCAOwrap(calc)
        orb_MG = calc.wfs.gd.zero_pad(wrapper.get_orbitals())
        C_nM = calc.wfs.kpt_u[0].C_nM

        self.nLCAO = C_nM.shape[1]
        self.nG = orb_MG.shape[1:]
        self.nG0 = self.nG[0] * self.nG[1] * self.nG[2]
        print 'grid size', self.nG
        # obtain the paw term for wfs
        #calc.wfs.kpt_u[k].P_ani
        
        gd = calc.wfs.gd
        setups = calc.wfs.setups
        rho_MM = np.zeros((self.nLCAO, self.nLCAO))
        P_aMi = calc.wfs.kpt_u[0].P_aMi

        # check orthonormalization of wfs
        phi_ii = {}
        for a in range(len(setups)):
            phi_p = setups[a].Delta_pL[:,0].copy()
            phi_ii[a] = unpack(phi_p) * sqrt(4*pi)
                    
        for mu in range(self.nLCAO):
            for nu in range(self.nLCAO):
                rho_MM[mu, nu] = gd.integrate(orb_MG[mu].conj() * orb_MG[nu])
                for a in range(len(setups)):
                    P_ii = np.outer(P_aMi[a][mu].conj(), P_aMi[a][nu])
                    rho_MM[mu, nu] += (P_ii * phi_ii[a]).sum()

        
        for n in range(self.nband):
            for m in range(self.nband):
                rho = (np.outer(C_nM[n].conj(), C_nM[m]) * rho_MM).sum()
                if n == m and np.abs(rho-1) > 1e-10:
                    print 'after PAW', (n, m), rho
                if n != m and np.abs(rho) > 1e-10:
                    print 'after PAW', (n, m), rho
                        
        # get dipole
        N_gd = self.nG
        self.h_c = calc.wfs.gd.h_c

        Li = np.array([3, 1, 2])
        d_MM = np.zeros((self.nLCAO, self.nLCAO, 3))
        r = np.zeros(self.nG)
        d_nn = np.zeros((self.nband, self.nband, 3))

        for ix in range(3):
            for i in range(N_gd[0]):
                for j in range(N_gd[1]):
                    for k in range(N_gd[2]):
                        if ix == 0:
                            r[i,j,k] = i*self.h_c[0]
                        elif ix == 1:
                            r[i,j,k] = j*self.h_c[1] 
                        else:
                            r[i,j,k] = k*self.h_c[2]

            for mu in range(self.nLCAO):
                for nu in range(self.nLCAO):
                    d_MM[mu, nu, ix] = gd.integrate(orb_MG[mu].conj() * orb_MG[nu] * r)
                    # print 'before PAW', (n, m), d_nn[n, m]
                    for a in range(len(setups)):
                        phi_p = setups[a].Delta_pL[:,Li[ix]].copy()
                        phi_ii[a] = unpack(phi_p) * sqrt(4*pi/3)
                        P_ii = np.outer(P_aMi[a][mu].conj(), P_aMi[a][nu])
                        d_MM[mu, nu, ix] += (P_ii * phi_ii[a]).sum()
                    # print 'after PAW', (n, m), d_nn[n, m]
                    
            for n in range(self.nband):
                for m in range(self.nband):
                    d_nn[n, m, ix] = (np.outer(C_nM[n].conj(), C_nM[m]) * d_MM[:,:,ix]).sum()
                
        e_n = e_n / Hartree
        wcut = wcut / Hartree
        wmin = 0 #wmin / Hartree
        wmax = wmax / Hartree
        self.dw = dw / Hartree
        self.Nw = int((wmax - wmin) / self.dw) + 1
        eta = eta / Hartree

        C_nn = np.zeros((self.nband, self.nband), dtype=complex)
        S = np.zeros((self.Nw, 3))
        for iw in range(self.Nw):
            w = iw * self.dw
            for n in range(self.nband):
                for m in range(self.nband):
                    C_nn[n, m] = (f_n[n] - f_n[m]) / (w + e_n[n] - e_n[m] + 1j * eta)
                    
            for ix in range(3):
                S[iw, ix] = - 2 * w / pi * np.imag((C_nn * d_nn[:,:,ix] * d_nn[:,:,ix].conj()).sum())

        print 'sum rule'
        N = S.sum() * self.dw / 3
        print 'N = ', N, (N - self.nvalence) / self.nvalence * 100 ,'% error'

        return

