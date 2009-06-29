from gpaw.xc_functional import XCFunctional
import numpy as npy
from itertools import izip
from gpaw.mpi import world
from gpaw.atom.generator import Generator, parameters
from gpaw.utilities import hartree
from math import pi

class NSCFSIC:
    def __init__(self, paw):
        self.paw = paw

    def calculate(self):
        assert world.size == 1 # Not parallelized

        ESIC = 0
        xc = XCFunctional('LDA', 2)

        # Calculate the contribution from the core orbitals
        for a in self.paw.density.D_asp:
            setup = self.paw.density.setups[a]
            # TODO: Use XC which has been used to calculate the actual calculation
            # TODO: Loop over setups, not atoms
            print "Atom core SIC for ", setup.symbol
            print "%10s%10s%10s" %  ("E_xc[n_i]", "E_Ha[n_i]", "E_SIC")
            g = Generator(setup.symbol, xcname='LDA',nofiles=True, txt=None)
            g.run(**parameters[setup.symbol])
            njcore = g.njcore
            for f, e, u in zip(g.f_j[:njcore], g.e_j[:njcore], g.u_j[:njcore]):
                # Calculate orbital density
                # NOTE: It's spherically symmetrized!
                #n = npy.dot(self.f_j,
                na = npy.where(abs(u) < 1e-160, 0,u)**2 / (4 * pi)
                na[1:] /= g.r[1:]**2
                na[0] = na[1]
                nb = npy.zeros(g.N)
                va = npy.zeros(g.N) 
                vb = npy.zeros(g.N)
                e_g = npy.zeros(g.N)
                vHr = npy.zeros(g.N)
                xc.calculate_spinpolarized(e_g, na, va, nb, vb)
                Exc = npy.dot(e_g, g.rgd.dv_g)
                hartree(0, na * g.r * g.dr, g.beta, g.N, vHr)
                EHa = 2*pi*npy.dot(vHr*na*g.r , g.dr)
                print "%10.2f%10.2f%10.2f" % (Exc*27.21, EHa*27.21, -f*(EHa+Exc)*27.21)
                ESIC += -f*(EHa+Exc)
                
        # SIC correction always spin-polarized!
        nt_sG = self.paw.gd.zeros(2)
        nt_sg = self.paw.finegd.zeros(2)
        vt_sg = self.paw.finegd.zeros(2)
        e_g = self.paw.finegd.zeros()
        vHt_g = self.paw.finegd.zeros()
        
        # For each state
        print "Valence electron sic "
        print "%10s%10s%10s%10s%10s%10s" % ("spin", "k-point", "band", "E_xc[n_i]", "E_Ha[n_i]", "E_SIC")
        for kpt in self.paw.wfs.kpt_u:
            for n, psit_G in enumerate(kpt.psit_nG):
                nt_sG[:] = 0.0

                # Add the density contribution of one orbital
                self.paw.wfs.add_orbital_density(nt_sG[0], kpt, n)
                
                # NO SYMMETRIZATION! Right?

                # Interpolate the density to finer grid
                self.paw.density.interpolator.apply(nt_sG[0], nt_sg[0])
                self.paw.density.interpolator.apply(nt_sG[1], nt_sg[1])

                # Calculate the spin-polarized LDA potential with other channel filled with zeros
                vt_sg[:] = 0.0
                xc.calculate_spinpolarized(e_g, nt_sg[0], vt_sg[0], nt_sg[1], vt_sg[1])
                Exc = e_g.ravel().sum() * self.paw.finegd.dv

                # Determine and add the compensation charge coefficients
                Q_aL={}
                for a in self.paw.density.D_asp:
                    # Obtain the atomic density matrix for state
                    D_sp = self.paw.wfs.get_orbital_density_matrix(a, kpt, n)
                    D_p = D_sp[kpt.s]
                    Q_aL[a] = npy.dot(D_p, self.paw.density.setups[a].Delta_pL)
                self.paw.density.ghat.add(nt_sg[kpt.s], Q_aL)

                # Solve the poisson equation
                self.paw.hamiltonian.poisson.solve(vHt_g, nt_sg[kpt.s], charge=1)

                # Calculate the pseudo Hartree-energy
                EH = 0.5 * self.paw.finegd.integrate(vHt_g * nt_sg[kpt.s])

                # Go though each atom
                for a in self.paw.density.D_asp:
                    xccorr = self.paw.density.setups[a].xc_correction

                    # Obtain the atomic density matrix for state
                    D_sp = self.paw.wfs.get_orbital_density_matrix(a, kpt, n)

                    # PAW correction to pseudo Hartree-energy
                    EH+= npy.sum([npy.dot(D_p, npy.dot(self.paw.density.setups[a].M_pp, D_p)) for D_p in D_sp])
                    # Expand the density matrix to spin-polarized case
                    if len(D_sp) == 1:
                        D_p2 = D_sp[0].copy()
                        D_p2[:] = 0.0
                        D_sp = [ D_sp[0], D_p2 ]
                        
                    vxc_sg = npy.zeros((2, xccorr.ng))
                    vxct_sg = npy.zeros((2, xccorr.ng))
                    integrator = xccorr.get_integrator(None)
                    e = npy.zeros((xccorr.ng))
                    e2 = npy.zeros((xccorr.ng))

                    # Loop over each sline
                    for n1_sg, n1t_sg, i_slice in izip(xccorr.expand_density(D_sp, core=False),
                                                     xccorr.expand_pseudo_density(D_sp, core=False),
                                                     integrator):
                        xc.calculate_spinpolarized(e, n1_sg[0], vxc_sg[0], n1_sg[1], vxc_sg[1])
                        E = npy.dot(e, xccorr.dv_g)
                        xc.calculate_spinpolarized(e2, n1t_sg[0], vxct_sg[0], n1t_sg[1], vxct_sg[1])
                        E2 = npy.dot(e2, xccorr.dv_g)
                        Exc += integrator.integrate_E(i_slice, E-E2)
                ESIC+=-kpt.f_n[n]*(Exc+EH)
                print "%10i%10i%10i%10.2f%10.2f%10.2f" % (kpt.s, kpt.k, n, Exc*27.21, EH*27.21, -kpt.f_n[n]*(Exc+EH)*27.21) 
        print "Total correction for self-interaction energy:"
        print "%10.2f eV" % (ESIC*27.21)
        print "New total energy:"
        total = (ESIC*27.21+self.paw.get_potential_energy()+self.paw.get_reference_energy())
        print "%10.2f eV" % (total)
        return total
