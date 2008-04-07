from gpaw.xc_functional import ZeroFunctional, XCFunctional, XC3DGrid
from gpaw.utilities.blas import axpy
from gpaw.utilities.complex import cc, real
from gpaw.utilities import pack, unpack2
from gpaw.gllb.gllb1d import GLLB1D
from gpaw.gllb import SMALL_NUMBER
import numpy as npy
from gpaw.mpi import world

class GLLBFunctional(ZeroFunctional, GLLB1D):
    def __init__(self, slater_xc_name, v_xc_name, K_G, relaxed_core_response=False):
        self.relaxed_core_response = relaxed_core_response
        self.K_G = K_G
        self.slater_xc = XCFunctional(slater_xc_name, 1)
        if v_xc_name is not None:
            self.v_xc = XCFunctional(v_xc_name, 1)
        else:
            self.v_xc = None
        
        GLLB1D.__init__(self)
        
    def get_functional(self):
        return self

    def pass_stuff(self, vt_sg, nt_sg, kpt_u, gd, finegd, interpolate, nspins, nuclei, all_nuclei, occupation, kpt_comm, symmetry, nvalence):
        self.vt_sg = vt_sg
        self.nt_sg = nt_sg
        self.kpt_u = kpt_u
        self.gd = gd
        self.finegd = finegd
        self.interpolate = interpolate
        self.nspins = nspins
        self.nuclei = nuclei
        self.all_nuclei = all_nuclei
        self.occupation = occupation
        self.kpt_comm = kpt_comm
        self.symmetry = symmetry
        self.nvalence = nvalence

        self.vt_G = gd.zeros() 
        self.vtp_sg = finegd.zeros(self.nspins)
        self.scr_sg = finegd.zeros(self.nspins)
        self.resp_sg = finegd.zeros(self.nspins)
        self.e_g = finegd.zeros()

        self.slater_xc3d = XC3DGrid(self.slater_xc, self.finegd, self.nspins)
        if self.v_xc is not None:
            self.v_xc3d = XC3DGrid(self.v_xc, self.finegd, self.nspins)
        else:
            self.v_xc3d = None

        # Allocate 'response-density' matrices for each nucleus
        for nucleus in self.nuclei:
            ni = nucleus.get_number_of_partial_waves()
            np = ni * (ni + 1) // 2
            nucleus.Dresp_sii = npy.zeros((self.nspins, ni, ni))
            nucleus.Dresp_sp = npy.zeros((self.nspins, np))

        # Allocate the 'response-weights' for each kpoint
        for kpt in self.kpt_u:
            kpt.wf_n = npy.zeros(kpt.nbands)

        self.ref_loc = self.nvalence // 2 - 1
        if self.ref_loc < 0: ref_loc = 0

    def update_xc_potential(self):
        assert(self.nspins == 1)

        # Locate the reference-level
        self.fermi_level = self.kpt_comm.max(max( [ kpt.eps_n[self.ref_loc] for kpt in self.kpt_u ]))

        print "Located reference level ", self.fermi_level

        # Update the response weights on each k-point
        for kpt in self.kpt_u:
            temp = self.fermi_level-kpt.eps_n[:]
            kpt.wf_n[:] = self.K_G * (npy.where(temp<1e-5, 0, temp))**(0.5) * kpt.f_n[:]
       
        # The smooth scr-part
        if self.nspins == 1:
            Exc = self.slater_xc3d.get_energy_and_potential_spinpaired(self.nt_sg[0], self.vtp_sg[0], e_g=self.e_g)
            self.scr_sg[0][:] = 2*self.e_g / (self.nt_sg[0]+SMALL_NUMBER)

            if self.v_xc3d is not None:
                self.vtp_sg[0][:] = 0.
                Exc += self.v_xc3d.get_energy_and_potential_spinpaired(self.nt_sg[0], self.vtp_sg[0], e_g = self.e_g)
                self.scr_sg[0][:] += self.vtp_sg[0]
                
        self.update_response_part()

        # Update the effective potential with screened and response part
        self.vt_sg[0] += self.scr_sg[0]
        self.vt_sg[0] += self.resp_sg[0]

        # Apply ALL PAW corrections!
        for nucleus in self.nuclei:
            Exc += nucleus.setup.xc_correction.GLLB(nucleus.D_sp, nucleus.Dresp_sp, nucleus.H_sp, nucleus.setup.extra_xc_data['core_response'])

        return Exc

    def update_response_part(self):
       
        # The smooth response part
        self.vt_G[:] = 0.0
        for kpt in self.kpt_u:
            # For each orbital, add the response part
            for wf, psit_G in zip(kpt.wf_n, kpt.psit_nG):
                if kpt.dtype is float:
                    axpy(wf, psit_G**2, self.vt_G)
                else:
                    self.vt_G += wf * (psit_G * npy.conjugate(psit_G)).real

        # Communicate the coarse-response part
        self.kpt_comm.sum(self.vt_G)

        # Include the symmetry to the response part also
        if self.symmetry is not None:
            self.symmetry.symmetrize(self.vt_G, self.gd)

        self.vtp_sg[:] = 0.0 # TODO: Is this needed for interpolate?
        self.interpolate(self.vt_G, self.vtp_sg[0])

        self.resp_sg[0][:] = self.vtp_sg[0] / (self.nt_sg[0] + SMALL_NUMBER)

        # Calculate the atomic 'response-density matricies'
        for nucleus in self.nuclei:
            D_sp = nucleus.Dresp_sp
            D_sii = nucleus.Dresp_sii
            D_sii[:] = 0.0
            for kpt in self.kpt_u:
                P_ni = nucleus.P_uni[kpt.u]
                D_sii[kpt.s] += real(npy.dot(cc(npy.transpose(P_ni)),
                                             P_ni * kpt.wf_n[:, npy.newaxis]))
            D_sp[:] = [pack(D_ii) for D_ii in D_sii]
            self.kpt_comm.sum(D_sp)

        # Symmetrize them
        if self.symmetry is not None:
            comm = self.gd.comm
            D_asp = []
            for nucleus in self.all_nuclei:
                if comm.rank == nucleus.rank:
                    D_sp = nucleus.Dresp_sp
                    comm.broadcast(D_sp, nucleus.rank)
                else:
                    ni = nucleus.get_number_of_partial_waves()
                    np = ni * (ni + 1) / 2
                    D_sp = npy.zeros((self.nspins, np))
                    comm.broadcast(D_sp, nucleus.rank)
                D_asp.append(D_sp)
            
            for s in range(self.nspins):
                D_aii = [unpack2(D_sp[s]) for D_sp in D_asp]
                for nucleus in self.nuclei:
                    nucleus.symmetrize(D_aii, self.symmetry.maps, s, response = True)
       
    
    def print_converged(self, output):
        # Locate HOMO-level
        homo = self.kpt_comm.max(max( [ kpt.eps_n[self.ref_loc] for kpt in self.kpt_u ]))

        # Locate LUMO-level
        lumo = -1.0 * self.kpt_comm.max(-1.0* min( [kpt.eps_n[self.ref_loc+1] for kpt in self.kpt_u] ))

        # Calculate the Kohn-Sham gap
        self.kohn_sham_gap = (lumo-homo)*output.Ha

        lumo_level = lumo

        lumo *= output.Ha
        homo *= output.Ha

        output.text("HOMO EIGENVALUE   :     %.4f eV" % homo)
        output.text("LUMO EIGENVALUE   :     %.4f eV" % lumo)
        output.text("------------------------------------")
        output.text("KS-BANDGAP        :     %.4f eV" % (lumo-homo))
        output.text("")
        output.text("Calculating perturbations to conduction band...")


        output.text("i) Updating band weights...")
        self.vt_G[:] = 0.0
        # Update the response weights on each k-point
        for kpt in self.kpt_u:
            temp = lumo_level-kpt.eps_n[:]
            kpt.wf_n[:] = self.K_G * (npy.where(temp<1e-5, 0, temp))**(0.5) * kpt.f_n[:] - kpt.wf_n[:]

        output.text("ii) Calculating perturbing potential...")
        self.update_response_part()

        output.text("iii) Calculating perturbation of each band...")
        output.text("BAND | ~E | E^a | NEW EIGENVALUE ")
        v_g = self.resp_sg[0]

        
        for kpt in self.kpt_u:
            psit_G = kpt.psit_nG[self.ref_loc+1]
            eps = kpt.eps_n[self.ref_loc+1]

            # Interpolate the wave functions to finer grid
            self.vt_G[:] = (psit_G * npy.conjugate(psit_G) ).real
            self.vtp_sg[:] = 0.0 # TODO: Is this needed for interpolate?
            self.interpolate(self.vt_G, self.vtp_sg[0])
            
            shift = self.finegd.integrate( self.vtp_sg[0] * v_g )

            paw_shift = 0.0
            for nucleus in self.nuclei:
                P_i = nucleus.P_uni[kpt.u][self.ref_loc+1]
                D_ii = npy.outer(P_i, cc(P_i)) 
                D_ii = real(D_ii)
                D_p = pack(D_ii)
                paw_shift += nucleus.setup.xc_correction.GLLBint(nucleus.D_sp[0], nucleus.Dresp_sp[0], D_p)

            paw_shift_all = self.gd.comm.sum(paw_shift)
            new_eig = eps + shift + paw_shift_all
            kpt.eps_n[self.ref_loc+1] = new_eig
            if self.gd.comm.rank == 0:
                # Gamma point calculation means kpt.k_c = None!!
                if kpt.k_c == None:
                    k_c = [ 0, 0, 0]
                else:
                    k_c = kpt.k_c

                print " (%.2f,%.2f,%.2f)  %.4f  %.4f  %.4f " % (k_c[0], k_c[1], k_c[2], shift*output.Ha, paw_shift*output.Ha, new_eig*output.Ha)

        # Locate LUMO-level again
        lumo = -1.0 * self.kpt_comm.max(-1.0* min([ kpt.eps_n[self.ref_loc+1] for kpt in self.kpt_u ] ))
        lumo *= output.Ha

        self.quasiparticle_gap = lumo-homo

        output.text("PERTURBED LUMO    :     %.4f eV" % lumo)
        output.text("DELTA X(C)        :     %.4f.eV" % self.get_discontinuity())
        output.text("QUASIPARTICLE GAP :     %.4f eV" % self.get_quasiparticle_gap())

    def get_kohn_sham_gap(self):
        return self.kohn_sham_gap

    def get_discontinuity(self):
        return self.quasiparticle_gap - self.kohn_sham_gap

    def get_quasiparticle_gap(self):
        return self.quasiparticle_gap
