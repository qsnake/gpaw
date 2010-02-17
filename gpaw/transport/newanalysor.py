from gpaw.transport.selfenergy import LeadSelfEnergy
from gpaw.transport.tools import get_matrix_index, aa1d, aa2d, sum_by_unit, \
                                dot, fermidistribution, eig_states_norm, \
                                find, get_atom_indices, dagger, write, gather_ndarray_dict
from gpaw.mpi import world
from gpaw import GPAW, Mixer, MixerDif, PoissonSolver
from ase.units import Hartree, Bohr
from gpaw.utilities.memory import maxrss
import numpy as np
import copy
import cPickle


class Structure_Info:
    def __init__(self, ion_step):
        self.ion_step = ion_step
    
    def initialize_data(self, positions, forces):
        self.positions = positions
        self.forces = forces

class Transmission_Info:
    # member variables:
    # ------------------------------
    # ep(energy points), tc(transmission coefficients), dos, bias
    # pdos, eigen_channel
    
    def __init__(self, ion_step, bias_step):
        self.ion_step, self.bias_step = ion_step, bias_step
    
    def initialize_data(self, bias, gate, ep, lead_pairs,
                        tc, dos, vt, nt, vtx, ntx, vty, nty,
                                 current, lead_fermis,
                                 time_cost, force, charge, contour):
        for name in ['bias', 'gate', 'ep', 'lead_pairs', 'tc', 'dos', 'vt',
                     'nt', 'vtx', 'ntx', 'vty', 'nty', 'current',
                     'lead_fermis', 'time_cost', 'force', 'charge', 'contour']:
            vars(self)[name] = eval(name)
        
    def initialize_data2(self, eig_tc_lead, eig_vc_lead, tp_tc, tp_vc,
                         dos_g, project_tc, left_tc, left_vc, lead_k, lead_vk,
                         tp_eig_w, tp_eig_v, tp_eig_vc, nk_on_energy):
        for name in ['eig_tc_lead', 'eig_vc_lead', 'tp_tc', 'tp_vc', 'dos_g'
                     'project_tc', 'left_tc', 'left_vc', 'lead_k', 'lead_vk',
                     'tp_eig_w', 'tp_eig_v', 'tp_eig_vc', 'nk_on_energy']:
            vars(self)[name] = eval(name)
            
    def initialize_data3(self, s00, h00, lead_fermi):
        self.s00 = s00
        self.h00 = h00
        self.lead_fermi = lead_fermi

class Electron_Step_Info:
    # member variables:
    # ------------------------------
    # ion_step
    # bias_step
    # step(index of electron step)
    # bias, gate
    # dd(diagonal element of d_spkmm), df(diagonal element of f_spkmm),
    # nt(density in a line along transport direction),
    # vt(potential in a line along transport direction)
    
   
    def __init__(self, ion_step, bias_step, ele_step):
        self.ion_step, self.bias_step, self.ele_step = ion_step, \
                                                         bias_step, ele_step
        
    def initialize_data(self, vt, nt, df, dd, vHt, rho, time_cost, mem_cost):
        for name in ['vt', 'nt', 'df', 'dd',
                     'vHt', 'rho', 'time_cost', 'mem_cost']:
            vars(self)[name] = eval(name)
    
class Transport_Analysor:
    def __init__(self, transport, restart=False):
        self.tp = transport
        self.restart = restart
        self.data = {}
        self.ele_steps = []
        self.bias_steps = []
        self.ion_steps = []
        self.n_ele_step = 0
        self.n_bias_step = 0
        self.n_ion_step = 0
        self.max_k_on_energy = 15
        self.matrix_foot_print = False
        self.reset = False
        self.scattering_states_initialized = False
        self.initialize()
        # --------------------------------------------------------------------
        #  analysis array name     &    size     &      units     & dtype
        # --------------------------------------------------------------------
        #  aX_X_X_nt  (X: int)          ns, nz                         float
        #         vt                    ns, nz                         float
        #         df                ns, npk, exnb               float || complex
        #         dd                ns, npk, exnb               float || complex
        #         vHt                  nz * 2                       float
        #         rho                  nz * 2                       float       
        #         dos               ns, npk, ne                     float
        #         tc                ns, npk, ne                     float
        #     
  
    def set_plot_option(self):
        tp = self.tp
        if tp.plot_option == None:
            ef = tp.lead_fermi[0]
            self.energies = np.linspace(ef - 5, ef + 5, 201) + 1e-4 * 1.j
            self.lead_pairs = [[0,1]]
        else:
            self.energies = tp.plot_option['energies'] + tp.lead_fermi[0] + \
                                                              1e-4 * 1.j
            self.lead_pairs = tp.plot_option['lead_pairs']
       
    def initialize(self):
        kw = self.tp.analysis_parameters
        if self.restart:
            self.initialize_selfenergy_and_green_function()
        else:
            self.selfenergies = self.tp.selfenergies
        p = self.set_default_analysis_parameters()
        for key in kw:
            if key in ['energies', 'lead_pairs', 'dos_project_atoms',
                       'project_equal_atoms', 'project_molecular_levels',
                       'isolate_atoms', 'dos_project_orbital',
                       'trans_project_orbital', 'eig_trans_channel_energies',
                        'eig_trans_channel_num', 'dos_realspace_energies']:
                p[key] = kw[key]
        for key in p:
            vars(self)[key] = p[key]       

        ef = self.tp.lead_fermi[0]
        #self.energies +=  ef + 1e-4 * 1.j
        path = self.tp.contour.get_plot_path()
        self.energies = path.energies
        self.my_energies = path.my_energies
        self.weights = path.weights
        self.my_weights = path.my_weights
        self.nids = path.nids
        self.my_nids = path.my_nids
       
        for variable in [self.eig_trans_channel_energies,
                         self.dos_realspace_energies]:
            if variable is not None:
                variable = np.array(variable) + ef

        setups = self.tp.inner_setups
        self.project_atoms_in_device = self.project_equal_atoms[0]
        self.project_atoms_in_molecule = self.project_equal_atoms[1]
        self.project_basis_in_device = get_atom_indices(
                                  self.project_atoms_in_device, setups)
        if self.isolate_atoms is not None:
            self.calculate_isolate_molecular_levels()
        self.overhead_data_saved = False
        self.nk_on_energy = None
        self.set_analysis_data_form()
 
    def save_overhead_data(self):
        contour_parameters = {}
        cp = contour_parameters
        cp['neintmethod'] = self.tp.neintmethod
        cp['neintstep'] = self.tp.neintstep
        cp['lead_fermi'] = self.tp.lead_fermi
        cp['kt'] = self.tp.occupations.width * Hartree
        if not self.tp.non_sc:
            cp['eqinttol'] = self.tp.intctrl.eqinttol
            cp['neinttol'] = self.tp.intctrl.neinttol
        
        basis_information = {}
        bi = basis_information
        bi['orbital_indices'] = self.tp.orbital_indices
        bi['lead_orbital_indices'] = self.tp.lead_orbital_indices
        bi['ll_index'] = self.tp.hsd.S[0].ll_index
        bi['ex_ll_index'] = self.tp.hsd.S[0].ex_ll_index
        
        if self.tp.analysis_mode == -2:
            lead_hs = self.collect_lead_hs()
        else:
            lead_hs = None
            
        atoms = self.tp.atoms.copy()
        if world.rank == 0:
            fd = file('analysis_overhead', 'wb')
            cPickle.dump((atoms, basis_information, contour_parameters), fd, 2)
            fd.close()
            fd = file('lead_hs', 'wb')
            cPickle.dump(lead_hs, fd, 2)
            fd.close()
      
    def reset_central_scattering_states(self):
        total_t = []
        total_vc = []
        total_k = []
        total_vl = []
        ns, npk = self.tp.my_nspins, self.tp.my_npk
        
        nl = self.tp.lead_num
        nb = np.max(self.tp.nblead)
        nbmol = self.tp.nbmol
        mkoe = self.max_k_on_energy
        dtype = complex
        self.nk_on_energy = np.zeros([ns, npk, len(self.energies)], dtype=int)
       
        for s in range(ns):
            total_t.append([])
            total_vc.append([])
            total_k.append([])
            total_vl.append([])
            for pk in range(npk):
                total_t[s].append([])
                total_vc[s].append([])
                total_k[s].append([])
                total_vl[s].append([])
                for e, energy in enumerate(self.energies):
                    t, vc, k, vl = self.central_scattering_states(
                                                           energy.real, s, pk)
                    t_array = np.zeros([nl, nl, mkoe, mkoe], dtype)
                    vc_array = np.zeros([nl, nbmol, mkoe], dtype)
                    k_array = np.zeros([nl, mkoe], dtype)
                    vl_array = np.zeros([nl, nb, mkoe], dtype)
                    nbl, nbloch = vl.shape[-2:]  
                    self.nk_on_energy[s, pk, e] = nbloch
                    t_array[:, :, :nbloch, :nbloch] = t
                    vc_array[:, :, :nbloch] = vc
                    k_array[:, :nbloch] = k
                    vl_array[:, :nbl, :nbloch] = vl                 

                    total_t[s][pk].append(t_array)
                    total_vc[s][pk].append(vc_array)                    
                    total_k[s][pk].append(k_array)
                    total_vl[s][pk].append(vl_array)
        self.total_scattering_t = np.array(total_t)            
        self.total_scattering_vc = np.array(total_vc)
        self.total_scattering_k = np.array(total_k)
        self.total_scattering_vl = np.array(total_vl)           

    def get_central_scattering_states(self, s, q, e):
        nbloch = self.nk_on_energy[s, q, e]
        return self.total_scattering_t[s, q, e, :, :, :nbloch, :nbloch], \
               self.total_scattering_vc[s, q, e, :, :, :nbloch], \
               self.total_scattering_k[s, q, e, :, :nbloch],  \
               self.total_scattering_vl[s, q, e, :, :, :nbloch]
           
    def set_default_analysis_parameters(self):
        p = {}
        p['energies'] = np.linspace(-5., 5., 61) 
        p['lead_pairs'] = [[0,1]]
        p['dos_project_atoms'] = None
        p['project_molecular_levels'] = None
        p['project_equal_atoms'] = [[], []]
        p['isolate_atoms'] = None
        p['dos_project_orbital'] = None
        p['trans_project_orbital'] = None
        p['eig_trans_channel_energies'] = None
        p['eig_trans_channel_num'] = 0
        p['dos_realspace_energies'] = None
        return p      
                
    def initialize_selfenergy_and_green_function(self):
        self.selfenergies = []
        tp = self.tp
        if tp.use_lead:
            directions = ['left', 'right']
            for i in range(tp.lead_num):
                self.selfenergies.append(LeadSelfEnergy(tp.lead_hsd[i],
                                                     tp.lead_couple_hsd[i],
                                                     tp.se_data_path,
                                                     directions[i]))
    
                self.selfenergies[i].set_bias(tp.bias[i])
            
    def reset_selfenergy_and_green_function(self, s, k):
        tp = self.tp
        for i in range(tp.lead_num):
            self.selfenergies[i].s = s
            self.selfenergies[i].pk = k
        tp.hsd.s = s
        tp.hsd.pk = k
      
    def calculate_green_function_of_k_point(self, s, k, energy, re_flag=0,
                                                    full=False, nid_flag=None):
        tp = self.tp 
        sigma = []
        for i in range(tp.lead_num):
            sigma.append(self.selfenergies[i](energy, nid_flag))
        gr = tp.hsd.calculate_eq_green_function(energy, sigma, False, full)
        if re_flag==0:
            return gr
        else:
            return gr, sigma 
    
    def calculate_transmission_and_dos(self, s, k, energies, nid_flags=None):
        self.reset_selfenergy_and_green_function(s, k)
        transmission_list = []
        dos_list = []
        for num, energy in enumerate(energies):
            if nid_flags is not None:
                nid_flag = nid_flags[num]
                nid_flag = str(self.tp.wfs.kpt_comm.rank) + '_' + str(nid_flag)
            else:
                nid_flag = None
            gr, sigma = self.calculate_green_function_of_k_point(s, k,
                                                energy, 1, nid_flag=nid_flag)
            trans_coff = []
            gamma = []
            for i in range(self.tp.lead_num):
                gamma.append(1.j * (sigma[i].recover() -
                                                sigma[i].recover().T.conj()))
        
            
            for i, lead_pair in enumerate(self.lead_pairs):
                l1, l2 = lead_pair
                if i == 0:
                    gr_sub, inv_mat = self.tp.hsd.abstract_sub_green_matrix(
                                                        energy, sigma, l1, l2)
                else:
                    gr_sub = self.tp.hsd.abstract_sub_green_matrix(energy,
                                                    sigma, l1, l2, inv_mat)                    
                transmission =  dot(dot(gamma[l1], gr_sub),
                                              dot(gamma[l2], gr_sub.T.conj()))
       
                trans_coff.append(np.real(np.trace(transmission)))
            transmission_list.append(trans_coff)
            del trans_coff
            
            dos = - np.imag(np.trace(dot(gr,
                                         self.tp.hsd.S[k].recover()))) / np.pi                
            dos_list.append(dos)
        
        ne = len(energies)
        npl = len(self.lead_pairs)
        transmission_list = np.array(transmission_list)
        transmission_list = np.resize(transmission_list.T, [npl, ne])
        return transmission_list, np.array(dos_list)

    def calculate_eigen_transport_channels(self, s, q):
        energies = self.eig_trans_channel_energies
        ne = len(energies)
        nl = self.tp.lead_num
        total_w = []
        total_v = []
        total_vc = []
        for n in range(ne):
            total_w.append([])
            total_v.append([])
            total_vc.append([])
            t, vc, k, vl = self.central_scattering_states(
                                                    energies[n], s, q)
            for i in range(nl):
                total_w[n].append([])
                total_v[n].append([])
                total_vc[n].append([])
                for j in range(nl):
                    zeta = t[i][j]
                    zeta2 = np.dot(zeta.T.conj(), zeta)
                    w, v = np.linalg.eig(zeta2)
                    total_w[n][i].append(w)
                    total_v[n][i].append(v)
                    total_vc[n][i].append(np.dot(vc[i], v))
        return np.array(total_w), np.array(total_v), np.array(total_vc)
    
    def calculate_charge_distribution(self, s, q):
        d_mm = self.tp.hsd.D[s][q].recover(True)
        s_mm = self.tp.hsd.S[q].recover(True)
        q_mm = np.dot(d_mm, s_mm)
        return np.diag(q_mm).real

    def calculate_isolate_molecular_levels(self):
        atoms = self.isolate_atoms
        atoms.pbc = True
        p = self.tp.gpw_kwargs.copy()
        p['nbands'] = None
        if 'mixer' in p:
            if not self.tp.spinpol:
                p['mixer'] = Mixer(0.1, 5, weight=100.0)
            else:
                p['mixer'] = MixerDif(0.1, 5, weight=100.0)
        p['poissonsolver'] = PoissonSolver(nn=2)
        if 'txt' in p and p['txt'] != '-':
            p['txt'] = 'isolate_' + p['txt']
        atoms.set_calculator(GPAW(**p))
        atoms.get_potential_energy()
        setups = atoms.calc.wfs.setups
        self.project_basis_in_molecule = get_atom_indices(
                                      self.project_atoms_in_molecule, setups)         
        kpt = atoms.calc.wfs.kpt_u[0]
        s_mm = atoms.calc.wfs.S_qMM[0]
        c_nm = eig_states_norm(kpt.C_nM, s_mm)
        self.isolate_eigen_values = kpt.eps_n
        self.isolate_eigen_vectors = c_nm
        self.isolate_s_mm = s_mm

    def calculate_project_transmission(self, s, q):
        project_transmission = []
        if self.project_molecular_levels is not None:
            eps_n, c_nm, s_mm = self.isolate_eigen_values, \
                                 self.isolate_eigen_vectors, self.isolate_s_mm
            ne = len(self.energies)
            nl = self.tp.lead_num
            T0 = np.zeros(c_nm.shape[0])
            ind1 = self.project_basis_in_molecule
            ind2 = self.project_basis_in_device
            for i in range(ne):
                total_t, total_vc, total_k, total_vl = \
                                   self.get_central_scattering_states(s, q, i)
                project_transmission.append([])
                for j in range(nl):
                    project_transmission[i].append([])
                    for k in range(nl):
                        vs = total_vc[k][ind2]
                        vm = np.dot(np.dot(c_nm.T.conj(), s_mm)[:, ind1], vs)
                        t0 = total_t[j][k]
                        if len(t0) > 0:
                            pt = vm * vm.conj() * \
                                     np.diag(np.dot(t0.T.conj(), t0)) \
                                          / np.diag(np.dot(vm.T.conj(), vm))
                        else:
                            pt = T0
                        project_transmission[i][j].append(pt)
        return np.array(project_transmission)

    def calculate_realspace_wave_functions(self, C_nm, q):
        #nl number of molecular levels
        wfs = self.tp.extended_calc.wfs
        nao = wfs.setups.nao
        nb, nl = C_nm.shape
        if wfs.dtype == float:
            C_nm = C_nm.real.copy()
        #extended_C_nm = np.zeros([nl, nao], wfs.dtype)
        total_psi_g = []
        for i in range(nl):
            psi_g = self.tp.gd.zeros(nl, dtype=wfs.dtype)
            c_nm = C_nm.reshape(1, -1)
            psi_g = psi_g.reshape(1, -1)
            wfs.basis_functions.lcao_to_grid(c_nm, psi_g, q)
            psi_g.shape = self.tp.gd.n_c
            total_psi_g.append(psi_g / Bohr**1.5)
        return np.array(total_psi_g)

    def get_left_channels(self, energy, s, k):
        g_s_ii, sigma = self.calculate_green_function_of_k_point(s, k,
                                                              energy, 1, True)
        nb = g_s_ii.shape[-1]
        dtype = g_s_ii.dtype
        lambda_l_ii = np.zeros([nb, nb], dtype)
        lambda_r_ii = np.zeros([nb, nb], dtype)
        ind = get_matrix_index(self.tp.hsd.S[0].ll_index[0][-1])
        lambda_l_ii[ind.T, ind] = 1.j * (sigma[0].recover() -
                                                  sigma[0].recover().T.conj())
        ind = get_matrix_index(self.tp.hsd.S[0].ll_index[1][-1])        
        lambda_r_ii[ind.T, ind] = 1.j * (sigma[1].recover() -
                                                  sigma[1].recover().T.conj())
        s_mm = self.tp.hsd.S[k].recover()
        s_s_i, s_s_ii = np.linalg.eig(s_mm)
        s_s_i = np.abs(s_s_i)
        s_s_sqrt_i = np.sqrt(s_s_i) # sqrt of eigenvalues  
        s_s_sqrt_ii = np.dot(s_s_ii * s_s_sqrt_i, dagger(s_s_ii))
        s_s_isqrt_ii = np.dot(s_s_ii / s_s_sqrt_i, dagger(s_s_ii))

        lambdab_r_ii = np.dot(np.dot(s_s_isqrt_ii, lambda_r_ii),s_s_isqrt_ii)
        a_l_ii = np.dot(np.dot(g_s_ii, lambda_l_ii), dagger(g_s_ii))
        ab_l_ii = np.dot(np.dot(s_s_sqrt_ii, a_l_ii), s_s_sqrt_ii)
        lambda_i, u_ii = np.linalg.eig(ab_l_ii)
        ut_ii = np.sqrt(lambda_i / (2.0 * np.pi)) * u_ii
        m_ii = 2 * np.pi * np.dot(np.dot(dagger(ut_ii), lambdab_r_ii),ut_ii)
        T_i,c_in = np.linalg.eig(m_ii)
        T_i = np.abs(T_i)
        channels = np.argsort(-T_i)
        c_in = np.take(c_in, channels, axis=1)
        T_n = np.take(T_i, channels)
        v_in = np.dot(np.dot(s_s_isqrt_ii, ut_ii), c_in)
        return T_n, v_in
    
    def calculate_realspace_dos(self, s, q):
        energies = self.dos_realspace_energies
        realspace_dos = []
        wfs = self.tp.extended_calc.wfs        
        for energy in energies:
            gr = self.calculate_green_function_of_k_point(s, q, energy)
            dos_mm = np.dot(gr, self.tp.hsd.S[q].recover())
            if wfs.dtype == float:
                dos_mm = np.real(dos_mm).copy()
                #kpt.rho_MM = dos_mm
                #wfs.add_to_density_from_k_point(dos_g, kpt)
            #wfs.kpt_comm.sum(dos_g)
            #total_dos_g = self.tp.gd1.collect(dos_g)
            #realspace_dos.append(total_dos_g)
            realspace_dos.append(dos_mm)
        return realspace_dos

    def calculate_partial_dos(self, s, q, atom_indices=None,
                                                          orbital_type='all'):
        energies = self.energies
        dos = []
        orbital_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
        orbital_indices = self.tp.orbital_indices     
        for energy in energies:
            gr = self.calculate_green_function_of_k_point(s, q, energy)
            dos_mm = -np.imag(np.dot(gr, self.tp.hsd.S[q].recover())) / np.pi
            dos_line = np.diag(dos_mm)
            tmp = 0
            if orbital_type == 'all':
                orbital_index = np.zeros([orbital_indices.shape[0]]) + 1
            else:
                orbital_index = orbital_indices[:, 1] - orbital_map[
                                                      orbital_type] ==  0 
            for j in atom_indices:
                atom_index = orbital_indices[:, 0] - j == 0
                tmp += np.sum(dos_line * atom_index * orbital_index)
            dos.append(tmp)
        return dos
       
    def lead_k_matrix(self, l, s, pk, k_vec, hors='S'):
        tp = self.tp
        if hors == 'S':
            h00 = tp.lead_hsd[l].S[pk].recover()
            h01 = tp.lead_couple_hsd[l].S[pk].recover()
        else:
            h00 = tp.lead_hsd[l].H[s][pk].recover()
            h01 = tp.lead_couple_hsd[l].H[s][pk].recover()
        h01 *=  np.exp(2 * np.pi * 1.j * k_vec)
        return h00 + h01 + h01.T.conj()
      
    def lead_scattering_states(self, energy, l, s, q):
        #Calculating the scattering states in electrodes
        #l ---- index of electrode
        #s ---- index of spin
        #q ---- index of local k point
        
        #if it is multi-terminal system, should add a part that can rotate
        # the lead hamiltonian
        MaxLambda = 1e2
        MinErr = 1e-8
        tp = self.tp
        energy += tp.bias[l]
        hes00 = tp.lead_hsd[l].H[s][q].recover() - \
                                    tp.lead_hsd[l].S[q].recover() * energy
        hes01 = tp.lead_couple_hsd[l].H[s][q].recover() - \
                             tp.lead_couple_hsd[l].S[q].recover() * energy
        nb = hes00.shape[-1]
        dtype = hes00.dtype
        A = np.zeros([2*nb, 2*nb], dtype)
        B = np.zeros([2*nb, 2*nb], dtype)
            
        A[:nb, nb:2*nb] = np.eye(nb)
        A[nb:2*nb, :nb] = hes01.T.conj()
        A[nb:2*nb, nb:2*nb] = hes00
            
        B[:nb, :nb] = np.eye(nb)
        B[nb:2*nb, nb:2*nb] = -hes01
            
        from scipy.linalg import eig
        D, V = eig(A, B)
        index = np.argsort(abs(D))
        D = D[index]
        V = V[:, index]
        
        #delete NaN
        index = find(np.abs(D) >= 0)
        D = D[index]
        V = V[:, index]
        
        #delete some unreasonable solutions
        index = find(abs(D) > MaxLambda)
        cutlen = len(D) - index[0]
        index = np.arange(cutlen, len(D) - cutlen)
        D = D[index]
        V = V[:, index]
            
        k = np.log(D) * (-1.j) / 2 / np.pi
        Vk = V[:nb]
            
        #sort scattering states 
        proindex = find(abs(k.imag) < MinErr)
            
        if len(proindex) > 0:
            k_sort = np.sort(k[proindex].real)
            index = np.argsort(k[proindex].real)
            k[proindex] = k[proindex[index]].real
            Vk[:, proindex] = Vk[:, proindex[index]]
                
            #normalization the scattering states
            j = proindex[0]
            while j <= proindex[-1]:
                same_k_index = find(abs((k - k[j]).real) < MinErr)
                sk = self.lead_k_matrix(l, s, q, k[j])
                Vk[:, same_k_index] = eig_states_norm(Vk[:, same_k_index],
                                                                         sk)
                j += len(same_k_index)
        return np.array(k[proindex]), np.array(Vk[:, proindex])
                   
    def central_scattering_states(self, energy, s, q):
        MaxLambda = 1e2
        MinErr = 1e-8
        tp = self.tp        
        bc = tp.inner_mol_index
        nc = len(bc)
        molhes = tp.hsd.H[s][q].recover(True) - \
                              tp.hsd.S[q].recover(True) * energy
        blead = []
        lead_hes = []
        lead_couple_hes = []
        for i in range(tp.lead_num):
            blead.append(tp.lead_layer_index[i][-1])
            lead_hes.append(tp.lead_hsd[i].H[s][q].recover() -
                                       tp.lead_hsd[i].S[q].recover() * energy)
            lead_couple_hes.append(tp.lead_couple_hsd[i].H[s][q].recover() -
                                tp.lead_couple_hsd[i].S[q].recover() * energy)

        ex_ll_index = tp.hsd.S[0].ex_ll_index    
        
        total_k = []
        total_vk = []
        total_lam = []
        total_v = []
        total_pro_right_index = []
        total_pro_left_index = []
        total_left_index = []
        total_right_index = []
        total_kr = []
        total_kt = []
        total_lambdar = []
        total_lambdat = []
        total_vkr = []
        total_vkt = []
        total_vr = []
        total_vt = []
        total_len_k = []
        total_nblead = []
        total_bA1 = []
        total_bA2 = []
        total_bproin = []
        total_nbB2 = []
        total_bB2 = []
                
        for i in range(tp.lead_num):
            k, vk = self.lead_scattering_states(energy, i, s, q)
            total_k.append(k)
            total_vk.append(vk)
            
            lam = np.exp(2 * np.pi * k * 1.j)
            total_lam.append(lam)
            
            #calculating v = dE/dk
            de2 = 1e-8
            k2, vk2 = self.lead_scattering_states(energy + de2, i, s, q)
            v = de2 / (k2 - k) / 2 / np.pi 
            total_v.append(v)
            
            #seperating left scaterring states and right scattering states
            #left scattering: lead->mol, right scattering: mol->lead
            proindex = find(abs(k.imag) < MinErr)
            pro_left_index = proindex[find(v[proindex].real < 0)]
            pro_left_index = pro_left_index[-np.arange(len(pro_left_index))
                                                                          - 1]
            
            pro_right_index = proindex[find(v[proindex].real > 0)]
               
            left_index = find(k.imag > MinErr)
            left_index = np.append(left_index, pro_left_index)
            
            right_index = pro_right_index.copy()
            right_index = np.append(right_index, find(k.imag < -MinErr))
            
            total_pro_left_index.append(pro_left_index)
            total_pro_right_index.append(pro_right_index)
            total_left_index.append(left_index)
            total_right_index.append(right_index)
           
            kr = k[left_index]
            kt = k[right_index]
            total_kr.append(kr)
            total_kt.append(kt)
            
            lambdar = np.diag(np.exp(2 * np.pi * kr * 1.j))
            lambdat = np.diag(np.exp(2 * np.pi * kt * 1.j))
            total_lambdar.append(lambdar)
            total_lambdat.append(lambdat)
            
            vkr = vk[:, left_index]
            vkt = vk[:, right_index]
            vr = v[pro_left_index]
            vt = v[pro_right_index]
            total_vkr.append(vkr)
            total_vkt.append(vkt)
            total_vr.append(vr)
            total_vt.append(vt)
            
            #abstract basis information
            len_k = len(right_index)
            total_len_k.append(len_k)
            
            #lead i basis seqeunce in whole matrix
            nblead = len(ex_ll_index[i][-1])
            total_nblead.append(nblead)
            bA1 = nc + int(np.sum(total_nblead[:i])) + np.arange(nblead)
            #sequence of the incident wave   
            bA2 = nc + int(np.sum(total_len_k[:i])) + np.arange(len_k)
            # the first n in vkt are scattering waves, the rest are the decaying ones
            # the first n in vkr are decaying waves, the rest are the scattering ones
            total_bA1.append(bA1)
            total_bA2.append(bA2)  
           
            bproin = np.arange(len(pro_right_index)) + \
                                          len(kr) - len(pro_right_index)
            ### this line need to check it...
            total_bproin.append(bproin)
            nbB2 = len(bproin)
            total_nbB2.append(nbB2)
            bB2 = int(np.sum(total_nbB2[:i])) + np.arange(nbB2)
            total_bB2.append(bB2)

        ind = get_matrix_index(bc)              
        Acc = molhes[ind.T, ind]
           
        total_Alc = []
        total_hcl = []
        total_All = []
        total_Acl = []
        total_Bc = []
        total_Bl = []    
            
        for i in range(tp.lead_num):
            ind1, ind2 = get_matrix_index(blead[i], bc)         
            Alc = molhes[ind1, ind2]
            ind1, ind2 = get_matrix_index(bc, blead[i])         
            hcl = molhes[ind1, ind2]
            vkt = total_vkt[i]
            All = np.dot(tp.lead_hsd[i].H[s][q].recover(), vkt) + \
                    np.dot(np.dot(tp.lead_couple_hsd[i].H[s][q].recover(), \
                                                     vkt), total_lambdat[i])
            Acl = np.dot(hcl, vkt)
            bpi = total_bproin[i]
            vkr = total_vkr[i]
            Bc = -np.dot(hcl, vkr[:, bpi])
            ind = get_matrix_index(bpi)
            Bl = -np.dot(tp.lead_hsd[i].H[s][q].recover(), vkr[:, bpi]) + \
                    np.dot(np.dot(tp.lead_couple_hsd[i].H[s][q].recover(),
                        vkr[:, bpi]) ,total_lambdar[i][ind.T, ind])
            total_Alc.append(Alc)
            total_Acl.append(Acl)
            total_hcl.append(hcl)
            total_All.append(All)
            total_Bc.append(Bc)
            total_Bl.append(Bl)
        total_bc = molhes.shape[-1]
        MatA = np.zeros([total_bc, nc + np.sum(total_len_k)], complex)    
        MatB = np.zeros([total_bc, np.sum(total_nbB2)], complex)
        ind = get_matrix_index(np.arange(nc))
        MatA[ind.T, ind] = Acc
        for i in range(tp.lead_num):
            ind1, ind2 = get_matrix_index(total_bA1[i], total_bA2[i])
            MatA[ind1, ind2] = total_All[i]
            ind1, ind2 = get_matrix_index(np.arange(nc), total_bA2[i])
            MatA[ind1, ind2] = total_Acl[i]
            ind1, ind2 = get_matrix_index(total_bA1[i], np.arange(nc))
            MatA[ind1, ind2] = total_Alc[i]
            ind1, ind2 = get_matrix_index(np.arange(nc), total_bB2[i])
            MatB[ind1, ind2] = total_Bc[i]
            ind1, ind2 = get_matrix_index(total_bA1[i], total_bB2[i])
            MatB[ind1, ind2] = total_Bl[i]
        Vx, residues, rank, singular = np.linalg.lstsq(MatA, MatB)
        
        total_vl = []
        for i in range(tp.lead_num):
            total_k[i] = total_k[i][total_pro_right_index[i]]
            total_vl.append(total_vk[i][:, total_pro_right_index[i]])
         
        total_vc = []
        t = []
        for i in range(tp.lead_num):
            t.append([])
            for j in range(tp.lead_num):
                t[i].append([])
        for i in range(tp.lead_num):
            ind1, ind2 = get_matrix_index(np.arange(nc), total_bB2[i])
            total_vc.append(Vx[ind1, ind2])
            for j in range(tp.lead_num):
                bx = total_bA2[j]
                bx = bx[:len(total_pro_right_index[j])]
                ind1, ind2 = get_matrix_index(bx, total_bB2[i])
                #t[j].append(Vx[ind1, ind2])
                t[j][i] = Vx[ind1, ind2]
                for m in range(len(total_pro_left_index[i])):
                    for n in range(len(total_pro_right_index[j])):
                        t[j][i][n, m] *= np.sqrt(abs(total_vt[i][n]/
                                                          total_vr[j][m]))
        return np.array(t), np.array(total_vc), \
                                         np.array(total_k), np.array(total_vl)
            
    def cal_sstates(self, energies, s, q):
        MinErr = 1e-8
        MaxLambda = 1 + MinErr
        
        energies = np.real(energies)
        ne = len(energies)
        tp  = self.tp
        t_all = np.zeros([ne, tp.lead_num, tp.lead_num])
        t_lead_all = []
        for i in range(tp.lead_num):
            t_lead_all.append([])
            for j in range(tp.lead_num):
                t_lead_all[i].append(np.zeros([tp.nblead[i],
                                                     tp.nblead[j], ne]))
        total_t = []
        total_vc = []
        total_k = []
        total_vl = []
        sk1 = []
        sk2 = []
        for n in range(ne):
            t, vc, k, vl = self.get_central_scattering_states(s, q, n)
            total_t.append(t)
            total_vc.append(vc)
            total_k.append(k)
            total_vl.append(vl)
            for i in range(tp.lead_num):
                for j in range(tp.lead_num):
                    t0 = t[i][j]
                    t_all[n, i, j] = np.sum(np.dot(t0.T.conj(), t0))
                    for m1 in range(t0.shape[0]):
                        sk1.append(self.lead_k_matrix(i, s, q, k[i][m1], 'S'))
                    for m2 in range(t0.shape[1]):
                        sk2.append(self.lead_k_matrix(i, s, q, k[j][m2], 'S'))
                    for m1 in range(t0.shape[0]):
                        for m2 in range(t0.shape[1]):
                            w1 = (vl[j][:, m1].conj() * np.dot(sk1[m1],
                                                        vl[j][:, m1])).real
                            w1[find(w1 < 0)] = 0
                            w1 /= np.sum(w1)
                            
                            w2 = (vl[i][:, m2].conj() * np.dot(sk2[m2],
                                                        vl[i][:, m2])).real
                            w2[find(w2 < 0)] = 0                        
                            w2 /= np.sum(w2)
                            
                            t_lead_all[i][j][:, :, n] += abs(t0[m1,
                                          m2]) ** 2 * np.dot(w2, w1.T.conj())
        return np.array(t_lead_all), np.array(t_all)
    
    def set_analysis_data_form(self):
        tp = self.tp
        ns, npk = tp.nspins, tp.npk
        nx, ny, nz = tp.gd.N_c
        exnb = tp.extended_calc.wfs.setups.nao
        nb = tp.nbmol
        ne = len(self.energies)
        na = len(tp.atoms)
        dtype = tp.wfs.dtype
        self.analysis_data_form = {}
        adf = self.analysis_data_form
        adf['nt'] = ((ns, nz), float)
        adf['vt'] = ((ns, nz), float)
        adf['df'] = ((ns, npk, exnb), dtype)
        adf['dd'] = ((ns, npk, exnb), dtype)
        adf['vHt'] = ((nz * 2,), float)
        adf['rho'] = ((nz * 2,), float)
        adf['dos'] = ((ns, npk, ne), float)
        adf['tc'] = ((ns, npk, ne), float)
        adf['force'] = ((na, 3), float)
        adf['charge'] = ((ns, nb), float)
        adf['ntx'] = ((ns, nx, nz), float)
        adf['nty'] = ((ns, ny, nz), float)
        adf['vtx'] = ((ns, nx, nz), float)
        adf['vty'] = ((ns, ny, nz), float)
        adf['current'] = ((1,), float)
 

    def save_ele_step(self):
        tp = self.tp
        step = Electron_Step_Info(self.n_ion_step, self.n_bias_step,
                                                           self.n_ele_step)
        dtype = tp.wfs.dtype
        nbmol = tp.extended_calc.wfs.setups.nao
        dd = np.empty([tp.my_nspins, tp.my_npk, nbmol], dtype)
        df = np.empty([tp.my_nspins, tp.my_npk, nbmol], dtype)
        
        if tp.wfs.kpt_comm.rank == 0:
            total_dd = np.empty([tp.nspins, tp.npk, nbmol], dtype)
            total_df = np.empty([tp.nspins, tp.npk, nbmol], dtype)
        else:
            total_dd = None
            total_df = None
        
        if not self.matrix_foot_print:
            fd = file('matrix_sample', 'wb')
            cPickle.dump(tp.hsd.S[0], fd, 2)
            fd.close()
            self.matrix_foot_print = True
            
        for s in range(tp.my_nspins):
            for k in range(tp.my_npk):
                dd[s, k] = np.diag(tp.hsd.D[s][k].recover(True))
                df[s, k] = np.diag(tp.hsd.H[s][k].recover(True))
       
        
        tp.wfs.kpt_comm.gather(dd, 0, total_dd)
        tp.wfs.kpt_comm.gather(df, 0, total_df)

        dim = tp.gd.N_c
        assert tp.d == 2
        
        calc = tp.extended_calc
        gd = calc.gd
        finegd = calc.hamiltonian.finegd

        nt_sG = tp.gd.collect(tp.density.nt_sG)
        vt_sG = gd.collect(calc.hamiltonian.vt_sG)
        
        if world.rank == 0:
            nt = []
            vt = []
            for s in range(tp.nspins): 
                nts = aa1d(nt_sG[s]) 
                vts = aa1d(vt_sG[s]) * Hartree
                nt1 = aa1d(tp.surround.sides['-'].boundary_nt_sG[s])
                nt2 = aa1d(tp.surround.sides['+'].boundary_nt_sG[s])
                nts = np.append(nt1, nts)
                nts = np.append(nts, nt2)
                nt.append(nts)
                vt.append(vts)
        else:
            nt = None
            vt = None
        nt = np.array(nt)
        vt = np.array(vt)
         
        gd = tp.finegd
        rhot_g = gd.collect(tp.density.rhot_g)

        if world.rank == 0:
            rho1 = tp.surround.sides['-'].boundary_rhot_g_line
            rho2 = tp.surround.sides['+'].boundary_rhot_g_line
            rho = aa1d(rhot_g)
            rho = np.append(rho1, rho)
            rho = np.append(rho, rho2)
        else:
            rho = None
            
        gd = finegd
        vHt_g = gd.collect(calc.hamiltonian.vHt_g)
        if world.rank == 0:
            vHt = aa1d(vHt_g) * Hartree
        else:
            vHt = None
          
        mem_cost = maxrss()
        if (not self.tp.non_sc) and (self.tp.analysis_mode >= 0) :  
            time_cost = self.ele_step_time_collect()
        else:
            time_cost = None
      
        step.initialize_data(vt, nt, df, dd, vHt, rho, time_cost, mem_cost)
        self.ele_steps.append(step)
        self.n_ele_step += 1
      
    def save_bias_step(self):
        if not self.overhead_data_saved:
            self.save_overhead_data()
            self.overhead_data_saved = True
        tp = self.tp
        step = Transmission_Info(self.n_ion_step, self.n_bias_step)
        time_cost = self.bias_step_time_collect()
        if 'tc' in tp.analysis_data_list:
            tc, dos = self.collect_transmission_and_dos()
        else:
            tc = None
            dos = None
        nt, vt, ntx, vtx, nty, vty = self.abstract_d_and_v()
                    
        if not tp.non_sc and 'current' in tp.analysis_data_list:
            current = self.calculate_current3(tc, 0)
        else:
            current = None
        if tp.non_sc or self.tp.analysis_mode < 0:
            force = None
        else:       
            force = tp.calculate_force()
            tp.F_av = None
        
        charge = self.collect_charge()
        if tp.non_sc or tp.analysis_mode < 0:
            contour = None
        else:    
            contour = self.collect_contour()
        dos_dict = {}
        tc_dict = {}
        if 'tc' in tp.analysis_data_list:            
            tc_dict['E' + str(tp.contour.comm.rank)] = tc
            dos_dict['E' + str(tp.contour.comm.rank)] = dos
        total_tc = gather_ndarray_dict(tc_dict, tp.contour.comm)
        total_dos = gather_ndarray_dict(dos_dict, tp.contour.comm)        
            
        step.initialize_data(tp.bias, tp.gate, self.energies, self.lead_pairs,
                              total_tc, total_dos, vt, nt, vtx, ntx, vty, nty,
                              current, tp.lead_fermi, time_cost, force, charge, contour)


        #prefix =  'Ab' + '_' + str(self.n_ion_step) + '_' \
        #                    + str(self.n_bias_step) + '_' \
                            
        #for name in ['nt', 'vt', 'ntx', 'vtx', 'nty', 'vty',
        #                    'dos', 'tc', 'current']:
        #    dimension, dtype = self.analysis_data_form[name]
        #    data_name = prefix + name
        #    write('analysis_data.nc', data_name, eval(name), dimension, dtype)
            
            
        if tp.analysis_mode == 2:  
            self.reset_central_scattering_states()
            eig_tc_lead, eig_vc_lead = \
                                       self.collect_lead_scattering_channels()
            tp_tc, tp_vc, lead_k, lead_vk = \
                                       self.collect_scat_scattering_channels()
            tp_eig_w, tp_eig_v, tp_eig_vc = \
                                       self.collect_eigen_transport_channels() 
        
            project_tc = self.collect_project_transmission()
            dos_g = self.collect_realspace_dos()
            left_tc, left_vc = self.collect_left_channels()
 
            step.initialize_data2(eig_tc_lead, eig_vc_lead, tp_tc, tp_vc,
                                 dos_g, project_tc, left_tc, left_vc, lead_k,
                                 lead_vk, tp_eig_w, tp_eig_v, tp_eig_vc)
        elif tp.analysis_mode == -2:
            #lead_s00, lead_s01, lead_h00, lead_h01 = self.collect_lead_hs()
            pass
            #s00, h00 = self.collect_scat_hs()
            #step.initialize_data3(s00, h00, tp.lead_fermi)
            
        #step.ele_steps = self.ele_steps
        #del self.ele_steps
        self.ele_steps = []
        self.n_ele_step = 0
        self.bias_steps.append(step)
        self.n_bias_step += 1

    def collect_lead_scattering_channels(self):
        tp = self.tp
        energies = self.eig_trans_channel_energies
        if energies is not None:
            nl = tp.lead_num
            ne = len(energies)
            ns, npk = tp.nspins, tp.npk
            nb = np.max(tp.nblead)
            mkoe = self.max_k_on_energy
            dtype = tp.wfs.dtype
            tc_array = np.zeros([ns, npk, ne, nl, mkoe], dtype)
            vc_array = np.zeros([ns, npk, ne, nl, nb, mkoe], dtype)

            ns, npk = tp.my_nspins, tp.my_npk
            local_tc_array = np.zeros([ns, npk, ne, nl, mkoe], dtype)
            local_vc_array = np.zeros([ns, npk, ne, nl, nb, mkoe], dtype)
            for s in range(ns):
                for q in range(npk):
                    for e, energy in enumerate(energies):
                        for l in range(nl):
                            tc, vc = self.lead_scattering_states(energy,
                                                                 l, s, q)
                            nbl, nbloch = vc.shape
                            local_tc_array[s, q, e, l, :nbloch] = tc
                            local_vc_array[s, q, e, l, :nbl, :nbloch] = vc 
                 
            kpt_comm = tp.wfs.kpt_comm
            kpt_comm.all_gather(local_tc_array, tc_array)
            kpt_comm.all_gather(local_vc_array, vc_array)             
            return tc_array, vc_array
        else:
            return None, None
       
    def collect_scat_scattering_channels(self):
        tp = self.tp
        energies = self.eig_trans_channel_energies
        if energies is not None:
            nl = tp.lead_num
            ne = len(energies)
            ns, npk = tp.nspins, tp.npk
            nbmol = tp.nbmol
            nb = np.max(tp.nblead)
            mkoe = self.max_k_on_energy
            dtype = tp.wfs.dtype
            tc_array = np.zeros([ns, npk, ne, nl, nl, mkoe, mkoe], dtype)
            vc_array = np.zeros([ns, npk, ne, nl, nbmol, mkoe], dtype)
            k_array = np.zeros([ns, npk, ne, nl, mkoe], dtype)
            vk_array = np.zeros([ns, npk, ne, nl, nb, mkoe], dtype)
            
            ns, npk = tp.my_nspins, tp.my_npk
            local_tc_array = np.zeros([ns, npk, ne, nl, nl, mkoe, mkoe],
                                                                        dtype)
            local_vc_array = np.zeros([ns, npk, ne, nl, nbmol, mkoe], dtype)
            local_k_array = np.zeros([ns, npk, ne, nl, mkoe], dtype)
            local_vk_array = np.zeros([ns, npk, ne, nl, nb, mkoe], dtype)
            for s in range(ns):
                for q in range(npk):
                    for e, energy in enumerate(energies):
                        tc, vc, k, vk = self.central_scattering_states(energy,
                                                                       s, q)
                        nbl, nbloch = vk.shape[-2:]
                        local_tc_array[s, q, e, :, :, :nbloch, :nbloch] = tc
                        local_vc_array[s, q, e, :, :, :nbloch] = vc
                        local_k_array[s, q, e, :, :nbloch] = k
                        local_vk_array[s, q, e, :, :nbl, :nbloch] = vk
                 
            kpt_comm = tp.wfs.kpt_comm
            kpt_comm.all_gather(local_tc_array, tc_array)
            kpt_comm.all_gather(local_vc_array, vc_array)
            kpt_comm.all_gather(local_k_array, k_array)
            kpt_comm.all_gather(local_vk_array, vk_array)            
            return tc_array, vc_array, k_array, vk_array 
        else:
            return None, None, None, None        

    def collect_charge(self):
        tp = self.tp
        ns, npk = tp.nspins, tp.npk
        nbmol = tp.nbmol + np.sum(tp.nblead)
        kpt_comm = tp.wfs.kpt_comm
        if kpt_comm.rank == 0:
            charge_array = np.zeros([ns, npk, nbmol])
        else:
            charge_array = None
        ns, npk = tp.my_nspins, tp.my_npk
        local_charge_array = np.zeros([ns, npk, nbmol])
        for s in range(ns):
            for q in range(npk):
                    local_charge_array[s, q] = \
                                     self.calculate_charge_distribution(s, q)
        kpt_comm.gather(local_charge_array, 0, charge_array)
        if world.rank == 0:
            charge_array = np.sum(charge_array, axis=1)
        return charge_array

    def collect_contour(self):
        tp = self.tp
        my_eq_contour = {}
        my_ne_contour = {}
        my_loc_contour = {}
        num = 0
        for s in range(tp.my_nspins):
            for q in range(tp.my_npk):
                flag = str(tp.my_ks_map[num, 0]) + str(tp.my_ks_map[num, 1])
                my_eq_contour[flag] = np.array(tp.eqpathinfo[s][q].energy)
                my_ne_contour[flag] = np.array(tp.nepathinfo[s][q].energy)
                if not tp.ground:
                    my_loc_contour[flag] = np.array(tp.locpathinfo[s][q].energy)        
        eq_contour = gather_ndarray_dict(my_eq_contour, tp.wfs.kpt_comm)
        ne_contour = gather_ndarray_dict(my_ne_contour, tp.wfs.kpt_comm)        
        if not tp.ground:
            loc_contour = gather_ndarray_dict(my_loc_contour, tp.wfs.kpt_comm)
        else:
            loc_contour = None
        contour = {'eq': eq_contour, 'ne': ne_contour, 'loc': loc_contour}
        return contour
        
    def collect_realspace_dos(self):
        if self.dos_realspace_energies is None:
            return None
        tp = self.tp
        ns, npk = tp.nspins, tp.npk
        nbmol = tp.nbmol
        dtype = tp.wfs.dtype
        energies = self.dos_realspace_energies
        ne = len(energies)
        dos_array = np.zeros([ns, npk, ne, nbmol, nbmol], dtype)
        ns, npk = tp.my_nspins, tp.my_npk
        local_dos_array = np.zeros([ns, npk, ne, nbmol, nbmol], dtype)
        for s in range(ns):
            for q in range(npk):
                local_dos_array[s, q] = self.calculate_realspace_dos(s, q)
        kpt_comm = tp.wfs.kpt_comm
        kpt_comm.all_gather(local_dos_array, dos_array)
        return dos_array        

    def collect_project_transmission(self):
        if self.isolate_atoms is None:
            return None
        tp = self.tp
        ns, npk = tp.nspins, tp.npk
        nbmol = tp.nbmol
        energies = self.energies
        ne = len(energies)
        nl = tp.lead_num
        nbi = len(self.isolate_eigen_values)
        project_tc = np.zeros([ns, npk, ne, nl, nl, nbi, 1])
        ns, npk = tp.my_nspins, tp.my_npk
        local_project_tc = np.zeros([ns, npk, ne, nl, nl, nbi, 1])
        for s in range(ns):
            for q in range(npk):
                local_project_tc[s, q] = \
                                     self.calculate_project_transmission(s, q)
        kpt_comm = tp.wfs.kpt_comm
        kpt_comm.all_gather(local_project_tc, project_tc)
        return project_tc

    def collect_left_channels(self):
        if self.eig_trans_channel_energies is None:
            return None, None
        tp = self.tp
        ns, npk = tp.nspins, tp.npk
        nbmol = tp.nbmol
        dtype = tp.wfs.dtype
        energies = self.eig_trans_channel_energies
        ne = len(energies)
        tc_array = np.zeros([ns, npk, ne, nbmol], dtype)
        vc_array = np.zeros([ns, npk, ne, nbmol, nbmol], dtype)        
        ns, npk = tp.my_nspins, tp.my_npk
        local_tc_array = np.zeros([ns, npk, ne, nbmol], dtype)
        local_vc_array = np.zeros([ns, npk, ne, nbmol, nbmol], dtype)
        for s in range(ns):
            for q in range(npk):
                for e, energy in enumerate(energies):
                    local_tc_array[s, q, e], local_vc_array[s, q, e] = \
                                          self.get_left_channels(energy, s, q)
        kpt_comm = tp.wfs.kpt_comm
        kpt_comm.all_gather(local_tc_array, tc_array)
        kpt_comm.all_gather(local_vc_array, vc_array)        
        return tc_array, vc_array 
       
    def collect_eigen_transport_channels(self):
        if self.eig_trans_channel_energies is None:
            return None, None, None
        tp = self.tp
        ns, npk = tp.nspins, tp.npk
        nbmol = tp.nbmol
        nl = tp.lead_num
        mkoe = self.max_k_on_energy
        dtype = tp.wfs.dtype
        energies = self.eig_trans_channel_energies
        ne = len(energies)
        w_array = np.zeros([ns, npk, ne, nl, nl, mkoe], dtype)
        v_array = np.zeros([ns, npk, ne, nl, nl, mkoe, mkoe], dtype)
        vc_array = np.zeros([ns, npk, ne, nl, nl, nbmol, mkoe], dtype)        
        ns, npk = tp.my_nspins, tp.my_npk
        local_w_array = np.zeros([ns, npk, ne, nl, nl, mkoe], dtype)
        local_v_array = np.zeros([ns, npk, ne, nl, nl, mkoe, mkoe], dtype)
        local_vc_array = np.zeros([ns, npk, ne, nl, nl, nbmol, mkoe], dtype) 
        for s in range(ns):
            for q in range(npk):
                w, v, vc = self.calculate_eigen_transport_channels(s, q)
                nbloch = w.shape[-1]
                local_w_array[s, q, :, :, :, :nbloch] = w
                local_v_array[s, q, :, :, :, :nbloch, :nbloch] = v
                local_vc_array[s, q, :, :, :, :, :nbloch] = vc
        kpt_comm = tp.wfs.kpt_comm
        kpt_comm.all_gather(local_w_array, w_array)
        kpt_comm.all_gather(local_v_array, v_array)        
        kpt_comm.all_gather(local_vc_array, vc_array) 
        return w_array, v_array, vc_array        

    def collect_lead_hs(self):
        tp = self.tp
        ns, npk = tp.nspins, tp.npk
        dtype = tp.wfs.dtype
        nl = tp.lead_num
        nb = tp.nblead[0]

        kpt_comm = tp.wfs.kpt_comm        
        if kpt_comm.rank == 0:
            s00 = np.zeros([npk, nl, nb, nb], dtype)
            s01 = np.zeros([npk, nl, nb, nb], dtype)        
            h00 = np.zeros([ns, npk, nl, nb, nb], dtype)
            h01 = np.zeros([ns, npk, nl, nb, nb], dtype)
        else:
            s00 = None
            s01 = None
            h00 = None
            h01 = None

        ns, npk = tp.my_nspins, tp.my_npk
        local_s00 = np.zeros([npk, nl, nb, nb], dtype)
        local_s01 = np.zeros([npk, nl, nb, nb], dtype)        
        local_h00 = np.zeros([ns, npk, nl, nb, nb], dtype)
        local_h01 = np.zeros([ns, npk, nl, nb, nb], dtype)        
        for q in range(npk):
            for l in range(nl):
                local_s00[q, l] = tp.lead_hsd[l].S[q].recover()
                local_s01[q, l] = tp.lead_couple_hsd[l].S[q].recover()                
                for s in range(ns):
                    local_h00[s, q, l] = tp.lead_hsd[l].H[s][q].recover()
                    local_h01[s, q, l] = \
                                       tp.lead_couple_hsd[l].H[s][q].recover()
        
        kpt_comm.gather(local_s00, 0, s00)
        kpt_comm.gather(local_s01, 0, s01)        
        kpt_comm.gather(local_h00, 0, h00)                     
        kpt_comm.gather(local_h01, 0, h01)                
        return s00, s01, h00, h01
    
    def collect_scat_hs(self):
        tp = self.tp
        ns, npk = tp.nspins, tp.npk
        dtype = tp.wfs.dtype
        nb = tp.nbmol
        kpt_comm = tp.wfs.kpt_comm
        
        if kpt_comm.rank == 0:
            s00 = np.zeros([npk, nb, nb], dtype)
            h00 = np.zeros([ns, npk, nb, nb], dtype)
        else:
            s00 = None
            h00 = None
            
        ns, npk = tp.my_nspins, tp.my_npk
        local_s00 = np.zeros([npk, nb, nb], dtype)
        local_h00 = np.zeros([ns, npk, nb, nb], dtype)
     
        for q in range(npk):
            local_s00[q] = tp.hsd.S[q].recover()
            for s in range(ns):
                local_h00[s, q] = tp.hsd.H[s][q].recover()
        kpt_comm.gather(local_s00, 0, s00)
        kpt_comm.gather(local_h00, 0, h00)                     
        return s00, h00       
        
    def bias_step_time_collect(self):
        timers = self.tp.timer.timers
        cost = {}
        if not self.tp.non_sc and self.tp.analysis_mode >= 0:
            cost['init scf'] = timers['init scf', ]
        return cost
        
    def ele_step_time_collect(self):    
        timers = self.tp.timer.timers
        cost = {}
        cost['eq fock2den'] = timers['DenMM', 'eq fock2den']
        cost['ne fock2den'] = timers['DenMM', 'ne fock2den']
        cost['Poisson'] = timers['HamMM', 'Hamiltonian', 'Poisson']
        cost['construct density'] = timers['HamMM', 'construct density']
        cost['atomic density'] = timers['HamMM', 'atomic density']
        cost['atomic hamiltonian'] = timers['HamMM', 'Hamiltonian',
                                                    'atomic hamiltonian']

        if self.tp.step == 0:
            cost['project hamiltonian'] = 0
            cost['record'] = 0
        else:
            cost['project hamiltonian'] = timers['HamMM',
                                                        'project hamiltonian']
            cost['record'] = self.tp.record_time_cost
        return cost

    def collect_transmission_and_dos(self, energies=None, nids=None):
        if energies == None:
            energies = self.my_energies
        if nids == None:
            nids = self.my_nids
        tp = self.tp
      
        nlp = len(self.lead_pairs)
        ne = len(energies)
        ns, npk = tp.my_nspins, tp.my_npk
        local_tc_array = np.empty([ns, npk, nlp, ne], float)
        local_dos_array = np.empty([ns, npk, ne], float)
        
        for s in range(ns):
            for q in range(npk):
                local_tc_array[s, q], local_dos_array[s, q] = \
                          self.calculate_transmission_and_dos(s, q, energies, nids)
        return local_tc_array, local_dos_array

    def save_ion_step(self):
        tp = self.tp
        step = Structure_Info(self.n_ion_step)
        step.initialize_data(tp.atoms.positions, tp.forces)
        step.bias_steps = self.bias_steps
        del self.bias_steps
        self.bias_steps = []
        self.n_bias_step = 0
        self.ion_steps.append(step)
        self.n_ion_step += 1
 
    def save_data_to_file(self, flag='bias', data_file=None):
        if flag == 'ion':
            steps = self.ion_steps
        elif flag == 'bias':
            steps = self.bias_steps
        else:
            steps = self.ele_steps
        if data_file is None:
            data_file = 'analysis_data_' + flag
        else:
            data_file += '_' + flag
        if world.rank == 0:
            fd = file(data_file, 'wb')
            cPickle.dump((steps, self.energies), fd, 2)
            fd.close()
   
    def abstract_d_and_v(self):
        tp = self.tp
        calc = tp.extended_calc
        gd = calc.gd        

        nt_sG = tp.gd.collect(tp.density.nt_sG)
        vt_sG = gd.collect(calc.hamiltonian.vt_sG)

        nt = []
        vt = []
        ntx = []
        nty = []
        vtx = []
        vty = []
        if world.rank == 0:        
            for s in range(tp.nspins):
                nts = aa1d(nt_sG[s])
                vts = aa1d(vt_sG[s])
                ntsx = aa2d(nt_sG[s], 0)            
                vtsx = aa2d(vt_sG[s], 0)
                ntsy = aa2d(nt_sG[s], 1)            
                vtsy = aa2d(vt_sG[s], 1)                
                nt.append(nts)
                vt.append(vts)
                ntx.append(ntsx)
                vtx.append(vtsx)
                nty.append(ntsy)
                vty.append(vtsy)
            nt = np.array(nt)
            vt = np.array(vt)
            ntx = np.array(ntx)
            vtx = np.array(vtx)
            nty = np.array(nty)
            vty = np.array(vty)
        return nt, vt, ntx, vtx, nty, vty
    
    def calculate_current(self):
        # temperary, because different pk point may have different ep,
        # and also for multi-terminal, energies is wrong
        tp = self.tp
        assert hasattr(tp, 'nepathinfo')
        ep = np.array(tp.nepathinfo[0][0].energy)
        weight = np.array(tp.nepathinfo[0][0].weight)
        fermi_factor = np.array(tp.nepathinfo[0][0].fermi_factor)

        nep = np.array(ep.shape)
        ff_dim = np.array(fermi_factor.shape)

        world.broadcast(nep, 0)
        world.broadcast(ff_dim, 0)

        if world.rank != 0: 
            ep = np.zeros(nep, ep.dtype) 
            weight = np.zeros(nep, weight.dtype)
            fermi_factor = np.zeros(ff_dim, fermi_factor.dtype)
        world.broadcast(ep, 0)
        world.broadcast(weight, 0)
        world.broadcast(fermi_factor,0)
        
        tc_array, dos_array = self.collect_transmission_and_dos(ep)
        current = np.zeros([tp.nspins, len(self.lead_pairs)])
        tc_all = np.sum(tc_array, axis=1) / tp.npk
        
        #attention here, pk weight should be changed
        for s in range(tp.nspins):
            for i in range(len(self.lead_pairs)):
                for j in range(len(ep)):
                    current[s, i] += tc_all[s, i, j] * weight[j] * \
                                                       fermi_factor[i][0][j]
        return current

    def calculate_current_of_energy(self, epts, lead_pair_index, s):
        # temperary, because different lead_pairs have different energy points
        tp = self.tp
        tc_array, dos_array = self.collect_transmission_and_dos(epts)
        if world.rank == 0:
            tc_all = np.sum(tc_array, axis=1) / tp.npk
            #attention here, pk weight should be changed
            fd = fermidistribution
            intctrl = tp.intctrl
            kt = intctrl.kt
            lead_ef1 = intctrl.leadfermi[self.lead_pairs[lead_pair_index][0]]
            lead_ef2 = intctrl.leadfermi[self.lead_pairs[lead_pair_index][1]]
            fermi_factor = fd(epts - lead_ef1, kt) - fd(epts - lead_ef2, kt)
            current = tc_all[s, lead_pair_index] * fermi_factor
        else:
            current = None
        return current
    
    def calculate_current2(self, lead_pair_index=0, s=0):
        from scipy.integrate import simps
        intctrl = self.tp.intctrl
        kt = intctrl.kt
        lead_ef1 = intctrl.leadfermi[self.lead_pairs[lead_pair_index][0]]
        lead_ef2 = intctrl.leadfermi[self.lead_pairs[lead_pair_index][1]]
        if lead_ef2 > lead_ef1:
            lead_ef1, lead_ef2 = lead_ef2, lead_ef1
        lead_ef1 += 2 * kt
        lead_ef2 -= 2 * kt
        ne = int(abs(lead_ef1 -lead_ef2) / 0.02)
        epts = np.linspace(lead_ef1, lead_ef2, ne) + 1e-4 * 1.j
        interval = epts[1] - epts[0]
        #ne = len(self.energies)
        #epts = self.energies
        #interval = self.energies[1] - self.energies[0]
        cures = self.calculate_current_of_energy(epts, lead_pair_index, s)
        if world.rank == 0:
            if ne != 0:
                current =  simps(cures, None, interval)
            else:
                current = 0
        else:
            current = None
        return current

    def calculate_current3(self, tc_array, lead_pair_index=0, s=0):
        from scipy.integrate import simps
        tp = self.tp
        intctrl = tp.intctrl
        kt = intctrl.kt
        fd = fermidistribution        
        lead_ef1 = intctrl.leadfermi[self.lead_pairs[lead_pair_index][0]]
        lead_ef2 = intctrl.leadfermi[self.lead_pairs[lead_pair_index][1]]
        if lead_ef2 > lead_ef1:
            lead_ef1, lead_ef2 = lead_ef2, lead_ef1
        interval = np.real(self.my_energies[1] - self.my_energies[0])
        tc_all = np.sum(tc_array, axis=1) / tp.npk 
        fermi_factor = fd(self.my_energies - lead_ef1, kt) - fd(
                                             self.my_energies - lead_ef2, kt)
        #current =  simps(tc_all[s, lead_pair_index] * fermi_factor,
        #                                                       None, interval)
        current = np.sum(tc_all[s, lead_pair_index] * fermi_factor * self.my_weights)
        current = np.array(current)
        
        tp.contour.comm.sum(current)
        tp.wfs.kpt_comm.sum(current)
        return current    
         
class Transport_Plotter:
    flags = ['b-o', 'r--']
    #xlabel, ylabel, title, legend, xtick, ytick,
    
    def __init__(self, flag='bias', data_file=None):
        if data_file is None:
            data_file = 'analysis_data'
        data_file += '_' + flag
        fd = file(data_file, 'r')
        data = cPickle.load(fd)
        if flag == 'ion':
            if len(data) == 2:
                self.ion_steps, self.energies = data
            else:
                self.ion_steps = data
        elif flag == 'bias':
            if len(data) == 2:
                self.bias_steps, self.energies = data
            else:
                self.bias_steps = data
        else:
            if len(data) == 2:
                self.ele_steps, self.energies = data
            else:
                self.ele_steps = data
        fd.close()
        self.my_options = False
        self.show_window = False

    def read_overhead(self):
        fd = file('analysis_overhead', 'r')
        atoms, basis_information, contour_information = cPickle.load(fd)
        fd.close()
        self.atoms = atoms
        self.basis = basis_information
        self.contour = contour_information
 
    def plot_setup(self):
        from matplotlib import rcParams
        rcParams['xtick.labelsize'] = 18
        rcParams['ytick.labelsize'] = 18
        rcParams['legend.fontsize'] = 18
        rcParams['axes.titlesize'] = 18
        rcParams['axes.labelsize'] = 18
        rcParams['font.size'] = 18
        
    def set_default_options(self):
        self.xlabel = []
        self.ylabel = []
        self.legend = []
        self.title = []

    def set_my_options(self, xlabel=None, ylabel=None, legend=None,
                                                            title=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend
        self.title = title
        self.my_options = True
            
    def set_options(self, xlabel=None, ylabel=None, legend=None, title=None):
        if self.my_options:
            pass
        else:
            self.xlabel = xlabel
            self.ylabel = ylabel
            self.legend = legend
            self.title = title
      
    def show(self, p, option='default'):
        if not self.show_window:
            if option == None:
                p.show()
            elif option == 'default':
                if self.legend != None:
                    p.legend(self.legend)
                if self.xlabel != None:
                    p.xlabel(self.xlabel)
                if self.ylabel != None:
                    p.ylabel(self.ylabel)
                if self.title != None:
                    p.title(self.title)
                p.show()
            else:
                pass

    def tc(self, bias_step, s=0, k=0, all=True):
        step = self.bias_steps[bias_step]
        tc_all = step.tc['E0']
        num = 1
        flag = True
        while flag:
            flag = False
            for name in step.tc:
                if name[0] == 'E' and name[1] == str(num):
                    tc_all = np.append(tc_all, step.tc[name], axis=-1)
                    flag = True
                    num += 1
        if all:
            tc_all = np.sum(tc_all, axis=1) / tc_all.shape[1]
            tc_all = np.sum(tc_all, axis=1)[s]
        else:
            tc_all = tc_all[s, k, 0]
        return tc_all
  
    def set_ele_steps(self, n_ion_step=None, n_bias_step=0):
        if n_ion_step != None:
            self.bias_steps = self.ion_steps[n_ion_step].bias_steps
        self.ele_steps = self.bias_steps[n_bias_step].ele_steps
       
    def compare_two_calculations(self, nstep, s, k):
        fd = file('analysis_data_cmp', 'r')
        self.ele_steps_cmp, self.energies_cmp = cPickle.load(fd)
        fd.close()
        ee = self.energies
        #ee = np.linspace(-3, 3, 61)
        step = self.ele_steps[nstep]
        step_cmp = self.ele_steps_cmp[nstep]
        
        import pylab as p
        p.plot(step.dd[s, k] - step_cmp.dd[s, k], 'b--o')
        p.title('density matrix')
        p.show()
        
        p.plot(step.df[s, k] - step_cmp.df[s, k], 'b--o')
        p.title('hamiltonian matrix')
        p.show()
        
        p.plot(step.nt - step_cmp.nt, 'b--o')
        p.title('density')
        p.show()
 
        p.plot(step.vt - step_cmp.vt, 'b--o')
        p.title('hamiltonian')
        p.show()
        
        p.plot(ee, step.tc[s, k, 0] - step_cmp.tc[s, k, 0], 'b--o')
        p.title('transmission')
        p.show()
        
        p.plot(ee, step.dos[s, k] - step_cmp.dos[s, k], 'b--o')
        p.title('dos')
        p.show()
       
    def plot_ele_step_info(self, info, steps_indices, s, k,
                                                     height=None, unit=None):
        xdata = self.energies
        #xdata = np.linspace(-3, 3, 61)
        energy_axis = False        
        import pylab as p
        legends = []
        if info == 'dd':
            data = 'dd[s, k]'
            title = 'density matrix diagonal elements'
        elif info == 'df':
            data = 'df[s, k]'
            title = 'hamiltonian matrix diagonal elements'
        elif info == 'den':
            data = 'nt[s]'
            title = 'density'
        elif info == 'ham':
            data = 'vt[s]'
            title = 'hamiltonian'
        elif info == 'rho':
            data = 'rho'
            title = 'total poisson density'
        elif info == 'vHt':
            data = 'vHt'
            title = 'total Hartree potential'            
        elif info == 'tc':
            data = 'tc[s, k, 0]'
            title = 'trasmission coefficeints'
            energy_axis = True
        elif info == 'dos':
            data = 'dos[s, k]'
            title = 'density of states'
            energy_axis = True
        else:
            raise ValueError('no this info type---' + info)        

        for i, step in enumerate(self.ele_steps):
            if i in steps_indices:
                ydata = eval('step.' + data)
                if unit != None:
                    ydata = sum_by_unit(ydata, unit)
                if not energy_axis:
                    p.plot(ydata)
                else:
                    p.plot(xdata, ydata)
                legends.append('step' + str(step.ele_step))
        self.set_options(None, None, legends, title)
        if height != None:
            p.axis([xdata[0], xdata[-1], 0, height])
        #p.show()
        self.show(p)

    def plot_tc(self, bias_step, s=0, k=0, all=True):
        tc = self.tc(bias_step, s=s, k=k, all=all)
        title = 'trasmission coefficeints'
        xlabel = 'Energy(eV)'
        ylabel = 'T'        
        energies = self.energies
        eye = np.zeros([10, 1]) + 1
        step = self.bias_steps[bias_step]
        f1 = (step.lead_fermis[0] + step.bias[0]) * eye
        f2 = (step.lead_fermis[1] + step.bias[1]) * eye        
        a1 = np.max(tc)
        l1 = np.linspace(0, a1, 10)
        import pylab as p
        flags = self.flags
        p.plot(energies, tc, flags[0], f1, l1, flags[1],f2, l1, flags[1])        
        p.xlabel(xlabel)
        p.ylabel(ylabel)
        p.show()
        
    def plot_bias_step_info(self, info, steps_indices, s, k,
                                            height=None, unit=None,
                                        all=False, show=True, dense_level=0):
        xdata = np.real(self.energies)
        #xdata = np.linspace(-3, 3, 61)
        energy_axis = False        
        import pylab as p
        legends = []
        if info == 'tc':
            data = 'tc[s, k, 0]'
            title = 'trasmission coefficeints'
            xlabel = 'Energy(eV)'
            ylabel = 'T'
            dim = 3
            energy_axis = True
        elif info == 'dos':
            data = 'dos[s, k]'
            title = 'density of states'
            xlabel = 'Energy(eV)'
            ylabel = 'DOS(Electron / eV)'
            dim = 2
            energy_axis = True
        elif info == 'den':
            data = "nt[s]"
            title = 'density'
            xlabel = 'Transport axis'
            ylabel = 'Electron'
            energy_axis = False
        elif info == 'ham':
            data = "vt[s]"
            title = 'hamiltonian'
            xlabel = 'Transport axis'
            ylabel = 'Energy(eV)'
            energy_axis = False
        else:
            raise ValueError('no this info type---' + info)        

        eye = np.zeros([10, 1]) + 1
        
        for i, step in enumerate(self.bias_steps):
            if i in steps_indices:
                if not all:
                    ydata = eval('step.' + data)
                else:
                    ydata = eval('step.' + info)
                    npk = ydata.shape[1]
                    for j in range(dim):
                        ydata = np.sum(ydata, axis=0)
                    if info == 'dos':
                        ydata /= npk
                if unit != None:
                    ydata = sum_by_unit(ydata, unit)
                f1 = (step.lead_fermis[0] + step.bias[0]) * eye
                f2 = (step.lead_fermis[1] + step.bias[1]) * eye
                a1 = np.max(ydata)
                l1 = np.linspace(0, a1, 10)
                flags = self.flags
                if not energy_axis:
                    xdata = np.arange(len(ydata))
                if dense_level != 0:
                    from scipy import interpolate
                    tck = interpolate.splrep(xdata, ydata, s=0)
                    num = len(xdata)
                    xdata = np.linspace(xdata[0], xdata[-1], num * (
                                                            dense_level + 1))
                    ydata = interpolate.splev(xdata, tck, der=0)                                
                if not energy_axis:
                    if info == 'ham':
                        p.plot(xdata, ydata * Hartree)
                    else:
                        p.plot(xdata, ydata)
                else:
                    p.plot(xdata, ydata, flags[0], f1, l1, flags[1],
                                                             f2, l1, flags[1])
                legends.append('step' + str(step.bias_step))
        self.set_options(xlabel, ylabel, legends, title)
        if height != None:
            p.axis([xdata[0], xdata[-1], 0, height])
        if show:
            self.show(p)
        return xdata, ydata            

    def compare_ele_step_info(self, info, steps_indices, s, k, height=None,
                                                                   unit=None):
        xdata = self.energies
        #xdata = np.linspace(-3, 3, 61)
        energy_axis = False        
        import pylab as p
        legends = []
        if info == 'dd':
            data = 'dd[s, k]'
            title = 'density matrix diagonal elements'
        elif info == 'df':
            data = 'df[s, k]'
            title = 'hamiltonian matrix diagonal elements'
        elif info == 'den':
            data = 'nt[s]'
            title = 'density'
        elif info == 'ham':
            data = 'vt[s]'
            title = 'hamiltonian'
        elif info == 'rho':
            data = 'rho'
            title = 'total poisson density'
        elif info == 'vHt':
            data = 'vHt'
            title = 'total Hartree potential'             
        elif info == 'tc':
            data = 'tc[s, k, 0]'
            title = 'trasmission coefficeints'
            energy_axis = True
        elif info == 'dos':
            data = 'dos[s, k]'
            title = 'density of states'
            energy_axis = True
        else:
            raise ValueError('no this info type---' + info)        

        for i, step in enumerate(self.ele_steps):
            if i == steps_indices[0]:
                ydata0 = eval('step.' + data)
                if unit != None:
                    ydata0 = sum_by_unit(ydata0, unit)
            elif i == steps_indices[1]:   
                ydata1 = eval('step.' + data)
                if unit != None:
                    ydata1 = sum_by_unit(ydata1, unit)                
        if not energy_axis:
            p.plot(ydata1 - ydata0)
        else:
            p.plot(xdata, ydata1 - ydata0)
        
        legends.append('step' + str(steps_indices[1]) +
                       'minus step' + str(steps_indices[0]))
        self.set_options(None, None, legends, title)
        if height != None:
            p.axis([xdata[0], xdata[-1], 0, height])
        self.show(p)

    def compare_bias_step_info(self, info, steps_indices, s, k,
                               height=None, unit=None):
        xdata = self.energies
        #xdata = np.linspace(-3, 3, 61)
        energy_axis = False        
        import pylab as p
        legends = []
        if info == 'dd':
            data = 'dd[s, k]'
            title = 'density matrix diagonal elements'
            xlabel = 'Basis Sequence'
            ylabel = 'Number'
            energy_axis = False      
        elif info == 'df':
            data = 'df[s, k]'
            title = 'hamiltonian matrix diagonal elements'
            xlabel = 'Basis Sequence'
            ylabel = 'Number'
            energy_axis = False           
        elif info == 'den':
            data = 'nt[s]'
            title = 'density'
            xlabel = 'Transport Axis'
            ylabel = 'Electron'
            energy_axis = False            
        elif info == 'ham':
            data = 'vt[s]'
            title = 'hamiltonian'
            xlabel = 'Transport Axis'
            ylabel = 'Energy(eV)'
            energy_axis = False             
        elif info == 'rho':
            data = 'rho'
            title = 'total poisson density'
            xlabel = 'Transport Axis'
            ylabel = 'Electron'
            energy_axis = False             
        elif info == 'vHt':
            data = 'vHt'
            title = 'total Hartree potential'
            xlabel = 'Transport Axis'
            ylabel = 'Energy(eV)'
            energy_axis = False             
        elif info == 'tc':
            data = 'tc[s, k, 0]'
            title = 'trasmission coefficeints'
            xlabel = 'Energy(eV)'
            ylabel = 'T'            
            energy_axis = True
        elif info == 'dos':
            data = 'dos[s, k]'
            title = 'density of states'
            xlabel = 'Energy(eV)'
            ylabel = 'DOS(Electron/eV)'              
            energy_axis = True
        else:
            raise ValueError('no this info type---' + info)        

        for i, step in enumerate(self.bias_steps):
            if i == steps_indices[0]:
                ydata0 = eval('step.ele_steps[-1].' + data)
                if unit != None:
                    ydata0 = sum_by_unit(ydata0, unit)
            elif i == steps_indices[1]:   
                ydata1 = eval('step.ele_steps[-1].' + data)
                if unit != None:
                    ydata1 = sum_by_unit(ydata1, unit)                
        if not energy_axis:
            p.plot(ydata1 - ydata0)
        else:
            p.plot(xdata, ydata1 - ydata0)
        
        legends.append('step' + str(steps_indices[1]) +
                       'minus step' + str(steps_indices[0]))
        self.set_options(xlabel, ylabel, legends, title)
        if height != None:
            p.axis([xdata[0], xdata[-1], 0, height])
        self.show(p)

    def show_bias_step_info(self, info, steps_indices, s, dense_level=0,
                                                                  shrink=1.0):
        import pylab as p
        if info[:2] == 'nt':
            title = 'density overview in axis ' + info[-1]
        elif info[:2] == 'vt':
            title = 'hamiltonian overview in axis ' + info[-1]
        data = 's' + str(s) + info
        for i, step in enumerate(self.bias_steps):
            if i in steps_indices:
                zdata = eval('step.' + info)[s]
                nx, ny = zdata.shape[:2]
                xdata, ydata = np.mgrid[0:nx:nx*1j,0:ny:ny*1j]
                if dense_level != 0:
                    #from scipy import interpolate
                    #tck = interpolate.bisplrep(xdata, ydata, zdata, s=0)
                    dl = dense_level + 1
                   #xdata, ydata = np.mgrid[0:nx:nx*dl*1j, 0:ny:ny*dl*1j]
                    #zdata = interpolate.bisplev(xdata[:,0],ydata[0,:],tck)
                    from gpaw.transport.tools import interpolate_2d
                    for i in range(dl):
                        zdata = interpolate_2d(zdata)
                p.matshow(zdata)
                #p.pcolor(xdata, ydata, zdata)
                self.set_options(None, None, None, title)
                p.colorbar(shrink=shrink)
                self.show(p)

    def plot_current(self, au=True, spinpol=False, dense_level=0, symm=False):
        bias = []
        current = []
        
        for step in self.bias_steps:
            bias.append(step.bias[0] - step.bias[1])
            current.append(-np.real(step.current))
        import pylab as p
        unit = 6.624 * 1e3 
        current = np.array(current) / (Hartree * 2 * np.pi)
        current = current.reshape(-1)
        if not spinpol:
            current *= 2
        ylabel = 'Current(au.)'
        if not au:
            current *= unit
            ylabel = 'Current($\mu$A)'
        bias = np.array(bias)    
        p.plot(bias, current, self.flags[0])
        if symm:
            p.plot(-bias, -current, self.flags[0])            
        
        if dense_level != 0:
            from scipy import interpolate
            tck = interpolate.splrep(bias, current, s=0)
            numb = len(bias)
            newbias = np.linspace(bias[0], bias[-1], numb * (dense_level + 1))
            newcurrent = interpolate.splev(newbias, tck, der=0)
            p.plot(newbias, newcurrent, self.flags[0])
        self.set_options('Bias(V)', ylabel)
        self.show(p)

    def plot_didv(self, au=True, spinpol=False, dense_level=0):
        bias = []
        current = []
        
        for step in self.bias_steps:
            bias.append(step.bias[0] - step.bias[1])
            current.append(np.real(step.current))
        import pylab as p
        unit = 6.624 * 1e3 
        current = np.array(current) / (Hartree * 2 * np.pi)
        current = current.reshape(-1)
        if not spinpol:
            current *= 2
        from scipy import interpolate
        tck = interpolate.splrep(bias, current, s=0)
        numb = len(bias)
        newbias = np.linspace(bias[0], bias[-1], numb * (dense_level + 1))
        newcurrent = interpolate.splev(newbias, tck, der=0)
        if not au:
            newcurrent *= unit
            ylabel = 'dI/dV($\mu$A/V)'
        else:
            newcurrent *= Hartree
            ylabel = 'dI/dV(au.)'            

        p.plot(newbias[:-1], np.diff(newcurrent), self.flags[0])
        self.set_options('Bias(V)', ylabel)
        self.show(p)

    def plot_tvs_curve(self, spinpol=False, dense_level=0):
        bias = []
        current = []
        for step in self.bias_steps:
            bias.append(step.bias[0] - step.bias[1])
            current.append(np.real(step.current))
        import pylab as p
        unit = 6.624 * 1e-3
        current = np.array(current) / (Hartree * 2 * np.pi)
        current = current.reshape(-1)
        if not spinpol:
            current *= 2
        from scipy import interpolate
        bias = abs(np.array(bias))
        #tck = interpolate.splrep(bias, current, s=0)
        #numb = len(bias)
        #newbias = np.linspace(bias[0], bias[-1], numb * (dense_level + 1))
        #newcurrent = interpolate.splev(newbias, tck, der=0)
        #newcurrent *= unit
        current *= unit
        ylabel = '$ln(I/V^2)$'
        ydata = np.log(abs(current) / (bias * bias))
        xdata = 1 / bias
        p.plot(xdata, ydata, self.flags[0])
        self.set_options('1/V', ylabel)
        self.show(p)      
   
   
    #def tc(self, bs, s=None, k=None):
        #transmission = self.bias_steps[bs].tc
        
        #for name in transmission:
   
    def plot_zero_bias_tvs(self, energies, transmission, fermi, kt=0.1, direction=-1):
        N = 121
        bias = np.linspace(0, 3., N)
        current = np.zeros([N])
        fd = fermidistribution
        unit = 6.624 * 1e-3        
        for i in range(N):
            fermi0 = fermi
            fermi1 = fermi
            if direction == 1:
                fermi0 += bias[i]
            elif direction == -1:
                fermi0 -= bias[i]
            elif direction == 0:
                fermi0 += bias[i] / 2
                fermi1 -= bias[i] / 2
                
            cc = 0    
            for energy, tc in zip(energies, transmission):
                ff = fd(energy - fermi0, kt) - fd(energy - fermi1, kt)
                cc += ff * tc
            current[i] = cc
        current *= energies[1] - energies[0]
        current *= 2 * unit /(Hartree * np.pi)
        import pylab as p        
        ylabel = '$ln(I/V^2)$'
        ydata = np.log(abs(current) / (bias * bias))
        xdata = 1 / bias
        p.plot(xdata, ydata, 'b--o')
        self.set_options('1/V', ylabel)
        self.show(p)              
         
    def compare_ele_step_info2(self, info, steps_indices, s, dense_level=0):
        import pylab as p
        if info[:2] == 'nt':
            title = 'density difference overview in axis ' + info[-1]
        elif info[:2] == 'vt':
            title = 'hamiltonian difference overview in axis ' + info[-1]
        assert steps_indices[0] < len(self.ele_steps)
        assert steps_indices[1] < len(self.ele_steps)
        step0 = self.ele_steps[steps_indices[0]]
        step1 = self.ele_steps[steps_indices[1]]
        data0 = 's' + str(s[0]) + info
        data1 = 's' + str(s[1]) + info        
        ydata = eval('step0.' + info)[s[0]] - eval('step1.' + info)[s[1]]
        p.matshow(ydata)
        self.set_options(None, None, None, title)
        self.show(p)
        p.colorbar()
        #p.legend([str(steps_indices[0]) + '-' + str(steps_indices[0])])
        self.show(p)
        
    def compare_bias_step_info2(self, info, steps_indices, s, shrink=1.0,
                                                              dense_level=0):
        import pylab as p
        if info[:2] == 'nt':
            title = 'density difference overview in axis ' + info[-1]
        elif info[:2] == 'vt':
            title = 'hamiltonian difference overview in axis ' + info[-1]
        assert steps_indices[0] < len(self.bias_steps)
        assert steps_indices[1] < len(self.bias_steps)
        step0 = self.bias_steps[steps_indices[0]]
        step1 = self.bias_steps[steps_indices[1]]
        data0 = 's' + str(s[0]) + info
        data1 = 's' + str(s[1]) + info        
        ydata = eval('step0.' + info)[s[0]] - eval('step1.' + info)[s[1]]
        if dense_level != 0:
            dl = dense_level + 1
            from gpaw.transport.tools import interpolate_2d
            for i in range(dl):
                ydata = interpolate_2d(ydata) 
        p.matshow(ydata)
        self.set_options(None, None, None, title)
        p.colorbar(shrink=shrink)
        #p.legend([str(steps_indices[0]) + '-' + str(steps_indices[0])])
        self.show(p)        
             
    def set_cmp_step0(self, ele_step, bias_step=None):
        if bias_step == None:
            self.cmp_step0 = self.ele_steps[ele_step]
        else:
            self.cmp_step0 = self.bias_steps[bias_step].ele_steps[ele_step]
                                  
    def cmp_steps(self, info, ele_step, bias_step=None):
        if bias_step == None:
            step = self.ele_steps[ele_step]
        else:
            step = self.bias_steps[bias_step].ele_steps[ele_step]            
        #xdata = np.linspace(-3, 3, 61)
        xdata = self.energies
        energy_axis = False        
        import pylab as p
        if info == 'dd':
            data = 'dd[s, k]'
            title = 'density matrix diagonal elements'
        elif info == 'df':
            data = 'df[s, k]'
            title = 'hamiltonian matrix diagonal elements'
        elif info == 'den':
            data = 'nt'
            title = 'density'
        elif info == 'ham':
            data = 'vt'
            title = 'hamiltonian'
        elif info == 'rho':
            data = 'rho'
            title = 'total poisson density'
        elif info == 'vHt':
            data = 'vHt'
            title = 'total Hartree potential'             
        elif info == 'tc':
            data = 'tc[s, k, 0]'
            title = 'trasmission coefficeints'
            energy_axis = True
        elif info == 'dos':
            data = 'dos[s, k]'
            title = 'density of states'
            energy_axis = True
        else:
            raise ValueError('no this info type---' + info)        

        ydata0 = eval('self.cmp_step0.' + data)
        ydata1 = eval('step.' + data)
           
        if not energy_axis:
            p.plot(ydata1 - ydata0)
        else:
            p.plot(xdata, ydata1 - ydata0)

        self.set_options(None, None, None, title)
        self.show(p)
 
    def plot_force(self, atom_indices, direction=2, bias_indices=None):
        if bias_indices == None:
            bias_indices = range(len(self.bias_steps))
        import pylab as p
        legends = []
        for i in atom_indices:
            forces = []
            bias = []
            legends.append('force of Atom' + str(i))
            for j in bias_indices:
                bias.append(self.bias_steps[j].bias[0] -
                                         self.bias_steps[j].bias[1])
                forces.append(self.bias_steps[j].force[i, direction])
            p.plot(bias, forces)
        self.set_options('Bias(V)', 'Force(au.)', legends)
        self.show(p)
 
    def plot_charge_on_bias(self, atom_indices=None, orbital_type=None,
                                            bias_indices=None, spin_type=None):
        if bias_indices == None:
            bias_indices = range(len(self.bias_steps))           
        if atom_indices == None:
            atom_indices = range(len(self.atoms))
        if orbital_type == None:
            orbital_type = 'all'
        import pylab as p
        charge = []
        bias = []
        orbital_indices = self.basis['orbital_indices']
        orbital_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
        for i in bias_indices:
            bias.append(self.bias_steps[i].bias[0] -
                                              self.bias_steps[i].bias[1])
            cc = 0
            for j in atom_indices:
                atom_index = orbital_indices[:, 0] - j == 0
                if orbital_type == 'all':
                    orbital_index = np.zeros([orbital_indices.shape[0]]) + 1
                else:
                    orbital_index = orbital_indices[:, 1] - orbital_map[
                                                     orbital_type] ==  0 
                tmp = np.sum(self.bias_steps[i].charge, axis =1) 
                ind = orbital_indices.shape[0]
                tmp = tmp[:, :ind]
                if spin_type == None:
                    tmp = np.sum(tmp, axis=0)
                elif spin_type == 'up':
                    tmp = tmp[0]
                else:
                    tmp = tmp[1]
                cc += np.sum(tmp * atom_index * orbital_index)
            charge.append(cc)
        p.plot(bias, charge)
        self.set_options('Bias(V)', 'Charge(au.)')
        self.show(p)
        return bias, charge

    def plot_charge_on_atoms(self, bias_step, atom_indices=None, orbital_type=None,
                                                            spin_type=None):
        if atom_indices == None:
            atom_indices = range(len(self.atoms))        
        if orbital_type == None:
            orbital_type = 'all'
        import pylab as p
        charge = []
        orbital_indices = self.basis['orbital_indices']
        orbital_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
        atom_symbols = []
        from ase.data import chemical_symbols
        for i, atom in zip(atom_indices, self.atoms[atom_indices]):
            atom_index = orbital_indices[:, 0] - i == 0
            if orbital_type == 'all':
                orbital_index = np.zeros([orbital_indices.shape[0]]) + 1
            else:
                orbital_index = orbital_indices[:, 1] - orbital_map[
                                                       orbital_type] ==  0             
            tmp = np.sum(self.bias_steps[bias_step].charge, axis =1) 
            ind = orbital_indices.shape[0]
            tmp = tmp[:, :ind]
            if spin_type == None:
                tmp = np.sum(tmp, axis=0)
            elif spin_type == 'up':
                tmp = tmp[0]
            else:
                tmp = tmp[1]
            charge.append(np.sum(tmp * atom_index * orbital_index))
            atom_symbols.append(chemical_symbols[atom.number])
        from matplotlib.ticker import MultipleLocator, FixedFormatter
        ax = p.subplot(111)
        p.plot(charge, 'b--o')
        p.xlabel('Project Atom')
        p.ylabel('Charge(au.)')
        majorLocator = MultipleLocator(1)
        majorFormatter = FixedFormatter(atom_symbols)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_major_formatter(majorFormatter)
        p.show()
        
    def show_force(self, bias_step, file=None):      
        import vtk
        from ase.visualize.vtk.atoms import vtkAtoms
        usewx = False
        try:
            import wx
            usewx = True
        except ImportError:
            pass
        if usewx:
            from vtk.wx.wxVTKRenderWindow import wxVTKRenderWindow
            app = wx.PySimpleApp()
            frame = wx.Frame(None, -1, 'wxVTKRenderWindow', size=(800,600))
            widget = wxVTKRenderWindow(frame, -1)
            win = widget.GetRenderWindow()
            ren = vtk.vtkRenderer()
            win.AddRenderer(ren)
        else:
            ren = vtk.vtkRenderer()
            win = vtk.vtkRenderWindow()
            win.AddRenderer(ren)
            win.SetSize(800,600)
            iren = vtk.vtkRenderWindowInteractor()
            iren.SetRenderWindow(win)
            style = vtk.vtkInteractorStyleTrackballCamera()
            iren.SetInteractorStyle(style)
        atoms = self.atoms.copy()
        calc = GPAW()
        atoms.set_calculator(calc)
        calc.initialize(atoms)
        calc.scf.converged = True
        calc.forces.F_av = self.bias_steps[bias_step].force

        va = vtkAtoms(atoms)
        va.add_cell()
        va.add_axes()
        va.add_forces()
        va.add_actors_to_renderer(ren)
        if usewx:
            frame.Show()
            app.MainLoop()
        else:
            iren.Initialize()
            win.OffScreenRenderingOff()
            win.Render()
            iren.Start()
        w2i=vtk.vtkWindowToImageFilter()
        w2i.SetInput(win)
        if file is not None:
            pw = vtk.vtkPNGWriter()
            pw.SetFileName(file)
            pw.SetInputConnection(w2i.GetOutputPort())
            pw.Write()

           

