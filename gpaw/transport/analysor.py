from gpaw.transport.selfenergy import LeadSelfEnergy, CellSelfEnergy
from gpaw.transport.greenfunction import GreenFunction
from gpaw.transport.tools import get_matrix_index, aa1d, aa2d, sum_by_unit, dot
import numpy as np
import copy
import pickle

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
                                   tc, dos, dv, current, lead_fermis, time_cost):
        self.bias = bias
        self.gate = gate
        self.ep = ep
        self.lead_pairs = lead_pairs
        self.tc = tc
        self.dos = dos
        self.dv = dv
        self.current = current
        self.lead_fermis = lead_fermis
        self.time_cost = time_cost

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
    
    # ------------------------------(optional)
    # ent(the nt variable in extended region), evt
    # ehot_g(the rho density in a line along transport direction)
    # evHt_g(the Hartree potential)
   
    def __init__(self, ion_step, bias_step, ele_step):
        self.ion_step, self.bias_step, self.ele_step = ion_step, bias_step, ele_step
        
    def initialize_data(self, bias, gate, dd, df, nt, vt, rho, vHt, D_asp, dH_asp, tc, dos, time_cost):
        self.bias = bias
        self.gate = gate
        self.dd = dd
        self.df = df
        self.nt = nt
        self.vt = vt
        self.rho = rho
        self.vHt = vHt
        self.D_asp = D_asp
        self.dH_asp = dH_asp
        self.tc = tc
        self.dos = dos
        self.time_cost = time_cost
    
    def initialize_extended_data(self, ent_G, evt_G, ent_g,
                                 evt_g, ehot_g, evHt_g, edmm):
        #capital 'e' represents 'extended region'
        self.ent_G = ent_G
        self.evt_G = evt_G
        self.ent_g = ent_g
        self.evt_g = evt_g
        self.ehot_g = ehot_g
        self.evHt_g = evHt_g
        self.edmm = edmm
    
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
        self.reset = False
        self.set_plot_option()
        self.initialize()

    def set_plot_option(self):
        tp = self.tp
        if tp.plot_option == None:
            ef = tp.lead_fermi[0]
            self.energies = np.linspace(ef - 100, ef + 100, 200) + 1e-4 * 1.j
            self.lead_pairs = [[0,1]]
        else:
            self.energies = tp.plot_option['energies']
            self.lead_pairs = tp.plot_option['lead_paris']
       
    def initialize(self):
        if self.restart:
            self.initialize_selfenergy_and_green_function()
        else:
            self.selfenergies = self.tp.selfenergies
            if self.tp.matrix_mode == 'full':
                self.greenfunction = self.tp.greenfunction
                
    def initialize_selfenergy_and_green_function(self):
        self.selfenergies = []
        tp = self.tp
        if tp.matrix_mode == 'full':
            if tp.use_lead:
                for i in range(tp.lead_num):
                    self.selfenergies.append(LeadSelfEnergy((tp.hl_spkmm[i][0,0],
                                                         tp.sl_pkmm[i][0]), 
                                                        (tp.hl_spkcmm[i][0,0],
                                                         tp.sl_pkcmm[i][0]),
                                                        (tp.hl_spkcmm[i][0,0],
                                                         tp.sl_pkcmm[i][0]),
                                                 1e-8))
    
                    self.selfenergies[i].set_bias(tp.bias[i])

            if tp.use_env:
                self.env_selfenergies = []
                for i in range(tp.env_num):
                    self.env_selfenergies.append(CellSelfEnergy((tp.he_skmm[i],
                                                             tp.se_kmm[i]),
                                                            (tp.he_smm[i],
                                                             tp.se_mm[i]),
                                                             tp.env_ibzk_kc[i],
                                                             tp.env_weight[i],
                                                            1e-8))
            self.greenfunction = GreenFunction(selfenergies=self.selfenergies,
                                               H=tp.h_spkmm[0,0],
                                               S=tp.s_pkmm[0], eta=0)
        else:  #sparse_matrix
            raise NotImplementError
            
    def reset_selfenergy_and_green_function(self, s, k):
        tp = self.tp
        if tp.matrix_mode == 'full':
            if tp.use_lead:    
                sg = self.selfenergies
                for i in range(tp.lead_num):
                    sg[i].h_ii = tp.hl_spkmm[i][s, k]
                    sg[i].s_ii = tp.sl_pkmm[i][k]
                    sg[i].h_ij = tp.hl_spkcmm[i][s, k]
                    sg[i].s_ij = tp.sl_pkcmm[i][k]
                    sg[i].h_im = tp.hl_spkcmm[i][s, k]
                    sg[i].s_im = tp.sl_pkcmm[i][k]         

            ind = get_matrix_index(tp.inner_mol_index)
            self.greenfunction.H = tp.h_spkmm[s, k, ind.T, ind]
            self.greenfunction.S = tp.s_pkmm[k, ind.T, ind]
        else:
            for i in range(tp.lead_num):
                self.selfenergies[i].s = s
                self.selfenergies[i].pk = k
            tp.hsd.s = s
            tp.hsd.pk = k
      
    def calculate_green_function_of_k_point(self, s, k, energy, re_flag=0):
        tp = self.tp 
        if tp.matrix_mode == 'full':
            nbmol = tp.nbmol_inner
            sigma_list = []
            sigma = np.zeros([nbmol, nbmol], complex)
            for i in range(tp.lead_num):
                ind = get_matrix_index(tp.inner_lead_index[i])
                sigma_list.append(self.selfenergies[i](energy))
                sigma[ind.T, ind] += sigma_list[i]
            if re_flag == 0:
                return self.greenfunction.calculate(energy, sigma)
            else:
                return self.greenfunction.calculate(energy, sigma), sigma_list
        else:
            sigma = []
            for i in range(tp.lead_num):
                sigma.append(self.selfenergies[i](energy))
            if re_flag==0:
                return tp.hsd.calculate_eq_green_function(energy, sigma, False)
            else:
                return tp.hsd.calculate_eq_green_function(energy, sigma,
                                                          False), sigma 
    
    def calculate_transmission_and_dos(self, s, k, energies):
        self.reset_selfenergy_and_green_function(s, k)
        transmission_list = []
        dos_list = []
        for energy in energies:
            gr, sigma = self.calculate_green_function_of_k_point(s, k, energy, 1)
            trans_coff = []
            gamma = []
            for i in range(self.tp.lead_num):
                if self.tp.matrix_mode == 'sparse':
                    gamma.append(1.j * (sigma[i].recover() -
                                                   sigma[i].recover().T.conj()))
                else:
                    gamma.append(1.j * (sigma[i] - sigma[i].T.conj()))            
            
            for i, lead_pair in enumerate(self.lead_pairs):
                l1, l2 = lead_pair
                if self.tp.matrix_mode == 'full':
                    ind1 = get_matrix_index(self.tp.lead_index[l1])
                    ind2 = get_matrix_index(self.tp.lead_index[l2])
                
                    gr_sub = gr[ind1.T, ind2]
                elif i == 0:
                    gr_sub, inv_mat = self.tp.hsd.abstract_sub_green_matrix(
                                                          energy, sigma, l1, l2)
                else:
                    gr_sub = self.tp.hsd.abstract_sub_green_matrix(energy,
                                                         sigma, l1, l2, inv_mat)                    
                transmission =  dot(dot(gamma[l1], gr_sub),
                                                dot(gamma[l2], gr_sub.T.conj()))
       
                trans_coff.append(np.trace(transmission))
            transmission_list.append(trans_coff)
            del trans_coff
            
            if self.tp.matrix_mode == 'full':
                dos = - np.imag(np.trace(dot(gr,
                                           self.greenfunction.S))) / np.pi
            else:
                dos = - np.imag(np.trace(dot(gr,
                                         self.tp.hsd.S[k].recover()))) / np.pi                
            dos_list.append(dos)
        
        ne = len(energies)
        npl = len(self.lead_pairs)
        transmission_list = np.array(transmission_list)
        transmission_list = np.resize(transmission_list.T, [npl, ne])
        return transmission_list, np.array(dos_list)

    def save_ele_step(self):
        tp = self.tp
        step = Electron_Step_Info(self.n_ion_step, self.n_bias_step, self.n_ele_step)
        dtype = tp.wfs.dtype
        nbmol = tp.wfs.setups.nao
        dd = np.empty([tp.my_nspins, tp.my_npk, nbmol], dtype)
        df = np.empty([tp.my_nspins, tp.my_npk, nbmol], dtype)
        
        total_dd = np.empty([tp.nspins, tp.npk, nbmol], dtype)
        total_df = np.empty([tp.nspins, tp.npk, nbmol], dtype)
        
        for s in range(tp.my_nspins):
            for k in range(tp.my_npk):
                if tp.matrix_mode == 'full':
                    dd[s, k] = np.diag(tp.d_spkmm[s, k])
                    df[s, k] = np.diag(tp.h_spkmm[s, k])
                else:
                    dd[s, k] = np.diag(tp.hsd.D[s][k].recover(True))
                    df[s, k] = np.diag(tp.hsd.H[s][k].recover(True))
        
        tp.wfs.kpt_comm.all_gather(dd, total_dd)
        tp.wfs.kpt_comm.all_gather(df, total_df)

        dim = tp.gd.N_c
        d1 = dim[0] // 2
        d2 = dim[1] // 2
        
        assert tp.d == 2
        
        gd = tp.gd
        nt_sG = gd.empty(tp.nspins, global_array=True)
        vt_sG = gd.empty(tp.nspins, global_array=True)
        
        nt_sG = gd.collect(tp.density.nt_sG - tp.density.nct_G, True)
        vt_sG = gd.collect(tp.hamiltonian.vt_sG, True)

        nt = nt_sG[0, d1, d2].copy()
        vt = vt_sG[0, d1, d2].copy()

        gd = tp.finegd
        rhot_g = gd.empty(tp.nspins, global_array=True)
        rhot_g = gd.collect(tp.density.rhot_g, True)
        rho = rhot_g[d1*2, d2*2].copy()
        
        vHt_g = gd.collect(tp.hamiltonian.vHt_g, True)
        vHt = vHt_g[d1*2, d2*2].copy()
        
        D_asp = copy.deepcopy(tp.density.D_asp)
        dH_asp = copy.deepcopy(tp.hamiltonian.dH_asp)
        
        tc_array, dos_array = self.collect_transmission_and_dos()
        time_cost = self.ele_step_time_collect()
        step.initialize_data(tp.bias, tp.gate, total_dd, total_df, nt, vt, rho, vHt, D_asp, dH_asp, tc_array, dos_array, time_cost)
        
        self.ele_steps.append(step)
        self.n_ele_step += 1
      
    def save_bias_step(self):
        tp = self.tp
        step = Transmission_Info(self.n_ion_step, self.n_bias_step)
        time_cost = self.bias_step_time_collect()
        tc_array, dos_array = self.collect_transmission_and_dos()
        dv = self.abstract_d_and_v()
        current = self.calculate_current()
        step.initialize_data(tp.bias, tp.gate, self.energies, self.lead_pairs,
                             tc_array, dos_array, dv, current, tp.lead_fermi, time_cost)
        step.ele_steps = self.ele_steps
        del self.ele_steps
        self.ele_steps = []
        self.n_ele_step = 0
        self.bias_steps.append(step)
        self.n_bias_step += 1

    def bias_step_time_collect(self):
        time = self.tp.timer.gettime
        cost = {}
        cost['update lead hamiltonian'] = time('update lead hamiltonian0') * self.tp.lead_num
        cost['init lead'] = time('init lead0') * self.tp.lead_num
        cost['scat guess'] = time('scat guess')
        cost['init scat'] = time('init scat')
        cost['init surround'] = time('init surround')
        cost['init scf'] = time('init scf')
        return cost
        
    def ele_step_time_collect(self):    
        time = self.tp.timer.gettime
        cost = {}
        cost['eq fock2den'] = time('eq fock2den')
        cost['ne fock2den'] = time('ne fock2den')
        cost['Poisson'] = time('Poisson')
        return cost

    def collect_transmission_and_dos(self, energies=None):
        if energies == None:
            energies = self.energies
        tp = self.tp
      
        nlp = len(self.lead_pairs)
        ne = len(energies)
        ns, npk = tp.nspins, tp.npk

        tc_array = np.empty([ns, npk, nlp, ne])
        dos_array = np.empty([ns, npk, ne])

        ns, npk = tp.my_nspins, tp.my_npk
        local_tc_array = np.empty([ns, npk, nlp, ne])
        local_dos_array = np.empty([ns, npk, ne])
        
        for s in range(ns):
            for q in range(npk):
                local_tc_array[s, q], local_dos_array[s, q] = \
                          self.calculate_transmission_and_dos(s, q, energies)

        kpt_comm = tp.wfs.kpt_comm
        kpt_comm.all_gather(local_tc_array, tc_array)
        kpt_comm.all_gather(local_dos_array, dos_array)            

        return tc_array, dos_array

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
 
    def save_data_to_file(self, flag='bias'):
        if flag == 'ion':
            steps = self.ion_steps
        elif flag == 'bias':
            steps = self.bias_steps
        else:
            steps = self.ele_steps
        fd = file('analysis_data_' + flag, 'wb')
        pickle.dump(steps, fd, 2)
        fd.close()
   
    def calculate_t_and_dos(self, E_range=[-6,2],
                            point_num = 60, leads=[[0,1]]):
        data = {}
        e_points = np.linspace(E_range[0], E_range[1], point_num)
        tcalc = self.set_calculator(e_points, leads)
        tcalc.get_transmission()
        tcalc.get_dos()
        f1 = self.intctrl.leadfermi[leads[0]] * (np.zeros([10, 1]) + 1)
        f2 = self.intctrl.leadfermi[leads[1]] * (np.zeros([10, 1]) + 1)
        a1 = np.max(tcalc.T_e)
        a2 = np.max(tcalc.dos_e)
        l1 = np.linspace(0, a1, 10)
        l2 = np.linspace(0, a2, 10)
        data['e_points'] = e_points
        data['T_e'] = tcalc.T_e
        data['dos_e'] = tcalc.dos_e
        data['f1'] = f1
        data['f2'] = f2
        data['l1'] = l1
        data['l2'] = l2
        return data
  
    def abstract_d_and_v(self):
        data = {}
        tp = self.tp
        nt = tp.gd.empty()
        vt = tp.gd.empty()
        for s in range(tp.nspins):
            nt = tp.gd.collect(tp.density.nt_sG[s], True)
            vt = tp.gd.collect(tp.hamiltonian.vt_sG[s], True)
            for name, d in [('x', 0), ('y', 1), ('z', 2)]:
                data['s' + str(s) + 'nt_1d_' + name] = aa1d(nt, d)
                data['s' + str(s) + 'nt_2d_' + name] = aa2d(nt, d)            
                data['s' + str(s) + 'vt_1d_' + name] = aa1d(vt, d)
                data['s' + str(s) + 'vt_2d_' + name] = aa2d(vt, d)
        return  data
   
    def calculate_current(self):
        # temperary, because different pk point may have different ep,
        # and also for multi-terminal, energies is wrong
        tp = self.tp
        assert hasattr(tp, 'nepathinfo')
        ep = tp.nepathinfo[0][0].energy
        weight = tp.nepathinfo[0][0].weight
        fermi_factor = tp.nepathinfo[0][0].fermi_factor
        
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

    def plot_transmission_and_dos(self, ni, nb, s, k, leads=[0,1]):
        l0, l1 = leads
        ep = self.energies
        bs = self.ion_steps[ni].bias_steps[nb]
        
        tc = bs.tc_array[s, k]
        dos = bs.dos_array[s, k]
        
        eye = np.zeros([10, 1]) + 1
        f1 = bs.lead_fermis[l0] * eye
        f2 = bs.lead_fermis[l1] * eye
        
        a1 = np.max(tc)
        a2 = np.max(dos)
        
        l1 = np.linspace(0, a1, 10)
        l2 = np.linspace(0, a2, 10)
        
        import pylab as p
        p.figure(1)
        p.subplot(211)
        p.plot(ep, tc, 'b-o', f1, l1, 'r--', f2, l1, 'r--')
        p.ylabel('Transmission Coefficients')
        p.subplot(212)
        p.plot(ep, dos, 'b-o', f1, l2, 'r--', f2, l2, 'r--')
        p.ylabel('Density of States')
        p.xlabel('Energy (eV)')
        p.savefig('tdos_i_' + str(ni) + '_b_' +  str(nb) + '_s_' + str(s)
                + '_k_' + str(k) + '_lp_' + str(l0) + '-' + str(l1) + '.png')
        if self.plot_show:
            p.show()
        
    def plot_dv(self, ni, nb, s, option, plot_d=2, overview_d=1):
        bs = self.ion_steps[ni].bias_steps[nb]
        dv = bs['dv']
    
        import pylab as p
        dd = ['x', 'y', 'z']        
        data_1d = dv['s' + str(s) + option + 't_1d_' + dd[plot_d]]
        data_2d = dv['s' + str(s) + option + 't_2d_' + dd[overview_d]]
        title1 = 's' + str(s) + 'bias=' + str(dv.bias) 
        title2 = 'overview=' + str(overview_d)

        if option == 'v':
            data_1d *= Hartree
            data_2d *= Hartree
            label = 'potential(eV)'
        else:
            label = 'density'
       
        p.figure(1)
        p.subplot(211)
        p.plot(data_1d, 'b--o')
        p.title(title1)
        p.subplot(212)
        p.matshow(data_2d)
        p.title(title2)
        p.ylabel(label)
        p.colorbar()
        p.savefig(option + '_' + str(ni) + '_b_' +  str(nb) +
                                                      '_s_' + str(s) + 'png')
        if self.plot_show:
            p.show()

    def compare_dv(self, ni, nb, s, option, cmp_option,
                                               cmp, plot_d=2, overview_d=1):
        bs = self.ion_steps[ni].bias_steps[nb]
        assert cmp_option == 'bias' or 'spin'
        if cmp_option == 'bias':
            bs1 = self.ion_steps[ni].bias_steps[cmp]
            scmp = s    
        else:
            bs1 = bs
            scmp = cmp
            
        import pylab as p
        dd = ['x', 'y', 'z']        
        data_1d = bs['dv']['s' + str(s) + option + 't_1d_' + dd[plot_d]]
        data_2d = bs['dv']['s' + str(s) + option + 't_2d_' + dd[overview_d]]            
        
        data_1d -= bs1['dv']['s' + str(scmp) + option + 't_1d_' + dd[plot_d]]
        data_2d -= bs1['dv']['s' + str(scmp) + option +
                                                    't_2d_' + dd[overview_d]]              
        if cmp_option == 'bias':
            filename = option + '_cmp_' + cmp_option + 'i_' + ni + 'b_' + \
                    str(nb) + '-' + str(cmp) + 's_' + str(s) + '.png'         
            
            title1 = 'compare_via_bias_s' + str(s) + 'bias=' \
                            + str(bs['dv'].bias) + '-' + str(bs1['dv'].bias)
        else:
            filename = option + '_cmp_' + cmp_option + 'i_' + ni + 'b_' + \
                    str(nb) + 's_' + str(s)  + '-' +  str(scmp) + '.png'
     
            title1 = 'compare_via_spin_s' + str(s) + '-' \
                               + str(scmp) + 'bias=' + str(bs['dv'].bias)

        title2 = 'overview=' + str(overview_d)

        if option == 'v':
            data_1d *= Hartree
            data_2d *= Hartree
            label = 'potential(eV)'
        else:
            label = 'density'
        
        p.figure(1)
        p.subplot(211)
        p.plot(data_1d, 'b--o')
        p.title(title1)
        p.subplot(212)
        p.matshow(data_2d)
        p.title(title2)
        p.ylabel(label)
        p.colorbar()
        p.savefig(filename)
        if self.plot_show:
            p.show()
     
    def plot_eigen_channel(self, energy=[0]):
        tcalc = self.set_calculator(energy)
        tcalc.initialize()
        tcalc.update()
        T_MM = tcalc.T_MM[0]
        from gpaw.utilities.lapack import diagonalize
        nmo = T_MM.shape[-1]
        T = np.zeros([nmo])
        info = diagonalize(T_MM, T)
        dmo = np.empty([nmo, nmo, nmo])
        for i in range(nmo):
            dmo[i] = dot(T_MM[i].T.conj(),T_MM[i])
        basis_functions = self.wfs.basis_functions
        for i in range(nmo):
            wt = self.gd.zeros(1)
            basis_functions.construct_density(dmo[i], wt[0], 0)
            import pylab
            wt=np.sum(wt, axis=2) / wt.shape[2] 
            if abs(T[i]) > 0.001:
                pylab.matshow(wt[0])
                pylab.title('T=' + str(T[i]))
                pylab.show()
     
    def calculate_real_dos(self, energy):
        ns = self.my_nspins
        nk = self.my_npk
        nb = self.nbmol_inner
        gr_skmm = np.zeros([ns, nk, nb, nb], complex)
        gr_mm = np.zeros([nb, nb], complex)
        self.initialize_green_function()
        for s in range(ns):
            for k in range(nk):
                gr_mm = self.calculate_green_function_of_k_point(s,
                                                                    k, energy)
                gr_skmm[s, k] =  (gr_mm - gr_mm.conj()) /2.
                print gr_skmm.dtype
                
        self.dos_sg = self.project_from_orbital_to_grid(gr_skmm)

    def plot_real_dos(self, direction=0, mode='average', nl=0):
        import pylab
        dim = self.dos_sg.shape
        print 'diff', np.max(abs(self.dos_sg[0] - self.dos_sg[1]))
        ns, nx, ny, nz = dim
        if mode == 'average':
            for s in range(ns):
                dos_g = np.sum(self.dos_sg[s], axis=direction)
                dos_g /= dim[direction]
                pylab.matshow(dos_g)
                pylab.show()
        elif mode == 'sl': # single layer mode
            for s in range(ns):
                if direction == 0:
                    dos_g = self.dos_sg[s, nl]
                elif direction == 1:
                    dos_g = self.dos_sg[s, :, nl]
                elif direction == 2:
                    dos_g = self.dos_g[s, :, :, nl]
                pylab.matshow(dos_g)
                pylab.show()
                
    def project_from_grid_to_orbital(self, vt_sg):
        wfs = self.wfs
        basis_functions = wfs.basis_functions
        nao = wfs.setups.nao
        if self.gamma:
            dtype = float
        else:
            dtype = complex
        vt_mm = np.empty((nao, nao), dtype)
        ns = self.my_nspins
        nk = len(self.my_kpts)
        vt_SqMM = np.empty((ns, nk, nao, nao), dtype)
        for kpt in wfs.kpt_u:
            basis_functions.calculate_potential_matrix(vt_sg[kpt.s],
                                                             vt_mm, kpt.q)
            vt_SqMM[kpt.s, kpt.q] = vt_mm      
        return 
        
    def project_from_orbital_to_grid(self, d_SqMM):
        wfs = self.wfs
        basis_functions = wfs.basis_functions
        nt_sG = self.gd.zeros(self.my_nspins)        
        for kpt in wfs.kpt_u:
            basis_functions.construct_density(d_SqMM[kpt.s, kpt.q],
                                              nt_sG[kpt.s], kpt.q)
        return nt_sG
    
    def restart_and_abstract_result(self, v_limit=3, num_v=16):
        bias = np.linspace(0, v_limit, num_v)
        current = np.empty([num_v])
        result = {}
        result['N'] = num_v
        for i in range(num_v):
            self.input('bias' + str(i))
            self.hamiltonian.vt_sG += self.get_linear_potential()
            result['step_data' + str(i)] = self.result_for_one_bias_step()            
            current[i] = self.current
        result['i_v'] = (bias, current)    
        if self.master:
            fd = file('result.dat', 'wb')
            pickle.dump(result, fd, 2)
            fd.close()
            
    def analysis(self):
        if self.master:
            fd = file('result.dat', 'r')
            result = pickle.load(fd)
            fd.close()
            num_v = result['N']
            for i in range(num_v):
                step_data = result['step_data' + str(i)]
                self.plot_step_data(step_data)
            self.plot_iv(result['i_v'])    
 
    def analysis_compare(self):
        if self.master:
            fd = file('result.dat', 'r')
            result = pickle.load(fd)
            fd.close()
            num_v = result['N']
            step_data1 = result['step_data' + str(0)]
            step_data2 = result['step_data' + str(1)]
            self.compare_step_data(step_data1, step_data2)
            
    def compare_step_data(self, step_data1, step_data2):
        overview_d = 1
        self.nspins = 1
        sd = step_data1['t_dos']
        bias = step_data1['bias']
        import pylab
        pylab.figure(1)
        pylab.subplot(211)
        pylab.plot(sd['e_points'], sd['T_e'], 'b-o', sd['f1'], sd['l1'],
                                    'r--', sd['f2'], sd['l1'], 'r--')
        pylab.ylabel('Transmission Coefficients')
        pylab.subplot(212)
        pylab.plot(sd['e_points'], sd['dos_e'], 'b-o', sd['f1'], sd['l2'],
                                      'r--', sd['f2'], sd['l2'], 'r--')
        pylab.ylabel('Density of States')
        pylab.xlabel('Energy (eV)')
        pylab.title('bias=' + str(bias))
        pylab.show()
        
        sd = step_data2['t_dos']
        bias = step_data2['bias']
        import pylab
        from mytools import gnu_save
        pylab.figure(1)
        pylab.subplot(211)
        pylab.plot(sd['e_points'], sd['T_e'], 'b-o', sd['f1'], sd['l1'],
                                    'r--', sd['f2'], sd['l1'], 'r--')
        pylab.ylabel('Transmission Coefficients')
        pylab.subplot(212)
        pylab.plot(sd['e_points'], sd['dos_e'], 'b-o', sd['f1'], sd['l2'],
                                      'r--', sd['f2'], sd['l2'], 'r--')
        pylab.ylabel('Density of States')
        pylab.xlabel('Energy (eV)')
        pylab.title('bias=' + str(bias))
        pylab.show()        
        
        sd1 = step_data1['v_d']
        sd2 = step_data2['v_d']
        bias1 = step_data1['bias']
        bias2 = step_data2['bias']
        dd = ['x', 'y', 'z']
        for s in range(self.nspins):
            tmp = sd1['s' + str(s) + 'vt_1d_' + dd[self.d]] - sd2['s' + str(s) + 'vt_1d_' + dd[self.d]]
            pylab.plot(tmp * Hartree, 'b--o') 
            pylab.ylabel('potential(eV)')
            pylab.title('spin' + str(s) + 'bias=' + str(bias1) + '-' + str(bias2))
            pylab.show()

            tmp = sd1['s' + str(s) + 'nt_1d_' + dd[self.d]] - sd2['s' + str(s) + 'nt_1d_' + dd[self.d]]        
            pylab.plot(tmp, 'b--o')
            pylab.ylabel('density')
            pylab.title('spin' + str(s) + 'bias=' + str(bias1) + '-' + str(bias2))
            pylab.show()

            tmp = sd1['s' + str(s) + 'vt_2d_' + dd[overview_d]] - sd2['s' + str(s) + 'vt_2d_' + dd[overview_d]]          
            cb = pylab.matshow(tmp * Hartree)
            pylab.title('spin' + str(s) + 'potential(eV) at bias=' + str(bias1) + '-' + str(bias2))
            pylab.colorbar()
            pylab.show()

            tmp = sd1['s' + str(s) + 'nt_2d_' + dd[overview_d]] - sd2['s' + str(s) + 'nt_2d_' + dd[overview_d]]              
            cb = pylab.matshow(tmp)
            gnu_save('diff_s', tmp) 
            pylab.title('spin' + str(s) + 'density at bias=' + str(bias1) + '-' + str(bias2))
            pylab.colorbar()                
            pylab.show()        
       
    def plot_step_data(self, step_data):
        overview_d = 1
        #self.d = 0
        self.nspins = 1
        sd = step_data['t_dos']
        bias = step_data['bias']
        import pylab
        pylab.figure(1)
        pylab.subplot(211)
        pylab.plot(sd['e_points'], sd['T_e'], 'b-o', sd['f1'], sd['l1'],
                                    'r--', sd['f2'], sd['l1'], 'r--')
        pylab.ylabel('Transmission Coefficients')
        pylab.subplot(212)
        pylab.plot(sd['e_points'], sd['dos_e'], 'b-o', sd['f1'], sd['l2'],
                                      'r--', sd['f2'], sd['l2'], 'r--')
        pylab.ylabel('Density of States')
        pylab.xlabel('Energy (eV)')
        pylab.title('bias=' + str(bias))
        pylab.show()
        
        sd = step_data['v_d']
        dd = ['x', 'y', 'z']
        for s in range(self.nspins):
            pylab.plot(sd['s' + str(s) + 'vt_1d_' + dd[self.d]] * Hartree, 'b--o') 
            pylab.ylabel('potential(eV)')
            pylab.title('spin' + str(s) + 'bias=' + str(bias))
            pylab.show()
        
            pylab.plot(sd['s' + str(s) + 'nt_1d_' + dd[self.d]], 'b--o')
            pylab.ylabel('density')
            pylab.title('spin' + str(s) + 'bias=' + str(bias))
            pylab.show()
        
            cb = pylab.matshow(sd['s' + str(s) + 'vt_2d_' + dd[overview_d]] * Hartree)
            pylab.title('spin' + str(s) + 'potential(eV) at bias=' + str(bias))
            pylab.colorbar()               
            pylab.show()
       
            cb = pylab.matshow(sd['s' + str(s) + 'nt_2d_' + dd[overview_d]])
            pylab.title('spin' + str(s) + 'density at bias=' + str(bias))
            pylab.colorbar()
            pylab.show()
        if self.nspins == 2:
            cb = pylab.matshow(sd['s' + str(0) + 'vt_2d_' + dd[overview_d]] * Hartree -
                           sd['s' + str(1) + 'vt_2d_' + dd[overview_d]] * Hartree)
            pylab.title('spin_diff' + 'potential(eV) at bias=' + str(bias))
            pylab.colorbar()                
            pylab.show()
       
            cb = pylab.matshow(sd['s' + str(0) + 'nt_2d_' + dd[overview_d]] -
                               sd['s' + str(1) + 'nt_2d_' + dd[overview_d]])
            pylab.title('spin_diff' + 'density at bias=' + str(bias))
            pylab.colorbar()                
            pylab.show()

    def plot_iv(self, i_v):
        v, i = i_v
        import pylab
        pylab.plot(v, i, 'b--o')
        pylab.xlabel('bias(V)')
        pylab.ylabel('current(au.)')
        pylab.show()
          
class Transport_Plotter:
    def __init__(self, flag='bias'):
        fd = file('analysis_data_' + flag, 'r')
        if flag == 'ion':
            self.ion_steps = pickle.load(fd)
        elif flag == 'bias':
            self.bias_steps = pickle.load(fd)
        else:
            self.ele_steps = pickle.load(fd)
        fd.close()

    def filter(self, n_ion_step=None, n_bias_step=0):
        if n_ion_step != None:
            self.bias_steps = self.ion_steps[n_ion_step].bias_steps
        self.ele_steps = self.bias_steps[n_bias_step].ele_steps
        
    def plot_ele_step(self, nstep, s, k):
        ee = np.linspace(-3, 5, 60)
        step = self.ele_steps[nstep]
        import pylab as p
        p.plot(step.dd[s, k], 'b--o')
        p.title('density matrix')
        p.show()
        
        p.plot(step.df[s, k], 'b--o')
        p.title('hamiltonian matrix')
        p.show()
        
        p.plot(step.nt, 'b--o')
        p.title('density')
        p.show()
 
        p.plot(step.vt, 'b--o')
        p.title('hamiltonian')
        p.show()
        
        p.plot(ee, step.tc[s, k, 0], 'b--o')
        p.title('transmission')
        p.show()
        
        p.plot(ee, step.dos[s, k], 'b--o')
        p.title('dos')
        p.show()
        
    def compare_two_calculations(self, nstep, s, k):
        fd = file('analysis_data_cmp', 'r')
        self.ele_steps_cmp = pickle.load(fd)
        fd.close()
        
        ee = np.linspace(-3, 5, 60)
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
        xdata = np.linspace(-3, 5, 60)
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
        p.title(title)
        p.legend(legends)
        if height != None:
            p.axis([xdata[0], xdata[-1], 0, height])
        p.show()
        
    def plot_ele_step_extended_info(self, info, steps_indices, s, k, unit=None):
        import pylab as p
        legends = []
        if info == 'edd':
            data = 'edmm[s, k]'
            title = 'extended density matrix diagonal elements'
        elif info == 'ent_G':
            data = 'ent_G'
            title = 'extended density on coarse gird'
        elif info == 'evt_G':
            data = 'evt_G'
            title = 'extended hamiltonian on coarse grid'
        elif info == 'ent_g':
            data = 'ent_g'
            title = 'extended density on fine grid'
        elif info == 'evt_g':
            data = 'evt_g'
            title = 'extended hamiltonian on fine grid'
        elif info == 'ehot_g':
            data = 'ehot_g'
            title = 'extended rho-density'
        elif info == 'vHt_g':
            data = 'evHt_g'
            title = 'extended Hartree potential'
        else:
            raise ValueError('no this info type---' + info)        

        for i, step in enumerate(self.ele_steps):
            if i in steps_indices:
                ydata = eval('step.' + data)
                if unit != None:
                    ydata = sum_by_unit(ydata, unit)
                p.plot(ydata)
                legends.append('step' + str(step.ele_step))
        p.title(title)
        p.legend(legends)
        p.show()        

    def compare_ele_step_info(self, info, steps_indices, s, k, height=None, unit=None):
        xdata = np.linspace(-3, 5, 60)
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
        p.title(title)
        p.legend(legends)
        if height != None:
            p.axis([xdata[0], xdata[-1], 0, height])
        p.show()
