from ase import Hartree
import numpy as np


class Side:
    def __init__(self, type, atoms, direction):
        self.type = type
        self.atoms = atoms
        self.direction = direction
        self.n_atoms = len(atoms)
        calc = atoms.calc
        self.N_c = calc.gd.N_c

    def abstract_boundary(self):
        calc = self.atoms.calc
        gd = calc.gd
        finegd = calc.finegd
        nn = finegd.N_c[2]
        ns = calc.wfs.nspins

        dim = gd.N_c
        d1 = dim[0] // 2
        d2 = dim[1] // 2
        
        vHt_g = finegd.collect(calc.hamiltonian.vHt_g, True)
        self.boundary_vHt_g = self.slice(nn, vHt_g)
        
        vt_sg = finegd.collect(calc.hamiltonian.vt_sg, True)
        self.boundary_vt_sg_line = self.slice(nn, vt_sg[:, d1 * 2, d2 * 2])
        
        nt_sg = finegd.collect(calc.density.nt_sg, True)
        self.boundary_nt_sg_line = self.slice(nn, nt_sg[:, d1 * 2, d2 * 2])        
        
        rhot_g = finegd.collect(calc.density.rhot_g, True)
        self.boundary_rhot_g_line = self.slice(nn, rhot_g[d1 * 2, d2 * 2])
  
        nn /= 2
        vt_sG = gd.collect(calc.hamiltonian.vt_sG, True)
        self.boundary_vt_sG_line = self.slice(nn, vt_sG[:, d1, d2])
        
        nt_sG = calc.gd.collect(calc.density.nt_sG, True)
        self.boundary_nt_sG_line = self.slice(nn, nt_sG[:, d1, d2])
        
    def slice(self, nn, in_array):
        if self.type == 'LR':
            seq1 = np.arange(-nn + 1, 1)
            seq2 = np.arange(nn)
            di = len(in_array.shape) - 1
            if self.direction == '-':
                slice_array = np.take(in_array, seq1, axis=di)
            else:
                slice_array = np.take(in_array, seq2, axis=di)
        return slice_array

class Surrounding:
    def __init__(self, tp, type='LR'):
        self.tp = tp
        self.type = type
        self.lead_num = tp.lead_num
        self.initialize()
        
    def initialize(self):
        if self.type == 'LR':
            self.sides = {}
            self.bias_index = {}
            self.side_basis_index = {}
            self.nn = []
            self.directions = ['-', '+']
            for i in range(self.lead_num):
                direction = self.directions[i]
                side = Side('LR', self.tp.atoms_l[i], direction)
                self.sides[direction] = side
                self.bias_index[direction] = self.tp.bias[i]
                self.side_basis_index[direction] = self.tp.lead_index[i]                
                self.nn.append(side.N_c[2])
            self.nn = np.array(self.nn)
            self.operator = self.tp.hamiltonian.poisson.operators[0]
            
        elif self.type == 'all':
            raise NotImplementError()
        self.calculate_sides()
        self.initialized = True

    def reset_bias(self, bias):
        self.bias = bias
        for i in range(self.lead_num):
            direction = self.directions[i]
            self.bias_index[direction] = bias[i]
        self.combine()
        self.get_extra_density()

    def calculate_sides(self):
        if self.type == 'LR':
            for name, in self.sides:
                self.sides[name].abstract_boundary()
        if self.type == 'all':
            raise NotImplementError('type all not yet')
            
    def get_extra_density(self):
        if self.type == 'LR':
            rhot_g = self.tp.finegd.zeros()
            self.operator.apply(self.tp.hamiltonian.vHt_g, rhot_g)
            nn = self.nn[0] * 2
            self.extra_rhot_g = self.uncapsule(nn, rhot_g)

    def capsule(self, nn, loc_in_array):
        ns = self.tp.nspins
        gd, gd0 = self.tp.finegd, self.tp.finegd0
        cap_array = self.tp.hamiltonian.vHt_g
        in_array = gd0.collect(loc_in_array, True)
        if len(in_array.shape) == 4:
            local_cap_array = gd.empty(ns)
            cap_array[:, :, :, nn:-nn] = in_array
        else:
            local_cap_array = gd.empty()
            cap_array[:, :, nn:-nn] = in_array
        gd.distribute(cap_array, local_cap_array)
        return local_cap_array
    
    def uncapsule(self, nn, in_array, nn2=None):
        gd, gd0 = self.tp.finegd, self.tp.finegd0
        nn1 = nn
        if nn2 == None:
            nn2 = nn1
        di = 2
        local_uncap_array = gd0.empty()
        global_in_array = gd.collect(in_array, True)
        seq = np.arange(nn1, global_in_array.shape[di] - nn2)    
        uncap_array = np.take(global_in_array, seq, axis=di)
        gd0.distribute(uncap_array, local_uncap_array)
        return local_uncap_array
      
    def combine(self):
        if self.type == 'LR':
            nn = self.nn[0] * 2
            ham = self.tp.hamiltonian
            if ham.vt_sg is None:
                ham.vt_sg = ham.finegd.empty(ham.nspins)
                ham.vHt_g = ham.finegd.zeros()
                ham.vt_sG = ham.gd.empty(ham.nspins)
                ham.poisson.initialize()            
            vHt_g = ham.vHt_g
            bias_shift0 = self.bias_index['-'] / Hartree
            bias_shift1 = self.bias_index['+'] / Hartree
            vHt_g.fill(0.0)                
            vHt_g[:, :, :nn] = self.sides['-'].boundary_vHt_g + bias_shift0
            vHt_g[:, :, -nn:] = self.sides['+'].boundary_vHt_g + bias_shift1
            self.get_extra_density()

    def combine_vHt_g(self, vHt_g):
        nn = self.nn[0] * 2
        self.tp.hamiltonian.vHt_g = self.capsule(nn, vHt_g)

    def abstract_inner_rhot(self):
        nn = self.nn[0] * 2
        rhot_g = self.uncapsule(nn, self.tp.density.rhot_g)
        rhot_g -= self.extra_rhot_g
        return rhot_g
        