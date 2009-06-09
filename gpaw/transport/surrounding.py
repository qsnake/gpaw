import numpy as np
from ase.units import Bohr, Hartree
from gpaw import GPAW, extra_parameters, debug
from gpaw.operators import Laplace
from gpaw.grid_descriptor import GridDescriptor
from gpaw.lcao.overlap import TwoCenterIntegrals
from gpaw.lfc import BasisFunctions, LFC
from gpaw.transformers import Transformer
from gpaw.xc_functional import XCFunctional, xcgrid
from gpaw.transport.tools import tri2full, count_tkpts_num, substract_pk
from gpaw.lcao.tools import get_realspace_hs
from gpaw.mpi import world
from gpaw.utilities import unpack
from gpaw.utilities.blas import gemm
from gpaw.setup import Setups

class Side:
    def __init__(self, type, atoms, nn, direction='x+'):
        self.type = type
        self.atoms = atoms
        self.nn = nn
        self.direction = direction

    def initialize(self):
        calc = self.atoms.calc
        rcut = []
        for setup in calc.wfs.setups:
            rcut.append(max(setup.rcut_j))
        rcutmax = max(rcut)
        nn_max = np.ceil(rcutmax / min(calc.gd.h_c))
        assert nn_max < self.nn

    def calculate(self):
        self.initialize()
        self.abstract_boundary_vHt_g()
        self.abstract_boundary_vt_sg()        
        self.abstract_boundary_vt_sG()
        self.abstract_boundary_nt_sg()
        self.abstract_boundary_nt_sG()            
        #self.get_inner_nt_sG()
        self.get_density_matrix()

    def abstract_boundary_vHt_g(self):
        vHt_g = self.atoms.calc.hamiltonian.vHt_g
        nn = self.nn
        self.boundary_vHt_g = self.slice(nn, vHt_g)

    def abstract_boundary_vt_sg(self):
        vt_sg = self.atoms.calc.hamiltonian.vt_sg
        nn = self.nn
        self.boundary_vt_sg = self.slice(nn, vt_sg)
        
    def abstract_boundary_nt_sg(self):
        nt_sg = self.atoms.calc.density.nt_sg
        nn = self.nn
        self.boundary_nt_sg = self.slice(nn, nt_sg)
        
    def abstract_boundary_vt_sG(self):
        vt_sG = self.atoms.calc.hamiltonian.vt_sG
        nn = self.nn / 2
        self.boundary_vt_sG = self.slice(nn, vt_sG)
    
    def abstract_boundary_nt_sG(self):
        nt_sG = self.atoms.calc.density.nt_sG
        nn = self.nn / 2
        self.boundary_nt_sG = self.slice(nn, nt_sG)

    def get_density_matrix(self):
        calc = self.atoms.calc
        gd = calc.gd
        wfs = calc.wfs
        nao = wfs.setups.nao
        ns = wfs.nspins
        kpts = wfs.ibzk_qc
        nq = len(kpts)
        self.d_skmm = np.empty([ns, nq, nao, nao], wfs.dtype)
        for kpt in wfs.kpt_u:
            wfs.calculate_density_matrix(kpt.f_n, kpt.C_nM,
                                            self.d_skmm[kpt.s, kpt.q])
        self.index = {'x-':-1, 'x+':1, 'y-':-2, 'y+':2, 'z-':-3, 'z+':3}  
        di = abs(self.index[self.direction]) - 1
        position = np.zeros(3)
        position[di] = 1
        ntk = count_tkpts_num(di, kpts)
        npk = len(wfs.ibzk_kc) / ntk
        my_npk = len(kpts) / ntk
        self.d_skmm *= ntk * npk
        self.d_spkmm = substract_pk(di, my_npk, ntk, kpts, self.d_skmm, 'h')
        self.d_spkcmm = substract_pk(di, my_npk, ntk, kpts, self.d_skmm, 'h',
                                     position)
   
    def get_inner_nt_sG(self):
        calc = self.atoms.calc
        gd = calc.gd
        wfs = calc.wfs
        nao = wfs.setups.nao
        ns = wfs.nspins
        nk = len(wfs.ibzk_kc)
        d_skmm = np.empty([ns, nk, nao, nao], wfs.dtype)
        for kpt in wfs.kpt_u:
            wfs.calculate_density_matrix(kpt.f_n, kpt.C_nM,
                                            d_skmm[kpt.s, kpt.k])
        weight_k = np.zeros([len(wfs.ibzk_kc)]) + 1
        d_srmm = get_realspace_hs(d_skmm, None, wfs.ibzk_kc,
                                    weight_k, R_c=(0,0,0), usesymm=False)        
        cell = gd.N_c * gd.h_c
        self.index = {'x-':-1, 'x+':1, 'y-':-2, 'y+':2, 'z-':-3, 'z+': 3}        
        pbc = gd.pbc_c.copy()
        pbc[abs(self.index[self.direction]) - 1] = False
        
        tmp_gd = GridDescriptor(gd.N_c, cell, pbc)        
        basis_functions = BasisFunctions(tmp_gd, 
                                        [setup.phit_j
                                        for setup in wfs.setups],
                                        None,
                                        cut=True)
        pos = self.atoms.positions
        spos_ac = np.linalg.solve(np.diag(cell) * Bohr, pos.T).T % 1.0
        basis_functions.set_positions(spos_ac)
        nt_sG = tmp_gd.zeros(ns)
        for s in range(ns):
            basis_functions.construct_density(d_srmm[s], nt_sG[s], 0)
        nt_sG0 = calc.density.nt_sG - calc.density.nct_G
        nn = self.nn / 2
        self.inner_nt_sG = self.slice_diff(nn, nt_sG0, nt_sG)
      
    def slice(self, nn, in_array):
        direction = self.direction
        if self.type == 'LR':
            seq = np.arange(-nn+1, 1)
            if len(in_array.shape) == 4:
                if direction == 'x-':
                    slice_array = in_array[:, seq, :, :]
                elif self.direction == 'x+':
                    slice_array = in_array[:, :nn, :, :]
                elif self.direction == 'y-':
                    slice_array = in_array[:, :, seq, :]
                elif self.direction == 'y+':
                    slice_array = in_array[:, :, :nn, :]
                elif self.direction == 'z-':
                    slice_array = in_array[:, :, :, seq]
                elif self.direction == 'z+':
                    slice_array = in_array[:, :, :, :nn]
                else:
                    raise ValueError('wrong direction value')
            elif len(in_array.shape) == 3:
                if direction == 'x-':
                    slice_array = in_array[seq, :, :]
                elif self.direction == 'x+':
                    slice_array = in_array[:nn, :, :]
                elif self.direction == 'y-':
                    slice_array = in_array[:, seq, :]
                elif self.direction == 'y+':
                    slice_array = in_array[:, :nn, :]
                elif self.direction == 'z-':
                    slice_array = in_array[:, :, seq]
                elif self.direction == 'z+':
                    slice_array = in_array[:, :, :nn]
                else:
                    raise ValueError('wrong direction value')            
            else:
                raise RuntimeError('wrong in_array')
        return slice_array

    def slice2(self, nn, in_array):
        direction = self.direction
        if self.type == 'LR':
            seq1 = np.arange(-nn + 1, 1)
            seq2 = np.arange(nn)
            di = abs(self.index[direction])
            sign = direction[1]
            assert len(in_array.shape) == 4 or len(in_array.shape) == 3
            if len(in_array.shape) == 3:                 
                di -= 1
            if sign == '-':
                slice_array = np.take(in_array, seq1, axis=di)
            else:
                slice_array = np.take(in_array, seq2, axis=di)
        return slice_array

    def slice_diff2(self, nn, in_array1, in_array2):
        direction = self.direction
        sign = direction[1]
        di = direction[0]
        if self.type == 'LR':
            assert len(in_array1.shape) == 4 and len(in_array2.shape) == 4
            seq1 = np.arange(-nn + 1, 1)
            seq2 = np.arange(nn)
            if sign == '-':
                slice_array = np.take(in_array1 - in_array2, seq1, axis=di)
            else:
                slice_array = np.take(in_array1 - in_array2, seq2, axis=di)
        return slice_array
    
    def slice_diff(self, nn, in_array1, in_array2):
        direction = self.direction
        if self.type == 'LR':
            if len(in_array1.shape) == 4 and len(in_array2.shape) == 4:
                if direction == 'x+':
                    slice_array = in_array1[:, -nn:] - in_array2[:, -nn:]
                elif self.direction == 'x-':
                    slice_array = in_array1[:, 1:nn+1] - in_array2[:, :nn]
                elif self.direction == 'y+':
                    slice_array = in_array1[:, :, -nn:] - \
                                                       in_array2[:, :, -nn:]
                elif self.direction == 'y-':
                    slice_array = in_array1[:, :, 1:nn+1] - \
                                                       in_array2[:, :, :nn]
                elif self.direction == 'z+':
                    slice_array = in_array1[:, :, :, -nn:] - \
                                                      in_array2[:, :, :, -nn:]
                elif self.direction == 'z-':
                    slice_array = in_array1[:, :, :, 1:nn+1] - \
                                                       in_array2[:, :, :, :nn]
                else:
                    raise ValueError('wrong direction value')
            else:
                raise RuntimeError('wrong in_array')
        return slice_array
 
class Surrounding:
    def __init__(self, **s_kwargs):
        self.set_kwargs(**s_kwargs)

    def set_kwargs(self, **s_kwargs):
        sk = s_kwargs
        self.gpw_kwargs = sk.copy()
        for key in sk:
            if key in ['type']:
                self.type = sk['type']
                del self.gpw_kwargs['type']
            if key in ['atoms']:
                self.atoms = sk['atoms']
                del self.gpw_kwargs['atoms']
            if key in ['atoms_l']:
                self.atoms_l = sk['atoms_l']
                del self.gpw_kwargs['atoms_l']
            if key in ['directions']:
                self.directions = sk['directions']
                del self.gpw_kwargs['directions']
            if key in ['lead_index']:
                self.lead_index = sk['lead_index']
            if key in ['bias']:
                self.bias = sk['bias']
                del self.gpw_kwargs['bias']
        self.sides_index = {'x-':-1, 'x+':1, 'y-':-2, 'y+':2, 'z-':-3, 'z+': 3}
        self.initialized = False
        self.nn = 32
        self.nspins = self.atoms.calc.wfs.nspins
        
    def initialize(self):
        if self.type == 'LR':
            self.lead_num = len(self.atoms_l)
            assert self.lead_num == len(self.bias)
            assert self.lead_num == len(self.directions)
            self.sides = {}
            self.bias_index = {}
            self.side_basis_index = {}
            for i in range(self.lead_num):
                direction = self.directions[i]
                self.sides[direction] = Side('LR',
                                             self.atoms_l[i],
                                             self.nn,
                                             direction)
                self.bias_index[direction] = self.bias[i]
                self.side_basis_index[direction] = self.lead_index[i]                
            di = direction
            di = abs(self.sides_index[di]) - 1

            gd = self.atoms.calc.gd
            domain_comm = gd.comm
            pbc = gd.pbc_c
            N_c = gd.N_c
            h_c = gd.h_c
            dim = N_c.copy()

            dim[di] += self.nn
            self.cell = np.array(dim) * h_c
            self.gd = GridDescriptor(dim, self.cell, pbc, domain_comm)
            
            self.finegd = GridDescriptor(dim * 2, self.cell, pbc, domain_comm)
            scale = -0.25 / np.pi
            self.operator = Laplace(self.finegd, scale, n=1)
            wfs = self.atoms.calc.wfs
            self.basis_functions = BasisFunctions(self.gd, 
                                                  [setup.phit_j
                                                   for setup in wfs.setups],
                                                  wfs.kpt_comm,
                                                  cut=True)
            pos = self.atoms.positions.copy()
            for i in range(len(self.atoms)):
                pos[i, di] += self.nn * h_c[di] * Bohr / 2
            spos_ac = np.linalg.solve(np.diag(self.cell) * Bohr, pos.T).T % 1.0
            if not wfs.gamma:
                self.basis_functions.set_k_points(wfs.ibzk_qc)            
            self.basis_functions.set_positions(spos_ac)
            stencil = self.atoms.calc.input_parameters.stencils[1]
            self.restrictor = Transformer(self.finegd, self.gd, stencil,
                                                               allocate=False)
            self.interpolator = Transformer(self.gd, self.finegd, stencil,
                                                               allocate=False)            
            xcfunc = self.atoms.calc.hamiltonian.xcfunc
            self.xc = xcgrid(xcfunc, self.finegd, self.nspins)
            self.restrictor.allocate()
            self.interpolator.allocate()
            self.xc.allocate()
            
            setups = wfs.setups
            self.nct = LFC(self.gd, [[setup.nct] for setup in setups],
                          integral=[setup.Nct for setup in setups],
                          forces=True, cut=True)
            self.ghat = LFC(self.finegd, [setup.ghat_l for setup in setups],
                        integral=np.sqrt(4 * np.pi), forces=True)
          
            self.nct.set_positions(spos_ac)
            self.ghat.set_positions(spos_ac)

            nq = len(wfs.ibzk_qc)
            nao = wfs.setups.nao            
            self.vt_MM = np.empty([nao, nao], wfs.dtype)       
            self.sah_spkmm = np.zeros((wfs.nspins, nq, nao, nao), wfs.dtype)        
            self.nct_G = self.gd.zeros()
            self.nct.add(self.nct_G, 1.0 / self.nspins)      
            
            import copy
            self.kpt_u = copy.deepcopy(wfs.kpt_u)
            self.get_extended_atoms()
            Za = self.extended_atoms.get_atomic_numbers()
            par = self.atoms.calc.input_parameters
            setups = Setups(Za, par.setups, par.basis,
                                           wfs.nspins, par.lmax, xcfunc)

            N_c = self.atoms.calc.gd.N_c.copy()
            for i in range(self.lead_num):
                N_c[di] += self.atoms_l[i].calc.gd.N_c[di]
            self.extended_gd = GridDescriptor(N_c, self.extend_cell / Bohr,
                                              self.extended_atoms._pbc,
                                              domain_comm)
            self.extended_basis_functions = BasisFunctions(self.extended_gd, 
                                                  [setup.phit_j
                                                   for setup in setups],
                                                  wfs.kpt_comm,
                                                  cut=True)
            
            self.tci = TwoCenterIntegrals(self.extended_gd, setups,
                                                      wfs.gamma, wfs.ibzk_qc)
            self.extended_setups = setups
            self.extended_P_aqMi = {}
            nao = setups.nao
            S_qMM = np.empty((nq, nao, nao), wfs.dtype)
            T_qMM = np.empty((nq, nao, nao), wfs.dtype)

            for a in range(len(Za)):
                ni = setups[a].ni
                self.extended_P_aqMi[a] = np.empty((nq, nao, ni), wfs.dtype)
            for kpt in self.kpt_u:
                q = kpt.q
                kpt.P_aMi = dict([(a, P_qMi[q])
                                for a, P_qMi in self.extended_P_aqMi.items()])                
            pos = self.extended_atoms.positions
            spos_ac = np.linalg.solve(np.diag(self.extend_cell),
                                                              pos.T).T % 1.0                
            if not wfs.gamma:
                self.extended_basis_functions.set_k_points(wfs.ibzk_qc)     
            self.extended_basis_functions.set_positions(spos_ac)
            self.tci.set_positions(spos_ac)
            self.tci.calculate(spos_ac, S_qMM, T_qMM, self.extended_P_aqMi)
            self.extended_nct = LFC(self.extended_gd, [[setup.nct]
                                                for setup in setups],
                                                integral=[setup.Nct
                                                for setup in setups],
                                                forces=True, cut=True)
            self.extended_nct.set_positions(spos_ac)
            self.extended_nct_G = self.extended_gd.zeros()
            self.extended_nct.add(self.extended_nct_G, 1.0 / self.nspins)                 
           
        elif self.type == 'all':
            self.sides = {}
            self.atoms._pbc = self.pbc
            self.sides['all'] = Side('all',
                                      self.nn,
                                      self.atoms,
                                      self.kpts,
                                      self.gpw_kwargs)
            dim = self.N_c[:]
            dim += self.nn
            self.cell = dim * self.h_c
            self.gd = GridDescriptor(self.N_c, self.cell, False)

            self.finegd = GridDescriptor(dim * 2, self.cell, False)

            scale = -0.25 / np.pi
            self.operator = Laplace(self.finegd, scale, n=1)
 
            wfs = self.atoms.calc.wfs
            self.basis_functions = BasisFunctions(self.gd, 
                                                   [setup.phit_j
                                                   for setup in wfs.setups],
                                                  None,
                                                  cut=True)
            
            pos = self.atoms.positions
            for i in range(len(self.atoms)):
                pos[i] += self.nn * self.h_c * Bohr  / 2
            spos_ac = np.linalg.solve(np.diag(self.cell) * Bohr, pos.T).T% 1.0
            self.basis_functions.set_positions(spos_ac) 
            nao = wfs.setups.nao
            #pay attention to dtype  
            self.vt_MM = np.empty([nao, nao], float)
            self.s_MM = np.empty([nao, nao])

        self.calculate_sides()
        self.combine()
        self.get_extra_density()
        self.initialized = True

    def reset_bias(self, bias):
        self.bias = bias
        for i in range(self.lead_num):
            direction = self.directions[i]
            self.bias_index[direction] = bias[i]
        self.combine()
        self.get_extra_density()

    def get_extended_cell(self):
        cell = self.atoms.cell
        if len(cell.shape) == 2:
            cell = np.diag(cell)
        dim = np.arange(3)
        if self.type == 'LR':
            for i in range(self.lead_num):
                direction = self.directions[i]
                cell_l = self.atoms_l[i].cell
                if len(cell_l.shape) == 2:
                    cell_l = np.diag(cell_l)
                di = abs(self.sides_index[direction]) - 1
                cell[di] += cell_l[di]        
                dim_left = np.delete(dim, [di])
                for j in dim_left:
                    assert cell[j] == cell_l[j]
        self.extend_cell = cell

    def get_extended_atoms(self):
        atoms = self.atoms.copy()
        cell = atoms.cell
        if len(cell.shape) == 2:
            cell = np.diag(cell)
        n_atoms = len(atoms)
        if self.type == 'LR':
            direction = self.directions[0][0]
            di = abs(self.sides_index[direction + '-']) - 1
            atoms_l = self.sides[direction + '-'].atoms.copy()
            cell_l = atoms_l.cell
            if len(cell_l.shape) == 2:
                cell_l = np.diag(cell_l)
            for atom in atoms:
                atom.position[di] += cell_l[di] 
            al_atoms = atoms_l.copy()
            al_atoms += atoms
            atoms_l = self.sides[direction + '+'].atoms.copy()
            for atom in atoms_l:
                atom.position[di] += cell_l[di] + cell[di]
            al_atoms += atoms_l
            self.get_extended_cell()
            al_atoms.set_cell(self.extend_cell)
            al_atoms.set_pbc(self.atoms._pbc)
            #al_atoms._pbc[di] = True
            self.extended_atoms = al_atoms
  
    def calculate_sides(self):
        if self.type == 'LR':
            for i in range(self.lead_num):
                direction = self.directions[i]
                side = self.sides[direction]
                side.calculate()
        if self.type == 'all':
            raise NotImplementError('type all not yet')
            
    def get_extra_density(self):
        if self.type == 'LR':
            direction = self.directions[0][0]
            rhot_g = self.finegd.zeros()
            self.operator.apply(self.vHt_g, rhot_g)
            self.extra_rhot_g = self.uncapsule(self.nn, direction, rhot_g)
            
    def combine_streching_atomic_hamiltonian(self):
        if self.type == 'LR':
            direction = self.directions[0][0]
            
            atoms1 = self.sides[direction + '-'].atoms
            bias_shift1 = self.bias_index[direction + '-'] / Hartree
            wfs = atoms1.calc.wfs
            nao1 = wfs.setups.nao
            S_qMM1 = wfs.S_qMM
            n_atoms1 = len(atoms1)

            atoms2 = self.sides[direction + '+'].atoms
            bias_shift2 = self.bias_index[direction + '+'] / Hartree            
            wfs = atoms2.calc.wfs
            nao2 = wfs.setups.nao
            S_qMM2 = wfs.S_qMM
            n_atoms2 = len(atoms2)
            
            wfs = self.atoms.calc.wfs
            nao0 = wfs.setups.nao
            n_atoms0 = len(self.atoms)
            nao = nao1 + nao2 + nao0
            
            n_atoms = n_atoms1 + n_atoms2 + n_atoms0
            
            s_qmm1 = np.zeros((nao, nao), wfs.dtype)
            s_qmm2 = np.zeros((nao, nao), wfs.dtype)            
            for kpt in self.kpt_u:
                s = kpt.s
                q = kpt.q
                sah_mm1 = np.zeros((nao, nao), wfs.dtype)
                sah_mm2 = np.zeros((nao, nao), wfs.dtype)
                s_qmm1[:nao1, :nao1] = S_qMM1[q]
                s_qmm2[-nao2:, -nao2:] = S_qMM2[q]                
                for a, P_Mi in kpt.P_aMi.items():
                    dtype = P_Mi.dtype
                    if a in range(n_atoms1):
                        ex_a = a
                        ham = atoms1.calc.hamiltonian
                        dH_ii = np.asarray(unpack(ham.dH_asp[ex_a][s]), dtype)
                        dHP_iM = np.empty((dH_ii.shape[1], P_Mi.shape[0]),
                                                                       dtype)
                        gemm(1.0, P_Mi, dH_ii, 0.0, dHP_iM, 'c')
                        gemm(1.0, dHP_iM, P_Mi, 1.0, sah_mm1)
                        sah_mm1 += bias_shift1 * s_qmm1[q]
                    elif a in range(n_atoms0 + n_atoms1, n_atoms):
                        ex_a = a - n_atoms0 - n_atoms1
                        ham = atoms2.calc.hamiltonian
                        dH_ii = np.asarray(unpack(ham.dH_asp[ex_a][s]), dtype)
                        dHP_iM = np.empty((dH_ii.shape[1], P_Mi.shape[0]),
                                                                       dtype)
                        gemm(1.0, P_Mi, dH_ii, 0.0, dHP_iM, 'c')
                        gemm(1.0, dHP_iM, P_Mi, 1.0, sah_mm2)
                        sah_mm2 += bias_shift2 * s_qmm2[q]
                sah_mm = sah_mm1 + sah_mm2        
                self.sah_spkmm[s, q] = sah_mm[nao1: nao1 + nao0,
                                           nao1: nao1 + nao0].copy()

    def combine_streching_density(self):
        if self.type == 'LR':
            direction = self.directions[0][0]
            di = abs(self.sides_index[direction + '-']) - 1
            side1 = self.sides[direction + '-']
            wfs = side1.atoms.calc.wfs
            nao1 = wfs.setups.nao
            ind1 = self.side_basis_index[direction + '-'] + nao1
            dim = len(ind1)
            ind1 = np.resize(ind1, [dim, dim])
            nn1 = side1.atoms.calc.gd.N_c[di]

            side2 = self.sides[direction + '+']
            wfs = side2.atoms.calc.wfs
            nao2 = wfs.setups.nao
            ind2 = self.side_basis_index[direction + '+'] + nao1
            dim = len(ind2)
            ind2 = np.resize(ind2, [dim, dim])            
            nn2 = side2.atoms.calc.gd.N_c[di]            
            
            wfs = self.atoms.calc.wfs
            nao0 = wfs.setups.nao
            nao = nao1 + nao2 + nao0
            kpts = wfs.ibzk_qc
            nq = len(kpts)
            self.d_spkmm = np.zeros([self.nspins, nq, nao, nao], wfs.dtype)
            self.extended_nt_sG = self.extended_gd.zeros(self.nspins)
            for kpt in self.kpt_u:
                s = kpt.s
                q = kpt.q
                self.d_spkmm[s, q, :nao1, :nao1] = side1.d_spkmm[s, q]
                self.d_spkmm[s, q, -nao2:, -nao2:] = side2.d_spkmm[s, q]
                
                ind = np.arange(nao1)
                dim = len(ind)
                ind = np.resize(ind, [dim, dim])                
                self.d_spkmm[s, q, ind.T, ind1] = side1.d_spkcmm[s, q]
                self.d_spkmm[s, q, ind1.T, ind] = side1.d_spkcmm[s, q].T.conj()
                
                ind = np.arange(nao2) + nao1 + nao0
                dim = len(ind)
                ind = np.resize(ind, [dim, dim])                       
                self.d_spkmm[s, q, ind2.T, ind] = side2.d_spkcmm[s, q]
                self.d_spkmm[s, q, ind.T, ind2] = side2.d_spkcmm[s, q].T.conj()
                self.extended_basis_functions.construct_density(
                                                    self.d_spkmm[s, q],
                                                    self.extended_nt_sG[s], q)
            wfs.band_comm.sum(self.extended_nt_sG)       
            wfs.kpt_comm.sum(self.extended_nt_sG)
            self.streching_nt_sG = self.uncapsule(nn1, direction,
                                                  self.extended_nt_sG, nn2)
 
    def calculate_potential_matrix(self, vt_sG0, s, q):
        nn = self.nn / 2
        direction = self.directions[0][0]
        vt_sG = self.capsule(nn, 'vt_sG', direction, vt_sG0)
        self.basis_functions.calculate_potential_matrix(vt_sG[s],
                                                            self.vt_MM, q)
        tri2full(self.vt_MM)
        return self.vt_MM
     
    def restrict(self, vt_sg, s):
        nn = self.nn
        direction = self.directions[0][0]
        vt_sg0 =  self.capsule(nn, 'vt_sg', direction, vt_sg)
        vt_G0 = self.gd.zeros()
        self.restrictor.apply(vt_sg0[s], vt_G0)
        nn /= 2
        return self.uncapsule(nn, direction, vt_G0)
    
    def get_xc(self, nt_sg0, vt_sg0):
        nn = self.nn
        ns = nt_sg0.shape[0]
        direction = self.directions[0][0]
        nt_sg = self.capsule(nn, 'nt_sg', direction, nt_sg0)
        vt_sg = self.capsule(nn, 'vt_sg', direction, vt_sg0)        
        if ns == 1:
            Exc = self.xc.get_energy_and_potential(nt_sg[0], vt_sg[0])
        else:
            Exc = self.xc.get_energy_and_potential(nt_sg[0], vt_sg[0],
                                                   nt_sg[1], vt_sg[1])
        vt_sg0 = self.uncapsule(nn, direction, vt_sg)
        return Exc, vt_sg0
    
    def interpolate_density(self, density, comp_charge=None):
        if comp_charge is None:
            comp_charge = density.calculate_multipole_moments()
        if density.nt_sg is None:
            density.nt_sg = density.finegd.empty(density.nspins)
        nt_sG0 = density.nt_sG
        nn = self.nn / 2
        direction = self.directions[0][0]
        nt_sG = self.capsule(nn, 'nt_sG', direction, nt_sG0)
        nt_sg = self.finegd.zeros(self.nspins)
        for s in range(density.nspins):
            self.interpolator.apply(nt_sG[s], nt_sg[s])
        nn *= 2
        density.nt_sg = self.uncapsule(nn, direction, nt_sg)

    def capsule(self, nn, array_name, direction, in_array):
        if array_name == 'nt_sg':
            cap_array = self.nt_sg.copy()
        elif array_name == 'vHt_g':
            cap_array = self.vHt_g.copy()            
        elif array_name == 'vt_sg':
            cap_array = self.vt_sg.copy()
        elif array_name == 'vt_sG':
            cap_array = self.vt_sG.copy()
        elif array_name == 'nt_sG':
            cap_array = self.nt_sG.copy()            
        else:
            raise ValueError('unknown array_name')
        if self.type == 'LR':
            if len(in_array.shape) == 4:
                if direction == 'x':
                    cap_array[:, nn:-nn] = in_array[:]
                elif direction == 'y':
                    cap_array[:, :, nn:-nn] = in_array[:]
                elif direction == 'z':
                    cap_array[:, :, :, nn:-nn] = in_array[:]
                else:
                    raise ValueError('unknown direction')
            else:
                if direction == 'x':
                    cap_array[nn:-nn] = in_array[:]
                elif direction == 'y':
                    cap_array[:, nn:-nn] = in_array[:]
                elif direction == 'z':
                    cap_array[:, :, nn:-nn] = in_array[:]
                else:
                    raise ValueError('unknown direction')                
        return cap_array
    
    def uncapsule(self, nn, direction, in_array, nn2=None):
        nn1 = nn
        if nn2 == None:
            nn2 = nn1
        if self.type == 'LR':
            if len(in_array.shape) == 4:
                if direction == 'x':
                    uncap_array = in_array[:, nn1:-nn2]
                elif direction == 'y':
                    uncap_array = in_array[:, :, nn1:-nn2]
                elif direction == 'z':
                    uncap_array = in_array[:, :, :, nn1:-nn2]
                else:
                    raise ValueError('unknown direction')
            else:
                if direction == 'x':
                    uncap_array = in_array[nn1:-nn2]
                elif direction == 'y':
                    uncap_array = in_array[:, nn1:-nn2]
                elif direction == 'z':
                    uncap_array = in_array[:, :, nn1:-nn2]
                else:
                    raise ValueError('unknown direction')                
        return uncap_array.copy()

    def uncapsule2(self, nn, direction, in_array, nn2=None):
        nn1 = nn
        if nn2 == None:
            nn2 = nn1
        if self.type == 'LR':
            seq = np.arange(nn1, -nn2)
            assert len(in_array.shape) == 4 or len(in_array.shape) == 3
            di = abs(self.sides_index[direction + '-'])
            if len(in_array.shape) == 3:
                di -= 1
            uncap_array = np.take(in_array, seq, axis=di)               
        return uncap_array
    
    def combine(self):
        if self.type == 'LR':
            ns = self.nspins
            direction = self.directions[0][0]
            gd0 = self.atoms.calc.gd
            self.streching_nt_sG = gd0.zeros(ns)
            self.nt_sG = self.gd.zeros(ns) 
            self.nt_sg = self.finegd.zeros(ns)
            self.vt_sG = self.gd.zeros(ns)
            self.vt_sg = self.finegd.zeros(ns)
            self.vHt_g = self.finegd.zeros()
            nn = self.nn
            nn2 = self.nn / 2
            if direction == 'x':
                assert gd0.N_c[0] > nn
                
                bias_shift0 = self.bias_index['x-'] / Hartree
                bias_shift1 = self.bias_index['x+'] / Hartree
                #self.streching_nt_sG[:, :nn2] = self.sides['x-'].inner_nt_sG 
                #self.streching_nt_sG[:, -nn2:] = self.sides['x+'].inner_nt_sG

                self.nt_sG[:, :nn2] = self.sides['x-'].boundary_nt_sG 
                self.nt_sG[:, -nn2:] = self.sides['x+'].boundary_nt_sG
                
                self.vt_sG[:, :nn2] = self.sides['x-'].boundary_vt_sG + bias_shift0
                self.vt_sG[:, -nn2:] = self.sides['x+'].boundary_vt_sG + bias_shift1
               
                self.nt_sg[:, :nn] = self.sides['x-'].boundary_nt_sg 
                self.nt_sg[:, -nn:] = self.sides['x+'].boundary_nt_sg

                self.vt_sg[:, :nn] = self.sides['x-'].boundary_vt_sg + bias_shift0
                self.vt_sg[:, -nn:] = self.sides['x+'].boundary_vt_sg + bias_shift1
                
                self.vHt_g[:nn] = self.sides['x-'].boundary_vHt_g + bias_shift0
                self.vHt_g[-nn:] = self.sides['x+'].boundary_vHt_g + bias_shift1
            
            elif direction == 'y':
                assert gd0.N_c[1] > nn
                bias_shift0 = self.bias_index['y-'] / Hartree
                bias_shift1 = self.bias_index['y+'] / Hartree
                
                #self.streching_nt_sG[:, :, :nn2] = self.sides['y-'].inner_nt_sG 
                #self.streching_nt_sG[:, :, -nn2:] = self.sides['y+'].inner_nt_sG

                self.nt_sG[:, :, :nn2] = self.sides['y-'].boundary_nt_sG 
                self.nt_sG[:, :, -nn2:] = self.sides['y+'].boundary_nt_sG
                
                self.vt_sG[:, :, :nn2] = self.sides['y-'].boundary_vt_sG + bias_shift0
                self.vt_sG[:, :, -nn2:] = self.sides['y+'].boundary_vt_sG + bias_shift1
                
                self.nt_sg[:, :, :nn] = self.sides['y-'].boundary_nt_sg 
                self.nt_sg[:, :, -nn:] = self.sides['y+'].boundary_nt_sg

                self.vt_sg[:, :, :nn] = self.sides['y-'].boundary_vt_sg + bias_shift0
                self.vt_sg[:, :, -nn:] = self.sides['y+'].boundary_vt_sg + bias_shift1
                
                self.vHt_g[:, :nn] = self.sides['y-'].boundary_vHt_g + bias_shift0
                self.vHt_g[:, -nn:] = self.sides['y+'].boundary_vHt_g + bias_shift1
            
            elif direction == 'z':
                assert gd0.N_c[2] > nn
                bias_shift0 = self.bias_index['z-'] / Hartree
                bias_shift1 = self.bias_index['z+'] / Hartree
                
                #self.streching_nt_sG[:, :, :, :nn2] = self.sides['z-'].inner_nt_sG 
                #self.streching_nt_sG[:, :, :, -nn2:] = self.sides['z+'].inner_nt_sG

                self.nt_sG[:, :, :, :nn2] = self.sides['z-'].boundary_nt_sG 
                self.nt_sG[:, :, :, -nn2:] = self.sides['z+'].boundary_nt_sG
                
                self.vt_sG[:, :, :, :nn2] = self.sides['z-'].boundary_vt_sG + bias_shift0
                self.vt_sG[:, :, :, -nn2:] = self.sides['z+'].boundary_vt_sG + bias_shift1
                
                self.nt_sg[:, :, :, :nn] = self.sides['z-'].boundary_nt_sg 
                self.nt_sg[:, :, :, -nn:] = self.sides['z+'].boundary_nt_sg

                self.vt_sg[:, :, :, :nn] = self.sides['z-'].boundary_vt_sg + bias_shift0
                self.vt_sg[:, :, :, -nn:] = self.sides['z+'].boundary_vt_sg + bias_shift1
                
                self.vHt_g[:, :, :nn] = self.sides['z-'].boundary_vHt_g + bias_shift0
                self.vHt_g[:, :, -nn:] = self.sides['z+'].boundary_vHt_g + bias_shift1
            self.combine_streching_density()
            self.combine_streching_atomic_hamiltonian()                   

    def calculate_pseudo_density(self, density, wfs):
        nn = self.nn / 2
        direction = self.directions[0][0]
        wfs.calculate_density_contribution(density.nt_sG)
        density.nt_sG += self.uncapsule(nn, direction, self.nct_G)
        
    def calculate_pseudo_charge(self, density, comp_charge):
        nn = self.nn
        direction = self.directions[0][0]
        density.nt_g = density.nt_sg.sum(axis=0)
        density.rhot_g = density.nt_g.copy()
        rhot_g = self.finegd.zeros()
        self.ghat.add(rhot_g, density.Q_aL)
        density.rhot_g += self.uncapsule(nn, direction, rhot_g)
        if debug:
            charge = self.finegd.integrate(self.rhot_g) + self.charge
            if abs(charge) > self.charge_eps:
                raise RuntimeError('Charge not conserved: excess=%.9f' %
                                   charge)        

    def set_positions(self, atoms=None):
        calc = self.atoms.calc
        wfs = calc.wfs
        density = calc.density
        hamiltonian = calc.hamiltonian
        charge = calc.input_parameters.charge
        hund = calc.input_parameters.hund
        spos_ac = calc.initialize_positions(atoms)        
        #self.wfs.initialize(self.density, self.hamiltonian, spos_ac)
        if density.nt_sG is None:
            if wfs.kpt_u[0].f_n is None or wfs.kpt_u[0].C_nM is None:
                self.initialize_from_atomic_densities(density, charge, hund)
            else:
                raise NonImplementError('missing the initialization from wfs')
        else:
            density.calculate_normalized_charges_and_mix()
        calc.update_hamiltonian(density)        
        calc.scf.reset()
        calc.forces.reset()
        calc.print_positions()        
                    
    def initialize_from_atomic_densities(self, density, charge, hund):
        f_sM = np.empty((self.nspins, self.extended_basis_functions.Mmax))
        self.extend_D_asp = {}
        density.D_asp = {}
        f_asi = {}
        setups = self.extended_setups
        basis_functions = self.extended_basis_functions
        magmom_a = self.extended_atoms.get_initial_magnetic_moments()
        charge += setups.core_charge
        c = charge / len(setups) 
        for a in basis_functions.atom_indices:
            f_si = setups[a].calculate_initial_occupation_numbers(
                       magmom_a[a], hund, charge=c)
            if a in basis_functions.my_atom_indices:
                self.extend_D_asp[a] = setups[a].initialize_density_matrix(f_si)
            f_asi[a] = f_si
        if self.type == 'LR':
            direction = self.directions[0][0]            
            di = abs(self.sides_index[direction + '-']) - 1
            atoms1 = self.sides[direction + '-'].atoms
            natoms1 = len(atoms1)
            atoms2 = self.sides[direction + '+'].atoms            
            natoms0 = len(self.atoms)
            for a in range(natoms0):
                density.D_asp[a] = self.extend_D_asp[a + natoms1].copy()
        self.extended_nt_sG = self.extended_gd.zeros(self.nspins)
        basis_functions.add_to_density(self.extended_nt_sG, f_asi)
        self.extended_nt_sG += self.extended_nct_G
        if self.type == 'LR':
            nn1 = atoms1.calc.gd.N_c[di]
            nn2 = atoms2.calc.gd.N_c[di]
            density.nt_sG = self.uncapsule(nn1, direction,
                                           self.extended_nt_sG, nn2)
        comp_charge = density.calculate_multipole_moments()
        if not density.mixer.mix_rho:
            density.mixer.mix(density)
            comp_charge = None
        self.interpolate_density(density, comp_charge)
        self.calculate_pseudo_charge(density, comp_charge)
        if density.mixer.mix_rho:
            density.mixer.mix(density)
        density.rhot_g -= self.extra_rhot_g              



