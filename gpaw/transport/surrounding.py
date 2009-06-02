import numpy as np
from ase.units import Bohr, Hartree
from gpaw import GPAW, extra_parameters, debug
from gpaw.operators import Laplace
from gpaw.grid_descriptor import GridDescriptor
from gpaw.lcao.overlap import TwoCenterIntegrals
from gpaw.lfc import BasisFunctions, LFC
from gpaw.transformers import Transformer
from gpaw.xc_functional import XCFunctional, xcgrid
from gpaw.transport.tools import tri2full
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
        self.get_inner_nt_sG()

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
            if key in ['bias']:
                self.bias = sk['bias']
                del self.gpw_kwargs['bias']
        self.sides_index = {'x-':-1, 'x+':1, 'y-':-2, 'y+':2, 'z-':-3, 'z+': 3}
        self.initialized = False
        self.nn = 64
        self.nspins = self.atoms.calc.wfs.nspins
        
    def initialize(self):
        if self.type == 'LR':
            self.lead_num = len(self.atoms_l)
            assert self.lead_num == len(self.bias)
            assert self.lead_num == len(self.directions)
            self.sides = {}
            self.bias_index = {}
            for i in range(self.lead_num):
                direction = self.directions[i]
                self.sides[direction] = Side('LR',
                                             self.atoms_l[i],
                                             self.nn,
                                             direction)
                self.bias_index[direction] = self.bias[i]
            di = direction
            di = abs(self.sides_index[di]) - 1

            gd = self.atoms.calc.gd
            pbc = gd.pbc_c
            N_c = gd.N_c
            h_c = gd.h_c
            dim = N_c[:]

            dim[di] += self.nn
            self.cell = np.array(dim) * h_c
            self.gd = GridDescriptor(dim, self.cell, pbc)
            
            self.finegd = GridDescriptor(dim * 2, self.cell, pbc)
            scale = -0.25 / np.pi
            self.operator = Laplace(self.finegd, scale, n=1)
            
            wfs = self.atoms.calc.wfs
            self.basis_functions = BasisFunctions(self.gd, 
                                                  [setup.phit_j
                                                   for setup in wfs.setups],
                                                  None,
                                                  cut=True)
            pos = self.atoms.positions.copy()
            for i in range(len(self.atoms)):
                pos[i, di] += self.nn * h_c[di] * Bohr / 2
            spos_ac = np.linalg.solve(np.diag(self.cell) * Bohr, pos.T).T % 1.0
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
              
            N_c = self.extend_cell // (self.gd.h_c)
            gd = GridDescriptor(N_c,
                                self.extend_cell / Bohr, self.extended_atoms._pbc)
            # the spacing infomation self.gd.N_c here can be arbitary 
            
            self.tci = TwoCenterIntegrals(gd, setups,
                                                      wfs.gamma, wfs.ibzk_qc)
            self.P_aqMi = {}
            nao = setups.nao
            S_qMM = np.empty((nq, nao, nao), wfs.dtype)
            T_qMM = np.empty((nq, nao, nao), wfs.dtype)

            for a in range(len(Za)):
                ni = setups[a].ni
                self.P_aqMi[a] = np.empty((nq, nao, ni), wfs.dtype)
            for kpt in self.kpt_u:
                q = kpt.q
                kpt.P_aMi = dict([(a, P_qMi[q])
                                  for a, P_qMi in self.P_aqMi.items()])                
            pos = self.extended_atoms.positions
            spos_ac = np.linalg.solve(np.diag(self.extend_cell),
                                                              pos.T).T % 1.0                
            self.tci.set_positions(spos_ac)
            self.tci.calculate(spos_ac, S_qMM, T_qMM, self.P_aqMi)
            
           
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
            al_atoms._pbc[di] = True
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
    
    def uncapsule(self, nn, direction, in_array):
        if self.type == 'LR':
            if len(in_array.shape) == 4:
                if direction == 'x':
                    uncap_array = in_array[:, nn:-nn]
                elif direction == 'y':
                    uncap_array = in_array[:, :, nn:-nn]
                elif direction == 'z':
                    uncap_array = in_array[:, :, :, nn:-nn]
                else:
                    raise ValueError('unknown direction')
            else:
                if direction == 'x':
                    uncap_array = in_array[nn:-nn]
                elif direction == 'y':
                    uncap_array = in_array[:, nn:-nn]
                elif direction == 'z':
                    uncap_array = in_array[:, :, nn:-nn]
                else:
                    raise ValueError('unknown direction')                
        return uncap_array.copy()
                
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
                self.streching_nt_sG[:, :nn2] = self.sides['x-'].inner_nt_sG 
                self.streching_nt_sG[:, -nn2:] = self.sides['x+'].inner_nt_sG

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
                
                self.streching_nt_sG[:, :, :nn2] = self.sides['y-'].inner_nt_sG 
                self.streching_nt_sG[:, :, -nn2:] = self.sides['y+'].inner_nt_sG

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
                
                self.streching_nt_sG[:, :, :, :nn2] = self.sides['z-'].inner_nt_sG 
                self.streching_nt_sG[:, :, :, -nn2:] = self.sides['z+'].inner_nt_sG

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
                