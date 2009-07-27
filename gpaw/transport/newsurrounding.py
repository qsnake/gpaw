import numpy as np
from ase.units import Bohr, Hartree
from gpaw import GPAW, extra_parameters, debug
from gpaw.operators import Laplace
from gpaw.grid_descriptor import GridDescriptor
from gpaw.lcao.overlap import TwoCenterIntegrals
from gpaw.lfc import BasisFunctions, LFC
from gpaw.transformers import Transformer
from gpaw.xc_functional import XCFunctional, xcgrid
from gpaw.transport.tools import tri2full, count_tkpts_num, substract_pk, get_matrix_index
from gpaw.wavefunctions import LCAOWaveFunctions
from gpaw.density import Density
from gpaw.lcao.tools import get_realspace_hs
from gpaw.mpi import world
from gpaw.utilities import unpack, pack
from gpaw.utilities.blas import gemm
from gpaw.setup import Setups
from gpaw import debug
import pickle
from pylab import *

class Side:
    def __init__(self, type, atoms, direction='x+'):
        self.type = type
        self.atoms = atoms
        self.direction = direction
        self.index = {'x-':-1, 'x+':1, 'y-':-2, 'y+':2, 'z-':-3, 'z+':3}
        self.axis = abs(self.index[self.direction]) - 1

        self.n_atoms = len(atoms)
        calc = atoms.calc
        self.N_c = calc.gd.N_c
        self.nao = calc.wfs.setups.nao
        self.S_qMM = calc.wfs.S_qMM
        self.dH_asp = calc.hamiltonian.dH_asp
        self.my_atom_indices = calc.wfs.basis_functions.my_atom_indices

    def calculate(self):
        self.abstract_boundary()
        self.get_density_matrix()
        self.get_scaling()
        
    def get_scaling(self):
        calc = self.atoms.calc
        density = calc.density
        wfs = calc.wfs
        comp_charge = density.calculate_multipole_moments()
        pseudo_charge = -(density.charge + comp_charge)
        density.calculate_pseudo_density(wfs)
        assert abs(pseudo_charge) > 1.0e-14
        self.scaling = pseudo_charge / density.gd.integrate(density.nt_sG).sum()

    def abstract_boundary(self):
        calc = self.atoms.calc
        gd = calc.gd
        finegd = calc.finegd
        nn = finegd.N_c[self.axis]
        ns = calc.wfs.nspins
        
        vHt_g = finegd.empty(global_array=True)
        vHt_g = finegd.collect(calc.hamiltonian.vHt_g, True)
        self.boundary_vHt_g = self.slice(nn, vHt_g)
        
        vt_sg = finegd.empty(ns, global_array=True)
        vt_sg = finegd.collect(calc.hamiltonian.vt_sg, True)
        self.boundary_vt_sg = self.slice(nn, vt_sg)
        
        nt_sg = finegd.empty(ns, global_array=True)
        nt_sg = finegd.collect(calc.density.nt_sg, True)
        self.boundary_nt_sg = self.slice(nn, nt_sg)        
            
        nn /= 2
        
        vt_sG = gd.empty(ns, global_array=True)
        vt_sG = gd.collect(calc.hamiltonian.vt_sG, True)
        self.boundary_vt_sG = self.slice(nn, vt_sG)
        
        nt_sG = gd.empty(ns, global_array=True)
        nt_sG = calc.gd.collect(calc.density.nt_sG, True)
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
        restart = False
        if not restart:
            for kpt in wfs.kpt_u:
                wfs.calculate_density_matrix(kpt.f_n, kpt.C_nM,
                                            self.d_skmm[kpt.s, kpt.q])
            di = abs(self.index[self.direction]) - 1
            position = np.zeros(3)
            position[di] = 1
            ntk = count_tkpts_num(di, kpts)
            npk = len(wfs.ibzk_kc) / ntk
            my_npk = len(kpts) / ntk
            self.d_skmm *=  ntk
            #self.d_skmm *= ntk * npk
            self.d_spkmm = substract_pk(di, my_npk, ntk, kpts, self.d_skmm, 'h')
            self.d_spkcmm = substract_pk(di, my_npk, ntk, kpts, self.d_skmm, 'h',
                                         position)
            fd = file('sidedk.dat', 'wb')
            pickle.dump((self.d_spkmm, self.d_spkcmm), fd, 2)
            fd.close()
        else:
            fd = file('sidedk.dat', 'r')
            self.d_spkmm, self.d_spkcmm = pickle.load(fd)
            fd.close()
        #self.d_spkmm *= self.scaling
        #self.d_spkcmm *= self.scaling

    def slice(self, nn, in_array):
        direction = self.direction
        if self.type == 'LR':
            seq1 = np.arange(-nn + 1, 1)
            seq2 = np.arange(nn)
            di = self.axis + 1
            sign = direction[1]
            assert len(in_array.shape) == 4 or len(in_array.shape) == 3
            if len(in_array.shape) == 3:                 
                di -= 1
            if sign == '-':
                slice_array = np.take(in_array, seq1, axis=di)
            else:
                slice_array = np.take(in_array, seq2, axis=di)
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
        self.sides_index = {'x-':-1, 'x+':1, 'y-':-2,
                            'y+':2, 'z-':-3, 'z+': 3}
        self.initialized = False
        self.nspins = self.atoms.calc.wfs.nspins
        
    def initialize(self):
        if self.type == 'LR':
            self.lead_num = len(self.atoms_l)
            assert self.lead_num == len(self.bias)
            assert self.lead_num == len(self.directions)
            self.sides = {}
            self.bias_index = {}
            self.side_basis_index = {}
            self.nn = []
            for i in range(self.lead_num):
                direction = self.directions[i]
                side = Side('LR', self.atoms_l[i], direction)
                self.sides[direction] = side
                self.bias_index[direction] = self.bias[i]
                self.side_basis_index[direction] = self.lead_index[i]                
                self.nn.append(side.N_c[side.axis])
            
            self.nn = np.array(self.nn)
            self.axis = side.axis

            calc = self.atoms.calc
            wfs = calc.wfs
            self.gd0 = calc.gd
            self.finegd0 = calc.finegd
            
            gd = self.gd0
            domain_comm = gd.comm
            pbc = gd.pbc_c
            N_c = gd.N_c
            h_c = gd.h_c
            dim = N_c.copy()

            dim[self.axis] += np.sum(self.nn)
            
            self.cell = np.array(dim) * h_c
            self.gd = self.set_grid_descriptor(dim, self.cell,
                                               pbc, domain_comm)
            
            self.finegd = self.set_grid_descriptor(dim * 2, self.cell,
                                                   pbc, domain_comm)
            
            scale = -0.25 / np.pi
            self.operator = Laplace(self.finegd, scale,
                                    n=calc.hamiltonian.poisson.nn)
            
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
            
            self.get_extended_atoms()
            Za = self.extended_atoms.get_atomic_numbers()
            par = self.atoms.calc.input_parameters
            setups = Setups(Za, par.setups, par.basis,
                                           wfs.nspins, par.lmax, xcfunc)
            
            args = (self.gd, self.nspins, setups, wfs.bd,
                    wfs.dtype, wfs.world, wfs.kpt_comm,
                    wfs.gamma, wfs.bzk_kc, wfs.ibzk_kc,
                    wfs.weight_k, wfs.symmetry, wfs.timer)            
            
            self.wfs = LCAOWaveFunctions(*args)
            
            self.density = Density(self.gd, self.finegd, self.nspins,
                                   par.charge + setups.core_charge)
            
            magmom_a = self.extended_atoms.get_initial_magnetic_moments()
            self.density.initialize(setups, stencil, wfs.timer,
                                                magmom_a, par.hund)
            self.density.set_mixer(par.mixer, par.fixmom, par.width)            
           
            
            pos = self.extended_atoms.positions
            spos_ac = np.linalg.solve(np.diag(self.cell) * Bohr,
                                                              pos.T).T % 1.0
            self.wfs.set_positions(spos_ac)
            self.density.set_positions(spos_ac)
       
        elif self.type == 'all':
            raise NotImplementError()
        self.calculate_sides()
        self.combine()
        self.get_extra_density()
        self.initialized = True
        del self.atoms_l
        for direction in self.sides:
            del self.sides[direction].atoms

    def reset_bias(self, bias):
        self.bias = bias
        for i in range(self.lead_num):
            direction = self.directions[i]
            self.bias_index[direction] = bias[i]
        self.combine()
        self.get_extra_density()

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
            all_atoms = atoms_l.copy()
            all_atoms += atoms
            atoms_l = self.sides[direction + '+'].atoms.copy()
            for atom in atoms_l:
                atom.position[di] += cell_l[di] + cell[di]
            all_atoms += atoms_l
            all_atoms.set_cell(self.cell)
            all_atoms.set_pbc(self.atoms._pbc)
            self.extended_atoms = all_atoms
  
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
            vHt_g = self.finegd.zeros()
            self.finegd.distribute(self.boundary_vHt_g, vHt_g)
            self.operator.apply(vHt_g, rhot_g)
            nn = self.nn[0] * 2
            self.extra_rhot_g = self.uncapsule(nn, 'vHt_g',
                                             direction, rhot_g, collect=True)
            
    def calculate_hamiltonian_atomic_matrix(self, kpt):
        if self.type == 'LR':
            wfs = self.wfs            
            nao = wfs.setups.nao
            ah_mm = np.zeros([nao, nao], wfs.dtype)
            for a, P_Mi in kpt.P_aMi.items():
                dH_ii = np.asarray(unpack(self.dH_asp[a][kpt.s]), P_Mi.dtype)
                dHP_iM = np.zeros((dH_ii.shape[1], P_Mi.shape[0]),
                                                               P_Mi.dtype)
                gemm(1.0, P_Mi, dH_ii, 0.0, dHP_iM, 'c')
                gemm(1.0, dHP_iM, P_Mi, 1.0, ah_mm)
        return ah_mm
            
    def combine_atomic_hamiltonian(self):
        if self.type == 'LR':
            direction = self.directions[0][0]
            side1 = self.sides[direction + '-']
            side2 = self.sides[direction + '+']
            
            for a in range(side1.n_atoms):
                self.dH_asp[a] = side1.dH_asp[a]
            
            for a in range(len(self.atoms)):
                self.dH_asp[a + side1.n_atoms] = self.atoms.calc.hamiltonian.dH_asp[a]

            for a in range(side2.n_atoms):
                self.dH_asp[a + side1.n_atoms + len(self.atoms)] = side2.dH_asp[a]

    def combine_atomic_density(self): 
        if self.type == 'LR':
            direction = self.directions[0][0]
            side1 = self.sides[direction + '-']
            side2 = self.sides[direction + '+']
            
            for a in range(side1.n_atoms):
                self.density.D_asp[a] = side1.D_asp[a]
            
            for a in range(len(self.atoms)):
                self.density.D_asp[a + side1.n_atoms] = self.atoms.calc.density.D_asp[a]

            for a in range(side2.n_atoms):
                self.density.D_asp[a + side1.n_atoms + len(self.atoms)] = side2.D_asp[a]

 
    def calculate_potential_matrix(self, vt_sG0, kpt):
        s = kpt.s
        q = kpt.q
        nn = self.nn[0]
        direction = self.directions[0][0]
        nao0 = self.atoms.calc.wfs.setups.nao
        nao = self.wfs.setups.nao
        self.vt_MM = np.zeros([nao, nao], self.wfs.dtype)
        vt_sG = self.capsule(nn, 'vt_sG', direction, vt_sG0)
        self.wfs.basis_functions.calculate_potential_matrix(vt_sG[s],
                                                                self.vt_MM, q)
        
        Mstart = self.wfs.basis_functions.Mstart
        Mstop = self.wfs.basis_functions.Mstop
        test = np.zeros([nao, nao], self.wfs.dtype)
        
        for a, P_Mi in kpt.P_aMi.items():
            dH_ii = np.asarray(unpack(self.dH_asp[a][s]), P_Mi.dtype)
            dHP_iM = np.zeros((dH_ii.shape[1], P_Mi.shape[0]), P_Mi.dtype)
            gemm(1.0, P_Mi, dH_ii, 0.0, dHP_iM, 'c')
            if Mstart != -1:
                P_Mi = P_Mi[Mstart:Mstop]
            gemm(1.0, dHP_iM, P_Mi, 1.0, self.vt_MM)
            gemm(1.0, dHP_iM, P_Mi, 1.0, test)            
        self.gd.comm.sum(self.vt_MM)
   
        direction = self.directions[0][0] + '-'   
        nao1 = self.sides[direction].nao
        ind = get_matrix_index(np.arange(nao0) +  nao1)
        
        return self.vt_MM[ind.T, ind].copy()
     
    def restrict(self, vt_sg, s):
        nn = self.nn[0] * 2
        direction = self.directions[0][0]        
        vt_sg0 =  self.capsule(nn, 'vt_sg', direction, vt_sg)
        vt_G0 = self.gd.zeros()
        self.restrictor.apply(vt_sg0[s], vt_G0)
        nn /= 2
        return self.uncapsule(nn, 'vt_sG', direction, vt_G0, collect=True)
    
    def get_xc(self, nt_sg0, vt_sg0):
        nn = self.nn[0] * 2
        ns = nt_sg0.shape[0]
        direction = self.directions[0][0]
        nt_sg = self.capsule(nn, 'nt_sg', direction, nt_sg0)
        vt_sg = self.capsule(nn, 'vt_sg', direction, vt_sg0)
          
        if ns == 1:
            Exc = self.xc.get_energy_and_potential(nt_sg[0], vt_sg[0])
        else:
            Exc = self.xc.get_energy_and_potential(nt_sg[0], vt_sg[0],
                                                   nt_sg[1], vt_sg[1])
        vt_sg0 = self.uncapsule(nn, 'vt_sg', direction, vt_sg, collect=True)
        return Exc, vt_sg0
    
    def interpolate_density(self, density, comp_charge=None):
        if comp_charge is None:
            comp_charge = density.calculate_multipole_moments()
        if density.nt_sg is None:
            density.nt_sg = density.finegd.empty(density.nspins)
  
        nn = self.nn[0]
        direction = self.directions[0][0]
        self.nt_sg = self.finegd.zeros(self.nspins)
        for s in range(density.nspins):
            self.interpolator.apply(self.density.nt_sG[s], self.nt_sg[s])
        nn *= 2
        density.nt_sg = self.uncapsule(nn, 'nt_sg', direction,
                                     self.nt_sg, collect=True)

    def choose_gd(self, array_name):
        if array_name == 'nt_sG' or array_name == 'vt_sG':
            gd, gd0 = self.gd, self.gd0
        else:
            gd, gd0 = self.finegd, self.finegd0
        return gd, gd0        

    def choose_array(self, array_name):
        if array_name == 'vHt_g':
            array = self.vHt_g
        elif array_name == 'vt_sg':
            array = self.boundary_vt_sg
        elif array_name == 'nt_sg':
            array = self.boundary_nt_sg
        elif array_name == 'vt_sG':
            array = self.boundary_vt_sG
        elif array_name == 'nt_sG':
            array = self.boundary_nt_sG
        return array

    def capsule(self, nn, array_name, direction, loc_in_array, distribute=True):
        ns = self.nspins
        gd, gd0 = self.choose_gd(array_name)
        cap_array = self.choose_array(array_name)
        in_array = gd0.collect(loc_in_array, True)
        if self.type == 'LR':
            if len(in_array.shape) == 4:
                local_cap_array = gd.empty(ns)
                if direction == 'x':
                    cap_array[:, nn:-nn] = in_array
                elif direction == 'y':
                    cap_array[:, :, nn:-nn] = in_array
                elif direction == 'z':
                    cap_array[:, :, :, nn:-nn] = in_array
                else:
                    raise ValueError('unknown direction')
            else:
                local_cap_array = gd.empty()
                if direction == 'x':
                    cap_array[nn:-nn] = in_array
                elif direction == 'y':
                    cap_array[:, nn:-nn] = in_array
                elif direction == 'z':
                    cap_array[:, :, nn:-nn] = in_array
                else:
                    raise ValueError('unknown direction')
        if distribute:
            gd.distribute(cap_array, local_cap_array)
            return local_cap_array
        else:
            return cap_array
    
    def uncapsule(self, nn, array_name, direction, in_array, collect=False,
                                                               nn2=None):
        ns = self.nspins
        gd, gd0 = self.choose_gd(array_name)
        nn1 = nn
        if nn2 == None:
            nn2 = nn1
        if self.type == 'LR':
            assert len(in_array.shape) == 4 or len(in_array.shape) == 3
            di = abs(self.sides_index[direction + '-'])
            if len(in_array.shape) == 3:
                di -= 1
                local_uncap_array = gd0.empty()
                if collect:
                    global_in_array = gd.empty(global_array=True)
                    global_in_array = gd.collect(in_array, True)
                else:
                    global_in_array = in_array
            else:
                local_uncap_array = gd0.empty(ns)
                if collect:
                    global_in_array = gd.empty(ns, global_array=True)
                    global_in_array = gd.collect(in_array, True)
                else:
                    global_in_array = in_array
            seq = np.arange(nn1, global_in_array.shape[di] - nn2)    
            uncap_array = np.take(global_in_array, seq, axis=di)
        gd0.distribute(uncap_array, local_uncap_array)
        return local_uncap_array

    def allocate_global_array(self):
        ns = self.nspins
        self.nt_sG = self.gd.zeros(ns, global_array=True) 
        self.boundary_nt_sG = self.gd.zeros(ns, global_array=True) 
        
        self.nt_sg = self.finegd.zeros(ns, global_array=True)
        self.boundary_nt_sg = self.finegd.zeros(ns, global_array=True)
        
        self.vt_sG = self.gd.zeros(ns, global_array=True)
        self.boundary_vt_sG = self.gd.zeros(ns, global_array=True)
        
        self.vt_sg = self.finegd.zeros(ns, global_array=True)
        self.boundary_vt_sg = self.finegd.zeros(ns, global_array =True)        
        
        self.vHt_g = self.finegd.zeros(global_array=True)
        self.boundary_vHt_g = self.finegd.zeros(global_array=True)
    
    def combine(self):
        self.allocate_global_array()  
        if self.type == 'LR':
            direction = self.directions[0][0]
            N_c = self.gd0.N_c
            nn = self.nn[0] * 2
            nn2 = nn / 2
            bias_shift0 = self.bias_index[direction + '-'] / Hartree
            bias_shift1 = self.bias_index[direction + '+'] / Hartree
            side0 = self.sides[direction + '-']
            side1 = self.sides[direction + '+']
            if direction == 'x':
                self.boundary_nt_sG[:, :nn2] = side0.boundary_nt_sG 
                self.boundary_nt_sG[:, -nn2:] = side1.boundary_nt_sG                
                self.boundary_vt_sG[:, :nn2] = side0.boundary_vt_sG + bias_shift0
                self.boundary_vt_sG[:, -nn2:] = side1.boundary_vt_sG + bias_shift1

                self.boundary_nt_sg[:, :nn] = side0.boundary_nt_sg 
                self.boundary_nt_sg[:, -nn:] = side1.boundary_nt_sg
                self.boundary_vt_sg[:, :nn] = side0.boundary_vt_sg + bias_shift0
                self.boundary_vt_sg[:, -nn:] = side1.boundary_vt_sg + bias_shift1

                self.boundary_vHt_g[:nn] = side0.boundary_vHt_g + bias_shift0
                self.boundary_vHt_g[-nn:] = side1.boundary_vHt_g + bias_shift1

            elif direction == 'y':
                self.boundary_nt_sG[:, :, :nn2] = side0.boundary_nt_sG 
                self.boundary_nt_sG[:, :, -nn2:] = side1.boundary_nt_sG
                self.boundary_vt_sG[:, :, :nn2] = side0.boundary_vt_sG + bias_shift0
                self.boundary_vt_sG[:, :, -nn2:] = side1.boundary_vt_sG + bias_shift1

                self.boundary_nt_sg[:, :, :nn] = side0.boundary_nt_sg 
                self.boundary_nt_sg[:, :, -nn:] = side1.boundary_nt_sg
                self.boundary_vt_sg[:, :, :nn] = side0.boundary_vt_sg + bias_shift0
                self.boundary_vt_sg[:, :, -nn:] = side1.boundary_vt_sg + bias_shift1
            
                self.boundary_vHt_g[:, :nn] = side0.boundary_vHt_g + bias_shift0
                self.boundary_vHt_g[:, -nn:] = side1.boundary_vHt_g + bias_shift1
            
            elif direction == 'z':
                self.boundary_nt_sG[:, :, :, :nn2] = side0.boundary_nt_sG 
                self.boundary_nt_sG[:, :, :, -nn2:] = side1.boundary_nt_sG
                self.boundary_vt_sG[:, :, :, :nn2] = side0.boundary_vt_sG + bias_shift0
                self.boundary_vt_sG[:, :, :, -nn2:] = side1.boundary_vt_sG + bias_shift1
                
                self.boundary_nt_sg[:, :, :, :nn] = side0.boundary_nt_sg 
                self.boundary_nt_sg[:, :, :, -nn:] = side1.boundary_nt_sg
                self.boundary_vt_sg[:, :, :, :nn] = side0.boundary_vt_sg + bias_shift0
                self.boundary_vt_sg[:, :, :, -nn:] = side1.boundary_vt_sg + bias_shift1
                
                self.boundary_vHt_g[:, :, :nn] = side0.boundary_vHt_g + bias_shift0
                self.boundary_vHt_g[:, :, -nn:] = side1.boundary_vHt_g + bias_shift1
        
        self.vHt_g = self.boundary_vHt_g.copy()

    def calculate_pseudo_density(self, density):
        nn = self.nn[0]
        direction = self.directions[0][0]
        self.wfs.calculate_density_contribution(self.density.nt_sG)
        self.density.nt_sG += self.density.nct_G
        density.nt_sG = self.uncapsule(nn, 'nt_sG', direction, self.density.nt_sG,
                                                               collect=True)
        
    def calculate_pseudo_charge(self, density, comp_charge):
        nn = self.nn[0] * 2
        direction = self.directions[0][0]
        self.nt_g = self.nt_sg.sum(axis=0)
        self.rhot_g = self.nt_g.copy()
        
        comp_charge = self.calculate_multipole_moments()
        
        self.density.ghat.add(self.rhot_g, self.Q_aL)
        density.nt_g = self.uncapsule(nn, 'vHt_g', direction,
                                         self.nt_g, collect=True)        
        density.rhot_g = self.uncapsule(nn, 'vHt_g', direction,
                                         self.rhot_g, collect=True)

    def calculate_multipole_moments(self):
        comp_charge = 0.0
        self.Q_aL = {}
        for a, D_sp in self.density.D_asp.items():
            Q_L = self.Q_aL[a] = np.dot(D_sp.sum(0), self.wfs.setups[a].Delta_pL)
            Q_L[0] += self.wfs.setups[a].Delta0
            comp_charge += Q_L[0]
        return self.gd.comm.sum(comp_charge) * np.sqrt(4 * np.pi)
    
    def set_positions(self, atoms=None):
        calc = self.atoms.calc
        density = calc.density
        charge = calc.input_parameters.charge
        hund = calc.input_parameters.hund
        spos_ac = calc.initialize_positions(atoms)        
        self.initialize_from_atomic_densities(density, charge, hund)
        calc.update_hamiltonian(density)        
        calc.scf.reset()
        calc.forces.reset()
        calc.print_positions()        

    def calculate_atomic_density_matrices(self, density):
        wfs = self.wfs
        kpt_u = self.atoms.calc.wfs.kpt_u
        f_un = [kpt.f_n for kpt in kpt_u]
        self.calculate_atomic_density_matrices_with_occupation(f_un)
        if self.type == 'LR':
            direction = self.directions[0][0]            
            side1 = self.sides[direction + '-']
            natoms1 = side1.n_atoms
            for a in self.atoms.calc.wfs.basis_functions.my_atom_indices:
                density.D_asp[a][:] = self.density.D_asp[a + natoms1]
             
    def combine_density_matrix(self):
        wfs = self.atoms.calc.wfs
        direction = self.directions[0][0]            
        side1 = self.sides[direction + '-']
        side2 = self.sides[direction + '+']        
        nao1 = side1.nao
        ind1 = self.side_basis_index[direction + '-']
        ind10 = get_matrix_index(ind1 + nao1)
        ind1 = get_matrix_index(ind1)
        
        nao0 = wfs.setups.nao
        ind0 = get_matrix_index(np.arange(nao0) + nao1)
 
        nao2 = side2.nao
        ind2 = self.side_basis_index[direction + '+'] + nao1 + nao2
        ind20 = get_matrix_index(ind2 - nao2)
        ind2 = get_matrix_index(ind2)
           
        for kpt0, kpt1 in zip(wfs.kpt_u, self.wfs.kpt_u):
            s = kpt0.s
            q = kpt0.q
            if self.type == 'LR':
                nao = self.wfs.setups.nao
                rho_MM = np.zeros((nao, nao), kpt0.rho_MM.dtype)
                rho_MM[ind0.T, ind0] = kpt0.rho_MM
                rho_MM[ind1.T, ind1] = side1.d_spkmm[s, q]
                rho_MM[ind2.T, ind2] = side2.d_spkmm[s, q]
                rho_MM[ind1.T, ind10] = side1.d_spkcmm[s, q]
                rho_MM[ind10.T, ind1] = side1.d_spkcmm[s, q].T.conj()
                rho_MM[ind2.T, ind20] = side2.d_spkcmm[s, q].T.conj()
                rho_MM[ind20.T, ind2] = side2.d_spkcmm[s, q]
                kpt1.rho_MM = rho_MM.copy()    
                
    def calculate_atomic_density_matrices_with_occupation(self, f_un):
        wfs = self.wfs
        
        for a, D_sp in self.density.D_asp.items():
            ni = wfs.setups[a].ni
            D_sii = np.zeros((self.nspins, ni, ni))
            for f_n, kpt in zip(f_un, wfs.kpt_u):
                #print world.rank, a, kpt.P_aMi[a]
                wfs.calculate_atomic_density_matrices_k_point(D_sii, kpt,
                                                              a, f_n)
            D_sp[:] = [pack(D_ii) for D_ii in D_sii]
            wfs.band_comm.sum(D_sp)
            wfs.kpt_comm.sum(D_sp)
        wfs.symmetrize_atomic_density_matrices(self.density.D_asp) 
                 
    def initialize_from_atomic_densities(self, density, charge, hund):
        f_sM = np.empty((self.nspins, self.wfs.basis_functions.Mmax))
        self.density.D_asp = {}
        density.D_asp = {}
        f_asi = {}
        setups = self.wfs.setups
        basis_functions = self.wfs.basis_functions
        magmom_a = self.extended_atoms.get_initial_magnetic_moments()
        charge += setups.core_charge
        c = charge / len(setups) 
        for a in basis_functions.atom_indices:
            f_si = setups[a].calculate_initial_occupation_numbers(
                       magmom_a[a], hund, charge=c)
            self.density.D_asp[a] =  setups[a].initialize_density_matrix(f_si)
            f_asi[a] = f_si
        if self.type == 'LR':
            direction = self.directions[0][0]            
            side1 = self.sides[direction + '-']
            side2 = self.sides[direction + '+']
            natoms1 = side1.n_atoms
            for a in self.atoms.calc.wfs.basis_functions.my_atom_indices:
                density.D_asp[a] = self.density.D_asp[a + natoms1].copy()

        self.density.nt_sG = self.gd.zeros(self.nspins)
        basis_functions.add_to_density(self.density.nt_sG, f_asi)
        self.density.nt_sG += self.density.nct_G
        
        if self.type == 'LR':
            di = side1.axis
            nn1 = side1.N_c[di]
            nn2 = side2.N_c[di]
            density.nt_sG = self.uncapsule(nn1, 'nt_sG', direction,
                                                        self.density.nt_sG, True, nn2)
        
        comp_charge = density.calculate_multipole_moments()
        self.interpolate_density(density, comp_charge)
        self.calculate_pseudo_charge(density, comp_charge)
        density.rhot_g -= self.extra_rhot_g

    def set_grid_descriptor(self, dim, cell, pbc, domain_comm):
        gd = GridDescriptor(dim, cell, pbc, domain_comm)
        gd.use_fixed_bc = True
        return gd
        
    def calculate_pseudo_charge2(self, nt_sG0):
        nn = self.nn[0]
        direction = self.directions[0][0]
        nt_sG = self.capsule(nn, 'nt_sG', direction, nt_sG0, distribute=False)
        nt_sG = nt_sG[:,:,:,nn-1:-nn].copy()
        return np.sum(nt_sG) * self.gd.dv

    def update_density(self, density):
        self.combine_density_matrix()
        self.calculate_pseudo_density(density)
        self.calculate_atomic_density_matrices(density)
        comp_charge = self.calculate_multipole_moments()
        if not self.density.mixer.mix_rho:
            self.density.mixer.mix(self.density)
            comp_charge = None
        self.interpolate_density(density, comp_charge)
        self.calculate_pseudo_charge(density, comp_charge)
        if self.density.mixer.mix_rho:
            self.density.mixer.mix(self.density)            
        density.rhot_g -= self.extra_rhot_g

    def combine_vHt_g(self, vHt_g):
        nn = self.nn[0] * 2
        direction = self.directions[0][0]
        tmp = self.capsule(nn, 'vHt_g', direction, vHt_g, False)

    def calculate_atomic_hamiltonian_matrix(self, ham, Ekin, Ebar, Epot, Exc):
        self.combine_vHt_g(ham.vHt_g)
        W_aL = {}
        for a in self.density.D_asp:
            W_aL[a] = np.empty((self.wfs.setups[a].lmax + 1)**2)
        self.density.ghat.integrate(self.vHt_g, W_aL)
        self.dH_asp = {}
        for a, D_sp in self.density.D_asp.items():
            W_L = W_aL[a]
            setup = self.wfs.setups[a]
            D_p = D_sp.sum(0)
            dH_p = (setup.K_p + setup.M_p +
                    setup.MB_p + 2.0 * np.dot(setup.M_pp, D_p) +
                    np.dot(setup.Delta_pL, W_L))
            Ekin += np.dot(setup.K_p, D_p) + setup.Kc
            Ebar += setup.MB + np.dot(setup.MB_p, D_p)
            Epot += setup.M + np.dot(D_p, (setup.M_p + np.dot(setup.M_pp, D_p)))

            assert setup.HubU == None

            self.dH_asp[a] = dH_sp = np.zeros_like(D_sp)
            Exc += setup.xc_correction.calculate_energy_and_derivatives(
                D_sp, dH_sp, a)
            dH_sp += dH_p
            Ekin -= (D_sp * dH_sp).sum()
        
        ham.dH_asp = {}
        if self.type == 'LR':
            direction = self.directions[0][0]            
            side1 = self.sides[direction + '-']
            natoms1 = side1.n_atoms
            for a in self.atoms.calc.wfs.basis_functions.my_atom_indices:
                ham.dH_asp[a] = self.dH_asp[a + natoms1]            

        
        
        