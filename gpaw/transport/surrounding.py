from gpaw import *
import numpy as np
from gpaw.operators import Laplace
from gpaw.grid_descriptor import GridDescriptor
from gpaw.lfc import BasisFunctions
from gpaw.transport.tools import tri2full
from gpaw.mpi import world
import pickle


class Side:
    def __init__(self, type, nn, atoms,
                 kpts, gpw_kwargs, direction='x+', bias=0):
        self.type = type
        self.atoms = atoms
        self.kpts = kpts
        self.bias = bias
        self.pbc = np.array(pbc)
        self.nn = nn
        self.gpw_kwargs = gpw_kwargs
        self.gpw_kwargs['kpts'] = self.kpts
        self.direction = direction
        self.boundary_vHt_g = None
        self.boundary_vt_sG = None

    def initialize(self):
        self.atoms.set_calculator(GPAW(**self.gpw_kwargs))
        self.atoms.center()
        calc = self.atoms.calc
        rcut = []
        for setup in calc.wfs.setups:
            rcut.append(max(setup.rcut_j))
        rcutmax = max(rcut)
        nn_max = np.ceil(rcutmax / min(calc.gd.h_c))
        assert nnmax < self.nn

    def calculate(self):
        self.initialize()
        self.atoms.get_potential_energy()

    def substract_boundary_vHt(self):
        calc = self.atoms.calc
        vHt_g = calc.hamiltonian.vHt_g
        dim = vHt_g.shape
        nx, ny, nz = dim
        nn = self.nn
        seq = np.array(range(-nn + 1, 1))
        if self.type == 'LR':
            if self.direction == 'x-':
                self.boundary_vHt_g = vHt_g[seq, :, :]
            elif self.direction == 'x+':
                self.boundary_vHt_g = vHt_g[:nn, :, :]
            elif self.direction == 'y-':
                self.boundary_vHt_g = vHt_g[:, seq, :]
            elif self.direction == 'y+':
                self.boundary_vHt_g = vHt_g[:, :nn, :]
            elif self.direction == 'z-':
                self.boundary_vHt_g = vHt_g[:, :, seq]
            elif self.direction == 'z+':
                self.boundary_vHt_g = vHt_g[:, :, :nn]
            else:
                raise ValueError('wrong direction value')
        elif self.type == 'all':
            self.boundary_vHt_g = vHt_g.copy()
        self.boundary_vHt_g += self.bias / Hartree

    def substract_boundary_vt_sG(self):
        calc = self.atoms.calc
        vt_sG = calc.hamiltonian.vt_sG
        dim = vt_sG.shape
        ns, nx, ny, nz = dim
        nn = self.nn
        seq = np.array(range(-nn + 1, 1))
        if self.type == 'LR':
            if self.direction == 'x-':
                self.boundary_vt_sG = vt_sG[:, seq, :, :]
            elif self.direction == 'x+':
                self.boundary_vt_sG = vt_sG[:, :nn, :, :]
            elif self.direction == 'y-':
                self.boundary_vt_sG = vt_sG[:, :, seq, :]
            elif self.direction == 'y+':
                self.boundary_vt_sG = vt_sG[:, :, :nn, :]
            elif self.direction == 'z-':
                self.boundary_vt_sG = vt_sG[:, :, :, seq]
            elif self.direction == 'z+':
                self.boundary_vt_sG = vt_sG[:, :, :, :nn]
            else:
                raise ValueError('wrong direction value')
        elif self.type == 'all':
            self.boundary_vt_sG = vt_sG.copy()
        self.boundary_vt_sG += self.bias / Hartree 

    def save_boundary(self):
        if world.rank == 0:
            fd = file(self.direction + '.bdv', 'wb')
            pickle.dump( (self.boundary_vHt_g,
                          self.boundary_vt_sG), fd, 2)
            fd.close()

    def revoke(self):
        fd = file(self.direction + '.bdv', 'r') 
        self.boundary_vHt_g, self.boundary_vt_sG = pickle.load(fd)
        fd.close()               
 
class Surrounding:
    def __init__(self, **s_kwargs):
        self.set_kwargs(**s_kwargs)

    def set_kwargs(self, **s_kwargs):
        sk = s_kwargs
        self.gpw_kwargs = sk.copy()
        for key in sk:
            if key in ['name']:
                self.name = sk['name']
                del self.gpw_kwargs['name']
            if key in ['type']:
                self.type = sk['type']
                del self.gpw_kwargs['type']
            if key in ['atoms']:
                self.atoms = sk['atoms']
                del self.gpw_kwargs['atoms']
            if key in ['pbc']:
                self.pbc = sk['pbc']
                del self.gpw_kwargs['pbc']
            if key in ['h_c']:
                self.h_c = sk['h_c']
                del self.gpw_kwargs['h_c']
            if key in ['directions']:
                self.directions = sk['directions']
                del self.gpw_kwargs['directions']
            if key in ['N_c']:
                self.N_c = sk['N_c']
                del self.gpw_kwargs['N_c']
            if key in ['kpts']:
                self.kpts = sk['kpts']
                del self.gpw_kwargs['kpts']
            if key in ['pl_atoms']:
                self.pl_atoms = sk['pl_atoms']
                del self.gpw_kwargs['pl_atoms']
            if key in ['pl_cells']:
                self.pl_cells = sk['pl_cells']
                del self.gpw_kwargs['pl_cells']
            if key in ['pl_kpts']:
                self.pl_kpts = sk['pl_kpts']
                del self.gpw_kwargs['pl_kpts']
            if key in ['pl_pbcs']:
                self.pl_pbcs = sk['pl_pbcs']
                del self.gpw_kwargs['pl_pbcs']
            if key in ['bias']:
                self.bias = sk['bias']
                del self.gpw_kwargs['bias']
        self.sides_index = {'x-':-1, 'x+':1, 'y-':-2, 'y+':2, 'z-':-3, 'z+': 3}
        self.initialized = False
        self.nn = 50
        self.nspins = self.atoms.calc.wfs.nspins
        
    def initialize(self):
        if self.type == 'LR':
            self.lead_num = len(self.pl_atoms)
            assert self.lead_num == len(self.pl_cells)
            assert self.lead_num == len(self.pl_pbcs)
            assert self.lead_num == len(self.bias)
            assert self.lead_num == len(self.directions)
            self.sides = {}
            for i in range(self.lead_num):
                direction = self.directions[i]
                atoms = self.get_side_atoms(i)
                self.sides[direction] = Side('LR',
                                             self.nn,
                                             atoms,
                                             self.pl_kpts,
                                             self.gpw_kwargs,
                                             direction,
                                             self.bias[i])
            di = direction
            di = abs(self.sides_index[di]) - 1
            dim = self.N_c[:]
            dim[di] += 2 * self.nn
            self.N_c = dim
            self.cell = np.array(dim) * self.h_c
            self.gd = GridDescriptor(self.N_c, self.cell, False)
            scale = -0.25 / np.pi
            self.operator = Laplace(self.gd, scale, n=1)
            
            wfs = self.atoms.calc.wfs
            self.basis_functions = BasisFunctions(self.gd, 
                                                  [setup.phit_j
                                                   for setup in wfs.setups],
                                                  wfs.kpt_comm,
                                                  cut=True)
            pos = self.atoms.positions
            for i in range(len(self.atoms)):
                pos[i, di] += self.nn * self.h_c[di]
            spos_ac = np.linalg.solve(np.diag(self.cell) * Bohr, pos.T).T
            self.basis_functions.set_positions(spos_ac) 

            nao = wfs.setups.nao
            vt_MM = np.empty([nap, nao])
            
        elif self.type == 'all':
            self.sides = {}
            self.atoms._pbc = self.pbc
            self.sides['all'] = Side('all',
                                     self.nn,
                                     self.atoms,
                                     self.kpts,
                                     self.gpw_kwargs)
            dim = self.N_c.copy()
            dim += 2 * self.nn
            self.N_c = dim
            self.cell = dim * self.h_c
            self.gd = GridDescriptor(self.N_c, self.cell, False)
            scale = -0.25 / np.pi
            self.operator = Laplace(self.gd, scale, n=1)
            
            wfs = self.atoms.calc.wfs
            self.basis_functions = BasisFunctions(self.gd, 
                                                  [setup.phit_j
                                                   for setup in wfs.setups],
                                                  wfs.kpt_comm,
                                                  cut=True)
            pos = self.atoms.positions
            for i in range(len(self.atoms)):
                pos[i] += self.nn * self.h_c
            spos_ac = np.linalg.solve(np.diag(self.cell) * Bohr, pos.T).T
            self.basis_functions.set_positions(spos_ac) 
            nao = wfs.setups.nao
            vt_MM = np.empty([nap, nao])

        self.initialized = True
        self.gpw_restart = False
        self.restart = False
  
    def get_side_atoms(self, l):
        atoms = self.atoms.copy()
        atomsl = atoms[self.pl_atoms[l]]
        atomsl.cell = self.pl_cells[l]
        atomsl._pbc = self.pl_pbc[l]
        atomsl.center()
        return atomsl
        
    def calculate_sides(self):
        if self.type == 'LR':
            for i in range(self.lead_num):
                direction = self.directions[i]
                side = self.sides[direction]
                if not self.restart:
                    side.calculate()
                    side.substract_boundary_vHt_g()
                    side.substract_boundary_vt_sG()
                    side.save_boundary()
                else:
                    side.revoke()
        if self.type == 'all':
            side = self.side['all']
            if not self.restart:
                side.calculate()
                side.substract_boundary_vHt_g()
                side.substract_boundary_vt_sG()
                side.save_boundary()
            else:
                side.revoke()
        self.combine_vHt()
        self.combine_vt_sG()

    def combine_vHt(self):
        nn = self.nn
        if self.type == 'LR':
            di = self.directions[0][0]
            self.vHt_g = self.gd.zeros(1)
            self.vHt_g.shape = self.vHt_g.shape[1:]
            if di == 'x':
                self.vHt_g[:nn] = self.sides['x-'].boundary_vHt_g 
                self.vHt_g[-nn:] = self.sides['x+'].boundary_vHt_g
            elif di == 'y':
                self.vHt_g[:, :nn] = self.sides['y-'].boundary_vHt_g
                self.vHt_g[:, -nn:] = self.sides['y+'].boundary_vHt_g
            elif di == 'z':
                self.vHt_g[:, :, :nn] = self.sides['z-'].boundary_vHt_g
                self.vHt_g[:, :, -nn:] = self.sides['z+'].boundary_vHt_g

        elif self.type == 'all':
            for i in range(1,4):
                dim[i] += 2 * self.nn
            self.vHt_g = self.gd.zeors(1)
            self.vHt_g.shape = self.vHt_g.shape[1:]

    def combine_vt_sG(self):
        nn = self.nn
        if self.type == 'LR':
            di = self.directions[0][0]
            self.vt_sG = self.gd.zeros(self.nspins)
            if di == 'x':
                self.vt_sG[:, :nn] = self.sides['x-'].boundary_vt_sG 
                self.vt_sG[:, -nn:] = self.sides['x+'].boundary_vt_sG
            elif di == 'y':
                self.vt_sG[:, :, :nn] = self.sides['y-'].boundary_vt_sG
                self.vt_sG[:, :, -nn:] = self.sides['y+'].boundary_vt_sG
            elif di == 'z':
                self.vt_sG[:, :, :, :nn] = self.sides['z-'].boundary_vt_sG
                self.vt_sG[:, :, :, -nn:] = self.sides['z+'].boundary_vt_sG

        elif self.type == 'all':
            for i in range(1,4):
                dim[i] += 2 * self.nn
            self.vHt_g = self.gd.zeors(1)
            self.vHt_g.shape = self.vHt_g.shape[1:]
             
    def get_extra_density(self):
        if self.type == 'LR':
            direction = self.directions[0][0]
            self.rhot_g = self.gd.zeros(1)
            self.rhot_g.shape = self.rhot_g.shape[1:]
            self.operator.apply(self.vHt_g, self.rhot_g)
            nn = self.nn
            if direction == 'x':
                self.rhot_g_inner = self.rhot_g[nn:-nn]
            elif direction == 'y':
                self.rhot_g_inner = self.rhot_g[:, nn:-nn]
            elif direction == 'z':
                self.rhot_g_inner = self.rhot_g[:, :, nn:-nn]
        return self.rhot_g_inner
         
    def get_matrix_projection(self):
        self.basis_functions.calculate_potential_matrix(self.vt_sG, self.vt_MM, 0)    
        tri2full(self.vt_MM)
        return self.vt_MM

