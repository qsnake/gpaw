import numpy as np

from ase.units import Hartree
from gpaw.transport.tools import aa1d, interpolate_array, \
                          collect_atomic_matrices, distribute_atomic_matrices



#       ---------------------------------------
#      side    |                     |  side
#          o  o| o  o  o  o  o  o  o |o  o
#       -      |                     |     +
#          o  o| o  o  o  o  o  o  o |o  o
#              |                     |
#       ---------------------------------------
#         Left |                     |Right
#         Lead | <-- Scattering -->  |Lead
#              |       Region        |
#
# class Surrounding is used to deal with the projection close to the
# boundary and combine the potential or density information of the
# scattering region and leads.


class Side:
    #Describe the electrode boundary
    def __init__(self, type, atoms, direction, h=None):
    #direction: '-' for the left electrode, '+' for the right electrode
        self.type = type
        self.atoms = atoms
        self.direction = direction
        self.n_atoms = len(atoms)
        calc = atoms.calc
        self.N_c = calc.gd.N_c.copy()
        self.h_cz = h

    def abstract_boundary(self):
    #abtract the effective potential, hartree potential, and average density
    #out from the electrode calculation.
        calc = self.atoms.calc
        gd = calc.gd
        finegd = calc.finegd
        nn = finegd.N_c[2]
        ns = calc.wfs.nspins

        dim = gd.N_c
        d1 = dim[0] // 2
        d2 = dim[1] // 2

        vHt_g = finegd.collect(calc.hamiltonian.vHt_g)
        vt_sg = finegd.collect(calc.hamiltonian.vt_sg)
        nt_sg = finegd.collect(calc.density.nt_sg)
        rhot_g = finegd.collect(calc.density.rhot_g)
        vt_sG = gd.collect(calc.hamiltonian.vt_sG)
        nt_sG = gd.collect(calc.density.nt_sG)

        self.boundary_vHt_g = None
        self.boundary_vHt_g1 = None
        self.boundary_vt_sg_line = None
        self.boundary_nt_sg = None
        self.boundary_rhot_g_line = None
        self.boundary_vt_sG = None
        self.boundary_nt_sG = None

        if gd.comm.rank == 0:
            self.boundary_vHt_g = self.slice(nn, vHt_g)
            self.boundary_nt_sg = self.slice(nn, nt_sg)
            if self.direction == '-':
                other_direction= '+'
            else:
                other_direction= '-'
            h = self.h_cz / 2.0
            b_vHt_g0 = self.boundary_vHt_g.copy()
            b_vHt_g1 = self.boundary_vHt_g.copy()

            self.boundary_vHt_g = interpolate_array(b_vHt_g0,
                                                    finegd, h, self.direction)
            self.boundary_vHt_g1 = interpolate_array(b_vHt_g1,
                                                   finegd, h, other_direction)

            vt_sg = interpolate_array(vt_sg, finegd, h, self.direction)
            self.boundary_vt_sg_line =  aa1d(vt_sg)
            self.boundary_nt_sg = interpolate_array(self.boundary_nt_sg,
                                                    finegd, h, self.direction)
            rhot_g = interpolate_array(rhot_g, finegd, h, self.direction)
            self.boundary_rhot_g_line = aa1d(rhot_g)

            nn /= 2
            h *= 2
            self.boundary_vt_sG = self.slice(nn, vt_sG)
            self.boundary_nt_sG = self.slice(nn, nt_sG)
            self.boundary_vt_sG = interpolate_array(self.boundary_vt_sG,
                                                    gd, h, self.direction)
            self.boundary_nt_sG = interpolate_array(self.boundary_nt_sG,
                                                    gd, h, self.direction)

        den, ham = calc.density, calc.hamiltonian
        self.D_asp = collect_atomic_matrices(den.D_asp, den.setups,
                                             den.nspins, den.gd.comm,
                                             den.rank_a)
        self.dH_asp = collect_atomic_matrices(ham.dH_asp, ham.setups,
                                              ham.nspins, ham.gd.comm,
                                              ham.rank_a)

        del self.atoms

    def slice(self, nn, in_array):
        if self.type == 'LR':
            seq1 = np.arange(nn)
            seq2 = np.arange(nn)
            di = len(in_array.shape) - 1
            if self.direction == '-':
                slice_array = np.take(in_array, seq1, axis=di)
            else:
                slice_array = np.take(in_array, seq2, axis=di)
        return slice_array


class Surrounding:
    #The potential and density enviroment of the scattering region
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
                side = Side('LR', self.tp.atoms_l[i], direction,
                            self.tp.gd.h_cv[2, 2])
                self.sides[direction] = side
                self.bias_index[direction] = self.tp.bias[i]
                self.side_basis_index[direction] = self.tp.lead_index[i]
                self.nn.append(side.N_c[2])
            self.nn = np.array(self.nn)
            self.operator = \
                        self.tp.extended_calc.hamiltonian.poisson.operators[0]
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

    def calculate_sides(self):
        if self.type == 'LR':
            for name, in self.sides:
                self.sides[name].abstract_boundary()
        if self.type == 'all':
            raise NotImplementError('type all not yet')

    def get_extra_density(self, vHt_g):
        #help to solve poisson euqation for different left and right electrode
        if self.type == 'LR':
            rhot_g = self.tp.finegd1.zeros()
            self.operator.apply(vHt_g, rhot_g)
            nn = self.nn * 2
            self.extra_rhot_g = self.uncapsule(nn, rhot_g, self.tp.finegd1,
                                               self.tp.finegd)

    def capsule(self, nn, loc_in_array, in_cap_array, gd, gd0):
        ns = self.tp.nspins
        cap_array = gd.collect(in_cap_array)
        in_array = gd0.collect(loc_in_array)

        if gd.comm.rank == 0:
            if len(loc_in_array.shape) == 4:
                local_cap_array = gd.zeros(ns)
                cap_array[:, :, :, nn[0]:-nn[1]] = in_array
            else:
                local_cap_array = gd.zeros()
                cap_array[:, :, nn[0]:-nn[1]] = in_array
        else:
            if len(loc_in_array.shape) == 4:
                local_cap_array = gd.zeros(ns)
            else:
                local_cap_array = gd.zeros()
        gd.distribute(cap_array, local_cap_array)
        return local_cap_array

    def uncapsule(self, nn, in_array, gd, gd0):
        nn1 = nn[0]
        nn2 = nn[1]
        ns = self.tp.nspins
        di = 2
        if len(in_array.shape) == 4:
            di += 1
            local_uncap_array = gd0.zeros(ns)
        else:
            local_uncap_array = gd0.zeros()
        global_in_array = gd.collect(in_array)
        if gd.comm.rank == 0:
            seq = np.arange(nn1, global_in_array.shape[di] - nn2)
            uncap_array = np.take(global_in_array, seq, axis=di)
        else:
            uncap_array = None
        gd0.distribute(uncap_array, local_uncap_array)
        return local_uncap_array

    def combine(self):
        if self.type == 'LR':
            nn = self.nn * 2
            ham = self.tp.extended_calc.hamiltonian
            if ham.vt_sg is None:
                ham.vt_sg = ham.finegd.empty(ham.nspins)
                ham.vHt_g = ham.finegd.zeros()
                ham.vt_sG = ham.gd.zeros(ham.nspins)
                ham.poisson.initialize()
                if not self.tp.fixed:
                    self.tp.inner_poisson.initialize()

            bias_shift0 = self.bias_index['-'] / Hartree
            bias_shift1 = self.bias_index['+'] / Hartree
            b_vHt_g0 = self.sides['-'].boundary_vHt_g
            b_vHt_g1 = self.sides['+'].boundary_vHt_g
            b_vHt_g01 = self.sides['-'].boundary_vHt_g1
            b_vHt_g11 = self.sides['+'].boundary_vHt_g1
            if self.tp.fixed:
                if self.tp.gd.comm.rank == 0:
                    self.tp.inner_poisson.initialize(b_vHt_g0 + bias_shift0,
                                                    b_vHt_g1 + bias_shift1)
                else:
                    self.tp.inner_poisson.initialize(None, None)

            if self.tp.gd.comm.rank == 0:
                vHt_g = ham.finegd.zeros(global_array=True)
                extra_vHt_g = ham.finegd.zeros(global_array=True)
                nt_sg = ham.finegd.zeros(self.tp.nspins, global_array=True)
                nt_sG = ham.gd.zeros(self.tp.nspins, global_array=True)

                vHt_g[:, :, :nn[0]] = b_vHt_g0 + bias_shift0
                vHt_g[:, :, -nn[1]:] = b_vHt_g1 + bias_shift1

                mnn = np.min(nn)
                extra_vHt_g[:, :, nn[0]-mnn:nn[0]] = bias_shift0 + \
                                 b_vHt_g0[:,:,-mnn:] - b_vHt_g11[:,:,-mnn:]
                if nn[1] != mnn:
                    extra_vHt_g[:, :, -nn[1]:-nn[1]+mnn] = bias_shift1 + \
                                 b_vHt_g1[:,:,:mnn] - b_vHt_g01[:,:,:mnn]
                else:
                    extra_vHt_g[:, :, -nn[1]:] = bias_shift1 + \
                                 b_vHt_g1[:,:,:mnn] - b_vHt_g01[:,:,:mnn]

                nt_sg[:, :, :, :nn[0]] = self.sides['-'].boundary_nt_sg
                nt_sg[:, :, :, -nn[1]:] = self.sides['+'].boundary_nt_sg

                nn /= 2
                nt_sG[:, :, :, :nn[0]] = self.sides['-'].boundary_nt_sG
                nt_sG[:, :, :, -nn[1]:] = self.sides['+'].boundary_nt_sG
            else:
                nt_sG = None
                nt_sg = None
                vHt_g = None
                extra_vHt_g = None

            loc_extra_vHt_g = ham.finegd.zeros()
            self.nt_sg = ham.finegd.zeros(self.tp.nspins)
            self.nt_sG = ham.gd.zeros(self.tp.nspins)

            ham.gd.distribute(nt_sG, self.nt_sG)
            ham.finegd.distribute(nt_sg, self.nt_sg)
            ham.finegd.distribute(vHt_g, ham.vHt_g)
            ham.finegd.distribute(extra_vHt_g, loc_extra_vHt_g)
            self.get_extra_density(loc_extra_vHt_g)

    def combine_vHt_g(self, vHt_g):
        nn = self.nn * 2
        extended_vHt_g = self.tp.extended_calc.hamiltonian.vHt_g
        self.tp.extended_calc.hamiltonian.vHt_g = self.capsule(nn, vHt_g,
                                                               extended_vHt_g,
                                                               self.tp.finegd1,
                                                               self.tp.finegd)

    def combine_nt_sG(self, nt_sG):
        nn = self.nn
        self.nt_sG = self.capsule(nn, nt_sG, self.nt_sG, self.tp.gd1,
                                  self.tp.gd)
        return self.nt_sG

    def combine_dH_asp(self, dH_asp):
        ham = self.tp.extended_calc.hamiltonian
        all_dH_asp = dH_asp[:]
        for i in range(self.lead_num):
            direction = self.directions[i]
            side = self.sides[direction]
            for n in range(side.n_atoms):
                all_dH_asp.append(side.dH_asp[n])
        ham.dH_asp = {}
        for a, D_sp in self.tp.extended_D_asp.items():
            ham.dH_asp[a] = np.zeros_like(D_sp)
        distribute_atomic_matrices(all_dH_asp, ham.dH_asp, ham.setups)

    def refresh_vt_sG(self):
        nn = self.nn
        gd = self.tp.extended_calc.gd
        bias_shift0 = self.bias_index['-'] / Hartree
        bias_shift1 = self.bias_index['+'] / Hartree
        vt_sG = gd.collect(self.tp.extended_calc.hamiltonian.vt_sG)
        if gd.comm.rank == 0:
            vt_sG[:, :, :, :nn[0]] = self.sides['-'].boundary_vt_sG + \
                                                                   bias_shift0
            vt_sG[:, :, :, -nn[1]:] = self.sides['+'].boundary_vt_sG + \
                                                                   bias_shift1
        gd.distribute(vt_sG, self.tp.extended_calc.hamiltonian.vt_sG)

