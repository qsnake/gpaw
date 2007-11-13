# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import Numeric as num
import copy
import gpaw.mpi as mpi
from gpaw.occupations import Dummy

class ControlOccupation(Dummy):
    """The occupation object for performing /_\SCF-calculations.
       It does the same as the usual FermiDirac object, except it fixes the
       occupation of some of the Kohn-Sham states through the
       controlled_states object."""

    def __init__(self, ne, nspins, kT, controlled_states = None):
       
        Dummy.__init__(self, ne, nspins)
        self.kT = kT
        self.controlled_states = controlled_states

    def calculate(self, kpts):

        self.stat_occs = self.controlled_states.get_states_and_occupations()

        if self.epsF is None:
            # Fermi level not set.  Make a good guess:
            self.guess_fermi_level(kpts)
            
        # Now find the correct Fermi level:
        self.find_fermi_level(kpts)

        S = 0.0
        for kpt in kpts:
            if self.fixmom:
                x = num.clip((kpt.eps_n - self.epsF[kpt.s]) / self.kT, -100.0, 100.0)
            else:
                x = num.clip((kpt.eps_n - self.epsF) / self.kT, -100.0, 100.0)
            y = num.exp(x)
            z = y + 1.0
            y *= x
            y /= z
            y -= num.log(z)
            S -= kpt.weight * num.sum(y)

        self.S = self.kpt_comm.sum(S) * self.kT
        self.calculate_band_energy(kpts)

    def guess_fermi_level(self, kpts):
        """Guesses the Fermi level, which sets the occupation of
           the non-controlled Kohn-Sham states."""

        stat_occs = self.stat_occs
        
        nu = len(kpts) * self.kpt_comm.size
        nb = len(kpts[0].eps_n)

        # Make a long array for all the eigenvalues:
        n = self.ne * nu
        list_eps_n = []
        n_rem = 0.
        for kpt in kpts:
            temp_eps_n = []
            for e in kpt.eps_n:
                temp_eps_n.append(e)
            removes = []
            for so in stat_occs[kpt.u]:
                if not so[0] in removes:
                    removes.append(so[0])
                    n_rem += so[1]
            removes.sort()
            removes.reverse()
            for rem in removes:
                del temp_eps_n[rem]
            list_eps_n += temp_eps_n

        list_eps_n = num.array(list_eps_n)

        if self.kpt_comm.size > 1:
            eps_n = mpi.all_gather_array(self.kpt_comm, list_eps_n)
        else:
            eps_n = list_eps_n.flat

        n_rem = self.kpt_comm.sum(n_rem)
 
        # Sort them:
        eps_n = num.sort(eps_n)
        n = int(n - n_rem + 0.5)
        try:
            self.epsF = 0.5 * (eps_n[n // 2] + eps_n[(n - 1) // 2])
        except:
            self.epsF = 0.0
            raise

        if self.fixmom:
            self.epsF = num.array([self.epsF, self.epsF])


    def find_fermi_level(self, kpts):
        """Find the Fermi level by integrating in energy until
        the number of electrons is correct. For fixed spin moment calculations
        a separate Fermi level for spin up and down electrons is set
        in order to fix also the magnetic moment"""

        stat_occs = self.stat_occs

        niter = 0
        while True:
            if self.fixmom:
                n = num.zeros(2, num.Float)
                dnde = num.zeros(2, num.Float)
            else:
                n = 0.0
                dnde = 0.0
            magmom = 0.0
            for kpt in kpts:
                sign = 1.0 - 2 * kpt.s
                if self.fixmom:
                    x = num.clip((kpt.eps_n - self.epsF[kpt.s]) / self.kT,
                                 -100.0, 100.0)
                    x = num.exp(x)
                    kpt.f_n[:] = kpt.weight / (x + 1.0)

                    dn_corr = 0.
                    dnde_corr = 0.
                    for so in stat_occs[kpt.u]:
                        kpt.f_n[so[0]] = (kpt.weight*self.nspins/2.) * so[1]
                        dn_corr += kpt.f_n[so[0]]
                        dnde_corr += kpt.f_n[so[0]]**2
                    
                    dn = num.sum(kpt.f_n)
                    n[kpt.s] += dn
                    dnde[kpt.s] += (((dn - dn_corr) - (num.sum(kpt.f_n**2)
                                                 - dnde_corr) / kpt.weight)
                                    / self.kT)
                else:
                    x = num.clip((kpt.eps_n - self.epsF) / self.kT,
                                 -100.0, 100.0)
                    x = num.exp(x)
                    kpt.f_n[:] = kpt.weight / (x + 1.0)

                    dn_corr = 0.
                    dnde_corr = 0.
                    for so in stat_occs[kpt.u]:
                        kpt.f_n[so[0]] = (kpt.weight*self.nspins/2.) * so[1]
                        dn_corr += kpt.f_n[so[0]]
                        dnde_corr += kpt.f_n[so[0]]**2
                    
                    dn = num.sum(kpt.f_n)
                    n += dn
                    dnde += (((dn - dn_corr) -
                              (num.sum(kpt.f_n**2) - dnde_corr) / kpt.weight)
                             / self.kT)

                magmom += sign * dn

            # comm.sum has to be called differently when summing scalars
            # than when summing arrays
            if self.fixmom:
                self.kpt_comm.sum(n)
                self.kpt_comm.sum(dnde)
            else:
                n = self.kpt_comm.sum(n)
                dnde = self.kpt_comm.sum(dnde)
            magmom = self.kpt_comm.sum(magmom)

            if self.fixmom:
                ne = num.array([(self.ne + self.M) / 2,
                                (self.ne - self.M) / 2])
                dn = ne - n
                if num.alltrue(abs(dn) < 1.0e-9):
                    if abs(magmom - self.M) > 1.0e-8:
                        raise RuntimeError, 'Magnetic moment not fixed'
                    break
                if num.sometrue(abs(dnde) <  1.0e-9):
                    self.guess_fermi_level(kpts)
                    continue
            else:
                dn = self.ne - n
                if abs(dn) < 1.0e-9:
                    break
                if abs(dnde) <  1.0e-9:
                    self.guess_fermi_level(kpts)
                    continue
            if niter == 100:
                self.guess_fermi_level(kpts)
            if niter > 1000:
                raise RuntimeError, 'Could not locate the Fermi level!'
            de = dn / dnde
            if abs(de) > self.kT:
                de *= self.kT / abs(de)
            self.epsF += de
            niter += 1

        if self.nspins == 1:
            magmom = 0.0

        self.niter = niter
        self.magmom = magmom

class SimpleControl:
    """The simplest possible controlled_states object. Example of use:
       >>> cs = SimpleControl([[[32,1.0],[21,0.0]],[[44,1.0]]], paw)
       >>> co = ControlOccupation(..., controlled_states = cs)
       >>> paw.occupations = co
       Here the occupation of state no 32 (starting from 0) in kpoint no 0 is
       fixed to 1.0, the occupation of state no 21 in kpoint no 0 is fixed to
       0.0 and the occupation of state no 44 in kpoint no 1 is fixed to 1.0"""

    def __init__(self, states_and_occupations, paw):

        self.paw = paw
        sao = states_and_occupations
        self.stat_occs = []
        if len(sao) == 0 or type(sao[0][0]) == type(int):
            for kpt in paw.kpt_u:
                self.stat_occs.append(sao)
        elif len(paw.kpt_u) == len(sao):
            self.stat_occs = sao
        else:
            for kpt in paw.kpt_u:
                self.stat_occs.append(sao[kpt.s*(len(sao)/paw.nspins)+kpt.k])

    def get_states_and_occupations(self):

        return self.stat_occs

class FixedKS:
    """Controlled states object. Recognizes the Kohn-Sham states between the
       SCF iterations, such that the occupations are fixed for the same
       Kohn-Sham states in all iterations. It requires a controlled_states
       object to give the occupations in the first iteration."""

    def __init__(self, paw = None, controlled_states = None):

        self.controlled_states = controlled_states
        if paw == None:
            self.projections = controlled_states.projections
        else:
            self.projections = Projections(paw)
        self.first_iteration = True

    def set_controlled_states(self, controlled_states):

        self.controlled_states = controlled_states
        self.first_iteration = True

    def get_states_and_occupations(self):

        cs = self.controlled_states
        proj = self.projections

        if self.first_iteration:
            self.stat_occs = cs.get_states_and_occupations()
            P_unai = proj.get_P_unai()
            self.prev_P = []
            for u in range(len(P_unai)):
                self.prev_P.append([])
                for so in self.stat_occs[u]:
                    self.prev_P[-1].append(P_unai[u][so[0]])
            self.first_iteration = False
        else:
            self.stat_occs, self.prev_P = proj.recognize(self.stat_occs,
                                                         self.prev_P)
            
        return self.stat_occs

class ControlMO:
    """Class to control the occupation by comparing the overlap of the
       Kohn-Sham states with a given molecular orbital. Example of use:
       >>> mo = N2_2pi([0,1], paw)
       >>> cmo = ControlMO(mo, [[0, 1.]], paw)
       >>> paw.occupations = cmo
       This projects all Kohn-Sham states to the 2pi orbital of nitrogen (as
       defined in N2_2pi) and sets the occupation of the one with the highest
       overlap to 1.0."""

    def __init__(self, molecular_orbital, dist, paw):

        self.MO = molecular_orbital
        if type(dist) == type([1]) or type(dist) == type(num.array([1])):
            self.distributor = SimpleDistribution(dist, paw)
        else:
            self.distributor = dist
        self.projections = Projections(paw)

    def get_states_and_occupations(self):

        P_unai = self.projections.get_P_unai()
        overlaps = []
        order = []
        for u in range(len(P_unai)):
            overlaps.append([])
            for n in range(len(P_unai[u])):
                overlaps[-1].append(self.MO.return_projection(P_unai[u][n]))
            order.append(range(len(overlaps[-1])))
            self.order_by_list = overlaps[-1]
            order[-1].sort(self.order_sort)
            
        dist = self.distributor.distribute(overlaps)
        stat_occs = []
        for u in range(len(dist)):
            stat_occs.append([])
            for n in range(len(dist[u])):
                stat_occs[-1].append([order[u][dist[u][n][0]],dist[u][n][1]])

        return stat_occs
            
    def order_sort(self, el1, el2):

        return cmp(self.order_by_list[el2], self.order_by_list[el1])
        

class ControlCombined:
    """Class to collect different controlled_states objects into one"""

    def __init__(self, list_cs):

        self.list_cs = list_cs

    def get_states_and_occupations(self):

        stat_occs = []
        for cs in self.list_cs:
            part_stat_occs = cs.get_states_and_occupations()
            if len(stat_occs) == 0:
                for pso in part_stat_occs:
                    stat_occs.append([])
            for u in range(len(part_stat_occs)):
                stat_occs[u] += part_stat_occs[u]

        return stat_occs


class ControlSpinSeparated:
    """Class to assign different controlled_states objects to different
       spin states (only makes sense in spinpolarized calculations"""

    def __init__(self, cs0, cs1, paw):

        self.paw = paw
        self.cs = [cs0, cs1]

    def get_states_and_occupations(self):

        so = [self.cs[0].get_states_and_occupations(),
              self.cs[1].get_states_and_occupations()]
        stat_occs = []
        for kpt in self.paw.kpt_u:
            stat_occs.append(so[kpt.s][kpt.u])

        return stat_occs


class Projections:
    """Central class to perform all operations which involves the projectors
       (P_uni)"""

    def __init__(self, paw):

        self.paw = paw

    def get_P_auni(self):

        paw = self.paw

        P_auni = []
        data_types = ['D', 'd']
        for nucleus in paw.nuclei:
            info = num.array([2, 0], num.Int)
            if nucleus.in_this_domain:
                if type(nucleus.P_uni[0][0][0]) == type(complex()):
                    info[0] = 0
                elif type(nucleus.P_uni[0][0][0]) == type(float()):
                    info[0] = 1
                info[1] = nucleus.get_number_of_partial_waves()
            paw.domain.comm.broadcast(info, nucleus.rank)

            if nucleus.in_this_domain:
                P_uni = nucleus.P_uni
            else:
                shape = (paw.nmyu, paw.nbands, info[1])
                P_uni = num.zeros(shape, data_types[info[0]])
            paw.domain.comm.broadcast(P_uni, nucleus.rank)

            P_auni.append(P_uni)

        return copy.deepcopy(P_auni)
            
    def get_P_unai(self):

        P_auni = self.get_P_auni()
        P_unai = []
        for u in range(len(P_auni[0])):
            P_unai.append([])
            for n in range(len(P_auni[0][0])):
                P_unai[u].append([])
                for a in range(len(P_auni)):
                    P_unai[u][n].append(P_auni[a][u][n])

        return P_unai

    def recognize(self, stat_occs, prev_P):

        P_unai = self.get_P_unai()
        next_P = []
        for u in range(len(P_unai)):
            next_P.append([])
            index = 0
            for prev_P_ai in prev_P[u]:
                state_no = self.state_no(prev_P_ai, P_unai[u])
                next_P[-1].append(P_unai[u][state_no])
                stat_occs[u][index][0] = state_no
                index += 1

        return stat_occs, next_P

    def state_no(self, P_ai, P_nai, return_all = False):

        deviations = []
        for n in range(len(P_nai)):
            deviations.append(self.compare(P_ai, P_nai[n]))

        order = range(len(deviations))
        self.order_by_list = deviations
        order.sort(self.order_sort)

        return order[0]

    def compare_old(self, P1, P2):

        max_val = 0.
        for a in range(len(P1)):
            for i in range(len(P1[a])):
                if abs(P1[a][i]) > max_val:
                    max_val = abs(P1[a][i])
                    ma, mi = a, i

        phase = P1[ma][mi] / P2[ma][mi]
        phase /= abs(phase)

        deviation = 0.
        for a in range(len(P1)):
            for i in range(len(P1[a])):
                deviation += abs(P1[a][i] - phase*P2[a][i])
        
        return deviation

    def compare(self, P1, P2):

        phases = []
        for a in range(len(P1)):
            max_val = 0.
            for i in range(len(P1[a])):
                if abs(P1[a][i]) > max_val:
                    max_val = abs(P1[a][i])
                    mi = i
            phases.append(P1[a][mi] / P2[a][mi])

            phases[-1] /= abs(phases[-1])

        deviation = 0.
        for a in range(len(P1)):
            for i in range(len(P1[a])):
                deviation += abs(P1[a][i] - phases[a]*P2[a][i])
        
        return deviation


    def order_sort(self, el1, el2):

        return cmp(self.order_by_list[el1], self.order_by_list[el2])


class SimpleDistribution:
    """Distributor class used by ControlMO"""

    def __init__(self, dist, paw):

        self.paw = paw
        self.dist = []
        if type(dist[0][0]) == type(0):
            for kpt in paw.kpt_u:
                self.dist.append(dist)
        else:
            for kpt in paw.kpt_u:
                self.dist.append(dist[kpt.s*(len(dist)/paw.nspins)+kpt.k])

    def distribute(self, overlaps):

        return self.dist


class DoubleDegenerateDistribution:
    """Distributor class, which can be passed to ControlMO. Helpful in some
       special situations, with double degenerate Kohn-Sham states."""

    def __init__(self, paw, remove = False):

        self.paw = paw
        self.remove = remove

    def distribute(self, overlaps):

        self.dist = []
        for ol in overlaps:
            ol.sort()
            ol.reverse()
            tot = abs(ol[0]) + abs(ol[1])
            self.dist.append([[0, abs(ol[0])/tot],
                              [1, abs(ol[1])/tot]])

        if self.remove:
            for i in range(len(self.dist)):
                self.dist[i][0][1] = 1. - self.dist[i][0][1]
                self.dist[i][1][1] = 1. - self.dist[i][1][1]
                if self.paw.nspins == 1:
                    self.dist[i][0][1] += 1.
                    self.dist[i][1][1] += 1.
        
        return self.dist


class N2_2pi:
    """Example of a molecular_orbital object, which can be passed to
       ControlMO"""

    def __init__(self, atomic_numbers):

        self.an = atomic_numbers

    def return_projection(self, P_ai):

        return (abs(P_ai[self.an[0]][3] - P_ai[self.an[1]][3])**2 +
                abs(P_ai[self.an[0]][1] - P_ai[self.an[1]][1])**2 )

