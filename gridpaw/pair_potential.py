# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import weakref

import Numeric as num

from gridpaw.interaction import GInteraction2 as GInteraction
from gridpaw.neighbor_list import NeighborList
from gridpaw.transrotation import rotate


class Neighbor:
    def __init__(self, v_LL, dvdr_LLc, nucleus):
        self.nucleus = weakref.ref(nucleus)
        self.v_LL = v_LL
        self.dvdr_LLc = dvdr_LLc


class PairPotential:
    def __init__(self, setups):
        # Collect the pair potential cutoffs in a list:
        self.cutoff_a = []
        for Z, setup in setups.items():
            self.cutoff_a.append((Z, setup.rcut2))
        
        # Make pair interactions:
        items = setups.items()
        self.interactions = {}
        for Z1, setup1 in items:
            for Z2, setup2 in items:
                interaction = GInteraction(setup1, setup2)
                self.interactions[(Z1, Z2)] = interaction

        self.neighborlist = None
        
    def update(self, pos_ac, nuclei, domain):
        if self.neighborlist is None:
            # Make a neighbor list object:
            Z_a = [nucleus.setup.Z for nucleus in nuclei]
            self.neighborlist = NeighborList(Z_a, pos_ac, domain,
                                             self.cutoff_a)
            updated = False
        else:
            updated = self.neighborlist.update_list(pos_ac)

        # Reset all pairs:
        for nucleus in nuclei:
            nucleus.neighbors = []

        # Make new pairs:
        cell_c = domain.cell_c
        angle = domain.angle
        for a1 in range(len(nuclei)):
            nucleus1 = nuclei[a1]
            Z1 = nucleus1.setup.Z
            for a2, offsets in self.neighborlist.neighbors(a1):
                nucleus2 = nuclei[a2]
                Z2 = nucleus2.setup.Z
                interaction = self.interactions[(Z1, Z2)]
                diff_c = pos_ac[a2] - pos_ac[a1]
                V_LL = num.zeros(interaction.v_LL.shape, num.Float)  #  XXXX!
                dVdr_LLc = num.zeros(interaction.dvdr_LLc.shape, num.Float)
                r_c = pos_ac[a2] - cell_c / 2
                for offset in offsets:
                    d_c = diff_c + offset
                    if angle is not None:
                        rotate(d_c, r_c, -angle * offset[0] / cell_c[0])
                    print a1, a2, d_c
                    v_LL, dvdr_LLc = interaction(d_c)
                    V_LL += v_LL
                    dVdr_LLc += dvdr_LLc
                nucleus1.neighbors.append(Neighbor(V_LL, dVdr_LLc, nucleus2))
                if nucleus2 is not nucleus1:
                    nucleus2.neighbors.append(
                        Neighbor(num.transpose(V_LL),
                                 -num.transpose(dVdr_LLc, (1, 0, 2)),
                                 nucleus1))
        return updated

    def print_info(self, out):
        pass
    """

        print >> out, 'pair potential:'
        print >> out, '  cutoffs:'
            print >> out, '   ', setup.symbol, setup.rcut2 * a0
        npairs = self.neighborlist.number_of_pairs() - len(pos_ac)
        if npairs == 0:
            print >> out, '  There are no pair interactions.'
        elif npairs == 1:
            print >> out, '  There is one pair interaction.'
        else:
            print >> out, '  There are %d pair interactions.' % npairs"""
