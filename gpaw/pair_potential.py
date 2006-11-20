# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import weakref

import Numeric as num

from gpaw.interaction import GInteraction2 as GInteraction
from gpaw.neighbor_list import NeighborList


class Neighbor:
    def __init__(self, v_LL, dvdr_LLc, nucleus):
        self.nucleus = weakref.ref(nucleus)
        self.v_LL = v_LL
        self.dvdr_LLc = dvdr_LLc


class PairPotential:
    def __init__(self, setups):
        # Collect the pair potential cutoffs in a list:
        self.cutoff_a = []
        for setup in setups:
            self.cutoff_a.append((setup.symbol, setup.rcutsoft))
        
        # Make pair interactions:
        self.interactions = {}
        for setup1 in setups:
            for setup2 in setups:
                interaction = GInteraction(setup1, setup2)
                self.interactions[(setup1.symbol, setup2.symbol)] = interaction

        self.neighborlist = None
        
    def update(self, pos_ac, nuclei, domain):
        if self.neighborlist is None:
            # Make a neighbor list object:
            symbol_a = [nucleus.setup.symbol for nucleus in nuclei]
            self.neighborlist = NeighborList(symbol_a, pos_ac, domain,
                                             self.cutoff_a)
        else:
            updated = self.neighborlist.update_list(pos_ac)
            if updated:
                print 'Neighbor list has been updated!' # XXXXX

        # Reset all pairs:
        for nucleus in nuclei:
            nucleus.neighbors = []

        # Make new pairs:
        cell_c = domain.cell_c
        for a1 in range(len(nuclei)):
            nucleus1 = nuclei[a1]
            symbol1 = nucleus1.setup.symbol
            for a2, offsets in self.neighborlist.neighbors(a1):
                nucleus2 = nuclei[a2]
                symbol2 = nucleus2.setup.symbol
                interaction = self.interactions[(symbol1, symbol2)]
                diff_c = pos_ac[a2] - pos_ac[a1]
                V_LL = num.zeros(interaction.v_LL.shape, num.Float)  #  XXXX!
                dVdr_LLc = num.zeros(interaction.dvdr_LLc.shape, num.Float)
                r_c = pos_ac[a2] - cell_c / 2
                for offset in offsets:
                    d_c = diff_c + offset
                    v_LL, dvdr_LLc = interaction(d_c)
                    V_LL += v_LL
                    dVdr_LLc += dvdr_LLc
                nucleus1.neighbors.append(Neighbor(V_LL, dVdr_LLc, nucleus2))
                if nucleus2 is not nucleus1:
                    nucleus2.neighbors.append(
                        Neighbor(num.transpose(V_LL),
                                 -num.transpose(dVdr_LLc, (1, 0, 2)),
                                 nucleus1))

    def print_info(self, out):
        pass
    """

        print >> out, 'pair potential:'
        print >> out, '  cutoffs:'
            print >> out, '   ', setup.symbol, setup.rcutsoft * a0
        npairs = self.neighborlist.number_of_pairs() - len(pos_ac)
        if npairs == 0:
            print >> out, '  There are no pair interactions.'
        elif npairs == 1:
            print >> out, '  There is one pair interaction.'
        else:
            print >> out, '  There are %d pair interactions.' % npairs"""
