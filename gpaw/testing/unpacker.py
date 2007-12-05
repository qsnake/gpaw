#!/usr/bin/env python

import sys
import pickle
from math import sqrt

import Numeric as num
from LinearAlgebra import solve_linear_equations as solve

import data
from data import molecules

# Sort all molecules by mass: ['H2', ... ]
allformulae = zip(*sorted([(sum(molecule.GetMasses()), name)
                           for name, molecule in molecules.items()]))[-1]

class DataUnentangler:
    def __init__(self, results, formulae=allformulae):
        self.results = results
        self.symbols = results['a'].keys()
        self.formulae = formulae

        self.refenergy = dict([(formula, -value['Earef'][0][0])
                               for formula, value
                               in data.moleculedata.items()])

        self.atomic_energy = dict([(symbol, value[0]) for (symbol, value)
                                   in results['a'].items()
                                   if value is not None])        
        
        self.dist_range = {} # For each formula, a range of distances
        self.molecular_energies = {} # Energies corresponding to dist_range
        self.atomization_energy = {} # Atomization energy at reference dist
        self.relaxed_dist = {} # Relaxed bond length
        self.relaxed_energy = {} # Relaxed energy
        self.eggbox_noise = {} # Maximum difference in eggbox test
        self.atomic_energy_sum = {} # yuck, get rid of this
        self.refdist = {} # Reference bond lenghts from molecule.py
        self.convergence_diff = {}        

        for formula, result in results['c'].items():
            if result is None:
                continue
            gpts, energies, niters = result
            # We want to consider only those energies for which h <= .2
            # 10 is default cell size
            # Obtain the cell size dynamically perhaps...
            # also make this number less than 10!
            # USE LESS MAGIC NUMBERS!!
            energies = [e for g, e in zip(gpts, energies)
                        if 10./g <= .201]
            
            self.convergence_diff[formula] = max(energies) - min(energies)

        for formula, result in results['e'].items():
            if result is None:
                continue
            self.eggbox_noise[formula] = max(result[1]) - min(result[1])
        
        for formula in self.formulae:
            positions = molecules[formula].GetCartesianPositions()
            if len(positions) == 2:
                self.refdist[formula] = positions[1, 0] - positions[0, 0]
        
        for formula in self.formulae:
            moleculetest_results = results['m'][formula]
            if moleculetest_results is None:
                continue
            dist, energy, iter = moleculetest_results
            self.dist_range[formula] = dist
            self.molecular_energies[formula] = energy
            self.get_atomization_energy(formula)

        for formula in results['m'].keys():
            system = molecules[formula]
            if len(system) == 2:
                self.get_relaxed_energy(formula)

    def get_relaxed_energy(self, formula):
        dists = self.dist_range.get(formula)
        energies = self.molecular_energies.get(formula)
        M = num.zeros((4, 5), num.Float)

        rel_d = self.dist_range.get(formula)
        if rel_d is None:
            return
        dists = self.refdist[formula]+ num.array(rel_d)
        for n in range(4):
            M[n] = dists**(-n)
        energies = num.array(self.molecular_energies[formula])
        atomic_energy = self.atomic_energy_sum[formula]
        a = solve(num.innerproduct(M, M),
                  num.dot(M, energies - atomic_energy))

        disc = 4 * a[2]**2 - 12 * a[1] * a[3]
        if disc < 0:
            print 'Bad things happen for %s' % formula
            return
        
        dmin = 1 / ((-2 * a[2] + sqrt(disc)) / (6 * a[3]))

        dfit = num.arange(dists[0] * 0.95, dists[4] * 1.05,
                          dists[2] * 0.005)

        efit = sum([a[n] * dfit**(-n) for n in range(4)])
        emin = sum([a[n] * dmin**(-n) for n in range(4)])
        self.relaxed_dist[formula] = dmin
        self.relaxed_energy[formula] = emin
        
    def get_atomization_energy(self, formula):
        system = molecules[formula]

        #print [self.atomic_energy[atom.symbol] for atom in system]
        atomic_energies = sum(filter(None, [self.atomic_energy.get(atom.symbol)
                                            for atom in system]))
            
        self.atomic_energy_sum[formula] = atomic_energies
        # Find entry with zero dislocation from reference value
        min_index = self.dist_range[formula].index(0.)
        molecular_energy = self.molecular_energies[formula][min_index]
        self.atomization_energy[formula] = molecular_energy - atomic_energies

def load(inputfile):
    results = pickle.load(open(inputfile))
    return DataUnentangler(results)
