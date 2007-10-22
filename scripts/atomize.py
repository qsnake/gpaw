#!/usr/bin/env python
import pickle
import sys
sys.path.append('.')

from ASE.Units import Convert
from ASE.Utilities.Parallel import paropen

from gpaw.utilities.molecule import Molecule
from gpaw.testing.atomization_data import \
     atomization, ex_atomization, atomization_vasp
from gpaw.mpi import world


def atomize(formulas, cellsize, gridspacing, relax=False, non_self_xcs=[],
            forcesymm=False, calc_parameters={}):
    """Determine atomization energies for list of molecules specified by
       string-list ``formulas``.
    """
    atom_energies = {}
    eas = {}
    errors = []
    kcal = Convert(1, 'eV', 'kcal/mol/Nav')
    for formula in formulas:
        print 'formula', formula
        if not load(formula, eas, errors):
            file = paropen(formula + '.pickle','w')
            mol = Molecule(formula, a=cellsize, h=gridspacing,
                           parameters=calc_parameters, forcesymm=forcesymm)
            check_atom_list(atom_energies)
            try:
                if relax:
                    mol.relax(verbose=True)
                ea = mol.atomize(verbose=True,
                                 atom_energies=atom_energies,
                                 xcs=non_self_xcs)
            except:
                eas[formula] = str(sys.exc_type) + ': ' + str(sys.exc_value)
                errors.append(formula)
            else:
                for i in range(len(non_self_xcs) + 1):
                    ea[i] *= kcal
                    if i > 0:
                        ea[i] += ea[0]
                eas[formula] = (atomization[formula][0],) + tuple(ea)
            check_atom_list(atom_energies)
            pickle.dump(eas[formula], file)
    return eas, errors

def load(formula, eas, errors):
    try:
        f = paropen(formula + '.pickle', 'r')
        eas[formula] = pickle.load(f)
        f.close()
        if type(eas[formula]) == str:
            errors.append(formula)
    except IOError:
        ok = False
    except EOFError:
        ok = True
    else:
        ok = True

    world.barrier()
    return ok

def check_atom_list(atom_energies):
    try:
        file = paropen('atom_energies.pickle', 'r')
        stored = pickle.load(file)
        for sym in stored:
            if sym not in atom_energies:
                atom_energies[sym] = stored[sym]
        file.close()
        update = False
        for sym in atom_energies:
            if sym not in stored:
                update = True
                stored[sym] = atom_energies[sym]
        if update:
            file = paropen('atom_energies.pickle', 'w')
            pickle.dump(atom_energies, file)
            file.close()                
    except IOError:
        file = paropen('atom_energies.pickle', 'w')
        pickle.dump(atom_energies, file)
        file.close()
    except EOFError:
        file = paropen('atom_energies.pickle', 'w')
        pickle.dump(atom_energies, file)
        file.close()

def pretty_print(eas, xcs, formulas=None):
    out = 'Mol  '
    for xc in xcs:
        out += '%5s '%xc
    if formulas == None:
        formulas = eas.keys()
    for formula in formulas:
        if type(eas[formula]) == str:
            out += '\n%s%s: %s'%(formula, (4 - len(formula)) * ' ',
                                 eas[formula])
        else:
            out += '\n%s%s:'%(formula, (4 - len(formula)) * ' ')
            for i in range(len(xcs)):
                out += ' %5.1f'%eas[formula][i]
    return out

def mean_error(eas, errors=[], exact=0):
    import Numeric as num
    d = eas.copy()
    for error in errors:
        d.pop(error)
    a = num.array(d.values())
    out = '\nMAE :'
    if len(a.shape) == 1:
        a.shape = (a.shape[0], 1)
    for i in range(a.shape[1]):
        if i == exact:
            mae = 0.0
        else:
            mae = num.sum(num.absolute(a[:,i] - a[:,exact])) / len(a)
        out += ' %5.1f'%mae
    return out               

def reference_blaha(molecules):
    eas = {}
    for formula in molecules:
        eas[formula] = atomization[formula][:4] + (ex_atomization[formula][0],)
    xcs = ('Expt', 'LDA', 'PBE', 'RPBE', 'EXX')
    return pretty_print(eas, xcs, molecules) + mean_error(eas)

def reference_vasp(molecules):
    eas = {}
    errors = []
    for formula in molecules:
        if formula not in ['H2', 'Be2']:
            eas[formula] = atomization_vasp[formula]
        else:
            errors.append(formula)
            eas[formula] = 'Not in data set'
    xcs = ('Expt', 'PBE_VASP', 'PBE_G03', 'PBE0_VASP', 'PBE0_G03')
    return pretty_print(eas, xcs, molecules) + mean_error(eas, errors)

if __name__ == '__main__':
    a = 8.0
    h = 0.2
    relax = False
    parameters = {'xc': 'PBE',
                  'txt': 'atomize.txt',
                  'convergence':{'energy': 0.001, # eV
                                 'density': 1.0e-3,
                                 'eigenstates': 1.0e-9,
                                 'bands': 'occupied'},
                  'lmax': 2,
                  'spinpol':True}
    
    molecules = ['HF',
                 'LiH',
                 'CH4',
                 'NH3',
                 'OH',
                 'H2O',
                 'HF',
                 'Li2',
                 'LiF',
                 'Be2',
                 'C2H2',
                 'C2H4',
                 'HCN',
                 'CO',
                 'N2',
                 'NO',
                 'O2',
                 'F2',
                 'P2',
                 'Cl2'
                 ]

    non_self_xcs = ('LDA', 'revPBE', 'RPBE', 'PBE0', 'EXX')
    eas, errors = atomize(molecules, a, h,
                          relax=relax,
                          non_self_xcs=non_self_xcs,
                          forcesymm=False,
                          calc_parameters=parameters)
    names = ('Expt', 'PBE',) + non_self_xcs

    print 'a =', a
    print 'h =', h
    print 'lmax =', parameters['lmax']
    print 'relax =', relax
    print ''
    print 'GPAW'
    print ''
    print pretty_print(eas, names, molecules) + mean_error(eas, errors,exact=0)
    print ''
    print 'Reference Blaha et. al.'
    print reference_blaha(molecules)
    print ''
    print 'Reference Kresse et. al.'
    print reference_vasp(molecules)
