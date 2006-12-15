import pickle
import sys
from gpaw.utilities.molecule import Molecule
from ASE.Units import Convert
from atomization_data import atomization

def atomize(formulas, cellsize, gridspacing, relax=False, non_self_xcs=(),
            forsymm=False, calc_parameters={}):
    """Determine atomization energies for list of molecule specified by
       string-list ``formulas``.
    """

    atom_energies = {}
    eas = {}
    errors = []
    kcal = Convert(1, 'eV', 'kcal/mol/Nav')
    for formula in formulas:
        if not load(formula, eas, errors):
            file = open(formula + '.pickle','w')
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
                eas[formula] = 'Failed: ' + sys.exc_type + sys.exc_value
                errors.append(formula)
            else:
                for i in range(len(xcs) + 1):
                    ea[i] *= kcal
                    if i > 0:
                        ea[i] += ea[0]
                eas[formula] = (atomization[formula][0],) + tuple(ea)
            check_atom_list(atom_energies)
            pickle.dump(eas[formula], file)
    return eas, errors

def load(formula, eas, errors):
    try:
        f = open(formula + '.pickle', 'r')
        eas[formula] = pickle.load(f)
        f.close()
        if type(eas[formula]) == str:
            errors.append(formula)
    except IOError:
        return False
    except EOFError:
        return True
    else:
        return True

def check_atom_list(atom_energies):
    try:
        file = open('atom_energies.pickle', 'r')
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
            file = open('atom_energies.pickle', 'w')
            pickle.dump(atom_energies, file)
            file.close()                
    except IOError:
        file = open('atom_energies.pickle', 'w')
        pickle.dump(atom_energies, file)
        file.close()
    except EOFError:
        file = open('atom_energies.pickle', 'w')
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

