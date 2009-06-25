from gpaw import *
from ase import *
from ase.parallel import paropen
from gpaw.utilities.tools import split_formula
from gpaw.utilities import equal

cell = [8., 8., 8.]
data = paropen('data.txt', 'w')

##Reference from J. Chem. Phys. Vol 120 No. 15, 15 April 2004, page 6898
tpss_de = {
'LiH': 59.1,
}
tpss_old = {
#'LiH': 56.1625,
'LiH': 57.4,
}

exp_bonds_dE = {
'LiH': (1.595,57.8),
} 

systems = ['LiH']

# Add atoms
for formula in systems:
    temp = split_formula(formula)
    for atom in temp:
        if atom not in systems:
            systems.append(atom)
energies = {}

# Calculate energies
for formula in systems:
    loa = molecule(formula)
    loa.set_cell(cell)
    loa.center()
    calc = GPAW(h=0.3,
                nbands=-2,
                xc='PBE',
                fixmom=True,
                txt=formula + '.txt')
    if len(loa) == 1:
        calc.set(hund=True)
    else:
        pos = loa.get_positions()
        pos[1,:] = pos[0,:] + [0.0, 0.0, exp_bonds_dE[formula][0]]
        loa.set_positions(pos)
    loa.set_calculator(calc)
    try:
        energy = loa.get_potential_energy()
        diff = calc.get_xc_difference('TPSS')
        energies[formula] = (energy, energy + diff)
    except:
        print >> data, formula, 'Error'
    else:
        print >> data, formula, energy, energy + diff
    data.flush()

#calculate atomization energies
file = paropen('tpss.txt', 'w')
print >> file, 'formula\tGPAW\tRef\tGPAW-Ref\tGPAW-exp'
mae_ref, mae_exp, mae_pbe, count = 0.0, 0.0, 0.0, 0
for formula in tpss_de.keys():
    try:
        atoms_formula = split_formula(formula)
        de_tpss = -1.0 * energies[formula][1]
        de_pbe = -1.0 * energies[formula][0]
        for atom_formula in atoms_formula:
            de_tpss += energies[atom_formula][1]
            de_pbe += energies[atom_formula][0]
    except:
        print >>file, formula, 'Error'
    else:
        de_tpss *= 627.5/27.211
        de_pbe *= 627.5/27.211
        mae_ref += abs(de_tpss-tpss_de[formula])
        mae_exp += abs(de_tpss-exp_bonds_dE[formula][1])
        mae_pbe += abs(de_pbe-exp_bonds_dE[formula][1])
        count += 1
        out = "%s\t%.1f\t%.1f\t%.1f\t%.1f kcal/mol"%(formula,de_tpss,tpss_de[formula],
                                            de_tpss-tpss_de[formula],de_tpss-exp_bonds_dE[formula][1])
        print >>file, out
        file.flush()


#comparison to gpaw 0.6.3798 version value in kcal/mol (note the grid:0.3 Ang)
    equal(de_tpss, tpss_old[formula], 0.1)

