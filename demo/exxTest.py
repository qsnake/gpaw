from ASE import Atom, ListOfAtoms
from gridpaw import Calculator
from gridpaw.atom.all_electron import AllElectron as AE
from gridpaw.exx import atomic_exact_exchange as aExx

# initialize output text string
out  = '|---------------------------------------------------|\n'
out += 'All units in Ang and eV\n'

# perform all electron calculation on isolated atom
# Note: all electron calculations are always spin saturated
#       and output is in atomic units!
atom = AE('Li')
atom.run()
atomExx = aExx(atom, 'all')
out += '\nExchange energy of atom (spin saturated):'
out += '\nvalence-valence: %s' %(aExx(atom, 'val-val') * 27.211395655517311)
out += '\nvalence-core   : %s' %(aExx(atom, 'val-core') * 27.211395655517311)
out += '\ncore-core      : %s' %(aExx(atom, 'core-core') * 27.211395655517311)
out += '\ntotal          : %s' %(aExx(atom, 'all') * 27.211395655517311)


# setup gridPAW calculation
a = 7.0    # size of unit cell
h = 0.2    # grid spacing
d = 2.6729 # binding lenght of Li2
b = a / 2  # middle of unit cell

Li  = ListOfAtoms([Atom('Li', (b, b, b), magmom=1)], cell=(a, a, a))
Li2 = ListOfAtoms([Atom('Li', (b, b, b - d / 2)),
                   Atom('Li', (b, b, b + d / 2))],
                  cell=(a, a, a))

calc = Calculator(xc='PBE',h=0.2,tolerance=1e-5)#reduced tolerance to save time

# perform gridPAW calculation on isolated atom (spin polarized)
Li.SetCalculator(calc)
P  = Li.GetPotentialEnergy()
XC = calc.GetXCEnergy()
XX = calc.GetExactExchange(decompose = True)
out += '\n\nExchange energy of atom (spin polarized):'
out += '\nvalence-valence: %s' %XX[1]
out += '\nvalence-core   : %s' %XX[2]
out += '\ncore-core      : %s' %XX[3]
out += '\ntotal          : %s' %XX[0]
XX = XX[0]

# perform gridPAW calculation on Li2 molecule
Li2.SetCalculator(calc)
P2  = Li2.GetPotentialEnergy()
XC2 = calc.GetXCEnergy()
XX2 = calc.GetExactExchange() # default: decompose = False
out += '\n\nAtomization energy %s' %(P2 - 2 * P)
out += '\nExchange-only atomization energy:  %s' %(
                 P2 - XC2 + XX2 - 2 * (P - XC + XX))

# print output
out += '\n|---------------------------------------------------|\n'
print out
