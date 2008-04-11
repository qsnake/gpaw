import os

from ase import *
from gpaw import Calculator
from gpaw import setup_paths
from gpaw.atom.all_electron import AllElectron
from gpaw.atom.generator import Generator, parameters

ref1 = 'R. v. Leeuwen PhysRevA 49, 2421 (1994)'
ref2 = 'Gritsenko IntJQuanChem 76, 407 (2000)'
# HOMO energy in mHa for closed shell atoms
e_HOMO_cs = { 'He': 851, 'Be': 321, 'Ne': 788,
              'Ar': 577, 'Kr': 529,# 'Xe': 474,
              'Mg' : 281 + 8 }
e_HOMO_cs = { 'Ne': 788 }
txt=None

print '--- Comparing LB94 with', ref1
print 'and', ref2

print '**** all electron calculations'
print 'atom [refs] -e_homo diff   all in mHa'
for atom in e_HOMO_cs.keys():
    ae = AllElectron(atom, 'LB94', txt=txt)
    ae.run()
    e_homo = int( ae.e_j[-1] * 10000 + .5 ) / 10.
    diff = e_HOMO_cs[atom] + e_homo
    print '%2s %8g %6.1f %4.1g' % (atom, e_HOMO_cs[atom], -e_homo, diff)
    assert( abs(diff) < 6 )

setup_paths.insert(0, '.')
setups = {}

print '**** 3D calculations'
print 'atom [refs] -e_homo diff   all in mHa'

for atom in e_HOMO_cs.keys():
    e_ref = e_HOMO_cs[atom]
    # generate setup for the atom
    if not setups.has_key(atom):
        g = Generator(atom, 'LB94', nofiles=True, txt=txt)
        g.run(**parameters[atom])
        setups[atom] = 1
    
    SS = Atoms([Atom(atom)], cell=(7, 7, 7), pbc=False)
    SS.center()
    c = Calculator(h=.3, xc='LB94', nbands=-2, txt=txt)
    c.calculate(SS)
    # find HOMO energy
    kpt = c.kpt_u[0]
    for e, f in zip(kpt.eps_n, kpt.f_n):
        if f < 0.99:
            break
        e_homo = e
    e_homo = int( e_homo * 10000 + .5 ) / 10.
    diff = e_ref + e_homo
    print '%2s %8g %6.1f %4.1f' % (atom, e_ref, -e_homo, diff)
    assert( abs(diff) < 7 )


# HOMO energy in mHa and magn. mom. for open shell atoms
e_HOMO_os = { 'He': [851, 0], # added for cross check
              'H': [440, 1],
              'N': [534-23, 3],
              'Na':[189+17, 1],
              'P': [385-16, 3] }
e_HOMO_os = { 'Ne': [788, 0], # added for cross check
              'H': [440, 1] }

for atom in e_HOMO_os.keys():
    e_ref = e_HOMO_os[atom][0]
    magmom =  e_HOMO_os[atom][1]
    # generate setup for the atom
    if not setups.has_key(atom):
        g = Generator(atom, 'LB94', nofiles=True, txt=txt)
        g.run(**parameters[atom])
        setups[atom] = 1

    SS = Atoms([Atom(atom, magmom=magmom)], cell=(7, 7, 7), pbc=False)
    SS.center()
    c = Calculator(h=.3, xc='LB94', nbands=-2, spinpol=True,
                   fixmom=True, txt=txt)
    c.calculate(SS)
    # find HOMO energy
    kpt = c.kpt_u[0]
    for e, f in zip(kpt.eps_n, kpt.f_n):
        if f < 0.99:
            break
        e_homo = e
    e_homo = int( e_homo * 10000 + .5 ) / 10.
    diff = e_ref + e_homo
    print '%2s %8g %6.1f %4.1f' % (atom, e_ref, -e_homo, diff)
    assert( abs(diff) < 14 )
        
        
# clean up
for atom in setups:
    os.remove(atom + '.LB94')
