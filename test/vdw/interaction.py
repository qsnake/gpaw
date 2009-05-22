import pickle
from ase import *
from gpaw import GPAW
from gpaw.vdw import FFTVDWFunctional, RealSpaceVDWFunctional

d = np.linspace(3.0, 5.5, 11)
for symbol in ['Ar', 'Kr']:
    vdw = FFTVDWFunctional(verbose=1)
    e = np.empty(11)
    de = np.empty(11)
    for i, r in enumerate(d):
        calc = GPAW('%s-dimer-%.2f.gpw' % (symbol, r), txt=None)
        e[i] = calc.get_atoms().get_potential_energy()
        de[i] = calc.get_xc_difference(vdw)
        print i, e, de
    calc = GPAW('%s-atom.gpw' % symbol, txt=None)
    e0 = calc.get_atoms().get_potential_energy()
    de0 = calc.get_xc_difference(vdw)
    print e, de, e0, de0
    pickle.dump((e, de, e0, de0), open(symbol + '.new.pckl', 'w'))

e = np.empty(11)
de = np.empty(11)
vdw = FFTVDWFunctional(verbose=1)
for i, r in enumerate(d):
    calc = GPAW('benzene-dimer-%.2f.gpw' % r, txt=None)
    e[i] = calc.get_atoms().get_potential_energy()
    de[i] = calc.get_xc_difference(vdw)
calc = GPAW('benzene.gpw', txt=None)
e0 = calc.get_atoms().get_potential_energy()
de0 = calc.get_xc_difference(vdw)
print e, de, e0, de0
pickle.dump((e, de, e0, de0), open('benzene.new.pckl', 'w'))
