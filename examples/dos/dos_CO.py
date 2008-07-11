import pylab as p
from gpaw import Calculator

calc = Calculator('CO.gpw', txt=None)
p.plot(*calc.get_dos(width=.1))
p.axis('tight')
p.xlabel(r'$\epsilon - \epsilon_F \ \rm{(eV)}$')
p.ylabel('Density of States (1/eV)')
p.show()
