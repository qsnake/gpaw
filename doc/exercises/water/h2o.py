from ase.data.molecules import molecule
from gpaw import GPAW

a = 10.0
h = 0.2

energies = {}

for symbol in ['H2O', 'H', 'O']:
    system = molecule(symbol)
    system.set_cell((a, a, a))
    system.center()

    calc = GPAW(h=h,
                txt='gpaw.%s.txt' % symbol)
    if symbol == 'H' or symbol == 'O':
        calc.set(hund=True)
    
    system.set_calculator(calc)
    
    resultfile = open('energy.%s.txt' % symbol, 'w')
    energy = system.get_potential_energy()
    energies[symbol] = energy
    print >> resultfile, energy

e_atomization = energies['H2O'] - 2 * energies['H'] - energies['O']
resultfile = open('energy.atomization.txt', 'w')
print >> resultfile, e_atomization
