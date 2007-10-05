from ASE import Crystal, Atom
from ASE.Units import units
from gpaw.utilities import equal
from gpaw import Calculator
from gpaw import setup_paths
from lxc_testsetups import Lxc_testsetups

setup_paths.insert(0, '.')

setups = Lxc_testsetups()
setups.create()

tolerance = 0.000003 # libxc must reproduce old gpaw energies
# zero Kelvin: in Hartree
reference_886 = { # version 886
    'X-C_PW': -7.3970905596, # 'LDA'
    'X_PBE-C_PBE': -7.51091709464, # 'PBE'
    'X_PBE_R-C_PBE': -7.5341232518, # 'revPBE'
    'RPBE': -7.53943687939, # 'RPBE'
    'PW91': -7.52300459704, # 'PW91'
    'oldLDA': -7.3970905596, # 'oldLDA'
    'LDA': -7.3970905596 # 'LDA'
    }

units.SetEnergyUnit('Hartree')
a = 5.0
n = 24
li = Crystal([Atom('Li', (0.0, 0.0, 0.0), magmom=1.0)], cell=(a, a, a))

calc = Calculator(gpts=(n, n, n), nbands=1, xc='X_PBE-C_PBE')
li.SetCalculator(calc)
e = li.GetPotentialEnergy() + calc.get_reference_energy()
equal(e, -7.462, 0.056)
equal(e, reference_886['X_PBE-C_PBE'], tolerance)

calc.set(xc='X_PBE_R-C_PBE')
erev = li.GetPotentialEnergy() + calc.get_reference_energy()
equal(erev, -7.487, 0.057)
equal(erev, reference_886['X_PBE_R-C_PBE'], tolerance)
equal(e - erev, 0.025, 0.002)

calc.set(xc='X-C_PW')
elda = li.GetPotentialEnergy() + calc.get_reference_energy()
equal(elda, reference_886['X-C_PW'], tolerance)

calc.set(xc='RPBE')
erpbe = li.GetPotentialEnergy() + calc.get_reference_energy()
equal(erpbe, reference_886['RPBE'], tolerance)

calc.set(xc='PW91')
epw91 = li.GetPotentialEnergy() + calc.get_reference_energy()
equal(epw91, reference_886['PW91'], tolerance)

calc.set(xc='oldLDA')
eoldlda = li.GetPotentialEnergy() + calc.get_reference_energy()
equal(eoldlda, reference_886['oldLDA'], tolerance)

calc.set(xc='LDA')
eldapw = li.GetPotentialEnergy() + calc.get_reference_energy()
equal(eldapw, reference_886['LDA'], tolerance)

setups.clean()
