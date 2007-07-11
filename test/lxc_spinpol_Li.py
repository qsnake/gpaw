from ASE import Crystal, Atom
from ASE.Units import units
from gpaw.utilities import equal
from gpaw import Calculator
from gpaw import setup_paths
from lxc_testsetups import Lxc_testsetups

setup_paths.insert(0, '.')

setups = Lxc_testsetups()
setups.create()

tolerance = 0.000001 # libxc must reproduce old gpaw energies
# zero Kelvin: in Hartree
reference_886 = { # version 886
    'lxcX-C_PW': -7.3970905596, # 'LDA'
    'lxcX_PBE-C_PBE': -7.51091709464, # 'PBE'
    'lxcX_PBE_R-C_PBE': -7.5341232518 # 'revPBE'
    }

units.SetEnergyUnit('Hartree')
a = 5.0
n = 24
li = Crystal([Atom('Li', (0.0, 0.0, 0.0), magmom=1.0)], cell=(a, a, a))

calc = Calculator(gpts=(n, n, n), nbands=1, xc='lxcX_PBE-C_PBE')
li.SetCalculator(calc)
e = li.GetPotentialEnergy() + calc.GetReferenceEnergy()
equal(e, -7.462, 0.056)
equal(e, reference_886['lxcX_PBE-C_PBE'], tolerance)

calc.Set(xc='lxcX_PBE_R-C_PBE')
erev = li.GetPotentialEnergy() + calc.GetReferenceEnergy()
equal(erev, -7.487, 0.057)
equal(erev, reference_886['lxcX_PBE_R-C_PBE'], tolerance)
equal(e - erev, 0.025, 0.002)

calc.Set(xc='lxcX-C_PW')
elda = li.GetPotentialEnergy() + calc.GetReferenceEnergy()
equal(elda, reference_886['lxcX-C_PW'], tolerance)

setups.clean()
