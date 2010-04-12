from ase import *
from gpaw.utilities import equal
from gpaw import GPAW
from gpaw import setup_paths
from lxc_testsetups import Lxc_testsetups

setup_paths.insert(0, '.')

setups = Lxc_testsetups()
setups.create()

tolerance = 0.000003 * Hartree # libxc must reproduce old gpaw energies
# zero Kelvin: in Hartree
reference_886 = { # version 886
    'X-C_PW': -7.3970905596 * Hartree, # 'LDA'
    'X_PBE-C_PBE': -7.51091709464 * Hartree, # 'PBE'
    'X_PBE_R-C_PBE': -7.5341232518 * Hartree, # 'revPBE'
    'oldRPBE': -7.53943687939 * Hartree, # 'oldRPBE'
    'PW91': -7.52300459704 * Hartree, # 'PW91'
    'oldLDA': -7.3970905596 * Hartree, # 'oldLDA'
    'LDA': -7.3970905596 * Hartree # 'LDA'
    }

a = 5.0
n = 24
li = Atoms([Atom('Li', (0.0, 0.0, 0.0), magmom=1.0)], cell=(a, a, a), pbc=True)

calc = GPAW(gpts=(n, n, n), nbands=1, xc='X_PBE-C_PBE')
li.set_calculator(calc)
e = li.get_potential_energy() + calc.get_reference_energy()
equal(e, -7.462 * Hartree, 0.056 * Hartree)
equal(e, reference_886['X_PBE-C_PBE'], tolerance)

calc.set(xc='X_PBE_R-C_PBE')
erev = li.get_potential_energy() + calc.get_reference_energy()
equal(erev, -7.487 * Hartree, 0.057 * Hartree)
equal(erev, reference_886['X_PBE_R-C_PBE'], tolerance)
equal(e - erev, 0.025 * Hartree, 0.002 * Hartree)

calc.set(xc='X-C_PW')
elda = li.get_potential_energy() + calc.get_reference_energy()
equal(elda, reference_886['X-C_PW'], tolerance)

calc.set(xc='RPBE')
erpbe = li.get_potential_energy() + calc.get_reference_energy()
equal(erpbe, reference_886['oldRPBE'], tolerance)

calc.set(xc='PW91')
epw91 = li.get_potential_energy() + calc.get_reference_energy()
equal(epw91, reference_886['PW91'], tolerance)

calc.set(xc='oldLDA')
eoldlda = li.get_potential_energy() + calc.get_reference_energy()
equal(eoldlda, reference_886['oldLDA'], tolerance)

calc.set(xc='LDA')
eldapw = li.get_potential_energy() + calc.get_reference_energy()
equal(eldapw, reference_886['LDA'], tolerance)

setups.clean()
