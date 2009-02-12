# This test calculates spherical harmonic expansion of all-electron kohn-sham potential of helium atom
# using nucleus.calculate_all_electron_potential method

from gpaw import *
from ase import *
from gpaw.atom.all_electron import AllElectron

# Calculate Helium atom using 3D-code
he = Atoms(positions=[(0,0,0)], symbols='He')
he.center(vacuum=3.0)
calc = GPAW(h=0.17)
he.set_calculator(calc)
he.get_potential_energy()

# Get the all-electron potential around the nucleus
vKS_sLg = calc.nuclei[0].calculate_all_electron_potential(calc.hamiltonian.vHt_g)

# Calculate Helium atom using 1D-code
he_atom =AllElectron('He')
he_atom.run()

# Get the KS-potential
vKS_atom = he_atom.vr / he_atom.r
vKS_atom[0] = vKS_atom[1]

# Get the spherical symmetric part and multiply with Y_00
vKS = vKS_sLg[0][0] / sqrt(4*pi)

# Compare
avg_diff = 0.0
for i, v in enumerate(vKS):
    avg_diff += abs(vKS_atom[i]-v)
avg_diff /= len(vKS)

print "Potential expansion is correct to", avg_diff * calc.Ha, " eV"
assert abs(avg_diff * calc.Ha < 0.02)
