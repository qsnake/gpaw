"""Diffusion along rows"""

from ase import *

a = 4.0614
b = a / sqrt(2)
h = b / 2
initial = Atoms([Atom('Al', (0, 0, 0)),
                 Atom('Al', (a / 2, b / 2, -h))],
                pbc=(1, 1, 0),
                cell=(a, b, 2 * h))
initial *= (2, 2, 2)
initial.append(Atom('Al', (a / 2, b / 2, 3 * h)))
initial.center(vacuum=4., axis=2)

final = initial.copy()
final.positions[-1, 1] += b

view([initial, final])
stop

# Construct a list of images:
images = [initial]
for i in range(5):
    images.append(initial.copy())
images.append(final)

# Make a mask of zeros and ones that select the dynamic atoms (the
# three topmost layers):
mask = initial.positions[:, 2] < 0.5 * h
constraint = FixAtoms(mask=mask)
print mask

for image in images:
    # Let all images use an EMT calculator:
    image.set_calculator(EMT())
    image.set_constraint(constraint)

# Relax the initial and final states:
QuasiNewton(initial).run(fmax=0.05)
QuasiNewton(final).run(fmax=0.05)

# Create a Nudged Elastic Band:
neb = NEB(images)

# Mak a starting guess for the minimum energy path (a straight line
# from the initial to the final state):
neb.interpolate()

# Relax the NEB path:
minimizer = QuasiNewton(neb)
minimizer.run(fmax=0.05)

# Write the path to a trajectory:
view(images)
write('jump1.traj', images)
